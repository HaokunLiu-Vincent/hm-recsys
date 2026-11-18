"""Tests for kaggle.ranker: training labels, reused M8 samples, model, Top-K, the M8 comparison.

LightGBM side (the file name starts with test_ranker_). The workspace is the tiny
one of final_example.py AFTER a rehearsal and the formal run of M8, so it holds
what the real project holds at tag v0.1: the sample files of V4 T1 T2 T3, the
Dataset record and the model of T4, and T4's saved Top-K. The tiny lock keeps
10 trees and a 4-week window.

    rehearsal    trains on V4 T1 T2 T3 (all reused) -> must give M8's T4 model, and
                 scoring T4's wide table (cut into two "shards") must give M8's Top-K
    submission   trains on T1 T2 T3 T4: T1-T3 reused, T4 sampled now (label file)
"""

import atexit
import dataclasses
import shutil
import tempfile
from pathlib import Path

import polars as pl
import pytest
from final_example import COMMIT, NOW, InProcessWorker, copy_workspace, load_context

from hmrec.config import load_base_config, load_kaggle_config, load_preprocess_config
from hmrec.csv_records import read_rows
from hmrec.feature_files import FeatureFiles, describe_part, read_feature_files, write_feature_files
from hmrec.final.run import run_formal
from hmrec.kaggle.context import KaggleContext
from hmrec.kaggle.plan import RUN_REHEARSAL, RUN_SUBMISSION, target_week
from hmrec.kaggle.ranker import (
    LABEL_ACCESS_COLUMNS,
    RankerContext,
    build_run_dataset,
    check_reused_sample,
    dataset_plan,
    load_ranker_context,
    load_run_model,
    m8_sample_sha256,
    m8_top,
    m8_trees_sha256,
    new_sample_folds,
    read_top,
    rehearsal_outcome,
    sample_file_of,
    sample_new_fold,
    saved_trees_sha256,
    score_shard,
    train_run_model,
)
from hmrec.locks import file_sha256
from hmrec.paths import ProjectPaths, build_paths
from hmrec.ranker.dataset import read_dataset_record
from hmrec.ranker.sampling import read_sample_record, record_file
from hmrec.run_records import RunStamp

REPO = Path(__file__).resolve().parents[2]
KAGGLE = load_kaggle_config(REPO / "configs" / "kaggle_submission.yaml")
PREPROCESS = load_preprocess_config(REPO / "configs" / "preprocess.yaml")
SCRIPT = "scripts/kaggle_ranker.py --step test"
T4_FIRST_QUERY = 12000
N_QUERIES = 40

_TEMPLATE: list[Path] = []


def _workspace(root: Path) -> ProjectPaths:
    """A fresh copy of the tiny workspace after M8's rehearsal AND formal run."""
    if not _TEMPLATE:
        folder = Path(tempfile.mkdtemp(prefix="hmrec_kaggle_ranker_"))
        atexit.register(shutil.rmtree, folder, ignore_errors=True)
        context = copy_workspace(folder, rehearsed=True)
        run_formal(context, InProcessWorker(context), folder, NOW)
        _TEMPLATE.append(folder)
    shutil.copytree(_TEMPLATE[0], root, dirs_exist_ok=True)
    return load_context(build_paths(load_base_config(root / "configs" / "base.yaml"), root)).paths


def _ranker(root: Path, run: str) -> RankerContext:
    paths = _workspace(root)
    final = load_context(paths)
    context = KaggleContext(
        run=run,
        paths=paths,
        base=final.base,
        config=dataclasses.replace(KAGGLE, shard_customers=N_QUERIES // 2),
        manifest=final.manifest,
        preprocess=PREPROCESS,
        target=target_week(run, KAGGLE, final.manifest),
        stamp=RunStamp(COMMIT, False),
    )
    return load_ranker_context(context)


def _access_rows(ranker: RankerContext) -> list[dict[str, str]]:
    return read_rows(ranker.context.paths.kaggle_label_access_file, LABEL_ACCESS_COLUMNS)


# ---------------------------------------------------------------- which folds, which files


def test_rehearsal_reuses_all_four_samples_and_the_submission_samples_t4(tmp_path: Path) -> None:
    rehearsal = _ranker(tmp_path / "r", RUN_REHEARSAL)
    assert [fold.name for fold in rehearsal.training_folds] == ["V4", "T1", "T2", "T3"]
    assert new_sample_folds(rehearsal) == ()
    submission = _ranker(tmp_path / "s", RUN_SUBMISSION)
    assert [fold.name for fold in submission.training_folds] == ["T1", "T2", "T3", "T4"]
    assert [fold.name for fold in new_sample_folds(submission)] == ["T4"]
    paths, ratio = submission.context.paths, submission.ratio
    t3, t4 = submission.training_folds[2], submission.training_folds[3]
    # T3: the file M8 wrote; T4: the run's own file under artifacts/kaggle/.
    assert sample_file_of(submission, t3) == paths.ranker_sample_file("full", "T3", ratio)
    assert sample_file_of(submission, t4) == paths.kaggle_sample_file(RUN_SUBMISSION, "T4", ratio)
    assert paths.kaggle_dir in sample_file_of(submission, t4).parents
    # The proof of what M8 trained on: the Dataset record of its T4 model.
    assert sorted(m8_sample_sha256(submission)) == ["T1", "T2", "T3", "V4"]


def test_a_reused_sample_must_be_the_file_m8_trained_on(tmp_path: Path) -> None:
    ranker = _ranker(tmp_path, RUN_REHEARSAL)
    t1 = ranker.training_folds[1]
    check_reused_sample(ranker, t1)  # untouched: passes
    sample = sample_file_of(ranker, t1)
    sample.write_bytes(sample.read_bytes() + b"x")
    with pytest.raises(ValueError, match="not the M8 sample file"):
        check_reused_sample(ranker, t1)
    with pytest.raises(ValueError, match="not the M8 sample file"):
        build_run_dataset(ranker, SCRIPT)
    # Nothing was built and no label access was recorded for the refused build.
    assert not ranker.context.paths.kaggle_dataset_dir(RUN_REHEARSAL).exists()
    assert _access_rows(ranker) == []


# ---------------------------------------------------------------- the rehearsal


def _write_shards(ranker: RankerContext) -> None:
    """Cut T4's frozen wide table into two "shards" (its two part files) with their records."""
    context = ranker.context
    paths, fold = context.paths, context.target.name
    source = read_feature_files(paths.features_dir(fold))
    for shard, part in enumerate(source.parts):
        directory = paths.kaggle_features_dir(context.run, shard)
        directory.mkdir(parents=True)
        shutil.copyfile(paths.features_dir(fold) / part.name, directory / "part-00000.parquet")
        described = describe_part(directory / "part-00000.parquet")
        record = FeatureFiles(
            fold, source.columns, source.categorical_features, (described,), described.n_rows,
            N_QUERIES // 2, {},
        )  # fmt: skip
        write_feature_files(directory, record)


def _write_queries(ranker: RankerContext) -> pl.DataFrame:
    """Give T4's queries a customer each, and the run the same customers under other query ids.

    Customer c is query 12000 + c of the fold. The run's queries keep the fold's
    query ids here, because the copied wide table carries them.
    """
    paths, context = ranker.context.paths, ranker.context
    query_ids = list(range(T4_FIRST_QUERY, T4_FIRST_QUERY + N_QUERIES))
    queries = pl.DataFrame(
        {"query_id": query_ids, "customer_idx": [q - T4_FIRST_QUERY for q in query_ids]},
        schema={"query_id": pl.Int32, "customer_idx": pl.Int32},
    )
    queries.with_columns(pl.lit("T4").alias("fold")).write_parquet(paths.queries_file("T4"))
    run_queries = paths.kaggle_queries_file(context.run)
    run_queries.parent.mkdir(parents=True, exist_ok=True)
    queries.write_parquet(run_queries)
    return queries


def _rehearse(root: Path) -> tuple[RankerContext, pl.DataFrame]:
    """Dataset, model and the Top-K of both shards of a rehearsal."""
    ranker = _ranker(root, RUN_REHEARSAL)
    build_run_dataset(ranker, SCRIPT)
    train_run_model(ranker)
    _write_shards(ranker)
    _write_queries(ranker)
    booster = load_run_model(ranker)
    paths, run = ranker.context.paths, ranker.context.run
    scored = [score_shard(ranker, booster, shard) for shard in (0, 1)]
    assert all(rows > 0 for rows in scored)
    tops = pl.concat([read_top(paths.kaggle_top_file(run, shard)) for shard in (0, 1)])
    return ranker, tops


def test_rehearsal_trains_the_model_m8_trained_for_t4(tmp_path: Path) -> None:
    ranker = _ranker(tmp_path, RUN_REHEARSAL)
    record = build_run_dataset(ranker, SCRIPT)
    paths = ranker.context.paths
    m8_record = read_dataset_record(paths.final_dataset_dir("t_phase", "full", "T4", ranker.ratio))
    # Same folds, same sample files, same binned data as M8's Dataset; a NEW file.
    assert record.train_folds == ("V4", "T1", "T2", "T3") == m8_record.train_folds
    assert record.sample_sha256 == m8_record.sample_sha256
    assert record.binary_sha256 == m8_record.binary_sha256
    assert paths.kaggle_dir in (paths.kaggle_dataset_dir(RUN_REHEARSAL)).parents
    model = train_run_model(ranker)
    assert model.trees_sha256 == m8_trees_sha256(ranker)
    assert model.model_sha256 == file_sha256(paths.kaggle_model_file(RUN_REHEARSAL))
    assert (model.n_train_rows, model.n_train_queries) == (record.n_rows, record.n_queries)
    # The model is never replaced.
    with pytest.raises(FileExistsError, match="never replaced"):
        train_run_model(ranker)
    # The Dataset build read the label column of the four sample files: one recorded read.
    rows = _access_rows(ranker)
    assert [(row["run"], row["folds"], row["script"]) for row in rows] == [
        ("rehearsal", "V4 T1 T2 T3", SCRIPT)
    ]
    assert rows[0]["git_commit"] == COMMIT and "Dataset build" in rows[0]["purpose"]


def test_rehearsal_scores_shards_to_the_top_k_m8_saved(tmp_path: Path) -> None:
    ranker, tops = _rehearse(tmp_path)
    saved = m8_top(ranker)
    assert saved.height == N_QUERIES * ranker.locked.config.top_k
    assert (
        tops.select(saved.columns).sort("query_id", "rank").equals(saved.sort("query_id", "rank"))
    )
    outcome = rehearsal_outcome(ranker, tops, 2, format_checks_passed=True)
    assert outcome.trees_identical and outcome.passed
    assert (outcome.comparison.n_queries, outcome.comparison.n_identical) == (N_QUERIES, N_QUERIES)
    assert (outcome.n_customers, outcome.n_shards) == (N_QUERIES, 2)
    # A scored shard is skipped, never rewritten.
    assert score_shard(ranker, load_run_model(ranker), 0) == 0


def test_rehearsal_fails_on_a_changed_list_or_a_failed_format_check(tmp_path: Path) -> None:
    ranker, tops = _rehearse(tmp_path)
    assert not rehearsal_outcome(ranker, tops, 2, format_checks_passed=False).passed
    # Swap rank 1 and rank 2 of one query: the same articles in another order.
    first = T4_FIRST_QUERY
    swapped = tops.with_columns(
        pl.when((pl.col("query_id") == first) & (pl.col("rank") <= 2))
        .then(3 - pl.col("rank"))
        .otherwise(pl.col("rank"))
        .cast(pl.Int32)
        .alias("rank")
    )
    outcome = rehearsal_outcome(ranker, swapped, 2, format_checks_passed=True)
    assert outcome.comparison.n_identical == N_QUERIES - 1
    assert outcome.comparison.share == pytest.approx(39 / 40)
    assert outcome.trees_identical and not outcome.top_identical_enough and not outcome.passed


def test_only_a_rehearsal_is_compared_with_m8(tmp_path: Path) -> None:
    ranker = _ranker(tmp_path, RUN_SUBMISSION)
    with pytest.raises(ValueError, match="only the rehearsal"):
        m8_trees_sha256(ranker)
    with pytest.raises(ValueError, match="only a rehearsal"):
        rehearsal_outcome(ranker, pl.DataFrame(), 1, True)


# ---------------------------------------------------------------- the real run


def test_submission_samples_t4_after_recording_the_label_read(tmp_path: Path) -> None:
    ranker = _ranker(tmp_path, RUN_SUBMISSION)
    paths = ranker.context.paths
    t4 = ranker.training_folds[-1]
    m8_files = {
        fold.name: file_sha256(sample_file_of(ranker, fold)) for fold in ranker.training_folds[:3]
    }
    record = sample_new_fold(ranker, t4, SCRIPT)
    sample = paths.kaggle_sample_file(RUN_SUBMISSION, "T4", ranker.ratio)
    assert record.sha256 == file_sha256(sample) and record.fold == "T4"
    # The locked seed and ratio; nothing was written next to M8's samples.
    assert (record.seed, record.ratio) == (ranker.locked.config.sampling.seed, ranker.ratio)
    assert not paths.ranker_sample_file("full", "T4", ranker.ratio).exists()
    assert read_sample_record(record_file(sample)) == record
    rows = _access_rows(ranker)
    assert [(row["run"], row["folds"]) for row in rows] == [("submission", "T4")]
    assert "negative sampling" in rows[0]["purpose"] and rows[0]["script"] == SCRIPT
    # A fold M8 sampled is never sampled again.
    with pytest.raises(ValueError, match="not sampled by this run"):
        sample_new_fold(ranker, ranker.training_folds[0], SCRIPT)
    assert {
        name: file_sha256(sample_file_of(ranker, fold))
        for name, fold in zip(m8_files, ranker.training_folds[:3], strict=True)
    } == m8_files


def test_submission_trains_on_t1_to_t4_from_scratch(tmp_path: Path) -> None:
    ranker = _ranker(tmp_path, RUN_SUBMISSION)
    paths = ranker.context.paths
    with pytest.raises(FileNotFoundError, match="not finished"):
        dataset_plan(ranker)  # T4 is not sampled yet
    sample_new_fold(ranker, ranker.training_folds[-1], SCRIPT)
    record = build_run_dataset(ranker, SCRIPT)
    assert record.train_folds == ("T1", "T2", "T3", "T4") and record.target_fold == "S"
    m8 = m8_sample_sha256(ranker)
    assert {fold: record.sample_sha256[fold] for fold in ("T1", "T2", "T3")} == {
        fold: m8[fold] for fold in ("T1", "T2", "T3")
    }
    model = train_run_model(ranker)
    assert load_run_model(ranker).num_trees() == ranker.locked.num_trees
    # Another training window than M8's T4 model: another model.
    t4_model = paths.ranker_model_file("t_phase", "full", ranker.locked.config_id, "T4")
    assert model.trees_sha256 != saved_trees_sha256(t4_model, ranker.locked.num_trees)
    # Two recorded reads: the label file of T4, then the label columns of the four samples.
    assert [row["folds"] for row in _access_rows(ranker)] == ["T4", "T1 T2 T3 T4"]
    # M8's own files are untouched: its T4 model and Top-K are still what they were.
    assert (
        m8_top(
            dataclasses.replace(
                ranker,
                context=dataclasses.replace(
                    ranker.context, target=ranker.context.manifest.get("T4")
                ),
            )
        ).height
        == N_QUERIES * ranker.locked.config.top_k
    )
