"""Tests for hmrec.kaggle_deepfm.train and .score on the tiny workspace of kaggle_deepfm_example.

The rehearsal of the workspace must reproduce the stage-2 record of T4 with seed
42: the same training loss and, for the T4 customers, the same Top-12 lists.
"""

import dataclasses
import json
import shutil
from pathlib import Path

import polars as pl
import pytest
from deepfm_example import RATIO
from kaggle_deepfm_example import (
    N_FOLD_CUSTOMERS,
    N_REHEARSAL_CUSTOMERS,
    N_SUBMISSION_CUSTOMERS,
    QUERY_ID_BASE,
    SCRIPT,
    SEED,
    build_kaggle_deepfm_workspace,
    kaggle_context,
    write_wide,
)
from t_eval_example import COMMIT, copy_workspace

from hmrec.csv_records import read_rows
from hmrec.deepfm.trial import TrialInputs
from hmrec.kaggle.plan import RUN_REHEARSAL, RUN_SUBMISSION
from hmrec.kaggle_deepfm.context import (
    LABEL_ACCESS_COLUMNS,
    DeepFMKaggleContext,
    training_folds,
)
from hmrec.kaggle_deepfm.report import (
    new_train_loss,
    read_all_tops,
    run_shards,
    stage2_top,
    stage2_train_loss,
)
from hmrec.kaggle_deepfm.score import load_run_model, score_run, score_shard
from hmrec.kaggle_deepfm.train import (
    TRAINING_FILE,
    chosen_inputs,
    read_training_summary,
    run_sample_files,
    train_run_model,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> TrialInputs:
    """The workspace before any DeepFM step of the Kaggle path (built once)."""
    return build_kaggle_deepfm_workspace(tmp_path_factory.mktemp("kaggle_deepfm") / "workspace")


@pytest.fixture(scope="module")
def rehearsed(
    built: TrialInputs, tmp_path_factory: pytest.TempPathFactory
) -> tuple[TrialInputs, DeepFMKaggleContext]:
    """A copy of the workspace with the rehearsal trained and scored."""
    inputs = copy_workspace(built, tmp_path_factory.mktemp("rehearsed") / "workspace")
    context = kaggle_context(inputs, RUN_REHEARSAL)
    chosen = chosen_inputs(context, inputs)
    train_run_model(context, chosen, SCRIPT)
    score_run(context, chosen, run_shards(context))
    return inputs, context


def _access_rows(inputs: TrialInputs) -> list[dict[str, str]]:
    return read_rows(inputs.paths.kaggle_deepfm_label_access_file, LABEL_ACCESS_COLUMNS)


def _lists(top: pl.DataFrame) -> dict[int, list[int]]:
    """query_id -> its articles in rank order."""
    table = top.group_by("query_id").agg(pl.col("article_idx").sort_by("rank"))
    return dict(zip(table["query_id"].to_list(), table["article_idx"].to_list(), strict=True))


# ---------------------------------------------------------------- configuration and files


def test_the_configuration_is_the_chosen_one_with_the_submissions_seed(built: TrialInputs) -> None:
    context = kaggle_context(built, RUN_SUBMISSION)
    chosen = chosen_inputs(context, built)
    assert (chosen.config.model.numeric_input, chosen.config.training.loss) == (
        "standardised",
        "listwise_softmax",
    )
    assert (chosen.config.training.seed, chosen.config.training.epochs) == (SEED, 1)
    # Another configuration id than the recorded one: nothing is trained.
    settings = dataclasses.replace(context.settings, config_id="0" * 12)
    with pytest.raises(ValueError, match="Nothing is trained"):
        chosen_inputs(dataclasses.replace(context, settings=settings), built)


def test_each_run_trains_on_the_sample_files_of_its_lightgbm_ranker(built: TrialInputs) -> None:
    paths = built.paths
    rehearsal = kaggle_context(built, RUN_REHEARSAL)
    submission = kaggle_context(built, RUN_SUBMISSION)
    rehearsal_files = run_sample_files(rehearsal, built, training_folds(rehearsal, 4))
    assert list(rehearsal_files) == ["V4", "T1", "T2", "T3"]
    assert all(
        file == paths.ranker_sample_file("full", fold, RATIO)
        for fold, file in rehearsal_files.items()
    )
    submission_files = run_sample_files(submission, built, training_folds(submission, 4))
    assert list(submission_files) == ["T1", "T2", "T3", "T4"]
    # T4 was drawn by the LightGBM submission itself: its file is in that run's folder.
    assert submission_files["T4"] == paths.kaggle_sample_file(RUN_SUBMISSION, "T4", RATIO)
    assert submission_files["T3"] == paths.ranker_sample_file("full", "T3", RATIO)


# ---------------------------------------------------------------- the rehearsal


def test_the_rehearsal_reproduces_the_stage_2_record_of_t4(
    rehearsed: tuple[TrialInputs, DeepFMKaggleContext],
) -> None:
    inputs, context = rehearsed
    # 1. The same mean training loss, as float64.
    summary = read_training_summary(inputs.paths.kaggle_deepfm_model_dir(RUN_REHEARSAL))
    assert summary.train_folds == ("V4", "T1", "T2", "T3")
    assert (summary.seed, summary.epochs, summary.config_id) == (
        SEED,
        1,
        context.settings.config_id,
    )
    assert new_train_loss(context) == summary.mean_loss == stage2_train_loss(context)
    # 2. The same Top-12 list for every T4 customer (customer k: the k-th query of T4).
    new_lists = _lists(read_all_tops(context))
    old_lists = _lists(stage2_top(context))
    assert len(new_lists) == N_REHEARSAL_CUSTOMERS and len(old_lists) == N_FOLD_CUSTOMERS
    first_old = min(old_lists)
    for customer in range(N_FOLD_CUSTOMERS):
        assert new_lists[QUERY_ID_BASE + customer] == old_lists[first_old + customer]
    # Every customer of the run has a complete Top-12, also the ones that are not T4 queries.
    assert all(len(articles) == 12 for articles in new_lists.values())


def test_the_label_read_is_recorded_before_the_training(
    rehearsed: tuple[TrialInputs, DeepFMKaggleContext],
) -> None:
    inputs, _ = rehearsed
    rows = _access_rows(inputs)
    assert [(row["run"], row["folds"], row["script"]) for row in rows] == [
        ("rehearsal", "V4 T1 T2 T3", SCRIPT)
    ]
    assert rows[0]["purpose"] == "DeepFM training labels: label column of the sample files"
    assert (rows[0]["git_commit"], rows[0]["git_dirty"]) == (COMMIT, "False")


def test_scoring_does_not_depend_on_the_shards_and_skips_finished_shards(
    rehearsed: tuple[TrialInputs, DeepFMKaggleContext], tmp_path: Path
) -> None:
    inputs, context = rehearsed
    chosen = chosen_inputs(context, inputs)
    tops = read_all_tops(context)
    assert tops["query_id"].n_unique() == N_REHEARSAL_CUSTOMERS
    # A second call scores nothing: both files exist and are left alone.
    assert score_run(context, chosen, run_shards(context)) == 0
    # The same customers scored as ONE shard (all 35 in one wide table) get the same lists.
    one = copy_workspace(inputs, tmp_path / "one_shard")
    paths = one.paths
    parts = [
        pl.read_parquet(part)
        for shard in range(2)
        for part in sorted(paths.kaggle_features_dir(RUN_REHEARSAL, shard).glob("part-*.parquet"))
    ]
    whole = pl.concat(parts).sort("query_id", maintain_order=True)
    for shard in range(2):
        paths.kaggle_deepfm_top_file(RUN_REHEARSAL, shard).unlink()
    shutil.rmtree(paths.kaggle_features_dir(RUN_REHEARSAL, 0))
    write_wide(paths.kaggle_features_dir(RUN_REHEARSAL, 0), "T4", whole)
    one_context = kaggle_context(one, RUN_REHEARSAL)
    transform, model = load_run_model(one_context, chosen_inputs(one_context, one))
    n_rows = score_shard(one_context, chosen_inputs(one_context, one), transform, model, 0)
    assert n_rows == whole.height
    one_top = pl.read_parquet(paths.kaggle_deepfm_top_file(RUN_REHEARSAL, 0))
    assert _lists(one_top) == _lists(tops)


# ---------------------------------------------------------------- refusals


def test_a_sample_file_that_is_not_the_recorded_one_is_refused(
    built: TrialInputs, tmp_path: Path
) -> None:
    inputs = copy_workspace(built, tmp_path / "copy")
    context = kaggle_context(inputs, RUN_REHEARSAL)
    sample = inputs.paths.ranker_sample_file("full", "T2", RATIO)
    pl.read_parquet(sample).head(50).write_parquet(sample)
    with pytest.raises(ValueError, match="sha256"):
        train_run_model(context, chosen_inputs(context, inputs), SCRIPT)
    assert not inputs.paths.kaggle_deepfm_model_dir(RUN_REHEARSAL).exists()
    # The read was recorded before the file was opened, also for this failed attempt.
    assert len(_access_rows(inputs)) == 1


def test_a_model_is_never_replaced_and_scoring_needs_a_finished_training(
    rehearsed: tuple[TrialInputs, DeepFMKaggleContext], built: TrialInputs
) -> None:
    inputs, context = rehearsed
    with pytest.raises(FileExistsError, match="never replaced"):
        train_run_model(context, chosen_inputs(context, inputs), SCRIPT)
    untrained = kaggle_context(built, RUN_REHEARSAL)
    with pytest.raises(FileNotFoundError, match="not trained yet"):
        load_run_model(untrained, chosen_inputs(untrained, built))


def test_the_submission_run_trains_on_t1_to_t4_and_scores_every_customer(
    built: TrialInputs, tmp_path: Path
) -> None:
    inputs = copy_workspace(built, tmp_path / "copy")
    context = kaggle_context(inputs, RUN_SUBMISSION)
    chosen = chosen_inputs(context, inputs)
    summary = train_run_model(context, chosen, SCRIPT)
    assert summary.train_folds == ("T1", "T2", "T3", "T4")
    model_dir = inputs.paths.kaggle_deepfm_model_dir(RUN_SUBMISSION)
    saved = json.loads((model_dir / TRAINING_FILE).read_text())
    assert saved["mean_loss"] == summary.mean_loss and list(saved["sample_sha256"]) == [
        "T1",
        "T2",
        "T3",
        "T4",
    ]
    assert [(row["run"], row["folds"]) for row in _access_rows(inputs)] == [
        ("submission", "T1 T2 T3 T4")
    ]
    assert score_run(context, chosen, run_shards(context)) > 0
    tops = read_all_tops(context)
    assert tops["query_id"].n_unique() == N_SUBMISSION_CUSTOMERS
    assert tops.group_by("query_id").len()["len"].to_list() == [12] * N_SUBMISSION_CUSTOMERS
    # Nothing of the LightGBM submission was written or replaced.
    assert not inputs.paths.kaggle_top_file(RUN_SUBMISSION, 0).exists()
    assert not inputs.paths.kaggle_model_file(RUN_SUBMISSION).exists()
