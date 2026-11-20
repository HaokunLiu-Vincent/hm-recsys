"""Tests for recall.generate: one channel of one post-lock fold, with its side files and checks.

R1-R5 use the REAL lock files (R5 with a smaller model, see _small_wals_lock) on the
hand examples of the channel tests. R6 uses the tiny model and the tiny chain data of
test_twotower_chain (anchor day 40 = "F1", anchor day 47 = "F2"); the function that
loads the real locked chain is replaced by one that builds the tiny chain.

R1 example (test_r1_popularity; anchor day 100, lock: A1, tau 3, quota 100):
    S(18_24, i): 10, 11, 12    S(25_34, i): 12, 13    S(UNKNOWN, i): 14 (not in A(a))
    q0 (18_24):   own 10, 11, 12, then fill-in 13        -> 3 non-fallback rows
    q1 (25_34):   own 12, 13, then fill-in 10, 11        -> 2 non-fallback rows
    q2 (UNKNOWN): fill-in 12, 10, 11, 13                 -> 0 non-fallback rows
    12 candidate rows; 5 are checked against the table; the table has 6 rows.
"""

import dataclasses
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from test_r1_popularity import ACTIVE_POOL, CUSTOMERS, ROWS, W1, W3
from test_r1_popularity import ANCHOR_DAY as R1_ANCHOR_DAY
from test_r1_popularity import QUERIES as R1_QUERIES
from test_r5_wals import ANCHOR_DAY as R5_ANCHOR_DAY
from test_r5_wals import RANDOM_ACTIVE, RANDOM_QUERIES, RANDOM_ROWS
from test_twotower_chain import ALARM, LINES, MEMORY
from test_twotower_chain import CONFIG as CHAIN_CONFIG
from test_twotower_infer import CHUNKS, _resources

from conftest import FIXTURE_DAY_ZERO, make_queries, make_recall_inputs
from hmrec.config import (
    BaseConfig,
    CandidateRunConfig,
    PreprocessConfig,
    load_base_config,
    load_preprocess_config,
)
from hmrec.data.day_index import day_index_to_date
from hmrec.data.folds import Fold
from hmrec.locks import check_file_sha256, file_sha256
from hmrec.paths import ProjectPaths, build_paths, get_paths
from hmrec.recall import generate as generate_module
from hmrec.recall.base import RecallInputs
from hmrec.recall.chain_records import read_chain_records
from hmrec.recall.generate import (
    FoldJob,
    check_fine_tune_loss,
    check_recorded_fine_tune_loss,
    check_scores_match_bucket_table,
    check_scores_match_vectors,
    generate_fold_channel,
    generate_statistical_channel,
    generate_twotower_fold,
    predict_twotower_fold,
    side_files,
    write_bucket_scores,
)
from hmrec.recall.locked import LockedRecall, ModelFile, load_locked_recall
from hmrec.recall.twotower.chain import TwoTowerChainChannel
from hmrec.recall.twotower.infer import (
    TwoTowerChannel,
    add_other_catalog_articles,
    query_vectors,
)
from hmrec.recall.twotower.train import load_checkpoint
from hmrec.run_records import FINE_TUNE_ONLY, GENERATED, SKIPPED, RunStamp
from hmrec.vectors import UserItemVectors, load_vectors

REPO = Path(__file__).resolve().parents[2]
BASE = load_base_config(REPO / "configs" / "base.yaml")
PREPROCESS = load_preprocess_config(REPO / "configs" / "preprocess.yaml")
LOCKED = load_locked_recall(get_paths())
STAMP = RunStamp("abc123", False)
FOLD_NAMES = ("F1", "F2")
R1, R2, R5, R6 = "r1_popularity", "r2_repurchase", "r5_wals", "r6_twotower"


def _fold(name: str, anchor_day: int) -> Fold:
    anchor = day_index_to_date(anchor_day, FIXTURE_DAY_ZERO)
    return Fold(name=name, phase="F", anchor=anchor, label_end=anchor + timedelta(days=6))


def _job(
    tmp_path: Path,
    fold: Fold,
    queries: pl.DataFrame,
    inputs: RecallInputs,
    locked: LockedRecall = LOCKED,
) -> FoldJob:
    """A job whose artifacts and results go under tmp_path."""
    paths = build_paths(BASE, tmp_path)
    return FoldJob(paths, BASE, PREPROCESS, locked, fold, FOLD_NAMES, queries, inputs)


def _r1_job(tmp_path: Path, rows: list[tuple[int, int, int]] = ROWS) -> FoldJob:
    inputs = make_recall_inputs(rows, R1_ANCHOR_DAY, customers=CUSTOMERS, active_pool=ACTIVE_POOL)
    return _job(tmp_path, _fold("F1", R1_ANCHOR_DAY), R1_QUERIES, inputs)


# ---------------------------------------------------------------- R1: candidates + bucket table


def test_r1_writes_candidates_and_the_full_bucket_score_table(tmp_path: Path) -> None:
    job = _r1_job(tmp_path)
    result = generate_statistical_channel(job, R1)
    assert (result.status, result.n_rows, result.checked_rows) == (GENERATED, 12, 5)
    candidates = pl.read_parquet(job.paths.candidates_file("F1", R1))
    assert candidates.height == 12
    table = pl.read_parquet(job.paths.r1_bucket_scores_file("F1"))
    # Every (bucket, article) bought in the window: NOT cut to a quota and NOT limited
    # to A(a) (article 14 is not active and is in no candidate list, but it is here).
    assert table.select("age_bucket", "article_idx").rows() == [
        ("AGE_18_24", 10),
        ("AGE_18_24", 11),
        ("AGE_18_24", 12),
        ("AGE_25_34", 12),
        ("AGE_25_34", 13),
        ("AGE_UNKNOWN", 14),
    ]
    assert table.schema == pl.Schema(
        {"age_bucket": pl.String, "article_idx": pl.Int32, "score": pl.Float32}
    )
    assert table["score"].to_list() == pytest.approx(
        [W1 + W3, W1, math.exp(-2), 2 * W1, math.exp(-2 / 3), W1], rel=1e-6
    )
    # q0's own rows carry exactly the table's float32 values of bucket 18_24.
    q0_own = candidates.filter((pl.col("query_id") == 0) & (pl.col("pop_is_fallback") == 0))
    assert q0_own.sort("article_idx")["score"].to_list() == table["score"].to_list()[:3]


def test_existing_candidates_are_skipped_and_never_replaced(tmp_path: Path) -> None:
    job = _r1_job(tmp_path)
    generate_statistical_channel(job, R1)
    files = [job.paths.candidates_file("F1", R1), job.paths.r1_bucket_scores_file("F1")]
    written_at = [path.stat().st_mtime_ns for path in files]
    again = generate_statistical_channel(job, R1)
    assert (again.status, again.n_rows, again.generate_seconds) == (SKIPPED, 12, 0.0)
    assert [path.stat().st_mtime_ns for path in files] == written_at


def test_side_file_without_candidates_is_a_half_finished_step(tmp_path: Path) -> None:
    job = _r1_job(tmp_path)
    table_file = job.paths.r1_bucket_scores_file("F1")
    table_file.parent.mkdir(parents=True)
    table_file.write_bytes(b"left over from a crashed run")
    with pytest.raises(FileExistsError, match="half-finished step"):
        generate_statistical_channel(job, R1)
    assert table_file.read_bytes() == b"left over from a crashed run"  # nothing was repaired


def test_r1_ignores_purchases_on_and_after_the_anchor(tmp_path: Path) -> None:
    # Poison on the anchor day (100) and later: would change bucket 18_24's ranking,
    # add article 13 to it, and add rows to the table if it leaked.
    poison = [(0, 13, 100)] * 5 + [(3, 13, 101)] * 5 + [(1, 11, 200)]
    clean = _r1_job(tmp_path / "clean")
    poisoned = _r1_job(tmp_path / "poisoned", ROWS + poison)
    generate_statistical_channel(clean, R1)
    generate_statistical_channel(poisoned, R1)
    for path_of in (
        lambda job: job.paths.candidates_file("F1", R1),
        lambda job: job.paths.r1_bucket_scores_file("F1"),
    ):
        assert pl.read_parquet(path_of(clean)).equals(pl.read_parquet(path_of(poisoned)))


def test_r2_has_no_side_files_and_no_consistency_check(tmp_path: Path) -> None:
    # Customer 0 bought 30 and 40, customer 1 bought 31; R2 returns what they bought.
    inputs = make_recall_inputs([(0, 30, 370), (0, 40, 399), (1, 31, 395)], 400)
    job = _job(tmp_path, _fold("F1", 400), make_queries([(1, 0), (2, 1)]), inputs)
    result = generate_statistical_channel(job, R2)
    assert (result.status, result.n_rows, result.checked_rows) == (GENERATED, 3, 0)
    assert side_files(job.paths, "F1", R2) == []
    written = pl.read_parquet(job.paths.candidates_file("F1", R2))
    assert written.sort("query_id", "article_idx").select("query_id", "article_idx").rows() == [
        (1, 30),
        (1, 40),
        (2, 31),
    ]


# ---------------------------------------------------------------- R5: candidates + vectors


def _small_wals_lock() -> LockedRecall:
    """The real lock with a small WALS model (4 factors, 5 iterations) for the tiny data."""
    channels = dict(LOCKED.channels.channels)
    r5 = channels[R5]
    channels[R5] = dataclasses.replace(r5, config={**r5.config, "factors": 4, "iterations": 5})
    return dataclasses.replace(
        LOCKED, channels=dataclasses.replace(LOCKED.channels, channels=channels)
    )


def test_r5_saves_query_users_and_all_items_and_scores_match_the_saved_vectors(
    tmp_path: Path,
) -> None:
    inputs = make_recall_inputs(RANDOM_ROWS, R5_ANCHOR_DAY, active_pool=RANDOM_ACTIVE)
    job = _job(tmp_path, _fold("F1", R5_ANCHOR_DAY), RANDOM_QUERIES, inputs, _small_wals_lock())
    result = generate_statistical_channel(job, R5)
    candidates = pl.read_parquet(job.paths.candidates_file("F1", R5))
    # Every candidate score was compared with the dot product of the vectors on disk.
    assert (result.status, result.checked_rows) == (GENERATED, candidates.height)
    assert candidates.height > 0
    saved = load_vectors(job.paths.wals_vectors_dir("F1"), "wals", "F1")
    # Users: only query customers that have a vector (40 and 41 never bought anything).
    query_customers = set(RANDOM_QUERIES["customer_idx"].to_list())
    assert set(saved.user_customer_idx.tolist()) == query_customers - {40, 41}
    # Articles: every article of the fit window, also those outside the active pool.
    bought = {article for _, article, _ in RANDOM_ROWS}
    assert set(saved.item_article_idx.tolist()) == bought
    assert not bought <= set(RANDOM_ACTIVE)


# ---------------------------------------------------------------- the two consistency checks


def _vectors() -> UserItemVectors:
    return UserItemVectors(
        user_customer_idx=np.array([0, 1], dtype=np.int32),
        user_vectors=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        item_article_idx=np.array([10, 11], dtype=np.int32),
        item_vectors=np.array([[0.5, 0.0], [0.0, 0.25]], dtype=np.float32),
    )


def _scored(rows: list[tuple[int, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "query_id": [row[0] for row in rows],
            "article_idx": [row[1] for row in rows],
            "score": [row[2] for row in rows],
        },
        schema={"query_id": pl.Int32, "article_idx": pl.Int32, "score": pl.Float32},
    )


def test_scores_must_equal_the_dot_products_of_the_vectors() -> None:
    queries = make_queries([(1, 0), (2, 1)])
    # u0 . v10 = 0.5, u0 . v11 = 0.5, u1 . v11 = 1.0
    good = _scored([(1, 10, 0.5), (1, 11, 0.5), (2, 11, 1.0)])
    assert check_scores_match_vectors(good, queries, _vectors(), 2, "test") == 3
    off_by_one_bit = _scored([(1, 10, float(np.nextafter(np.float32(0.5), np.float32(1.0))))])
    with pytest.raises(ValueError, match="1 of 1 candidate scores differ"):
        check_scores_match_vectors(off_by_one_bit, queries, _vectors(), 2, "test")
    # Article 12 has no saved vector: its dot product is NaN, which also fails.
    without_vector = _scored([(1, 12, 0.5)])
    with pytest.raises(ValueError, match="differ from the dot"):
        check_scores_match_vectors(without_vector, queries, _vectors(), 2, "test")


def test_non_fallback_scores_must_equal_the_bucket_table() -> None:
    queries = make_queries([(1, 0)])
    customers = pl.DataFrame(
        {"customer_idx": [0], "age_bucket": ["AGE_18_24"]},
        schema={"customer_idx": pl.Int32, "age_bucket": pl.String},
    )
    table = pl.DataFrame(
        {"age_bucket": ["AGE_18_24", "AGE_18_24"], "article_idx": [10, 11], "score": [1.5, 0.5]},
        schema={"age_bucket": pl.String, "article_idx": pl.Int32, "score": pl.Float32},
    )

    def candidates(scores: list[float], fallback: list[int], articles: list[int]) -> pl.DataFrame:
        return _scored(
            [(1, article, score) for article, score in zip(articles, scores, strict=True)]
        ).with_columns(pl.Series("pop_is_fallback", fallback, dtype=pl.Int8))

    # The fallback row (global score 9.0, article not in the table) is not checked.
    good = candidates([1.5, 0.5, 9.0], [0, 0, 1], [10, 11, 99])
    assert check_scores_match_bucket_table(good, queries, customers, table, "test") == 2
    wrong_score = candidates([1.5, 0.75], [0, 0], [10, 11])
    with pytest.raises(ValueError, match="1 of 2 non-fallback candidates differ"):
        check_scores_match_bucket_table(wrong_score, queries, customers, table, "test")
    not_in_table = candidates([1.5], [0], [99])
    with pytest.raises(ValueError, match="missing in"):
        check_scores_match_bucket_table(not_in_table, queries, customers, table, "test")


def test_bucket_score_table_is_validated_and_never_replaced(tmp_path: Path) -> None:
    good = pl.DataFrame(
        {"age_bucket": ["A", "A"], "article_idx": [1, 2], "score": [1.0, 2.0]},
        schema={"age_bucket": pl.String, "article_idx": pl.Int32, "score": pl.Float32},
    )
    path = tmp_path / "F1" / "r1_bucket_scores.parquet"
    with pytest.raises(ValueError, match="schema"):
        write_bucket_scores(good.with_columns(pl.col("score").cast(pl.Float64)), path)
    with pytest.raises(ValueError, match="repeat"):
        write_bucket_scores(pl.concat([good, good]), path)
    with pytest.raises(ValueError, match="null, NaN or infinite"):
        write_bucket_scores(
            good.with_columns(pl.lit(float("nan"), dtype=pl.Float32).alias("score")), path
        )
    assert not path.exists()
    write_bucket_scores(good, path)
    with pytest.raises(FileExistsError, match="never replaced"):
        write_bucket_scores(good, path)


# ---------------------------------------------------------------- R6: predict, save, then fine-tune

CHAIN_POOL = [0, 1, 3, 4]  # articles 2 and 5 are outside A(a)
F1 = _fold("F1", 40)  # week [40, 47): the "D1 week" of test_twotower_chain
F2 = _fold("F2", 47)  # week [47, 54)


def _chain_job(tmp_path: Path, fold: Fold, lines: list[tuple[int, int, int]] = LINES) -> FoldJob:
    anchor_day = (fold.anchor - FIXTURE_DAY_ZERO).days
    queries = make_queries([(1, 0), (2, 1), (3, 2)], fold=fold.name)
    return _job(
        tmp_path, fold, queries, make_recall_inputs(lines, anchor_day, active_pool=CHAIN_POOL)
    )


@pytest.fixture
def starts(monkeypatch: pytest.MonkeyPatch) -> list[ModelFile]:
    """Replace the loader of the real locked chain by one that builds the tiny chain.

    Like the real loader it checks the start model's sha256 before loading it. The
    locked f1_start file does not exist under tmp_path: there the freshly built tiny
    model stands in for it. Returns the list of start models it was asked for.
    """
    asked: list[ModelFile] = []

    def load_tiny_chain(
        locked: LockedRecall,
        base: BaseConfig,
        paths: ProjectPaths,
        preprocess: PreprocessConfig,
        start: ModelFile,
    ) -> TwoTowerChainChannel:
        asked.append(start)
        resources = _resources()
        if start.path.exists():
            check_file_sha256(start.path, start.sha256)
            load_checkpoint(resources.model, start.path)
        return TwoTowerChainChannel(
            CHAIN_CONFIG,
            resources,
            CHUNKS,
            paths.twotower_locked_chain_checkpoint_file,
            ALARM,
            MEMORY,
            {"model_identity": {}},
        )

    monkeypatch.setattr(generate_module, "load_locked_twotower_chain", load_tiny_chain)
    return asked


def _catalog_vectors(job: FoldJob, checkpoint: Path | None) -> UserItemVectors:
    """Vectors of the job's fold with the start weights, or with a saved model's weights."""
    resources = _resources()
    if checkpoint is not None:
        load_checkpoint(resources.model, checkpoint)
    pool_vectors = query_vectors(CHAIN_CONFIG, resources, job.queries, job.inputs, 2)
    return add_other_catalog_articles(pool_vectors, resources, 2)


def test_r6_saves_vectors_and_candidates_before_it_fine_tunes(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    job = _chain_job(tmp_path, F1)
    result = generate_twotower_fold(job, STAMP)
    model_file = job.paths.twotower_locked_chain_checkpoint_file("F1")
    assert model_file.exists()  # the fold WAS fine-tuned ...
    saved = load_vectors(job.paths.twotower_vectors_dir("F1"), "twotower", "F1")
    before = _catalog_vectors(job, checkpoint=None)
    after = _catalog_vectors(job, checkpoint=model_file)
    # ... but the saved vectors are those of the model BEFORE the fine-tune,
    assert np.array_equal(saved.user_vectors, before.user_vectors)
    assert np.array_equal(saved.item_vectors, before.item_vectors)
    # and the fine-tune really changed the model (so the order matters).
    assert not np.array_equal(saved.item_vectors, after.item_vectors)
    # The candidates are also the start model's: the same as the frozen D-phase channel.
    frozen = TwoTowerChannel(CHAIN_CONFIG, _resources(), CHUNKS)
    candidates = pl.read_parquet(job.paths.candidates_file("F1", R6))
    assert candidates.equals(frozen.generate(job.queries, job.inputs))
    assert (result.status, result.n_rows, result.checked_rows) == (GENERATED, 6, 6)
    assert "fine-tune 3 new + 3 replay rows" in result.detail


def test_r6_vectors_cover_query_users_and_the_whole_catalog(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    job = _chain_job(tmp_path, F1)
    generate_twotower_fold(job, STAMP)
    saved = load_vectors(job.paths.twotower_vectors_dir("F1"), "twotower", "F1")
    assert saved.user_customer_idx.tolist() == [0, 1, 2]  # the fold's query customers
    # All 6 catalog articles, although only 4 are in A(a) and can be R6 candidates.
    assert saved.item_article_idx.tolist() == [0, 1, 2, 3, 4, 5]
    candidates = pl.read_parquet(job.paths.candidates_file("F1", R6))
    assert set(candidates["article_idx"].to_list()) <= set(CHAIN_POOL)


def test_r6_chain_record_has_the_models_sha256_and_fine_tune_numbers(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    job = _chain_job(tmp_path, F1)
    generate_twotower_fold(job, STAMP)
    records = read_chain_records(job.paths.m6_twotower_chain_results_file)
    assert list(records) == ["F1"]
    record = records["F1"]
    model_file = job.paths.twotower_locked_chain_checkpoint_file("F1")
    assert record.sha256 == file_sha256(model_file)
    assert (record.start_model, record.model) == ("D4.pt", "F1.pt")
    assert record.start_sha256 == LOCKED.twotower.checkpoints["f1_start"].sha256
    # 3 new rows of non-alarm users in the week [40, 47), 1 : 1 replay, 1 alarm row.
    assert (record.n_new_rows, record.n_replay_rows, record.n_train_samples) == (3, 3, 6)
    assert (record.n_alarm_samples, record.epochs) == (1, CHAIN_CONFIG.e_update)
    assert math.isfinite(record.mean_train_loss) and math.isfinite(record.alarm_loss)
    assert (record.git_commit, record.git_dirty) == ("abc123", False)


def test_r6_next_fold_uses_the_recorded_model_and_the_last_fold_is_not_fine_tuned(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    generate_twotower_fold(_chain_job(tmp_path, F1), STAMP)
    job = _chain_job(tmp_path, F2)  # F2 is the last fold of FOLD_NAMES
    result = generate_twotower_fold(job, STAMP)
    f1_model = job.paths.twotower_locked_chain_checkpoint_file("F1")
    # F1 started from the locked f1_start, F2 from the model saved after F1.
    assert [start.path.name for start in starts] == ["D4.pt", "F1.pt"]
    assert starts[1] == ModelFile(f1_model, file_sha256(f1_model))
    after_f1 = _resources()
    load_checkpoint(after_f1.model, f1_model)
    expected = TwoTowerChannel(CHAIN_CONFIG, after_f1, CHUNKS).generate(job.queries, job.inputs)
    assert pl.read_parquet(job.paths.candidates_file("F2", R6)).equals(expected)
    # The last fold is only predicted: no model, no row.
    assert not job.paths.twotower_locked_chain_checkpoint_file("F2").exists()
    assert list(read_chain_records(job.paths.m6_twotower_chain_results_file)) == ["F1"]
    assert (result.status, result.detail) == (GENERATED, "last fold: predicted only, no fine-tune")


def test_r6_folds_must_be_processed_in_time_order(tmp_path: Path, starts: list[ModelFile]) -> None:
    with pytest.raises(ValueError, match="F1 has no row"):
        generate_twotower_fold(_chain_job(tmp_path, F2), STAMP)
    assert starts == []  # refused before any model was loaded
    assert not _chain_job(tmp_path, F2).paths.candidates_file("F2", R6).exists()


def test_r6_refuses_a_previous_model_whose_sha256_changed(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    job = _chain_job(tmp_path, F1)
    generate_twotower_fold(job, STAMP)
    f1_model = job.paths.twotower_locked_chain_checkpoint_file("F1")
    # The model file is replaced after its sha256 was recorded (here: other weights).
    other = _resources()
    with torch.no_grad():
        other.model.item_tower.article_embedding.weight[:6].add_(1.0)
    f1_model.unlink()
    torch.save(
        {"state_dict": other.model.state_dict(), "metadata": {"model_identity": {}}}, f1_model
    )
    with pytest.raises(ValueError, match="This is not the locked file"):
        generate_twotower_fold(_chain_job(tmp_path, F2), STAMP)
    assert not job.paths.candidates_file("F2", R6).exists()


def test_r6_finished_fold_is_skipped(tmp_path: Path, starts: list[ModelFile]) -> None:
    job = _chain_job(tmp_path, F1)
    generate_twotower_fold(job, STAMP)
    files = [
        job.paths.candidates_file("F1", R6),
        job.paths.twotower_locked_chain_checkpoint_file("F1"),
        job.paths.twotower_vectors_dir("F1") / "item_vectors.npy",
    ]
    written_at = [path.stat().st_mtime_ns for path in files]
    again = generate_twotower_fold(job, STAMP)
    assert again.status == SKIPPED
    assert len(starts) == 1  # no model was loaded for the skipped run
    assert [path.stat().st_mtime_ns for path in files] == written_at
    assert list(read_chain_records(job.paths.m6_twotower_chain_results_file)) == ["F1"]


def test_r6_does_only_the_missing_fine_tune_when_candidates_exist(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    # A run that stopped after the prediction: candidates and vectors exist, no model.
    job = _chain_job(tmp_path, F1)
    chain = generate_module.load_locked_twotower_chain(
        job.locked, BASE, job.paths, PREPROCESS, ModelFile(tmp_path / "none.pt", "0" * 64)
    )
    candidates_file = job.paths.candidates_file("F1", R6)
    predict_twotower_fold(
        chain, "F1", job.queries, job.inputs, candidates_file, job.paths.twotower_vectors_dir("F1")
    )
    written_at = candidates_file.stat().st_mtime_ns
    result = generate_twotower_fold(job, STAMP)
    assert result.status == FINE_TUNE_ONLY
    assert candidates_file.stat().st_mtime_ns == written_at  # the candidates were not touched
    assert job.paths.twotower_locked_chain_checkpoint_file("F1").exists()
    assert list(read_chain_records(job.paths.m6_twotower_chain_results_file)) == ["F1"]


def test_r6_vectors_without_candidates_are_a_half_finished_step(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    job = _chain_job(tmp_path, F1)
    metadata = job.paths.twotower_vectors_dir("F1") / "vectors.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="half-finished step"):
        generate_twotower_fold(job, STAMP)


def test_r6_prediction_ignores_purchases_on_and_after_the_anchor(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    # Extra purchases inside the fold's own week [40, 47) and later. They are
    # fine-tune data, so they may change the saved MODEL, but never the fold's
    # candidates or vectors, which are written before the fine-tune.
    # (Two in-week lines only: the tiny replay window has just 5 rows to draw from.)
    poison = [(0, 3, 40), (1, 4, 43), (0, 1, 47), (1, 3, 60)]
    clean = _chain_job(tmp_path / "clean", F1)
    poisoned = _chain_job(tmp_path / "poisoned", F1, LINES + poison)
    generate_twotower_fold(clean, STAMP)
    generate_twotower_fold(poisoned, STAMP)
    clean_candidates = pl.read_parquet(clean.paths.candidates_file("F1", R6))
    assert clean_candidates.equals(pl.read_parquet(poisoned.paths.candidates_file("F1", R6)))
    clean_vectors = load_vectors(clean.paths.twotower_vectors_dir("F1"), "twotower", "F1")
    poisoned_vectors = load_vectors(poisoned.paths.twotower_vectors_dir("F1"), "twotower", "F1")
    assert np.array_equal(clean_vectors.user_vectors, poisoned_vectors.user_vectors)
    assert np.array_equal(clean_vectors.item_vectors, poisoned_vectors.item_vectors)
    # The two in-week lines did reach the fine-tune: 2 more new rows. The lines on day
    # 47 and 60 are after the week [40, 47) and are not used at all.
    clean_record = read_chain_records(clean.paths.m6_twotower_chain_results_file)["F1"]
    poisoned_record = read_chain_records(poisoned.paths.m6_twotower_chain_results_file)["F1"]
    assert poisoned_record.n_new_rows == clean_record.n_new_rows + 2


# ---------------------------------------------------------------- loss check and entry


def test_fine_tune_loss_must_be_finite_and_below_the_limit(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    job = _chain_job(tmp_path, F1)
    result = generate_twotower_fold(job, STAMP)
    record = read_chain_records(job.paths.m6_twotower_chain_results_file)["F1"]
    check_fine_tune_loss(record, max_loss=1e9)  # no error
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="fine-tune training loss"):
            check_fine_tune_loss(dataclasses.replace(record, mean_train_loss=bad), 1e9)
    with pytest.raises(ValueError, match=r"fine-tune alarm loss is 7\.5"):
        check_fine_tune_loss(dataclasses.replace(record, alarm_loss=7.5), max_loss=7.0)
    # The script's check reads the recorded row: with an absurdly low limit it stops ...
    strict = CandidateRunConfig(4, 0.10, max_fine_tune_loss=1e-6)
    with pytest.raises(ValueError, match="the chain stops for a human decision"):
        check_recorded_fine_tune_loss(job, result, strict)
    # ... but not for a skipped run, which trained nothing.
    skipped = generate_twotower_fold(job, STAMP)
    check_recorded_fine_tune_loss(job, skipped, strict)


def test_entry_refuses_a_fold_that_is_not_post_lock(tmp_path: Path) -> None:
    d_fold = dataclasses.replace(_fold("D1", R1_ANCHOR_DAY), phase="D")
    job = dataclasses.replace(_r1_job(tmp_path), fold=d_fold)
    with pytest.raises(ValueError, match="D1 is not a post-lock fold"):
        generate_fold_channel(job, R1, STAMP)
    # A post-lock fold goes to the channel's generator.
    assert generate_fold_channel(_r1_job(tmp_path), R1, STAMP).status == GENERATED
