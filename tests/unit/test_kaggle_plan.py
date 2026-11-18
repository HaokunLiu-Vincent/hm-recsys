"""Tests for kaggle.plan: the predicted week, the customers, query ids and shards."""

import dataclasses
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hmrec.config import load_kaggle_config, load_queries_config
from hmrec.data.folds import Fold, load_fold_manifest
from hmrec.data.queries import QUERIES_SCHEMA
from hmrec.kaggle.plan import (
    RUN_REHEARSAL,
    RUN_SUBMISSION,
    build_queries,
    check_query_id_range,
    fine_tune_fold,
    number_of_shards,
    ranker_training_folds,
    read_queries,
    rehearsal_customers,
    shard_queries,
    submission_week,
    target_week,
    write_queries,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = load_kaggle_config(REPO / "configs" / "kaggle_submission.yaml")
MANIFEST = load_fold_manifest(REPO / "configs" / "fold_manifest.yaml")
OFFSET = load_queries_config(REPO / "configs" / "queries.yaml").query_id_fold_offset
WEEK = Fold("S", "S", date(2020, 9, 23), date(2020, 9, 29))


def test_submission_week_follows_the_last_fold_and_is_not_a_fold() -> None:
    week = submission_week(CONFIG, MANIFEST)
    # T4 ends 2020-09-22, the last day of the data; the week is the 7 days after it.
    assert week == WEEK and week.label_days == 7
    assert week.name not in MANIFEST.names and week.phase not in ("D", "F", "V", "T")
    with pytest.raises(ValueError, match="which is a fold name"):
        submission_week(dataclasses.replace(CONFIG, week_name="T4"), MANIFEST)
    with pytest.raises(ValueError, match="expected the day after"):
        submission_week(dataclasses.replace(CONFIG, week_anchor=date(2020, 9, 24)), MANIFEST)
    with pytest.raises(ValueError, match="has 6 days"):
        submission_week(dataclasses.replace(CONFIG, week_label_end=date(2020, 9, 28)), MANIFEST)


def test_target_week_of_each_run() -> None:
    assert target_week(RUN_SUBMISSION, CONFIG, MANIFEST) == WEEK
    assert target_week(RUN_REHEARSAL, CONFIG, MANIFEST) == MANIFEST.get("T4")
    not_last = dataclasses.replace(
        CONFIG, rehearsal=dataclasses.replace(CONFIG.rehearsal, target_fold="T3")
    )
    with pytest.raises(ValueError, match="must predict the last fold T4"):
        target_week(RUN_REHEARSAL, not_last, MANIFEST)
    with pytest.raises(ValueError, match="run must be one of"):
        target_week("formal", CONFIG, MANIFEST)


def test_training_folds_and_fine_tune_week() -> None:
    def names(run: str) -> list[str]:
        return [fold.name for fold in ranker_training_folds(run, CONFIG, MANIFEST, 4)]

    # The 4 folds right before the predicted week.
    assert names(RUN_SUBMISSION) == ["T1", "T2", "T3", "T4"]
    assert names(RUN_REHEARSAL) == ["V4", "T1", "T2", "T3"]
    # A window that reaches into the V folds is refused for the submission.
    with pytest.raises(ValueError, match="trains on T folds only"):
        ranker_training_folds(RUN_SUBMISSION, CONFIG, MANIFEST, 5)
    # R6: one more weekly fine-tune, on T4, for the submission only.
    assert fine_tune_fold(RUN_SUBMISSION, MANIFEST) == MANIFEST.get("T4")
    assert fine_tune_fold(RUN_REHEARSAL, MANIFEST) is None


def test_rehearsal_customers_are_the_fold_plus_a_seeded_draw() -> None:
    everyone = np.arange(100, dtype=np.int32)
    fold = np.array([3, 50, 7], dtype=np.int32)
    drawn = rehearsal_customers(fold, everyone, 10, seed=5)
    assert drawn.size == 13 and drawn.dtype == np.int32
    assert np.all(np.diff(drawn) > 0) and set(fold.tolist()) <= set(drawn.tolist())
    # The same seed gives the same customers; another seed gives others.
    assert np.array_equal(drawn, rehearsal_customers(fold[::-1], everyone[::-1], 10, seed=5))
    assert not np.array_equal(drawn, rehearsal_customers(fold, everyone, 10, seed=6))
    with pytest.raises(ValueError, match="cannot draw 98"):
        rehearsal_customers(fold, everyone, 98, seed=5)


def test_queries_are_numbered_by_customer_from_the_base() -> None:
    queries = build_queries(np.array([9, 2, 5], dtype=np.int32), WEEK, 160_000_000)
    assert queries.schema == QUERIES_SCHEMA
    assert queries.rows() == [
        (160_000_000, "S", 2, date(2020, 9, 23)),
        (160_000_001, "S", 5, date(2020, 9, 23)),
        (160_000_002, "S", 9, date(2020, 9, 23)),
    ]
    with pytest.raises(ValueError, match="appears twice"):
        build_queries(np.array([1, 1], dtype=np.int32), WEEK, 0)


def test_query_ids_must_be_above_the_folds_and_fit_in_int32() -> None:
    # 16 folds x 10,000,000: the real base starts exactly where the folds' ids end.
    assert len(MANIFEST.folds) * OFFSET == CONFIG.query_id_base == 160_000_000
    check_query_id_range(CONFIG.query_id_base, 1_371_980, MANIFEST, OFFSET)
    with pytest.raises(ValueError, match="is below 160000000"):
        check_query_id_range(159_999_999, 10, MANIFEST, OFFSET)
    with pytest.raises(ValueError, match="do not fit in int32"):
        check_query_id_range(2_147_483_000, 1_000, MANIFEST, OFFSET)


def test_shards_partition_the_queries_in_order() -> None:
    queries = build_queries(np.arange(10, dtype=np.int32), WEEK, 100)
    assert number_of_shards(10, 4) == 3 and number_of_shards(8, 4) == 2
    # 1,371,980 customers in shards of 70,000: 19 full shards and one of 41,980.
    assert number_of_shards(1_371_980, CONFIG.shard_customers) == 20
    shards = [shard_queries(queries, shard, 4) for shard in range(3)]
    assert [shard.height for shard in shards] == [4, 4, 2]
    assert pl.concat(shards).equals(queries)
    with pytest.raises(ValueError, match="shard must be in 0..2"):
        shard_queries(queries, 3, 4)
    with pytest.raises(ValueError, match="shard_customers must be > 0"):
        number_of_shards(10, 0)
    with pytest.raises(ValueError, match="at least one query"):
        number_of_shards(0, 4)


def test_query_list_is_written_once_and_never_replaced(tmp_path: Path) -> None:
    path = tmp_path / "run" / "queries.parquet"
    with pytest.raises(FileNotFoundError, match="run the prepare step"):
        read_queries(path)
    queries = build_queries(np.arange(5, dtype=np.int32), WEEK, 100)
    write_queries(queries, path)
    written_at = path.stat().st_mtime_ns
    write_queries(queries, path)  # the same queries again: only compared
    assert path.stat().st_mtime_ns == written_at and read_queries(path).equals(queries)
    with pytest.raises(ValueError, match="holds other queries"):
        write_queries(build_queries(np.arange(6, dtype=np.int32), WEEK, 100), path)
    with pytest.raises(ValueError, match="queries schema"):
        write_queries(queries.select("query_id"), tmp_path / "other.parquet")
