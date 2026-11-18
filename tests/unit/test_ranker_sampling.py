"""Tests for hmrec.ranker.sampling: in-query negative sampling (proposal §9.1).

Hand example (labels: q1 -> {12}, q2 -> {20, 22}; q3 has no positive candidate):
    q1: candidates 10 11 12 13 14 15   1 positive, 5 negatives
    q2: candidates 20 21 22            2 positives, 1 negative
    q3: candidates 30 31               0 positives -> the query is dropped
With ratio 2:
    q1 keeps its positive and 2 x 1 = 2 of its 5 negatives     -> 3 rows
    q2 wants 2 x 2 = 4 negatives but has only 1: all are taken -> 3 rows
"""

from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
from ranker_example import LOCK, MANIFEST, WIDE_COLUMNS, wide_table, write_features, write_labels

from hmrec.config import RankerSamplingConfig
from hmrec.ranker.ablation import build_experiment
from hmrec.ranker.sampling import (
    LABEL,
    RowGroupWriter,
    SampleJob,
    load_training_labels,
    negative_orders,
    read_sample_record,
    record_file,
    sample_chunk,
    sample_fold,
    verify_sample_file,
)

SEED = 42
Q1 = [(1, article) for article in (10, 11, 12, 13, 14, 15)]
PAIRS = [*Q1, (2, 20), (2, 21), (2, 22), (3, 30), (3, 31)]
ROWS = wide_table(
    [{"query_id": q, "article_idx": a, "pop_rank": float(a)} for q, a in PAIRS]
).select("query_id", "article_idx", "pop_rank")
LABELS = pl.DataFrame(
    {"query_id": [1, 2, 2, 9], "article_idx": [12, 20, 22, 99]},
    schema={"query_id": pl.Int32, "article_idx": pl.Int32},
)


def _pairs(table: pl.DataFrame) -> list[tuple[int, int]]:
    return list(zip(table["query_id"].to_list(), table["article_idx"].to_list(), strict=True))


def test_hand_example_counts_and_labels() -> None:
    sample = sample_chunk(ROWS, LABELS, SEED, [2])[2]
    assert sample.columns == ["query_id", "article_idx", "label", "pop_rank"]
    assert sample["label"].dtype == pl.Int8
    per_query = sample.group_by("query_id").agg(pl.len(), pl.col(LABEL).sum()).sort("query_id")
    # q1: 1 positive + 2 negatives; q2: 2 positives + its only negative; q3 is gone.
    assert per_query.rows() == [(1, 3, 1), (2, 3, 2)]
    positives = _pairs(sample.filter(pl.col(LABEL) == 1))
    assert positives == [(1, 12), (2, 20), (2, 22)]  # every positive is kept
    assert _pairs(sample) == sorted(_pairs(sample))  # sorted by (query_id, article_idx)


def test_negatives_come_only_from_the_same_query() -> None:
    sample = sample_chunk(ROWS, LABELS, SEED, [2])[2]
    candidates = set(PAIRS)
    # Every sampled row is one of that query's own candidates, with its own features.
    assert set(_pairs(sample)) <= candidates
    assert sample["pop_rank"].to_list() == [float(a) for _, a in _pairs(sample)]


def test_enough_negatives_per_positive_or_all_of_them() -> None:
    for ratio, expected_q1 in ((1, 2), (2, 3), (4, 5), (5, 6), (80, 6)):
        sample = sample_chunk(ROWS, LABELS, SEED, [ratio])[ratio]
        assert sample.filter(pl.col("query_id") == 1).height == expected_q1


def test_same_seed_same_sample_and_another_seed_another_sample() -> None:
    first = sample_chunk(ROWS, LABELS, SEED, [2])[2]
    again = sample_chunk(ROWS, LABELS, SEED, [2])[2]
    assert first.equals(again)
    q1_negatives = set()
    for seed in range(20):
        sample = sample_chunk(ROWS, LABELS, seed, [2])[2]
        negatives = sample.filter((pl.col("query_id") == 1) & (pl.col(LABEL) == 0))
        q1_negatives.add(tuple(negatives["article_idx"].to_list()))
    assert len(q1_negatives) > 1


def test_smaller_ratio_is_a_subset_of_a_larger_ratio() -> None:
    samples = sample_chunk(ROWS, LABELS, SEED, [1, 2, 4])
    assert set(_pairs(samples[1])) < set(_pairs(samples[2])) < set(_pairs(samples[4]))


def test_a_query_sample_does_not_depend_on_chunking_or_row_order() -> None:
    together = sample_chunk(ROWS, LABELS, SEED, [2])[2]
    only_q1 = sample_chunk(ROWS.filter(pl.col("query_id") == 1), LABELS, SEED, [2])[2]
    only_q2 = sample_chunk(ROWS.filter(pl.col("query_id") == 2), LABELS, SEED, [2])[2]
    assert pl.concat([only_q1, only_q2]).equals(together)
    shuffled = ROWS.sample(fraction=1.0, shuffle=True, seed=7)
    assert sample_chunk(shuffled, LABELS, SEED, [2])[2].equals(together)


def test_negatives_are_drawn_uniformly() -> None:
    # 1 positive (article 0) and 10 negatives per query, ratio 1: each query keeps one
    # negative. Over 3,000 queries every negative should be kept about 300 times
    # (binomial sd = sqrt(3000 * 0.1 * 0.9) = 16.4; 5 sd = 82).
    n_queries = 3000
    rows = pl.DataFrame(
        {
            "query_id": np.repeat(np.arange(n_queries, dtype=np.int32), 11),
            "article_idx": np.tile(np.arange(11, dtype=np.int32), n_queries),
        }
    )
    labels = pl.DataFrame(
        {
            "query_id": np.arange(n_queries, dtype=np.int32),
            "article_idx": np.zeros(n_queries, dtype=np.int32),
        }
    )
    sample = sample_chunk(rows, labels, SEED, [1])[1]
    counts = (
        sample.filter(pl.col(LABEL) == 0).group_by("article_idx").agg(pl.len()).sort("article_idx")
    )
    assert counts["article_idx"].to_list() == list(range(1, 11))
    assert all(abs(count - 300) < 82 for count in counts["len"].to_list())


def test_negative_orders_values() -> None:
    query_ids = np.array([5, 5, 5, 5, 8, 8], dtype=np.int32)
    labels = np.array([0, 1, 0, 0, 1, 0], dtype=np.int8)
    order = negative_orders(query_ids, labels, SEED)
    assert order.dtype == np.int32
    assert order[1] == -1 and order[4] == -1  # positives
    assert sorted(order[[0, 2, 3]].tolist()) == [0, 1, 2]  # a permutation of the 3 negatives
    assert order[5] == 0
    with pytest.raises(ValueError, match="sorted by query_id"):
        negative_orders(np.array([2, 1], dtype=np.int32), np.array([0, 0], dtype=np.int8), SEED)
    assert negative_orders(np.zeros(0, np.int32), np.zeros(0, np.int8), SEED).size == 0


def test_no_positive_query_gives_an_empty_sample() -> None:
    rows = ROWS.filter(pl.col("query_id") == 3)
    assert sample_chunk(rows, LABELS, SEED, [2])[2].height == 0


# ---------------------------------------------------------------- labels: phase guard


def test_t_labels_are_refused_before_the_file_is_opened(tmp_path: Path) -> None:
    missing_file = tmp_path / "does_not_exist.parquet"
    # ValueError (the phase), not FileNotFoundError: the file is never touched.
    with pytest.raises(ValueError, match="labels of fold T1 .* must not be read"):
        load_training_labels(MANIFEST.get("T1"), missing_file, ("F", "V"))


def test_labels_of_another_fold_are_refused(tmp_path: Path) -> None:
    labels_file = tmp_path / "labels" / "F1.parquet"
    write_labels(labels_file, "F2", [(1, 12)])
    with pytest.raises(ValueError, match="holds folds"):
        load_training_labels(MANIFEST.get("F1"), labels_file, ("F", "V"))


# ---------------------------------------------------------------- writer


def test_row_group_writer_writes_fixed_row_groups(tmp_path: Path) -> None:
    table = pl.DataFrame({"x": list(range(15))}, schema={"x": pl.Int32})
    path = tmp_path / "out.parquet"
    writer = RowGroupWriter(path, 4)
    writer.append(table.head(7))
    writer.append(table.tail(8))
    writer.close()
    metadata = pq.ParquetFile(path).metadata
    sizes = [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)]
    assert sizes == [4, 4, 4, 3]  # only the last row group is smaller
    assert pl.read_parquet(path).equals(table)
    assert writer.n_rows == 15


# ---------------------------------------------------------------- one fold


def _job(tmp_path: Path, fold: str = "F1", ratios: tuple[int, ...] = (1, 2)) -> SampleJob:
    wide = wide_table([{"query_id": q, "article_idx": a, "pop_rank": float(a)} for q, a in PAIRS])
    features_dir = tmp_path / "features" / fold
    if not features_dir.exists():
        parts = [wide.filter(pl.col("query_id") == 1), wide.filter(pl.col("query_id") > 1)]
        write_features(features_dir, fold, parts)
        write_labels(tmp_path / "labels" / f"{fold}.parquet", fold, [(1, 12), (2, 20), (2, 22)])
    return SampleJob(
        fold=MANIFEST.get(fold),
        features_dir=features_dir,
        labels_file=tmp_path / "labels" / f"{fold}.parquet",
        experiment=build_experiment("full", LOCK),
        wide_columns=WIDE_COLUMNS,
        sample_files={ratio: tmp_path / "samples" / fold / f"r{ratio}.parquet" for ratio in ratios},
        allowed_phases=("F", "V"),
        git_commit="abc",
    )


SETTINGS = RankerSamplingConfig(seed=SEED, ratios=(1, 2), row_group_rows=4)


def test_sample_fold_writes_files_and_records(tmp_path: Path) -> None:
    job = _job(tmp_path)
    records = sample_fold(job, SETTINGS)
    record = records[2]
    assert (record.n_rows, record.n_queries, record.n_positives) == (6, 2, 3)
    assert record.positive_share == 0.5
    assert (record.rows_per_query_mean, record.rows_per_query_p50) == (3.0, 3.0)
    assert record.columns[:3] == ("query_id", "article_idx", "label")
    assert len(record.columns) == 3 + 71
    sample = pl.read_parquet(job.sample_files[2])
    assert sample.columns == list(record.columns)
    assert sample.height == 6 and sample["label"].sum() == 3
    assert records[1].n_rows == 5  # q1: 1 + 1, q2: 2 + 1
    assert verify_sample_file(job.sample_files[2]) == record
    assert read_sample_record(record_file(job.sample_files[1])).ratio == 1


def test_a_finished_sample_is_skipped_not_replaced(tmp_path: Path) -> None:
    job = _job(tmp_path)
    sample_fold(job, SETTINGS)
    before = {ratio: path.stat().st_mtime_ns for ratio, path in job.sample_files.items()}
    again = sample_fold(job, SETTINGS)
    assert {ratio: path.stat().st_mtime_ns for ratio, path in job.sample_files.items()} == before
    assert again[2].n_rows == 6


def test_a_half_written_sample_is_an_error(tmp_path: Path) -> None:
    job = _job(tmp_path)
    job.sample_files[2].parent.mkdir(parents=True)
    job.sample_files[2].write_bytes(b"half")
    with pytest.raises(ValueError, match="stopped half"):
        sample_fold(job, SETTINGS)


def test_a_changed_sample_file_is_detected(tmp_path: Path) -> None:
    job = _job(tmp_path)
    sample_fold(job, SETTINGS)
    pl.read_parquet(job.sample_files[2]).head(2).write_parquet(job.sample_files[2])
    with pytest.raises(ValueError, match="not the file that was written"):
        verify_sample_file(job.sample_files[2])


def test_sample_fold_refuses_a_t_fold_before_reading_anything(tmp_path: Path) -> None:
    job = _job(tmp_path)
    t_job = SampleJob(
        fold=MANIFEST.get("T1"),
        features_dir=tmp_path / "nothing",
        labels_file=tmp_path / "nothing.parquet",
        experiment=job.experiment,
        wide_columns=WIDE_COLUMNS,
        sample_files={2: tmp_path / "samples" / "T1" / "r2.parquet"},
        allowed_phases=("F", "V"),
        git_commit="abc",
    )
    with pytest.raises(ValueError, match="labels of fold T1"):
        sample_fold(t_job, SETTINGS)
    assert not (tmp_path / "samples" / "T1").exists()
