"""Tests for kaggle.features: shared tables once, shards of customers, the same wide table.

The fold is the one of feature_example.py (3 queries: q1 customer 0, q2 customer 1,
q3 customer 2) with the candidates of test_feature_build.py (9 distinct pairs).
The property under test: cutting the queries into shards gives, row for row and
bit for bit, the wide table of all queries in one shard, which in turn is the
table M6's build_fold_features writes.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from feature_example import ARTICLES, CONFIG, CUSTOMERS, LINES, POISON, QUERIES, make_fold
from test_feature_build import make_candidates
from test_feature_dense import BUCKET_SCORES, IMAGE, IMAGE_MASK, TEXT, TEXT_MASK, TWOTOWER, WALS

from hmrec.config import load_base_config
from hmrec.data.histories import recent_purchase_histories
from hmrec.feature_files import verify_feature_files
from hmrec.features.base import FoldData, HistoryArrays
from hmrec.features.build import FoldInputs, build_fold_features
from hmrec.features.categories import build_category_codes
from hmrec.features.dense import DenseSources, similarity_vectors
from hmrec.kaggle.features import (
    QUERY_STATS_SCHEMA,
    ShardFiles,
    ShardInputs,
    SharedTables,
    compute_shared_tables,
    count_distinct_candidates,
    query_stats,
    shard_is_finished,
    shard_tables,
    write_shard_features,
)
from hmrec.paths import build_paths
from hmrec.run_records import RunStamp

REPO = Path(__file__).resolve().parents[2]
BASE = load_base_config(REPO / "configs" / "base.yaml")
KEYS = ["query_id", "article_idx"]
DETAILS = {"run": "rehearsal", "shard": 0, "git_commit": "abc123", "git_dirty": False}


def _history(data: FoldData) -> HistoryArrays:
    """The last purchases of the customers of data.queries (the function R6 uses)."""
    customers = data.query_customers
    history, mask = recent_purchase_histories(
        data.transactions,
        customers,
        data.anchor,
        data.day_zero,
        CONFIG.history_length,
        padding_idx=ARTICLES.height,
    )
    return HistoryArrays(customers, history, mask)


def _shard_inputs(query_ids: list[int], lines: list = LINES) -> ShardInputs:
    """The inputs of a shard that holds the given queries (everything else is the fold's)."""
    queries = QUERIES.filter(pl.col("query_id").is_in(query_ids))
    data = dataclasses.replace(make_fold(lines), queries=queries)
    history = _history(data)
    arrays = (history.customer_idx, history.history, history.history_mask)
    in_shard = pl.col("query_id").is_in(query_ids)
    dense = DenseSources(
        twotower=TWOTOWER,
        wals=WALS,
        bucket_scores=BUCKET_SCORES,
        image=similarity_vectors(*arrays, IMAGE, IMAGE_MASK, 2),
        text=similarity_vectors(*arrays, TEXT, TEXT_MASK, 2),
        history=history,
        history_similarity_top_k=5,
        pair_chunk_size=3,
        embedding_pair_chunk_size=2,
    )
    candidates = {name: table.filter(in_shard) for name, table in make_candidates().items()}
    return ShardInputs(data, history, candidates, dense)


def _shared(lines: list = LINES) -> SharedTables:
    data = make_fold(lines)
    return compute_shared_tables(data, CONFIG, build_category_codes(ARTICLES, CUSTOMERS))


def _build(tmp_path: Path, shards: list[list[int]], lines: list = LINES) -> pl.DataFrame:
    """Build every shard into its own folder and return all rows in (query, article) order."""
    shared = _shared(lines)
    tables = []
    for number, query_ids in enumerate(shards):
        files = ShardFiles(tmp_path / f"{number:03d}" / "features", tmp_path / f"{number}.parquet")
        files.stats_file.parent.mkdir(parents=True, exist_ok=True)
        write_shard_features(_shard_inputs(query_ids, lines), shared, CONFIG, 2, files, DETAILS)
        tables += [pl.read_parquet(part) for part in sorted(files.features_dir.glob("part-*"))]
    return pl.concat(tables).sort(KEYS)


def _same(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    """True if two wide tables are identical, NaN in the same places included."""
    if left.columns != right.columns or left.schema != right.schema:
        return False
    return all(
        np.array_equal(left[name].to_numpy(), right[name].to_numpy(), equal_nan=True)
        for name in left.columns
    )


@pytest.mark.parametrize("shards", [[[1], [2], [3]], [[1, 2], [3]], [[1], [2, 3]]])
def test_wide_table_does_not_depend_on_the_shards(tmp_path: Path, shards: list[list[int]]) -> None:
    one_shard = _build(tmp_path / "one", [[1, 2, 3]])
    sharded = _build(tmp_path / "many", shards)
    assert one_shard.height == 9 and _same(sharded, one_shard)


def test_one_shard_equals_the_table_m6_builds_for_the_fold(tmp_path: Path) -> None:
    one_shard = _build(tmp_path / "kaggle", [[1, 2, 3]])
    all_queries = _shard_inputs([1, 2, 3])
    inputs = FoldInputs(
        data=all_queries.data,
        history=all_queries.history,
        candidates=all_queries.candidates,
        dense=all_queries.dense,
        expected_rows=9,
    )
    paths = build_paths(BASE, tmp_path / "m6")
    build_fold_features(paths, inputs, CONFIG, 2, RunStamp("abc123", False))
    parts = sorted(paths.features_dir("F1").glob("part-*.parquet"))
    fold_table = pl.concat([pl.read_parquet(part) for part in parts]).sort(KEYS)
    assert _same(one_shard, fold_table)
    assert one_shard.columns == [*KEYS, *CONFIG.names] and len(CONFIG.names) == 71


def test_shared_tables_come_from_all_customers_not_from_the_shard(tmp_path: Path) -> None:
    shared = _shared()
    # One row per article of the catalog, whatever the queries are.
    assert shared.item["article_idx"].to_list() == [0, 1, 2, 3]
    data = _shard_inputs([3])  # customer 2: no purchase at all
    tables = shard_tables(
        data.data,
        CONFIG,
        shared,
        pl.DataFrame(
            {"customer_idx": [], "article_idx": []},
            schema={"customer_idx": pl.Int32, "article_idx": pl.Int32},
        ),
    )
    # The shard's own tables hold only its customers; the shared ones are passed through.
    assert tables.user["customer_idx"].to_list() == [2]
    assert tables.item.equals(shared.item)
    assert tables.user_item.global_type_share.equals(shared.global_type_share)
    # The shares are over ALL customers' lines, so they do not sum over the shard only.
    assert shared.global_type_share.height > 0


def test_features_ignore_purchases_on_and_after_the_anchor(tmp_path: Path) -> None:
    clean = _build(tmp_path / "clean", [[1, 2], [3]])
    poisoned = _build(tmp_path / "poison", [[1, 2], [3]], LINES + POISON)
    assert _same(poisoned, clean)


def test_features_json_is_written_last_and_verifies(tmp_path: Path) -> None:
    files = ShardFiles(tmp_path / "features", tmp_path / "query_stats.parquet")
    assert not shard_is_finished(files.features_dir)
    result = write_shard_features(_shard_inputs([1, 2, 3]), _shared(), CONFIG, 2, files, DETAILS)
    assert shard_is_finished(files.features_dir)
    assert (result.n_rows, result.n_parts, result.n_queries, result.skipped) == (9, 2, 3, False)
    # tt_dot / wals_dot were compared with the R6 / R5 scores on every returned row.
    assert (result.tt_dot_rows_checked, result.wals_dot_rows_checked) == (6, 4)
    record = verify_feature_files(files.features_dir, (*KEYS, *CONFIG.names))
    assert (record.n_rows, record.n_queries, record.fold) == (9, 3, "F1")
    content = json.loads((files.features_dir / "features.json").read_text())
    assert content["details"]["run"] == "rehearsal" and content["details"]["shard"] == 0
    # Part files without features.json are a half-finished shard.
    (files.features_dir / "features.json").unlink()
    with pytest.raises(FileExistsError, match="half-finished"):
        shard_is_finished(files.features_dir)


def test_row_count_must_equal_the_distinct_candidates(tmp_path: Path) -> None:
    inputs = _shard_inputs([1, 2, 3])
    assert count_distinct_candidates(inputs.candidates) == 9
    # Candidates of a query that is not in the shard are not built, so the counts differ.
    broken = dataclasses.replace(inputs, data=_shard_inputs([1, 2]).data)
    files = ShardFiles(tmp_path / "features", tmp_path / "query_stats.parquet")
    with pytest.raises(ValueError, match="6 rows, but the six channels hold 9 distinct"):
        write_shard_features(broken, _shared(), CONFIG, 5, files, DETAILS)
    assert not (files.features_dir / "features.json").exists()


def test_query_stats_count_candidates_and_mark_customers_without_history(tmp_path: Path) -> None:
    files = ShardFiles(tmp_path / "features", tmp_path / "query_stats.parquet")
    write_shard_features(_shard_inputs([1, 2, 3]), _shared(), CONFIG, 2, files, DETAILS)
    stats = pl.read_parquet(files.stats_file)
    assert stats.schema == QUERY_STATS_SCHEMA
    # q1 {0, 1, 2}, q2 {0, 1, 2}, q3 {1, 2, 3}: 3 candidates each; customer 2 never bought.
    assert stats.rows() == [(1, 0, 3, True), (2, 1, 3, True), (3, 2, 3, False)]


def test_query_without_candidates_counts_zero() -> None:
    user = pl.DataFrame(
        {"customer_idx": [0, 1], "user_days_since_last_purchase": [3.0, float("nan")]},
        schema={"customer_idx": pl.Int32, "user_days_since_last_purchase": pl.Float32},
    )
    queries = QUERIES.filter(pl.col("query_id") <= 2)
    counts = pl.DataFrame(
        {"query_id": [1], "n_candidates": [4]},
        schema={"query_id": pl.Int32, "n_candidates": pl.Int32},
    )
    assert query_stats(queries, counts, user).rows() == [(1, 0, 4, True), (2, 1, 0, False)]
    with pytest.raises(ValueError, match="no row in the customer table"):
        query_stats(QUERIES, counts, user)
