"""Tests for hmrec.data.queries. Expected values: tests/fixtures/asof/README.md."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from conftest import D1, D2, FIXTURE_DAY_ZERO, make_transactions
from hmrec.config import load_queries_config
from hmrec.data.folds import Fold, FoldManifest, load_fold_manifest
from hmrec.data.queries import (
    LABELS_SCHEMA,
    QUERIES_SCHEMA,
    FoldQueries,
    build_fold_queries,
    check_fold_queries,
    check_query_id_capacity,
    decode_query_id,
    fold_ordinal,
    query_id_base,
    write_fold_queries,
)

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"
OFFSET = 1000  # small offset for readable expected values; the repo uses 10,000,000
CATALOG = pl.DataFrame({"article_idx": [0, 1, 2, 3, 4, 5]}, schema={"article_idx": pl.Int32})


def _build(
    transactions: pl.DataFrame,
    fold: Fold,
    manifest: FoldManifest,
    catalog: pl.DataFrame = CATALOG,
    offset: int = OFFSET,
) -> FoldQueries:
    return build_fold_queries(
        transactions.lazy(), catalog, fold, manifest, FIXTURE_DAY_ZERO, offset
    )


# ---------------------------------------------------------------- query_id


def test_repo_offset_and_fold_ordinals() -> None:
    offset = load_queries_config(REPO_CONFIGS / "queries.yaml").query_id_fold_offset
    manifest = load_fold_manifest(REPO_CONFIGS / "fold_manifest.yaml")
    assert offset == 10_000_000
    assert fold_ordinal(manifest, "D1") == 0
    assert fold_ordinal(manifest, "T4") == 15
    # 16 folds x 10M = 160M, far below the int32 limit of about 2.1 billion.
    check_query_id_capacity(manifest, offset)
    assert query_id_base(manifest, "V2", offset) == 9 * 10_000_000


def test_decode_query_id_round_trip() -> None:
    manifest = load_fold_manifest(REPO_CONFIGS / "fold_manifest.yaml")
    offset = 10_000_000
    assert decode_query_id(0, manifest, offset) == ("D1", 0)
    assert decode_query_id(10_000_002, manifest, offset) == ("D2", 2)
    assert decode_query_id(15 * offset + 70_000, manifest, offset) == ("T4", 70_000)
    for fold_name in manifest.names:
        query_id = query_id_base(manifest, fold_name, offset) + 123
        assert decode_query_id(query_id, manifest, offset) == (fold_name, 123)


def test_decode_query_id_rejects_bad_values(fixture_manifest: FoldManifest) -> None:
    with pytest.raises(ValueError, match=">= 0"):
        decode_query_id(-1, fixture_manifest, OFFSET)
    with pytest.raises(ValueError, match="fold ordinal 2"):
        decode_query_id(2 * OFFSET, fixture_manifest, OFFSET)


def test_capacity_check_rejects_int32_overflow(fixture_manifest: FoldManifest) -> None:
    with pytest.raises(ValueError, match="int32"):
        # Largest query_id = 2 folds x (2^30 + 1) - 1 = 2^31 + 1 > int32 max (2^31 - 1).
        check_query_id_capacity(fixture_manifest, 2**30 + 1)
    # Exactly at the limit is allowed: 2 x 2^30 - 1 = 2^31 - 1.
    check_query_id_capacity(fixture_manifest, 2**30)


# ---------------------------------------------------------------- building


def test_d1_queries_and_labels_hand_computed(
    fixture_transactions: pl.DataFrame, fixture_manifest: FoldManifest
) -> None:
    result = _build(fixture_transactions, D1, fixture_manifest)
    # Label week days 42..48: customers 0 and 2. D1 is ordinal 0 -> query_id = position.
    assert result.queries.rows() == [(0, "D1", 0, D1.anchor), (1, "D1", 2, D1.anchor)]
    # Customer 0 bought article 3 (day 42) and article 4 twice (days 45, 48): 2 labels.
    assert result.labels.rows() == [(0, "D1", 3), (0, "D1", 4), (1, "D1", 0)]
    assert result.queries.schema == QUERIES_SCHEMA
    assert result.labels.schema == LABELS_SCHEMA


def test_d2_query_ids_start_at_offset(
    fixture_transactions: pl.DataFrame, fixture_manifest: FoldManifest
) -> None:
    result = _build(fixture_transactions, D2, fixture_manifest)
    # D2 is ordinal 1 -> query_id = 1000 + position. Customers 0 and 1.
    assert result.queries["query_id"].to_list() == [1000, 1001]
    assert result.queries["customer_idx"].to_list() == [0, 1]
    assert result.labels.rows() == [(1000, "D2", 2), (1001, "D2", 1)]


def test_queries_ignore_poison_outside_label_week(
    fixture_transactions: pl.DataFrame, fixture_manifest: FoldManifest
) -> None:
    clean = _build(fixture_transactions, D1, fixture_manifest)
    # Before the anchor (history) and after label_end (day 48): new customers 7, 8, 9
    # and new articles for customer 0. None of them may appear in D1.
    poison = make_transactions([(9, 5, 41), (9, 5, 0), (8, 5, 49), (0, 5, 49), (7, 1, 733)])
    poisoned = _build(pl.concat([fixture_transactions, poison]), D1, fixture_manifest)
    assert poisoned.queries.equals(clean.queries)
    assert poisoned.labels.equals(clean.labels)


def test_only_catalog_articles_count(
    fixture_transactions: pl.DataFrame, fixture_manifest: FoldManifest
) -> None:
    # Without article 0, customer 2 (who bought only article 0) is not a query,
    # and customer 0 keeps position 0.
    catalog_without_0 = CATALOG.filter(pl.col("article_idx") != 0)
    result = _build(fixture_transactions, D1, fixture_manifest, catalog=catalog_without_0)
    assert result.queries["customer_idx"].to_list() == [0]
    assert result.labels.rows() == [(0, "D1", 3), (0, "D1", 4)]


def test_too_many_queries_for_offset(
    fixture_transactions: pl.DataFrame, fixture_manifest: FoldManifest
) -> None:
    # D1 has 2 queries; offset 2 would let position 2 of D1 collide with D2.
    with pytest.raises(ValueError, match="not below"):
        _build(fixture_transactions, D1, fixture_manifest, offset=2)


def test_check_fold_queries_rejects_query_without_labels(
    fixture_transactions: pl.DataFrame, fixture_manifest: FoldManifest
) -> None:
    result = _build(fixture_transactions, D1, fixture_manifest)
    broken = FoldQueries(
        queries=result.queries, labels=result.labels.filter(pl.col("query_id") == 0)
    )
    with pytest.raises(ValueError, match="every query needs"):
        check_fold_queries(broken)


# ---------------------------------------------------------------- IO


def test_write_refuses_to_overwrite(
    tmp_path: Path, fixture_transactions: pl.DataFrame, fixture_manifest: FoldManifest
) -> None:
    result = _build(fixture_transactions, D1, fixture_manifest)
    queries_path = tmp_path / "queries" / "D1.parquet"
    labels_path = tmp_path / "labels" / "D1.parquet"
    write_fold_queries(result, queries_path, labels_path, overwrite=False)
    assert pl.read_parquet(queries_path).equals(result.queries)
    assert pl.read_parquet(labels_path).equals(result.labels)
    with pytest.raises(FileExistsError):
        write_fold_queries(result, queries_path, labels_path, overwrite=False)
    write_fold_queries(result, queries_path, labels_path, overwrite=True)


def test_anchor_column_is_a_date(
    fixture_transactions: pl.DataFrame, fixture_manifest: FoldManifest
) -> None:
    result = _build(fixture_transactions, D1, fixture_manifest)
    assert result.queries["anchor"].unique().to_list() == [date(2020, 2, 12)]
