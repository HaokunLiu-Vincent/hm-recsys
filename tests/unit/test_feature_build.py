"""Tests for features.build: one fold from the six candidate tables to the part files.

The fold is the one of feature_example.py; vectors, embeddings and the bucket table are
those of test_feature_dense.py. Candidates (the R5 / R6 scores are the dot products of
those vectors, as recall stores them):
    pop   q1: art 0 (rank 1, own), art 2 (rank 2, fill-in); q2: art 2 (1, fill-in);
          q3: art 3 (1, own)
    rep   q1: art 0        sib   q1: art 1        icf   q2: art 0
    wals  q1: art 1 (u0 . v1 = 4), art 0 (0.75);  q2: art 1 (2), art 0 (1)
    tt    q1: art 2 (3), art 1 (2);  q2: art 2 (7), art 1 (4);  q3: art 2 (11), art 1 (6)
Distinct (query, article): q1 {0, 1, 2}, q2 {0, 1, 2}, q3 {1, 2, 3} -> 9 rows.
"""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from feature_example import CONFIG, LINES, POISON, make_fold, make_history
from test_feature_dense import TWOTOWER, WALS, make_dense_sources
from test_fusion import _candidates

from hmrec.config import load_base_config
from hmrec.feature_files import verify_feature_files
from hmrec.features.build import (
    BuildResult,
    FoldInputs,
    build_fold_features,
    check_dot_equals_score,
    order_columns,
)
from hmrec.features.stats import STATS_COLUMNS
from hmrec.paths import ProjectPaths, build_paths
from hmrec.run_records import RunStamp
from hmrec.vectors import dot_pairs

REPO = Path(__file__).resolve().parents[2]
BASE = load_base_config(REPO / "configs" / "base.yaml")
STAMP = RunStamp("abc123", False)
KEYS = ["query_id", "article_idx"]
EXPECTED_ROWS = 9


def _vector_candidates(vectors, rows: list[tuple[int, int, int, int]]) -> pl.DataFrame:
    """(query_id, customer_idx, article_idx, rank) rows scored with the vectors' dot product."""
    customers = np.array([row[1] for row in rows], dtype=np.int32)
    articles = np.array([row[2] for row in rows], dtype=np.int32)
    scores = dot_pairs(vectors, customers, articles, 10)
    return _candidates(
        [(row[0], row[2], row[3], float(score)) for row, score in zip(rows, scores, strict=True)]
    )


def make_candidates() -> dict[str, pl.DataFrame]:
    return {
        "r1_popularity": _candidates(
            [(1, 0, 1, 1.5), (1, 2, 2, 0.3), (2, 2, 1, 0.3), (3, 3, 1, 0.25)], fallback=[0, 1, 1, 0]
        ),
        "r2_repurchase": _candidates([(1, 0, 1, 0.9)]),
        "r3_sibling": _candidates([(1, 1, 1, 0.7)]),
        "r4_itemcf": _candidates([(2, 0, 1, 0.2)]),
        "r5_wals": _vector_candidates(
            WALS, [(1, 0, 1, 1), (1, 0, 0, 2), (2, 1, 1, 1), (2, 1, 0, 2)]
        ),
        "r6_twotower": _vector_candidates(
            TWOTOWER,
            [(1, 0, 2, 1), (1, 0, 1, 2), (2, 1, 2, 1), (2, 1, 1, 2), (3, 2, 2, 1), (3, 2, 1, 2)],
        ),
    }


def _inputs(lines: list = LINES, expected_rows: int = EXPECTED_ROWS) -> FoldInputs:
    return FoldInputs(
        data=make_fold(lines),
        history=make_history(lines),
        candidates=make_candidates(),
        dense=make_dense_sources(lines),
        expected_rows=expected_rows,
    )


def _build(
    tmp_path: Path, chunk: int = 2, inputs: FoldInputs | None = None
) -> tuple[ProjectPaths, BuildResult]:
    paths = build_paths(BASE, tmp_path)
    return paths, build_fold_features(paths, inputs or _inputs(), CONFIG, chunk, STAMP)


def _parts(paths: ProjectPaths) -> list[pl.DataFrame]:
    return [
        pl.read_parquet(path) for path in sorted(paths.features_dir("F1").glob("part-*.parquet"))
    ]


def test_wide_table_columns_dtypes_and_no_label_column(tmp_path: Path) -> None:
    paths, result = _build(tmp_path)
    table = pl.concat(_parts(paths))
    # query_id, article_idx, then EXACTLY the 71 features of configs/features.yaml, in order.
    assert table.columns == [*KEYS, *CONFIG.names] and len(CONFIG.names) == 71
    for feature in CONFIG.features:
        assert str(table.schema[feature.name]) == ("Int16" if feature.categorical else "Float32")
    assert all(table[name].null_count() == 0 for name in table.columns)  # NaN or code 0, never null
    assert not [name for name in table.columns if "label" in name]
    assert "customer_idx" not in table.columns  # raw ids are not features (proposal §8.1)
    assert (result.n_rows, result.n_parts, result.n_queries, result.skipped) == (9, 2, 3, False)


def test_rows_are_sorted_and_a_query_is_never_split_over_files(tmp_path: Path) -> None:
    paths, _ = _build(tmp_path, chunk=2)
    parts = _parts(paths)
    assert [part["query_id"].unique().sort().to_list() for part in parts] == [[1, 2], [3]]
    for part in parts:
        assert part.select(KEYS).equals(part.select(KEYS).sort(KEYS))
    table = pl.concat(parts)
    assert table.select(KEYS).rows() == [
        (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3),
    ]  # fmt: skip
    assert not table.select(KEYS).is_duplicated().any()


def test_one_value_of_every_group(tmp_path: Path) -> None:
    paths, _ = _build(tmp_path)
    table = pl.concat(_parts(paths))
    row = table.filter((pl.col("query_id") == 1) & (pl.col("article_idx") == 0)).to_dicts()[0]
    # A: returned by pop (rank 1), rep (rank 1) and wals (rank 2) -> 3 channels.
    assert (row["pop_rank"], row["rep_rank"], row["wals_rank"], row["n_channels"]) == (1, 1, 2, 3)
    assert math.isnan(row["tt_rank"]) and row["rep_score"] == pytest.approx(0.9)
    assert row["rrf_score"] == pytest.approx(2 / 61 + 1 / 62, rel=1e-6)
    # B: tt_dot is there although R6 did not return the article.
    assert (row["tt_dot"], row["wals_dot"], row["pop_bucket_score"]) == (1.0, 0.75, 1.5)
    # C, D, E: the hand-computed values of the group tests.
    assert (row["item_cnt_1w"], row["user_cnt_52w"], row["ui_cnt_item"]) == (3, 3, 2)
    assert (row["product_type_no"], row["club_member_status"]) == (1, 1)
    assert row["ui_price_ratio"] == pytest.approx((1.25 / 3) / 0.35, rel=1e-6)
    # The features added on 2026-11-17 (one of each group; values of the group tests).
    assert (row["tt_sim_max_hist"], row["item_discount_ratio"]) == (1.0, 1.0)
    assert (row["user_purchase_days_52w"], row["ui_attr_match_cnt"]) == (3, 6)
    assert row["ui_decay_garment"] == pytest.approx(math.exp(-5 / 30) + math.exp(-1 / 30), rel=1e-6)


def test_chunk_size_does_not_change_the_table(tmp_path: Path) -> None:
    tables = []
    for chunk in (1, 2, 10):
        paths, result = _build(tmp_path / f"chunk{chunk}", chunk=chunk)
        assert result.n_parts == {1: 3, 2: 2, 10: 1}[chunk]
        tables.append(pl.concat(_parts(paths)))
    assert tables[0].equals(tables[1]) and tables[1].equals(tables[2])


def test_poison_on_and_after_the_anchor_changes_nothing(tmp_path: Path) -> None:
    clean_paths, _ = _build(tmp_path / "clean")
    poisoned_paths, _ = _build(tmp_path / "poisoned", inputs=_inputs(LINES + POISON))
    assert pl.concat(_parts(clean_paths)).equals(pl.concat(_parts(poisoned_paths)))


def test_dot_checks_metadata_category_codes_and_stats(tmp_path: Path) -> None:
    paths, result = _build(tmp_path)
    # Every R6 / R5 row was compared with the channel's score.
    assert (result.tt_dot_rows_checked, result.wals_dot_rows_checked) == (6, 4)
    # features.json: every part file with its sha256 and rows, and the column list.
    record = json.loads(paths.features_metadata_file("F1").read_text())
    assert (record["n_rows"], record["n_queries"]) == (9, 3)
    assert record["columns"] == [*KEYS, *CONFIG.names]
    assert record["categorical_features"] == list(CONFIG.categorical_names)
    assert [(part["name"], part["n_rows"]) for part in record["parts"]] == [
        ("part-00000.parquet", 6),
        ("part-00001.parquet", 3),
    ]
    assert (record["details"]["tt_dot_rows_checked"], record["details"]["git_commit"]) == (
        6,
        "abc123",
    )
    assert result.history_similarity_seconds > 0
    assert record["details"]["history_similarity_seconds"] == result.history_similarity_seconds
    codes = json.loads(paths.category_codes_file.read_text())["codes"]
    assert codes["club_member_status"] == ["ACTIVE", "PRE-CREATE"]
    assert codes["product_type_no"] == [5, 7]
    # One stats row per feature.
    assert [row["feature"] for row in result.stats_rows] == list(CONFIG.names)
    assert all(tuple(row) == STATS_COLUMNS for row in result.stats_rows)
    stats = {row["feature"]: row for row in result.stats_rows}
    # tt_rank: 6 of 9 rows were returned by R6 -> 3 missing; ranks 1 and 2 -> mean 1.5.
    assert (stats["tt_rank"]["n_rows"], stats["tt_rank"]["n_missing"]) == ("9", "3")
    assert (stats["tt_rank"]["min"], stats["tt_rank"]["max"], stats["tt_rank"]["mean"]) == (
        "1.0",
        "2.0",
        "1.5",
    )
    # A category: missing = code 0 (customer 1 has no club status: its 3 rows); no mean.
    club = stats["club_member_status"]
    assert (club["n_missing"], club["min"], club["max"], club["mean"]) == ("3", "0.0", "2.0", "")


def test_finished_fold_is_skipped_and_never_replaced(tmp_path: Path) -> None:
    paths, _ = _build(tmp_path)
    files = sorted(paths.features_dir("F1").iterdir())
    written_at = [path.stat().st_mtime_ns for path in files]
    again = build_fold_features(paths, _inputs(), CONFIG, 2, STAMP)
    assert (again.skipped, again.stats_rows) == (True, [])
    assert [path.stat().st_mtime_ns for path in files] == written_at


def test_part_files_without_metadata_are_a_half_finished_fold(tmp_path: Path) -> None:
    paths = build_paths(BASE, tmp_path)
    part = paths.features_part_file("F1", 0)
    part.parent.mkdir(parents=True)
    part.write_bytes(b"left over")
    with pytest.raises(FileExistsError, match="half-finished fold"):
        build_fold_features(paths, _inputs(), CONFIG, 2, STAMP)
    assert part.read_bytes() == b"left over"


def test_wrong_row_count_fails_and_leaves_the_fold_unfinished(tmp_path: Path) -> None:
    paths = build_paths(BASE, tmp_path)
    with pytest.raises(ValueError, match="has 9 rows, but results/m6_candidates.csv records 10"):
        build_fold_features(paths, _inputs(expected_rows=10), CONFIG, 2, STAMP)
    assert not paths.features_metadata_file("F1").exists()


def test_a_dot_that_is_not_the_channels_score_fails(tmp_path: Path) -> None:
    inputs = _inputs()
    tampered = dict(inputs.candidates)
    # One R6 score off by one unit in the last place.
    r6 = tampered["r6_twotower"]
    bumped = (
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("score") + 1e-6 * 10)
        .otherwise(pl.col("score"))
    )
    tampered["r6_twotower"] = r6.with_columns(bumped.cast(pl.Float32).alias("score"))
    paths = build_paths(BASE, tmp_path)
    with pytest.raises(ValueError, match="tt_dot differs from the channel's score on 1 of"):
        build_fold_features(
            paths, dataclasses.replace(inputs, candidates=tampered), CONFIG, 2, STAMP
        )
    assert not paths.features_metadata_file("F1").exists()


def test_dot_check_requires_every_candidate_row() -> None:
    wide = pl.DataFrame(
        {"query_id": [1], "article_idx": [2], "tt_rank": [1.0], "tt_dot": [3.0]},
        schema={
            "query_id": pl.Int32,
            "article_idx": pl.Int32,
            "tt_rank": pl.Float32,
            "tt_dot": pl.Float32,
        },
    )
    candidates = _candidates([(1, 2, 1, 3.0)])
    assert check_dot_equals_score(wide, candidates, "r6_twotower", "tt_dot") == 1
    two = _candidates([(1, 2, 1, 3.0), (1, 3, 2, 2.0)])  # a candidate that is not in the table
    with pytest.raises(ValueError, match="2 candidates, but 1 wide rows"):
        check_dot_equals_score(wide, two, "r6_twotower", "tt_dot")


def test_order_columns_refuses_missing_columns_and_nulls() -> None:
    features = CONFIG.features[:2]  # pop_rank, rep_rank
    table = pl.DataFrame(
        {"query_id": [2, 1], "article_idx": [5, 5], "pop_rank": [1.0, None], "rep_rank": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="null values in the wide table columns"):
        order_columns(table, features)
    with pytest.raises(ValueError, match="lacks the feature columns"):
        order_columns(table.drop("rep_rank"), features)
    ordered = order_columns(
        table.with_columns(pl.col("pop_rank").fill_null(float("nan"))), features
    )
    assert ordered["query_id"].to_list() == [1, 2] and str(ordered.schema["pop_rank"]) == "Float32"


# ---------------------------------------------------------------- no recall import, no labels

FORBIDDEN = (
    "hmrec.recall",
    "labels_file",
    "FoldEvaluator",
    "load_label_evaluator",
    "label_week_transactions",
)


def test_feature_and_fusion_code_never_imports_recall_or_reads_labels() -> None:
    files = sorted((REPO / "src" / "hmrec" / "features").rglob("*.py"))
    files += sorted((REPO / "src" / "hmrec" / "fusion").rglob("*.py"))
    files += [
        REPO / "scripts" / "m6_build_features.py",
        REPO / "src" / "hmrec" / "data" / "histories.py",
    ]
    assert len(files) >= 14
    offenders = [
        f"{path.relative_to(REPO)}: {name}"
        for path in files
        for name in FORBIDDEN
        if name in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_written_fold_passes_the_readers_verification(tmp_path: Path) -> None:
    # What M7 calls before it reads a fold's features.
    paths, _ = _build(tmp_path)
    directory = paths.features_dir("F1")
    files = verify_feature_files(directory, [*KEYS, *CONFIG.names])
    assert (files.fold, files.n_rows, len(files.parts)) == ("F1", 9, 2)
    # A part file changed after it was written is caught by its sha256.
    part = directory / "part-00001.parquet"
    pl.read_parquet(part).with_columns(pl.col("pop_rank") + 1).write_parquet(part)
    with pytest.raises(ValueError, match="This is not the frozen file"):
        verify_feature_files(directory, [*KEYS, *CONFIG.names])
