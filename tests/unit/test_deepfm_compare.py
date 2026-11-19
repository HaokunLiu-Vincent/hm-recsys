"""Tests for deepfm.compare: query groups, the paired bootstrap, the group table.

Example (k irrelevant here; AP values are given directly). Two folds:
    V1: q1 (customer 1, 5 days), q2 (customer 2, never bought), q3 (customer 3, 31 days)
    V2: q4 (customer 1, 30 days), q5 (customer 4, 400 days)
    DeepFM AP:   q1 0.5, q2 0.0, q3 0.2, q4 1.0, q5 0.3
    LightGBM AP: q1 0.4, q2 0.1, q3 0.2, q4 0.5, q5 0.5
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hmrec.deepfm.compare import (
    DEEPFM,
    DIFFERENCE,
    LIGHTGBM,
    CompareSettings,
    FoldFiles,
    assign_groups,
    compare_folds,
    fold_table,
    group_rows,
    paired_bootstrap,
    read_deepfm_ap,
    read_lightgbm_ap,
)
from hmrec.diagnostics.bootstrap import cluster_bootstrap, customer_totals
from hmrec.feature_files import FeatureFiles, describe_part, write_feature_files

NAN = float("nan")
DEEPFM_AP = {1: 0.5, 2: 0.0, 3: 0.2, 4: 1.0, 5: 0.3}
LIGHTGBM_AP = {1: 0.4, 2: 0.1, 3: 0.2, 4: 0.5, 5: 0.5}
CUSTOMER = {1: 1, 2: 2, 3: 3, 4: 1, 5: 4}
DAYS = {1: 5.0, 2: NAN, 3: 31.0, 4: 30.0, 5: 400.0}
FOLD_QUERIES = {"V1": [1, 2, 3], "V2": [4, 5]}


def _frame(query_ids: list[int], column: str, values: dict[int, float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"query_id": query_ids, column: [values[q] for q in query_ids]},
        schema={"query_id": pl.Int32, column: pl.Float64},
    )


def _queries(query_ids: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"query_id": query_ids, "customer_idx": [CUSTOMER[q] for q in query_ids]},
        schema={"query_id": pl.Int32, "customer_idx": pl.Int32},
    )


def _days(query_ids: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"query_id": query_ids, "user_days_since_last_purchase": [DAYS[q] for q in query_ids]},
        schema={"query_id": pl.Int32, "user_days_since_last_purchase": pl.Float32},
    )


def _fold(fold: str) -> pl.DataFrame:
    ids = FOLD_QUERIES[fold]
    return fold_table(
        fold,
        _queries(ids),
        assign_groups(_days(ids), 30),
        _frame(ids, "deepfm_ap", DEEPFM_AP),
        _frame(ids, "lightgbm_ap", LIGHTGBM_AP),
    )


def _all() -> pl.DataFrame:
    return pl.concat([_fold("V1"), _fold("V2")])


def test_groups_follow_the_days_since_the_last_purchase() -> None:
    groups = assign_groups(_days([1, 2, 3, 4, 5]), 30)
    # NaN = never bought; 5 and exactly 30 days are recent; 31 and 400 are "other".
    assert groups.rows() == [
        (1, "recent_purchase"),
        (2, "no_history"),
        (3, "other"),
        (4, "recent_purchase"),
        (5, "other"),
    ]
    with pytest.raises(ValueError, match="must be > 0"):
        assign_groups(_days([1]), 0)


def test_fold_table_needs_the_same_queries_from_both_models() -> None:
    table = _fold("V1")
    assert table.columns == [
        "query_id", "fold", "customer_idx", "query_group", "deepfm_ap", "lightgbm_ap",
    ]  # fmt: skip
    assert table.height == 3 and table["fold"].unique().to_list() == ["V1"]
    with pytest.raises(ValueError, match="not evaluated on the same queries"):
        fold_table(
            "V1", _queries([1, 2, 3]), assign_groups(_days([1, 2, 3]), 30),
            _frame([1, 2, 3], "deepfm_ap", DEEPFM_AP), _frame([1, 2], "lightgbm_ap", LIGHTGBM_AP),
        )  # fmt: skip
    groups = assign_groups(_days([1, 2, 3]), 30)
    both = (
        _frame([1, 2, 3], "deepfm_ap", DEEPFM_AP),
        _frame([1, 2, 3], "lightgbm_ap", LIGHTGBM_AP),
    )
    with pytest.raises(ValueError, match="no customer or no group"):
        fold_table("V1", _queries([1, 2]), groups, *both)


def test_group_table_hand_computed() -> None:
    rows = {(row["fold"], row["query_group"]): row for row in group_rows(_all())}
    # Pooled over both folds: recent = q1, q4; no history = q2; other = q3, q5.
    recent = rows[("pooled", "recent_purchase")]
    assert (recent["n_queries"], float(recent["query_share"])) == ("2", 0.4)
    assert float(recent["deepfm_map"]) == pytest.approx(0.75)  # (0.5 + 1.0) / 2
    assert float(recent["lightgbm_map"]) == pytest.approx(0.45)
    assert float(recent["difference"]) == pytest.approx(0.30)
    other = rows[("pooled", "other")]
    assert float(other["deepfm_map"]) == pytest.approx(0.25)  # (0.2 + 0.3) / 2
    assert float(other["lightgbm_map"]) == pytest.approx(0.35)
    everyone = rows[("pooled", "all")]
    assert float(everyone["deepfm_map"]) == pytest.approx(0.4)  # 2.0 / 5
    assert float(everyone["lightgbm_map"]) == pytest.approx(0.34)
    # Per fold: V2 has no customer without history, so that row does not exist.
    assert ("V2", "no_history") not in rows and rows[("V1", "no_history")]["n_queries"] == "1"
    assert float(rows[("V2", "all")]["deepfm_map"]) == pytest.approx(0.65)
    assert list(rows)[:4] == [
        ("V1", "no_history"), ("V1", "recent_purchase"), ("V1", "other"), ("V1", "all"),
    ]  # fmt: skip


def test_bootstrap_estimates_are_the_pooled_means_and_the_difference_is_paired() -> None:
    intervals, n_customers = paired_bootstrap(_all(), 200, 7, 0.95)
    assert n_customers == 4  # customer 1 has two queries and is drawn as one unit
    assert intervals[DEEPFM].estimate == pytest.approx(0.4)
    assert intervals[LIGHTGBM].estimate == pytest.approx(0.34)
    assert intervals[DIFFERENCE].estimate == pytest.approx(0.06)
    for interval in intervals.values():
        assert interval.low <= interval.estimate <= interval.high
    again, _ = paired_bootstrap(_all(), 200, 7, 0.95)
    assert again == intervals  # a fixed seed gives the same intervals
    other_seed, _ = paired_bootstrap(_all(), 200, 8, 0.95)
    assert other_seed[DIFFERENCE] != intervals[DIFFERENCE]
    with pytest.raises(ValueError, match="confidence_level"):
        paired_bootstrap(_all(), 200, 7, 1.0)


def test_bootstrap_equals_the_bootstrap_of_the_diagnostics() -> None:
    # 300 random customers with 1-3 queries each; the same input for both implementations.
    rng = np.random.default_rng(3)
    customers = np.repeat(np.arange(300), rng.integers(1, 4, size=300))
    per_query = pl.DataFrame(
        {
            "customer_idx": customers,
            "deepfm_ap": rng.random(customers.size),
            "lightgbm_ap": rng.random(customers.size),
        },
        schema={"customer_idx": pl.Int32, "deepfm_ap": pl.Float64, "lightgbm_ap": pl.Float64},
    )
    intervals, n_customers = paired_bootstrap(per_query, 1000, 42, 0.95)
    # The diagnostics call the two systems "ranker" and "baseline".
    renamed = per_query.rename({"deepfm_ap": "ranker_ap", "lightgbm_ap": "baseline_ap"})
    expected = cluster_bootstrap(customer_totals(renamed), 1000, 42, 0.95)
    pairs = (
        (DEEPFM, "lambdamart"),
        (LIGHTGBM, "rrf_baseline"),
        (DIFFERENCE, "lambdamart_minus_rrf_baseline"),
    )
    for mine, theirs in pairs:
        assert intervals[mine].estimate == expected[theirs].estimate
        assert intervals[mine].low == expected[theirs].low
        assert intervals[mine].high == expected[theirs].high
    assert n_customers == 300


# ---------------------------------------------------------------- from files

FEATURES = ("user_days_since_last_purchase",)
WIDE_COLUMNS = ("query_id", "article_idx", *FEATURES)


def _write_fold(root: Path, fold: str) -> FoldFiles:
    """The saved files of one fold of the example: per-query values, queries, a wide table."""
    ids = FOLD_QUERIES[fold]
    directory = root / fold
    features_dir = directory / "features"
    features_dir.mkdir(parents=True)
    # Two candidate rows per query; both carry the query's days value.
    wide = pl.DataFrame(
        {
            "query_id": [q for q in ids for _ in range(2)],
            "article_idx": [a for _ in ids for a in (7, 8)],
            "user_days_since_last_purchase": [DAYS[q] for q in ids for _ in range(2)],
        },
        schema={"query_id": pl.Int32, "article_idx": pl.Int32, FEATURES[0]: pl.Float32},
    )
    wide.write_parquet(features_dir / "part-00000.parquet")
    part = describe_part(features_dir / "part-00000.parquet")
    record = FeatureFiles(fold, WIDE_COLUMNS, (), (part,), wide.height, len(ids), {})
    write_feature_files(features_dir, record)
    deepfm = pl.concat(
        [
            _frame(ids, "map@12", DEEPFM_AP).with_columns(pl.lit(3, dtype=pl.Int32).alias("epoch")),
            _frame(ids, "map@12", LIGHTGBM_AP).with_columns(
                pl.lit(1, dtype=pl.Int32).alias("epoch")
            ),
        ]
    )
    deepfm.write_parquet(directory / "deepfm.parquet")
    lightgbm = _frame(ids, "map@12", LIGHTGBM_AP).with_columns(
        pl.lit(500, dtype=pl.Int32).alias("num_trees")
    )
    lightgbm.write_parquet(directory / "lightgbm.parquet")
    _queries(ids).write_parquet(directory / "queries.parquet")
    return FoldFiles(
        fold,
        directory / "deepfm.parquet",
        directory / "lightgbm.parquet",
        directory / "queries.parquet",
        features_dir,
    )


SETTINGS = CompareSettings("map@12", 3, 500, 30, 200, 7, 0.95, WIDE_COLUMNS)


def test_comparison_from_saved_files(tmp_path: Path) -> None:
    folds = [_write_fold(tmp_path, "V1"), _write_fold(tmp_path, "V2")]
    intervals, n_customers, n_queries, groups = compare_folds(folds, SETTINGS)
    expected, _ = paired_bootstrap(_all(), 200, 7, 0.95)
    assert intervals == expected and (n_customers, n_queries) == (4, 5)
    assert groups == group_rows(_all())
    # The epoch and the number of trees pick the rows that are compared.
    assert read_deepfm_ap(folds[0].deepfm_per_query, 3, "map@12")["deepfm_ap"].to_list() == [
        0.5,
        0.0,
        0.2,
    ]
    with pytest.raises(ValueError, match="no rows for epoch 9"):
        read_deepfm_ap(folds[0].deepfm_per_query, 9, "map@12")
    with pytest.raises(ValueError, match="no rows for 300 trees"):
        read_lightgbm_ap(folds[0].lightgbm_per_query, 300, "map@12")


def test_a_wide_table_that_is_not_the_recorded_one_is_refused(tmp_path: Path) -> None:
    files = _write_fold(tmp_path, "V1")
    record = files.features_dir / "features.json"
    record.write_text(record.read_text().replace('"sha256": "', '"sha256": "0'))
    with pytest.raises(ValueError, match="sha256"):
        compare_folds([files], SETTINGS)
