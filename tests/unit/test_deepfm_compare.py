"""Tests for deepfm.compare: the arm table, the paired bootstrap, the query groups.

Example (AP values are given directly). Two folds, five queries:
    V1: q1 (customer 1, 5 days), q2 (customer 2, never bought), q3 (customer 3, 31 days)
    V2: q4 (customer 1, 30 days), q5 (customer 4, 400 days)

Four arms with seeds 42 and 43; AP of q1 ... q5 and the pooled mean over the 5 queries:
    lgb_lambdarank     42: 0.4 0.1 0.2 0.5 0.5 -> 0.34     43: 0.6 0.1 0.4 0.5 0.3 -> 0.38
    lgb_binary         42: 0.2 everywhere      -> 0.20     43: 0.3 everywhere      -> 0.30
    deepfm_listwise    42: 0.1 0.1 0.1 0.1 0.6 -> 0.20     43: 0.2 0.2 0.2 0.2 0.7 -> 0.30
    deepfm_pointwise   42: 0.5 0.0 0.2 1.0 0.3 -> 0.40     43: 0.5 0.2 0.2 1.0 0.3 -> 0.44
"""

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hmrec.deepfm.compare import (
    ARMS,
    CONTRASTS,
    DIFFERENCE,
    FIRST,
    SECOND,
    SEED_MEAN,
    ArmRuns,
    CompareSettings,
    FoldFiles,
    arm_fold_values,
    arm_rows,
    assign_groups,
    compare_arms,
    contrast_rows,
    fold_table,
    group_rows,
    paired_bootstrap,
    read_saved_values,
    seed_basis,
    seed_summary,
    table_columns,
    value_column,
)
from hmrec.diagnostics.bootstrap import cluster_bootstrap, customer_totals
from hmrec.feature_files import FeatureFiles, describe_part, write_feature_files

NAN = float("nan")
AP = {
    "lgb_lambdarank": {42: [0.4, 0.1, 0.2, 0.5, 0.5], 43: [0.6, 0.1, 0.4, 0.5, 0.3]},
    "lgb_binary": {42: [0.2] * 5, 43: [0.3] * 5},
    "deepfm_listwise": {42: [0.1, 0.1, 0.1, 0.1, 0.6], 43: [0.2, 0.2, 0.2, 0.2, 0.7]},
    "deepfm_pointwise": {42: [0.5, 0.0, 0.2, 1.0, 0.3], 43: [0.5, 0.2, 0.2, 1.0, 0.3]},
}
STEP_COLUMN = {
    "lgb_lambdarank": "num_trees",
    "lgb_binary": "num_trees",
    "deepfm_listwise": "epoch",
    "deepfm_pointwise": "epoch",
}
STEP = {"lgb_lambdarank": 500, "lgb_binary": 300, "deepfm_listwise": 2, "deepfm_pointwise": 1}
CUSTOMER = {1: 1, 2: 2, 3: 3, 4: 1, 5: 4}
DAYS = {1: 5.0, 2: NAN, 3: 31.0, 4: 30.0, 5: 400.0}
FOLD_QUERIES = {"V1": [1, 2, 3], "V2": [4, 5]}
METRICS = ("map@12", "recall@12")
FEATURES = ("user_days_since_last_purchase",)
WIDE_COLUMNS = ("query_id", "article_idx", *FEATURES)
SETTINGS = CompareSettings(METRICS, 42, 30, 200, 7, 0.95, WIDE_COLUMNS)
MAP = "map@12"


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


def _saved(arm: str, seed: int, query_ids: list[int], step: int) -> pl.DataFrame:
    """A per-query file of one training: the AP as map@12 and half of it as recall@12."""
    values = [AP[arm][seed][q - 1] for q in query_ids]
    return pl.DataFrame(
        {
            "query_id": query_ids,
            STEP_COLUMN[arm]: [step] * len(query_ids),
            "map@12": values,
            "recall@12": [value / 2 for value in values],
        },
        schema={
            "query_id": pl.Int32,
            STEP_COLUMN[arm]: pl.Int32,
            "map@12": pl.Float64,
            "recall@12": pl.Float64,
        },
    )


def _write(root: Path) -> tuple[list[FoldFiles], list[ArmRuns]]:
    """The saved files of the example: per arm, seed and fold; queries; a wide table per fold."""
    folds = []
    for fold, ids in FOLD_QUERIES.items():
        features_dir = root / fold / "features"
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
        _queries(ids).write_parquet(root / fold / "queries.parquet")
        folds.append(FoldFiles(fold, root / fold / "queries.parquet", features_dir))
    arms = []
    for arm in ARMS:
        files: dict[int, dict[str, Path]] = {}
        for seed in (42, 43):
            files[seed] = {}
            for fold, ids in FOLD_QUERIES.items():
                table = _saved(arm, seed, ids, STEP[arm])
                if arm == "lgb_binary" and seed == 42:
                    # This training was evaluated at several checkpoints; only one is the arm's.
                    other = _saved(arm, seed, ids, 100).with_columns(pl.lit(0.9).alias("map@12"))
                    table = pl.concat([other, table])
                path = root / fold / f"{arm}_s{seed}.parquet"
                table.write_parquet(path)
                files[seed][fold] = path
        arms.append(ArmRuns(arm, STEP_COLUMN[arm], dict.fromkeys((42, 43), STEP[arm]), files))
    return folds, arms


@pytest.fixture(scope="module")
def example(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[FoldFiles], list[ArmRuns]]:
    return _write(tmp_path_factory.mktemp("compare"))


def _per_query(example: tuple[list[FoldFiles], list[ArmRuns]]) -> pl.DataFrame:
    _, arms = example
    tables = []
    for fold, ids in FOLD_QUERIES.items():
        values = [arm_fold_values(arm, fold, METRICS) for arm in arms]
        tables.append(fold_table(fold, _queries(ids), assign_groups(_days(ids), 30), values))
    return pl.concat(tables)


# ---------------------------------------------------------------- reading


def test_saved_values_are_read_for_the_selected_model_only(
    example: tuple[list[FoldFiles], list[ArmRuns]],
) -> None:
    _, arms = example
    binary = arms[1]
    file = binary.files[42]["V1"]
    # The file holds two checkpoints; the arm's k = 300 picks the 0.2 rows.
    assert read_saved_values(file, "num_trees", 300, METRICS)["map@12"].to_list() == [0.2] * 3
    assert read_saved_values(file, "num_trees", 100, METRICS)["map@12"].to_list() == [0.9] * 3
    with pytest.raises(ValueError, match="no rows with num_trees = 700"):
        read_saved_values(file, "num_trees", 700, METRICS)
    values = arm_fold_values(arms[0], "V1", METRICS)
    assert values.columns == [
        "query_id",
        "lgb_lambdarank|seed_42|map@12",
        "lgb_lambdarank|seed_42|recall@12",
        "lgb_lambdarank|seed_43|map@12",
        "lgb_lambdarank|seed_43|recall@12",
        "lgb_lambdarank|seed_mean|map@12",
        "lgb_lambdarank|seed_mean|recall@12",
    ]
    # Seed mean of every query: (0.4 + 0.6) / 2, (0.1 + 0.1) / 2, (0.2 + 0.4) / 2.
    assert values["lgb_lambdarank|seed_mean|map@12"].to_list() == pytest.approx([0.5, 0.1, 0.3])
    assert (seed_basis(42), value_column("a", SEED_MEAN, "m")) == ("seed_42", "a|seed_mean|m")


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


def test_fold_table_needs_the_same_queries_from_every_arm(
    example: tuple[list[FoldFiles], list[ArmRuns]],
) -> None:
    _, arms = example
    table = _per_query(example)
    assert table.columns[:4] == ["query_id", "fold", "customer_idx", "query_group"]
    assert table.height == 5 and len(table.columns) == 4 + 4 * 3 * 2
    values = [arm_fold_values(arm, "V1", METRICS) for arm in arms]
    groups = assign_groups(_days([1, 2, 3]), 30)
    with pytest.raises(ValueError, match="not evaluated on the same queries"):
        fold_table("V1", _queries([1, 2, 3]), groups, [values[0], values[1].head(2)])
    with pytest.raises(ValueError, match="no customer or no group"):
        fold_table("V1", _queries([1, 2]), groups, values)
    # A seed that was evaluated on other queries is refused too.
    broken = ArmRuns(
        "lgb_lambdarank",
        "num_trees",
        {42: 500, 43: 500},
        {42: arms[0].files[42], 43: {"V1": arms[0].files[43]["V2"]}},
    )
    with pytest.raises(ValueError, match="evaluated on other queries"):
        arm_fold_values(broken, "V1", METRICS)


# ---------------------------------------------------------------- the arm table


def test_seed_summary_hand_computed() -> None:
    # 0.34 and 0.38: mean 0.36; sample std = sqrt(0.02^2 + 0.02^2) = 0.0283; range 0.04 / 0.34.
    summary = seed_summary([0.34, 0.38])
    assert summary["mean"] == pytest.approx(0.36)
    assert summary["std"] == pytest.approx(math.sqrt(0.0008))
    assert summary["range"] == pytest.approx(0.04 / 0.34)
    # Three seeds: the divisor of the variance is n - 1 = 2.
    assert seed_summary([1.0, 2.0, 3.0])["std"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="at least two"):
        seed_summary([0.3])


def test_arm_table_hand_computed(example: tuple[list[FoldFiles], list[ArmRuns]]) -> None:
    _, arms = example
    rows = {(row["arm"], row["row"]): row for row in arm_rows(_per_query(example), arms, METRICS)}
    assert list(rows)[:5] == [
        ("lgb_lambdarank", "seed_42"),
        ("lgb_lambdarank", "seed_43"),
        ("lgb_lambdarank", "mean"),
        ("lgb_lambdarank", "std"),
        ("lgb_lambdarank", "range"),
    ]
    pooled = {
        "lgb_lambdarank": (0.34, 0.38),
        "lgb_binary": (0.20, 0.30),
        "deepfm_listwise": (0.20, 0.30),
        "deepfm_pointwise": (0.40, 0.44),
    }
    for arm, (first, second) in pooled.items():
        assert float(rows[(arm, "seed_42")][MAP]) == pytest.approx(first)
        assert float(rows[(arm, "seed_43")][MAP]) == pytest.approx(second)
        assert float(rows[(arm, "mean")][MAP]) == pytest.approx((first + second) / 2)
        assert float(rows[(arm, "range")][MAP]) == pytest.approx((second - first) / first)
        # recall@12 was saved as half of the AP.
        assert float(rows[(arm, "mean")]["recall@12"]) == pytest.approx((first + second) / 4)
        assert rows[(arm, "seed_42")]["step"] == str(STEP[arm])
    lambdarank = rows[("lgb_lambdarank", "std")]
    assert float(lambdarank[MAP]) == pytest.approx(math.sqrt(0.0008))
    # Per fold (main metric): V1 = q1 ... q3 -> (0.4 + 0.1 + 0.2) / 3 and (0.6 + 0.1 + 0.4) / 3.
    assert float(rows[("lgb_lambdarank", "seed_42")]["map@12|V1"]) == pytest.approx(0.7 / 3)
    assert float(rows[("lgb_lambdarank", "seed_43")]["map@12|V1"]) == pytest.approx(1.1 / 3)
    assert float(rows[("lgb_lambdarank", "mean")]["map@12|V1"]) == pytest.approx(0.3)
    assert lambdarank["map@12|V1"] == "" and lambdarank["n_queries"] == "5"
    columns = table_columns(list(rows.values()))
    assert columns[:5] == ("arm", "row", "step_column", "step", "n_queries")
    assert columns[5:] == ("map@12", "recall@12", "map@12|V1", "map@12|V2")


# ---------------------------------------------------------------- the bootstrap


def test_bootstrap_estimates_are_the_pooled_means_and_the_difference_is_paired(
    example: tuple[list[FoldFiles], list[ArmRuns]],
) -> None:
    per_query = _per_query(example)
    first = value_column("deepfm_pointwise", "seed_42", MAP)
    second = value_column("lgb_lambdarank", "seed_42", MAP)
    intervals, n_customers = paired_bootstrap(per_query, first, second, 200, 7, 0.95)
    assert n_customers == 4  # customer 1 has two queries and is drawn as one unit
    assert intervals[FIRST].estimate == pytest.approx(0.4)
    assert intervals[SECOND].estimate == pytest.approx(0.34)
    assert intervals[DIFFERENCE].estimate == pytest.approx(0.06)
    for interval in intervals.values():
        assert interval.low <= interval.estimate <= interval.high
    again, _ = paired_bootstrap(per_query, first, second, 200, 7, 0.95)
    assert again == intervals  # a fixed seed gives the same intervals
    other_seed, _ = paired_bootstrap(per_query, first, second, 200, 8, 0.95)
    assert other_seed[DIFFERENCE] != intervals[DIFFERENCE]
    # Swapping the arms flips the sign of the difference and of its interval.
    swapped, _ = paired_bootstrap(per_query, second, first, 200, 7, 0.95)
    assert swapped[DIFFERENCE].estimate == pytest.approx(-0.06)
    assert swapped[DIFFERENCE].low == pytest.approx(-intervals[DIFFERENCE].high)
    with pytest.raises(ValueError, match="confidence_level"):
        paired_bootstrap(per_query, first, second, 200, 7, 1.0)


def test_bootstrap_equals_the_bootstrap_of_the_diagnostics() -> None:
    # 300 random customers with 1-3 queries each; the same input for both implementations.
    rng = np.random.default_rng(3)
    customers = np.repeat(np.arange(300), rng.integers(1, 4, size=300))
    per_query = pl.DataFrame(
        {
            "customer_idx": customers,
            "ranker_ap": rng.random(customers.size),
            "baseline_ap": rng.random(customers.size),
        },
        schema={"customer_idx": pl.Int32, "ranker_ap": pl.Float64, "baseline_ap": pl.Float64},
    )
    intervals, n_customers = paired_bootstrap(per_query, "ranker_ap", "baseline_ap", 1000, 42, 0.95)
    # The diagnostics call the two systems "ranker" and "baseline".
    expected = cluster_bootstrap(customer_totals(per_query), 1000, 42, 0.95)
    pairs = (
        (FIRST, "lambdamart"),
        (SECOND, "rrf_baseline"),
        (DIFFERENCE, "lambdamart_minus_rrf_baseline"),
    )
    for mine, theirs in pairs:
        assert intervals[mine].estimate == expected[theirs].estimate
        assert intervals[mine].low == expected[theirs].low
        assert intervals[mine].high == expected[theirs].high
    assert n_customers == 300


def test_contrasts_on_one_seed_and_on_the_seed_means(
    example: tuple[list[FoldFiles], list[ArmRuns]],
) -> None:
    per_query = _per_query(example)
    rows = contrast_rows(per_query, SETTINGS)
    # 4 contrasts x 2 bases x 3 quantities.
    assert len(rows) == 24 and SETTINGS.bases == ("seed_42", "seed_mean")
    assert [(a, b) for a, b in CONTRASTS] == [
        ("deepfm_pointwise", "lgb_lambdarank"),
        ("deepfm_listwise", "lgb_lambdarank"),
        ("lgb_lambdarank", "lgb_binary"),
        ("deepfm_listwise", "deepfm_pointwise"),
    ]
    difference = {
        (row["first_arm"], row["second_arm"], row["basis"]): float(row["estimate"])
        for row in rows
        if row["quantity"] == DIFFERENCE
    }
    # Seed 42: 0.40 - 0.34. Seed means: pointwise (0.40 + 0.44) / 2, lambdarank (0.34 + 0.38) / 2.
    key = ("deepfm_pointwise", "lgb_lambdarank")
    assert difference[(*key, "seed_42")] == pytest.approx(0.06)
    assert difference[(*key, "seed_mean")] == pytest.approx(0.42 - 0.36)
    assert difference[("lgb_lambdarank", "lgb_binary", "seed_mean")] == pytest.approx(0.11)
    assert difference[("deepfm_listwise", "deepfm_pointwise", "seed_42")] == pytest.approx(-0.2)
    one = rows[2]
    assert (one["n_customers"], one["n_queries"], one["n_replicates"]) == ("4", "5", "200")
    assert float(one["ci_low"]) <= float(one["estimate"]) <= float(one["ci_high"])


# ---------------------------------------------------------------- the group table


def test_group_table_hand_computed(example: tuple[list[FoldFiles], list[ArmRuns]]) -> None:
    table = group_rows(_per_query(example), ARMS, ("seed_42", "seed_mean"), MAP)
    rows = {(row["basis"], row["fold"], row["query_group"]): row for row in table}
    # Pooled over both folds: recent = q1, q4; no history = q2; other = q3, q5.
    recent = rows[("seed_42", "pooled", "recent_purchase")]
    assert (recent["n_queries"], float(recent["query_share"])) == ("2", 0.4)
    assert float(recent["deepfm_pointwise"]) == pytest.approx(0.75)  # (0.5 + 1.0) / 2
    assert float(recent["lgb_lambdarank"]) == pytest.approx(0.45)
    assert float(recent["lgb_binary"]) == pytest.approx(0.2)
    other = rows[("seed_42", "pooled", "other")]
    assert float(other["deepfm_pointwise"]) == pytest.approx(0.25)  # (0.2 + 0.3) / 2
    assert float(other["deepfm_listwise"]) == pytest.approx(0.35)  # (0.1 + 0.6) / 2
    everyone = rows[("seed_42", "pooled", "all")]
    assert float(everyone["deepfm_pointwise"]) == pytest.approx(0.4)
    assert float(everyone["lgb_lambdarank"]) == pytest.approx(0.34)
    # Seed means: lambdarank q1 (0.4 + 0.6) / 2 and q4 0.5 -> 0.5.
    assert float(rows[("seed_mean", "pooled", "recent_purchase")]["lgb_lambdarank"]) == (
        pytest.approx(0.5)
    )
    # Per fold: V2 has no customer without history, so that row does not exist.
    assert ("seed_42", "V2", "no_history") not in rows
    assert rows[("seed_42", "V1", "no_history")]["n_queries"] == "1"
    assert float(rows[("seed_42", "V2", "all")]["deepfm_pointwise"]) == pytest.approx(0.65)
    assert list(rows)[:4] == [
        ("seed_42", "V1", "no_history"),
        ("seed_42", "V1", "recent_purchase"),
        ("seed_42", "V1", "other"),
        ("seed_42", "V1", "all"),
    ]


# ---------------------------------------------------------------- from files


def test_comparison_from_saved_files(example: tuple[list[FoldFiles], list[ArmRuns]]) -> None:
    folds, arms = example
    comparison = compare_arms(folds, arms, SETTINGS)
    per_query = _per_query(example)
    assert comparison.n_queries == 5
    assert comparison.arms == arm_rows(per_query, arms, METRICS)
    assert comparison.contrasts == contrast_rows(per_query, SETTINGS)
    assert comparison.groups == group_rows(per_query, ARMS, SETTINGS.bases, MAP)
    # A contrast needs both of its arms; an arm may be given only once.
    with pytest.raises(ValueError, match="that are not given"):
        compare_arms(folds, arms[:3], SETTINGS)
    with pytest.raises(ValueError, match="an arm name repeats"):
        compare_arms(folds, [*arms, arms[0]], SETTINGS)
    only_43 = [
        ArmRuns(arm.arm, arm.step_column, {43: arm.steps[43]}, {43: arm.files[43]}) for arm in arms
    ]
    with pytest.raises(ValueError, match="have no seed 42"):
        compare_arms(folds, only_43, SETTINGS)
    with pytest.raises(ValueError, match="same columns"):
        table_columns([{"a": "1"}, {"b": "2"}])
    with pytest.raises(ValueError, match="no rows"):
        table_columns([])


def test_the_caller_can_give_other_arms_and_contrasts(
    example: tuple[list[FoldFiles], list[ArmRuns]],
) -> None:
    # The extension (proposal §16.10) compares more arms with its own contrasts; the
    # numbers of an arm or of a contrast do not depend on which other arms are compared.
    folds, arms = example
    whole = compare_arms(folds, arms, SETTINGS)
    pair = ("deepfm_pointwise", "lgb_binary")
    part = compare_arms(folds, [arms[1], arms[3]], SETTINGS, contrasts=[pair])
    assert {row["arm"] for row in part.arms} == set(pair)
    assert part.arms == [row for row in whole.arms if row["arm"] in pair]
    assert {(row["first_arm"], row["second_arm"]) for row in part.contrasts} == {pair}
    # 2 bases x 3 quantities, and the same method as for a contrast of the 2x2 study.
    assert len(part.contrasts) == 6
    reversed_pair = compare_arms(folds, arms, SETTINGS, contrasts=[(pair[1], pair[0])])
    by_key = {(row["basis"], row["quantity"]): row for row in reversed_pair.contrasts}
    for row in part.contrasts:
        if row["quantity"] == "difference":
            other = by_key[(row["basis"], "difference")]
            assert float(row["estimate"]) == pytest.approx(-float(other["estimate"]), abs=1e-15)
            assert float(row["ci_low"]) == pytest.approx(-float(other["ci_high"]), abs=1e-15)
    # The group table has one column per given arm, in the given order.
    arm_columns = [name for name in part.groups[0] if name in ARMS]
    assert arm_columns == ["lgb_binary", "deepfm_pointwise"]


def test_a_wide_table_that_is_not_the_recorded_one_is_refused(tmp_path: Path) -> None:
    folds, arms = _write(tmp_path)
    record = folds[0].features_dir / "features.json"
    record.write_text(record.read_text().replace('"sha256": "', '"sha256": "0'))
    with pytest.raises(ValueError, match="sha256"):
        compare_arms(folds, arms, SETTINGS)
