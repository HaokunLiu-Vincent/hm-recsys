"""Tests for hmrec.t_eval.compare on the tiny workspace (proposal §16.12).

The DeepFM values come from formal trainings on the tiny T folds. The "M8 record"
is made up here: per-query values of one seed, saved where M8 saved its own, and
the rows of results/t_phase_metrics.csv that belong to them. Expected values are
computed in the tests with plain sums over the saved per-query files.
"""

import math
import statistics
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from t_eval_example import (
    METRICS,
    NOW,
    SEEDS,
    T_FOLDS,
    build_t_eval_workspace,
    context_of,
    record_rehearsal,
    run_and_record,
    t_eval_config,
)

from hmrec.deepfm.compare import CompareSettings
from hmrec.deepfm.trial import TrialInputs
from hmrec.t_eval.compare import (
    check_m8_values,
    compare_on_t,
    deepfm_arm_runs,
    m8_arm_runs,
    per_query_table,
    single_seed_rows,
    v_and_t_rows,
)
from hmrec.t_eval.gate import begin_formal_run
from hmrec.t_eval.plan import STAGE_FORMAL, TEvalContext, TrialKey

LOCKED_ID = "ca1a3cdfce49"
CONFIG = t_eval_config()
ARM, M8_ARM = CONFIG.arm, CONFIG.m8_record_arm
SETTINGS = CompareSettings(
    metrics=METRICS,
    seed=42,
    recent_purchase_days=30,
    n_replicates=200,
    bootstrap_seed=7,
    confidence_level=0.95,
    wide_columns=(),
)
TOLERANCE = 1e-12


def write_m8_record(inputs: TrialInputs) -> list[dict[str, str]]:
    """Save made-up per-query values of the M8 record; returns its rows of t_phase_metrics."""
    rng = np.random.default_rng(5)
    paths = inputs.paths
    rows = []
    for fold in T_FOLDS:
        query_ids = pl.read_parquet(paths.queries_file(fold))["query_id"]
        values = {metric: rng.random(query_ids.len()) for metric in METRICS}
        table = pl.DataFrame({"query_id": query_ids, "fold": [fold] * query_ids.len(), **values})
        # Like M8's file: also rows of another number of trees, which must be left out.
        both = pl.concat(
            [
                table.with_columns(pl.lit(100, dtype=pl.Int32).alias("num_trees")),
                table.with_columns(pl.lit(500, dtype=pl.Int32).alias("num_trees")),
            ]
        )
        file = paths.ranker_per_query_file("t_phase", "full", LOCKED_ID, fold)
        file.parent.mkdir(parents=True, exist_ok=True)
        both.write_parquet(file)
        row = {"experiment": "full", "config_id": LOCKED_ID, "fold": fold, "num_trees": "500"}
        for metric in METRICS:
            row[metric] = repr(math.fsum(values[metric].tolist()) / query_ids.len())
        rows.append(row)
    return rows


def v_rows() -> list[dict[str, str]]:
    """Made-up rows of results/m9_ext_arms.csv: what the comparison reads of the V table."""
    return [
        {"arm": ARM, "row": "mean", "map@12": "0.5"},
        {"arm": ARM, "row": "seed_42", "map@12": "0.51"},
        {"arm": M8_ARM, "row": "seed_42", "map@12": "0.4"},
        {"arm": M8_ARM, "row": "mean", "map@12": "0.41"},
        {"arm": "deepfm_pointwise", "row": "mean", "map@12": "0.3"},
    ]


@pytest.fixture(scope="module")
def evaluated(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[TrialInputs, TEvalContext, list[dict[str, str]]]:
    """A workspace after the formal run, with a made-up M8 record."""
    inputs = build_t_eval_workspace(tmp_path_factory.mktemp("t_eval_compare"))
    context = context_of(inputs)
    record_rehearsal(context)
    begin_formal_run(context, NOW, "scripts/m9_t_eval.py")
    for seed in SEEDS:
        for fold in T_FOLDS:
            run_and_record(inputs, context, TrialKey(STAGE_FORMAL, seed, fold))
    return inputs, context, write_m8_record(inputs)


def _saved(inputs: TrialInputs, seed: int, fold: str, metric: str) -> list[float]:
    directory = inputs.paths.deepfm_trial_dir("m9_t", CONFIG.config_id, seed, fold)
    return pl.read_parquet(directory / "per_query.parquet").sort("query_id")[metric].to_list()


def test_the_arm_table_holds_the_seeds_their_summary_and_the_m8_record(
    evaluated: tuple[TrialInputs, TEvalContext, list[dict[str, str]]],
) -> None:
    inputs, context, m8_rows = evaluated
    result = compare_on_t(context, SETTINGS, LOCKED_ID, m8_rows, v_rows(), TOLERANCE)
    assert result.n_queries == 30 * len(T_FOLDS)
    assert [(row["arm"], row["row"]) for row in result.arms] == [
        (ARM, "seed_42"),
        (ARM, "seed_43"),
        (ARM, "mean"),
        (ARM, "std"),
        (ARM, "range"),
        (M8_ARM, "seed_42"),
    ]
    assert all(tuple(row) == tuple(result.arms[0]) for row in result.arms)
    # Pooled MAP@12 of a seed = the plain mean over the 120 saved per-query values.
    pooled = {}
    for seed in SEEDS:
        values = [v for fold in T_FOLDS for v in _saved(inputs, seed, fold, "map@12")]
        pooled[seed] = math.fsum(values) / len(values)
    by_row = {(row["arm"], row["row"]): row for row in result.arms}
    for seed in SEEDS:
        assert float(by_row[(ARM, f"seed_{seed}")]["map@12"]) == pytest.approx(pooled[seed], abs=0)
    seed_values = list(pooled.values())
    assert float(by_row[(ARM, "mean")]["map@12"]) == math.fsum(seed_values) / len(seed_values)
    assert float(by_row[(ARM, "std")]["map@12"]) == statistics.stdev(seed_values)
    expected_range = (max(seed_values) - min(seed_values)) / min(seed_values)
    assert float(by_row[(ARM, "range")]["map@12"]) == expected_range
    # MAP@12 per week: of a seed, and the seed mean in the mean row.
    t2 = [math.fsum(_saved(inputs, seed, "T2", "map@12")) / 30 for seed in SEEDS]
    assert float(by_row[(ARM, "seed_43")]["map@12|T2"]) == t2[1]
    assert float(by_row[(ARM, "mean")]["map@12|T2"]) == math.fsum(t2) / 2
    assert by_row[(ARM, "std")]["map@12|T2"] == ""
    # The M8 record: its recorded value per week, and the 500-tree rows only.
    m8 = by_row[(M8_ARM, "seed_42")]
    assert (m8["step_column"], m8["step"]) == ("num_trees", "500")
    for recorded in m8_rows:
        assert float(m8[f"map@12|{recorded['fold']}"]) == pytest.approx(
            float(recorded["map@12"]), abs=TOLERANCE
        )


def test_the_contrast_is_deepfm_minus_the_m8_record_on_both_bases(
    evaluated: tuple[TrialInputs, TEvalContext, list[dict[str, str]]],
) -> None:
    _, context, m8_rows = evaluated
    result = compare_on_t(context, SETTINGS, LOCKED_ID, m8_rows, v_rows(), TOLERANCE)
    assert [(row["basis"], row["quantity"]) for row in result.contrasts] == [
        (basis, quantity)
        for basis in ("seed_42", "seed_mean")
        for quantity in ("first", "second", "difference")
    ]
    assert {(row["first_arm"], row["second_arm"]) for row in result.contrasts} == {(ARM, M8_ARM)}
    by_row = {(row["arm"], row["row"]): float(row["map@12"]) for row in result.arms}
    by_contrast = {(row["basis"], row["quantity"]): row for row in result.contrasts}
    mean_difference = by_contrast[("seed_mean", "difference")]
    # The estimate is the difference of the two pooled values of the arm table.
    assert float(mean_difference["estimate"]) == pytest.approx(
        by_row[(ARM, "mean")] - by_row[(M8_ARM, "seed_42")], abs=1e-15
    )
    assert float(by_contrast[("seed_42", "difference")]["estimate"]) == pytest.approx(
        by_row[(ARM, "seed_42")] - by_row[(M8_ARM, "seed_42")], abs=1e-15
    )
    # The M8 record has one seed: its "seed mean" is that seed.
    assert (
        by_contrast[("seed_mean", "second")]["estimate"]
        == (by_contrast[("seed_42", "second")]["estimate"])
    )
    assert float(mean_difference["ci_low"]) <= float(mean_difference["estimate"])
    assert float(mean_difference["estimate"]) <= float(mean_difference["ci_high"])
    assert (mean_difference["n_replicates"], mean_difference["bootstrap_seed"]) == ("200", "7")
    # 20 customers have queries in the four T folds; all 120 queries are used.
    assert (mean_difference["n_customers"], mean_difference["n_queries"]) == ("20", "120")


def test_v_and_t_side_by_side(
    evaluated: tuple[TrialInputs, TEvalContext, list[dict[str, str]]],
) -> None:
    _, context, m8_rows = evaluated
    result = compare_on_t(context, SETTINGS, LOCKED_ID, m8_rows, v_rows(), TOLERANCE)
    by_row = {(row["arm"], row["row"]): row["map@12"] for row in result.arms}
    assert result.v_and_t == [
        {
            "arm": ARM,
            "metric": "map@12",
            "v_basis": "seed_mean",
            "v_n_seeds": "2",
            "v_value": "0.5",
            "t_basis": "seed_mean",
            "t_n_seeds": "2",
            "t_value": by_row[(ARM, "mean")],
        },
        {
            "arm": M8_ARM,
            "metric": "map@12",
            "v_basis": "seed_42",
            "v_n_seeds": "1",
            "v_value": "0.4",
            "t_basis": "seed_42",
            "t_n_seeds": "1",
            "t_value": by_row[(M8_ARM, "seed_42")],
        },
        {
            "arm": M8_ARM,
            "metric": "map@12",
            "v_basis": "seed_mean",
            "v_n_seeds": "2",
            "v_value": "0.41",
            "t_basis": "seed_42",
            "t_n_seeds": "1",
            "t_value": by_row[(M8_ARM, "seed_42")],
        },
    ]
    with pytest.raises(ValueError, match="has 0 rows"):
        v_and_t_rows(context, v_rows()[:1], result.arms, "map@12")


def test_m8_values_that_do_not_give_the_recorded_metrics_are_refused(
    evaluated: tuple[TrialInputs, TEvalContext, list[dict[str, str]]],
) -> None:
    _, context, m8_rows = evaluated
    changed = [dict(row) for row in m8_rows]
    changed[2]["map@12"] = repr(float(changed[2]["map@12"]) + 1e-9)
    with pytest.raises(ValueError, match="T3: the saved per-query values of M8"):
        compare_on_t(context, SETTINGS, LOCKED_ID, changed, v_rows(), TOLERANCE)
    with pytest.raises(ValueError, match="has 0 rows"):
        compare_on_t(context, SETTINGS, LOCKED_ID, m8_rows[:3], v_rows(), TOLERANCE)
    m8 = m8_arm_runs(context, LOCKED_ID)
    per_query = per_query_table(context, (deepfm_arm_runs(context), m8), METRICS)
    row = single_seed_rows(per_query, m8, METRICS)[0]
    check_m8_values(row, m8_rows, LOCKED_ID, T_FOLDS, "map@12", TOLERANCE)
    with pytest.raises(ValueError, match="expected exactly one"):
        single_seed_rows(per_query, deepfm_arm_runs(context), METRICS)


def test_the_single_seed_basis_must_be_the_seed_of_the_m8_record(
    evaluated: tuple[TrialInputs, TEvalContext, list[dict[str, str]]],
) -> None:
    _, context, m8_rows = evaluated
    other = CompareSettings(METRICS, 43, 30, 200, 7, 0.95, ())
    with pytest.raises(ValueError, match="must be the M8 record's seed 42"):
        compare_on_t(context, other, LOCKED_ID, m8_rows, v_rows(), TOLERANCE)


def test_a_missing_formal_training_stops_the_comparison(tmp_path: Path) -> None:
    inputs = build_t_eval_workspace(tmp_path)
    context = context_of(inputs)
    m8_rows = write_m8_record(inputs)
    with pytest.raises(FileNotFoundError):
        compare_on_t(context, SETTINGS, LOCKED_ID, m8_rows, v_rows(), TOLERANCE)
