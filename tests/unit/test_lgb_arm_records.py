"""Tests for hpo.lgb_arm_records: the two tables and the number-of-trees rule of the binary arm.

Hand example (two folds, V1 with 100 queries and V2 with 300), binary arm, seed 42:
    100 trees: V1 0.20, V2 0.10 -> pooled (100 x 0.20 + 300 x 0.10) / 400 = 0.125
    200 trees: V1 0.22, V2 0.12 -> pooled 0.145
    300 trees: V1 0.21, V2 0.1205 -> pooled 0.142875 (1.47% below the best)
    with a 1% tolerance only 200 trees is "about equal" to the best -> k* = 200;
    with a 20% tolerance 100 trees is too (13.8% below) -> k* = 100.
"""

from pathlib import Path

import pytest

from hmrec.hpo.lgb_arm_records import (
    ARM_BINARY,
    ARM_LAMBDARANK,
    ARMS,
    TRIAL_COLUMNS,
    append_metric_row,
    append_trial_row,
    choose_binary_num_trees,
    is_recorded,
    metric_columns,
    pooled_curve,
    read_metric_rows,
    read_trial_rows,
)

METRICS = ("map@12", "recall@12")
N_QUERIES = {"V1": 100, "V2": 300}
CURVE = {
    100: {"V1": 0.20, "V2": 0.10},
    200: {"V1": 0.22, "V2": 0.12},
    300: {"V1": 0.21, "V2": 0.1205},
}


def _metric_rows(arm: str, seed: int, values: dict[int, dict[str, float]]) -> list[dict[str, str]]:
    return [
        {
            "arm": arm,
            "seed": str(seed),
            "config_id": "abc",
            "fold": fold,
            "num_trees": str(num_trees),
            "n_queries": str(N_QUERIES[fold]),
            "map@12": repr(value),
            "recall@12": repr(2 * value),
            "train_metric": "",
            "git_commit": "c",
            "finished_at": "t",
        }
        for num_trees, by_fold in values.items()
        for fold, value in by_fold.items()
    ]


ROWS = _metric_rows(ARM_BINARY, 42, CURVE)


def test_tables_are_appended_and_read_back(tmp_path: Path) -> None:
    assert ARMS == ("lambdarank", "binary")
    metrics_file, trials_file = tmp_path / "metrics.csv", tmp_path / "trials.csv"
    assert read_metric_rows(metrics_file, METRICS) == [] and read_trial_rows(trials_file) == []
    for row in ROWS:
        append_metric_row(metrics_file, METRICS, row)
    assert read_metric_rows(metrics_file, METRICS) == ROWS
    assert metric_columns(METRICS)[:6] == (
        "arm", "seed", "config_id", "fold", "num_trees", "n_queries",
    )  # fmt: skip
    trial = dict.fromkeys(TRIAL_COLUMNS, "x")
    trial.update({"arm": ARM_LAMBDARANK, "seed": "43", "fold": "V1"})
    append_trial_row(trials_file, trial)
    rows = read_trial_rows(trials_file)
    assert is_recorded(rows, ARM_LAMBDARANK, 43, "V1")
    assert not is_recorded(rows, ARM_LAMBDARANK, 44, "V1")
    assert not is_recorded(rows, ARM_BINARY, 43, "V1")
    append_trial_row(trials_file, trial)
    with pytest.raises(ValueError, match="recorded twice"):
        is_recorded(read_trial_rows(trials_file), ARM_LAMBDARANK, 43, "V1")
    assert "process_max_rss_gb" in TRIAL_COLUMNS and "libomp_version" in TRIAL_COLUMNS


def test_pooled_curve_hand_computed() -> None:
    curve = pooled_curve(ROWS, ARM_BINARY, 42, ("V1", "V2"), "map@12")
    assert curve == pytest.approx({100: 0.125, 200: 0.145, 300: 0.142875})
    # Rows of another arm, another seed or a fold that is not asked for do not count.
    others = _metric_rows(ARM_LAMBDARANK, 42, {100: {"V1": 0.9, "V2": 0.9}})
    others += _metric_rows(ARM_BINARY, 43, {100: {"V1": 0.9, "V2": 0.9}})
    assert pooled_curve(ROWS + others, ARM_BINARY, 42, ("V1", "V2"), "map@12") == curve
    # A number of trees that one fold lacks is not in the curve.
    partial = _metric_rows(ARM_BINARY, 42, {400: {"V1": 0.3}})
    assert sorted(pooled_curve(ROWS + partial, ARM_BINARY, 42, ("V1", "V2"), "map@12")) == [
        100,
        200,
        300,
    ]
    with pytest.raises(ValueError, match="recorded twice"):
        pooled_curve(ROWS + ROWS[:1], ARM_BINARY, 42, ("V1", "V2"), "map@12")


def test_number_of_trees_rule_of_the_binary_arm() -> None:
    folds, checkpoints = ("V1", "V2"), (100, 200, 300)
    choice = choose_binary_num_trees(ROWS, 42, folds, "map@12", 0.01, checkpoints)
    assert (choice.num_trees, choice.best_num_trees) == (200, 200)
    assert choice.score == pytest.approx(0.145) and not choice.uses_all_trees
    wide = choose_binary_num_trees(ROWS, 42, folds, "map@12", 0.2, checkpoints)
    assert (wide.num_trees, wide.best_num_trees) == (100, 200)
    # If the curve still rises at the end, k* is the last checkpoint and that is flagged.
    rising = _metric_rows(ARM_BINARY, 42, {100: CURVE[100], 200: CURVE[200]})
    last = choose_binary_num_trees(rising, 42, folds, "map@12", 0.01, (100, 200))
    assert last.num_trees == 200 and last.uses_all_trees
    # Every checkpoint of every fold must be there: the first seed must be finished.
    with pytest.raises(ValueError, match=r"no complete metric rows at \[400\]"):
        choose_binary_num_trees(ROWS, 42, folds, "map@12", 0.01, (100, 200, 300, 400))
    with pytest.raises(ValueError, match="not finished"):
        choose_binary_num_trees(ROWS, 43, folds, "map@12", 0.01, checkpoints)
