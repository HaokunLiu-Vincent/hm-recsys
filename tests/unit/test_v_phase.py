"""Tests for hmrec.hpo.v_phase: the grid file, "k first, then parameters", and the driver.

Selection examples use 4 folds with 100 queries each, so the pooled MAP@12 is the
plain mean of the 4 fold values. Checkpoints are 100, 200, 300 trees.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ranker_example import REPO, build_workspace, tiny_inputs
from v_phase_example import N_QUERIES, RULE, TINY_GRID_RAW, FakeLauncher, run_and_record_trial

from hmrec.config import RankerTunedParams, load_yaml
from hmrec.csv_records import read_rows
from hmrec.hpo.selection import KEEP, SWITCH
from hmrec.hpo.v_phase import (
    BASELINE_SCRIPT,
    DATASET_SCRIPT,
    SAMPLE_SCRIPT,
    TRIAL_SCRIPT,
    ConfigScores,
    GridContext,
    VPhaseGrid,
    check_default_against_speed_test,
    check_grid,
    compare_systems,
    config_scores,
    decide_step,
    final_winner,
    goal_lines,
    load_m7_context,
    load_v_phase_grid,
    parse_v_phase_grid,
    record_step,
    refuse_to_launch,
    run_baseline,
    run_grid,
    run_minus_r6,
    step_columns,
    trial_arguments,
)
from hmrec.hpo.v_records import (
    AREA_SPEED_TEST,
    AREA_V_PHASE,
    SPEED_TEST_STEP,
    MetricRow,
    read_metric_rows,
)
from hmrec.hpo.v_trial import (
    TrialInputs,
    TrialSpec,
    build_and_save_dataset,
    load_planned_dataset,
    model_config,
    plan_dataset,
    ranker_config_id,
)
from hmrec.locks import file_sha256
from hmrec.paths import get_paths
from hmrec.ranker.train import save_model, train_ranker, training_params

REAL_GRID_FILE = REPO / "configs" / "hpo" / "v_phase.yaml"
FOLDS = ("V1", "V2", "V3", "V4")
TREES = (100, 200, 300)
TUNED = RankerTunedParams(31, 500, 40)


# ---------------------------------------------------------------- the grid file


def test_real_grid_is_the_pre_registered_one() -> None:
    grid = load_v_phase_grid(REAL_GRID_FILE)
    assert grid.target_folds == FOLDS
    assert grid.metric == "map@12" and grid.metrics == ("map@12", "recall@12", "ndcg@12")
    assert grid.stop_if_winner_uses_all_trees is True
    assert grid.g1_min_pooled_score == 0.032
    assert grid.experiments == ("rrf_baseline", "minus_r6")
    assert [step.name for step in grid.steps] == ["leaves_x_min_data", "negatives_per_positive"]
    # The priority order of step 1: smaller num_leaves first, 500 before 100.
    assert [tuple(c.values()) for c in grid.steps[0].candidates] == [
        (31, 500), (31, 100), (63, 500), (63, 100), (127, 500), (127, 100),
    ]  # fmt: skip
    assert [c["negatives_per_positive"] for c in grid.steps[1].candidates] == [40, 60, 80]


def test_real_configs_fit_together() -> None:
    # Reads only config files: the grid's metric is the rule's Ranker metric, the
    # target folds are 4 V folds, and every ratio of step 2 is sampled.
    context = load_m7_context(get_paths(), refuse_to_launch, "abc")
    assert context.inputs.metrics == ("map@12", "recall@12", "ndcg@12")
    assert context.inputs.ranker.tuned_defaults == TUNED
    assert context.scripts_dir.name == "scripts"
    for script in (SAMPLE_SCRIPT, DATASET_SCRIPT, TRIAL_SCRIPT, BASELINE_SCRIPT):
        assert (context.scripts_dir / script).is_file()


def _raw_grid(**changes: Any) -> dict[str, Any]:
    raw = load_yaml(REAL_GRID_FILE)
    raw.update(changes)
    return raw


def test_grid_file_errors() -> None:
    with pytest.raises(ValueError, match="phase must be 'V'"):
        parse_v_phase_grid(_raw_grid(phase="T"))
    with pytest.raises(ValueError, match="unknown \\['minus_r5'\\]"):
        parse_v_phase_grid(_raw_grid(experiments=["minus_r5"]))
    with pytest.raises(ValueError, match="unknown keys"):
        parse_v_phase_grid(_raw_grid(extra=1))
    bad_key = [{"name": "s", "kind": "select", "order": "listed",
                "candidates": [{"learning_rate": 0.1}, {"learning_rate": 0.2}]}]  # fmt: skip
    with pytest.raises(ValueError, match="are not tuned parameters"):
        parse_v_phase_grid(_raw_grid(steps=bad_key))
    nearest = [{"name": "s", "kind": "select", "order": "nearest_to_current",
                "candidates": [{"num_leaves": 31}, {"num_leaves": 63}]}]  # fmt: skip
    with pytest.raises(ValueError, match="only listed select steps"):
        parse_v_phase_grid(_raw_grid(steps=nearest))


def test_check_grid_refuses_t_folds_and_unsampled_ratios(tmp_path: Path) -> None:
    inputs = tiny_inputs(tmp_path)  # samples ratios 2 and 4 only
    grid = load_v_phase_grid(REAL_GRID_FILE)
    with pytest.raises(ValueError, match="ratio 40 is not in sampling.ratios"):
        check_grid(grid, RULE, inputs)
    with_t = replace(grid, target_folds=("V2", "V3", "V4", "T1"))
    with pytest.raises(ValueError, match="M7 predicts and evaluates only V folds"):
        check_grid(with_t, RULE, inputs)
    with pytest.raises(ValueError, match="must name 4 folds"):
        check_grid(replace(grid, target_folds=("V1",)), RULE, inputs)
    with pytest.raises(ValueError, match="selection_rule.yaml says"):
        check_grid(replace(grid, metric="ndcg@12"), RULE, inputs)


# ---------------------------------------------------------------- k first, then parameters


def _rows(config_id: str, curve: dict[int, list[float]]) -> list[MetricRow]:
    """Metric rows of one configuration: number of trees -> the 4 folds' MAP@12."""
    return [
        MetricRow("full", config_id, fold, k, N_QUERIES, {"map@12": value}, None)
        for k, values in curve.items()
        for fold, value in zip(FOLDS, values, strict=True)
    ]


GRID = replace(load_v_phase_grid(REAL_GRID_FILE), extra_metrics=())


def _scores(config_id: str, curve: dict[int, list[float]]) -> ConfigScores:
    return config_scores(_rows(config_id, curve), "full", TUNED, config_id, GRID, TREES, 0.01)


def _flat(values: dict[int, float]) -> dict[int, list[float]]:
    """The same value on all 4 folds."""
    return {k: [value] * 4 for k, value in values.items()}


DEFAULT = _flat({100: 0.0300, 200: 0.0301, 300: 0.0290})


def test_every_configuration_gets_its_own_number_of_trees() -> None:
    default = _scores("default", DEFAULT)
    # 0.0300 is within 1% of the best (0.0301), so the smaller k = 100 is chosen.
    assert default.choice.num_trees == 100 and default.choice.best_num_trees == 200
    assert default.scores.pooled_score == pytest.approx(0.0300)
    assert default.scores.fold_scores == dict.fromkeys(FOLDS, 0.0300)
    assert default.equal_fold == pytest.approx(0.0300)
    assert default.choice.uses_all_trees is False


def test_candidate_wins_at_its_own_number_of_trees_and_the_run_stops() -> None:
    default = _scores("default", DEFAULT)
    # The candidate is bad with 100 trees but best with 300: it is compared at 300,
    # the default at 100. 0.0310 vs 0.0300 = +3.3% and better on all 4 folds.
    candidate = _scores("slow", _flat({100: 0.0200, 200: 0.0280, 300: 0.0310}))
    assert candidate.choice.num_trees == 300 and candidate.choice.uses_all_trees is True
    result = decide_step("step", default, [candidate], RULE, stop_if_winner_uses_all_trees=True)
    assert result.decision.action == SWITCH and result.winner.config_id == "slow"
    assert result.decision.checks[0].relative_improvement == pytest.approx(1 / 30)
    # The winner needs all trees: more trees might be better still -> stop for a human.
    assert result.stop is True
    relaxed = decide_step("step", default, [candidate], RULE, stop_if_winner_uses_all_trees=False)
    assert relaxed.stop is False


def test_the_number_of_trees_is_fixed_before_the_comparison() -> None:
    default = _scores("default", DEFAULT)
    # A candidate is NOT compared at its best point but at k*, the smallest k within 1%
    # of its best. Candidate a: best 0.0306 at 300; 0.03035 at 200 is within 1% of it, so
    # k* = 200 and it is compared with 0.03035 = +1.17% over the default: a pass.
    # Candidate b: best 0.0304 at 300 (+1.3%, would pass); but k* = 200 with 0.0302 =
    # +0.67%: a fail.
    passing = _scores("a", _flat({100: 0.0200, 200: 0.03035, 300: 0.0306}))
    assert passing.choice.num_trees == 200
    assert decide_step("s", default, [passing], RULE, True).decision.action == SWITCH
    failing = _scores("b", _flat({100: 0.0200, 200: 0.0302, 300: 0.0304}))
    assert failing.choice.num_trees == 200
    result = decide_step("s", default, [failing], RULE, True)
    assert result.decision.action == KEEP and result.winner.config_id == "default"


def test_two_percent_better_but_only_two_folds_better_keeps_the_default() -> None:
    default = _scores("default", DEFAULT)
    # At 100 trees: +0.0024 on V1 and V2, equal on V3 and V4 -> pooled +4%, 2 folds better.
    uneven = {100: [0.0324, 0.0324, 0.0300, 0.0300], 200: [0.03] * 4, 300: [0.03] * 4}
    candidate = _scores("uneven", uneven)
    assert candidate.choice.num_trees == 100
    result = decide_step("s", default, [candidate], RULE, True)
    check = result.decision.checks[0]
    assert check.relative_improvement == pytest.approx(0.04) and check.folds_better == 2
    assert result.decision.action == KEEP and result.stop is False


def test_two_passing_candidates_within_one_percent_follow_the_priority_order() -> None:
    default = _scores("default", DEFAULT)
    first = _scores("first", _flat({100: 0.0310, 200: 0.0310, 300: 0.0310}))
    second = _scores("second", _flat({100: 0.0312, 200: 0.0312, 300: 0.0312}))
    result = decide_step("s", default, [first, second], RULE, True)
    # Both pass; 0.0310 is within 1% of 0.0312, so the one listed first wins.
    assert [check.passes for check in result.decision.checks] == [True, True]
    assert result.winner.config_id == "first"
    clear = _scores("clear", _flat({100: 0.0330, 200: 0.0330, 300: 0.0330}))
    assert decide_step("s", default, [first, clear], RULE, True).winner.config_id == "clear"


def test_a_missing_checkpoint_is_an_error_not_a_shorter_curve() -> None:
    rows = _rows("c", _flat({100: 0.03, 200: 0.03, 300: 0.03}))
    without_one = [row for row in rows if not (row.fold == "V3" and row.num_trees == 300)]
    with pytest.raises(ValueError, match="complete are \\[100, 200\\]"):
        config_scores(without_one, "full", TUNED, "c", GRID, TREES, 0.01)


def test_step_rows_are_recorded_once(tmp_path: Path) -> None:
    default = _scores("default", DEFAULT)
    candidate = _scores("slow", _flat({100: 0.0200, 200: 0.0280, 300: 0.0310}))
    result = decide_step("step", default, [candidate], RULE, True)
    path = tmp_path / "v_phase_steps.csv"
    record_step(path, result, FOLDS, "abc")
    record_step(path, result, FOLDS, "abc")  # the same decision again: nothing is added
    rows = read_rows(path, step_columns(FOLDS))
    assert [(row["role"], row["config_id"], row["selected"]) for row in rows] == [
        ("default", "default", "False"),
        ("candidate", "slow", "True"),
    ]
    assert rows[1]["num_trees"] == "300" and rows[1]["uses_all_trees"] == "True"
    assert rows[1]["passes"] == "True" and rows[0]["passes"] == ""
    assert float(rows[1]["score_V3"]) == 0.0310 and rows[0]["best_num_trees"] == "200"
    # A decision that changed after it was written must never be overwritten silently.
    other = decide_step("step", default, [], RULE, True)
    with pytest.raises(ValueError, match="recorded with winner"):
        record_step(path, other, FOLDS, "abc")


# ---------------------------------------------------------------- the driver (fake processes)

FOLD_OFFSET = {"V1": 0.0, "V2": 0.001, "V3": 0.002, "V4": 0.003}


def flat_curves(tuned: RankerTunedParams, fold: str, k: int) -> float:
    """(8, 5) is clearly best; every curve is flat, so k* is the first checkpoint (10)."""
    base = {(4, 5): 0.0300, (4, 2): 0.0302, (8, 5): 0.0310}[
        (tuned.num_leaves, tuned.min_data_in_leaf)
    ]
    bonus = 0.0001 if tuned.negatives_per_positive == 4 else 0.0
    return base + bonus + FOLD_OFFSET[fold]


def rising_curves(tuned: RankerTunedParams, fold: str, k: int) -> float:
    """Like flat_curves, but (8, 5) keeps improving up to the last checkpoint (20)."""
    value = flat_curves(tuned, fold, k)
    return value + (0.002 if (tuned.num_leaves, k) == (8, 20) else 0.0)


def _context(inputs: TrialInputs, launcher: FakeLauncher) -> GridContext:
    grid: VPhaseGrid = parse_v_phase_grid(TINY_GRID_RAW)
    check_grid(grid, RULE, inputs)
    # FakeLauncher trains nothing, so there is no model to compare with the speed test.
    return GridContext(
        inputs, grid, RULE, Path("scripts"), launcher, "abc", check_speed_test_model=False
    )


def test_grid_runs_every_trial_once_and_selects(tmp_path: Path) -> None:
    inputs = tiny_inputs(tmp_path)
    launcher = FakeLauncher(inputs, flat_curves)
    context = _context(inputs, launcher)
    results = run_grid(context, None)
    assert [result.step for result in results] == ["leaves_x_min_data", "negatives_per_positive"]
    first, second = results
    # Step 1: (4, 2) is +0.67% (fails), (8, 5) is +3.2% on all folds (passes).
    assert first.decision.action == SWITCH
    assert first.winner.tuned == RankerTunedParams(8, 5, 2)
    assert [check.passes for check in first.decision.checks] == [False, True]
    assert first.winner.choice.num_trees == 10 and first.stop is False
    # Step 2 starts from the winner; ratio 4 is only +0.3% better: the default stays.
    assert second.default.tuned == RankerTunedParams(8, 5, 2)
    assert second.decision.action == KEEP and second.winner.tuned == RankerTunedParams(8, 5, 2)
    # Processes: 3 configurations x 4 folds, then 1 new configuration x 4 folds.
    assert launcher.count(TRIAL_SCRIPT) == 12 + 4
    # Every (target fold, ratio) Dataset once; every training fold sampled once (F1 ... V3).
    assert launcher.count(DATASET_SCRIPT) == 4 + 4
    sampled = [args["fold"][0] for name, args in launcher.calls if name == SAMPLE_SCRIPT]
    assert sorted(sampled) == ["F1", "F2", "F3", "F4", "V1", "V2", "V3"]
    assert all(
        args["ratios"] == ["2", "4"] for name, args in launcher.calls if name == SAMPLE_SCRIPT
    )
    trial_trees = {args["num-trees"][0] for name, args in launcher.calls if name == TRIAL_SCRIPT}
    assert trial_trees == {"20"}  # every grid model is trained to num_trees
    steps = read_rows(inputs.paths.v_phase_steps_file, step_columns(FOLDS))
    assert len(steps) == 3 + 2

    # Running again starts nothing: every trial is recorded.
    calls_before = len(launcher.calls)
    assert run_grid(context, None)[1].winner.config_id == second.winner.config_id
    assert len(launcher.calls) == calls_before
    assert final_winner(context).tuned == RankerTunedParams(8, 5, 2)


def test_grid_stops_when_the_winner_needs_all_trees(tmp_path: Path) -> None:
    inputs = tiny_inputs(tmp_path)
    launcher = FakeLauncher(inputs, rising_curves)
    context = _context(inputs, launcher)
    results = run_grid(context, None)
    assert len(results) == 1 and results[0].stop is True
    assert results[0].winner.choice.num_trees == 20
    # Step 2 never ran: no ratio-4 trial was started.
    assert launcher.count(TRIAL_SCRIPT) == 12
    with pytest.raises(ValueError, match="stopped early"):
        final_winner(context)


def test_last_step_and_unfinished_grid(tmp_path: Path) -> None:
    inputs = tiny_inputs(tmp_path)
    launcher = FakeLauncher(inputs, flat_curves)
    context = _context(inputs, launcher)
    assert len(run_grid(context, "leaves_x_min_data")) == 1
    assert launcher.count(TRIAL_SCRIPT) == 12
    # The final winner is read from finished results only: it must not start trials.
    with pytest.raises(RuntimeError, match="the grid is not finished"):
        final_winner(context)
    with pytest.raises(ValueError, match="unknown step"):
        run_grid(context, "no_such_step")


def test_grid_refuses_to_run_while_the_histogram_mode_is_undecided(tmp_path: Path) -> None:
    undecided = {"mode": "undecided", "comparison_trees": 6, "warmup_trees": 2,
                 "min_row_wise_speedup": 0.15}  # fmt: skip
    inputs = tiny_inputs(tmp_path, histogram=undecided)
    launcher = FakeLauncher(inputs, flat_curves)
    with pytest.raises(ValueError, match="still 'undecided'"):
        run_grid(_context(inputs, launcher), None)
    assert launcher.calls == []


def test_baseline_ablation_and_goals(tmp_path: Path) -> None:
    inputs = tiny_inputs(tmp_path)
    launcher = FakeLauncher(inputs, flat_curves, baseline=0.02)
    context = _context(inputs, launcher)
    run_grid(context, None)
    run_baseline(context, AREA_V_PHASE, FOLDS)
    run_baseline(context, AREA_V_PHASE, FOLDS)  # already there: nothing is started
    assert launcher.count(BASELINE_SCRIPT) == 4
    winner = final_winner(context)
    calls_before = len(launcher.calls)
    run_minus_r6(context, winner)
    ablation_calls = launcher.calls[calls_before:]
    samples = [args for name, args in ablation_calls if name == SAMPLE_SCRIPT]
    # The ablation samples only the selected ratio, for every training fold again.
    assert len(samples) == 7 and all(args["experiment"] == ["minus_r6"] for args in samples)
    assert all(args["ratios"] == ["2"] for args in samples)
    trials = [args for name, args in ablation_calls if name == TRIAL_SCRIPT]
    assert len(trials) == 4 and all(args["experiment"] == ["minus_r6"] for args in trials)
    # It trains exactly the selected number of trees (10), not num_trees (20).
    assert all(args["num-trees"] == ["10"] for args in trials)

    metric_rows = read_metric_rows(inputs.paths.v_phase_metrics_file, context.grid.metrics)
    comparison = compare_systems(context, winner, metric_rows)
    assert comparison.num_trees == 10 and comparison.folds_above_baseline == 4
    assert comparison.baseline.pooled_score == pytest.approx(0.02)
    assert comparison.full.pooled_score == pytest.approx(0.0310 + 0.0015)
    assert comparison.minus_r6 is not None
    lines = goal_lines(comparison, RULE, context.grid.g1_min_pooled_score)
    assert lines[0].startswith("G1 met") and lines[1].startswith("G3 met")
    assert goal_lines(comparison, RULE, 0.05)[0].startswith("G1 NOT met")
    below = replace(comparison, folds_above_baseline=2)
    assert goal_lines(below, RULE, 0.03)[1].startswith("G3 NOT met")


def test_grid_stops_right_after_the_default_trial_if_the_speed_test_is_missing(
    tmp_path: Path,
) -> None:
    # With the model check on (the real driver), the default configuration on the speed
    # test's fold (V1) runs first and is compared with the speed test's model at once.
    # Here no speed test ran: the grid stops after that ONE trial instead of going on.
    inputs = tiny_inputs(tmp_path)
    launcher = FakeLauncher(inputs, flat_curves)
    context = replace(_context(inputs, launcher), check_speed_test_model=True)
    with pytest.raises(ValueError, match="no recorded trial of configuration"):
        run_grid(context, None)
    assert launcher.count(TRIAL_SCRIPT) == 1
    (trial,) = [args for name, args in launcher.calls if name == TRIAL_SCRIPT]
    assert trial["target"] == ["V1"] and trial["num-leaves"] == ["4"]


def test_trial_arguments_match_the_trial_script() -> None:
    spec = TrialSpec(AREA_V_PHASE, "full", "s", "V1", TUNED, 1000, "row_wise", True)
    arguments = trial_arguments(spec)
    assert arguments[arguments.index("--num-leaves") + 1] == "31"
    assert arguments[arguments.index("--min-data-in-leaf") + 1] == "500"
    assert arguments[arguments.index("--ratio") + 1] == "40"
    assert "--no-eval" not in arguments
    assert "--no-eval" in trial_arguments(replace(spec, evaluate=False))
    script = (REPO / "scripts" / "m7_train_eval.py").read_text(encoding="utf-8")
    for argument in arguments:
        if argument.startswith("--"):
            assert f'"{argument}"' in script


# ---------------------------------------------------------------- first trees = speed test


def _speed_test_and_grid_models(tmp_path: Path) -> tuple[GridContext, TrialSpec]:
    """Real tiny models: the speed test's (10 trees) and the grid's default (20 trees) on V1."""
    inputs = build_workspace(tmp_path)
    build_and_save_dataset(plan_dataset(inputs, "full", "V1", 2, "c"), 64)
    defaults = inputs.ranker.tuned_defaults
    speed = TrialSpec(
        AREA_SPEED_TEST, "full", SPEED_TEST_STEP, "V1", defaults, 10, "col_wise", True
    )
    grid = TrialSpec(
        AREA_V_PHASE, "full", "leaves_x_min_data", "V1", defaults, 20, "col_wise", True
    )
    run_and_record_trial(inputs, speed)
    run_and_record_trial(inputs, grid)
    context = GridContext(
        inputs, parse_v_phase_grid(TINY_GRID_RAW), RULE, Path("scripts"), refuse_to_launch, "abc"
    )
    return context, speed


def test_default_model_starts_with_the_speed_test_model(tmp_path: Path) -> None:
    context, _ = _speed_test_and_grid_models(tmp_path)
    assert context.check_speed_test_model is True  # the default of the real driver
    common = check_default_against_speed_test(context)
    assert len(common) == 64


def test_a_different_speed_test_model_stops_the_grid(tmp_path: Path) -> None:
    # Everything that defines a model is in its config id, so in practice only another
    # environment (LightGBM, libomp) can make the trees differ. Simulated here: the speed
    # test's model file is replaced by a model with other trees, and its recorded sha256
    # is changed to match, as if the speed test had really produced that model.
    context, speed = _speed_test_and_grid_models(tmp_path)
    inputs = context.inputs
    config_id = ranker_config_id(
        model_config(inputs.ranker, speed.tuned, "full", speed.histogram_mode)
    )
    model_file = inputs.paths.ranker_model_file(AREA_SPEED_TEST, "full", config_id, "V1")
    old_sha256 = file_sha256(model_file)
    other_params = training_params(inputs.ranker, replace(speed.tuned, num_leaves=8), "col_wise", 2)
    dataset, _ = load_planned_dataset(plan_dataset(inputs, "full", "V1", 2, "c"), other_params)
    other_model, _ = train_ranker(dataset, other_params, 10, 10)
    model_file.unlink()
    new_sha256 = save_model(other_model, model_file)
    trial_file = inputs.paths.m7_speed_test_file
    trial_file.write_text(trial_file.read_text().replace(old_sha256, new_sha256))
    with pytest.raises(ValueError, match="first 10 trees .* differ from the speed test"):
        check_default_against_speed_test(context)


def test_a_model_file_that_is_not_the_recorded_one_stops_the_grid(tmp_path: Path) -> None:
    context, speed = _speed_test_and_grid_models(tmp_path)
    inputs = context.inputs
    config_id = ranker_config_id(
        model_config(inputs.ranker, speed.tuned, "full", speed.histogram_mode)
    )
    model_file = inputs.paths.ranker_model_file(AREA_V_PHASE, "full", config_id, "V1")
    with model_file.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="is not the model recorded"):
        check_default_against_speed_test(context)
