"""Tests for hpo.lgb_arms_run: the order of the arms' trainings, k*, and continuing a run.

The trainings run in this process on the tiny M8 workspace (final_example), which
has M7's Dataset for V4 only; so the run here has one fold, V4. Tiny lock: 10
trees, checkpoints every 10; the binary arm's first seed is trained to 20 trees.
"""

from pathlib import Path

from final_example import copy_workspace
from ranker_example import METRICS

from hmrec.hpo.lgb_arm_records import (
    ARM_BINARY,
    ARM_LAMBDARANK,
    append_metric_row,
    append_trial_row,
    read_metric_rows,
    read_trial_rows,
)
from hmrec.hpo.lgb_arms import (
    ArmContext,
    ArmSpec,
    load_arm_context,
    metric_rows,
    run_arm_trial,
    trial_row,
)
from hmrec.hpo.lgb_arms_run import (
    ArmRequest,
    ArmsPlan,
    binary_choice,
    ensure_trainings,
    request_arguments,
    run_arms,
)
from hmrec.ranker.runtime import read_lightgbm_runtime

FOLD = "V4"


def _plan(root: Path, calls: list[ArmRequest]) -> tuple[ArmsPlan, ArmContext]:
    paths = copy_workspace(root).paths
    runtime = read_lightgbm_runtime()
    context = load_arm_context(paths, runtime, METRICS)
    every = context.locked.config.checkpoint_every

    def launch(request: ArmRequest) -> None:
        """What scripts/m9_lgb_trial.py does, in this process."""
        calls.append(request)
        last = request.num_trees
        evaluated = tuple(range(every, last + 1, every)) if request.evaluate_all else (last,)
        spec = ArmSpec(request.arm, request.seed, request.target_fold, last, evaluated)
        outcome = run_arm_trial(context, spec, "commit")
        for row in metric_rows(spec, outcome, METRICS, "commit", "now"):
            append_metric_row(paths.m9_lgb_metrics_file, METRICS, row)
        measured = {"total_seconds": 1.0, "process_max_rss_gb": 0.5}
        stamp = {"git_commit": "commit", "git_dirty": "False", "finished_at": "now"}
        append_trial_row(
            paths.m9_lgb_trials_file, trial_row(context, spec, outcome, runtime, measured, stamp)
        )

    plan = ArmsPlan(
        paths=paths,
        folds=(FOLD,),
        seeds=(42, 43, 44),
        metrics=METRICS,
        locked_num_trees=context.locked.num_trees,
        checkpoint_every=every,
        binary_num_trees=20,
        tolerance=0.01,
        launcher=launch,
    )
    return plan, context


def test_request_arguments() -> None:
    assert request_arguments(ArmRequest(ARM_LAMBDARANK, 43, "V2", 500, False)) == [
        "--arm", "lambdarank", "--seed", "43", "--target", "V2", "--num-trees", "500",
    ]  # fmt: skip
    assert request_arguments(ArmRequest(ARM_BINARY, 42, "V1", 1000, True))[-1] == "--evaluate-all"


def test_arms_run_in_order_and_continue_where_they_stopped(tmp_path: Path) -> None:
    calls: list[ArmRequest] = []
    plan, _ = _plan(tmp_path, calls)
    assert plan.binary_checkpoints == (10, 20)
    choice = run_arms(plan)
    # lambdarank: seeds 43, 44 only (42 is the M7 record), the locked 10 trees.
    assert calls[:2] == [
        ArmRequest(ARM_LAMBDARANK, 43, FOLD, 10, False),
        ArmRequest(ARM_LAMBDARANK, 44, FOLD, 10, False),
    ]
    # binary: the first seed to 20 trees with every checkpoint, then the others to k*.
    assert calls[2] == ArmRequest(ARM_BINARY, 42, FOLD, 20, True)
    assert choice.num_trees in (10, 20) and choice == binary_choice(plan)
    assert calls[3:] == [
        ArmRequest(ARM_BINARY, 43, FOLD, choice.num_trees, False),
        ArmRequest(ARM_BINARY, 44, FOLD, choice.num_trees, False),
    ]
    trials = read_trial_rows(plan.paths.m9_lgb_trials_file)
    assert [(row["arm"], row["seed"]) for row in trials] == [
        ("lambdarank", "43"),
        ("lambdarank", "44"),
        ("binary", "42"),
        ("binary", "43"),
        ("binary", "44"),
    ]
    metrics = read_metric_rows(plan.paths.m9_lgb_metrics_file, METRICS)
    binary_42 = [
        row["num_trees"] for row in metrics if (row["arm"], row["seed"]) == ("binary", "42")
    ]
    assert binary_42 == ["10", "20"]
    # The seeds of an arm give different scores or at least different models.
    hashes = {row["model_sha256"] for row in trials}
    assert len(hashes) == 5
    # A second run starts nothing.
    again = run_arms(plan)
    assert len(calls) == 5 and again == choice
    assert ensure_trainings(plan, [ArmRequest(ARM_BINARY, 43, FOLD, choice.num_trees, False)]) == 0
