"""Check configs/locks/ranker.lock.yaml against the recorded V phase (proposal §9.6).

The lock file is written by hand, once. This module derives what it MUST say,
without reading it:

    1. replay the selection rule over results/v_phase_metrics.csv, step by step
       (replay_selection; no trial is run, a missing trial is an error);
    2. take the winner of the last step: its tuned values, its number of trees k*,
       its config_id and its recorded scores;
    3. take the fixed settings from the configs the V phase ran with
       (configs/ranker.yaml, num_threads of configs/base.yaml);
    4. take the LightGBM and libomp versions from the winner's trial rows in
       results/v_phase.csv.

check_lock_matches_v_phase compares that, value by value, with the lock file
and fails on any difference. It is called by tests/unit/test_v_lock.py.

This is a check, not a source of configuration: code after the lock gets the
Ranker configuration only from hmrec.ranker.locked and never from here.
Every input is a small file in git (configs/ and results/); nothing under data/
or artifacts/ is read, and no label is read.
"""

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from hmrec.hpo.v_phase import GridContext, StepResult, refuse_to_launch, run_step
from hmrec.hpo.v_records import AREA_V_PHASE, find_trial, read_trial_rows, results_files
from hmrec.locks import RANKER_SAMPLING_METHOD, RULE_SELECTION, RankerLock
from hmrec.ranker.ablation import FULL

RUNTIME_ROW_COLUMNS = ("lightgbm_version", "libomp_version")


def replay_selection(context: GridContext) -> list[StepResult]:
    """Decide every step of the grid again from the recorded results only.

    Nothing is launched (a trial without a recorded row raises) and nothing is
    written. The comparison of the default model with the speed test's model is
    skipped, because it reads model files under artifacts/.

    Returns:
        One StepResult per step of configs/hpo/v_phase.yaml, in order.

    Raises:
        RuntimeError: If a trial of the grid has no recorded row.
        ValueError: If a step's winner uses all trees: the grid then stopped
            for a human decision and there is no final configuration.
    """
    replay = replace(context, launch=refuse_to_launch, check_speed_test_model=False)
    current = replay.inputs.ranker.tuned_defaults
    results = []
    for step in replay.grid.steps:
        result = run_step(replay, step, current)
        if result.stop:
            raise ValueError(
                f"step {step.name}: the winner's number of trees is the last checkpoint; "
                "the grid stopped there and has no final configuration"
            )
        results.append(result)
        current = result.winner.tuned
    return results


def _winner_runtime(context: GridContext, config_id: str) -> dict[str, str]:
    """LightGBM and libomp versions of the winner's trials; all folds must agree."""
    trial_file, _ = results_files(context.inputs.paths, AREA_V_PHASE)
    rows = read_trial_rows(trial_file)
    seen = set()
    for fold_name in context.grid.target_folds:
        row = find_trial(rows, FULL, config_id, fold_name)
        if row is None:
            raise ValueError(f"{trial_file}: no trial row of {config_id} on {fold_name}")
        seen.add(tuple(row[column] for column in RUNTIME_ROW_COLUMNS))
    if len(seen) != 1:
        raise ValueError(
            f"{trial_file}: the trials of {config_id} ran with different "
            f"{RUNTIME_ROW_COLUMNS}: {sorted(seen)}"
        )
    return dict(zip(RUNTIME_ROW_COLUMNS, seen.pop(), strict=True))


def expected_lock_content(context: GridContext) -> dict[str, dict[str, Any]]:
    """What the Ranker lock must say, derived from the recorded V phase (module docstring).

    Returns:
        Section name -> values, in the shape of lock_content: selection,
        runtime, training_window, sampling, model, scoring.
    """
    winner = replay_selection(context)[-1].winner
    ranker, paths = context.inputs.ranker, context.inputs.paths
    _, metrics_file = results_files(paths, AREA_V_PHASE)
    runtime = _winner_runtime(context, winner.config_id)
    return {
        "selection": {
            "basis": RULE_SELECTION,
            "config_id": winner.config_id,
            "rule_selected_config_id": winner.config_id,
            "default_kept": winner.tuned == ranker.tuned_defaults,
            "v_phase_record.source": metrics_file.relative_to(paths.project_root).as_posix(),
            "v_phase_record.metric": context.grid.metric,
            "v_phase_record.pooled": winner.scores.pooled_score,
            "v_phase_record.folds": dict(winner.scores.fold_scores),
        },
        "runtime": {**runtime, "num_threads": context.inputs.base.num_threads},
        "training_window": {"window_weeks": ranker.window_weeks},
        "sampling": {
            "method": RANKER_SAMPLING_METHOD,
            "seed": ranker.sampling.seed,
            "negatives_per_positive": winner.tuned.negatives_per_positive,
            "row_group_rows": ranker.sampling.row_group_rows,
        },
        "model": {
            "num_leaves": winner.tuned.num_leaves,
            "min_data_in_leaf": winner.tuned.min_data_in_leaf,
            "num_trees": winner.choice.num_trees,
            "seed": ranker.seed,
            "histogram_mode": ranker.histogram.mode,
            "dataset_params": dict(ranker.dataset_params),
            "train_params": dict(ranker.train_params),
        },
        "scoring": {"checkpoint_every": ranker.checkpoint_every, "top_k": ranker.top_k},
    }


def lock_content(lock: RankerLock) -> dict[str, dict[str, Any]]:
    """The values of a parsed Ranker lock, in the shape of expected_lock_content.

    Left out: the `lock` stamp, selection.decision_record (a pointer to
    docs/decisions.md) and `references` (checked against the files themselves by
    hmrec.ranker.locked.check_referenced_locks).
    """
    selection, record = lock.selection, lock.selection.v_phase_record
    return {
        "selection": {
            "basis": selection.basis,
            "config_id": selection.config_id,
            "rule_selected_config_id": selection.rule_selected_config_id,
            "default_kept": selection.default_kept,
            "v_phase_record.source": record.source,
            "v_phase_record.metric": record.metric,
            "v_phase_record.pooled": record.pooled,
            "v_phase_record.folds": dict(record.folds),
        },
        "runtime": {
            "lightgbm_version": lock.runtime.lightgbm_version,
            "libomp_version": lock.runtime.libomp_version,
            "num_threads": lock.runtime.num_threads,
        },
        "training_window": {"window_weeks": lock.window_weeks},
        "sampling": {
            "method": lock.sampling.method,
            "seed": lock.sampling.seed,
            "negatives_per_positive": lock.sampling.negatives_per_positive,
            "row_group_rows": lock.sampling.row_group_rows,
        },
        "model": {
            "num_leaves": lock.model.num_leaves,
            "min_data_in_leaf": lock.model.min_data_in_leaf,
            "num_trees": lock.model.num_trees,
            "seed": lock.model.seed,
            "histogram_mode": lock.model.histogram_mode,
            "dataset_params": dict(lock.model.dataset_params),
            "train_params": dict(lock.model.train_params),
        },
        "scoring": {
            "checkpoint_every": lock.scoring.checkpoint_every,
            "top_k": lock.scoring.top_k,
        },
    }


def content_differences(
    in_lock: Mapping[str, Mapping[str, Any]], expected: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Every value that differs between two contents, as "section.key: lock ..., expected ...".

    Values are compared through their JSON text. That is exact for numbers (the
    lock file quotes the recorded scores with all their digits, so there is
    nothing to round) and it tells 0, 0.0 and false apart, which == does not:
    they are different settings and give different config ids.
    """
    differences = []
    for section in sorted(set(in_lock) | set(expected)):
        lock_values, expected_values = in_lock.get(section, {}), expected.get(section, {})
        for key in sorted(set(lock_values) | set(expected_values)):
            lock_value, expected_value = lock_values.get(key), expected_values.get(key)
            lock_text = json.dumps(lock_value, sort_keys=True)
            expected_text = json.dumps(expected_value, sort_keys=True)
            if lock_text != expected_text:
                differences.append(
                    f"{section}.{key}: {lock_value!r} in the lock, {expected_value!r} from the "
                    "recorded V phase"
                )
    return differences


def check_lock_matches_v_phase(lock: RankerLock, context: GridContext) -> None:
    """Fail unless the Ranker lock says exactly what the recorded V phase decided.

    Args:
        lock: The parsed configs/locks/ranker.lock.yaml.
        context: The M7 context (configs and paths; hmrec.hpo.v_phase.load_m7_context).

    Raises:
        ValueError: If any value differs; the message lists every difference.
        RuntimeError: If a trial of the grid has no recorded row.
    """
    differences = content_differences(lock_content(lock), expected_lock_content(context))
    if differences:
        raise ValueError(
            "configs/locks/ranker.lock.yaml differs from the recorded V phase "
            "(results/v_phase*.csv, configs/ranker.yaml, configs/base.yaml): "
            + "; ".join(differences)
        )
