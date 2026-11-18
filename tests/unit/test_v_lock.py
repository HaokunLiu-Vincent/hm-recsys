"""The Ranker lock must say exactly what the recorded V phase decided (M7, proposal §9.6).

Expected content of configs/locks/ranker.lock.yaml, derived WITHOUT the lock file
(hmrec.hpo.v_lock.expected_lock_content):

    1. replay the selection rule over results/v_phase_metrics.csv, both steps
       (no trial is run);
    2. the winner of the last step gives the tuned values, the number of trees,
       the config_id and the recorded scores;
    3. the fixed settings come from configs/ranker.yaml and configs/base.yaml;
    4. the LightGBM and libomp versions come from the winner's rows in
       results/v_phase.csv.

Then every value of the lock file is changed, one at a time, and each change
must be refused by one of the checks. Inputs are small files in git only
(configs/, results/, docs/); nothing under data/ or artifacts/ is read.
"""

import copy
import json
import math
from dataclasses import replace
from datetime import date
from typing import Any

import pytest

from hmrec.config import RankerConfig, load_ranker_config, load_yaml
from hmrec.hpo.v_lock import (
    check_lock_matches_v_phase,
    content_differences,
    expected_lock_content,
    lock_content,
    replay_selection,
)
from hmrec.hpo.v_phase import GridContext, load_m7_context, refuse_to_launch
from hmrec.hpo.v_records import AREA_V_PHASE, find_trial, read_trial_rows, results_files
from hmrec.locks import load_ranker_lock, parse_ranker_lock
from hmrec.paths import ProjectPaths, get_paths
from hmrec.ranker.identity import model_config
from hmrec.ranker.locked import (
    check_lock_config_id,
    check_referenced_locks,
    locked_ranker_config,
    locked_tuned_params,
)

# Values nothing can be compared with: when the file was written and the pointer to
# docs/decisions.md. tests/unit/test_locks.py pins them (the date, the commit, and that
# the decision record is a heading of docs/decisions.md), so a change is still caught.
PINNED_ELSEWHERE = {
    ("lock", "locked_on"),
    ("lock", "git_commit"),
    ("selection", "decision_record"),
}


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return get_paths()


@pytest.fixture(scope="module")
def context(paths: ProjectPaths) -> GridContext:
    return load_m7_context(paths, refuse_to_launch, "test")


@pytest.fixture(scope="module")
def expected(context: GridContext) -> dict[str, dict[str, Any]]:
    return expected_lock_content(context)


@pytest.fixture(scope="module")
def ranker_yaml(paths: ProjectPaths) -> RankerConfig:
    return load_ranker_config(paths.ranker_config_file)


# ---------------------------------------------------------------- the replay


def test_replay_keeps_the_default_in_both_steps(context: GridContext) -> None:
    results = replay_selection(context)
    assert [result.step for result in results] == ["leaves_x_min_data", "negatives_per_positive"]
    # No candidate passed the §4.2 rule in either step (docs/decisions.md 2026-11-18).
    assert [result.decision.action for result in results] == ["keep", "keep"]
    assert all(not check.passes for result in results for check in result.decision.checks)
    winner = results[-1].winner
    assert winner.tuned == context.inputs.ranker.tuned_defaults
    assert winner.config_id == "ca1a3cdfce49"
    # k* = the smallest k within 1% of the best pooled MAP@12 (which is at 1000 trees).
    assert (winner.choice.num_trees, winner.choice.best_num_trees) == (500, 1000)
    assert not winner.choice.uses_all_trees


def test_replay_starts_no_process_and_fails_on_a_missing_trial(context: GridContext) -> None:
    # A grid with one more candidate than was run: its trials have no recorded row,
    # so the replay must stop instead of training anything.
    step = context.grid.steps[1]
    more = replace(step, candidates=(*step.candidates, {"negatives_per_positive": 100}))
    grid = replace(context.grid, steps=(context.grid.steps[0], more))
    with pytest.raises(RuntimeError, match="has not run yet"):
        replay_selection(replace(context, grid=grid))


# ---------------------------------------------------------------- lock == replay


def test_the_lock_file_equals_the_replayed_selection(
    paths: ProjectPaths, context: GridContext, expected: dict[str, dict[str, Any]]
) -> None:
    lock = load_ranker_lock(paths.ranker_lock_file)
    assert content_differences(lock_content(lock), expected) == []
    check_lock_matches_v_phase(lock, context)  # the same comparison, as the public check
    # Spelled out for the reader: what the replay gives.
    assert expected["model"]["num_leaves"] == 31
    assert expected["model"]["min_data_in_leaf"] == 500
    assert expected["model"]["num_trees"] == 500
    assert expected["sampling"]["negatives_per_positive"] == 40
    assert expected["selection"]["default_kept"] is True
    assert expected["runtime"] == {
        "lightgbm_version": "4.7.0",
        "libomp_version": "22.1.4",
        "num_threads": 6,
    }


def test_the_locked_settings_are_the_recorded_settings_of_the_winning_trials(
    paths: ProjectPaths, context: GridContext, ranker_yaml: RankerConfig
) -> None:
    # Every trial row stores the full model_config it was trained with (config_json).
    # The lock must give exactly that mapping for each of the 4 folds.
    lock = load_ranker_lock(paths.ranker_lock_file)
    config = locked_ranker_config(lock, ranker_yaml)
    from_lock = model_config(config, locked_tuned_params(lock), "full", lock.model.histogram_mode)
    trial_file, _ = results_files(paths, AREA_V_PHASE)
    rows = read_trial_rows(trial_file)
    for fold_name in context.grid.target_folds:
        row = find_trial(rows, "full", lock.selection.config_id, fold_name)
        assert row is not None
        assert json.loads(row["config_json"]) == from_lock
        assert json.dumps(from_lock, sort_keys=True) == row["config_json"]
        # The grid trained 1000 trees; the first 500 are the locked model.
        assert int(row["num_trees"]) >= lock.model.num_trees
        assert row["git_dirty"] == "False"


def test_the_recorded_score_is_the_pooled_map_of_the_four_folds(
    paths: ProjectPaths,
) -> None:
    lock = load_ranker_lock(paths.ranker_lock_file)
    record = lock.selection.v_phase_record
    # Query counts of V1-V4 (results/v_phase_metrics.csv); pooled = sum(n * score) / sum(n).
    queries = {"V1": 82802, "V2": 74833, "V3": 71094, "V4": 72035}
    pooled = math.fsum(queries[fold] * record.folds[fold] for fold in queries) / sum(
        queries.values()
    )
    assert record.pooled == pytest.approx(pooled, rel=1e-12)
    assert record.pooled == pytest.approx(0.032713, abs=5e-7)


# ---------------------------------------------------------------- any change is refused


def _leaves(node: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Paths of all values of a nested mapping (a list is one value)."""
    if not isinstance(node, dict):
        return [prefix]
    found = []
    for key, value in node.items():
        found.extend(_leaves(value, (*prefix, str(key))))
    return found


def _changed(value: Any) -> Any:
    """Another value of the same type, as close to the original as possible."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return math.nextafter(value, math.inf)  # the next float: the smallest change
    if isinstance(value, list):
        return [*value, 1]
    if isinstance(value, date):
        return value.replace(day=value.day - 1)
    text = str(value)
    is_hex = len(text) in (12, 40, 64) and all(c in "0123456789abcdef" for c in text)
    if is_hex:
        # Still a well-formed id / hash, so the check behind the format check must catch it.
        return text[:-1] + ("0" if text[-1] != "0" else "1")
    return text + "x"


def _is_refused(
    raw: dict[str, Any],
    expected: dict[str, dict[str, Any]],
    paths: ProjectPaths,
    ranker_yaml: RankerConfig,
) -> bool:
    """True if any check refuses this lock content.

    The checks, in the order code would meet them: the strict parser, the
    config_id of the lock's own settings, the sha256 of the referenced lock
    files, the comparison with the recorded V phase.
    """
    try:
        lock = parse_ranker_lock(raw)
        check_lock_config_id(lock, locked_ranker_config(lock, ranker_yaml))
        check_referenced_locks(lock, paths)
    except (ValueError, FileNotFoundError):
        return True
    return content_differences(lock_content(lock), expected) != []


def test_changing_any_value_of_the_lock_is_refused(
    paths: ProjectPaths, expected: dict[str, dict[str, Any]], ranker_yaml: RankerConfig
) -> None:
    original = load_yaml(paths.ranker_lock_file)
    assert not _is_refused(copy.deepcopy(original), expected, paths, ranker_yaml)
    leaves = _leaves(original)
    # lock 3 + selection 12 (5 values, source, metric, pooled, 4 folds) + references 6
    # (path and sha256 of the features, channels and twotower locks) + runtime 3 +
    # training_window 1 + sampling 4 + model 36 (5 values, 7 bin parameters, 24 training
    # parameters) + scoring 2 = 67: every value of the file is visited.
    assert len(leaves) == 67
    assert sum(leaf[0] == "references" for leaf in leaves) == 6
    accepted = []
    for leaf in leaves:
        if leaf in PINNED_ELSEWHERE:
            continue
        raw = copy.deepcopy(original)
        node = raw
        for key in leaf[:-1]:
            node = node[key]
        node[leaf[-1]] = _changed(node[leaf[-1]])
        if not _is_refused(raw, expected, paths, ranker_yaml):
            accepted.append(".".join(leaf))
    assert accepted == []


def test_the_public_check_names_what_differs(paths: ProjectPaths, context: GridContext) -> None:
    raw = load_yaml(paths.ranker_lock_file)
    raw["model"]["num_trees"] = 600  # a checkpoint, but not the one the rule chose
    raw["runtime"]["libomp_version"] = "23.0.0"
    with pytest.raises(ValueError) as error:
        check_lock_matches_v_phase(parse_ranker_lock(raw), context)
    message = str(error.value)
    assert "model.num_trees: 600 in the lock, 500 from the recorded V phase" in message
    assert "runtime.libomp_version: '23.0.0' in the lock, '22.1.4' from" in message


def test_values_are_compared_exactly_and_by_type() -> None:
    same = {"model": {"lambda_l1": 0.0, "eval_at": [12], "deterministic": True}}
    assert content_differences(same, copy.deepcopy(same)) == []
    for key, other in (("lambda_l1", 0), ("eval_at", [12.0]), ("deterministic", 1)):
        changed = copy.deepcopy(same)
        changed["model"][key] = other
        assert len(content_differences(changed, same)) == 1
    # A value that only one side has is a difference too.
    assert content_differences({"model": {}}, same) != []
    assert content_differences(same, {"scoring": {"top_k": 12}}) != []
