"""The lock files must say exactly what the recorded D phase decided (M5).

Expected configuration of each channel, derived here WITHOUT the lock files:

    1. replay the §4.2 selection rule over the trials in results/d_phase.csv
       (hpo.union_precheck.select_channels; no trial is run);
    2. R6: apply the manual decision of configs/m4_union_check.yaml
       (`manual_selection`: replay 0.5 instead of the rule's 0);
    3. R3: apply the outcome of the pre-registered M4 rule
       (`variants.r3_active_pool`: active_pool_only true).

The lock files must hold these configs, their trial ids must be the recorded
trials, and the scores they quote must be the recorded scores.
"""

from pathlib import Path

import pytest

from hmrec.config import load_selection_rule, load_union_check_config
from hmrec.data.folds import load_fold_manifest
from hmrec.hpo.grid import PhaseGrid, apply_overrides, load_phase_grid
from hmrec.hpo.lock_check import read_recorded_trial
from hmrec.hpo.runner import make_trial_id, parsed_config
from hmrec.hpo.union_precheck import SelectedChannel, select_channels
from hmrec.locks import (
    ALL_CHANNELS,
    BASE_CHECKPOINT,
    F1_START_CHECKPOINT,
    MANUAL_DECISION,
    R6_CHANNEL,
    RULE_EXECUTION,
    RULE_SELECTION,
)
from hmrec.paths import ProjectPaths, get_paths
from hmrec.recall.locked import LockedRecall, load_locked_recall, locked_checkpoint_file

R3_CHANNEL = "r3_sibling"
R3_VARIANT = "r3_active_pool"


@pytest.fixture(scope="module")
def paths() -> ProjectPaths:
    return get_paths()


@pytest.fixture(scope="module")
def grid(paths: ProjectPaths) -> PhaseGrid:
    return load_phase_grid(paths.d_phase_grid_file)


@pytest.fixture(scope="module")
def d_folds(paths: ProjectPaths) -> list[str]:
    return [fold.name for fold in load_fold_manifest(paths.fold_manifest_file).by_phase("D")]


@pytest.fixture(scope="module")
def locked(paths: ProjectPaths) -> LockedRecall:
    return load_locked_recall(paths)


@pytest.fixture(scope="module")
def rule_selection(
    paths: ProjectPaths, grid: PhaseGrid, d_folds: list[str]
) -> dict[str, SelectedChannel]:
    """Step 1 only: what the §4.2 rule selected for each channel."""
    rule = load_selection_rule(paths.selection_rule_file)
    selected = select_channels(paths, ALL_CHANNELS, grid, rule, d_folds)
    return {choice.channel: choice for choice in selected}


@pytest.fixture(scope="module")
def expected_configs(
    paths: ProjectPaths, rule_selection: dict[str, SelectedChannel]
) -> dict[str, dict]:
    """Steps 1-3: the configuration every channel must be locked with."""
    settings = load_union_check_config(paths.union_check_config_file)
    configs = {channel: dict(choice.raw_config) for channel, choice in rule_selection.items()}
    # Only R6 has a manual decision and only R3 has a rule outcome; anything else
    # in the config file would be a change this test does not know about.
    assert set(settings.manual_selection) == {R6_CHANNEL}
    assert set(settings.variants[R3_VARIANT]) == {R3_CHANNEL}
    configs[R6_CHANNEL] = apply_overrides(
        configs[R6_CHANNEL], settings.manual_selection[R6_CHANNEL]
    )
    configs[R3_CHANNEL] = apply_overrides(
        configs[R3_CHANNEL], settings.variants[R3_VARIANT][R3_CHANNEL]
    )
    return configs


@pytest.mark.parametrize("channel", ALL_CHANNELS)
def test_locked_config_is_the_replayed_selection(
    channel: str, locked: LockedRecall, expected_configs: dict[str, dict]
) -> None:
    lock = locked.channel(channel)
    # Same raw keys and values ...
    assert lock.config == expected_configs[channel]
    # ... and the same config after parsing (what the channel really runs with).
    assert parsed_config(channel, lock.config) == parsed_config(channel, expected_configs[channel])


@pytest.mark.parametrize("channel", ALL_CHANNELS)
def test_locked_trial_id_is_the_id_of_the_locked_config(
    channel: str, locked: LockedRecall, expected_configs: dict[str, dict]
) -> None:
    lock = locked.channel(channel)
    assert lock.trial_id == make_trial_id(channel, lock.config)
    assert lock.trial_id == make_trial_id(channel, expected_configs[channel])


@pytest.mark.parametrize("channel", ALL_CHANNELS)
def test_locked_trial_and_scores_are_the_recorded_ones(
    channel: str, locked: LockedRecall, paths: ProjectPaths, grid: PhaseGrid, d_folds: list[str]
) -> None:
    lock = locked.channel(channel)
    # read_recorded_trial fails unless the results file named by the lock holds this
    # trial with the same parsed config, metric, pooled score and fold scores.
    record = read_recorded_trial(paths, lock, d_folds, grid)
    assert record.trial_id == lock.trial_id
    assert record.summary.pooled == lock.d_phase_record.pooled
    assert record.summary.fold_scores == lock.d_phase_record.folds
    assert list(lock.d_phase_record.folds) == d_folds
    # No locked trial may come from a run with uncommitted code.
    assert record.git_dirty is False


def test_rule_selected_trial_and_basis_of_every_channel(
    locked: LockedRecall, rule_selection: dict[str, SelectedChannel]
) -> None:
    for channel in ALL_CHANNELS:
        lock = locked.channel(channel)
        assert lock.rule_selected_trial_id == rule_selection[channel].record.trial_id
    basis = {channel: locked.channel(channel).selection_basis for channel in ALL_CHANNELS}
    assert basis == {
        "r1_popularity": RULE_SELECTION,
        "r2_repurchase": RULE_SELECTION,
        R3_CHANNEL: RULE_EXECUTION,
        "r4_itemcf": RULE_SELECTION,
        "r5_wals": RULE_SELECTION,
        R6_CHANNEL: MANUAL_DECISION,
    }


def test_r3_and_r6_differ_from_the_rule_selection_in_one_key_only(
    locked: LockedRecall, rule_selection: dict[str, SelectedChannel]
) -> None:
    def changed_keys(channel: str) -> dict[str, tuple[object, object]]:
        rule_config = parsed_config(channel, rule_selection[channel].raw_config)
        lock_config = parsed_config(channel, locked.channel(channel).config)
        keys = sorted(set(rule_config) | set(lock_config))
        return {
            key: (rule_config.get(key), lock_config.get(key))
            for key in keys
            if rule_config.get(key) != lock_config.get(key)
        }

    # R3: the D-phase runs had no such option (None); gamma and tau are not re-tuned.
    assert changed_keys(R3_CHANNEL) == {"active_pool_only": (None, True)}
    # R6: the rule selected (E_update 2, replay 0); (2, 50%) is the manual decision.
    assert changed_keys(R6_CHANNEL) == {"replay": (0.0, 0.5)}


def test_recorded_checkpoint_paths_are_the_paths_of_the_locked_config(
    locked: LockedRecall, paths: ProjectPaths
) -> None:
    # locked_checkpoint_file fails if the path written in the lock file is not the
    # file that hmrec.paths gives for the locked base_epoch / trial_id.
    base = locked_checkpoint_file(paths, locked.twotower, BASE_CHECKPOINT)
    f1_start = locked_checkpoint_file(paths, locked.twotower, F1_START_CHECKPOINT)
    assert base == paths.twotower_checkpoint_file(4)
    assert f1_start == paths.twotower_chain_checkpoint_file("835ef201ee4d", "D4")


def test_base_window_ends_where_the_first_d_fold_starts(
    locked: LockedRecall, paths: ProjectPaths
) -> None:
    # The base model must not have seen D1: its window is [end_anchor - 182d, end_anchor)
    # and end_anchor is the D1 anchor (proposal §6.2).
    first_d_fold = load_fold_manifest(paths.fold_manifest_file).by_phase("D")[0]
    assert locked.twotower.channel.config["base_window"]["end_anchor"] == first_d_fold.anchor


def test_lock_files_do_not_point_outside_the_repository(locked: LockedRecall) -> None:
    for channel in ALL_CHANNELS:
        source = Path(locked.channel(channel).d_phase_record.source)
        assert not source.is_absolute()
        assert source.parts[0] == "results"
