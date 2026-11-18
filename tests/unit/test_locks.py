"""Tests for hmrec.locks: strict parsing of the lock files and the sha256 check.

The parsing tests start from the REAL lock files (configs/locks/) and break one
thing at a time, so every error path is checked against the real layout.
"""

import copy
import hashlib
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from hmrec.config import load_yaml
from hmrec.locks import (
    BASE_CHECKPOINT,
    F1_START_CHECKPOINT,
    MANUAL_DECISION,
    RULE_EXECUTION,
    RULE_SELECTION,
    STATISTICAL_CHANNELS,
    check_file_sha256,
    file_sha256,
    load_channels_lock,
    load_ranker_lock,
    load_twotower_lock,
    parse_channels_lock,
    parse_ranker_lock,
    parse_twotower_lock,
)

LOCKS = Path(__file__).resolve().parents[2] / "configs" / "locks"
CHANNELS_FILE = LOCKS / "channels.lock.yaml"
TWOTOWER_FILE = LOCKS / "twotower.lock.yaml"
RANKER_FILE = LOCKS / "ranker.lock.yaml"


def _channels_raw() -> dict[str, Any]:
    return copy.deepcopy(load_yaml(CHANNELS_FILE))


def _twotower_raw() -> dict[str, Any]:
    return copy.deepcopy(load_yaml(TWOTOWER_FILE))


def _ranker_raw() -> dict[str, Any]:
    return copy.deepcopy(load_yaml(RANKER_FILE))


# ---------------------------------------------------------------- the real files


def test_channels_lock_holds_r1_to_r5_with_their_basis() -> None:
    lock = load_channels_lock(CHANNELS_FILE)
    assert lock.stamp.name == "channels"
    assert lock.stamp.locked_on == date(2026, 11, 17)
    assert tuple(lock.channels) == STATISTICAL_CHANNELS
    basis = {name: channel.selection_basis for name, channel in lock.channels.items()}
    assert basis == {
        "r1_popularity": RULE_SELECTION,
        "r2_repurchase": RULE_SELECTION,
        "r3_sibling": RULE_EXECUTION,  # the pre-registered M4 rule switched it to A(a) only
        "r4_itemcf": RULE_SELECTION,
        "r5_wals": RULE_SELECTION,
    }
    quotas = {name: channel.quota for name, channel in lock.channels.items()}
    assert quotas == {
        "r1_popularity": 100,
        "r2_repurchase": 30,
        "r3_sibling": 50,
        "r4_itemcf": 100,
        "r5_wals": 100,
    }
    assert lock.channels["r3_sibling"].config["active_pool_only"] is True
    assert lock.channels["r3_sibling"].d_phase_record.source == "results/m4_trials.csv"


def test_twotower_lock_holds_the_manual_choice_and_both_model_files() -> None:
    lock = load_twotower_lock(TWOTOWER_FILE)
    assert lock.stamp.name == "twotower"
    assert lock.channel.selection_basis == MANUAL_DECISION
    assert lock.channel.trial_id != lock.channel.rule_selected_trial_id
    config = lock.channel.config
    chosen = (config["base_epoch"], config["e_update"], config["replay"], config["chain_seed"])
    assert chosen == (4, 2, 0.5, 42)
    assert config["training"]["fine_tune_learning_rate"] == 0.0001
    assert config["base_window"] == {"end_anchor": date(2020, 6, 3), "window_days": 182}
    assert set(lock.checkpoints) == {BASE_CHECKPOINT, F1_START_CHECKPOINT}
    assert lock.checkpoints[BASE_CHECKPOINT].path == "artifacts/models/twotower/base/epoch_4.pt"
    assert lock.checkpoints[F1_START_CHECKPOINT].path == (
        "artifacts/models/twotower/chain/835ef201ee4d/D4.pt"
    )
    assert (lock.alarm_users.share, lock.alarm_users.seed) == (0.01, 42)


# ---------------------------------------------------------------- strict parsing


def test_unknown_and_missing_keys_are_refused() -> None:
    raw = _channels_raw()
    raw["channels"]["r1_popularity"]["note"] = "typo"
    with pytest.raises(ValueError, match="unknown keys"):
        parse_channels_lock(raw)
    raw = _channels_raw()
    del raw["channels"]["r5_wals"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_channels_lock(raw)


def test_a_config_the_channel_would_refuse_is_refused() -> None:
    raw = _channels_raw()
    raw["channels"]["r4_itemcf"]["config"]["seed_windw_days"] = 365  # misspelled key
    with pytest.raises(ValueError, match="unknown keys"):
        parse_channels_lock(raw)
    raw = _channels_raw()
    del raw["channels"]["r3_sibling"]["config"]["active_pool_only"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_channels_lock(raw)


def test_quota_must_repeat_the_config_quota() -> None:
    raw = _channels_raw()
    raw["channels"]["r2_repurchase"]["quota"] = 50
    with pytest.raises(ValueError, match=r"quota 50 differs from config\.quota 30"):
        parse_channels_lock(raw)


def test_selection_basis_must_fit_the_two_trial_ids() -> None:
    raw = _channels_raw()
    raw["channels"]["r1_popularity"]["selection_basis"] = "best_guess"
    with pytest.raises(ValueError, match="selection_basis must be one of"):
        parse_channels_lock(raw)
    # R3's locked trial is not the rule's trial, so it cannot be called a rule selection.
    raw = _channels_raw()
    raw["channels"]["r3_sibling"]["selection_basis"] = RULE_SELECTION
    with pytest.raises(ValueError, match="does not fit"):
        parse_channels_lock(raw)
    # ... and a rule selection cannot be labelled a manual decision.
    raw = _channels_raw()
    raw["channels"]["r1_popularity"]["selection_basis"] = MANUAL_DECISION
    with pytest.raises(ValueError, match="does not fit"):
        parse_channels_lock(raw)


def test_ids_and_hashes_must_have_the_right_shape() -> None:
    raw = _channels_raw()
    raw["channels"]["r1_popularity"]["trial_id"] = "d2094d45"  # too short
    with pytest.raises(ValueError, match="expected 12 lowercase hex"):
        parse_channels_lock(raw)
    raw = _channels_raw()
    raw["lock"]["git_commit"] = "db801c6"  # short hash
    with pytest.raises(ValueError, match="expected 40 lowercase hex"):
        parse_channels_lock(raw)
    raw = _twotower_raw()
    raw["checkpoints"]["base"]["sha256"] = "0ABC"
    with pytest.raises(ValueError, match="expected 64 lowercase hex"):
        parse_twotower_lock(raw)


def test_a_lock_file_of_the_other_kind_is_refused() -> None:
    raw = _channels_raw()
    raw["lock"]["name"] = "twotower"
    with pytest.raises(ValueError, match="expected 'channels'"):
        parse_channels_lock(raw)


def test_twotower_lock_refuses_a_frozen_config_and_other_alarm_settings() -> None:
    raw = _twotower_raw()
    raw["channel"]["config"]["e_update"] = 0  # frozen: there would be no chain to continue
    with pytest.raises(ValueError, match="e_update must be > 0"):
        parse_twotower_lock(raw)
    raw = _twotower_raw()
    raw["alarm_users"]["seed"] = 7  # not the base training seed -> other users
    with pytest.raises(ValueError, match="differ from config.training"):
        parse_twotower_lock(raw)
    raw = _twotower_raw()
    del raw["checkpoints"]["f1_start"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_twotower_lock(raw)


# ---------------------------------------------------------------- ranker.lock.yaml


def test_ranker_lock_holds_the_final_configuration() -> None:
    lock = load_ranker_lock(RANKER_FILE)
    assert lock.stamp.name == "ranker"
    assert lock.stamp.locked_on == date(2026, 11, 18)
    assert lock.stamp.git_commit == "1de35313f2df8149d74d54ee19b05791f4aa3993"
    # The V phase kept the defaults: 31 leaves, min_data_in_leaf 500, 40 negatives per
    # positive, and chose 500 trees (docs/decisions.md 2026-11-18).
    assert (lock.model.num_leaves, lock.model.min_data_in_leaf) == (31, 500)
    assert lock.sampling.negatives_per_positive == 40
    assert lock.model.num_trees == 500
    assert lock.model.histogram_mode == "row_wise"
    assert lock.selection.basis == RULE_SELECTION
    assert lock.selection.config_id == lock.selection.rule_selected_config_id == "ca1a3cdfce49"
    assert lock.selection.default_kept is True
    assert (lock.window_weeks, lock.scoring.checkpoint_every, lock.scoring.top_k) == (4, 100, 12)
    assert (lock.model.seed, lock.sampling.seed, lock.sampling.row_group_rows) == (42, 42, 100000)
    assert lock.runtime.lightgbm_version == "4.7.0"  # text, not the number 4.7
    assert lock.runtime.libomp_version == "22.1.4"
    assert lock.runtime.num_threads == 6
    assert tuple(lock.references) == ("features_lock", "channels_lock", "twotower_lock")
    assert lock.references["features_lock"].path == "configs/locks/features.lock.yaml"
    assert lock.references["twotower_lock"].path == "configs/locks/twotower.lock.yaml"
    assert len(lock.model.dataset_params) == 7 and len(lock.model.train_params) == 24
    assert lock.selection.v_phase_record.metric == "map@12"
    assert tuple(lock.selection.v_phase_record.folds) == ("V1", "V2", "V3", "V4")


def test_the_ranker_lock_points_to_an_entry_of_the_decision_log() -> None:
    lock = load_ranker_lock(RANKER_FILE)
    prefix = "docs/decisions.md "
    assert lock.selection.decision_record.startswith(prefix)
    heading = "## " + lock.selection.decision_record.removeprefix(prefix)
    decisions = (LOCKS.parents[1] / "docs" / "decisions.md").read_text(encoding="utf-8")
    assert sum(line.startswith(heading) for line in decisions.splitlines()) == 1


def test_ranker_lock_refuses_missing_and_unknown_keys_in_every_section() -> None:
    sections = ("selection", "references", "runtime", "training_window", "sampling", "model")
    for section in (*sections, "scoring"):
        raw = _ranker_raw()
        raw[section]["made_up"] = 1
        with pytest.raises(ValueError, match="unknown keys"):
            parse_ranker_lock(raw)
        raw = _ranker_raw()
        del raw[section]
        with pytest.raises(ValueError, match="missing keys"):
            parse_ranker_lock(raw)
    # Every LightGBM parameter must be written out: none may be dropped or added.
    raw = _ranker_raw()
    del raw["model"]["train_params"]["lambdarank_truncation_level"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_ranker_lock(raw)
    raw = _ranker_raw()
    raw["model"]["dataset_params"]["max_bin_by_feature"] = [255]
    with pytest.raises(ValueError, match="unknown keys"):
        parse_ranker_lock(raw)
    raw = _ranker_raw()
    for name in ("features_lock", "channels_lock", "twotower_lock"):
        raw = _ranker_raw()
        del raw["references"][name]
        with pytest.raises(ValueError, match="missing keys"):
            parse_ranker_lock(raw)


def test_ranker_lock_refuses_values_a_model_cannot_be_built_with() -> None:
    cases = [
        (("model", "histogram_mode"), "undecided", "histogram_mode must be one of"),
        (("model", "num_trees"), 550, "must be a multiple of"),  # 550 is not 100, 200, ...
        (("model", "num_trees"), 0, "num_trees must be > 0"),
        (("model", "num_leaves"), "31", "expected"),  # text instead of a number
        (("model", "seed"), -1, "seed must be >= 0"),
        (("sampling", "method"), "uniform", "method must be"),
        (("sampling", "negatives_per_positive"), 0, "negatives_per_positive must be > 0"),
        (("sampling", "row_group_rows"), 0, "row_group_rows must be > 0"),
        (("scoring", "checkpoint_every"), 0, "checkpoint_every must be > 0"),
        (("scoring", "top_k"), 0, "top_k must be > 0"),
        (("training_window", "window_weeks"), 0, "window_weeks must be > 0"),
        (("runtime", "num_threads"), 0, "num_threads must be > 0"),
        (("runtime", "lightgbm_version"), 4.7, "expected"),  # unquoted in YAML: a number
        (("runtime", "libomp_version"), " ", "must not be empty"),
        (("selection", "default_kept"), 1, "expected true or false"),
        (("selection", "decision_record"), "", "expected a non-empty string"),
        (("selection", "config_id"), "ca1a3cdf", "expected 12 lowercase hex"),
        (("references", "features_lock"), {"path": "x", "sha256": "0ABC"}, "expected 64 lowercase"),
    ]
    for (section, key), value, message in cases:
        raw = _ranker_raw()
        raw[section][key] = value
        with pytest.raises(ValueError, match=message):
            parse_ranker_lock(raw)


def test_ranker_lock_basis_must_fit_the_configuration_ids() -> None:
    raw = _ranker_raw()
    raw["selection"]["basis"] = "guess"
    with pytest.raises(ValueError, match="basis must be one of"):
        parse_ranker_lock(raw)
    # The rule's own configuration cannot be labelled a manual decision ...
    raw = _ranker_raw()
    raw["selection"]["basis"] = MANUAL_DECISION
    with pytest.raises(ValueError, match="does not fit"):
        parse_ranker_lock(raw)
    # ... and another configuration cannot be called the rule's selection.
    raw = _ranker_raw()
    raw["selection"]["rule_selected_config_id"] = "3f2285d99ba3"
    with pytest.raises(ValueError, match="does not fit"):
        parse_ranker_lock(raw)
    raw["selection"]["basis"] = MANUAL_DECISION
    assert parse_ranker_lock(raw).selection.basis == MANUAL_DECISION


def test_a_ranker_lock_of_another_kind_is_refused() -> None:
    raw = _ranker_raw()
    raw["lock"]["name"] = "features"
    with pytest.raises(ValueError, match="expected 'ranker'"):
        parse_ranker_lock(raw)
    raw = _ranker_raw()
    raw["lock"]["git_commit"] = "1de3531"  # short hash
    with pytest.raises(ValueError, match="expected 40 lowercase hex"):
        parse_ranker_lock(raw)


# ---------------------------------------------------------------- sha256


def test_file_sha256_of_known_bytes(tmp_path: Path) -> None:
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")
    # The standard test vector of SHA-256 for "abc".
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert file_sha256(path) == expected
    check_file_sha256(path, expected)  # no error


def test_file_sha256_reads_files_larger_than_one_piece(tmp_path: Path) -> None:
    # 2.5 MB: three reads of the 1 MB pieces; must equal hashing all bytes at once.
    content = bytes(range(256)) * 10_240
    path = tmp_path / "big.bin"
    path.write_bytes(content)
    assert file_sha256(path) == hashlib.sha256(content).hexdigest()


def test_check_file_sha256_fails_on_another_file_and_on_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    path.write_bytes(b"abc")
    locked = file_sha256(path)
    path.write_bytes(b"abd")  # one byte changed after the lock
    with pytest.raises(ValueError, match="This is not the locked file") as error:
        check_file_sha256(path, locked)
    assert locked in str(error.value)  # the message shows the locked hash ...
    assert file_sha256(path) in str(error.value)  # ... and the actual one
    with pytest.raises(FileNotFoundError, match="locked file not found"):
        check_file_sha256(tmp_path / "missing.pt", locked)
