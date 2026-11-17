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
    load_twotower_lock,
    parse_channels_lock,
    parse_twotower_lock,
)

LOCKS = Path(__file__).resolve().parents[2] / "configs" / "locks"
CHANNELS_FILE = LOCKS / "channels.lock.yaml"
TWOTOWER_FILE = LOCKS / "twotower.lock.yaml"


def _channels_raw() -> dict[str, Any]:
    return copy.deepcopy(load_yaml(CHANNELS_FILE))


def _twotower_raw() -> dict[str, Any]:
    return copy.deepcopy(load_yaml(TWOTOWER_FILE))


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
