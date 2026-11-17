"""Tests for the features lock: strict parsing and the comparison with configs/features.yaml.

Like test_locks.py, the parsing tests start from the REAL lock file and break one
thing at a time. The comparison tests start from the REAL configs/features.yaml.
"""

import copy
import dataclasses
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from hmrec.config import (
    N_PROPOSAL_FEATURES,
    PROPOSAL_CATEGORICAL_FEATURES,
    FeaturesConfig,
    load_features_config,
    load_yaml,
)
from hmrec.locks import (
    LOCKED_FEATURE_NUMBERS,
    check_features_match_lock,
    load_features_lock,
    parse_features_lock,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
LOCK_FILE = CONFIGS / "locks" / "features.lock.yaml"
CONFIG_FILE = CONFIGS / "features.yaml"


def _lock_raw() -> dict[str, Any]:
    return copy.deepcopy(load_yaml(LOCK_FILE))


def _config() -> FeaturesConfig:
    return load_features_config(CONFIG_FILE)


# ---------------------------------------------------------------- the real files


def test_features_lock_holds_the_71_features_the_fixed_values_and_the_settings() -> None:
    lock = load_features_lock(LOCK_FILE)
    assert lock.stamp.name == "features"
    assert lock.stamp.locked_on == date(2026, 11, 18)
    assert len(lock.features) == N_PROPOSAL_FEATURES == 71
    assert lock.categorical_features == PROPOSAL_CATEGORICAL_FEATURES
    assert len(lock.categorical_features) == 8
    assert (lock.decay_tau_days, lock.history_similarity_top_k) == (30.0, 5)
    assert (lock.history_length, lock.rrf_k, lock.online_sales_channel_id) == (20, 60, 2)
    assert (lock.week_days, lock.four_week_days, lock.year_days) == (7, 28, 364)
    # First and last column, and the dtype of a category code.
    assert (lock.features[0].name, lock.features[0].dtype) == ("pop_rank", "float32")
    assert (lock.features[-1].name, lock.features[-1].dtype) == ("ui_attr_match_cnt", "float32")
    dtypes = {feature.name: feature.dtype for feature in lock.features}
    assert dtypes["section_no"] == "int16"


def test_features_yaml_matches_the_lock() -> None:
    # The check itself passes ...
    check_features_match_lock(_config(), load_features_lock(LOCK_FILE))
    # ... and, spelled out, the two files list the same columns in the same order.
    config = _config()
    lock = load_features_lock(LOCK_FILE)
    in_config = [(feature.name, feature.dtype) for feature in config.features]
    in_lock = [(feature.name, feature.dtype) for feature in lock.features]
    assert in_config == in_lock


def test_every_number_of_features_yaml_is_locked_except_the_chunk_size() -> None:
    # A new number added to FeaturesConfig must be either locked or listed here as
    # "memory only"; otherwise it could change feature values without the lock noticing.
    all_fields = {field.name for field in dataclasses.fields(FeaturesConfig)}
    not_numbers = {"features"}
    memory_only = {"embedding_pair_chunk_size"}
    assert set(LOCKED_FEATURE_NUMBERS) == all_fields - not_numbers - memory_only
    assert len(LOCKED_FEATURE_NUMBERS) == 8  # 2 fixed values + 3 settings + 3 window lengths


# ---------------------------------------------------------------- strict parsing


def test_unknown_and_missing_keys_are_refused() -> None:
    raw = _lock_raw()
    raw["features"][0]["group"] = "A"  # not a key of the lock
    with pytest.raises(ValueError, match="unknown keys"):
        parse_features_lock(raw)
    raw = _lock_raw()
    del raw["fixed_values"]["decay_tau_days"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_features_lock(raw)
    raw = _lock_raw()
    del raw["categorical_features"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_features_lock(raw)
    raw = _lock_raw()
    del raw["settings"]["rrf_k"]
    with pytest.raises(ValueError, match=r"features\.lock\.settings: missing keys \['rrf_k'\]"):
        parse_features_lock(raw)
    raw = _lock_raw()
    raw["settings"]["windows"]["two_week_days"] = 14  # not a window of the proposal
    with pytest.raises(ValueError, match="unknown keys"):
        parse_features_lock(raw)


def test_a_setting_of_the_wrong_type_is_refused() -> None:
    raw = _lock_raw()
    raw["settings"]["windows"]["year_days"] = 364.0  # a number of days is an integer
    with pytest.raises(ValueError, match="expected an integer"):
        parse_features_lock(raw)


def test_a_lock_file_of_the_other_kind_is_refused() -> None:
    raw = _lock_raw()
    raw["lock"]["name"] = "channels"
    with pytest.raises(ValueError, match="expected 'features'"):
        parse_features_lock(raw)


def test_unknown_dtype_and_repeated_name_are_refused() -> None:
    raw = _lock_raw()
    raw["features"][0]["dtype"] = "float64"
    with pytest.raises(ValueError, match="dtype must be one of"):
        parse_features_lock(raw)
    raw = _lock_raw()
    raw["features"][1]["name"] = raw["features"][0]["name"]
    with pytest.raises(ValueError, match=r"repeated feature names \['pop_rank'\]"):
        parse_features_lock(raw)


def test_categorical_list_must_be_the_int16_features_in_order() -> None:
    raw = _lock_raw()
    raw["categorical_features"].remove("section_no")
    with pytest.raises(ValueError, match="must be the int16 features in column order"):
        parse_features_lock(raw)
    raw = _lock_raw()
    raw["categorical_features"].reverse()
    with pytest.raises(ValueError, match="must be the int16 features in column order"):
        parse_features_lock(raw)


# ---------------------------------------------------------------- comparison


# (attribute of FeaturesConfig, the locked value, another value). One row per locked number.
CHANGED_NUMBERS = [
    ("decay_tau_days", 30.0, 14.0),
    ("history_similarity_top_k", 5, 3),
    ("history_length", 20, 10),
    ("rrf_k", 60, 10),
    ("online_sales_channel_id", 2, 1),
    ("week_days", 7, 14),
    ("four_week_days", 28, 30),
    ("year_days", 364, 365),
]


def test_the_change_test_covers_every_locked_number() -> None:
    assert [name for name, _, _ in CHANGED_NUMBERS] == list(LOCKED_FEATURE_NUMBERS)


@pytest.mark.parametrize(("name", "locked_value", "other_value"), CHANGED_NUMBERS)
def test_a_changed_number_in_features_yaml_is_refused(
    name: str, locked_value: float, other_value: float
) -> None:
    lock = load_features_lock(LOCK_FILE)
    config = _config()
    assert getattr(config, name) == locked_value == getattr(lock, name)
    changed = dataclasses.replace(config, **{name: other_value})
    expected = f"{name}: {other_value} in features.yaml, {locked_value} in the lock"
    with pytest.raises(ValueError, match="differs from the locked features") as error:
        check_features_match_lock(changed, lock)
    # Only this one item is reported.
    assert str(error.value).endswith(": " + expected)


def test_a_changed_chunk_size_is_accepted() -> None:
    # Memory only (proposal §1.3): the same features come out, so the lock does not hold it.
    changed = dataclasses.replace(_config(), embedding_pair_chunk_size=12_345)
    check_features_match_lock(changed, load_features_lock(LOCK_FILE))


def test_a_removed_reordered_or_retyped_feature_is_refused() -> None:
    lock = load_features_lock(LOCK_FILE)
    config = _config()
    # One feature fewer.
    shorter = dataclasses.replace(config, features=config.features[:-1])
    with pytest.raises(ValueError, match="70 features in features.yaml, 71 in the lock"):
        check_features_match_lock(shorter, lock)
    # The first two columns swapped: same set, other order.
    swapped = (config.features[1], config.features[0], *config.features[2:])
    with pytest.raises(ValueError, match=r"column 0 \(name, dtype\): \('rep_rank', 'float32'\)"):
        check_features_match_lock(dataclasses.replace(config, features=swapped), lock)
    # A category code turned into a number: the dtype and the categorical list both differ.
    position = config.names.index("section_no")
    retyped_feature = dataclasses.replace(
        config.features[position], dtype="float32", categorical=False
    )
    retyped = (*config.features[:position], retyped_feature, *config.features[position + 1 :])
    with pytest.raises(ValueError, match="categorical features") as error:
        check_features_match_lock(dataclasses.replace(config, features=retyped), lock)
    assert f"column {position} (name, dtype): ('section_no', 'float32')" in str(error.value)
