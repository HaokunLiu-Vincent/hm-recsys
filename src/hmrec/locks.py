"""Read the locked configuration (configs/locks/, proposal §6.5, §8.7 and §11).

    configs/locks/channels.lock.yaml  -> ChannelsLock   (R1-R5)
    configs/locks/twotower.lock.yaml  -> TwoTowerLock   (R6: config, alarm users, model files)
    configs/locks/features.lock.yaml  -> FeaturesLock   (the 71 Ranker features)

This module belongs to the bottom layer (it needs only hmrec.config and the
standard library), so recall/ and features/ may both import it. It only READS
and CHECKS the lock files. Building channels from them is hmrec.recall.locked.

Every loader is strict, like hmrec.config: a missing key, an unknown key or a
value of the wrong type raises ValueError. A locked channel config must also pass
the channel's own parser (hmrec.config.parse_*_config), so a lock file cannot
hold a config the channel would refuse.

Model files live under artifacts/, which is not in git. The lock records their
sha256; check_file_sha256 fails when a file is not the one that was locked.
"""

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import Any

from hmrec.config import (
    FEATURE_DTYPES,
    FeaturesConfig,
    check_keys,
    get_date,
    get_float,
    get_int,
    get_mapping,
    get_str,
    get_str_list,
    load_yaml,
    parse_itemcf_config,
    parse_popularity_config,
    parse_repurchase_config,
    parse_sibling_config,
    parse_twotower_config,
    parse_wals_config,
)

R6_CHANNEL = "r6_twotower"
# The channels of channels.lock.yaml, in the order R1 ... R5.
STATISTICAL_CHANNELS = (
    "r1_popularity",
    "r2_repurchase",
    "r3_sibling",
    "r4_itemcf",
    "r5_wals",
)
ALL_CHANNELS = (*STATISTICAL_CHANNELS, R6_CHANNEL)

RULE_SELECTION = "rule_selection"
RULE_EXECUTION = "rule_execution"
MANUAL_DECISION = "manual_decision"
SELECTION_BASES = (RULE_SELECTION, RULE_EXECUTION, MANUAL_DECISION)

# Names of the model files in twotower.lock.yaml (`checkpoints`).
BASE_CHECKPOINT = "base"  # base model after E_base epochs; predicted D1
F1_START_CHECKPOINT = "f1_start"  # chain model after the D4 fine-tune (M4); predicts F1
CHECKPOINT_NAMES = (BASE_CHECKPOINT, F1_START_CHECKPOINT)

TRIAL_ID_LENGTH = 12  # hmrec.hpo.runner.make_trial_id: 12 hex characters
GIT_COMMIT_LENGTH = 40  # a full git commit hash
SHA256_LENGTH = 64  # hex characters of a sha256 digest
HASH_READ_BYTES = 1024 * 1024  # files are hashed 1 MB at a time, never read whole

# The numbers of configs/features.yaml that define feature values and are therefore
# in features.lock.yaml: two fixed values, then four settings (three window lengths).
# Attribute names of both FeaturesConfig and FeaturesLock.
LOCKED_FEATURE_NUMBERS = (
    "decay_tau_days",
    "history_similarity_top_k",
    "history_length",
    "rrf_k",
    "online_sales_channel_id",
    "week_days",
    "four_week_days",
    "year_days",
)

# The parser every locked config must pass. Each raises ValueError on a missing
# key, an unknown key or a value out of range.
_CONFIG_PARSERS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "r1_popularity": parse_popularity_config,
    "r2_repurchase": parse_repurchase_config,
    "r3_sibling": parse_sibling_config,
    "r4_itemcf": parse_itemcf_config,
    "r5_wals": parse_wals_config,
    R6_CHANNEL: parse_twotower_config,
}


# ---------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class LockStamp:
    """When a lock file was written.

    Attributes:
        name: "channels", "twotower" or "features".
        locked_on: The day the file was written.
        git_commit: HEAD at that time (full hash). The commit that adds the lock
            file is a child of this commit.
    """

    name: str
    locked_on: date
    git_commit: str


@dataclass(frozen=True)
class RecordedScore:
    """The D-phase score of a locked trial, copied from a results file. A record only.

    Nothing reads these numbers to make a choice. D-phase scores are optimistic,
    because the configurations were chosen on these weeks (proposal §12).

    Attributes:
        source: Results file that holds the trial, e.g. "results/d_phase.csv"
            (relative to the repository root).
        metric: The channel's selection metric, e.g. "recall@100".
        pooled: Score over all queries of the folds together (proposal §4.1). float64.
        folds: Fold name -> score of that fold, e.g. {"D1": 0.0749, ...}. float64.
    """

    source: str
    metric: str
    pooled: float
    folds: dict[str, float]


@dataclass(frozen=True)
class LockedChannel:
    """One channel's locked configuration.

    Attributes:
        channel: Channel name, e.g. "r4_itemcf".
        selection_basis: "rule_selection" (the §4.2 rule), "rule_execution"
            (outcome of a pre-registered rule on top of the rule's selection) or
            "manual_decision" (chosen by hand, recorded in docs/decisions.md).
        trial_id: Id of the locked configuration (hmrec.hpo.runner.make_trial_id).
        rule_selected_trial_id: The trial the §4.2 rule selected in the D phase;
            equal to trial_id exactly when selection_basis is "rule_selection".
        decision_record: The docs/decisions.md entry that explains the choice.
        quota: Max candidates per query (equal to config["quota"]).
        d_phase_record: Recorded D-phase score of the trial (a record only).
        config: The complete raw config, with the same keys as
            configs/recall/{channel}.yaml. It already passed the channel's
            parser; parse it again (hmrec.config.parse_*_config) to get the
            typed dataclass.
    """

    channel: str
    selection_basis: str
    trial_id: str
    rule_selected_trial_id: str
    decision_record: str
    quota: int
    d_phase_record: RecordedScore
    config: dict[str, Any]


@dataclass(frozen=True)
class ChannelsLock:
    """configs/locks/channels.lock.yaml: the locked configs of R1-R5.

    Attributes:
        stamp: When the file was written.
        channels: Channel name -> its lock; exactly STATISTICAL_CHANNELS.
    """

    stamp: LockStamp
    channels: dict[str, LockedChannel]


@dataclass(frozen=True)
class LockedCheckpoint:
    """A locked model file.

    Attributes:
        name: "base" or "f1_start" (CHECKPOINT_NAMES).
        path: Path relative to the repository root, e.g.
            "artifacts/models/twotower/base/epoch_4.pt". A record: code gets the
            real path from hmrec.paths and checks that it is this one.
        sha256: sha256 of the file's bytes (64 hex characters).
    """

    name: str
    path: str
    sha256: str


@dataclass(frozen=True)
class AlarmUsersLock:
    """How the 1% alarm users of the Two-Tower chain are fixed (proposal §6.4).

    Attributes:
        share: Share of distinct base-window users (0.01).
        seed: Seed of the draw (the base training seed).
        source: The array the users are drawn from, relative to the repository
            root: the customer_idx of every base training sample.
        source_sha256: sha256 of that file; the same users are drawn only from
            the same array.
        n_alarm_users: Number of alarm users drawn at lock time.
    """

    share: float
    seed: int
    source: str
    source_sha256: str
    n_alarm_users: int


@dataclass(frozen=True)
class TwoTowerLock:
    """configs/locks/twotower.lock.yaml: the locked config and model files of R6.

    Attributes:
        stamp: When the file was written.
        channel: The locked R6 configuration (channel "r6_twotower").
        alarm_users: The alarm-user settings.
        checkpoints: Name -> locked model file; exactly CHECKPOINT_NAMES.
    """

    stamp: LockStamp
    channel: LockedChannel
    alarm_users: AlarmUsersLock
    checkpoints: dict[str, LockedCheckpoint]


@dataclass(frozen=True)
class LockedFeature:
    """One feature of configs/locks/features.lock.yaml.

    Attributes:
        name: Feature name, as in proposal §8.
        dtype: "float32" (a number, missing = NaN) or "int16" (a category code,
            0 = missing).
    """

    name: str
    dtype: str


@dataclass(frozen=True)
class FeaturesLock:
    """configs/locks/features.lock.yaml: the locked Ranker features (proposal §8.7).

    Attributes:
        stamp: When the file was written.
        features: The 71 features in wide-table column order.
        categorical_features: Names of the categorical features, in column order.
        decay_tau_days: Time constant of the ui_decay_* features (fixed, not tuned).
        history_similarity_top_k: tt_sim_top5_hist averages this many of the
            largest similarities (fixed, not tuned).
        history_length: Past purchases used by the *_hist features.
        rrf_k: The constant of rrf_score = sum of 1 / (rrf_k + rank).
        online_sales_channel_id: sales_channel_id of the online channel.
        week_days: Length in days of the 1w and prev_1w windows.
        four_week_days: Length in days of the 4w window.
        year_days: Length in days of the 52w window.
    """

    stamp: LockStamp
    features: tuple[LockedFeature, ...]
    categorical_features: tuple[str, ...]
    decay_tau_days: float
    history_similarity_top_k: int
    history_length: int
    rrf_k: int
    online_sales_channel_id: int
    week_days: int
    four_week_days: int
    year_days: int


# ---------------------------------------------------------------- field checks


def _check_hex(value: str, length: int, where: str) -> str:
    """Return `value` if it is exactly `length` lowercase hex characters, else fail."""
    is_hex = all(character in "0123456789abcdef" for character in value)
    if len(value) != length or not is_hex:
        raise ValueError(f"{where}: expected {length} lowercase hex characters, got {value!r}")
    return value


def _parse_stamp(raw: Mapping[str, Any], expected_name: str) -> LockStamp:
    """Parse the `lock` section and check that the file is the expected kind of lock."""
    where = "lock"
    check_keys(raw, {"name", "locked_on", "git_commit"}, where)
    name = get_str(raw, "name", where)
    if name != expected_name:
        raise ValueError(f"{where}.name is {name!r}, expected {expected_name!r}")
    return LockStamp(
        name=name,
        locked_on=get_date(raw, "locked_on", where),
        git_commit=_check_hex(
            get_str(raw, "git_commit", where), GIT_COMMIT_LENGTH, f"{where}.git_commit"
        ),
    )


def _parse_recorded_score(raw: Mapping[str, Any], where: str) -> RecordedScore:
    """Parse a `d_phase_record` section; it needs at least one fold score."""
    check_keys(raw, {"source", "metric", "pooled", "folds"}, where)
    raw_folds = get_mapping(raw, "folds", where)
    if not raw_folds:
        raise ValueError(f"{where}.folds must list at least one fold")
    folds = {str(fold): get_float(raw_folds, fold, f"{where}.folds") for fold in raw_folds}
    return RecordedScore(
        source=get_str(raw, "source", where),
        metric=get_str(raw, "metric", where),
        pooled=get_float(raw, "pooled", where),
        folds=folds,
    )


def _check_basis(channel: LockedChannel, where: str) -> None:
    """The basis must be known, and agree with whether the rule's own trial was locked."""
    if channel.selection_basis not in SELECTION_BASES:
        raise ValueError(
            f"{where}.selection_basis must be one of {SELECTION_BASES}, "
            f"got {channel.selection_basis!r}"
        )
    is_rule_trial = channel.trial_id == channel.rule_selected_trial_id
    if is_rule_trial != (channel.selection_basis == RULE_SELECTION):
        raise ValueError(
            f"{where}: selection_basis {channel.selection_basis!r} does not fit trial_id "
            f"{channel.trial_id} and rule_selected_trial_id {channel.rule_selected_trial_id}; "
            f"they are equal exactly when the basis is {RULE_SELECTION!r}"
        )


def _check_complete_config(channel: str, config: Mapping[str, Any], where: str) -> None:
    """The config must pass the channel's parser AND spell out every setting.

    The parser is the check of keys and ranges. On top of that, a lock file may
    not rely on a parser default: a key the parser treats as optional (R3's
    active_pool_only, absent = false) must still be written, so the lock file
    alone says what runs.
    """
    parsed = _CONFIG_PARSERS[channel](config)
    missing = sorted({field.name for field in fields(parsed)} - set(config))
    if missing:
        raise ValueError(f"{where}: missing keys {missing}; a lock must state every setting")


def parse_locked_channel(channel: str, raw: Mapping[str, Any], where: str) -> LockedChannel:
    """Parse one channel entry of a lock file.

    Args:
        channel: Channel name; decides which config parser is applied.
        raw: The entry (see configs/locks/channels.lock.yaml for the keys).
        where: Name of the entry for error messages.

    Raises:
        ValueError: On a missing / unknown key, a wrong type, a config the
            channel's parser refuses, a quota that differs from config.quota, or
            a selection_basis that does not fit the two trial ids.
    """
    if channel not in _CONFIG_PARSERS:
        raise ValueError(f"{where}: unknown channel {channel!r}")
    expected = {
        "selection_basis",
        "trial_id",
        "rule_selected_trial_id",
        "decision_record",
        "quota",
        "d_phase_record",
        "config",
    }
    check_keys(raw, expected, where)
    config = dict(get_mapping(raw, "config", where))
    _check_complete_config(channel, config, f"{where}.config")
    locked = LockedChannel(
        channel=channel,
        selection_basis=get_str(raw, "selection_basis", where),
        trial_id=_check_hex(get_str(raw, "trial_id", where), TRIAL_ID_LENGTH, f"{where}.trial_id"),
        rule_selected_trial_id=_check_hex(
            get_str(raw, "rule_selected_trial_id", where),
            TRIAL_ID_LENGTH,
            f"{where}.rule_selected_trial_id",
        ),
        decision_record=get_str(raw, "decision_record", where),
        quota=get_int(raw, "quota", where),
        d_phase_record=_parse_recorded_score(
            get_mapping(raw, "d_phase_record", where), f"{where}.d_phase_record"
        ),
        config=config,
    )
    _check_basis(locked, where)
    if locked.quota != config["quota"]:
        raise ValueError(
            f"{where}: quota {locked.quota} differs from config.quota {config['quota']}"
        )
    return locked


# ---------------------------------------------------------------- channels.lock.yaml


def parse_channels_lock(raw: Mapping[str, Any]) -> ChannelsLock:
    """Build a ChannelsLock from the parsed content of configs/locks/channels.lock.yaml.

    Raises:
        ValueError: If the file is not a "channels" lock, does not hold exactly
            R1-R5, or an entry is invalid (parse_locked_channel).
    """
    check_keys(raw, {"lock", "channels"}, "channels.lock")
    stamp = _parse_stamp(get_mapping(raw, "lock", "channels.lock"), "channels")
    raw_channels = get_mapping(raw, "channels", "channels.lock")
    check_keys(raw_channels, set(STATISTICAL_CHANNELS), "channels.lock.channels")
    channels = {
        channel: parse_locked_channel(
            channel,
            get_mapping(raw_channels, channel, "channels.lock.channels"),
            f"channels.lock.channels.{channel}",
        )
        for channel in STATISTICAL_CHANNELS
    }
    return ChannelsLock(stamp, channels)


def load_channels_lock(path: Path) -> ChannelsLock:
    """Read and check configs/locks/channels.lock.yaml."""
    return parse_channels_lock(load_yaml(path))


# ---------------------------------------------------------------- twotower.lock.yaml


def _parse_alarm_users(raw: Mapping[str, Any], config: Mapping[str, Any]) -> AlarmUsersLock:
    """Parse `alarm_users`; share and seed must repeat the locked training settings."""
    where = "twotower.lock.alarm_users"
    check_keys(raw, {"share", "seed", "source", "source_sha256", "n_alarm_users"}, where)
    alarm = AlarmUsersLock(
        share=get_float(raw, "share", where),
        seed=get_int(raw, "seed", where),
        source=get_str(raw, "source", where),
        source_sha256=_check_hex(
            get_str(raw, "source_sha256", where), SHA256_LENGTH, f"{where}.source_sha256"
        ),
        n_alarm_users=get_int(raw, "n_alarm_users", where),
    )
    training = config["training"]
    if alarm.share != training["alarm_user_share"] or alarm.seed != training["seed"]:
        raise ValueError(
            f"{where}: share {alarm.share} / seed {alarm.seed} differ from "
            f"config.training (alarm_user_share {training['alarm_user_share']}, "
            f"seed {training['seed']})"
        )
    if alarm.n_alarm_users <= 0:
        raise ValueError(f"{where}.n_alarm_users must be > 0, got {alarm.n_alarm_users}")
    return alarm


def _parse_checkpoints(raw: Mapping[str, Any]) -> dict[str, LockedCheckpoint]:
    """Parse `checkpoints`: exactly the names in CHECKPOINT_NAMES, each with path and sha256."""
    where = "twotower.lock.checkpoints"
    check_keys(raw, set(CHECKPOINT_NAMES), where)
    checkpoints: dict[str, LockedCheckpoint] = {}
    for name in CHECKPOINT_NAMES:
        entry = get_mapping(raw, name, where)
        check_keys(entry, {"path", "sha256"}, f"{where}.{name}")
        checkpoints[name] = LockedCheckpoint(
            name=name,
            path=get_str(entry, "path", f"{where}.{name}"),
            sha256=_check_hex(
                get_str(entry, "sha256", f"{where}.{name}"), SHA256_LENGTH, f"{where}.{name}.sha256"
            ),
        )
    return checkpoints


def parse_twotower_lock(raw: Mapping[str, Any]) -> TwoTowerLock:
    """Build a TwoTowerLock from the parsed content of configs/locks/twotower.lock.yaml.

    Raises:
        ValueError: If the file is not a "twotower" lock, a section is invalid,
            or the locked config is frozen (e_update = 0): the lock is for the
            weekly training chain, which needs e_update > 0.
    """
    where = "twotower.lock"
    check_keys(raw, {"lock", "channel", "alarm_users", "checkpoints"}, where)
    stamp = _parse_stamp(get_mapping(raw, "lock", where), "twotower")
    channel = parse_locked_channel(
        R6_CHANNEL, get_mapping(raw, "channel", where), f"{where}.channel"
    )
    if channel.config["e_update"] <= 0:
        raise ValueError(f"{where}.channel.config.e_update must be > 0 (the weekly chain)")
    return TwoTowerLock(
        stamp=stamp,
        channel=channel,
        alarm_users=_parse_alarm_users(get_mapping(raw, "alarm_users", where), channel.config),
        checkpoints=_parse_checkpoints(get_mapping(raw, "checkpoints", where)),
    )


def load_twotower_lock(path: Path) -> TwoTowerLock:
    """Read and check configs/locks/twotower.lock.yaml."""
    return parse_twotower_lock(load_yaml(path))


# ---------------------------------------------------------------- features.lock.yaml


def _parse_locked_feature(raw: Any, position: int) -> LockedFeature:
    """Parse one entry of `features`: a mapping with exactly `name` and `dtype`."""
    where = f"features.lock.features[{position}]"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {raw!r}")
    check_keys(raw, {"name", "dtype"}, where)
    feature = LockedFeature(name=get_str(raw, "name", where), dtype=get_str(raw, "dtype", where))
    if feature.dtype not in FEATURE_DTYPES:
        raise ValueError(f"{where}.dtype must be one of {FEATURE_DTYPES}, got {feature.dtype!r}")
    return feature


def _parse_locked_features(raw_features: Any, where: str) -> tuple[LockedFeature, ...]:
    """Parse the `features` list; every feature name may appear only once."""
    if not isinstance(raw_features, list):
        raise ValueError(f"{where}: expected a list, got {type(raw_features).__name__}")
    features = tuple(_parse_locked_feature(item, i) for i, item in enumerate(raw_features))
    names = [feature.name for feature in features]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"{where}: repeated feature names {repeated}")
    return features


def _parse_categorical_features(
    raw: Mapping[str, Any], features: tuple[LockedFeature, ...], where: str
) -> tuple[str, ...]:
    """Parse `categorical_features`; it must be exactly the int16 features in column order."""
    categorical = get_str_list(raw, "categorical_features", where)
    # A category code is stored as int16 and every int16 column is a category
    # code (proposal §8.1), so the two parts of the file must say the same thing.
    int16_names = tuple(feature.name for feature in features if feature.dtype == "int16")
    if categorical != int16_names:
        raise ValueError(
            f"{where}.categorical_features must be the int16 features in column order "
            f"{int16_names}, got {categorical}"
        )
    return categorical


def parse_features_lock(raw: Mapping[str, Any]) -> FeaturesLock:
    """Build a FeaturesLock from the parsed content of configs/locks/features.lock.yaml.

    Raises:
        ValueError: If the file is not a "features" lock, a key is missing or
            unknown, a number has the wrong type, a feature name is repeated,
            or `categorical_features` is not exactly the int16 features in
            column order.
    """
    where = "features.lock"
    expected = {"lock", "fixed_values", "settings", "categorical_features", "features"}
    check_keys(raw, expected, where)
    fixed = get_mapping(raw, "fixed_values", where)
    check_keys(fixed, {"decay_tau_days", "history_similarity_top_k"}, f"{where}.fixed_values")
    settings = get_mapping(raw, "settings", where)
    setting_keys = {"history_length", "rrf_k", "online_sales_channel_id", "windows"}
    check_keys(settings, setting_keys, f"{where}.settings")
    windows = get_mapping(settings, "windows", f"{where}.settings")
    window_keys = {"week_days", "four_week_days", "year_days"}
    check_keys(windows, window_keys, f"{where}.settings.windows")
    features = _parse_locked_features(raw["features"], f"{where}.features")
    return FeaturesLock(
        stamp=_parse_stamp(get_mapping(raw, "lock", where), "features"),
        features=features,
        categorical_features=_parse_categorical_features(raw, features, where),
        decay_tau_days=get_float(fixed, "decay_tau_days", f"{where}.fixed_values"),
        history_similarity_top_k=get_int(
            fixed, "history_similarity_top_k", f"{where}.fixed_values"
        ),
        history_length=get_int(settings, "history_length", f"{where}.settings"),
        rrf_k=get_int(settings, "rrf_k", f"{where}.settings"),
        online_sales_channel_id=get_int(settings, "online_sales_channel_id", f"{where}.settings"),
        week_days=get_int(windows, "week_days", f"{where}.settings.windows"),
        four_week_days=get_int(windows, "four_week_days", f"{where}.settings.windows"),
        year_days=get_int(windows, "year_days", f"{where}.settings.windows"),
    )


def load_features_lock(path: Path) -> FeaturesLock:
    """Read and check configs/locks/features.lock.yaml."""
    return parse_features_lock(load_yaml(path))


def _feature_list_difference(
    in_config: list[tuple[str, str]], in_lock: list[tuple[str, str]]
) -> str | None:
    """Describe how two (name, dtype) lists differ, or return None if they are equal."""
    if in_config == in_lock:
        return None
    if len(in_config) != len(in_lock):
        only_config = sorted(set(in_config) - set(in_lock))
        only_lock = sorted(set(in_lock) - set(in_config))
        return (
            f"{len(in_config)} features in features.yaml, {len(in_lock)} in the lock; "
            f"only in features.yaml {only_config}, only in the lock {only_lock}"
        )
    for position, (config_item, lock_item) in enumerate(zip(in_config, in_lock, strict=True)):
        if config_item != lock_item:
            return (
                f"column {position} (name, dtype): {config_item} in features.yaml, "
                f"{lock_item} in the lock"
            )
    return None


def check_features_match_lock(config: FeaturesConfig, lock: FeaturesLock) -> None:
    """Fail unless configs/features.yaml says exactly what the features lock says.

    Compared: the feature names and dtypes in column order, the categorical
    features, the two fixed values (decay_tau_days, history_similarity_top_k)
    and the four settings (history_length, rrf_k, online_sales_channel_id, the
    three window lengths). Only embedding_pair_chunk_size is not compared: it
    changes memory, not results. The code keeps reading configs/features.yaml;
    this check is what stops an edit of that file after the lock.

    Args:
        config: The loaded configs/features.yaml.
        lock: The loaded configs/locks/features.lock.yaml.

    Raises:
        ValueError: If any compared item differs; the message names each one.
    """
    differences: list[str] = []
    feature_difference = _feature_list_difference(
        [(feature.name, feature.dtype) for feature in config.features],
        [(feature.name, feature.dtype) for feature in lock.features],
    )
    if feature_difference is not None:
        differences.append(feature_difference)
    if config.categorical_names != lock.categorical_features:
        differences.append(
            f"categorical features: {config.categorical_names} in features.yaml, "
            f"{lock.categorical_features} in the lock"
        )
    # FeaturesConfig and FeaturesLock use the same attribute names for these numbers.
    for name in LOCKED_FEATURE_NUMBERS:
        in_config, in_lock = getattr(config, name), getattr(lock, name)
        if in_config != in_lock:
            differences.append(f"{name}: {in_config} in features.yaml, {in_lock} in the lock")
    if differences:
        raise ValueError(
            "configs/features.yaml differs from the locked features "
            "(configs/locks/features.lock.yaml): " + "; ".join(differences)
        )


# ---------------------------------------------------------------- sha256 of files


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes as 64 hex characters (read in 1 MB pieces)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while piece := handle.read(HASH_READ_BYTES):
            digest.update(piece)
    return digest.hexdigest()


def check_file_sha256(path: Path, expected_sha256: str) -> None:
    """Fail unless `path` exists and has exactly the locked sha256.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If its sha256 differs: it is not the file that was locked
            (regenerated, replaced or damaged), so it must not be used.
    """
    if not path.is_file():
        raise FileNotFoundError(f"locked file not found: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{path}: sha256 is {actual}, but the lock file records {expected_sha256}. "
            "This is not the locked file; do not use it."
        )
