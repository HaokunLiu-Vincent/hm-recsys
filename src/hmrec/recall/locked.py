"""The ONLY entry to the six recall channels after M5 (proposal §6.5, §11).

From M6 on, every F / V / T fold gets its channels from here:

    locked = load_locked_recall(paths)                      # reads configs/locks/
    channel = build_locked_channel("r4_itemcf", locked, base, paths, preprocess)
    candidates = channel.generate(queries, inputs)

Code that runs after the lock must NOT read configs/recall/*.yaml (the D-phase
defaults) or results/d_phase.csv to decide a configuration.
tests/unit/test_locked.py scans the post-lock code for those names.

R6 comes in two forms, both starting from a model file whose sha256 is checked
BEFORE its weights are loaded (a mismatch is an error: artifacts/ is not in git,
so the hash is the only proof that the file is the one that was recorded):

  - load_locked_twotower_channel: predict only, from a file named in the lock
    ("base" = the model that predicted D1; "f1_start" = the chain model after
    the D4 fine-tune, which predicts F1). Nothing is trained.
  - load_locked_twotower_chain: the weekly training chain of proposal §6.6 with
    the locked settings (E_update, replay, fine-tune lr, chain seed, alarm
    users). It returns the SAME TwoTowerChainChannel class the D phase used, so
    the weekly fine-tune is the same code. The caller gives the start model
    (hmrec.recall.chain_records decides which file that is for a fold) and
    drives "predict, save, then fine-tune" (hmrec.recall.generate).
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hmrec.config import BaseConfig, PreprocessConfig, parse_twotower_config
from hmrec.data.folds import Fold, FoldManifest
from hmrec.locks import (
    ALL_CHANNELS,
    BASE_CHECKPOINT,
    CHECKPOINT_NAMES,
    F1_START_CHECKPOINT,
    R6_CHANNEL,
    STATISTICAL_CHANNELS,
    ChannelsLock,
    LockedChannel,
    TwoTowerLock,
    check_file_sha256,
    load_channels_lock,
    load_twotower_lock,
)
from hmrec.logging_utils import get_logger
from hmrec.paths import ProjectPaths
from hmrec.recall.base import RecallChannel
from hmrec.recall.build import build_statistical_channel, twotower_chunks
from hmrec.recall.twotower.chain import TwoTowerChainChannel
from hmrec.recall.twotower.data import read_training_array
from hmrec.recall.twotower.infer import TwoTowerChannel, load_twotower_resources
from hmrec.recall.twotower.model import TwoTowerModel
from hmrec.recall.twotower.train import check_model_identity, choose_alarm_users, load_checkpoint

LOGGER = get_logger("recall.locked")

# Lock files write artifact paths as "artifacts/..." (the default artifacts root).
ARTIFACTS_PREFIX = "artifacts"
# The chain model that predicts F1 was saved after the fine-tune on the LAST D fold
# (M4 in proposal §6.4), as chain/{trial_id}/D4.pt.
F1_START_FOLD = "D4"
# Phases whose folds are generated after the lock, in time order (proposal §3).
POST_LOCK_PHASES = ("F", "V", "T")
# The alarm users are drawn from the base-window training arrays (proposal §6.4).
BASE_WINDOW_NAME = "base"
ALARM_SOURCE_ARRAY = "customer_idx"


@dataclass(frozen=True)
class LockedRecall:
    """Both lock files.

    Attributes:
        channels: configs/locks/channels.lock.yaml (R1-R5).
        twotower: configs/locks/twotower.lock.yaml (R6).
    """

    channels: ChannelsLock
    twotower: TwoTowerLock

    def channel(self, name: str) -> LockedChannel:
        """The lock of one of the six channels.

        Raises:
            ValueError: If `name` is not one of the six channels.
        """
        if name == R6_CHANNEL:
            return self.twotower.channel
        if name in self.channels.channels:
            return self.channels.channels[name]
        raise ValueError(f"unknown channel {name!r}; expected one of {ALL_CHANNELS}")


def load_locked_recall(paths: ProjectPaths) -> LockedRecall:
    """Read and check both lock files (strict; see hmrec.locks)."""
    return LockedRecall(
        channels=load_channels_lock(paths.channels_lock_file),
        twotower=load_twotower_lock(paths.twotower_lock_file),
    )


def post_lock_folds(manifest: FoldManifest) -> tuple[Fold, ...]:
    """The folds generated with the locked configuration: F1 ... T4, in time order.

    Raises:
        ValueError: If the fold right before the first of them is not D4. The
            locked "f1_start" model is the chain model after the D4 fine-tune,
            so it may only predict the week that directly follows D4.
    """
    folds = tuple(fold for fold in manifest.folds if fold.phase in POST_LOCK_PHASES)
    if not folds:
        raise ValueError(f"the fold manifest has no folds of phases {POST_LOCK_PHASES}")
    first_position = manifest.folds.index(folds[0])
    before = manifest.folds[first_position - 1].name if first_position > 0 else None
    if before != F1_START_FOLD:
        raise ValueError(
            f"the fold before {folds[0].name} is {before}, expected {F1_START_FOLD}: "
            "the locked f1_start model would not fit the first post-lock fold"
        )
    if manifest.folds[first_position:] != folds:
        raise ValueError("folds of phases F, V, T must be the last folds of the manifest")
    return folds


# ---------------------------------------------------------------- R6 model files


@dataclass(frozen=True)
class ModelFile:
    """A model file together with the sha256 it must have.

    Attributes:
        path: The checkpoint file.
        sha256: The recorded sha256 (lock file or results/m6_twotower_chain.csv).
    """

    path: Path
    sha256: str


def _artifact_path(paths: ProjectPaths, recorded: str) -> Path:
    """Turn an "artifacts/..." path of a lock file into a path under the artifacts root."""
    parts = Path(recorded).parts
    if not parts or parts[0] != ARTIFACTS_PREFIX:
        raise ValueError(f"lock path {recorded!r} must start with {ARTIFACTS_PREFIX!r}/")
    return paths.artifacts_root.joinpath(*parts[1:])


def locked_checkpoint_file(paths: ProjectPaths, lock: TwoTowerLock, name: str) -> Path:
    """Path of a locked R6 model file, from hmrec.paths.

    The path is derived from the locked config (base_epoch, trial_id), never
    taken from the lock file's text; the recorded path must then be the same
    file, otherwise the lock file and the code disagree about where models live.

    Args:
        paths: Project paths.
        lock: The R6 lock.
        name: "base" or "f1_start".

    Raises:
        ValueError: If `name` is unknown or the recorded path is another file.
    """
    if name not in CHECKPOINT_NAMES:
        raise ValueError(f"unknown locked checkpoint {name!r}; expected one of {CHECKPOINT_NAMES}")
    if name == BASE_CHECKPOINT:
        path = paths.twotower_checkpoint_file(lock.channel.config["base_epoch"])
    else:
        path = paths.twotower_chain_checkpoint_file(lock.channel.trial_id, F1_START_FOLD)
    recorded = _artifact_path(paths, lock.checkpoints[name].path)
    if recorded != path:
        raise ValueError(
            f"locked checkpoint {name!r}: the lock file records {recorded}, "
            f"but the locked config gives {path}"
        )
    return path


def load_verified_checkpoint(
    model: TwoTowerModel, path: Path, expected_sha256: str, identity: dict[str, Any]
) -> None:
    """Load a locked model file into `model`, but only if its sha256 is the locked one.

    Args:
        model: The model whose weights are replaced (in place).
        path: The checkpoint file.
        expected_sha256: The sha256 recorded in the lock file.
        identity: The code vocabularies of the current data (train.model_identity).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the sha256 differs (nothing is loaded), or the checkpoint
            was trained with other article / profile codes.
    """
    # Hash FIRST: a file that is not the locked one must never reach the model.
    check_file_sha256(path, expected_sha256)
    metadata = load_checkpoint(model, path)
    check_model_identity(metadata["model_identity"], identity, path)
    LOGGER.info("loaded locked checkpoint %s (sha256 %s)", path, expected_sha256)


def load_locked_twotower_channel(
    locked: LockedRecall,
    base: BaseConfig,
    paths: ProjectPaths,
    preprocess: PreprocessConfig,
    start: str,
) -> TwoTowerChannel:
    """R6 as a predict-only channel that starts from a locked model file.

    The base checkpoint is always checked and loaded first (every chain model
    was trained from it); for start = "f1_start" its weights are then replaced
    by the checked chain model.

    Args:
        locked: Both lock files.
        base: configs/base.yaml (chunk sizes).
        paths: Project paths.
        preprocess: Current preprocess config.
        start: "base" (the model that predicted D1) or "f1_start" (predicts F1).

    Raises:
        ValueError: On an unknown `start` or a sha256 mismatch.
    """
    lock = locked.twotower
    config = parse_twotower_config(lock.channel.config)
    start_file = locked_checkpoint_file(paths, lock, start)
    base_file = locked_checkpoint_file(paths, lock, BASE_CHECKPOINT)
    # load_twotower_resources reads the base checkpoint, so it is hashed before that.
    check_file_sha256(base_file, lock.checkpoints[BASE_CHECKPOINT].sha256)
    resources = load_twotower_resources(paths, preprocess, config)
    LOGGER.info("loaded locked base checkpoint %s", base_file)
    if start != BASE_CHECKPOINT:
        expected_sha256 = lock.checkpoints[start].sha256
        load_verified_checkpoint(resources.model, start_file, expected_sha256, resources.identity)
        resources.model.eval()
    return TwoTowerChannel(config, resources, twotower_chunks(base))


def load_locked_twotower_chain(
    locked: LockedRecall,
    base: BaseConfig,
    paths: ProjectPaths,
    preprocess: PreprocessConfig,
    start: ModelFile,
) -> TwoTowerChainChannel:
    """The weekly training chain with the locked settings, starting from a checked model.

    Returns the D-phase chain class (hmrec.recall.twotower.chain), configured
    only from the lock: E_update, replay, fine-tune learning rate, chain seed,
    and the locked alarm users (load_locked_alarm_users). Its
    fine_tune_on_week(fold, inputs) fine-tunes on the week [anchor, anchor + 7)
    with a NEW Adam and saves the model to
    artifacts/models/twotower/chain_locked/{fold}.pt (never overwritten).

    Args:
        locked: Both lock files.
        base: configs/base.yaml (chunk sizes, memory settings).
        paths: Project paths.
        preprocess: Current preprocess config.
        start: The model to start from and its recorded sha256: the locked
            f1_start for F1, else the model saved after the previous fold.

    Raises:
        FileNotFoundError: If a model file is missing.
        ValueError: If a sha256 differs from its record (nothing is loaded).
    """
    lock = locked.twotower
    config = parse_twotower_config(lock.channel.config)
    base_file = locked_checkpoint_file(paths, lock, BASE_CHECKPOINT)
    # load_twotower_resources reads the base checkpoint, so it is hashed before that.
    check_file_sha256(base_file, lock.checkpoints[BASE_CHECKPOINT].sha256)
    resources = load_twotower_resources(paths, preprocess, config)
    load_verified_checkpoint(resources.model, start.path, start.sha256, resources.identity)
    resources.model.eval()
    metadata = {
        "trial_id": lock.channel.trial_id,
        "config": asdict(config),
        "model_identity": resources.identity,
        "start": f"{start.path.name} (sha256 {start.sha256})",
        "lock_git_commit": lock.stamp.git_commit,
    }
    return TwoTowerChainChannel(
        config,
        resources,
        twotower_chunks(base),
        paths.twotower_locked_chain_checkpoint_file,
        load_locked_alarm_users(paths, lock),
        base.memory,
        metadata,
    )


def load_locked_alarm_users(paths: ProjectPaths, lock: TwoTowerLock) -> np.ndarray:
    """The alarm users of the training chain, drawn exactly as when the lock was written.

    The users are drawn from the base training array with the locked share and
    seed. The array's sha256 and the number of users are compared with the
    lock, so a regenerated array cannot silently change who is kept out.

    Returns:
        Sorted int32 array [n_alarm_users] of customer_idx.

    Raises:
        ValueError: If the source array is not the locked file, or the draw
            gives another number of users.
    """
    alarm = lock.alarm_users
    source = paths.twotower_data_dir(BASE_WINDOW_NAME) / f"{ALARM_SOURCE_ARRAY}.npy"
    recorded = _artifact_path(paths, alarm.source)
    if recorded != source:
        raise ValueError(f"alarm users: the lock file records {recorded}, the code reads {source}")
    check_file_sha256(source, alarm.source_sha256)
    customer_idx = read_training_array(
        paths.twotower_data_dir(BASE_WINDOW_NAME), ALARM_SOURCE_ARRAY
    )
    alarm_users = choose_alarm_users(customer_idx, alarm.share, alarm.seed)
    if alarm_users.size != alarm.n_alarm_users:
        raise ValueError(
            f"alarm users: drew {alarm_users.size} users, the lock records {alarm.n_alarm_users}"
        )
    return alarm_users


# ---------------------------------------------------------------- the entry


def build_locked_channel(
    channel: str,
    locked: LockedRecall,
    base: BaseConfig,
    paths: ProjectPaths,
    preprocess: PreprocessConfig,
    r6_start: str = F1_START_CHECKPOINT,
) -> RecallChannel:
    """Build one recall channel from the lock files.

    Args:
        channel: One of the six channel names.
        locked: Both lock files (load_locked_recall).
        base: configs/base.yaml (chunk sizes, threads; they do not change results).
        paths: Project paths (R6 model files and tables).
        preprocess: Current preprocess config (R6 tables).
        r6_start: Only for "r6_twotower": the locked model file to predict with.

    Raises:
        ValueError: On an unknown channel, or (R6) a model file whose sha256 is
            not the locked one.
    """
    if channel == R6_CHANNEL:
        return load_locked_twotower_channel(locked, base, paths, preprocess, r6_start)
    if channel not in STATISTICAL_CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; expected one of {ALL_CHANNELS}")
    return build_statistical_channel(channel, locked.channel(channel).config, base)


def build_locked_channels(
    locked: LockedRecall,
    base: BaseConfig,
    paths: ProjectPaths,
    preprocess: PreprocessConfig,
    r6_start: str = F1_START_CHECKPOINT,
    channels: Sequence[str] = ALL_CHANNELS,
) -> dict[str, RecallChannel]:
    """build_locked_channel for several channels (default: all six, R1 ... R6)."""
    return {
        channel: build_locked_channel(channel, locked, base, paths, preprocess, r6_start)
        for channel in channels
    }
