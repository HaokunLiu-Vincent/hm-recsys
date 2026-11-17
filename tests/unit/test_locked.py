"""Tests for recall.locked: channels from the lock files, sha256 before loading, alarm users.

R1-R5 are built from the REAL lock files (building a channel reads no data).
The R6 model tests use the tiny model of test_twotower_infer and checkpoints in
tmp_path; the real model files are only checked by scripts/m5_verify_locks.py.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
from test_twotower_infer import _resources

from hmrec.config import (
    BaseConfig,
    ItemCFConfig,
    WalsConfig,
    load_base_config,
    load_preprocess_config,
)
from hmrec.data.folds import FoldManifest, load_fold_manifest
from hmrec.locks import (
    ALL_CHANNELS,
    BASE_CHECKPOINT,
    F1_START_CHECKPOINT,
    R6_CHANNEL,
    STATISTICAL_CHANNELS,
    AlarmUsersLock,
    LockedCheckpoint,
    TwoTowerLock,
    file_sha256,
)
from hmrec.paths import ProjectPaths, build_paths, get_paths
from hmrec.recall.build import build_statistical_channel
from hmrec.recall.locked import (
    LockedRecall,
    build_locked_channel,
    build_locked_channels,
    load_locked_alarm_users,
    load_locked_recall,
    load_verified_checkpoint,
    locked_checkpoint_file,
    post_lock_folds,
)
from hmrec.recall.r1_popularity import PopularityChannel
from hmrec.recall.r2_repurchase import RepurchaseChannel
from hmrec.recall.r3_sibling import SiblingChannel
from hmrec.recall.r4_itemcf import ItemCFChannel
from hmrec.recall.r5_wals import WalsChannel
from hmrec.recall.twotower.train import choose_alarm_users, save_checkpoint

REPO = Path(__file__).resolve().parents[2]
PATHS = get_paths()
BASE = load_base_config(REPO / "configs" / "base.yaml")
PREPROCESS = load_preprocess_config(REPO / "configs" / "preprocess.yaml")
LOCKED = load_locked_recall(PATHS)


def _tmp_paths(tmp_path: Path) -> ProjectPaths:
    """Project paths whose artifacts root is tmp_path/artifacts."""
    return build_paths(BASE, tmp_path)


# ---------------------------------------------------------------- R1-R5 from the lock files


def test_channel_lookup_covers_the_six_channels() -> None:
    assert [LOCKED.channel(name).channel for name in ALL_CHANNELS] == list(ALL_CHANNELS)
    with pytest.raises(ValueError, match="unknown channel"):
        LOCKED.channel("r7_graph")


def test_statistical_channels_are_built_with_the_locked_configs() -> None:
    channels = build_locked_channels(LOCKED, BASE, PATHS, PREPROCESS, channels=STATISTICAL_CHANNELS)
    assert list(channels) == list(STATISTICAL_CHANNELS)
    r1, r2, r3, r4, r5 = channels.values()
    assert isinstance(r1, PopularityChannel)
    assert (r1.config.variant, r1.config.tau_days, r1.config.quota) == ("A1", 3.0, 100)
    assert isinstance(r2, RepurchaseChannel)
    assert (r2.config.variant, r2.config.lookback_days, r2.config.quota) == (
        "decayed_frequency",
        365,
        30,
    )
    assert isinstance(r3, SiblingChannel)
    assert (r3.config.gamma, r3.config.tau_days, r3.config.active_pool_only) == (1.0, 30.0, True)
    assert isinstance(r4, ItemCFChannel)
    assert r4.config == ItemCFConfig(
        variant="CF3",
        fit_window_days=365,
        seed_window_days=365,
        alpha=0.7,
        neighbors_per_seed=100,
        seed_decay=True,
        seed_tau_days=30.0,
        quota=100,
    )
    assert isinstance(r5, WalsChannel)
    assert r5.config == WalsConfig(
        variant="W2",
        window_days=180,
        decay_tau_days=60.0,
        alpha_conf=60.0,
        regularization=0.1,
        factors=128,
        iterations=15,
        random_seed=42,
        quota=100,
    )


def test_chunk_sizes_come_from_base_yaml_not_from_the_lock() -> None:
    chunking = dataclasses.replace(BASE.chunking, item_chunk_size=7)
    base: BaseConfig = dataclasses.replace(BASE, chunking=chunking)
    r4 = build_locked_channel("r4_itemcf", LOCKED, base, PATHS, PREPROCESS)
    assert isinstance(r4, ItemCFChannel)
    assert r4.item_chunk_size == 7


def test_unknown_channel_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        build_locked_channel("r7_graph", LOCKED, BASE, PATHS, PREPROCESS)


# ---------------------------------------------------------------- R6 model files


def test_locked_checkpoint_paths_follow_the_artifacts_root(tmp_path: Path) -> None:
    paths = _tmp_paths(tmp_path)
    models = tmp_path / "artifacts" / "models" / "twotower"
    lock = LOCKED.twotower
    assert locked_checkpoint_file(paths, lock, BASE_CHECKPOINT) == models / "base" / "epoch_4.pt"
    assert locked_checkpoint_file(paths, lock, F1_START_CHECKPOINT) == (
        models / "chain" / "835ef201ee4d" / "D4.pt"
    )
    with pytest.raises(ValueError, match="unknown locked checkpoint"):
        locked_checkpoint_file(paths, lock, "v1_start")


def _with_checkpoint_path(lock: TwoTowerLock, name: str, path: str) -> TwoTowerLock:
    checkpoints = dict(lock.checkpoints)
    checkpoints[name] = LockedCheckpoint(name, path, checkpoints[name].sha256)
    return dataclasses.replace(lock, checkpoints=checkpoints)


def test_a_recorded_path_that_is_another_file_is_refused(tmp_path: Path) -> None:
    paths = _tmp_paths(tmp_path)
    # The locked config says base_epoch 4, but the lock file points at epoch 5.
    other_epoch = _with_checkpoint_path(
        LOCKED.twotower, BASE_CHECKPOINT, "artifacts/models/twotower/base/epoch_5.pt"
    )
    with pytest.raises(ValueError, match="but the locked config gives"):
        locked_checkpoint_file(paths, other_epoch, BASE_CHECKPOINT)
    outside = _with_checkpoint_path(LOCKED.twotower, BASE_CHECKPOINT, "models/base/epoch_4.pt")
    with pytest.raises(ValueError, match="must start with 'artifacts'"):
        locked_checkpoint_file(paths, outside, BASE_CHECKPOINT)


def _saved_model_file(tmp_path: Path) -> tuple[Path, torch.Tensor]:
    """A checkpoint whose article embedding differs from a freshly built tiny model."""
    saved = _resources()
    with torch.no_grad():
        saved.model.item_tower.article_embedding.weight[:6].normal_(0.0, 2.0)
    path = tmp_path / "model.pt"
    save_checkpoint(saved.model, path, {"model_identity": {}})
    return path, saved.model.item_tower.article_embedding.weight.detach().clone()


def test_checkpoint_with_the_locked_sha256_is_loaded(tmp_path: Path) -> None:
    path, saved_weight = _saved_model_file(tmp_path)
    model = _resources().model
    assert not torch.equal(model.item_tower.article_embedding.weight, saved_weight)
    load_verified_checkpoint(model, path, file_sha256(path), identity={})
    assert torch.equal(model.item_tower.article_embedding.weight, saved_weight)


def test_checkpoint_with_another_sha256_is_refused_and_nothing_is_loaded(tmp_path: Path) -> None:
    path, _ = _saved_model_file(tmp_path)
    locked_sha256 = file_sha256(path)
    # The file is replaced after the lock (here: one more byte at the end).
    with path.open("ab") as handle:
        handle.write(b"\0")
    model = _resources().model
    before = model.item_tower.article_embedding.weight.detach().clone()
    with pytest.raises(ValueError, match="This is not the locked file"):
        load_verified_checkpoint(model, path, locked_sha256, identity={})
    # The check runs BEFORE torch.load: the model is untouched.
    assert torch.equal(model.item_tower.article_embedding.weight, before)


def test_checkpoint_trained_with_other_codes_is_refused(tmp_path: Path) -> None:
    path, _ = _saved_model_file(tmp_path)  # saved with model_identity {}
    with pytest.raises(ValueError, match="n_articles differs from the current data"):
        load_verified_checkpoint(
            _resources().model, path, file_sha256(path), identity={"n_articles": 6}
        )


# ---------------------------------------------------------------- alarm users


def _alarm_lock(tmp_path: Path, n_alarm_users: int) -> tuple[ProjectPaths, TwoTowerLock]:
    """A lock for a tiny base array: customers 0..9 with several samples each."""
    paths = _tmp_paths(tmp_path)
    source = paths.twotower_data_dir("base") / "customer_idx.npy"
    source.parent.mkdir(parents=True)
    np.save(source, np.repeat(np.arange(10, dtype=np.int32), 3))
    alarm = AlarmUsersLock(
        share=0.3,
        seed=5,
        source="artifacts/twotower/data/base/customer_idx.npy",
        source_sha256=file_sha256(source),
        n_alarm_users=n_alarm_users,
    )
    return paths, dataclasses.replace(LOCKED.twotower, alarm_users=alarm)


def test_alarm_users_are_the_same_draw_as_in_training(tmp_path: Path) -> None:
    # share 0.3 of 10 distinct users -> 3 users, drawn with seed 5.
    paths, lock = _alarm_lock(tmp_path, n_alarm_users=3)
    alarm_users = load_locked_alarm_users(paths, lock)
    expected = choose_alarm_users(np.arange(10, dtype=np.int32), 0.3, 5)
    assert alarm_users.tolist() == expected.tolist()
    assert alarm_users.size == 3


def test_alarm_users_from_a_changed_array_are_refused(tmp_path: Path) -> None:
    paths, lock = _alarm_lock(tmp_path, n_alarm_users=3)
    source = paths.twotower_data_dir("base") / "customer_idx.npy"
    np.save(source, np.repeat(np.arange(11, dtype=np.int32), 3))  # regenerated: one more user
    with pytest.raises(ValueError, match="This is not the locked file"):
        load_locked_alarm_users(paths, lock)


def test_alarm_user_count_must_match_the_lock(tmp_path: Path) -> None:
    paths, lock = _alarm_lock(tmp_path, n_alarm_users=4)
    with pytest.raises(ValueError, match="drew 3 users, the lock records 4"):
        load_locked_alarm_users(paths, lock)


# ---------------------------------------------------------------- the entry is the only way

# Code that runs after the lock. It must take every recall configuration from
# configs/locks/ (through hmrec.locks / hmrec.recall.locked).
POST_LOCK_FILES = (
    "src/hmrec/locks.py",
    "src/hmrec/recall/locked.py",
    "src/hmrec/recall/build.py",
    "src/hmrec/recall/generate.py",
    "src/hmrec/recall/chain_records.py",
    "src/hmrec/recall/inputs.py",
    "src/hmrec/eval/monitor.py",
    "src/hmrec/pipeline.py",
    "src/hmrec/run_records.py",
    "src/hmrec/csv_records.py",
    "scripts/m6_generate_candidates.py",
    "scripts/m6_monitor_candidates.py",
    "scripts/m6_run_folds.py",
    "scripts/m6_build_features.py",
    "scripts/m7_ranker_hpo.py",
    "scripts/m8_final_eval.py",
)
POST_LOCK_FOLDERS = ("src/hmrec/fusion", "src/hmrec/features", "src/hmrec/ranker")
# Accessors of the pre-lock sources: the D-phase default configs and the D-phase results.
PRE_LOCK_NAMES = (
    "recall_config_file",
    "d_phase_results_file",
    "m4_trials_file",
    "load_popularity_config",
    "load_repurchase_config",
    "load_sibling_config",
    "load_itemcf_config",
    "load_wals_config",
    "load_twotower_config",
)


def test_post_lock_code_never_reads_default_configs_or_d_phase_results() -> None:
    files = [REPO / name for name in POST_LOCK_FILES]
    for folder in POST_LOCK_FOLDERS:
        files.extend(sorted((REPO / folder).rglob("*.py")))
    assert all(path.is_file() for path in files)
    offenders = [
        f"{path.relative_to(REPO)}: {name}"
        for path in files
        for name in PRE_LOCK_NAMES
        if name in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_r6_is_never_built_without_a_model_file_check() -> None:
    # The entry routes R6 to load_locked_twotower_channel (sha256 first). A config
    # alone must not be enough to get an R6 channel from the R1-R5 builder.
    with pytest.raises(ValueError, match="not one of the channels R1-R5"):
        build_statistical_channel(R6_CHANNEL, LOCKED.channel(R6_CHANNEL).config, BASE)


def test_locked_recall_is_a_plain_pair_of_the_two_lock_files() -> None:
    assert isinstance(LOCKED, LockedRecall)
    assert LOCKED.channels.stamp.git_commit == LOCKED.twotower.stamp.git_commit


# ---------------------------------------------------------------- post-lock folds, no labels


def test_post_lock_folds_are_f1_to_t4_right_after_d4() -> None:
    manifest = load_fold_manifest(PATHS.fold_manifest_file)
    folds = post_lock_folds(manifest)
    assert [fold.name for fold in folds] == [
        "F1", "F2", "F3", "F4", "V1", "V2", "V3", "V4", "T1", "T2", "T3", "T4",
    ]  # fmt: skip
    # Without D4 in front of F1 the locked f1_start model would predict the wrong week.
    without_d4 = FoldManifest(tuple(fold for fold in manifest.folds if fold.name != "D4"))
    with pytest.raises(ValueError, match="the fold before F1 is D3, expected D4"):
        post_lock_folds(without_d4)
    only_d = FoldManifest(tuple(fold for fold in manifest.folds if fold.phase == "D"))
    with pytest.raises(ValueError, match="no folds of phases"):
        post_lock_folds(only_d)


# Candidate generation: everything under recall/, its script and the fold driver.
# None of it may read labels (AGENTS.md §3.1). In M6 only the monitor script does,
# and only through hmrec.eval.monitor.load_label_evaluator (F and V folds).
LABEL_NAMES = ("labels_file", "FoldEvaluator", "load_label_evaluator", "label_week_transactions")


def test_candidate_generation_code_never_reads_labels() -> None:
    files = sorted((REPO / "src" / "hmrec" / "recall").rglob("*.py"))
    files += [
        REPO / "scripts" / "m6_generate_candidates.py",
        REPO / "scripts" / "m6_run_folds.py",
        REPO / "src" / "hmrec" / "pipeline.py",
        REPO / "src" / "hmrec" / "run_records.py",
        REPO / "src" / "hmrec" / "locks.py",
    ]
    offenders = [
        f"{path.relative_to(REPO)}: {name}"
        for path in files
        for name in LABEL_NAMES
        if name in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_the_monitor_script_gets_labels_only_through_the_phase_guard() -> None:
    script = (REPO / "scripts" / "m6_monitor_candidates.py").read_text(encoding="utf-8")
    # It may name the labels file, but it must hand it to load_label_evaluator (which
    # refuses T folds) and never build a FoldEvaluator itself.
    assert "load_label_evaluator(" in script
    assert "FoldEvaluator" not in script
    assert "label_week_transactions" not in script
