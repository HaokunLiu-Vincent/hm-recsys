"""Tests for hmrec.ranker.locked: the only entry to the Ranker configuration after the lock.

The tests copy the real configs/ folder to a temporary folder and change one
thing at a time there, so every check runs against the real lock file's layout.
No data file and no result table is read.
"""

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from hmrec.config import BaseConfig, load_base_config, load_ranker_config, load_yaml
from hmrec.locks import load_ranker_lock
from hmrec.paths import ProjectPaths, build_paths
from hmrec.ranker.locked import (
    check_lock_config_id,
    check_referenced_locks,
    check_runtime_matches_lock,
    load_locked_ranker,
    locked_config_id,
    locked_ranker_config,
    locked_tuned_params,
)
from hmrec.ranker.runtime import LightGBMRuntime
from hmrec.ranker.train import dataset_params, training_params

REPO = Path(__file__).resolve().parents[2]
# The runtime the V phase ran with (every row of results/v_phase.csv records it).
RECORDED_RUNTIME = LightGBMRuntime(
    lightgbm_version="4.7.0",
    libomp_path="/opt/homebrew/Cellar/libomp/22.1.4/lib/libomp.dylib",
    libomp_version="22.1.4",
)


def _workspace(root: Path) -> tuple[ProjectPaths, BaseConfig]:
    """A copy of the repository's configs/ under `root`, with paths that point to it."""
    shutil.copytree(REPO / "configs", root / "configs")
    base = load_base_config(root / "configs" / "base.yaml")
    return build_paths(base, root), base


def _rewrite(path: Path, section: str, key: str, value: Any) -> None:
    """Set raw[section][key] = value in a YAML file (comments are lost; the copy is a test file)."""
    raw = load_yaml(path)
    raw[section][key] = value
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


# ---------------------------------------------------------------- the real lock file


def test_the_entry_builds_the_locked_configuration(tmp_path: Path) -> None:
    paths, base = _workspace(tmp_path)
    locked = load_locked_ranker(paths, base, RECORDED_RUNTIME)
    assert locked.config_id == "ca1a3cdfce49"
    assert (locked.tuned.num_leaves, locked.tuned.min_data_in_leaf) == (31, 500)
    assert locked.tuned.negatives_per_positive == 40
    assert locked.num_trees == locked.config.num_trees == 500
    assert locked.histogram_mode == locked.config.histogram.mode == "row_wise"
    assert locked.num_threads == base.num_threads == 6
    # Only the locked ratio is left, so no other sample can be asked for.
    assert locked.config.sampling.ratios == (40,)
    assert locked.config.tuned_defaults == locked.tuned
    assert (locked.config.window_weeks, locked.config.top_k) == (4, 12)
    assert locked.config.checkpoint_every == 100


def test_the_locked_configuration_gives_the_lightgbm_parameters_of_the_v_phase(
    tmp_path: Path,
) -> None:
    paths, base = _workspace(tmp_path)
    locked = load_locked_ranker(paths, base, RECORDED_RUNTIME)
    params = training_params(locked.config, locked.tuned, locked.histogram_mode, locked.num_threads)
    # 7 bin parameters + 24 fixed training parameters + num_leaves, min_data_in_leaf,
    # seed, num_threads and the two histogram switches.
    assert params["num_leaves"] == 31 and params["min_data_in_leaf"] == 500
    assert params["learning_rate"] == 0.05 and params["objective"] == "lambdarank"
    assert params["force_row_wise"] is True
    assert params["num_threads"] == 6 and params["seed"] == 42
    bins = dataset_params(locked.config, locked.num_threads)
    locked_bins = locked.lock.model.dataset_params
    assert {key: bins[key] for key in locked_bins} == locked_bins
    assert bins["max_bin"] == 255 and bins["seed"] == 42 and bins["num_threads"] == 6


def test_result_defining_values_come_from_the_lock_not_from_ranker_yaml(tmp_path: Path) -> None:
    paths, base = _workspace(tmp_path)
    before = load_locked_ranker(paths, base, RECORDED_RUNTIME)
    # Edits of configs/ranker.yaml after the lock: the tuned defaults, a fixed
    # parameter, the grid's tree limit and the window. None of them may reach M8.
    ranker_file = paths.ranker_config_file
    _rewrite(ranker_file, "tuned_defaults", "num_leaves", 63)
    _rewrite(ranker_file, "train_params", "learning_rate", 0.1)
    _rewrite(ranker_file, "histogram", "mode", "col_wise")
    raw = load_yaml(ranker_file)
    raw["num_trees"], raw["window_weeks"], raw["seed"] = 2000, 6, 7
    ranker_file.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    after = load_locked_ranker(paths, base, RECORDED_RUNTIME)
    assert after.config == before.config and after.tuned == before.tuned
    # The two memory / logging settings DO come from configs/ranker.yaml.
    raw["scoring_query_chunk_size"], raw["train_metric_every"] = 1000, 50
    ranker_file.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    smaller = load_locked_ranker(paths, base, RECORDED_RUNTIME)
    assert smaller.config.scoring_query_chunk_size == 1000
    assert smaller.config.train_metric_every == 50
    assert smaller.config_id == before.config_id  # memory settings are not part of the id


# ---------------------------------------------------------------- the checks


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("model", "num_leaves", 63),
        ("model", "min_data_in_leaf", 100),
        ("model", "seed", 43),
        ("model", "histogram_mode", "col_wise"),
        ("sampling", "negatives_per_positive", 80),
        ("sampling", "seed", 7),
        ("training_window", "window_weeks", 5),
    ],
)
def test_an_edited_model_setting_no_longer_gives_the_recorded_id(
    tmp_path: Path, section: str, key: str, value: Any
) -> None:
    paths, base = _workspace(tmp_path)
    _rewrite(paths.ranker_lock_file, section, key, value)
    with pytest.raises(ValueError, match="The locked settings were changed"):
        load_locked_ranker(paths, base, RECORDED_RUNTIME)


def test_an_edited_lightgbm_parameter_no_longer_gives_the_recorded_id(tmp_path: Path) -> None:
    paths, base = _workspace(tmp_path)
    raw = load_yaml(paths.ranker_lock_file)
    raw["model"]["train_params"]["lambdarank_truncation_level"] = 15
    paths.ranker_lock_file.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    lock = load_ranker_lock(paths.ranker_lock_file)
    config = locked_ranker_config(lock, load_ranker_config(paths.ranker_config_file))
    assert locked_config_id(lock, config) != lock.selection.config_id
    with pytest.raises(ValueError, match="records ca1a3cdfce49"):
        check_lock_config_id(lock, config)


def test_a_changed_features_channels_or_twotower_lock_is_refused(tmp_path: Path) -> None:
    paths, base = _workspace(tmp_path)
    lock = load_ranker_lock(paths.ranker_lock_file)
    check_referenced_locks(lock, paths)  # the real files: no error
    referenced = (paths.features_lock_file, paths.channels_lock_file, paths.twotower_lock_file)
    for file in referenced:
        original = file.read_bytes()
        file.write_bytes(original + b"# one more comment\n")
        with pytest.raises(ValueError, match="This is not the locked file") as error:
            load_locked_ranker(paths, base, RECORDED_RUNTIME)
        assert file.name in str(error.value)  # the message names the changed file
        file.write_bytes(original)
        load_locked_ranker(paths, base, RECORDED_RUNTIME)  # restored: accepted again
    for file in referenced:
        original = file.read_bytes()
        file.unlink()
        with pytest.raises(FileNotFoundError, match="locked file not found"):
            load_locked_ranker(paths, base, RECORDED_RUNTIME)
        file.write_bytes(original)


def test_a_reference_to_another_file_is_refused(tmp_path: Path) -> None:
    paths, base = _workspace(tmp_path)
    raw = load_yaml(paths.ranker_lock_file)
    # Another real lock file under the name "features_lock": a real file, the wrong one.
    raw["references"]["features_lock"]["path"] = "configs/locks/twotower.lock.yaml"
    paths.ranker_lock_file.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 'configs/locks/features.lock.yaml'"):
        load_locked_ranker(paths, base, RECORDED_RUNTIME)


def test_another_runtime_or_thread_count_is_refused(tmp_path: Path) -> None:
    paths, base = _workspace(tmp_path)
    lock = load_ranker_lock(paths.ranker_lock_file)
    check_runtime_matches_lock(lock, RECORDED_RUNTIME, 6)  # no error
    newer_lightgbm = LightGBMRuntime("4.8.0", RECORDED_RUNTIME.libomp_path, "22.1.4")
    with pytest.raises(ValueError, match="LightGBM version: '4.8.0' here, '4.7.0' in the lock"):
        load_locked_ranker(paths, base, newer_lightgbm)
    upgraded = LightGBMRuntime(
        "4.7.0", "/opt/homebrew/Cellar/libomp/23.0.0/lib/libomp.dylib", "23.0.0"
    )
    with pytest.raises(ValueError, match="libomp version: '23.0.0' here") as error:
        load_locked_ranker(paths, base, upgraded)
    assert "/opt/homebrew/Cellar/libomp/23.0.0" in str(error.value)  # where it was loaded from
    # A libomp outside Homebrew has no readable version: also not the recorded one.
    unknown = LightGBMRuntime("4.7.0", "/x/lib/libomp.dylib", "unknown")
    with pytest.raises(ValueError, match="libomp version: 'unknown' here"):
        load_locked_ranker(paths, base, unknown)
    with pytest.raises(ValueError, match="num_threads: 8 here, 6 in the lock"):
        check_runtime_matches_lock(lock, RECORDED_RUNTIME, 8)


def test_a_linux_container_is_not_the_locked_runtime(tmp_path: Path) -> None:
    # In a Linux container LightGBM runs on GNU libgomp, not on the Homebrew libomp the
    # V phase ran on (hmrec.ranker.runtime records it as "<package> <version>"). The lock
    # is not changed and the check is not relaxed for it: the entry refuses, so a Linux
    # result is never taken for a bit-identical repeat of the laptop (docs/mlops.md §8.3).
    paths, base = _workspace(tmp_path)
    container = LightGBMRuntime(
        "4.7.0", "/usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0", "libgomp1 14.2.0-19"
    )
    with pytest.raises(ValueError, match="libomp version: 'libgomp1 14.2.0-19' here") as error:
        load_locked_ranker(paths, base, container)
    assert "'22.1.4' in the lock" in str(error.value)
    assert "/usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0" in str(error.value)
    # The same LightGBM version and thread count do not make it pass.
    assert "LightGBM version" not in str(error.value)
    assert "num_threads" not in str(error.value)


def test_several_differences_are_all_named(tmp_path: Path) -> None:
    paths, _ = _workspace(tmp_path)
    lock = load_ranker_lock(paths.ranker_lock_file)
    # All three at once: the message names each one.
    with pytest.raises(ValueError) as error:
        check_runtime_matches_lock(lock, LightGBMRuntime("5.0.0", "/x", "1.0"), 4)
    message = str(error.value)
    assert "LightGBM version" in message and "libomp version" in message
    assert "num_threads" in message


def test_tuned_params_are_read_from_two_sections_of_the_lock() -> None:
    lock = load_ranker_lock(REPO / "configs" / "locks" / "ranker.lock.yaml")
    tuned = locked_tuned_params(lock)
    assert (tuned.num_leaves, tuned.min_data_in_leaf) == (31, 500)  # `model`
    assert tuned.negatives_per_positive == 40  # `sampling`
