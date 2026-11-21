"""Tests for hmrec.ranker.platform_entry: the tolerance-accepted entry to the locked Ranker.

As in test_ranker_locked.py, the real configs/ folder is copied to a temporary
folder and one thing is changed at a time. The entry takes the platform
(sys.platform) and the runtime as arguments, so the Linux branch is tested on
the laptop too, and the macOS branch inside a container.
"""

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from hmrec.config import BaseConfig, load_base_config, load_yaml
from hmrec.paths import ProjectPaths, build_paths, calibration_output_paths
from hmrec.ranker.locked import load_locked_ranker
from hmrec.ranker.platform_entry import (
    MODE_ACCEPTED,
    MODE_CALIBRATION,
    PlatformNotAccepted,
    load_locked_ranker_for_platform,
    require_calibration_outputs,
)
from hmrec.ranker.runtime import LightGBMRuntime

REPO = Path(__file__).resolve().parents[2]
# What hmrec.ranker.runtime.read_lightgbm_runtime gives in the arm64 image (measured).
CONTAINER_RUNTIME = LightGBMRuntime(
    lightgbm_version="4.7.0",
    libomp_path="/usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0",
    libomp_version="libgomp1 14.2.0-19",
)
# The runtime the V phase ran with on the laptop (every row of results/v_phase.csv).
LAPTOP_RUNTIME = LightGBMRuntime(
    lightgbm_version="4.7.0",
    libomp_path="/opt/homebrew/Cellar/libomp/22.1.4/lib/libomp.dylib",
    libomp_version="22.1.4",
)


def _workspace(root: Path) -> tuple[ProjectPaths, BaseConfig, ProjectPaths]:
    """A copy of configs/ under `root`; project paths; the calibration output paths."""
    shutil.copytree(REPO / "configs", root / "configs")
    base = load_base_config(root / "configs" / "base.yaml")
    paths = build_paths(base, root)
    return paths, base, calibration_output_paths(paths, "linux-arm64", "run1", "main")


def _rewrite(path: Path, section: str, key: str, value: Any) -> None:
    """Set raw[section][key] = value in a YAML file of the copy."""
    raw = load_yaml(path)
    raw[section][key] = value
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_calibration_mode_on_linux_gives_the_same_configuration_as_the_strict_entry(
    tmp_path: Path,
) -> None:
    paths, base, output = _workspace(tmp_path)
    result = load_locked_ranker_for_platform(
        paths, base, CONTAINER_RUNTIME, "linux", MODE_CALIBRATION, output
    )
    strict = load_locked_ranker(paths, base, LAPTOP_RUNTIME)
    # Every value of the configuration is the one the strict entry returns on the laptop.
    assert result.locked == strict
    assert result.locked.config_id == "ca1a3cdfce49"
    # The OpenMP runtime is recorded as found; it is not the one in the lock.
    assert result.mode == MODE_CALIBRATION
    assert result.runtime == CONTAINER_RUNTIME
    assert result.runtime.libomp_version != result.locked.lock.runtime.libomp_version


def test_the_strict_entry_still_refuses_the_container_runtime(tmp_path: Path) -> None:
    # This module adds an entry; it does not weaken the first one.
    paths, base, _ = _workspace(tmp_path)
    with pytest.raises(ValueError, match="libomp version: 'libgomp1 14.2.0-19' here"):
        load_locked_ranker(paths, base, CONTAINER_RUNTIME)


@pytest.mark.parametrize("system", ["darwin", "win32"])
def test_the_entry_refuses_everything_that_is_not_linux(tmp_path: Path, system: str) -> None:
    # Also with the laptop's own runtime: on macOS the strict entry is the only one.
    paths, base, output = _workspace(tmp_path)
    for runtime in (LAPTOP_RUNTIME, CONTAINER_RUNTIME):
        with pytest.raises(RuntimeError, match="runs on Linux only"):
            load_locked_ranker_for_platform(paths, base, runtime, system, MODE_CALIBRATION, output)


def test_accepted_mode_refuses_because_no_platform_record_exists(tmp_path: Path) -> None:
    paths, base, output = _workspace(tmp_path)
    with pytest.raises(PlatformNotAccepted, match="no platform record exists"):
        load_locked_ranker_for_platform(
            paths, base, CONTAINER_RUNTIME, "linux", MODE_ACCEPTED, output
        )


def test_an_unknown_mode_is_refused(tmp_path: Path) -> None:
    paths, base, output = _workspace(tmp_path)
    with pytest.raises(ValueError, match="mode must be one of"):
        load_locked_ranker_for_platform(paths, base, CONTAINER_RUNTIME, "linux", "replay", output)


def test_another_lightgbm_version_or_thread_count_is_refused(tmp_path: Path) -> None:
    paths, base, output = _workspace(tmp_path)
    other_version = replace(CONTAINER_RUNTIME, lightgbm_version="4.6.0")
    with pytest.raises(ValueError, match="LightGBM version: '4.6.0' here, '4.7.0' in the lock"):
        load_locked_ranker_for_platform(
            paths, base, other_version, "linux", MODE_CALIBRATION, output
        )
    with pytest.raises(ValueError, match="num_threads: 4 here, 6 in the lock"):
        load_locked_ranker_for_platform(
            paths,
            replace(base, num_threads=4),
            CONTAINER_RUNTIME,
            "linux",
            MODE_CALIBRATION,
            output,
        )


def test_the_config_id_check_is_the_one_of_the_strict_entry(tmp_path: Path) -> None:
    paths, base, output = _workspace(tmp_path)
    _rewrite(paths.ranker_lock_file, "model", "num_leaves", 63)
    with pytest.raises(ValueError, match="The locked settings were changed"):
        load_locked_ranker_for_platform(
            paths, base, CONTAINER_RUNTIME, "linux", MODE_CALIBRATION, output
        )


def test_a_changed_referenced_lock_file_is_refused(tmp_path: Path) -> None:
    paths, base, output = _workspace(tmp_path)
    with paths.channels_lock_file.open("a", encoding="utf-8") as handle:
        handle.write("\n# edited\n")
    with pytest.raises(ValueError, match="sha256"):
        load_locked_ranker_for_platform(
            paths, base, CONTAINER_RUNTIME, "linux", MODE_CALIBRATION, output
        )


@pytest.mark.parametrize(
    "runtime",
    [
        # No OpenMP runtime found in the process.
        LightGBMRuntime("4.7.0", "not loaded", "unknown"),
        # A file no Debian package owns, e.g. the copy of libgomp inside the torch wheel.
        LightGBMRuntime(
            "4.7.0", "/app/.venv/lib/python3.12/site-packages/torch/lib/x.so", "unknown"
        ),
    ],
)
def test_a_runtime_that_cannot_be_named_is_refused(
    tmp_path: Path, runtime: LightGBMRuntime
) -> None:
    paths, base, output = _workspace(tmp_path)
    with pytest.raises(ValueError, match="cannot be identified"):
        load_locked_ranker_for_platform(paths, base, runtime, "linux", MODE_CALIBRATION, output)


def test_calibration_mode_may_only_write_into_the_calibration_folders(tmp_path: Path) -> None:
    paths, base, output = _workspace(tmp_path)
    require_calibration_outputs(paths, output)
    # The stage-1 locations themselves: both roots are refused.
    with pytest.raises(ValueError, match="may write artifacts only under"):
        load_locked_ranker_for_platform(
            paths, base, CONTAINER_RUNTIME, "linux", MODE_CALIBRATION, paths
        )
    stage_one_results = replace(output, results_root=paths.results_root)
    with pytest.raises(ValueError, match="may write result tables only under"):
        require_calibration_outputs(paths, stage_one_results)
    # A folder that only starts with the same name is not inside the calibration folder.
    sibling = replace(output, artifacts_root=paths.artifacts_root / "m10_calibration_other")
    with pytest.raises(ValueError, match="may write artifacts only under"):
        require_calibration_outputs(paths, sibling)
