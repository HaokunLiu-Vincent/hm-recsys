"""Tests for hmrec.ranker.platform_entry: the tolerance-accepted entry to the locked Ranker.

As in test_ranker_locked.py, the real configs/ folder is copied to a temporary
folder and one thing is changed at a time. The entry takes the platform
(sys.platform) and the runtime as arguments, so the Linux branch is tested on
the laptop too, and the macOS branch inside a container.
"""

import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from hmrec.config import BaseConfig, load_base_config, load_yaml
from hmrec.paths import (
    PIPELINE_STATE_ROOT_VARIABLE,
    PipelinePaths,
    ProjectPaths,
    StateLayout,
    build_paths,
    calibration_output_paths,
    pipeline_paths,
    tick_root_paths,
)
from hmrec.ranker.locked import load_locked_ranker
from hmrec.ranker.platform_entry import (
    MODE_ACCEPTED,
    MODE_CALIBRATION,
    PlatformNotAccepted,
    load_locked_ranker_for_platform,
    platform_record_name,
    require_calibration_outputs,
)
from hmrec.ranker.runtime import LightGBMRuntime

REPO = Path(__file__).resolve().parents[2]
V1_ANCHOR = date(2020, 7, 29)
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
        paths, base, CONTAINER_RUNTIME, "linux", "aarch64", MODE_CALIBRATION, output
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
            load_locked_ranker_for_platform(
                paths, base, runtime, system, "aarch64", MODE_CALIBRATION, output
            )


def test_an_unknown_mode_is_refused(tmp_path: Path) -> None:
    paths, base, output = _workspace(tmp_path)
    with pytest.raises(ValueError, match="mode must be one of"):
        load_locked_ranker_for_platform(
            paths, base, CONTAINER_RUNTIME, "linux", "aarch64", "replay", output
        )


def test_another_lightgbm_version_or_thread_count_is_refused(tmp_path: Path) -> None:
    paths, base, output = _workspace(tmp_path)
    other_version = replace(CONTAINER_RUNTIME, lightgbm_version="4.6.0")
    with pytest.raises(ValueError, match="LightGBM version: '4.6.0' here, '4.7.0' in the lock"):
        load_locked_ranker_for_platform(
            paths, base, other_version, "linux", "aarch64", MODE_CALIBRATION, output
        )
    with pytest.raises(ValueError, match="num_threads: 4 here, 6 in the lock"):
        load_locked_ranker_for_platform(
            paths,
            replace(base, num_threads=4),
            CONTAINER_RUNTIME,
            "linux",
            "aarch64",
            MODE_CALIBRATION,
            output,
        )


def test_the_config_id_check_is_the_one_of_the_strict_entry(tmp_path: Path) -> None:
    paths, base, output = _workspace(tmp_path)
    _rewrite(paths.ranker_lock_file, "model", "num_leaves", 63)
    with pytest.raises(ValueError, match="The locked settings were changed"):
        load_locked_ranker_for_platform(
            paths, base, CONTAINER_RUNTIME, "linux", "aarch64", MODE_CALIBRATION, output
        )


def test_a_changed_referenced_lock_file_is_refused(tmp_path: Path) -> None:
    paths, base, output = _workspace(tmp_path)
    with paths.channels_lock_file.open("a", encoding="utf-8") as handle:
        handle.write("\n# edited\n")
    with pytest.raises(ValueError, match="sha256"):
        load_locked_ranker_for_platform(
            paths, base, CONTAINER_RUNTIME, "linux", "aarch64", MODE_CALIBRATION, output
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
        load_locked_ranker_for_platform(
            paths, base, runtime, "linux", "aarch64", MODE_CALIBRATION, output
        )


def test_calibration_mode_may_only_write_into_the_calibration_folders(tmp_path: Path) -> None:
    paths, base, output = _workspace(tmp_path)
    require_calibration_outputs(paths, output)
    # The stage-1 locations themselves: both roots are refused.
    with pytest.raises(ValueError, match="may write artifacts only under"):
        load_locked_ranker_for_platform(
            paths, base, CONTAINER_RUNTIME, "linux", "aarch64", MODE_CALIBRATION, paths
        )
    stage_one_results = replace(output, results_root=paths.results_root)
    with pytest.raises(ValueError, match="may write result tables only under"):
        require_calibration_outputs(paths, stage_one_results)
    # A folder that only starts with the same name is not inside the calibration folder.
    sibling = replace(output, artifacts_root=paths.artifacts_root / "m10_calibration_other")
    with pytest.raises(ValueError, match="may write artifacts only under"):
        require_calibration_outputs(paths, sibling)


# ---------------------------------------------------------------- mode "accepted"


def _pipeline_output(paths: ProjectPaths) -> ProjectPaths:
    """Output locations of a tolerance-accepted run: new folders, not the stage-1 ones."""
    return replace(
        paths,
        artifacts_root=paths.artifacts_root / "weekly",
        results_root=paths.results_root / "weekly",
    )


def _accepted(
    paths: ProjectPaths,
    base: BaseConfig,
    runtime: LightGBMRuntime = CONTAINER_RUNTIME,
    system: str = "linux",
    machine: str = "aarch64",
) -> Any:
    return load_locked_ranker_for_platform(
        paths, base, runtime, system, machine, MODE_ACCEPTED, _pipeline_output(paths)
    )


def test_platform_record_name() -> None:
    assert platform_record_name("linux", "aarch64") == "linux-arm64"
    assert platform_record_name("linux", "x86_64") == "linux-amd64"


def test_accepted_mode_accepts_the_platform_its_record_describes(tmp_path: Path) -> None:
    # The real record configs/platforms/linux-arm64.yaml and the runtime of the arm64 image.
    paths, base, _ = _workspace(tmp_path)
    result = _accepted(paths, base)
    assert result.mode == MODE_ACCEPTED
    assert result.record is not None and result.record.platform == "linux-arm64"
    assert result.record.acceptance.max_relative_difference == 0.01
    # The configuration is the one the strict entry returns on the laptop.
    assert result.locked == load_locked_ranker(paths, base, LAPTOP_RUNTIME)
    # Calibration mode runs before a platform is accepted and carries no record.
    calibration = load_locked_ranker_for_platform(
        paths,
        base,
        CONTAINER_RUNTIME,
        "linux",
        "aarch64",
        MODE_CALIBRATION,
        calibration_output_paths(paths, "linux-arm64", "run1", "main"),
    )
    assert calibration.record is None


def test_accepted_mode_refuses_a_platform_without_a_record(tmp_path: Path) -> None:
    # Until M13 this test used x86-64, which had no record then. It was calibrated on
    # Cloud Run and has one now (next test), so a platform that has none stands in.
    paths, base, _ = _workspace(tmp_path)
    with pytest.raises(PlatformNotAccepted, match="no platform record .*linux-riscv64.yaml"):
        _accepted(paths, base, machine="riscv64")
    # A platform whose record is missing is refused as before: without its record, x86-64
    # is not accepted.
    paths.platform_record_file("linux-amd64").unlink()
    with pytest.raises(PlatformNotAccepted, match="no platform record .*linux-amd64.yaml"):
        _accepted(paths, base, machine="x86_64")


def test_accepted_mode_accepts_x86_64_by_its_own_record(tmp_path: Path) -> None:
    # configs/platforms/linux-amd64.yaml (M13): the same LightGBM and the same libgomp1
    # package version as on arm64, in the x86-64 library folder.
    paths, base, _ = _workspace(tmp_path)
    runtime = replace(CONTAINER_RUNTIME, libomp_path="/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0")
    result = _accepted(paths, base, runtime=runtime, machine="x86_64")
    assert result.mode == MODE_ACCEPTED
    assert result.record is not None and result.record.platform == "linux-amd64"
    assert result.record.architecture == "x86_64"
    # An arm64 process is not accepted by the x86-64 record, and the other way round:
    # each record is looked up by the architecture of the process.
    assert _accepted(paths, base).record.platform == "linux-arm64"
    with pytest.raises(PlatformNotAccepted, match="OpenMP runtime: 'libgomp1 14.2.0-20' here"):
        _accepted(
            paths,
            base,
            runtime=replace(runtime, libomp_version="libgomp1 14.2.0-20"),
            machine="x86_64",
        )


def test_accepted_mode_refuses_a_process_that_is_not_the_recorded_platform(tmp_path: Path) -> None:
    paths, base, _ = _workspace(tmp_path)
    # Another version of the OpenMP runtime than the record names (a rebuilt image after
    # Debian replaced the package): a new record is needed, after item D was run again.
    other_openmp = replace(CONTAINER_RUNTIME, libomp_version="libgomp1 14.2.0-20")
    with pytest.raises(PlatformNotAccepted, match="OpenMP runtime: 'libgomp1 14.2.0-20' here"):
        _accepted(paths, base, runtime=other_openmp)
    with pytest.raises(PlatformNotAccepted, match="LightGBM version: '4.6.0' here"):
        _accepted(paths, base, runtime=replace(CONTAINER_RUNTIME, lightgbm_version="4.6.0"))
    with pytest.raises(PlatformNotAccepted, match="num_threads: 4 here, 6 in the record"):
        _accepted(paths, replace(base, num_threads=4))
    # The laptop's own runtime inside a Linux process is not the recorded platform either.
    with pytest.raises(PlatformNotAccepted, match="OpenMP runtime: '22.1.4' here"):
        _accepted(paths, base, runtime=LAPTOP_RUNTIME)


def test_accepted_mode_refuses_another_ranker_lock_than_the_record_was_made_for(
    tmp_path: Path,
) -> None:
    paths, base, _ = _workspace(tmp_path)
    with paths.ranker_lock_file.open("a", encoding="utf-8") as handle:
        handle.write("\n# edited\n")
    with pytest.raises(PlatformNotAccepted, match="sha256 of the Ranker lock file"):
        _accepted(paths, base)


def test_accepted_mode_refuses_a_record_copied_to_another_platform_name(tmp_path: Path) -> None:
    paths, base, _ = _workspace(tmp_path)
    shutil.copy(
        paths.platform_record_file("linux-arm64"), paths.platform_record_file("linux-amd64")
    )
    with pytest.raises(ValueError, match="records platform 'linux-arm64', expected 'linux-amd64'"):
        _accepted(paths, base, machine="x86_64")


def test_accepted_mode_never_writes_into_the_stage_1_folders(tmp_path: Path) -> None:
    paths, base, _ = _workspace(tmp_path)
    with pytest.raises(ValueError, match="may not write its artifacts into the stage-1 folder"):
        load_locked_ranker_for_platform(
            paths, base, CONTAINER_RUNTIME, "linux", "aarch64", MODE_ACCEPTED, paths
        )
    only_results = replace(_pipeline_output(paths), results_root=paths.results_root)
    with pytest.raises(ValueError, match="may not write result tables into the stage-1 folder"):
        load_locked_ranker_for_platform(
            paths, base, CONTAINER_RUNTIME, "linux", "aarch64", MODE_ACCEPTED, only_results
        )


def test_accepted_mode_refuses_macos_before_it_looks_for_a_record(tmp_path: Path) -> None:
    paths, base, _ = _workspace(tmp_path)
    with pytest.raises(RuntimeError, match="runs on Linux only"):
        _accepted(paths, base, runtime=LAPTOP_RUNTIME, system="darwin", machine="arm64")


# ---------------------------------------------------------------- M13: a calibration tick


def _tick_view(paths: ProjectPaths, state: Path, results: Path | None = None) -> PipelinePaths:
    """The paths of a step of a tick whose state is `state` (tables in it unless given)."""
    view = pipeline_paths(paths, StateLayout(state), V1_ANCHOR, (("V1", V1_ANCHOR),), None)
    return view if results is None else replace(view, results_root=results)


def test_calibration_mode_accepts_the_calibration_state_of_a_tick(tmp_path: Path) -> None:
    paths, base, _ = _workspace(tmp_path)
    tick = tick_root_paths(paths, {PIPELINE_STATE_ROOT_VARIABLE: "/mnt/pipeline"}, True)
    view = _tick_view(tick, tick.pipeline_state_dir)
    assert view.layout.root == Path("/mnt/pipeline-calibration")
    assert view.results_root == Path("/mnt/pipeline-calibration/results")
    # Mode "calibration" is how a platform is measured before it has a record (x86-64
    # in M13); it does not read a record and returns none.
    amd64_runtime = replace(
        CONTAINER_RUNTIME, libomp_path="/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0"
    )
    result = load_locked_ranker_for_platform(
        tick, base, amd64_runtime, "linux", "x86_64", MODE_CALIBRATION, view
    )
    assert result.mode == MODE_CALIBRATION and result.record is None
    assert result.locked == load_locked_ranker(paths, base, LAPTOP_RUNTIME)
    assert result.runtime == amd64_runtime


def test_calibration_mode_refuses_a_tick_that_writes_anywhere_else(tmp_path: Path) -> None:
    paths, base, _ = _workspace(tmp_path)
    tick = tick_root_paths(paths, {}, calibration=True)
    state = tick.pipeline_state_dir
    require_calibration_outputs(tick, _tick_view(tick, state))
    # The pipeline's own state: a tick that is not a calibration tick.
    with pytest.raises(ValueError, match="needs a calibration tick"):
        require_calibration_outputs(paths, _tick_view(paths, paths.pipeline_state_dir))
    # A calibration tick whose step would write into the pipeline's state ...
    with pytest.raises(ValueError, match="may write state only under"):
        require_calibration_outputs(tick, _tick_view(tick, paths.pipeline_state_dir))
    # ... or its result tables into the M12 tables.
    m12_tables = _tick_view(tick, state, results=paths.replay_results_dir)
    with pytest.raises(ValueError, match="may write result tables only under"):
        require_calibration_outputs(tick, m12_tables)
    # The entry itself refuses it too, before anything is read.
    with pytest.raises(ValueError, match="needs a calibration tick"):
        load_locked_ranker_for_platform(
            paths,
            base,
            CONTAINER_RUNTIME,
            "linux",
            "aarch64",
            MODE_CALIBRATION,
            _tick_view(paths, paths.pipeline_state_dir),
        )
