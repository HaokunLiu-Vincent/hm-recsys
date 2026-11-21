"""A second entry to the locked Ranker configuration, for tolerance-accepted platforms.

Why it exists (docs/mlops.md §8.3; decision of 2026-11-22 in docs/decisions.md):
the strict entry, hmrec.ranker.locked.load_locked_ranker, refuses every process
whose OpenMP runtime is not the Homebrew libomp the lock records. That is
right: the lock says which runtime the V phase ran on, and a result of the
strict entry may be compared bit by bit with stage 1. A Linux container can
never pass it. This module is how a Linux process gets the SAME locked
configuration without changing what the lock means. The strict entry and every
lock file are untouched.

What is checked, compared with the strict entry:

    the lock's settings give its config_id          the same check, unchanged
    sha256 of the features / channels / twotower    the same check, unchanged
    lock files
    LightGBM version, num_threads                   must equal the lock, as before
    OpenMP runtime                                  NOT compared with the lock.
                                                    mode "calibration": recorded as
                                                    found; mode "accepted": must
                                                    equal a platform record

Rules of this entry:

  - It refuses macOS. On the laptop the strict entry is the only one.
  - Mode "accepted" needs a platform record (hmrec.platform_records,
    configs/platforms/{platform}.yaml): the file that says which runtime of
    which platform the user accepted, and by which thresholds. The process must
    be the platform the record describes: the same OS and architecture, the
    same LightGBM version, the same OpenMP runtime, the same thread count and
    the same Ranker lock file (by sha256). A platform without a record is
    refused (PlatformNotAccepted). Its outputs may go anywhere EXCEPT the
    stage-1 locations.
  - Mode "calibration" is for measuring a platform before it is accepted
    (docs/m10_calibration_plan.md). It records the OpenMP runtime it finds and
    may write ONLY into the calibration folders; the caller passes where it
    will write, and anything else is refused here.
  - Everything produced through this entry carries the platform fields of
    hmrec.platform_fields (OS, architecture, OpenMP library and version, image
    commit) and goes to new tables. It is never compared bit by bit with a
    laptop result and never written to a stage-1 table.

Only the calibration code may import this module (hmrec.calibration and
scripts/m10_calibrate.py); tests/unit/test_calibration_code_rules.py checks it.
"""

from dataclasses import dataclass
from pathlib import Path

from hmrec.config import BaseConfig, load_ranker_config
from hmrec.locks import RankerLock, file_sha256, load_ranker_lock
from hmrec.paths import ProjectPaths
from hmrec.platform_fields import ARCHITECTURE_NAMES
from hmrec.platform_records import PlatformRecord, load_platform_record
from hmrec.ranker.locked import (
    LockedRanker,
    check_lock_config_id,
    check_referenced_locks,
    locked_ranker_config,
    locked_tuned_params,
)
from hmrec.ranker.runtime import LINUX, NOT_LOADED, UNKNOWN, LightGBMRuntime

MODE_CALIBRATION = "calibration"
MODE_ACCEPTED = "accepted"
MODES = (MODE_CALIBRATION, MODE_ACCEPTED)


class PlatformNotAccepted(RuntimeError):
    """The platform of this process has no accepted platform record."""


@dataclass(frozen=True)
class PlatformLockedRanker:
    """The locked Ranker configuration on a tolerance-accepted platform.

    Attributes:
        locked: The configuration; the same values the strict entry returns.
        mode: "calibration" or "accepted".
        runtime: The LightGBM version and the OpenMP runtime of this process,
            as recorded (not compared with the lock).
        record: The platform record the process was accepted by; None in
            calibration mode, which runs before a platform is accepted.
    """

    locked: LockedRanker
    mode: str
    runtime: LightGBMRuntime
    record: PlatformRecord | None


def require_tolerance_platform(system: str) -> None:
    """Fail unless this process runs on Linux.

    Args:
        system: sys.platform of the process.

    Raises:
        RuntimeError: On macOS (and anything that is not Linux). The laptop is
            the reference of every bit-identical check and uses only the strict
            entry, which compares the OpenMP runtime with the lock.
    """
    if system != LINUX:
        raise RuntimeError(
            f"the tolerance-accepted entry runs on Linux only, this process is {system!r}. "
            "On the laptop use hmrec.ranker.locked.load_locked_ranker, the strict entry."
        )


def check_version_and_threads_match_lock(
    lock: RankerLock, runtime: LightGBMRuntime, num_threads: int
) -> None:
    """Fail unless this process has the locked LightGBM version and thread count.

    These two are the same on every platform (uv.lock pins LightGBM;
    configs/base.yaml gives num_threads), so a tolerance never covers them.

    Raises:
        ValueError: If one of them differs; the message names each one.
    """
    locked = lock.runtime
    pairs = (
        ("LightGBM version", runtime.lightgbm_version, locked.lightgbm_version),
        ("num_threads", num_threads, locked.num_threads),
    )
    differences = [
        f"{name}: {actual!r} here, {recorded!r} in the lock"
        for name, actual, recorded in pairs
        if actual != recorded
    ]
    if differences:
        raise ValueError(
            "this process does not match configs/locks/ranker.lock.yaml (runtime): "
            + "; ".join(differences)
            + ". The tolerance of docs/mlops.md 8.3 covers the OpenMP runtime only."
        )


def require_known_openmp_runtime(runtime: LightGBMRuntime) -> None:
    """Fail unless the OpenMP runtime of this process can be named in a record.

    Raises:
        ValueError: If no runtime is loaded or no package owns the loaded file.
            A record that says "unknown" would not tell which runtime a result
            was computed on.
    """
    if runtime.libomp_path == NOT_LOADED or runtime.libomp_version == UNKNOWN:
        raise ValueError(
            f"the OpenMP runtime of this process cannot be identified (path "
            f"{runtime.libomp_path!r}, version {runtime.libomp_version!r}); a "
            "tolerance-accepted result must record the runtime it ran on"
        )


def require_calibration_outputs(paths: ProjectPaths, output_paths: ProjectPaths) -> None:
    """Fail unless `output_paths` writes only into the calibration folders.

    Args:
        paths: The project paths (the stage-1 locations).
        output_paths: Where the caller will write artifacts and result tables.

    Raises:
        ValueError: If the artifacts root is not inside
            artifacts/m10_calibration/ or the results root is not
            results/m10_calibration/ of `paths`.
    """
    allowed_artifacts = paths.calibration_artifacts_dir
    allowed_results = paths.calibration_results_dir
    if not _is_inside(output_paths.artifacts_root, allowed_artifacts):
        raise ValueError(
            f"calibration mode may write artifacts only under {allowed_artifacts}, "
            f"got {output_paths.artifacts_root}"
        )
    if output_paths.results_root != allowed_results:
        raise ValueError(
            f"calibration mode may write result tables only under {allowed_results}, "
            f"got {output_paths.results_root}"
        )


def require_new_output_locations(paths: ProjectPaths, output_paths: ProjectPaths) -> None:
    """Fail if `output_paths` is a stage-1 location (accepted mode).

    A tolerance-accepted result is never written among the stage-1 artifacts or
    into a stage-1 result table.

    Raises:
        ValueError: If the artifacts root or the results root is the one of `paths`.
    """
    if output_paths.artifacts_root == paths.artifacts_root:
        raise ValueError(
            f"a tolerance-accepted run may not write its artifacts into the stage-1 folder "
            f"{paths.artifacts_root}"
        )
    if output_paths.results_root == paths.results_root:
        raise ValueError(
            f"a tolerance-accepted run may not write result tables into the stage-1 folder "
            f"{paths.results_root}"
        )


def platform_record_name(system: str, machine: str) -> str:
    """The name of a platform's record file, e.g. ("linux", "aarch64") -> "linux-arm64"."""
    return f"{system}-{ARCHITECTURE_NAMES.get(machine, machine)}"


def accepted_platform_record(
    paths: ProjectPaths, system: str, machine: str, runtime: LightGBMRuntime, num_threads: int
) -> PlatformRecord:
    """The platform record this process is accepted by, after comparing the two.

    Args:
        paths: Project paths.
        system: sys.platform of the process.
        machine: platform.machine() of the process.
        runtime: The LightGBM version and OpenMP runtime of the process.
        num_threads: num_threads of configs/base.yaml.

    Raises:
        PlatformNotAccepted: If the platform has no record, or the process is
            not the platform its record describes (another OS, architecture,
            LightGBM version, OpenMP runtime or thread count), or the Ranker
            lock file is not the one the record was made for.
    """
    record_file = paths.platform_record_file(platform_record_name(system, machine))
    if not record_file.is_file():
        raise PlatformNotAccepted(
            f"no platform record {record_file}: the platform {system} / {machine} has not "
            "been accepted (docs/mlops.md 8.3). A platform is accepted by the user after its "
            "calibration; only mode 'calibration' can run before that."
        )
    record = load_platform_record(record_file)
    lock_file = paths.project_root / record.ranker_lock_path
    pairs = (
        ("os", system, record.os),
        ("architecture", machine, record.architecture),
        ("LightGBM version", runtime.lightgbm_version, record.lightgbm_version),
        ("OpenMP runtime", runtime.libomp_version, record.openmp_version),
        ("num_threads", num_threads, record.num_threads),
        ("Ranker lock file", lock_file, paths.ranker_lock_file),
        (
            "sha256 of the Ranker lock file",
            file_sha256(paths.ranker_lock_file),
            record.ranker_lock_sha256,
        ),
    )
    differences = [
        f"{name}: {actual!r} here, {recorded!r} in the record"
        for name, actual, recorded in pairs
        if actual != recorded
    ]
    if differences:
        raise PlatformNotAccepted(
            f"this process is not the platform {record_file.name} describes: "
            + "; ".join(differences)
            + ". A changed runtime needs a new platform record (they are append-only)."
        )
    return record


def _is_inside(path: Path, folder: Path) -> bool:
    """True if `path` is `folder` itself or lies below it (no file system access)."""
    return path == folder or folder in path.parents


def load_locked_ranker_for_platform(
    paths: ProjectPaths,
    base: BaseConfig,
    runtime: LightGBMRuntime,
    system: str,
    machine: str,
    mode: str,
    output_paths: ProjectPaths,
) -> PlatformLockedRanker:
    """Read the Ranker lock on a tolerance-accepted platform (see the module docstring).

    Args:
        paths: Project paths.
        base: configs/base.yaml (its num_threads must equal the lock's).
        runtime: hmrec.ranker.runtime.read_lightgbm_runtime() of this process.
        system: sys.platform of this process.
        machine: platform.machine() of this process, e.g. "aarch64".
        mode: "calibration" or "accepted".
        output_paths: Where everything produced with this configuration is
            written (hmrec.paths.calibration_output_paths in calibration mode).

    Returns:
        The locked configuration, the mode and the recorded runtime.

    Raises:
        RuntimeError: On macOS.
        PlatformNotAccepted: In mode "accepted": the platform has no record, or
            the process is not the platform its record describes.
        ValueError: On an unknown mode; if the lock's settings do not give its
            config_id; if a referenced lock file has another sha256; if the
            LightGBM version or num_threads differ from the lock; if the OpenMP
            runtime cannot be identified; if calibration mode would write
            outside the calibration folders, or accepted mode into a stage-1 folder.
    """
    require_tolerance_platform(system)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    record = None
    if mode == MODE_ACCEPTED:
        require_new_output_locations(paths, output_paths)
        record = accepted_platform_record(paths, system, machine, runtime, base.num_threads)
    else:
        require_calibration_outputs(paths, output_paths)
    lock = load_ranker_lock(paths.ranker_lock_file)
    config = locked_ranker_config(lock, load_ranker_config(paths.ranker_config_file))
    # The first two checks are the strict entry's own functions, unchanged.
    check_lock_config_id(lock, config)
    check_referenced_locks(lock, paths)
    check_version_and_threads_match_lock(lock, runtime, base.num_threads)
    require_known_openmp_runtime(runtime)
    locked = LockedRanker(
        lock=lock,
        config=config,
        tuned=locked_tuned_params(lock),
        num_trees=lock.model.num_trees,
        histogram_mode=lock.model.histogram_mode,
        num_threads=lock.runtime.num_threads,
        config_id=lock.selection.config_id,
    )
    return PlatformLockedRanker(locked=locked, mode=mode, runtime=runtime, record=record)
