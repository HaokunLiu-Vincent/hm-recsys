"""What every calibration process starts and ends with (docs/m10_calibration_plan.md §2).

A calibration process does ONE item, as in stage 1 (one step per process, so
ru_maxrss is that step's true memory peak):

    context = start_calibration(run, label)      # paths, configs, the code state
    ... the item computes, then compares ...
    finish_item(context, outcome)                # comparison rows, the run row

Where things are read and written:

    context.paths            the stage-1 locations: inputs and the stage-1 outputs
                             that are compared with. Read only. In the container
                             they are mounted read-only.
    context.main_paths       artifacts/m10_calibration/{platform}/{run}/main/...
    context.recorded_model_paths
                             .../{run}/recorded_model/... (item C1)
    result tables            results/m10_calibration/ (hmrec.calibration.records)

This module imports neither torch nor lightgbm: both sides start here.
"""

import logging
import platform
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from hmrec.calibration.records import (
    Measurement,
    ProcessRecord,
    RepeatCheck,
    RunIdentity,
    append_comparisons,
    append_repeat_checks,
    append_run,
)
from hmrec.config import (
    BaseConfig,
    CalibrationConfig,
    load_base_config,
    load_calibration_config,
)
from hmrec.git_state import git_state
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.memory import (
    MemoryTracker,
    ProcessPeakMemory,
    bytes_to_gb,
    log_process_peak_memory,
)
from hmrec.paths import ProjectPaths, calibration_output_paths, get_paths
from hmrec.platform_fields import ARCHITECTURE_NAMES, read_platform_fields
from hmrec.platform_report import read_machine_info
from hmrec.run_records import RunStamp, check_memory_limit

LOGGER = get_logger("calibration")

MAIN_TREE = "main"
RECORDED_MODEL_TREE = "recorded_model"


@dataclass(frozen=True)
class CalibrationContext:
    """Everything a calibration process works with.

    Attributes:
        paths: The stage-1 locations (inputs and references; read only).
        main_paths: Where the run's own artifacts go (the "main" tree).
        recorded_model_paths: Where item C1 writes (the "recorded_model" tree).
        base: configs/base.yaml (threads, chunk sizes, memory limit).
        config: configs/m10_calibration.yaml.
        run: Name of this container run, e.g. "run1".
        platform_name: e.g. "linux-arm64"; part of every output path.
        stamp: The commit of the code and whether it is dirty.
        started: time.perf_counter() when the process started its work.
    """

    paths: ProjectPaths
    main_paths: ProjectPaths
    recorded_model_paths: ProjectPaths
    base: BaseConfig
    config: CalibrationConfig
    run: str
    platform_name: str
    stamp: RunStamp
    started: float


@dataclass(frozen=True)
class ItemOutcome:
    """What one item produced; finish_item writes it to the tables.

    Attributes:
        item: The calibration item: "A", "B", "C1", "C2", "union" or "D".
        scope: What inside the item, e.g. "r5_wals" or "fine_tune".
        item_seconds: Wall time of the item's own computation.
        sampled_peak_rss_gb: SAMPLED peak RSS during the computation.
        item_peak: ru_maxrss read right after the computation, before the
            comparison loaded any stage-1 file.
        measurements: The comparison rows.
        detail: Short free text for the run row.
    """

    item: str
    scope: str
    item_seconds: float
    sampled_peak_rss_gb: float
    item_peak: ProcessPeakMemory
    measurements: Sequence[Measurement]
    detail: str


def platform_name(system: str, machine: str) -> str:
    """The folder name of a platform, e.g. ("linux", "aarch64") -> "linux-arm64"."""
    return f"{system}-{ARCHITECTURE_NAMES.get(machine, machine)}"


def require_known_run(run: str, config: CalibrationConfig) -> None:
    """Fail unless `run` is one of the run names of configs/m10_calibration.yaml."""
    if run not in config.runs:
        raise ValueError(f"--run must be one of {list(config.runs)}, got {run!r}")


def require_committed_code(stamp: RunStamp) -> None:
    """Fail if the code differs from its commit (AGENTS.md §3.4, docs/mlops.md §4).

    A calibration row must belong to one definite commit. Inside an image the
    code state was already compared with the tree hash stored at build time.

    Raises:
        RuntimeError: If the working tree has uncommitted changes.
    """
    if stamp.git_dirty:
        raise RuntimeError(
            f"the working tree differs from commit {stamp.git_commit[:12]}: a calibration "
            "runs only from committed code (an image built with `make image-build`)"
        )


def start_calibration(run: str, label: str) -> CalibrationContext:
    """Read the configs and the code state, set up logging, and name the output trees.

    Args:
        run: Name of this container run ("run1" or "run2").
        label: Name of the process for its log file, e.g. "recall_r5_wals".

    Raises:
        ValueError: If `run` is unknown.
        RuntimeError: If the code is not a committed tree, or (inside an image)
            is not the code the image was built with.
    """
    started = time.perf_counter()
    paths = get_paths()
    config = load_calibration_config(paths.m10_calibration_config_file)
    require_known_run(run, config)
    name = platform_name(sys.platform, platform.machine())
    setup_logging(logging.INFO, log_file=paths.calibration_log_file(name, run, label))
    base = load_base_config(paths.base_config_file)
    commit, dirty = git_state(paths.project_root, paths.results_root)
    stamp = RunStamp(commit, dirty)
    require_committed_code(stamp)
    LOGGER.info("calibration %s on %s, run %s, commit %s", label, name, run, commit)
    return CalibrationContext(
        paths=paths,
        main_paths=calibration_output_paths(paths, name, run, MAIN_TREE),
        recorded_model_paths=calibration_output_paths(paths, name, run, RECORDED_MODEL_TREE),
        base=base,
        config=config,
        run=run,
        platform_name=name,
        stamp=stamp,
        started=started,
    )


def timed[Value](
    context: CalibrationContext, label: str, compute: Callable[[], Value]
) -> tuple[Value, float, float]:
    """Run `compute` inside a MemoryTracker.

    Args:
        context: The calibration context (memory settings of configs/base.yaml).
        label: Name of the block in the log.
        compute: The computation to time.

    Returns:
        (its result, wall seconds, SAMPLED peak RSS in GB).
    """
    memory = context.base.memory
    with MemoryTracker(
        label,
        LOGGER,
        limit_gb=memory.limit_gb,
        sample_interval_seconds=memory.sample_interval_seconds,
    ) as tracker:
        started = time.perf_counter()
        value = compute()
        seconds = time.perf_counter() - started
    return value, seconds, bytes_to_gb(tracker.peak_rss_bytes)


def run_identity(context: CalibrationContext) -> RunIdentity:
    """The identity columns of this process, with the libraries loaded by NOW."""
    return RunIdentity(
        run=context.run,
        git_commit=context.stamp.git_commit,
        git_dirty=context.stamp.git_dirty,
        platform=read_platform_fields(context.paths.project_root),
    )


def finish_item(context: CalibrationContext, outcome: ItemOutcome) -> None:
    """Write the comparison rows and the run row of an item; then check the memory limit.

    The run row is written last, so a process without a run row did not finish.

    Raises:
        MemoryError: If the true memory peak of the process is above the limit
            of configs/base.yaml (after the rows are written).
    """
    paths = context.paths
    identity = run_identity(context)
    append_comparisons(paths.calibration_comparisons_file, outcome.measurements, identity)
    label = f"calibration {outcome.item} {outcome.scope}"
    limit_gb = context.base.memory.limit_gb
    peak = log_process_peak_memory(LOGGER, label, limit_gb)
    machine = read_machine_info()
    memory_limit = machine.memory_limit_bytes
    record = ProcessRecord(
        item=outcome.item,
        scope=outcome.scope,
        item_seconds=outcome.item_seconds,
        total_seconds=time.perf_counter() - context.started,
        sampled_peak_rss_gb=outcome.sampled_peak_rss_gb,
        item_max_rss_gb=outcome.item_peak.process_max_rss_gb,
        process_max_rss_gb=peak.process_max_rss_gb,
        num_threads=context.base.num_threads,
        usable_cpus=machine.cpu_quota if machine.cpu_quota is not None else machine.usable_cpus,
        memory_limit_gb=0.0 if memory_limit is None else bytes_to_gb(memory_limit),
        detail=outcome.detail,
    )
    append_run(paths.calibration_runs_file, record, identity)
    LOGGER.info(
        "%s: %d comparison rows -> %s; run row -> %s",
        label,
        len(outcome.measurements),
        paths.calibration_comparisons_file,
        paths.calibration_runs_file,
    )
    check_memory_limit(peak, limit_gb, label)


def finish_repeat(
    context: CalibrationContext, checks: Sequence[RepeatCheck], first_run: str, second_run: str
) -> None:
    """Write the rows of a repeatability check (two container runs compared bit by bit)."""
    path = context.paths.calibration_repeatability_file
    append_repeat_checks(path, checks, first_run, second_run, run_identity(context))
    for check in checks:
        LOGGER.info(
            "repeatability %s | %s | bit-identical: %s | %s",
            check.item,
            check.artifact,
            check.bit_identical,
            check.detail,
        )
    LOGGER.info("%d repeatability rows -> %s", len(checks), path)
