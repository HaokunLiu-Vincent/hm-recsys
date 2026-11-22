"""What every step process of a tick starts and ends with (docs/mlops.md §6.1).

A step process does ONE step, so that its process max RSS is the true memory
peak of that step (hmrec.memory) and torch and LightGBM never share a process:

    context, step = start_step(anchor, mode, run_id, step_name)
    ... the step computes and checks; it writes into its staging folder ...
    finish_step(context, step, started, detail, metrics, files)

finish_step writes step.json into the step's folder as the LAST action: the
step's record for the run manifest (seconds, true peak, the platform fields of
this process), the metric values the step measured and the output files it wants
recorded. The driver renames the folder afterwards. A step that raises writes
nothing; its staging folder is set aside by the driver.

A tick works on the full data, so start_step REFUSES code that differs from its
commit (AGENTS.md §3.4, docs/mlops.md invariant 7).

This module imports neither torch nor lightgbm: both sides start here.
"""

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from hmrec.config import (
    BaseConfig,
    PipelineConfig,
    PreprocessConfig,
    load_base_config,
    load_pipeline_config,
    load_preprocess_config,
    load_versioning_config,
)
from hmrec.data.folds import FoldManifest, load_fold_manifest
from hmrec.git_state import git_state
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.memory import log_process_peak_memory
from hmrec.paths import PipelinePaths, ProjectPaths, StateLayout, get_paths, pipeline_paths
from hmrec.platform_fields import PlatformFields
from hmrec.run_manifest import MetricRecord, step_record_of_this_process
from hmrec.run_records import RunStamp, check_memory_limit
from hmrec.threads import check_polars_thread_pool
from hmrec.weekly.calendar import Week, previous_week, week_at, week_named, weeks_from
from hmrec.weekly.steps import STEP_RECORD_FILE, Step, check_mode, staging_dir, step_by_name

LOGGER = get_logger("weekly.context")


@dataclass(frozen=True)
class TickContext:
    """Everything a process of a tick works with.

    Attributes:
        paths: The project paths: configs, data/processed/ (the SOURCE of the
            transactions) and the bootstrap state. Only C1 and the `eval`-mode
            query list read the source transactions.
        layout: The layout of the pipeline's state.
        view: The paths the stage-1 functions get (hmrec.paths.PipelinePaths):
            the tick's ingested transactions, the week folders of the state.
        base: configs/base.yaml (threads, chunk sizes, memory limit).
        preprocess: configs/preprocess.yaml (day_zero).
        pipeline: configs/m12_pipeline.yaml.
        manifest: configs/fold_manifest.yaml (read only).
        tick: The week the tick opens (its anchor is the clock).
        closing: The week right before it, which the tick closes.
        first_week: The week the first tick opens; earlier weeks are closed in
            the bootstrap state.
        mode: "eval" or "serve".
        run_id: The run of the tick this process belongs to.
        stamp: The commit of the code and whether it is dirty.
        ranker_experiment: The Ranker experiment of the samples ("full").
    """

    paths: ProjectPaths
    layout: StateLayout
    view: PipelinePaths
    base: BaseConfig
    preprocess: PreprocessConfig
    pipeline: PipelineConfig
    manifest: FoldManifest
    tick: Week
    closing: Week
    first_week: Week
    mode: str
    run_id: str
    stamp: RunStamp
    ranker_experiment: str

    @property
    def closes_a_week(self) -> bool:
        """False for the first tick: the week before it is closed in the bootstrap state."""
        return self.tick.ordinal > self.first_week.ordinal


@dataclass(frozen=True)
class OutputFile:
    """An output file a step wants recorded in the run manifest.

    Attributes:
        role: What the file is, e.g. "candidates/r4_itemcf" (unique within a tick).
        name: The file's path inside the step's folder, with "/".
    """

    role: str
    name: str


def require_committed_code(stamp: RunStamp) -> None:
    """Fail if the code differs from its commit.

    Raises:
        RuntimeError: If the working tree has uncommitted changes. Inside an
            image the code tree was already compared with the build-time hash.
    """
    if stamp.git_dirty:
        raise RuntimeError(
            f"the working tree differs from commit {stamp.git_commit[:12]}: a tick runs only "
            "from committed code (an image built with `make image-build`)"
        )


def load_tick_context(
    paths: ProjectPaths,
    anchor: date,
    mode: str,
    run_id: str,
    stamp: RunStamp,
    step: Step | None = None,
) -> TickContext:
    """Read the configs of a tick and build its paths (no data is read).

    Args:
        paths: The project paths.
        anchor: The tick's anchor: the first day of the week it opens.
        mode: "eval" or "serve".
        run_id: The run of the tick.
        stamp: The code state.
        step: The step this process runs; its folder becomes the staging folder
            of the returned view. None for the driver, which writes no step output.

    Raises:
        ValueError: If the mode is unknown or the anchor is not on the weekly grid.
    """
    check_mode(mode)
    manifest = load_fold_manifest(paths.fold_manifest_file)
    versioning = load_versioning_config(paths.versioning_config_file)
    tick = week_at(anchor, manifest)
    first_week = week_named(versioning.bootstrap.first_tick_fold, manifest)
    layout = StateLayout(paths.pipeline_state_dir)
    opened = weeks_from(first_week, tick, manifest) if tick.ordinal >= first_week.ordinal else ()
    week_anchors = tuple((week.name, week.anchor) for week in opened)
    staging = None if step is None or step.step_dir == "" else (step.week.anchor, step.step_dir)
    return TickContext(
        paths=paths,
        layout=layout,
        view=pipeline_paths(paths, layout, tick.anchor, week_anchors, staging),
        base=load_base_config(paths.base_config_file),
        preprocess=load_preprocess_config(paths.preprocess_config_file),
        pipeline=load_pipeline_config(paths.pipeline_config_file),
        manifest=manifest,
        tick=tick,
        closing=previous_week(tick, manifest),
        first_week=first_week,
        mode=mode,
        run_id=run_id,
        stamp=stamp,
        ranker_experiment=versioning.bootstrap.ranker_experiment,
    )


def start_step(anchor: date, mode: str, run_id: str, step_name: str) -> tuple[TickContext, Step]:
    """Set up logging, refuse uncommitted code, and build the context of one step.

    Args:
        anchor: The tick's anchor.
        mode: "eval" or "serve".
        run_id: The run of the tick (names the log file).
        step_name: The step this process runs, e.g. "O1_recall_r4_itemcf".

    Raises:
        RuntimeError: If the code is not a committed tree.
        ValueError: If the tick has no such step.
    """
    paths = get_paths()
    layout = StateLayout(paths.pipeline_state_dir)
    setup_logging(logging.INFO, log_file=layout.step_log_file(run_id, step_name))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    stamp = RunStamp(commit, dirty)
    require_committed_code(stamp)
    plain = load_tick_context(paths, anchor, mode, run_id, stamp)
    step = step_by_name(step_name, plain.tick, plain.closing, mode)
    context = load_tick_context(paths, anchor, mode, run_id, stamp, step)
    threads = check_polars_thread_pool(context.base.num_threads)
    LOGGER.info(
        "tick %s (%s, %s) run %s | step %s | commit %s | polars threads %d",
        context.tick.name,
        anchor,
        mode,
        run_id,
        step_name,
        commit,
        threads,
    )
    staging_dir(layout, step).mkdir(parents=True, exist_ok=True)
    return context, step


def step_record_json(
    context: TickContext,
    step: Step,
    seconds: float,
    detail: str,
    metrics: Sequence[MetricRecord],
    files: Sequence[OutputFile],
) -> dict[str, Any]:
    """The content of step.json for a step that ran in THIS process."""
    record = step_record_of_this_process(step.name, seconds, detail, context.paths.project_root)
    if record.platform is None:
        raise ValueError("a step that ran has platform fields")
    return {
        "name": record.name,
        "seconds": record.seconds,
        "process_max_rss_gb": record.process_max_rss_gb,
        "platform": record.platform.columns(),
        "detail": record.detail,
        "metrics": [
            {"name": metric.name, "week": metric.week.isoformat(), "value": metric.value}
            for metric in metrics
        ],
        "files": [{"role": file.role, "name": file.name} for file in files],
    }


@dataclass(frozen=True)
class StepOutput:
    """What a finished step left in step.json (read by the driver).

    Attributes:
        name: The step's name.
        seconds: Wall time of the step's process.
        process_max_rss_gb: True peak RSS of the step's process.
        platform: The platform fields of that process.
        detail: Short free text.
        metrics: The metric values the step measured.
        files: The output files to record in the run manifest.
    """

    name: str
    seconds: float
    process_max_rss_gb: float
    platform: PlatformFields
    detail: str
    metrics: tuple[MetricRecord, ...]
    files: tuple[OutputFile, ...]


def read_step_output(folder: Path) -> StepOutput:
    """Read the step.json of a step's folder.

    Raises:
        FileNotFoundError: If the step ended without writing its record.
    """
    record_file = folder / STEP_RECORD_FILE
    if not record_file.is_file():
        raise FileNotFoundError(f"the step left no {STEP_RECORD_FILE} in {folder}")
    raw = json.loads(record_file.read_text(encoding="utf-8"))
    return StepOutput(
        name=raw["name"],
        seconds=float(raw["seconds"]),
        process_max_rss_gb=float(raw["process_max_rss_gb"]),
        platform=PlatformFields(**raw["platform"]),
        detail=raw["detail"],
        metrics=tuple(
            MetricRecord(item["name"], date.fromisoformat(item["week"]), float(item["value"]))
            for item in raw["metrics"]
        ),
        files=tuple(OutputFile(item["role"], item["name"]) for item in raw["files"]),
    )


def write_step_record(folder: Path, content: dict[str, Any]) -> None:
    """Write step.json into a step's folder (it is written once)."""
    target = folder / STEP_RECORD_FILE
    if target.exists():
        raise FileExistsError(f"{target} already exists; a step record is never replaced")
    target.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finish_step(
    context: TickContext,
    step: Step,
    started: float,
    detail: str,
    metrics: Sequence[MetricRecord] = (),
    files: Sequence[OutputFile] = (),
) -> None:
    """Write the step's record into its folder, then check the memory limit.

    Call it as the LAST action of a step process: the true peak (ru_maxrss) is
    read here.

    Args:
        context: The process's context.
        step: The step that ran.
        started: time.perf_counter() at the start of the process.
        detail: Short free text for the run manifest.
        metrics: Metric values the step measured, each with the week it describes.
        files: Output files to record in the run manifest (inside the step's folder).

    Raises:
        MemoryError: If the true peak RSS of the process is above memory.limit_gb.
            The step then counts as failed: the limit is a stop condition.
    """
    folder = staging_dir(context.layout, step)
    missing = [file.name for file in files if not (folder / file.name).is_file()]
    if missing:
        raise FileNotFoundError(f"step {step.name} did not write its outputs {missing}")
    seconds = time.perf_counter() - started
    content = step_record_json(context, step, seconds, detail, metrics, files)
    limit_gb = context.base.memory.limit_gb
    peak = log_process_peak_memory(LOGGER, f"tick {context.tick.name} {step.name}", limit_gb)
    # Checked BEFORE the record is written: a step above the limit leaves no record,
    # so the driver sees a failed step and sets its folder aside.
    check_memory_limit(peak, limit_gb, f"tick {context.tick.name} {step.name}")
    write_step_record(folder, content)
    LOGGER.info("step %s finished in %.1f s: %s", step.name, seconds, detail)
