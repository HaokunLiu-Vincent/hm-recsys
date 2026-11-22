"""The driver of one tick: every step in its own process, in order (docs/mlops.md §6).

    run_tick(paths, anchor, mode, runner, stamp)

closes the week before the anchor and opens the week of the anchor. The rules of
§6.4 are implemented here and in hmrec.weekly.storage:

    refusal     a tick starts only if the tick before it has finished: the
                previous week has published predictions (hmrec.weekly.state). A
                week cannot be skipped.
    staging     a step writes into weeks/{anchor}/.staging/{step}/; the folder is
                renamed only after the step's process ended without an error
                (its checks passed) and the driver's own gate passed. An output
                that exists is never overwritten.
    resume      running the same tick again skips every step whose output
                exists. The staging folder of a failed attempt is moved to
                .failed/, never deleted and never reused.
    stop        the first failed step or check stops the tick. The reason goes
                into the run manifest ("failed") and into the monitoring table
                as an alert. `current` is not touched.
    publish     state/current.json is replaced only after O5's folder is in
                place, which needs every earlier step.

Every run of a tick writes a run manifest (hmrec.run_manifest), also a failed one;
a resumed tick gets a new run id.

Labels: before a step reads the answers of a week (the `eval` query list, C2, C3),
the driver appends a row to results/m12_replay/label_access.csv. The replay reads
T labels for monitoring and verification only (docs/mlops.md invariant 3).

`serve` weeks cannot be closed in this milestone: closing needs the features of
the customers who bought, and a `serve` tick removes a shard's features after
scoring (docs/mlops.md §8.4). The replay ends with its `serve` tick.

This module imports neither torch nor lightgbm. Steps run through a StepRunner:
subprocesses in production (subprocess_runner), a function in tests.
"""

import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from hmrec.config import MonitoringConfig, load_monitoring_config
from hmrec.csv_records import append_row
from hmrec.locks import ALL_CHANNELS
from hmrec.logging_utils import get_logger
from hmrec.memory import read_process_peak_memory
from hmrec.paths import FEATURES_DIR, ProjectPaths, shard_step_dir
from hmrec.platform_fields import read_platform_fields
from hmrec.run_manifest import (
    EVAL_MODE,
    FAILED,
    SERVE_MODE,
    STEP_DONE,
    STEP_SKIPPED,
    SUCCEEDED,
    FileRecord,
    MetricRecord,
    RunManifest,
    StepRecord,
    file_record,
    read_run_code,
    write_run_manifest,
)
from hmrec.run_records import RunStamp
from hmrec.weekly.context import (
    TickContext,
    load_tick_context,
    read_step_output,
    require_committed_code,
)
from hmrec.weekly.data_steps import read_ingest_record
from hmrec.weekly.history import (
    OUTPUT_SIGNALS,
    append_findings,
    earlier_labelled_flags,
    feature_history,
    output_history,
    own_feature_summaries,
    own_output_signals,
    recall_history,
    week_mode,
    write_findings,
)
from hmrec.weekly.monitoring import (
    Finding,
    feature_findings,
    labelled_findings,
    output_findings,
    step_failed_finding,
    transaction_findings,
)
from hmrec.weekly.outputs import (
    chain_model_pointer,
    ranker_model_pointer,
    read_week_evaluation,
)
from hmrec.weekly.shards import number_of_shards
from hmrec.weekly.state import (
    CurrentPointer,
    next_run_id,
    require_same_mode,
    require_tick_may_start,
    switch_current,
    write_marker,
)
from hmrec.weekly.steps import (
    C1_INGEST,
    C2_EVALUATE,
    C3_SAMPLE,
    C4_FINE_TUNE,
    O1_QUERIES,
    PUBLISH,
    SHARD_FEATURES,
    SHARD_SCORE,
    Step,
    final_dir,
    shard_score_step,
    staging_dir,
    steps_before_shards,
    steps_of_shards,
)
from hmrec.weekly.storage import LocalStateStore, StateStore

LOGGER = get_logger("weekly.driver")

STEP_SCRIPT = "m12_step.py"
CLOSING_STEPS = (C2_EVALUATE, C3_SAMPLE, C4_FINE_TUNE)
LABEL_ACCESS_COLUMNS: tuple[str, ...] = (
    "recorded_at",
    "run_id",
    "tick_anchor",
    "mode",
    "week",
    "phase",
    "step",
    "purpose",
    "git_commit",
)
STEP_COLUMNS: tuple[str, ...] = (
    "run_id",
    "tick_anchor",
    "mode",
    "step",
    "status",
    "seconds",
    "process_max_rss_gb",
    "os",
    "architecture",
    "openmp_library",
    "openmp_version",
    "detail",
    "git_commit",
)
METRIC_COLUMNS: tuple[str, ...] = ("run_id", "tick_anchor", "mode", "week", "name", "value")
LABEL_PURPOSES = {
    O1_QUERIES: "eval-mode query list: WHO buys in the week the tick opens (docs/mlops.md 6.2)",
    C2_EVALUATE: "closing: label sets of the closed week, evaluation of its predictions",
    C3_SAMPLE: "closing: Ranker training sample of the closed week (labels as training labels)",
}


class StepFailed(RuntimeError):
    """A step of a tick failed, or a check stopped the tick after it."""

    def __init__(self, step: str, reason: str) -> None:
        """Remember which step stopped the tick and why."""
        super().__init__(f"step {step} stopped the tick: {reason}")
        self.step = step
        self.reason = reason


@dataclass(frozen=True)
class StepCall:
    """One step to run for a tick.

    Attributes:
        anchor: The tick's anchor.
        mode: "eval" or "serve".
        run_id: The run of the tick.
        step: The step.
    """

    anchor: date
    mode: str
    run_id: str
    step: Step


# Runs one step to its end; raises if the step failed. The step leaves step.json in its
# staging folder (hmrec.weekly.context.finish_step).
StepRunner = Callable[[StepCall], None]


def subprocess_runner(scripts_dir: Path) -> StepRunner:
    """A runner that starts scripts/m12_step.py in a new process for every step.

    Args:
        scripts_dir: The folder that holds m12_step.py.
    """

    def run(call: StepCall) -> None:
        command = [
            sys.executable,
            str(scripts_dir / STEP_SCRIPT),
            "--anchor",
            call.anchor.isoformat(),
            "--mode",
            call.mode,
            "--run-id",
            call.run_id,
            "--step",
            call.step.name,
        ]
        LOGGER.info("starting: %s", " ".join(command[1:]))
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"the step's process ended with exit code {completed.returncode}")

    return run


@dataclass
class TickRun:
    """What a run of a tick collects while it goes.

    Attributes:
        context: The tick's context.
        store: The state.
        runner: Runs the steps.
        monitoring: The monitoring rules.
        steps: The step records so far, in order.
        metrics: The metric values the steps measured.
        outputs: The output files the steps recorded.
        findings: The monitoring findings so far.
    """

    context: TickContext
    store: StateStore
    runner: StepRunner
    monitoring: MonitoringConfig
    steps: list[StepRecord]
    metrics: list[MetricRecord]
    outputs: list[FileRecord]
    findings: list[Finding]


@dataclass(frozen=True)
class TickResult:
    """The outcome of one run of a tick.

    Attributes:
        manifest: The run manifest that was written.
        manifest_file: Where it is.
        findings: The monitoring findings of the run.
    """

    manifest: RunManifest
    manifest_file: Path
    findings: tuple[Finding, ...]

    @property
    def succeeded(self) -> bool:
        """True if every step passed and the week was published."""
        return self.manifest.status == SUCCEEDED


# ---------------------------------------------------------------- one step


def relative(context: TickContext, path: Path) -> str:
    """A path relative to the repository root, with "/"."""
    return path.relative_to(context.paths.project_root).as_posix()


def collect_step(run: TickRun, step: Step, folder: Path, status: str) -> None:
    """Add a finished step's record, metrics and output files to the run.

    Args:
        run: The run.
        step: The step.
        folder: The step's FINAL folder (its step.json is read from there).
        status: "done" for a step that ran now; "skipped" for one whose output
            existed (a resumed tick): its metrics and files are still collected,
            so the manifest of the resumed run describes the whole tick.
    """
    output = read_step_output(folder)
    if status == STEP_DONE:
        record = StepRecord(
            step.name,
            status,
            output.seconds,
            output.process_max_rss_gb,
            output.platform,
            output.detail,
        )
    else:
        detail = f"output exists, not run again ({output.detail})"
        record = StepRecord(step.name, status, 0.0, 0.0, None, detail)
    run.steps.append(record)
    run.metrics.extend(output.metrics)
    root = run.context.paths.project_root
    for file in output.files:
        run.outputs.append(file_record(file.role, root, relative(run.context, folder / file.name)))


def skipped(run: TickRun, step: Step, detail: str) -> None:
    """Record a step that did not run and has no folder to read."""
    run.steps.append(StepRecord(step.name, STEP_SKIPPED, 0.0, 0.0, None, detail))


def record_label_access(run: TickRun, step: Step) -> None:
    """Append a row to the label-access table BEFORE a step reads the answers of a week.

    Steps that read answers: the `eval` query list (who buys in the tick's week),
    C2 and C3 (the closed week). Other steps write no row.
    """
    context = run.context
    if step.name not in LABEL_PURPOSES or (step.name == O1_QUERIES and context.mode != EVAL_MODE):
        return
    row = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": context.run_id,
        "tick_anchor": context.tick.anchor.isoformat(),
        "mode": context.mode,
        "week": step.week.name,
        "phase": step.week.fold.phase,
        "step": step.name,
        "purpose": LABEL_PURPOSES[step.name],
        "git_commit": context.stamp.git_commit,
    }
    append_row(context.paths.replay_table_file("label_access"), LABEL_ACCESS_COLUMNS, row)


def failure_reason(run: TickRun, step: Step, error: Exception) -> str:
    """Why a step failed: what its process wrote before it ended, else the runner's error."""
    failure_file = run.context.layout.step_failure_file(run.context.run_id, step.name)
    if failure_file.is_file():
        return failure_file.read_text(encoding="utf-8").strip()
    return f"{type(error).__name__}: {error}"


def set_staging_aside(run: TickRun, step: Step) -> None:
    """Move what a failed or interrupted attempt of a step left behind to .failed/."""
    staging = staging_dir(run.context.layout, step)
    if run.store.exists(staging):
        failed = run.context.layout.failed_dir(step.week.anchor)
        moved = run.store.set_aside(staging, failed, step.name)
        LOGGER.warning("step %s: an unfinished attempt was moved to %s", step.name, moved)


def ingest_gate(run: TickRun, step: Step) -> None:
    """The gate after C1: the input-data rules. An alert stops the tick (docs/mlops.md §6.1).

    Raises:
        StepFailed: If the new week's row count is outside its band or its date
            range is wrong.
    """
    folder = staging_dir(run.context.layout, step)
    record = read_ingest_record(folder / "ingest.json")
    right_range = (
        record.new_week_first_date == record.expected_first_date
        and record.new_week_last_date == record.expected_last_date
    )
    findings = transaction_findings(
        run.context.tick.anchor,
        record.new_week_rows,
        record.previous_week_rows,
        right_range,
        run.monitoring,
    )
    run.findings.extend(findings)
    alerts = [finding for finding in findings if finding.alert]
    if alerts:
        reasons = "; ".join(
            f"{finding.rule}: value {finding.value:g}, rule: {finding.limit}" for finding in alerts
        )
        raise StepFailed(step.name, f"input data check failed: {reasons}")


def after_commit(run: TickRun, step: Step) -> None:
    """What the driver does once a step's folder is in place.

    O5: switch the `current` pointer, then mark the week as published.
    A shard's score: remove the shard's features (docs/mlops.md §8.4).
    C4: mark the closed week as closed (C2 and C3 are before it in the order).
    """
    context, store, layout = run.context, run.store, run.context.layout
    if step.kind == PUBLISH:
        publish(run)
    if step.kind == SHARD_SCORE and step.shard is not None:
        store.discard_dir(
            layout.step_dir(step.week.anchor, shard_step_dir(step.shard, FEATURES_DIR))
        )
    if step.name == C4_FINE_TUNE and not store.exists(layout.closed_file(step.week.anchor)):
        content = {"week": step.week.name, "anchor": step.week.anchor.isoformat()}
        write_marker(
            store,
            layout.closed_file(step.week.anchor),
            {**content, "closed_by_run": context.run_id},
        )


def publish(run: TickRun) -> None:
    """Switch state/current.json to the new prediction set and mark the week as published."""
    context, store, layout = run.context, run.store, run.context.layout
    week = context.tick
    folder = layout.prediction_set_dir(week.anchor)
    pointer = CurrentPointer(
        anchor=week.anchor,
        mode=context.mode,
        closed_week=context.closing.anchor,
        prediction_set=folder.relative_to(layout.root).as_posix(),
        models=(ranker_model_pointer(context), chain_model_pointer(context)),
        run_id=context.run_id,
    )
    switch_current(store, layout, pointer)
    if not store.exists(layout.published_file(week.anchor)):
        content = {"week": week.name, "anchor": week.anchor.isoformat(), "mode": context.mode}
        write_marker(
            store, layout.published_file(week.anchor), {**content, "run_id": context.run_id}
        )
    LOGGER.info("O5: current -> %s", pointer.prediction_set)


def step_output_exists(run: TickRun, step: Step) -> bool:
    """True if a step is finished: its output folder exists.

    A shard's features are finished when the shard is SCORED: the features
    themselves were removed then.
    """
    layout = run.context.layout
    if run.store.exists(final_dir(layout, step)):
        return True
    if step.kind == SHARD_FEATURES and step.shard is not None:
        return run.store.exists(final_dir(layout, shard_score_step(step.week, step.shard)))
    return False


def run_step(run: TickRun, step: Step) -> None:
    """Run one step, or skip it if its output exists (see the module docstring).

    Raises:
        StepFailed: If the step's process failed, or the gate after it did.
    """
    context, layout = run.context, run.context.layout
    if step.name in CLOSING_STEPS and not context.closes_a_week:
        skipped(run, step, f"week {step.week.name} is closed in the bootstrap state")
        return
    if step_output_exists(run, step):
        folder = final_dir(layout, step)
        if run.store.exists(folder):
            collect_step(run, step, folder, STEP_SKIPPED)
        else:
            skipped(run, step, "the shard is scored; its features were removed")
        if step.kind == PUBLISH and not run.store.exists(layout.published_file(step.week.anchor)):
            # An earlier run stopped between the prediction set and the pointer.
            publish(run)
        return
    set_staging_aside(run, step)
    record_label_access(run, step)
    try:
        run.runner(StepCall(context.tick.anchor, context.mode, context.run_id, step))
        if step.name == C1_INGEST:
            ingest_gate(run, step)
        run.store.commit_dir(staging_dir(layout, step), final_dir(layout, step))
    except StepFailed:
        set_staging_aside(run, step)
        raise
    except Exception as error:
        reason = failure_reason(run, step, error)
        set_staging_aside(run, step)
        raise StepFailed(step.name, reason) from error
    collect_step(run, step, final_dir(layout, step), STEP_DONE)
    after_commit(run, step)
    LOGGER.info("step %s done", step.name)


# ---------------------------------------------------------------- monitoring of a tick


def same_week_findings(run: TickRun) -> list[Finding]:
    """The feature and output-signal rules for the week the tick opened."""
    context, config = run.context, run.monitoring
    week = context.tick
    findings: list[Finding] = []
    features = own_feature_summaries(context, week)
    if features is not None:
        findings += feature_findings(
            week.anchor, features, feature_history(context, config), config
        )
    signals = own_output_signals(context, week)
    current = {name: signals[name] for name in OUTPUT_SIGNALS if name in signals}
    findings += output_findings(week.anchor, current, output_history(context, config), config)
    return findings


def closed_week_findings(run: TickRun) -> list[Finding]:
    """The labelled rules for the week the tick closed; empty for the first tick."""
    context, config = run.context, run.monitoring
    if not context.closes_a_week:
        return []
    closing = context.closing
    evaluation = read_week_evaluation(context.view.evaluation_file(closing.name))
    metric = context.pipeline.metrics[0]
    return labelled_findings(
        closing.anchor,
        evaluation.channel_recall,
        recall_history(context, config, ALL_CHANNELS),
        evaluation.union_recall,
        evaluation.ranker[metric],
        evaluation.rrf_baseline[metric],
        evaluation.hit_value,
        earlier_labelled_flags(context, config),
        config,
    )


def monitoring_metrics(anchor: date, findings: Sequence[Finding]) -> list[MetricRecord]:
    """The lines of the run manifest for the monitoring: two counts, then every raised finding.

    A raised flag or alert is one metric with the value 1.0 and the week it
    describes. Metric names are unique per week, so the rule and the subject are
    both in the name.
    """
    flags = [finding for finding in findings if finding.flag]
    alerts = [finding for finding in findings if finding.alert]
    metrics = [
        MetricRecord("monitor/flags_raised", anchor, float(len(flags))),
        MetricRecord("monitor/alerts_raised", anchor, float(len(alerts))),
    ]
    for kind, raised in (("flag", flags), ("alert", alerts)):
        for finding in raised:
            subject = f"/{finding.subject}" if finding.subject else ""
            metrics.append(
                MetricRecord(f"monitor/{kind}/{finding.rule}{subject}", finding.week, 1.0)
            )
    return metrics


def record_findings(run: TickRun) -> None:
    """Keep the tick's findings in the state (once) and append them to the monitoring table.

    A resumed tick whose findings are already in the state writes nothing again.
    """
    context = run.context
    target = context.layout.monitoring_file(context.tick.anchor)
    if run.store.exists(target):
        LOGGER.warning(
            "%s exists: the findings of this tick were recorded by an earlier run", target
        )
        return
    write_findings(run.store, target, run.findings)
    append_findings(context, context.paths.replay_table_file("monitoring"), run.findings)


# ---------------------------------------------------------------- the manifest


def tick_inputs(context: TickContext) -> list[FileRecord]:
    """The files a tick reads that it did not write itself, with size and sha256."""
    paths = context.paths
    files = {
        "processed/transactions": paths.transactions_file,
        "processed/customers": paths.customers_file,
        "processed/articles": paths.articles_file,
        "processed/metadata": paths.processed_metadata_file,
        "bootstrap/manifest": paths.bootstrap_manifest_file,
        "lock/channels": paths.channels_lock_file,
        "lock/twotower": paths.twotower_lock_file,
        "lock/features": paths.features_lock_file,
        "lock/ranker": paths.ranker_lock_file,
        "config/pipeline": paths.pipeline_config_file,
        "config/monitoring": paths.monitoring_config_file,
    }
    return [
        file_record(role, paths.project_root, relative(context, path))
        for role, path in files.items()
        if path.is_file()
    ]


def build_manifest(
    run: TickRun, status: str, failure: str, started_at: datetime, started: float
) -> RunManifest:
    """The run manifest of a tick that ended (succeeded or failed)."""
    context = run.context
    paths = context.paths
    metrics = [*run.metrics, *monitoring_metrics(context.tick.anchor, run.findings)]
    return RunManifest(
        run_id=context.run_id,
        anchor=context.tick.anchor,
        mode=context.mode,
        status=status,
        failure=failure,
        code=read_run_code(paths.project_root, paths.results_root),
        platform=read_platform_fields(paths.project_root),
        inputs=tuple(tick_inputs(context)),
        outputs=tuple(run.outputs),
        metrics=tuple(metrics),
        steps=tuple(run.steps),
        started_at_utc=started_at,
        finished_at_utc=datetime.now(UTC).replace(microsecond=0),
        total_seconds=time.perf_counter() - started,
        driver_max_rss_gb=read_process_peak_memory().process_max_rss_gb,
    )


def append_run_tables(context: TickContext, manifest: RunManifest) -> None:
    """Append the run's steps and metric values to the result tables (results/m12_replay/)."""
    identity = {
        "run_id": manifest.run_id,
        "tick_anchor": manifest.anchor.isoformat(),
        "mode": manifest.mode,
    }
    for step in manifest.steps:
        platform = step.platform
        row = {
            **identity,
            "step": step.name,
            "status": step.status,
            "seconds": repr(step.seconds),
            "process_max_rss_gb": repr(step.process_max_rss_gb),
            "os": "" if platform is None else platform.os,
            "architecture": "" if platform is None else platform.architecture,
            "openmp_library": "" if platform is None else platform.openmp_library,
            "openmp_version": "" if platform is None else platform.openmp_version,
            "detail": step.detail,
            "git_commit": manifest.code.git_commit,
        }
        append_row(context.paths.replay_table_file("steps"), STEP_COLUMNS, row)
    for metric in manifest.metrics:
        row = {
            **identity,
            "week": metric.week.isoformat(),
            "name": metric.name,
            "value": repr(metric.value),
        }
        append_row(context.paths.replay_table_file("metrics"), METRIC_COLUMNS, row)


# ---------------------------------------------------------------- the tick


def require_closable(context: TickContext) -> None:
    """Fail if the week the tick would close was opened in `serve` mode (module docstring).

    Raises:
        RuntimeError: If so.
    """
    if context.closes_a_week and week_mode(context, context.closing) == SERVE_MODE:
        raise RuntimeError(
            f"week {context.closing.name} was opened in serve mode: it cannot be closed. "
            "Closing needs the features of the customers who bought, and a serve tick removes "
            "a shard's features after scoring (docs/mlops.md 8.4)."
        )


def all_steps(run: TickRun) -> None:
    """Run the steps of the tick in order; the shard steps once the query list exists."""
    context = run.context
    for step in steps_before_shards(context.tick, context.closing, context.mode):
        run_step(run, step)
    if context.mode != SERVE_MODE:
        return
    queries_file = context.view.queries_file(context.tick.name)
    n_queries = int(pl.scan_parquet(queries_file).select(pl.len()).collect().item())
    n_shards = number_of_shards(n_queries, context.pipeline.shard_customers)
    LOGGER.info("serve: %d customers in %d shards", n_queries, n_shards)
    for step in steps_of_shards(context.tick, n_shards):
        run_step(run, step)


def run_tick(
    paths: ProjectPaths,
    anchor: date,
    mode: str,
    runner: StepRunner,
    stamp: RunStamp,
    store: StateStore | None = None,
) -> TickResult:
    """Run one tick: close the week before `anchor`, open the week of `anchor`.

    Args:
        paths: The project paths.
        anchor: The tick's anchor, the only clock of the pipeline.
        mode: "eval" or "serve".
        runner: Runs one step (subprocess_runner in production).
        stamp: The commit of the code and whether it is dirty.
        store: The state; None means the local folder artifacts/pipeline/.

    Returns:
        The run manifest (status "succeeded" or "failed") and the findings.

    Raises:
        RuntimeError: If the tick may not start: uncommitted code, a skipped
            week, another mode than before, or a `serve` week to close. Nothing
            is written then, not even a manifest: the tick did not start.
    """
    require_committed_code(stamp)
    plain = load_tick_context(paths, anchor, mode, "pending", stamp)
    state = store if store is not None else LocalStateStore(plain.layout.root)
    require_tick_may_start(state, plain.layout, plain.tick, plain.closing, plain.first_week)
    require_closable(plain)
    context = replace(plain, run_id=next_run_id(state, plain.layout, anchor, mode))
    require_same_mode(state, context.layout, context.tick, mode, context.run_id)
    monitoring = load_monitoring_config(paths.monitoring_config_file)
    run = TickRun(context, state, runner, monitoring, [], [], [], [])
    started_at, started = datetime.now(UTC).replace(microsecond=0), time.perf_counter()
    LOGGER.info("tick %s (%s, %s), run %s", context.tick.name, anchor, mode, context.run_id)
    status, failure = SUCCEEDED, ""
    try:
        all_steps(run)
        run.findings += same_week_findings(run) + closed_week_findings(run)
        record_findings(run)
    except StepFailed as error:
        status, failure = FAILED, str(error)
        run.findings.append(step_failed_finding(context.tick.anchor, error.step))
        append_findings(context, paths.replay_table_file("monitoring"), run.findings)
        LOGGER.error("tick %s stopped: %s", context.tick.name, failure)
    manifest = build_manifest(run, status, failure, started_at, started)
    manifest_file = context.layout.run_manifest_file(context.run_id)
    write_run_manifest(manifest_file, manifest)
    append_run_tables(context, manifest)
    LOGGER.info(
        "tick %s %s in %.0f s; manifest %s",
        context.tick.name,
        status,
        manifest.total_seconds,
        manifest_file,
    )
    return TickResult(manifest, manifest_file, tuple(run.findings))
