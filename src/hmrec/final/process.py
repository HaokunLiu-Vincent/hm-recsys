"""One step of the final evaluation in its own process (M8, called by scripts/m8_step.py).

A worker process does exactly one step for one fold, records it, and exits:

    monitor    rows -> results/t_phase_recall.csv (or the rehearsal's table)
    baseline   one metric row of the RRF baseline -> the area's metric table
    sample     a new sample file of one training fold (formal run only)
    dataset    the binary Dataset of one target fold
    trial      one model trained and its fold scored: metric rows, then the trial
               row (written last: it marks the trial as done)

Every step except "trial" also appends a row to results/m8_runs.csv (time,
sampled and true memory peak, LightGBM / libomp versions, git commit); a trial's
row holds the same information. The true peak is checked against
memory.limit_gb as the LAST thing, after the rows are written.

In the formal mode the worker passes the gate first
(hmrec.final.gate.formal_run_permit): no passed rehearsal, no recorded start or
changed code means it stops before it reads anything.
"""

import time
from dataclasses import asdict
from datetime import datetime

from hmrec.csv_records import append_row
from hmrec.eval.final_eval import FormalRunPermit
from hmrec.eval.monitor import MONITOR_COLUMNS
from hmrec.final.context import MODE_FORMAL, FinalContext, area_of, load_final_context
from hmrec.final.gate import formal_run_permit, require_clean_tree
from hmrec.final.monitor import monitor_fold, recall_file
from hmrec.final.run import (
    STEP_BASELINE,
    STEP_DATASET,
    STEP_MONITOR,
    STEP_SAMPLE,
    STEP_TRIAL,
    StepRequest,
    check_request,
)
from hmrec.final.trial import (
    baseline_row,
    build_final_dataset,
    run_final_trial,
    run_sampling,
    trial_spec,
)
from hmrec.git_state import git_state, warn_uncommitted_changes
from hmrec.hpo.v_records import (
    RankerTrialRecord,
    append_m7_run_row,
    append_metric_row,
    append_trial,
    results_files,
)
from hmrec.hpo.v_trial import TrialOutcome
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.memory import MemoryTracker, ProcessPeakMemory, bytes_to_gb, log_process_peak_memory
from hmrec.paths import ProjectPaths, get_paths
from hmrec.ranker.runtime import read_lightgbm_runtime
from hmrec.run_records import GENERATED, RunStamp, StepResult, check_memory_limit, run_row

LOGGER = get_logger("final.process")


def _now() -> str:
    """The current time as ISO text, to the second."""
    return datetime.now().isoformat(timespec="seconds")


def start_final_process(run_name: str) -> FinalContext:
    """Common start of every M8 script: logging, git state, LightGBM / OpenMP runtime, configs.

    Args:
        run_name: Name of the run's log folder, artifacts/runs/{run_name}/.

    Returns:
        The context. Loading it checks the Ranker lock, including that this
        process runs the recorded LightGBM and libomp versions.
    """
    paths = get_paths()
    setup_logging(log_file=paths.run_dir(run_name) / "run.log")
    return load_process_context(paths)


def load_process_context(paths: ProjectPaths) -> FinalContext:
    """The context of THIS process: its git state, its LightGBM / OpenMP runtime, the configs.

    Also used by scripts/checks/check_ranker_eval.py, which sets up its own logging.
    """
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("git HEAD %s, uncommitted code/config changes: %s", commit, dirty)
    if dirty:
        warn_uncommitted_changes(commit)
    # Fails here if a second OpenMP runtime (e.g. torch's) is loaded in this process.
    runtime = read_lightgbm_runtime()
    LOGGER.info(
        "LightGBM %s; OpenMP runtime %s (version %s)",
        runtime.lightgbm_version,
        runtime.libomp_path,
        runtime.libomp_version,
    )
    return load_final_context(paths, RunStamp(commit, dirty), runtime)


def step_label(request: StepRequest) -> str:
    """Name of a step in logs and in results/m8_runs.csv, e.g. "formal_dataset_full"."""
    return f"{request.mode}_{request.step}_{request.experiment}"


def _helper_step(
    context: FinalContext, request: StepRequest, permit: FormalRunPermit | None
) -> tuple[int, str]:
    """Run a step that is not a trial; returns (rows written, a short detail text)."""
    mode, fold, experiment = request.mode, request.fold, request.experiment
    if request.step == STEP_MONITOR:
        rows = monitor_fold(context, mode, fold, permit)
        for row in rows:
            append_row(recall_file(context, mode), MONITOR_COLUMNS, row)
        return len(rows), "channels and union"
    if request.step == STEP_BASELINE:
        row = baseline_row(context, mode, fold, permit)
        metrics_file = results_files(context.paths, area_of(mode))[1]
        append_metric_row(
            metrics_file, row, context.config.metrics, context.stamp.git_commit, _now()
        )
        return row.n_queries, "n_rows = queries of the fold"
    if request.step == STEP_SAMPLE:
        if permit is None:
            raise TypeError("sampling belongs to the formal run and needs a FormalRunPermit")
        record = run_sampling(context, experiment, fold, permit)
        return record.n_rows, f"{record.n_queries} queries, {record.n_positives} positives"
    if request.step != STEP_DATASET:
        raise ValueError(f"not a helper step: {request.step!r}")
    dataset = build_final_dataset(context, mode, experiment, fold)
    train_folds = " ".join(dataset.train_folds)
    return dataset.n_rows, f"{dataset.n_queries} queries; train folds {train_folds}"


def _record_trial(
    context: FinalContext,
    request: StepRequest,
    outcome: TrialOutcome,
    seconds: float,
    sampled_peak_gb: float,
    peak: ProcessPeakMemory,
) -> None:
    """Append a trial's metric rows, then its trial row (last: it marks the trial as done)."""
    trial_file, metrics_file = results_files(context.paths, area_of(request.mode))
    finished_at = _now()
    for metric_row in outcome.metric_rows:
        append_metric_row(
            metrics_file, metric_row, context.config.metrics, context.stamp.git_commit, finished_at
        )
        LOGGER.info("%d trees: %s", metric_row.num_trees, metric_row.scores)
    spec = trial_spec(context, request.mode, request.experiment, request.fold)
    record = RankerTrialRecord(
        phase=context.manifest.get(request.fold).phase,
        experiment=request.experiment,
        step=request.mode,
        config_id=outcome.config_id,
        fold=request.fold,
        train_folds=outcome.train_folds,
        params=asdict(spec.tuned),
        config=outcome.config,
        histogram_mode=spec.histogram_mode,
        num_trees=spec.num_trees,
        evaluated=True,
        n_train_rows=outcome.n_train_rows,
        n_train_queries=outcome.n_train_queries,
        n_scored_rows=outcome.n_scored_rows,
        n_queries=outcome.n_queries,
        load_seconds=outcome.load_seconds,
        train_seconds=outcome.train_seconds,
        seconds_per_tree=outcome.seconds_per_tree,
        score_seconds=outcome.score_seconds,
        total_seconds=seconds,
        sampled_peak_rss_gb=sampled_peak_gb,
        process_max_rss_gb=peak.process_max_rss_gb,
        model_sha256=outcome.model_sha256,
        runtime=context.runtime,
        git_commit=context.stamp.git_commit,
        git_dirty=context.stamp.git_dirty,
        finished_at=finished_at,
    )
    append_trial(trial_file, record)
    LOGGER.info("trial row -> %s (config %s)", trial_file, outcome.config_id)


def run_worker(context: FinalContext, request: StepRequest, started: float) -> None:
    """Do one step, record it and check the memory limit.

    Args:
        context: The context of this process.
        request: The step.
        started: time.perf_counter() at the start of the process (for the total time).

    Raises:
        FinalRunRefused: If the tree is dirty or, in the formal mode, the gate refuses.
        MemoryError: If the true peak is above memory.limit_gb (after the rows are written).
    """
    check_request(request)
    require_clean_tree(context, f"an M8 step ({step_label(request)})")
    # The gate, before anything of the step is read. A rehearsal needs no permit and
    # cannot reach a T fold (hmrec.final.trial.training_window).
    permit = formal_run_permit(context) if request.mode == MODE_FORMAL else None
    label = f"m8 {request.fold} {step_label(request)}"
    memory = context.base.memory
    outcome, n_rows, detail = None, 0, ""
    with MemoryTracker(
        label,
        LOGGER,
        limit_gb=memory.limit_gb,
        sample_interval_seconds=memory.sample_interval_seconds,
    ) as tracker:
        if request.step == STEP_TRIAL:
            outcome = run_final_trial(
                context, request.mode, request.experiment, request.fold, permit
            )
        else:
            n_rows, detail = _helper_step(context, request, permit)
    sampled_peak_gb = bytes_to_gb(tracker.peak_rss_bytes)
    # Last step of the run: the true peak of the whole process.
    peak = log_process_peak_memory(LOGGER, label, memory.limit_gb)
    seconds = time.perf_counter() - started
    if outcome is not None:
        _record_trial(context, request, outcome, seconds, sampled_peak_gb, peak)
    else:
        result = StepResult(
            fold=request.fold,
            step=step_label(request),
            status=GENERATED,
            n_rows=n_rows,
            generate_seconds=tracker.elapsed_seconds,
            sampled_peak_rss_gb=sampled_peak_gb,
            checked_rows=0,
            detail=detail,
        )
        row = run_row(result, seconds, peak, context.stamp)
        append_m7_run_row(context.paths.m8_runs_file, row, context.runtime)
        LOGGER.info("run row -> %s: %s", context.paths.m8_runs_file, row)
    check_memory_limit(peak, memory.limit_gb, label)
