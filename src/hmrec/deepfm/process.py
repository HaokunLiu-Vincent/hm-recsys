"""What a DeepFM process does around its one job: configs, code state, result rows.

A process trains one model or benchmarks one device, so its process max RSS is
the true memory peak of that job (hmrec.memory). Every process refuses a
working tree with uncommitted code or config changes: the runs read the full
data, and every result row must belong to one commit (AGENTS.md §3.4).
"""

import json
import logging
import time
from dataclasses import replace

import torch

from hmrec.config import DeepFMConfig, load_base_config, load_deepfm_config
from hmrec.csv_records import append_row
from hmrec.data.folds import load_fold_manifest
from hmrec.deepfm.benchmark import BenchmarkResult
from hmrec.deepfm.records import (
    BENCHMARK_COLUMNS,
    TRIAL_COLUMNS,
    ReferenceScores,
    metric_columns,
    model_settings,
    stamp_columns,
)
from hmrec.deepfm.trial import AREA_SPEED_TEST, TrialInputs, TrialResult, TrialSpec
from hmrec.git_state import git_state
from hmrec.locks import load_features_lock
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.memory import ProcessPeakMemory, log_process_peak_memory
from hmrec.paths import ProjectPaths, get_paths
from hmrec.run_records import RunStamp, check_memory_limit
from hmrec.threads import check_polars_thread_pool

LOGGER = get_logger("deepfm.process")


def require_clean_tree(stamp: RunStamp) -> None:
    """Fail if the working tree has uncommitted code or config changes."""
    if stamp.git_dirty:
        raise RuntimeError(
            f"the working tree has uncommitted changes (HEAD {stamp.git_commit[:12]}): a DeepFM "
            "run reads the full data and starts only from committed code"
        )


def load_trial_inputs(paths: ProjectPaths) -> TrialInputs:
    """Read the configs a DeepFM process needs (no data is read)."""
    return TrialInputs(
        paths=paths,
        base=load_base_config(paths.base_config_file),
        config=load_deepfm_config(paths.deepfm_config_file),
        manifest=load_fold_manifest(paths.fold_manifest_file),
        lock=load_features_lock(paths.features_lock_file),
    )


def start_process(label: str) -> tuple[TrialInputs, RunStamp]:
    """Set up logging, read the configs and refuse a dirty working tree.

    Args:
        label: Name of this process; names its log file under artifacts/deepfm/logs/.
    """
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.deepfm_log_file(label))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    stamp = RunStamp(commit, dirty)
    LOGGER.info("deepfm %s | git HEAD %s, uncommitted changes: %s", label, commit, dirty)
    require_clean_tree(stamp)
    inputs = load_trial_inputs(paths)
    threads = check_polars_thread_pool(inputs.base.num_threads)
    LOGGER.info("polars thread pool size = %d, torch %s", threads, torch.__version__)
    return inputs, stamp


def with_seed(config: DeepFMConfig, seed: int) -> DeepFMConfig:
    """The configuration with another training seed."""
    return replace(config, training=replace(config.training, seed=seed))


def speed_test_spec(inputs: TrialInputs) -> TrialSpec:
    """The one training of the speed test: the default configuration, 1 epoch, its target fold."""
    speed_test, training = inputs.config.speed_test, inputs.config.training
    return TrialSpec(AREA_SPEED_TEST, speed_test.target_fold, training.seed, speed_test.epochs)


def finish(inputs: TrialInputs, label: str) -> ProcessPeakMemory:
    """Read and log the process's true memory peak (call it after the job, before the rows)."""
    return log_process_peak_memory(LOGGER, label, inputs.base.memory.limit_gb)


# ---------------------------------------------------------------- result rows


def record_benchmark(
    inputs: TrialInputs, result: BenchmarkResult, peak: ProcessPeakMemory, stamp: RunStamp
) -> None:
    """Append one row to results/deepfm_benchmark.csv."""
    row = {
        "device": result.device,
        "config_id": result.config_id,
        "target_fold": result.target_fold,
        "batch_rows": str(inputs.config.training.batch_rows),
        "steps": str(result.steps),
        "discarded_steps": str(result.discarded_steps),
        "median_step_ms": repr(result.median_step_ms),
        "mean_step_ms": repr(result.mean_step_ms),
        "same_seed_losses_identical": str(result.same_seed_identical),
        "same_seed_max_loss_difference": repr(result.same_seed_max_difference),
        "first_loss": repr(result.first_loss),
        "last_loss": repr(result.last_loss),
        "scoring_rows": str(result.scoring_rows),
        "scoring_rows_per_second": repr(result.scoring_rows_per_second),
        "num_threads": str(inputs.base.num_threads),
        "torch_version": torch.__version__,
        "process_max_rss_gb": repr(peak.process_max_rss_gb),
        **stamp_columns(stamp),
    }
    append_row(inputs.paths.deepfm_benchmark_file, BENCHMARK_COLUMNS, row)


def record_trial(
    inputs: TrialInputs,
    spec: TrialSpec,
    result: TrialResult,
    started: float,
    peak: ProcessPeakMemory,
    stamp: RunStamp,
) -> None:
    """Append the trial's metric rows (one per epoch), then its trial row.

    The trial row is written LAST and marks the trial as recorded.
    """
    paths, metrics = inputs.paths, inputs.config.metrics
    for epoch in result.metrics:
        row = {
            "area": spec.area,
            "config_id": result.config_id,
            "seed": str(spec.seed),
            "fold": spec.target_fold,
            "epoch": str(epoch.epoch),
            "n_queries": str(epoch.n_queries),
            **{metric: repr(epoch.scores[metric]) for metric in metrics},
            "train_loss": repr(epoch.train_loss),
            "git_commit": stamp.git_commit,
            "finished_at": stamp_columns(stamp)["finished_at"],
        }
        append_row(paths.deepfm_metrics_file, metric_columns(metrics), row)
    train_seconds = sum(epoch.seconds for epoch in result.epochs)
    row = {
        "area": spec.area,
        "config_id": result.config_id,
        "config": json.dumps(model_settings(inputs.config), sort_keys=True),
        "seed": str(spec.seed),
        "fold": spec.target_fold,
        "train_folds": " ".join(result.train_folds),
        "device": result.device,
        "epochs": str(spec.epochs),
        "n_train_rows": str(result.n_train_rows),
        "n_train_queries": str(result.n_train_queries),
        "n_scored_rows": str(result.n_scored_rows),
        "n_queries": str(result.metrics[0].n_queries),
        "prepare_seconds": repr(result.prepare_seconds),
        "train_seconds": repr(train_seconds),
        "seconds_per_epoch": repr(train_seconds / len(result.epochs)),
        "score_seconds": repr(result.score_seconds),
        "score_seconds_per_model": repr(result.score_seconds / len(result.epochs)),
        "total_seconds": repr(time.perf_counter() - started),
        "process_max_rss_gb": repr(peak.process_max_rss_gb),
        "num_threads": str(inputs.base.num_threads),
        "torch_version": torch.__version__,
        **stamp_columns(stamp),
    }
    append_row(paths.deepfm_trials_file, TRIAL_COLUMNS, row)
    check_memory_limit(peak, inputs.base.memory.limit_gb, f"deepfm {spec.area} {spec.target_fold}")


def below_baseline(result: TrialResult, reference: ReferenceScores) -> bool:
    """True if the LAST epoch's value of the reference metric is below the RRF baseline."""
    return result.metrics[-1].scores[reference.metric] < reference.rrf_baseline
