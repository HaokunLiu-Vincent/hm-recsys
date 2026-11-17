"""CLI for M6: monitor the generated candidates of one fold (reports only, proposal §6.6).

Usage:
    caffeinate -i uv run python scripts/m6_monitor_candidates.py --fold F1

Reads the six candidate files of the fold and appends one row per channel and
one for their union to results/m6_candidates.csv:
    every fold      rows and candidates per query;
    F and V folds   also Recall@quota, union recall and the share of queries
                    without any hit (through eval/);
    T folds         NO label is read. T metrics are computed in M8.
Nothing is selected or changed. A fold that already has rows is skipped.
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import logging  # noqa: E402
import time  # noqa: E402

import polars as pl  # noqa: E402

from hmrec.config import load_base_config, load_candidate_run_config  # noqa: E402
from hmrec.csv_records import append_row, read_rows  # noqa: E402
from hmrec.data.folds import load_fold_manifest  # noqa: E402
from hmrec.eval.monitor import (  # noqa: E402
    LABEL_PHASES,
    MONITOR_COLUMNS,
    FoldCandidates,
    load_label_evaluator,
    monitor_rows,
)
from hmrec.hpo.runner import git_state, warn_uncommitted_changes  # noqa: E402
from hmrec.locks import ALL_CHANNELS  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import MemoryTracker, bytes_to_gb, log_process_peak_memory  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.pipeline import MONITOR_STEP  # noqa: E402
from hmrec.recall.inputs import read_queries  # noqa: E402
from hmrec.recall.locked import load_locked_recall, post_lock_folds  # noqa: E402
from hmrec.run_records import (  # noqa: E402
    GENERATED,
    SKIPPED,
    RunStamp,
    StepResult,
    append_run_row,
    check_memory_limit,
    run_row,
)

LOGGER = get_logger("m6_monitor_candidates")
CANDIDATE_COLUMNS = ["query_id", "article_idx", "rank"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, help="a post-lock fold: F1 ... T4")
    return parser.parse_args()


def main() -> None:
    """Compute and append the monitoring rows of one fold."""
    started = time.perf_counter()
    fold_name = parse_args().fold
    paths = get_paths()
    setup_logging(
        logging.INFO, log_file=paths.run_dir(f"m6_{fold_name}_{MONITOR_STEP}") / "run.log"
    )
    base = load_base_config(paths.base_config_file)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(base.num_threads))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("git HEAD %s, uncommitted code/config changes: %s", commit, dirty)
    if dirty:
        warn_uncommitted_changes(commit)
    stamp = RunStamp(commit, dirty)
    config = load_candidate_run_config(paths.m6_config_file)
    manifest = load_fold_manifest(paths.fold_manifest_file)
    if fold_name not in [fold.name for fold in post_lock_folds(manifest)]:
        raise ValueError(f"--fold must be a post-lock fold (F1 ... T4), got {fold_name!r}")
    fold = manifest.get(fold_name)
    results_file = paths.m6_candidates_results_file
    earlier_rows = read_rows(results_file, MONITOR_COLUMNS)
    label = f"m6 {fold_name} {MONITOR_STEP}"
    if any(row["fold"] == fold_name for row in earlier_rows):
        LOGGER.warning("%s already has rows in %s; SKIPPED", fold_name, results_file)
        result = StepResult(fold_name, MONITOR_STEP, SKIPPED, 0, 0.0, 0.0, 0, "")
    else:
        locked = load_locked_recall(paths)
        with MemoryTracker(
            label,
            LOGGER,
            limit_gb=base.memory.limit_gb,
            sample_interval_seconds=base.memory.sample_interval_seconds,
        ) as tracker:
            computing_started = time.perf_counter()
            data = FoldCandidates(
                fold=fold_name,
                phase=fold.phase,
                queries=read_queries(paths, fold_name),
                candidates_by_channel={
                    channel: pl.read_parquet(
                        paths.candidates_file(fold_name, channel), columns=CANDIDATE_COLUMNS
                    )
                    for channel in ALL_CHANNELS
                },
                locked={channel: locked.channel(channel) for channel in ALL_CHANNELS},
                fold_names=manifest.names,
            )
            # Labels are read for F and V folds only; a T fold gets no evaluator.
            evaluator = None
            if fold.phase in LABEL_PHASES:
                evaluator = load_label_evaluator(
                    fold_name, fold.phase, paths.labels_file(fold_name)
                )
            rows = monitor_rows(
                data, evaluator, earlier_rows, config, base.chunking.query_chunk_size, stamp
            )
            seconds = time.perf_counter() - computing_started
        for row in rows:
            append_row(results_file, MONITOR_COLUMNS, row)
            LOGGER.info("%s %s: %s", fold_name, row["scope"], row)
        sampled_peak = bytes_to_gb(tracker.peak_rss_bytes)
        result = StepResult(
            fold_name, MONITOR_STEP, GENERATED, len(rows), seconds, sampled_peak, 0, ""
        )
    peak = log_process_peak_memory(LOGGER, label, base.memory.limit_gb)
    append_run_row(paths.m6_runs_file, run_row(result, time.perf_counter() - started, peak, stamp))
    check_memory_limit(peak, base.memory.limit_gb, label)


if __name__ == "__main__":
    main()
