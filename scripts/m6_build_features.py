"""CLI for M6: build the Ranker's wide feature table of one fold (proposal §7.3, §8).

Usage:
    caffeinate -i uv run python scripts/m6_build_features.py --fold F1

Reads the fold's six candidate files, R1's bucket score table and the R5 / R6
vectors (all written by scripts/m6_generate_candidates.py), computes the
features of configs/features.yaml and writes
artifacts/features/{fold}/part-*.parquet: one row per (query_id, article_idx),
sorted, one part per chunk of queries. No label is read; the table has no label
column. A finished fold is skipped, never replaced.

Each run appends the per-feature summary (missing rate, min, max, mean) to
results/m6_features.csv and one row (time, true memory peak) to results/m6_runs.csv.

Run it on a clean working tree (AGENTS.md §3.4). One fold per process.
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import logging  # noqa: E402
import time  # noqa: E402

from hmrec.config import (  # noqa: E402
    load_base_config,
    load_features_config,
    load_preprocess_config,
)
from hmrec.csv_records import append_row, read_rows  # noqa: E402
from hmrec.data.folds import load_fold_manifest  # noqa: E402
from hmrec.eval.monitor import MONITOR_COLUMNS, UNION  # noqa: E402
from hmrec.features.build import (  # noqa: E402
    FoldInputs,
    build_fold_features,
    load_candidates,
    load_dense_sources,
    load_fold_data,
    load_history,
)
from hmrec.features.stats import STATS_COLUMNS  # noqa: E402
from hmrec.hpo.runner import git_state, warn_uncommitted_changes  # noqa: E402
from hmrec.locks import (  # noqa: E402
    check_features_match_lock,
    load_features_lock,
    load_twotower_lock,
)
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import MemoryTracker, bytes_to_gb, log_process_peak_memory  # noqa: E402
from hmrec.paths import ProjectPaths, get_paths  # noqa: E402
from hmrec.run_records import (  # noqa: E402
    GENERATED,
    SKIPPED,
    RunStamp,
    StepResult,
    append_run_row,
    check_memory_limit,
    run_row,
)

LOGGER = get_logger("m6_build_features")
STEP = "features"
POST_LOCK_PHASES = ("F", "V", "T")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, help="a post-lock fold: F1 ... T4")
    return parser.parse_args()


def expected_rows(paths: ProjectPaths, fold: str) -> int:
    """The fold's number of deduplicated candidates recorded by the M6 monitor."""
    rows = read_rows(paths.m6_candidates_results_file, MONITOR_COLUMNS)
    matches = [row for row in rows if row["fold"] == fold and row["scope"] == UNION]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {UNION!r} row of fold {fold} in "
            f"{paths.m6_candidates_results_file}, found {len(matches)}"
        )
    return int(matches[0]["n_rows"])


def main() -> None:
    """Build the wide table of one fold and record the run."""
    started = time.perf_counter()
    fold_name = parse_args().fold
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.run_dir(f"m6_{fold_name}_{STEP}") / "run.log")
    base = load_base_config(paths.base_config_file)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(base.num_threads))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("git HEAD %s, uncommitted code/config changes: %s", commit, dirty)
    if dirty:
        warn_uncommitted_changes(commit)
    stamp = RunStamp(commit, dirty)
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    config = load_features_config(paths.features_config_file)
    # The features are locked (proposal §8.7): stop if configs/features.yaml was changed.
    check_features_match_lock(config, load_features_lock(paths.features_lock_file))
    fold = load_fold_manifest(paths.fold_manifest_file).get(fold_name)
    if fold.phase not in POST_LOCK_PHASES:
        raise ValueError(f"--fold must be a post-lock fold (F1 ... T4), got {fold_name!r}")
    # The feature history must be the one R6 was locked with (proposal §8.3).
    locked_history = load_twotower_lock(paths.twotower_lock_file).channel.config["history_length"]
    if config.history_length != locked_history:
        raise ValueError(
            f"features.history_length is {config.history_length}, but the locked R6 uses "
            f"{locked_history}"
        )
    label = f"m6 {fold_name} {STEP}"
    with MemoryTracker(
        label,
        LOGGER,
        limit_gb=base.memory.limit_gb,
        sample_interval_seconds=base.memory.sample_interval_seconds,
    ) as tracker:
        data = load_fold_data(paths, preprocess, fold)
        history = load_history(data, config)
        inputs = FoldInputs(
            data=data,
            history=history,
            candidates=load_candidates(paths, fold_name),
            dense=load_dense_sources(paths, preprocess, data, history, config, base),
            expected_rows=expected_rows(paths, fold_name),
        )
        result = build_fold_features(paths, inputs, config, base.chunking.query_chunk_size, stamp)
    for row in result.stats_rows:
        append_row(paths.m6_features_results_file, STATS_COLUMNS, row)
    checked = result.tt_dot_rows_checked + result.wals_dot_rows_checked
    detail = (
        f"{result.n_parts} parts, {result.n_queries} queries; tt_dot = R6 score on "
        f"{result.tt_dot_rows_checked} rows, wals_dot = R5 score on "
        f"{result.wals_dot_rows_checked} rows; tt_sim_* step "
        f"{result.history_similarity_seconds:.1f} s"
    )
    step_result = StepResult(
        fold=fold_name,
        step=STEP,
        status=SKIPPED if result.skipped else GENERATED,
        n_rows=result.n_rows,
        generate_seconds=result.seconds,
        sampled_peak_rss_gb=bytes_to_gb(tracker.peak_rss_bytes),
        checked_rows=checked,
        detail="" if result.skipped else detail,
    )
    # Last step of the run: the true peak of the whole process, into the log and the row.
    peak = log_process_peak_memory(LOGGER, label, base.memory.limit_gb)
    row = run_row(step_result, time.perf_counter() - started, peak, stamp)
    append_run_row(paths.m6_runs_file, row)
    LOGGER.info("run row -> %s: %s", paths.m6_runs_file, row)
    check_memory_limit(peak, base.memory.limit_gb, label)


if __name__ == "__main__":
    main()
