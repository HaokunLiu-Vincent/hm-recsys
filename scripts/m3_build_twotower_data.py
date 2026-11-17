"""CLI for M3: build the Two-Tower training arrays of the base window (proposal §6.1-6.2).

Usage:
    uv run python scripts/m3_build_twotower_data.py [--run-id m3_twotower_data]

Reads data/processed/ (through scan_processed_parquet), builds the samples and
histories of the base window in configs/recall/r6_twotower.yaml and writes them
to artifacts/twotower/data/base/ (refuses to write into a non-empty directory).
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import logging  # noqa: E402
from dataclasses import asdict  # noqa: E402

from hmrec.config import (  # noqa: E402
    load_base_config,
    load_preprocess_config,
    load_twotower_config,
)
from hmrec.data.folds import load_fold_manifest  # noqa: E402
from hmrec.data.metadata import scan_processed_parquet  # noqa: E402
from hmrec.hpo.runner import git_state  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import MemoryTracker, bytes_to_gb  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.recall.twotower.data import (  # noqa: E402
    build_window_arrays,
    check_window_before_tuning,
    summarize_training_arrays,
    write_training_arrays,
)

WINDOW_NAME = "base"
LOGGER = get_logger("m3_build_twotower_data")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", default="m3_twotower_data", help="log goes to artifacts/runs/{run_id}/run.log"
    )
    return parser.parse_args()


def main() -> None:
    """Build and write the base-window arrays, then log the summary."""
    args = parse_args()
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.run_dir(args.run_id) / "run.log")
    base = load_base_config(paths.base_config_file)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(base.num_threads))
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    twotower = load_twotower_config(paths.recall_config_file("r6_twotower"))
    window = twotower.base_window
    first_d_anchor = load_fold_manifest(paths.fold_manifest_file).by_phase("D")[0].anchor
    check_window_before_tuning(window, first_d_anchor)
    metadata_file = paths.processed_metadata_file
    transactions = scan_processed_parquet(paths.transactions_file, metadata_file, preprocess)
    n_articles = (
        scan_processed_parquet(paths.articles_file, metadata_file, preprocess)
        .select("article_idx")
        .collect()
        .height
    )
    with MemoryTracker(
        "two-tower base arrays",
        LOGGER,
        limit_gb=base.memory.limit_gb,
        sample_interval_seconds=base.memory.sample_interval_seconds,
    ) as tracker:
        arrays = build_window_arrays(
            transactions, window, preprocess.day_zero, twotower.history_length, n_articles
        )
    summary = summarize_training_arrays(arrays)
    commit, dirty = git_state(paths.project_root, paths.results_root)
    metadata = {
        "window": asdict(window),
        "history_length": twotower.history_length,
        "padding_idx": n_articles,
        "summary": asdict(summary),
        "build_seconds": tracker.elapsed_seconds,
        "build_peak_rss_gb": bytes_to_gb(tracker.peak_rss_bytes),
        "git_commit": commit,
        "git_dirty": dirty,
    }
    directory = paths.twotower_data_dir(WINDOW_NAME)
    write_training_arrays(arrays, directory, metadata)
    file_bytes = sum(path.stat().st_size for path in directory.glob("*.npy"))
    LOGGER.info("summary: %s", summary)
    LOGGER.info("files: %.1f MB in %s", file_bytes / 1024**2, directory)


if __name__ == "__main__":
    main()
