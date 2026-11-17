"""CLI for M6: generate the locked channels' candidates for one fold (proposal §6.6, §7.3).

Usage:
    caffeinate -i uv run python scripts/m6_generate_candidates.py --fold F1 --channel r4_itemcf
    caffeinate -i uv run python scripts/m6_generate_candidates.py --fold F1

With --channel: that channel is generated in THIS process. Without it: the six
channels are generated one after another, each in its own process (so each
one's process max RSS is its true memory peak).

The channel is built only from configs/locks/ (hmrec.recall.locked). Output:
artifacts/candidates/{fold}/{channel}.parquet, plus the files the features need
(R1 bucket scores, R5 and R6 vectors). R6 also fine-tunes on the fold's week
after predicting it, and must be run fold by fold in time order (checked by
sha256). Existing candidates are never replaced: the step is skipped. Each run
appends one row to results/m6_runs.csv. No label is read here; the label-based
monitoring is scripts/m6_monitor_candidates.py.

Run it on a clean working tree (AGENTS.md §3.4).
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import logging  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from hmrec.config import (  # noqa: E402
    load_base_config,
    load_candidate_run_config,
    load_catalog_config,
    load_preprocess_config,
)
from hmrec.data.folds import load_fold_manifest  # noqa: E402
from hmrec.hpo.runner import git_state, warn_uncommitted_changes  # noqa: E402
from hmrec.locks import ALL_CHANNELS  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import log_process_peak_memory  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.pipeline import run_script  # noqa: E402
from hmrec.recall.generate import (  # noqa: E402
    FoldJob,
    check_recorded_fine_tune_loss,
    generate_fold_channel,
)
from hmrec.recall.inputs import (  # noqa: E402
    build_recall_inputs,
    load_shared_tables,
    read_queries,
)
from hmrec.recall.locked import load_locked_recall, post_lock_folds  # noqa: E402
from hmrec.run_records import (  # noqa: E402
    RunStamp,
    append_run_row,
    check_memory_limit,
    run_row,
)

LOGGER = get_logger("m6_generate_candidates")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, help="a post-lock fold: F1 ... T4")
    parser.add_argument("--channel", default=None, choices=ALL_CHANNELS)
    return parser.parse_args()


def generate_one(fold_name: str, channel: str) -> None:
    """Generate one channel of one fold in this process and record the run."""
    started = time.perf_counter()
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.run_dir(f"m6_{fold_name}_{channel}") / "run.log")
    base = load_base_config(paths.base_config_file)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(base.num_threads))
    torch.set_num_threads(base.num_threads)
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("git HEAD %s, uncommitted code/config changes: %s", commit, dirty)
    if dirty:
        warn_uncommitted_changes(commit)
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    catalog = load_catalog_config(paths.catalog_config_file)
    config = load_candidate_run_config(paths.m6_config_file)
    manifest = load_fold_manifest(paths.fold_manifest_file)
    fold_names = tuple(fold.name for fold in post_lock_folds(manifest))
    if fold_name not in fold_names:
        raise ValueError(f"--fold must be one of {list(fold_names)}, got {fold_name!r}")
    fold = manifest.get(fold_name)
    tables = load_shared_tables(paths, preprocess)
    job = FoldJob(
        paths=paths,
        base=base,
        preprocess=preprocess,
        locked=load_locked_recall(paths),
        fold=fold,
        fold_names=fold_names,
        queries=read_queries(paths, fold_name),
        inputs=build_recall_inputs(tables, fold, preprocess, catalog),
    )
    LOGGER.info(
        "%s: anchor %s, %d queries, active pool %d",
        fold_name,
        fold.anchor,
        job.queries.height,
        job.inputs.active_pool.height,
    )
    stamp = RunStamp(commit, dirty)
    result = generate_fold_channel(job, channel, stamp)
    # Last step of the run: the true peak of the whole process, into the log and the row.
    label = f"m6 {fold_name} {channel}"
    peak = log_process_peak_memory(LOGGER, label, base.memory.limit_gb)
    row = run_row(result, time.perf_counter() - started, peak, stamp)
    append_run_row(paths.m6_runs_file, row)
    LOGGER.info("run row -> %s: %s", paths.m6_runs_file, row)
    # Checks that stop a driver. They come after the row, so the run is on record.
    check_memory_limit(peak, base.memory.limit_gb, label)
    check_recorded_fine_tune_loss(job, result, config)


def main() -> None:
    """One channel in this process, or all six channels in one process each."""
    args = parse_args()
    if args.channel is not None:
        generate_one(args.fold, args.channel)
        return
    setup_logging(logging.INFO)
    for channel in ALL_CHANNELS:
        run_script(Path(__file__), ["--fold", args.fold, "--channel", channel])


if __name__ == "__main__":
    main()
