"""CLI for M5: check that the lock files reproduce the recorded D1 scores (one channel).

Usage (one process per channel, so /usr/bin/time reports each channel's true peak):
    /usr/bin/time -l caffeinate -i uv run python scripts/m5_verify_locks.py --channel r1_popularity
    ... --channel r2_repurchase | r3_sibling | r4_itemcf | r5_wals | r6_twotower

The channel is built ONLY from configs/locks/ (hmrec.recall.locked). Its D1
candidates are generated again and written to artifacts/m5/lock_check/D1/, a
folder no later step reads. Its selection metric must equal the D1 score of the
locked trial exactly (R1, R2, R4, R5, R6: results/d_phase.csv; R3:
results/m4_trials.csv), otherwise the script fails. R6 predicts with the locked
base checkpoint (the model that predicted D1 in the chain); nothing is trained.

Run it on a clean working tree (after the lock files are committed).
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import logging  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from hmrec.config import (  # noqa: E402
    load_base_config,
    load_catalog_config,
    load_preprocess_config,
)
from hmrec.data.folds import load_fold_manifest  # noqa: E402
from hmrec.hpo.d_phase import PHASE, load_fold_contexts  # noqa: E402
from hmrec.hpo.grid import load_phase_grid  # noqa: E402
from hmrec.hpo.lock_check import check_locked_channel, read_recorded_trial  # noqa: E402
from hmrec.hpo.runner import git_state, warn_uncommitted_changes  # noqa: E402
from hmrec.locks import ALL_CHANNELS, BASE_CHECKPOINT  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import MemoryTracker  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.recall.locked import build_locked_channel, load_locked_recall  # noqa: E402

LOGGER = get_logger("m5_verify_locks")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True, choices=ALL_CHANNELS)
    parser.add_argument("--run-id", default=None, help="log goes to artifacts/runs/{run_id}/")
    return parser.parse_args()


def main() -> None:
    """Build one channel from the lock files, regenerate D1, compare with the record."""
    args = parse_args()
    channel = args.channel
    paths = get_paths()
    run_id = args.run_id or f"m5_verify_locks_{channel}"
    setup_logging(logging.INFO, log_file=paths.run_dir(run_id) / "run.log")
    base = load_base_config(paths.base_config_file)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(base.num_threads))
    torch.set_num_threads(base.num_threads)
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("git HEAD %s, uncommitted code/config changes: %s", commit, dirty)
    if dirty:
        warn_uncommitted_changes(commit)
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    catalog = load_catalog_config(paths.catalog_config_file)
    grid = load_phase_grid(paths.d_phase_grid_file)
    folds = load_fold_manifest(paths.fold_manifest_file).by_phase(PHASE)
    # D1 only: it is the fold the locked BASE checkpoint predicted in the chain, so
    # all six channels can be checked on it without any fine-tune.
    fold = folds[0]
    locked = load_locked_recall(paths)
    locked_channel = locked.channel(channel)
    record = read_recorded_trial(paths, locked_channel, [f.name for f in folds], grid)
    with MemoryTracker(
        f"m5 lock check {channel}",
        LOGGER,
        limit_gb=base.memory.limit_gb,
        sample_interval_seconds=base.memory.sample_interval_seconds,
    ):
        context = load_fold_contexts(paths, [fold], preprocess, catalog)[0]
        started = time.perf_counter()
        recall_channel = build_locked_channel(
            channel, locked, base, paths, preprocess, r6_start=BASE_CHECKPOINT
        )
        LOGGER.info(
            "%s: built from the lock files in %.1f s", channel, time.perf_counter() - started
        )
        result = check_locked_channel(
            recall_channel,
            locked_channel,
            context,
            record.summary.fold_scores[fold.name],
            paths.lock_check_candidates_file(fold.name, channel),
            base.memory,
        )
    LOGGER.info("lock check passed: %s", result)


if __name__ == "__main__":
    main()
