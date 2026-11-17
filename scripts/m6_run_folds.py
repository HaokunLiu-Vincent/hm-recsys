"""CLI for M6: generate and monitor post-lock folds one after another, in time order.

Usage:
    caffeinate -i uv run python scripts/m6_run_folds.py --first-fold F1 --last-fold F1
    caffeinate -i uv run python scripts/m6_run_folds.py --first-fold F2 --last-fold T4

For each fold from --first-fold to --last-fold: the six channels (R6 last,
including its weekly fine-tune), then the monitor; every step in its own
process (scripts/m6_generate_candidates.py, scripts/m6_monitor_candidates.py).
After each fold one summary line is logged.

The run STOPS at the first step that fails. A step fails on: a schema or
consistency check, a model whose sha256 differs from its record, a process
whose true memory peak is above memory.limit_gb, or an R6 fine-tune loss that
is not finite or above twotower_chain.max_fine_tune_loss. Finished steps are
kept; after a human decision the same command continues where it stopped
(existing candidates are skipped, never replaced).
"""

import argparse
import logging
import time
from pathlib import Path

from hmrec.csv_records import read_rows
from hmrec.data.folds import load_fold_manifest
from hmrec.eval.monitor import MONITOR_COLUMNS
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.paths import get_paths
from hmrec.pipeline import fold_steps, fold_summary, run_script
from hmrec.recall.locked import post_lock_folds
from hmrec.run_records import RUN_COLUMNS

LOGGER = get_logger("m6_run_folds")
SCRIPTS = Path(__file__).resolve().parent
GENERATE_SCRIPT = SCRIPTS / "m6_generate_candidates.py"
MONITOR_SCRIPT = SCRIPTS / "m6_monitor_candidates.py"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-fold", required=True, help="first fold to run, e.g. F1")
    parser.add_argument("--last-fold", required=True, help="last fold to run, e.g. T4")
    return parser.parse_args()


def main() -> None:
    """Run the folds from --first-fold to --last-fold in time order."""
    args = parse_args()
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.run_dir("m6_run_folds") / "run.log")
    names = [fold.name for fold in post_lock_folds(load_fold_manifest(paths.fold_manifest_file))]
    for name in (args.first_fold, args.last_fold):
        if name not in names:
            raise ValueError(f"unknown post-lock fold {name!r}; expected one of {names}")
    first, last = names.index(args.first_fold), names.index(args.last_fold)
    if first > last:
        raise ValueError(f"--first-fold {args.first_fold} is after --last-fold {args.last_fold}")
    started = time.perf_counter()
    for fold in names[first : last + 1]:
        for step, script, arguments in fold_steps(fold, GENERATE_SCRIPT, MONITOR_SCRIPT):
            seconds = run_script(script, arguments)
            LOGGER.info("%s %s: finished in %.1f s", fold, step, seconds)
        summary = fold_summary(
            fold,
            read_rows(paths.m6_runs_file, RUN_COLUMNS),
            read_rows(paths.m6_candidates_results_file, MONITOR_COLUMNS),
        )
        LOGGER.info("FOLD SUMMARY | %s", summary)
    LOGGER.info(
        "folds %s to %s finished in %.0f s",
        args.first_fold,
        args.last_fold,
        time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
