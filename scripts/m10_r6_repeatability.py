"""CLI: check whether an R6 fine-tune is repeatable on this platform (M10 part 3).

Usage:
    python scripts/m10_r6_repeatability.py --setting none --rows 8192
    python scripts/m10_r6_repeatability.py --setting one_thread --rows 8192 --record

Runs the real fine-tuning code on SYNTHETIC data (drawn from a seed; no competition
data is read) twice, each time in a fresh process, and compares the two sets of
weights bit by bit (hmrec.calibration.r6_repeatability). --setting chooses the
runtime setting of the fine-tune (hmrec.recall.twotower.fine_tune_runtime);
--rows the size of the synthetic fine-tune set (configs/m10_r6_repeatability.yaml).
--record appends the result to results/m10_calibration/r6_repeatability.csv.

The script prints one summary line and ends with exit code 0 whatever the
outcome: it reports, it does not judge. (--worker is the mode it starts itself in.)
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from hmrec.calibration.r6_repeatability import (  # noqa: E402
    run_check,
    run_worker,
    summary_line,
    worker_exit_code,
)
from hmrec.logging_utils import setup_logging  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.recall.twotower.fine_tune_runtime import SETTINGS  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Check the repeatability of an R6 fine-tune.")
    parser.add_argument("--setting", required=True, choices=SETTINGS)
    parser.add_argument("--rows", required=True, type=int, help="rows of the synthetic set")
    parser.add_argument("--record", action="store_true", help="append the row to the table")
    parser.add_argument("--worker", action="store_true", help="internal: fine-tune once")
    parser.add_argument("--state-file", type=Path, default=None, help="internal: worker output")
    return parser.parse_args()


def main() -> None:
    """Run the check, or one worker."""
    args = parse_args()
    setup_logging(logging.INFO)
    paths = get_paths()
    if args.worker:
        try:
            run_worker(paths, args.setting, args.rows, args.state_file)
        except Exception as error:
            logging.getLogger("hmrec").exception("worker stopped")
            sys.exit(worker_exit_code(error))
        return
    row = run_check(paths, Path(__file__).resolve(), args.setting, args.rows, args.record)
    sys.stdout.write(summary_line(row) + "\n")


if __name__ == "__main__":
    main()
