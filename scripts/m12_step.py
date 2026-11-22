"""CLI: ONE step of a tick of the weekly pipeline, in its own process (docs/mlops.md §6.1).

Started by the driver (scripts/weekly.py), one process per step, so that a
step's process max RSS is its true memory peak and torch and LightGBM never
share a process. It is not meant to be called by hand, but it can be:

    python scripts/m12_step.py --anchor 2020-07-29 --mode eval \
        --run-id 2020-07-29-eval-a1 --step O1_recall_r4_itemcf

The step writes into its staging folder under artifacts/pipeline/weeks/; the
driver renames the folder when the step ended without an error.
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
from datetime import date  # noqa: E402

from hmrec.run_manifest import MODES  # noqa: E402
from hmrec.weekly.run_step import run_one_step  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="One step of a tick of the weekly pipeline.")
    parser.add_argument("--anchor", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--step", required=True, help="a step name, e.g. C1_ingest")
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="a step of a calibration tick (the driver passes it on)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the step."""
    args = parse_args()
    run_one_step(args.anchor, args.mode, args.run_id, args.step, args.calibration)


if __name__ == "__main__":
    main()
