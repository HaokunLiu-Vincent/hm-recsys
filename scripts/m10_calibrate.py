"""CLI for M10: run ONE calibration item of week V4 in a container (docs/mlops.md §8.3).

Usage, inside the image (docs/m10_calibration_plan.md §2; `make image-calibrate` adds the mounts):
    python scripts/m10_calibrate.py --run run1 --item recall --channel r1_popularity
    python scripts/m10_calibrate.py --run run1 --item recall --channel r5_wals
    python scripts/m10_calibrate.py --run run1 --item r6-predict --start recorded
    python scripts/m10_calibrate.py --run run1 --item r6-fine-tune
    python scripts/m10_calibrate.py --run run1 --item r6-predict --start calibration
    python scripts/m10_calibrate.py --run run1 --item union
    python scripts/m10_calibrate.py --run run1 --item ranker-dataset
    python scripts/m10_calibrate.py --run run1 --item ranker
    python scripts/m10_calibrate.py --run run2 --item repeat-ranker

One item per process. Each item runs the stage-1 functions with the locked
configuration, writes ONLY under artifacts/m10_calibration/ and
results/m10_calibration/, and compares its output with the stage-1 record of the
week. Labels of the week (a V fold) are read for recall and MAP@12; no label of
a T week is read, and nothing is selected or tuned.

Run it from an image built from a clean, committed tree (`make image-build`).
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)

from hmrec.calibration.run import ITEMS, run_calibration_item  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run one item of the M10 calibration.")
    parser.add_argument("--run", required=True, help="name of the container run: run1 or run2")
    parser.add_argument("--item", required=True, choices=ITEMS)
    parser.add_argument(
        "--channel", default=None, help="for --item recall: r1_popularity ... r5_wals"
    )
    parser.add_argument(
        "--start", default=None, help="for --item r6-predict: recorded or calibration"
    )
    return parser.parse_args()


def main() -> None:
    """Run the item."""
    args = parse_args()
    run_calibration_item(args.item, args.run, args.channel, args.start)


if __name__ == "__main__":
    main()
