"""CLI for M10: print what this process runs on (docs/mlops.md §8).

    uv run python scripts/image_info.py --side torch
    uv run python scripts/image_info.py --side lightgbm

Prints the commit of the container image (if this is one), the platform and
whether it is the reference platform, Python, the CPUs and memory the process
can use, package versions, the OpenMP runtimes and BLAS libraries that are
really loaded, and the thread settings. One "name: value" line per fact.

torch and LightGBM never share a process (AGENTS.md §5), so each side is its own
run; `make image-info` runs both inside the image. Reads no data and writes no file.
"""

import argparse
import os
import sys

from hmrec.platform_report import SIDES, report_lines


def parse_arguments() -> argparse.Namespace:
    """Parse --side."""
    parser = argparse.ArgumentParser(description="Print what this process runs on.")
    parser.add_argument("--side", choices=SIDES, required=True)
    return parser.parse_args()


def main() -> None:
    """Write the report of one side to standard output."""
    arguments = parse_arguments()
    # Standard output on purpose, not the logger: the report is the result of this
    # command, and `make image-info` shows it without log prefixes.
    sys.stdout.write("\n".join(report_lines(arguments.side, os.environ)) + "\n")


if __name__ == "__main__":
    main()
