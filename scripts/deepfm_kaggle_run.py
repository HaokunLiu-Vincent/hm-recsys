"""CLI for the DeepFM Kaggle submission: every step of a run, each in its own process.

    uv run python scripts/deepfm_kaggle_run.py --run rehearsal
    uv run python scripts/deepfm_kaggle_run.py --run submission

Steps, in order (scripts/deepfm_kaggle_step.py; a finished step is skipped):
    train, score, submission, and for the rehearsal: check.

A step that fails (a failed check, a NaN loss, a memory peak above the limit)
stops the run. The real run starts only after a passed rehearsal with the same
code. Nothing is uploaded: the submission itself is made once, by hand.
"""

import argparse
import logging
from pathlib import Path

from hmrec.kaggle.plan import RUN_REHEARSAL, RUNS
from hmrec.kaggle.run import run_script
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.paths import get_paths

LOGGER = get_logger("deepfm_kaggle_run")
STEP_SCRIPT = Path(__file__).with_name("deepfm_kaggle_step.py")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=RUNS)
    return parser.parse_args()


def main() -> None:
    """Run the steps of one run in order."""
    run = parse_args().run
    setup_logging(logging.INFO, log_file=get_paths().kaggle_deepfm_log_file(run, "driver"))
    steps = ["train", "score", "submission"]
    if run == RUN_REHEARSAL:
        steps.append("check")
    total = 0.0
    for step in steps:
        seconds = run_script(STEP_SCRIPT, ["--run", run, "--step", step])
        LOGGER.info("finished in %.1f s: %s %s", seconds, run, step)
        total += seconds
    LOGGER.info("%s run finished in %.1f s", run, total)


if __name__ == "__main__":
    main()
