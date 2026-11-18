"""CLI for the Kaggle late submission: run every step of a run, one process per step.

    caffeinate -i uv run python scripts/kaggle_run.py --run rehearsal
    caffeinate -i uv run python scripts/kaggle_run.py --run submission

Order and steps: hmrec.kaggle.run. Run it on a clean working tree; the real run
also needs a passed rehearsal of the same code. A failed step stops the run; the
same command continues it (finished files are skipped, never replaced). This
script uploads nothing.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

from hmrec.config import load_kaggle_config  # noqa: E402
from hmrec.kaggle.plan import RUNS, number_of_shards, read_queries  # noqa: E402
from hmrec.kaggle.run import run_steps, steps_after_recall, steps_before_shards  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402

LOGGER = get_logger("kaggle_run")
SCRIPTS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=RUNS)
    return parser.parse_args()


def main() -> None:
    """Run all steps of one run in order."""
    run = parse_args().run
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.kaggle_log_file(run, "driver"))
    seconds = run_steps(SCRIPTS_DIR, run, steps_before_shards(run))
    config = load_kaggle_config(paths.kaggle_config_file)
    queries = read_queries(paths.kaggle_queries_file(run))
    n_shards = number_of_shards(queries.height, config.shard_customers)
    seconds += run_steps(SCRIPTS_DIR, run, steps_after_recall(run, n_shards))
    LOGGER.info("kaggle %s: all steps finished in %.0f s", run, seconds)


if __name__ == "__main__":
    main()
