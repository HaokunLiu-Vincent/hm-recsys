"""CLI for the Kaggle late submission: the first step of a run (docs/decisions.md 2026-11-19).

    uv run python scripts/kaggle_prepare.py --run rehearsal

Checks the free disk space and that sample_submission.csv holds exactly the
customers of the customer table, then writes the run's query list
(artifacts/kaggle/{run}/queries.parquet). The real run (--run submission) starts
only after a passed rehearsal of the same code. Started by scripts/kaggle_run.py.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402

from hmrec.kaggle.context import ALL_SHARDS, DONE, finish_process, start_process  # noqa: E402
from hmrec.kaggle.plan import RUNS, number_of_shards  # noqa: E402
from hmrec.kaggle.prepare import prepare_run  # noqa: E402

STEP = "prepare"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=RUNS)
    return parser.parse_args()


def main() -> None:
    """Prepare one run and record the step."""
    started = time.perf_counter()
    context = start_process(parse_args().run, STEP)
    queries = prepare_run(context)
    n_shards = number_of_shards(queries.height, context.config.shard_customers)
    detail = f"{queries.height} customers, {n_shards} shards, week {context.target.name}"
    finish_process(context, STEP, ALL_SHARDS, DONE, queries.height, started, detail)


if __name__ == "__main__":
    main()
