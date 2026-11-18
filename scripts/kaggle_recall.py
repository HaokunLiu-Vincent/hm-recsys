"""CLI for the Kaggle late submission: ONE recall channel for all shards of a run.

    uv run python scripts/kaggle_recall.py --run rehearsal --channel r4_itemcf

The channel is built from the lock files only, fitted once with all history
before the predicted week's anchor, and then writes the candidates of every
shard (hmrec.kaggle.recall). For --run submission, r6_twotower first fine-tunes
the locked chain's last model once on the T4 week and records the new model's
sha256 in results/kaggle_twotower.csv. Reads no label. One channel per process.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402

from hmrec.kaggle.context import ALL_SHARDS, DONE, SKIPPED, finish_process, start_process  # noqa: E402
from hmrec.kaggle.plan import RUNS  # noqa: E402
from hmrec.kaggle.recall import generate_channel, load_recall_job  # noqa: E402
from hmrec.locks import ALL_CHANNELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=RUNS)
    parser.add_argument("--channel", required=True, choices=ALL_CHANNELS)
    return parser.parse_args()


def main() -> None:
    """Generate one channel for every shard of the run and record the step."""
    started = time.perf_counter()
    args = parse_args()
    step = f"recall_{args.channel}"
    context = start_process(args.run, step)
    result = generate_channel(load_recall_job(context), args.channel, context.stamp)
    status = DONE if result.n_generated > 0 else SKIPPED
    detail = (
        f"{result.n_generated} of {result.n_shards} shards written, "
        f"{result.checked_rows} rows checked"
    )
    if result.detail:
        detail = f"{detail}; {result.detail}"
    finish_process(context, step, ALL_SHARDS, status, result.n_rows, started, detail)


if __name__ == "__main__":
    main()
