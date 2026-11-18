"""CLI for the Kaggle late submission: the wide feature table of a run.

    uv run python scripts/kaggle_features.py --run rehearsal --prepare
    uv run python scripts/kaggle_features.py --run rehearsal --shard 0

--prepare computes the tables over ALL customers once (the article table, the
global product-type shares). --shard builds the 71 locked features of one shard
of customers from the shard's candidate files (hmrec.kaggle.features). Reads no
label. One step per process.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402

from hmrec.kaggle.context import ALL_SHARDS, DONE, SKIPPED, finish_process, start_process  # noqa: E402
from hmrec.kaggle.features import (  # noqa: E402
    build_shard_features,
    load_locked_features_config,
    prepare_shared_tables,
)
from hmrec.kaggle.plan import RUNS  # noqa: E402

PREPARE_STEP = "features_prepare"
SHARD_STEP = "features"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=RUNS)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--prepare", action="store_true", help="the tables over all customers")
    what.add_argument("--shard", type=int, help="0-based shard number")
    return parser.parse_args()


def main() -> None:
    """Run one feature step and record it."""
    started = time.perf_counter()
    args = parse_args()
    label = PREPARE_STEP if args.prepare else f"{SHARD_STEP}_{args.shard:03d}"
    context = start_process(args.run, label)
    config = load_locked_features_config(context)
    if args.prepare:
        tables = prepare_shared_tables(context, config)
        detail = f"{tables.global_type_share.height} product types"
        finish_process(context, PREPARE_STEP, ALL_SHARDS, DONE, tables.item.height, started, detail)
        return
    result = build_shard_features(context, config, args.shard)
    detail = (
        f"{result.n_parts} parts, {result.n_queries} queries; tt_dot = R6 score on "
        f"{result.tt_dot_rows_checked} rows, wals_dot = R5 score on "
        f"{result.wals_dot_rows_checked} rows"
    )
    status = SKIPPED if result.skipped else DONE
    finish_process(context, SHARD_STEP, str(args.shard), status, result.n_rows, started, detail)


if __name__ == "__main__":
    main()
