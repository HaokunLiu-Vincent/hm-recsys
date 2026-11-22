"""CLI: compare the weeks the replay closed with their stage-1 records (docs/mlops.md §8.3).

Usage, after one or more ticks:
    make image-replay-compare        in a container of its own (see the Makefile for the mounts)
    make replay-compare              on this machine (needs a clean working tree)

For every `eval` week the pipeline has closed and that is not compared yet, the
rows go to results/m12_replay/comparison.csv: the acceptance quantities (recall
of each channel, union recall, MAP@12; relative difference against the record,
inside or outside the threshold of the platform record), the two structural
checks, and the reported quantities (pair overlap of the candidates, the
deduplicated candidate count, the RRF baseline). When every week of the replay is
closed, the pooled MAP@12 is compared too. The chain records of the R6
fine-tunes are copied to results/m12_replay/twotower_chain.csv, and the `serve`
tick's Top-12 is compared with the saved Kaggle prediction set
(results/m12_replay/serve_overlap.csv; reported only).

No label is read: the pipeline's numbers come from what its step C2 measured, the
records from the stage-1 result tables. Nothing is selected or tuned.

A replay in the cloud (M13) is compared on the laptop from a copy of the SMALL
records of its state (`bash infra/operate.sh download-records`): manifests, step
records, the metrics of step C2. The large files stay in the bucket, so the pair
overlap of the candidates and the `serve` overlap are not measured:

    HMREC_PIPELINE_STATE_ROOT=<the copy> uv run python scripts/m12_replay_compare.py \
        --platform linux-amd64 --without-candidates

The comparison table is then {state}/results/comparison.csv (hmrec.paths.tick_root_paths).
--calibration reads the copy of a calibration state ("{state folder}-calibration").

Exit code 1 if an acceptance quantity is outside its threshold or a structural
check failed: the milestone stops and the difference is investigated.
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

from hmrec.config import load_pipeline_config  # noqa: E402
from hmrec.data.folds import load_fold_manifest  # noqa: E402
from hmrec.git_state import git_state  # noqa: E402
from hmrec.locks import load_ranker_lock  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import get_paths, tick_root_paths  # noqa: E402
from hmrec.platform_records import load_platform_record  # noqa: E402
from hmrec.weekly.replay_compare import compare_replay  # noqa: E402

LOGGER = get_logger("scripts.m12_replay_compare")
# The platform whose record holds the acceptance threshold when none is named (the arm64
# container on the laptop, the replay of M12; docs/mlops.md §8.3).
REPLAY_PLATFORM = "linux-arm64"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare closed weeks with stage-1 records.")
    parser.add_argument(
        "--platform",
        default=REPLAY_PLATFORM,
        help="the platform record whose acceptance threshold is used (configs/platforms/)",
    )
    parser.add_argument(
        "--without-candidates",
        action="store_true",
        help="the state holds only small records: leave out pair overlap and serve overlap",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="compare the calibration state instead of the state",
    )
    return parser.parse_args()


def main() -> None:
    """Run the comparison and report what is outside its criterion."""
    args = parse_args()
    setup_logging(logging.INFO)
    paths = tick_root_paths(get_paths(), os.environ, args.calibration)
    commit, dirty = git_state(paths.project_root, paths.results_root)
    if dirty:
        raise RuntimeError(
            f"the working tree differs from commit {commit[:12]}: the comparison reads "
            "full-data outputs and runs only from committed code"
        )
    manifest = load_fold_manifest(paths.fold_manifest_file)
    config = load_pipeline_config(paths.pipeline_config_file)
    record = load_platform_record(paths.platform_record_file(args.platform))
    result = compare_replay(
        paths,
        [manifest.get(name) for name in config.replay_eval_folds],
        config.replay_serve_anchor,
        load_ranker_lock(paths.ranker_lock_file),
        record.acceptance.max_relative_difference,
        commit,
        with_candidates=not args.without_candidates,
    )
    LOGGER.info(
        "%d of %d weeks closed; compared now: %s; %d chain record(s) copied; table %s",
        len(result.closed),
        result.n_weeks,
        ", ".join(result.compared_now) or "none",
        result.chain_records_copied,
        paths.replay_table_file("comparison"),
    )
    if result.serve_row is not None:
        LOGGER.info("serve tick against the Kaggle prediction set (reported): %s", result.serve_row)
    for week_name, comparison in result.failed:
        LOGGER.error(
            "%s: %s %s %s is OUTSIDE: value %r, record %r, relative difference %.4f%%",
            week_name,
            comparison.kind,
            comparison.quantity,
            comparison.subject,
            comparison.value,
            comparison.record,
            100 * comparison.relative_difference,
        )
    if result.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
