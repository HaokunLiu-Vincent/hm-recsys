"""CLI for M7: draw the Ranker's training samples of one fold (proposal §9.1).

Usage:
    caffeinate -i uv run python scripts/m7_sample.py --fold F1
    caffeinate -i uv run python scripts/m7_sample.py --fold F1 --experiment minus_r6 --ratios 40

Reads the fold's frozen wide table (verified first) and its labels, keeps the
queries with at least one positive candidate, and writes one sample file per
negatives-per-positive ratio:
    artifacts/ranker/samples/{experiment}/{fold}/r{ratio}.parquet  (+ .json record)
A finished sample is skipped, never replaced. Appends one row (time, true memory
peak, LightGBM version, loaded libomp) to results/m7_runs.csv.

M7 samples F and V folds only: any other fold is refused before a file is opened.
Run it on a clean working tree (AGENTS.md §3.4). One fold per process.
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import time  # noqa: E402

from hmrec.hpo.v_phase import refuse_to_launch, sample_job, start_m7_run  # noqa: E402
from hmrec.hpo.v_records import append_m7_run_row  # noqa: E402
from hmrec.logging_utils import get_logger  # noqa: E402
from hmrec.memory import MemoryTracker, bytes_to_gb, log_process_peak_memory  # noqa: E402
from hmrec.ranker.ablation import EXPERIMENTS, FULL  # noqa: E402
from hmrec.ranker.runtime import read_lightgbm_runtime  # noqa: E402
from hmrec.ranker.sampling import pending_ratios, sample_fold  # noqa: E402
from hmrec.run_records import (  # noqa: E402
    GENERATED,
    SKIPPED,
    StepResult,
    check_memory_limit,
    run_row,
)

LOGGER = get_logger("m7_sample")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, help="a training fold of M7: F1 ... F4, V1 ... V3")
    parser.add_argument("--experiment", default=FULL, choices=EXPERIMENTS)
    parser.add_argument(
        "--ratios", type=int, nargs="+", help="default: sampling.ratios of configs/ranker.yaml"
    )
    return parser.parse_args()


def main() -> None:
    """Sample one fold and record the run."""
    started = time.perf_counter()
    args = parse_args()
    step = f"sample_{args.experiment}"
    context, stamp = start_m7_run(f"m7_{args.fold}_{step}", refuse_to_launch)
    inputs = context.inputs
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(inputs.base.num_threads))
    ratios = args.ratios or list(inputs.ranker.sampling.ratios)
    # sample_job checks the fold's phase first: a T fold stops here.
    job = sample_job(inputs, args.fold, args.experiment, ratios, stamp.git_commit)
    written = pending_ratios(job.sample_files)
    label = f"m7 {args.fold} {step}"
    with MemoryTracker(
        label,
        LOGGER,
        limit_gb=inputs.base.memory.limit_gb,
        sample_interval_seconds=inputs.base.memory.sample_interval_seconds,
    ) as tracker:
        records = sample_fold(job, inputs.ranker.sampling)
    for ratio, record in records.items():
        LOGGER.info(
            "r=%d: %d rows, %d queries, %d positives (%.4f%%), rows per query mean %.1f",
            ratio,
            record.n_rows,
            record.n_queries,
            record.n_positives,
            100 * record.positive_share,
            record.rows_per_query_mean,
        )
    result = StepResult(
        fold=args.fold,
        step=step,
        status=GENERATED if written else SKIPPED,
        n_rows=sum(records[ratio].n_rows for ratio in written),
        generate_seconds=tracker.elapsed_seconds if written else 0.0,
        sampled_peak_rss_gb=bytes_to_gb(tracker.peak_rss_bytes),
        checked_rows=0,
        detail=f"ratios written: {written}" if written else "",
    )
    # Last step of the run: the true peak of the whole process, into the log and the row.
    peak = log_process_peak_memory(LOGGER, label, inputs.base.memory.limit_gb)
    row = run_row(result, time.perf_counter() - started, peak, stamp)
    append_m7_run_row(inputs.paths.m7_runs_file, row, read_lightgbm_runtime())
    LOGGER.info("run row -> %s: %s", inputs.paths.m7_runs_file, row)
    check_memory_limit(peak, inputs.base.memory.limit_gb, label)


if __name__ == "__main__":
    main()
