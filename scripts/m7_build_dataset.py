"""CLI for M7: build the binary LightGBM Dataset of one target fold (proposal §1.3, §9.2).

Usage:
    caffeinate -i uv run python scripts/m7_build_dataset.py --target V1 --ratio 40

Streams the sample files of the 4 weeks before the target fold through
lgb.Sequence (one row group at a time), bins them and writes
    artifacts/lgb_datasets/{experiment}/{target}_r{ratio}/train.bin  (+ dataset.json)
The samples must exist (scripts/m7_sample.py). A finished Dataset is skipped,
never replaced. Appends one row (time, true memory peak, LightGBM version, loaded
libomp) to results/m7_runs.csv.

The target must be a V fold and its training folds F / V folds: anything else is
refused before a file is opened. One Dataset per process.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402

from hmrec.hpo.v_phase import refuse_to_launch, start_m7_run  # noqa: E402
from hmrec.hpo.v_records import append_m7_run_row  # noqa: E402
from hmrec.hpo.v_trial import build_and_save_dataset, plan_dataset  # noqa: E402
from hmrec.logging_utils import get_logger  # noqa: E402
from hmrec.memory import MemoryTracker, bytes_to_gb, log_process_peak_memory  # noqa: E402
from hmrec.ranker.ablation import EXPERIMENTS, FULL  # noqa: E402
from hmrec.ranker.dataset import dataset_is_built, read_dataset_record  # noqa: E402
from hmrec.ranker.runtime import read_lightgbm_runtime  # noqa: E402
from hmrec.run_records import (  # noqa: E402
    GENERATED,
    SKIPPED,
    StepResult,
    check_memory_limit,
    run_row,
)

LOGGER = get_logger("m7_build_dataset")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="the fold to predict: V1 ... V4")
    parser.add_argument("--ratio", required=True, type=int, help="negatives per positive")
    parser.add_argument("--experiment", default=FULL, choices=EXPERIMENTS)
    return parser.parse_args()


def main() -> None:
    """Build one Dataset and record the run."""
    started = time.perf_counter()
    args = parse_args()
    step = f"dataset_{args.experiment}_r{args.ratio}"
    context, stamp = start_m7_run(f"m7_{args.target}_{step}", refuse_to_launch)
    inputs = context.inputs
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(inputs.base.num_threads))
    # plan_dataset checks the folds first: a T fold stops here.
    plan = plan_dataset(inputs, args.experiment, args.target, args.ratio, stamp.git_commit)
    already_built = dataset_is_built(plan.directory)
    label = f"m7 {args.target} {step}"
    with MemoryTracker(
        label,
        LOGGER,
        limit_gb=inputs.base.memory.limit_gb,
        sample_interval_seconds=inputs.base.memory.sample_interval_seconds,
    ) as tracker:
        if already_built:
            record = read_dataset_record(plan.directory)
        else:
            record = build_and_save_dataset(plan, inputs.ranker.sampling.row_group_rows)
    LOGGER.info(
        "%s: %d rows, %d queries, folds %s -> %s",
        "skipped (exists)" if already_built else "built",
        record.n_rows,
        record.n_queries,
        list(record.train_folds),
        plan.directory,
    )
    result = StepResult(
        fold=args.target,
        step=step,
        status=SKIPPED if already_built else GENERATED,
        n_rows=record.n_rows,
        generate_seconds=0.0 if already_built else tracker.elapsed_seconds,
        sampled_peak_rss_gb=bytes_to_gb(tracker.peak_rss_bytes),
        checked_rows=0,
        detail=f"{record.n_queries} queries; train folds {' '.join(record.train_folds)}",
    )
    peak = log_process_peak_memory(LOGGER, label, inputs.base.memory.limit_gb)
    row = run_row(result, time.perf_counter() - started, peak, stamp)
    append_m7_run_row(inputs.paths.m7_runs_file, row, read_lightgbm_runtime())
    LOGGER.info("run row -> %s: %s", inputs.paths.m7_runs_file, row)
    check_memory_limit(peak, inputs.base.memory.limit_gb, label)


if __name__ == "__main__":
    main()
