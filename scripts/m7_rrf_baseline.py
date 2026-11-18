"""CLI for M7: the RRF baseline of one V fold (proposal §9.5).

Usage:
    uv run python scripts/m7_rrf_baseline.py --fold V1 --area v_phase

No model: every query's Top-12 of the same candidates by rrf_score DESC,
best_rank ASC, article_idx ASC, evaluated with the Ranker's metrics. Appends one
metric row (results/v_phase_metrics.csv, or results/m7_speed_test_metrics.csv
for --area speed_test) and one row to results/m7_runs.csv.

The fold must be a V fold: anything else is refused before a file is opened.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

from hmrec.hpo.v_phase import refuse_to_launch, start_m7_run  # noqa: E402
from hmrec.hpo.v_records import (  # noqa: E402
    AREAS,
    RRF_BASELINE,
    append_m7_run_row,
    append_metric_row,
    results_files,
)
from hmrec.hpo.v_trial import rrf_baseline_row  # noqa: E402
from hmrec.logging_utils import get_logger  # noqa: E402
from hmrec.memory import MemoryTracker, bytes_to_gb, log_process_peak_memory  # noqa: E402
from hmrec.ranker.runtime import read_lightgbm_runtime  # noqa: E402
from hmrec.run_records import (  # noqa: E402
    GENERATED,
    StepResult,
    check_memory_limit,
    run_row,
)

LOGGER = get_logger("m7_rrf_baseline")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, help="a V fold: V1 ... V4")
    parser.add_argument("--area", required=True, choices=AREAS)
    return parser.parse_args()


def main() -> None:
    """Evaluate the RRF baseline of one fold and record it."""
    started = time.perf_counter()
    args = parse_args()
    step = f"{RRF_BASELINE}_{args.area}"
    context, stamp = start_m7_run(f"m7_{args.fold}_{step}", refuse_to_launch)
    inputs = context.inputs
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(inputs.base.num_threads))
    label = f"m7 {args.fold} {step}"
    with MemoryTracker(
        label,
        LOGGER,
        limit_gb=inputs.base.memory.limit_gb,
        sample_interval_seconds=inputs.base.memory.sample_interval_seconds,
    ) as tracker:
        # rrf_baseline_row checks the fold's phase first: a T fold stops here.
        metric_row = rrf_baseline_row(inputs, args.fold)
    _, metrics_file = results_files(inputs.paths, args.area)
    finished_at = datetime.now().isoformat(timespec="seconds")
    append_metric_row(metrics_file, metric_row, inputs.metrics, stamp.git_commit, finished_at)
    LOGGER.info("RRF baseline %s: %s -> %s", args.fold, metric_row.scores, metrics_file)
    result = StepResult(
        fold=args.fold,
        step=step,
        status=GENERATED,
        n_rows=metric_row.n_queries,
        generate_seconds=tracker.elapsed_seconds,
        sampled_peak_rss_gb=bytes_to_gb(tracker.peak_rss_bytes),
        checked_rows=0,
        detail="n_rows = queries of the fold",
    )
    peak = log_process_peak_memory(LOGGER, label, inputs.base.memory.limit_gb)
    row = run_row(result, time.perf_counter() - started, peak, stamp)
    append_m7_run_row(inputs.paths.m7_runs_file, row, read_lightgbm_runtime())
    check_memory_limit(peak, inputs.base.memory.limit_gb, label)


if __name__ == "__main__":
    main()
