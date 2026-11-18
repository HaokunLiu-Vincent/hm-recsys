"""CLI for M7: ONE Ranker trial = one configuration on one V fold (proposal §9.2-9.4).

Usage (normally started by scripts/m7_ranker_hpo.py or scripts/m7_speed_test.py):
    caffeinate -i uv run python scripts/m7_train_eval.py --area v_phase --experiment full \
        --step leaves_x_min_data --target V1 --num-leaves 31 --min-data-in-leaf 500 \
        --ratio 40 --num-trees 2000 --histogram-mode col_wise

Loads the binary Dataset of the 4 weeks before the target fold, trains from
scratch to --num-trees trees, then scores ALL candidates of the target fold and
evaluates the Top-12 after every checkpoint (hmrec.hpo.v_trial). With --no-eval
it only trains and times (no label of the target fold is read).

Writes, in this order: the Top-12 and per-query files, the model, the metric rows
(results/v_phase_metrics.csv) and LAST the trial row (results/v_phase.csv) with
the true memory peak of this process, the LightGBM version and the libomp it
loaded. The speed test (--area speed_test) writes
to results/m7_speed_test*.csv instead.

The target must be a V fold: anything else is refused before a file is opened.
One trial per process, so the process max RSS is the trial's own peak.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from datetime import datetime  # noqa: E402

from hmrec.config import HISTOGRAM_COL_WISE, HISTOGRAM_ROW_WISE, RankerTunedParams  # noqa: E402
from hmrec.hpo.v_phase import PHASE, refuse_to_launch, start_m7_run  # noqa: E402
from hmrec.hpo.v_records import (  # noqa: E402
    AREAS,
    RankerTrialRecord,
    append_metric_row,
    append_trial,
    results_files,
)
from hmrec.hpo.v_trial import TrialSpec, run_trial  # noqa: E402
from hmrec.logging_utils import get_logger  # noqa: E402
from hmrec.memory import MemoryTracker, bytes_to_gb, log_process_peak_memory  # noqa: E402
from hmrec.ranker.ablation import EXPERIMENTS  # noqa: E402
from hmrec.ranker.runtime import read_lightgbm_runtime  # noqa: E402
from hmrec.run_records import check_memory_limit  # noqa: E402

LOGGER = get_logger("m7_train_eval")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, choices=AREAS)
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    parser.add_argument("--step", required=True, help="step name written to the trial row")
    parser.add_argument("--target", required=True, help="the fold to predict: V1 ... V4")
    parser.add_argument("--num-leaves", required=True, type=int)
    parser.add_argument("--min-data-in-leaf", required=True, type=int)
    parser.add_argument("--ratio", required=True, type=int, help="negatives per positive")
    parser.add_argument("--num-trees", required=True, type=int)
    parser.add_argument(
        "--histogram-mode", required=True, choices=(HISTOGRAM_COL_WISE, HISTOGRAM_ROW_WISE)
    )
    parser.add_argument("--no-eval", action="store_true", help="train and time only")
    return parser.parse_args()


def main() -> None:
    """Run one trial and record it."""
    started = time.perf_counter()
    args = parse_args()
    tuned = RankerTunedParams(args.num_leaves, args.min_data_in_leaf, args.ratio)
    spec = TrialSpec(
        area=args.area,
        experiment=args.experiment,
        step=args.step,
        target_fold=args.target,
        tuned=tuned,
        num_trees=args.num_trees,
        histogram_mode=args.histogram_mode,
        evaluate=not args.no_eval,
    )
    run_name = (
        f"m7_{args.area}_{args.experiment}_{args.step}_{args.target}_"
        f"l{args.num_leaves}_m{args.min_data_in_leaf}_r{args.ratio}_{args.histogram_mode}"
    )
    context, stamp = start_m7_run(run_name, refuse_to_launch)
    inputs = context.inputs
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(inputs.base.num_threads))
    label = f"m7 trial {args.step} {args.target}"
    with MemoryTracker(
        label,
        LOGGER,
        limit_gb=inputs.base.memory.limit_gb,
        sample_interval_seconds=inputs.base.memory.sample_interval_seconds,
    ) as tracker:
        # run_trial checks the target fold first: a T fold stops here.
        outcome = run_trial(inputs, spec, stamp.git_commit)
    # Last step of the run: the true peak of the whole process.
    peak = log_process_peak_memory(LOGGER, label, inputs.base.memory.limit_gb)
    finished_at = datetime.now().isoformat(timespec="seconds")
    trial_file, metrics_file = results_files(inputs.paths, args.area)
    for metric_row in outcome.metric_rows:
        append_metric_row(metrics_file, metric_row, inputs.metrics, stamp.git_commit, finished_at)
        LOGGER.info("%d trees: %s", metric_row.num_trees, metric_row.scores)
    record = RankerTrialRecord(
        phase=PHASE,
        experiment=args.experiment,
        step=args.step,
        config_id=outcome.config_id,
        fold=args.target,
        train_folds=outcome.train_folds,
        params=asdict(tuned),
        config=outcome.config,
        histogram_mode=args.histogram_mode,
        num_trees=args.num_trees,
        evaluated=spec.evaluate,
        n_train_rows=outcome.n_train_rows,
        n_train_queries=outcome.n_train_queries,
        n_scored_rows=outcome.n_scored_rows,
        n_queries=outcome.n_queries,
        load_seconds=outcome.load_seconds,
        train_seconds=outcome.train_seconds,
        seconds_per_tree=outcome.seconds_per_tree,
        score_seconds=outcome.score_seconds,
        total_seconds=time.perf_counter() - started,
        sampled_peak_rss_gb=bytes_to_gb(tracker.peak_rss_bytes),
        process_max_rss_gb=peak.process_max_rss_gb,
        model_sha256=outcome.model_sha256,
        runtime=read_lightgbm_runtime(),
        git_commit=stamp.git_commit,
        git_dirty=stamp.git_dirty,
        finished_at=finished_at,
    )
    append_trial(trial_file, record)
    LOGGER.info("trial row -> %s (config %s)", trial_file, outcome.config_id)
    check_memory_limit(peak, inputs.base.memory.limit_gb, label)


if __name__ == "__main__":
    main()
