"""CLI for M9: ONE training of a LightGBM arm on one V fold (proposal §16.8).

    uv run python scripts/m9_lgb_trial.py --arm lambdarank --seed 43 --target V1 --num-trees 500
    uv run python scripts/m9_lgb_trial.py --arm binary --seed 42 --target V1 --num-trees 1000 \
        --evaluate-all

Normally started by scripts/m9_lgb_comparison.py. Trains the locked Ranker
configuration (configs/locks/ranker.lock.yaml) with the arm's objective and seed
on the binary Dataset M7 built for the target fold, scores all candidates of the
fold and evaluates the Top-12 (hmrec.hpo.lgb_arms). Appends rows to
results/m9_lgb_metrics.csv and, LAST, to results/m9_lgb_trials.csv; nothing is
written to results/v_phase*.csv.

The target must be a V fold: anything else is refused before a file is opened.
One training per process, so the process max RSS is its own peak. The working
tree must be clean (the run reads the full data).
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

from hmrec.config import load_deepfm_config  # noqa: E402
from hmrec.git_state import git_state  # noqa: E402
from hmrec.hpo.lgb_arm_records import ARMS, append_metric_row, append_trial_row  # noqa: E402
from hmrec.hpo.lgb_arms import (  # noqa: E402
    ArmSpec,
    load_arm_context,
    metric_rows,
    run_arm_trial,
    trial_row,
)
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import log_process_peak_memory  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.ranker.runtime import read_lightgbm_runtime  # noqa: E402
from hmrec.run_records import check_memory_limit  # noqa: E402

LOGGER = get_logger("m9_lgb_trial")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int, help="the LightGBM seed")
    parser.add_argument("--target", required=True, help="the fold to predict: V1 ... V4")
    parser.add_argument("--num-trees", required=True, type=int)
    parser.add_argument(
        "--evaluate-all",
        action="store_true",
        help="evaluate every checkpoint (default: only the last one)",
    )
    return parser.parse_args()


def main() -> None:
    """Run one training and record it."""
    started = time.perf_counter()
    args = parse_args()
    paths = get_paths()
    label = f"m9_lgb_{args.arm}_s{args.seed}_{args.target}"
    setup_logging(log_file=paths.run_dir(label) / "run.log")
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("%s | git HEAD %s, uncommitted changes: %s", label, commit, dirty)
    if dirty:
        raise SystemExit("the working tree has uncommitted changes: the training does not start")
    # Fails here if a second OpenMP runtime (e.g. torch's) is loaded in this process.
    runtime = read_lightgbm_runtime()
    # The metrics of the comparison are the ones of the DeepFM arms (configs/deepfm.yaml).
    metrics = load_deepfm_config(paths.deepfm_config_file).metrics
    context = load_arm_context(paths, runtime, metrics)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(context.base.num_threads))
    every = context.locked.config.checkpoint_every
    evaluated = (
        tuple(range(every, args.num_trees + 1, every)) if args.evaluate_all else (args.num_trees,)
    )
    spec = ArmSpec(args.arm, args.seed, args.target, args.num_trees, evaluated)
    # run_arm_trial checks the target fold first: a T fold stops here.
    outcome = run_arm_trial(context, spec, commit)
    # Last step of the run: the true peak of the whole process.
    peak = log_process_peak_memory(LOGGER, label, context.base.memory.limit_gb)
    finished_at = datetime.now().isoformat(timespec="seconds")
    for row in metric_rows(spec, outcome, metrics, commit, finished_at):
        append_metric_row(paths.m9_lgb_metrics_file, metrics, row)
        LOGGER.info("%s trees: %s = %s", row["num_trees"], metrics[0], row[metrics[0]])
    measured = {
        "total_seconds": time.perf_counter() - started,
        "process_max_rss_gb": peak.process_max_rss_gb,
    }
    stamp = {"git_commit": commit, "git_dirty": str(dirty), "finished_at": finished_at}
    append_trial_row(
        paths.m9_lgb_trials_file, trial_row(context, spec, outcome, runtime, measured, stamp)
    )
    LOGGER.info("trial row -> %s (config %s)", paths.m9_lgb_trials_file, outcome.config_id)
    check_memory_limit(peak, context.base.memory.limit_gb, label)


if __name__ == "__main__":
    main()
