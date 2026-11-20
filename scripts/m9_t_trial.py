"""CLI for M9: ONE training of the evaluation on T1-T4 (proposal §16.12).

    uv run python scripts/m9_t_trial.py --stage rehearsal --seed 42 --target V4
    uv run python scripts/m9_t_trial.py --stage formal --seed 43 --target T2

Normally started by scripts/m9_t_eval.py. Trains the DeepFM configuration chosen
on V (configs/m9_t_eval.yaml) for one target week, scores all candidates of the
week and evaluates the Top-12. Appends rows to results/m9_t_metrics.csv and, LAST,
to results/m9_t_trials.csv; results/deepfm_*.csv are not written.

--stage rehearsal predicts only the configured V fold and reads no T label.
--stage formal predicts a T fold: it runs only if the rehearsal passed at this
commit and the driver recorded the read of T labels (hmrec.t_eval.gate); the
permit is taken before any file is opened. One training per process; clean tree
only.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402

from hmrec.deepfm.process import finish, record_trial, start_process  # noqa: E402
from hmrec.deepfm.trial import check_scores_are_finite  # noqa: E402
from hmrec.logging_utils import get_logger  # noqa: E402
from hmrec.t_eval.gate import worker_permit  # noqa: E402
from hmrec.t_eval.plan import (  # noqa: E402
    STAGE_FORMAL,
    STAGES,
    TrialKey,
    check_trial_key,
    load_t_eval_context,
)
from hmrec.t_eval.trial import run_t_eval_trial  # noqa: E402

LOGGER = get_logger("m9_t_trial")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--seed", required=True, type=int, help="the training seed")
    parser.add_argument("--target", required=True, help="the fold to predict")
    return parser.parse_args()


def main() -> None:
    """Run one training and record it."""
    started = time.perf_counter()
    args = parse_args()
    label = f"m9_t_{args.stage}_{args.target}_seed{args.seed}"
    # start_process refuses a working tree with uncommitted code or config changes.
    inputs, stamp = start_process(f"{label}_{time.strftime('%Y%m%dT%H%M%S')}")
    context = load_t_eval_context(inputs.paths, stamp)
    key = TrialKey(args.stage, args.seed, args.target)
    check_trial_key(context, key)
    # The gate, before anything is read: no permit, no training on or of a T fold.
    permit = worker_permit(context) if key.stage == STAGE_FORMAL else None
    chosen, spec, result = run_t_eval_trial(context, inputs, key, permit)
    peak = finish(chosen, label)
    paths = chosen.paths
    record_trial(
        chosen, spec, result, started, peak, stamp, paths.m9_t_trials_file, paths.m9_t_metrics_file
    )
    check_scores_are_finite(result)
    metric = chosen.config.metric
    last = result.metrics[-1]
    LOGGER.info(
        "%s | %s seed %d, %d epochs: %s = %.6f",
        key.stage,
        key.target_fold,
        key.seed,
        last.epoch,
        metric,
        last.scores[metric],
    )


if __name__ == "__main__":
    main()
