"""CLI for the DeepFM Kaggle submission: ONE step of a run (torch side, no LightGBM).

    uv run python scripts/deepfm_kaggle_step.py --run rehearsal --step train

Steps (hmrec.kaggle_deepfm), with the DeepFM configuration chosen on V and seed 42:
    train        check the sample files of the run's LightGBM ranker, train the model
                 from scratch and save it. Reads the label column of the sample
                 files as TRAINING labels; a row is appended to
                 results/kaggle_deepfm_label_access.csv first.
    score        score every shard of the run and write its Top-12
    submission   write the submission file and make the nine format checks
    check        the rehearsal only: compare with the stage-2 evaluation (training
                 loss, Top-12 lists, file format), record the outcome and exit with
                 an error if it did not pass

The real run (--run submission) starts a step only if the last rehearsal passed
with this code. One step per process; clean working tree only. Nothing is
uploaded here.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402

from hmrec.git_state import code_changed_since  # noqa: E402
from hmrec.kaggle.plan import RUN_SUBMISSION, RUNS  # noqa: E402
from hmrec.kaggle_deepfm.context import (  # noqa: E402
    DONE,
    SKIPPED,
    DeepFMKaggleContext,
    finish_process,
    start_process,
)
from hmrec.kaggle_deepfm.report import (  # noqa: E402
    build_submission,
    format_checks_passed,
    record_files,
    record_rehearsal,
    rehearsal_outcome,
    require_passed_rehearsal,
    run_shards,
)
from hmrec.kaggle_deepfm.score import score_run  # noqa: E402
from hmrec.kaggle_deepfm.train import (  # noqa: E402
    TRAINING_FILE,
    chosen_inputs,
    read_base_inputs,
    read_training_summary,
    train_run_model,
)
from hmrec.logging_utils import get_logger  # noqa: E402

LOGGER = get_logger("deepfm_kaggle_step")
STEPS = ("train", "score", "submission", "check")
SCRIPT = "scripts/deepfm_kaggle_step.py"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=RUNS)
    parser.add_argument("--step", required=True, choices=STEPS)
    return parser.parse_args()


def run_train(context: DeepFMKaggleContext) -> tuple[str, int, str]:
    """Train and save the model unless it is finished. Returns (status, rows, detail)."""
    model_dir = context.paths.kaggle_deepfm_model_dir(context.run)
    if (model_dir / TRAINING_FILE).is_file():
        summary = read_training_summary(model_dir)
        return SKIPPED, summary.n_train_rows, "the model exists; no label was read"
    inputs = chosen_inputs(context, read_base_inputs(context))
    summary = train_run_model(context, inputs, f"{SCRIPT} --step train")
    detail = (
        f"folds {' '.join(summary.train_folds)}, {summary.n_train_queries} queries, "
        f"{summary.steps} steps, mean loss {summary.mean_loss!r}, "
        f"{summary.train_seconds:.1f} s"
    )
    return DONE, summary.n_train_rows, detail


def run_score(context: DeepFMKaggleContext) -> tuple[str, int, str]:
    """Score every shard that has no Top-K file yet. Returns (status, rows, detail)."""
    n_shards = run_shards(context)
    n_rows = score_run(context, chosen_inputs(context, read_base_inputs(context)), n_shards)
    return DONE, n_rows, f"{n_shards} shards"


def run_submission(context: DeepFMKaggleContext) -> tuple[str, int, str]:
    """Write and check the submission file and record it. Returns (status, rows, detail).

    A failed format check is recorded here; main stops the process after its run row.
    """
    check = build_submission(context)
    record_files(context, check)
    detail = (
        f"csv sha256 {check.csv_sha256}, gzip sha256 {check.gzip_sha256}, "
        f"{len(check.checks)} format checks, failed: {check.failed or 'none'}"
    )
    return DONE, check.n_rows, detail


def run_check(context: DeepFMKaggleContext) -> tuple[str, int, str]:
    """Compare the rehearsal with stage 2 and record the outcome. Returns (status, rows, detail)."""
    outcome = rehearsal_outcome(context)
    record_rehearsal(context, outcome)
    comparison = outcome.comparison
    detail = (
        f"training loss {outcome.new_loss!r} (stage 2: {outcome.stage2_loss!r}), equal: "
        f"{outcome.loss_equal}; identical Top-K lists: {comparison.n_identical} of "
        f"{comparison.n_queries} ({comparison.share:.6f}, needed {outcome.min_identical_share}); "
        f"format checks passed: {outcome.format_checks_passed}; passed: {outcome.passed}"
    )
    LOGGER.info("rehearsal | %s", detail)
    if not outcome.passed:
        raise SystemExit(f"the rehearsal did NOT pass: {detail}")
    return DONE, comparison.n_queries, detail


def main() -> None:
    """Run one step and record it."""
    started = time.perf_counter()
    args = parse_args()
    context = start_process(args.run, args.step)
    if args.run == RUN_SUBMISSION:
        # The gate of the real run, before anything is read.
        root = context.paths.project_root
        commit = require_passed_rehearsal(
            context, lambda recorded: code_changed_since(root, recorded)
        )
        LOGGER.info("the last rehearsal passed at commit %s with this code", commit[:12])
    runners = {
        "train": run_train,
        "score": run_score,
        "submission": run_submission,
        "check": run_check,
    }
    status, n_rows, detail = runners[args.step](context)
    finish_process(context, args.step, status, n_rows, started, detail)
    if args.step == "submission" and not format_checks_passed(context):
        raise SystemExit(f"format checks FAILED: {detail}")


if __name__ == "__main__":
    main()
