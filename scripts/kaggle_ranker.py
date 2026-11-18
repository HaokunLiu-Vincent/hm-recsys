"""CLI for the Kaggle late submission: one ranker step of a run (LightGBM side, no torch).

    uv run python scripts/kaggle_ranker.py --run rehearsal --step dataset

Steps (hmrec.kaggle.ranker), all with the locked configuration:
    sample    draw the training sample of every fold M8 did not sample (T4)
    dataset   check the reused M8 sample files, build the binary Dataset
    train     train the model from scratch and save it
    score     score every shard and write its Top-K
    check     the rehearsal only: compare with M8 (trees, Top-K lists, file format),
              record the outcome and exit with an error if it did not pass

sample and dataset read T labels as TRAINING labels; each first appends a row to
results/kaggle_label_access.csv. One step per process.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402

from hmrec.kaggle.context import ALL_SHARDS, DONE, SKIPPED, finish_process, start_process  # noqa: E402
from hmrec.kaggle.plan import RUNS  # noqa: E402
from hmrec.kaggle.ranker import (  # noqa: E402
    RankerContext,
    build_run_dataset,
    load_ranker_context,
    load_run_model,
    new_sample_folds,
    rehearsal_outcome,
    sample_new_fold,
    score_shard,
    train_run_model,
)
from hmrec.kaggle.rehearsal import record_rehearsal  # noqa: E402
from hmrec.kaggle.report import format_checks_passed, read_all_tops, run_shards  # noqa: E402
from hmrec.logging_utils import get_logger  # noqa: E402
from hmrec.ranker.dataset import dataset_is_built, read_dataset_record  # noqa: E402

LOGGER = get_logger("kaggle_ranker")
STEPS = ("sample", "dataset", "train", "score", "check")
SCRIPT = "scripts/kaggle_ranker.py"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, choices=RUNS)
    parser.add_argument("--step", required=True, choices=STEPS)
    return parser.parse_args()


def run_sample(ranker: RankerContext) -> tuple[str, int, str]:
    """Sample the new folds. Returns (status, rows, detail)."""
    records = [
        sample_new_fold(ranker, fold, f"{SCRIPT} --step sample")
        for fold in new_sample_folds(ranker)
    ]
    detail = "; ".join(
        f"{record.fold}: {record.n_rows} rows, {record.n_queries} queries, sha256 {record.sha256}"
        for record in records
    )
    return DONE, sum(record.n_rows for record in records), detail or "no fold to sample"


def run_dataset(ranker: RankerContext) -> tuple[str, int, str]:
    """Build the Dataset unless it exists. Returns (status, rows, detail)."""
    directory = ranker.context.paths.kaggle_dataset_dir(ranker.context.run)
    if dataset_is_built(directory):
        record = read_dataset_record(directory)
        return SKIPPED, record.n_rows, "the Dataset exists; no label was read"
    record = build_run_dataset(ranker, f"{SCRIPT} --step dataset")
    detail = (
        f"folds {' '.join(record.train_folds)}, {record.n_queries} queries, "
        f"binary sha256 {record.binary_sha256}"
    )
    return DONE, record.n_rows, detail


def run_train(ranker: RankerContext) -> tuple[str, int, str]:
    """Train and save the model unless it exists. Returns (status, rows, detail)."""
    if ranker.context.paths.kaggle_model_file(ranker.context.run).exists():
        return SKIPPED, 0, "the model exists"
    model = train_run_model(ranker)
    detail = (
        f"{ranker.locked.num_trees} trees in {model.train_seconds:.1f} s, model sha256 "
        f"{model.model_sha256}, trees sha256 {model.trees_sha256}"
    )
    return DONE, model.n_train_rows, detail


def run_score(ranker: RankerContext) -> tuple[str, int, str]:
    """Score every shard that has no Top-K file yet. Returns (status, rows, detail)."""
    booster = load_run_model(ranker)
    n_shards = run_shards(ranker.context)
    n_rows = sum(score_shard(ranker, booster, shard) for shard in range(n_shards))
    return DONE, n_rows, f"{n_shards} shards"


def run_check(ranker: RankerContext) -> tuple[str, int, str]:
    """Compare the rehearsal with M8 and record the outcome. Returns (status, rows, detail)."""
    context = ranker.context
    outcome = rehearsal_outcome(
        ranker, read_all_tops(context), run_shards(context), format_checks_passed(context)
    )
    record_rehearsal(context.paths, outcome, context.stamp)
    comparison = outcome.comparison
    detail = (
        f"trees identical: {outcome.trees_identical}; identical Top-K lists: "
        f"{comparison.n_identical} of {comparison.n_queries} ({comparison.share:.6f}, needed "
        f"{outcome.min_identical_share}); format checks passed: "
        f"{outcome.format_checks_passed}; passed: {outcome.passed}"
    )
    LOGGER.info("rehearsal | %s", detail)
    if not outcome.passed:
        raise SystemExit(f"the rehearsal did NOT pass: {detail}")
    return DONE, comparison.n_queries, detail


def main() -> None:
    """Run one ranker step and record it."""
    started = time.perf_counter()
    args = parse_args()
    step = f"ranker_{args.step}"
    context = start_process(args.run, step)
    ranker = load_ranker_context(context)
    runners = {
        "sample": run_sample,
        "dataset": run_dataset,
        "train": run_train,
        "score": run_score,
        "check": run_check,
    }
    status, n_rows, detail = runners[args.step](ranker)
    finish_process(context, step, ALL_SHARDS, status, n_rows, started, detail)


if __name__ == "__main__":
    main()
