"""Independent cross-check of one DeepFM trial's evaluation (see README.md).

Usage:
    uv run python scripts/checks/check_deepfm_eval.py --area v_phase \
        --config-id 0123456789ab --seed 42 --fold V1 --epoch 3

Reads the trial's saved Top-12 (artifacts/deepfm/.../predictions.parquet) and
recomputes MAP@12 for --epoch with plain Python loops. Nothing from
hmrec.eval.metrics or hmrec.deepfm is used for the computation, so a bug there
cannot hide itself. Then compares with the row of results/deepfm_metrics.csv.

Checks:
    1. every query has at most 12 rows, ranks 1..n, no repeated article;
    2. inside a query the rows are ordered by score DESC, then article_idx ASC;
    3. the recomputed MAP@12 equals the recorded one (difference < 1e-12).

AP@12 of a query with label set G and ranked list r_1 ... r_n (n <= 12):
    AP = (1 / min(12, |G|)) * sum over k with r_k in G of (hits among r_1..r_k) / k
MAP@12 is the mean over ALL labelled queries of the fold; a query without any
prediction or hit counts as 0.

Labels: the label table of the V fold is obtained through
hmrec.eval.ranker_eval.load_m7_evaluator, which refuses every fold that is not
a V fold. Any failure raises ValueError (non-zero exit).
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import csv  # noqa: E402
import logging  # noqa: E402
from collections import defaultdict  # noqa: E402

import polars as pl  # noqa: E402

from hmrec.data.folds import load_fold_manifest  # noqa: E402
from hmrec.eval.ranker_eval import load_m7_evaluator  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402

LOGGER = get_logger("check_deepfm_eval")
TOP_K = 12
METRIC = "map@12"
TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    return parser.parse_args()


def ranked_lists(predictions: pl.DataFrame) -> dict[int, list[int]]:
    """query_id -> its articles in rank order, after checking ranks, order and repeats."""
    rows_by_query: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for query_id, rank, article_idx, score in predictions.select(
        "query_id", "rank", "article_idx", "score"
    ).iter_rows():
        rows_by_query[query_id].append((rank, article_idx, score))
    lists: dict[int, list[int]] = {}
    for query_id, rows in rows_by_query.items():
        rows.sort()
        ranks = [row[0] for row in rows]
        articles = [row[1] for row in rows]
        if ranks != list(range(1, len(rows) + 1)) or len(rows) > TOP_K:
            raise ValueError(
                f"query {query_id}: ranks are {ranks}, expected 1..n with n <= {TOP_K}"
            )
        if len(set(articles)) != len(articles):
            raise ValueError(f"query {query_id}: an article is repeated")
        for (_, article, score), (_, next_article, next_score) in zip(rows, rows[1:], strict=False):
            in_order = score > next_score or (score == next_score and article < next_article)
            if not in_order:
                raise ValueError(f"query {query_id}: not ordered by score DESC, article_idx ASC")
        lists[query_id] = articles
    return lists


def mean_average_precision(lists: dict[int, list[int]], labels: dict[int, set[int]]) -> float:
    """MAP@12 over all labelled queries, with plain loops."""
    total = 0.0
    for query_id, wanted in labels.items():
        hits = 0
        precision_sum = 0.0
        for position, article in enumerate(lists.get(query_id, []), start=1):
            if article in wanted:
                hits += 1
                precision_sum += hits / position
        total += precision_sum / min(TOP_K, len(wanted))
    return total / len(labels)


def recorded_value(metrics_file: str, args: argparse.Namespace) -> float:
    """The recorded MAP@12 of the trial's epoch; exactly one row must match."""
    with open(metrics_file, newline="", encoding="utf-8") as handle:  # noqa: PTH123
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["area"] == args.area
            and row["config_id"] == args.config_id
            and int(row["seed"]) == args.seed
            and row["fold"] == args.fold
            and int(row["epoch"]) == args.epoch
        ]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one recorded row, found {len(rows)}")
    return float(rows[0][METRIC])


def main() -> None:
    """Recompute MAP@12 of one trial's epoch and compare it with the record."""
    args = parse_args()
    setup_logging(logging.INFO)
    paths = get_paths()
    fold = load_fold_manifest(paths.fold_manifest_file).get(args.fold)
    evaluator = load_m7_evaluator(fold, paths.labels_file(args.fold))  # refuses non-V folds
    labels: dict[int, set[int]] = defaultdict(set)
    for query_id, article_idx in evaluator.labels.select("query_id", "article_idx").iter_rows():
        labels[query_id].add(article_idx)
    directory = paths.deepfm_trial_dir(args.area, args.config_id, args.seed, args.fold)
    predictions = pl.read_parquet(directory / "predictions.parquet").filter(
        pl.col("epoch") == args.epoch
    )
    if predictions.height == 0:
        raise ValueError(f"{directory} has no predictions for epoch {args.epoch}")
    lists = ranked_lists(predictions)
    unknown = [query_id for query_id in lists if query_id not in labels]
    if unknown:
        raise ValueError(f"{len(unknown)} predicted queries are not queries of fold {args.fold}")
    recomputed = mean_average_precision(lists, labels)
    recorded = recorded_value(str(paths.deepfm_metrics_file), args)
    difference = abs(recomputed - recorded)
    LOGGER.info(
        "%s %s seed %d epoch %d: recomputed %s = %.15f, recorded %.15f, difference %.3e, "
        "%d queries",
        args.fold, args.config_id, args.seed, args.epoch, METRIC, recomputed, recorded,
        difference, len(labels),
    )  # fmt: skip
    if difference >= TOLERANCE:
        raise ValueError(f"recomputed {recomputed!r} differs from the recorded {recorded!r}")
    LOGGER.info("OK")


if __name__ == "__main__":
    main()
