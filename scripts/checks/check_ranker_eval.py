"""Independent cross-check of one Ranker trial's evaluation (see README.md).

Usage:
    uv run python scripts/checks/check_ranker_eval.py --area speed_test \
        --experiment full --config-id 0123456789ab --fold V1 --num-trees 300

Reads the trial's saved Top-12 (artifacts/predictions/...) and recomputes MAP@12,
Recall@12 and NDCG@12 for --num-trees with plain Python loops. Nothing from
hmrec.eval.metrics or hmrec.ranker is used for the computation, so a bug there
cannot hide itself. Then compares with the row of the results table.

Checks:
    1. every query has at most 12 rows, ranks 1..n, no repeated article;
    2. inside a query the rows are ordered by score DESC, then article_idx ASC;
    3. the recomputed fold means equal the recorded ones (difference < 1e-9).

Labels: for a V fold (areas speed_test, v_phase, m8_rehearsal) the label table is
obtained through hmrec.eval.ranker_eval, which refuses every fold that is not a V
fold. For a trial of the final evaluation (area t_phase, a T fold) it is obtained
through hmrec.eval.final_eval with a FormalRunPermit from hmrec.final.gate: the
gate refuses unless the rehearsal passed and the formal run was started with this
code, so this script cannot be the first reader of a T label.
Any failure raises ValueError (non-zero exit).
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
from collections import defaultdict  # noqa: E402

import polars as pl  # noqa: E402

from hmrec.data.folds import Fold, load_fold_manifest  # noqa: E402
from hmrec.eval.final_eval import load_final_evaluator  # noqa: E402
from hmrec.eval.ranker_eval import load_m7_evaluator  # noqa: E402
from hmrec.final.gate import formal_run_permit  # noqa: E402
from hmrec.final.process import load_process_context  # noqa: E402
from hmrec.hpo.v_records import (  # noqa: E402
    AREA_T_PHASE,
    AREAS,
    FINAL_AREAS,
    read_metric_rows,
    results_files,
)
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import ProjectPaths, get_paths  # noqa: E402

LOGGER = get_logger("check_ranker_eval")
TOP_K = 12
METRICS = ("map@12", "recall@12", "ndcg@12")
TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, choices=(*AREAS, *FINAL_AREAS))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--num-trees", required=True, type=int)
    return parser.parse_args()


def check_ranking(rows: list[tuple[int, int, float]], query_id: int) -> None:
    """One query's (rank, article_idx, score) rows must be a clean Top-12 in the right order."""
    ranks = [rank for rank, _, _ in rows]
    articles = [article for _, article, _ in rows]
    if ranks != list(range(1, len(rows) + 1)) or len(rows) > TOP_K:
        raise ValueError(f"query {query_id}: ranks are {ranks}, expected 1..n with n <= {TOP_K}")
    if len(set(articles)) != len(articles):
        raise ValueError(f"query {query_id}: an article repeats")
    for (_, article_a, score_a), (_, article_b, score_b) in zip(rows, rows[1:], strict=False):
        in_order = score_a > score_b or (score_a == score_b and article_a < article_b)
        if not in_order:
            raise ValueError(f"query {query_id}: rows are not ordered by score DESC, article ASC")


def query_metrics(ranked_articles: list[int], labels: set[int]) -> tuple[float, float, float]:
    """(AP@12, Recall@12, NDCG@12) of one query, from the definitions in proposal §2.5.

    `labels` is G_q: everything the customer bought in the label week, whether or
    not it was among the candidates. With K = 12 and hits = labels found in the
    first K positions:
        AP@K     = sum over hit positions r of (hits up to r) / r, divided by min(K, |G_q|)
        Recall@K = hits / |G_q|                  (the denominator is NOT cut at K)
        NDCG@K   = DCG@K / IDCG@K, DCG@K = sum over hit positions r of 1 / log2(r + 1),
                   IDCG@K = the DCG of a perfect list = min(K, |G_q|) hits at ranks 1, 2, ...
    """
    hits = 0
    precision_sum = 0.0
    dcg = 0.0
    for position, article in enumerate(ranked_articles[:TOP_K], start=1):
        if article in labels:
            hits += 1
            precision_sum += hits / position
            dcg += 1.0 / math.log2(position + 1)
    # A perfect Top-K list can hold at most K labels: min(K, |G_q|) hits, not |G_q|.
    ideal_hits = min(TOP_K, len(labels))
    ideal_dcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    average_precision = precision_sum / min(TOP_K, len(labels))
    return average_precision, hits / len(labels), dcg / ideal_dcg


def recompute_fold_means(top: pl.DataFrame, label_table: pl.DataFrame) -> tuple[list[float], int]:
    """Recompute the fold means of METRICS from one checkpoint's Top-12 rows.

    Args:
        top: The saved Top-12 of ONE number of trees. Columns used: query_id, rank,
            article_idx, score (other columns, e.g. num_trees, are ignored).
        label_table: One row per (query_id, article_idx) of the fold's labels.

    Returns:
        ([MAP@12, Recall@12, NDCG@12] averaged over ALL labelled queries, their number).
        A query without any prediction counts as 0.

    Raises:
        ValueError: If a query's ranking is not a clean Top-12 in the right order, or a
            predicted query has no label.
    """
    labels: dict[int, set[int]] = defaultdict(set)
    for query_id, article in zip(
        label_table["query_id"].to_list(), label_table["article_idx"].to_list(), strict=True
    ):
        labels[query_id].add(article)
    ranked: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    # Select the four columns by name: the file also has num_trees, in another position.
    ordered = top.select("query_id", "rank", "article_idx", "score").sort("query_id", "rank")
    for query_id, rank, article, score in ordered.iter_rows():
        ranked[query_id].append((rank, article, score))
    unknown = set(ranked) - set(labels)
    if unknown:
        raise ValueError(f"{len(unknown)} predicted queries have no label")
    values: list[list[float]] = [[], [], []]
    for query_id, query_labels in labels.items():
        rows = ranked.get(query_id, [])
        check_ranking(rows, query_id)
        for index, value in enumerate(query_metrics([row[1] for row in rows], query_labels)):
            values[index].append(value)
    return [math.fsum(column) / len(labels) for column in values], len(labels)


def read_label_table(paths: ProjectPaths, fold: Fold, area: str) -> pl.DataFrame:
    """The fold's labels, one row per (query_id, article_idx), through the area's guarded loader.

    Area t_phase: the gate of the final evaluation is passed first (a passed
    rehearsal and a started formal run, with this code, clean tree); only then
    are the labels of the T fold read. Every other area: the M7 loader, which
    accepts V folds only.
    """
    labels_file = paths.labels_file(fold.name)
    if area == AREA_T_PHASE:
        permit = formal_run_permit(load_process_context(paths))
        return load_final_evaluator(fold, labels_file, permit).labels
    return load_m7_evaluator(fold, labels_file).labels


def main() -> None:
    """Recompute one trial's metrics at one number of trees and compare."""
    args = parse_args()
    setup_logging(logging.INFO)
    paths = get_paths()
    fold = load_fold_manifest(paths.fold_manifest_file).get(args.fold)
    # The loader checks the phase (and for a T fold the gate) first; only the label
    # table is used here.
    label_table = read_label_table(paths, fold, args.area)
    predictions_file = paths.ranker_predictions_file(
        args.area, args.experiment, args.config_id, args.fold
    )
    top = pl.read_parquet(predictions_file).filter(pl.col("num_trees") == args.num_trees)
    if top.height == 0:
        raise ValueError(f"{predictions_file} has no rows with num_trees = {args.num_trees}")
    means, n_queries = recompute_fold_means(top, label_table)
    _, metrics_file = results_files(paths, args.area)
    recorded = [
        row
        for row in read_metric_rows(metrics_file, METRICS)
        if (row.experiment, row.config_id, row.fold, row.num_trees)
        == (args.experiment, args.config_id, args.fold, args.num_trees)
    ]
    if len(recorded) != 1:
        raise ValueError(f"expected one recorded metric row, found {len(recorded)}")
    for metric, value in zip(METRICS, means, strict=True):
        difference = abs(value - recorded[0].scores[metric])
        LOGGER.info(
            "%s: recomputed %.12f, recorded %.12f", metric, value, recorded[0].scores[metric]
        )
        if difference >= TOLERANCE:
            raise ValueError(f"{metric}: recomputed {value} differs from the recorded value")
    LOGGER.info("OK: %d queries, ranking order and all %d metrics agree", n_queries, len(METRICS))


if __name__ == "__main__":
    main()
