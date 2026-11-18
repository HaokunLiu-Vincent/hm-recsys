"""Score ranked Top-K tables of the Ranker on a V fold (M7; proposal §2.5, §9.4).

Labels are read here (AGENTS.md §3.1: only eval/ and ranker/sampling.py read
labels). load_m7_evaluator is the ONLY way M7 gets the labels of the fold it
predicts, and it refuses every fold that is not a V fold: T labels are not read
before M8.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hmrec.data.folds import Fold
from hmrec.eval.metrics import FOLD, QUERY_ID, VALUE, macro_mean
from hmrec.eval.report import FoldEvaluator

M7_EVALUATION_PHASE = "V"


def load_m7_evaluator(fold: Fold, labels_file: Path) -> FoldEvaluator:
    """The evaluator of a V fold. The phase is checked before the file is opened.

    Raises:
        ValueError: If the fold is not a V fold.
    """
    if fold.phase != M7_EVALUATION_PHASE:
        raise ValueError(
            f"M7 evaluates only {M7_EVALUATION_PHASE} folds, got fold {fold.name} "
            f"(phase {fold.phase}); its labels are not read"
        )
    return FoldEvaluator.from_file(fold.name, labels_file)


@dataclass(frozen=True)
class RankingScores:
    """Metric values of one ranked table on one fold.

    Attributes:
        n_queries: Queries of the fold (every query with a label counts, also
            one with no hit or no prediction).
        means: metric name -> mean over the fold's queries (float64, math.fsum).
        per_query: One row per query. Columns: query_id, fold, then one float64
            column per metric; sorted by query_id.
    """

    n_queries: int
    means: dict[str, float]
    per_query: pl.DataFrame


def ranking_scores(
    evaluator: FoldEvaluator, ranked: pl.DataFrame, metrics: Sequence[str]
) -> RankingScores:
    """Evaluate a ranked table with several metrics.

    Args:
        evaluator: The fold's evaluator.
        ranked: One row per (query_id, article_idx) with rank (1 = best), e.g.
            the Top-12 of every query.
        metrics: e.g. ["map@12", "recall@12", "ndcg@12"].

    Returns:
        The fold means and the per-query values.
    """
    tables = evaluator.per_query_many(ranked, metrics)
    means = {metric: macro_mean(tables[metric]) for metric in metrics}
    per_query = tables[metrics[0]].select(QUERY_ID, FOLD)
    for metric in metrics:
        # Every table has one row per labelled query, sorted by query_id (eval.metrics).
        if tables[metric][QUERY_ID].to_list() != per_query[QUERY_ID].to_list():
            raise ValueError(f"{metric}: per-query rows are not aligned")
        per_query = per_query.with_columns(tables[metric][VALUE].alias(metric))
    return RankingScores(evaluator.n_queries, means, per_query)
