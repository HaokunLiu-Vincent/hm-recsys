"""Score candidates of one fold and summarise several folds (proposal §2.5, §4.1, §4.4).

Labels are read here (AGENTS.md §3.1: only eval/ and ranker/sampling.py read
labels). Callers such as hmrec.hpo.runner get a FoldEvaluator and never touch
the label table themselves.

Metric names: "recall@K", "ndcg@K", "map@K" (map@K = mean AP@K).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hmrec.eval.metrics import (
    ARTICLE_IDX,
    FOLD,
    QUERY_ID,
    VALUE,
    average_precision_at_k,
    check_labels,
    macro_mean,
    ndcg_at_k,
    per_fold_means,
    query_weighted_mean,
    recall_at_k,
)

IS_HIT = "is_hit"
METRIC_KINDS = ("recall", "ndcg", "map")


def parse_metric(name: str) -> tuple[str, int]:
    """Split "recall@100" into ("recall", 100)."""
    kind, separator, k_text = name.partition("@")
    if separator != "@" or kind not in METRIC_KINDS or not k_text.isdigit() or int(k_text) <= 0:
        raise ValueError(f"metric must look like recall@K, ndcg@K or map@K, got {name!r}")
    return kind, int(k_text)


@dataclass(frozen=True)
class FoldEvaluator:
    """Scores candidates against one fold's labels.

    Attributes:
        fold: Fold name, e.g. "D1".
        labels: One row per (query_id, article_idx) in G_q; columns query_id, fold, article_idx.
    """

    fold: str
    labels: pl.DataFrame

    @classmethod
    def from_file(cls, fold: str, labels_file: Path) -> "FoldEvaluator":
        """Read artifacts/labels/{fold}.parquet (written by data/queries.py)."""
        labels = pl.read_parquet(labels_file, columns=[QUERY_ID, FOLD, ARTICLE_IDX])
        check_labels(labels)
        folds_in_file = labels[FOLD].unique().to_list()
        if folds_in_file != [fold]:
            raise ValueError(f"{labels_file} holds folds {folds_in_file}, expected [{fold!r}]")
        return cls(fold, labels)

    @property
    def n_queries(self) -> int:
        """Number of queries (queries with at least one label)."""
        return self.labels[QUERY_ID].n_unique()

    def per_query(self, candidates: pl.DataFrame, metric: str) -> pl.DataFrame:
        """One metric value per query of this fold.

        Args:
            candidates: One row per (query_id, article_idx) with rank.
            metric: e.g. "recall@100".

        Returns:
            Columns query_id, fold, value (float64); one row per labelled query.
        """
        kind, k = parse_metric(metric)
        predictions = candidates.select(QUERY_ID, ARTICLE_IDX, "rank")
        if kind == "recall":
            return recall_at_k(predictions, self.labels, k)
        if kind == "ndcg":
            return ndcg_at_k(predictions, self.labels, k)
        return average_precision_at_k(predictions, self.labels, k)

    def mark_hits(self, candidates: pl.DataFrame) -> pl.DataFrame:
        """`candidates` with a bool column is_hit: True if (query_id, article_idx) is a label."""
        label_pairs = self.labels.select(QUERY_ID, ARTICLE_IDX).with_columns(
            pl.lit(True).alias(IS_HIT)
        )
        return candidates.join(label_pairs, on=[QUERY_ID, ARTICLE_IDX], how="left").with_columns(
            pl.col(IS_HIT).fill_null(False)
        )


@dataclass(frozen=True)
class MultiFoldSummary:
    """Per-fold and pooled scores of one metric (proposal §4.1).

    Attributes:
        fold_scores: Fold -> mean over that fold's queries, in fold order.
        fold_queries: Fold -> number of queries.
        pooled: Query-weighted mean over all folds (the primary score).
        equal_fold: Mean of the fold means (reported only).
        group_scores: Group name -> query-weighted mean over the group's folds
            (D-phase diagnostics D1-D2 / D3-D4; reported only).
    """

    fold_scores: dict[str, float]
    fold_queries: dict[str, int]
    pooled: float
    equal_fold: float
    group_scores: dict[str, float]


def summarize_folds(
    per_query: pl.DataFrame, fold_groups: Mapping[str, Sequence[str]]
) -> MultiFoldSummary:
    """Summarise per-query values of several folds.

    Args:
        per_query: One row per query of all folds; columns query_id, fold, value.
        fold_groups: Group name -> fold names, e.g. {"D1_D2": ["D1", "D2"]}.

    Returns:
        The summary; every number is float64 and exactly reproducible (math.fsum).
    """
    fold_table = per_fold_means(per_query)
    fold_scores = dict(zip(fold_table[FOLD].to_list(), fold_table[VALUE].to_list(), strict=True))
    fold_queries = dict(
        zip(fold_table[FOLD].to_list(), fold_table["n_queries"].to_list(), strict=True)
    )
    group_scores: dict[str, float] = {}
    for group, folds in fold_groups.items():
        group_rows = per_query.filter(pl.col(FOLD).is_in(list(folds)))
        group_scores[group] = macro_mean(group_rows)
    return MultiFoldSummary(
        fold_scores=fold_scores,
        fold_queries=fold_queries,
        pooled=query_weighted_mean(per_query),
        equal_fold=math.fsum(fold_scores.values()) / len(fold_scores),
        group_scores=group_scores,
    )
