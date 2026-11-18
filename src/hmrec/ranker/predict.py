"""Scoring candidates with the first k trees and Top-K per query (proposal §9.4, §14.2).

THE DEFINITION of "the score with k trees" used by M7 and M8 (staged_raw_scores):

    score_k = segment_1 + segment_2 + ... + segment_(k / checkpoint_every)
    segment_j = raw score of trees (j - 1) * checkpoint_every ... j * checkpoint_every - 1

computed with booster.predict(start_iteration=..., num_iteration=checkpoint_every,
raw_score=True) and added up in float64, in this order. Every tree is evaluated
once, and one pass gives the scores at every checkpoint (100, 200, ..., 2000).

It is NOT bit-identical to booster.predict(num_iteration=k): the same trees are
added in another order, so the two differ by floating-point rounding (about
1e-14 on test data). That is why the definition is fixed here and both M7 and
M8 call this one function.

Ranking order inside a query: score DESC, then article_idx ASC (a total order,
because article_idx is unique within a query).
"""

from collections.abc import Iterator, Sequence

import lightgbm as lgb
import numpy as np
import polars as pl

QUERY_ID = "query_id"
ARTICLE_IDX = "article_idx"
SCORE = "score"
RANK = "rank"


def checkpoints(num_trees: int, checkpoint_every: int) -> tuple[int, ...]:
    """The numbers of trees at which scores are produced: every, 2 * every, ..., num_trees.

    Raises:
        ValueError: If num_trees is not a positive multiple of checkpoint_every.
    """
    if checkpoint_every <= 0 or num_trees <= 0 or num_trees % checkpoint_every != 0:
        raise ValueError(
            f"num_trees ({num_trees}) must be a positive multiple of checkpoint_every "
            f"({checkpoint_every})"
        )
    return tuple(range(checkpoint_every, num_trees + 1, checkpoint_every))


def feature_matrix(rows: pl.DataFrame, feature_names: Sequence[str]) -> np.ndarray:
    """The feature columns as one float32 array [n_rows, n_features], in model order.

    float32 for every column: the int16 category codes are exact in float32, and
    it is the dtype the training data had.
    """
    values = rows.select(list(feature_names)).cast(pl.Float32).to_numpy()
    return np.ascontiguousarray(values, dtype=np.float32)


def staged_raw_scores(
    booster: lgb.Booster,
    features: np.ndarray,
    num_trees: int,
    checkpoint_every: int,
    num_threads: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (k, score with the first k trees) for every checkpoint k (module docstring).

    Args:
        booster: The trained model.
        features: float32 [n_rows, n_features], columns in model order.
        num_trees: The last checkpoint; at most the model's number of trees.
        checkpoint_every: Trees per segment.
        num_threads: CPU threads for prediction.

    Yields:
        (k, scores): scores is a NEW float64 array [n_rows] for every k.
    """
    if num_trees > booster.num_trees():
        raise ValueError(f"the model has {booster.num_trees()} trees, {num_trees} were requested")
    if features.ndim != 2 or features.shape[1] != booster.num_feature():
        raise ValueError(
            f"features must be [n_rows, {booster.num_feature()}], got shape {features.shape}"
        )
    total = np.zeros(features.shape[0], dtype=np.float64)
    for k in checkpoints(num_trees, checkpoint_every):
        segment = booster.predict(
            features,
            start_iteration=k - checkpoint_every,
            num_iteration=checkpoint_every,
            raw_score=True,
            num_threads=num_threads,
        )
        total += np.asarray(segment, dtype=np.float64)
        yield k, total.copy()


def top_k_per_query(
    query_ids: np.ndarray, article_idx: np.ndarray, scores: np.ndarray, top_k: int
) -> pl.DataFrame:
    """Rank every query's candidates and keep the Top-`top_k`.

    Order inside a query: score DESC, then article_idx ASC.

    Args:
        query_ids: int32 [n_rows].
        article_idx: int32 [n_rows], unique within a query.
        scores: float64 [n_rows].
        top_k: Rows kept per query (fewer if the query has fewer candidates).

    Returns:
        One row per kept (query_id, article_idx). Columns: query_id (int32),
        article_idx (int32), score (float64), rank (int32, 1 = best); sorted by
        (query_id, rank).
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN: the order would not be defined")
    table = pl.DataFrame(
        {QUERY_ID: query_ids, ARTICLE_IDX: article_idx, SCORE: scores},
        schema={QUERY_ID: pl.Int32, ARTICLE_IDX: pl.Int32, SCORE: pl.Float64},
    )
    ordered = table.sort([QUERY_ID, SCORE, ARTICLE_IDX], descending=[False, True, False])
    ranked = ordered.with_columns(
        pl.int_range(1, pl.len() + 1, dtype=pl.Int32).over(QUERY_ID).alias(RANK)
    )
    return ranked.filter(pl.col(RANK) <= top_k)


def score_rows(
    booster: lgb.Booster,
    rows: pl.DataFrame,
    feature_names: Sequence[str],
    num_trees: int,
    checkpoint_every: int,
    top_k: int,
    num_threads: int,
) -> dict[int, pl.DataFrame]:
    """Score a chunk of whole queries and return its Top-K at every checkpoint.

    Args:
        booster: The trained model.
        rows: One row per (query_id, article_idx) with the feature columns; all
            candidates of the chunk's queries (no sampling).
        feature_names: The model's features in order.
        num_trees: The last checkpoint.
        checkpoint_every: Trees between two checkpoints.
        top_k: Rows kept per query.
        num_threads: CPU threads for prediction.

    Returns:
        k -> the Top-K table of top_k_per_query, using the first k trees.
    """
    features = feature_matrix(rows, feature_names)
    query_ids = rows[QUERY_ID].to_numpy()
    article_idx = rows[ARTICLE_IDX].to_numpy()
    staged = staged_raw_scores(booster, features, num_trees, checkpoint_every, num_threads)
    return {k: top_k_per_query(query_ids, article_idx, scores, top_k) for k, scores in staged}
