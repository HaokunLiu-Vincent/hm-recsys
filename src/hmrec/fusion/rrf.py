"""Reciprocal rank fusion (proposal §8.2, §9.5).

    rrf_score = sum over the channels that returned the article of 1 / (k + rank), k = 60

Also the two other aggregate columns of the A group (n_channels, best_rank) and
the RRF baseline ranking: per query, order by rrf_score DESC, then best_rank
ASC, then article_idx ASC, and keep the Top-K.

Ranks are float32 columns with NaN for "not returned by that channel"
(hmrec.fusion.union).
"""

from collections.abc import Sequence

import polars as pl

QUERY_ID = "query_id"
ARTICLE_IDX = "article_idx"
RRF_SCORE = "rrf_score"
BEST_RANK = "best_rank"
N_CHANNELS = "n_channels"
RANK = "rank"


def rrf_score_expr(rank_columns: Sequence[str], rrf_k: int) -> pl.Expr:
    """rrf_score = sum of 1 / (rrf_k + rank) over the channels with a rank (proposal §8.2).

    Computed in float64 and summed in the order of `rank_columns` (always the
    same order, so the result is reproducible), then stored as float32.
    """
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be > 0, got {rrf_k}")
    terms = [
        pl.when(pl.col(name).is_nan())
        .then(pl.lit(0.0, dtype=pl.Float64))
        .otherwise(1.0 / (rrf_k + pl.col(name).cast(pl.Float64)))
        for name in rank_columns
    ]
    return pl.sum_horizontal(terms).cast(pl.Float32).alias(RRF_SCORE)


def n_channels_expr(rank_columns: Sequence[str]) -> pl.Expr:
    """n_channels = how many channels returned the article (1 ... 6), float32."""
    returned = [pl.col(name).is_not_nan().cast(pl.Int32) for name in rank_columns]
    return pl.sum_horizontal(returned).cast(pl.Float32).alias(N_CHANNELS)


def best_rank_expr(rank_columns: Sequence[str]) -> pl.Expr:
    """best_rank = the smallest rank over the channels that returned the article, float32."""
    # NaN -> null first: min_horizontal skips nulls, so missing channels are ignored.
    ranks = [pl.col(name).fill_nan(None) for name in rank_columns]
    return pl.min_horizontal(ranks).cast(pl.Float32).alias(BEST_RANK)


def rrf_baseline(wide: pl.DataFrame, top_k: int) -> pl.DataFrame:
    """The RRF baseline ranking (proposal §9.5): Top-`top_k` per query by rrf_score.

    Order within a query: rrf_score DESC, then best_rank ASC, then article_idx
    ASC (unique within a query, so the order is total). The stored float32
    rrf_score is used, so the order always matches the stored numbers.

    Args:
        wide: One row per (query_id, article_idx) with rrf_score and best_rank.
        top_k: Rows kept per query (12 for MAP@12).

    Returns:
        query_id, article_idx, rank (int32, 1 = best), score (float32 =
        rrf_score); sorted by (query_id, rank).
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    ordered = wide.select(QUERY_ID, ARTICLE_IDX, RRF_SCORE, BEST_RANK).sort(
        [QUERY_ID, RRF_SCORE, BEST_RANK, ARTICLE_IDX], descending=[False, True, False, False]
    )
    ranked = ordered.with_columns(
        pl.int_range(1, pl.len() + 1, dtype=pl.Int32).over(QUERY_ID).alias(RANK)
    ).filter(pl.col(RANK) <= top_k)
    return ranked.select(QUERY_ID, ARTICLE_IDX, RANK, pl.col(RRF_SCORE).alias("score"))
