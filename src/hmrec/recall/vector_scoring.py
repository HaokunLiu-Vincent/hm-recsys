"""Top-quota scoring from user / article vectors, shared by R5 WALS and R6 Two-Tower.

Both channels give each user and each active article a vector and keep, per
query, the `quota` articles with the largest dot product (proposal §5 R5, R6):

  1. Selection: a float32 matrix product per chunk of users finds every article
     whose score is >= the user's quota-th best score (ties all kept).
  2. Stored score: recomputed with hmrec.vectors.dot_pairs, the same function
     that features/ uses for wals_dot / tt_dot, so a candidate's score equals
     its dot-product feature exactly. rank_candidates then orders by
     (score DESC, article_idx ASC) and cuts to quota.
The matrix product and dot_pairs may differ in the last float32 bit, so in a
near-tie at the quota-th place the kept set can differ from an exact ranking by
the last item; the stored scores and order are always exact.
"""

import numpy as np
import polars as pl

from hmrec.recall.base import (
    ARTICLE_IDX,
    CUSTOMER_IDX,
    QUERY_ID,
    SCORE,
    OutputSpec,
    rank_candidates,
)
from hmrec.vectors import UserItemVectors, dot_pairs, lookup_rows


def select_top_items(
    user_vectors: np.ndarray, item_vectors: np.ndarray, quota: int
) -> tuple[np.ndarray, np.ndarray]:
    """For each user row, every item whose score is >= that user's quota-th best score.

    Ties at the quota-th score are all kept (the exact cut happens later in
    rank_candidates), so the result does not depend on how ties are ordered here.

    Args:
        user_vectors: float32 [n_users_chunk, dim].
        item_vectors: float32 [n_items, dim].
        quota: Number of items wanted per user.

    Returns:
        (user_rows, item_rows): int64 arrays of the same length, positions into
        the two inputs.
    """
    scores = user_vectors @ item_vectors.T  # float32 [n_users_chunk, n_items]
    n_items = item_vectors.shape[0]
    if n_items <= quota:
        user_rows, item_rows = np.nonzero(np.ones_like(scores, dtype=bool))
        return user_rows, item_rows
    # np.partition puts the quota-th largest value of each row at position n_items - quota.
    threshold = np.partition(scores, n_items - quota, axis=1)[:, n_items - quota]
    user_rows, item_rows = np.nonzero(scores >= threshold[:, None])
    return user_rows, item_rows


def score_query_chunk(
    queries: pl.DataFrame,
    vectors: UserItemVectors,
    active_item_rows: np.ndarray,
    quota: int,
    pair_chunk_size: int,
    spec: OutputSpec,
) -> pl.DataFrame:
    """Top-quota active articles for a chunk of queries whose customers have a vector.

    Args:
        queries: One row per query; columns query_id, customer_idx. Every
            customer must have a vector.
        vectors: The channel's vectors.
        active_item_rows: Rows of vectors.item_vectors that are in A(a), int64.
        quota: Candidates per query.
        pair_chunk_size: Passed to dot_pairs.
        spec: The channel's output declaration.

    Returns:
        Candidates valid for `spec`.
    """
    user_rows = lookup_rows(vectors.user_customer_idx, queries[CUSTOMER_IDX].to_numpy())
    chunk_rows, item_positions = select_top_items(
        vectors.user_vectors[user_rows], vectors.item_vectors[active_item_rows], quota
    )
    customers = queries[CUSTOMER_IDX].to_numpy()[chunk_rows]
    articles = vectors.item_article_idx[active_item_rows[item_positions]]
    # The stored score comes from dot_pairs, the same function as the *_dot feature.
    scores = dot_pairs(vectors, customers, articles, pair_chunk_size)
    scored = pl.DataFrame(
        {
            QUERY_ID: queries[QUERY_ID].to_numpy()[chunk_rows],
            ARTICLE_IDX: articles,
            SCORE: scores,
        },
        schema={QUERY_ID: pl.Int32, ARTICLE_IDX: pl.Int32, SCORE: pl.Float32},
    )
    return rank_candidates(scored, spec, quota)


def candidates_from_vectors(
    queries: pl.DataFrame,
    vectors: UserItemVectors,
    active_pool: pl.DataFrame,
    quota: int,
    query_chunk_size: int,
    pair_chunk_size: int,
    spec: OutputSpec,
) -> pl.DataFrame:
    """Candidates for all queries, in chunks of `query_chunk_size` queries (§1.3).

    Queries whose customer has no vector get no rows.

    Returns:
        Candidates valid for `spec`, sorted by (query_id, rank).
    """
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be > 0, got {query_chunk_size}")
    active_rows = lookup_rows(vectors.item_article_idx, active_pool[ARTICLE_IDX].to_numpy())
    active_rows = np.sort(active_rows[active_rows >= 0])
    has_vector = lookup_rows(vectors.user_customer_idx, queries[CUSTOMER_IDX].to_numpy()) >= 0
    scorable = queries.select(QUERY_ID, CUSTOMER_IDX).filter(pl.Series(has_vector)).sort(QUERY_ID)
    if scorable.height == 0 or active_rows.size == 0:
        return spec.schema.to_frame()
    chunks: list[pl.DataFrame] = []
    for start in range(0, scorable.height, query_chunk_size):
        chunk = scorable.slice(start, query_chunk_size)
        scored = score_query_chunk(chunk, vectors, active_rows, quota, pair_chunk_size, spec)
        chunks.append(scored)
    return pl.concat(chunks)
