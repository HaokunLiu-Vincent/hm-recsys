"""Group B: dense scores computed for EVERY candidate, 7 columns (proposal §8.3).

    tt_dot            R6 user vector . article vector, from the vectors saved for the
                      fold (the model that predicted the fold, before its fine-tune).
                      For a row that R6 returned it equals the R6 score exactly.
    wals_dot          R5 user vector . article vector; NaN if the user or the
                      article has no vector. Equals the R5 score for R5's rows.
    pop_bucket_score  R1's decayed popularity S(g, i) of the article in the user's
                      age bucket g (not cut to the Top-100, not limited to A(a));
                      0 if nobody of that bucket bought the article in the window.
    img_sim_hist      cosine between the article's CLIP image vector and the mean
                      image vector of the user's last 20 purchases before the
                      anchor (only those of the 20 that have an image vector).
                      NaN if the article has no image vector, or none of the 20 has.
    txt_sim_hist      the same with the MiniLM text vector (every article has one);
                      NaN only if the user has no purchase before the anchor.
    tt_sim_max_hist   the LARGEST similarity between the candidate and one of the
                      user's last 20 purchases, with the fold's R6 article vectors
                      (L2-normalised, so the dot product is the cosine). Unlike the
                      mean-vector features it is high when ONE past purchase is
                      close to the candidate. NaN if the user has no purchase.
    tt_sim_top5_hist  the mean of the 5 largest of those similarities (of all of
                      them if the history has fewer than 5). NaN likewise.

The dot products come from hmrec.vectors.dot_pairs, the function the R5 / R6
scores were computed with. "The user's last 20 purchases" is
hmrec.data.histories.recent_purchase_histories, the function R6 uses.

Mean vector: the plain mean of the (up to 20) embedding vectors, then
L2-normalised. The article's vector is L2-normalised as well, so their dot
product is the cosine.

Inputs are files written by recall (read in features/build.py); this module
does not import recall/.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from hmrec.features.base import AGE_BUCKET, ARTICLE_IDX, CUSTOMER_IDX, HistoryArrays
from hmrec.vectors import UserItemVectors, dot_pairs, lookup_rows

TT_DOT = "tt_dot"
WALS_DOT = "wals_dot"
POP_BUCKET_SCORE = "pop_bucket_score"
IMG_SIM_HIST = "img_sim_hist"
TXT_SIM_HIST = "txt_sim_hist"
TT_SIM_MAX_HIST = "tt_sim_max_hist"
TT_SIM_TOP_HIST = "tt_sim_top5_hist"
BUCKET_TABLE_SCORE = "score"


@dataclass(frozen=True)
class DenseSources:
    """What the B group reads for one fold.

    Attributes:
        twotower: R6 vectors saved for the fold (query customers, all articles).
        wals: R5 vectors saved for the fold (query customers that have one,
            articles that have one).
        bucket_scores: R1's S(g, i) table: age_bucket, article_idx, score (float32).
        image: Users = L2-normalised mean image vector of each customer's
            history (customers without one are absent); items = L2-normalised
            image vector of every article that has one.
        text: The same for the text vectors.
        history: The query customers' last purchases (for the tt_sim_* features).
        history_similarity_top_k: How many of the largest similarities
            tt_sim_top5_hist averages (5).
        pair_chunk_size: Pairs per dot_pairs chunk for tt_dot / wals_dot.
        embedding_pair_chunk_size: Pairs per chunk for the image / text cosine.
    """

    twotower: UserItemVectors
    wals: UserItemVectors
    bucket_scores: pl.DataFrame
    image: UserItemVectors
    text: UserItemVectors
    history: HistoryArrays
    history_similarity_top_k: int
    pair_chunk_size: int
    embedding_pair_chunk_size: int


def normalize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalise the rows of a float32 matrix.

    Returns:
        (normalised float32 [n, dim], has_norm bool [n]). A row whose norm is
        0 cannot be normalised: it stays 0 and has_norm is False.
    """
    norms = np.linalg.norm(matrix, axis=1)
    has_norm = norms > 0
    safe_norms = np.where(has_norm, norms, 1.0).astype(np.float32)
    return (matrix / safe_norms[:, None]).astype(np.float32), has_norm


def history_mean_vectors(
    history: np.ndarray,
    history_mask: np.ndarray,
    embeddings: np.ndarray,
    embedding_mask: np.ndarray,
    users_per_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalised mean embedding of each user's history.

    mean_u = (1 / n_u) * sum of embeddings[i] over the history slots i of user u
    that are filled AND whose article has an embedding; n_u = number of such
    slots. An article bought on two different days fills two slots and counts
    twice. Then mean_u is divided by its L2 norm.

    Args:
        history: int32 [Q, L] from recent_purchase_histories; empty slots hold
            the padding index n_articles.
        history_mask: bool [Q, L], True for filled slots.
        embeddings: float32 [n_articles, dim], row = article_idx.
        embedding_mask: bool [n_articles], True where the article has a vector.
        users_per_chunk: Users per chunk; bounds memory to
            users_per_chunk * L * dim * 4 bytes.

    Returns:
        (vectors float32 [Q, dim], has_vector bool [Q]). has_vector is False
        for a user with no usable slot (no history, or no article of the
        history has an embedding); that user's row is all 0.
    """
    if users_per_chunk <= 0:
        raise ValueError(f"users_per_chunk must be > 0, got {users_per_chunk}")
    n_users = history.shape[0]
    # The padding index n_articles gets a False mask, so a padded slot is never used.
    padded_mask = np.append(embedding_mask, False)
    means = np.zeros((n_users, embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(n_users, dtype=np.int32)
    for start in range(0, n_users, users_per_chunk):
        stop = start + users_per_chunk
        slots = history[start:stop]
        usable = history_mask[start:stop] & padded_mask[slots]
        # Unusable slots point at row 0 and get weight 0, so they add nothing.
        gathered = embeddings[np.where(usable, slots, 0)]  # float32 [q, L, dim]
        totals = np.einsum("qld,ql->qd", gathered, usable.astype(np.float32))
        chunk_counts = usable.sum(axis=1).astype(np.int32)
        counts[start:stop] = chunk_counts
        means[start:stop] = totals / np.maximum(chunk_counts, 1)[:, None]
    normalised, has_norm = normalize_rows(means)
    return normalised, (counts > 0) & has_norm


def similarity_vectors(
    customer_idx: np.ndarray,
    history: np.ndarray,
    history_mask: np.ndarray,
    embeddings: np.ndarray,
    embedding_mask: np.ndarray,
    users_per_chunk: int,
) -> UserItemVectors:
    """The vectors whose dot product is img_sim_hist / txt_sim_hist.

    Args:
        customer_idx: int32 [Q], strictly increasing; row q of `history` is
            this customer's history.
        history: int32 [Q, L], see history_mean_vectors.
        history_mask: bool [Q, L].
        embeddings: float32 [n_articles, dim].
        embedding_mask: bool [n_articles].
        users_per_chunk: Users per chunk when averaging.

    Returns:
        Users: the customers that have a mean vector, with it (L2-normalised).
        Items: the articles that have an embedding, L2-normalised. A user or
        article that is absent gives NaN in dot_pairs, which is the "missing"
        rule of proposal §8.3.
    """
    user_vectors, has_vector = history_mean_vectors(
        history, history_mask, embeddings, embedding_mask, users_per_chunk
    )
    item_vectors, has_norm = normalize_rows(embeddings)
    item_rows = np.flatnonzero(embedding_mask & has_norm)
    return UserItemVectors(
        user_customer_idx=customer_idx[has_vector].astype(np.int32),
        user_vectors=np.ascontiguousarray(user_vectors[has_vector]),
        item_article_idx=item_rows.astype(np.int32),
        item_vectors=np.ascontiguousarray(item_vectors[item_rows]),
    )


def add_dense_features(
    chunk: pl.DataFrame, sources: DenseSources, customers: pl.DataFrame
) -> pl.DataFrame:
    """Add the five B-group columns to a chunk of candidates.

    Args:
        chunk: One row per (query_id, article_idx) with customer_idx.
        sources: The fold's vectors and score table.
        customers: customer_idx -> age_bucket.

    Returns:
        `chunk` with tt_dot, wals_dot, pop_bucket_score, img_sim_hist and
        txt_sim_hist (float32) added; row order unchanged.
    """
    customer_idx = chunk[CUSTOMER_IDX].to_numpy()
    article_idx = chunk[ARTICLE_IDX].to_numpy()
    pairs, embedding_pairs = sources.pair_chunk_size, sources.embedding_pair_chunk_size
    dots = {
        TT_DOT: dot_pairs(sources.twotower, customer_idx, article_idx, pairs),
        WALS_DOT: dot_pairs(sources.wals, customer_idx, article_idx, pairs),
        IMG_SIM_HIST: dot_pairs(sources.image, customer_idx, article_idx, embedding_pairs),
        TXT_SIM_HIST: dot_pairs(sources.text, customer_idx, article_idx, embedding_pairs),
    }
    with_dots = chunk.with_columns(
        [pl.Series(name, values, dtype=pl.Float32) for name, values in dots.items()]
    )
    bucket_table = sources.bucket_scores.rename({BUCKET_TABLE_SCORE: POP_BUCKET_SCORE})
    with_bucket = with_dots.join(
        customers.select(CUSTOMER_IDX, AGE_BUCKET),
        on=CUSTOMER_IDX,
        how="left",
        maintain_order="left",
    ).join(bucket_table, on=[AGE_BUCKET, ARTICLE_IDX], how="left", maintain_order="left")
    # No table entry = nobody of the user's age bucket bought the article in R1's
    # window, so its decayed popularity there is 0 (a sum over no purchase lines).
    return with_bucket.with_columns(pl.col(POP_BUCKET_SCORE).cast(pl.Float32).fill_null(0.0)).drop(
        AGE_BUCKET
    )


# ---------------------------------------------------------------- similarity to past purchases


def _query_runs(customer_idx: np.ndarray) -> list[tuple[int, int]]:
    """(start, stop) of every run of consecutive rows with the same customer."""
    if customer_idx.size == 0:
        return []
    changes = np.flatnonzero(customer_idx[1:] != customer_idx[:-1]) + 1
    starts = np.concatenate([[0], changes])
    stops = np.concatenate([changes, [customer_idx.size]])
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def history_similarities(
    customer_idx: np.ndarray,
    article_idx: np.ndarray,
    item_vectors: UserItemVectors,
    history: HistoryArrays,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Largest and mean-of-top-k similarity between each candidate and the user's past purchases.

    For a candidate article c of a user with history articles h_1 ... h_n (n <= 20):
        sim_j = v_c . v_{h_j}                      (R6 article vectors)
        tt_sim_max_hist  = max_j sim_j
        tt_sim_top5_hist = mean of the min(top_k, n) largest sim_j
    An article that fills two history slots (bought on two days) counts twice.

    The rows are processed one user at a time (the candidates of a query are
    consecutive rows of the wide table), so memory is one small
    [candidates of the query, n] matrix, whatever the chunk size.

    Args:
        customer_idx: int [n_rows], the customer of each candidate row.
        article_idx: int [n_rows], the candidate article of each row.
        item_vectors: The fold's R6 vectors; only the article side is used and
            it must hold every candidate and every history article.
        history: The customers' last purchases.
        top_k: Number of largest similarities averaged (5).

    Returns:
        (largest float32 [n_rows], top_mean float32 [n_rows]); NaN for a user
        without any purchase.

    Raises:
        ValueError: If a customer has no history row, or an article has no vector.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    candidate_rows = lookup_rows(item_vectors.item_article_idx, article_idx)
    user_rows = lookup_rows(history.customer_idx, customer_idx)
    if (candidate_rows < 0).any() or (user_rows < 0).any():
        raise ValueError("a candidate article has no R6 vector, or a customer has no history row")
    largest = np.full(customer_idx.size, np.nan, dtype=np.float32)
    top_mean = np.full(customer_idx.size, np.nan, dtype=np.float32)
    for start, stop in _query_runs(customer_idx):
        user_row = user_rows[start]
        bought = history.history[user_row][history.history_mask[user_row]]
        if bought.size == 0:
            continue  # no purchase before the anchor: both features stay NaN
        history_rows = lookup_rows(item_vectors.item_article_idx, bought)
        if (history_rows < 0).any():
            raise ValueError("a history article has no R6 vector")
        candidates = item_vectors.item_vectors[candidate_rows[start:stop]]
        # float32 [candidates of the query, n history articles]
        similarities = candidates @ item_vectors.item_vectors[history_rows].T
        largest[start:stop] = similarities.max(axis=1)
        kept = min(top_k, bought.size)
        # Sorted ascending, so the last `kept` columns are the largest; summing them
        # in sorted order makes the mean independent of the order of the history.
        top_mean[start:stop] = np.sort(similarities, axis=1)[:, -kept:].mean(axis=1)
    return largest, top_mean


def add_history_similarity_features(chunk: pl.DataFrame, sources: DenseSources) -> pl.DataFrame:
    """Add tt_sim_max_hist and tt_sim_top5_hist to a chunk of candidates.

    Args:
        chunk: One row per (query_id, article_idx) with customer_idx; the rows
            of a query are consecutive.
        sources: The fold's R6 vectors and histories.

    Returns:
        `chunk` with the two float32 columns added; row order unchanged.
    """
    largest, top_mean = history_similarities(
        chunk[CUSTOMER_IDX].to_numpy(),
        chunk[ARTICLE_IDX].to_numpy(),
        sources.twotower,
        sources.history,
        sources.history_similarity_top_k,
    )
    return chunk.with_columns(
        pl.Series(TT_SIM_MAX_HIST, largest, dtype=pl.Float32),
        pl.Series(TT_SIM_TOP_HIST, top_mean, dtype=pl.Float32),
    )
