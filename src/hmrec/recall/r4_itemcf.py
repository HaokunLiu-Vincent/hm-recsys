"""R4 item-based collaborative filtering (proposal §5 R4), channel "icf".

Windows (configs/recall/r4_itemcf.yaml):
    fit window  [a - fit_window_days, a)   (default 365 days): C_w, n_w, d_u
    seed window [a - seed_window_days, a)  (default 90 days):  the user's seeds
                                           (HPO grid {90, 180, 365})

Interactions are 0/1: R_ui = 1 if user u bought article i at least once in the
fit window (repeat purchases do not count twice).

Formulas (§5 R4):
    w_u       = 1 / ln(1 + d_u)  for CF3 (IUF), d_u = distinct articles of u
              = 1                for CF0
    C_w(i, j) = sum_u w_u * R_ui * R_uj           (weighted co-occurrence)
    n_w(i)    = sum_u w_u * R_ui                  (weighted margin)
    s(i -> j) = C_w(i, j) / (n_w(i)^alpha * n_w(j)^(1 - alpha)),   j != i
    S(u, j)   = sum of q_ui * s(i -> j) over the distinct seeds i of u
    q_ui      = 1                                  seed decay off (default)
              = exp(-age_ui / seed_tau_days)       seed decay on; age_ui = days since
                                                   u LAST bought i (yesterday = 1)
The seed decay only weights the seeds; the graph (C_w, n_w, d_u) is never decayed.
s is not symmetric when alpha != 0.5: s(i -> j) != s(j -> i).

Active pool (§2.4, §5 R4): only j in A(a) can be a neighbour, and the filter is
applied BEFORE the Top-K_n cut, so each seed keeps its best K_n ACTIVE
neighbours. C_w, n_w and d_u use all articles of the fit window (A(a) does not
change them), and a seed does not have to be active itself.

Memory (§1.3): C_w is never built in full. Seeds are processed in chunks of
`item_chunk_size`; for each chunk the rows C_w(i, .) are computed, cut to the
Top-K_n active neighbours, and freed. Only the neighbour tables are kept.

Several alphas in one pass: C_w and n_w do not depend on alpha, so each chunk's
C_w rows are re-normalised and re-sorted once per alpha (generate_for_alphas).

Exactness: C_w(i, .) is a sparse product (scipy). Each entry is summed over
users in the same order whatever the chunk size, so chunked and unchunked
results are bit-identical (tested).

Order: default (score DESC, article_idx ASC); ties between neighbours are also
broken by article_idx ASC. Already-bought articles are NOT excluded.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import sparse

from hmrec.config import ItemCFConfig
from hmrec.data.asof import history_window
from hmrec.logging_utils import get_logger
from hmrec.recall.base import (
    AGE_DAYS,
    ARTICLE_IDX,
    CUSTOMER_IDX,
    QUERY_ID,
    SCORE,
    OutputSpec,
    RecallInputs,
    add_age_days,
    rank_candidates,
)

SEED_IDX = "seed_idx"
NEIGHBOR_IDX = "neighbor_idx"
COOCCURRENCE = "cooccurrence"
SIMILARITY = "similarity"
USER_WEIGHT = "user_weight"
DISTINCT_ITEMS = "distinct_items"
NEIGHBOR_COUNT = "neighbor_count"
SEED_WEIGHT = "seed_weight"

ICF_SPEC = OutputSpec(channel="icf")

PAIRS_SCHEMA = pl.Schema({SEED_IDX: pl.Int32, NEIGHBOR_IDX: pl.Int32, COOCCURRENCE: pl.Float64})
NEIGHBORS_SCHEMA = pl.Schema({SEED_IDX: pl.Int32, NEIGHBOR_IDX: pl.Int32, SIMILARITY: pl.Float64})

LOGGER = get_logger("recall.r4_itemcf")


# ---------------------------------------------------------------- interactions


@dataclass(frozen=True)
class WeightedInteractions:
    """The 0/1 user-article matrix of the fit window, with user weights.

    Users are indexed by customer_idx directly (rows of customers who bought
    nothing are empty), articles by article_idx.

    Attributes:
        weighted_item_user: CSR, shape [n_articles, n_users], float64.
            Entry (i, u) = w_u if u bought i in the fit window.
        user_item: CSR, shape [n_users, n_articles], float64. Entry (u, j) = 1.0
            if u bought j in the fit window.
        item_weight_sum: n_w(i), shape [n_articles], float64. 0 for articles not
            bought in the fit window.
        n_articles: Size of the article axis (catalog size).
    """

    weighted_item_user: sparse.csr_array
    user_item: sparse.csr_array
    item_weight_sum: np.ndarray
    n_articles: int


def distinct_user_items(fit_history: pl.DataFrame) -> pl.DataFrame:
    """R_ui = 1: one row per (customer_idx, article_idx) bought at least once.

    Args:
        fit_history: One row per purchase line in the fit window.

    Returns:
        One row per distinct (customer_idx, article_idx), sorted by both columns.
    """
    return fit_history.select(CUSTOMER_IDX, ARTICLE_IDX).unique().sort(CUSTOMER_IDX, ARTICLE_IDX)


def user_weights(user_items: pl.DataFrame, variant: str) -> pl.DataFrame:
    """User weight w_u (§5 R4): 1 / ln(1 + d_u) for CF3, 1 for CF0.

    d_u >= 1 for every user in `user_items`, so ln(1 + d_u) >= ln 2 > 0.

    Args:
        user_items: One row per distinct (customer_idx, article_idx).
        variant: "CF0" or "CF3".

    Returns:
        One row per customer. Columns customer_idx, distinct_items (d_u),
        user_weight (float64).
    """
    distinct_counts = user_items.group_by(CUSTOMER_IDX).agg(pl.len().alias(DISTINCT_ITEMS))
    if variant == "CF0":
        weight = pl.lit(1.0, dtype=pl.Float64)
    elif variant == "CF3":
        weight = 1.0 / (1.0 + pl.col(DISTINCT_ITEMS).cast(pl.Float64)).log()
    else:
        raise ValueError(f"unknown ItemCF variant {variant!r}, expected 'CF0' or 'CF3'")
    return distinct_counts.with_columns(weight.alias(USER_WEIGHT))


def build_weighted_interactions(
    user_items: pl.DataFrame, weights: pl.DataFrame, n_articles: int
) -> WeightedInteractions:
    """Build the sparse matrices and n_w (see WeightedInteractions).

    Args:
        user_items: One row per distinct (customer_idx, article_idx).
        weights: customer_idx -> user_weight, from user_weights.
        n_articles: Size of the article axis (catalog size).

    Returns:
        The interaction matrices of the fit window.

    Raises:
        ValueError: If an article_idx is outside [0, n_articles).
    """
    with_weight = user_items.join(weights.select(CUSTOMER_IDX, USER_WEIGHT), on=CUSTOMER_IDX)
    users = with_weight[CUSTOMER_IDX].to_numpy().astype(np.int32)
    items = with_weight[ARTICLE_IDX].to_numpy().astype(np.int32)
    if items.size > 0 and (items.min() < 0 or items.max() >= n_articles):
        raise ValueError(f"article_idx outside [0, {n_articles}) in the fit window")
    n_users = int(users.max()) + 1 if users.size > 0 else 0
    weight_values = with_weight[USER_WEIGHT].to_numpy().astype(np.float64)
    weighted_item_user = sparse.csr_array(
        (weight_values, (items, users)), shape=(n_articles, n_users)
    )
    user_item = sparse.csr_array(
        (np.ones(users.size, dtype=np.float64), (users, items)), shape=(n_users, n_articles)
    )
    # Sorted column indices fix the order in which each C_w entry is summed,
    # which makes chunked and unchunked results bit-identical.
    weighted_item_user.sort_indices()
    user_item.sort_indices()
    # n_w(i) = sum_u w_u * R_ui = row sums of weighted_item_user (§5 R4).
    item_weight_sum = np.asarray(weighted_item_user.sum(axis=1), dtype=np.float64).ravel()
    return WeightedInteractions(weighted_item_user, user_item, item_weight_sum, n_articles)


# ---------------------------------------------------------------- neighbours


def cooccurrence_pairs(
    interactions: WeightedInteractions,
    seed_items: np.ndarray,
    active_mask: np.ndarray | None,
) -> pl.DataFrame:
    """C_w(i, j) for the given seeds i and every j with C_w(i, j) > 0 and j != i.

    One sparse product: C_w(seeds, .) = weighted_item_user[seeds] @ user_item.

    Args:
        interactions: From build_weighted_interactions.
        seed_items: Seed article_idx values, shape [n_seeds], int.
        active_mask: Shape [n_articles], bool; True = article is in A(a). Only
            active neighbours are kept. None keeps every neighbour (used by tests
            to build the full matrix).

    Returns:
        One row per (seed_idx, neighbor_idx). Columns seed_idx (int32),
        neighbor_idx (int32), cooccurrence (float64).
    """
    rows = interactions.weighted_item_user[seed_items] @ interactions.user_item
    coo = sparse.coo_array(rows)
    seeds = np.asarray(seed_items, dtype=np.int32)[coo.row]
    neighbors = coo.col.astype(np.int32)
    keep = neighbors != seeds  # s(i -> i) is not a neighbour (§5 R4)
    if active_mask is not None:
        keep &= active_mask[neighbors]
    return pl.DataFrame(
        {SEED_IDX: seeds[keep], NEIGHBOR_IDX: neighbors[keep], COOCCURRENCE: coo.data[keep]},
        schema=PAIRS_SCHEMA,
    )


def add_similarity(pairs: pl.DataFrame, item_weight_sum: np.ndarray, alpha: float) -> pl.DataFrame:
    """s(i -> j) = C_w(i, j) / (n_w(i)^alpha * n_w(j)^(1 - alpha))  (§5 R4).

    Args:
        pairs: From cooccurrence_pairs.
        item_weight_sum: n_w, shape [n_articles], float64.
        alpha: Exponent in [0, 1].

    Returns:
        One row per (seed_idx, neighbor_idx); columns seed_idx, neighbor_idx,
        similarity (float64).
    """
    seeds = pairs[SEED_IDX].to_numpy()
    neighbors = pairs[NEIGHBOR_IDX].to_numpy()
    seed_margin = np.power(item_weight_sum[seeds], alpha)
    neighbor_margin = np.power(item_weight_sum[neighbors], 1.0 - alpha)
    similarity = pairs[COOCCURRENCE].to_numpy() / (seed_margin * neighbor_margin)
    return pairs.select(SEED_IDX, NEIGHBOR_IDX).with_columns(
        pl.Series(SIMILARITY, similarity, dtype=pl.Float64)
    )


def top_neighbors(similarities: pl.DataFrame, neighbors_per_seed: int) -> pl.DataFrame:
    """Keep the Top-K_n neighbours of each seed: similarity DESC, neighbor_idx ASC.

    Returns:
        At most `neighbors_per_seed` rows per seed, sorted by
        (seed_idx ASC, similarity DESC, neighbor_idx ASC).
    """
    ordered = similarities.sort(
        [SEED_IDX, SIMILARITY, NEIGHBOR_IDX], descending=[False, True, False]
    )
    position = pl.int_range(pl.len()).over(SEED_IDX)
    return ordered.filter(position < neighbors_per_seed)


def build_neighbor_tables(
    interactions: WeightedInteractions,
    seed_items: np.ndarray,
    active_mask: np.ndarray,
    alphas: Sequence[float],
    neighbors_per_seed: int,
    item_chunk_size: int,
) -> dict[float, pl.DataFrame]:
    """Top-K_n active neighbours of every seed, for each alpha, in seed chunks (§1.3).

    Each chunk's C_w rows are computed once and re-used for all alphas, then freed.

    Args:
        interactions: From build_weighted_interactions.
        seed_items: Distinct seed article_idx values, shape [n_seeds].
        active_mask: Shape [n_articles], bool; True = in A(a).
        alphas: Alphas to normalise with (distinct values).
        neighbors_per_seed: K_n.
        item_chunk_size: Seeds per chunk.

    Returns:
        alpha -> neighbour table (NEIGHBORS_SCHEMA), sorted by
        (seed_idx, similarity DESC, neighbor_idx).
    """
    if item_chunk_size <= 0:
        raise ValueError(f"item_chunk_size must be > 0, got {item_chunk_size}")
    if len(set(alphas)) != len(alphas):
        raise ValueError(f"alphas must be distinct, got {list(alphas)}")
    parts: dict[float, list[pl.DataFrame]] = {alpha: [] for alpha in alphas}
    n_chunks = (len(seed_items) + item_chunk_size - 1) // item_chunk_size
    for chunk_number, start in enumerate(range(0, len(seed_items), item_chunk_size), 1):
        chunk = seed_items[start : start + item_chunk_size]
        pairs = cooccurrence_pairs(interactions, chunk, active_mask)
        for alpha in alphas:
            similarities = add_similarity(pairs, interactions.item_weight_sum, alpha)
            parts[alpha].append(top_neighbors(similarities, neighbors_per_seed))
        LOGGER.debug("neighbour chunk %d / %d: %d pairs", chunk_number, n_chunks, pairs.height)
    return {
        alpha: pl.concat(tables) if tables else NEIGHBORS_SCHEMA.to_frame()
        for alpha, tables in parts.items()
    }


def count_neighbors(neighbors: pl.DataFrame, seed_items: np.ndarray) -> pl.DataFrame:
    """Number of kept neighbours per seed, including seeds with none.

    Returns:
        One row per seed. Columns seed_idx (int32), neighbor_count (uint32).
    """
    seeds = pl.DataFrame({SEED_IDX: seed_items}, schema={SEED_IDX: pl.Int32})
    counts = neighbors.group_by(SEED_IDX).agg(pl.len().alias(NEIGHBOR_COUNT))
    return (
        seeds.join(counts, on=SEED_IDX, how="left")
        .with_columns(pl.col(NEIGHBOR_COUNT).fill_null(0))
        .sort(SEED_IDX)
    )


# ---------------------------------------------------------------- user scores


def seed_weights(
    seed_history: pl.DataFrame, anchor_day: int, seed_tau_days: float | None
) -> pl.DataFrame:
    """Seed weight q_ui of each distinct (customer, seed) pair (§5 R4).

    Args:
        seed_history: Purchase lines in the seed window; columns customer_idx,
            article_idx, t_dat_day (int16).
        anchor_day: The anchor as a day index.
        seed_tau_days: None = no decay (q_ui = 1). Otherwise
            q_ui = exp(-age_ui / seed_tau_days), where age_ui is the age of the
            customer's MOST RECENT purchase of the seed. Buying the same seed
            several times does not add up: only the last purchase counts.

    Returns:
        One row per distinct (customer_idx, article_idx). Column seed_weight (float64).
    """
    with_age = add_age_days(seed_history, anchor_day)
    last_purchase = with_age.group_by(CUSTOMER_IDX, ARTICLE_IDX).agg(pl.col(AGE_DAYS).min())
    if seed_tau_days is None:
        weight = pl.lit(1.0, dtype=pl.Float64)
    else:
        weight = (-pl.col(AGE_DAYS).cast(pl.Float64) / seed_tau_days).exp()
    return last_purchase.select(CUSTOMER_IDX, ARTICLE_IDX, weight.alias(SEED_WEIGHT))


def query_seed_items(customer_seeds: pl.DataFrame, queries: pl.DataFrame) -> pl.DataFrame:
    """The distinct seeds of each query, with their weights.

    Args:
        customer_seeds: From seed_weights.
        queries: One row per query; columns query_id, customer_idx.

    Returns:
        One row per (query_id, article_idx), sorted by both; column seed_weight.
        Queries without a purchase in the seed window have no rows.
    """
    return (
        queries.select(QUERY_ID, CUSTOMER_IDX)
        .join(customer_seeds, on=CUSTOMER_IDX)
        .select(QUERY_ID, ARTICLE_IDX, SEED_WEIGHT)
        .sort(QUERY_ID, ARTICLE_IDX)
    )


def similarity_matrix(neighbors: pl.DataFrame, n_articles: int) -> sparse.csr_array:
    """The kept s(i -> j) as a sparse matrix, shape [n_articles, n_articles], float64."""
    matrix = sparse.csr_array(
        (
            neighbors[SIMILARITY].to_numpy(),
            (neighbors[SEED_IDX].to_numpy(), neighbors[NEIGHBOR_IDX].to_numpy()),
        ),
        shape=(n_articles, n_articles),
    )
    matrix.sort_indices()
    return matrix


def score_query_chunk(
    query_seeds: pl.DataFrame, similarity: sparse.csr_array, n_articles: int
) -> pl.DataFrame:
    """S(u, j) = sum of q_ui * s(i -> j) over the query's distinct seeds i  (§5 R4).

    Computed as one sparse product: seed weights [n_queries, n_articles] @ similarity.
    With the seed decay off every weight is exactly 1.0.

    Args:
        query_seeds: One row per (query_id, article_idx) seed with seed_weight,
            for a chunk of queries.
        similarity: From similarity_matrix, shape [n_articles, n_articles].
        n_articles: Catalog size.

    Returns:
        One row per (query_id, article_idx) with S > 0; columns query_id,
        article_idx, score (float64).
    """
    query_ids, query_rows = np.unique(query_seeds[QUERY_ID].to_numpy(), return_inverse=True)
    seeds = sparse.csr_array(
        (
            query_seeds[SEED_WEIGHT].to_numpy(),
            (query_rows, query_seeds[ARTICLE_IDX].to_numpy()),
        ),
        shape=(query_ids.size, n_articles),
    )
    seeds.sort_indices()
    scores = sparse.coo_array(seeds @ similarity)
    return pl.DataFrame(
        {QUERY_ID: query_ids[scores.row], ARTICLE_IDX: scores.col, SCORE: scores.data},
        schema={QUERY_ID: pl.Int32, ARTICLE_IDX: pl.Int32, SCORE: pl.Float64},
    )


def score_queries(
    query_seeds: pl.DataFrame,
    similarity: sparse.csr_array,
    n_articles: int,
    quota: int,
    query_chunk_size: int,
) -> pl.DataFrame:
    """Score and rank all queries in chunks of `query_chunk_size` queries (§1.3).

    Returns:
        Candidates valid for ICF_SPEC (at most `quota` per query).
    """
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be > 0, got {query_chunk_size}")
    query_ids = query_seeds[QUERY_ID].unique().sort()
    ranked_chunks: list[pl.DataFrame] = []
    for start in range(0, query_ids.len(), query_chunk_size):
        chunk_ids = query_ids.slice(start, query_chunk_size)
        chunk = query_seeds.filter(pl.col(QUERY_ID).is_in(chunk_ids.implode()))
        scored = score_query_chunk(chunk, similarity, n_articles)
        ranked_chunks.append(rank_candidates(scored, ICF_SPEC, quota))
    if not ranked_chunks:
        return ICF_SPEC.schema.to_frame()
    return pl.concat(ranked_chunks)


# ---------------------------------------------------------------- channel


def catalog_size(articles: pl.DataFrame) -> int:
    """Size of the article axis: max(article_idx) + 1 of the catalog."""
    if articles.height == 0:
        raise ValueError("the article catalog is empty")
    max_idx = articles[ARTICLE_IDX].max()
    if not isinstance(max_idx, int):
        raise ValueError(f"article_idx max is {max_idx!r}, expected an int")
    return max_idx + 1


def active_pool_mask(active_pool: pl.DataFrame, n_articles: int) -> np.ndarray:
    """Boolean array, shape [n_articles]: True for articles in A(a)."""
    mask = np.zeros(n_articles, dtype=bool)
    mask[active_pool[ARTICLE_IDX].to_numpy()] = True
    return mask


@dataclass(frozen=True)
class ItemCFNeighbors:
    """Everything needed to score one anchor's queries, for one or more alphas.

    Attributes:
        query_seeds: One row per (query_id, article_idx) seed, with seed_weight.
        seed_items: Distinct seed article_idx values, shape [n_seeds], int32.
        tables: alpha -> Top-K_n active neighbour table (NEIGHBORS_SCHEMA).
        n_articles: Catalog size.
    """

    query_seeds: pl.DataFrame
    seed_items: np.ndarray
    tables: dict[float, pl.DataFrame]
    n_articles: int


class ItemCFChannel:
    """R4 recall channel, see the module docstring."""

    def __init__(self, config: ItemCFConfig, item_chunk_size: int, query_chunk_size: int) -> None:
        """Store the R4 config and the chunk sizes (chunking.* in configs/base.yaml)."""
        self.config = config
        self.item_chunk_size = item_chunk_size
        self.query_chunk_size = query_chunk_size

    @property
    def output_spec(self) -> OutputSpec:
        """R4 output: base columns only, default order."""
        return ICF_SPEC

    def fit_interactions(self, inputs: RecallInputs) -> WeightedInteractions:
        """Build R_ui, w_u and n_w from the fit window. Does not use A(a)."""
        fit_history = history_window(
            inputs.transactions,
            inputs.anchor,
            inputs.day_zero,
            self.config.fit_window_days,
            columns=[CUSTOMER_IDX, ARTICLE_IDX],
        )
        user_items = distinct_user_items(fit_history)
        weights = user_weights(user_items, self.config.variant)
        return build_weighted_interactions(user_items, weights, catalog_size(inputs.articles))

    def build_neighbors(
        self, queries: pl.DataFrame, inputs: RecallInputs, alphas: Sequence[float]
    ) -> ItemCFNeighbors:
        """Neighbour tables for the seeds of `queries`, one per alpha.

        Only seeds that some query uses are processed; the output is the same as
        processing every article of the fit window and then looking them up.
        """
        interactions = self.fit_interactions(inputs)
        seed_history = history_window(
            inputs.transactions,
            inputs.anchor,
            inputs.day_zero,
            self.config.seed_window_days,
            columns=[CUSTOMER_IDX, ARTICLE_IDX],
        )
        # Only the queried customers' seeds are needed.
        seed_history = seed_history.join(queries.select(CUSTOMER_IDX), on=CUSTOMER_IDX, how="semi")
        seed_tau_days = self.config.seed_tau_days if self.config.seed_decay else None
        customer_seeds = seed_weights(seed_history, inputs.anchor_day, seed_tau_days)
        query_seeds = query_seed_items(customer_seeds, queries)
        seed_items = query_seeds[ARTICLE_IDX].unique().sort().to_numpy().astype(np.int32)
        tables = build_neighbor_tables(
            interactions,
            seed_items,
            active_pool_mask(inputs.active_pool, interactions.n_articles),
            alphas,
            self.config.neighbors_per_seed,
            self.item_chunk_size,
        )
        return ItemCFNeighbors(query_seeds, seed_items, tables, interactions.n_articles)

    def candidates_from_neighbors(self, neighbors: ItemCFNeighbors, alpha: float) -> pl.DataFrame:
        """Score and rank the queries with the neighbour table of `alpha`."""
        similarity = similarity_matrix(neighbors.tables[alpha], neighbors.n_articles)
        return score_queries(
            neighbors.query_seeds,
            similarity,
            neighbors.n_articles,
            self.config.quota,
            self.query_chunk_size,
        )

    def generate_for_alphas(
        self, queries: pl.DataFrame, inputs: RecallInputs, alphas: Sequence[float]
    ) -> dict[float, pl.DataFrame]:
        """Candidates for several alphas, re-using C_w and n_w (§5 R4 HPO).

        Returns:
            alpha -> candidates valid for ICF_SPEC.
        """
        neighbors = self.build_neighbors(queries, inputs, alphas)
        return {alpha: self.candidates_from_neighbors(neighbors, alpha) for alpha in alphas}

    def generate(self, queries: pl.DataFrame, inputs: RecallInputs) -> pl.DataFrame:
        """Return R4 candidates for `queries` with the configured alpha."""
        alpha = self.config.alpha
        return self.generate_for_alphas(queries, inputs, [alpha])[alpha]
