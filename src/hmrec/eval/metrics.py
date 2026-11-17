"""Ranking metrics per query, and their averages (proposal §2.5, §4.1).

Per-query metrics (one value per query in `labels`):
    Recall@K      = |TopK ∩ G_q| / |G_q|
    AP@K          = sum over hit positions r <= K of P(r), divided by min(K, |G_q|),
                    where P(r) = (hits in the top r) / r. K = 12 gives the Kaggle AP@12.
    NDCG@K        = DCG@K / IDCG@K, binary relevance,
                    DCG@K  = sum over hit positions r <= K of 1 / log2(r + 1),
                    IDCG@K = sum over r = 1 .. min(K, |G_q|) of 1 / log2(r + 1).
    Union recall  = |candidates ∩ G_q| / |G_q| (all candidates, no cut-off).

Rules (proposal §2.5):
    - The set of queries is the set of query_ids in `labels`. A query without
      any prediction scores 0; it is never dropped from the average.
    - Short outputs are not padded: a missing slot is simply a miss.
    - A prediction for a query_id that is not in `labels` is an error (it means
      the predictions and labels come from different folds or a bug).

Averages:
    - macro_mean: average of per-query values (proposal §2.5).
    - query_weighted_mean: pool the queries of several folds and average them;
      the primary 4-week score (proposal §4.1).
    - equal_fold_mean: average of the per-fold means; reported only.

Floating point: metric values are float64, not float32. They are small, computed
on CPU, and the §4.2 "strictly better" comparison is sensitive to rounding
(AGENTS.md §4). Results are exactly reproducible: inside a query, terms are
summed in rank order; across queries, the sum uses math.fsum (exactly rounded,
so independent of order) after sorting by query_id. Shuffling the input rows
never changes any result.
"""

import math

import polars as pl

QUERY_ID = "query_id"
FOLD = "fold"
ARTICLE_IDX = "article_idx"
RANK = "rank"
VALUE = "value"
N_QUERIES = "n_queries"

METRIC_DTYPE = pl.Float64

# Internal column names.
_N_LABELS = "n_labels"
_DISCOUNT = "discount"


# ---------------------------------------------------------------- input checks


def _require_columns(table: pl.DataFrame, columns: tuple[str, ...], name: str) -> None:
    """Fail if `table` lacks any of `columns`."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}; it has {table.columns}")


def check_labels(labels: pl.DataFrame) -> None:
    """Check the label table: one row per (query_id, article_idx).

    Raises:
        ValueError: If columns are missing, the table is empty (there are no
            queries to average over), a (query_id, article_idx) pair repeats, or
            a query_id belongs to more than one fold.
    """
    _require_columns(labels, (QUERY_ID, FOLD, ARTICLE_IDX), "labels")
    if labels.height == 0:
        raise ValueError("labels is empty: there are no queries to evaluate")
    if labels.select(QUERY_ID, ARTICLE_IDX).is_duplicated().any():
        raise ValueError("labels: (query_id, article_idx) must be unique (G_q is a set)")
    folds_per_query = labels.group_by(QUERY_ID).agg(pl.col(FOLD).n_unique())
    if (folds_per_query[FOLD] > 1).any():
        raise ValueError("labels: a query_id appears in more than one fold")


def _check_query_ids_known(table: pl.DataFrame, labels: pl.DataFrame, name: str) -> None:
    """Fail if `table` has a query_id that is not in `labels` (a query without G_q)."""
    unknown = table.join(labels.select(QUERY_ID).unique(), on=QUERY_ID, how="anti")
    if unknown.height > 0:
        examples = unknown[QUERY_ID].unique().sort().head(5).to_list()
        raise ValueError(
            f"{name} has query_ids without labels (empty or unknown G_q), e.g. {examples}"
        )


def check_predictions(predictions: pl.DataFrame, labels: pl.DataFrame) -> None:
    """Check a ranked prediction table: one row per (query_id, article_idx).

    Raises:
        ValueError: If columns are missing, an article repeats inside a query,
            the ranks of a query are not exactly 1, 2, ..., n, or a query_id is
            not in `labels`.
    """
    _require_columns(predictions, (QUERY_ID, ARTICLE_IDX, RANK), "predictions")
    if predictions.select(QUERY_ID, ARTICLE_IDX).is_duplicated().any():
        raise ValueError("predictions: article_idx must be unique within each query")
    rank_check = predictions.group_by(QUERY_ID).agg(
        pl.col(RANK).min().alias("min_rank"),
        pl.col(RANK).max().alias("max_rank"),
        pl.col(RANK).n_unique().alias("n_ranks"),
        pl.len().alias("n_rows"),
    )
    # Ranks 1..n without gaps or repeats <=> min = 1, max = n rows, all distinct.
    bad = rank_check.filter(
        (pl.col("min_rank") != 1)
        | (pl.col("max_rank") != pl.col("n_rows"))
        | (pl.col("n_ranks") != pl.col("n_rows"))
    )
    if bad.height > 0:
        examples = bad[QUERY_ID].sort().head(5).to_list()
        raise ValueError(f"predictions: ranks must be 1..n in every query; bad queries {examples}")
    _check_query_ids_known(predictions, labels, "predictions")


# ---------------------------------------------------------------- shared helpers


def _check_k(k: int) -> None:
    """Fail unless k is a positive integer."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")


def _label_counts(labels: pl.DataFrame) -> pl.DataFrame:
    """Return one row per query: query_id, fold, n_labels = |G_q|, sorted by query_id."""
    return labels.group_by(QUERY_ID, FOLD).agg(pl.len().alias(_N_LABELS)).sort(QUERY_ID)


def _hits_in_top_k(predictions: pl.DataFrame, labels: pl.DataFrame, k: int) -> pl.DataFrame:
    """Return the predictions with rank <= k that are in G_q: columns query_id, rank."""
    top_k = predictions.filter(pl.col(RANK) <= k).select(QUERY_ID, ARTICLE_IDX, RANK)
    hits = top_k.join(labels.select(QUERY_ID, ARTICLE_IDX), on=[QUERY_ID, ARTICLE_IDX], how="semi")
    return hits.select(QUERY_ID, RANK)


def _attach_to_all_queries(per_hit_query: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Left-join per-query sums onto every labelled query; queries without hits get 0.

    Returns columns query_id, fold, n_labels, plus the columns of `per_hit_query`.
    """
    all_queries = _label_counts(labels)
    joined = all_queries.join(per_hit_query, on=QUERY_ID, how="left")
    value_columns = [column for column in per_hit_query.columns if column != QUERY_ID]
    return joined.with_columns(pl.col(value_columns).fill_null(0))


def _finish(per_query: pl.DataFrame, value: pl.Expr) -> pl.DataFrame:
    """Return the final per-query table: query_id, fold, value (float64), sorted by query_id."""
    return per_query.select(QUERY_ID, FOLD, value.cast(METRIC_DTYPE).alias(VALUE)).sort(QUERY_ID)


# ---------------------------------------------------------------- per-query metrics


def recall_at_k(predictions: pl.DataFrame, labels: pl.DataFrame, k: int) -> pl.DataFrame:
    """Recall@K per query: |TopK ∩ G_q| / |G_q|.

    Args:
        predictions: One row per (query_id, article_idx), with rank (1-based,
            consecutive). Rows with rank > k are ignored.
        labels: One row per (query_id, article_idx) in G_q, with column fold.
        k: Cut-off, > 0.

    Returns:
        One row per query in `labels`. Columns: query_id, fold, value (float64),
        sorted by query_id.
    """
    _check_k(k)
    check_labels(labels)
    check_predictions(predictions, labels)
    return _recall_from_top_k_hits(_hits_in_top_k(predictions, labels, k), labels)


def _recall_from_top_k_hits(hits: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Recall@K from the hits with rank <= K (columns query_id, rank)."""
    hit_counts = hits.group_by(QUERY_ID).agg(pl.len().alias("n_hits"))
    per_query = _attach_to_all_queries(hit_counts, labels)
    return _finish(per_query, pl.col("n_hits") / pl.col(_N_LABELS))


def average_precision_at_k(predictions: pl.DataFrame, labels: pl.DataFrame, k: int) -> pl.DataFrame:
    """AP@K per query, with denominator min(K, |G_q|) (Kaggle H&M definition for K = 12).

    For the i-th hit (i = 1, 2, ...) at rank r_i <= K, the precision at that
    rank is i / r_i, because the hits above it are exactly hits 1..i.
    AP@K = sum_i (i / r_i) / min(K, |G_q|).

    Args:
        predictions: See recall_at_k.
        labels: See recall_at_k.
        k: Cut-off, > 0 (12 for MAP@12).

    Returns:
        One row per query in `labels`: query_id, fold, value (float64), sorted by query_id.
    """
    _check_k(k)
    check_labels(labels)
    check_predictions(predictions, labels)
    return _average_precision_from_top_k_hits(_hits_in_top_k(predictions, labels, k), labels, k)


def _average_precision_from_top_k_hits(
    hits: pl.DataFrame, labels: pl.DataFrame, k: int
) -> pl.DataFrame:
    """AP@K from the hits with rank <= K (columns query_id, rank)."""
    sorted_ranks = pl.col(RANK).sort()
    hit_number = pl.int_range(1, pl.len() + 1)
    # Sum the precisions in rank order (cum_sum().last()), so the result does
    # not depend on the input row order.
    precision_sums = hits.group_by(QUERY_ID).agg(
        (hit_number / sorted_ranks).cum_sum().last().alias("precision_sum")
    )
    per_query = _attach_to_all_queries(precision_sums, labels)
    denominator = pl.min_horizontal(pl.col(_N_LABELS), pl.lit(k))
    return _finish(per_query, pl.col("precision_sum") / denominator)


def _discount_table(k: int) -> pl.DataFrame:
    """Return rank 1..k with discount 1 / log2(rank + 1) (float64).

    DCG and IDCG both read their discounts from this one table, so a perfect
    ranking gives exactly NDCG = 1.0 (no tiny difference from two log functions).
    """
    ranks = list(range(1, k + 1))
    discounts = [1.0 / math.log2(rank + 1) for rank in ranks]
    return pl.DataFrame(
        {RANK: ranks, _DISCOUNT: discounts}, schema={RANK: pl.Int64, _DISCOUNT: METRIC_DTYPE}
    )


def _ideal_dcg_table(discounts: pl.DataFrame) -> pl.DataFrame:
    """Return n_ideal = 1..k with IDCG = discount(1) + ... + discount(n_ideal).

    Summed left to right, the same order as DCG, so a perfect ranking matches exactly.
    """
    running_sum = 0.0
    ideal_dcgs = []
    for discount in discounts[_DISCOUNT].to_list():
        running_sum += discount
        ideal_dcgs.append(running_sum)
    return pl.DataFrame(
        {"n_ideal": discounts[RANK].to_list(), "ideal_dcg": ideal_dcgs},
        schema={"n_ideal": pl.Int64, "ideal_dcg": METRIC_DTYPE},
    )


def ndcg_at_k(predictions: pl.DataFrame, labels: pl.DataFrame, k: int) -> pl.DataFrame:
    """NDCG@K per query with binary relevance.

    DCG@K = sum over hits at rank r <= K of 1 / log2(r + 1).
    IDCG@K = DCG of a perfect list: min(K, |G_q|) hits at ranks 1, 2, ...

    Args:
        predictions: See recall_at_k.
        labels: See recall_at_k.
        k: Cut-off, > 0.

    Returns:
        One row per query in `labels`: query_id, fold, value (float64), sorted by query_id.
    """
    _check_k(k)
    check_labels(labels)
    check_predictions(predictions, labels)
    return _ndcg_from_top_k_hits(_hits_in_top_k(predictions, labels, k), labels, k)


def _ndcg_from_top_k_hits(hits: pl.DataFrame, labels: pl.DataFrame, k: int) -> pl.DataFrame:
    """NDCG@K from the hits with rank <= K (columns query_id, rank)."""
    discounts = _discount_table(k)
    hits = hits.with_columns(pl.col(RANK).cast(pl.Int64))
    hits = hits.join(discounts, on=RANK)
    dcg = hits.group_by(QUERY_ID).agg(pl.col(_DISCOUNT).sort_by(RANK).cum_sum().last().alias("dcg"))
    per_query = _attach_to_all_queries(dcg, labels)
    per_query = per_query.with_columns(
        pl.min_horizontal(pl.col(_N_LABELS), pl.lit(k)).cast(pl.Int64).alias("n_ideal")
    ).join(_ideal_dcg_table(discounts), on="n_ideal")
    return _finish(per_query, pl.col("dcg") / pl.col("ideal_dcg"))


# ---------------------------------------------------------------- several metrics, one join

METRIC_KINDS = ("recall", "ndcg", "map")


def ranked_hits(predictions: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Check both tables ONCE and return every prediction that is in G_q.

    Evaluating several metrics on the same predictions needs the input checks
    and the join with the labels only once: every metric here is a function of
    the hits' ranks. On 8-12 million candidate rows this avoids repeating the
    checks and the join for each metric.

    Args:
        predictions: See recall_at_k.
        labels: See recall_at_k.

    Returns:
        One row per hit (any rank). Columns: query_id, rank.
    """
    check_labels(labels)
    check_predictions(predictions, labels)
    hits = predictions.select(QUERY_ID, ARTICLE_IDX, RANK).join(
        labels.select(QUERY_ID, ARTICLE_IDX), on=[QUERY_ID, ARTICLE_IDX], how="semi"
    )
    return hits.select(QUERY_ID, RANK)


def metric_from_hits(kind: str, hits: pl.DataFrame, labels: pl.DataFrame, k: int) -> pl.DataFrame:
    """One metric per query from the hits of ranked_hits.

    Gives exactly the same values as recall_at_k / ndcg_at_k /
    average_precision_at_k on the full predictions: the hits with rank <= k are
    the same rows, and the same code turns them into the metric.

    Args:
        kind: "recall", "ndcg" or "map" (map = AP@K per query).
        hits: Output of ranked_hits for these labels.
        labels: See recall_at_k (already checked by ranked_hits).
        k: Cut-off, > 0.

    Returns:
        One row per query in `labels`: query_id, fold, value (float64), sorted by query_id.
    """
    _check_k(k)
    top_k_hits = hits.filter(pl.col(RANK) <= k)
    if kind == "recall":
        return _recall_from_top_k_hits(top_k_hits, labels)
    if kind == "ndcg":
        return _ndcg_from_top_k_hits(top_k_hits, labels, k)
    if kind == "map":
        return _average_precision_from_top_k_hits(top_k_hits, labels, k)
    raise ValueError(f"metric kind must be one of {METRIC_KINDS}, got {kind!r}")


def label_hits(candidates: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """The candidate rows whose (query_id, article_idx) is in G_q (for union recall, §5.7).

    Only these rows matter for union recall, and they are few (about 1 in 400
    candidates), so unions and leave-one-out unions can be built from them
    instead of from tens of millions of candidate rows.

    Args:
        candidates: Rows with query_id, article_idx (other columns are kept).
        labels: See recall_at_k.

    Returns:
        The rows of `candidates` that hit a label, all columns kept.
    """
    _require_columns(candidates, (QUERY_ID, ARTICLE_IDX), "candidates")
    _check_query_ids_known(candidates, labels, "candidates")
    return candidates.join(
        labels.select(QUERY_ID, ARTICLE_IDX), on=[QUERY_ID, ARTICLE_IDX], how="semi"
    )


def union_recall_from_hits(hits: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Union recall per query from rows that are already known to hit a label (label_hits).

    The same pair may appear several times (several channels); it counts once.
    Gives exactly the values of union_recall on the full candidate rows.

    Returns:
        One row per query in `labels`: query_id, fold, value (float64), sorted by query_id.
    """
    _require_columns(hits, (QUERY_ID, ARTICLE_IDX), "hits")
    pairs = hits.select(QUERY_ID, ARTICLE_IDX).unique()
    hit_counts = pairs.group_by(QUERY_ID).agg(pl.len().alias("n_hits"))
    per_query = _attach_to_all_queries(hit_counts, labels)
    return _finish(per_query, pl.col("n_hits") / pl.col(_N_LABELS))


def union_recall(candidates: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Union recall per query: |candidates ∩ G_q| / |G_q|, with no cut-off.

    Args:
        candidates: Candidate articles per query; column query_id, article_idx.
            The same article may appear several times (e.g. from several
            channels); it counts once.
        labels: See recall_at_k.

    Returns:
        One row per query in `labels`: query_id, fold, value (float64), sorted by query_id.
    """
    check_labels(labels)
    _require_columns(candidates, (QUERY_ID, ARTICLE_IDX), "candidates")
    _check_query_ids_known(candidates, labels, "candidates")
    unique_candidates = candidates.select(QUERY_ID, ARTICLE_IDX).unique()
    hits = unique_candidates.join(
        labels.select(QUERY_ID, ARTICLE_IDX), on=[QUERY_ID, ARTICLE_IDX], how="semi"
    )
    hit_counts = hits.group_by(QUERY_ID).agg(pl.len().alias("n_hits"))
    per_query = _attach_to_all_queries(hit_counts, labels)
    return _finish(per_query, pl.col("n_hits") / pl.col(_N_LABELS))


# ---------------------------------------------------------------- averages


def _sorted_values(per_query: pl.DataFrame) -> list[float]:
    """Return the per-query values sorted by query_id, after basic checks."""
    _require_columns(per_query, (QUERY_ID, FOLD, VALUE), "per_query")
    if per_query.height == 0:
        raise ValueError("per_query is empty: cannot average over zero queries")
    if per_query[QUERY_ID].is_duplicated().any():
        raise ValueError("per_query: query_id must be unique (one value per query)")
    if per_query[VALUE].null_count() > 0:
        raise ValueError("per_query: value has nulls")
    return per_query.sort(QUERY_ID)[VALUE].cast(METRIC_DTYPE).to_list()


def macro_mean(per_query: pl.DataFrame) -> float:
    """Average per-query values (proposal §2.5 "macro").

    Uses math.fsum, which is exactly rounded, so the result does not depend on
    the row order.

    Args:
        per_query: One row per query; columns query_id, fold, value.

    Returns:
        The mean value as a Python float (float64).
    """
    values = _sorted_values(per_query)
    return math.fsum(values) / len(values)


def query_weighted_mean(per_query: pl.DataFrame) -> float:
    """Primary 4-week score (proposal §4.1): pool the queries of all folds, then average.

    A fold with more queries counts more. This is macro_mean over the pooled
    table; query_ids are globally unique, so no two folds collide.

    Args:
        per_query: One row per query from several folds; columns query_id, fold, value.

    Returns:
        The pooled mean (float64).
    """
    return macro_mean(per_query)


def per_fold_means(per_query: pl.DataFrame) -> pl.DataFrame:
    """Mean value of each fold.

    Args:
        per_query: One row per query; columns query_id, fold, value.

    Returns:
        One row per fold. Columns: fold (str), n_queries (int64), value (float64).
        Folds are in time order (by their smallest query_id, which encodes the
        fold position in the manifest).
    """
    _sorted_values(per_query)
    rows = []
    for fold_table in per_query.sort(QUERY_ID).partition_by(FOLD, maintain_order=True):
        rows.append(
            {
                FOLD: fold_table[FOLD][0],
                N_QUERIES: fold_table.height,
                VALUE: macro_mean(fold_table),
            }
        )
    return pl.DataFrame(rows, schema={FOLD: pl.String, N_QUERIES: pl.Int64, VALUE: METRIC_DTYPE})


def equal_fold_mean(per_query: pl.DataFrame) -> float:
    """Average of the per-fold means; every fold counts the same (reported, not used to select).

    Args:
        per_query: One row per query from several folds; columns query_id, fold, value.

    Returns:
        The mean of the fold means (float64).
    """
    fold_means = per_fold_means(per_query)[VALUE].to_list()
    return math.fsum(fold_means) / len(fold_means)
