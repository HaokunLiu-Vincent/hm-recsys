"""R1 age-bucketed decayed popularity (proposal §5 R1), channel "pop".

Variants (configs/recall/r1_popularity.yaml):

A1 (default), per age bucket g, history [a - window_days, a):
    S(g, i) = sum over purchase lines of article i by customers in bucket g
              of exp(-age_days / tau)                                  (§5 R1)
    Each query gets the Top-quota articles of its customer's bucket. If the
    bucket has fewer than `quota` active-pool articles, the list is filled with
    global decayed popularity
        S(i) = sum over ALL purchase lines of article i of exp(-age_days / tau)
    (same tau), in S(i) order, skipping articles already in the list. Fill-in
    rows have pop_is_fallback = 1.

G0 (ablation baseline): one global list for everyone,
    S(i) = number of purchase lines of article i in [a - window_days, a)
    (raw count, no decay, no buckets); pop_is_fallback = 0 everywhere.

Both variants only output articles in the active pool A(a) (proposal §2.4).

Order and score (the one exception to the default sort, see docs/decisions.md):
rows are sorted by (pop_is_fallback ASC, score DESC, article_idx ASC). Bucket
rows carry S(g, i), fill-in rows carry the global S(i), so `score` mixes two
scales; pop_is_fallback tells them apart. The B-group feature pop_bucket_score
(proposal §8.3) gives every candidate the same S(g, i) scale.
"""

import polars as pl

from hmrec.config import PopularityConfig
from hmrec.data.asof import history_window
from hmrec.logging_utils import get_logger
from hmrec.recall.base import (
    ARTICLE_IDX,
    CUSTOMER_IDX,
    QUERY_ID,
    SCORE,
    OutputSpec,
    RecallInputs,
    SortKey,
    add_age_days,
    decay_weight_expr,
    rank_candidates,
)

AGE_BUCKET = "age_bucket"
FOLD = "fold"
IS_FALLBACK = "pop_is_fallback"

POP_SPEC = OutputSpec(
    channel="pop",
    extra_columns={IS_FALLBACK: pl.Int8()},
    sort_keys=(
        SortKey(IS_FALLBACK, descending=False),
        SortKey(SCORE, descending=True),
        SortKey(ARTICLE_IDX, descending=False),
    ),
)

LOGGER = get_logger("recall.r1_popularity")


# ---------------------------------------------------------------- scores


def bucket_decayed_scores(
    history: pl.DataFrame, customers: pl.DataFrame, tau_days: float
) -> pl.DataFrame:
    """S(g, i) = sum of exp(-age_days / tau) over bucket g's purchase lines of i (§5 R1).

    Args:
        history: Purchase lines with age_days (from add_age_days); columns
            customer_idx, article_idx, age_days.
        customers: customer_idx -> age_bucket.
        tau_days: Decay time constant.

    Returns:
        One row per (age_bucket, article_idx) bought by that bucket; column score (float64).
    """
    with_bucket = history.join(customers.select(CUSTOMER_IDX, AGE_BUCKET), on=CUSTOMER_IDX)
    return with_bucket.group_by(AGE_BUCKET, ARTICLE_IDX).agg(
        decay_weight_expr(tau_days).sum().alias(SCORE)
    )


def global_decayed_scores(history: pl.DataFrame, tau_days: float) -> pl.DataFrame:
    """S(i) = sum of exp(-age_days / tau) over all purchase lines of i (fill-in for A1).

    Returns:
        One row per article in `history`; columns article_idx, score (float64).
    """
    return history.group_by(ARTICLE_IDX).agg(decay_weight_expr(tau_days).sum().alias(SCORE))


def global_raw_counts(history: pl.DataFrame) -> pl.DataFrame:
    """G0: S(i) = number of purchase lines of i in the window (no decay).

    Returns:
        One row per article in `history`; columns article_idx, score (float64).
    """
    return history.group_by(ARTICLE_IDX).agg(pl.len().cast(pl.Float64).alias(SCORE))


# ---------------------------------------------------------------- lists per bucket


def _keep_active(scores: pl.DataFrame, active_pool: pl.DataFrame) -> pl.DataFrame:
    """Keep only articles in the active pool A(a) (proposal §2.4)."""
    return scores.join(active_pool.select(ARTICLE_IDX), on=ARTICLE_IDX, how="semi")


def _top_per_group(
    scores: pl.DataFrame, group_column: str, quota: int, position: str
) -> pl.DataFrame:
    """Keep the `quota` best rows per group by (score DESC, article_idx ASC), numbered 1..n.

    The score is cast to float32 first, the precision it is stored with.
    """
    ordered = scores.with_columns(pl.col(SCORE).cast(pl.Float32)).sort(
        [group_column, SCORE, ARTICLE_IDX], descending=[False, True, False]
    )
    numbered = ordered.with_columns(
        pl.int_range(1, pl.len() + 1).over(group_column).alias(position)
    )
    return numbered.filter(pl.col(position) <= quota)


def build_bucket_lists(
    bucket_scores: pl.DataFrame,
    global_scores: pl.DataFrame,
    buckets: list[str],
    quota: int,
) -> pl.DataFrame:
    """Build the A1 list of every bucket: own Top-quota, then global fill-in.

    Steps for each bucket g:
        1. Own list: Top-quota of S(g, i).
        2. If it has n < quota rows, take the global list in S(i) order, drop the
           articles already in the own list, and keep the first quota - n.

    Args:
        bucket_scores: (age_bucket, article_idx, score), already limited to A(a).
        global_scores: (article_idx, score), already limited to A(a).
        buckets: Every bucket that needs a list (buckets of the queried customers).
        quota: Maximum list length.

    Returns:
        One row per (age_bucket, article_idx); columns age_bucket, article_idx,
        score (float32), pop_is_fallback (int8: 0 = own bucket, 1 = fill-in).
    """
    own = _top_per_group(bucket_scores, AGE_BUCKET, quota, "position")
    own_counts = own.group_by(AGE_BUCKET).agg(pl.len().alias("n_own"))
    bucket_frame = pl.DataFrame({AGE_BUCKET: buckets}, schema={AGE_BUCKET: pl.String})
    # Every bucket x every global article, minus the bucket's own articles.
    fill_pool = bucket_frame.join(global_scores, how="cross").join(
        own.select(AGE_BUCKET, ARTICLE_IDX), on=[AGE_BUCKET, ARTICLE_IDX], how="anti"
    )
    fill_ranked = _top_per_group(fill_pool, AGE_BUCKET, quota, "position")
    fill = (
        fill_ranked.join(own_counts, on=AGE_BUCKET, how="left")
        .with_columns(pl.col("n_own").fill_null(0))
        .filter(pl.col("position") <= quota - pl.col("n_own"))
    )
    own_rows = own.select(AGE_BUCKET, ARTICLE_IDX, SCORE).with_columns(
        pl.lit(0, dtype=pl.Int8).alias(IS_FALLBACK)
    )
    fill_rows = fill.select(AGE_BUCKET, ARTICLE_IDX, SCORE).with_columns(
        pl.lit(1, dtype=pl.Int8).alias(IS_FALLBACK)
    )
    return pl.concat([own_rows, fill_rows])


# ---------------------------------------------------------------- channel


class PopularityChannel:
    """R1 recall channel (variant A1 or G0), see the module docstring."""

    def __init__(self, config: PopularityConfig) -> None:
        """Store the R1 config."""
        self.config = config

    @property
    def output_spec(self) -> OutputSpec:
        """R1 output: extra column pop_is_fallback; fallback rows sorted last."""
        return POP_SPEC

    def generate(self, queries: pl.DataFrame, inputs: RecallInputs) -> pl.DataFrame:
        """Return R1 candidates for `queries` (see RecallChannel.generate)."""
        history = history_window(
            inputs.transactions,
            inputs.anchor,
            inputs.day_zero,
            self.config.window_days,
            columns=[CUSTOMER_IDX, ARTICLE_IDX],
        )
        history = add_age_days(history, inputs.anchor_day)
        if self.config.variant == "G0":
            candidates = self._generate_g0(queries, history, inputs)
        else:
            candidates = self._generate_a1(queries, history, inputs)
            log_fallback_diagnostics(fallback_diagnostics(candidates, queries, inputs.customers))
        return candidates

    def _generate_a1(
        self, queries: pl.DataFrame, history: pl.DataFrame, inputs: RecallInputs
    ) -> pl.DataFrame:
        """A1: bucket lists with global fill-in, joined to each query by age bucket."""
        tau = self.config.tau_days
        bucket_scores = _keep_active(
            bucket_decayed_scores(history, inputs.customers, tau), inputs.active_pool
        )
        global_scores = _keep_active(global_decayed_scores(history, tau), inputs.active_pool)
        query_buckets = queries.select(QUERY_ID, CUSTOMER_IDX).join(
            inputs.customers.select(CUSTOMER_IDX, AGE_BUCKET), on=CUSTOMER_IDX, how="left"
        )
        if query_buckets[AGE_BUCKET].null_count() > 0:
            raise ValueError("some query customers have no age_bucket in inputs.customers")
        buckets = query_buckets[AGE_BUCKET].unique().sort().to_list()
        lists = build_bucket_lists(bucket_scores, global_scores, buckets, self.config.quota)
        scored = query_buckets.join(lists, on=AGE_BUCKET).select(
            QUERY_ID, ARTICLE_IDX, SCORE, IS_FALLBACK
        )
        return rank_candidates(scored, POP_SPEC, self.config.quota)

    def _generate_g0(
        self, queries: pl.DataFrame, history: pl.DataFrame, inputs: RecallInputs
    ) -> pl.DataFrame:
        """G0: one global raw-count list for every query; no fill-in."""
        counts = _keep_active(global_raw_counts(history), inputs.active_pool)
        top = _top_per_group(
            counts.with_columns(pl.lit("ALL").alias(AGE_BUCKET)),
            AGE_BUCKET,
            self.config.quota,
            "position",
        )
        scored = (
            queries.select(QUERY_ID)
            .join(top.select(ARTICLE_IDX, SCORE), how="cross")
            .with_columns(pl.lit(0, dtype=pl.Int8).alias(IS_FALLBACK))
        )
        return rank_candidates(scored, POP_SPEC, self.config.quota)


# ---------------------------------------------------------------- diagnostics


def fallback_diagnostics(
    candidates: pl.DataFrame, queries: pl.DataFrame, customers: pl.DataFrame
) -> pl.DataFrame:
    """Count fill-in rows per fold and age bucket (diagnostic only).

    Args:
        candidates: R1 output.
        queries: query_id, customer_idx and (optionally) fold.
        customers: customer_idx -> age_bucket.

    Returns:
        One row per (fold, age_bucket) of the queried customers. Columns: fold,
        age_bucket, n_queries, n_queries_with_fallback, n_fallback_rows.
        fold is "?" if `queries` has no fold column.
    """
    with_fold = (
        queries if FOLD in queries.columns else queries.with_columns(pl.lit("?").alias(FOLD))
    )
    query_info = with_fold.select(QUERY_ID, FOLD, CUSTOMER_IDX).join(
        customers.select(CUSTOMER_IDX, AGE_BUCKET), on=CUSTOMER_IDX, how="left"
    )
    fallback_per_query = candidates.group_by(QUERY_ID).agg(
        pl.col(IS_FALLBACK).cast(pl.Int64).sum().alias("n_fallback_rows")
    )
    per_query = query_info.join(fallback_per_query, on=QUERY_ID, how="left").with_columns(
        pl.col("n_fallback_rows").fill_null(0)
    )
    return (
        per_query.group_by(FOLD, AGE_BUCKET)
        .agg(
            pl.len().alias("n_queries"),
            (pl.col("n_fallback_rows") > 0).sum().alias("n_queries_with_fallback"),
            pl.col("n_fallback_rows").sum().alias("n_fallback_rows"),
        )
        .sort(FOLD, AGE_BUCKET)
    )


def log_fallback_diagnostics(diagnostics: pl.DataFrame) -> None:
    """Log one line per (fold, age_bucket) with the fill-in counts."""
    for row in diagnostics.iter_rows(named=True):
        LOGGER.info(
            "pop fallback | fold=%s bucket=%s queries=%d queries_with_fallback=%d rows=%d",
            row[FOLD],
            row[AGE_BUCKET],
            row["n_queries"],
            row["n_queries_with_fallback"],
            row["n_fallback_rows"],
        )
