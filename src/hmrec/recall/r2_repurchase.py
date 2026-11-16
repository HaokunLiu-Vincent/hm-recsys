"""R2 repurchase (proposal §5 R2), channel "rep".

Only articles the customer bought in [a - lookback_days, a) are recommended
(default 180 days). Not limited to the active pool (proposal §2.4).

Variants (configs/recall/r2_repurchase.yaml):

decayed_frequency (default):
    S(u, i) = sum over u's purchase lines of i of exp(-age_days / tau)     (§5 R2)
    Duplicate lines count separately: buying twice is a stronger signal.

recency_only (ablation control):
    S(u, i) = 1 / age_days(last purchase of i by u)
    Only the most recent purchase matters; yesterday = 1.0, two days ago = 0.5.
    Articles bought on the same day tie and are ordered by article_idx ASC only;
    no other information (e.g. purchase count) breaks the tie, so this stays a
    pure recency baseline.

Order: default (score DESC, article_idx ASC). At most `quota` rows, never
padded; customers without purchases in the window get no rows.
"""

import polars as pl

from hmrec.config import RepurchaseConfig
from hmrec.data.asof import history_window
from hmrec.recall.base import (
    AGE_DAYS,
    ARTICLE_IDX,
    CUSTOMER_IDX,
    QUERY_ID,
    SCORE,
    OutputSpec,
    RecallInputs,
    add_age_days,
    decay_weight_expr,
    rank_candidates,
)

REP_SPEC = OutputSpec(channel="rep")


def decayed_frequency_scores(history: pl.DataFrame, tau_days: float) -> pl.DataFrame:
    """S(u, i) = sum of exp(-age_days / tau) over u's purchase lines of i (§5 R2).

    Args:
        history: Purchase lines with age_days; columns customer_idx, article_idx, age_days.
        tau_days: Decay time constant.

    Returns:
        One row per (customer_idx, article_idx); column score (float64).
    """
    return history.group_by(CUSTOMER_IDX, ARTICLE_IDX).agg(
        decay_weight_expr(tau_days).sum().alias(SCORE)
    )


def recency_only_scores(history: pl.DataFrame) -> pl.DataFrame:
    """S(u, i) = 1 / min(age_days) over u's purchase lines of i (recency-only control).

    min(age_days) is the most recent purchase; earlier purchases of the same
    article are ignored on purpose.

    Returns:
        One row per (customer_idx, article_idx); column score (float64).
    """
    return history.group_by(CUSTOMER_IDX, ARTICLE_IDX).agg(
        (1.0 / pl.col(AGE_DAYS).min().cast(pl.Float64)).alias(SCORE)
    )


class RepurchaseChannel:
    """R2 recall channel, see the module docstring."""

    def __init__(self, config: RepurchaseConfig) -> None:
        """Store the R2 config."""
        self.config = config

    @property
    def output_spec(self) -> OutputSpec:
        """R2 output: base columns only, default order."""
        return REP_SPEC

    def generate(self, queries: pl.DataFrame, inputs: RecallInputs) -> pl.DataFrame:
        """Return R2 candidates for `queries` (see RecallChannel.generate)."""
        history = history_window(
            inputs.transactions,
            inputs.anchor,
            inputs.day_zero,
            self.config.lookback_days,
            columns=[CUSTOMER_IDX, ARTICLE_IDX],
        )
        # Only the queried customers matter; dropping the rest early saves memory.
        history = history.join(queries.select(CUSTOMER_IDX), on=CUSTOMER_IDX, how="semi")
        history = add_age_days(history, inputs.anchor_day)
        if self.config.variant == "recency_only":
            scores = recency_only_scores(history)
        else:
            scores = decayed_frequency_scores(history, self.config.tau_days)
        scored = queries.select(QUERY_ID, CUSTOMER_IDX).join(scores, on=CUSTOMER_IDX)
        return rank_candidates(
            scored.select(QUERY_ID, ARTICLE_IDX, SCORE), REP_SPEC, self.config.quota
        )
