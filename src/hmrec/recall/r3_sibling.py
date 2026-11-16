"""R3 sibling: same product_code, other colour (proposal §5 R3, formula S2), channel "sib".

Windows (configs/recall/r3_sibling.yaml):
    style window  [a - style_lookback_days, a)  (default 180 days)
    market window [a - market_window_days, a)   (default 7 days)

Formulas (§5 R3):
    P(u, c) = sum over u's purchase lines in the style window with product_code c
              of exp(-age_days / tau)                      (style preference)
    M(j)    = number of purchase lines of article j in the market window,
              over ALL customers                           (market heat)
    S(u, j) = P(u, c(j)) * (1 + M(j)) ** gamma
    gamma = 0 gives S = P (style preference only).

Candidates for u: every catalog article j whose product_code c(j) u bought in
the style window, EXCLUDING articles u bought in the style window. An article
u bought only before the style window is not excluded.
Not limited to the active pool (proposal §2.4).

Order: default (score DESC, article_idx ASC). At most `quota` rows, never padded.
"""

import polars as pl

from hmrec.config import SiblingConfig
from hmrec.data.asof import history_window
from hmrec.recall.base import (
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

PRODUCT_CODE = "product_code"
STYLE = "style_preference"
MARKET = "market_count"

SIB_SPEC = OutputSpec(channel="sib")


def style_preference(
    style_history: pl.DataFrame, articles: pl.DataFrame, tau_days: float
) -> pl.DataFrame:
    """P(u, c) = sum of exp(-age_days / tau) over u's purchase lines of product_code c (§5 R3).

    Args:
        style_history: Purchase lines with age_days; columns customer_idx, article_idx, age_days.
        articles: article_idx -> product_code.
        tau_days: Decay time constant.

    Returns:
        One row per (customer_idx, product_code); column style_preference (float64).
    """
    with_code = style_history.join(articles.select(ARTICLE_IDX, PRODUCT_CODE), on=ARTICLE_IDX)
    return with_code.group_by(CUSTOMER_IDX, PRODUCT_CODE).agg(
        decay_weight_expr(tau_days).sum().alias(STYLE)
    )


def market_counts(market_history: pl.DataFrame) -> pl.DataFrame:
    """M(j) = number of purchase lines of article j in the market window, all customers (§5 R3).

    Returns:
        One row per article bought in the window; columns article_idx, market_count (int64).
    """
    return market_history.group_by(ARTICLE_IDX).agg(pl.len().cast(pl.Int64).alias(MARKET))


def sibling_scores(
    preference: pl.DataFrame,
    style_history: pl.DataFrame,
    articles: pl.DataFrame,
    market: pl.DataFrame,
    gamma: float,
) -> pl.DataFrame:
    """S(u, j) = P(u, c(j)) * (1 + M(j)) ** gamma for every sibling j not bought by u (§5 R3).

    Args:
        preference: P(u, c) from style_preference.
        style_history: The style-window purchase lines (for the exclusion).
        articles: article_idx -> product_code (the catalog).
        market: M(j) from market_counts; articles not in it have M(j) = 0.
        gamma: Exponent of (1 + M(j)), >= 0.

    Returns:
        One row per (customer_idx, article_idx); column score (float64).
    """
    siblings = preference.join(articles.select(ARTICLE_IDX, PRODUCT_CODE), on=PRODUCT_CODE)
    bought = style_history.select(CUSTOMER_IDX, ARTICLE_IDX).unique()
    not_bought = siblings.join(bought, on=[CUSTOMER_IDX, ARTICLE_IDX], how="anti")
    with_market = not_bought.join(market, on=ARTICLE_IDX, how="left").with_columns(
        pl.col(MARKET).fill_null(0)
    )
    heat = (1.0 + pl.col(MARKET).cast(pl.Float64)).pow(gamma)
    return with_market.select(CUSTOMER_IDX, ARTICLE_IDX, (pl.col(STYLE) * heat).alias(SCORE))


class SiblingChannel:
    """R3 recall channel, see the module docstring."""

    def __init__(self, config: SiblingConfig) -> None:
        """Store the R3 config."""
        self.config = config

    @property
    def output_spec(self) -> OutputSpec:
        """R3 output: base columns only, default order."""
        return SIB_SPEC

    def generate(self, queries: pl.DataFrame, inputs: RecallInputs) -> pl.DataFrame:
        """Return R3 candidates for `queries` (see RecallChannel.generate)."""
        style_history = history_window(
            inputs.transactions,
            inputs.anchor,
            inputs.day_zero,
            self.config.style_lookback_days,
            columns=[CUSTOMER_IDX, ARTICLE_IDX],
        )
        style_history = style_history.join(
            queries.select(CUSTOMER_IDX), on=CUSTOMER_IDX, how="semi"
        )
        style_history = add_age_days(style_history, inputs.anchor_day)
        # The market count uses every customer, not only the queried ones.
        market_history = history_window(
            inputs.transactions,
            inputs.anchor,
            inputs.day_zero,
            self.config.market_window_days,
            columns=[ARTICLE_IDX],
        )
        preference = style_preference(style_history, inputs.articles, self.config.tau_days)
        scores = sibling_scores(
            preference,
            style_history,
            inputs.articles,
            market_counts(market_history),
            self.config.gamma,
        )
        scored = queries.select(QUERY_ID, CUSTOMER_IDX).join(scores, on=CUSTOMER_IDX)
        return rank_candidates(
            scored.select(QUERY_ID, ARTICLE_IDX, SCORE), SIB_SPEC, self.config.quota
        )
