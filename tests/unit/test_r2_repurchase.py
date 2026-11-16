"""Tests for R2 repurchase with a hand-computed example.

Anchor day 200, lookback 180 -> history days 20..199, tau = 30.
Queries: q0 -> customer 0, q1 -> customer 1 (no purchases), q2 -> customer 2.

Customer 0 purchase lines (article, day, age_days, weight exp(-age/30)):
    20  199    1  e^(-1/30)  = 0.967216
    20  170   30  e^(-1)     = 0.367879
    21  190   10  e^(-1/3)   = 0.716531   (twice: duplicate line)
    23  199    1  0.967216
    22   15  185  outside the window -> ignored

Decayed frequency (tau 30):
    21: 2 x 0.716531           = 1.433063
    20: 0.967216 + 0.367879    = 1.335096
    23: 0.967216
    -> order 21, 20, 23

Recency only (1 / age_days of the most recent purchase):
    20: last bought at age 1  -> 1/1  = 1.0   (the purchase at age 30 is ignored)
    23: last bought at age 1  -> 1.0  (same day as 20: tie -> article_idx ASC)
    21: age 10                -> 1/10 = 0.1  (bought twice; the count does NOT help)
    -> order 20, 23, 21

Customer 2: article 24 on day 20 (age 180, first day of the window, included)
and article 25 on day 19 (age 181, excluded).
"""

import math

import polars as pl
import pytest
from conftest import make_queries, make_recall_inputs

from hmrec.config import RepurchaseConfig
from hmrec.recall.base import add_age_days, validate_candidates
from hmrec.recall.r2_repurchase import REP_SPEC, RepurchaseChannel

ANCHOR_DAY = 200
ROWS = [
    (0, 20, 199),
    (0, 20, 170),
    (0, 21, 190),
    (0, 21, 190),
    (0, 23, 199),
    (0, 22, 15),
    (2, 24, 20),
    (2, 25, 19),
    (9, 20, 199),  # customer 9 is not a query
]
QUERIES = make_queries([(0, 0), (1, 1), (2, 2)])
DECAYED = RepurchaseConfig(variant="decayed_frequency", lookback_days=180, tau_days=30.0, quota=30)
RECENCY = RepurchaseConfig(variant="recency_only", lookback_days=180, tau_days=30.0, quota=30)


def _run(config: RepurchaseConfig, rows: list[tuple[int, int, int]] = ROWS) -> pl.DataFrame:
    return RepurchaseChannel(config).generate(QUERIES, make_recall_inputs(rows, ANCHOR_DAY))


def _query(candidates: pl.DataFrame, query_id: int) -> tuple[list[int], list[float]]:
    one = candidates.filter(pl.col("query_id") == query_id).sort("rank")
    return one["article_idx"].to_list(), one["score"].to_list()


def test_decayed_frequency_hand_computed() -> None:
    candidates = _run(DECAYED)
    articles, scores = _query(candidates, 0)
    assert articles == [21, 20, 23]
    expected = [2 * math.exp(-1 / 3), math.exp(-1 / 30) + math.exp(-1), math.exp(-1 / 30)]
    assert scores == pytest.approx(expected, rel=1e-6)
    validate_candidates(candidates, REP_SPEC, quota=30)


def test_recency_only_uses_last_purchase_and_breaks_ties_by_article() -> None:
    articles, scores = _query(_run(RECENCY), 0)
    assert articles == [20, 23, 21]
    assert scores == pytest.approx([1.0, 1.0, 0.1], rel=1e-6)


def test_window_edges_and_empty_customers() -> None:
    candidates = _run(DECAYED)
    # Customer 2: age 180 is inside, age 181 is not.
    articles, scores = _query(candidates, 2)
    assert articles == [24]
    assert scores == pytest.approx([math.exp(-6)], rel=1e-6)
    # Customer 1 has no purchases: no rows (never padded), and customer 9 is not a query.
    assert candidates.filter(pl.col("query_id") == 1).height == 0
    assert sorted(candidates["query_id"].unique().to_list()) == [0, 2]


def test_quota_keeps_the_best() -> None:
    config = RepurchaseConfig(
        variant="decayed_frequency", lookback_days=180, tau_days=30.0, quota=2
    )
    assert _query(_run(config), 0)[0] == [21, 20]


def test_ignores_poison_on_and_after_anchor() -> None:
    # Five lines of article 23 on the anchor day would put 23 first; customer 1
    # buying on day 205 would give q1 rows.
    poison = [(0, 23, 200)] * 5 + [(1, 26, 205)]
    for config in (DECAYED, RECENCY):
        assert _run(config, ROWS + poison).equals(_run(config))


def test_age_days_below_one_raises() -> None:
    # age_days 0 would mean a row from the anchor day is in the history.
    leaked = pl.DataFrame({"t_dat_day": [199, 200]}, schema={"t_dat_day": pl.Int16})
    with pytest.raises(ValueError, match="age_days=0"):
        add_age_days(leaked, anchor_day=200)
