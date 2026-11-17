"""Tests for R3 sibling with a hand-computed "black / blue / white" example.

Catalog: product_code 500 = one T-shirt in three colours:
    article 30 = black, 31 = blue, 32 = white
product_code 600 = one jumper in two colours: 40 = grey, 41 = green.

Anchor day 200; style window 180 days (days 20..199), market window 7 days
(days 193..199); tau = 30, gamma = 0.5.

Customer 0 (the only query):
    bought black 30 on day 170 (age 30) -> weight e^(-30/30) = e^(-1)    = 0.367879
    bought grey  40 on day 199 (age 1)  -> weight e^(-1/30)              = 0.967216
    bought blue  31 on day 10  (age 190, BEFORE the style window: not excluded,
                                and not part of P)

Style preference:
    P(0, 500) = 0.367879
    P(0, 600) = 0.967216

Market heat M(j), purchase lines of all customers in days 193..199:
    blue 31: 3 lines (customer 5)      -> M = 3
    white 32: 0 lines                  -> M = 0 (one more line on day 150 is outside)
    green 41: 8 lines (customer 6)     -> M = 8

Candidates: siblings not bought in the style window -> 31, 32 (code 500), 41 (code 600).
Black 30 and grey 40 are excluded (bought in the window).

gamma = 0.5:
    S(0, 41) = 0.967216 x (1 + 8)^0.5 = 0.967216 x 3 = 2.901648
    S(0, 31) = 0.367879 x (1 + 3)^0.5 = 0.367879 x 2 = 0.735759
    S(0, 32) = 0.367879 x (1 + 0)^0.5 = 0.367879
    -> order 41, 31, 32

gamma = 0 (style only): 41 = 0.967216, 31 = 32 = 0.367879 (tie -> article_idx ASC)
    -> order 41, 31, 32
gamma = 1: 41 = 0.967216 x 9 = 8.704945, 31 = 0.367879 x 4 = 1.471518, 32 = 0.367879

(Values are rounded to 6 decimals after computing with the exact weights.)
"""

import dataclasses
import math

import polars as pl
import pytest
from conftest import make_queries, make_recall_inputs

from hmrec.config import SiblingConfig
from hmrec.recall.base import rank_candidates, validate_candidates
from hmrec.recall.r3_sibling import SIB_SPEC, SiblingChannel

ANCHOR_DAY = 200
PRODUCT_CODES = {30: 500, 31: 500, 32: 500, 40: 600, 41: 600}
ROWS = [
    (0, 30, 170),
    (0, 40, 199),
    (0, 31, 10),
    *[(5, 31, 195)] * 3,
    *[(6, 41, 199)] * 8,
    (7, 32, 150),
]
QUERIES = make_queries([(0, 0)])
P_500 = math.exp(-1)
P_600 = math.exp(-1 / 30)


def _config(gamma: float = 0.5, quota: int = 50) -> SiblingConfig:
    return SiblingConfig(
        style_lookback_days=180, market_window_days=7, tau_days=30.0, gamma=gamma, quota=quota
    )


def _run(config: SiblingConfig, rows: list[tuple[int, int, int]] = ROWS) -> pl.DataFrame:
    inputs = make_recall_inputs(rows, ANCHOR_DAY, product_codes=PRODUCT_CODES)
    return SiblingChannel(config).generate(QUERIES, inputs)


def _result(candidates: pl.DataFrame) -> tuple[list[int], list[float]]:
    ordered = candidates.sort("query_id", "rank")
    return ordered["article_idx"].to_list(), ordered["score"].to_list()


def test_black_blue_white_gamma_half() -> None:
    candidates = _run(_config(gamma=0.5))
    articles, scores = _result(candidates)
    assert articles == [41, 31, 32]
    assert scores == pytest.approx([P_600 * 3, P_500 * 2, P_500], rel=1e-6)
    validate_candidates(candidates, SIB_SPEC, quota=50)


def test_gamma_zero_is_style_only_with_article_tie_break() -> None:
    articles, scores = _result(_run(_config(gamma=0.0)))
    assert articles == [41, 31, 32]
    assert scores == pytest.approx([P_600, P_500, P_500], rel=1e-6)


def test_gamma_one() -> None:
    articles, scores = _result(_run(_config(gamma=1.0)))
    assert articles == [41, 31, 32]
    assert scores == pytest.approx([P_600 * 9, P_500 * 4, P_500], rel=1e-6)


def test_bought_articles_in_window_are_excluded() -> None:
    articles, _ = _result(_run(_config()))
    assert 30 not in articles  # black, bought on day 170
    assert 40 not in articles  # grey, bought on day 199
    # Blue 31 was bought on day 10, before the 180-day window: still a candidate.
    assert 31 in articles


def test_quota() -> None:
    assert _result(_run(_config(quota=2)))[0] == [41, 31]


def test_ignores_poison_on_and_after_anchor() -> None:
    # If these leaked: customer 0 buying white 32 on the anchor day would exclude it,
    # and 10 lines of 32 would raise M(32); a day-201 purchase of code 500 would raise P.
    poison = [(0, 32, 200), *[(5, 32, 200)] * 10, (0, 31, 201)]
    assert _run(_config(), ROWS + poison).equals(_run(_config()))


# ---------------------------------------------------------------- active pool only (M4 rule)
#
# Same example, gamma 0.5: unfiltered order 41 (2.901648), 31 (0.735759), 32 (0.367879).
# Active pool A(a) = {31, 32, 40}: green 41 is NOT in the pool.
#   active_pool_only -> 41 is dropped BEFORE the cut: order 31, 32.
#   With quota 2: unfiltered Top-2 = 41, 31; filtered Top-2 = 31, 32, so the place
#   freed by 41 goes to the next in-pool sibling (32), it is not left empty.


def _pool_config(gamma: float = 0.5, quota: int = 50) -> SiblingConfig:
    return dataclasses.replace(_config(gamma, quota), active_pool_only=True)


def _run_with_pool(
    config: SiblingConfig, pool: list[int], rows: list[tuple[int, int, int]] = ROWS
) -> pl.DataFrame:
    inputs = make_recall_inputs(rows, ANCHOR_DAY, product_codes=PRODUCT_CODES, active_pool=pool)
    return SiblingChannel(config).generate(QUERIES, inputs)


def test_active_pool_only_drops_out_of_pool_siblings_and_keeps_scores() -> None:
    result = _run_with_pool(_pool_config(), [31, 32, 40])
    validate_candidates(result, SIB_SPEC, quota=50)
    assert result["article_idx"].to_list() == [31, 32]
    # Scores are unchanged: P(u, c) and M(j) still use every article and customer.
    assert result["score"].to_list() == pytest.approx([P_500 * 2, P_500], rel=1e-6)


def test_freed_places_go_to_in_pool_siblings() -> None:
    assert _run_with_pool(_config(quota=2), [31, 32, 40])["article_idx"].to_list() == [41, 31]
    assert _run_with_pool(_pool_config(quota=2), [31, 32, 40])["article_idx"].to_list() == [31, 32]


@pytest.mark.parametrize("pool", [[31, 32, 40], [41], [30, 31, 41], []])
@pytest.mark.parametrize("quota", [1, 2, 50])
def test_filter_before_cut_equals_full_scoring_then_filter(pool: list[int], quota: int) -> None:
    # Reference: score EVERY sibling (no quota limit), delete the out-of-pool rows, then
    # rank and cut to the quota. The option must give exactly this table.
    everything = _run_with_pool(_config(quota=1000), pool)
    in_pool = everything.filter(
        pl.col("article_idx").is_in(pl.Series(pool, dtype=pl.Int32).implode())
    )
    reference = rank_candidates(in_pool.select("query_id", "article_idx", "score"), SIB_SPEC, quota)
    assert _run_with_pool(_pool_config(quota=quota), pool).equals(reference)


def test_option_off_ignores_the_active_pool() -> None:
    assert _run_with_pool(_config(), [31]).equals(_run(_config()))


def test_active_pool_only_ignores_poison_on_and_after_anchor() -> None:
    poison = [(0, 32, 200), (0, 41, 205), *[(8, 32, 200)] * 50, *[(8, 31, 201)] * 50]
    clean = _run_with_pool(_pool_config(), [31, 32, 40])
    assert _run_with_pool(_pool_config(), [31, 32, 40], ROWS + poison).equals(clean)
