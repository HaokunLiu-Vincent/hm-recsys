"""Tests for R1 popularity with a hand-computed example.

Anchor day 100, window 7 days -> history days 93..99, tau = 3, quota 3.
Active pool A(a) = {10, 11, 12, 13} (article 14 is not active).

Customers: 0 and 1 -> AGE_18_24, 2 -> AGE_25_34, 3 -> AGE_UNKNOWN.
Queries: q0 -> customer 0, q1 -> customer 2, q2 -> customer 3.

Purchase lines (customer, article, day, age_days, weight exp(-age/3)):
    0  10  99  1  e^(-1/3) = 0.716531
    1  10  97  3  e^(-1)   = 0.367879
    1  11  99  1  0.716531
    0  12  94  6  e^(-2)   = 0.135335
    2  12  99  1  0.716531   (twice: duplicate line -> 1.433063)
    2  13  98  2  e^(-2/3) = 0.513417
    3  14  99  1  not active -> never output
    2  10  92  8  outside the 7-day window -> ignored

S(18_24, i): 10 = 0.716531 + 0.367879 = 1.084411, 11 = 0.716531, 12 = 0.135335
S(25_34, i): 12 = 1.433063, 13 = 0.513417
S(UNKNOWN, i): only article 14, which is not active -> empty list
Global S(i) (active only): 12 = 0.135335 + 1.433063 = 1.568398, 10 = 1.084411,
    11 = 0.716531, 13 = 0.513417

Expected A1 lists:
    q0 (18_24): 10 (1.084411), 11 (0.716531), 12 (0.135335)       no fill-in
    q1 (25_34): 12 (1.433063), 13 (0.513417), then fill-in 10 (1.084411, global)
    q2 (UNKNOWN): fill-in 12 (1.568398), 10 (1.084411), 11 (0.716531)

(Values are rounded to 6 decimals after summing the exact weights.)
"""

import math

import polars as pl
import pytest

from conftest import make_queries, make_recall_inputs
from hmrec.config import PopularityConfig
from hmrec.recall.base import validate_candidates
from hmrec.recall.r1_popularity import POP_SPEC, PopularityChannel, fallback_diagnostics

ANCHOR_DAY = 100
ROWS = [
    (0, 10, 99),
    (1, 10, 97),
    (1, 11, 99),
    (0, 12, 94),
    (2, 12, 99),
    (2, 12, 99),
    (2, 13, 98),
    (3, 14, 99),
    (2, 10, 92),
]
CUSTOMERS = {0: "AGE_18_24", 1: "AGE_18_24", 2: "AGE_25_34", 3: "AGE_UNKNOWN"}
ACTIVE_POOL = [10, 11, 12, 13]
QUERIES = make_queries([(0, 0), (1, 2), (2, 3)])
A1 = PopularityConfig(variant="A1", window_days=7, tau_days=3.0, quota=3)
G0 = PopularityConfig(variant="G0", window_days=7, tau_days=3.0, quota=3)

W1 = math.exp(-1 / 3)  # age 1
W2 = math.exp(-2 / 3)  # age 2
W3 = math.exp(-1)  # age 3
W6 = math.exp(-2)  # age 6


def _run(config: PopularityConfig, rows: list[tuple[int, int, int]] = ROWS) -> pl.DataFrame:
    inputs = make_recall_inputs(rows, ANCHOR_DAY, customers=CUSTOMERS, active_pool=ACTIVE_POOL)
    return PopularityChannel(config).generate(QUERIES, inputs)


def _rows(candidates: pl.DataFrame, query_id: int) -> list[tuple[int, int, int]]:
    """(rank, article_idx, pop_is_fallback) of one query."""
    one = candidates.filter(pl.col("query_id") == query_id).sort("rank")
    return list(zip(one["rank"], one["article_idx"], one["pop_is_fallback"], strict=True))


def _scores(candidates: pl.DataFrame, query_id: int) -> list[float]:
    return candidates.filter(pl.col("query_id") == query_id).sort("rank")["score"].to_list()


def test_a1_hand_computed() -> None:
    candidates = _run(A1)
    assert _rows(candidates, 0) == [(1, 10, 0), (2, 11, 0), (3, 12, 0)]
    assert _scores(candidates, 0) == pytest.approx([W1 + W3, W1, W6], rel=1e-6)
    # q1: own bucket has 2 active articles; the 3rd slot is global fill-in.
    # Its score 1.084411 is larger than 0.513417 but it still ranks last: fallback rows
    # are sorted after own rows (pop_is_fallback ASC first).
    assert _rows(candidates, 1) == [(1, 12, 0), (2, 13, 0), (3, 10, 1)]
    assert _scores(candidates, 1) == pytest.approx([2 * W1, W2, W1 + W3], rel=1e-6)
    # q2: AGE_UNKNOWN only bought inactive article 14 -> the whole list is fill-in.
    assert _rows(candidates, 2) == [(1, 12, 1), (2, 10, 1), (3, 11, 1)]
    assert _scores(candidates, 2) == pytest.approx([W6 + 2 * W1, W1 + W3, W1], rel=1e-6)
    validate_candidates(candidates, POP_SPEC, quota=3)


def test_g0_hand_computed() -> None:
    # Raw 7-day counts (active only): 10 -> 2, 11 -> 1, 12 -> 3 (1 + duplicate 2), 13 -> 1.
    # Top 3: 12 (3), 10 (2), then 11 and 13 tie at 1 -> article_idx ASC -> 11.
    candidates = _run(G0)
    for query_id in (0, 1, 2):
        assert _rows(candidates, query_id) == [(1, 12, 0), (2, 10, 0), (3, 11, 0)]
        assert _scores(candidates, query_id) == [3.0, 2.0, 1.0]


def test_inactive_article_never_output() -> None:
    assert 14 not in _run(A1)["article_idx"].to_list()
    assert 14 not in _run(G0)["article_idx"].to_list()


def test_ignores_poison_on_and_after_anchor() -> None:
    # Many lines on day 100 (the anchor) and later would change every ranking if
    # they leaked: article 13 bought 5 times by bucket 18_24 and by customer 3.
    poison = [(0, 13, 100)] * 5 + [(3, 13, 101)] * 5 + [(1, 11, 200)]
    for config in (A1, G0):
        assert _run(config, ROWS + poison).equals(_run(config))


def test_fallback_diagnostics_per_bucket() -> None:
    diagnostics = fallback_diagnostics(
        _run(A1), QUERIES, make_recall_inputs([], 1, CUSTOMERS).customers
    )
    assert diagnostics.rows() == [
        ("D1", "AGE_18_24", 1, 0, 0),
        ("D1", "AGE_25_34", 1, 1, 1),
        ("D1", "AGE_UNKNOWN", 1, 1, 3),
    ]


def test_quota_larger_than_pool_is_not_padded() -> None:
    # Only 4 active articles exist, so even quota 100 gives at most 4 rows.
    config = PopularityConfig(variant="A1", window_days=7, tau_days=3.0, quota=100)
    candidates = _run(config)
    counts = candidates.group_by("query_id").len().sort("query_id")["len"].to_list()
    assert counts == [4, 4, 4]


# ---------------------------------------------------------------- full bucket score table (M6)


def _with_table(
    rows: list[tuple[int, int, int]] = ROWS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    inputs = make_recall_inputs(rows, ANCHOR_DAY, customers=CUSTOMERS, active_pool=ACTIVE_POOL)
    return PopularityChannel(A1).generate_with_bucket_scores(QUERIES, inputs)


def test_bucket_score_table_is_not_cut_to_the_quota_or_the_active_pool() -> None:
    candidates, table = _with_table()
    # The candidates are exactly those of generate().
    assert candidates.equals(_run(A1))
    # S(g, i) of EVERY (bucket, article) in the window (see the module docstring):
    # also (UNKNOWN, 14), although article 14 is not active and is never a candidate.
    assert table.rows() == [
        ("AGE_18_24", 10, pytest.approx(W1 + W3, rel=1e-6)),
        ("AGE_18_24", 11, pytest.approx(W1, rel=1e-6)),
        ("AGE_18_24", 12, pytest.approx(W6, rel=1e-6)),
        ("AGE_25_34", 12, pytest.approx(2 * W1, rel=1e-6)),
        ("AGE_25_34", 13, pytest.approx(W2, rel=1e-6)),
        ("AGE_UNKNOWN", 14, pytest.approx(W1, rel=1e-6)),
    ]
    assert table.schema == pl.Schema(
        {"age_bucket": pl.String, "article_idx": pl.Int32, "score": pl.Float32}
    )
    # Quota 1 keeps one candidate per query, but the table is unchanged.
    quota_one = PopularityConfig(variant="A1", window_days=7, tau_days=3.0, quota=1)
    inputs = make_recall_inputs(ROWS, ANCHOR_DAY, customers=CUSTOMERS, active_pool=ACTIVE_POOL)
    cut, same_table = PopularityChannel(quota_one).generate_with_bucket_scores(QUERIES, inputs)
    assert cut.height == 3 and same_table.equals(table)


def test_non_fallback_scores_are_the_table_entries_bit_for_bit() -> None:
    candidates, table = _with_table()
    own = candidates.filter(pl.col("pop_is_fallback") == 0)
    # q0 is bucket 18_24, q1 is bucket 25_34 (q2 has only fill-in rows).
    bucket_of_query = {0: "AGE_18_24", 1: "AGE_25_34"}
    entries = {(bucket, article): score for bucket, article, score in table.rows()}
    assert own.height == 5
    for query_id, article, score in own.select("query_id", "article_idx", "score").rows():
        assert entries[(bucket_of_query[query_id], article)] == score  # exact, no tolerance


def test_bucket_score_table_ignores_poison_and_g0_has_none() -> None:
    poison = [(0, 13, 100)] * 5 + [(3, 13, 101)] * 5 + [(1, 11, 200)]
    assert _with_table(ROWS + poison)[1].equals(_with_table()[1])
    inputs = make_recall_inputs(ROWS, ANCHOR_DAY, customers=CUSTOMERS, active_pool=ACTIVE_POOL)
    with pytest.raises(ValueError, match="A1 only"):
        PopularityChannel(G0).generate_with_bucket_scores(QUERIES, inputs)
