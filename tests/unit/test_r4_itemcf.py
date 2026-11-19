"""Tests for R4 ItemCF with a hand-computed 4-user example.

Articles: A = 0, B = 1, C = 2, D = 3, E = 4 (E is never bought).
Anchor day 400; fit window 365 days (days 35..399); seed window 90 days (days 310..399).

Users (distinct articles in the fit window):
    U1 (customer 1): A, B     d = 2   (A bought twice: still R = 1)
    U2 (customer 2): A, B, C  d = 3
    U3 (customer 3): A, C, D  d = 3
    U4 (customer 4): A        d = 1
Customer 5 bought B on day 30 (age 370): outside the fit window, ignored.

IUF weights w_u = 1 / ln(1 + d_u):
    w1 = 1/ln 3 = 0.910239    w2 = w3 = 1/ln 4 = 0.721348    w4 = 1/ln 2 = 1.442695

n_w(i) = sum of w_u over users who bought i:
    n_w(A) = w1 + w2 + w3 + w4 = 3.795629
    n_w(B) = w1 + w2           = 1.631587
    n_w(C) = w2 + w3           = 1.442695
    n_w(D) = w3                = 0.721348
C_w(i, j) = sum of w_u over users who bought both:
    C_w(A, B) = w1 + w2 = 1.631587    C_w(A, C) = w2 + w3 = 1.442695
    C_w(A, D) = w3      = 0.721348    C_w(B, C) = w2      = 0.721348

s(i -> j) = C_w(i, j) / (n_w(i)^alpha * n_w(j)^(1 - alpha)):
    alpha = 0.5: s(A -> B) = s(B -> A) = 1.631587 / sqrt(3.795629 * 1.631587) = 0.655637
    alpha = 0.7: s(A -> B) = 1.631587 / (3.795629^0.7 * 1.631587^0.3) = 0.553769
                 s(B -> A) = 1.631587 / (1.631587^0.7 * 3.795629^0.3) = 0.776243  (!= s(A -> B))
    CF0 (w_u = 1, alpha = 0.5): s(A -> B) = 2 / sqrt(4 * 2) = 0.707107

User score, U1 with alpha = 0.7 (seeds A and B):
    from A: B 0.553769, C 0.508070, D 0.312754
    from B: A 0.776243, C 0.458738
    S: C = 0.508070 + 0.458738 = 0.966808, A = 0.776243, B = 0.553769, D = 0.312754
    -> order C, A, B, D (already-bought A and B are NOT excluded)

Seed decay on (tau_seed = 30), U1 again. Seed weight q = exp(-age / 30) of the
LAST purchase of each seed:
    A: bought on days 390 and 392 -> last at age 8 -> q_A = e^(-8/30) = 0.765928
       (NOT e^(-10/30) + e^(-8/30): repeat purchases do not add up)
    B: bought on day 395 (age 5)  -> q_B = e^(-5/30) = 0.846482
    S: C = q_A * 0.508070 + q_B * 0.458738 = 0.777459
       A = q_B * 0.776243 = 0.657076
       B = q_A * 0.553769 = 0.424147
       D = q_A * 0.312754 = 0.239547

(Values are rounded to 6 decimals; the tests compare with exact formulas.)
"""

import dataclasses
import math

import numpy as np
import polars as pl
import pytest

from conftest import make_queries, make_recall_inputs
from hmrec.config import ItemCFConfig
from hmrec.recall.base import RecallInputs, validate_candidates
from hmrec.recall.r4_itemcf import (
    ICF_SPEC,
    NEIGHBOR_IDX,
    SEED_IDX,
    SIMILARITY,
    ItemCFChannel,
    add_similarity,
    build_neighbor_tables,
    cooccurrence_pairs,
    count_neighbors,
    seed_weights,
)

A, B, C, D, E = 0, 1, 2, 3, 4
ANCHOR_DAY = 400
ROWS = [
    (1, A, 390),
    (1, A, 392),
    (1, B, 395),
    (2, A, 380),
    (2, B, 385),
    (2, C, 399),
    (3, A, 350),
    (3, C, 360),
    (3, D, 370),
    (4, A, 398),
    (5, B, 30),
]
CATALOG = {article: 900 + article for article in (A, B, C, D, E)}
ALL_ACTIVE = [A, B, C, D, E]
QUERIES = make_queries([(10, 1), (11, 4), (12, 9)])  # customer 9 has no purchases

W1 = 1 / math.log(3)
W2 = W3 = 1 / math.log(4)
W4 = 1 / math.log(2)
N_A = W1 + W2 + W3 + W4
N_B = W1 + W2
C_AB = W1 + W2


def _config(
    variant: str = "CF3",
    alpha: float = 0.7,
    quota: int = 100,
    seed_window_days: int = 90,
    seed_decay: bool = False,
) -> ItemCFConfig:
    return ItemCFConfig(
        variant=variant,
        fit_window_days=365,
        seed_window_days=seed_window_days,
        alpha=alpha,
        neighbors_per_seed=100,
        seed_decay=seed_decay,
        seed_tau_days=30.0,
        quota=quota,
    )


def _inputs(
    rows: list[tuple[int, int, int]] = ROWS, active: list[int] = ALL_ACTIVE
) -> RecallInputs:
    return make_recall_inputs(rows, ANCHOR_DAY, product_codes=CATALOG, active_pool=active)


def _channel(config: ItemCFConfig, item_chunk_size: int = 2, query_chunk_size: int = 2):
    return ItemCFChannel(config, item_chunk_size, query_chunk_size)


def _similarity(table: pl.DataFrame, seed: int, neighbor: int) -> float:
    row = table.filter((pl.col(SEED_IDX) == seed) & (pl.col(NEIGHBOR_IDX) == neighbor))
    assert row.height == 1
    return row[SIMILARITY].item()


def _query(candidates: pl.DataFrame, query_id: int) -> tuple[list[int], list[float]]:
    one = candidates.filter(pl.col("query_id") == query_id).sort("rank")
    return one["article_idx"].to_list(), one["score"].to_list()


# ---------------------------------------------------------------- hand-computed values


def test_iuf_weights_margins_and_cooccurrence() -> None:
    interactions = _channel(_config()).fit_interactions(_inputs())
    expected_margins = [N_A, N_B, W2 + W3, W3, 0.0]
    assert interactions.item_weight_sum.tolist() == pytest.approx(expected_margins, rel=1e-12)
    pairs = cooccurrence_pairs(interactions, np.array([A]), active_mask=None).sort(NEIGHBOR_IDX)
    assert pairs[NEIGHBOR_IDX].to_list() == [B, C, D]  # A -> A is never a pair
    assert pairs["cooccurrence"].to_list() == pytest.approx([C_AB, W2 + W3, W3], rel=1e-12)


def test_similarity_for_two_alphas_in_one_pass_and_asymmetry() -> None:
    interactions = _channel(_config()).fit_interactions(_inputs())
    tables = build_neighbor_tables(
        interactions,
        seed_items=np.array([A, B, C, D]),
        active_mask=np.ones(5, dtype=bool),
        alphas=[0.5, 0.7],
        neighbors_per_seed=100,
        item_chunk_size=2,
    )
    cosine = C_AB / math.sqrt(N_A * N_B)  # 0.655637
    assert _similarity(tables[0.5], A, B) == pytest.approx(cosine, rel=1e-12)
    assert _similarity(tables[0.5], B, A) == pytest.approx(cosine, rel=1e-12)
    a_to_b = C_AB / (N_A**0.7 * N_B**0.3)  # 0.553769
    b_to_a = C_AB / (N_B**0.7 * N_A**0.3)  # 0.776243
    assert _similarity(tables[0.7], A, B) == pytest.approx(a_to_b, rel=1e-12)
    assert _similarity(tables[0.7], B, A) == pytest.approx(b_to_a, rel=1e-12)
    assert abs(a_to_b - b_to_a) > 0.2


def test_cf0_is_unweighted_cosine() -> None:
    interactions = _channel(_config(variant="CF0", alpha=0.5)).fit_interactions(_inputs())
    assert interactions.item_weight_sum.tolist() == [4.0, 2.0, 2.0, 1.0, 0.0]
    pairs = cooccurrence_pairs(interactions, np.array([A]), active_mask=None)
    similarities = add_similarity(pairs, interactions.item_weight_sum, alpha=0.5)
    assert _similarity(similarities, A, B) == pytest.approx(2 / math.sqrt(8), rel=1e-12)


def test_user_scores_sum_over_seeds_and_keep_bought_articles() -> None:
    candidates = _channel(_config()).generate(QUERIES, _inputs())
    articles, scores = _query(candidates, 10)
    assert articles == [C, A, B, D]
    s_a_to_c = (W2 + W3) / (N_A**0.7 * (W2 + W3) ** 0.3)
    s_b_to_c = W2 / (N_B**0.7 * (W2 + W3) ** 0.3)
    expected = [
        s_a_to_c + s_b_to_c,  # 0.966808
        C_AB / (N_B**0.7 * N_A**0.3),  # 0.776243
        C_AB / (N_A**0.7 * N_B**0.3),  # 0.553769
        W3 / (N_A**0.7 * W3**0.3),  # 0.312754
    ]
    assert scores == pytest.approx(expected, rel=1e-6)
    # Customer 9 has no seeds -> no rows (never padded).
    assert candidates.filter(pl.col("query_id") == 12).height == 0
    validate_candidates(candidates, ICF_SPEC, quota=100)


# ---------------------------------------------------------------- active pool


def test_inactive_seed_still_contributes() -> None:
    # A is not active: it is never output, but U4's only seed A still gives B, C, D.
    candidates = _channel(_config()).generate(QUERIES, _inputs(active=[B, C, D]))
    articles, _ = _query(candidates, 11)
    assert articles == [B, C, D]
    assert A not in candidates["article_idx"].to_list()


def test_active_pool_does_not_change_cooccurrence_or_margins() -> None:
    channel = _channel(_config())
    full = channel.fit_interactions(_inputs(active=ALL_ACTIVE))
    small = channel.fit_interactions(_inputs(active=[C]))
    assert np.array_equal(full.item_weight_sum, small.item_weight_sum)
    seeds = np.array([A, B, C, D])
    full_pairs = cooccurrence_pairs(full, seeds, active_mask=None)
    small_pairs = cooccurrence_pairs(small, seeds, active_mask=None)
    assert full_pairs.equals(small_pairs)
    # The similarity of a kept neighbour does not depend on which others are active.
    tables_full = channel.build_neighbors(QUERIES, _inputs(active=ALL_ACTIVE), [0.7]).tables
    tables_small = channel.build_neighbors(QUERIES, _inputs(active=[C]), [0.7]).tables
    assert _similarity(tables_small[0.7], A, C) == _similarity(tables_full[0.7], A, C)


# ---------------------------------------------------------------- equivalence and chunking


def _random_rows(seed: int) -> list[tuple[int, int, int]]:
    """40 customers x 15 articles, random purchases in days 300..399."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int, int]] = []
    for customer in range(40):
        n_lines = int(rng.integers(1, 8))
        for _ in range(n_lines):
            rows.append((customer, int(rng.integers(0, 15)), int(rng.integers(300, 400))))
    return rows


RANDOM_ROWS = _random_rows(seed=7)
RANDOM_CATALOG = {article: article for article in range(15)}
RANDOM_ACTIVE = [1, 2, 4, 5, 8, 9, 11, 14]
RANDOM_QUERIES = make_queries([(100 + customer, customer) for customer in range(0, 40, 3)])


def _random_inputs() -> RecallInputs:
    return make_recall_inputs(
        RANDOM_ROWS, ANCHOR_DAY, product_codes=RANDOM_CATALOG, active_pool=RANDOM_ACTIVE
    )


def test_filter_before_top_k_equals_full_matrix_then_filter() -> None:
    # Reference: the full seed x all-article similarity matrix in one piece, then
    # drop inactive articles, then take the Top-K_n. Must match the chunked code
    # (which drops inactive articles inside each chunk) exactly.
    k = 3
    interactions = _channel(_config()).fit_interactions(_random_inputs())
    seeds = np.arange(15)
    full = add_similarity(
        cooccurrence_pairs(interactions, seeds, active_mask=None), interactions.item_weight_sum, 0.7
    )
    reference = (
        full.filter(pl.col(NEIGHBOR_IDX).is_in(RANDOM_ACTIVE))
        .sort([SEED_IDX, SIMILARITY, NEIGHBOR_IDX], descending=[False, True, False])
        .group_by(SEED_IDX, maintain_order=True)
        .head(k)
    )
    mask = np.zeros(15, dtype=bool)
    mask[RANDOM_ACTIVE] = True
    chunked = build_neighbor_tables(interactions, seeds, mask, [0.7], k, item_chunk_size=4)[0.7]
    assert chunked.equals(reference.select(chunked.columns))


@pytest.mark.parametrize(("item_chunk_size", "query_chunk_size"), [(1, 1), (2, 3), (4, 5)])
def test_chunked_equals_unchunked(item_chunk_size: int, query_chunk_size: int) -> None:
    config = ItemCFConfig(
        variant="CF3",
        fit_window_days=365,
        seed_window_days=60,
        alpha=0.7,
        neighbors_per_seed=3,
        seed_decay=True,  # chunking must not change decayed scores either
        seed_tau_days=30.0,
        quota=5,
    )
    one_chunk = ItemCFChannel(config, item_chunk_size=1000, query_chunk_size=1000)
    chunked = ItemCFChannel(config, item_chunk_size, query_chunk_size)
    expected = one_chunk.build_neighbors(RANDOM_QUERIES, _random_inputs(), [0.5, 0.7])
    actual = chunked.build_neighbors(RANDOM_QUERIES, _random_inputs(), [0.5, 0.7])
    for alpha in (0.5, 0.7):
        assert actual.tables[alpha].equals(expected.tables[alpha])
    assert chunked.generate(RANDOM_QUERIES, _random_inputs()).equals(
        one_chunk.generate(RANDOM_QUERIES, _random_inputs())
    )


def test_generate_for_alphas_matches_single_alpha_runs() -> None:
    channel = _channel(_config())
    together = channel.generate_for_alphas(RANDOM_QUERIES, _random_inputs(), [0.5, 0.7, 0.9])
    for alpha in (0.5, 0.7, 0.9):
        alone = _channel(_config(alpha=alpha)).generate(RANDOM_QUERIES, _random_inputs())
        assert together[alpha].equals(alone)


# ---------------------------------------------------------------- windows and leakage


def test_ignores_poison_on_and_after_anchor() -> None:
    # On the anchor day: U4 buys D five times and customer 9 buys B. If these
    # leaked, n_w / C_w would change and query 12 would get candidates.
    poison = [(4, D, 400)] * 5 + [(9, B, 400), (1, E, 401), (3, B, 450)]
    channel = _channel(_config())
    assert channel.generate(QUERIES, _inputs(ROWS + poison)).equals(
        channel.generate(QUERIES, _inputs())
    )


def test_seed_window_edge() -> None:
    # Customer 6 bought E on day 310 (age 90, inside the seed window) and C on
    # day 309 (age 91, fit window only). Only E is a seed.
    rows = [*ROWS, (6, E, 310), (6, C, 309), (2, E, 380)]
    neighbors = _channel(_config()).build_neighbors(make_queries([(20, 6)]), _inputs(rows), [0.7])
    assert neighbors.query_seeds["article_idx"].to_list() == [E]


def test_count_neighbors_includes_seeds_without_neighbours() -> None:
    neighbors = pl.DataFrame(
        {SEED_IDX: [1, 1, 3], NEIGHBOR_IDX: [2, 3, 1], SIMILARITY: [0.5, 0.4, 0.3]},
        schema={SEED_IDX: pl.Int32, NEIGHBOR_IDX: pl.Int32, SIMILARITY: pl.Float64},
    )
    counts = count_neighbors(neighbors, np.array([1, 2, 3], dtype=np.int32))
    assert counts.rows() == [(1, 2), (2, 0), (3, 1)]


def test_bad_arguments_raise() -> None:
    interactions = _channel(_config()).fit_interactions(_inputs())
    mask = np.ones(5, dtype=bool)
    with pytest.raises(ValueError, match="alphas must be distinct"):
        build_neighbor_tables(interactions, np.array([A]), mask, [0.7, 0.7], 10, 2)
    with pytest.raises(ValueError, match="item_chunk_size must be > 0"):
        build_neighbor_tables(interactions, np.array([A]), mask, [0.7], 10, 0)


# ---------------------------------------------------------------- seed window and seed decay

S_A_TO_B = C_AB / (N_A**0.7 * N_B**0.3)
S_A_TO_C = (W2 + W3) / (N_A**0.7 * (W2 + W3) ** 0.3)
S_A_TO_D = W3 / (N_A**0.7 * W3**0.3)
S_B_TO_A = C_AB / (N_B**0.7 * N_A**0.3)
S_B_TO_C = W2 / (N_B**0.7 * (W2 + W3) ** 0.3)


def test_seed_decay_hand_computed() -> None:
    candidates = _channel(_config(seed_decay=True)).generate(QUERIES, _inputs())
    articles, scores = _query(candidates, 10)
    q_a = math.exp(-8 / 30)  # last purchase of A at age 8
    q_b = math.exp(-5 / 30)
    assert articles == [C, A, B, D]
    expected = [
        q_a * S_A_TO_C + q_b * S_B_TO_C,  # 0.777459
        q_b * S_B_TO_A,  # 0.657076
        q_a * S_A_TO_B,  # 0.424147
        q_a * S_A_TO_D,  # 0.239547
    ]
    assert scores == pytest.approx(expected, rel=1e-6)
    validate_candidates(candidates, ICF_SPEC, quota=100)


def test_seed_weight_uses_last_purchase_only() -> None:
    history = pl.DataFrame(
        {"customer_idx": [1, 1, 1], "article_idx": [A, A, B], "t_dat_day": [390, 392, 395]},
        schema={"customer_idx": pl.Int32, "article_idx": pl.Int32, "t_dat_day": pl.Int16},
    )
    decayed = seed_weights(history, ANCHOR_DAY, seed_tau_days=30.0).sort("article_idx")
    assert decayed["seed_weight"].to_list() == pytest.approx(
        [math.exp(-8 / 30), math.exp(-5 / 30)], rel=1e-12
    )
    flat = seed_weights(history, ANCHOR_DAY, seed_tau_days=None).sort("article_idx")
    assert flat["seed_weight"].to_list() == [1.0, 1.0]  # exactly 1, not a count


def test_decay_off_scores_are_plain_sums() -> None:
    # Decay off = the original definition S(u, j) = sum of s(i -> j), bit for bit:
    # every seed weight is exactly 1.0. Same expected values as
    # test_user_scores_sum_over_seeds_and_keep_bought_articles.
    _, scores = _query(_channel(_config(seed_decay=False)).generate(QUERIES, _inputs()), 10)
    expected = [S_A_TO_C + S_B_TO_C, S_B_TO_A, S_A_TO_B, S_A_TO_D]
    assert scores == pytest.approx(expected, rel=1e-6)
    # The decay time constant is ignored when the decay is off.
    other_tau = dataclasses.replace(_config(), seed_tau_days=5.0)
    assert (
        _channel(other_tau)
        .generate(QUERIES, _inputs())
        .equals(_channel(_config()).generate(QUERIES, _inputs()))
    )


def test_longer_seed_window_adds_older_seeds() -> None:
    # Customer 8 bought D on day 200 (age 200): a seed only with L_seed = 365.
    rows = [*ROWS, (8, D, 200)]
    queries = make_queries([(30, 8)])
    short = _channel(_config(seed_window_days=90)).generate(queries, _inputs(rows))
    long = _channel(_config(seed_window_days=365)).generate(queries, _inputs(rows))
    assert short.height == 0
    # D's neighbours: C (via U3) and A (via U3).
    assert sorted(long["article_idx"].to_list()) == [A, C]
