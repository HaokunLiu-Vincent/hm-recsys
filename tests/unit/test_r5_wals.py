"""Tests for R5 WALS.

Confidence matrix (hand-computed). Anchor day 200, window 180 days (days 20..199),
decay tau 60, alpha_conf 20.

    customer 1, article 10: day 199 twice (age 1, a repeat line) and day 140 (age 60)
        W2: r = 2 * e^(-1/60) + e^(-1) = 2 * 0.983471 + 0.367879 = 2.334822
            c = 1 + 20 * 2.334822 = 47.696447
        W0: r = 1 -> c = 21
    customer 1, article 12: day 80 (age 120) -> r = e^(-2) = 0.135335, c = 3.706706
    customer 3, article 12: day 20 (age 180, first day of the window) -> r = e^(-3) = 0.049787,
        c = 1.995741
    customer 4, article 11: day 19 (age 181) -> outside the window: customer 4 and
        article 11 get no row / column at all.

Matrix rows = customers [1, 3], columns = articles [10, 12].
"""

import math

import numpy as np
import polars as pl
import pytest
from conftest import make_queries, make_recall_inputs

from hmrec.config import WalsConfig
from hmrec.recall.base import RecallInputs, validate_candidates
from hmrec.recall.r5_wals import WALS_SPEC, WalsChannel, build_als_model, select_top_items
from hmrec.vectors import dot_pairs

ANCHOR_DAY = 200
HAND_ROWS = [(1, 10, 199), (1, 10, 199), (1, 10, 140), (1, 12, 80), (3, 12, 20), (4, 11, 19)]


def _config(variant: str = "W2", seed: int = 42, quota: int = 5) -> WalsConfig:
    return WalsConfig(
        variant=variant,
        window_days=180,
        decay_tau_days=60.0,
        alpha_conf=20.0,
        regularization=0.1,
        factors=4,
        iterations=5,
        random_seed=seed,
        quota=quota,
    )


def _channel(config: WalsConfig) -> WalsChannel:
    # 2 threads: reproducibility must also hold with several threads.
    return WalsChannel(config, num_threads=2, query_chunk_size=3, pair_chunk_size=7)


def _random_rows(seed: int) -> list[tuple[int, int, int]]:
    """30 customers x 12 articles, random purchases in days 100..199."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int, int]] = []
    for customer in range(30):
        for _ in range(int(rng.integers(1, 9))):
            rows.append((customer, int(rng.integers(0, 12)), int(rng.integers(100, 200))))
    return rows


RANDOM_ROWS = _random_rows(seed=11)
RANDOM_ACTIVE = [0, 2, 3, 5, 6, 8, 9, 11]
# Customers 0..29 have purchases; 40 and 41 do not.
RANDOM_QUERIES = make_queries([(500 + c, c) for c in [*range(0, 30, 2), 40, 41]])


def _random_inputs(rows: list[tuple[int, int, int]] = RANDOM_ROWS) -> RecallInputs:
    return make_recall_inputs(rows, ANCHOR_DAY, active_pool=RANDOM_ACTIVE)


# ---------------------------------------------------------------- confidence matrix


def test_w2_confidence_sums_repeat_lines_with_decay() -> None:
    confidence = _channel(_config("W2")).build_confidence(make_recall_inputs(HAND_ROWS, ANCHOR_DAY))
    assert confidence.user_customer_idx.tolist() == [1, 3]
    assert confidence.item_article_idx.tolist() == [10, 12]
    assert confidence.matrix.dtype == np.float32
    dense = confidence.matrix.toarray()
    r_1_10 = 2 * math.exp(-1 / 60) + math.exp(-1)
    expected = [
        [1 + 20 * r_1_10, 1 + 20 * math.exp(-2)],  # 47.696447, 3.706706
        [0.0, 1 + 20 * math.exp(-3)],  # customer 3 never bought article 10
    ]
    assert dense.ravel().tolist() == pytest.approx(np.ravel(expected).tolist(), rel=1e-6)


def test_w0_confidence_ignores_repeats_and_age() -> None:
    confidence = _channel(_config("W0")).build_confidence(make_recall_inputs(HAND_ROWS, ANCHOR_DAY))
    assert confidence.matrix.toarray().tolist() == [[21.0, 21.0], [0.0, 21.0]]


def test_library_alpha_is_one() -> None:
    # The CSR values already are c_ui = 1 + alpha_conf * r_ui; a library alpha != 1
    # would scale them a second time.
    model = build_als_model(_config(), num_threads=2)
    assert model.alpha == 1.0
    assert model.factors == 4
    assert model.regularization == 0.1


# ---------------------------------------------------------------- fitting and output


def test_fixed_seed_is_reproducible() -> None:
    first = _channel(_config()).fit(_random_inputs())
    second = _channel(_config()).fit(_random_inputs())
    assert np.array_equal(first.user_vectors, second.user_vectors)
    assert np.array_equal(first.item_vectors, second.item_vectors)
    other_seed = _channel(_config(seed=7)).fit(_random_inputs())
    assert not np.array_equal(first.user_vectors, other_seed.user_vectors)
    assert (
        _channel(_config())
        .generate(RANDOM_QUERIES, _random_inputs())
        .equals(_channel(_config()).generate(RANDOM_QUERIES, _random_inputs()))
    )


def test_output_schema_active_pool_and_empty_users() -> None:
    candidates = _channel(_config(quota=5)).generate(RANDOM_QUERIES, _random_inputs())
    validate_candidates(candidates, WALS_SPEC, quota=5)
    assert set(candidates["article_idx"].to_list()) <= set(RANDOM_ACTIVE)
    # Customers 40 and 41 have no purchase in the window: no rows.
    assert candidates.filter(pl.col("query_id").is_in([540, 541])).height == 0
    counts = candidates.group_by("query_id").len()["len"].to_list()
    assert counts == [5] * 15


def test_bought_articles_are_not_excluded() -> None:
    # Quota >= |A(a)|: every active article is a candidate for every scored query,
    # including the ones the customer already bought.
    candidates = _channel(_config(quota=50)).generate(RANDOM_QUERIES, _random_inputs())
    per_query = candidates.group_by("query_id").agg(pl.col("article_idx").sort())
    for articles in per_query["article_idx"].to_list():
        assert articles == RANDOM_ACTIVE


def test_score_equals_vectors_dot_exactly() -> None:
    channel = _channel(_config())
    inputs = _random_inputs()
    vectors = channel.fit(inputs)
    candidates = channel.candidates_from_vectors(RANDOM_QUERIES, inputs, vectors)
    customers = candidates.join(RANDOM_QUERIES, on="query_id")["customer_idx"].to_numpy()
    dots = dot_pairs(vectors, customers, candidates["article_idx"].to_numpy(), chunk_size=3)
    assert np.array_equal(candidates["score"].to_numpy(), dots)


def test_selection_matches_exact_ranking() -> None:
    # Brute force: score every active article with dot_pairs and sort by
    # (score DESC, article_idx ASC); the channel must keep the same Top-5.
    channel = _channel(_config(quota=5))
    inputs = _random_inputs()
    vectors = channel.fit(inputs)
    candidates = channel.candidates_from_vectors(RANDOM_QUERIES, inputs, vectors)
    active = np.array(RANDOM_ACTIVE)
    for customer in range(0, 30, 2):
        dots = dot_pairs(vectors, np.full(active.size, customer), active, chunk_size=100)
        order = sorted(range(active.size), key=lambda k: (-dots[k], active[k]))
        expected = active[order[:5]].tolist()
        actual = candidates.filter(pl.col("query_id") == 500 + customer).sort("rank")
        assert actual["article_idx"].to_list() == expected


def test_select_top_items_keeps_ties_at_the_cut() -> None:
    users = np.array([[1.0, 0.0]], dtype=np.float32)
    items = np.array([[3.0, 0.0], [1.0, 5.0], [1.0, -2.0], [2.0, 0.0]], dtype=np.float32)
    # Scores 3, 1, 1, 2: with quota 3 the cut is at 1, and both items with score 1 stay.
    _, item_rows = select_top_items(users, items, quota=3)
    assert sorted(item_rows.tolist()) == [0, 1, 2, 3]


def test_ignores_poison_on_and_after_anchor() -> None:
    poison = [(0, 3, 200)] * 5 + [(40, 5, 200), (1, 2, 230)]
    channel = _channel(_config())
    assert channel.generate(RANDOM_QUERIES, _random_inputs(RANDOM_ROWS + poison)).equals(
        channel.generate(RANDOM_QUERIES, _random_inputs())
    )
