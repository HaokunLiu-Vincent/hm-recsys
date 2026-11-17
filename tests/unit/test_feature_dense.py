"""Tests for features.dense (group B) on the fold of feature_example.py.

Histories before day 400 (one per (day, article), newest first; hmrec.data.histories):
    customer 0: day 399 art 0, day 395 art 0, day 390 art 1   -> [0, 0, 1]
    customer 1: day 399 art 0, day 380 art 1, day 30 art 2    -> [0, 1, 2]
    customer 2: nothing                                       -> []

Text vectors (every article has one): t0 = (1, 0), t1 = (0, 1), t2 = (0.6, 0.8), t3 = (-1, 0)
    customer 0: mean (t0 + t0 + t1) / 3 = (2/3, 1/3) -> unit (2, 1) / sqrt(5)
        cosine with t0 = 2 / sqrt(5) = 0.894427; with t2 = (1.2 + 0.8) / sqrt(5) = 0.894427;
        with t3 = -0.894427
    customer 1: mean (1.6, 1.8) / 3 -> unit (1.6, 1.8) / sqrt(5.8);
        with t2 = 2.4 / sqrt(5.8) = 0.996546
    customer 2: no history -> NaN
Image vectors: only articles 0 and 2 have one: i0 = (1, 0), i2 = (0, 1)
    customer 0: of [0, 0, 1] only the two art-0 slots are usable -> mean (1, 0)
        cosine with i0 = 1, with i2 = 0; article 1 has no image vector -> NaN
    customer 1: of [0, 1, 2] usable 0 and 2 -> mean (0.5, 0.5) -> unit (0.7071, 0.7071);
        with i2 = 0.707107
    customer 2: NaN

R6 vectors: users 0, 1, 2 and all four articles. R5 vectors: users 0, 1 and articles 0, 1 only.
R1 bucket table: (AGE_25_34, art 0) = 1.5 and (AGE_45_54, art 3) = 0.25; nothing else.

Similarity to single past purchases (R6 article vectors v0 = (1, 0), v1 = (0, 1), v2 = (1, 1),
v3 = (2, -1); the function only takes dot products, unit length is checked when vectors are read):
    customer 0, history [0, 0, 1]:
        candidate 0: (1, 1, 0)  -> max 1, mean of all three (fewer than 5) = 2/3
        candidate 2: (1, 1, 1)  -> max 1, mean 1
        candidate 3: (2, 2, -1) -> max 2, mean 1
    customer 1, history [0, 1, 2]: candidate 3: (2, -1, 1) -> max 2, mean 2/3
    customer 2, no history: NaN
"""

import dataclasses
import math

import numpy as np
import polars as pl
import pytest
from conftest import FIXTURE_DAY_ZERO
from feature_example import ANCHOR, ARTICLES, CUSTOMERS, LINES, POISON, QUERIES, make_transactions

from hmrec.data.histories import recent_purchase_histories
from hmrec.features.base import HistoryArrays
from hmrec.features.dense import (
    DenseSources,
    add_dense_features,
    add_history_similarity_features,
    history_mean_vectors,
    history_similarities,
    normalize_rows,
    similarity_vectors,
)
from hmrec.vectors import UserItemVectors

N_ARTICLES = ARTICLES.height
TEXT = np.array([[1, 0], [0, 1], [0.6, 0.8], [-1, 0]], dtype=np.float32)
TEXT_MASK = np.ones(N_ARTICLES, dtype=np.bool_)
IMAGE = np.array([[1, 0], [0, 0], [0, 1], [0, 0]], dtype=np.float32)
IMAGE_MASK = np.array([True, False, True, False])
CUSTOMER_IDX = np.array([0, 1, 2], dtype=np.int32)
SQRT5 = math.sqrt(5)
TWOTOWER = UserItemVectors(
    user_customer_idx=CUSTOMER_IDX,
    user_vectors=np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
    item_article_idx=np.arange(4, dtype=np.int32),
    item_vectors=np.array([[1, 0], [0, 1], [1, 1], [2, -1]], dtype=np.float32),
)
WALS = UserItemVectors(
    user_customer_idx=np.array([0, 1], dtype=np.int32),
    user_vectors=np.array([[1, 1], [2, 0]], dtype=np.float32),
    item_article_idx=np.array([0, 1], dtype=np.int32),
    item_vectors=np.array([[0.5, 0.25], [1, 3]], dtype=np.float32),
)
BUCKET_SCORES = pl.DataFrame(
    {"age_bucket": ["AGE_25_34", "AGE_45_54"], "article_idx": [0, 3], "score": [1.5, 0.25]},
    schema={"age_bucket": pl.String, "article_idx": pl.Int32, "score": pl.Float32},
)


def _histories(lines: list = LINES) -> tuple[np.ndarray, np.ndarray]:
    return recent_purchase_histories(
        make_transactions(lines).lazy(), CUSTOMER_IDX, ANCHOR, FIXTURE_DAY_ZERO, 20, N_ARTICLES
    )


def make_dense_sources(lines: list = LINES, users_per_chunk: int = 2) -> DenseSources:
    history, mask = _histories(lines)
    return DenseSources(
        twotower=TWOTOWER,
        wals=WALS,
        bucket_scores=BUCKET_SCORES,
        image=similarity_vectors(CUSTOMER_IDX, history, mask, IMAGE, IMAGE_MASK, users_per_chunk),
        text=similarity_vectors(CUSTOMER_IDX, history, mask, TEXT, TEXT_MASK, users_per_chunk),
        history=HistoryArrays(CUSTOMER_IDX, history, mask),
        history_similarity_top_k=5,
        pair_chunk_size=3,
        embedding_pair_chunk_size=2,
    )


def _chunk(pairs: list[tuple[int, int]]) -> pl.DataFrame:
    table = pl.DataFrame(
        {"query_id": [q for q, _ in pairs], "article_idx": [a for _, a in pairs]},
        schema={"query_id": pl.Int32, "article_idx": pl.Int32},
    )
    return table.join(QUERIES, on="query_id", maintain_order="left")


ALL_PAIRS = [(query, article) for query in (1, 2, 3) for article in range(4)]


def _dense(lines: list = LINES, users_per_chunk: int = 2) -> pl.DataFrame:
    sources = make_dense_sources(lines, users_per_chunk)
    return add_dense_features(_chunk(ALL_PAIRS), sources, CUSTOMERS)


def _value(table: pl.DataFrame, query_id: int, article_idx: int, name: str) -> float:
    row = table.filter((pl.col("query_id") == query_id) & (pl.col("article_idx") == article_idx))
    return row[name][0]


def test_histories_are_the_shared_definition() -> None:
    history, mask = _histories()
    assert history[0, :3].tolist() == [0, 0, 1] and mask[0].sum() == 3
    assert history[1, :3].tolist() == [0, 1, 2] and mask[1].sum() == 3
    assert mask[2].sum() == 0 and history[2, 0] == N_ARTICLES  # padding, not a real article


def test_mean_vector_is_normalised_and_skips_articles_without_a_vector() -> None:
    history, mask = _histories()
    vectors, has_vector = history_mean_vectors(history, mask, IMAGE, IMAGE_MASK, users_per_chunk=2)
    assert has_vector.tolist() == [True, True, False]
    assert vectors[0].tolist() == pytest.approx([1.0, 0.0])
    assert vectors[1].tolist() == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2)])
    text_vectors, text_has = history_mean_vectors(history, mask, TEXT, TEXT_MASK, users_per_chunk=5)
    assert text_has.tolist() == [True, True, False]
    assert text_vectors[0].tolist() == pytest.approx([2 / SQRT5, 1 / SQRT5])
    normalised, has_norm = normalize_rows(np.array([[3, 4], [0, 0]], dtype=np.float32))
    assert normalised.tolist() == [pytest.approx([0.6, 0.8]), [0.0, 0.0]]
    assert has_norm.tolist() == [True, False]


def test_text_and_image_similarity_hand_computed() -> None:
    table = _dense()
    assert _value(table, 1, 0, "txt_sim_hist") == pytest.approx(2 / SQRT5, rel=1e-6)
    assert _value(table, 1, 2, "txt_sim_hist") == pytest.approx(2 / SQRT5, rel=1e-6)
    assert _value(table, 1, 3, "txt_sim_hist") == pytest.approx(-2 / SQRT5, rel=1e-6)
    assert _value(table, 2, 2, "txt_sim_hist") == pytest.approx(2.4 / math.sqrt(5.8), rel=1e-6)
    assert _value(table, 1, 0, "img_sim_hist") == pytest.approx(1.0, rel=1e-6)
    assert _value(table, 1, 2, "img_sim_hist") == pytest.approx(0.0, abs=1e-6)
    assert _value(table, 2, 2, "img_sim_hist") == pytest.approx(1 / math.sqrt(2), rel=1e-6)


def test_missing_rules_of_the_similarities() -> None:
    table = _dense()
    # The article has no image vector: NaN, whoever the user is.
    assert math.isnan(_value(table, 1, 1, "img_sim_hist"))
    assert math.isnan(_value(table, 2, 3, "img_sim_hist"))
    # The user has no history: both similarities are NaN.
    for article in range(4):
        assert math.isnan(_value(table, 3, article, "img_sim_hist"))
        assert math.isnan(_value(table, 3, article, "txt_sim_hist"))
    # Text covers every article: with a history, txt_sim_hist is never NaN.
    assert table.filter(pl.col("query_id") != 3)["txt_sim_hist"].is_nan().sum() == 0


def test_dots_bucket_score_and_their_missing_rules() -> None:
    table = _dense()
    # tt_dot: every candidate has one. u0 = (1, 2); v2 = (1, 1) -> 3; v3 = (2, -1) -> 0.
    assert _value(table, 1, 2, "tt_dot") == 3.0 and _value(table, 1, 3, "tt_dot") == 0.0
    assert _value(table, 3, 0, "tt_dot") == 5.0  # u2 = (5, 6) . (1, 0)
    assert table["tt_dot"].is_nan().sum() == 0
    # wals_dot: u0 = (1, 1) . v0 = (0.5, 0.25) -> 0.75; NaN without a user or article vector.
    assert _value(table, 1, 0, "wals_dot") == 0.75
    assert math.isnan(_value(table, 1, 2, "wals_dot"))  # article 2 has no WALS vector
    assert math.isnan(_value(table, 3, 0, "wals_dot"))  # customer 2 has no WALS vector
    # pop_bucket_score: the table entry of the user's bucket, 0 if there is none.
    assert _value(table, 1, 0, "pop_bucket_score") == 1.5  # (AGE_25_34, 0)
    assert _value(table, 3, 3, "pop_bucket_score") == 0.25  # (AGE_45_54, 3)
    assert _value(table, 1, 3, "pop_bucket_score") == 0.0  # other bucket's entry does not count
    assert _value(table, 2, 0, "pop_bucket_score") == 0.0  # AGE_UNKNOWN has no entry


def test_rows_dtypes_and_no_extra_columns() -> None:
    table = _dense()
    assert table.select("query_id", "article_idx").rows() == ALL_PAIRS  # order unchanged
    names = ["tt_dot", "wals_dot", "img_sim_hist", "txt_sim_hist", "pop_bucket_score"]
    assert table.columns == ["query_id", "article_idx", "customer_idx", *names]
    for name in names:
        assert str(table.schema[name]) == "Float32" and table[name].null_count() == 0, name


def test_poison_on_and_after_the_anchor_changes_nothing() -> None:
    # The poison gives customer 2 purchases on day 400 / 401 and customer 0 an article-1
    # purchase on day 400: histories, and so both similarities, must not move.
    assert _dense(LINES + POISON).equals(_dense())


def test_chunk_sizes_do_not_change_the_similarities() -> None:
    assert _dense(users_per_chunk=1).equals(_dense(users_per_chunk=50))
    history, mask = _histories()
    with pytest.raises(ValueError, match="users_per_chunk must be > 0"):
        history_mean_vectors(history, mask, TEXT, TEXT_MASK, users_per_chunk=0)


# ---------------------------------------------------------------- tt_sim_max_hist, tt_sim_top5_hist


def _similarity(lines: list = LINES) -> pl.DataFrame:
    return add_history_similarity_features(_chunk(ALL_PAIRS), make_dense_sources(lines))


def test_largest_and_top_similarity_to_past_purchases_hand_computed() -> None:
    table = _similarity()
    assert table.select("query_id", "article_idx").rows() == ALL_PAIRS  # order unchanged
    expected = {(1, 0): (1.0, 2 / 3), (1, 2): (1.0, 1.0), (1, 3): (2.0, 1.0), (2, 3): (2.0, 2 / 3)}
    for (query_id, article_idx), (largest, top_mean) in expected.items():
        assert _value(table, query_id, article_idx, "tt_sim_max_hist") == largest
        assert _value(table, query_id, article_idx, "tt_sim_top5_hist") == pytest.approx(top_mean)
    for name in ("tt_sim_max_hist", "tt_sim_top5_hist"):
        assert str(table.schema[name]) == "Float32" and table[name].null_count() == 0
        # Customer 2 has no purchase: NaN on all four candidates, and only there.
        assert table.filter(pl.col("query_id") == 3)[name].is_nan().all()
        assert table.filter(pl.col("query_id") != 3)[name].is_nan().sum() == 0


def test_top_similarity_averages_only_the_five_largest() -> None:
    # One user with 7 past purchases (articles 0..6) and candidate article 7. With
    # one-dimensional vectors v_j = 0.1 * (j + 1) and v_7 = 1, the similarities are
    # 0.1, 0.2, ..., 0.7: max 0.7; the five largest are 0.3 ... 0.7, mean 0.5.
    values = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [1.0]], dtype=np.float32)
    vectors = UserItemVectors(
        user_customer_idx=np.array([0], dtype=np.int32),
        user_vectors=np.array([[1.0]], dtype=np.float32),
        item_article_idx=np.arange(8, dtype=np.int32),
        item_vectors=values,
    )
    seven = HistoryArrays(
        np.array([0], dtype=np.int32),
        np.arange(7, dtype=np.int32).reshape(1, 7),
        np.ones((1, 7), dtype=np.bool_),
    )
    customer, candidate = np.array([0], dtype=np.int32), np.array([7], dtype=np.int32)
    largest, top_mean = history_similarities(customer, candidate, vectors, seven, top_k=5)
    assert largest[0] == pytest.approx(0.7) and top_mean[0] == pytest.approx(0.5)
    # The order of the history does not matter.
    reversed_history = dataclasses.replace(seven, history=seven.history[:, ::-1].copy())
    again = history_similarities(customer, candidate, vectors, reversed_history, top_k=5)
    assert again[0][0] == largest[0] and again[1][0] == top_mean[0]
    # top_k = 2: the two largest, 0.6 and 0.7.
    assert history_similarities(customer, candidate, vectors, seven, 2)[1][0] == pytest.approx(0.65)
    with pytest.raises(ValueError, match="top_k must be > 0"):
        history_similarities(customer, candidate, vectors, seven, 0)


def test_similarity_needs_a_vector_for_every_article_and_a_row_for_every_customer() -> None:
    sources = make_dense_sources()
    unknown_article = np.array([9], dtype=np.int32)
    with pytest.raises(ValueError, match="has no R6 vector"):
        history_similarities(CUSTOMER_IDX[:1], unknown_article, TWOTOWER, sources.history, 5)
    unknown_customer = np.array([7], dtype=np.int32)
    with pytest.raises(ValueError, match="no history row"):
        history_similarities(unknown_customer, CUSTOMER_IDX[:1], TWOTOWER, sources.history, 5)


def test_similarity_ignores_purchases_on_and_after_the_anchor() -> None:
    # The poison gives customer 2 (no history) purchases on day 400 and 401: still NaN.
    assert _similarity(LINES + POISON).equals(_similarity())
