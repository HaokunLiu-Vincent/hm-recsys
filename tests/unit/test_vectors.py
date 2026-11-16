"""Tests for hmrec.vectors: lookups, dot products with NaN for missing, save / load.

Hand example (dim 2):
    users:    customer 2 -> (1, 0), customer 5 -> (0, 2), customer 9 -> (1, 1)
    articles: article 10 -> (3, 4), article 20 -> (-1, 0.5)
    dot(5, 10) = 0*3 + 2*4 = 8      dot(2, 20) = 1*(-1) + 0*0.5 = -1
    dot(9, 10) = 3 + 4 = 7          dot(9, 20) = -1 + 0.5 = -0.5
    customer 7 and article 30 have no vector -> NaN
"""

import json
from pathlib import Path

import numpy as np
import pytest

from hmrec.vectors import (
    METADATA_FILE,
    UserItemVectors,
    dot_pairs,
    load_vectors,
    lookup_rows,
    save_vectors,
    select_users,
)


def _hand_vectors() -> UserItemVectors:
    return UserItemVectors(
        user_customer_idx=np.array([2, 5, 9], dtype=np.int32),
        user_vectors=np.array([[1, 0], [0, 2], [1, 1]], dtype=np.float32),
        item_article_idx=np.array([10, 20], dtype=np.int32),
        item_vectors=np.array([[3, 4], [-1, 0.5]], dtype=np.float32),
    )


def _random_vectors(seed: int = 3) -> UserItemVectors:
    rng = np.random.default_rng(seed)
    return UserItemVectors(
        user_customer_idx=np.arange(0, 60, 2, dtype=np.int32),
        user_vectors=rng.standard_normal((30, 8)).astype(np.float32),
        item_article_idx=np.arange(0, 40, 3, dtype=np.int32),
        item_vectors=rng.standard_normal((14, 8)).astype(np.float32),
    )


# ---------------------------------------------------------------- dot products


def test_dot_pairs_hand_values_and_missing() -> None:
    customers = np.array([5, 2, 9, 9, 7, 2])
    articles = np.array([10, 20, 10, 20, 10, 30])
    result = dot_pairs(_hand_vectors(), customers, articles, chunk_size=4)
    assert result.dtype == np.float32
    assert result[:4].tolist() == [8.0, -1.0, 7.0, -0.5]
    assert np.isnan(result[4])  # customer 7 has no vector
    assert np.isnan(result[5])  # article 30 has no vector


def test_dot_pairs_do_not_depend_on_chunk_size() -> None:
    vectors = _random_vectors()
    rng = np.random.default_rng(0)
    customers = rng.integers(0, 62, size=500)
    articles = rng.integers(0, 42, size=500)
    one_chunk = dot_pairs(vectors, customers, articles, chunk_size=10_000)
    for chunk_size in (1, 7, 128):
        chunked = dot_pairs(vectors, customers, articles, chunk_size)
        assert np.array_equal(chunked, one_chunk, equal_nan=True)
    # Some pairs are missing (odd customers, articles not divisible by 3) and some are not.
    assert 0 < int(np.isnan(one_chunk).sum()) < 500


def test_lookup_rows_edges() -> None:
    index = np.array([3, 8, 20], dtype=np.int32)
    assert lookup_rows(index, np.array([3, 20, 1, 21, 8, 9])).tolist() == [0, 2, -1, -1, 1, -1]
    empty = np.array([], dtype=np.int32)
    assert lookup_rows(empty, np.array([1, 2])).tolist() == [-1, -1]


def test_dot_pairs_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="same length"):
        dot_pairs(_hand_vectors(), np.array([2, 5]), np.array([10]), chunk_size=1)
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        dot_pairs(_hand_vectors(), np.array([2]), np.array([10]), chunk_size=0)


# ---------------------------------------------------------------- validation


def test_unsorted_or_duplicate_index_raises() -> None:
    hand = _hand_vectors()
    with pytest.raises(ValueError, match="strictly increasing"):
        UserItemVectors(
            np.array([5, 2, 9], dtype=np.int32),
            hand.user_vectors,
            hand.item_article_idx,
            hand.item_vectors,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        UserItemVectors(
            hand.user_customer_idx,
            hand.user_vectors,
            np.array([10, 10], dtype=np.int32),
            hand.item_vectors,
        )


def test_wrong_dtype_shape_or_values_raise() -> None:
    hand = _hand_vectors()
    with pytest.raises(ValueError, match="int32"):
        UserItemVectors(
            hand.user_customer_idx.astype(np.int64),
            hand.user_vectors,
            hand.item_article_idx,
            hand.item_vectors,
        )
    with pytest.raises(ValueError, match="float32"):
        UserItemVectors(
            hand.user_customer_idx,
            hand.user_vectors.astype(np.float64),
            hand.item_article_idx,
            hand.item_vectors,
        )
    with pytest.raises(ValueError, match="rows"):
        UserItemVectors(
            hand.user_customer_idx,
            hand.user_vectors[:2],
            hand.item_article_idx,
            hand.item_vectors,
        )
    with pytest.raises(ValueError, match="NaN or infinite"):
        bad = hand.item_vectors.copy()
        bad[0, 0] = np.nan
        UserItemVectors(hand.user_customer_idx, hand.user_vectors, hand.item_article_idx, bad)
    with pytest.raises(ValueError, match="dim"):
        UserItemVectors(
            hand.user_customer_idx,
            hand.user_vectors,
            hand.item_article_idx,
            np.zeros((2, 3), dtype=np.float32),
        )


# ---------------------------------------------------------------- save / load


def test_save_then_load_is_identical(tmp_path: Path) -> None:
    vectors = _random_vectors()
    directory = tmp_path / "models" / "wals" / "D1"
    save_vectors(vectors, directory, "wals", "D1", keep_customer_idx=None, overwrite=False)
    loaded = load_vectors(directory, "wals", "D1")
    assert np.array_equal(loaded.user_customer_idx, vectors.user_customer_idx)
    assert np.array_equal(loaded.user_vectors, vectors.user_vectors)
    assert np.array_equal(loaded.item_article_idx, vectors.item_article_idx)
    assert np.array_equal(loaded.item_vectors, vectors.item_vectors)
    metadata = json.loads((directory / METADATA_FILE).read_text())
    assert metadata["source"] == "wals"
    assert metadata["fold"] == "D1"
    assert metadata["all_users_saved"] is True


def test_load_checks_source_and_fold(tmp_path: Path) -> None:
    save_vectors(_hand_vectors(), tmp_path, "wals", "F1", keep_customer_idx=None, overwrite=False)
    with pytest.raises(ValueError, match="expected twotower / F1"):
        load_vectors(tmp_path, "twotower", "F1")
    with pytest.raises(ValueError, match="expected wals / F2"):
        load_vectors(tmp_path, "wals", "F2")


def test_save_refuses_overwrite_and_unknown_source(tmp_path: Path) -> None:
    save_vectors(_hand_vectors(), tmp_path, "wals", "D1", keep_customer_idx=None, overwrite=False)
    with pytest.raises(FileExistsError):
        save_vectors(
            _hand_vectors(), tmp_path, "wals", "D1", keep_customer_idx=None, overwrite=False
        )
    save_vectors(_hand_vectors(), tmp_path, "wals", "D1", keep_customer_idx=None, overwrite=True)
    with pytest.raises(ValueError, match="source must be one of"):
        save_vectors(_hand_vectors(), tmp_path / "x", "als", "D1", None, overwrite=False)


def test_partial_user_save_keeps_dots_and_gives_nan_for_others(tmp_path: Path) -> None:
    vectors = _random_vectors()
    keep = np.array([0, 10, 58, 999])  # 999 has no vector at all: simply absent
    save_vectors(vectors, tmp_path, "wals", "F1", keep_customer_idx=keep, overwrite=False)
    loaded = load_vectors(tmp_path, "wals", "F1")
    assert loaded.user_customer_idx.tolist() == [0, 10, 58]
    assert np.array_equal(loaded.item_vectors, vectors.item_vectors)  # all articles kept
    customers = np.array([0, 10, 58, 2, 4, 0])
    articles = np.array([0, 3, 39, 0, 3, 6])
    full = dot_pairs(vectors, customers, articles, chunk_size=4)
    partial = dot_pairs(loaded, customers, articles, chunk_size=4)
    saved = np.array([True, True, True, False, False, True])
    assert np.array_equal(partial[saved], full[saved])
    assert np.isnan(partial[~saved]).all()
    assert not np.isnan(full).any()


def test_select_users_keeps_all_articles() -> None:
    selected = select_users(_hand_vectors(), np.array([9, 2]))
    assert selected.user_customer_idx.tolist() == [2, 9]
    assert selected.user_vectors.tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert selected.item_article_idx.tolist() == [10, 20]
