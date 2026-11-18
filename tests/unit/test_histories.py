"""Tests for data.histories: the one definition of "a customer's last purchases".

Lines (customer, article, day), on FIXTURE_DAY_ZERO:
    customer 0: (0, 1, 3), (0, 2, 8), (0, 2, 8), (0, 4, 8), (0, 1, 9)
        one purchase per (day, article): day 9 art 1; day 8 art 2 and art 4; day 3 art 1
        newest day first, ascending article within a day -> [1, 2, 4, 1]
    customer 1: (1, 4, 9) -> [4]
    customer 2: nothing
Poison (anchor day 10 and later): (0, 5, 10), (1, 3, 12), (2, 0, 10).
"""

import numpy as np
import pytest

from conftest import FIXTURE_DAY_ZERO, make_queries, make_recall_inputs, make_transactions
from hmrec.data.day_index import day_index_to_date
from hmrec.data.histories import (
    build_histories,
    daily_purchases,
    load_customer_lines_before,
    recent_purchase_histories,
)
from hmrec.recall.twotower.infer import query_histories

LINES = [(0, 1, 3), (0, 2, 8), (0, 2, 8), (0, 4, 8), (0, 1, 9), (1, 4, 9)]
POISON = [(0, 5, 10), (1, 3, 12), (2, 0, 10)]
ANCHOR = day_index_to_date(10, FIXTURE_DAY_ZERO)
CUSTOMERS = np.array([0, 1, 2], dtype=np.int32)
PAD = 6


def _histories(lines: list, length: int = 5) -> tuple[np.ndarray, np.ndarray]:
    transactions = make_transactions(lines).lazy()
    return recent_purchase_histories(transactions, CUSTOMERS, ANCHOR, FIXTURE_DAY_ZERO, length, PAD)


def test_last_purchases_hand_computed() -> None:
    history, mask = _histories(LINES)
    # The same article on two days fills two slots; two lines on one day fill one.
    assert history.tolist() == [[1, 2, 4, 1, PAD], [4, PAD, PAD, PAD, PAD], [PAD] * 5]
    assert mask.sum(axis=1).tolist() == [4, 1, 0]
    assert history.dtype == np.int32 and mask.dtype == np.bool_


def test_only_the_most_recent_purchases_are_kept() -> None:
    history, mask = _histories(LINES, length=2)
    assert history.tolist() == [[1, 2], [4, PAD], [PAD, PAD]]
    assert mask.tolist() == [[True, True], [True, False], [False, False]]


def test_purchases_on_and_after_the_anchor_are_never_in_a_history() -> None:
    clean_history, clean_mask = _histories(LINES)
    poisoned_history, poisoned_mask = _histories(LINES + POISON)
    assert np.array_equal(clean_history, poisoned_history)
    assert np.array_equal(clean_mask, poisoned_mask)
    lines = load_customer_lines_before(
        make_transactions(LINES + POISON).lazy(), CUSTOMERS, ANCHOR, FIXTURE_DAY_ZERO
    )
    assert lines["t_dat_day"].max() == 9


def test_r6_inference_uses_this_very_function() -> None:
    # The feature code calls recent_purchase_histories; R6 inference (query_histories)
    # must give the same arrays for the same customers and anchor.
    inputs = make_recall_inputs(LINES + POISON, 10)
    queries = make_queries([(1, 0), (2, 1), (3, 2)])
    r6_history, r6_mask = query_histories(inputs, queries["customer_idx"].to_numpy(), 5, PAD)
    history, mask = _histories(LINES + POISON)
    assert np.array_equal(r6_history, history) and np.array_equal(r6_mask, mask)


def test_an_article_equal_to_the_padding_index_is_refused() -> None:
    events = daily_purchases(make_transactions([(0, PAD, 3)]))
    with pytest.raises(ValueError, match="article_idx must be in"):
        build_histories(events, CUSTOMERS[:1], np.array([10], dtype=np.int32), 5, PAD)


def test_customers_without_any_purchase_get_empty_histories() -> None:
    # Customers 7 and 8 never bought anything, so there is not a single purchase line
    # to read: every slot holds the padding index and the mask is False everywhere.
    never_bought = np.array([7, 8], dtype=np.int32)
    history, mask = recent_purchase_histories(
        make_transactions(LINES).lazy(), never_bought, ANCHOR, FIXTURE_DAY_ZERO, 3, PAD
    )
    assert history.tolist() == [[PAD, PAD, PAD], [PAD, PAD, PAD]]
    assert not mask.any() and mask.shape == (2, 3)
