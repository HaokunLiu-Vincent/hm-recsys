"""Tests for hmrec.diagnostics.concentration (table B), with hand-computed numbers."""

import polars as pl
import pytest

from hmrec.diagnostics.concentration import (
    concentration,
    concentration_columns,
    concentration_row,
    first_sale_days,
    top_pair_share,
)

ANCHOR_DAY = 100


def _label_counts(counts: dict[int, int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"article_idx": list(counts), "n_label_pairs": list(counts.values())},
        schema={"article_idx": pl.Int32, "n_label_pairs": pl.Int64},
    )


def _first_sale(days: dict[int, int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"article_idx": list(days), "first_sale_day": list(days.values())},
        schema={"article_idx": pl.Int32, "first_sale_day": pl.Int16},
    )


def test_first_sale_days_takes_the_earliest_line_of_each_article() -> None:
    lines = pl.DataFrame(
        {"article_idx": [7, 3, 7, 3, 9], "t_dat_day": [50, 20, 12, 21, 99]},
        schema={"article_idx": pl.Int32, "t_dat_day": pl.Int16},
    )
    first = first_sale_days(lines)
    assert first.schema == {"article_idx": pl.Int32, "first_sale_day": pl.Int16}
    assert first.rows() == [(3, 20), (7, 12), (9, 99)]


def test_top_pair_share_by_hand_with_ties_and_short_lists() -> None:
    counts = _label_counts({5: 4, 2: 3, 9: 3, 1: 2})  # 12 pairs
    assert top_pair_share(counts, 1) == 4 / 12
    # Articles 2 and 9 tie at 3 pairs: the smaller article_idx comes first, and the
    # share is the same either way (4 + 3) / 12.
    assert top_pair_share(counts, 2) == 7 / 12
    assert top_pair_share(counts, 3) == 10 / 12
    assert top_pair_share(counts, 100) == 1.0  # fewer articles than N: all of them
    assert top_pair_share(counts.reverse(), 2) == 7 / 12  # the row order does not matter
    with pytest.raises(ValueError, match="top_n must be > 0"):
        top_pair_share(counts, 0)


def test_concentration_by_hand() -> None:
    # 10 label pairs on 5 articles. The window of "new" is [100 - 28, 100) = days 72 ... 99.
    counts = _label_counts({1: 4, 2: 3, 3: 1, 4: 1, 5: 1})
    first_sale = _first_sale(
        {
            1: 0,  # old
            2: 72,  # the first day of the window: new
            3: 71,  # one day too early: old
            4: 99,  # the day before the anchor: new
            # article 5 has no sale before the anchor: first sold in the label week
            8: 80,  # sold before, but no label this week: not counted anywhere
        }
    )
    stats = concentration(counts, first_sale, ANCHOR_DAY, (1, 2), 28)
    assert (stats.n_label_pairs, stats.n_label_articles) == (10, 5)
    assert stats.top_pair_shares == {1: 4 / 10, 2: 7 / 10}
    assert (stats.n_new_articles, stats.new_pair_share) == (2, 4 / 10)  # articles 2 and 4
    assert (stats.n_unsold_before_articles, stats.unsold_before_pair_share) == (1, 1 / 10)
    row = concentration_row("V1", "V", "2020-07-29", stats, 28)
    assert list(row) == list(concentration_columns((1, 2), 28))
    assert row["top2_pair_share"] == "0.7"
    assert row["n_articles_first_sale_within_28d"] == "2"
    assert row["pair_share_first_sale_in_label_week"] == "0.1"


def test_a_first_sale_on_or_after_the_anchor_is_refused() -> None:
    # Such a day could only come from label-week data: the function must not accept it.
    counts = _label_counts({1: 1})
    for leaked_day in (ANCHOR_DAY, ANCHOR_DAY + 3):
        with pytest.raises(ValueError, match="not before the anchor"):
            concentration(counts, _first_sale({1: leaked_day}), ANCHOR_DAY, (1,), 28)


def test_concentration_needs_labels() -> None:
    with pytest.raises(ValueError, match="no label pairs"):
        concentration(_label_counts({}), _first_sale({}), ANCHOR_DAY, (1,), 28)
