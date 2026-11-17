"""A customer's most recent purchases before a day: ONE definition for the whole project.

Used by R6 Two-Tower (training samples, weekly fine-tune, inference) and by
the feature code (img_sim_hist / txt_sim_hist, proposal §8.3). features/ may
not import recall/, so this code lives in the lower data/ layer and both
import it. That is what guarantees "the user's last 20 purchases" means the
same thing for the model and for the features.

The definition (proposal §6.1):
    - one purchase per (customer_idx, t_dat_day, article_idx): several purchase
      lines of the same article on the same day count once (daily_purchases);
      the same article on different days counts each time;
    - only purchases STRICTLY before the query day (the day itself is excluded);
      no lower time limit;
    - order: most recent day first; several articles on the same day in
      ascending article_idx, so the result never depends on file order;
    - the last `history_length` of them; shorter histories are padded at the END
      with a padding index that is not a real article, with history_mask False.

Dates are filtered only through hmrec.data.asof.
"""

from datetime import date

import numpy as np
import polars as pl

from hmrec.data.asof import TRANSACTION_DAY, history_window
from hmrec.data.day_index import date_to_day_index
from hmrec.data.id_maps import ARTICLE_IDX, CUSTOMER_IDX

# Key for sorting/searching (customer, day) pairs as one int64:
# key = customer_idx * DAY_KEY_SPAN + day. t_dat_day is int16 (0..32767), so
# 2**16 is larger than any day and keys of different customers never overlap.
DAY_KEY_SPAN = 2**16


# ---------------------------------------------------------------- reading purchase lines (IO)


def load_all_lines_before(
    transactions: pl.LazyFrame, end_anchor: date, day_zero: date
) -> pl.DataFrame:
    """Every purchase line before end_anchor: t_dat_day in [0, end_anchor) (the history source).

    The window starts at day 0 (day_zero, the first day of the data), so a
    history may reach back as far as the data goes (proposal §5 R6: no time limit).
    """
    days_since_day_zero = date_to_day_index(end_anchor, day_zero)
    if days_since_day_zero <= 0:
        raise ValueError(f"end_anchor {end_anchor} must be after day_zero {day_zero}")
    return history_window(
        transactions,
        end_anchor,
        day_zero,
        days_since_day_zero,
        columns=[CUSTOMER_IDX, ARTICLE_IDX],
    )


def load_customer_lines_before(
    transactions: pl.LazyFrame, customer_idx: np.ndarray, end_anchor: date, day_zero: date
) -> pl.DataFrame:
    """Every purchase line before end_anchor of the given customers only (inference histories).

    Filtering by customer first (not by date, so asof still does every date
    filter) keeps the table small: one fold's ~82k query customers instead of
    all 1.37M customers.
    """
    customers = pl.Series(CUSTOMER_IDX, customer_idx, dtype=pl.Int32).implode()
    of_customers = transactions.filter(pl.col(CUSTOMER_IDX).is_in(customers))
    return load_all_lines_before(of_customers, end_anchor, day_zero)


# ---------------------------------------------------------------- one purchase per day


def daily_purchases(lines: pl.DataFrame) -> pl.DataFrame:
    """Collapse purchase lines to one row per (customer_idx, t_dat_day, article_idx).

    Several lines of the same article by the same customer on the same day
    (e.g. two pieces of one item) become one row: for the Two-Tower model they
    are one positive, not two (proposal §6.1).

    Returns:
        Columns customer_idx (int32), t_dat_day (int16), article_idx (int32);
        unsorted.
    """
    return lines.select(CUSTOMER_IDX, TRANSACTION_DAY, ARTICLE_IDX).unique()


# ---------------------------------------------------------------- histories


def _event_keys(customer_idx: np.ndarray, day: np.ndarray) -> np.ndarray:
    """int64 key that sorts by customer, then day (see DAY_KEY_SPAN)."""
    if day.size and (day.min() < 0 or day.max() >= DAY_KEY_SPAN):
        raise ValueError(f"t_dat_day must be in [0, {DAY_KEY_SPAN}); got {day.min()}..{day.max()}")
    return customer_idx.astype(np.int64) * DAY_KEY_SPAN + day.astype(np.int64)


def _sorted_events(events: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Events as numpy arrays (keys int64, article_idx int32, day int32).

    Sorted by customer, day ascending, and article_idx DESCENDING within a day.
    Reading this order BACKWARDS from a position gives: most recent day first,
    and ascending article_idx within a day, which is the history order we want.
    """
    ordered = events.sort(
        [CUSTOMER_IDX, TRANSACTION_DAY, ARTICLE_IDX], descending=[False, False, True]
    )
    customer = ordered[CUSTOMER_IDX].to_numpy()
    day = ordered[TRANSACTION_DAY].to_numpy().astype(np.int32)
    article = ordered[ARTICLE_IDX].to_numpy().astype(np.int32)
    return _event_keys(customer, day), article, day


def _check_history_before_query(
    history_day: np.ndarray, history_mask: np.ndarray, query_day: np.ndarray
) -> None:
    """Fail if a filled history slot is not strictly before its query day (a leak)."""
    leaked = history_mask & (history_day >= query_day[:, None])
    if leaked.any():
        row = int(np.argwhere(leaked)[0, 0])
        raise ValueError(
            f"history of query row {row} (day {query_day[row]}) contains a purchase on day "
            f"{history_day[row][leaked[row]].max()}, which is not before the query day"
        )


def check_below_padding(name: str, article_idx: np.ndarray, padding_idx: int) -> None:
    """Fail unless every article_idx is in [0, padding_idx): only padding may equal padding_idx."""
    if article_idx.size and (article_idx.min() < 0 or article_idx.max() >= padding_idx):
        raise ValueError(
            f"{name}: article_idx must be in [0, {padding_idx}) (padding_idx = n_articles), "
            f"got {article_idx.min()}..{article_idx.max()}"
        )


def build_histories(
    events: pl.DataFrame,
    query_customer: np.ndarray,
    query_day: np.ndarray,
    history_length: int,
    padding_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each query (customer, day): the last `history_length` purchases before that day.

    Args:
        events: Daily purchases (daily_purchases) to take histories from; must
            contain every purchase before the latest query day.
        query_customer: int32 [Q], customer_idx of each query.
        query_day: int16/int32 [Q], t_dat_day of each query.
        history_length: L, e.g. 20.
        padding_idx: Value of empty slots: n_articles, which is not a real article.

    Returns:
        (history int32 [Q, L], history_mask bool [Q, L]). Slot 0 is the most
        recent purchase; empty slots hold padding_idx with mask False.

    Raises:
        ValueError: If an event's article_idx is not below padding_idx (it
            would be confused with padding), or a history leaks (see
            _check_history_before_query).
    """
    keys, articles, days = _sorted_events(events)
    check_below_padding("events", articles, padding_idx)
    query_day = query_day.astype(np.int32)
    # Events of this customer before the query day are exactly keys[start:end]:
    #   start = first event of this customer (key >= customer * SPAN + 0),
    #   end   = first event of this customer ON or AFTER the query day.
    start = np.searchsorted(keys, _event_keys(query_customer, np.zeros_like(query_day)), "left")
    end = np.searchsorted(keys, _event_keys(query_customer, query_day), "left")
    n_queries = query_customer.shape[0]
    history = np.full((n_queries, history_length), padding_idx, dtype=np.int32)
    history_day = np.full((n_queries, history_length), -1, dtype=np.int32)
    history_mask = np.zeros((n_queries, history_length), dtype=np.bool_)
    for slot in range(history_length):
        # Slot k holds the (k+1)-th most recent event: position end - 1 - k.
        position = end - 1 - slot
        filled = position >= start
        safe_position = np.where(filled, position, 0)  # any valid index; masked anyway
        history[:, slot] = np.where(filled, articles[safe_position], padding_idx)
        history_day[:, slot] = np.where(filled, days[safe_position], -1)
        history_mask[:, slot] = filled
    _check_history_before_query(history_day, history_mask, query_day)
    return history, history_mask


def recent_purchase_histories(
    transactions: pl.LazyFrame,
    customer_idx: np.ndarray,
    anchor: date,
    day_zero: date,
    history_length: int,
    padding_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The last `history_length` purchases of each customer before `anchor`.

    This is the history R6 predicts with and the history the img_sim_hist /
    txt_sim_hist features average over: both call this function.

    Args:
        transactions: Processed transactions (lazy); only t_dat < anchor is read.
        customer_idx: int32 [Q], distinct customers.
        anchor: First day of the label week; the anchor day is NOT included.
        day_zero: Day 0 of t_dat_day.
        history_length: L, e.g. 20.
        padding_idx: Value of empty slots: n_articles, which is not a real article.

    Returns:
        (history int32 [Q, L], history_mask bool [Q, L]); see build_histories.
    """
    lines = load_customer_lines_before(transactions, customer_idx, anchor, day_zero)
    anchor_day = date_to_day_index(anchor, day_zero)
    anchor_days = np.full(customer_idx.shape, anchor_day, dtype=np.int32)
    return build_histories(
        daily_purchases(lines), customer_idx, anchor_days, history_length, padding_idx
    )
