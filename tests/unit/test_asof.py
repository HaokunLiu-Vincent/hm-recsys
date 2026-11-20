"""Tests for hmrec.data.asof. Expected rows are listed in tests/fixtures/asof/README.md."""

from pathlib import Path

import polars as pl
import pytest

from conftest import D1, D1_ANCHOR, D2, FIXTURE_DAY_ZERO, make_transactions
from hmrec.data.asof import (
    _assert_inside_window,
    history_window,
    label_week_transactions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# The only files allowed to mention label_week_transactions (AGENTS.md §3.1).
LABEL_WEEK_ALLOWED_FILES = {
    REPO_ROOT / "src" / "hmrec" / "data" / "asof.py",
    REPO_ROOT / "src" / "hmrec" / "data" / "queries.py",
}

# Rows on or after the D1 anchor (day 42), including the last day of the real data
# (733). None of them may change a history computed for anchor day 42.
POISON_ROWS = [(9, 1, 42), (9, 1, 42), (9, 3, 43), (8, 5, 60), (7, 0, 733)]


def _days(frame: pl.DataFrame) -> list[int]:
    return sorted(frame["t_dat_day"].to_list())


def test_history_window_7_days_is_days_35_to_41(fixture_transactions: pl.DataFrame) -> None:
    window = history_window(fixture_transactions.lazy(), D1_ANCHOR, FIXTURE_DAY_ZERO, 7)
    # Rows 3, 6, 9, 10, 11 of the fixture: days 41, 41, 35, 40, 41.
    assert _days(window) == [35, 40, 41, 41, 41]


def test_history_window_includes_start_and_excludes_anchor(
    fixture_transactions: pl.DataFrame,
) -> None:
    window = history_window(fixture_transactions.lazy(), D1_ANCHOR, FIXTURE_DAY_ZERO, 28)
    days = _days(window)
    # 28 days before day 42 -> [14, 42): day 14 (row 1) is in, day 13 (row 4) and 42 are out.
    assert days[0] == 14
    assert days[-1] == 41
    assert 13 not in days
    assert 42 not in days
    assert window.height == 10


def test_history_window_keeps_duplicate_lines(fixture_transactions: pl.DataFrame) -> None:
    window = history_window(fixture_transactions.lazy(), D1_ANCHOR, FIXTURE_DAY_ZERO, 28)
    day_20_rows = window.filter(pl.col("t_dat_day") == 20)
    assert day_20_rows.height == 2  # rows 7 and 8 are identical; both stay


def test_history_window_ignores_poison_on_and_after_anchor(
    fixture_transactions: pl.DataFrame,
) -> None:
    clean = history_window(fixture_transactions.lazy(), D1_ANCHOR, FIXTURE_DAY_ZERO, 28)
    poisoned_input = pl.concat([fixture_transactions, make_transactions(POISON_ROWS)])
    poisoned = history_window(poisoned_input.lazy(), D1_ANCHOR, FIXTURE_DAY_ZERO, 28)
    assert poisoned.equals(clean)


def test_history_window_returns_requested_columns_plus_day(
    fixture_transactions: pl.DataFrame,
) -> None:
    window = history_window(
        fixture_transactions.lazy(), D1_ANCHOR, FIXTURE_DAY_ZERO, 7, columns=["article_idx"]
    )
    assert window.columns == ["article_idx", "t_dat_day"]


def test_history_window_rejects_non_positive_lookback(fixture_transactions: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="lookback_days"):
        history_window(fixture_transactions.lazy(), D1_ANCHOR, FIXTURE_DAY_ZERO, 0)


def test_history_window_rejects_missing_or_wrong_day_column(
    fixture_transactions: pl.DataFrame,
) -> None:
    without_day = fixture_transactions.drop("t_dat_day").lazy()
    with pytest.raises(ValueError, match="no 't_dat_day' column"):
        history_window(without_day, D1_ANCHOR, FIXTURE_DAY_ZERO, 7)
    as_int32 = fixture_transactions.with_columns(pl.col("t_dat_day").cast(pl.Int32)).lazy()
    with pytest.raises(ValueError, match="dtype"):
        history_window(as_int32, D1_ANCHOR, FIXTURE_DAY_ZERO, 7)


def test_window_check_raises_on_rows_outside_window() -> None:
    # The filter makes this impossible today; the check guards against future edits.
    leaked = make_transactions([(0, 0, 41), (0, 0, 42)])
    with pytest.raises(ValueError, match="not before 42"):
        _assert_inside_window(leaked, start_day=35, end_day_exclusive=42)
    too_early = make_transactions([(0, 0, 34)])
    with pytest.raises(ValueError, match="before 35"):
        _assert_inside_window(too_early, start_day=35, end_day_exclusive=42)


def test_label_week_is_anchor_to_label_end(fixture_transactions: pl.DataFrame) -> None:
    d1_week = label_week_transactions(fixture_transactions.lazy(), D1, FIXTURE_DAY_ZERO)
    # Rows 12-15: days 42, 45, 48, 48. Day 49 (row 16) belongs to D2.
    assert _days(d1_week) == [42, 45, 48, 48]
    d2_week = label_week_transactions(fixture_transactions.lazy(), D2, FIXTURE_DAY_ZERO)
    assert _days(d2_week) == [49, 55]


def test_history_and_label_week_do_not_overlap(fixture_transactions: pl.DataFrame) -> None:
    # Use a lookback that covers the whole fixture, so every pre-anchor row is history.
    history = history_window(fixture_transactions.lazy(), D1.anchor, FIXTURE_DAY_ZERO, 1000)
    label_week = label_week_transactions(fixture_transactions.lazy(), D1, FIXTURE_DAY_ZERO)
    history_days = set(history["t_dat_day"].to_list())
    label_days = set(label_week["t_dat_day"].to_list())
    assert history_days.isdisjoint(label_days)
    # The anchor day (42) is only in the label week.
    assert 42 in label_days
    assert 42 not in history_days
    assert max(history_days) == 41


def test_label_week_transactions_used_only_by_asof_and_queries() -> None:
    # label_week_transactions reads the answers of a fold. Recall, feature and ranker
    # code must never call it (AGENTS.md §3.1). This scan is the real guard; the
    # docstring alone is only a reminder.
    offenders = []
    for folder in (REPO_ROOT / "src" / "hmrec", REPO_ROOT / "scripts"):
        for path in sorted(folder.rglob("*.py")):
            if path in LABEL_WEEK_ALLOWED_FILES:
                continue
            if "label_week_transactions" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []
