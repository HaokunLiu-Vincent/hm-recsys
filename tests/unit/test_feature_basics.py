"""Tests for features.base, features.categories and features.stats (fold: feature_example.py)."""

import dataclasses
import json
from pathlib import Path

import polars as pl
import pytest
from feature_example import ARTICLES, CONFIG, CUSTOMERS, LINES, POISON, make_fold, make_history

from hmrec.features.base import as_feature, history_rows, lines_before, window_lines
from hmrec.features.categories import (
    CategoryCodes,
    build_category_codes,
    encode_categories,
    save_or_check_category_codes,
)
from hmrec.features.stats import STATS_COLUMNS, FeatureStats
from hmrec.run_records import RunStamp

# ---------------------------------------------------------------- windows (base)


def _days(lines: pl.DataFrame) -> list[int]:
    return sorted(lines["t_dat_day"].to_list())


def test_windows_end_before_the_anchor() -> None:
    fold = make_fold(LINES + POISON)  # with poison: it must never show up
    # Anchor day 400. 1w = [393, 400), prev_1w = [386, 393), 4w = [372, 400), 52w = [36, 400).
    assert _days(window_lines(fold, 7, ["article_idx"])) == [395, 399, 399]
    assert _days(window_lines(fold, 7, ["article_idx"], days_before_anchor=7)) == [390]
    assert _days(window_lines(fold, 28, ["article_idx"])) == [380, 390, 395, 399, 399]
    assert _days(window_lines(fold, 364, ["article_idx"])) == [380, 390, 395, 399, 399]
    assert _days(lines_before(fold, ["article_idx"])) == [30, 380, 390, 395, 399, 399]


def test_age_days_counts_from_the_folds_anchor() -> None:
    fold = make_fold()
    this_week = window_lines(fold, 7, ["article_idx"]).sort("t_dat_day")
    assert this_week["age_days"].to_list() == [5, 1, 1]  # day 395 -> 5, day 399 -> 1
    # A window that ends 7 days before the anchor still measures age from the anchor.
    previous = window_lines(fold, 7, ["article_idx"], days_before_anchor=7)
    assert previous["age_days"].to_list() == [10]  # day 390
    assert str(this_week.schema["age_days"]) == "Int32"


def test_customer_filter_and_bad_arguments() -> None:
    fold = make_fold()
    only_one = lines_before(fold, ["customer_idx"], customers=fold.query_customers[1:2])
    assert only_one["customer_idx"].unique().to_list() == [1] and only_one.height == 3
    assert fold.query_customers.tolist() == [0, 1, 2] and fold.anchor_day == 400
    with pytest.raises(ValueError, match="days_before_anchor must be >= 0"):
        window_lines(fold, 7, ["article_idx"], days_before_anchor=-1)


def test_as_feature_missing_count_is_zero_and_other_missing_is_nan() -> None:
    table = pl.DataFrame({"n": [3, None], "mean": [0.5, None]})
    result = table.select(as_feature("n", count=True), as_feature("mean"))
    assert result["n"].to_list() == [3.0, 0.0]
    assert result["mean"][0] == 0.5 and result["mean"].is_nan().to_list() == [False, True]
    assert [str(dtype) for dtype in result.dtypes] == ["Float32", "Float32"]


def test_history_rows_are_one_row_per_filled_slot() -> None:
    # Customer 0: [0, 0, 1] (article 0 on two days fills two slots); customer 1: [0, 1, 2];
    # customer 2 has no purchase and so no row. Padded slots are dropped.
    rows = history_rows(make_history())
    assert rows.rows() == [(0, 0), (0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    assert [str(dtype) for dtype in rows.dtypes] == ["Int32", "Int32"]
    assert history_rows(make_history(LINES + POISON)).equals(rows)  # nothing on / after the anchor


# ---------------------------------------------------------------- category codes


def test_codes_are_the_sorted_distinct_values_from_one() -> None:
    codes = build_category_codes(ARTICLES, CUSTOMERS)
    assert codes.values["product_type_no"] == (5, 7)
    assert codes.values["colour_group_code"] == (8, 9)
    assert codes.values["product_group_name"] == ("A", "B")
    # null is not a value: it gets code 0.
    assert codes.values["club_member_status"] == ("ACTIVE", "PRE-CREATE")
    assert codes.values["fashion_news_frequency"] == ("Monthly", "NONE", "Regularly")


def test_encode_keeps_rows_and_maps_missing_to_zero() -> None:
    codes = build_category_codes(ARTICLES, CUSTOMERS)
    encoded = encode_categories(CUSTOMERS, ("club_member_status", "fashion_news_frequency"), codes)
    assert encoded["customer_idx"].to_list() == [0, 1, 2]  # order unchanged
    assert encoded["club_member_status"].to_list() == [1, 0, 2]  # ACTIVE, missing, PRE-CREATE
    assert encoded["fashion_news_frequency"].to_list() == [2, 3, 1]
    assert str(encoded.schema["club_member_status"]) == "Int16"
    assert encoded["age"].to_list() == CUSTOMERS["age"].to_list()  # other columns untouched


def test_a_value_outside_the_code_table_is_an_error() -> None:
    codes = build_category_codes(ARTICLES, CUSTOMERS)
    other = ARTICLES.with_columns(pl.lit(99, dtype=pl.Int16).alias("product_type_no"))
    with pytest.raises(ValueError, match=r"product_type_no: values \[99\] are not in the category"):
        encode_categories(other, ("product_type_no",), codes)


def test_code_table_is_saved_once_and_must_stay_the_same(tmp_path: Path) -> None:
    codes = build_category_codes(ARTICLES, CUSTOMERS)
    path = tmp_path / "features" / "category_codes.json"
    save_or_check_category_codes(path, codes)
    written = path.read_text()
    assert json.loads(written)["codes"]["section_no"] == [3, 4]
    save_or_check_category_codes(path, codes)  # the same table again: accepted, not rewritten
    assert path.read_text() == written
    changed = CategoryCodes({**codes.values, "section_no": (3, 4, 5)})
    with pytest.raises(ValueError, match="holds another category code table"):
        save_or_check_category_codes(path, changed)


# ---------------------------------------------------------------- stats


def test_stats_accumulate_over_chunks() -> None:
    numeric = CONFIG.features[0]  # pop_rank, float32
    category = next(feature for feature in CONFIG.features if feature.categorical)
    stats = FeatureStats((numeric, category))
    nan = float("nan")

    def chunk(ranks: list[float], codes: list[int]) -> pl.DataFrame:
        return pl.DataFrame(
            {numeric.name: ranks, category.name: codes},
            schema={numeric.name: pl.Float32, category.name: pl.Int16},
        )

    stats.update(chunk([1.0, nan, 4.0], [0, 2, 5]))
    stats.update(chunk([nan, 7.0], [3, 0]))
    rows = {row["feature"]: row for row in stats.rows("F1", RunStamp("abc", False))}
    assert all(tuple(row) == STATS_COLUMNS for row in rows.values())
    first = rows[numeric.name]
    # 5 rows, 2 NaN; the other three are 1, 4, 7 -> min 1, max 7, mean 4.
    assert (first["n_rows"], first["n_missing"], first["missing_rate"]) == ("5", "2", "0.4")
    assert (first["min"], first["max"], first["mean"]) == ("1.0", "7.0", "4.0")
    second = rows[category.name]
    # Category: missing = code 0 (2 of 5); min / max over all codes; no mean.
    assert (second["n_missing"], second["min"], second["max"], second["mean"]) == (
        "2",
        "0.0",
        "5.0",
        "",
    )
    assert (second["categorical"], second["dtype"], second["group"]) == ("True", "int16", "C")


def test_stats_of_an_all_missing_feature_have_no_range() -> None:
    numeric = CONFIG.features[0]
    stats = FeatureStats((numeric,))
    stats.update(
        pl.DataFrame({numeric.name: [float("nan")] * 2}, schema={numeric.name: pl.Float32})
    )
    row = stats.rows("T1", RunStamp("abc", True))[0]
    assert (row["missing_rate"], row["min"], row["max"], row["mean"]) == ("1.0", "", "", "")
    assert (row["fold"], row["git_dirty"]) == ("T1", "True")
    assert dataclasses.is_dataclass(RunStamp)
