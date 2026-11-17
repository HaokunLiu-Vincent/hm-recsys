"""Group E: customer x article features, 21 columns (proposal §8.6).

Purchase LINES are counted. "History" is every purchase of the customer with
t_dat < a; "52w" is [a - 364, a). "Same X" = an article whose field X has the
candidate's value.

    ui_cnt_item            the customer's history lines of this article; 0 if none
    ui_days_since_item     a - (the last day they bought this article); NaN if never
    ui_cnt_pcode           history lines of ANY article with the candidate's
                           product_code (the candidate itself included); 0 if none
    ui_days_since_pcode    a - (the last day they bought such an article); NaN if never
    ui_share_ptype_52w     (the customer's 52w lines of the same product_type_no) /
                           (all their 52w lines); NaN if they have no line in 52w,
                           else 0 if none matches
    ui_share_garment_52w, ui_share_index_52w, ui_share_colour_52w,
    ui_share_section_52w, ui_share_pgroup_52w
                           the same for garment_group_no, index_group_no,
                           colour_group_code, section_no, product_group_name
    ui_price_ratio         item_price_mean_4w / user_price_mean_52w; NaN if either is missing
    ui_age_gap             |age - item_buyer_age_mean_4w|; NaN if either is missing
    ui_decay_ptype         sum of exp(-age_days / 30) over the customer's HISTORY lines
                           of the same product_type_no; 0 if none (tau fixed, not tuned)
    ui_decay_garment, ui_decay_section, ui_decay_colour
                           the same for garment_group_no, section_no, colour_group_code
    ui_days_since_ptype    a - (the last day they bought the same product_type_no);
                           NaN if never
    ui_days_since_section  the same for section_no
    ui_lift_ptype_52w      ui_share_ptype_52w / (the share of that product_type_no among
                           ALL customers' 52w lines); NaN if the customer has no 52w
                           line or that site-wide share is 0
    ui_channel_gap         |user_online_share_52w - item_online_share_4w|; NaN if either
                           is missing
    ui_attr_match_cnt      of the six category fields of §8.4: for how many does the
                           candidate's value occur among the customer's last 20
                           purchases (0 ... 6); NaN if they have no purchase

The small tables are computed once per fold for the query customers only (the
site-wide product_type share is one row per type), then joined to each chunk.
"""

from dataclasses import dataclass

import polars as pl

from hmrec.config import FeaturesConfig
from hmrec.features.base import (
    AGE,
    AGE_DAYS,
    ARTICLE_CATEGORY_COLUMNS,
    ARTICLE_IDX,
    CUSTOMER_IDX,
    NAN,
    PRODUCT_CODE,
    FoldData,
    lines_before,
    window_lines,
)

# Feature -> the article category field it compares.
SHARE_FEATURES: dict[str, str] = {
    "ui_share_ptype_52w": "product_type_no",
    "ui_share_garment_52w": "garment_group_no",
    "ui_share_index_52w": "index_group_no",
    "ui_share_colour_52w": "colour_group_code",
    "ui_share_section_52w": "section_no",
    "ui_share_pgroup_52w": "product_group_name",
}
DECAY_FEATURES: dict[str, str] = {
    "ui_decay_ptype": "product_type_no",
    "ui_decay_garment": "garment_group_no",
    "ui_decay_section": "section_no",
    "ui_decay_colour": "colour_group_code",
}
DAYS_SINCE_FEATURES: dict[str, str] = {
    "ui_days_since_ptype": "product_type_no",
    "ui_days_since_section": "section_no",
}
LIFT_FEATURE = "ui_lift_ptype_52w"
LIFT_COLUMN = "product_type_no"
# Column order of proposal §8.6.
USER_ITEM_FEATURES = (
    "ui_cnt_item",
    "ui_days_since_item",
    "ui_cnt_pcode",
    "ui_days_since_pcode",
    "ui_share_ptype_52w",
    "ui_share_garment_52w",
    "ui_share_index_52w",
    "ui_share_colour_52w",
    "ui_price_ratio",
    "ui_age_gap",
    *DECAY_FEATURES,
    *DAYS_SINCE_FEATURES,
    "ui_share_section_52w",
    "ui_share_pgroup_52w",
    LIFT_FEATURE,
    "ui_channel_gap",
    "ui_attr_match_cnt",
)
YEAR_LINES = "__user_lines_52w"
GLOBAL_SHARE = "__global_share"
HAS_HISTORY = "__has_history"


def _raw(name: str) -> str:
    """Temporary column name of an article field joined onto the candidates."""
    return f"__{name}"


def _decay(column: str) -> str:
    """Temporary column: the decayed line count of a category field."""
    return f"__decay_{column}"


def _days(column: str) -> str:
    """Temporary column: days since the last purchase of a category field."""
    return f"__days_{column}"


def _seen(column: str) -> str:
    """Temporary column: 1 if the value occurs in the customer's last purchases."""
    return f"__seen_{column}"


@dataclass(frozen=True)
class UserItemTables:
    """The fold's aggregates of the query customers' purchases.

    Attributes:
        by_article: One row per (customer_idx, article_idx) ever bought:
            ui_cnt_item, ui_days_since_item.
        by_product_code: One row per (customer_idx, product_code) ever bought:
            ui_cnt_pcode, ui_days_since_pcode.
        history_by_category: Category field -> one row per (customer_idx,
            value) ever bought: the decayed line count and the days since the
            last such purchase (temporary column names).
        year_by_category: Share feature -> one row per (customer_idx, value)
            bought in 52w, with the number of lines in a column named after the feature.
        year_lines: One row per customer with a line in 52w: the number of lines.
        global_type_share: One row per product_type_no sold in 52w by anyone:
            its share of all 52w lines.
        recent_values: Category field -> one row per (customer_idx, value) that
            occurs among the customer's last purchases.
        with_history: One row per customer that has at least one purchase.
        article_fields: One row per article: article_idx, product_code and the
            six category fields under temporary names.
    """

    by_article: pl.DataFrame
    by_product_code: pl.DataFrame
    history_by_category: dict[str, pl.DataFrame]
    year_by_category: dict[str, pl.DataFrame]
    year_lines: pl.DataFrame
    global_type_share: pl.DataFrame
    recent_values: dict[str, pl.DataFrame]
    with_history: pl.DataFrame
    article_fields: pl.DataFrame


def article_field_table(articles: pl.DataFrame) -> pl.DataFrame:
    """article_idx with product_code and the six category fields under temporary names."""
    fields = [pl.col(column).alias(_raw(column)) for column in ARTICLE_CATEGORY_COLUMNS]
    return articles.select(ARTICLE_IDX, pl.col(PRODUCT_CODE).alias(_raw(PRODUCT_CODE)), *fields)


def history_aggregates(
    history: pl.DataFrame, decay_tau_days: float
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, pl.DataFrame]]:
    """Aggregates over ALL history lines of the query customers.

    Args:
        history: One row per purchase line with customer_idx, article_idx,
            age_days and the temporary article fields.
        decay_tau_days: Time constant of ui_decay_* (30).

    Returns:
        (by_article, by_product_code, history_by_category); see UserItemTables.
    """
    count_and_last = [pl.len().alias("cnt"), pl.col(AGE_DAYS).min().alias("days")]
    by_article = history.group_by(CUSTOMER_IDX, ARTICLE_IDX).agg(count_and_last)
    by_article = by_article.rename({"cnt": "ui_cnt_item", "days": "ui_days_since_item"})
    by_code = history.group_by(CUSTOMER_IDX, _raw(PRODUCT_CODE)).agg(count_and_last)
    by_code = by_code.rename({"cnt": "ui_cnt_pcode", "days": "ui_days_since_pcode"})
    # See proposal §8.6: weight of a line = exp(-age_days / tau), summed in float64.
    weight = (-pl.col(AGE_DAYS).cast(pl.Float64) / decay_tau_days).exp()
    by_category = {
        column: history.group_by(CUSTOMER_IDX, _raw(column)).agg(
            weight.sum().alias(_decay(column)), pl.col(AGE_DAYS).min().alias(_days(column))
        )
        for column in dict.fromkeys(DECAY_FEATURES.values())
    }
    return by_article, by_code, by_category


def global_type_shares(
    data: FoldData, config: FeaturesConfig, article_fields: pl.DataFrame
) -> pl.DataFrame:
    """Share of each product_type_no among ALL customers' purchase lines of [a - 364, a).

    Returns:
        One row per product_type_no sold in the window: the type (temporary
        name) and its share (float64, > 0). A type that is absent has share 0.
    """
    lines = window_lines(data, config.year_days, [ARTICLE_IDX])
    typed = lines.join(
        article_fields.select(ARTICLE_IDX, _raw(LIFT_COLUMN)), on=ARTICLE_IDX, how="left"
    )
    counts = typed.group_by(_raw(LIFT_COLUMN)).agg(pl.len().alias("n"))
    total = pl.col("n").sum().cast(pl.Float64)
    return counts.select(
        _raw(LIFT_COLUMN), (pl.col("n").cast(pl.Float64) / total).alias(GLOBAL_SHARE)
    )


def recent_value_tables(
    recent_purchases: pl.DataFrame, article_fields: pl.DataFrame
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """The category values that occur among each customer's last purchases.

    Args:
        recent_purchases: One row per filled slot of the customers' histories
            (hmrec.data.histories): customer_idx, article_idx.
        article_fields: See article_field_table.

    Returns:
        (field -> distinct (customer_idx, value) with a column of 1s, the
        customers that have at least one purchase).
    """
    with_fields = recent_purchases.join(article_fields, on=ARTICLE_IDX, how="left")
    values = {
        column: with_fields.select(CUSTOMER_IDX, _raw(column))
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int32).alias(_seen(column)))
        for column in ARTICLE_CATEGORY_COLUMNS
    }
    with_history = (
        recent_purchases.select(CUSTOMER_IDX).unique().with_columns(pl.lit(True).alias(HAS_HISTORY))
    )
    return values, with_history


def build_user_item_tables(
    data: FoldData, config: FeaturesConfig, recent_purchases: pl.DataFrame
) -> UserItemTables:
    """Aggregate the query customers' purchases before the anchor (see UserItemTables).

    Args:
        data: The fold.
        config: configs/features.yaml (52w window, decay tau).
        recent_purchases: The query customers' last purchases before the
            anchor, one row per history slot: customer_idx, article_idx.
    """
    customers = data.query_customers
    article_fields = article_field_table(data.articles)
    history = lines_before(data, [CUSTOMER_IDX, ARTICLE_IDX], customers=customers)
    history = history.join(article_fields, on=ARTICLE_IDX, how="left")
    by_article, by_product_code, history_by_category = history_aggregates(
        history, config.decay_tau_days
    )
    year = window_lines(data, config.year_days, [CUSTOMER_IDX, ARTICLE_IDX], customers=customers)
    year = year.join(article_fields, on=ARTICLE_IDX, how="left")
    year_by_category = {
        feature: year.group_by(CUSTOMER_IDX, _raw(column)).agg(pl.len().alias(feature))
        for feature, column in SHARE_FEATURES.items()
    }
    recent_values, with_history = recent_value_tables(recent_purchases, article_fields)
    return UserItemTables(
        by_article=by_article,
        by_product_code=by_product_code,
        history_by_category=history_by_category,
        year_by_category=year_by_category,
        year_lines=year.group_by(CUSTOMER_IDX).agg(pl.len().alias(YEAR_LINES)),
        global_type_share=global_type_shares(data, config, article_fields),
        recent_values=recent_values,
        with_history=with_history,
        article_fields=article_fields,
    )


# ---------------------------------------------------------------- joining onto a chunk


def _left_join(frame: pl.DataFrame, table: pl.DataFrame, on: list[str]) -> pl.DataFrame:
    """Left join that keeps the rows of `frame` in their order (one table row per key)."""
    return frame.join(table, on=on, how="left", maintain_order="left")


def _join_tables(chunk: pl.DataFrame, tables: UserItemTables) -> pl.DataFrame:
    """Join every aggregate of `tables` onto the candidates (temporary columns included)."""
    joined = _left_join(chunk, tables.article_fields, [ARTICLE_IDX])
    joined = _left_join(joined, tables.by_article, [CUSTOMER_IDX, ARTICLE_IDX])
    joined = _left_join(joined, tables.by_product_code, [CUSTOMER_IDX, _raw(PRODUCT_CODE)])
    joined = _left_join(joined, tables.year_lines, [CUSTOMER_IDX])
    joined = _left_join(joined, tables.with_history, [CUSTOMER_IDX])
    joined = _left_join(joined, tables.global_type_share, [_raw(LIFT_COLUMN)])
    for feature, column in SHARE_FEATURES.items():
        joined = _left_join(joined, tables.year_by_category[feature], [CUSTOMER_IDX, _raw(column)])
    for column, table in tables.history_by_category.items():
        joined = _left_join(joined, table, [CUSTOMER_IDX, _raw(column)])
    for column, table in tables.recent_values.items():
        joined = _left_join(joined, table, [CUSTOMER_IDX, _raw(column)])
    return joined


def _share_expr(feature: str) -> pl.Expr:
    """Share = lines in the category / all 52w lines; NaN without 52w lines, 0 if none matches."""
    total = pl.col(YEAR_LINES)
    share = pl.col(feature).fill_null(0).cast(pl.Float64) / total.cast(pl.Float64)
    return (
        pl.when(total.is_null() | (total == 0))
        .then(pl.lit(NAN, dtype=pl.Float64))
        .otherwise(share)
        .cast(pl.Float32)
        .alias(feature)
    )


def _price_ratio_expr() -> pl.Expr:
    """item_price_mean_4w / user_price_mean_52w; NaN if a side is missing or the divisor is 0."""
    item = pl.col("item_price_mean_4w").cast(pl.Float64)
    user = pl.col("user_price_mean_52w").cast(pl.Float64)
    # NaN on either side already gives NaN; only a zero divisor needs its own rule.
    return (
        pl.when(user == 0)
        .then(pl.lit(NAN, dtype=pl.Float64))
        .otherwise(item / user)
        .cast(pl.Float32)
        .alias("ui_price_ratio")
    )


def _lift_expr() -> pl.Expr:
    """ui_share_ptype_52w / site-wide share of the type; NaN if the site-wide share is 0."""
    site_share = pl.col(GLOBAL_SHARE)
    # A type nobody bought in 52w has no row (null). The user's share is NaN when
    # they have no 52w line, and NaN / x stays NaN.
    lift = pl.col("ui_share_ptype_52w").cast(pl.Float64) / site_share
    return (
        pl.when(site_share.is_null() | (site_share == 0))
        .then(pl.lit(NAN, dtype=pl.Float64))
        .otherwise(lift)
        .cast(pl.Float32)
        .alias(LIFT_FEATURE)
    )


def _attr_match_expr() -> pl.Expr:
    """Number of the six category fields whose candidate value is in the last purchases."""
    seen = [pl.col(_seen(column)).fill_null(0) for column in ARTICLE_CATEGORY_COLUMNS]
    count = pl.sum_horizontal(seen).cast(pl.Float32)
    return (
        pl.when(pl.col(HAS_HISTORY).is_null())
        .then(pl.lit(NAN, dtype=pl.Float32))
        .otherwise(count)
        .alias("ui_attr_match_cnt")
    )


def add_user_item_features(chunk: pl.DataFrame, tables: UserItemTables) -> pl.DataFrame:
    """Add the 21 E-group columns to a chunk of candidates.

    Args:
        chunk: One row per (query_id, article_idx) with customer_idx and the
            already joined features age, item_price_mean_4w,
            user_price_mean_52w, item_buyer_age_mean_4w, user_online_share_52w
            and item_online_share_4w (float32, NaN = missing).
        tables: The fold's aggregates (build_user_item_tables).

    Returns:
        `chunk` with the 21 float32 columns added; row order unchanged.
    """
    joined = _join_tables(chunk, tables)
    age_gap = (pl.col(AGE) - pl.col("item_buyer_age_mean_4w")).abs()
    channel_gap = (pl.col("user_online_share_52w") - pl.col("item_online_share_4w")).abs()
    features = [
        pl.col("ui_cnt_item").cast(pl.Float32).fill_null(0.0),
        pl.col("ui_days_since_item").cast(pl.Float32).fill_null(NAN),
        pl.col("ui_cnt_pcode").cast(pl.Float32).fill_null(0.0),
        pl.col("ui_days_since_pcode").cast(pl.Float32).fill_null(NAN),
        *[_share_expr(feature) for feature in SHARE_FEATURES],
        _price_ratio_expr(),
        age_gap.cast(pl.Float32).alias("ui_age_gap"),
        channel_gap.cast(pl.Float32).alias("ui_channel_gap"),
        _attr_match_expr(),
    ]
    for feature, column in DECAY_FEATURES.items():
        features.append(pl.col(_decay(column)).cast(pl.Float32).fill_null(0.0).alias(feature))
    for feature, column in DAYS_SINCE_FEATURES.items():
        features.append(pl.col(_days(column)).cast(pl.Float32).fill_null(NAN).alias(feature))
    # The lift needs the finished share, so it is added in a second step.
    with_features = joined.with_columns(features).with_columns(_lift_expr())
    temporary = [name for name in with_features.columns if name.startswith("__")]
    return with_features.drop(temporary)
