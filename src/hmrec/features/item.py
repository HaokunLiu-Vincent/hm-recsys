"""Group C: article features, 17 columns (proposal §8.4). One row per article and anchor.

Purchase LINES are counted (two identical lines count twice, as in R1 / R2).

    item_cnt_1w                  purchase lines in [a - 7, a); 0 if none
    item_cnt_prev_1w             purchase lines in [a - 14, a - 7); 0 if none
    item_cnt_4w                  purchase lines in [a - 28, a); 0 if none
    item_uniq_buyers_4w          distinct customers in [a - 28, a); 0 if none
    item_days_since_first_sale   a - (first purchase day in all history); NaN if never sold
    item_days_since_last_sale    a - (last purchase day before a); NaN if never sold
    item_price_mean_4w           mean price over the purchase lines of [a - 28, a); NaN if none
    item_online_share_4w         share of those lines with the online sales channel; NaN if none
    item_buyer_age_mean_4w       mean age over the DISTINCT buyers of [a - 28, a) whose age
                                 is known (each buyer counts once); NaN if there is none
    item_repurchase_ratio_52w    among the customers who bought the article in
                                 [a - 364, a): the share who bought it on 2 or more
                                 DIFFERENT DAYS; NaN if nobody bought it
    item_discount_ratio          (mean price of the lines in [a - 7, a)) /
                                 (mean price of the lines in [a - 364, a)); below 1 = it
                                 sold cheaper last week than over the year; NaN if either
                                 window has no sale
    product_type_no, product_group_name, garment_group_no, index_group_no,
    section_no, colour_group_code   int16 category codes (features/categories.py)

"Days since" is age_days = anchor_day - t_dat_day, so a sale yesterday gives 1.

Price means: `price` is float32 in the data. It is cast to float64 BEFORE it is
averaged and the mean is cast to float32 at the end. A multi-threaded group-by
may add the values in another order from run to run; in float64 that changes
the mean around the 16th digit, far below what float32 keeps, so the stored
float32 value does not depend on the row order (test_feature_item.py).
"""

import polars as pl

from hmrec.config import FeaturesConfig
from hmrec.data.asof import TRANSACTION_DAY
from hmrec.features.base import (
    AGE,
    AGE_DAYS,
    ARTICLE_CATEGORY_COLUMNS,
    ARTICLE_IDX,
    CUSTOMER_IDX,
    PRICE,
    SALES_CHANNEL,
    FoldData,
    as_feature,
    lines_before,
    window_lines,
)
from hmrec.features.categories import CategoryCodes, encode_categories

COUNT_FEATURES = ("item_cnt_1w", "item_cnt_prev_1w", "item_cnt_4w", "item_uniq_buyers_4w")
OTHER_FEATURES = (
    "item_days_since_first_sale",
    "item_days_since_last_sale",
    "item_price_mean_4w",
    "item_online_share_4w",
    "item_buyer_age_mean_4w",
    "item_repurchase_ratio_52w",
    "item_discount_ratio",
)


def weekly_counts(data: FoldData, config: FeaturesConfig) -> pl.DataFrame:
    """item_cnt_1w and item_cnt_prev_1w per article that has a line in either week.

    Returns:
        One row per article; a count is null where the article has no line in that week.
    """
    week = config.week_days
    this_week = window_lines(data, week, [ARTICLE_IDX])
    previous_week = window_lines(data, week, [ARTICLE_IDX], days_before_anchor=week)
    current = this_week.group_by(ARTICLE_IDX).agg(pl.len().alias("item_cnt_1w"))
    previous = previous_week.group_by(ARTICLE_IDX).agg(pl.len().alias("item_cnt_prev_1w"))
    return current.join(previous, on=ARTICLE_IDX, how="full", coalesce=True)


def four_week_stats(data: FoldData, config: FeaturesConfig) -> pl.DataFrame:
    """The five features of the window [a - 28, a), per article sold in it."""
    columns = [CUSTOMER_IDX, ARTICLE_IDX, PRICE, SALES_CHANNEL]
    lines = window_lines(data, config.four_week_days, columns)
    is_online = pl.col(SALES_CHANNEL) == config.online_sales_channel_id
    per_article = lines.group_by(ARTICLE_IDX).agg(
        pl.len().alias("item_cnt_4w"),
        pl.col(CUSTOMER_IDX).n_unique().alias("item_uniq_buyers_4w"),
        pl.col(PRICE).cast(pl.Float64).mean().alias("item_price_mean_4w"),
        is_online.cast(pl.Float64).mean().alias("item_online_share_4w"),
    )
    # Each buyer counts once, however many lines they bought; unknown ages are
    # skipped by mean(), and an article whose buyers all lack an age gets null.
    buyers = lines.select(ARTICLE_IDX, CUSTOMER_IDX).unique()
    buyer_ages = buyers.join(data.customers.select(CUSTOMER_IDX, AGE), on=CUSTOMER_IDX, how="left")
    age_mean = buyer_ages.group_by(ARTICLE_IDX).agg(
        pl.col(AGE).cast(pl.Float64).mean().alias("item_buyer_age_mean_4w")
    )
    return per_article.join(age_mean, on=ARTICLE_IDX, how="left")


def sale_days(data: FoldData) -> pl.DataFrame:
    """Days since the first and the last sale, per article ever sold before the anchor."""
    lines = lines_before(data, [ARTICLE_IDX])
    return lines.group_by(ARTICLE_IDX).agg(
        # The first sale is the oldest line: the largest age_days.
        pl.col(AGE_DAYS).max().alias("item_days_since_first_sale"),
        pl.col(AGE_DAYS).min().alias("item_days_since_last_sale"),
    )


def repurchase_ratio(data: FoldData, config: FeaturesConfig) -> pl.DataFrame:
    """item_repurchase_ratio_52w per article sold in [a - 364, a).

    ratio = (customers who bought it on >= 2 different days) / (customers who bought it).
    Two lines on the same day are one purchase occasion (e.g. two pieces in one
    basket), not a repurchase.
    """
    lines = window_lines(data, config.year_days, [CUSTOMER_IDX, ARTICLE_IDX])
    purchase_days = lines.select(CUSTOMER_IDX, ARTICLE_IDX, TRANSACTION_DAY).unique()
    per_buyer = purchase_days.group_by(ARTICLE_IDX, CUSTOMER_IDX).agg(pl.len().alias("n_days"))
    return per_buyer.group_by(ARTICLE_IDX).agg(
        (pl.col("n_days") >= 2).cast(pl.Float64).mean().alias("item_repurchase_ratio_52w")
    )


def discount_ratio(data: FoldData, config: FeaturesConfig) -> pl.DataFrame:
    """item_discount_ratio per article sold both in [a - 7, a) and in [a - 364, a).

    ratio = mean price of last week's lines / mean price of the year's lines.
    The year window contains the last week, so an article sold last week always
    has a year mean; an article without a sale last week is not in the result
    (its ratio is NaN).
    """
    week = window_lines(data, config.week_days, [ARTICLE_IDX, PRICE])
    year = window_lines(data, config.year_days, [ARTICLE_IDX, PRICE])
    # float64 before the mean: see the module docstring.
    week_mean = week.group_by(ARTICLE_IDX).agg(pl.col(PRICE).cast(pl.Float64).mean().alias("week"))
    year_mean = year.group_by(ARTICLE_IDX).agg(pl.col(PRICE).cast(pl.Float64).mean().alias("year"))
    both = week_mean.join(year_mean, on=ARTICLE_IDX, how="inner")
    ratio = pl.when(pl.col("year") == 0).then(None).otherwise(pl.col("week") / pl.col("year"))
    return both.select(ARTICLE_IDX, ratio.alias("item_discount_ratio"))


def build_item_table(data: FoldData, config: FeaturesConfig, codes: CategoryCodes) -> pl.DataFrame:
    """The C-group features of every article of the catalog, as of the fold's anchor.

    Uses only purchases before the anchor (see the module docstring for the windows).

    Returns:
        One row per article in data.articles, sorted by article_idx: article_idx
        (int32), the eleven numeric features (float32) and the six category codes (int16).
    """
    table = encode_categories(
        data.articles.select(ARTICLE_IDX, *ARTICLE_CATEGORY_COLUMNS),
        ARTICLE_CATEGORY_COLUMNS,
        codes,
    )
    for part in (
        weekly_counts(data, config),
        four_week_stats(data, config),
        sale_days(data),
        repurchase_ratio(data, config),
        discount_ratio(data, config),
    ):
        table = table.join(part, on=ARTICLE_IDX, how="left")
    numeric = [as_feature(name, count=True) for name in COUNT_FEATURES]
    numeric += [as_feature(name) for name in OTHER_FEATURES]
    return table.select(ARTICLE_IDX, *numeric, *ARTICLE_CATEGORY_COLUMNS).sort(ARTICLE_IDX)
