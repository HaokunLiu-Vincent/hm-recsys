"""Group D: customer features, 12 columns (proposal §8.5). One row per query customer and anchor.

    age                              the customer's age as stored; NaN if unknown
    club_member_status,
    fashion_news_frequency           int16 category codes (0 = missing)
    fn, active                       0 / 1
    user_cnt_4w                      purchase lines in [a - 28, a); 0 if none
    user_cnt_52w                     purchase lines in [a - 364, a); 0 if none
    user_days_since_last_purchase    a - (last purchase day before a); NaN if never bought
    user_price_mean_52w              mean price over the lines of [a - 364, a); NaN if none
    user_online_share_52w            share of those lines with the online channel; NaN if none
    user_uniq_items_52w              distinct articles in [a - 364, a); 0 if none
    user_purchase_days_52w           distinct days with a purchase in [a - 364, a); 0 if none

user_price_mean_52w: `price` is cast to float64 before the mean and the mean to
float32 at the end, so the stored value does not depend on the row order (see
features/item.py).

Only the fold's query customers are computed.
"""

import polars as pl

from hmrec.config import FeaturesConfig
from hmrec.data.asof import TRANSACTION_DAY
from hmrec.features.base import (
    AGE,
    AGE_DAYS,
    ARTICLE_IDX,
    CUSTOMER_CATEGORY_COLUMNS,
    CUSTOMER_IDX,
    PRICE,
    SALES_CHANNEL,
    FoldData,
    as_feature,
    lines_before,
    window_lines,
)
from hmrec.features.categories import CategoryCodes, encode_categories

PROFILE_FEATURES = (AGE, "fn", "active")
COUNT_FEATURES = (
    "user_cnt_4w",
    "user_cnt_52w",
    "user_uniq_items_52w",
    "user_purchase_days_52w",
)
OTHER_FEATURES = (
    "user_days_since_last_purchase",
    "user_price_mean_52w",
    "user_online_share_52w",
)
# Column order of proposal §8.5.
USER_FEATURES = (
    AGE,
    "club_member_status",
    "fashion_news_frequency",
    "fn",
    "active",
    "user_cnt_4w",
    "user_cnt_52w",
    "user_days_since_last_purchase",
    "user_price_mean_52w",
    "user_online_share_52w",
    "user_uniq_items_52w",
    "user_purchase_days_52w",
)


def purchase_stats(data: FoldData, config: FeaturesConfig) -> pl.DataFrame:
    """The purchase statistics of the query customers that have any purchase before the anchor.

    Returns:
        One row per such customer; a statistic is null where its window is empty.
    """
    customers = data.query_customers
    four_weeks = window_lines(data, config.four_week_days, [CUSTOMER_IDX], customers=customers)
    year_columns = [CUSTOMER_IDX, ARTICLE_IDX, PRICE, SALES_CHANNEL]
    year = window_lines(data, config.year_days, year_columns, customers=customers)
    everything = lines_before(data, [CUSTOMER_IDX], customers=customers)
    is_online = pl.col(SALES_CHANNEL) == config.online_sales_channel_id
    last_purchase = everything.group_by(CUSTOMER_IDX).agg(
        pl.col(AGE_DAYS).min().alias("user_days_since_last_purchase")
    )
    count_4w = four_weeks.group_by(CUSTOMER_IDX).agg(pl.len().alias("user_cnt_4w"))
    year_stats = year.group_by(CUSTOMER_IDX).agg(
        pl.len().alias("user_cnt_52w"),
        pl.col(PRICE).cast(pl.Float64).mean().alias("user_price_mean_52w"),
        is_online.cast(pl.Float64).mean().alias("user_online_share_52w"),
        pl.col(ARTICLE_IDX).n_unique().alias("user_uniq_items_52w"),
        pl.col(TRANSACTION_DAY).n_unique().alias("user_purchase_days_52w"),
    )
    return last_purchase.join(count_4w, on=CUSTOMER_IDX, how="left").join(
        year_stats, on=CUSTOMER_IDX, how="left"
    )


def build_user_table(data: FoldData, config: FeaturesConfig, codes: CategoryCodes) -> pl.DataFrame:
    """The D-group features of every query customer, as of the fold's anchor.

    Returns:
        One row per distinct query customer, sorted by customer_idx: customer_idx
        (int32) and the 12 features in the order of proposal §8.5 (float32;
        the two category codes int16).

    Raises:
        ValueError: If a query customer is not in the customers table.
    """
    wanted = pl.DataFrame({CUSTOMER_IDX: data.query_customers}, schema={CUSTOMER_IDX: pl.Int32})
    profile_columns = [CUSTOMER_IDX, *PROFILE_FEATURES, *CUSTOMER_CATEGORY_COLUMNS]
    profiles = wanted.join(data.customers.select(profile_columns), on=CUSTOMER_IDX, how="inner")
    if profiles.height != wanted.height:
        raise ValueError(
            f"{wanted.height - profiles.height} query customers are not in the customers table"
        )
    table = encode_categories(profiles, CUSTOMER_CATEGORY_COLUMNS, codes)
    table = table.join(purchase_stats(data, config), on=CUSTOMER_IDX, how="left")
    numeric = [as_feature(name) for name in (*PROFILE_FEATURES, *OTHER_FEATURES)]
    numeric += [as_feature(name, count=True) for name in COUNT_FEATURES]
    return table.with_columns(numeric).select(CUSTOMER_IDX, *USER_FEATURES).sort(CUSTOMER_IDX)
