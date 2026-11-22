"""The hit value of a week's recommendations (docs/mlops.md §10, decision D12).

    hit value = the sum of the prices of recommended articles that the customer
                bought in the week, in the dataset's normalised price units

It is tracked in monitoring only, with no threshold. It is closer to business
language than MAP@12, but it is NOT revenue caused by the recommendations: the
purchases in the data were made without them (docs/mlops.md §16).

A purchase counts with every purchase line: a customer who bought a recommended
article twice in the week adds its price twice.

This module reads the answers of a week (through a table built by
hmrec.data.queries.label_week_purchase_values), so it lives in eval/.
"""

import math

import polars as pl

from hmrec.data.queries import PURCHASE_VALUE

QUERY_ID = "query_id"
CUSTOMER_IDX = "customer_idx"
ARTICLE_IDX = "article_idx"


def hit_value(
    recommended: pl.DataFrame, queries: pl.DataFrame, purchase_values: pl.DataFrame
) -> float:
    """The hit value of one week's recommendations.

    Args:
        recommended: One row per recommended (query_id, article_idx): the Top-K
            of every query. A pair must not repeat.
        queries: query_id -> customer_idx of the week.
        purchase_values: One row per (customer_idx, article_idx) bought in the
            week, with purchase_value (float64).

    Returns:
        The sum over the recommended pairs the customer bought, as float64
        (math.fsum over the pairs in (query_id, article_idx) order, so the
        result does not depend on row order).

    Raises:
        ValueError: If a recommended pair repeats or a query_id is unknown.
    """
    pairs = recommended.select(QUERY_ID, ARTICLE_IDX)
    if pairs.is_duplicated().any():
        raise ValueError("a recommended (query_id, article_idx) pair repeats")
    with_customer = pairs.join(queries.select(QUERY_ID, CUSTOMER_IDX), on=QUERY_ID, how="left")
    if with_customer[CUSTOMER_IDX].null_count() > 0:
        raise ValueError("a recommended query_id is not a query of the week")
    hits = with_customer.join(
        purchase_values.select(CUSTOMER_IDX, ARTICLE_IDX, PURCHASE_VALUE),
        on=[CUSTOMER_IDX, ARTICLE_IDX],
        how="inner",
    ).sort(QUERY_ID, ARTICLE_IDX)
    return math.fsum(hits[PURCHASE_VALUE].to_list())
