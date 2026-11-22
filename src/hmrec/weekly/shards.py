"""Shards of customers in `serve` mode (docs/mlops.md §6.2, §8.4).

A `serve` tick predicts for every customer (1.37 million). Its queries are sorted
by query_id (= by customer_idx) and cut into shards:

    shard k = the queries at positions [k x shard_customers, (k + 1) x shard_customers)

A shard only partitions the customers. Every model and every statistic over all
customers is fitted ONCE per week and used for all shards
(hmrec.weekly.shard_recall, hmrec.weekly.shard_features), so a customer gets the
same candidates, features and scores whatever the shards are.

Brought into main from branch kaggle-late-submission (hmrec.kaggle.plan) without
its Kaggle-specific parts. This module imports neither torch nor lightgbm.
"""

import polars as pl


def number_of_shards(n_queries: int, shard_customers: int) -> int:
    """How many shards n_queries customers need: ceil(n_queries / shard_customers).

    Raises:
        ValueError: If shard_customers <= 0 or there is no query.
    """
    if shard_customers <= 0:
        raise ValueError(f"shard_customers must be > 0, got {shard_customers}")
    if n_queries <= 0:
        raise ValueError(f"a week needs at least one query, got {n_queries}")
    return -(-n_queries // shard_customers)


def shard_queries(queries: pl.DataFrame, shard: int, shard_customers: int) -> pl.DataFrame:
    """The queries of one shard: positions [shard x size, (shard + 1) x size).

    Args:
        queries: The week's query list, sorted by query_id.
        shard: 0-based shard number.
        shard_customers: Customers per shard.

    Raises:
        ValueError: If the shard does not exist.
    """
    n_shards = number_of_shards(queries.height, shard_customers)
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard must be in 0..{n_shards - 1}, got {shard}")
    return queries.slice(shard * shard_customers, shard_customers)
