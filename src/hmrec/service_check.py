"""What the recommendation service SHOULD answer, derived without its serving form (M14).

The service answers from predictions/{anchor}/serving/ (hmrec.serving_set). To
check its answers, the expected ones are derived here from other files of the
same tick, on another path:

    a known customer    the rows of the prediction set itself (top.parquet:
                        customer_idx, rank, article_idx), translated with the two
                        id maps of data/processed/;
    an unknown customer the top_k articles of R1's score table S(g, i) for the
                        "age missing" bucket (r1_bucket_scores.parquet), ordered
                        by (score DESC, article_idx ASC).

The second one is NOT the definition of R1's list (which keeps only articles of
the active pool and fills up with global popularity); it equals the list when
the bucket's best top_k articles are all in the active pool, which is the normal
case. A difference is reported, not hidden.

Customer ids are written to the expectations file only; that file is not put
into the repository.

This module belongs to the upper layer and imports polars; it loads neither
torch nor lightgbm, and nothing imports it but its script.
"""

import polars as pl

CUSTOMER_IDX = "customer_idx"
CUSTOMER_ID = "customer_id"
ARTICLE_IDX = "article_idx"
ARTICLE_ID = "article_id"
AGE_BUCKET = "age_bucket"
RANK = "rank"
SCORE = "score"


def expected_known(
    predictions: pl.DataFrame,
    customer_id_map: pl.DataFrame,
    article_id_map: pl.DataFrame,
    n_customers: int,
    seed: int,
) -> dict[str, list[str]]:
    """The Top-K of some customers of a prediction set, in original ids.

    Args:
        predictions: One row per (customer, rank). Columns: customer_idx, rank,
            article_idx.
        customer_id_map: customer_idx -> customer_id.
        article_id_map: article_idx -> article_id.
        n_customers: How many customers to draw (all, if the set has fewer).
        seed: Seed of the draw.

    Returns:
        customer_id -> its article ids, rank 1 first.
    """
    customers = predictions.select(pl.col(CUSTOMER_IDX).unique().sort())
    drawn = customers.sample(n=min(n_customers, customers.height), seed=seed)
    rows = (
        predictions.join(drawn, on=CUSTOMER_IDX, how="semi")
        .join(customer_id_map.select(CUSTOMER_IDX, CUSTOMER_ID), on=CUSTOMER_IDX, how="left")
        .join(article_id_map.select(ARTICLE_IDX, ARTICLE_ID), on=ARTICLE_IDX, how="left")
        .sort(CUSTOMER_ID, RANK)
    )
    if rows[CUSTOMER_ID].null_count() > 0 or rows[ARTICLE_ID].null_count() > 0:
        raise ValueError("a prediction row has no customer id or no article id in the id maps")
    grouped = rows.group_by(CUSTOMER_ID, maintain_order=True).agg(pl.col(ARTICLE_ID))
    return dict(zip(grouped[CUSTOMER_ID].to_list(), grouped[ARTICLE_ID].to_list(), strict=True))


def expected_fallback(
    bucket_scores: pl.DataFrame, article_id_map: pl.DataFrame, bucket: str, top_k: int
) -> list[str]:
    """The top_k articles of one age bucket in R1's score table, in original ids.

    Args:
        bucket_scores: One row per (age_bucket, article_idx) with its score S(g, i).
        article_id_map: article_idx -> article_id.
        bucket: The age bucket, e.g. "AGE_UNKNOWN".
        top_k: Length of the list.

    Raises:
        ValueError: If the bucket has fewer than top_k articles in the table.
    """
    best = (
        bucket_scores.filter(pl.col(AGE_BUCKET) == bucket)
        .sort([SCORE, ARTICLE_IDX], descending=[True, False])
        .head(top_k)
        .join(article_id_map.select(ARTICLE_IDX, ARTICLE_ID), on=ARTICLE_IDX, how="left")
    )
    if best.height < top_k or best[ARTICLE_ID].null_count() > 0:
        raise ValueError(f"bucket {bucket!r} has no {top_k} known articles in the score table")
    return best[ARTICLE_ID].to_list()


def unknown_customer_ids(customer_id_map: pl.DataFrame, n_customers: int) -> list[str]:
    """Ids that are NOT customers: 64 hexadecimal characters that the id map does not hold.

    The candidates are the numbers 0, 1, 2, ... written with 64 hexadecimal
    digits; one that happens to be a customer id is skipped.
    """
    known = set(customer_id_map[CUSTOMER_ID].to_list())
    unknown: list[str] = []
    number = 0
    while len(unknown) < n_customers:
        candidate = f"{number:064x}"
        if candidate not in known:
            unknown.append(candidate)
        number += 1
    return unknown
