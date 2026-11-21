"""Convert the raw Kaggle files to compact Parquet / NumPy files (proposal §1.3, §2.1).

Inputs (data/raw/ and data/processed/*_features/):
    transactions_train.csv, customers.csv, articles.csv, and the precomputed
    image / text embeddings with their id tables.

Outputs (data/processed/), all keyed by the int32 indexes from hmrec.data.id_maps:
    - transactions: one row per purchase line. Exact duplicate lines are KEPT,
      because they mean the customer bought the item more than once.
      Columns: customer_idx (int32), article_idx (int32), t_dat_day (int16,
      days since day_zero, see hmrec.data.day_index), price (float32),
      sales_channel_id (int8). Sorted by (t_dat_day, customer_idx).
    - customers: one row per customer. Columns: customer_idx (int32), age (int8,
      null if missing), age_bucket (str), fn (int8 0/1), active (int8 0/1),
      club_member_status (str, nullable), fashion_news_frequency (str, nullable).
    - articles: one row per article. Columns: article_idx (int32), product_code
      (int32) and the §8.4 category columns.
    - embeddings: float32 arrays [n_articles, dim] where row i is article_idx i,
      plus a bool mask [n_articles] that is False for articles without a vector.

Every ID column is read from CSV as a string, so article_id keeps its leading zeros.
Computation functions take and return DataFrames / arrays; the few functions that
read or write files are grouped at the bottom ("IO").
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from hmrec.config import PreprocessConfig
from hmrec.data.day_index import DAY_INDEX_DTYPE, date_to_day_index_expr
from hmrec.data.id_maps import (
    ARTICLE_ID,
    ARTICLE_IDX,
    CUSTOMER_ID,
    CUSTOMER_IDX,
    IDX_DTYPE,
    MAX_EXAMPLES_IN_ERROR,
    article_ids_to_idx,
    attach_idx_lazy,
    customer_ids_to_idx,
)

# ---------------------------------------------------------------- raw file formats

# Full column list of transactions_train.csv, in file order. IDs are read as
# strings (leading zeros); t_dat is parsed later with an explicit format.
TRANSACTIONS_CSV_SCHEMA: dict[str, pl.DataType] = {
    "t_dat": pl.String(),
    CUSTOMER_ID: pl.String(),
    ARTICLE_ID: pl.String(),
    "price": pl.Float64(),
    "sales_channel_id": pl.Int64(),
}
RAW_DATE_FORMAT = "%Y-%m-%d"

# Output schema of the processed transactions table.
# int16 day index: see hmrec.data.day_index. float32 price and int8 channel
# (values 1 and 2) follow the memory plan in proposal §1.3.
TRANSACTIONS_SCHEMA: dict[str, pl.DataType] = {
    CUSTOMER_IDX: IDX_DTYPE(),
    ARTICLE_IDX: IDX_DTYPE(),
    "t_dat_day": DAY_INDEX_DTYPE(),
    "price": pl.Float32(),
    "sales_channel_id": pl.Int8(),
}
TRANSACTIONS_SORT_COLUMNS = ["t_dat_day", CUSTOMER_IDX]

# The largest value an int16 column can hold.
INT16_MAX = int(np.iinfo(np.int16).max)

# fashion_news_frequency has 877,711 "NONE" and 2 "None" in customers.csv.
# They mean the same thing, so the rare spelling is folded into the common one.
FASHION_NEWS_SPELLING_FIXES = {"None": "NONE"}

# The §8.4 category columns (plus product_code for R3 / R6) and their storage dtypes.
# The integer ranges in articles.csv (checked 2026-11-15) fit these dtypes; the
# casts are strict, so a value that does not fit raises instead of wrapping around.
#   product_code 108775..959461, product_type_no -1..762, garment_group_no 1001..1025,
#   index_group_no 1..26, section_no 2..97, colour_group_code -1..93.
ARTICLE_COLUMN_DTYPES: dict[str, pl.DataType] = {
    "product_code": pl.Int32(),
    "product_type_no": pl.Int16(),
    "product_group_name": pl.String(),
    "garment_group_no": pl.Int16(),
    "index_group_no": pl.Int8(),
    "section_no": pl.Int8(),
    "colour_group_code": pl.Int8(),
}
# product_code is the first 7 digits of the 10-digit article_id (Kaggle format).
PRODUCT_CODE_LENGTH = 7

# Columns of the precomputed embedding id tables that we use.
EMBEDDING_ROW = "embedding_row"


# ---------------------------------------------------------------- transactions


def build_transactions(
    raw: pl.LazyFrame,
    customer_map: pl.DataFrame,
    article_map: pl.DataFrame,
    day_zero: date,
) -> pl.LazyFrame:
    """Turn raw transaction lines into the compact, int-keyed table (lazy).

    No rows are removed or merged: exact duplicate lines stay, because each one
    is a separate purchase. Unknown IDs become NULL indexes here and are caught
    by validate_transactions.

    The result is sorted by (t_dat_day, customer_idx). maintain_order=True keeps
    ties (including duplicate lines) in their original file order, so the output
    is deterministic. Sorting by date lets later date filters skip whole Parquet
    row groups. The string IDs are already replaced by int32 at this point, so
    the sort only handles ~15 bytes per row (~0.5 GB for the full data).

    Args:
        raw: One row per purchase line, columns as TRANSACTIONS_CSV_SCHEMA.
        customer_map: Columns customer_idx (int32), customer_id (str).
        article_map: Columns article_idx (int32), article_id (str).
        day_zero: Date that becomes t_dat_day = 0.

    Returns:
        LazyFrame with columns and dtypes TRANSACTIONS_SCHEMA, one row per purchase line.
    """
    purchase_dates = pl.col("t_dat").str.to_date(RAW_DATE_FORMAT, strict=True)
    with_day_index = raw.with_columns(
        date_to_day_index_expr(purchase_dates, day_zero).alias("t_dat_day")
    )
    with_customer_idx = attach_idx_lazy(with_day_index, customer_map, CUSTOMER_ID, CUSTOMER_IDX)
    with_both_idx = attach_idx_lazy(with_customer_idx, article_map, ARTICLE_ID, ARTICLE_IDX)
    typed = with_both_idx.select(
        pl.col(CUSTOMER_IDX),
        pl.col(ARTICLE_IDX),
        pl.col("t_dat_day"),
        pl.col("price").cast(pl.Float32),
        pl.col("sales_channel_id").cast(pl.Int8, strict=True),
    )
    return typed.sort(TRANSACTIONS_SORT_COLUMNS, maintain_order=True)


@dataclass(frozen=True)
class TransactionStats:
    """Summary numbers of a processed transactions table.

    Attributes:
        n_rows: Number of purchase lines.
        min_day: Smallest t_dat_day.
        max_day: Largest t_dat_day.
    """

    n_rows: int
    min_day: int
    max_day: int


def _check_transaction_schema(transactions: pl.LazyFrame) -> None:
    """Fail if the columns or dtypes differ from TRANSACTIONS_SCHEMA."""
    actual_schema = dict(transactions.collect_schema())
    if actual_schema != TRANSACTIONS_SCHEMA:
        raise ValueError(
            f"transactions: expected schema {TRANSACTIONS_SCHEMA}, got {actual_schema}"
        )


def validate_transactions(transactions: pl.LazyFrame, expected_rows: int) -> TransactionStats:
    """Check the processed transactions table in one aggregate pass.

    Checks:
        - columns and dtypes equal TRANSACTIONS_SCHEMA;
        - row count equals the raw CSV row count (nothing dropped, duplicates kept);
        - no NULL index or day (a NULL index means the ID was not in the ID map);
        - min t_dat_day == 0, i.e. the earliest date equals day_zero;
        - max t_dat_day <= INT16_MAX;
        - t_dat_day never decreases from one row to the next (file is date-sorted).

    Args:
        transactions: The processed table, usually pl.scan_parquet of the written file.
        expected_rows: Number of rows in the raw CSV.

    Returns:
        TransactionStats of the table.

    Raises:
        ValueError: If any check fails. The message names the check.
    """
    _check_transaction_schema(transactions)
    stats = transactions.select(
        pl.len().alias("n_rows"),
        pl.col(CUSTOMER_IDX).null_count().alias("null_customer_idx"),
        pl.col(ARTICLE_IDX).null_count().alias("null_article_idx"),
        pl.col("t_dat_day").null_count().alias("null_t_dat_day"),
        pl.col("t_dat_day").min().alias("min_day"),
        pl.col("t_dat_day").max().alias("max_day"),
        # diff() < 0 marks a row whose day is smaller than the previous row's day.
        (pl.col("t_dat_day").diff() < 0).sum().alias("n_day_decreases"),
    ).collect()
    row = stats.row(0, named=True)

    if row["n_rows"] != expected_rows:
        raise ValueError(f"transactions: {row['n_rows']} rows, expected {expected_rows}")
    for column in ("null_customer_idx", "null_article_idx", "null_t_dat_day"):
        if row[column] > 0:
            raise ValueError(
                f"transactions: {row[column]} rows have {column.removeprefix('null_')}"
                " = NULL (ID not in the ID map, or unparsable date)"
            )
    if row["min_day"] != 0:
        raise ValueError(
            f"transactions: earliest t_dat_day is {row['min_day']}, expected 0; "
            f"the first date in the data must equal day_zero in configs/preprocess.yaml"
        )
    if row["max_day"] > INT16_MAX:
        raise ValueError(f"transactions: max t_dat_day {row['max_day']} exceeds int16")
    if row["n_day_decreases"] > 0:
        raise ValueError(
            f"transactions: t_dat_day decreases {row['n_day_decreases']} times; "
            f"the file must be sorted by {TRANSACTIONS_SORT_COLUMNS}"
        )
    return TransactionStats(n_rows=row["n_rows"], min_day=row["min_day"], max_day=row["max_day"])


# ---------------------------------------------------------------- customers


def assign_age_bucket_expr(age: pl.Expr, config: PreprocessConfig) -> pl.Expr:
    """Map an integer age to its R1 age bucket name (proposal §5 R1, frozen).

    Rules:
        - age is NULL                      -> config.unknown_age_bucket (AGE_UNKNOWN)
        - min_age <= age <= max_age of a bucket -> that bucket's name (both ends inclusive)
        - any other age (< 18 or > 99)     -> config.invalid_age_bucket (AGE_INVALID)

    Args:
        age: Integer expression, NULL when the age is missing.
        config: Preprocess config with the bucket edges.

    Returns:
        String expression with exactly one bucket name per row (never NULL).
    """
    # Each when/then gives the bucket name where its rule matches and NULL
    # elsewhere. The buckets do not overlap (checked by the config loader), so
    # at most one of them is non-NULL; coalesce picks it.
    unknown_rule = pl.when(age.is_null()).then(pl.lit(config.unknown_age_bucket))
    bucket_rules = [
        pl.when(age.is_between(bucket.min_age, bucket.max_age, closed="both")).then(
            pl.lit(bucket.name)
        )
        for bucket in config.age_buckets
    ]
    matched_bucket = pl.coalesce([unknown_rule, *bucket_rules])
    # Nothing matched: the age is present but outside every bucket.
    return matched_bucket.fill_null(pl.lit(config.invalid_age_bucket))


def flag_to_int8_expr(column: str) -> pl.Expr:
    """Turn a "1.0"-or-empty flag column (FN, Active) into int8 0/1.

    In customers.csv the only values are "1.0" and empty. Empty means the flag
    is not set, so it becomes 0 (proposal §8.5 lists fn / active as 0/1).
    Any other value is rejected later by _check_flags.
    """
    return pl.col(column).cast(pl.Float32, strict=True).fill_null(0.0).cast(pl.Int8)


def _check_flags(customers: pl.DataFrame, columns: tuple[str, ...]) -> None:
    """Fail if a flag column holds anything other than 0 or 1."""
    for column in columns:
        bad_values = customers.filter(~pl.col(column).is_in([0, 1]))[column].unique()
        if bad_values.len() > 0:
            raise ValueError(f"customers.{column}: expected only 0/1, got {bad_values.to_list()}")


def build_customers(
    raw: pl.LazyFrame, customer_map: pl.DataFrame, config: PreprocessConfig
) -> pl.DataFrame:
    """Build the processed customers table.

    postal_code is dropped on purpose: no recall channel or feature uses it
    (decision 2026-11-15), and it is the largest column of the file.

    Args:
        raw: customers.csv with every column read as a string.
        customer_map: Columns customer_idx (int32), customer_id (str).
        config: Preprocess config (age buckets).

    Returns:
        One row per customer, sorted by customer_idx. Columns:
        customer_idx (int32), age (int8, NULL if missing), age_bucket (str),
        fn (int8), active (int8), club_member_status (str, nullable),
        fashion_news_frequency (str, nullable).
    """
    # Ages in the data are 16..99. int8 holds up to 127; the strict cast raises
    # on anything larger instead of wrapping around.
    age = pl.col("age").cast(pl.Int8, strict=True)
    selected = raw.select(
        pl.col(CUSTOMER_ID),
        age.alias("age"),
        assign_age_bucket_expr(age, config).alias("age_bucket"),
        flag_to_int8_expr("FN").alias("fn"),
        flag_to_int8_expr("Active").alias("active"),
        pl.col("club_member_status"),
        pl.col("fashion_news_frequency").replace(FASHION_NEWS_SPELLING_FIXES),
    ).collect()
    _check_flags(selected, ("fn", "active"))
    customers = customer_ids_to_idx(selected, customer_map)
    return customers.sort(CUSTOMER_IDX)


def count_by_age_bucket(customers: pl.DataFrame, config: PreprocessConfig) -> pl.DataFrame:
    """Count customers per age bucket, listing every bucket even when its count is 0.

    Args:
        customers: Processed customers table (needs column age_bucket).
        config: Preprocess config; defines the bucket order.

    Returns:
        One row per bucket in config order. Columns: age_bucket (str), n_customers (int64).
    """
    all_buckets = pl.DataFrame({"age_bucket": list(config.all_age_bucket_names)})
    counts = customers.group_by("age_bucket").agg(pl.len().cast(pl.Int64).alias("n_customers"))
    unexpected = counts.join(all_buckets, on="age_bucket", how="anti")
    if unexpected.height > 0:
        raise ValueError(f"unexpected age_bucket values: {unexpected['age_bucket'].to_list()}")
    joined = all_buckets.join(counts, on="age_bucket", how="left", maintain_order="left")
    return joined.with_columns(pl.col("n_customers").fill_null(0))


# ---------------------------------------------------------------- articles


def check_product_code_matches_article_id(raw_articles: pl.DataFrame) -> None:
    """Fail unless product_code equals the first 7 characters of article_id on every row.

    R3 (sibling) and the Two-Tower product_code embedding rely on product_code
    grouping the colour variants of one product, so a mismatch must stop the run.

    Args:
        raw_articles: Columns article_id (str) and product_code (str), as in the CSV.

    Raises:
        ValueError: Listing a few mismatching article_ids.
    """
    derived_code = pl.col(ARTICLE_ID).str.slice(0, PRODUCT_CODE_LENGTH)
    mismatches = raw_articles.filter(derived_code != pl.col("product_code"))
    if mismatches.height > 0:
        examples = mismatches.head(MAX_EXAMPLES_IN_ERROR).select(ARTICLE_ID, "product_code")
        raise ValueError(
            f"articles: {mismatches.height} rows have product_code != article_id[:"
            f"{PRODUCT_CODE_LENGTH}], e.g. {examples.rows()}"
        )


def build_articles(raw: pl.LazyFrame, article_map: pl.DataFrame) -> pl.DataFrame:
    """Build the processed articles table with product_code and the §8.4 category columns.

    Args:
        raw: articles.csv with every column read as a string.
        article_map: Columns article_idx (int32), article_id (str).

    Returns:
        One row per article, sorted by article_idx. Columns: article_idx (int32)
        and ARTICLE_COLUMN_DTYPES (product_code int32, product_type_no int16,
        product_group_name str, garment_group_no int16, index_group_no int8,
        section_no int8, colour_group_code int8).
    """
    raw_articles = raw.select(ARTICLE_ID, *ARTICLE_COLUMN_DTYPES.keys()).collect()
    check_product_code_matches_article_id(raw_articles)
    # Casting "0100000" to int32 drops the leading zero. That is fine: product_code
    # is only used as a grouping key, and the 7-digit check above already passed.
    typed = raw_articles.with_columns(
        pl.col(column).cast(dtype, strict=True) for column, dtype in ARTICLE_COLUMN_DTYPES.items()
    )
    articles = article_ids_to_idx(typed, article_map)
    return articles.sort(ARTICLE_IDX)


# ---------------------------------------------------------------- embeddings


def _check_embedding_array(vectors: np.ndarray, expected_dim: int) -> None:
    """Fail unless `vectors` is a 2-D float32 array with `expected_dim` columns."""
    if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
        raise ValueError(f"embeddings: expected shape [n, {expected_dim}], got {vectors.shape}")
    if vectors.dtype != np.float32:
        raise ValueError(f"embeddings: expected float32, got {vectors.dtype}")


def _check_embedding_row_ids(row_ids: pl.DataFrame, n_vectors: int) -> None:
    """Fail unless embedding_row is exactly 0..n_vectors-1 and article_id is unique."""
    # Compare as int64: the source tables store embedding_row as int64.
    rows = row_ids[EMBEDDING_ROW].cast(pl.Int64).sort()
    expected_rows = pl.int_range(0, n_vectors, dtype=pl.Int64, eager=True)
    if not rows.equals(expected_rows):
        raise ValueError(
            f"embedding id table: {EMBEDDING_ROW} must be exactly 0..{n_vectors - 1} "
            f"(one id row per vector), got {rows.len()} rows"
        )
    duplicated = row_ids.filter(pl.col(ARTICLE_ID).is_duplicated())[ARTICLE_ID].unique()
    if duplicated.len() > 0:
        examples = duplicated.head(MAX_EXAMPLES_IN_ERROR).to_list()
        raise ValueError(f"embedding id table: duplicated article_id, e.g. {examples}")


def check_unit_norms(aligned: np.ndarray, mask: np.ndarray, tolerance: float) -> None:
    """Fail unless every present vector has L2 norm within 1 +/- tolerance.

    The embeddings are documented as L2-normalized (proposal §2.1), so dot
    products are cosine similarities. This checks ALL rows, not a sample.

    Args:
        aligned: float32 [n_articles, dim].
        mask: bool [n_articles]; only rows with True are checked.
        tolerance: Allowed absolute deviation of the norm from 1.

    Raises:
        ValueError: With the number of bad rows and the worst article_idx.
    """
    present_idx = np.flatnonzero(mask)
    norms = np.linalg.norm(aligned[present_idx], axis=1)
    deviation = np.abs(norms - 1.0)
    n_bad = int(np.count_nonzero(deviation > tolerance))
    if n_bad > 0:
        worst_position = int(np.argmax(deviation))
        raise ValueError(
            f"embeddings: {n_bad} vectors have L2 norm outside 1 +/- {tolerance}; worst is "
            f"article_idx {int(present_idx[worst_position])} with norm "
            f"{float(norms[worst_position]):.6f}"
        )


def align_embeddings(
    vectors: np.ndarray,
    row_ids: pl.DataFrame,
    article_map: pl.DataFrame,
    expected_dim: int,
    norm_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder precomputed embeddings so that row i belongs to article_idx i.

    The source file's row order is NOT article_idx order (proposal §2.1): row r
    of `vectors` belongs to the article_id with embedding_row = r in `row_ids`.

    Articles without a vector get an all-zero row and mask False. The zeros are
    only a placeholder; consumers must use the mask, not treat zero as a real input.

    Args:
        vectors: float32 [n_vectors, expected_dim], source file order.
        row_ids: One row per vector. Columns: embedding_row (int, 0..n_vectors-1),
            article_id (10-digit str).
        article_map: Columns article_idx (int32), article_id (str); n_articles rows.
        expected_dim: Required vector width.
        norm_tolerance: Allowed deviation of each present vector's L2 norm from 1.

    Returns:
        aligned: float32 [n_articles, expected_dim].
        mask: bool [n_articles], True where the article has a vector.

    Raises:
        ValueError: On a wrong shape / dtype, bad embedding_row values, duplicated
            or unknown article_id, or a vector that is not unit length.
    """
    _check_embedding_array(vectors, expected_dim)
    _check_embedding_row_ids(row_ids, vectors.shape[0])
    with_idx = article_ids_to_idx(row_ids.select(EMBEDDING_ROW, ARTICLE_ID), article_map)
    source_rows = with_idx[EMBEDDING_ROW].to_numpy()
    target_rows = with_idx[ARTICLE_IDX].to_numpy()

    n_articles = article_map.height
    aligned = np.zeros((n_articles, expected_dim), dtype=np.float32)
    aligned[target_rows] = vectors[source_rows]
    mask = np.zeros(n_articles, dtype=bool)
    mask[target_rows] = True

    check_unit_norms(aligned, mask, norm_tolerance)
    return aligned, mask


# ---------------------------------------------------------------- IO


def scan_transactions_csv(path: Path) -> pl.LazyFrame:
    """Lazily scan transactions_train.csv with an explicit schema (IDs as strings)."""
    return pl.scan_csv(path, schema=TRANSACTIONS_CSV_SCHEMA)


def scan_all_string_csv(path: Path) -> pl.LazyFrame:
    """Lazily scan a CSV with every column as a string (customers.csv, articles.csv).

    Reading everything as text keeps leading zeros ("0100000015") and lets the
    build_* functions do every cast explicitly and strictly.
    """
    return pl.scan_csv(path, infer_schema=False)


def count_rows(frame: pl.LazyFrame) -> int:
    """Count the rows of a lazy table with the streaming engine."""
    return int(frame.select(pl.len()).collect(engine="streaming").item())


def write_transactions(transactions: pl.LazyFrame, path: Path, row_group_size: int) -> None:
    """Stream the processed transactions to Parquet.

    Small row groups (from configs/preprocess.yaml) plus min/max statistics let
    readers that filter on t_dat_day skip row groups outside the date range,
    because the file is sorted by t_dat_day.

    Args:
        transactions: Output of build_transactions.
        path: Destination file (parent folder is created).
        row_group_size: Rows per Parquet row group.
    """
    transactions.sink_parquet(
        path,
        row_group_size=row_group_size,
        statistics=True,
        maintain_order=True,
        mkdir=True,
        engine="streaming",
    )


def write_table(table: pl.DataFrame, path: Path) -> None:
    """Write a small table (customers, articles) to Parquet, creating the parent folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(path)


def read_embedding_source(vectors_path: Path, ids_path: Path) -> tuple[np.ndarray, pl.DataFrame]:
    """Read one precomputed embedding file and its id table (read-only inputs).

    Returns:
        vectors: float32 [n_vectors, dim] in source order.
        row_ids: Columns embedding_row, article_id; one row per vector.
    """
    vectors = np.load(vectors_path)
    row_ids = pl.read_parquet(ids_path, columns=[EMBEDDING_ROW, ARTICLE_ID])
    return vectors, row_ids


def write_aligned_embeddings(
    aligned: np.ndarray, mask: np.ndarray, vectors_path: Path, mask_path: Path
) -> None:
    """Save an aligned embedding array and its mask as .npy files."""
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, aligned)
    np.save(mask_path, mask)


@dataclass(frozen=True)
class TableSummary:
    """Size of one processed output, for the M0 report.

    Attributes:
        n_rows: Number of rows.
        file_bytes: Size of the file on disk.
        memory_bytes: Size when fully loaded (Polars estimated_size, or ndarray.nbytes).
    """

    n_rows: int
    file_bytes: int
    memory_bytes: int


def summarize_parquet_table(path: Path) -> TableSummary:
    """Load a Parquet table fully and report its rows, file size and in-memory size."""
    table = pl.read_parquet(path)
    return TableSummary(
        n_rows=table.height,
        file_bytes=path.stat().st_size,
        memory_bytes=int(table.estimated_size()),
    )


def summarize_npy_array(path: Path) -> TableSummary:
    """Load a .npy array and report its rows, file size and in-memory size (nbytes)."""
    array = np.load(path)
    return TableSummary(
        n_rows=int(array.shape[0]),
        file_bytes=path.stat().st_size,
        memory_bytes=int(array.nbytes),
    )
