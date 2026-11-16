"""Map raw string IDs to dense int32 indexes and back (proposal §2.1).

Inputs: customer_id (64-char hex string), article_id (10-digit string with
leading zeros, e.g. "0108775015").

Outputs: two mapping tables, one row per entity:
    - customer_id (str) <-> customer_idx (int32)
    - article_id (str) <-> article_idx (int32)

How indexes are assigned: the string IDs are sorted and numbered 0..N-1. So the
mapping depends only on the SET of IDs, not on the row order of the CSV file.

Why int32: 1.37M customers and 105K articles fit easily, and int32 halves the
memory of int64 (AGENTS.md §4: all IDs are int32).

The embedding files are aligned to article_idx in hmrec.data.preprocess, using
the article map built here.
"""

from pathlib import Path

import polars as pl

CUSTOMER_ID = "customer_id"
CUSTOMER_IDX = "customer_idx"
ARTICLE_ID = "article_id"
ARTICLE_IDX = "article_idx"

# Storage dtype of every *_idx column (AGENTS.md §4).
IDX_DTYPE = pl.Int32

# article_id is a 10-digit string with leading zeros (proposal §2.1). This is a
# property of the Kaggle file format, not a tunable parameter.
ARTICLE_ID_PATTERN = r"^[0-9]{10}$"

# How many bad values to show in an error message.
MAX_EXAMPLES_IN_ERROR = 5


# ---------------------------------------------------------------- building maps


def build_id_map(ids: pl.Series, id_column: str, idx_column: str) -> pl.DataFrame:
    """Number unique string IDs 0..N-1 in sorted order.

    Args:
        ids: String IDs, one per entity. Must not contain nulls or duplicates.
        id_column: Name of the string ID column in the output.
        idx_column: Name of the int32 index column in the output.

    Returns:
        One row per entity, sorted by idx (which is also sorted by ID).
        Columns: idx_column (int32, 0..N-1), id_column (str).

    Raises:
        ValueError: If `ids` contains nulls or duplicates.
    """
    if ids.null_count() > 0:
        raise ValueError(f"{id_column}: found {ids.null_count()} null IDs")
    duplicates = ids.filter(ids.is_duplicated()).unique().sort()
    if duplicates.len() > 0:
        examples = duplicates.head(MAX_EXAMPLES_IN_ERROR).to_list()
        raise ValueError(
            f"{id_column}: found {duplicates.len()} duplicated IDs, e.g. {examples}; "
            f"each entity must appear exactly once"
        )
    sorted_ids = ids.cast(pl.String).sort()
    return pl.DataFrame(
        {
            idx_column: pl.int_range(0, sorted_ids.len(), dtype=IDX_DTYPE, eager=True),
            id_column: sorted_ids,
        }
    )


def build_customer_id_map(customer_ids: pl.Series) -> pl.DataFrame:
    """Build the customer map from the customer_id column of customers.csv.

    Args:
        customer_ids: One customer_id (str) per customer.

    Returns:
        Columns: customer_idx (int32), customer_id (str). One row per customer.
    """
    return build_id_map(customer_ids, CUSTOMER_ID, CUSTOMER_IDX)


def check_article_id_format(article_ids: pl.Series) -> None:
    """Fail if any article_id is not exactly 10 digits.

    A shorter ID almost always means the CSV was read as an integer and the
    leading zeros were lost ("0108775015" -> 108775015).

    Raises:
        ValueError: If any value does not match ARTICLE_ID_PATTERN.
    """
    is_valid = article_ids.cast(pl.String).str.contains(ARTICLE_ID_PATTERN)
    bad_ids = article_ids.filter(~is_valid.fill_null(False))
    if bad_ids.len() > 0:
        examples = bad_ids.head(MAX_EXAMPLES_IN_ERROR).to_list()
        raise ValueError(
            f"article_id must be a 10-digit string with leading zeros; "
            f"found {bad_ids.len()} bad values, e.g. {examples}"
        )


def build_article_id_map(article_ids: pl.Series) -> pl.DataFrame:
    """Build the article map from the article_id column of articles.csv.

    Args:
        article_ids: One article_id (10-digit str) per article.

    Returns:
        Columns: article_idx (int32), article_id (str). One row per article.
    """
    check_article_id_format(article_ids)
    return build_id_map(article_ids, ARTICLE_ID, ARTICLE_IDX)


# ---------------------------------------------------------------- converting


def attach_idx_lazy(
    frame: pl.LazyFrame, id_map: pl.DataFrame, id_column: str, idx_column: str
) -> pl.LazyFrame:
    """Replace a string ID column by its int32 index, lazily.

    This is a left join, so an ID that is not in the map gets a NULL index.
    The caller must check for nulls afterwards (ids_to_idx does it for you).
    The map is small (at most 1.37M rows), so it is the build side of the join
    and `frame` can be streamed.

    Args:
        frame: Any table with a string column `id_column`.
        id_map: Columns idx_column (int32), id_column (str).
        id_column: Name of the string ID column to replace.
        idx_column: Name of the int32 index column to add.

    Returns:
        Same rows in the same order; `id_column` is replaced by `idx_column`
        at the same position.
    """
    output_columns = [
        idx_column if name == id_column else name for name in frame.collect_schema().names()
    ]
    joined = frame.join(id_map.lazy(), on=id_column, how="left", maintain_order="left")
    return joined.select(output_columns)


def _raise_if_unmapped(
    original: pl.DataFrame, converted: pl.DataFrame, from_column: str, to_column: str
) -> None:
    """Fail if some non-null input value got a null output value (it was not in the map)."""
    is_unmapped = converted[to_column].is_null() & original[from_column].is_not_null()
    if is_unmapped.any():
        unknown = original.filter(is_unmapped)[from_column].unique().sort()
        examples = unknown.head(MAX_EXAMPLES_IN_ERROR).to_list()
        raise ValueError(
            f"{unknown.len()} {from_column} values are not in the ID map, e.g. {examples}"
        )


def ids_to_idx(
    frame: pl.DataFrame, id_map: pl.DataFrame, id_column: str, idx_column: str
) -> pl.DataFrame:
    """Replace a string ID column by its int32 index and fail on unknown IDs.

    Args:
        frame: Any table with a string column `id_column`.
        id_map: Columns idx_column (int32), id_column (str).
        id_column: Name of the string ID column.
        idx_column: Name of the int32 index column.

    Returns:
        Same rows in the same order, with `id_column` replaced by `idx_column`.

    Raises:
        ValueError: If any ID is not in the map.
    """
    converted = attach_idx_lazy(frame.lazy(), id_map, id_column, idx_column).collect()
    _raise_if_unmapped(frame, converted, id_column, idx_column)
    return converted


def idx_to_ids(
    frame: pl.DataFrame, id_map: pl.DataFrame, id_column: str, idx_column: str
) -> pl.DataFrame:
    """Replace an int32 index column by the original string ID and fail on unknown indexes.

    Args:
        frame: Any table with an integer column `idx_column`.
        id_map: Columns idx_column (int32), id_column (str).
        id_column: Name of the string ID column.
        idx_column: Name of the int32 index column.

    Returns:
        Same rows in the same order, with `idx_column` replaced by `id_column`.

    Raises:
        ValueError: If any index is not in the map.
    """
    with_int32_idx = frame.with_columns(pl.col(idx_column).cast(IDX_DTYPE))
    converted = attach_idx_lazy(with_int32_idx.lazy(), id_map, idx_column, id_column).collect()
    _raise_if_unmapped(with_int32_idx, converted, idx_column, id_column)
    return converted


def customer_ids_to_idx(frame: pl.DataFrame, customer_map: pl.DataFrame) -> pl.DataFrame:
    """Replace customer_id (str) by customer_idx (int32). See ids_to_idx."""
    return ids_to_idx(frame, customer_map, CUSTOMER_ID, CUSTOMER_IDX)


def customer_idx_to_ids(frame: pl.DataFrame, customer_map: pl.DataFrame) -> pl.DataFrame:
    """Replace customer_idx (int32) by customer_id (str). See idx_to_ids."""
    return idx_to_ids(frame, customer_map, CUSTOMER_ID, CUSTOMER_IDX)


def article_ids_to_idx(frame: pl.DataFrame, article_map: pl.DataFrame) -> pl.DataFrame:
    """Replace article_id (str) by article_idx (int32). See ids_to_idx."""
    return ids_to_idx(frame, article_map, ARTICLE_ID, ARTICLE_IDX)


def article_idx_to_ids(frame: pl.DataFrame, article_map: pl.DataFrame) -> pl.DataFrame:
    """Replace article_idx (int32) by article_id (str). See idx_to_ids."""
    return idx_to_ids(frame, article_map, ARTICLE_ID, ARTICLE_IDX)


# ---------------------------------------------------------------- IO


def write_id_map(id_map: pl.DataFrame, path: Path) -> None:
    """Write an ID map to Parquet, creating the parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    id_map.write_parquet(path)


def read_id_map(path: Path, id_column: str, idx_column: str) -> pl.DataFrame:
    """Read an ID map and check its schema and that idx is exactly 0..N-1.

    Args:
        path: Parquet file written by write_id_map.
        id_column: Expected string ID column.
        idx_column: Expected int32 index column.

    Returns:
        Columns: idx_column (int32), id_column (str), sorted by idx.

    Raises:
        ValueError: If the columns or dtypes are wrong or idx is not 0..N-1.
    """
    id_map = pl.read_parquet(path)
    expected_schema = {idx_column: IDX_DTYPE, id_column: pl.String}
    if dict(id_map.schema) != expected_schema:
        raise ValueError(f"{path}: expected schema {expected_schema}, got {dict(id_map.schema)}")
    expected_idx = pl.int_range(0, id_map.height, dtype=IDX_DTYPE, eager=True)
    if not id_map[idx_column].equals(expected_idx):
        raise ValueError(f"{path}: {idx_column} must be exactly 0..{id_map.height - 1} in order")
    return id_map
