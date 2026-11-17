"""Integer codes of the categorical features (proposal §8.1, §8.4, §8.5).

A category field becomes an int16 code:
    0      = missing (null)
    1 .. K = the distinct non-null values of the field in sorted order

The code table is built from ALL rows of the articles and customers tables, so
it does not depend on a fold: every fold uses the same codes, and a code means
the same value in the Ranker's training weeks and prediction weeks. It is saved
to artifacts/features/category_codes.json; later folds must reproduce it exactly.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hmrec.features.base import ARTICLE_CATEGORY_COLUMNS, CUSTOMER_CATEGORY_COLUMNS

MISSING_CODE = 0
CODE_DTYPE = pl.Int16
INT16_MAX = 32767
FORMAT_VERSION = 1


@dataclass(frozen=True)
class CategoryCodes:
    """The code table of every categorical feature.

    Attributes:
        values: Field name -> its distinct non-null values in sorted order.
            The value at position p has code p + 1; null has code 0.
    """

    values: dict[str, tuple[int | str, ...]]


def _sorted_values(column: pl.Series) -> tuple[int | str, ...]:
    """Distinct non-null values of a column in ascending order."""
    values = tuple(column.drop_nulls().unique().sort().to_list())
    if len(values) >= INT16_MAX:
        raise ValueError(f"{column.name} has {len(values)} values: codes would not fit in int16")
    return values


def build_category_codes(articles: pl.DataFrame, customers: pl.DataFrame) -> CategoryCodes:
    """Build the code table from the full articles and customers tables.

    Args:
        articles: One row per article with the six category fields of §8.4.
        customers: One row per customer with club_member_status and
            fashion_news_frequency.
    """
    values: dict[str, tuple[int | str, ...]] = {}
    for name in ARTICLE_CATEGORY_COLUMNS:
        values[name] = _sorted_values(articles[name])
    for name in CUSTOMER_CATEGORY_COLUMNS:
        values[name] = _sorted_values(customers[name])
    return CategoryCodes(values)


def encode_categories(
    table: pl.DataFrame, columns: tuple[str, ...], codes: CategoryCodes
) -> pl.DataFrame:
    """Replace category columns by their int16 codes (0 = missing).

    Args:
        table: Any table holding `columns` with their original values.
        columns: The category columns to encode.
        codes: The code table.

    Returns:
        `table` with each of `columns` replaced by its int16 code; the row
        order and the other columns are unchanged.

    Raises:
        ValueError: If a non-null value is not in the code table.
    """
    encoded = table
    for name in columns:
        values = codes.values[name]
        mapping = pl.DataFrame(
            {
                name: pl.Series(name, list(values), dtype=table.schema[name]),
                "__code": pl.Series("__code", range(1, len(values) + 1), dtype=CODE_DTYPE),
            }
        )
        joined = encoded.join(mapping, on=name, how="left", maintain_order="left")
        unknown = joined.filter(pl.col(name).is_not_null() & pl.col("__code").is_null())
        if unknown.height > 0:
            examples = unknown[name].unique().head(5).to_list()
            raise ValueError(f"{name}: values {examples} are not in the category code table")
        encoded = joined.with_columns(pl.col("__code").fill_null(MISSING_CODE).alias(name)).drop(
            "__code"
        )
    return encoded


def save_or_check_category_codes(path: Path, codes: CategoryCodes) -> None:
    """Write the code table, or (if the file exists) require it to be the same table.

    The first fold writes the file. Every later fold builds the table again
    from the same two tables and must get exactly the saved one; a difference
    means the processed data changed, and features of different folds would
    no longer use the same codes.

    Raises:
        ValueError: If the saved table differs from `codes`.
    """
    content = {
        "format_version": FORMAT_VERSION,
        "codes": {k: list(v) for k, v in codes.values.items()},
    }
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != content:
            raise ValueError(
                f"{path} holds another category code table than the current articles / "
                "customers tables give; features of different folds would not be comparable"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")
