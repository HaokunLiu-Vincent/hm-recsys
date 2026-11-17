"""Append-only CSV tables for small result records (results/*.csv).

A results file gets one row per event (a run, a fold, a saved model). Rows are
only ever appended, and the header is fixed: appending a row with other
columns, or to a file with another header, is an error. All values are text;
write floats with repr() so reading them back gives the same float64.
"""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path


def read_rows(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    """All rows of a results file as dicts (column -> text); empty if the file does not exist.

    Raises:
        ValueError: If the file's header is not exactly `columns`.
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != list(columns):
            raise ValueError(f"{path} has columns {reader.fieldnames}, expected {list(columns)}")
        return list(reader)


def append_row(path: Path, columns: Sequence[str], row: Mapping[str, str]) -> None:
    """Append one row (and the header if the file is new). Existing rows are never changed.

    Args:
        path: The results file.
        columns: The file's fixed header.
        row: Column -> text, with exactly the keys of `columns`.

    Raises:
        ValueError: If the row, or an existing file, has other columns.
    """
    if list(row) != list(columns):
        raise ValueError(f"row has columns {list(row)}, expected {list(columns)}")
    read_rows(path, columns)  # checks the header of an existing file
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        if is_new:
            writer.writeheader()
        writer.writerow(dict(row))
