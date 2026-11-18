"""Shared pieces of the diagnostics' result tables (results/t_diag_*.csv).

Every row ends with two columns that say which run wrote it:

    git_commit       HEAD of the run (the working tree is clean, or no T label is read)
    run_started_at   the time the run started; for a run that reads T labels it is
                     the `recorded_at` of its row in results/t_diag_label_access.csv

The tables are append-only (hmrec.csv_records). A second run appends its rows
with its own run_started_at; rows are never changed.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

from hmrec.csv_records import append_row

PROVENANCE_COLUMNS: tuple[str, ...] = ("git_commit", "run_started_at")
NOT_COMPUTED = ""  # an empty cell, e.g. the mean of a group with no query


def float_text(value: float | None) -> str:
    """A float as CSV text (repr, so it reads back exactly); None as an empty cell."""
    return NOT_COMPUTED if value is None else repr(value)


def append_rows(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    git_commit: str,
    run_started_at: str,
) -> None:
    """Append rows to a diagnostics table, adding the two provenance columns.

    Args:
        path: The table, results/t_diag_*.csv.
        columns: The table's columns WITHOUT the provenance columns.
        rows: Rows with exactly `columns`.
        git_commit: HEAD of the run.
        run_started_at: The start time of the run.
    """
    header = (*columns, *PROVENANCE_COLUMNS)
    for row in rows:
        stamped = {**row, "git_commit": git_commit, "run_started_at": run_started_at}
        append_row(path, header, stamped)
