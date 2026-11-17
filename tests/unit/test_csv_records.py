"""Tests for hmrec.csv_records: append-only result tables with a fixed header."""

from pathlib import Path

import pytest

from hmrec.csv_records import append_row, read_rows

COLUMNS = ("fold", "value")


def test_rows_are_appended_and_read_back_as_text(tmp_path: Path) -> None:
    path = tmp_path / "results" / "table.csv"
    assert read_rows(path, COLUMNS) == []  # no file yet
    append_row(path, COLUMNS, {"fold": "F1", "value": repr(0.1 + 0.2)})
    append_row(path, COLUMNS, {"fold": "F2", "value": "x,y"})  # a comma is quoted, not split
    rows = read_rows(path, COLUMNS)
    assert rows == [{"fold": "F1", "value": "0.30000000000000004"}, {"fold": "F2", "value": "x,y"}]
    # repr() text reads back as exactly the same float64.
    assert float(rows[0]["value"]) == 0.1 + 0.2
    # One header line and two rows: the first row was not rewritten.
    assert path.read_text(encoding="utf-8").splitlines()[0] == "fold,value"
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_other_columns_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    with pytest.raises(ValueError, match="row has columns"):
        append_row(path, COLUMNS, {"fold": "F1"})
    with pytest.raises(ValueError, match="row has columns"):
        append_row(path, COLUMNS, {"value": "1", "fold": "F1"})  # wrong order
    assert not path.exists()  # nothing was written
    append_row(path, COLUMNS, {"fold": "F1", "value": "1"})
    with pytest.raises(ValueError, match="has columns"):
        read_rows(path, ("fold", "value", "extra"))
    with pytest.raises(ValueError, match="has columns"):
        append_row(path, ("fold", "other"), {"fold": "F1", "other": "1"})
