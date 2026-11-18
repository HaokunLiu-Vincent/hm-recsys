"""Tests for hmrec.diagnostics.records and hmrec.diagnostics.decomposition (table A)."""

from pathlib import Path

import pytest

from hmrec.csv_records import read_rows
from hmrec.diagnostics.decomposition import (
    decomposition_columns,
    decomposition_row,
    recall_over_union,
)
from hmrec.diagnostics.recorded import RecordedFold
from hmrec.diagnostics.records import PROVENANCE_COLUMNS, append_rows, float_text


def test_float_text_reads_back_exactly_and_none_is_an_empty_cell() -> None:
    assert float(float_text(0.1 + 0.2)) == 0.1 + 0.2
    assert float_text(None) == ""


def test_append_rows_adds_the_provenance_columns_and_never_changes_rows(tmp_path: Path) -> None:
    path = tmp_path / "t_diag_example.csv"
    append_rows(path, ("fold", "value"), [{"fold": "V1", "value": "1"}], "commit_a", "run_1")
    append_rows(path, ("fold", "value"), [{"fold": "V1", "value": "2"}], "commit_b", "run_2")
    assert read_rows(path, ("fold", "value", *PROVENANCE_COLUMNS)) == [
        {"fold": "V1", "value": "1", "git_commit": "commit_a", "run_started_at": "run_1"},
        {"fold": "V1", "value": "2", "git_commit": "commit_b", "run_started_at": "run_2"},
    ]
    with pytest.raises(ValueError, match="columns"):
        append_rows(path, ("fold",), [{"fold": "V1"}], "commit_c", "run_3")


def test_decomposition_row_by_hand() -> None:
    recorded = RecordedFold(
        fold="T1",
        phase="T",
        n_queries=80,
        union_recall=0.25,
        ranker={"map@12": 0.04, "recall@12": 0.1, "ndcg@12": 0.06},
        baseline={"map@12": 0.02, "recall@12": 0.05, "ndcg@12": 0.03},
    )
    row = decomposition_row(recorded, "map@12", "recall@12")
    assert list(row) == list(decomposition_columns("map@12", "recall@12"))
    # 0.1 / 0.25 = 0.4 of the found labels reach the Ranker's Top-12; 0.05 / 0.25 = 0.2.
    assert row == {
        "fold": "T1",
        "phase": "T",
        "n_queries": "80",
        "union_recall": "0.25",
        "lambdamart_recall@12": "0.1",
        "lambdamart_recall@12_over_union": "0.4",
        "lambdamart_map@12": "0.04",
        "rrf_baseline_recall@12": "0.05",
        "rrf_baseline_recall@12_over_union": "0.2",
        "rrf_baseline_map@12": "0.02",
    }


def test_recall_over_union_needs_a_positive_union_recall() -> None:
    assert recall_over_union(0.1, 0.25) == 0.4
    with pytest.raises(ValueError, match="union recall must be > 0"):
        recall_over_union(0.0, 0.0)
