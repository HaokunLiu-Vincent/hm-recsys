"""Tests for hmrec.eval.diagnostic_labels: the diagnostics' label door and the label counts."""

from pathlib import Path

import polars as pl
import pytest
from diag_example import COMMIT, FORMAL_COMMIT, MANIFEST, NOW, build_diag_workspace

from hmrec.eval.diagnostic_labels import (
    DiagnosticAccess,
    label_pair_counts,
    load_diagnostic_evaluator,
)
from hmrec.eval.final_eval import FormalRunPermit

ACCESS = DiagnosticAccess(COMMIT, FORMAL_COMMIT, NOW, "descriptive diagnostics")


def test_a_t_fold_is_refused_without_access_before_the_file_is_opened(tmp_path: Path) -> None:
    # The file does not exist: a FileNotFoundError would mean the door tried to open it.
    missing = tmp_path / "labels" / "T1.parquet"
    permit = FormalRunPermit(COMMIT, COMMIT, NOW, NOW)  # the M8 permit is not an access
    for not_an_access in (None, "yes", permit):
        with pytest.raises(TypeError, match="DiagnosticAccess"):
            load_diagnostic_evaluator(MANIFEST.get("T1"), missing, not_an_access)  # type: ignore[arg-type]  # wrong type on purpose


def test_a_t_fold_opens_with_access_and_other_phases_need_none(tmp_path: Path) -> None:
    paths = build_diag_workspace(tmp_path).paths
    t_fold = load_diagnostic_evaluator(MANIFEST.get("T1"), paths.labels_file("T1"), ACCESS)
    assert (t_fold.fold, t_fold.n_queries) == ("T1", 4)
    for name in ("D1", "F1", "V1"):
        evaluator = load_diagnostic_evaluator(MANIFEST.get(name), paths.labels_file(name), None)
        assert (evaluator.fold, evaluator.n_queries) == (name, 4)


def test_label_pair_counts_counts_pairs_per_article(tmp_path: Path) -> None:
    paths = build_diag_workspace(tmp_path).paths
    evaluator = load_diagnostic_evaluator(MANIFEST.get("V1"), paths.labels_file("V1"), None)
    counts = label_pair_counts(evaluator)
    # Labels: q0 {1, 2}, q1 {1}, q2 {4, 8}, q3 {5} -> article 1 twice, the others once.
    assert counts.schema == {"article_idx": pl.Int32, "n_label_pairs": pl.Int64}
    assert counts.rows() == [(1, 2), (2, 1), (4, 1), (5, 1), (8, 1)]
