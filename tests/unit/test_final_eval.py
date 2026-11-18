"""Tests for hmrec.eval.final_eval: T labels are read only through the M8 entry, with a permit.

Also here: the guards of M6 and M7 still refuse every T fold, exactly as before M8.
"""

from pathlib import Path

import pytest
from final_example import FOLD_QUERY_IDS
from ranker_example import write_labels

from hmrec.data.folds import load_fold_manifest
from hmrec.eval.final_eval import (
    FormalRunPermit,
    final_training_label_phases,
    load_final_evaluator,
    load_rehearsal_evaluator,
)
from hmrec.eval.monitor import load_label_evaluator, require_label_phase
from hmrec.eval.ranker_eval import load_m7_evaluator
from hmrec.ranker.sampling import load_training_labels
from hmrec.ranker.window import M7_TRAINING_PHASES, m7_training_window, require_m7_training_fold

REPO = Path(__file__).resolve().parents[2]
MANIFEST = load_fold_manifest(REPO / "configs" / "fold_manifest.yaml")
PERMIT = FormalRunPermit("a" * 40, "a" * 40, "2026-11-18T10:00:00", "2026-11-18T11:00:00")
T1, V4, F4 = MANIFEST.get("T1"), MANIFEST.get("V4"), MANIFEST.get("F4")


def _labels(tmp_path: Path, fold: str) -> Path:
    path = tmp_path / f"{fold}.parquet"
    write_labels(path, fold, [(FOLD_QUERY_IDS[fold], 3), (FOLD_QUERY_IDS[fold] + 1, 5)])
    return path


def test_t_labels_are_read_with_a_permit(tmp_path: Path) -> None:
    evaluator = load_final_evaluator(T1, _labels(tmp_path, "T1"), PERMIT)
    assert evaluator.fold == "T1" and evaluator.n_queries == 2


@pytest.mark.parametrize("not_a_permit", [None, True, "permit", {"git_commit": "a" * 40}])
def test_without_a_permit_no_t_label_file_is_opened(tmp_path: Path, not_a_permit: object) -> None:
    # The file does not exist: if the loader tried to open it, the error would be
    # FileNotFoundError, not the TypeError of the permit check.
    missing = tmp_path / "not-written.parquet"
    with pytest.raises(TypeError, match="only with a FormalRunPermit"):
        load_final_evaluator(T1, missing, not_a_permit)  # type: ignore[arg-type]  # the test


def test_the_final_loader_refuses_folds_that_are_not_t_folds(tmp_path: Path) -> None:
    missing = tmp_path / "not-written.parquet"
    for fold in (V4, F4, MANIFEST.get("D1")):
        with pytest.raises(ValueError, match="evaluates only T folds"):
            load_final_evaluator(fold, missing, PERMIT)


def test_training_labels_of_a_t_fold_need_a_permit() -> None:
    assert final_training_label_phases(V4, None) == ("V",)  # V labels were read in M7 already
    assert final_training_label_phases(V4, PERMIT) == ("V",)
    assert final_training_label_phases(T1, PERMIT) == ("T",)
    with pytest.raises(TypeError, match="only with a FormalRunPermit"):
        final_training_label_phases(T1, None)
    with pytest.raises(TypeError, match="only with a FormalRunPermit"):
        final_training_label_phases(T1, "permit")  # type: ignore[arg-type]  # the test
    with pytest.raises(ValueError, match="samples only V and T folds"):
        final_training_label_phases(F4, PERMIT)


def test_the_phases_of_a_v_fold_do_not_open_a_t_label_file(tmp_path: Path) -> None:
    # The sampling code checks the fold against the allowed phases BEFORE it opens the
    # file: with the phases of a V fold, a T fold's (missing) file is never touched.
    allowed = final_training_label_phases(V4, None)
    with pytest.raises(ValueError, match="must not be read here"):
        load_training_labels(T1, tmp_path / "not-written.parquet", allowed)


def test_the_rehearsal_loader_gives_only_the_configured_v_fold(tmp_path: Path) -> None:
    evaluator = load_rehearsal_evaluator(V4, _labels(tmp_path, "V4"), "V4")
    assert evaluator.fold == "V4"
    missing = tmp_path / "not-written.parquet"
    # Another V fold, a T fold, and a T fold configured as the rehearsal fold: all refused
    # before the file is opened.
    for fold, configured in ((MANIFEST.get("V3"), "V4"), (T1, "V4"), (T1, "T1")):
        with pytest.raises(ValueError, match="its labels are not read"):
            load_rehearsal_evaluator(fold, missing, configured)


# ---------------------------------------------------------------- M6 and M7 are unchanged


def test_the_m7_entries_still_refuse_every_t_fold(tmp_path: Path) -> None:
    missing = tmp_path / "not-written.parquet"
    for name in ("T1", "T2", "T3", "T4"):
        fold = MANIFEST.get(name)
        with pytest.raises(ValueError, match="M7 evaluates only V folds"):
            load_m7_evaluator(fold, missing)
        with pytest.raises(ValueError, match="M7 predicts and evaluates only V folds"):
            m7_training_window(MANIFEST, name, 4)
        with pytest.raises(ValueError, match="M7 trains only on folds of phases"):
            require_m7_training_fold(fold)
        with pytest.raises(ValueError, match="must not be read here"):
            load_training_labels(fold, missing, M7_TRAINING_PHASES)


def test_the_m6_monitor_still_refuses_every_t_fold(tmp_path: Path) -> None:
    missing = tmp_path / "not-written.parquet"
    for name in ("T1", "T2", "T3", "T4"):
        with pytest.raises(ValueError, match="M6 must not read the labels"):
            require_label_phase(name, "T")
        with pytest.raises(ValueError, match="M6 must not read the labels"):
            load_label_evaluator(name, "T", missing)
