"""Tests for hmrec.ranker.final_window: the training windows and fold guards of M8."""

from pathlib import Path

import pytest

from hmrec.data.folds import load_fold_manifest
from hmrec.ranker.final_window import (
    final_sampling_folds,
    final_training_window,
    rehearsal_training_window,
    require_final_target_fold,
    require_final_training_fold,
)
from hmrec.ranker.window import m7_training_window

REPO = Path(__file__).resolve().parents[2]
MANIFEST = load_fold_manifest(REPO / "configs" / "fold_manifest.yaml")
T_FOLDS = [fold.name for fold in MANIFEST.by_phase("T")]
V_FOLDS = [fold.name for fold in MANIFEST.by_phase("V")]


def _names(folds: tuple) -> list[str]:
    return [fold.name for fold in folds]


def test_the_four_windows_of_the_final_evaluation() -> None:
    # Every T fold is trained on the 4 folds right before it (proposal §9.2, §10).
    t1, t2, t3, t4 = T_FOLDS
    v1, v2, v3, v4 = V_FOLDS
    assert _names(final_training_window(MANIFEST, t1, 4)) == [v1, v2, v3, v4]
    assert _names(final_training_window(MANIFEST, t2, 4)) == [v2, v3, v4, t1]
    assert _names(final_training_window(MANIFEST, t3, 4)) == [v3, v4, t1, t2]
    assert _names(final_training_window(MANIFEST, t4, 4)) == [v4, t1, t2, t3]


def test_every_training_fold_ends_before_the_predicted_week() -> None:
    for target in T_FOLDS:
        anchor = MANIFEST.get(target).anchor
        for fold in final_training_window(MANIFEST, target, 4):
            assert fold.label_end < anchor  # its label week is complete, nothing leaks


def test_only_t_folds_are_predicted() -> None:
    for name in (*V_FOLDS, "F1", "D4"):
        with pytest.raises(ValueError, match="predicts only T folds"):
            final_training_window(MANIFEST, name, 4)
        with pytest.raises(ValueError, match="predicts only T folds"):
            require_final_target_fold(MANIFEST.get(name))
    require_final_target_fold(MANIFEST.get(T_FOLDS[0]))  # no error


def test_only_v_and_t_folds_train_a_final_model() -> None:
    require_final_training_fold(MANIFEST.get(V_FOLDS[0]))
    require_final_training_fold(MANIFEST.get(T_FOLDS[0]))
    for name in ("F4", "D1"):
        with pytest.raises(ValueError, match="trains only on folds of phases"):
            require_final_training_fold(MANIFEST.get(name))
    # A window of 5 weeks would reach back to F4 for the first T fold: refused.
    with pytest.raises(ValueError, match="trains only on folds of phases"):
        final_training_window(MANIFEST, T_FOLDS[0], 5)


def test_sampling_folds_are_the_four_v_folds_and_the_first_three_t_folds() -> None:
    folds = final_sampling_folds(MANIFEST, T_FOLDS, 4)
    assert _names(folds) == [*V_FOLDS, *T_FOLDS[:3]]
    # The last T fold trains no model, so its labels are never training labels.
    assert T_FOLDS[3] not in _names(folds)
    # One target only: just its own window.
    assert _names(final_sampling_folds(MANIFEST, T_FOLDS[:1], 4)) == V_FOLDS


def test_the_rehearsal_window_is_the_m7_window_of_the_rehearsal_fold() -> None:
    rehearsal_fold = V_FOLDS[3]
    window = rehearsal_training_window(MANIFEST, rehearsal_fold, rehearsal_fold, 4)
    assert _names(window) == ["F4", *V_FOLDS[:3]]
    assert window == m7_training_window(MANIFEST, rehearsal_fold, 4)
    assert all(fold.phase in ("F", "V") for fold in window)  # no T fold can be reached


def test_a_rehearsal_predicts_only_the_configured_fold() -> None:
    rehearsal_fold = V_FOLDS[3]
    # Another V fold: not the configured one.
    with pytest.raises(ValueError, match="only the configured rehearsal fold"):
        rehearsal_training_window(MANIFEST, V_FOLDS[0], rehearsal_fold, 4)
    # A T fold can never be rehearsed, neither as "another fold" ...
    with pytest.raises(ValueError, match="only the configured rehearsal fold"):
        rehearsal_training_window(MANIFEST, T_FOLDS[0], rehearsal_fold, 4)
    # ... nor by configuring it as the rehearsal fold: the M7 guard refuses it.
    with pytest.raises(ValueError, match="M7 predicts and evaluates only V folds"):
        rehearsal_training_window(MANIFEST, T_FOLDS[0], T_FOLDS[0], 4)
