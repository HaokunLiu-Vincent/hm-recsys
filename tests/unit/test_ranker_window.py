"""Tests for hmrec.ranker.window: the 4-week training window and the M7 fold guard."""

from datetime import date

import pytest
from ranker_example import MANIFEST

from hmrec.data.folds import Fold, FoldManifest
from hmrec.ranker.window import (
    m7_sampling_folds,
    m7_training_window,
    require_m7_target_fold,
    require_m7_training_fold,
    training_folds,
)


def _names(folds: tuple[Fold, ...]) -> list[str]:
    return [fold.name for fold in folds]


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        # proposal §9.2: the 4 folds right before the predicted fold.
        ("V1", ["F1", "F2", "F3", "F4"]),
        ("V2", ["F2", "F3", "F4", "V1"]),
        ("V3", ["F3", "F4", "V1", "V2"]),
        ("V4", ["F4", "V1", "V2", "V3"]),
    ],
)
def test_training_window_of_every_v_fold(target: str, expected: list[str]) -> None:
    assert _names(m7_training_window(MANIFEST, target, 4)) == expected


def test_training_folds_all_end_before_the_target_anchor() -> None:
    target = MANIFEST.get("V2")
    for fold in training_folds(MANIFEST, "V2", 4):
        assert fold.label_end < target.anchor


@pytest.mark.parametrize("target", ["T1", "T2", "T3", "T4"])
def test_a_t_fold_is_refused_as_target(target: str) -> None:
    # training_folds itself would return V / T folds for a T target (M8 needs that),
    # but the M7 entry refuses the target before anything else.
    with pytest.raises(ValueError, match="M7 predicts and evaluates only V folds"):
        m7_training_window(MANIFEST, target, 4)


def test_f_and_d_folds_are_refused_as_target() -> None:
    for target in ("F4", "D4"):
        with pytest.raises(ValueError, match="only V folds"):
            m7_training_window(MANIFEST, target, 4)


def test_a_t_fold_is_refused_as_training_fold() -> None:
    with pytest.raises(ValueError, match="M7 trains only on folds of phases"):
        require_m7_training_fold(MANIFEST.get("T1"))
    with pytest.raises(ValueError, match="M7 trains only on folds of phases"):
        require_m7_training_fold(MANIFEST.get("D4"))
    require_m7_training_fold(MANIFEST.get("F1"))
    require_m7_training_fold(MANIFEST.get("V3"))
    require_m7_target_fold(MANIFEST.get("V4"))


def test_a_window_that_reaches_into_d_folds_is_refused() -> None:
    # With 5 weeks, V1 would train on D4: D folds never train the Ranker (proposal §3).
    with pytest.raises(ValueError, match="got fold D4"):
        m7_training_window(MANIFEST, "V1", 5)


def test_not_enough_folds_before_the_target() -> None:
    week = [
        Fold("V1", "V", date(2020, 1, 8), date(2020, 1, 14)),
        Fold("V2", "V", date(2020, 1, 15), date(2020, 1, 21)),
    ]
    with pytest.raises(ValueError, match="has only 1 folds before it"):
        training_folds(FoldManifest(tuple(week)), "V2", 4)
    with pytest.raises(ValueError, match="unknown fold"):
        training_folds(MANIFEST, "X9", 4)


def test_m7_samples_f1_to_v3_and_never_v4_or_t() -> None:
    folds = m7_sampling_folds(MANIFEST, ("V1", "V2", "V3", "V4"), 4)
    # V4 trains only models that predict T folds (M8), so M7 does not sample it.
    assert _names(folds) == ["F1", "F2", "F3", "F4", "V1", "V2", "V3"]
