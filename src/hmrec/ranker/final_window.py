"""The training windows and fold guards of the final evaluation (M8, proposal §9.2, §10).

The final evaluation predicts the T folds. A model that predicts a fold is
trained from scratch on the `window_weeks` folds right before it:

    T1 <- V1 V2 V3 V4      T2 <- V2 V3 V4 T1      T3 <- V3 V4 T1 T2      T4 <- V4 T1 T2 T3

So the labels of T1, T2 and T3 are also TRAINING labels (of the models that
predict later weeks); every training fold ends before the predicted week
starts, so no label of the predicted week or later is used.

The guards here belong to M8 only. The M7 guards (hmrec.ranker.window) are
unchanged: an M7 entry still refuses every T fold. The rehearsal of M8 predicts
a V fold and uses the M7 window function, so a rehearsal can never reach a T
fold either.
"""

from collections.abc import Sequence

from hmrec.data.folds import Fold, FoldManifest
from hmrec.ranker.window import m7_training_window, training_folds

FINAL_TARGET_PHASE = "T"
FINAL_TRAINING_PHASES = ("V", "T")


def require_final_target_fold(fold: Fold) -> None:
    """Fail unless the final evaluation may predict and evaluate this fold (T folds only)."""
    if fold.phase != FINAL_TARGET_PHASE:
        raise ValueError(
            f"the final evaluation predicts only {FINAL_TARGET_PHASE} folds, got fold "
            f"{fold.name} (phase {fold.phase})"
        )


def require_final_training_fold(fold: Fold) -> None:
    """Fail unless a model of the final evaluation may train on this fold (V and T folds)."""
    if fold.phase not in FINAL_TRAINING_PHASES:
        raise ValueError(
            f"the final evaluation trains only on folds of phases {FINAL_TRAINING_PHASES}, "
            f"got fold {fold.name} (phase {fold.phase})"
        )


def final_training_window(
    manifest: FoldManifest, target_fold: str, window_weeks: int
) -> tuple[Fold, ...]:
    """The training folds of the model that predicts a T fold, after checking every fold.

    Args:
        manifest: All folds in time order (configs/fold_manifest.yaml).
        target_fold: Name of the T fold to predict.
        window_weeks: Number of training weeks (4, from the Ranker lock).

    Returns:
        The `window_weeks` folds right before the target, in time order.

    Raises:
        ValueError: If the target is not a T fold or a training fold is not a V / T fold.
    """
    require_final_target_fold(manifest.get(target_fold))
    folds = training_folds(manifest, target_fold, window_weeks)
    for fold in folds:
        require_final_training_fold(fold)
    return folds


def final_sampling_folds(
    manifest: FoldManifest, target_folds: Sequence[str], window_weeks: int
) -> tuple[Fold, ...]:
    """Every fold that trains at least one of the target folds' models, in time order.

    For the four T folds these are V1 V2 V3 V4 and the first three T folds. The
    last T fold is not among them: it trains no model.
    """
    needed: set[str] = set()
    for target_fold in target_folds:
        window = final_training_window(manifest, target_fold, window_weeks)
        needed.update(fold.name for fold in window)
    return tuple(fold for fold in manifest.folds if fold.name in needed)


def rehearsal_training_window(
    manifest: FoldManifest, target_fold: str, rehearsal_fold: str, window_weeks: int
) -> tuple[Fold, ...]:
    """The training folds of the rehearsal model: the M7 window of the rehearsal fold.

    The rehearsal runs the M8 code path on ONE V fold (configs/m8_final.yaml) and
    must reproduce the recorded V-phase result. It uses the M7 window function, so
    the target must be a V fold and every training fold an F or V fold.

    Raises:
        ValueError: If `target_fold` is not the configured rehearsal fold, or the
            M7 fold check fails.
    """
    if target_fold != rehearsal_fold:
        raise ValueError(
            f"a rehearsal predicts only the configured rehearsal fold {rehearsal_fold}, "
            f"got {target_fold}"
        )
    return m7_training_window(manifest, target_fold, window_weeks)
