"""The rolling training window and the fold guard of M7 (proposal §9.2, AGENTS.md §3.2).

A model that predicts a fold is trained on the `window_weeks` folds right before
it in configs/fold_manifest.yaml:

    V1 <- F1 F2 F3 F4      V2 <- F2 F3 F4 V1      V3 <- F3 F4 V1 V2      V4 <- F4 V1 V2 V3

M7 tunes the Ranker, so it works with F and V folds ONLY. Every M7 entry point
calls m7_training_window (or require_m7_training_fold) FIRST: a T fold, as the
predicted fold or as a training fold, raises before any file is opened. T labels
are read in M8 only, through its own entry point.
"""

from hmrec.data.folds import Fold, FoldManifest

M7_TARGET_PHASE = "V"
M7_TRAINING_PHASES = ("F", "V")


def training_folds(manifest: FoldManifest, target_fold: str, window_weeks: int) -> tuple[Fold, ...]:
    """The `window_weeks` folds right before `target_fold`, in time order.

    Args:
        manifest: All folds in time order (configs/fold_manifest.yaml).
        target_fold: Name of the fold to predict, e.g. "V2".
        window_weeks: Number of training weeks (4).

    Returns:
        E.g. (F2, F3, F4, V1) for V2. Every fold ends before the target's anchor,
        so its label week is complete when the target is predicted.

    Raises:
        ValueError: If the manifest has fewer than `window_weeks` folds before the target.
    """
    if window_weeks <= 0:
        raise ValueError(f"window_weeks must be > 0, got {window_weeks}")
    names = list(manifest.names)
    if target_fold not in names:
        raise ValueError(f"unknown fold {target_fold!r}; the manifest has {names}")
    position = names.index(target_fold)
    if position < window_weeks:
        raise ValueError(
            f"fold {target_fold} has only {position} folds before it, need {window_weeks}"
        )
    return tuple(manifest.folds[position - window_weeks : position])


def require_m7_target_fold(fold: Fold) -> None:
    """Fail unless M7 may predict and evaluate this fold (V folds only).

    Raises:
        ValueError: For any other phase. In particular a T fold: it is evaluated in M8.
    """
    if fold.phase != M7_TARGET_PHASE:
        raise ValueError(
            f"M7 predicts and evaluates only {M7_TARGET_PHASE} folds, got fold {fold.name} "
            f"(phase {fold.phase}); T folds are evaluated in M8"
        )


def require_m7_training_fold(fold: Fold) -> None:
    """Fail unless M7 may read this fold's labels to train (F and V folds only).

    Raises:
        ValueError: For any other phase. D folds never train the Ranker (proposal
            §3); T labels are not read in M7.
    """
    if fold.phase not in M7_TRAINING_PHASES:
        raise ValueError(
            f"M7 trains only on folds of phases {M7_TRAINING_PHASES}, got fold {fold.name} "
            f"(phase {fold.phase})"
        )


def m7_training_window(
    manifest: FoldManifest, target_fold: str, window_weeks: int
) -> tuple[Fold, ...]:
    """The training folds of an M7 model, after checking the target and every training fold.

    Raises:
        ValueError: If the target is not a V fold or a training fold is not an F / V fold.
    """
    require_m7_target_fold(manifest.get(target_fold))
    folds = training_folds(manifest, target_fold, window_weeks)
    for fold in folds:
        require_m7_training_fold(fold)
    return folds


def m7_sampling_folds(
    manifest: FoldManifest, target_folds: tuple[str, ...], window_weeks: int
) -> tuple[Fold, ...]:
    """Every fold that trains at least one of the target folds' models, in time order.

    For V1 ... V4 these are F1 F2 F3 F4 V1 V2 V3. V4 is not among them: it trains
    only models that predict T folds (M8).
    """
    needed: set[str] = set()
    for target_fold in target_folds:
        needed.update(fold.name for fold in m7_training_window(manifest, target_fold, window_weeks))
    return tuple(fold for fold in manifest.folds if fold.name in needed)
