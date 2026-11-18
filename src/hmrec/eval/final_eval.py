"""The ONLY way the labels of a T fold are read (M8, the final evaluation; proposal §10).

Until M8 no label of a T fold was read: the M6 monitor and every M7 entry refuse
a T fold (hmrec.eval.monitor.require_label_phase, hmrec.eval.ranker_eval,
hmrec.ranker.window). Those guards are unchanged. This module is the one door
for T labels, and it opens only with a FormalRunPermit:

    load_final_evaluator(fold, labels_file, permit)   labels of a T fold, to evaluate it
    final_training_label_phases(fold, permit)         the phases a sampling job may read
                                                      when a T fold is a TRAINING fold

A FormalRunPermit is proof that (1) the rehearsal passed with the code that is
running now and (2) the start of the formal run was recorded ("first read of a T
label": time and git commit) BEFORE any T label was opened. Only
hmrec.final.gate creates one, after checking both
(tests/unit/test_final_code_rules.py scans the code for other places).

The rehearsal itself reads no T label: load_rehearsal_evaluator gives the
labels of the configured rehearsal fold only, and that fold must be a V fold.
"""

from dataclasses import dataclass
from pathlib import Path

from hmrec.data.folds import Fold
from hmrec.eval.report import FoldEvaluator

FINAL_EVALUATION_PHASE = "T"
REHEARSAL_PHASE = "V"


@dataclass(frozen=True)
class FormalRunPermit:
    """Proof that this process may read the labels of T folds (module docstring).

    Attributes:
        git_commit: HEAD of this process.
        rehearsal_commit: The commit the passed rehearsal ran at (same code as HEAD).
        rehearsal_checked_at: When that rehearsal was compared with the M7 records.
        first_label_read_at: The recorded start of the formal run.
    """

    git_commit: str
    rehearsal_commit: str
    rehearsal_checked_at: str
    first_label_read_at: str


def _require_permit(permit: FormalRunPermit) -> None:
    """Fail unless `permit` really is a FormalRunPermit (not None, not another object)."""
    if not isinstance(permit, FormalRunPermit):
        raise TypeError(
            "labels of a T fold are read only with a FormalRunPermit from hmrec.final.gate "
            f"(the rehearsal must have passed), got {type(permit).__name__}"
        )


def load_final_evaluator(fold: Fold, labels_file: Path, permit: FormalRunPermit) -> FoldEvaluator:
    """The evaluator of a T fold. Permit and phase are checked before the file is opened.

    Raises:
        TypeError: If no FormalRunPermit is given.
        ValueError: If the fold is not a T fold.
    """
    _require_permit(permit)
    if fold.phase != FINAL_EVALUATION_PHASE:
        raise ValueError(
            f"the final evaluation evaluates only {FINAL_EVALUATION_PHASE} folds, got fold "
            f"{fold.name} (phase {fold.phase})"
        )
    return FoldEvaluator.from_file(fold.name, labels_file)


def final_training_label_phases(fold: Fold, permit: FormalRunPermit | None) -> tuple[str, ...]:
    """The phases whose labels a sampling job of the final evaluation may read for `fold`.

    The result is passed to hmrec.ranker.sampling as `allowed_phases`, which checks
    the fold's phase before it opens the label file.

    Args:
        fold: The training fold to sample (a V or T fold).
        permit: Required for a T fold; not needed for a V fold (V labels were
            already read in M7).

    Returns:
        ("V",) for a V fold, ("T",) for a T fold with a permit.

    Raises:
        TypeError: If the fold is a T fold and no FormalRunPermit is given.
        ValueError: If the fold is neither a V nor a T fold.
    """
    if fold.phase == REHEARSAL_PHASE:
        return (REHEARSAL_PHASE,)
    if fold.phase != FINAL_EVALUATION_PHASE:
        raise ValueError(
            f"the final evaluation samples only V and T folds, got fold {fold.name} "
            f"(phase {fold.phase})"
        )
    if permit is None:
        raise TypeError(
            f"fold {fold.name} is a T fold: its labels are read only with a FormalRunPermit"
        )
    _require_permit(permit)
    return (FINAL_EVALUATION_PHASE,)


def load_rehearsal_evaluator(fold: Fold, labels_file: Path, rehearsal_fold: str) -> FoldEvaluator:
    """The evaluator of the rehearsal fold. Name and phase are checked before the file is opened.

    Raises:
        ValueError: If the fold is not the configured rehearsal fold or not a V fold.
    """
    if fold.name != rehearsal_fold or fold.phase != REHEARSAL_PHASE:
        raise ValueError(
            f"a rehearsal evaluates only the configured rehearsal fold {rehearsal_fold} "
            f"(a {REHEARSAL_PHASE} fold), got fold {fold.name} (phase {fold.phase}); "
            "its labels are not read"
        )
    return FoldEvaluator.from_file(fold.name, labels_file)
