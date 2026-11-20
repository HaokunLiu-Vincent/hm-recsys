"""The label door of the M9 evaluation on T1-T4 (proposal §16.12, AGENTS.md §3.2).

The labels of the T folds were read by the formal run of M8 (hmrec.eval.final_eval)
and by the descriptive diagnostics after it (hmrec.eval.diagnostic_labels). Both
doors stay as they are. This module is a THIRD door, used only by the single
evaluation on T1-T4 of the DeepFM configuration chosen on V (hmrec.t_eval, scripts/m9_t_*.py):

    load_t_eval_evaluator(fold, labels_file, permit)          the labels of a T fold
    load_t_eval_rehearsal_evaluator(fold, labels_file, name)  the labels of the rehearsal fold

For a T fold it needs a TEvalPermit. Only hmrec.t_eval.gate creates one, after it
has checked that the rehearsal passed at the commit of HEAD on a clean working
tree and that a row in results/m9_t_label_access.csv (time, commit, purpose,
script) was written BEFORE the permit exists. The rehearsal reads no T label: its
loader gives the labels of the configured rehearsal fold only, and that fold
must be a V fold.

The evaluation is run once and reported only. Nothing that comes out of this
module may be used to select, tune or change anything
(tests/unit/test_t_eval_code_rules.py checks who imports it).
"""

from dataclasses import dataclass
from pathlib import Path

from hmrec.data.folds import Fold
from hmrec.eval.report import FoldEvaluator

T_EVAL_PHASE = "T"
T_EVAL_REHEARSAL_PHASE = "V"


@dataclass(frozen=True)
class TEvalPermit:
    """Proof that this process may read the labels of T folds for the M9 evaluation.

    Attributes:
        git_commit: HEAD of this process.
        rehearsal_checked_at: When the passed rehearsal of this commit was compared.
        recorded_at: The time of the row in results/m9_t_label_access.csv.
        purpose: The recorded purpose.
        script: The recorded script.
    """

    git_commit: str
    rehearsal_checked_at: str
    recorded_at: str
    purpose: str
    script: str


def require_t_eval_permit(permit: TEvalPermit | None) -> TEvalPermit:
    """Fail unless `permit` really is a TEvalPermit (not None, not another object).

    Raises:
        TypeError: Otherwise.
    """
    if not isinstance(permit, TEvalPermit):
        raise TypeError(
            "labels of a T fold are read by the M9 evaluation only with a TEvalPermit from "
            f"hmrec.t_eval.gate (the rehearsal must have passed), got {type(permit).__name__}"
        )
    return permit


def load_t_eval_evaluator(
    fold: Fold, labels_file: Path, permit: TEvalPermit | None
) -> FoldEvaluator:
    """The evaluator of a T fold. Permit and phase are checked before the file is opened.

    Raises:
        TypeError: If no TEvalPermit is given.
        ValueError: If the fold is not a T fold.
    """
    require_t_eval_permit(permit)
    if fold.phase != T_EVAL_PHASE:
        raise ValueError(
            f"the M9 evaluation on T evaluates only {T_EVAL_PHASE} folds, got fold "
            f"{fold.name} (phase {fold.phase})"
        )
    return FoldEvaluator.from_file(fold.name, labels_file)


def load_t_eval_rehearsal_evaluator(
    fold: Fold, labels_file: Path, rehearsal_fold: str
) -> FoldEvaluator:
    """The evaluator of the rehearsal fold. Name and phase are checked before the file is opened.

    Raises:
        ValueError: If the fold is not the configured rehearsal fold or not a V fold.
    """
    if fold.name != rehearsal_fold or fold.phase != T_EVAL_REHEARSAL_PHASE:
        raise ValueError(
            f"a rehearsal evaluates only the configured rehearsal fold {rehearsal_fold} "
            f"(a {T_EVAL_REHEARSAL_PHASE} fold), got fold {fold.name} (phase {fold.phase}); "
            "its labels are not read"
        )
    return FoldEvaluator.from_file(fold.name, labels_file)
