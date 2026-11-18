"""The label door of the descriptive diagnostics after M8 (AGENTS.md §3.2).

The formal run of M8 read the labels of the T folds once, through
hmrec.eval.final_eval. That door stays as it is and stays closed for any code
that is not the code of the formal run. This module is a SECOND door, used only
by the descriptive diagnostics (hmrec.diagnostics, scripts/m8_diagnostics.py):

    load_diagnostic_evaluator(fold, labels_file, access)

For a T fold it needs a DiagnosticAccess. Only hmrec.diagnostics.access creates
one, after it has checked that the formal run is finished (the verdict on G1-G3
is recorded, so reading the labels again cannot change it), that the working
tree is clean and the M8 code and configs are unchanged, and after it has
appended a row to results/t_diag_label_access.csv (time, git commit, purpose).
The labels of D, F and V folds were read many times before M8 and need no access.

What is read here is used to DESCRIBE recorded results. Nothing that comes out
of this module may be used to select, tune or change anything
(tests/unit/test_diag_code_rules.py checks who imports it).
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hmrec.data.folds import Fold
from hmrec.eval.metrics import ARTICLE_IDX
from hmrec.eval.report import FoldEvaluator

RECORDED_PHASE = "T"  # the phase whose labels are read only with a recorded access
N_LABEL_PAIRS = "n_label_pairs"


@dataclass(frozen=True)
class DiagnosticAccess:
    """Proof that this read of T labels is recorded and is for description only.

    Attributes:
        git_commit: HEAD of this process.
        formal_run_commit: The commit of the finished formal run of M8.
        recorded_at: The time written to results/t_diag_label_access.csv.
        purpose: The recorded purpose, "descriptive diagnostics".
    """

    git_commit: str
    formal_run_commit: str
    recorded_at: str
    purpose: str


def load_diagnostic_evaluator(
    fold: Fold, labels_file: Path, access: DiagnosticAccess | None
) -> FoldEvaluator:
    """The evaluator of a fold, for descriptive diagnostics only.

    The phase and the access are checked BEFORE the file is opened.

    Args:
        fold: Any fold of the manifest.
        labels_file: artifacts/labels/{fold}.parquet.
        access: Required for a T fold; ignored for D, F and V folds.

    Raises:
        TypeError: If the fold is a T fold and `access` is not a DiagnosticAccess.
    """
    if fold.phase == RECORDED_PHASE and not isinstance(access, DiagnosticAccess):
        raise TypeError(
            f"fold {fold.name} is a T fold: the diagnostics read its labels only with a "
            "DiagnosticAccess from hmrec.diagnostics.access (the read must be recorded first), "
            f"got {type(access).__name__}"
        )
    return FoldEvaluator.from_file(fold.name, labels_file)


def label_pair_counts(evaluator: FoldEvaluator) -> pl.DataFrame:
    """How many label pairs each article has in the fold's label week.

    A label pair is one (query, article): a customer who bought the article in
    the label week, counted once however many pieces they bought (G_q is a
    set, hmrec.data.queries). Only catalog articles have labels.

    Returns:
        One row per article with at least one label. Columns: article_idx
        (int32), n_label_pairs (int64); sorted by article_idx.
    """
    counts = evaluator.labels.group_by(ARTICLE_IDX).agg(
        pl.len().cast(pl.Int64).alias(N_LABEL_PAIRS)
    )
    return counts.sort(ARTICLE_IDX)
