"""How the weekly pipeline gets the labels of a week it closes (docs/mlops.md §6.1, §4).

A tick at anchor `a` closes the week before `a`. Under the simulated clock that
week is over, so its purchases are known: step C2 builds the week's label sets
from the transactions the tick ingested (every row is before `a`) and reads them
back through this module. The rule checked here is the clock itself:

    the labels of a week may be read only by a tick whose anchor is after the
    week's last day.

The replay runs over V1-V4 AND T1-T4. T labels are read by the replay only
(the closing steps, the `eval`-mode query list and the comparison with the M8
records), never to select or tune anything: docs/mlops.md invariant 3, approved
by the user on 2026-11-21 (decision D1). Every tick that reads the labels of a
week records it first in results/m12_replay/label_access.csv (the driver does).

The stage-1 label files (artifacts/labels/) are NOT opened by the pipeline: this
module is given a file of the pipeline's own state. The entry points of M6, M7,
M8, the diagnostics and the M9 evaluation are untouched.
"""

from datetime import date
from pathlib import Path

from hmrec.data.folds import Fold
from hmrec.eval.report import FoldEvaluator


def require_week_is_over(week: Fold, tick_anchor: date) -> None:
    """Fail unless the week ended before the tick's anchor.

    Raises:
        ValueError: If the tick's anchor is on or before the week's last day:
            under the tick's clock the week's purchases do not exist yet.
    """
    if tick_anchor <= week.label_end:
        raise ValueError(
            f"the labels of week {week.name} ({week.anchor} .. {week.label_end}) cannot be read "
            f"by the tick at {tick_anchor}: the week is not over at that anchor"
        )


def load_closed_week_evaluator(week: Fold, tick_anchor: date, labels_file: Path) -> FoldEvaluator:
    """The evaluator of a week the tick closes. The clock is checked before the file is opened.

    Args:
        week: The closed week.
        tick_anchor: The anchor of the tick that reads the labels.
        labels_file: The week's label file in the PIPELINE's state (written by C2).

    Raises:
        ValueError: If the week is not over at the tick's anchor.
    """
    require_week_is_over(week, tick_anchor)
    return FoldEvaluator.from_file(week.name, labels_file)
