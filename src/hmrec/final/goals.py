"""The verdict on the goals G1-G3 of the final evaluation (proposal §0.1, §10).

Pure functions: numbers in, verdict out. The thresholds come from
configs/m8_final.yaml, which was committed before any label of a T fold was
read. A goal is "met" or "NOT met"; there is nothing in between.

    G1  pooled MAP@12 of the Ranker on the target folds >= g1_min_pooled_map
    G2  union recall of the six channels >= g2_min_union_recall on EVERY target fold
    G3  the Ranker's MAP@12 is STRICTLY above the RRF baseline's on at least
        g3_min_folds_above_baseline target folds
"""

from collections.abc import Mapping
from dataclasses import dataclass

GOAL_COLUMNS: tuple[str, ...] = (
    "goal",
    "criterion",
    "value",
    "threshold",
    "met",
    "detail",
    "git_commit",
    "decided_at",
)


@dataclass(frozen=True)
class GoalVerdict:
    """The verdict on one goal.

    Attributes:
        goal: "G1", "G2" or "G3".
        criterion: The rule in words.
        value: The number the rule is applied to, as text (repr of a float, or a count).
        threshold: The threshold, as text.
        met: True = the goal is met.
        detail: The per-fold numbers behind the value.
    """

    goal: str
    criterion: str
    value: str
    threshold: str
    met: bool
    detail: str


def _per_fold_text(values: Mapping[str, float]) -> str:
    """Fold values as one line of text, e.g. T1=0.031 T2=0.033, with all digits (repr)."""
    return " ".join(f"{fold}={value!r}" for fold, value in values.items())


def judge_g1(pooled_map: float, map_by_fold: Mapping[str, float], threshold: float) -> GoalVerdict:
    """G1: the pooled (query-weighted) MAP@12 of the target folds is >= the threshold.

    Args:
        pooled_map: Mean MAP@12 over all queries of the target folds, float64.
        map_by_fold: Fold -> MAP@12 (for the record only; the verdict uses pooled_map).
        threshold: g1_min_pooled_map.
    """
    return GoalVerdict(
        goal="G1",
        criterion="pooled MAP@12 of the Ranker on the target folds >= threshold",
        value=repr(pooled_map),
        threshold=repr(threshold),
        met=pooled_map >= threshold,
        detail=_per_fold_text(map_by_fold),
    )


def judge_g2(union_recall_by_fold: Mapping[str, float], threshold: float) -> GoalVerdict:
    """G2: the union recall of the six channels is >= the threshold on EVERY target fold.

    Args:
        union_recall_by_fold: Fold -> union recall (mean over the fold's queries).
        threshold: g2_min_union_recall.

    Returns:
        value is the LOWEST fold's union recall: the goal is met exactly when it
        is >= the threshold.
    """
    if not union_recall_by_fold:
        raise ValueError("G2 needs the union recall of at least one fold")
    lowest = min(union_recall_by_fold.values())
    return GoalVerdict(
        goal="G2",
        criterion="union recall of the six channels >= threshold on every target fold",
        value=repr(lowest),
        threshold=repr(threshold),
        met=all(value >= threshold for value in union_recall_by_fold.values()),
        detail=_per_fold_text(union_recall_by_fold),
    )


def judge_g3(
    ranker_map_by_fold: Mapping[str, float],
    baseline_map_by_fold: Mapping[str, float],
    min_folds: int,
) -> GoalVerdict:
    """G3: the Ranker's MAP@12 is strictly above the RRF baseline's on >= min_folds folds.

    A tie on a fold does not count for the Ranker.

    Args:
        ranker_map_by_fold: Fold -> MAP@12 of the Ranker.
        baseline_map_by_fold: Fold -> MAP@12 of the RRF baseline; the same folds.
        min_folds: g3_min_folds_above_baseline.

    Raises:
        ValueError: If the two mappings do not have the same folds.
    """
    if list(ranker_map_by_fold) != list(baseline_map_by_fold):
        raise ValueError(
            f"G3 needs the same folds for both systems, got {list(ranker_map_by_fold)} and "
            f"{list(baseline_map_by_fold)}"
        )
    above = [
        fold for fold, score in ranker_map_by_fold.items() if score > baseline_map_by_fold[fold]
    ]
    pairs = " ".join(
        f"{fold}={ranker_map_by_fold[fold]!r}>{baseline_map_by_fold[fold]!r}"
        if fold in above
        else f"{fold}={ranker_map_by_fold[fold]!r}<={baseline_map_by_fold[fold]!r}"
        for fold in ranker_map_by_fold
    )
    return GoalVerdict(
        goal="G3",
        criterion="Ranker MAP@12 strictly above the RRF baseline on at least threshold folds",
        value=str(len(above)),
        threshold=str(min_folds),
        met=len(above) >= min_folds,
        detail=pairs,
    )
