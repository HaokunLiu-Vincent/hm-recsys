"""The §4 selection rule (proposal §4.1-4.2, configs/selection_rule.yaml).

One candidate against the default (§4.2): the candidate replaces the default
only if BOTH hold:
    1. (candidate - default) / default >= min_relative_improvement (1%),
       on the primary (query-weighted, pooled) score;
    2. the candidate is STRICTLY better on at least min_folds_better (3) of
       n_folds (4) folds. Equal scores do not count as better.

Several candidates in one step (proposal §4.2, docs/decisions.md 2026-11-17):
    a. check every candidate against the current default with the rule above;
    b. none passes -> keep the default;
    c. otherwise, among the passing candidates, those whose pooled score is
       less than 1% (relative) below the best passing score are "about equal";
       pick the one with the highest priority (candidates arrive in priority
       order, see hmrec.hpo.grid).

R5 iterations is a convergence check, not a choice (convergence_decision).

Ranker (V phase, configs/hpo/v_phase.yaml): every configuration first gets its
own number of trees (choose_num_trees), then the configurations are compared
with choose_winner, each one at its own number of trees.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from hmrec.config import SelectionRule

# Two scores whose absolute difference is below this value are treated as EQUAL.
# This only removes floating-point noise (metric values are float64, so real
# differences are many orders of magnitude larger). It does NOT relax "strictly
# better" in proposal §4.2: a candidate is better on a fold only if
# candidate - default >= SCORE_EQUALITY_TOLERANCE. configs/selection_rule.yaml is
# committed and must not change, so this constant lives here instead.
# See docs/decisions.md (2026-11-16).
SCORE_EQUALITY_TOLERANCE = 1e-12

KEEP = "keep"
SWITCH = "switch"
STOP = "stop"


@dataclass(frozen=True)
class TrialScores:
    """Selection-metric scores of one trial.

    Attributes:
        trial_id: Identifier of the trial.
        fold_scores: Fold name -> score of that fold (e.g. Recall@100 of D1).
        pooled_score: Query-weighted score over all folds (the primary score).
    """

    trial_id: str
    fold_scores: dict[str, float]
    pooled_score: float


@dataclass(frozen=True)
class CandidateCheck:
    """Result of the §4.2 rule for one candidate against the default.

    Attributes:
        trial_id: The candidate.
        relative_improvement: (candidate - default) / default on the pooled score.
        folds_better: Folds where the candidate is strictly better.
        passes: True if both §4.2 conditions hold.
    """

    trial_id: str
    relative_improvement: float
    folds_better: int
    passes: bool


@dataclass(frozen=True)
class StepDecision:
    """Outcome of one step.

    Attributes:
        winner_trial_id: The selected trial (the default if nothing passes or
            the step is a convergence check that keeps it).
        action: "keep" (default stays), "switch" (a candidate wins) or "stop"
            (convergence check failed; a human must decide).
        checks: One CandidateCheck per candidate, in priority order.
        reason: Plain-language explanation for the report.
    """

    winner_trial_id: str
    action: str
    checks: tuple[CandidateCheck, ...]
    reason: str


# ---------------------------------------------------------------- one candidate


def _check_same_folds(default: TrialScores, candidate: TrialScores, rule: SelectionRule) -> None:
    """Both trials must be scored on the same n_folds folds."""
    if list(default.fold_scores) != list(candidate.fold_scores):
        raise ValueError(
            f"{candidate.trial_id} folds {list(candidate.fold_scores)} != "
            f"default folds {list(default.fold_scores)}"
        )
    if len(default.fold_scores) != rule.n_folds:
        raise ValueError(f"expected {rule.n_folds} folds, got {list(default.fold_scores)}")


def relative_change(reference: float, value: float) -> float:
    """(value - reference) / reference. The reference must be > 0."""
    if reference <= 0:
        raise ValueError(f"reference score must be > 0 for a relative change, got {reference}")
    return (value - reference) / reference


def count_folds_better(reference: TrialScores, other: TrialScores) -> int:
    """Number of folds where `other` is strictly better than `reference` (beyond float noise)."""
    better = 0
    for fold, reference_score in reference.fold_scores.items():
        if other.fold_scores[fold] - reference_score >= SCORE_EQUALITY_TOLERANCE:
            better += 1
    return better


def check_candidate(
    default: TrialScores, candidate: TrialScores, rule: SelectionRule
) -> CandidateCheck:
    """Apply the two §4.2 conditions to one candidate."""
    _check_same_folds(default, candidate, rule)
    improvement = relative_change(default.pooled_score, candidate.pooled_score)
    folds_better = count_folds_better(default, candidate)
    passes = improvement >= rule.min_relative_improvement and folds_better >= rule.min_folds_better
    return CandidateCheck(candidate.trial_id, improvement, folds_better, passes)


# ---------------------------------------------------------------- several candidates


def choose_winner(
    default: TrialScores,
    candidates: list[TrialScores],
    rule: SelectionRule,
) -> StepDecision:
    """The multi-candidate rule (module docstring a-c).

    Args:
        default: The current default (winner of the previous step).
        candidates: The step's other configurations, highest priority first.
        rule: configs/selection_rule.yaml.

    Returns:
        The decision with one check per candidate.
    """
    checks = tuple(check_candidate(default, candidate, rule) for candidate in candidates)
    passing = [
        candidate for candidate, check in zip(candidates, checks, strict=True) if check.passes
    ]
    if not passing:
        reason = "no candidate passes the §4.2 rule: keep the default"
        return StepDecision(default.trial_id, KEEP, checks, reason)
    best_score = max(candidate.pooled_score for candidate in passing)
    # "About equal": less than min_relative_improvement (1%) below the best passing score.
    about_equal = [
        candidate
        for candidate in passing
        if -relative_change(best_score, candidate.pooled_score) < rule.min_relative_improvement
    ]
    winner = about_equal[0]  # `passing` keeps the priority order, so this is the highest priority
    reason = (
        f"{len(passing)} candidate(s) pass; {len(about_equal)} within 1% of the best "
        f"passing score {best_score:.6f}; {winner.trial_id} has the highest priority among them"
    )
    return StepDecision(winner.trial_id, SWITCH, checks, reason)


# ---------------------------------------------------------------- Ranker: number of trees


@dataclass(frozen=True)
class NumTreesChoice:
    """The number of trees chosen for one Ranker configuration.

    Attributes:
        num_trees: k*, the chosen number of trees.
        score: Pooled score with k* trees.
        best_num_trees: The smallest k with the highest pooled score.
        best_score: The highest pooled score of the curve.
        uses_all_trees: True if k* is the last checkpoint: the curve may still be
            rising, so more trees could be better.
    """

    num_trees: int
    score: float
    best_num_trees: int
    best_score: float
    uses_all_trees: bool


def choose_num_trees(curve: Mapping[int, float], tolerance: float) -> NumTreesChoice:
    """Pick a configuration's number of trees from its pooled score curve.

    The rule (configs/hpo/v_phase.yaml): with P(k) the pooled score with k trees,

        k* = the smallest k with (max P - P(k)) / max P < tolerance

    So among all k that are "about equal" to the best one, the cheapest wins. A k
    whose score is EXACTLY tolerance below the best is not about equal.

    Args:
        curve: number of trees -> pooled score (e.g. MAP@12 of the 4 folds).
        tolerance: Relative tolerance (min_relative_improvement of
            configs/selection_rule.yaml, 0.01).

    Returns:
        The choice.
    """
    if not curve:
        raise ValueError("the curve is empty: there is no number of trees to choose from")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}")
    tree_counts = sorted(curve)
    best_score = max(curve.values())
    best_num_trees = next(k for k in tree_counts if curve[k] == best_score)
    about_equal = [k for k in tree_counts if -relative_change(best_score, curve[k]) < tolerance]
    chosen = about_equal[0]
    return NumTreesChoice(
        num_trees=chosen,
        score=curve[chosen],
        best_num_trees=best_num_trees,
        best_score=best_score,
        uses_all_trees=chosen == tree_counts[-1],
    )


# ---------------------------------------------------------------- R5 iterations


def is_clearly_worse(reference: TrialScores, other: TrialScores, rule: SelectionRule) -> bool:
    """Mirror of §4.2: pooled drop >= 1% AND strictly worse on >= 3 of 4 folds."""
    _check_same_folds(reference, other, rule)
    drop = -relative_change(reference.pooled_score, other.pooled_score)
    folds_worse = count_folds_better(other, reference)
    return drop >= rule.min_relative_improvement and folds_worse >= rule.min_folds_better


def convergence_decision(
    current: TrialScores,
    shorter: TrialScores,
    longer: TrialScores,
    rule: SelectionRule,
) -> StepDecision:
    """R5 iterations check {shorter, current, longer} = {10, 15, 25} (proposal §5 R5, §4.2).

    - "stop" if the longer run is clearly worse than the current one, or the
      shorter run passes the §4.2 rule against the current one: scores fall
      after fewer iterations, so lambda may be too small. Nothing is changed
      automatically; a human decides whether to go back to the lambda step.
    - "switch" to the longer run if it passes the §4.2 rule.
    - "keep" the current number of iterations otherwise.
    """
    shorter_check = check_candidate(current, shorter, rule)
    longer_check = check_candidate(current, longer, rule)
    checks = (shorter_check, longer_check)
    if is_clearly_worse(current, longer, rule):
        reason = f"{longer.trial_id} is clearly worse than {current.trial_id}: stop and report"
        return StepDecision(current.trial_id, STOP, checks, reason)
    if shorter_check.passes:
        reason = f"{shorter.trial_id} passes the rule against {current.trial_id}: stop and report"
        return StepDecision(current.trial_id, STOP, checks, reason)
    if longer_check.passes:
        reason = f"{longer.trial_id} passes the §4.2 rule: use more iterations"
        return StepDecision(longer.trial_id, SWITCH, checks, reason)
    reason = "the current iterations are on the plateau: keep them"
    return StepDecision(current.trial_id, KEEP, checks, reason)
