"""Tests for the §4.2 selection rule, the multi-candidate rule and the R5 convergence check.

Rule (configs/selection_rule.yaml): relative improvement >= 1% on the pooled
score AND strictly better on >= 3 of 4 folds. Fold scores below are chosen so
that the pooled score is given explicitly (the rule never recomputes it).
"""

from pathlib import Path

import pytest

from hmrec.config import load_selection_rule
from hmrec.hpo.selection import (
    KEEP,
    STOP,
    SWITCH,
    TrialScores,
    check_candidate,
    choose_winner,
    convergence_decision,
    is_clearly_worse,
)

RULE = load_selection_rule(Path(__file__).resolve().parents[2] / "configs" / "selection_rule.yaml")
FOLDS = ("D1", "D2", "D3", "D4")


def _scores(trial_id: str, folds: tuple[float, float, float, float], pooled: float) -> TrialScores:
    return TrialScores(trial_id, dict(zip(FOLDS, folds, strict=True)), pooled)


DEFAULT = _scores("default", (0.100, 0.100, 0.100, 0.100), 0.100)


# ---------------------------------------------------------------- one candidate


def test_rule_values_are_the_committed_ones() -> None:
    assert (RULE.min_relative_improvement, RULE.min_folds_better, RULE.n_folds) == (0.01, 3, 4)


def test_passes_with_enough_improvement_on_three_folds() -> None:
    check = check_candidate(DEFAULT, _scores("c", (0.104, 0.103, 0.102, 0.099), 0.102), RULE)
    assert check.folds_better == 3
    assert check.relative_improvement == pytest.approx(0.02)
    assert check.passes


def test_enough_improvement_but_only_two_folds_better_fails() -> None:
    # +5% pooled (driven by D3, D4) but only 2 of 4 folds better.
    check = check_candidate(DEFAULT, _scores("c", (0.099, 0.098, 0.115, 0.110), 0.105), RULE)
    assert check.relative_improvement == pytest.approx(0.05)
    assert check.folds_better == 2
    assert not check.passes


def test_all_folds_better_but_improvement_below_one_percent_fails() -> None:
    check = check_candidate(DEFAULT, _scores("c", (0.1005, 0.1005, 0.1005, 0.1005), 0.1005), RULE)
    assert check.folds_better == 4
    assert not check.passes


def test_equal_fold_score_is_not_better() -> None:
    # D1 equal (up to float noise), D2 and D3 better, D4 worse -> 2 folds better.
    candidate = _scores("c", (0.100 + 1e-15, 0.110, 0.110, 0.095), 0.104)
    check = check_candidate(DEFAULT, candidate, RULE)
    assert check.folds_better == 2
    assert not check.passes


def test_mismatched_folds_or_zero_default_raise() -> None:
    other_folds = TrialScores("c", {"D1": 0.1, "D2": 0.1, "D3": 0.1, "D5": 0.1}, 0.1)
    with pytest.raises(ValueError, match="folds"):
        check_candidate(DEFAULT, other_folds, RULE)
    zero = _scores("zero", (0.0, 0.0, 0.0, 0.0), 0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        check_candidate(zero, DEFAULT, RULE)


# ---------------------------------------------------------------- several candidates


def test_no_candidate_passes_keeps_default() -> None:
    weak = _scores("weak", (0.101, 0.101, 0.099, 0.099), 0.100)
    two_folds = _scores("two_folds", (0.090, 0.090, 0.130, 0.130), 0.110)
    decision = choose_winner(DEFAULT, [weak, two_folds], RULE)
    assert decision.action == KEEP
    assert decision.winner_trial_id == "default"
    assert [check.passes for check in decision.checks] == [False, False]


def test_top_scorer_fails_but_another_passes() -> None:
    # "top" has the best pooled score but is better on only 2 folds.
    top = _scores("top", (0.090, 0.095, 0.140, 0.140), 0.120)
    steady = _scores("steady", (0.103, 0.103, 0.103, 0.103), 0.103)
    decision = choose_winner(DEFAULT, [top, steady], RULE)
    assert decision.action == SWITCH
    assert decision.winner_trial_id == "steady"


def test_close_passing_candidates_follow_priority_order() -> None:
    # Both pass; "second" scores higher, but "first" is within 1% of it and has
    # the higher priority (listed first) -> "first".
    first = _scores("first", (0.105, 0.105, 0.105, 0.105), 0.1050)
    second = _scores("second", (0.106, 0.106, 0.106, 0.106), 0.1055)
    decision = choose_winner(DEFAULT, [first, second], RULE)
    assert decision.winner_trial_id == "first"
    # Reversed priority -> "second".
    assert choose_winner(DEFAULT, [second, first], RULE).winner_trial_id == "second"


def test_clearly_better_candidate_beats_priority() -> None:
    # "first" passes but is more than 1% below "second" -> not "about equal".
    first = _scores("first", (0.103, 0.103, 0.103, 0.103), 0.103)
    second = _scores("second", (0.110, 0.110, 0.110, 0.110), 0.110)
    assert choose_winner(DEFAULT, [first, second], RULE).winner_trial_id == "second"


def test_failing_candidate_never_wins_even_if_listed_first() -> None:
    failing = _scores("failing", (0.120, 0.120, 0.090, 0.090), 0.106)
    passing = _scores("passing", (0.105, 0.105, 0.105, 0.105), 0.105)
    assert choose_winner(DEFAULT, [failing, passing], RULE).winner_trial_id == "passing"


# ---------------------------------------------------------------- R5 iterations

CURRENT = _scores("it15", (0.100, 0.100, 0.100, 0.100), 0.100)
SHORT_PLATEAU = _scores("it10", (0.0995, 0.0995, 0.0995, 0.0995), 0.0995)
LONG_PLATEAU = _scores("it25", (0.1003, 0.1003, 0.1003, 0.1003), 0.1003)


def test_convergence_plateau_keeps_current() -> None:
    decision = convergence_decision(CURRENT, SHORT_PLATEAU, LONG_PLATEAU, RULE)
    assert (decision.action, decision.winner_trial_id) == (KEEP, "it15")


def test_convergence_switches_when_longer_passes() -> None:
    longer = _scores("it25", (0.103, 0.103, 0.103, 0.103), 0.103)
    decision = convergence_decision(CURRENT, SHORT_PLATEAU, longer, RULE)
    assert (decision.action, decision.winner_trial_id) == (SWITCH, "it25")


def test_convergence_stops_when_longer_is_clearly_worse() -> None:
    longer = _scores("it25", (0.097, 0.097, 0.097, 0.101), 0.098)
    assert is_clearly_worse(CURRENT, longer, RULE)
    decision = convergence_decision(CURRENT, SHORT_PLATEAU, longer, RULE)
    assert (decision.action, decision.winner_trial_id) == (STOP, "it15")


def test_convergence_stops_instead_of_choosing_fewer_iterations() -> None:
    # 10 iterations pass the rule against 15: never switch to 10 automatically.
    shorter = _scores("it10", (0.104, 0.104, 0.104, 0.104), 0.104)
    decision = convergence_decision(CURRENT, shorter, LONG_PLATEAU, RULE)
    assert (decision.action, decision.winner_trial_id) == (STOP, "it15")


def test_slightly_worse_longer_run_is_not_clearly_worse() -> None:
    # -2% pooled but only 2 folds worse -> not "clearly worse".
    longer = _scores("it25", (0.090, 0.090, 0.101, 0.101), 0.098)
    assert not is_clearly_worse(CURRENT, longer, RULE)
