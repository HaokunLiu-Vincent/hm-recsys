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
    choose_num_trees,
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


# ---------------------------------------------------------------- Ranker: number of trees
#
# k* = the smallest k with (max P - P(k)) / max P < 1% (configs/hpo/v_phase.yaml).


def test_num_trees_takes_the_smallest_k_on_the_plateau() -> None:
    # Best is 0.0320 at 300. Within 1% means P(k) > 0.03168: 200 (0.0318) and 400 qualify,
    # 100 (0.0300, 6.25% below) does not. The smallest of them is 200.
    curve = {100: 0.0300, 200: 0.0318, 300: 0.0320, 400: 0.0319}
    choice = choose_num_trees(curve, 0.01)
    assert (choice.num_trees, choice.score) == (200, 0.0318)
    assert (choice.best_num_trees, choice.best_score) == (300, 0.0320)
    assert choice.uses_all_trees is False


def test_num_trees_exactly_one_percent_below_is_not_about_equal() -> None:
    # 0.99 is exactly 1% below 1.0: "< 1%" excludes it, so 200 (0.5% below) is chosen.
    choice = choose_num_trees({100: 0.99, 200: 0.995, 300: 1.0}, 0.01)
    assert choice.num_trees == 200


def test_num_trees_rising_curve_uses_all_trees() -> None:
    choice = choose_num_trees({100: 0.020, 200: 0.025, 300: 0.030}, 0.01)
    assert (choice.num_trees, choice.best_num_trees) == (300, 300)
    assert choice.uses_all_trees is True  # the caller stops and asks for more trees


def test_num_trees_early_peak_then_overfitting() -> None:
    choice = choose_num_trees({100: 0.030, 200: 0.028, 300: 0.020}, 0.01)
    assert choice.num_trees == 100 and choice.uses_all_trees is False


def test_num_trees_ties_at_the_best_score_take_the_smallest_k() -> None:
    choice = choose_num_trees({300: 0.03, 100: 0.03, 200: 0.03}, 0.01)
    assert (choice.num_trees, choice.best_num_trees) == (100, 100)


def test_num_trees_needs_a_curve_and_a_positive_tolerance() -> None:
    with pytest.raises(ValueError, match="curve is empty"):
        choose_num_trees({}, 0.01)
    with pytest.raises(ValueError, match="tolerance must be > 0"):
        choose_num_trees({100: 0.03}, 0.0)
