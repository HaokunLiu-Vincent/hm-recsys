"""Tests for hmrec.final.goals: the verdict on G1-G3, with examples worked out by hand."""

import pytest

from hmrec.final.goals import judge_g1, judge_g2, judge_g3

FOLDS = ("T1", "T2", "T3", "T4")


def _by_fold(*values: float) -> dict[str, float]:
    return dict(zip(FOLDS, values, strict=True))


# ---------------------------------------------------------------- G1


def test_g1_compares_the_pooled_map_with_the_threshold() -> None:
    by_fold = _by_fold(0.030, 0.031, 0.035, 0.034)
    met = judge_g1(0.0325, by_fold, 0.032)
    assert met.goal == "G1" and met.met is True
    assert (met.value, met.threshold) == ("0.0325", "0.032")
    assert met.detail == "T1=0.03 T2=0.031 T3=0.035 T4=0.034"
    assert judge_g1(0.0319, by_fold, 0.032).met is False


def test_g1_is_met_exactly_at_the_threshold_and_not_just_below_it() -> None:
    assert judge_g1(0.032, _by_fold(0.032, 0.032, 0.032, 0.032), 0.032).met is True
    just_below = 0.03199999999999999  # the float right below 0.032 at 17 digits
    assert just_below < 0.032
    assert judge_g1(just_below, _by_fold(0.03, 0.03, 0.03, 0.03), 0.032).met is False


def test_g1_uses_the_pooled_value_not_the_fold_values() -> None:
    # Two folds below 0.032, but the pooled value passed in is above: met.
    # (Whether pooled is computed correctly is tested in test_final_report.py.)
    assert judge_g1(0.0321, _by_fold(0.029, 0.030, 0.036, 0.035), 0.032).met is True
    # Three folds above, pooled below (a large weak fold): not met.
    assert judge_g1(0.0318, _by_fold(0.033, 0.033, 0.033, 0.028), 0.032).met is False


# ---------------------------------------------------------------- G2


def test_g2_needs_every_fold_at_or_above_the_threshold() -> None:
    met = judge_g2(_by_fold(0.25, 0.26, 0.24, 0.27), 0.24)
    assert met.goal == "G2" and met.met is True
    assert met.value == "0.24"  # the lowest fold, exactly at the threshold
    assert met.detail == "T1=0.25 T2=0.26 T3=0.24 T4=0.27"


def test_g2_fails_if_a_single_fold_is_below() -> None:
    # Mean 0.2575 >= 0.24, but T3 = 0.23 < 0.24: the goal is per fold, so NOT met.
    verdict = judge_g2(_by_fold(0.27, 0.27, 0.23, 0.26), 0.24)
    assert verdict.met is False and verdict.value == "0.23"
    assert judge_g2({"T1": 0.2399}, 0.24).met is False
    with pytest.raises(ValueError, match="at least one fold"):
        judge_g2({}, 0.24)


# ---------------------------------------------------------------- G3


def test_g3_counts_the_folds_where_the_ranker_is_strictly_above_the_baseline() -> None:
    ranker = _by_fold(0.030, 0.031, 0.020, 0.034)
    baseline = _by_fold(0.019, 0.019, 0.021, 0.020)
    # Above on T1, T2, T4 (3 folds); below on T3. 3 >= 3: met.
    verdict = judge_g3(ranker, baseline, 3)
    assert verdict.goal == "G3" and verdict.met is True
    assert (verdict.value, verdict.threshold) == ("3", "3")
    assert verdict.detail == "T1=0.03>0.019 T2=0.031>0.019 T3=0.02<=0.021 T4=0.034>0.02"
    # The same numbers with a threshold of 4 folds: NOT met.
    assert judge_g3(ranker, baseline, 4).met is False


def test_g3_a_tie_does_not_count_for_the_ranker() -> None:
    ranker = _by_fold(0.030, 0.030, 0.020, 0.020)
    baseline = _by_fold(0.020, 0.020, 0.020, 0.021)
    # Above on T1 and T2; EQUAL on T3 (not "strictly above"); below on T4 -> 2 folds.
    verdict = judge_g3(ranker, baseline, 3)
    assert verdict.value == "2" and verdict.met is False
    assert "T3=0.02<=0.02" in verdict.detail


def test_g3_needs_the_same_folds_for_both_systems() -> None:
    with pytest.raises(ValueError, match="the same folds"):
        judge_g3({"T1": 0.03, "T2": 0.03}, {"T1": 0.02, "T3": 0.02}, 1)
