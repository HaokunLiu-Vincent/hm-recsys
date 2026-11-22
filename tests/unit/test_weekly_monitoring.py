"""Tests for weekly.monitoring: the rules of configs/m12_monitoring.yaml on hand-made numbers.

The thresholds are the real ones (the user's decision of 2026-11-22): history of
4 weeks; channel recall 10% below the mean; union recall below 0.24; the Ranker
not above RRF; alert after 3 consecutive weeks; transactions outside 50%..200%;
feature missing rate +-0.05, feature mean +-50%; output signals +-25%.
"""

from datetime import date
from pathlib import Path

import pytest

from hmrec.config import load_monitoring_config, load_yaml, parse_monitoring_config
from hmrec.weekly.monitoring import (
    LABELLED,
    RECORDED_ONLY,
    RULE_CHANNEL_RECALL,
    RULE_FEATURE_MEAN,
    RULE_FEATURE_MISSING_RATE,
    RULE_HIT_VALUE,
    RULE_RANKER_ABOVE_RRF,
    RULE_UNION_RECALL,
    FeatureSummary,
    Finding,
    consecutive_weeks,
    feature_findings,
    labelled_findings,
    output_findings,
    raised_flags,
    relative_change,
    step_failed_finding,
    transaction_findings,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO / "configs" / "m12_monitoring.yaml"
CONFIG = load_monitoring_config(CONFIG_FILE)
WEEK = date(2020, 8, 5)


def _by_key(findings: list[Finding]) -> dict[tuple[str, str], Finding]:
    return {(finding.rule, finding.subject): finding for finding in findings}


def test_the_config_holds_the_decided_thresholds() -> None:
    assert (CONFIG.history_weeks, CONFIG.consecutive_weeks_for_alert) == (4, 3)
    assert (CONFIG.channel_recall_max_relative_drop, CONFIG.union_recall_min) == (0.10, 0.24)
    assert CONFIG.ranker_must_beat_rrf is True
    assert (CONFIG.transactions_min_ratio, CONFIG.transactions_max_ratio) == (0.5, 2.0)
    assert CONFIG.feature_missing_rate_max_change == 0.05
    assert CONFIG.feature_mean_max_relative_change == 0.5
    assert CONFIG.output_signal_max_relative_change == 0.25


def test_config_values_are_validated() -> None:
    raw = load_yaml(CONFIG_FILE)
    with pytest.raises(ValueError, match="min_ratio must be below"):
        parse_monitoring_config(
            {**raw, "same_week": {**raw["same_week"], "transactions_min_ratio": 3}}
        )
    with pytest.raises(ValueError, match="history_weeks"):
        parse_monitoring_config({**raw, "history_weeks": 0})


# ---------------------------------------------------------------- input data


@pytest.mark.parametrize(
    ("rows", "flag"),
    [(1000, False), (500, False), (499, True), (2000, False), (2001, True)],
)
def test_transaction_rows_alert_outside_half_to_double_the_mean(rows: int, flag: bool) -> None:
    # Mean of the previous 4 weeks = (900 + 1100 + 1000 + 1000) / 4 = 1000: band 500 .. 2000.
    findings = _by_key(transaction_findings(WEEK, rows, [900, 1100, 1000, 1000], True, CONFIG))
    rule = findings[("input/transaction_rows", "")]
    assert (rule.reference, rule.n_history, rule.flag, rule.alert) == (1000.0, 4, flag, flag)
    assert not findings[("input/date_range", "")].flag


def test_a_wrong_date_range_is_an_alert_and_short_history_raises_nothing() -> None:
    findings = _by_key(transaction_findings(WEEK, 5, [1000, 1000, 1000], False, CONFIG))
    assert findings[("input/date_range", "")].alert
    rows = findings[("input/transaction_rows", "")]
    # 3 weeks of history only: 5 rows would be far below the band, but nothing is raised.
    assert (rows.limit, rows.n_history, rows.flag, rows.alert) == (RECORDED_ONLY, 3, False, False)


# ---------------------------------------------------------------- features


def test_feature_rules_compare_with_the_mean_of_four_weeks() -> None:
    # Missing rates 0.20, 0.22, 0.18, 0.20 -> mean 0.20; means 10, 12, 8, 10 -> mean 10.
    history = [
        {"f": FeatureSummary(rate, mean), "code": FeatureSummary(0.0, None)}
        for rate, mean in ((0.20, 10.0), (0.22, 12.0), (0.18, 8.0), (0.20, 10.0))
    ]
    quiet = _by_key(
        feature_findings(
            WEEK,
            {"f": FeatureSummary(0.25, 15.0), "code": FeatureSummary(0.04, None)},
            history,
            CONFIG,
        )
    )
    # 0.25 is exactly 5 points above 0.20 and 15 exactly 50% above 10: not MORE than the limit.
    assert not quiet[(RULE_FEATURE_MISSING_RATE, "f")].flag
    assert not quiet[(RULE_FEATURE_MEAN, "f")].flag
    assert quiet[(RULE_FEATURE_MEAN, "f")].reference == pytest.approx(10.0)
    # A categorical feature has a missing rate but no mean.
    assert (RULE_FEATURE_MEAN, "code") not in quiet
    loud = _by_key(feature_findings(WEEK, {"f": FeatureSummary(0.26, 4.9)}, history, CONFIG))
    assert loud[(RULE_FEATURE_MISSING_RATE, "f")].flag and loud[(RULE_FEATURE_MEAN, "f")].flag
    # A feature flag is never an alert.
    assert not any(finding.alert for finding in loud.values())


def test_features_with_less_than_four_weeks_are_recorded_only() -> None:
    history = [{"f": FeatureSummary(0.2, 10.0)}] * 3
    findings = feature_findings(WEEK, {"f": FeatureSummary(0.9, 99.0)}, history, CONFIG)
    assert [(f.limit, f.flag, f.n_history) for f in findings] == [(RECORDED_ONLY, False, 3)] * 2
    # A feature that is new (not in the earlier weeks) has no history either.
    new = feature_findings(
        WEEK, {"new": FeatureSummary(0.9, 1.0)}, [{"f": FeatureSummary(0.2, 1.0)}] * 4, CONFIG
    )
    assert {finding.limit for finding in new} == {RECORDED_ONLY}


def test_relative_change_handles_a_zero_reference() -> None:
    assert relative_change(0.0, 0.0) == 0.0
    assert relative_change(1.0, 0.0) == float("inf")
    assert relative_change(-3.0, -2.0) == pytest.approx(-0.5)


# ---------------------------------------------------------------- output signals


def test_output_signals_are_flagged_a_quarter_away_from_the_mean() -> None:
    history = {"candidates_per_query": [300.0, 310.0, 290.0, 300.0], "distinct_articles": [40.0]}
    current = {"candidates_per_query": 224.0, "distinct_articles": 5.0, "top_popular_share": 0.9}
    findings = _by_key(output_findings(WEEK, current, history, CONFIG))
    per_query = findings[("output/signal", "candidates_per_query")]
    # 224 is 25.3% below the mean 300: flagged, never an alert.
    assert (per_query.reference, per_query.flag, per_query.alert) == (300.0, True, False)
    assert not _by_key(output_findings(WEEK, {"candidates_per_query": 225.0}, history, CONFIG))[
        ("output/signal", "candidates_per_query")
    ].flag
    # One week of history, and no history at all: recorded only.
    assert findings[("output/signal", "distinct_articles")].limit == RECORDED_ONLY
    assert findings[("output/signal", "top_popular_share")].n_history == 0


# ---------------------------------------------------------------- labelled


def _labelled(
    recall: float = 0.10,
    union: float = 0.30,
    ranker: float = 0.03,
    rrf: float = 0.02,
    earlier: list[frozenset[tuple[str, str]]] | None = None,
    history: list[float] | None = None,
) -> dict[tuple[str, str], Finding]:
    past = [0.10, 0.12, 0.08, 0.10] if history is None else history
    findings = labelled_findings(
        WEEK,
        {"r4_itemcf": recall},
        {"r4_itemcf": past},
        union,
        ranker,
        rrf,
        1.5,
        earlier or [],
        CONFIG,
    )
    return _by_key(findings)


def test_channel_recall_is_flagged_more_than_ten_percent_below_the_mean() -> None:
    # Mean of the previous 4 weeks = 0.10: flagged below 0.09 (9.9% below is not flagged).
    assert not _labelled(recall=0.0901)[(RULE_CHANNEL_RECALL, "r4_itemcf")].flag
    low = _labelled(recall=0.0899)[(RULE_CHANNEL_RECALL, "r4_itemcf")]
    assert (low.flag, low.consecutive_weeks, low.alert, low.layer) == (True, 1, False, LABELLED)
    assert low.reference == pytest.approx(0.10)
    short = _labelled(recall=0.01, history=[0.10, 0.10, 0.10])[(RULE_CHANNEL_RECALL, "r4_itemcf")]
    assert (short.limit, short.flag) == (RECORDED_ONLY, False)


def test_union_recall_and_ranker_rules_have_fixed_limits() -> None:
    quiet = _labelled(union=0.24, ranker=0.021, rrf=0.02)
    assert not quiet[(RULE_UNION_RECALL, "")].flag and not quiet[(RULE_RANKER_ABOVE_RRF, "")].flag
    loud = _labelled(union=0.2399, ranker=0.02, rrf=0.02)
    assert loud[(RULE_UNION_RECALL, "")].flag
    # "Not above" includes equal.
    assert loud[(RULE_RANKER_ABOVE_RRF, "")].flag
    assert loud[(RULE_RANKER_ABOVE_RRF, "")].reference == 0.02
    # The hit value is recorded only: no threshold exists for it.
    hit = quiet[(RULE_HIT_VALUE, "")]
    assert (hit.value, hit.limit, hit.flag) == (1.5, RECORDED_ONLY, False)


def test_the_same_flag_in_three_consecutive_weeks_is_an_alert() -> None:
    union_flag = frozenset({(RULE_UNION_RECALL, "")})
    other_flag = frozenset({(RULE_CHANNEL_RECALL, "r4_itemcf")})
    one_before = _labelled(union=0.2, earlier=[union_flag, other_flag])[(RULE_UNION_RECALL, "")]
    assert (one_before.consecutive_weeks, one_before.alert) == (2, False)
    two_before = _labelled(union=0.2, earlier=[union_flag, union_flag])[(RULE_UNION_RECALL, "")]
    assert (two_before.consecutive_weeks, two_before.alert) == (3, True)
    # A week without the flag in between breaks the run.
    broken = _labelled(union=0.2, earlier=[other_flag, union_flag, union_flag])
    assert broken[(RULE_UNION_RECALL, "")].consecutive_weeks == 1
    # Not raised this week: no count, no alert, whatever the weeks before were.
    quiet = _labelled(union=0.3, earlier=[union_flag, union_flag])[(RULE_UNION_RECALL, "")]
    assert (quiet.flag, quiet.consecutive_weeks, quiet.alert) == (False, 0, False)
    assert consecutive_weeks(RULE_UNION_RECALL, "", []) == 1


def test_raised_flags_and_the_step_failed_alert() -> None:
    findings = list(_labelled(union=0.2, recall=0.01).values())
    assert raised_flags(findings, LABELLED) == {
        (RULE_UNION_RECALL, ""),
        (RULE_CHANNEL_RECALL, "r4_itemcf"),
    }
    failed = step_failed_finding(WEEK, "O2_features")
    assert (failed.rule, failed.subject, failed.flag, failed.alert) == (
        "pipeline/step_failed",
        "O2_features",
        True,
        True,
    )
