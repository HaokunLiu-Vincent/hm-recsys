"""The monitoring rules of the weekly pipeline (docs/mlops.md §10).

The thresholds are configs/m12_monitoring.yaml: the user's decision, fixed before
the first replay tick and not changed afterwards. NOTHING here changes a
configuration: a rule produces a Finding, which becomes a row of the monitoring
table and (when raised) a line of the run manifest.

Two layers:

    same week (no labels)       input data, features, output signals of the week
                                the tick opens
    one week late (labelled)    recall, MAP@12 against the RRF baseline, hit value
                                of the week the tick closes

"The previous weeks" of a rule are the `history_weeks` (4) weeks right before the
week that is judged. A rule with fewer weeks of history RECORDS its value and
raises nothing; the caller passes only the weeks it has, oldest first.

Flag and alert:
    flag     the rule's condition holds for this week
    alert    input data: the flag itself (and the tick stops);
             labelled rules: the same flag (rule and subject) in
             `consecutive_weeks_for_alert` (3) consecutive weeks;
             features and output signals: never (flag only).

This module holds the rules only (pure functions on numbers). Reading the
history from the state is hmrec.weekly.history. Neither torch nor lightgbm.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from hmrec.config import MonitoringConfig

SAME_WEEK = "same_week"
LABELLED = "labelled"
PIPELINE = "pipeline"

RULE_TRANSACTION_ROWS = "input/transaction_rows"
RULE_DATE_RANGE = "input/date_range"
RULE_FEATURE_MISSING_RATE = "feature/missing_rate"
RULE_FEATURE_MEAN = "feature/mean"
RULE_OUTPUT_SIGNAL = "output/signal"
RULE_CHANNEL_RECALL = "recall/channel"
RULE_UNION_RECALL = "recall/union"
RULE_RANKER_ABOVE_RRF = "ranker/above_rrf_baseline"
RULE_HIT_VALUE = "hit_value"
RULE_STEP_FAILED = "pipeline/step_failed"

NO_SUBJECT = ""
RECORDED_ONLY = "recorded only"


@dataclass(frozen=True)
class Finding:
    """What one rule found for one subject in one week.

    Attributes:
        layer: "same_week", "labelled" or "pipeline".
        rule: The rule's name, e.g. "recall/channel".
        subject: What the rule looked at (a channel, a feature, a signal); "" if
            the rule has one subject only.
        week: Anchor of the week the value describes.
        value: The week's value.
        reference: What it was compared with: the mean of the previous weeks, or
            a fixed limit; None if nothing was compared.
        n_history: Previous weeks that were available (0 for a fixed limit).
        limit: The condition in words, e.g. "more than 10% below the mean of the
            previous 4 weeks"; "recorded only" if nothing can be raised.
        flag: True if the condition holds.
        consecutive_weeks: For a raised labelled flag: weeks in a row with this
            flag, this week included; else 0.
        alert: True if the finding is an alert.
    """

    layer: str
    rule: str
    subject: str
    week: date
    value: float
    reference: float | None
    n_history: int
    limit: str
    flag: bool
    consecutive_weeks: int
    alert: bool


def mean_of(values: Sequence[float]) -> float:
    """The mean of a non-empty list (math.fsum, so the order does not matter)."""
    return math.fsum(values) / len(values)


def has_full_history(history: Sequence[float], config: MonitoringConfig) -> bool:
    """True if every one of the previous `history_weeks` weeks has a value."""
    return len(history) >= config.history_weeks


def recorded(layer: str, rule: str, subject: str, week: date, value: float, n: int) -> Finding:
    """A finding that only records a value (too little history, or a rule without a threshold)."""
    return Finding(layer, rule, subject, week, value, None, n, RECORDED_ONLY, False, 0, False)


def relative_change(value: float, reference: float) -> float:
    """(value - reference) / |reference|; infinite if the reference is 0 and the value is not."""
    if reference == 0:
        return 0.0 if value == 0 else math.inf
    return (value - reference) / abs(reference)


# ---------------------------------------------------------------- same week: input data


def transaction_findings(
    week: date,
    new_week_rows: int,
    previous_week_rows: Sequence[int],
    date_range_is_right: bool,
    config: MonitoringConfig,
) -> list[Finding]:
    """The two input-data rules: the row count of the new week and its date range.

    Args:
        week: The tick's anchor (the same-week layer describes the tick).
        new_week_rows: Purchase lines of the 7 days before the anchor.
        previous_week_rows: Purchase lines of the weeks right before, oldest first.
        date_range_is_right: True if the new week has rows on its first and on
            its last day, and none outside.
        config: The thresholds.

    Returns:
        Two findings. Both are alerts when raised: the tick stops on them.
    """
    range_limit = "the new week has rows from anchor - 7 days to anchor - 1 day"
    range_flag = not date_range_is_right
    findings = [
        Finding(
            SAME_WEEK,
            RULE_DATE_RANGE,
            NO_SUBJECT,
            week,
            1.0 if date_range_is_right else 0.0,
            1.0,
            0,
            range_limit,
            range_flag,
            0,
            range_flag,
        )
    ]
    history = [float(rows) for rows in previous_week_rows]
    rows = float(new_week_rows)
    if not has_full_history(history, config):
        findings.append(
            recorded(SAME_WEEK, RULE_TRANSACTION_ROWS, NO_SUBJECT, week, rows, len(history))
        )
        return findings
    reference = mean_of(history[-config.history_weeks :])
    low, high = config.transactions_min_ratio, config.transactions_max_ratio
    flag = rows < low * reference or rows > high * reference
    limit = (
        f"below {low:g} x or above {high:g} x the mean of the previous {config.history_weeks} weeks"
    )
    findings.append(
        Finding(
            SAME_WEEK,
            RULE_TRANSACTION_ROWS,
            NO_SUBJECT,
            week,
            rows,
            reference,
            config.history_weeks,
            limit,
            flag,
            0,
            flag,
        )
    )
    return findings


# ---------------------------------------------------------------- same week: features, output


@dataclass(frozen=True)
class FeatureSummary:
    """The two monitored numbers of one feature in one week.

    Attributes:
        missing_rate: Share of rows without a value.
        mean: Mean of the present values; None for a categorical feature or a
            feature with no present value.
    """

    missing_rate: float
    mean: float | None


def feature_findings(
    week: date,
    current: Mapping[str, FeatureSummary],
    history: Sequence[Mapping[str, FeatureSummary]],
    config: MonitoringConfig,
) -> list[Finding]:
    """The feature rules: missing rate (absolute change) and mean (relative change).

    Args:
        week: The tick's anchor.
        current: Feature -> its summary in the week the tick opens.
        history: The same for the previous weeks, oldest first; fewer than
            `history_weeks` entries means every value is recorded only.
        config: The thresholds.

    Returns:
        One finding per feature for the missing rate, and one for the mean of
        every feature that has one. Flags only, never alerts.
    """
    weeks = list(history)[-config.history_weeks :]
    full = len(history) >= config.history_weeks
    n = config.history_weeks
    findings: list[Finding] = []
    for feature, summary in current.items():
        past = [past_week[feature] for past_week in weeks if feature in past_week]
        if not full or len(past) < n:
            findings.append(
                recorded(
                    SAME_WEEK,
                    RULE_FEATURE_MISSING_RATE,
                    feature,
                    week,
                    summary.missing_rate,
                    len(past),
                )
            )
            if summary.mean is not None:
                findings.append(
                    recorded(SAME_WEEK, RULE_FEATURE_MEAN, feature, week, summary.mean, len(past))
                )
            continue
        reference = mean_of([item.missing_rate for item in past])
        change = config.feature_missing_rate_max_change
        flag = abs(summary.missing_rate - reference) > change
        limit = (
            f"moves by more than {change:g} (absolute) against the mean of the previous {n} weeks"
        )
        findings.append(
            Finding(
                SAME_WEEK,
                RULE_FEATURE_MISSING_RATE,
                feature,
                week,
                summary.missing_rate,
                reference,
                n,
                limit,
                flag,
                0,
                False,
            )
        )
        past_means = [item.mean for item in past if item.mean is not None]
        if summary.mean is None:
            continue
        if len(past_means) < n:
            findings.append(
                recorded(SAME_WEEK, RULE_FEATURE_MEAN, feature, week, summary.mean, len(past_means))
            )
            continue
        mean_reference = mean_of(past_means)
        share = config.feature_mean_max_relative_change
        mean_flag = abs(relative_change(summary.mean, mean_reference)) > share
        mean_limit = f"changes by more than {share:.0%} against the mean of the previous {n} weeks"
        findings.append(
            Finding(
                SAME_WEEK,
                RULE_FEATURE_MEAN,
                feature,
                week,
                summary.mean,
                mean_reference,
                n,
                mean_limit,
                mean_flag,
                0,
                False,
            )
        )
    return findings


def output_findings(
    week: date,
    current: Mapping[str, float],
    history: Mapping[str, Sequence[float]],
    config: MonitoringConfig,
) -> list[Finding]:
    """The output-signal rule: more than a share away from the mean of the previous weeks.

    Args:
        week: The tick's anchor.
        current: Signal name -> its value in the week the tick opens.
        history: Signal name -> its values in the previous weeks, oldest first.
        config: The thresholds.

    Returns:
        One finding per signal. Flags only, never alerts.
    """
    n, share = config.history_weeks, config.output_signal_max_relative_change
    limit = f"more than {share:.0%} away from the mean of the previous {n} weeks"
    findings = []
    for signal, value in current.items():
        past = list(history.get(signal, ()))
        if not has_full_history(past, config):
            findings.append(recorded(SAME_WEEK, RULE_OUTPUT_SIGNAL, signal, week, value, len(past)))
            continue
        reference = mean_of(past[-n:])
        flag = abs(relative_change(value, reference)) > share
        findings.append(
            Finding(
                SAME_WEEK,
                RULE_OUTPUT_SIGNAL,
                signal,
                week,
                value,
                reference,
                n,
                limit,
                flag,
                0,
                False,
            )
        )
    return findings


# ---------------------------------------------------------------- one week late: labelled


def consecutive_weeks(
    rule: str, subject: str, earlier_flags: Sequence[frozenset[tuple[str, str]]]
) -> int:
    """Weeks in a row with this flag, the judged week included.

    Args:
        rule: The rule of the flag that is raised now.
        subject: Its subject.
        earlier_flags: The raised (rule, subject) pairs of the weeks before the
            judged week, the NEWEST first; the list ends at the first week that
            has no monitoring record.
    """
    count = 1
    for flags in earlier_flags:
        if (rule, subject) not in flags:
            break
        count += 1
    return count


def labelled_finding(
    rule: str,
    subject: str,
    week: date,
    value: float,
    reference: float,
    n_history: int,
    limit: str,
    flag: bool,
    earlier_flags: Sequence[frozenset[tuple[str, str]]],
    config: MonitoringConfig,
) -> Finding:
    """A labelled finding with its consecutive-week count and alert."""
    count = consecutive_weeks(rule, subject, earlier_flags) if flag else 0
    alert = count >= config.consecutive_weeks_for_alert
    return Finding(
        LABELLED, rule, subject, week, value, reference, n_history, limit, flag, count, alert
    )


def labelled_findings(
    week: date,
    channel_recall: Mapping[str, float],
    recall_history: Mapping[str, Sequence[float]],
    union_recall: float,
    ranker_score: float,
    rrf_score: float,
    hit_value: float,
    earlier_flags: Sequence[frozenset[tuple[str, str]]],
    config: MonitoringConfig,
) -> list[Finding]:
    """The labelled rules for the week a tick closed.

    Args:
        week: Anchor of the closed week.
        channel_recall: Channel -> its recall in the closed week.
        recall_history: Channel -> its recall in the weeks before, oldest first.
        union_recall: Union recall of the six channels in the closed week.
        ranker_score: The Ranker's MAP@12 of the closed week.
        rrf_score: The RRF baseline's MAP@12 of the closed week.
        hit_value: The hit value of the closed week (recorded only).
        earlier_flags: Raised labelled (rule, subject) pairs of the weeks before
            the closed week, newest first (see consecutive_weeks).
        config: The thresholds.
    """
    n, drop = config.history_weeks, config.channel_recall_max_relative_drop
    recall_limit = f"more than {drop:.0%} below the mean of the previous {n} weeks"
    findings = []
    for channel, recall in channel_recall.items():
        past = list(recall_history.get(channel, ()))
        if not has_full_history(past, config):
            findings.append(
                recorded(LABELLED, RULE_CHANNEL_RECALL, channel, week, recall, len(past))
            )
            continue
        reference = mean_of(past[-n:])
        # The formula of the stage-1 rule (hmrec.eval.monitor.compare_with_previous).
        flag = reference > 0 and recall / reference - 1.0 < -drop
        findings.append(
            labelled_finding(
                RULE_CHANNEL_RECALL,
                channel,
                week,
                recall,
                reference,
                n,
                recall_limit,
                flag,
                earlier_flags,
                config,
            )
        )
    union_limit = f"below {config.union_recall_min:g}"
    findings.append(
        labelled_finding(
            RULE_UNION_RECALL,
            NO_SUBJECT,
            week,
            union_recall,
            config.union_recall_min,
            0,
            union_limit,
            union_recall < config.union_recall_min,
            earlier_flags,
            config,
        )
    )
    if config.ranker_must_beat_rrf:
        findings.append(
            labelled_finding(
                RULE_RANKER_ABOVE_RRF,
                NO_SUBJECT,
                week,
                ranker_score,
                rrf_score,
                0,
                "not above the MAP@12 of the RRF baseline",
                not ranker_score > rrf_score,
                earlier_flags,
                config,
            )
        )
    findings.append(recorded(LABELLED, RULE_HIT_VALUE, NO_SUBJECT, week, hit_value, 0))
    return findings


def step_failed_finding(week: date, step: str) -> Finding:
    """The alert of a tick that stopped: a step failed or a check stopped it (mlops §10)."""
    limit = "a step failed or a check stopped the tick"
    return Finding(PIPELINE, RULE_STEP_FAILED, step, week, 1.0, None, 0, limit, True, 0, True)


def raised_flags(findings: Sequence[Finding], layer: str) -> frozenset[tuple[str, str]]:
    """The (rule, subject) pairs of the raised flags of one layer."""
    return frozenset(
        (finding.rule, finding.subject)
        for finding in findings
        if finding.layer == layer and finding.flag
    )
