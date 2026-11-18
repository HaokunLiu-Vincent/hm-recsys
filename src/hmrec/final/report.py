"""The numbers of the final evaluation and the verdict on G1-G3, from the result tables (M8).

Everything here reads results/t_phase_metrics.csv and results/t_phase_recall.csv
only: no label, no model, no artifact. The verdict is written once to
results/t_phase_goals.csv by code, with the thresholds of configs/m8_final.yaml.

Pooled score of the target folds (proposal §4.1, query-weighted), as in the V
phase: pooled = sum_f (n_queries_f * mean_f) / sum_f n_queries_f (math.fsum).
The plain mean of the fold means ("equal fold") is reported next to it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hmrec.csv_records import append_row, read_rows
from hmrec.eval.monitor import MONITOR_COLUMNS, NOT_COMPUTED, UNION
from hmrec.final.context import FinalContext
from hmrec.final.goals import GOAL_COLUMNS, GoalVerdict, judge_g1, judge_g2, judge_g3
from hmrec.final.trial import experiment_config_id
from hmrec.hpo.v_records import (
    AREA_T_PHASE,
    NO_TREES,
    RRF_BASELINE,
    MetricRow,
    config_rows,
    equal_fold_score,
    pooled_score,
    read_metric_rows,
    results_files,
    rows_at,
)
from hmrec.ranker.ablation import FULL


@dataclass(frozen=True)
class SystemSummary:
    """One system's metric values on the target folds.

    Attributes:
        experiment: "full", "minus_r6" or "rrf_baseline".
        n_queries: Fold -> queries of the fold.
        by_fold: Metric -> {fold -> mean over the fold's queries}.
        pooled: Metric -> query-weighted mean over all target folds.
        equal_fold: Metric -> plain mean of the fold means.
    """

    experiment: str
    n_queries: dict[str, int]
    by_fold: dict[str, dict[str, float]]
    pooled: dict[str, float]
    equal_fold: dict[str, float]


def system_summary(
    metric_rows: Sequence[MetricRow],
    experiment: str,
    config_id: str,
    folds: Sequence[str],
    num_trees: int,
    metrics: Sequence[str],
) -> SystemSummary:
    """Per-fold, pooled and equal-fold values of one system.

    Args:
        metric_rows: All rows of the metric table.
        experiment: The system.
        config_id: Its configuration id ("rrf_baseline" for the baseline).
        folds: The target folds; every one must have exactly one row.
        num_trees: The number of trees of the rows (0 for the baseline).
        metrics: The metrics to summarise.

    Raises:
        ValueError: If a fold has no row or more than one (hmrec.hpo.v_records.rows_at).
    """
    fold_rows = rows_at(config_rows(metric_rows, experiment, config_id), folds, num_trees)
    return SystemSummary(
        experiment=experiment,
        n_queries={fold: row.n_queries for fold, row in fold_rows.items()},
        by_fold={
            metric: {fold: row.scores[metric] for fold, row in fold_rows.items()}
            for metric in metrics
        },
        pooled={metric: pooled_score(fold_rows, metric) for metric in metrics},
        equal_fold={metric: equal_fold_score(fold_rows, metric) for metric in metrics},
    )


def final_summaries(context: FinalContext) -> dict[str, SystemSummary]:
    """The summaries of the Ranker, the RRF baseline and every secondary experiment on T.

    Raises:
        ValueError: If a system's rows are not complete (the formal run is not finished).
    """
    config, num_trees = context.config, context.locked.num_trees
    rows = read_metric_rows(results_files(context.paths, AREA_T_PHASE)[1], config.metrics)
    folds, metrics = config.target_folds, config.metrics
    summaries = {
        FULL: system_summary(
            rows, FULL, experiment_config_id(context, FULL), folds, num_trees, metrics
        ),
        RRF_BASELINE: system_summary(rows, RRF_BASELINE, RRF_BASELINE, folds, NO_TREES, metrics),
    }
    for experiment in config.secondary_experiments:
        config_id = experiment_config_id(context, experiment)
        summaries[experiment] = system_summary(
            rows, experiment, config_id, folds, num_trees, metrics
        )
    return summaries


def union_recall_by_fold(
    recall_rows: Sequence[Mapping[str, str]], folds: Sequence[str]
) -> dict[str, float]:
    """Fold -> union recall of the six channels, from the monitoring rows.

    Raises:
        ValueError: If a fold does not have exactly one union row with a recall.
    """
    values = {}
    for fold in folds:
        matches = [
            row
            for row in recall_rows
            if row["fold"] == fold and row["scope"] == UNION and row["recall"] != NOT_COMPUTED
        ]
        if len(matches) != 1:
            raise ValueError(f"fold {fold} has {len(matches)} union recall rows, expected 1")
        values[fold] = float(matches[0]["recall"])
    return values


def final_verdicts(context: FinalContext) -> list[GoalVerdict]:
    """Judge G1, G2 and G3 on the target folds with the thresholds of configs/m8_final.yaml."""
    config, goals = context.config, context.config.goals
    summaries = final_summaries(context)
    full, baseline = summaries[FULL], summaries[RRF_BASELINE]
    recall_rows = read_rows(context.paths.t_phase_recall_file, MONITOR_COLUMNS)
    union = union_recall_by_fold(recall_rows, config.target_folds)
    metric = config.metric
    return [
        judge_g1(full.pooled[metric], full.by_fold[metric], goals.g1_min_pooled_map),
        judge_g2(union, goals.g2_min_union_recall),
        judge_g3(full.by_fold[metric], baseline.by_fold[metric], goals.g3_min_folds_above_baseline),
    ]


def record_verdicts(context: FinalContext, verdicts: Sequence[GoalVerdict], now: str) -> None:
    """Write the verdict to results/t_phase_goals.csv, once.

    If the table already has rows, they must say the same (goal, value,
    threshold, met); then nothing is written.

    Raises:
        ValueError: If a recorded verdict differs from the one computed now: the
            results or the thresholds changed after the verdict was written.
    """
    path = context.paths.t_phase_goals_file
    key_columns = ("goal", "value", "threshold", "met")
    computed = [(v.goal, v.value, v.threshold, str(v.met)) for v in verdicts]
    recorded = [tuple(row[c] for c in key_columns) for row in read_rows(path, GOAL_COLUMNS)]
    if recorded:
        if recorded != computed:
            raise ValueError(
                f"{path} records the verdict {recorded}, but the results now give {computed}"
            )
        return
    for verdict in verdicts:
        row = {
            "goal": verdict.goal,
            "criterion": verdict.criterion,
            "value": verdict.value,
            "threshold": verdict.threshold,
            "met": str(verdict.met),
            "detail": verdict.detail,
            "git_commit": context.stamp.git_commit,
            "decided_at": now,
        }
        append_row(path, GOAL_COLUMNS, row)


def summary_lines(context: FinalContext, verdicts: Sequence[GoalVerdict]) -> list[str]:
    """Plain-language lines for the log: every system's numbers, then the verdict."""
    lines = []
    for summary in final_summaries(context).values():
        for metric in context.config.metrics:
            folds = " ".join(f"{f}={v:.6f}" for f, v in summary.by_fold[metric].items())
            lines.append(
                f"{summary.experiment} {metric}: {folds} | pooled {summary.pooled[metric]:.6f} "
                f"| equal fold {summary.equal_fold[metric]:.6f}"
            )
    for verdict in verdicts:
        status = "met" if verdict.met else "NOT met"
        lines.append(
            f"{verdict.goal} {status}: {verdict.value} against {verdict.threshold} "
            f"({verdict.criterion}); {verdict.detail}"
        )
    return lines
