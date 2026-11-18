"""Result records of the Ranker's V phase (proposal §4.4, §9.4).

A Ranker "trial" is one configuration on ONE fold (a model trained on the 4
weeks before the fold, then the fold scored), run in its own process. This is a
different shape from a D-phase trial (one configuration on 4 folds, one score),
so the V phase has its own record instead of hmrec.hpo.records.TrialRecord;
results/d_phase.csv and its columns are untouched.

Three append-only tables (all in results/, in git):

    v_phase.csv           one row per trial: what ran, how long, how much memory.
                          TWO memory columns, never to be mixed up (hmrec.memory):
                            sampled_peak_rss_gb   sampled, a lower bound
                            process_max_rss_gb    ru_maxrss of the trial's process,
                                                  the TRUE peak
    v_phase_metrics.csv   one row per (configuration, fold, number of trees): the
                          fold's metric values with the first k trees. Every
                          selection is computed from this table alone.
    v_phase_steps.csv     one row per configuration of a step: its number of
                          trees, scores, whether it passed, and the decision.

The speed test writes the first two shapes to results/m7_speed_test*.csv, so its
numbers never mix with the grid's.

Every trial row, and every row of results/m7_runs.csv (the helper processes:
sampling, Dataset building, RRF baseline), also names the LightGBM version and
the path and version of the OpenMP runtime (libomp) the process loaded
(hmrec.ranker.runtime): libomp is a system library outside uv.lock.

Pooled score of several folds (proposal §4.1, query-weighted): this module
computes it from the fold rows as

    pooled = sum_f (n_queries_f * mean_f) / sum_f n_queries_f      (math.fsum)

which is the mean over all queries of the folds, up to float rounding (about
1e-16 relative; the rule's thresholds are 1e-2). Per-fold comparisons use the
fold means themselves.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hmrec.csv_records import append_row, read_rows
from hmrec.hpo.selection import TrialScores
from hmrec.paths import ProjectPaths
from hmrec.ranker.runtime import RUNTIME_COLUMNS, LightGBMRuntime
from hmrec.run_records import RUN_COLUMNS

AREA_V_PHASE = "v_phase"
AREA_SPEED_TEST = "speed_test"
AREAS = (AREA_V_PHASE, AREA_SPEED_TEST)

RRF_BASELINE = "rrf_baseline"  # experiment name and config_id of the RRF baseline rows
SPEED_TEST_STEP = "speed_test"  # step name of the speed test's evaluated trial
NO_TREES = 0  # num_trees of a row that has no model (the RRF baseline)
TRAIN_METRIC_COLUMN = "train_ndcg@12"
NOT_COMPUTED = ""

TRIAL_COLUMNS: tuple[str, ...] = (
    "phase",
    "experiment",
    "step",
    "config_id",
    "fold",
    "train_folds",
    "params_json",
    "config_json",
    "histogram_mode",
    "num_trees",
    "evaluated",
    "n_train_rows",
    "n_train_queries",
    "n_scored_rows",
    "n_queries",
    "load_seconds",
    "train_seconds",
    "seconds_per_tree",
    "score_seconds",
    "total_seconds",
    "sampled_peak_rss_gb",
    "process_max_rss_gb",
    "model_sha256",
    *RUNTIME_COLUMNS,
    "git_commit",
    "git_dirty",
    "finished_at",
)

# results/m7_runs.csv: the columns of results/m6_runs.csv plus the runtime columns.
M7_RUN_COLUMNS: tuple[str, ...] = (*RUN_COLUMNS, *RUNTIME_COLUMNS)


def results_files(paths: ProjectPaths, area: str) -> tuple[Path, Path]:
    """(trial table, metric table) of an area: the grid's files or the speed test's."""
    if area == AREA_V_PHASE:
        return paths.v_phase_results_file, paths.v_phase_metrics_file
    if area == AREA_SPEED_TEST:
        return paths.m7_speed_test_file, paths.m7_speed_test_metrics_file
    raise ValueError(f"area must be one of {AREAS}, got {area!r}")


# ---------------------------------------------------------------- trial rows


@dataclass(frozen=True)
class RankerTrialRecord:
    """Everything recorded about one Ranker trial (one configuration on one fold).

    Attributes:
        phase: "V".
        experiment: "full" or "minus_r6".
        step: The step the trial ran in, e.g. "leaves_x_min_data".
        config_id: Hash of everything that defines the model (hpo.v_phase.ranker_config_id).
        fold: The predicted fold.
        train_folds: The training folds in time order.
        params: The tuned values (num_leaves, min_data_in_leaf, negatives_per_positive).
        config: The full configuration the config_id was computed from.
        histogram_mode: "col_wise" or "row_wise".
        num_trees: Trees trained.
        evaluated: False for a timing-only run (no scoring, no label read).
        n_train_rows: Rows of the training Dataset.
        n_train_queries: Queries (groups) of the training Dataset.
        n_scored_rows: Candidate rows scored on the fold (0 if not evaluated).
        n_queries: Queries of the fold (0 if not evaluated).
        load_seconds: Seconds to verify and load the binary Dataset.
        train_seconds: Seconds of training.
        seconds_per_tree: Mean seconds per tree after the warm-up trees.
        score_seconds: Seconds of scoring and evaluation (0 if not evaluated).
        total_seconds: Wall time of the whole process.
        sampled_peak_rss_gb: SAMPLED peak RSS of the trial (a lower bound).
        process_max_rss_gb: TRUE peak RSS of the trial's process (ru_maxrss).
        model_sha256: sha256 of the saved model ("" if the model was not saved).
        runtime: LightGBM version and the loaded OpenMP runtime of the trial's process.
        git_commit: HEAD when the trial ran.
        git_dirty: True if code or configs differed from HEAD.
        finished_at: ISO timestamp.
    """

    phase: str
    experiment: str
    step: str
    config_id: str
    fold: str
    train_folds: tuple[str, ...]
    params: dict[str, Any]
    config: dict[str, Any]
    histogram_mode: str
    num_trees: int
    evaluated: bool
    n_train_rows: int
    n_train_queries: int
    n_scored_rows: int
    n_queries: int
    load_seconds: float
    train_seconds: float
    seconds_per_tree: float
    score_seconds: float
    total_seconds: float
    sampled_peak_rss_gb: float
    process_max_rss_gb: float
    model_sha256: str
    runtime: LightGBMRuntime
    git_commit: str
    git_dirty: bool
    finished_at: str


def trial_row(record: RankerTrialRecord) -> dict[str, str]:
    """One row of the trial table (all values as text; floats with repr())."""
    return {
        "phase": record.phase,
        "experiment": record.experiment,
        "step": record.step,
        "config_id": record.config_id,
        "fold": record.fold,
        "train_folds": " ".join(record.train_folds),
        "params_json": json.dumps(record.params, sort_keys=True),
        "config_json": json.dumps(record.config, sort_keys=True),
        "histogram_mode": record.histogram_mode,
        "num_trees": str(record.num_trees),
        "evaluated": str(record.evaluated),
        "n_train_rows": str(record.n_train_rows),
        "n_train_queries": str(record.n_train_queries),
        "n_scored_rows": str(record.n_scored_rows),
        "n_queries": str(record.n_queries),
        "load_seconds": repr(record.load_seconds),
        "train_seconds": repr(record.train_seconds),
        "seconds_per_tree": repr(record.seconds_per_tree),
        "score_seconds": repr(record.score_seconds),
        "total_seconds": repr(record.total_seconds),
        "sampled_peak_rss_gb": repr(record.sampled_peak_rss_gb),
        "process_max_rss_gb": repr(record.process_max_rss_gb),
        "model_sha256": record.model_sha256,
        **record.runtime.columns(),
        "git_commit": record.git_commit,
        "git_dirty": str(record.git_dirty),
        "finished_at": record.finished_at,
    }


def append_trial(path: Path, record: RankerTrialRecord) -> None:
    """Append one trial row. Existing rows are never changed."""
    append_row(path, TRIAL_COLUMNS, trial_row(record))


def append_m7_run_row(path: Path, row: Mapping[str, str], runtime: LightGBMRuntime) -> None:
    """Append one row to results/m7_runs.csv: a run_records row plus the runtime columns.

    Args:
        path: results/m7_runs.csv.
        row: Output of hmrec.run_records.run_row.
        runtime: The LightGBM / OpenMP runtime of this process.
    """
    append_row(path, M7_RUN_COLUMNS, {**row, **runtime.columns()})


def read_trial_rows(path: Path) -> list[dict[str, str]]:
    """All trial rows as text dicts; empty if the file does not exist yet."""
    return read_rows(path, TRIAL_COLUMNS)


def find_trial(
    rows: Sequence[Mapping[str, str]],
    experiment: str,
    config_id: str,
    fold: str,
    step: str | None = None,
) -> Mapping[str, str] | None:
    """The row of one trial, or None if it has not run.

    In the grid a configuration runs once per fold, whatever the step: the
    default of a later step is the winner of an earlier one and is NOT trained
    again, so the grid looks trials up without a step. The speed test runs the
    same configuration in several steps (timing only, then evaluated) and passes
    the step.

    Raises:
        ValueError: If the trial is recorded more than once.
    """
    matches = [
        row
        for row in rows
        if (row["experiment"], row["config_id"], row["fold"]) == (experiment, config_id, fold)
        and (step is None or row["step"] == step)
    ]
    if len(matches) > 1:
        raise ValueError(f"trial ({experiment}, {config_id}, {fold}, {step}) is recorded twice")
    return matches[0] if matches else None


# ---------------------------------------------------------------- metric rows


@dataclass(frozen=True)
class MetricRow:
    """A fold's metric values of one configuration with the first k trees.

    Attributes:
        experiment: "full", "minus_r6" or "rrf_baseline".
        config_id: The configuration ("rrf_baseline" for the baseline).
        fold: The evaluated fold.
        num_trees: k (0 for the baseline).
        n_queries: Queries of the fold.
        scores: metric name -> mean over the fold's queries.
        train_metric: Training-set NDCG@12 with k trees, if it was computed.
    """

    experiment: str
    config_id: str
    fold: str
    num_trees: int
    n_queries: int
    scores: dict[str, float]
    train_metric: float | None


def metric_columns(metrics: Sequence[str]) -> tuple[str, ...]:
    """Header of a metric table for these metrics."""
    return (
        "experiment",
        "config_id",
        "fold",
        "num_trees",
        "n_queries",
        *metrics,
        TRAIN_METRIC_COLUMN,
        "git_commit",
        "finished_at",
    )


def append_metric_row(
    path: Path, row: MetricRow, metrics: Sequence[str], git_commit: str, finished_at: str
) -> None:
    """Append one metric row (floats with repr(), so they read back exactly)."""
    if list(row.scores) != list(metrics):
        raise ValueError(f"row has metrics {list(row.scores)}, expected {list(metrics)}")
    train_metric = NOT_COMPUTED if row.train_metric is None else repr(row.train_metric)
    text_row = {
        "experiment": row.experiment,
        "config_id": row.config_id,
        "fold": row.fold,
        "num_trees": str(row.num_trees),
        "n_queries": str(row.n_queries),
        **{metric: repr(row.scores[metric]) for metric in metrics},
        TRAIN_METRIC_COLUMN: train_metric,
        "git_commit": git_commit,
        "finished_at": finished_at,
    }
    append_row(path, metric_columns(metrics), text_row)


def read_metric_rows(path: Path, metrics: Sequence[str]) -> list[MetricRow]:
    """All metric rows; empty if the file does not exist yet."""
    rows = read_rows(path, metric_columns(metrics))
    return [
        MetricRow(
            experiment=row["experiment"],
            config_id=row["config_id"],
            fold=row["fold"],
            num_trees=int(row["num_trees"]),
            n_queries=int(row["n_queries"]),
            scores={metric: float(row[metric]) for metric in metrics},
            train_metric=(
                None
                if row[TRAIN_METRIC_COLUMN] == NOT_COMPUTED
                else float(row[TRAIN_METRIC_COLUMN])
            ),
        )
        for row in rows
    ]


# ---------------------------------------------------------------- curves and pooled scores


def config_rows(rows: Sequence[MetricRow], experiment: str, config_id: str) -> list[MetricRow]:
    """The metric rows of one configuration of one experiment."""
    return [row for row in rows if row.experiment == experiment and row.config_id == config_id]


def rows_at(
    rows: Sequence[MetricRow], folds: Sequence[str], num_trees: int
) -> dict[str, MetricRow]:
    """Fold -> the row with `num_trees` trees, for exactly the given folds.

    Args:
        rows: The metric rows of ONE configuration (see config_rows).
        folds: The folds that must all be present.
        num_trees: k.

    Raises:
        ValueError: If a fold is missing or recorded twice at this k.
    """
    by_fold: dict[str, MetricRow] = {}
    for row in rows:
        if row.num_trees != num_trees or row.fold not in folds:
            continue
        if row.fold in by_fold:
            raise ValueError(f"fold {row.fold} is recorded twice at {num_trees} trees")
        by_fold[row.fold] = row
    missing = [fold for fold in folds if fold not in by_fold]
    if missing:
        raise ValueError(f"no metric row at {num_trees} trees for folds {missing}")
    return {fold: by_fold[fold] for fold in folds}


def pooled_score(fold_rows: Mapping[str, MetricRow], metric: str) -> float:
    """Query-weighted mean of the folds' scores (module docstring)."""
    total_queries = sum(row.n_queries for row in fold_rows.values())
    if total_queries <= 0:
        raise ValueError("the folds have no queries")
    weighted = math.fsum(row.n_queries * row.scores[metric] for row in fold_rows.values())
    return weighted / total_queries


def equal_fold_score(fold_rows: Mapping[str, MetricRow], metric: str) -> float:
    """Plain mean of the folds' scores (reported only, proposal §4.1)."""
    return math.fsum(row.scores[metric] for row in fold_rows.values()) / len(fold_rows)


def complete_tree_counts(rows: Sequence[MetricRow], folds: Sequence[str]) -> list[int]:
    """The numbers of trees for which EVERY fold has a row, ascending."""
    folds_at: dict[int, set[str]] = {}
    for row in rows:
        if row.fold in folds:
            folds_at.setdefault(row.num_trees, set()).add(row.fold)
    return sorted(k for k, present in folds_at.items() if present == set(folds))


def pooled_curve(rows: Sequence[MetricRow], folds: Sequence[str], metric: str) -> dict[int, float]:
    """Number of trees -> pooled score, for every k that all folds have.

    Args:
        rows: The metric rows of ONE configuration.
        folds: The folds to pool (V1 ... V4).
        metric: e.g. "map@12".
    """
    return {
        k: pooled_score(rows_at(rows, folds, k), metric) for k in complete_tree_counts(rows, folds)
    }


def scores_at(
    rows: Sequence[MetricRow], folds: Sequence[str], metric: str, num_trees: int, trial_id: str
) -> TrialScores:
    """The per-fold and pooled scores of one configuration with `num_trees` trees."""
    fold_rows = rows_at(rows, folds, num_trees)
    return TrialScores(
        trial_id=trial_id,
        fold_scores={fold: row.scores[metric] for fold, row in fold_rows.items()},
        pooled_score=pooled_score(fold_rows, metric),
    )
