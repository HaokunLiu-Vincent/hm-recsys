"""The recorded numbers the diagnostics describe, and the check against them.

Everything here comes from result tables (results/*.csv). No label is read.

    V folds   results/v_phase_metrics.csv (Ranker, RRF baseline) and
              results/m6_candidates.csv (union recall)
    T folds   results/t_phase_metrics.csv and results/t_phase_recall.csv

The tables are only READ. The diagnostics never write to them.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hmrec.csv_records import read_rows
from hmrec.data.folds import Fold
from hmrec.diagnostics.context import DiagnosticsContext
from hmrec.eval.monitor import MONITOR_COLUMNS, NOT_COMPUTED, UNION
from hmrec.hpo.selection import SCORE_EQUALITY_TOLERANCE
from hmrec.hpo.v_records import (
    AREA_T_PHASE,
    AREA_V_PHASE,
    NO_TREES,
    RRF_BASELINE,
    config_rows,
    read_metric_rows,
    results_files,
    rows_at,
)
from hmrec.paths import ProjectPaths
from hmrec.ranker.ablation import FULL

# Phase -> the results area that holds the phase's Ranker trials and metric rows.
AREA_OF_PHASE: dict[str, str] = {"V": AREA_V_PHASE, "T": AREA_T_PHASE}


@dataclass(frozen=True)
class RecordedFold:
    """What the result tables record for one fold.

    Attributes:
        fold: Fold name.
        phase: "V" or "T".
        n_queries: Queries of the fold.
        union_recall: Mean over queries of the union recall of the six channels.
        ranker: Metric name -> the locked Ranker's fold mean (locked number of trees).
        baseline: Metric name -> the RRF baseline's fold mean.
    """

    fold: str
    phase: str
    n_queries: int
    union_recall: float
    ranker: dict[str, float]
    baseline: dict[str, float]


def area_of_phase(phase: str) -> str:
    """The results area of a phase ("v_phase" or "t_phase").

    Raises:
        ValueError: For a phase without recorded Ranker results (D, F).
    """
    if phase not in AREA_OF_PHASE:
        raise ValueError(f"phase {phase!r} has no recorded Ranker results; expected V or T")
    return AREA_OF_PHASE[phase]


def recall_file(paths: ProjectPaths, phase: str) -> Path:
    """The monitoring table that holds a phase's union recall."""
    area_of_phase(phase)  # refuses other phases
    return paths.t_phase_recall_file if phase == "T" else paths.m6_candidates_results_file


def recorded_union_recall(rows: Sequence[Mapping[str, str]], fold: str) -> tuple[float, int]:
    """(union recall, queries) of a fold from monitoring rows.

    Args:
        rows: Rows with MONITOR_COLUMNS (results/m6_candidates.csv or t_phase_recall.csv).
        fold: The fold.

    Raises:
        ValueError: Unless exactly one union row of the fold has a recall. (The M6
            table has a union row for T folds too, without a recall: it is skipped.)
    """
    matches = [
        row
        for row in rows
        if row["fold"] == fold and row["scope"] == UNION and row["recall"] != NOT_COMPUTED
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one recorded union recall of fold {fold}, found {len(matches)}"
        )
    return float(matches[0]["recall"]), int(matches[0]["n_queries"])


def recorded_fold(context: DiagnosticsContext, fold: Fold) -> RecordedFold:
    """The recorded numbers of one V or T fold.

    Raises:
        ValueError: If a row is missing or recorded twice, or the three sources
            disagree on the number of queries.
    """
    paths, ranker = context.paths, context.ranker
    metrics_file = results_files(paths, area_of_phase(fold.phase))[1]
    metric_rows = read_metric_rows(metrics_file, context.metrics)
    ranker_rows = config_rows(metric_rows, FULL, ranker.config_id)
    ranker_row = rows_at(ranker_rows, [fold.name], ranker.num_trees)[fold.name]
    baseline_rows = config_rows(metric_rows, RRF_BASELINE, RRF_BASELINE)
    baseline_row = rows_at(baseline_rows, [fold.name], NO_TREES)[fold.name]
    monitor_rows = read_rows(recall_file(paths, fold.phase), MONITOR_COLUMNS)
    union_recall, union_queries = recorded_union_recall(monitor_rows, fold.name)
    counts = {ranker_row.n_queries, baseline_row.n_queries, union_queries}
    if len(counts) != 1:
        raise ValueError(f"fold {fold.name}: the result tables disagree on n_queries: {counts}")
    return RecordedFold(
        fold=fold.name,
        phase=fold.phase,
        n_queries=ranker_row.n_queries,
        union_recall=union_recall,
        ranker=dict(ranker_row.scores),
        baseline=dict(baseline_row.scores),
    )


def check_close(name: str, computed: float, recorded: float) -> None:
    """Fail unless a recomputed number equals the recorded one (tolerance 1e-12, absolute).

    The diagnostics recompute per-query values from saved files. If their mean
    is not the recorded mean, the files are not the ones the record was made
    from, and nothing may be reported.

    Raises:
        ValueError: If |computed - recorded| > hmrec.hpo.selection.SCORE_EQUALITY_TOLERANCE.
    """
    if abs(computed - recorded) > SCORE_EQUALITY_TOLERANCE:
        raise ValueError(
            f"{name}: recomputed {computed!r} differs from the recorded {recorded!r} by more "
            f"than {SCORE_EQUALITY_TOLERANCE}"
        )
