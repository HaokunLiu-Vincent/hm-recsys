"""Run the descriptive diagnostics: compute the four tables, then write them.

    A  decomposition   results/t_diag_decomposition.csv   result tables only
    B  concentration   results/t_diag_concentration.csv   labels + history before the anchor
    C  query groups    results/t_diag_query_groups.csv    saved files + labels
    D  bootstrap       results/t_diag_bootstrap.csv       the per-query values of C

Order of a run:
    1. If a T fold's labels are needed, pass the gate and RECORD the read
       (hmrec.diagnostics.access) before any label file is opened.
    2. Compute all four tables in memory. Every per-query value that is
       recomputed is checked against the recorded fold mean (tolerance 1e-12).
       A failed check stops the run before anything is written.
    3. Write the four tables (append-only).

Nothing is trained, scored with a model, selected or changed. The M8 result
tables are read, never written.
"""

import math
from dataclasses import dataclass

import polars as pl

from hmrec.data.day_index import date_to_day_index
from hmrec.data.folds import Fold
from hmrec.diagnostics.access import issue_diagnostic_access
from hmrec.diagnostics.bootstrap import (
    BASELINE_MAP,
    BOOTSTRAP_COLUMNS,
    LAMBDAMART_MAP,
    bootstrap_rows,
    cluster_bootstrap,
    customer_totals,
)
from hmrec.diagnostics.concentration import (
    concentration,
    concentration_columns,
    concentration_row,
    first_sale_days,
)
from hmrec.diagnostics.context import DiagnosticsContext
from hmrec.diagnostics.decomposition import decomposition_columns, decomposition_row
from hmrec.diagnostics.inputs import (
    read_lines_before_anchor,
    read_queries,
    read_ranker_per_query,
    read_union_hits,
    read_wide_summary,
)
from hmrec.diagnostics.query_groups import (
    BASELINE_AP,
    GROUPS,
    QUERY_GROUP,
    RANKER_AP,
    UNION_RECALL,
    assign_query_groups,
    build_per_query,
    column_mean,
    query_group_columns,
    query_group_rows,
)
from hmrec.diagnostics.recorded import RecordedFold, check_close, recorded_fold
from hmrec.diagnostics.records import append_rows
from hmrec.eval.diagnostic_labels import (
    RECORDED_PHASE,
    DiagnosticAccess,
    label_pair_counts,
    load_diagnostic_evaluator,
)
from hmrec.logging_utils import get_logger

LOGGER = get_logger("diagnostics.run")

Rows = list[dict[str, str]]


@dataclass(frozen=True)
class DiagnosticsTables:
    """The rows of the four tables of one run (without the provenance columns)."""

    decomposition: Rows
    concentration: Rows
    query_groups: Rows
    bootstrap: Rows


def needs_t_labels(context: DiagnosticsContext) -> bool:
    """True if a table that reads labels covers the T phase (table A reads no label)."""
    config = context.config
    label_phases = (
        *config.concentration_phases,
        *config.query_group_phases,
        *config.bootstrap_phases,
    )
    return RECORDED_PHASE in label_phases


# ---------------------------------------------------------------- A and B


def decomposition_table(context: DiagnosticsContext) -> Rows:
    """Table A: one row per fold, from the recorded result tables."""
    return [
        decomposition_row(recorded_fold(context, fold), context.map_metric, context.recall_metric)
        for fold in context.folds_of(context.config.decomposition_phases)
    ]


def concentration_table(context: DiagnosticsContext, access: DiagnosticAccess | None) -> Rows:
    """Table B: one row per fold. Reads each fold's labels and the lines before its anchor."""
    config = context.config
    rows: Rows = []
    for fold in context.folds_of(config.concentration_phases):
        evaluator = load_diagnostic_evaluator(fold, context.paths.labels_file(fold.name), access)
        first_sale = first_sale_days(read_lines_before_anchor(context, fold))
        stats = concentration(
            label_pair_counts(evaluator),
            first_sale,
            date_to_day_index(fold.anchor, context.preprocess.day_zero),
            config.top_article_counts,
            config.new_article_days,
        )
        rows.append(
            concentration_row(
                fold.name, fold.phase, fold.anchor.isoformat(), stats, config.new_article_days
            )
        )
        LOGGER.info("concentration | %s | %d label pairs", fold.name, stats.n_label_pairs)
    return rows


# ---------------------------------------------------------------- per-query values (C and D)


def check_group_counts(per_query: pl.DataFrame, recorded: RecordedFold) -> None:
    """Fail unless the three groups together hold exactly the recorded number of queries.

    Raises:
        ValueError: If a query has an unknown group, or the group sizes do not add
            up to the n_queries the result tables record for the fold.
    """
    counts = dict(per_query.group_by(QUERY_GROUP).len().iter_rows())
    unknown = sorted(set(counts) - set(GROUPS))
    if unknown:
        raise ValueError(f"fold {recorded.fold}: unknown query groups {unknown}")
    total = sum(counts.get(group, 0) for group in GROUPS)
    if total != recorded.n_queries:
        raise ValueError(
            f"fold {recorded.fold}: the query groups hold {total} queries, but the result "
            f"tables record {recorded.n_queries}"
        )


def check_per_query(
    context: DiagnosticsContext, per_query: pl.DataFrame, recorded: RecordedFold
) -> None:
    """Fail unless the per-query values reproduce the fold's recorded numbers.

    Checked: the group sizes add up to the recorded n_queries; the means of the
    union recall, the Ranker's AP and the baseline's AP equal the recorded fold
    means (hmrec.diagnostics.recorded.check_close, tolerance 1e-12).
    """
    check_group_counts(per_query, recorded)
    fold, metric = recorded.fold, context.map_metric
    check_close(f"{fold} union recall", column_mean(per_query, UNION_RECALL), recorded.union_recall)
    check_close(
        f"{fold} Ranker {metric}", column_mean(per_query, RANKER_AP), recorded.ranker[metric]
    )
    check_close(
        f"{fold} RRF baseline {metric}",
        column_mean(per_query, BASELINE_AP),
        recorded.baseline[metric],
    )


def fold_per_query(
    context: DiagnosticsContext, fold: Fold, access: DiagnosticAccess | None
) -> pl.DataFrame:
    """The checked per-query table of one V or T fold (hmrec.diagnostics.query_groups).

    The Ranker's AP is the SAVED per-query value. The RRF baseline's AP and the
    union recall were not saved per query; they are recomputed from the frozen
    wide table and the candidate files with the project's own functions, and
    their fold means must equal the recorded ones.
    """
    recorded = recorded_fold(context, fold)
    evaluator = load_diagnostic_evaluator(fold, context.paths.labels_file(fold.name), access)
    wide = read_wide_summary(context, fold)
    per_query = build_per_query(
        queries=read_queries(context, fold),
        groups=assign_query_groups(wide.days, context.config.recent_purchase_days),
        union_recall=evaluator.union_recall_of_hits(read_union_hits(context, fold, evaluator)),
        ranker_ap=read_ranker_per_query(context, fold),
        baseline_ap=evaluator.per_query(wide.baseline_top, context.map_metric),
    )
    check_per_query(context, per_query, recorded)
    LOGGER.info(
        "per-query values | %s | %d queries, all checks passed", fold.name, per_query.height
    )
    return per_query


def per_query_tables(
    context: DiagnosticsContext, access: DiagnosticAccess | None
) -> dict[str, pl.DataFrame]:
    """Fold name -> per-query table, for every fold tables C or D need, in time order."""
    phases = {*context.config.query_group_phases, *context.config.bootstrap_phases}
    return {
        fold.name: fold_per_query(context, fold, access) for fold in context.folds_of(tuple(phases))
    }


# ---------------------------------------------------------------- C and D


def query_group_table(context: DiagnosticsContext, per_query: dict[str, pl.DataFrame]) -> Rows:
    """Table C: three rows per fold."""
    rows: Rows = []
    for fold in context.folds_of(context.config.query_group_phases):
        rows.extend(
            query_group_rows(per_query[fold.name], fold.name, fold.phase, context.map_metric)
        )
    return rows


def recorded_pooled(recorded: list[RecordedFold], scores: str, metric: str) -> float:
    """Query-weighted mean of the folds' recorded scores (proposal §4.1).

    pooled = sum_f n_queries_f * mean_f / sum_f n_queries_f, as in hmrec.hpo.v_records.

    Args:
        recorded: The recorded numbers of the folds to pool.
        scores: "ranker" or "baseline".
        metric: e.g. "map@12".
    """
    total = sum(fold.n_queries for fold in recorded)
    weighted = math.fsum(fold.n_queries * getattr(fold, scores)[metric] for fold in recorded)
    return weighted / total


def bootstrap_table(context: DiagnosticsContext, per_query: dict[str, pl.DataFrame]) -> Rows:
    """Table D: three rows per phase (both systems and their difference)."""
    config, metric = context.config, context.map_metric
    rows: Rows = []
    for phase in config.bootstrap_phases:
        folds = context.folds_of((phase,))
        pooled = pl.concat([per_query[fold.name] for fold in folds])
        totals = customer_totals(pooled)
        intervals = cluster_bootstrap(
            totals, config.bootstrap_replicates, config.bootstrap_seed, config.confidence_level
        )
        recorded = [recorded_fold(context, fold) for fold in folds]
        check_close(
            f"{phase} pooled Ranker {metric}",
            intervals[LAMBDAMART_MAP].estimate,
            recorded_pooled(recorded, "ranker", metric),
        )
        check_close(
            f"{phase} pooled RRF baseline {metric}",
            intervals[BASELINE_MAP].estimate,
            recorded_pooled(recorded, "baseline", metric),
        )
        rows.extend(
            bootstrap_rows(
                phase,
                tuple(fold.name for fold in folds),
                metric,
                totals,
                intervals,
                config.bootstrap_replicates,
                config.bootstrap_seed,
                config.confidence_level,
            )
        )
        LOGGER.info("bootstrap | phase %s | %d customers", phase, totals.height)
    return rows


# ---------------------------------------------------------------- the run


def compute_tables(
    context: DiagnosticsContext, access: DiagnosticAccess | None
) -> DiagnosticsTables:
    """Compute the four tables in memory. Nothing is written."""
    decomposition = decomposition_table(context)
    concentration_rows = concentration_table(context, access)
    per_query = per_query_tables(context, access)
    return DiagnosticsTables(
        decomposition=decomposition,
        concentration=concentration_rows,
        query_groups=query_group_table(context, per_query),
        bootstrap=bootstrap_table(context, per_query),
    )


def write_tables(
    context: DiagnosticsContext, tables: DiagnosticsTables, run_started_at: str
) -> None:
    """Append the four tables to results/t_diag_*.csv (the only files written)."""
    paths, config = context.paths, context.config
    commit = context.stamp.git_commit
    outputs = (
        (
            paths.t_diag_decomposition_file,
            decomposition_columns(context.map_metric, context.recall_metric),
            tables.decomposition,
        ),
        (
            paths.t_diag_concentration_file,
            concentration_columns(config.top_article_counts, config.new_article_days),
            tables.concentration,
        ),
        (
            paths.t_diag_query_groups_file,
            query_group_columns(context.map_metric),
            tables.query_groups,
        ),
        (paths.t_diag_bootstrap_file, BOOTSTRAP_COLUMNS, tables.bootstrap),
    )
    for path, columns, rows in outputs:
        append_rows(path, columns, rows, commit, run_started_at)
        LOGGER.info("wrote %d rows to %s", len(rows), path)


def run_diagnostics(context: DiagnosticsContext, now: str, script: str) -> DiagnosticsTables:
    """One run of the diagnostics (module docstring).

    Args:
        context: The context of this process.
        now: The current time as ISO text: the run's start, written to the label
            access record and to every row.
        script: The calling script, e.g. "scripts/m8_diagnostics.py" (for the record).

    Returns:
        The rows that were written.

    Raises:
        DiagnosticsRefused: If T labels are needed and the gate refuses. Nothing
            was read or written then.
        ValueError: If a check against the recorded numbers fails. The four
            tables are not written then (the label access row stays: the labels
            were read).
    """
    access = issue_diagnostic_access(context, now, script) if needs_t_labels(context) else None
    tables = compute_tables(context, access)
    write_tables(context, tables, now)
    return tables
