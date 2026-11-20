"""The chosen DeepFM configuration next to the M8 record on T1-T4 (report only; §16.12).

Two arms:

    the DeepFM arm      the chosen configuration, every seed, from the per-query
                        values its formal trainings saved when they were evaluated
    the M8 record       LightGBM lambdarank, one seed, from the per-query values the
                        formal run of M8 saved (never trained again)

Only saved values are read: no label is read and no model is loaded.

    1. Arm table. DeepFM: per seed the pooled metrics (the mean over ALL queries
       of the T folds) and MAP per fold; then the mean, the sample standard
       deviation and the range of the seeds (hmrec.deepfm.compare.arm_rows). The
       M8 record: one row with the same columns.
    2. Contrast. The paired cluster bootstrap over customers used on V
       (hmrec.deepfm.compare.contrast_rows; proposal §10.2): DeepFM minus the M8
       record, on the seed mean of DeepFM and on the single seed of both. The M8
       record has one seed, so its "seed mean" is that seed.
    3. V and T side by side: pooled MAP of the two arms on V1-V4 (read from
       results/m9_ext_arms.csv) and on T1-T4.

Before the saved values of M8 are used, their mean per fold must equal the
recorded value in results/t_phase_metrics.csv (check_m8_values).

This module imports neither torch nor LightGBM.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from hmrec.deepfm.compare import (
    ALL_QUERIES,
    CUSTOMER_IDX,
    FOLD,
    QUERY_GROUP,
    QUERY_ID,
    SEED_MEAN,
    ArmRuns,
    CompareSettings,
    arm_fold_values,
    arm_rows,
    contrast_rows,
    fold_table,
    seed_basis,
    value_column,
)
from hmrec.t_eval.plan import FORMAL_AREA, TEvalContext

EPOCH = "epoch"
NUM_TREES = "num_trees"
PER_QUERY_FILE = "per_query.parquet"
M8_AREA = "t_phase"
M8_EXPERIMENT = "full"
MEAN_ROW = "mean"


def deepfm_arm_runs(context: TEvalContext) -> ArmRuns:
    """The saved per-query files of the formal trainings of the chosen configuration."""
    paths, config = context.paths, context.config
    files = {
        seed: {
            fold: paths.deepfm_trial_dir(FORMAL_AREA, config.config_id, seed, fold) / PER_QUERY_FILE
            for fold in config.target_folds
        }
        for seed in config.seeds
    }
    return ArmRuns(config.arm, EPOCH, dict.fromkeys(config.seeds, config.epochs), files)


def m8_arm_runs(context: TEvalContext, locked_config_id: str) -> ArmRuns:
    """The saved per-query files of the M8 formal run (LightGBM lambdarank, one seed).

    Args:
        context: The context of this process.
        locked_config_id: The configuration id of the locked Ranker (from its lock file).
    """
    paths, config = context.paths, context.config
    files = {
        config.m8_record_seed: {
            fold: paths.ranker_per_query_file(M8_AREA, M8_EXPERIMENT, locked_config_id, fold)
            for fold in config.target_folds
        }
    }
    steps = {config.m8_record_seed: config.m8_record_num_trees}
    return ArmRuns(config.m8_record_arm, NUM_TREES, steps, files)


def per_query_table(
    context: TEvalContext, arms: Sequence[ArmRuns], metrics: Sequence[str]
) -> pl.DataFrame:
    """One row per labelled query of the T folds with every arm's saved values and its customer.

    The query groups of the V comparison are not computed on T: every query gets
    the group "all".

    Returns:
        Columns query_id, fold, customer_idx, query_group, then the value columns
        of every arm (hmrec.deepfm.compare.value_column); folds in time order.

    Raises:
        ValueError: If the arms were not evaluated on exactly the same queries.
    """
    tables = []
    for fold in context.config.target_folds:
        queries = pl.read_parquet(
            context.paths.queries_file(fold), columns=[QUERY_ID, CUSTOMER_IDX]
        )
        groups = queries.select(QUERY_ID, pl.lit(ALL_QUERIES).alias(QUERY_GROUP))
        values = [arm_fold_values(arm, fold, metrics) for arm in arms]
        tables.append(fold_table(fold, queries, groups, values))
    return pl.concat(tables)


def _mean(table: pl.DataFrame, column: str) -> float:
    """Mean of a column over the rows, summed with math.fsum in (fold, query_id) order."""
    values = table.sort([FOLD, QUERY_ID])[column].to_list()
    return math.fsum(values) / len(values)


def single_seed_rows(
    per_query: pl.DataFrame, arm: ArmRuns, metrics: Sequence[str]
) -> list[dict[str, str]]:
    """The arm-table row of an arm with ONE seed, with the columns of arm_rows.

    Raises:
        ValueError: If the arm does not have exactly one seed.
    """
    if len(arm.seeds) != 1:
        raise ValueError(f"{arm.arm} has {len(arm.seeds)} seeds, expected exactly one")
    seed = arm.seeds[0]
    basis = seed_basis(seed)
    main = metrics[0]
    row = {
        "arm": arm.arm,
        "row": basis,
        "step_column": arm.step_column,
        "step": str(arm.steps[seed]),
        "n_queries": str(per_query.height),
    }
    for metric in metrics:
        row[metric] = repr(_mean(per_query, value_column(arm.arm, basis, metric)))
    for fold in per_query[FOLD].unique(maintain_order=True).to_list():
        in_fold = per_query.filter(pl.col(FOLD) == fold)
        row[f"{main}|{fold}"] = repr(_mean(in_fold, value_column(arm.arm, basis, main)))
    return [row]


def check_m8_values(
    arm_row: Mapping[str, str],
    recorded: Sequence[Mapping[str, str]],
    locked_config_id: str,
    folds: Sequence[str],
    metric: str,
    tolerance: float,
) -> None:
    """Fail unless the saved per-query values of M8 give M8's recorded metric of every fold.

    Args:
        arm_row: The M8 record's row of the arm table (single_seed_rows).
        recorded: Rows of results/t_phase_metrics.csv.
        locked_config_id: The configuration id of the locked Ranker.
        folds: The T folds.
        metric: The main metric.
        tolerance: Largest allowed absolute difference (1e-12).

    Raises:
        ValueError: If a fold is not recorded exactly once or a value differs.
    """
    for fold in folds:
        wanted = (M8_EXPERIMENT, locked_config_id, fold)
        rows = [
            row for row in recorded if (row["experiment"], row["config_id"], row["fold"]) == wanted
        ]
        if len(rows) != 1:
            raise ValueError(f"results/t_phase_metrics.csv has {len(rows)} rows for {wanted}")
        saved, written = float(arm_row[f"{metric}|{fold}"]), float(rows[0][metric])
        if abs(saved - written) > tolerance:
            raise ValueError(
                f"{fold}: the saved per-query values of M8 give {metric} = {saved!r}, but M8 "
                f"recorded {written!r}"
            )


def _v_value(rows: Sequence[Mapping[str, str]], arm: str, row_name: str, metric: str) -> float:
    """One pooled value of an arm on V from results/m9_ext_arms.csv.

    Raises:
        ValueError: If the table has not exactly one such row.
    """
    found = [row for row in rows if (row["arm"], row["row"]) == (arm, row_name)]
    if len(found) != 1:
        raise ValueError(f"results/m9_ext_arms.csv has {len(found)} rows for ({arm}, {row_name})")
    return float(found[0][metric])


def _t_value(rows: Sequence[Mapping[str, str]], arm: str, row_name: str, metric: str) -> float:
    """One pooled value of an arm on T from the arm table of this comparison."""
    found = [row for row in rows if (row["arm"], row["row"]) == (arm, row_name)]
    if len(found) != 1:
        raise ValueError(f"the T arm table has {len(found)} rows for ({arm}, {row_name})")
    return float(found[0][metric])


def v_and_t_rows(
    context: TEvalContext,
    v_rows: Sequence[Mapping[str, str]],
    t_rows: Sequence[Mapping[str, str]],
    metric: str,
) -> list[dict[str, str]]:
    """Pooled values of the two arms on V1-V4 and on T1-T4, side by side.

    DeepFM: its seed means on both sides. The M8 record has one seed on T; it is
    shown with that seed on both sides, and once more with its seed mean on V.

    Args:
        context: The context of this process.
        v_rows: Rows of results/m9_ext_arms.csv (the eight-arm table on V).
        t_rows: The arm table of this comparison.
        metric: The main metric.

    Returns:
        One dict per row: arm, metric, v_basis, v_n_seeds, v_value, t_basis,
        t_n_seeds, t_value (floats with repr()).
    """
    config = context.config
    n_seeds = len(config.seeds)
    m8_basis = seed_basis(config.m8_record_seed)
    lines = (
        (config.arm, MEAN_ROW, n_seeds, MEAN_ROW, n_seeds),
        (config.m8_record_arm, m8_basis, 1, m8_basis, 1),
        (config.m8_record_arm, MEAN_ROW, n_seeds, m8_basis, 1),
    )
    rows = []
    for arm, v_row, v_seeds, t_row, t_seeds in lines:
        rows.append(
            {
                "arm": arm,
                "metric": metric,
                "v_basis": SEED_MEAN if v_row == MEAN_ROW else v_row,
                "v_n_seeds": str(v_seeds),
                "v_value": repr(_v_value(v_rows, arm, v_row, metric)),
                "t_basis": SEED_MEAN if t_row == MEAN_ROW else t_row,
                "t_n_seeds": str(t_seeds),
                "t_value": repr(_t_value(t_rows, arm, t_row, metric)),
            }
        )
    return rows


@dataclass(frozen=True)
class TComparison:
    """The three tables of the comparison on T, as rows of text.

    Attributes:
        arms: The arm table: DeepFM (seeds, mean, std, range), then the M8 record.
        contrasts: The bootstrap intervals of DeepFM minus the M8 record.
        v_and_t: The two arms on V and on T.
        n_queries: Queries of the T folds together.
    """

    arms: list[dict[str, str]]
    contrasts: list[dict[str, str]]
    v_and_t: list[dict[str, str]]
    n_queries: int


def compare_on_t(
    context: TEvalContext,
    settings: CompareSettings,
    locked_config_id: str,
    m8_metric_rows: Sequence[Mapping[str, str]],
    v_arm_rows: Sequence[Mapping[str, str]],
    tolerance: float,
) -> TComparison:
    """Compare the chosen DeepFM configuration with the M8 record on the T folds.

    Args:
        context: The context of this process.
        settings: Metrics, the single-seed basis and the bootstrap settings;
            settings.seed must be the seed of the M8 record.
        locked_config_id: The configuration id of the locked Ranker.
        m8_metric_rows: Rows of results/t_phase_metrics.csv.
        v_arm_rows: Rows of results/m9_ext_arms.csv.
        tolerance: Tolerance of check_m8_values.

    Raises:
        ValueError: If settings.seed is not the M8 record's seed, a saved file is
            missing a model, the arms were evaluated on other queries, or the M8
            values do not give the recorded metrics.
    """
    config = context.config
    if settings.seed != config.m8_record_seed:
        raise ValueError(
            f"the single-seed basis must be the M8 record's seed {config.m8_record_seed}, "
            f"got {settings.seed}"
        )
    deepfm, m8 = deepfm_arm_runs(context), m8_arm_runs(context, locked_config_id)
    metrics, main = settings.metrics, settings.metrics[0]
    per_query = per_query_table(context, (deepfm, m8), metrics)
    m8_rows = single_seed_rows(per_query, m8, metrics)
    check_m8_values(
        m8_rows[0], m8_metric_rows, locked_config_id, config.target_folds, main, tolerance
    )
    arms = [*arm_rows(per_query, [deepfm], metrics), *m8_rows]
    contrasts = contrast_rows(per_query, settings, ((deepfm.arm, m8.arm),))
    return TComparison(
        arms=arms,
        contrasts=contrasts,
        v_and_t=v_and_t_rows(context, v_arm_rows, arms, main),
        n_queries=per_query.height,
    )
