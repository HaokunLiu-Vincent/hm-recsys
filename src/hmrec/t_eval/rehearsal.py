"""Compare a rehearsal of the M9 evaluation with the V4 records of the configuration (§16.12).

The rehearsal ran the new code path with the rehearsal fold (V4) as the target
week, for every seed of the evaluation. Each result must be BIT-IDENTICAL to the
record the configuration has for that fold and seed from the V phase:

    code_state   the rehearsal training is recorded once, at HEAD, on a clean tree
    values       the number of queries, every metric and the mean training loss
                 (float64 ==), against results/deepfm_metrics.csv
    top_k        the saved Top-K, row by row, with its float64 scores
    per_query    the saved per-query metric values

The outcome is one row in results/m9_t_rehearsal_check.csv: "passed" only if every
check of every seed passed. The gate (hmrec.t_eval.gate) reads that row. Nothing
here reads a label: only result tables and saved prediction files.

This module imports neither torch nor LightGBM.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from hmrec.csv_records import append_row
from hmrec.t_eval.gate import FAILED, PASSED, REHEARSAL_CHECK_COLUMNS
from hmrec.t_eval.plan import STAGE_REHEARSAL, TEvalContext, area_of, read_csv_table

V_AREA = "v_phase"
PREDICTIONS_FILE = "predictions.parquet"
PER_QUERY_FILE = "per_query.parquet"
N_QUERIES = "n_queries"
TRAIN_LOSS = "train_loss"


@dataclass(frozen=True)
class Check:
    """One comparison of the rehearsal with the V record.

    Attributes:
        name: "seed {seed}: {what}", e.g. "seed 42: values".
        passed: True if the two are exactly equal.
        detail: What was compared, or what differs.
    """

    name: str
    passed: bool
    detail: str


def trial_rows_of(
    rows: Sequence[Mapping[str, str]], area: str, config_id: str, seed: int, fold: str
) -> list[Mapping[str, str]]:
    """The rows of one training in a trial table or a metric table (they share these columns)."""
    wanted = (area, config_id, str(seed), fold)
    return [
        row for row in rows if (row["area"], row["config_id"], row["seed"], row["fold"]) == wanted
    ]


def compare_values(
    name: str, new: Mapping[str, str], recorded: Mapping[str, str], metrics: Sequence[str]
) -> Check:
    """Compare the rehearsal's metric row with the V record's, exactly.

    Args:
        name: The check's name.
        new: The rehearsal's row of results/m9_t_metrics.csv.
        recorded: The V record's row of results/deepfm_metrics.csv.
        metrics: The metrics to compare (float64 ==).

    Returns:
        The check; its detail names every value that differs.
    """
    different = []
    for column in ("epoch", N_QUERIES):
        if int(new[column]) != int(recorded[column]):
            different.append(f"{column}: {new[column]} != {recorded[column]}")
    for column in (*metrics, TRAIN_LOSS):
        if float(new[column]) != float(recorded[column]):
            different.append(f"{column}: {new[column]} != {recorded[column]}")
    if different:
        return Check(name, False, "; ".join(different))
    compared = ", ".join(f"{column} = {new[column]}" for column in (*metrics, TRAIN_LOSS))
    return Check(name, True, f"{N_QUERIES} = {new[N_QUERIES]}, {compared}")


def compare_tables(name: str, new: pl.DataFrame, recorded: pl.DataFrame) -> Check:
    """Compare two saved tables exactly: columns, dtypes, row order and every value.

    Returns:
        The check; its detail says what differs first (the schema, the number of
        rows, or how many rows differ in some column).
    """
    if new.schema != recorded.schema:
        return Check(name, False, f"schema {dict(new.schema)} != {dict(recorded.schema)}")
    if new.height != recorded.height:
        return Check(name, False, f"{new.height} rows != {recorded.height} rows")
    if new.equals(recorded):
        return Check(name, True, f"{new.height} rows equal")
    different = max(int((new[column] != recorded[column]).sum()) for column in new.columns)
    return Check(name, False, f"up to {different} rows differ in a column")


def seed_checks(
    context: TEvalContext,
    seed: int,
    new_trials: Sequence[Mapping[str, str]],
    new_metrics: Sequence[Mapping[str, str]],
    recorded_metrics: Sequence[Mapping[str, str]],
    metrics: Sequence[str],
) -> list[Check]:
    """The four checks of one seed (module docstring).

    Args:
        context: The context of this process.
        seed: The seed.
        new_trials: Rows of results/m9_t_trials.csv.
        new_metrics: Rows of results/m9_t_metrics.csv.
        recorded_metrics: Rows of results/deepfm_metrics.csv (the V records).
        metrics: The metrics to compare.

    Returns:
        The checks. If the rehearsal training or the V record is not there
        exactly once (or the training ran on a dirty tree), only the failed
        code_state check is returned.
    """
    paths, config, commit = context.paths, context.config, context.stamp.git_commit
    fold, config_id = config.rehearsal_target_fold, config.config_id
    area = area_of(STAGE_REHEARSAL, commit)
    trials = trial_rows_of(new_trials, area, config_id, seed, fold)
    values = trial_rows_of(new_metrics, area, config_id, seed, fold)
    recorded = trial_rows_of(recorded_metrics, V_AREA, config_id, seed, fold)
    recorded = [row for row in recorded if int(row["epoch"]) == config.epochs]
    at_head = [row for row in trials if row["git_commit"] == commit and row["git_dirty"] == "False"]
    if len(trials) != 1 or len(at_head) != 1 or len(values) != 1 or len(recorded) != 1:
        detail = (
            f"{len(trials)} trial rows ({len(at_head)} at HEAD {commit[:12]} on a clean tree), "
            f"{len(values)} metric rows and {len(recorded)} V records; expected one of each"
        )
        return [Check(f"seed {seed}: code_state", False, detail)]
    checks = [Check(f"seed {seed}: code_state", True, f"recorded at HEAD {commit[:12]}, clean")]
    checks.append(compare_values(f"seed {seed}: values", values[0], recorded[0], metrics))
    new_dir = paths.deepfm_trial_dir(area, config_id, seed, fold)
    recorded_dir = paths.deepfm_trial_dir(V_AREA, config_id, seed, fold)
    for what, file_name in (("top_k", PREDICTIONS_FILE), ("per_query", PER_QUERY_FILE)):
        checks.append(
            compare_tables(
                f"seed {seed}: {what}",
                pl.read_parquet(new_dir / file_name),
                pl.read_parquet(recorded_dir / file_name),
            )
        )
    return checks


def check_rehearsal(context: TEvalContext, metrics: Sequence[str]) -> list[Check]:
    """All checks of the rehearsal at HEAD: every seed against its V record."""
    paths = context.paths
    new_trials = read_csv_table(paths.m9_t_trials_file)
    new_metrics = read_csv_table(paths.m9_t_metrics_file)
    recorded_metrics = read_csv_table(paths.deepfm_metrics_file)
    checks: list[Check] = []
    for seed in context.config.seeds:
        checks += seed_checks(context, seed, new_trials, new_metrics, recorded_metrics, metrics)
    return checks


def write_check_row(context: TEvalContext, checks: Sequence[Check], now: str) -> str:
    """Append the outcome of a rehearsal to results/m9_t_rehearsal_check.csv.

    Returns:
        "passed" if there are checks and every one passed, else "failed".
    """
    config = context.config
    failed = [check.name for check in checks if not check.passed]
    status = FAILED if failed or not checks else PASSED
    row = {
        "status": status,
        "target_fold": config.rehearsal_target_fold,
        "seeds": " ".join(str(seed) for seed in config.seeds),
        "config_id": config.config_id,
        "n_checks": str(len(checks)),
        "failed_checks": "; ".join(failed),
        "checks_json": json.dumps(
            {check.name: {"passed": check.passed, "detail": check.detail} for check in checks}
        ),
        "git_commit": context.stamp.git_commit,
        "git_dirty": str(context.stamp.git_dirty),
        "checked_at": now,
    }
    append_row(context.paths.m9_t_rehearsal_check_file, REHEARSAL_CHECK_COLUMNS, row)
    return status
