"""Compare a rehearsal with the recorded M7 results (M8). It must be EXACTLY the same.

The rehearsal runs the M8 code path with the locked configuration on the
configured V fold (configs/m8_final.yaml), reading no label of a T fold. M7
predicted the same fold with the same configuration, so every number must be
the same, bit for bit. The checks:

    metrics       every metric of the full system at every checkpoint (100, ...,
                  the locked number of trees), of the RRF baseline, and of every
                  secondary experiment (minus_r6), against results/v_phase_metrics.csv:
                  the same number of queries and == on float64
    model         the first `num_trees` trees of the rehearsal model are the trees
                  of the M7 model (hmrec.ranker.train.first_trees_sha256); both
                  model files must be the recorded files (sha256)
    Top-K         the saved Top-12 at the locked number of trees: the same rows,
                  articles and float64 scores as M7's saved Top-12
    recall        the monitoring rows of the fold against results/m6_candidates.csv
    code          the rehearsal's trials ran with the code at HEAD, clean

One row per comparison run goes to results/m8_rehearsal_check.csv, with the
sha256 of the sample files the rehearsal trained on. hmrec.final.gate lets the
formal run start only if the latest row says "passed".

This is the one place in M8 that reads the V-phase result tables. It compares
recorded numbers of the LOCKED configuration on the rehearsal fold; it selects
nothing and takes no configuration from them.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from hmrec.csv_records import append_row, read_rows
from hmrec.eval.monitor import MONITOR_COLUMNS
from hmrec.final.context import MODE_REHEARSAL, FinalContext
from hmrec.final.gate import FAILED, PASSED, REHEARSAL_CHECK_COLUMNS
from hmrec.final.monitor import recall_file
from hmrec.final.trial import experiment_config_id
from hmrec.hpo.v_records import (
    AREA_M8_REHEARSAL,
    AREA_V_PHASE,
    NO_TREES,
    RRF_BASELINE,
    MetricRow,
    config_rows,
    find_trial,
    read_metric_rows,
    read_trial_rows,
    results_files,
)
from hmrec.locks import file_sha256
from hmrec.ranker.ablation import FULL
from hmrec.ranker.dataset import read_dataset_record
from hmrec.ranker.predict import checkpoints
from hmrec.ranker.train import first_trees_sha256, load_model

NUM_TREES = "num_trees"
# Columns of a monitoring row that describe the run, not the fold's candidates.
RUN_ONLY_MONITOR_COLUMNS = ("git_commit", "git_dirty", "finished_at")


@dataclass(frozen=True)
class RehearsalCheck:
    """One comparison of the rehearsal with M7.

    Attributes:
        name: What was compared, e.g. "full metrics at 500 trees".
        passed: True if it is exactly the same.
        detail: What differs (empty if nothing does).
    """

    name: str
    passed: bool
    detail: str


def model_experiments(context: FinalContext) -> tuple[str, ...]:
    """The experiments that train models: "full", then the secondary experiments."""
    return (FULL, *context.config.secondary_experiments)


# ---------------------------------------------------------------- metrics


def _row_at(rows: Sequence[MetricRow], fold: str, num_trees: int) -> MetricRow | None:
    """The single row of a fold at a number of trees; None if missing or recorded twice."""
    matches = [row for row in rows if row.fold == fold and row.num_trees == num_trees]
    return matches[0] if len(matches) == 1 else None


def compare_metric_rows(
    rehearsal: Sequence[MetricRow],
    reference: Sequence[MetricRow],
    fold: str,
    tree_counts: Sequence[int],
    label: str,
) -> list[RehearsalCheck]:
    """Compare one system's metric rows on the rehearsal fold, one check per number of trees.

    Args:
        rehearsal: The rehearsal's rows of ONE configuration (hmrec.hpo.v_records.config_rows).
        reference: M7's rows of the same configuration.
        fold: The rehearsal fold.
        tree_counts: The numbers of trees to compare (0 for the RRF baseline).
        label: Name of the system in the check names, e.g. "full".

    Returns:
        A check passes if both tables have exactly one row at that number of
        trees, with the same number of queries and == metric values (float64).
    """
    checks = []
    for num_trees in tree_counts:
        name = f"{label} metrics at {num_trees} trees"
        ours, theirs = _row_at(rehearsal, fold, num_trees), _row_at(reference, fold, num_trees)
        if ours is None or theirs is None:
            missing = "the rehearsal" if ours is None else "the M7 record"
            checks.append(RehearsalCheck(name, False, f"{missing} has no single row for {fold}"))
            continue
        differences = []
        if ours.n_queries != theirs.n_queries:
            differences.append(f"n_queries {ours.n_queries} != {theirs.n_queries}")
        for metric, value in ours.scores.items():
            # Exact equality on purpose: the same trees on the same rows give the same floats.
            if value != theirs.scores[metric]:
                differences.append(f"{metric} {value!r} != {theirs.scores[metric]!r}")
        checks.append(RehearsalCheck(name, not differences, "; ".join(differences)))
    return checks


def metric_checks(context: FinalContext) -> list[RehearsalCheck]:
    """The metric comparisons of the full system, the secondary experiments and the baseline."""
    paths, fold = context.paths, context.config.rehearsal_target_fold
    metrics = context.config.metrics
    rehearsal = read_metric_rows(results_files(paths, AREA_M8_REHEARSAL)[1], metrics)
    reference = read_metric_rows(results_files(paths, AREA_V_PHASE)[1], metrics)
    ranker = context.locked.config
    tree_counts = checkpoints(ranker.num_trees, ranker.checkpoint_every)
    checks = []
    for experiment in model_experiments(context):
        config_id = experiment_config_id(context, experiment)
        checks += compare_metric_rows(
            config_rows(rehearsal, experiment, config_id),
            config_rows(reference, experiment, config_id),
            fold,
            tree_counts,
            experiment,
        )
    checks += compare_metric_rows(
        config_rows(rehearsal, RRF_BASELINE, RRF_BASELINE),
        config_rows(reference, RRF_BASELINE, RRF_BASELINE),
        fold,
        (NO_TREES,),
        RRF_BASELINE,
    )
    return checks


# ---------------------------------------------------------------- models and Top-K


def _recorded_trial(
    context: FinalContext, area: str, experiment: str
) -> tuple[Mapping[str, str] | None, str]:
    """(trial row of the experiment on the rehearsal fold or None, its config_id)."""
    config_id = experiment_config_id(context, experiment)
    rows = read_trial_rows(results_files(context.paths, area)[0])
    fold = context.config.rehearsal_target_fold
    return find_trial(rows, experiment, config_id, fold), config_id


def model_check(context: FinalContext, experiment: str) -> RehearsalCheck:
    """The rehearsal model's trees must be the first trees of the M7 model.

    Both model files are first checked against the sha256 in their trial rows:
    artifacts/ is not in git, so the hash is the proof that a file is the
    recorded one.
    """
    num_trees = context.locked.num_trees
    name = f"{experiment} model: first {num_trees} trees"
    fold, paths = context.config.rehearsal_target_fold, context.paths
    hashes = {}
    for area in (AREA_M8_REHEARSAL, AREA_V_PHASE):
        row, config_id = _recorded_trial(context, area, experiment)
        if row is None:
            return RehearsalCheck(name, False, f"no trial row in area {area}")
        model_file = paths.ranker_model_file(area, experiment, config_id, fold)
        if not model_file.is_file() or file_sha256(model_file) != row["model_sha256"]:
            return RehearsalCheck(name, False, f"{model_file} is not the recorded model file")
        hashes[area] = first_trees_sha256(load_model(model_file), num_trees)
    same = hashes[AREA_M8_REHEARSAL] == hashes[AREA_V_PHASE]
    detail = "" if same else f"rehearsal {hashes[AREA_M8_REHEARSAL]}, M7 {hashes[AREA_V_PHASE]}"
    return RehearsalCheck(name, same, detail)


def top_k_check(context: FinalContext, experiment: str) -> RehearsalCheck:
    """The saved Top-K at the locked number of trees must equal M7's saved Top-K exactly.

    Compared: num_trees, query_id, rank, article_idx and the float64 score of
    every row (polars DataFrame.equals).
    """
    num_trees = context.locked.num_trees
    name = f"{experiment} Top-{context.locked.config.top_k} at {num_trees} trees"
    fold, paths = context.config.rehearsal_target_fold, context.paths
    config_id = experiment_config_id(context, experiment)
    tables = []
    for area in (AREA_M8_REHEARSAL, AREA_V_PHASE):
        predictions_file = paths.ranker_predictions_file(area, experiment, config_id, fold)
        if not predictions_file.is_file():
            return RehearsalCheck(name, False, f"{predictions_file} not found")
        table = pl.read_parquet(predictions_file).filter(pl.col(NUM_TREES) == num_trees)
        tables.append(table)
    same = tables[0].height > 0 and tables[0].equals(tables[1])
    detail = "" if same else f"rehearsal {tables[0].height} rows, M7 {tables[1].height} rows"
    return RehearsalCheck(name, same, detail)


def code_check(context: FinalContext, experiment: str) -> RehearsalCheck:
    """The rehearsal trial must have run with the code at HEAD, from a clean tree."""
    name = f"{experiment} trial ran with the code at HEAD"
    row, _ = _recorded_trial(context, AREA_M8_REHEARSAL, experiment)
    if row is None:
        return RehearsalCheck(name, False, "no rehearsal trial row")
    if row["git_dirty"] != "False" or context.code_changed(row["git_commit"]):
        detail = f"it ran at {row['git_commit'][:12]} (dirty: {row['git_dirty']})"
        return RehearsalCheck(name, False, detail)
    return RehearsalCheck(name, True, "")


# ---------------------------------------------------------------- recall monitoring


def monitor_check(context: FinalContext) -> RehearsalCheck:
    """The rehearsal's monitoring rows must equal the rows M6 recorded for the fold.

    Every column is compared as text except git_commit, git_dirty and
    finished_at, which describe the run.
    """
    name = "recall monitoring rows"
    fold = context.config.rehearsal_target_fold
    compared = [column for column in MONITOR_COLUMNS if column not in RUN_ONLY_MONITOR_COLUMNS]
    tables = []
    for path in (recall_file(context, MODE_REHEARSAL), context.paths.m6_candidates_results_file):
        rows = [row for row in read_rows(path, MONITOR_COLUMNS) if row["fold"] == fold]
        tables.append([{column: row[column] for column in compared} for row in rows])
    if not tables[0] or len(tables[0]) != len(tables[1]):
        detail = f"rehearsal {len(tables[0])} rows, M6 {len(tables[1])} rows"
        return RehearsalCheck(name, False, detail)
    differences = [
        f"{ours['scope']}.{column}: {ours[column]!r} != {theirs[column]!r}"
        for ours, theirs in zip(tables[0], tables[1], strict=True)
        for column in compared
        if ours[column] != theirs[column]
    ]
    return RehearsalCheck(name, not differences, "; ".join(differences))


# ---------------------------------------------------------------- the record


def rehearsal_checks(context: FinalContext) -> list[RehearsalCheck]:
    """Every comparison of the module docstring, in a fixed order."""
    checks = metric_checks(context)
    for experiment in model_experiments(context):
        checks.append(code_check(context, experiment))
        checks.append(model_check(context, experiment))
        checks.append(top_k_check(context, experiment))
    checks.append(monitor_check(context))
    return checks


def trained_sample_sha256(context: FinalContext) -> dict[str, dict[str, str]]:
    """Experiment -> {training fold -> sha256 of the sample file the rehearsal trained on}.

    Read from the rehearsal Datasets' records (dataset.json), which were written
    after every sample file had been checked against its own record.
    """
    fold = context.config.rehearsal_target_fold
    ratio = context.locked.tuned.negatives_per_positive
    recorded = {}
    for experiment in model_experiments(context):
        directory = context.paths.final_dataset_dir(AREA_M8_REHEARSAL, experiment, fold, ratio)
        recorded[experiment] = dict(read_dataset_record(directory).sample_sha256)
    return recorded


def record_rehearsal_check(context: FinalContext, now: str) -> dict[str, str]:
    """Run every comparison and append the outcome to results/m8_rehearsal_check.csv.

    Args:
        context: The context of this process.
        now: The current time as ISO text.

    Returns:
        The appended row; row["status"] is "passed" only if EVERY check passed.
    """
    checks = rehearsal_checks(context)
    failed = [check.name for check in checks if not check.passed]
    row = {
        "status": FAILED if failed else PASSED,
        "target_fold": context.config.rehearsal_target_fold,
        "config_id": context.locked.config_id,
        "num_trees": str(context.locked.num_trees),
        "n_checks": str(len(checks)),
        "failed_checks": "; ".join(failed),
        "checks_json": json.dumps(
            [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in checks]
        ),
        "sample_sha256_json": json.dumps(trained_sample_sha256(context), sort_keys=True),
        "git_commit": context.stamp.git_commit,
        "git_dirty": str(context.stamp.git_dirty),
        "checked_at": now,
    }
    append_row(context.paths.m8_rehearsal_check_file, REHEARSAL_CHECK_COLUMNS, row)
    return row
