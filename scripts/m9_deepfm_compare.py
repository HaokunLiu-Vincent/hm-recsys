"""CLI for M9: compare the four arms of the 2x2 study on V1-V4 (report only; proposal §16.8).

    uv run python scripts/m9_deepfm_compare.py

Arms: lgb_lambdarank (the locked Ranker), lgb_binary, deepfm_listwise,
deepfm_pointwise, each at its selected configuration with the final seeds. Uses
only saved values: the per-query metric values every training wrote when it was
evaluated on the V folds, the query lists and one feature column of the frozen
wide tables. No label is read, no model is loaded, nothing is trained.

Writes results/m9_arms.csv (per arm: every seed, the mean, the sample standard
deviation and the range; run time and true peak per seed), results/m9_contrasts.csv
(the customer-level paired bootstrap of four differences, on the selection seed
and on the seed means, with the settings of configs/m8_diagnostics.yaml) and
results/m9_groups.csv (MAP@12 of the four arms in the three query groups). Run
it after both studies have finished, on a clean working tree.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import logging  # noqa: E402

from hmrec.config import (  # noqa: E402
    load_deepfm_config,
    load_deepfm_study_config,
    load_diagnostics_config,
    load_selection_rule,
)
from hmrec.csv_records import append_row  # noqa: E402
from hmrec.deepfm.compare import (  # noqa: E402
    ARM_DEEPFM_LISTWISE,
    ARM_DEEPFM_POINTWISE,
    ARM_LGB_BINARY,
    ARM_LGB_LAMBDARANK,
    ArmRuns,
    CompareSettings,
    FoldFiles,
    compare_arms,
    seed_basis,
    table_columns,
)
from hmrec.deepfm.records import (  # noqa: E402
    deepfm_run_cost,
    lightgbm_arm_run_cost,
    locked_lightgbm,
    locked_lightgbm_run_cost,
    stamp_columns,
)
from hmrec.deepfm.study import Variant, variant_config_id  # noqa: E402
from hmrec.deepfm.study_run import (  # noqa: E402
    AREA_V_PHASE,
    STAGES,
    StudyContext,
    TrialRequest,
    run_study,
)
from hmrec.git_state import git_state  # noqa: E402
from hmrec.hpo.lgb_arm_records import (  # noqa: E402
    ARM_BINARY,
    ARM_LAMBDARANK,
    choose_binary_num_trees,
    read_metric_rows,
)
from hmrec.locks import load_features_lock, load_ranker_lock  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import ProjectPaths, get_paths  # noqa: E402
from hmrec.ranker.ablation import wide_table_columns  # noqa: E402
from hmrec.run_records import RunStamp  # noqa: E402

LOGGER = get_logger("m9_deepfm_compare")
STAMP_COLUMNS = ("git_commit", "git_dirty", "finished_at")
COST_COLUMNS = ("total_seconds", "process_max_rss_gb", "cost_note")
PER_QUERY_FILE = "per_query.parquet"
EPOCH = "epoch"
NUM_TREES = "num_trees"
M7_COST_NOTE = "the M7 grid trial: trained to 1000 trees and evaluated at 10 checkpoints"


def refuse_training(request: TrialRequest) -> None:
    """The comparison trains nothing: a missing training means the study is not finished."""
    raise SystemExit(f"a training of the study is missing ({request}): finish the study first")


def deepfm_arm(context: StudyContext, arm: str, variant: Variant) -> ArmRuns:
    """The saved files of a DeepFM arm: its selected configuration with the final seeds."""
    paths, study = context.paths, context.study
    config_id = variant_config_id(context.config, variant)
    files = {
        seed: {
            fold: paths.deepfm_trial_dir(AREA_V_PHASE, config_id, seed, fold) / PER_QUERY_FILE
            for fold in study.target_folds
        }
        for seed in study.final_seeds
    }
    return ArmRuns(arm, EPOCH, dict.fromkeys(study.final_seeds, variant.epochs), files)


def lightgbm_arms(context: StudyContext, metrics: tuple[str, ...]) -> tuple[ArmRuns, ArmRuns]:
    """The saved files of the two LightGBM arms (lambdarank: its first seed is the M7 record)."""
    paths, study = context.paths, context.study
    locked_id, locked_trees = locked_lightgbm(paths)
    lock = load_ranker_lock(paths.ranker_lock_file)
    every = lock.scoring.checkpoint_every
    checkpoints = tuple(range(every, study.lightgbm_binary_num_trees + 1, every))
    first_seed, *other_seeds = study.lightgbm_seeds
    choice = choose_binary_num_trees(
        read_metric_rows(paths.m9_lgb_metrics_file, metrics),
        first_seed,
        study.target_folds,
        study.metric,
        study.epoch_tolerance,
        checkpoints,
    )
    folds = study.target_folds
    lambdarank_files = {
        first_seed: {
            fold: paths.ranker_per_query_file(AREA_V_PHASE, "full", locked_id, fold)
            for fold in folds
        }
    }
    for seed in other_seeds:
        lambdarank_files[seed] = {
            fold: paths.m9_lgb_per_query_file(ARM_LAMBDARANK, seed, fold) for fold in folds
        }
    binary_files = {
        seed: {fold: paths.m9_lgb_per_query_file(ARM_BINARY, seed, fold) for fold in folds}
        for seed in study.lightgbm_seeds
    }
    seeds = study.lightgbm_seeds
    return (
        ArmRuns(
            ARM_LGB_LAMBDARANK, NUM_TREES, dict.fromkeys(seeds, locked_trees), lambdarank_files
        ),
        ArmRuns(ARM_LGB_BINARY, NUM_TREES, dict.fromkeys(seeds, choice.num_trees), binary_files),
    )


def run_costs(
    context: StudyContext, pointwise: Variant, listwise: Variant
) -> dict[tuple[str, str], tuple[float, float, str]]:
    """(arm, "seed_S") -> (total seconds, highest true peak in GB, note), from the trial tables."""
    paths, study = context.paths, context.study
    folds = study.target_folds
    costs: dict[tuple[str, str], tuple[float, float, str]] = {}
    for arm, variant in ((ARM_DEEPFM_POINTWISE, pointwise), (ARM_DEEPFM_LISTWISE, listwise)):
        config_id = variant_config_id(context.config, variant)
        for seed in study.final_seeds:
            seconds, peak = deepfm_run_cost(paths, AREA_V_PHASE, config_id, seed, folds)
            costs[(arm, seed_basis(seed))] = (seconds, peak, "")
    first_seed, *other_seeds = study.lightgbm_seeds
    seconds, peak = locked_lightgbm_run_cost(paths, folds)
    costs[(ARM_LGB_LAMBDARANK, seed_basis(first_seed))] = (seconds, peak, M7_COST_NOTE)
    for seed in other_seeds:
        seconds, peak = lightgbm_arm_run_cost(paths, ARM_LAMBDARANK, seed, folds)
        costs[(ARM_LGB_LAMBDARANK, seed_basis(seed))] = (seconds, peak, "")
    for seed in study.lightgbm_seeds:
        seconds, peak = lightgbm_arm_run_cost(paths, ARM_BINARY, seed, folds)
        note = "trained to the full number of trees, every checkpoint evaluated"
        costs[(ARM_LGB_BINARY, seed_basis(seed))] = (
            seconds,
            peak,
            note if seed == first_seed else "",
        )
    return costs


def write_table(path, rows: list[dict[str, str]], stamp: dict[str, str]) -> None:
    """Write the rows of one comparison table, each with the code state and the time."""
    columns = (*table_columns(rows), *STAMP_COLUMNS)
    for row in rows:
        append_row(path, columns, {**row, **stamp})
    LOGGER.info("%d rows -> %s", len(rows), path)


def output_files(paths: ProjectPaths) -> tuple:
    """The three tables of the comparison; none of them may exist yet."""
    files = (paths.m9_arms_file, paths.m9_contrasts_file, paths.m9_groups_file)
    if any(path.exists() for path in files):
        raise SystemExit("a comparison table exists already; they are never replaced")
    return files


def main() -> None:
    """Compare the four arms at their selected configurations."""
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.deepfm_log_file("compare"))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    if dirty:
        raise SystemExit("the working tree has uncommitted changes: the comparison does not start")
    arms_file, contrasts_file, groups_file = output_files(paths)
    config = load_deepfm_config(paths.deepfm_config_file)
    context = StudyContext(
        paths=paths,
        config=config,
        study=load_deepfm_study_config(paths.deepfm_study_config_file),
        rule=load_selection_rule(paths.selection_rule_file),
        launcher=refuse_training,
    )
    study = context.study
    result = run_study(context, STAGES[-1])
    pointwise, listwise = result.pointwise_variant, result.listwise_variant
    if pointwise is None or listwise is None:
        raise SystemExit("the DeepFM study has not selected both configurations yet")
    lambdarank_arm, binary_arm = lightgbm_arms(context, config.metrics)
    arms = (
        lambdarank_arm,
        binary_arm,
        deepfm_arm(context, ARM_DEEPFM_LISTWISE, listwise),
        deepfm_arm(context, ARM_DEEPFM_POINTWISE, pointwise),
    )
    diagnostics = load_diagnostics_config(paths.m8_diagnostics_config_file)
    lock = load_features_lock(paths.features_lock_file)
    settings = CompareSettings(
        metrics=config.metrics,
        seed=study.selection_seed,
        recent_purchase_days=diagnostics.recent_purchase_days,
        n_replicates=diagnostics.bootstrap_replicates,
        bootstrap_seed=diagnostics.bootstrap_seed,
        confidence_level=diagnostics.confidence_level,
        wide_columns=wide_table_columns(lock),
    )
    folds = [
        FoldFiles(fold, paths.queries_file(fold), paths.features_dir(fold))
        for fold in study.target_folds
    ]
    comparison = compare_arms(folds, arms, settings)
    costs = run_costs(context, pointwise, listwise)
    arm_table = []
    for row in comparison.arms:
        seconds, peak, note = costs.get((row["arm"], row["row"]), (None, None, ""))
        cost = {
            "total_seconds": "" if seconds is None else repr(seconds),
            "process_max_rss_gb": "" if peak is None else repr(peak),
            "cost_note": note,
        }
        arm_table.append({**row, **cost})
    stamp = stamp_columns(RunStamp(commit, dirty))
    write_table(arms_file, arm_table, stamp)
    write_table(contrasts_file, comparison.contrasts, stamp)
    write_table(groups_file, comparison.groups, stamp)
    for row in arm_table:
        LOGGER.info("%s %s: %s = %s", row["arm"], row["row"], study.metric, row[study.metric])
    for row in comparison.contrasts:
        if row["quantity"] == "difference":
            LOGGER.info(
                "%s - %s (%s): %s [%s, %s]",
                row["first_arm"],
                row["second_arm"],
                row["basis"],
                row["estimate"],
                row["ci_low"],
                row["ci_high"],
            )


if __name__ == "__main__":
    main()
