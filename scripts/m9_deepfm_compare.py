"""CLI for M9: compare the selected DeepFM configuration with LightGBM on V1-V4 (report only).

    uv run python scripts/m9_deepfm_compare.py

Uses only saved values: the per-query AP@12 both models wrote when they were
evaluated on the V folds, the query lists and one feature column of the frozen
wide tables. No label is read, no model is loaded, nothing is trained.

Writes results/deepfm_compare.csv (the 95% interval of DeepFM, of LightGBM and
of DeepFM minus LightGBM, with the customer-level paired bootstrap and the
settings of configs/m8_diagnostics.yaml) and results/deepfm_groups.csv (MAP@12
of both models in the three query groups, per fold and pooled). Run it after
the study has finished, on a clean working tree.
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
    COMPARE_COLUMNS,
    GROUP_COLUMNS,
    QUANTITIES,
    CompareSettings,
    FoldFiles,
    compare_folds,
)
from hmrec.deepfm.records import locked_lightgbm, stamp_columns  # noqa: E402
from hmrec.deepfm.study_run import (  # noqa: E402
    StudyContext,
    TrialRequest,
    run_final,
    run_tuning,
    selected_variant,
)
from hmrec.git_state import git_state  # noqa: E402
from hmrec.locks import load_features_lock  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.ranker.ablation import wide_table_columns  # noqa: E402
from hmrec.run_records import RunStamp  # noqa: E402

LOGGER = get_logger("m9_deepfm_compare")
STAMP_COLUMNS = ("git_commit", "git_dirty", "finished_at")
PER_QUERY_FILE = "per_query.parquet"


def refuse_training(request: TrialRequest) -> None:
    """The comparison trains nothing: a missing training means the study is not finished."""
    raise SystemExit(f"a training of the study is missing ({request}): finish the study first")


def main() -> None:
    """Compare the selected configuration (selection seed) with the locked LightGBM ranker."""
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.deepfm_log_file("compare"))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    if dirty:
        raise SystemExit("the working tree has uncommitted changes: the comparison does not start")
    if paths.deepfm_compare_file.exists() or paths.deepfm_groups_file.exists():
        raise SystemExit("the comparison tables exist already; they are never replaced")
    context = StudyContext(
        paths=paths,
        config=load_deepfm_config(paths.deepfm_config_file),
        study=load_deepfm_study_config(paths.deepfm_study_config_file),
        rule=load_selection_rule(paths.selection_rule_file),
        launcher=refuse_training,
    )
    study = context.study
    final = run_final(context, selected_variant(run_tuning(context), context.default))
    lightgbm_id, num_trees = locked_lightgbm(paths)
    diagnostics = load_diagnostics_config(paths.m8_diagnostics_config_file)
    lock = load_features_lock(paths.features_lock_file)
    settings = CompareSettings(
        metric=study.metric,
        deepfm_epoch=final.score.selected_epoch,
        lightgbm_num_trees=num_trees,
        recent_purchase_days=diagnostics.recent_purchase_days,
        n_replicates=diagnostics.bootstrap_replicates,
        seed=diagnostics.bootstrap_seed,
        confidence_level=diagnostics.confidence_level,
        wide_columns=wide_table_columns(lock),
    )
    seed, config_id = study.selection_seed, final.score.config_id
    folds = [
        FoldFiles(
            fold=fold,
            deepfm_per_query=paths.deepfm_trial_dir(final.area, config_id, seed, fold)
            / PER_QUERY_FILE,
            lightgbm_per_query=paths.ranker_per_query_file("v_phase", "full", lightgbm_id, fold),
            queries=paths.queries_file(fold),
            features_dir=paths.features_dir(fold),
        )
        for fold in study.target_folds
    ]
    intervals, n_customers, n_queries, groups = compare_folds(folds, settings)
    stamp = stamp_columns(RunStamp(commit, dirty))
    for quantity in QUANTITIES:
        interval = intervals[quantity]
        row = {
            "folds": " ".join(study.target_folds),
            "metric": settings.metric,
            "quantity": quantity,
            "estimate": repr(interval.estimate),
            "ci_low": repr(interval.low),
            "ci_high": repr(interval.high),
            "confidence_level": repr(settings.confidence_level),
            "n_replicates": str(settings.n_replicates),
            "seed": str(settings.seed),
            "n_customers": str(n_customers),
            "n_queries": str(n_queries),
            "deepfm_config_id": config_id,
            "deepfm_seed": str(seed),
            "deepfm_epoch": str(settings.deepfm_epoch),
            "lightgbm_config_id": lightgbm_id,
            "lightgbm_num_trees": str(num_trees),
            **stamp,
        }
        append_row(paths.deepfm_compare_file, (*COMPARE_COLUMNS, *STAMP_COLUMNS), row)
        LOGGER.info(
            "%s: %.6f [%.6f, %.6f]", quantity, interval.estimate, interval.low, interval.high
        )
    for group in groups:
        append_row(paths.deepfm_groups_file, (*GROUP_COLUMNS, *STAMP_COLUMNS), {**group, **stamp})


if __name__ == "__main__":
    main()
