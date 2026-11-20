"""CLI for M9: the chosen DeepFM configuration next to the M8 record on T1-T4 (§16.12).

    uv run python scripts/m9_t_compare.py

Report only. Uses saved values: the per-query metric values the formal trainings
of scripts/m9_t_eval.py wrote, the per-query values the formal run of M8 saved for
LightGBM lambdarank, the query lists, results/t_phase_metrics.csv (a check of the
M8 values) and results/m9_ext_arms.csv (the V values). No label is read, no model
is loaded, nothing is trained.

Writes results/m9_t_arms.csv (DeepFM: every seed with MAP@12 per week and pooled,
the mean, the sample standard deviation and the range; then the M8 record),
results/m9_t_contrasts.csv (the customer-level paired bootstrap of DeepFM minus the
M8 record, with the settings of configs/m8_diagnostics.yaml) and
results/m9_t_v_and_t.csv (the two models on V and on T). Run it after the formal
run has finished, on a clean working tree; the tables are never replaced.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

from hmrec.config import load_deepfm_config, load_diagnostics_config  # noqa: E402
from hmrec.csv_records import append_row  # noqa: E402
from hmrec.deepfm.compare import CompareSettings, table_columns  # noqa: E402
from hmrec.deepfm.records import locked_lightgbm, stamp_columns  # noqa: E402
from hmrec.git_state import git_state  # noqa: E402
from hmrec.hpo.selection import SCORE_EQUALITY_TOLERANCE  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.run_records import RunStamp  # noqa: E402
from hmrec.t_eval.compare import compare_on_t  # noqa: E402
from hmrec.t_eval.plan import STAGE_FORMAL, load_t_eval_context, read_csv_table  # noqa: E402
from hmrec.t_eval.run import missing_keys, stage_keys  # noqa: E402

LOGGER = get_logger("m9_t_compare")
STAMP_COLUMNS = ("git_commit", "git_dirty", "finished_at")


def write_table(path: Path, rows: list[dict[str, str]], stamp: dict[str, str]) -> None:
    """Write the rows of one comparison table, each with the code state and the time."""
    columns = (*table_columns(rows), *STAMP_COLUMNS)
    for row in rows:
        append_row(path, columns, {**row, **stamp})
    LOGGER.info("%d rows -> %s", len(rows), path)


def main() -> None:
    """Compare the two arms on T1-T4 from saved values."""
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.deepfm_log_file("m9_t_compare"))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    if dirty:
        raise SystemExit("the working tree has uncommitted changes: the comparison does not start")
    tables = (paths.m9_t_arms_file, paths.m9_t_contrasts_file, paths.m9_t_v_and_t_file)
    if any(table.exists() for table in tables):
        raise SystemExit("a comparison table exists already; they are never replaced")
    context = load_t_eval_context(paths, RunStamp(commit, dirty))
    missing = missing_keys(context, stage_keys(context, STAGE_FORMAL))
    if missing:
        raise SystemExit(f"{len(missing)} trainings of the formal run are not recorded: {missing}")
    locked_config_id, locked_trees = locked_lightgbm(paths)
    if locked_trees != context.config.m8_record_num_trees:
        raise SystemExit(
            f"the locked Ranker has {locked_trees} trees, configs/m9_t_eval.yaml says "
            f"{context.config.m8_record_num_trees}"
        )
    diagnostics = load_diagnostics_config(paths.m8_diagnostics_config_file)
    settings = CompareSettings(
        metrics=load_deepfm_config(paths.deepfm_config_file).metrics,
        seed=context.config.m8_record_seed,
        recent_purchase_days=diagnostics.recent_purchase_days,
        n_replicates=diagnostics.bootstrap_replicates,
        bootstrap_seed=diagnostics.bootstrap_seed,
        confidence_level=diagnostics.confidence_level,
        # No wide table is read: the query groups are not computed on T.
        wide_columns=(),
    )
    comparison = compare_on_t(
        context,
        settings,
        locked_config_id,
        read_csv_table(paths.t_phase_metrics_file),
        read_csv_table(paths.m9_ext_arms_file),
        SCORE_EQUALITY_TOLERANCE,
    )
    stamp = stamp_columns(RunStamp(commit, dirty))
    # The M8 record has one seed, so its row has no mean / std / range rows; the DeepFM
    # rows and the M8 row share their columns.
    write_table(paths.m9_t_arms_file, comparison.arms, stamp)
    write_table(paths.m9_t_contrasts_file, comparison.contrasts, stamp)
    write_table(paths.m9_t_v_and_t_file, comparison.v_and_t, stamp)
    metric = settings.metrics[0]
    for row in comparison.arms:
        LOGGER.info("%s %s: %s = %s", row["arm"], row["row"], metric, row[metric])
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
