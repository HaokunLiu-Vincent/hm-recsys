"""CLI: R1-R5 union pre-check on D1-D4 (diagnostics only, before M4; proposal §5.7 items 1-3).

Usage:
    uv run python scripts/m2_union_precheck.py [--run-id m2_union_precheck]

Replays the D-phase selection over results/d_phase.csv (no new trials), generates
the selected configs' candidates on D1-D4 into artifacts/hpo/D/union_precheck/
(existing files are read back, never overwritten) and writes
results/d_phase_union_precheck.csv. Nothing is selected or locked, and the R3
active-pool decision (§5.7 item 4) is NOT made here: that is M4, after R6.
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import logging  # noqa: E402

import polars as pl  # noqa: E402

from hmrec.config import (  # noqa: E402
    load_base_config,
    load_catalog_config,
    load_preprocess_config,
    load_selection_rule,
)
from hmrec.data.folds import load_fold_manifest  # noqa: E402
from hmrec.hpo.d_phase import PHASE, load_fold_contexts  # noqa: E402
from hmrec.hpo.grid import load_phase_grid  # noqa: E402
from hmrec.hpo.runner import git_state, warn_uncommitted_changes  # noqa: E402
from hmrec.hpo.union_precheck import (  # noqa: E402
    run_union_precheck,
    select_channels,
    write_precheck_table,
)
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import MemoryTracker  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402

CHANNELS = ("r1_popularity", "r2_repurchase", "r3_sibling", "r4_itemcf", "r5_wals")
LOGGER = get_logger("m2_union_precheck")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", default="m2_union_precheck", help="log goes to artifacts/runs/{run_id}/run.log"
    )
    return parser.parse_args()


def main() -> None:
    """Run the pre-check and write the summary table."""
    args = parse_args()
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.run_dir(args.run_id) / "run.log")
    base = load_base_config(paths.base_config_file)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(base.num_threads))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("git HEAD %s, uncommitted code/config changes: %s", commit, dirty)
    if dirty:
        warn_uncommitted_changes(commit)
    grid = load_phase_grid(paths.d_phase_grid_file)
    rule = load_selection_rule(paths.selection_rule_file)
    folds = load_fold_manifest(paths.fold_manifest_file).by_phase(PHASE)
    fold_names = [fold.name for fold in folds]
    selected = select_channels(paths, CHANNELS, grid, rule, fold_names)
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    catalog = load_catalog_config(paths.catalog_config_file)
    contexts = load_fold_contexts(paths, folds, preprocess, catalog)
    with MemoryTracker(
        "union pre-check",
        LOGGER,
        limit_gb=base.memory.limit_gb,
        sample_interval_seconds=base.memory.sample_interval_seconds,
    ):
        table = run_union_precheck(
            selected, contexts, base, paths.union_precheck_candidates_file, grid.fold_groups
        )
    write_precheck_table(table, paths.union_precheck_results_file)
    # Print every row and column (polars shortens long tables by default).
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=5):
        LOGGER.info("summary table -> %s\n%s", paths.union_precheck_results_file, table)


if __name__ == "__main__":
    main()
