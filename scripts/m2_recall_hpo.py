"""CLI for M2: D-phase ablation + HPO of recall channels R1-R5 (proposal §4, §5).

Usage:
    uv run python scripts/m2_recall_hpo.py --channels r1_popularity r2_repurchase r3_sibling
        [--run-id m2_recall_hpo]

Runs the steps of configs/hpo/d_phase.yaml on folds D1-D4 (full data) and writes
results/d_phase.csv (appended; finished trials are skipped on a re-run),
results/d_phase_steps/{channel}.csv and the R2 / R3 pool-flag files under
artifacts/hpo/D/. The selected configs are only logged; nothing is written to
configs/locks/ (a human confirms first).
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import logging  # noqa: E402

from hmrec.config import (  # noqa: E402
    load_base_config,
    load_catalog_config,
    load_preprocess_config,
    load_selection_rule,
)
from hmrec.data.folds import load_fold_manifest  # noqa: E402
from hmrec.hpo.d_phase import PHASE, load_fold_contexts, run_d_phase  # noqa: E402
from hmrec.hpo.grid import load_phase_grid  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402

LOGGER = get_logger("m2_recall_hpo")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channels", nargs="+", required=True, help="e.g. r1_popularity r2_repurchase"
    )
    parser.add_argument(
        "--run-id", default="m2_recall_hpo", help="log goes to artifacts/runs/{run_id}/run.log"
    )
    return parser.parse_args()


def main() -> None:
    """Run the D phase for the requested channels."""
    args = parse_args()
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.run_dir(args.run_id) / "run.log")
    base = load_base_config(paths.base_config_file)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(base.num_threads))
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    catalog = load_catalog_config(paths.catalog_config_file)
    rule = load_selection_rule(paths.selection_rule_file)
    grid = load_phase_grid(paths.d_phase_grid_file)
    folds = load_fold_manifest(paths.fold_manifest_file).by_phase(PHASE)
    contexts = load_fold_contexts(paths, folds, preprocess, catalog)
    outcomes = run_d_phase(paths, args.channels, contexts, grid, rule, base)
    for outcome in outcomes:
        LOGGER.info("%s: selected (NOT locked) %s", outcome.channel, outcome.selected_config)


if __name__ == "__main__":
    main()
