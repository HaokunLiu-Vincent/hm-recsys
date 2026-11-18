"""CLI for the descriptive diagnostics after M8 (AGENTS.md §3.2).

The diagnostics describe recorded results; they select and change nothing.

Usage:
    caffeinate -i uv run python scripts/m8_diagnostics.py

Writes four tables (append-only; configs/m8_diagnostics.yaml says what they cover):
    results/t_diag_decomposition.csv   A: union recall, Recall@12, their ratio, MAP@12
    results/t_diag_concentration.csv   B: how concentrated each label week's purchases are
    results/t_diag_query_groups.csv    C: query groups by purchase history, and their scores
    results/t_diag_bootstrap.csv       D: bootstrap intervals of the pooled MAP@12

Reading the labels of the T folds again is recorded FIRST, in
results/t_diag_label_access.csv (time, git commit, purpose). The run refuses to
start unless the formal run of M8 is finished, the working tree is clean and the
M8 code and configs are unchanged since the formal run (hmrec.diagnostics.access).

No model is trained or loaded, M8 is not run again, and no M8 result table is
written. Only saved files are read: result tables, the saved per-query values,
the frozen wide tables, candidate files, labels, and purchase lines before each
anchor (for the first sale day of an article).
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
from datetime import datetime  # noqa: E402

from hmrec.diagnostics.context import load_diagnostics_context  # noqa: E402
from hmrec.diagnostics.run import run_diagnostics  # noqa: E402
from hmrec.git_state import git_state, warn_uncommitted_changes  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import log_process_peak_memory  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.ranker.runtime import read_lightgbm_runtime  # noqa: E402
from hmrec.run_records import RunStamp, check_memory_limit  # noqa: E402

LOGGER = get_logger("m8_diagnostics")
RUN_NAME = "m8_diagnostics"
SCRIPT = "scripts/m8_diagnostics.py"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (there are none; --help prints the description)."""
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> None:
    """Run the diagnostics once."""
    parse_args()
    paths = get_paths()
    setup_logging(log_file=paths.run_dir(RUN_NAME) / "run.log")
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("git HEAD %s, uncommitted code/config changes: %s", commit, dirty)
    if dirty:
        warn_uncommitted_changes(commit)
    context = load_diagnostics_context(paths, RunStamp(commit, dirty), read_lightgbm_runtime())
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(context.base.num_threads))
    now = datetime.now().isoformat(timespec="seconds")
    tables = run_diagnostics(context, now, SCRIPT)
    LOGGER.info(
        "rows written: decomposition %d, concentration %d, query groups %d, bootstrap %d",
        len(tables.decomposition),
        len(tables.concentration),
        len(tables.query_groups),
        len(tables.bootstrap),
    )
    limit_gb = context.base.memory.limit_gb
    peak = log_process_peak_memory(LOGGER, RUN_NAME, limit_gb)
    check_memory_limit(peak, limit_gb, RUN_NAME)


if __name__ == "__main__":
    main()
