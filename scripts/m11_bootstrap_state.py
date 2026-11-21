"""CLI: build the bootstrap state of the replay, data/bootstrap/ (docs/mlops.md §6.3).

Usage:
    python scripts/m11_bootstrap_state.py --plan      # list the files; copy nothing
    python scripts/m11_bootstrap_state.py             # build data/bootstrap/
    python scripts/m11_bootstrap_state.py --verify    # compare the state with its manifest

The state is the one at the end of F4 that the first tick (V1) starts from: the R6
chain model, the Ranker samples of F1-F4 and the monitoring history
(hmrec.bootstrap_state says which files and why). Stage-1 files are COPIED, never
moved or changed; each is compared with its stage-1 sha256 record before the copy.
An existing data/bootstrap/ is never replaced.

Building needs a clean working tree: the manifest records the commit of the code
that built the state (AGENTS.md §3.4).
"""

import argparse
import logging
import sys

from hmrec.bootstrap_state import (
    build_bootstrap_state,
    plan_from_repository,
    read_bootstrap_manifest,
    verify_bootstrap_state,
)
from hmrec.git_state import git_state
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.paths import get_paths

LOGGER = get_logger("scripts.m11_bootstrap_state")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the bootstrap state of the replay.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plan", action="store_true", help="list the files; copy nothing")
    group.add_argument("--verify", action="store_true", help="compare the state with its manifest")
    return parser.parse_args()


def main() -> None:
    """Plan, build or verify the state."""
    args = parse_args()
    setup_logging(logging.INFO)
    paths = get_paths()
    if args.verify:
        entries = read_bootstrap_manifest(paths.bootstrap_manifest_file)
        problems = verify_bootstrap_state(paths.bootstrap_dir, entries)
        for problem in problems:
            LOGGER.error("%s", problem)
        LOGGER.info("%d file(s) checked, %d problem(s)", len(entries), len(problems))
        sys.exit(1 if problems else 0)
    plan, fold_names = plan_from_repository(paths)
    LOGGER.info("first tick %s; the state is the end of %s", plan.first_tick_fold, plan.closed_fold)
    for item in plan.items:
        folds = f" rows of {' '.join(item.folds)}" if item.folds else ""
        LOGGER.info("%-4s %s <- %s%s", item.kind, item.destination, item.source, folds)
    if args.plan:
        return
    commit, dirty = git_state(paths.project_root, paths.results_root)
    if dirty:
        raise RuntimeError("the working tree has uncommitted changes; commit before building")
    entries = build_bootstrap_state(paths, plan, fold_names, commit)
    total_bytes = sum(entry.size_bytes for entry in entries)
    LOGGER.info(
        "wrote %d file(s), %d bytes, and the manifest %s",
        len(entries),
        total_bytes,
        paths.bootstrap_manifest_file,
    )


if __name__ == "__main__":
    main()
