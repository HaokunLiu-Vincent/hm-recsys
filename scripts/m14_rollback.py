"""CLI: point `current` at another published prediction set (docs/mlops.md §9, §11; M14).

    python scripts/m14_rollback.py --to previous     # roll back one prediction set
    python scripts/m14_rollback.py --to next         # roll forward again, one set
    python scripts/m14_rollback.py --to 2020-09-16   # the set of that anchor

The service follows the pointer without a restart (hmrec.serving.store). A target
that does not exist, was never published, fails its checksum or has no loadable
serving form is refused, and the pointer stays as it was (exit code 1). A move
is one row of the table pointer_moves.csv. Nothing else is written; no model and
no prediction set is changed. hmrec.weekly.rollback holds the rules.

The state is found as for a tick (HMREC_PIPELINE_STATE_ROOT; not set on the
laptop, where the state is artifacts/pipeline/).

In the cloud it runs as one execution of the pipeline's job, which has the state
mounted:
    bash infra/operate.sh rollback previous
"""

import argparse
import logging
import os
import sys

from hmrec.git_state import git_state
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.paths import get_paths, tick_root_paths
from hmrec.run_records import RunStamp
from hmrec.weekly.driver import state_store
from hmrec.weekly.rollback import RollbackRefused, move_pointer

LOGGER = get_logger("scripts.m14_rollback")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Move the `current` pointer.")
    parser.add_argument(
        "--to", required=True, help='"previous", "next", or the anchor of a set (YYYY-MM-DD)'
    )
    return parser.parse_args()


def main() -> None:
    """Move the pointer, or say why it was not moved."""
    args = parse_args()
    setup_logging(logging.INFO)
    paths = tick_root_paths(get_paths(), os.environ, calibration=False)
    commit, dirty = git_state(paths.project_root, paths.results_root)
    try:
        move = move_pointer(paths, state_store(paths), args.to, RunStamp(commit, dirty))
    except RollbackRefused as refusal:
        LOGGER.error("refused, the pointer is unchanged: %s", refusal)
        sys.exit(1)
    LOGGER.info(
        "current now names the prediction set of %s (%s); it named %s before",
        move.to_anchor,
        move.record.set_id,
        move.from_anchor,
    )


if __name__ == "__main__":
    main()
