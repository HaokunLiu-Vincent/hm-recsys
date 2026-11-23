"""CLI: build the serving form of a prediction set that is already published (mlops §9; M14).

    python scripts/m14_serving_set.py --anchor 2020-09-23

From M14 on, step O5 of every tick writes predictions/{anchor}/serving/ itself
(hmrec.weekly.serving_build). The prediction sets of the replay were published
before that; this command builds the same files for one of them, ONCE, from the
state as it is: the prediction set, the R1 candidates of its tick (the fallback
list) and the two id maps. No step of the tick runs again and no existing file
is changed; an existing serving folder is never replaced.

It reads no label. The state and the inputs are found as for a tick
(HMREC_PIPELINE_INPUTS_ROOT, HMREC_PIPELINE_STATE_ROOT; neither is set on the
laptop, where the state is artifacts/pipeline/).

In the cloud it runs as one execution of the pipeline's job:
    bash infra/operate.sh execute scripts/m14_serving_set.py --anchor 2020-09-23
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import logging  # noqa: E402
import os  # noqa: E402
from datetime import date  # noqa: E402

from hmrec.config import load_base_config  # noqa: E402
from hmrec.git_state import git_state  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import log_process_peak_memory  # noqa: E402
from hmrec.paths import get_paths, tick_root_paths  # noqa: E402
from hmrec.run_records import RunStamp  # noqa: E402
from hmrec.weekly.context import require_committed_code  # noqa: E402
from hmrec.weekly.serving_build import backfill_serving_set  # noqa: E402

LOGGER = get_logger("scripts.m14_serving_set")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Serving form of a published prediction set.")
    parser.add_argument("--anchor", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    return parser.parse_args()


def main() -> None:
    """Build the serving set and say what was written."""
    args = parse_args()
    setup_logging(logging.INFO)
    paths = tick_root_paths(get_paths(), os.environ, calibration=False)
    commit, dirty = git_state(paths.project_root, paths.results_root)
    stamp = RunStamp(commit, dirty)
    require_committed_code(stamp)
    LOGGER.info("state %s, anchor %s, commit %s", paths.pipeline_state_dir, args.anchor, commit)
    record = backfill_serving_set(paths, args.anchor, stamp)
    fallback = record.description.fallback
    LOGGER.info(
        "serving set %s written: %d customers x %d articles; "
        "fallback list (bucket %s, from %s): %s",
        record.set_id,
        record.n_customers,
        record.top_k,
        fallback.age_bucket,
        fallback.source,
        " ".join(fallback.article_ids),
    )
    limit_gb = load_base_config(paths.base_config_file).memory.limit_gb
    log_process_peak_memory(LOGGER, f"serving set {args.anchor}", limit_gb)


if __name__ == "__main__":
    main()
