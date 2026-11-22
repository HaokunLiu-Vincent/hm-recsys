"""CLI: one tick of the weekly pipeline (docs/mlops.md §6).

    weekly --anchor DATE --mode eval|serve
    weekly --next --mode eval|serve

    python scripts/weekly.py --anchor 2020-07-29 --mode eval

closes the week before the anchor (C1 ingest, C2 evaluate, C3 sample, C4 fine-tune
R6) and opens the week of the anchor (O1 recall, O2 features, O3 train the
Ranker, O4 score, O5 publish, O6 record). The anchor is the only clock: every
step reads only purchases before it.

Every step runs in its own process (scripts/m12_step.py). The tick stops at the
first failed step or check and says why; running the same command again skips
the finished steps. A tick refuses to start if the tick before it has not
finished. State: artifacts/pipeline/. Result tables: results/m12_replay/.

In the container (the mounts are fixed there):
    make image-weekly ARGS="--anchor 2020-07-29 --mode eval"

--next takes the anchor from the state instead of the command line: the week
after the current prediction set (the first tick if nothing is published yet).
A scheduler can then start the pipeline with the same arguments every week; a
tick that stopped is resumed, because the state did not move.

Where the inputs and the state lie (M13). Neither variable is set on the laptop.
    HMREC_PIPELINE_INPUTS_ROOT   a folder with data/processed/, data/bootstrap/ and
                                 artifacts/features/category_codes.json
    HMREC_PIPELINE_STATE_ROOT    the folder of the state and of the result tables
    HMREC_IMAGE_DIGEST           the registry digest of the image; it goes into
                                 the run manifest

--calibration runs a calibration tick (docs/mlops.md §8.3): it measures a
platform that has no platform record yet. Everything is written under a separate
state, next to the state ("{state folder}-calibration"), and the Ranker goes
through the tolerance-accepted entry in calibration mode.

Exit code 0: the week was published. 1: the tick stopped (see the run manifest).
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

from hmrec.git_state import git_state  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import get_paths, tick_root_paths  # noqa: E402
from hmrec.run_manifest import MODES  # noqa: E402
from hmrec.run_records import RunStamp  # noqa: E402
from hmrec.weekly.driver import next_anchor, run_tick, subprocess_runner  # noqa: E402

LOGGER = get_logger("scripts.weekly")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="One tick of the weekly pipeline.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--anchor", type=date.fromisoformat, help="YYYY-MM-DD")
    which.add_argument(
        "--next",
        action="store_true",
        help="the tick the state says is due: one week after the current prediction set",
    )
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="a calibration tick: a separate state, the Ranker entry in calibration mode",
    )
    return parser.parse_args()


def main() -> None:
    """Run the tick and report how it ended."""
    args = parse_args()
    setup_logging(logging.INFO)
    paths = tick_root_paths(get_paths(), os.environ, args.calibration)
    commit, dirty = git_state(paths.project_root, paths.results_root)
    anchor = next_anchor(paths) if args.next else args.anchor
    LOGGER.info(
        "anchor %s (%s); state %s%s",
        anchor,
        "from the state" if args.next else "from the command line",
        paths.pipeline_state_dir,
        " (calibration tick)" if args.calibration else "",
    )
    runner = subprocess_runner(Path(__file__).resolve().parent)
    result = run_tick(paths, anchor, args.mode, runner, RunStamp(commit, dirty))
    manifest = result.manifest
    alerts = [finding for finding in result.findings if finding.alert]
    flags = [finding for finding in result.findings if finding.flag and not finding.alert]
    LOGGER.info(
        "tick %s %s: %s in %.0f s, largest process %.2f GB, %d flag(s), %d alert(s); %s",
        anchor,
        args.mode,
        manifest.status,
        manifest.total_seconds,
        manifest.max_process_rss_gb,
        len(flags),
        len(alerts),
        result.manifest_file,
    )
    for finding in [*alerts, *flags]:
        kind = "ALERT" if finding.alert else "flag"
        LOGGER.info(
            "%s | %s %s | week %s | value %r, reference %r | %s",
            kind,
            finding.rule,
            finding.subject,
            finding.week,
            finding.value,
            finding.reference,
            finding.limit,
        )
    if not result.succeeded:
        LOGGER.error("the tick stopped: %s", manifest.failure)
        sys.exit(1)


if __name__ == "__main__":
    main()
