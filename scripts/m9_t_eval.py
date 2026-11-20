"""CLI for M9: the single evaluation on T1-T4 of the DeepFM configuration chosen on V (§16.12).

    uv run python scripts/m9_t_eval.py --stage rehearsal
    uv run python scripts/m9_t_eval.py --stage formal

--stage rehearsal trains the chosen configuration (configs/m9_t_eval.yaml) with V4
as the target week for every seed and compares each result with the V4 record of
the configuration; everything must be bit-identical. It reads no T label and
writes one row to results/m9_t_rehearsal_check.csv. A failed rehearsal exits with
an error.

--stage formal runs only if the latest rehearsal passed at THIS commit on a clean
working tree. It first appends a row to results/m9_t_label_access.csv (time,
commit, purpose, script), then trains every seed on T1-T4, one process per
training. The evaluation is run once and is reported only. If it is interrupted,
run the same command again: recorded trainings are skipped; at another commit it
is refused.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import logging  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from hmrec.config import load_deepfm_config  # noqa: E402
from hmrec.git_state import git_state  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.run_records import RunStamp  # noqa: E402
from hmrec.t_eval.gate import PASSED  # noqa: E402
from hmrec.t_eval.plan import STAGE_REHEARSAL, STAGES, load_t_eval_context  # noqa: E402
from hmrec.t_eval.run import run_formal, run_rehearsal, subprocess_launcher  # noqa: E402

LOGGER = get_logger("m9_t_eval")
TRIAL_SCRIPT = Path(__file__).with_name("m9_t_trial.py")
SCRIPT_NAME = "scripts/m9_t_eval.py"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    return parser.parse_args()


def now() -> str:
    """The current local time as text, to the second."""
    return datetime.now().isoformat(timespec="seconds")


def main() -> None:
    """Run the rehearsal or the formal run."""
    args = parse_args()
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.deepfm_log_file(f"m9_t_eval_{args.stage}"))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("m9_t_eval %s | git HEAD %s, uncommitted changes: %s", args.stage, commit, dirty)
    if dirty:
        raise SystemExit("the working tree has uncommitted changes: the evaluation does not start")
    context = load_t_eval_context(paths, RunStamp(commit, dirty))
    launcher = subprocess_launcher(TRIAL_SCRIPT)
    if args.stage == STAGE_REHEARSAL:
        metrics = load_deepfm_config(paths.deepfm_config_file).metrics
        status, checks = run_rehearsal(context, launcher, metrics, now)
        for check in checks:
            LOGGER.info(
                "%s | %s | %s", "ok" if check.passed else "FAILED", check.name, check.detail
            )
        LOGGER.info("rehearsal %s -> %s", status, paths.m9_t_rehearsal_check_file)
        if status != PASSED:
            raise SystemExit("the rehearsal FAILED: no T label is read. Stop and report.")
        return
    started = run_formal(context, launcher, now, SCRIPT_NAME)
    LOGGER.info("formal run: %d trainings were run; all of them are recorded", started)


if __name__ == "__main__":
    main()
