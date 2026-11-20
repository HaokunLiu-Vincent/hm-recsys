"""CLI for M9: train the runs of the input-representation extension on V1-V4 (proposal §16.10).

    uv run python scripts/m9_deepfm_extension.py

Runs E1-E4 of configs/hpo/deepfm_extension.yaml (255 bins and the standardised
input, each with the pointwise and the listwise loss) with every seed, one
process per training (scripts/m9_deepfm_trial.py). A training that is already
recorded is not started again, so an interrupted run continues where it stopped.
Nothing is tuned and nothing is selected; a T fold is refused by every training.
Clean working tree only. The comparison is scripts/m9_deepfm_compare.py --extension.
"""

import logging
from pathlib import Path

from hmrec.config import (
    load_deepfm_config,
    load_deepfm_extension_config,
    load_deepfm_study_config,
    load_selection_rule,
)
from hmrec.deepfm.extension import run_extension
from hmrec.deepfm.study_run import StudyContext, TrialRequest, subprocess_launcher
from hmrec.git_state import git_state
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.paths import get_paths

LOGGER = get_logger("m9_deepfm_extension")
TRIAL_SCRIPT = Path(__file__).with_name("m9_deepfm_trial.py")


def no_launcher(request: TrialRequest) -> None:
    """The study context's own launcher is never used: every run has its own."""
    raise SystemExit(f"a training was started without its number of bins ({request})")


def main() -> None:
    """Train the missing trainings of the extension and log every run's result."""
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.deepfm_log_file("extension"))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("extension | git HEAD %s, uncommitted changes: %s", commit, dirty)
    if dirty:
        raise SystemExit("the working tree has uncommitted changes: the extension does not start")
    context = StudyContext(
        paths=paths,
        config=load_deepfm_config(paths.deepfm_config_file),
        study=load_deepfm_study_config(paths.deepfm_study_config_file),
        rule=load_selection_rule(paths.selection_rule_file),
        launcher=no_launcher,
    )
    extension = load_deepfm_extension_config(paths.deepfm_extension_config_file)
    results = run_extension(
        context, extension, lambda extra: subprocess_launcher(TRIAL_SCRIPT, extra)
    )
    for result in results:
        LOGGER.info(
            "%s %s: pooled %s = %.6f",
            result.run.name,
            result.run.arm,
            context.metric,
            result.score.pooled,
        )


if __name__ == "__main__":
    main()
