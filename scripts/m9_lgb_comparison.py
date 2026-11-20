"""CLI for M9: run the two LightGBM arms of the 2x2 comparison on V1-V4 (proposal §16.8).

    caffeinate -i uv run python scripts/m9_lgb_comparison.py

Arm "lambdarank": the locked Ranker with the seeds after the first (the first
seed is the M7 record). Arm "binary": the locked Ranker with the binary
objective; its first seed to `binary_num_trees` trees, k* by the number-of-trees
rule, then the other seeds to k* trees. Every training runs in its own process
(scripts/m9_lgb_trial.py); recorded trainings are not repeated, so the same
command continues an interrupted run. T folds are never touched.

Before anything is trained, the locked parameters that make training random are
logged. If there were none, every seed would give the same model: the run stops
and says so instead of training seeds that cannot differ.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

from pathlib import Path  # noqa: E402

from hmrec.config import (  # noqa: E402
    load_deepfm_config,
    load_deepfm_study_config,
)
from hmrec.git_state import git_state  # noqa: E402
from hmrec.hpo.lgb_arms import load_arm_context, random_parameters  # noqa: E402
from hmrec.hpo.lgb_arms_run import ArmsPlan, run_arms, subprocess_launcher  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.ranker.runtime import read_lightgbm_runtime  # noqa: E402

LOGGER = get_logger("m9_lgb_comparison")
TRIAL_SCRIPT = Path(__file__).resolve().parent / "m9_lgb_trial.py"


def main() -> None:
    """Run the missing trainings of the two arms and log the binary arm's k*."""
    paths = get_paths()
    setup_logging(log_file=paths.run_dir("m9_lgb_comparison") / "run.log")
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("git HEAD %s, uncommitted changes: %s", commit, dirty)
    if dirty:
        raise SystemExit("the working tree has uncommitted changes: the run does not start")
    study = load_deepfm_study_config(paths.deepfm_study_config_file)
    metrics = load_deepfm_config(paths.deepfm_config_file).metrics
    context = load_arm_context(paths, read_lightgbm_runtime(), metrics)
    locked = context.locked
    if study.lightgbm_seeds[0] != locked.config.seed:
        raise SystemExit(
            f"the first LightGBM seed must be the locked seed {locked.config.seed}, "
            f"got {study.lightgbm_seeds[0]}"
        )
    randomness = random_parameters(locked.config.train_params)
    LOGGER.info("locked parameters that make training random: %s", randomness)
    LOGGER.info(
        "a seed is set through the parameter `seed` only (as in the locked training); "
        "sampling seed %d for every arm",
        locked.config.sampling.seed,
    )
    if not randomness:
        raise SystemExit(
            "STOP: no locked parameter makes training random, so every seed gives the same "
            "model. The seed runs are not started."
        )
    plan = ArmsPlan(
        paths=paths,
        folds=study.target_folds,
        seeds=study.lightgbm_seeds,
        metrics=metrics,
        locked_num_trees=locked.num_trees,
        checkpoint_every=locked.config.checkpoint_every,
        binary_num_trees=study.lightgbm_binary_num_trees,
        tolerance=study.epoch_tolerance,
        launcher=subprocess_launcher(TRIAL_SCRIPT),
    )
    choice = run_arms(plan)
    LOGGER.info(
        "done | binary arm k* = %d (uses the last checkpoint: %s)",
        choice.num_trees,
        choice.uses_all_trees,
    )


if __name__ == "__main__":
    main()
