"""CLI for M9: run the DeepFM study on V1-V4 up to a stage (proposal §16.5).

    caffeinate -i uv run python scripts/m9_deepfm_study.py --through step0
    caffeinate -i uv run python scripts/m9_deepfm_study.py --through seeds --record

Stages, in order: step0 (noise), tuning (learning rate, embedding dimension,
dropout), final (epoch rule of the selected configuration, with the epoch-cap
extension), ablations, seeds. Every training runs in its own process
(scripts/m9_deepfm_trial.py); recorded trainings are not repeated, so the same
command continues an interrupted study. A stop condition (seed range above the
limit, default below the RRF baseline, a diverged or failed training, a memory
peak above the limit) ends the run with an error.

--record writes results/deepfm_study.csv once, after the last stage. Run it on
a clean working tree. T folds are never touched.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

from hmrec.config import (  # noqa: E402
    load_deepfm_config,
    load_deepfm_study_config,
    load_selection_rule,
)
from hmrec.deepfm.study import StudyStop  # noqa: E402
from hmrec.deepfm.study_run import (  # noqa: E402
    StudyContext,
    final_rows,
    reference,
    run_ablations,
    run_final,
    run_seeds,
    run_step0,
    run_tuning,
    selected_variant,
    step0_rows,
    subprocess_launcher,
    tuning_rows,
    write_study_table,
)
from hmrec.git_state import git_state  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.run_records import RunStamp  # noqa: E402

LOGGER = get_logger("m9_deepfm_study")
STAGES = ("step0", "tuning", "final", "ablations", "seeds")
TRIAL_SCRIPT = Path(__file__).resolve().parent / "m9_deepfm_trial.py"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", required=True, choices=STAGES)
    parser.add_argument("--record", action="store_true", help="write results/deepfm_study.csv")
    return parser.parse_args()


def main() -> None:
    """Run the stages up to --through and log what the rules decided."""
    args = parse_args()
    last = STAGES.index(args.through)
    if args.record and args.through != STAGES[-1]:
        raise SystemExit("--record needs --through seeds: the table records the finished study")
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.deepfm_log_file("study"))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    if dirty:
        raise SystemExit("the working tree has uncommitted changes: the study does not start")
    context = StudyContext(
        paths=paths,
        config=load_deepfm_config(paths.deepfm_config_file),
        study=load_deepfm_study_config(paths.deepfm_study_config_file),
        rule=load_selection_rule(paths.selection_rule_file),
        launcher=subprocess_launcher(TRIAL_SCRIPT),
    )
    try:
        step0 = run_step0(context)
    except StudyStop as stop:
        raise SystemExit(f"STOP: {stop}") from stop
    for result in step0.scores:
        LOGGER.info(
            "step 0 | seed %d: epoch %d, pooled %.6f, folds %s",
            result.seed,
            result.selected_epoch,
            result.pooled,
            result.fold_scores,
        )
    LOGGER.info(
        "step 0 | seed range %.4f%%, RRF baseline %.6f", 100 * step0.seed_range, step0.baseline
    )
    if last < STAGES.index("tuning"):
        return
    outcomes = run_tuning(context)
    selected = selected_variant(outcomes, context.default)
    LOGGER.info("tuning | selected configuration: %s", selected)
    if last < STAGES.index("final"):
        return
    final = run_final(context, selected)
    LOGGER.info(
        "final | extended: %s, epochs %d, selected epoch %d, pooled %.6f",
        final.extended,
        final.epochs,
        final.score.selected_epoch,
        final.score.pooled,
    )
    if last < STAGES.index("ablations"):
        return
    ablations = run_ablations(context, final)
    for name, result in ablations.items():
        LOGGER.info(
            "ablation | %s: epoch %d, pooled %.6f", name, result.selected_epoch, result.pooled
        )
    if last < STAGES.index("seeds"):
        return
    seeds = run_seeds(context, final)
    for result in seeds:
        LOGGER.info(
            "seeds | seed %d: epoch %d, pooled %.6f",
            result.seed,
            result.selected_epoch,
            result.pooled,
        )
    if args.record:
        rows = step0_rows(step0) + tuning_rows(outcomes)
        rows += final_rows(final, ablations, seeds, reference(context).lightgbm)
        write_study_table(paths, rows, RunStamp(commit, dirty))
        LOGGER.info("study table -> %s (%d rows)", paths.deepfm_study_file, len(rows))


if __name__ == "__main__":
    main()
