"""CLI for M9: run the DeepFM part of the study on V1-V4 up to a stage (proposal §16.8).

    caffeinate -i uv run python scripts/m9_deepfm_study.py --through step0
    caffeinate -i uv run python scripts/m9_deepfm_study.py --through ablations --record

Stages, in order: step0 (the default configuration with three seeds and every
schedule length; the noise rule), tuning (pointwise: learning rate with its
schedule length, embedding dimension, dropout), pointwise (the final seeds),
listwise (its learning-rate step and the final seeds), ablations. Every training
runs in its own process (scripts/m9_deepfm_trial.py); recorded trainings are not
repeated, so the same command continues an interrupted study. A stop condition
(the default below the RRF baseline, a diverged or failed training, a memory
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
from hmrec.deepfm.study import ConfigurationScore, StudyStop  # noqa: E402
from hmrec.deepfm.study_run import (  # noqa: E402
    STAGES,
    StepOutcome,
    StudyContext,
    StudyResult,
    reference,
    run_study,
    study_rows,
    subprocess_launcher,
    write_study_table,
)
from hmrec.git_state import git_state  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.run_records import RunStamp  # noqa: E402

LOGGER = get_logger("m9_deepfm_study")
TRIAL_SCRIPT = Path(__file__).resolve().parent / "m9_deepfm_trial.py"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", required=True, choices=STAGES)
    parser.add_argument("--record", action="store_true", help="write results/deepfm_study.csv")
    return parser.parse_args()


def log_configuration(label: str, result: ConfigurationScore) -> None:
    """Log one scored configuration: its T, the seeds' pooled scores and their mean."""
    by_seed = ", ".join(f"seed {seed.seed}: {seed.pooled:.6f}" for seed in result.seeds)
    LOGGER.info(
        "%s | %s, T = %d: pooled %.6f (%s)",
        label,
        result.config_id,
        result.epochs,
        result.pooled,
        by_seed,
    )


def log_step(label: str, outcome: StepOutcome) -> None:
    """Log one tuning step: every candidate and the decision."""
    for value in outcome.step.candidates:
        log_configuration(f"{label} | {outcome.step.parameter} = {value}", outcome.scores[value])
    LOGGER.info(
        "%s | selected %s (%s); on the grid edge: %s",
        label,
        outcome.selected_value,
        outcome.decision.reason,
        outcome.on_grid_edge,
    )


def log_result(result: StudyResult) -> None:
    """Log what every finished stage decided."""
    step0 = result.step0
    for epochs, score in step0.schedule.scores.items():
        log_configuration(f"step 0 | T = {epochs}", score)
    LOGGER.info(
        "step 0 | selected T = %d, seed range %.4f%%, RRF baseline %.6f, tuning seeds %s",
        step0.schedule.selected_epochs,
        100 * step0.seed_range,
        step0.baseline,
        list(step0.tuning_seeds),
    )
    for outcome in result.tuning or ():
        log_step(f"tuning {outcome.step.name}", outcome)
    if result.pointwise is not None:
        LOGGER.info("pointwise | selected configuration: %s", result.pointwise_variant)
        log_configuration("pointwise | final seeds", result.pointwise)
    if result.listwise_step is not None and result.listwise is not None:
        log_step("listwise", result.listwise_step)
        LOGGER.info("listwise | selected configuration: %s", result.listwise_variant)
        log_configuration("listwise | final seeds", result.listwise)
    for name, score in (result.ablations or {}).items():
        log_configuration(f"ablation | {name}", score)


def main() -> None:
    """Run the stages up to --through and log what the rules decided."""
    args = parse_args()
    if args.record and args.through != STAGES[-1]:
        raise SystemExit(f"--record needs --through {STAGES[-1]}: it records the finished study")
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
        result = run_study(context, args.through)
    except StudyStop as stop:
        raise SystemExit(f"STOP: {stop}") from stop
    log_result(result)
    if args.record:
        rows = study_rows(result, context.study.selection_seed, reference(context).lightgbm)
        write_study_table(paths, rows, RunStamp(commit, dirty))
        LOGGER.info("study table -> %s (%d rows)", paths.deepfm_study_file, len(rows))


if __name__ == "__main__":
    main()
