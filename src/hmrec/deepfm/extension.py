"""The input-representation extension of the M9 study on V1-V4 (proposal §16.10).

Added on 2026-11-20, after the results of the 2x2 study were seen. Four runs,
all with the settings the 2x2 study kept for the 32-bin pointwise arm and
T = 1, each with every seed of configs/hpo/deepfm_extension.yaml:

    E1   binned with 255 bins, pointwise
    E2   binned with 255 bins, listwise
    E3   standardised input, pointwise
    E4   standardised input, listwise

The runs are trained and reported. Nothing is tuned and nothing is selected.

A run is a Variant (the seven study settings) plus its number of bins. The number
of bins is not one of a Variant's settings, so a run gets its own StudyContext:
the config with `n_bins` replaced, and a launcher that passes the same number to
the training process. The helpers of hmrec.deepfm.study_run then work unchanged:
a training that is already recorded is not started again (E3 with seed 42 is the
ablation run of the 2x2 study, because its configuration id is the same).

Every training goes through scripts/m9_deepfm_trial.py, which refuses T folds.
This module imports neither torch nor LightGBM.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from hmrec.config import DeepFMConfig, DeepFMExtensionConfig, DeepFMExtensionRun, DeepFMStudyConfig
from hmrec.deepfm.study import ConfigurationScore, Variant, variant_config_id
from hmrec.deepfm.study_run import StudyContext, TrialRequest, run_configuration
from hmrec.logging_utils import get_logger

LOGGER = get_logger("deepfm.extension")

N_BINS_ARGUMENT = "--n-bins"
# Arm names must not collide with the four arms of the 2x2 study.
BASE_ARMS = ("lgb_lambdarank", "lgb_binary", "deepfm_listwise", "deepfm_pointwise")

Launcher = Callable[[TrialRequest], None]


def run_variant(extension: DeepFMExtensionConfig, run: DeepFMExtensionRun) -> Variant:
    """The seven study settings of one run: the fixed ones plus its input and loss."""
    return Variant(
        learning_rate=extension.learning_rate,
        embedding_dim=extension.embedding_dim,
        dropout=extension.dropout,
        components=extension.components,
        numeric_input=run.numeric_input,
        loss=run.loss,
        epochs=extension.epochs,
    )


def run_config(config: DeepFMConfig, run: DeepFMExtensionRun) -> DeepFMConfig:
    """configs/deepfm.yaml with the run's number of bins (its other settings come by Variant)."""
    return replace(config, n_bins=run.n_bins)


def run_config_id(
    config: DeepFMConfig, extension: DeepFMExtensionConfig, run: DeepFMExtensionRun
) -> str:
    """The configuration id of one run."""
    return variant_config_id(run_config(config, run), run_variant(extension, run))


def n_bins_arguments(run: DeepFMExtensionRun) -> tuple[str, str]:
    """The extra command-line arguments of a run's trainings: ("--n-bins", "N")."""
    return (N_BINS_ARGUMENT, str(run.n_bins))


def check_extension(
    extension: DeepFMExtensionConfig, study: DeepFMStudyConfig, config: DeepFMConfig
) -> None:
    """Fail unless the extension can be compared with the arms of the 2x2 study.

    The eight-arm comparison puts the extension's arms next to the 2x2 arms, so
    both must use the same folds and the same seeds; an arm name must be new;
    and two runs must not be the same configuration.

    Raises:
        ValueError: If one of these does not hold.
    """
    if extension.target_folds != study.target_folds:
        raise ValueError(
            f"the extension's folds {extension.target_folds} are not the study's "
            f"{study.target_folds}"
        )
    if extension.seeds != study.final_seeds:
        raise ValueError(
            f"the extension's seeds {extension.seeds} are not the study's final seeds "
            f"{study.final_seeds}"
        )
    taken = [run.arm for run in extension.runs if run.arm in BASE_ARMS]
    if taken:
        raise ValueError(f"arm names {taken} belong to the 2x2 study")
    ids = [run_config_id(config, extension, run) for run in extension.runs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"two runs of the extension are the same configuration: {ids}")
    known = {*BASE_ARMS, *(run.arm for run in extension.runs)}
    unknown = [arm for pair in extension.contrasts for arm in pair if arm not in known]
    if unknown:
        raise ValueError(f"contrasts name unknown arms {unknown}; the arms are {sorted(known)}")


def run_context(
    context: StudyContext,
    run: DeepFMExtensionRun,
    launcher_for: Callable[[Sequence[str]], Launcher],
) -> StudyContext:
    """The StudyContext of one run: its number of bins in the config AND in the launcher.

    Args:
        context: The study's context (configs/deepfm.yaml as it is).
        run: The run.
        launcher_for: Extra command-line arguments -> a launcher that passes them
            to every training (hmrec.deepfm.study_run.subprocess_launcher).
    """
    return replace(
        context,
        config=run_config(context.config, run),
        launcher=launcher_for(n_bins_arguments(run)),
    )


@dataclass(frozen=True)
class RunResult:
    """One run of the extension with all its seeds.

    Attributes:
        run: The run.
        variant: Its seven study settings.
        score: Its recorded result with every seed.
    """

    run: DeepFMExtensionRun
    variant: Variant
    score: ConfigurationScore


def run_extension(
    context: StudyContext,
    extension: DeepFMExtensionConfig,
    launcher_for: Callable[[Sequence[str]], Launcher],
) -> tuple[RunResult, ...]:
    """Train what is missing of every run (in order) and return the recorded results.

    Args:
        context: The study's context; its launcher is not used.
        extension: configs/hpo/deepfm_extension.yaml.
        launcher_for: See run_context.

    Raises:
        ValueError: If the extension does not fit the study (check_extension).
    """
    check_extension(extension, context.study, context.config)
    results = []
    for run in extension.runs:
        own = run_context(context, run, launcher_for)
        variant = run_variant(extension, run)
        score = run_configuration(own, variant, extension.seeds)
        by_seed = ", ".join(f"seed {seed.seed}: {seed.pooled:.6f}" for seed in score.seeds)
        LOGGER.info(
            "%s (%s) | %s, T = %d: pooled %.6f (%s), seed range %.4f%%",
            run.name,
            run.arm,
            score.config_id,
            score.epochs,
            score.pooled,
            by_seed,
            100 * score.seed_range,
        )
        results.append(RunResult(run, variant, score))
    return tuple(results)
