"""One DeepFM training of the M9 evaluation on T1-T4 (proposal §16.12). Torch side.

The training, scoring and evaluation code is the one of the V phase
(hmrec.deepfm.trial.run_trial_through). What differs is the door:

    rehearsal   the M7 window of the rehearsal fold, the Dataset record of the M8
                rehearsal, labels through load_t_eval_rehearsal_evaluator (the
                configured V fold only). No T fold can be reached.
    formal      the window of M8 (V and T folds), the Dataset record of the M8
                formal run, labels through load_t_eval_evaluator, which needs a
                TEvalPermit (hmrec.t_eval.gate).

In the formal run the permit is required BEFORE the door exists, so before any
file is opened: the sample files of T1-T3 hold T labels too.

The configuration is built from configs/deepfm.yaml and the settings of
configs/m9_t_eval.yaml; its configuration id must be the one recorded for the
chosen arm on V, otherwise nothing is trained.

This module must not import LightGBM or a module that loads it.
"""

from dataclasses import replace

from hmrec.config import TEvalConfig
from hmrec.deepfm.process import with_n_bins, with_seed
from hmrec.deepfm.records import deepfm_config_id
from hmrec.deepfm.study import Variant, apply_variant
from hmrec.deepfm.trial import TrialDoor, TrialInputs, TrialResult, TrialSpec, run_trial_through
from hmrec.eval.t_eval_labels import (
    TEvalPermit,
    load_t_eval_evaluator,
    load_t_eval_rehearsal_evaluator,
    require_t_eval_permit,
)
from hmrec.t_eval.plan import (
    EXPERIMENT,
    STAGE_FORMAL,
    TEvalContext,
    TrialKey,
    area_of,
    m8_dataset_dir,
    training_window,
)


def chosen_variant(config: TEvalConfig) -> Variant:
    """The seven study settings of the chosen configuration (configs/m9_t_eval.yaml)."""
    return Variant(
        learning_rate=config.learning_rate,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
        components=config.components,
        numeric_input=config.numeric_input,
        loss=config.loss,
        epochs=config.epochs,
    )


def chosen_trial_inputs(inputs: TrialInputs, config: TEvalConfig, seed: int) -> TrialInputs:
    """The inputs of one training: configs/deepfm.yaml with the chosen settings and the seed.

    Args:
        inputs: The inputs read from the config files (hmrec.deepfm.process).
        config: configs/m9_t_eval.yaml.
        seed: The training seed.

    Raises:
        ValueError: If the configuration id is not the one recorded for the chosen
            arm on V, or the configuration is not for the "full" experiment.
    """
    changed = with_n_bins(apply_variant(inputs.config, chosen_variant(config)), config.n_bins)
    config_id = deepfm_config_id(changed)
    if config_id != config.config_id:
        raise ValueError(
            f"{config.arm}: the configuration has id {config_id}, but the arm was chosen on V "
            f"with id {config.config_id}. Nothing is trained."
        )
    if changed.experiment != EXPERIMENT:
        raise ValueError(f"experiment must be {EXPERIMENT!r}, got {changed.experiment!r}")
    return replace(inputs, config=with_seed(changed, seed))


def t_eval_door(
    context: TEvalContext, inputs: TrialInputs, key: TrialKey, permit: TEvalPermit | None
) -> TrialDoor:
    """The door of one training: its window, its M8 Dataset record and its label loader.

    The fold checks come first; in the formal run the permit is required before
    the Dataset record (and so any sample file) is looked at.

    Raises:
        ValueError: If the training is not part of the evaluation, a fold check
            fails or the Dataset record is not the expected one.
        TypeError: In the formal run without a TEvalPermit.
    """
    config = inputs.config
    folds = training_window(context, key, config.window_weeks)
    if key.stage == STAGE_FORMAL:
        require_t_eval_permit(permit)
    dataset_dir = m8_dataset_dir(context, key, folds, config.negatives_per_positive)
    target = context.manifest.get(key.target_fold)
    labels_file = context.paths.labels_file(target.name)
    if key.stage == STAGE_FORMAL:
        return TrialDoor(
            folds, dataset_dir, lambda: load_t_eval_evaluator(target, labels_file, permit)
        )
    rehearsal_fold = context.config.rehearsal_target_fold
    return TrialDoor(
        folds,
        dataset_dir,
        lambda: load_t_eval_rehearsal_evaluator(target, labels_file, rehearsal_fold),
    )


def run_t_eval_trial(
    context: TEvalContext, inputs: TrialInputs, key: TrialKey, permit: TEvalPermit | None
) -> tuple[TrialInputs, TrialSpec, TrialResult]:
    """Train the chosen configuration for one target week and evaluate its final model.

    Args:
        context: The context of this process.
        inputs: The inputs read from the config files.
        key: The training (stage, seed, target fold).
        permit: The TEvalPermit (formal run); None in a rehearsal.

    Returns:
        (the inputs of the chosen configuration with the seed, the trial's spec,
        the result); the caller writes the result rows.
    """
    chosen = chosen_trial_inputs(inputs, context.config, key.seed)
    door = t_eval_door(context, chosen, key, permit)  # fold checks and permit first
    area = area_of(key.stage, context.stamp.git_commit)
    spec = TrialSpec(area, key.target_fold, key.seed, context.config.epochs)
    return chosen, spec, run_trial_through(chosen, spec, door)
