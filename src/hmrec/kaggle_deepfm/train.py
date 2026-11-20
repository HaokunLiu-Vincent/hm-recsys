"""Train the DeepFM model of a Kaggle run (docs/decisions.md 2026-11-20). Torch side.

The model is the DeepFM configuration chosen on V (configs/m9_t_eval.yaml) with the
seed of configs/kaggle_deepfm.yaml, trained from scratch on the 4 folds before
the predicted week, with the SAME sample files the LightGBM ranker of the run was
trained on:

    submission   T1 T2 T3 T4: the files of T1-T3 that M8 wrote, and the file of T4
                 that the LightGBM submission drew (artifacts/kaggle/submission/).
    rehearsal    V4 T1 T2 T3: the four files M8 and the stage-2 evaluation trained
                 their T4 models on.

A sample file is used only if its sha256 equals the one in its own record AND the
one in the Dataset record of the run's LightGBM ranker
(artifacts/kaggle/{run}/ranker/dataset/dataset.json): that record is the proof of
which files the LightGBM submission trained on. Nothing is sampled here.

The steps are the ones of a stage-2 trial, with the same functions
(hmrec.deepfm.trial, hmrec.deepfm.data): check the sample files, read them, fit
the input transform ON THESE TRAINING ROWS ONLY, encode, train with the run's
schedule, save the model after the last epoch. Only the target week differs: it
is scored by hmrec.kaggle_deepfm.score, in another process.

Labels: the label column of the sample files holds T labels, used here as
TRAINING labels. The read is recorded first (record_label_access). No label file
is opened and nothing is evaluated.

Files of a model, under artifacts/kaggle/deepfm/{run}/model/ (never replaced):
    transform.json   the fitted standardisation statistics
    epoch{T}.pt      the model after the last epoch T
    training.json    written LAST: marks the training as finished; holds the mean
                     training loss, the steps and the sha256 of the sample files

This module must not import LightGBM or a module that loads it.
"""

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from hmrec.data.folds import Fold
from hmrec.deepfm.data import (
    check_sample_files,
    feature_columns,
    lightgbm_sample_sha256,
    load_training_rows,
)
from hmrec.deepfm.inputs import encode_rows, save_transform
from hmrec.deepfm.process import load_trial_inputs, with_n_bins, with_seed
from hmrec.deepfm.records import deepfm_config_id
from hmrec.deepfm.study import Variant, apply_variant
from hmrec.deepfm.train import training_set
from hmrec.deepfm.trial import (
    TRANSFORM_FILE,
    PreparedTraining,
    TrialInputs,
    TrialSpec,
    check_run_settings,
    fit_window_transform,
    train_and_save,
)
from hmrec.kaggle_deepfm.context import DeepFMKaggleContext, record_label_access, training_folds
from hmrec.logging_utils import get_logger
from hmrec.memory import log_memory

LOGGER = get_logger("kaggle_deepfm.train")

AREA = "kaggle_deepfm"
EXPERIMENT = "full"
TRAINING_FILE = "training.json"
# The area of the LightGBM Datasets of the M8 formal run (hmrec.paths.final_dataset_dir).
M8_DATASET_AREA = "t_phase"
LABEL_PURPOSE = "DeepFM training labels: label column of the sample files"


def chosen_inputs(context: DeepFMKaggleContext, inputs: TrialInputs) -> TrialInputs:
    """The inputs of the run's training: configs/deepfm.yaml with the chosen settings and seed.

    Args:
        context: The process's context (the chosen settings, the seed, the id).
        inputs: The inputs read from the config files (read_base_inputs).

    Raises:
        ValueError: If the configuration id is not the recorded one, or the
            configuration is not for the "full" experiment.
    """
    chosen, settings = context.chosen, context.settings
    variant = Variant(
        learning_rate=chosen.learning_rate,
        embedding_dim=chosen.embedding_dim,
        dropout=chosen.dropout,
        components=chosen.components,
        numeric_input=chosen.numeric_input,
        loss=chosen.loss,
        epochs=chosen.epochs,
    )
    config = with_n_bins(apply_variant(inputs.config, variant), chosen.n_bins)
    config_id = deepfm_config_id(config)
    if config_id != settings.config_id:
        raise ValueError(
            f"the configuration has id {config_id}, but the configuration chosen on V has id "
            f"{settings.config_id}. Nothing is trained."
        )
    if config.experiment != EXPERIMENT:
        raise ValueError(f"experiment must be {EXPERIMENT!r}, got {config.experiment!r}")
    return replace(inputs, config=with_seed(config, settings.seed))


def read_base_inputs(context: DeepFMKaggleContext) -> TrialInputs:
    """The DeepFM inputs as the config files give them (no data is read)."""
    return load_trial_inputs(context.paths)


def run_sample_files(
    context: DeepFMKaggleContext, inputs: TrialInputs, folds: tuple[Fold, ...]
) -> dict[str, Path]:
    """Training fold -> the sample file the run's LightGBM ranker was trained on, in time order.

    A fold M8 sampled (it is in the Dataset record of M8's last model) has its
    file in the shared place; a fold the LightGBM submission drew itself (T4 in
    the real run) has its file under artifacts/kaggle/{run}/.
    """
    paths, ratio = context.paths, inputs.config.negatives_per_positive
    last_fold = context.kaggle.manifest.folds[-1].name
    m8_record = paths.final_dataset_dir(M8_DATASET_AREA, EXPERIMENT, last_fold, ratio)
    sampled_by_m8 = set(lightgbm_sample_sha256(m8_record))
    files = {}
    for fold in folds:
        if fold.name in sampled_by_m8:
            files[fold.name] = paths.ranker_sample_file(EXPERIMENT, fold.name, ratio)
        else:
            files[fold.name] = paths.kaggle_sample_file(context.run, fold.name, ratio)
    return files


@dataclass(frozen=True)
class TrainingSummary:
    """What the training of a run's model gave (also the content of training.json).

    Attributes:
        config_id: Id of the configuration.
        seed: The training seed.
        epochs: Epochs trained (the schedule length).
        train_folds: The training folds.
        sample_sha256: Fold -> the checked sha256 of its sample file.
        n_train_rows: Training rows.
        n_train_queries: Training queries.
        steps: Optimizer steps of the last epoch.
        mean_loss: Mean training loss of the last epoch (float64).
        train_seconds: Seconds of training (all epochs).
    """

    config_id: str
    seed: int
    epochs: int
    train_folds: tuple[str, ...]
    sample_sha256: dict[str, str]
    n_train_rows: int
    n_train_queries: int
    steps: int
    mean_loss: float
    train_seconds: float


def read_training_summary(model_dir: Path) -> TrainingSummary:
    """The summary of a finished training.

    Raises:
        FileNotFoundError: If the training is not finished.
    """
    file = model_dir / TRAINING_FILE
    if not file.is_file():
        raise FileNotFoundError(f"{file} not found: the model is not trained yet")
    raw = json.loads(file.read_text(encoding="utf-8"))
    return TrainingSummary(**{**raw, "train_folds": tuple(raw["train_folds"])})


def prepare_run_training(
    context: DeepFMKaggleContext, inputs: TrialInputs, script: str
) -> tuple[PreparedTraining, dict[str, str]]:
    """Record the label read, check the sample files, read and encode the training rows.

    Returns:
        (the encoded training rows with their transform, fold -> checked sha256).

    Raises:
        ValueError: If a sample file is not the one the run's LightGBM ranker
            trained on (sha256), or the folds differ from that Dataset's.
    """
    config = inputs.config
    folds = training_folds(context, config.window_weeks)
    files = run_sample_files(context, inputs, folds)
    dataset_dir = context.paths.kaggle_dataset_dir(context.run)
    # Recorded BEFORE a sample file is opened: its label column holds T labels.
    record_label_access(context, tuple(files), LABEL_PURPOSE, script)
    checked = check_sample_files(files, lightgbm_sample_sha256(dataset_dir))
    columns = feature_columns(inputs.lock)
    rows = load_training_rows(list(files.values()), columns)
    # The transform (standardisation statistics) is fitted on these training rows only.
    transform = fit_window_transform(inputs, rows, columns)
    # reuse_numeric: the standardised values are written over the raw values, as in a
    # stage-2 trial; `rows.raw` is not read after this line.
    encoded = encode_rows(rows.raw, transform, reuse_numeric=True)
    log_memory(LOGGER, f"{context.run}: {rows.raw.n_rows} training rows encoded")
    device = torch.device(config.training.device)
    data = training_set(encoded, rows.labels, rows.query_ids, device)
    return PreparedTraining(transform, data, data.boundaries.size - 1), checked


def train_run_model(
    context: DeepFMKaggleContext, inputs: TrialInputs, script: str
) -> TrainingSummary:
    """Train the run's model from scratch and save it.

    Args:
        context: The process's context.
        inputs: The run's configuration (chosen_inputs).
        script: The script that runs the training (written to the label access table).

    Returns:
        The summary; training.json exists when this function returns.

    Raises:
        FileExistsError: If the model folder exists (a model is never replaced).
        ValueError: If a check fails or the training loss is not finite.
    """
    config = inputs.config
    spec = TrialSpec(
        AREA, context.kaggle.target.name, context.settings.seed, config.training.epochs
    )
    check_run_settings(inputs, spec)
    model_dir = context.paths.kaggle_deepfm_model_dir(context.run)
    if model_dir.exists():
        raise FileExistsError(f"{model_dir} already exists; a model is never replaced")
    # As in a stage-2 trial: the thread count is set before anything is computed.
    torch.set_num_threads(inputs.base.num_threads)
    prepared, checked = prepare_run_training(context, inputs, script)
    model_dir.mkdir(parents=True)
    save_transform(prepared.transform, model_dir / TRANSFORM_FILE)
    started = time.perf_counter()
    epochs = train_and_save(inputs, spec, prepared, model_dir)
    last = epochs[-1]
    summary = TrainingSummary(
        config_id=deepfm_config_id(config),
        seed=spec.seed,
        epochs=spec.epochs,
        train_folds=tuple(checked),
        sample_sha256=checked,
        n_train_rows=prepared.data.n_rows,
        n_train_queries=prepared.n_queries,
        steps=last.steps,
        mean_loss=last.mean_loss,
        train_seconds=time.perf_counter() - started,
    )
    if not all(math.isfinite(epoch.mean_loss) for epoch in epochs):
        raise ValueError(f"the training loss is not finite: {[e.mean_loss for e in epochs]}")
    content = {**summary.__dict__, "train_folds": list(summary.train_folds)}
    (model_dir / TRAINING_FILE).write_text(json.dumps(content), encoding="utf-8")
    return summary
