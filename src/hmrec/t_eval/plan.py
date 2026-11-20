"""What the M9 evaluation on T1-T4 trains, on which folds, from which files (proposal §16.12).

ONE configuration is trained: the DeepFM configuration chosen on V
(configs/m9_t_eval.yaml). Two stages run the same code path:

    rehearsal   predicts the configured V fold (V4) for every seed, with the M7
                window (F4 V1 V2 V3). Reads no T label. Its results must be
                bit-identical to the V4 records of the configuration
                (hmrec.t_eval.rehearsal). Its files go to the area
                "m9_t_rehearsal/{commit}", so a rehearsal belongs to one commit.
    formal      predicts the T folds for every seed, with the windows of M8:

                    T1 <- V1 V2 V3 V4      T2 <- V2 V3 V4 T1
                    T3 <- V3 V4 T1 T2      T4 <- V4 T1 T2 T3

                So the labels of T1, T2 and T3 are also TRAINING labels, as in M8;
                every training fold ends before the predicted week starts. Its
                files go to the area "m9_t".

Nothing is sampled here. The training rows are the sample files M8 trained on,
named by the record of the LightGBM Dataset M8 built for the target week
(artifacts/lgb_datasets/t_phase/full/ for a T fold, .../m8_rehearsal/full/ for
the rehearsal fold). m8_dataset_dir checks that the record is the one of the M8
formal run; the sha256 of every sample file is checked when it is read
(hmrec.deepfm.data.check_sample_files).

This module imports neither torch nor LightGBM.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from hmrec.config import TEvalConfig, load_t_eval_config
from hmrec.data.folds import Fold, FoldManifest, load_fold_manifest
from hmrec.paths import ProjectPaths
from hmrec.ranker.window import m7_training_window, training_folds
from hmrec.run_records import RunStamp

STAGE_REHEARSAL = "rehearsal"
STAGE_FORMAL = "formal"
STAGES = (STAGE_REHEARSAL, STAGE_FORMAL)
FORMAL_AREA = "m9_t"
REHEARSAL_AREA = "m9_t_rehearsal"
COMMIT_LENGTH = 12
# The areas of the LightGBM Datasets M8 built (hmrec.paths.final_dataset_dir).
M8_FORMAL_DATASET_AREA = "t_phase"
M8_REHEARSAL_DATASET_AREA = "m8_rehearsal"
EXPERIMENT = "full"
TARGET_PHASE = "T"
REHEARSAL_PHASE = "V"
TRAINING_PHASES = ("V", "T")
DATASET_RECORD_FILE = "dataset.json"


@dataclass(frozen=True)
class TEvalContext:
    """What every process of the evaluation works with.

    Attributes:
        paths: Project paths.
        config: configs/m9_t_eval.yaml.
        manifest: configs/fold_manifest.yaml.
        stamp: HEAD and whether the working tree is dirty.
    """

    paths: ProjectPaths
    config: TEvalConfig
    manifest: FoldManifest
    stamp: RunStamp


@dataclass(frozen=True)
class TrialKey:
    """One training of the evaluation.

    Attributes:
        stage: "rehearsal" or "formal".
        seed: The training seed.
        target_fold: The fold to predict.
    """

    stage: str
    seed: int
    target_fold: str


def check_config_folds(config: TEvalConfig, manifest: FoldManifest) -> None:
    """Fail unless the configured folds fit the fold manifest.

    Checked: every target fold is a T fold, they are in time order, and the
    rehearsal fold is a V fold.

    Raises:
        ValueError: On any violation.
    """
    names = list(manifest.names)
    for name in config.target_folds:
        if manifest.get(name).phase != TARGET_PHASE:
            raise ValueError(f"m9_t_eval.target_folds: {name} is not a {TARGET_PHASE} fold")
    positions = [names.index(name) for name in config.target_folds]
    if positions != sorted(positions):
        raise ValueError(f"m9_t_eval.target_folds must be in time order: {config.target_folds}")
    rehearsal_fold = manifest.get(config.rehearsal_target_fold)
    if rehearsal_fold.phase != REHEARSAL_PHASE:
        raise ValueError(
            f"m9_t_eval.rehearsal.target_fold {rehearsal_fold.name} is not a {REHEARSAL_PHASE} fold"
        )


def load_t_eval_context(paths: ProjectPaths, stamp: RunStamp) -> TEvalContext:
    """Read and check the configs of the evaluation (no data is read)."""
    config = load_t_eval_config(paths.t_eval_config_file)
    manifest = load_fold_manifest(paths.fold_manifest_file)
    check_config_folds(config, manifest)
    return TEvalContext(paths, config, manifest, stamp)


def area_of(stage: str, git_commit: str) -> str:
    """The area of a stage's files: "m9_t", or "m9_t_rehearsal/{first 12 of the commit}".

    Raises:
        ValueError: If the stage is unknown.
    """
    if stage == STAGE_FORMAL:
        return FORMAL_AREA
    if stage == STAGE_REHEARSAL:
        return f"{REHEARSAL_AREA}/{git_commit[:COMMIT_LENGTH]}"
    raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")


def check_trial_key(context: TEvalContext, key: TrialKey) -> None:
    """Refuse a training the evaluation does not contain, before anything is read.

    A rehearsal trains only the configured rehearsal fold; the formal run trains
    only the configured T folds. Both use only the configured seeds.

    Raises:
        ValueError: On any violation.
    """
    config = context.config
    if key.stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {key.stage!r}")
    if key.seed not in config.seeds:
        raise ValueError(f"the evaluation trains seeds {config.seeds}, got seed {key.seed}")
    if key.stage == STAGE_REHEARSAL and key.target_fold != config.rehearsal_target_fold:
        raise ValueError(
            f"a rehearsal predicts only the configured rehearsal fold "
            f"{config.rehearsal_target_fold}, got {key.target_fold}"
        )
    if key.stage == STAGE_FORMAL and key.target_fold not in config.target_folds:
        raise ValueError(
            f"the formal run predicts only the folds {config.target_folds}, got {key.target_fold}"
        )


def formal_training_window(
    manifest: FoldManifest, target_fold: str, window_weeks: int
) -> tuple[Fold, ...]:
    """The training folds of a model that predicts a T fold: the window of M8.

    Raises:
        ValueError: If the target is not a T fold or a training fold is not a V / T fold.
    """
    target = manifest.get(target_fold)
    if target.phase != TARGET_PHASE:
        raise ValueError(
            f"the formal run predicts only {TARGET_PHASE} folds, got fold {target.name} "
            f"(phase {target.phase})"
        )
    folds = training_folds(manifest, target_fold, window_weeks)
    for fold in folds:
        if fold.phase not in TRAINING_PHASES:
            raise ValueError(
                f"the formal run trains only on folds of phases {TRAINING_PHASES}, got fold "
                f"{fold.name} (phase {fold.phase})"
            )
    return folds


def training_window(context: TEvalContext, key: TrialKey, window_weeks: int) -> tuple[Fold, ...]:
    """The training folds of one training, after the stage's fold checks.

    The rehearsal uses the M7 window function (target V, training folds F / V), so
    a rehearsal can never reach a T fold.

    Raises:
        ValueError: If the training is not part of the evaluation or a fold check fails.
    """
    check_trial_key(context, key)
    if key.stage == STAGE_REHEARSAL:
        return m7_training_window(context.manifest, key.target_fold, window_weeks)
    return formal_training_window(context.manifest, key.target_fold, window_weeks)


def read_csv_table(path: Path) -> list[dict[str, str]]:
    """All rows of a result table as text dicts; empty if the file does not exist.

    Used for tables whose owning module imports torch or LightGBM, so that a
    module that loads neither can read them.
    """
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------- the M8 Datasets


def m8_formal_run_commit(paths: ProjectPaths) -> str:
    """The commit of the M8 formal run, from the first row of results/t_phase_label_access.csv.

    Raises:
        ValueError: If the table does not exist or is empty (M8 has not run).
    """
    access_file = paths.t_phase_label_access_file
    rows = read_csv_table(access_file)
    if not rows:
        raise ValueError(f"{access_file} has no row: the formal run of M8 is not recorded")
    return rows[0]["git_commit"]


def m8_dataset_dir(
    context: TEvalContext, key: TrialKey, folds: tuple[Fold, ...], ratio: int
) -> Path:
    """The folder of the LightGBM Dataset M8 built for the target week, after checking its record.

    For a T fold this is the Dataset of the M8 formal run; for the rehearsal fold
    the one of the M8 rehearsal (the same sample files as the M7 Dataset of that
    fold). Only the record (dataset.json) is used: it names, by sha256, the
    sample files M8 trained on. The record must name this target, this ratio and
    exactly these training folds, and must have been written at the commit of
    the M8 formal run.

    Args:
        context: The context of this process.
        key: The training.
        folds: Its training folds (training_window).
        ratio: Negatives per positive of the sample files.

    Raises:
        FileNotFoundError: If the record does not exist.
        ValueError: If the record is not the expected one.
    """
    area = M8_REHEARSAL_DATASET_AREA if key.stage == STAGE_REHEARSAL else M8_FORMAL_DATASET_AREA
    directory = context.paths.final_dataset_dir(area, EXPERIMENT, key.target_fold, ratio)
    record_file = directory / DATASET_RECORD_FILE
    if not record_file.is_file():
        raise FileNotFoundError(f"{record_file} not found: M8 built no Dataset for this week")
    record = json.loads(record_file.read_text(encoding="utf-8"))
    expected = {
        "experiment": EXPERIMENT,
        "target_fold": key.target_fold,
        "ratio": ratio,
        "train_folds": [fold.name for fold in folds],
        "git_commit": m8_formal_run_commit(context.paths),
    }
    different = {
        name: (record.get(name), value)
        for name, value in expected.items()
        if record.get(name) != value
    }
    if different:
        raise ValueError(
            f"{record_file}: not the expected Dataset record of M8; (recorded, expected) of "
            f"what differs: {different}"
        )
    if list(record["sample_sha256"]) != expected["train_folds"]:
        raise ValueError(f"{record_file}: sample_sha256 does not name the training folds")
    return directory
