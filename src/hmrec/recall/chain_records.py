"""The record of the locked R6 training chain: results/m6_twotower_chain.csv (proposal §6.6).

The chain moves one fold at a time, F1 -> F2 -> ... -> T4:

    model that predicts F1   = the locked f1_start (sha256 in configs/locks/twotower.lock.yaml)
    model that predicts fold k (k > F1)
                             = the model saved after the fine-tune on fold k - 1,
                               artifacts/models/twotower/chain_locked/{k - 1}.pt

Every saved model gets ONE row in results/m6_twotower_chain.csv with its sha256
and the numbers of its fine-tune. start_model_for enforces the order: a fold can
only be processed if the previous fold's model exists AND still has the
recorded sha256. So no fold can be predicted by a model that skipped a week,
came from another run, or was replaced on disk.

The last fold (T4) is only predicted: it has no fine-tune, no model and no row.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hmrec.csv_records import append_row, read_rows
from hmrec.locks import check_file_sha256
from hmrec.recall.locked import ModelFile

CHAIN_COLUMNS: tuple[str, ...] = (
    "fold",
    "anchor",
    "start_model",
    "start_sha256",
    "model",
    "sha256",
    "n_new_rows",
    "n_replay_rows",
    "epochs",
    "steps",
    "n_train_samples",
    "n_alarm_samples",
    "mean_train_loss",
    "alarm_loss",
    "fine_tune_seconds",
    "sampled_peak_rss_gb",
    "git_commit",
    "git_dirty",
    "finished_at",
)


@dataclass(frozen=True)
class ChainRecord:
    """One saved chain model: the row of results/m6_twotower_chain.csv for one fold.

    Attributes:
        fold: The fold whose week the model was fine-tuned on, e.g. "F1".
        anchor: The fold's anchor (ISO date); the fine-tune week is [anchor, anchor + 7).
        start_model: File name of the model the fine-tune started from (the
            model that predicted this fold), e.g. "D4.pt" or "F1.pt".
        start_sha256: sha256 of that start model.
        model: File name of the saved model, "{fold}.pt".
        sha256: sha256 of the saved model. The next fold checks it before loading.
        n_new_rows: New-week rows of the fine-tune set (alarm users excluded).
        n_replay_rows: Replay rows (from the 4 weeks before the week).
        epochs: E_update.
        steps: Optimizer steps.
        n_train_samples: n_new_rows + n_replay_rows.
        n_alarm_samples: The alarm users' new-week samples (loss only).
        mean_train_loss: Mean training loss of the last epoch.
        alarm_loss: Mean loss of the alarm users' samples after the update.
        fine_tune_seconds: Wall time of the update steps.
        sampled_peak_rss_gb: SAMPLED peak RSS during the fine-tune (a lower
            bound; the true peak of the process is in results/m6_runs.csv).
        git_commit: HEAD when the run started.
        git_dirty: Uncommitted code / config changes when the run started.
        finished_at: ISO timestamp.
    """

    fold: str
    anchor: str
    start_model: str
    start_sha256: str
    model: str
    sha256: str
    n_new_rows: int
    n_replay_rows: int
    epochs: int
    steps: int
    n_train_samples: int
    n_alarm_samples: int
    mean_train_loss: float
    alarm_loss: float
    fine_tune_seconds: float
    sampled_peak_rss_gb: float
    git_commit: str
    git_dirty: bool
    finished_at: str


def record_to_row(record: ChainRecord) -> dict[str, str]:
    """A ChainRecord as one CSV row (floats with repr(), so they read back exactly)."""
    values = {name: getattr(record, name) for name in CHAIN_COLUMNS}
    return {
        name: repr(value) if isinstance(value, float) else str(value)
        for name, value in values.items()
    }


def row_to_record(row: Mapping[str, str]) -> ChainRecord:
    """Rebuild a ChainRecord from a CSV row written by record_to_row."""
    return ChainRecord(
        fold=row["fold"],
        anchor=row["anchor"],
        start_model=row["start_model"],
        start_sha256=row["start_sha256"],
        model=row["model"],
        sha256=row["sha256"],
        n_new_rows=int(row["n_new_rows"]),
        n_replay_rows=int(row["n_replay_rows"]),
        epochs=int(row["epochs"]),
        steps=int(row["steps"]),
        n_train_samples=int(row["n_train_samples"]),
        n_alarm_samples=int(row["n_alarm_samples"]),
        mean_train_loss=float(row["mean_train_loss"]),
        alarm_loss=float(row["alarm_loss"]),
        fine_tune_seconds=float(row["fine_tune_seconds"]),
        sampled_peak_rss_gb=float(row["sampled_peak_rss_gb"]),
        git_commit=row["git_commit"],
        git_dirty=row["git_dirty"] == "True",
        finished_at=row["finished_at"],
    )


def read_chain_records(path: Path) -> dict[str, ChainRecord]:
    """Fold -> its record; empty if the file does not exist yet.

    Raises:
        ValueError: If a fold has more than one row. A model is saved once and
            never replaced, so two rows mean the record cannot be trusted.
    """
    records: dict[str, ChainRecord] = {}
    for row in read_rows(path, CHAIN_COLUMNS):
        record = row_to_record(row)
        if record.fold in records:
            raise ValueError(f"{path} has more than one row for fold {record.fold}")
        records[record.fold] = record
    return records


def append_chain_record(path: Path, record: ChainRecord) -> None:
    """Append the row of a newly saved model.

    Raises:
        ValueError: If the fold already has a row.
    """
    if record.fold in read_chain_records(path):
        raise ValueError(f"{path} already has a row for fold {record.fold}")
    append_row(path, CHAIN_COLUMNS, record_to_row(record))


def start_model_for(
    fold: str,
    fold_names: Sequence[str],
    f1_start: ModelFile,
    records: Mapping[str, ChainRecord],
    checkpoint_file: Callable[[str], Path],
) -> ModelFile:
    """The model that predicts `fold`, after checking that the chain reached it in order.

    Args:
        fold: The fold to process.
        fold_names: The post-lock folds in time order (F1 ... T4).
        f1_start: The locked model that predicts the first fold, with its sha256.
        records: Fold -> record of results/m6_twotower_chain.csv.
        checkpoint_file: Fold -> artifacts/models/twotower/chain_locked/{fold}.pt.

    Returns:
        The start model and the sha256 it must have. For the first fold this is
        `f1_start` (its file is hashed by the loader); for a later fold the
        previous fold's model, already checked here.

    Raises:
        ValueError: If `fold` is unknown, the previous fold has no record yet
            (it was not fine-tuned: the chain may not skip a week), or the
            previous model's sha256 differs from its record.
        FileNotFoundError: If the previous fold's model file is missing.
    """
    if fold not in fold_names:
        raise ValueError(f"unknown fold {fold!r}; expected one of {list(fold_names)}")
    position = list(fold_names).index(fold)
    if position == 0:
        return f1_start
    previous = fold_names[position - 1]
    if previous not in records:
        raise ValueError(
            f"fold {fold} needs the model fine-tuned on {previous}, but {previous} has no row "
            "in results/m6_twotower_chain.csv: process the folds in time order"
        )
    path = checkpoint_file(previous)
    # Checked here as well as by the loader, so the order rule does not depend on
    # which function loads the weights.
    check_file_sha256(path, records[previous].sha256)
    return ModelFile(path, records[previous].sha256)


def check_model_and_record_agree(
    fold: str, records: Mapping[str, ChainRecord], checkpoint_file: Callable[[str], Path]
) -> bool:
    """True if `fold` was fine-tuned (model file and record both exist), False if neither.

    Raises:
        ValueError: If only one of the two exists. A model without a row, or a
            row without a model, is a half-finished or tampered state that a
            human must look at; nothing is repaired automatically.
    """
    has_model = checkpoint_file(fold).exists()
    has_record = fold in records
    if has_model != has_record:
        raise ValueError(
            f"fold {fold}: model file exists = {has_model}, but row in "
            f"results/m6_twotower_chain.csv exists = {has_record}; they must agree"
        )
    return has_model
