"""Base training of the Two-Tower model (proposal §5 R6, §6.2).

- Samples: the base-window arrays (data.py). 1% of users (alarm users) are
  kept out of training; their loss is logged after every epoch as an alarm
  for bugs or divergence, and is NEVER used to choose anything (§6.2).
- Order: each epoch a new random permutation of the training samples, made on
  the CPU with seed (seed, epoch), so it is the same on every device.
  Consecutive slices of batch_size samples; the last partial batch is kept.
- Step: encode the batch (each distinct article once), multi-positive LogQ
  loss (loss.py), Adam with a CONSTANT learning rate, no weight decay, no
  gradient clipping.
- After each epoch: save artifacts/models/twotower/base/epoch_{k}.pt, so
  checkpoint k = "trained k epochs" (E_base candidates, §6.3).
"""

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hmrec.config import MemoryConfig, TwoTowerTrainingConfig
from hmrec.logging_utils import get_logger
from hmrec.memory import MemoryTracker, bytes_to_gb
from hmrec.recall.twotower.data import (
    Batch,
    CategoryCodes,
    DeviceItemFeatures,
    DeviceSamples,
    TrainingArrays,
    get_batch,
    samples_to_device,
    synchronize,
)
from hmrec.recall.twotower.loss import article_log_q, multi_positive_logq_loss
from hmrec.recall.twotower.model import TwoTowerModel

LOGGER = get_logger("recall.twotower.train")


# ---------------------------------------------------------------- samples and order


@dataclass(frozen=True)
class TrainingSplit:
    """Sample indices (int64) of the training set and of the alarm users.

    Attributes:
        train_indices: Samples used for training.
        alarm_indices: Samples of the alarm users (loss logged only).
        n_alarm_users: Number of alarm users.
    """

    train_indices: np.ndarray
    alarm_indices: np.ndarray
    n_alarm_users: int


def choose_alarm_users(customer_idx: np.ndarray, share: float, seed: int) -> np.ndarray:
    """The alarm users: round(share x distinct users) customers drawn with `seed`.

    The training chain calls this again on the BASE arrays, so it gets exactly
    the same users as base training and keeps them out of every fine-tune.

    Returns:
        Sorted int array of customer_idx.
    """
    users = np.unique(customer_idx)
    n_alarm_users = round(share * users.size)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(users, size=n_alarm_users, replace=False))


def split_alarm_users(customer_idx: np.ndarray, share: float, seed: int) -> TrainingSplit:
    """Keep round(share x distinct users) users, drawn with `seed`, out of training.

    Args:
        customer_idx: int [N], customer of each sample.
        share: Share of distinct users to keep out (0.01).
        seed: Seed of the draw.
    """
    alarm_users = choose_alarm_users(customer_idx, share, seed)
    n_alarm_users = alarm_users.size
    is_alarm = np.isin(customer_idx, alarm_users)
    return TrainingSplit(
        train_indices=np.flatnonzero(~is_alarm).astype(np.int64),
        alarm_indices=np.flatnonzero(is_alarm).astype(np.int64),
        n_alarm_users=n_alarm_users,
    )


def epoch_order(train_indices: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    """A random permutation of train_indices for one epoch (seeded by (seed, epoch), CPU)."""
    rng = np.random.default_rng([seed, epoch])
    return train_indices[rng.permutation(train_indices.size)]


def batch_slices(n_samples: int, batch_size: int) -> Iterator[slice]:
    """Consecutive slices of batch_size; the last one may be shorter (it is kept)."""
    for start in range(0, n_samples, batch_size):
        yield slice(start, min(start + batch_size, n_samples))


# ---------------------------------------------------------------- one step


@dataclass(frozen=True)
class TrainingContext:
    """Everything a step needs besides the model, all on the training device.

    Attributes:
        samples: Training arrays on the device.
        profiles: Profile codes int32 [n_customers, 3].
        features: Item feature table (padding row last).
        log_q: float32 [n_articles + 1], log Q(k) of the current training set.
        temperature: tau.
        device: The device of all tensors.
    """

    samples: DeviceSamples
    profiles: torch.Tensor
    features: DeviceItemFeatures
    log_q: torch.Tensor
    temperature: float
    device: torch.device


def batch_loss(model: TwoTowerModel, batch: Batch, context: TrainingContext) -> torch.Tensor:
    """Forward pass and loss of one batch (a scalar tensor with a graph)."""
    encoding = model.encode_batch(batch, context.features)
    return multi_positive_logq_loss(
        users=encoding.users,
        columns=encoding.columns,
        target_column=encoding.target_column,
        group_id=batch.group_id,
        column_log_q=context.log_q[encoding.column_article_idx],
        temperature=context.temperature,
    )


def train_step(
    model: TwoTowerModel,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    context: TrainingContext,
) -> torch.Tensor:
    """One forward + backward + Adam update; returns the loss (detached, on the device)."""
    optimizer.zero_grad(set_to_none=True)
    loss = batch_loss(model, batch, context)
    loss.backward()
    optimizer.step()
    return loss.detach()


def make_optimizer(
    model: TwoTowerModel, training: TwoTowerTrainingConfig, learning_rate: float
) -> torch.optim.Adam:
    """Adam with the fixed betas / epsilon / weight decay; constant learning rate."""
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(training.adam_beta1, training.adam_beta2),
        eps=training.adam_epsilon,
        weight_decay=training.weight_decay,
    )


@torch.no_grad()
def mean_loss(
    model: TwoTowerModel, indices: np.ndarray, batch_size: int, context: TrainingContext
) -> float:
    """Mean loss over `indices` in fixed order, no update (alarm users). 0.0 if empty."""
    losses: list[float] = []
    for part in batch_slices(indices.size, batch_size):
        index_tensor = torch.from_numpy(indices[part]).to(context.device)
        batch = get_batch(context.samples, context.profiles, index_tensor)
        losses.append(float(batch_loss(model, batch, context).item()))
    return float(np.mean(losses)) if losses else 0.0


# ---------------------------------------------------------------- epochs


@dataclass(frozen=True)
class EpochResult:
    """Numbers of one epoch.

    Attributes:
        epoch: 1-based epoch number.
        steps: Steps in the epoch.
        seconds: Wall time of the training steps.
        mean_train_loss: Mean loss of the epoch's steps.
        alarm_loss: Mean loss of the alarm users after the epoch (no update).
        peak_rss_gb: Sampled peak RSS of the process during the epoch (a lower bound).
    """

    epoch: int
    steps: int
    seconds: float
    mean_train_loss: float
    alarm_loss: float
    peak_rss_gb: float


def train_epoch(
    model: TwoTowerModel,
    optimizer: torch.optim.Optimizer,
    order: np.ndarray,
    batch_size: int,
    context: TrainingContext,
    log_every_steps: int,
) -> tuple[int, float]:
    """Run one pass over `order`; returns (steps, mean training loss)."""
    loss_sum = torch.zeros((), device=context.device)
    steps = 0
    for part in batch_slices(order.size, batch_size):
        index_tensor = torch.from_numpy(order[part]).to(context.device)
        batch = get_batch(context.samples, context.profiles, index_tensor)
        # Summed on the device: reading .item() every step would force a sync.
        loss_sum = loss_sum + train_step(model, optimizer, batch, context)
        steps += 1
        if steps % log_every_steps == 0:
            LOGGER.info("step %d: mean loss so far %.4f", steps, float(loss_sum.item()) / steps)
    return steps, float(loss_sum.item()) / steps


def run_base_training(
    model: TwoTowerModel,
    split: TrainingSplit,
    context: TrainingContext,
    training: TwoTowerTrainingConfig,
    memory: MemoryConfig,
    save_epoch: Callable[[int, EpochResult], None],
) -> list[EpochResult]:
    """Train `training.epochs` epochs; after each, compute the alarm loss and call save_epoch.

    Args:
        model: The model, already on context.device.
        split: Training / alarm sample indices.
        context: Device tensors and log Q of the training set.
        training: Batch size, learning rate, epochs, seed, logging.
        memory: Memory limit and sampling interval (configs/base.yaml).
        save_epoch: (epoch, result) -> None; saves the checkpoint and logs.
    """
    optimizer = make_optimizer(model, training, training.learning_rate)
    results: list[EpochResult] = []
    for epoch in range(1, training.epochs + 1):
        order = epoch_order(split.train_indices, training.seed, epoch)
        with MemoryTracker(
            f"two-tower epoch {epoch}",
            LOGGER,
            limit_gb=memory.limit_gb,
            sample_interval_seconds=memory.sample_interval_seconds,
        ) as tracker:
            synchronize(context.device)
            started = time.perf_counter()
            steps, train_loss = train_epoch(
                model, optimizer, order, training.batch_size, context, training.log_every_steps
            )
            synchronize(context.device)
            seconds = time.perf_counter() - started
            alarm_loss = mean_loss(model, split.alarm_indices, training.batch_size, context)
        peak = bytes_to_gb(tracker.peak_rss_bytes)
        result = EpochResult(epoch, steps, seconds, train_loss, alarm_loss, peak)
        save_epoch(epoch, result)
        results.append(result)
    return results


# ---------------------------------------------------------------- checkpoints


def save_checkpoint(model: TwoTowerModel, path: Path, metadata: dict[str, Any]) -> None:
    """Save the weights (on the CPU) and metadata; never overwrite a checkpoint.

    Raises:
        FileExistsError: If `path` exists (artifacts are never overwritten).
    """
    if path.exists():
        raise FileExistsError(f"{path} already exists; checkpoints are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    # Metadata goes through JSON so it holds only plain values (dates -> text).
    plain_metadata = json.loads(json.dumps(metadata, default=str))
    torch.save({"state_dict": state, "metadata": plain_metadata}, path)


def load_checkpoint(model: TwoTowerModel, path: Path) -> dict[str, Any]:
    """Load weights into `model` (strict: every tensor must match); returns the metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return checkpoint["metadata"]


def epoch_metadata(result: EpochResult) -> dict[str, Any]:
    """The EpochResult as a plain dict (for checkpoint metadata and run logs)."""
    return asdict(result)


def model_identity(
    n_articles: int, article_codes: CategoryCodes, profile_codes: CategoryCodes
) -> dict[str, Any]:
    """What the weights depend on besides their shapes: the code vocabularies.

    Stored in every checkpoint; check_model_identity compares it when loading,
    so a checkpoint is never used with codes that mean something else.
    """
    return {
        "n_articles": n_articles,
        "article_code_columns": list(article_codes.columns),
        "article_vocabularies": {k: list(v) for k, v in article_codes.vocabularies.items()},
        "profile_columns": list(profile_codes.columns),
        "profile_vocabularies": {k: list(v) for k, v in profile_codes.vocabularies.items()},
    }


def check_model_identity(stored: dict[str, Any], current: dict[str, Any], path: Path) -> None:
    """Fail if a checkpoint was trained with other article / profile codes than the current data."""
    for key, value in current.items():
        if stored.get(key) != value:
            raise ValueError(f"{path}: {key} differs from the current data; retrain the model")


# ---------------------------------------------------------------- weekly fine-tune (§6.4)


@dataclass(frozen=True)
class FineTuneArrays:
    """The arrays of one weekly fine-tune.

    Attributes:
        train: New-week rows + replay rows, alarm users excluded.
        alarm: The alarm users' new-week rows (loss only, never trained on).
        n_new: New-week rows in `train`.
        n_replay: Replay rows in `train`.
    """

    train: TrainingArrays
    alarm: TrainingArrays
    n_new: int
    n_replay: int


@dataclass(frozen=True)
class DeviceTables:
    """Tables every step reads, on the training device.

    Attributes:
        profiles: Profile codes int32 [n_customers, 3].
        features: Item feature table (padding row last).
    """

    profiles: torch.Tensor
    features: DeviceItemFeatures


@dataclass(frozen=True)
class FineTuneSchedule:
    """How long and with which randomness one week is fine-tuned.

    Attributes:
        epochs: E_update.
        seed: chain_seed.
        week_day: t_dat_day of the week's anchor (makes every week's shuffle different).
    """

    epochs: int
    seed: int
    week_day: int


@dataclass(frozen=True)
class FineTuneResult:
    """Numbers of one weekly fine-tune.

    Attributes:
        epochs: E_update.
        steps: Optimizer steps in total.
        n_train_samples: Fine-tune set size (new + replay, alarm users excluded).
        n_alarm_samples: Alarm users' samples of the new week (loss only).
        seconds: Wall time of the update steps.
        mean_train_loss: Mean loss of the last epoch's steps.
        alarm_loss: Mean loss of the alarm users' new-week samples after the update.
        peak_rss_gb: Sampled peak RSS during the fine-tune (a lower bound).
    """

    epochs: int
    steps: int
    n_train_samples: int
    n_alarm_samples: int
    seconds: float
    mean_train_loss: float
    alarm_loss: float
    peak_rss_gb: float


def fine_tune_order(n_samples: int, seed: int, week_day: int, epoch: int) -> np.ndarray:
    """A random permutation of 0..n-1 for one fine-tune epoch, seeded by (seed, week, epoch)."""
    rng = np.random.default_rng([seed, week_day, epoch])
    return rng.permutation(n_samples).astype(np.int64)


def _alarm_loss(
    model: TwoTowerModel,
    alarm_arrays: TrainingArrays,
    context: TrainingContext,
    batch_size: int,
) -> float:
    """Mean loss of the alarm samples (no update), with the fine-tune set's log Q."""
    if alarm_arrays.n_samples == 0:
        return 0.0
    alarm_context = replace(context, samples=samples_to_device(alarm_arrays, context.device))
    indices = np.arange(alarm_arrays.n_samples, dtype=np.int64)
    return mean_loss(model, indices, batch_size, alarm_context)


def fine_tune(
    model: TwoTowerModel,
    sets: FineTuneArrays,
    tables: DeviceTables,
    training: TwoTowerTrainingConfig,
    schedule: FineTuneSchedule,
    memory: MemoryConfig,
) -> FineTuneResult:
    """One weekly fine-tune of `model` in place (proposal §6.4).

    A NEW Adam optimizer (lr = fine_tune_learning_rate, other settings as in
    base training) is created on every call, so no optimizer state carries over
    from the previous week. log Q is counted on this fine-tune set only.

    Args:
        model: The current model (changed in place).
        sets: The week's fine-tune set and alarm samples.
        tables: Profile codes and item features on the training device.
        training: Batch size, temperature, Adam settings, fine-tune lr.
        schedule: Epochs (E_update), seed and the week's anchor day (for the shuffles).
        memory: Memory limit and sampling interval.
    """
    device = tables.features.image.device
    n_rows = int(tables.features.image.shape[0])  # n_articles + 1
    log_q = article_log_q(sets.train.article_idx, n_rows)
    context = TrainingContext(
        samples=samples_to_device(sets.train, device),
        profiles=tables.profiles,
        features=tables.features,
        log_q=torch.from_numpy(log_q).to(device),
        temperature=training.temperature,
        device=device,
    )
    optimizer = make_optimizer(model, training, training.fine_tune_learning_rate)
    model.train()
    with MemoryTracker(
        f"fine-tune week {schedule.week_day}",
        LOGGER,
        limit_gb=memory.limit_gb,
        sample_interval_seconds=memory.sample_interval_seconds,
    ) as tracker:
        started = time.perf_counter()
        steps, train_loss = 0, 0.0
        for epoch in range(1, schedule.epochs + 1):
            order = fine_tune_order(sets.train.n_samples, schedule.seed, schedule.week_day, epoch)
            epoch_steps, train_loss = train_epoch(
                model, optimizer, order, training.batch_size, context, training.log_every_steps
            )
            steps += epoch_steps
        synchronize(device)
        seconds = time.perf_counter() - started
        alarm_loss = _alarm_loss(model, sets.alarm, context, training.batch_size)
    model.eval()
    return FineTuneResult(
        epochs=schedule.epochs,
        steps=steps,
        n_train_samples=sets.train.n_samples,
        n_alarm_samples=sets.alarm.n_samples,
        seconds=seconds,
        mean_train_loss=train_loss,
        alarm_loss=alarm_loss,
        peak_rss_gb=bytes_to_gb(tracker.peak_rss_bytes),
    )
