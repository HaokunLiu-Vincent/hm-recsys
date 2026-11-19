"""Train a DeepFM model for a number of epochs (proposal §16.2, §16.3).

    pointwise   every epoch the training rows are put in a new random order and
                cut into batches of `batch_rows` rows (the last one may be smaller).
    listwise    every epoch the QUERIES are put in a new random order and packed,
                in that order, into batches of at most `batch_rows` rows; a query
                is never split, and a query larger than a batch is a batch alone.

Reproducibility: torch's seed is set before the model is built (initial weights,
dropout); the order of epoch e comes from numpy's default_rng([seed, e]). The same
seed on CPU with the same number of threads gives the same weights
(tests/unit/test_deepfm_train.py).

The whole encoded training set lives on the device as one uint8 tensor (0.76 GB
for 10.7 million rows); a batch is gathered from it by row index and only then
turned into int64 / float32.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from hmrec.config import DeepFMModelConfig, DeepFMTrainingConfig
from hmrec.deepfm.inputs import EncodedRows, FieldLayout
from hmrec.deepfm.loss import listwise_loss, pointwise_loss
from hmrec.deepfm.model import DeepFM
from hmrec.logging_utils import get_logger

LOGGER = get_logger("deepfm.train")

LISTWISE = "listwise_softmax"
MPS = "mps"


def synchronize(device: torch.device) -> None:
    """Wait until the device has finished its queued work (MPS runs asynchronously)."""
    if device.type == MPS:
        torch.mps.synchronize()


@dataclass(frozen=True)
class DeviceRows:
    """Encoded rows as tensors on the training device.

    Attributes:
        codes: uint8 [n_rows, n_code_fields].
        values: float32 [n_rows, n_value_fields], or None.
        missing: uint8 [n_rows, n_value_fields], or None.
    """

    codes: torch.Tensor
    values: torch.Tensor | None
    missing: torch.Tensor | None

    def batch(
        self, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """The model inputs of the given rows: (codes int64, values float32, missing float32)."""
        codes = self.codes[rows].long()
        if self.values is None or self.missing is None:
            return codes, None, None
        return codes, self.values[rows], self.missing[rows].float()


def rows_to_device(rows: EncodedRows, device: torch.device) -> DeviceRows:
    """Move encoded rows to the device (the arrays are not copied on CPU)."""
    values = None if rows.values is None else torch.from_numpy(rows.values).to(device)
    missing = None if rows.missing is None else torch.from_numpy(rows.missing).to(device)
    return DeviceRows(torch.from_numpy(rows.codes).to(device), values, missing)


# ---------------------------------------------------------------- batches


def pointwise_batches(n_rows: int, batch_rows: int, seed: int, epoch: int) -> Iterator[np.ndarray]:
    """Row indexes of the batches of one epoch: a random order cut into pieces of batch_rows.

    Args:
        n_rows: Training rows.
        batch_rows: Rows per batch.
        seed: The training seed.
        epoch: 1-based epoch number (a new order every epoch).

    Yields:
        int64 arrays; together they hold every row exactly once.
    """
    order = np.random.default_rng([seed, epoch]).permutation(n_rows)
    for start in range(0, n_rows, batch_rows):
        yield order[start : start + batch_rows]


def query_boundaries(query_ids: np.ndarray) -> np.ndarray:
    """First row of every query, plus the number of rows at the end: int64 [n_queries + 1].

    Raises:
        ValueError: If query_id decreases (the rows of a query would not be adjacent).
    """
    if query_ids.size == 0:
        raise ValueError("there are no training rows")
    steps = np.diff(query_ids)
    if np.any(steps < 0):
        raise ValueError("rows must be sorted by query_id")
    return np.concatenate(([0], np.flatnonzero(steps) + 1, [query_ids.size])).astype(np.int64)


def listwise_batches(
    boundaries: np.ndarray, batch_rows: int, seed: int, epoch: int
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Batches of whole queries for one epoch.

    Args:
        boundaries: From query_boundaries.
        batch_rows: Maximum rows per batch (a larger query is a batch alone).
        seed: The training seed.
        epoch: 1-based epoch number.

    Yields:
        (row indexes int64 [B], query index of each row int64 [B] in
        0 ... n_queries - 1, n_queries of the batch).
    """
    n_queries = boundaries.size - 1
    order = np.random.default_rng([seed, epoch]).permutation(n_queries)
    sizes = np.diff(boundaries)
    batch: list[int] = []
    rows_in_batch = 0
    for query in order.tolist():
        if batch and rows_in_batch + int(sizes[query]) > batch_rows:
            yield _whole_query_batch(batch, boundaries)
            batch, rows_in_batch = [], 0
        batch.append(query)
        rows_in_batch += int(sizes[query])
    if batch:
        yield _whole_query_batch(batch, boundaries)


def _whole_query_batch(
    queries: list[int], boundaries: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Row indexes and per-row query index of a batch made of the given queries."""
    rows = [np.arange(boundaries[query], boundaries[query + 1]) for query in queries]
    sizes = [piece.size for piece in rows]
    query_index = np.repeat(np.arange(len(queries), dtype=np.int64), sizes)
    return np.concatenate(rows), query_index, len(queries)


# ---------------------------------------------------------------- training


def build_model(
    layout: FieldLayout, config: DeepFMModelConfig, seed: int, device: torch.device
) -> DeepFM:
    """Build the model with initial weights fixed by `seed`, on the device."""
    torch.manual_seed(seed)
    return DeepFM(layout, config).to(device)


def build_optimizer(model: DeepFM, config: DeepFMTrainingConfig) -> torch.optim.Adam:
    """Adam with the configured learning rate and weight decay (default betas)."""
    return torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


@dataclass(frozen=True)
class TrainingSet:
    """The training rows on the device.

    Attributes:
        rows: The encoded inputs.
        labels: float32 [n_rows] on the device.
        boundaries: Query boundaries (query_boundaries); needed by the listwise loss.
    """

    rows: DeviceRows
    labels: torch.Tensor
    boundaries: np.ndarray

    @property
    def n_rows(self) -> int:
        """Training rows."""
        return int(self.labels.shape[0])


def training_set(
    rows: EncodedRows, labels: np.ndarray, query_ids: np.ndarray, device: torch.device
) -> TrainingSet:
    """Put encoded training rows and their labels on the device."""
    return TrainingSet(
        rows=rows_to_device(rows, device),
        labels=torch.from_numpy(labels.astype(np.float32)).to(device),
        boundaries=query_boundaries(query_ids),
    )


def training_step(
    model: DeepFM,
    optimizer: torch.optim.Optimizer,
    data: TrainingSet,
    rows: torch.Tensor,
    query_index: torch.Tensor | None,
    n_queries: int,
) -> torch.Tensor:
    """One optimizer step on the given rows; returns the batch loss (a 0-d tensor)."""
    codes, values, missing = data.rows.batch(rows)
    logits = model(codes, values, missing)
    labels = data.labels[rows]
    if query_index is None:
        loss = pointwise_loss(logits, labels)
    else:
        loss = listwise_loss(logits, labels, query_index, n_queries)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()


@dataclass(frozen=True)
class EpochResult:
    """What one epoch did.

    Attributes:
        epoch: 1-based epoch number.
        steps: Optimizer steps.
        mean_loss: Mean of the batch losses.
        seconds: Wall time of the epoch.
    """

    epoch: int
    steps: int
    mean_loss: float
    seconds: float


def train_epoch(
    model: DeepFM,
    optimizer: torch.optim.Optimizer,
    data: TrainingSet,
    config: DeepFMTrainingConfig,
    seed: int,
    epoch: int,
) -> EpochResult:
    """Run one epoch over all training rows (see the module docstring for the batches).

    Args:
        model: The model, changed in place.
        optimizer: Its optimizer.
        data: The training rows on the device.
        config: Loss, batch size.
        seed: The training seed (fixes the order of this epoch together with `epoch`).
        epoch: 1-based epoch number.
    """
    device = data.labels.device
    model.train()
    started = time.perf_counter()
    total_loss = torch.zeros((), device=device)
    steps = 0
    if config.loss == LISTWISE:
        for rows, query_index, n_queries in listwise_batches(
            data.boundaries, config.batch_rows, seed, epoch
        ):
            row_tensor = torch.from_numpy(rows).to(device)
            index_tensor = torch.from_numpy(query_index).to(device)
            total_loss += training_step(model, optimizer, data, row_tensor, index_tensor, n_queries)
            steps += 1
    else:
        for rows in pointwise_batches(data.n_rows, config.batch_rows, seed, epoch):
            row_tensor = torch.from_numpy(rows).to(device)
            total_loss += training_step(model, optimizer, data, row_tensor, None, 0)
            steps += 1
    synchronize(device)
    seconds = time.perf_counter() - started
    mean_loss = float(total_loss.item()) / steps
    LOGGER.info("epoch %d: %d steps, mean loss %.5f, %.1f s", epoch, steps, mean_loss, seconds)
    return EpochResult(epoch, steps, mean_loss, seconds)
