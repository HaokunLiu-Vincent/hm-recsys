"""Pre-training speed benchmark with a PLACEHOLDER Two-Tower model (proposal §1.4).

Not the R6 model (that is model.py / loss.py, still to be written). The
placeholder has the same inputs and roughly the same shape of work, so it shows
whether fetching batches or the model itself is the slow part:

    item tower: Linear(512 -> D) on the image vector (times its mask)
              + Linear(384 -> D) on the text vector (times its mask)
              + article ID embedding (padding_idx = n_articles)
              + one embedding per product_code / §8.4 category column (padding_idx = 0)
              -> L2-normalised [.., D]
    user tower: masked mean of the item tower over the 20 history slots
              + one embedding per profile column -> L2-normalised [B, D]
    loss:       in-batch softmax over the batch's positive articles. Another
                sample of the same group (same customer, same day) is removed
                from the softmax instead of being used as a negative. No LogQ.

Timing per step (§1.4): the device is synchronised before and after each part,
so asynchronous MPS work is counted in the part that queued it.
    fetch:   get_batch (index the device tensors)
    compute: forward + backward + optimizer step
"""

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from hmrec.config import TwoTowerBenchmarkConfig
from hmrec.recall.twotower.data import Batch, DeviceItemFeatures, DeviceSamples, get_batch


def synchronize(device: torch.device) -> None:
    """Wait until all queued work on the device is done (no-op on CPU)."""
    if device.type == "mps":
        torch.mps.synchronize()


class PlaceholderTwoTower(nn.Module):
    """A minimal two-tower model for timing only (see the module docstring)."""

    def __init__(
        self,
        items: DeviceItemFeatures,
        code_cardinalities: tuple[int, ...],
        profile_cardinalities: tuple[int, ...],
        embedding_dim: int,
    ) -> None:
        """Build the layers; `items` stays on its device and is not trained.

        Args:
            items: Item feature table with the padding row last (n_articles + 1 rows).
            code_cardinalities: Codes per article column (incl. code 0).
            profile_cardinalities: Codes per profile column (incl. code 0).
            embedding_dim: D, the output width.
        """
        super().__init__()
        self.items = items
        n_rows = int(items.image.shape[0])  # n_articles + 1 (padding row)
        self.image_projection = nn.Linear(int(items.image.shape[1]), embedding_dim)
        self.text_projection = nn.Linear(int(items.text.shape[1]), embedding_dim)
        self.article_embedding = nn.Embedding(n_rows, embedding_dim, padding_idx=n_rows - 1)
        self.code_embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim, padding_idx=0) for size in code_cardinalities
        )
        self.profile_embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim) for size in profile_cardinalities
        )

    def encode_items(self, article_idx: torch.Tensor) -> torch.Tensor:
        """Item vectors for int64 indices of any shape [...] -> float32 [..., D]."""
        items = self.items
        image_mask = items.image_mask[article_idx].unsqueeze(-1)
        text_mask = items.text_mask[article_idx].unsqueeze(-1)
        vector = self.image_projection(items.image[article_idx]) * image_mask
        vector = vector + self.text_projection(items.text[article_idx]) * text_mask
        vector = vector + self.article_embedding(article_idx)
        codes = items.codes[article_idx].long()
        for column, embedding in enumerate(self.code_embeddings):
            vector = vector + embedding(codes[..., column])
        return functional.normalize(vector, dim=-1)

    def encode_users(self, batch: Batch) -> torch.Tensor:
        """User vectors float32 [B, D]: masked mean of the history plus the profile."""
        history = self.encode_items(batch.history)  # [B, L, D]
        mask = batch.history_mask.unsqueeze(-1).float()
        # Mean over filled slots; an empty history gives 0 (count clamped to 1).
        pooled = (history * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        for column, embedding in enumerate(self.profile_embeddings):
            pooled = pooled + embedding(batch.profile[:, column])
        return functional.normalize(pooled, dim=-1)

    def forward(self, batch: Batch, temperature: float) -> torch.Tensor:
        """In-batch softmax loss (scalar) of one batch."""
        users = self.encode_users(batch)
        positives = self.encode_items(batch.article_idx)
        return in_batch_softmax_loss(users, positives, batch.group_id, temperature)


def make_placeholder_model(
    items: DeviceItemFeatures,
    code_cardinalities: tuple[int, ...],
    profile_cardinalities: tuple[int, ...],
    config: TwoTowerBenchmarkConfig,
) -> PlaceholderTwoTower:
    """Build the placeholder with seeded weights, on the device of `items`."""
    torch.manual_seed(config.seed)
    model = PlaceholderTwoTower(
        items, code_cardinalities, profile_cardinalities, config.embedding_dim
    )
    return model.to(items.image.device)


def in_batch_softmax_loss(
    users: torch.Tensor, items: torch.Tensor, group_id: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Cross-entropy of each user against all items of the batch; item i is user i's positive.

    Args:
        users: float32 [B, D].
        items: float32 [B, D], the positive article of each sample.
        group_id: int64 [B]. Item j with the same group as user i (j != i) is
            ANOTHER positive of user i, so it is removed from i's softmax.
        temperature: Logits are divided by it.
    """
    logits = users @ items.T / temperature  # [B, B]
    same_group = group_id.unsqueeze(1) == group_id.unsqueeze(0)
    is_self = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same_group & ~is_self, float("-inf"))
    targets = torch.arange(logits.shape[0], device=logits.device)
    return functional.cross_entropy(logits, targets)


@dataclass(frozen=True)
class BenchmarkTimes:
    """Seconds per step, one entry per step.

    Attributes:
        fetch_seconds: Time of get_batch.
        compute_seconds: Time of forward + backward + optimizer step.
        losses: Loss of each step (to see that training does something).
    """

    fetch_seconds: list[float]
    compute_seconds: list[float]
    losses: list[float]


def _train_step(
    model: PlaceholderTwoTower,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    temperature: float,
) -> torch.Tensor:
    """One forward + backward + update; returns the loss (still on the device)."""
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch, temperature)
    loss.backward()
    optimizer.step()
    return loss.detach()


def run_benchmark(
    model: PlaceholderTwoTower,
    samples: DeviceSamples,
    profiles: torch.Tensor,
    config: TwoTowerBenchmarkConfig,
    device: torch.device,
) -> BenchmarkTimes:
    """Train `config.steps` steps on random batches and time fetch and compute separately.

    The sample order is one random permutation (seeded, made on the CPU so it is
    the same on every device), created before timing starts.
    """
    generator = torch.Generator().manual_seed(config.seed)
    n_samples = int(samples.article_idx.shape[0])
    order = torch.randperm(n_samples, generator=generator).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    times = BenchmarkTimes([], [], [])
    for step in range(config.steps):
        start = (step * config.batch_size) % max(n_samples - config.batch_size, 1)
        synchronize(device)
        began = time.perf_counter()
        batch = get_batch(samples, profiles, order[start : start + config.batch_size])
        synchronize(device)
        fetched = time.perf_counter()
        loss = _train_step(model, optimizer, batch, config.temperature)
        synchronize(device)
        finished = time.perf_counter()
        times.fetch_seconds.append(fetched - began)
        times.compute_seconds.append(finished - fetched)
        times.losses.append(float(loss.item()))
    return times


@dataclass(frozen=True)
class BenchmarkSummary:
    """Totals of a benchmark; "steady" excludes the first warmup_steps steps.

    Attributes:
        steps: Number of steps.
        fetch_total: Seconds spent fetching batches (all steps).
        compute_total: Seconds of forward + backward + update (all steps).
        steady_fetch_mean: Mean fetch seconds per step after warm-up.
        steady_compute_mean: Mean compute seconds per step after warm-up.
        steady_fetch_share: fetch / (fetch + compute) after warm-up.
        first_loss: Loss of the first step.
        last_loss: Loss of the last step.
    """

    steps: int
    fetch_total: float
    compute_total: float
    steady_fetch_mean: float
    steady_compute_mean: float
    steady_fetch_share: float
    first_loss: float
    last_loss: float


def summarize_benchmark(times: BenchmarkTimes, warmup_steps: int) -> BenchmarkSummary:
    """Totals and steady-state means of the step times."""
    steady_fetch = times.fetch_seconds[warmup_steps:]
    steady_compute = times.compute_seconds[warmup_steps:]
    steady_fetch_sum = sum(steady_fetch)
    steady_total = steady_fetch_sum + sum(steady_compute)
    return BenchmarkSummary(
        steps=len(times.fetch_seconds),
        fetch_total=sum(times.fetch_seconds),
        compute_total=sum(times.compute_seconds),
        steady_fetch_mean=steady_fetch_sum / len(steady_fetch),
        steady_compute_mean=sum(steady_compute) / len(steady_compute),
        steady_fetch_share=steady_fetch_sum / steady_total if steady_total > 0 else 0.0,
        first_loss=times.losses[0],
        last_loss=times.losses[-1],
    )
