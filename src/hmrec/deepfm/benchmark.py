"""The device benchmark of DeepFM and the device rule (proposal §16.2, §16.6).

For one device (CPU with the configured threads, or MPS), in its own process:

    - the REAL default model and the real encoded training rows of the speed
      test's target week, batches of `batch_rows` rows;
    - `benchmark_steps` training steps, each timed after the device has finished
      its work; the first `benchmark_discard_steps` are not counted (warm-up);
    - the same steps a second time from the same seed: are the losses identical?
    - scoring throughput: `benchmark_scoring_rows` rows through the model without
      gradients.

Device rule: MPS is used only if its median training step is at least
`min_mps_speedup` times faster than the CPU's; otherwise CPU.
"""

import time
from dataclasses import dataclass, replace

import numpy as np
import torch

from hmrec.deepfm.records import deepfm_config_id
from hmrec.deepfm.score import score_device_rows
from hmrec.deepfm.train import (
    TrainingSet,
    build_model,
    build_optimizer,
    pointwise_batches,
    synchronize,
    training_step,
)
from hmrec.deepfm.trial import PreparedTraining, TrialInputs, prepare_training
from hmrec.logging_utils import get_logger
from hmrec.ranker.window import m7_training_window

LOGGER = get_logger("deepfm.benchmark")

CPU = "cpu"
MPS = "mps"
BENCHMARK_DEVICES = (CPU, MPS)
MS_PER_SECOND = 1000.0
# The benchmark's batches are those of epoch 1 of a training with the same seed.
BENCHMARK_EPOCH = 1


@dataclass(frozen=True)
class BenchmarkResult:
    """The measurements of one device.

    Attributes:
        device: "cpu" or "mps".
        config_id: Id of the benchmarked configuration.
        target_fold: The fold whose training rows were used.
        steps: Training steps run.
        discarded_steps: First steps left out of the timing.
        median_step_ms: Median time of a timed step, in milliseconds.
        mean_step_ms: Mean time of a timed step.
        same_seed_identical: True if two runs from the same seed gave identical losses.
        same_seed_max_difference: Largest absolute difference between their losses.
        first_loss: Loss of the first step.
        last_loss: Loss of the last step.
        scoring_rows: Rows of the scoring measurement.
        scoring_rows_per_second: Its throughput.
    """

    device: str
    config_id: str
    target_fold: str
    steps: int
    discarded_steps: int
    median_step_ms: float
    mean_step_ms: float
    same_seed_identical: bool
    same_seed_max_difference: float
    first_loss: float
    last_loss: float
    scoring_rows: int
    scoring_rows_per_second: float


def timed_steps(
    inputs: TrialInputs, data: TrainingSet, prepared: PreparedTraining, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run `steps` training steps of a fresh model; returns (losses, seconds per step).

    Every step is timed until the device has finished it (MPS works
    asynchronously, so without the wait a step would look faster than it is).

    Returns:
        (float64 [steps] losses, float64 [steps] seconds).
    """
    training, device = inputs.config.training, data.labels.device
    model = build_model(prepared.transform.layout, inputs.config.model, training.seed, device)
    optimizer = build_optimizer(model, training)
    model.train()
    batches = pointwise_batches(data.n_rows, training.batch_rows, training.seed, BENCHMARK_EPOCH)
    losses, seconds = [], []
    for _, rows in zip(range(steps), batches, strict=False):
        started = time.perf_counter()
        row_tensor = torch.from_numpy(rows).to(device)
        loss = training_step(model, optimizer, data, row_tensor, None, 0)
        synchronize(device)
        seconds.append(time.perf_counter() - started)
        losses.append(float(loss.item()))
    if len(losses) != steps:
        raise ValueError(f"the training rows give only {len(losses)} batches, need {steps}")
    return np.array(losses, dtype=np.float64), np.array(seconds, dtype=np.float64)


def scoring_throughput(inputs: TrialInputs, data: TrainingSet, prepared: PreparedTraining) -> float:
    """Rows per second when scoring the first `benchmark_scoring_rows` training rows."""
    config, device = inputs.config, data.labels.device
    n_rows = min(config.speed_test.benchmark_scoring_rows, data.n_rows)
    model = build_model(prepared.transform.layout, config.model, config.training.seed, device)
    synchronize(device)
    started = time.perf_counter()
    score_device_rows(model, data.rows, n_rows, config.scoring_rows_per_chunk)
    synchronize(device)
    return n_rows / (time.perf_counter() - started)


def run_benchmark(inputs: TrialInputs, device_name: str) -> BenchmarkResult:
    """Benchmark the default configuration on one device (see the module docstring).

    Args:
        inputs: Configuration; its training.device is replaced by `device_name`.
        device_name: "cpu" or "mps".

    Raises:
        ValueError: On another device name, or if the target is not a V fold.
    """
    if device_name not in BENCHMARK_DEVICES:
        raise ValueError(f"device must be one of {BENCHMARK_DEVICES}, got {device_name!r}")
    config = inputs.config
    on_device = replace(
        inputs, config=replace(config, training=replace(config.training, device=device_name))
    )
    speed_test = config.speed_test
    torch.set_num_threads(inputs.base.num_threads)
    device = torch.device(device_name)
    folds = m7_training_window(inputs.manifest, speed_test.target_fold, config.window_weeks)
    prepared = prepare_training(on_device, speed_test.target_fold, folds, device)
    steps, discard = speed_test.benchmark_steps, speed_test.benchmark_discard_steps
    losses, seconds = timed_steps(on_device, prepared.data, prepared, steps)
    again, _ = timed_steps(on_device, prepared.data, prepared, steps)
    timed = seconds[discard:] * MS_PER_SECOND
    result = BenchmarkResult(
        device=device_name,
        config_id=deepfm_config_id(config),
        target_fold=speed_test.target_fold,
        steps=steps,
        discarded_steps=discard,
        median_step_ms=float(np.median(timed)),
        mean_step_ms=float(timed.mean()),
        same_seed_identical=bool(np.array_equal(losses, again)),
        same_seed_max_difference=float(np.abs(losses - again).max()),
        first_loss=float(losses[0]),
        last_loss=float(losses[-1]),
        scoring_rows=min(speed_test.benchmark_scoring_rows, prepared.data.n_rows),
        scoring_rows_per_second=scoring_throughput(on_device, prepared.data, prepared),
    )
    LOGGER.info("benchmark %s", result)
    return result


def choose_device(cpu_step_ms: float, mps_step_ms: float, min_mps_speedup: float) -> str:
    """The device rule: "mps" only if MPS is at least `min_mps_speedup` times faster than CPU.

    Args:
        cpu_step_ms: Median time of a training step on CPU.
        mps_step_ms: Median time of a training step on MPS.
        min_mps_speedup: From configs/deepfm.yaml (1.5).
    """
    if cpu_step_ms <= 0 or mps_step_ms <= 0:
        raise ValueError("step times must be > 0")
    return MPS if cpu_step_ms / mps_step_ms >= min_mps_speedup else CPU
