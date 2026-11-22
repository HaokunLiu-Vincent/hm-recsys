"""What an execution of the Cloud Run job cost, from its run time (docs/mlops.md §12, M13).

A Cloud Run job is billed for the time its instance runs: the vCPUs and the
memory of the job's shape, each per second. This module turns the start and end
of an execution into billed vCPU-seconds, GiB-seconds and a cost, and keeps one
row per execution in results/m13_cloud/executions.csv.

It is an ESTIMATE from run times and list prices (configs/m13_cloud.yaml): the
bill itself is not read, the monthly free tier is not subtracted, and storage and
network are not included. The run time is the time from the start of the
execution to its completion as Cloud Run reports them.

Bottom layer: this module imports only the standard library and hmrec.config. It
names no project, bucket or account.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hmrec.config import check_keys, get_float, get_int, get_mapping, load_yaml

EXECUTION_COLUMNS: tuple[str, ...] = (
    "label",
    "kind",
    "execution",
    "started_by",
    "image_commit",
    "started_at_utc",
    "finished_at_utc",
    "wall_seconds",
    "cpu",
    "memory_gib",
    "vcpu_seconds",
    "gib_seconds",
    "cost_usd",
    "max_memory_fraction",
    "max_memory_gib",
    "state_bytes",
    "outcome",
)


@dataclass(frozen=True)
class CloudConfig:
    """configs/m13_cloud.yaml.

    Attributes:
        cpu: vCPUs of the job.
        memory_gib: Memory of the job in GiB.
        usd_per_vcpu_second: List price of one vCPU for one second.
        usd_per_gib_second: List price of one GiB of memory for one second.
    """

    cpu: int
    memory_gib: int
    usd_per_vcpu_second: float
    usd_per_gib_second: float


def parse_cloud_config(raw: Mapping[str, object]) -> CloudConfig:
    """Build a CloudConfig from the parsed YAML.

    Raises:
        ValueError: If a key is missing or unknown, or a value is not positive.
    """
    check_keys(raw, {"job", "prices"}, "m13_cloud.yaml")
    job = get_mapping(raw, "job", "m13_cloud.yaml")
    prices = get_mapping(raw, "prices", "m13_cloud.yaml")
    check_keys(job, {"cpu", "memory_gib"}, "m13_cloud.yaml: job")
    check_keys(prices, {"usd_per_vcpu_second", "usd_per_gib_second"}, "m13_cloud.yaml: prices")
    config = CloudConfig(
        cpu=get_int(job, "cpu", "job"),
        memory_gib=get_int(job, "memory_gib", "job"),
        usd_per_vcpu_second=get_float(prices, "usd_per_vcpu_second", "prices"),
        usd_per_gib_second=get_float(prices, "usd_per_gib_second", "prices"),
    )
    values = (config.cpu, config.memory_gib, config.usd_per_vcpu_second, config.usd_per_gib_second)
    if min(values) <= 0:
        raise ValueError(f"m13_cloud.yaml: every value must be > 0, got {config}")
    return config


def load_cloud_config(path: Path) -> CloudConfig:
    """Read configs/m13_cloud.yaml."""
    return parse_cloud_config(load_yaml(path))


@dataclass(frozen=True)
class ExecutionCost:
    """The billed quantities of one execution.

    Attributes:
        wall_seconds: Seconds from the start of the execution to its completion.
        vcpu_seconds: wall_seconds x vCPUs.
        gib_seconds: wall_seconds x GiB.
        cost_usd: vcpu_seconds x price + gib_seconds x price.
    """

    wall_seconds: float
    vcpu_seconds: float
    gib_seconds: float
    cost_usd: float


def execution_cost(started: datetime, finished: datetime, config: CloudConfig) -> ExecutionCost:
    """The cost of an execution that ran from `started` to `finished`.

    Example: 900 s at 6 vCPU and 24 GiB = 5,400 vCPU-s and 21,600 GiB-s
    = 5,400 x 0.000018 + 21,600 x 0.000002 = 0.0972 + 0.0432 = 0.1404 USD.

    Args:
        started: Start of the execution (timezone-aware).
        finished: Completion of the execution (timezone-aware).
        config: The job's shape and the prices.

    Raises:
        ValueError: If a time has no timezone or the end is before the start.
    """
    if started.tzinfo is None or finished.tzinfo is None:
        raise ValueError("the start and the end of an execution need a timezone")
    seconds = (finished - started).total_seconds()
    if seconds < 0:
        raise ValueError(f"the execution ended ({finished}) before it started ({started})")
    vcpu_seconds = seconds * config.cpu
    gib_seconds = seconds * config.memory_gib
    cost = vcpu_seconds * config.usd_per_vcpu_second + gib_seconds * config.usd_per_gib_second
    return ExecutionCost(seconds, vcpu_seconds, gib_seconds, cost)


def execution_row(
    label: str,
    kind: str,
    execution: str,
    started_by: str,
    image_commit: str,
    started: datetime,
    finished: datetime,
    max_memory_fraction: float | None,
    state_bytes: int | None,
    outcome: str,
    config: CloudConfig,
) -> dict[str, str]:
    """One row of results/m13_cloud/executions.csv.

    Args:
        label: What ran, e.g. "V1" or "smoke test".
        kind: "smoke", "calibration", "eval" or "serve".
        execution: The name Cloud Run gave the execution.
        started_by: "operator" or "scheduler".
        image_commit: The commit the job's image was built from.
        started: Start of the execution.
        finished: Completion of the execution.
        max_memory_fraction: The highest memory use of the instance as a share
            of its limit (Cloud Monitoring), or None if it was not read.
        state_bytes: Bytes under the state prefix after the execution, or None.
        outcome: "succeeded" or "failed".
        config: The job's shape and the prices.
    """
    cost = execution_cost(started, finished, config)
    fraction = "" if max_memory_fraction is None else repr(max_memory_fraction)
    memory = "" if max_memory_fraction is None else repr(max_memory_fraction * config.memory_gib)
    return {
        "label": label,
        "kind": kind,
        "execution": execution,
        "started_by": started_by,
        "image_commit": image_commit,
        "started_at_utc": started.isoformat(timespec="seconds"),
        "finished_at_utc": finished.isoformat(timespec="seconds"),
        "wall_seconds": repr(cost.wall_seconds),
        "cpu": str(config.cpu),
        "memory_gib": str(config.memory_gib),
        "vcpu_seconds": repr(cost.vcpu_seconds),
        "gib_seconds": repr(cost.gib_seconds),
        "cost_usd": repr(cost.cost_usd),
        "max_memory_fraction": fraction,
        "max_memory_gib": memory,
        "state_bytes": "" if state_bytes is None else str(state_bytes),
        "outcome": outcome,
    }
