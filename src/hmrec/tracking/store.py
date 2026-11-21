"""Writing to and reading from the local MLflow store (docs/mlops.md §7.3).

The store is one SQLite file, artifacts/mlflow/mlflow.db (not in git). Recent
MLflow versions refuse a plain folder of files as a store, so the backend is
SQLite; no server runs anywhere.

MLflow is loaded HERE and only when a function needs it (load_mlflow): it lives
in the dependency group "tracking", which the pipeline image does not install.
A static `import mlflow` at the top of a module would fail the type check and
the import inside the image.

Rules of the sync:
    - a record that is already in the store (same hmrec.record_id in the same
      experiment) is skipped: the sync never changes or duplicates a run;
    - every metric gets a tag "source.<key>" that says where its value came from
      (hmrec.tracking.records.MetricSource); hmrec.tracking.check uses it.
"""

import importlib
import importlib.util
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hmrec.tracking.records import RECORD_ID_TAG, TrackedRun

MLFLOW_MODULE = "mlflow"
# MLflow collects anonymous usage data by default. This project sends nothing to outside
# services, so it is switched off before MLflow is imported. The third variable silences a
# log line MLflow prints on every start (a hint for coding assistants; not telemetry).
TELEMETRY_OFF = {
    "MLFLOW_DISABLE_TELEMETRY": "true",
    "DO_NOT_TRACK": "true",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
}
# Limits of one MLflow log_batch call.
MAX_METRICS_PER_BATCH = 1000
MAX_PARAMS_PER_BATCH = 100
MAX_TAGS_PER_BATCH = 100
SEARCH_PAGE_SIZE = 500
FINISHED = "FINISHED"
MILLISECONDS = 1000


@dataclass(frozen=True)
class LoggedRun:
    """One run as it is in the MLflow store.

    Attributes:
        experiment: Name of its experiment.
        record_id: Its hmrec.record_id tag ("" if the run has none).
        mlflow_run_id: MLflow's id of the run.
        metrics: Metric key -> value.
        tags: Every tag of the run.
    """

    experiment: str
    record_id: str
    mlflow_run_id: str
    metrics: dict[str, float]
    tags: dict[str, str]


@dataclass(frozen=True)
class SyncSummary:
    """What a sync did.

    Attributes:
        logged: Record ids that were written to the store.
        skipped: Record ids that were already there and were left untouched.
    """

    logged: tuple[str, ...]
    skipped: tuple[str, ...]


def mlflow_is_installed() -> bool:
    """True if the dependency group "tracking" is installed in this environment."""
    return importlib.util.find_spec(MLFLOW_MODULE) is not None


def load_mlflow() -> Any:
    """Import MLflow with its usage telemetry switched off.

    Returns:
        The mlflow module (typed as Any: it is not installed in every environment).

    Raises:
        RuntimeError: If MLflow is not installed, with the command that installs it.
    """
    if not mlflow_is_installed():
        raise RuntimeError(
            "MLflow is not installed in this environment. It is in the dependency group "
            "'tracking' (not in the pipeline image): run `uv sync --group tracking`, or use "
            "`make mlflow-sync`."
        )
    for name, value in TELEMETRY_OFF.items():
        os.environ.setdefault(name, value)
    return importlib.import_module(MLFLOW_MODULE)


def store_uri(store_file: Path) -> str:
    """The MLflow tracking URI of a SQLite store file (an absolute path gives four slashes)."""
    return f"sqlite:///{store_file.resolve().as_posix()}"


def open_client(store_file: Path) -> Any:
    """An MlflowClient on the local store; the store's folder is created if needed."""
    mlflow = load_mlflow()
    store_file.parent.mkdir(parents=True, exist_ok=True)
    return mlflow.MlflowClient(tracking_uri=store_uri(store_file))


def ensure_experiment(client: Any, name: str, artifacts_dir: Path) -> str:
    """The id of the experiment `name`, creating it if it does not exist."""
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return str(experiment.experiment_id)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    location = artifacts_dir.resolve().as_uri()
    return str(client.create_experiment(name, artifact_location=location))


def read_experiment_runs(client: Any, experiment_name: str) -> list[LoggedRun]:
    """Every run of one experiment as it is in the store; empty if the experiment is missing."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    runs: list[LoggedRun] = []
    page_token = None
    while True:
        page = client.search_runs(
            [experiment.experiment_id], max_results=SEARCH_PAGE_SIZE, page_token=page_token
        )
        for run in page:
            tags = {str(key): str(value) for key, value in run.data.tags.items()}
            runs.append(
                LoggedRun(
                    experiment=experiment_name,
                    record_id=tags.get(RECORD_ID_TAG, ""),
                    mlflow_run_id=str(run.info.run_id),
                    metrics={str(key): float(value) for key, value in run.data.metrics.items()},
                    tags=tags,
                )
            )
        page_token = page.token
        if not page_token:
            return runs


def _batches(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    """`items` cut into pieces of at most `size`."""
    return [items[start : start + size] for start in range(0, len(items), size)]


def log_run(client: Any, experiment_id: str, run: TrackedRun) -> str:
    """Write one run to the store and close it.

    Args:
        client: An MlflowClient (open_client).
        experiment_id: The id of the run's experiment (ensure_experiment).
        run: The run to write.

    Returns:
        MLflow's id of the new run.
    """
    entities = load_mlflow().entities
    now_ms = int(time.time() * MILLISECONDS)
    created = client.create_run(experiment_id, run_name=run.run_name)
    run_id = str(created.info.run_id)
    metrics = [entities.Metric(metric.key, metric.value, now_ms, 0) for metric in run.metrics]
    params = [entities.Param(key, value) for key, value in run.params.items()]
    tags = [entities.RunTag(key, value) for key, value in run.all_tags().items()]
    for batch in _batches(metrics, MAX_METRICS_PER_BATCH):
        client.log_batch(run_id, metrics=batch)
    for batch in _batches(params, MAX_PARAMS_PER_BATCH):
        client.log_batch(run_id, params=batch)
    for batch in _batches(tags, MAX_TAGS_PER_BATCH):
        client.log_batch(run_id, tags=batch)
    client.set_terminated(run_id, FINISHED)
    return run_id


def sync_runs(client: Any, runs: Sequence[TrackedRun], artifacts_dir: Path) -> SyncSummary:
    """Write the runs that are not in the store yet; leave the others untouched.

    Args:
        client: An MlflowClient (open_client).
        runs: The runs the view should hold.
        artifacts_dir: Folder for the experiments' run files (artifacts/mlflow/artifacts/).

    Raises:
        ValueError: If two of the given runs have the same record id in one experiment.
    """
    identities = [(run.experiment, run.record_id) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("two runs of one experiment have the same record id")
    logged: list[str] = []
    skipped: list[str] = []
    present: dict[str, set[str]] = {}
    for run in runs:
        if run.experiment not in present:
            in_store = read_experiment_runs(client, run.experiment)
            present[run.experiment] = {logged_run.record_id for logged_run in in_store}
        if run.record_id in present[run.experiment]:
            skipped.append(run.record_id)
            continue
        experiment_id = ensure_experiment(client, run.experiment, artifacts_dir)
        log_run(client, experiment_id, run)
        present[run.experiment].add(run.record_id)
        logged.append(run.record_id)
    return SyncSummary(logged=tuple(logged), skipped=tuple(skipped))
