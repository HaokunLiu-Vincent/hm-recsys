"""Tests for hmrec.tracking.cloud: the replay on Cloud Run in the local MLflow view (M13).

The records of a state in a bucket are copied to artifacts/m13_cloud/. Here the
copy is made by two ticks of the fake steps of weekly_example.py whose state root
is that folder: the run manifests are real ones, written by the driver.

MLflow is in the dependency group "tracking": reading manifests is tested
everywhere, writing to a real registry only where MLflow is installed
(`make test-tools`), as in test_tracking_registry.py.
"""

from pathlib import Path

import pytest
from weekly_example import STAMP, V1, V2, FakeSteps, build_workspace, patch_driver_environment

from hmrec.paths import PIPELINE_STATE_ROOT_VARIABLE, ProjectPaths, tick_root_paths
from hmrec.run_manifest import EVAL_MODE
from hmrec.tracking.cloud import (
    CLOUD_CALIBRATION_EXPERIMENT,
    CLOUD_CHAIN_MODEL,
    CLOUD_EXPERIMENT,
    CLOUD_RANKER_MODEL,
    cloud_models,
    cloud_runs,
)
from hmrec.tracking.registry import (
    CHAIN_MODEL,
    RANKER_MODEL,
    REMOTE_SOURCE_SCHEME,
    check_registry,
    read_alias,
    read_versions,
    sync_registry,
)
from hmrec.tracking.store import mlflow_is_installed, open_client
from hmrec.weekly.driver import run_tick


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectPaths:
    """A workspace with cloud records: two ticks of a replay and one calibration tick."""
    patch_driver_environment(monkeypatch)
    paths = build_workspace(tmp_path)
    environment = {PIPELINE_STATE_ROOT_VARIABLE: str(paths.cloud_state_records_dir)}
    state = tick_root_paths(paths, environment, calibration=False)
    for anchor in (V1, V2):
        assert run_tick(state, anchor, EVAL_MODE, FakeSteps(state), STAMP).succeeded
    calibration = tick_root_paths(paths, environment, calibration=True)
    assert calibration.pipeline_state_dir == paths.cloud_calibration_records_dir
    assert run_tick(calibration, V1, EVAL_MODE, FakeSteps(calibration), STAMP).succeeded
    return paths


def test_without_downloaded_records_the_view_has_no_cloud_run(tmp_path: Path) -> None:
    # A fresh clone, CI, an image: nothing was copied from a bucket.
    paths = build_workspace(tmp_path)
    assert cloud_runs(paths) == []
    assert cloud_models(paths) == ([], {})


def test_cloud_manifests_become_runs_with_ids_and_experiments_of_their_own(
    paths: ProjectPaths,
) -> None:
    runs = cloud_runs(paths)
    assert [(run.experiment, run.record_id) for run in runs] == [
        (CLOUD_EXPERIMENT, "pipeline-cloud/2020-07-29-eval-a1"),
        (CLOUD_EXPERIMENT, "pipeline-cloud/2020-08-05-eval-a1"),
        (CLOUD_CALIBRATION_EXPERIMENT, "pipeline-cloud-calibration/2020-07-29-eval-a1"),
    ]
    # The run id is the one of the laptop's replay; the record id is not, so the two
    # replays never share a record of the view.
    assert all(not run.record_id.startswith("pipeline/") for run in runs)
    # Every metric names the manifest it was read from, inside the copy of the records.
    first = runs[0]
    assert first.tags["hmrec.manifest"] == (
        "artifacts/m13_cloud/pipeline/runs/2020-07-29-eval-a1/manifest.json"
    )
    assert all(metric.source.file == first.tags["hmrec.manifest"] for metric in first.metrics)
    # The calibration tick is marked by its metric.
    assert any(metric.key.endswith("tick/calibration") for metric in runs[2].metrics)


def test_cloud_models_are_mirrored_under_registered_models_of_their_own(
    paths: ProjectPaths, tmp_path: Path
) -> None:
    models, current = cloud_models(paths)
    # Two Rankers (V1, V2) and the chain model the V2 tick wrote; nothing of the calibration.
    assert [(model.name, model.run_id) for model in models] == [
        (CLOUD_RANKER_MODEL, "2020-07-29-eval-a1"),
        (CLOUD_CHAIN_MODEL, "2020-08-05-eval-a1"),
        (CLOUD_RANKER_MODEL, "2020-08-05-eval-a1"),
    ]
    assert all(model.name not in (RANKER_MODEL, CHAIN_MODEL) for model in models)
    assert "calibration" not in " ".join(model.path for model in models)
    assert current == {CLOUD_RANKER_MODEL: models[2].sha256, CLOUD_CHAIN_MODEL: models[1].sha256}
    if not mlflow_is_installed():
        # The pipeline image and the default environment: only the reading is tested.
        return
    # The model files are in the bucket, not on this machine: remove them from the copy.
    for model in models:
        (paths.project_root / model.path).unlink()
    client = open_client(tmp_path / "mlflow" / "mlflow.db")
    summary = sync_registry(client, models, current, paths.project_root, files_on_disk=False)
    assert len(summary.registered) == 3
    assert read_alias(client, CLOUD_RANKER_MODEL) == models[2].sha256
    assert read_alias(client, CLOUD_CHAIN_MODEL) == models[1].sha256
    versions = read_versions(client, CLOUD_RANKER_MODEL)
    assert versions[models[0].sha256]["hmrec.path"] == models[0].path
    sources = {
        version.source for version in client.search_model_versions(f"name = '{CLOUD_CHAIN_MODEL}'")
    }
    assert sources == {f"{REMOTE_SOURCE_SCHEME}{models[1].path}"}
    # The check compares with the manifests and the pointer; it does not ask for the files.
    assert check_registry(client, models, current, paths.project_root, files_on_disk=False) == []
    # The same check for a state on this machine would miss every file.
    problems = check_registry(client, models, current, paths.project_root)
    assert len(problems) == 3 and all("is not on disk" in problem for problem in problems)
    # A pointer that names another model than the alias is still found.
    moved = {**current, CLOUD_RANKER_MODEL: models[0].sha256}
    assert check_registry(client, models, moved, paths.project_root, files_on_disk=False) != []
