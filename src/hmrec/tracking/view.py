"""The whole MLflow view of this repository: which runs it holds, sync, check, UI.

The functions here put the pieces together for the scripts (scripts/m11_mlflow.py):

    view_runs    every run the view should hold, read from the repository's
                 result tables and from the run manifests that exist
    sync_view    write the runs that are missing to the local store
    check_view   compare the store with its sources
    view_models  the pipeline's model files the registry mirrors, and which of
                 them are behind the current prediction set
    sync_models  create the missing model versions; point the alias "current"
    check_models compare the registry with the manifests, the files and the state
    ui_command   the command that serves the local UI
"""

import sys

from hmrec.config import MlflowViewConfig
from hmrec.data.folds import FoldManifest
from hmrec.locks import RankerLock
from hmrec.paths import ProjectPaths
from hmrec.run_manifest import find_run_manifests, read_run_manifest
from hmrec.tracking.check import CheckResult, check_logged_runs
from hmrec.tracking.records import (
    TrackedRun,
    WeeklyTables,
    calibration_runs,
    pipeline_run,
    stage1_weekly_run,
)
from hmrec.tracking.registry import (
    MirroredModel,
    RegistrySummary,
    check_registry,
    current_models,
    mirrored_models,
    sync_registry,
)
from hmrec.tracking.store import (
    SyncSummary,
    open_client,
    read_experiment_runs,
    store_uri,
    sync_runs,
)

# The phases whose weeks have a stage-1 weekly record, in time order (the eight weeks of
# the replay, docs/mlops.md §6.3).
VALIDATION_PHASE = "V"
TEST_PHASE = "T"


def view_runs(
    paths: ProjectPaths, config: MlflowViewConfig, manifest: FoldManifest, lock: RankerLock
) -> list[TrackedRun]:
    """Every run the view should hold now.

    Args:
        paths: Project paths (result tables, run manifests).
        config: The names of the three experiments.
        manifest: All folds in time order.
        lock: The Ranker lock (which configuration's rows are the weekly records).

    Returns:
        The stage-1 weekly records of V1-V4 and T1-T4, the M10 calibration
        records, and one run per run manifest under artifacts/pipeline/runs/
        (none before M12).
    """
    root = paths.project_root
    weekly_tables = {
        VALIDATION_PHASE: WeeklyTables(
            paths.m6_candidates_results_file, paths.v_phase_metrics_file
        ),
        TEST_PHASE: WeeklyTables(paths.t_phase_recall_file, paths.t_phase_metrics_file),
    }
    runs: list[TrackedRun] = []
    for phase, tables in weekly_tables.items():
        for fold in manifest.by_phase(phase):
            runs.append(stage1_weekly_run(fold, tables, lock, config.experiment_stage1, root))
    runs.extend(
        calibration_runs(
            paths.calibration_comparisons_file,
            paths.calibration_runs_file,
            config.experiment_calibration,
            root,
        )
    )
    for manifest_file in find_run_manifests(paths.pipeline_runs_dir):
        tick = read_run_manifest(manifest_file)
        runs.append(pipeline_run(tick, manifest_file, config.experiment_pipeline, root))
    return runs


def sync_view(paths: ProjectPaths, runs: list[TrackedRun]) -> SyncSummary:
    """Write the runs that are not in the local store yet (hmrec.tracking.store.sync_runs)."""
    client = open_client(paths.mlflow_store_file)
    return sync_runs(client, runs, paths.mlflow_artifacts_dir)


def check_view(
    paths: ProjectPaths, config: MlflowViewConfig, runs: list[TrackedRun]
) -> CheckResult:
    """Compare every metric of the local store with its source (hmrec.tracking.check)."""
    client = open_client(paths.mlflow_store_file)
    logged = [run for name in config.experiments for run in read_experiment_runs(client, name)]
    return check_logged_runs(logged, runs, paths.project_root)


def view_models(paths: ProjectPaths) -> tuple[list[MirroredModel], dict[str, str]]:
    """The models the registry should mirror now, and the ones behind the current prediction set.

    Returns:
        (every model file a succeeded tick recorded in its run manifest;
        registered model -> sha256 of the model state/current.json names).
    """
    manifests = [read_run_manifest(file) for file in find_run_manifests(paths.pipeline_runs_dir)]
    return mirrored_models(manifests), current_models(paths)


def sync_models(
    paths: ProjectPaths, models: list[MirroredModel], current: dict[str, str]
) -> RegistrySummary:
    """Create the model versions that are missing and point the alias "current"."""
    client = open_client(paths.mlflow_store_file)
    return sync_registry(client, models, current, paths.project_root)


def check_models(
    paths: ProjectPaths, models: list[MirroredModel], current: dict[str, str]
) -> list[str]:
    """Compare the registry with the manifests, the files on disk and state/current.json."""
    client = open_client(paths.mlflow_store_file)
    return check_registry(client, models, current, paths.project_root)


def ui_command(paths: ProjectPaths, config: MlflowViewConfig) -> list[str]:
    """The command that serves the MLflow UI on the local store (this machine only)."""
    return [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        store_uri(paths.mlflow_store_file),
        "--host",
        config.ui_host,
        "--port",
        str(config.ui_port),
    ]
