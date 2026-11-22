"""The replay on Cloud Run in the local MLflow view (docs/mlops.md §7.3; M13).

The pipeline's state in the cloud lies in a bucket. Its SMALL records (run
manifests, markers, tables) are copied to the laptop by
`bash infra/operate.sh download-records`; this module turns the run manifests of
that copy into runs of the view, and the model files they record into model
versions:

    experiment "weekly-pipeline (cloud run)"              one run per run of a tick
    experiment "weekly-pipeline calibration (cloud run)"  the calibration ticks
    registered models "hmrec-ranker-cloud-run" and        one version per model file a
    "hmrec-twotower-chain-cloud-run"                      tick of the cloud replay wrote

The laptop's replay (M12) and the cloud's have the same run ids and their own
models, so the cloud's records have ids, experiments and registered models of
their own. The model files stay in the bucket: a version's source is the key the
run manifest recorded, and its sha256 is the manifest's; the file is not hashed
again on this machine (hmrec.tracking.registry, `files_on_disk=False`).

As for every record of the view: the manifests are the record, MLflow is a view.
Neither torch nor lightgbm; MLflow only through hmrec.tracking.store.
"""

from pathlib import Path

from hmrec.paths import ProjectPaths
from hmrec.run_manifest import find_run_manifests, read_run_manifest
from hmrec.tracking.records import TrackedRun, pipeline_run
from hmrec.tracking.registry import MirroredModel, current_models_of, mirrored_models

CLOUD_EXPERIMENT = "weekly-pipeline (cloud run)"
CLOUD_CALIBRATION_EXPERIMENT = "weekly-pipeline calibration (cloud run)"
CLOUD_EXPERIMENTS = (CLOUD_EXPERIMENT, CLOUD_CALIBRATION_EXPERIMENT)
CLOUD_RECORD_PREFIX = "pipeline-cloud"
CLOUD_CALIBRATION_RECORD_PREFIX = "pipeline-cloud-calibration"
CLOUD_RANKER_MODEL = "hmrec-ranker-cloud-run"
CLOUD_CHAIN_MODEL = "hmrec-twotower-chain-cloud-run"
# Output role of a run manifest -> registered model of the cloud replay.
CLOUD_ROLE_TO_MODEL = {"ranker_model": CLOUD_RANKER_MODEL, "r6_chain_model": CLOUD_CHAIN_MODEL}
# Role of a model in state/current.json -> registered model of the cloud replay.
CLOUD_CURRENT_ROLE_TO_MODEL = {"ranker": CLOUD_RANKER_MODEL, "twotower_chain": CLOUD_CHAIN_MODEL}


def runs_of(records_dir: Path, experiment: str, prefix: str, repo_root: Path) -> list[TrackedRun]:
    """One run of the view per run manifest under {records_dir}/runs/ (none if it is absent)."""
    return [
        pipeline_run(read_run_manifest(file), file, experiment, repo_root, prefix)
        for file in find_run_manifests(records_dir / "runs")
    ]


def cloud_runs(paths: ProjectPaths) -> list[TrackedRun]:
    """The runs of the cloud replay and of its calibration, from the copies of their records.

    Empty on a machine that never downloaded them (a fresh clone, CI, an image).
    """
    root = paths.project_root
    return [
        *runs_of(paths.cloud_state_records_dir, CLOUD_EXPERIMENT, CLOUD_RECORD_PREFIX, root),
        *runs_of(
            paths.cloud_calibration_records_dir,
            CLOUD_CALIBRATION_EXPERIMENT,
            CLOUD_CALIBRATION_RECORD_PREFIX,
            root,
        ),
    ]


def cloud_models(paths: ProjectPaths) -> tuple[list[MirroredModel], dict[str, str]]:
    """The model files of the cloud replay, and the ones behind its current prediction set.

    The calibration state is not mirrored: its models were written to measure a
    platform, and no prediction set of it is in use.
    """
    records = paths.cloud_state_records_dir
    manifests = [read_run_manifest(file) for file in find_run_manifests(records / "runs")]
    models = mirrored_models(manifests, CLOUD_ROLE_TO_MODEL)
    return models, current_models_of(records, CLOUD_CURRENT_ROLE_TO_MODEL)
