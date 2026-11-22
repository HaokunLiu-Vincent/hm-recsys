"""Tests for hmrec.tracking.registry: the model registry as a mirror of the pipeline's models.

The example: two ticks. The first trains a Ranker; the second (a resumed run,
so it lists the first step's output again) trains another Ranker and advances
the R6 chain. state/current.json names the second Ranker and that chain model.

MLflow is in the dependency group "tracking": the functions that read manifests
and the state are tested everywhere; writing to a real registry only where MLflow
is installed (`make test-tools`), each case with its own assertions.
"""

import dataclasses
import json
from datetime import date
from pathlib import Path

import pytest
from run_manifest_example import example_manifest

from hmrec.config import load_base_config
from hmrec.data_manifest import hash_file
from hmrec.paths import ProjectPaths, StateLayout, build_paths
from hmrec.run_manifest import FAILED, FileRecord, RunManifest
from hmrec.tracking.registry import (
    CHAIN_MODEL,
    CURRENT_ALIAS,
    RANKER_MODEL,
    MirroredModel,
    check_registry,
    current_models,
    mirrored_models,
    read_alias,
    read_versions,
    sync_registry,
    version_tags,
)
from hmrec.tracking.store import load_mlflow, mlflow_is_installed, open_client

REPO = Path(__file__).resolve().parents[2]
BASE = load_base_config(REPO / "configs" / "base.yaml")
MODELS = {
    "ranker_v1": "artifacts/pipeline/weeks/2020-07-29/ranker_model/model.txt",
    "ranker_v2": "artifacts/pipeline/weeks/2020-08-05/ranker_model/model.txt",
    "chain_v1": "artifacts/pipeline/weeks/2020-07-29/twotower/V1.pt",
}


def _workspace(root: Path) -> tuple[ProjectPaths, dict[str, FileRecord]]:
    """Three model files under root, and their records as a tick would write them."""
    paths = build_paths(BASE, root)
    records = {}
    for key, relative in MODELS.items():
        file = root / relative
        file.parent.mkdir(parents=True)
        file.write_text(f"weights of {key}")
        size, sha256, _ = hash_file(file)
        role = "r6_chain_model" if key.startswith("chain") else "ranker_model"
        records[key] = FileRecord(role, relative, size, sha256)
    return paths, records


def _tick(
    run_id: str, anchor: date, outputs: list[FileRecord], status: str = "succeeded"
) -> RunManifest:
    failure = "" if status == "succeeded" else "step O3_train stopped the tick"
    other = FileRecord("queries", f"artifacts/pipeline/{run_id}/queries.parquet", 10, "f" * 64)
    return dataclasses.replace(
        example_manifest(),
        run_id=run_id,
        anchor=anchor,
        status=status,
        failure=failure,
        outputs=(other, *outputs),
        metrics=(),
    )


def _manifests(records: dict[str, FileRecord]) -> list[RunManifest]:
    return [
        _tick("2020-07-29-eval-a1", date(2020, 7, 29), [records["ranker_v1"]]),
        # A failed run records what it wrote, but nothing of it is mirrored.
        _tick("2020-08-05-eval-a1", date(2020, 8, 5), [records["chain_v1"]], FAILED),
        _tick("2020-08-05-eval-a2", date(2020, 8, 5), [records["chain_v1"], records["ranker_v2"]]),
        # A resumed tick lists the outputs of its skipped steps again.
        _tick("2020-08-05-eval-a3", date(2020, 8, 5), [records["chain_v1"], records["ranker_v2"]]),
    ]


def _write_current(paths: ProjectPaths, records: dict[str, FileRecord]) -> None:
    current = StateLayout(paths.pipeline_state_dir).current_file
    current.parent.mkdir(parents=True)
    models = [
        {
            "role": "ranker",
            "path": "weeks/2020-08-05/ranker_model/model.txt",
            "sha256": records["ranker_v2"].sha256,
        },
        {
            "role": "twotower_chain",
            "path": "weeks/2020-07-29/twotower/V1.pt",
            "sha256": records["chain_v1"].sha256,
        },
    ]
    current.write_text(json.dumps({"models": models}))


def test_every_recorded_model_file_is_mirrored_once(tmp_path: Path) -> None:
    _, records = _workspace(tmp_path)
    models = mirrored_models(_manifests(records))
    assert [(model.name, model.path, model.run_id) for model in models] == [
        (RANKER_MODEL, MODELS["ranker_v1"], "2020-07-29-eval-a1"),
        (CHAIN_MODEL, MODELS["chain_v1"], "2020-08-05-eval-a2"),
        (RANKER_MODEL, MODELS["ranker_v2"], "2020-08-05-eval-a2"),
    ]
    # Other outputs (queries, candidates, ...) are not models.
    assert all(model.sha256 != "f" * 64 for model in models)
    tags = version_tags(models[1])
    assert tags["hmrec.sha256"] == records["chain_v1"].sha256
    assert (tags["hmrec.path"], tags["hmrec.tick_anchor"]) == (MODELS["chain_v1"], "2020-08-05")


def test_current_models_come_from_the_state_pointer(tmp_path: Path) -> None:
    paths, records = _workspace(tmp_path)
    assert current_models(paths) == {}  # no tick has published yet
    _write_current(paths, records)
    assert current_models(paths) == {
        RANKER_MODEL: records["ranker_v2"].sha256,
        CHAIN_MODEL: records["chain_v1"].sha256,
    }


def test_the_registry_mirrors_the_models_and_follows_the_current_pointer(tmp_path: Path) -> None:
    paths, records = _workspace(tmp_path)
    models = mirrored_models(_manifests(records))
    _write_current(paths, records)
    current = current_models(paths)
    if mlflow_is_installed():
        # The laptop with the "tracking" group: a real SQLite store in a temporary folder.
        client = open_client(tmp_path / "mlflow" / "mlflow.db")
        summary = sync_registry(client, models, current, tmp_path)
        assert len(summary.registered) == 3 and summary.present == ()
        rankers = read_versions(client, RANKER_MODEL)
        assert set(rankers) == {records["ranker_v1"].sha256, records["ranker_v2"].sha256}
        assert rankers[records["ranker_v2"].sha256]["hmrec.path"] == MODELS["ranker_v2"]
        # The alias names the models behind the current prediction set.
        assert read_alias(client, RANKER_MODEL) == records["ranker_v2"].sha256
        assert read_alias(client, CHAIN_MODEL) == records["chain_v1"].sha256
        assert check_registry(client, models, current, tmp_path) == []
        # A second sync creates nothing; the alias is set again.
        again = sync_registry(client, models, current, tmp_path)
        assert again.registered == () and len(again.present) == 3
        assert set(again.aliases) == {RANKER_MODEL, CHAIN_MODEL}
        # The pointer moves back (a rollback): the alias follows at the next sync.
        earlier = {**current, RANKER_MODEL: records["ranker_v1"].sha256}
        assert check_registry(client, models, earlier, tmp_path) != []
        sync_registry(client, models, earlier, tmp_path)
        assert read_alias(client, RANKER_MODEL) == records["ranker_v1"].sha256
        # A file that changed on disk is found by the check: the mirror names a sha256.
        (tmp_path / MODELS["chain_v1"]).write_text("other weights")
        problems = check_registry(client, models, earlier, tmp_path)
        assert problems == [
            f"{CHAIN_MODEL}: {MODELS['chain_v1']} no longer has the recorded sha256"
        ]
        assert CURRENT_ALIAS == "current"
    else:
        # The pipeline image: MLflow is not installed. Nothing can be written, and the
        # code says how to get MLflow instead of failing with an import error.
        with pytest.raises(RuntimeError, match="dependency group 'tracking'"):
            load_mlflow()
        with pytest.raises(RuntimeError, match="dependency group 'tracking'"):
            open_client(tmp_path / "mlflow" / "mlflow.db")
        assert all(isinstance(model, MirroredModel) for model in models)


def test_a_current_model_that_no_tick_wrote_gets_no_alias(tmp_path: Path) -> None:
    paths, records = _workspace(tmp_path)
    # The first tick predicts with the bootstrap state's chain model: no tick recorded it.
    models = mirrored_models(_manifests(records)[:1])
    current = {RANKER_MODEL: records["ranker_v1"].sha256, CHAIN_MODEL: "b" * 64}
    if mlflow_is_installed():
        client = open_client(tmp_path / "mlflow" / "mlflow.db")
        summary = sync_registry(client, models, current, tmp_path)
        assert summary.aliases == {RANKER_MODEL: "1"}
        assert read_alias(client, CHAIN_MODEL) is None
        assert check_registry(client, models, current, tmp_path) == []
    else:
        assert [model.name for model in models] == [RANKER_MODEL]
