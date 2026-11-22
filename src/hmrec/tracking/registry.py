"""The model registry of the local MLflow view: a MIRROR of the pipeline's models (mlops §7.3).

Every tick of the weekly pipeline trains one Ranker and (when it closes a week)
advances the R6 chain by one model. Their files are pipeline state; each is
recorded in the tick's run manifest as an output with its path and sha256
(roles "ranker_model" and "r6_chain_model"). The registry mirrors exactly that:

    registered model "hmrec-ranker"            one version per weekly Ranker
    registered model "hmrec-twotower-chain"    one version per R6 chain model

A version is a POINTER, not a copy: its source is the file's path and its tags
hold the sha256, the week and the run that wrote it. No model file is uploaded
or moved.

One alias, "current", marks the version behind the current prediction set, as
state/current.json says (the Ranker and the chain model it names, matched by
sha256).

**The registry is a mirror.** The pipeline and the endpoint never read MLflow;
what is deployed is decided by state/current.json. The registry can be deleted
with the store and rebuilt by `make mlflow-sync`.

This module needs MLflow only through the client it is given; mirrored_models
and current_models read files and need none. Neither torch nor lightgbm.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hmrec.data_manifest import hash_file
from hmrec.paths import ProjectPaths, StateLayout
from hmrec.run_manifest import SUCCEEDED, RunManifest

RANKER_MODEL = "hmrec-ranker"
CHAIN_MODEL = "hmrec-twotower-chain"
# Output role of a run manifest -> registered model.
ROLE_TO_MODEL = {"ranker_model": RANKER_MODEL, "r6_chain_model": CHAIN_MODEL}
# Role of a model in state/current.json -> registered model.
CURRENT_ROLE_TO_MODEL = {"ranker": RANKER_MODEL, "twotower_chain": CHAIN_MODEL}
CURRENT_ALIAS = "current"
SHA256_TAG = "hmrec.sha256"
PATH_TAG = "hmrec.path"
RUN_TAG = "hmrec.pipeline_run_id"
ANCHOR_TAG = "hmrec.tick_anchor"
SIZE_TAG = "hmrec.size_bytes"
MIRROR_TAG = "hmrec.mirror"
MIRROR_NOTE = "pointer to pipeline state; nothing reads the registry to deploy"
SEARCH_PAGE_SIZE = 200


@dataclass(frozen=True)
class MirroredModel:
    """One model file of the pipeline, as the registry shows it.

    Attributes:
        name: The registered model: "hmrec-ranker" or "hmrec-twotower-chain".
        path: The file, relative to the repository root, with "/".
        sha256: sha256 of the file as its tick recorded it.
        size_bytes: Size of the file as its tick recorded it.
        run_id: The run of the tick that wrote the file.
        tick_anchor: Anchor of that tick (ISO date).
    """

    name: str
    path: str
    sha256: str
    size_bytes: int
    run_id: str
    tick_anchor: str


@dataclass(frozen=True)
class RegistrySummary:
    """What a sync of the registry did.

    Attributes:
        registered: (model name, sha256) of the versions that were created.
        present: (model name, sha256) of the versions that were already there.
        aliases: Model name -> the version number the alias "current" points to now.
    """

    registered: tuple[tuple[str, str], ...]
    present: tuple[tuple[str, str], ...]
    aliases: dict[str, str]


def mirrored_models(manifests: Sequence[RunManifest]) -> list[MirroredModel]:
    """The models the registry should hold: every model file a succeeded tick recorded.

    A resumed tick lists the outputs of its skipped steps again, so the same file
    can appear in several manifests; it is mirrored once (by model name and
    sha256), with the first run that recorded it.

    Args:
        manifests: The run manifests of the pipeline, in the order of their run ids.

    Returns:
        The models in the order they were first recorded.
    """
    models: dict[tuple[str, str], MirroredModel] = {}
    for manifest in manifests:
        if manifest.status != SUCCEEDED:
            continue
        for output in manifest.outputs:
            name = ROLE_TO_MODEL.get(output.role)
            if name is None or (name, output.sha256) in models:
                continue
            models[(name, output.sha256)] = MirroredModel(
                name=name,
                path=output.path,
                sha256=output.sha256,
                size_bytes=output.size_bytes,
                run_id=manifest.run_id,
                tick_anchor=manifest.anchor.isoformat(),
            )
    return list(models.values())


def current_models(paths: ProjectPaths) -> dict[str, str]:
    """Registered model -> sha256 of the model behind the current prediction set.

    Read from state/current.json; empty if no tick has published yet.
    """
    current_file = StateLayout(paths.pipeline_state_dir).current_file
    if not current_file.is_file():
        return {}
    raw = json.loads(current_file.read_text(encoding="utf-8"))
    return {
        CURRENT_ROLE_TO_MODEL[model["role"]]: model["sha256"]
        for model in raw["models"]
        if model["role"] in CURRENT_ROLE_TO_MODEL
    }


def version_tags(model: MirroredModel) -> dict[str, str]:
    """The tags of a model version (everything that identifies the file)."""
    return {
        SHA256_TAG: model.sha256,
        PATH_TAG: model.path,
        SIZE_TAG: str(model.size_bytes),
        RUN_TAG: model.run_id,
        ANCHOR_TAG: model.tick_anchor,
        MIRROR_TAG: MIRROR_NOTE,
    }


def read_versions(client: Any, name: str) -> dict[str, dict[str, str]]:
    """sha256 -> {"version": number, plus the tags} of every version of a registered model.

    Empty if the registered model does not exist.
    """
    versions: dict[str, dict[str, str]] = {}
    page_token = None
    while True:
        page = client.search_model_versions(
            f"name = '{name}'", max_results=SEARCH_PAGE_SIZE, page_token=page_token
        )
        for version in page:
            tags = {str(key): str(value) for key, value in version.tags.items()}
            versions[tags.get(SHA256_TAG, "")] = {**tags, "version": str(version.version)}
        page_token = page.token
        if not page_token:
            return versions


def ensure_registered_model(client: Any, name: str) -> None:
    """Create the registered model if it does not exist yet."""
    names = {model.name for model in client.search_registered_models(f"name = '{name}'")}
    if name not in names:
        client.create_registered_model(name, tags={MIRROR_TAG: MIRROR_NOTE})


def sync_registry(
    client: Any, models: Sequence[MirroredModel], current: dict[str, str], repo_root: Path
) -> RegistrySummary:
    """Create the missing model versions and point the alias "current" (see the module docstring).

    A version that is already there (same registered model, same sha256) is left
    untouched; the alias is moved, because "current" changes with every tick.

    Args:
        client: An MlflowClient (hmrec.tracking.store.open_client).
        models: The models the registry should hold (mirrored_models).
        current: Registered model -> sha256 of the current model (current_models).
        repo_root: The repository root; a version's source is the file's URI under it.
    """
    registered: list[tuple[str, str]] = []
    present: list[tuple[str, str]] = []
    for name in sorted({model.name for model in models}):
        ensure_registered_model(client, name)
        existing = read_versions(client, name)
        for model in (item for item in models if item.name == name):
            if model.sha256 in existing:
                present.append((name, model.sha256))
                continue
            source = (repo_root / model.path).resolve().as_uri()
            client.create_model_version(name, source, tags=version_tags(model))
            registered.append((name, model.sha256))
    aliases: dict[str, str] = {}
    for name, sha256 in sorted(current.items()):
        version = read_versions(client, name).get(sha256)
        if version is None:
            # The model behind the prediction set is not a tick's output: the first
            # tick predicts with the chain model of the bootstrap state.
            continue
        client.set_registered_model_alias(name, CURRENT_ALIAS, version["version"])
        aliases[name] = version["version"]
    return RegistrySummary(tuple(registered), tuple(present), aliases)


def read_alias(client: Any, name: str) -> str | None:
    """The sha256 of the version the alias "current" points to; None if there is no alias."""
    registered = client.search_registered_models(f"name = '{name}'")
    for model in registered:
        version = dict(model.aliases).get(CURRENT_ALIAS)
        if version is not None:
            tags = client.get_model_version(name, str(version)).tags
            return str(tags.get(SHA256_TAG, ""))
    return None


def check_registry(
    client: Any, models: Sequence[MirroredModel], current: dict[str, str], repo_root: Path
) -> list[str]:
    """Compare the registry with the manifests, the files on disk and state/current.json.

    For every model a manifest recorded: a version with its sha256 exists, the
    version's path tag is the recorded path, and the file on disk still has that
    sha256 (it is hashed again here). For the alias: it points to the model
    state/current.json names.

    Returns:
        One line per difference; empty if the registry mirrors the state.
    """
    problems: list[str] = []
    for name in sorted({model.name for model in models}):
        versions = read_versions(client, name)
        wanted = {model.sha256: model for model in models if model.name == name}
        for sha256 in sorted(set(versions) - set(wanted)):
            problems.append(f"{name}: version with sha256 {sha256!r} is not a recorded model")
        for sha256, model in wanted.items():
            version = versions.get(sha256)
            if version is None:
                problems.append(f"{name}: {model.path} ({sha256}) has no version in the registry")
                continue
            if version.get(PATH_TAG) != model.path:
                problems.append(f"{name}: version {version['version']} names another path")
            file = repo_root / model.path
            if not file.is_file():
                problems.append(f"{name}: {model.path} is not on disk")
            elif hash_file(file)[1] != sha256:
                problems.append(f"{name}: {model.path} no longer has the recorded sha256")
    for name, sha256 in sorted(current.items()):
        known = any(model.name == name and model.sha256 == sha256 for model in models)
        alias = read_alias(client, name)
        if known and alias != sha256:
            problems.append(
                f"{name}: alias {CURRENT_ALIAS!r} points to {alias!r}, expected {sha256}"
            )
    return problems
