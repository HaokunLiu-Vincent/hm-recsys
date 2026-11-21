"""The run manifest: the record one tick of the weekly pipeline writes (docs/mlops.md §6.1 O6).

A tick is one call of the weekly pipeline for an anchor date. Its last step
writes ONE JSON file, runs/{run_id}/manifest.json (docs/mlops.md §7.2), that says
what ran and what it produced:

    run_id, anchor, mode      which tick: "eval" or "serve" (docs/mlops.md §6.2)
    status, failure           "succeeded", or "failed" with the reason
    git_commit, git_dirty     the code (inside an image: the image's commit, False)
    code_tree_sha256          the verified code-tree hash of the image
                              ("not an image" on the laptop)
    image_digest              the registry digest of the image, when the runner
                              was told it (null otherwise)
    platform                  what the driver process ran on (hmrec.platform_fields)
    inputs, outputs           files with role, path, size and sha256
    metrics                   name, week and value; a metric names the week it
                              DESCRIBES, because a week's predictions are scored
                              by the next tick
    steps                     one entry per step: status, seconds, the TRUE peak
                              memory of the step's process and its platform fields
    started_at_utc, finished_at_utc, total_seconds
    driver_max_rss_gb         true peak of the driver process
    max_process_rss_gb        the largest true peak of any process of the tick

The manifest files are the source of truth for runs; MLflow is a view built from
them (docs/mlops.md §7.3, hmrec.tracking).

Why platform fields per step: every step runs in its own process, and the
OpenMP runtime differs between them (torch's copy for recall, the system libgomp
for LightGBM, none for the feature build; docs/mlops.md §8.1).

A manifest is written once and never replaced. It is written under a temporary
name and renamed, so a file named manifest.json is always complete.

This module is at the bottom layer. It reuses hmrec.memory (true peak),
hmrec.platform_fields and hmrec.git_state, and must stay free of torch and
lightgbm: both sides write step records.
"""

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from hmrec.data_manifest import hash_file
from hmrec.git_state import NOT_AN_IMAGE, code_tree_of_run, git_state
from hmrec.memory import read_process_peak_memory
from hmrec.platform_fields import PLATFORM_COLUMNS, PlatformFields, read_platform_fields

SCHEMA_VERSION = 1

EVAL_MODE = "eval"
SERVE_MODE = "serve"
MODES = (EVAL_MODE, SERVE_MODE)

SUCCEEDED = "succeeded"
FAILED = "failed"
RUN_STATUSES = (SUCCEEDED, FAILED)

STEP_DONE = "done"
STEP_SKIPPED = "skipped"
STEP_FAILED = "failed"
STEP_STATUSES = (STEP_DONE, STEP_SKIPPED, STEP_FAILED)

# The runner of a container may pass the registry digest of the image in this variable
# (an image cannot know its own digest when it is built). Not set: the digest is unknown.
IMAGE_DIGEST_VARIABLE = "HMREC_IMAGE_DIGEST"

RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
COMMIT = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

FILE_KEYS = ("role", "path", "size_bytes", "sha256")
METRIC_KEYS = ("name", "week", "value")
STEP_KEYS = ("name", "status", "seconds", "process_max_rss_gb", "platform", "detail")
MANIFEST_KEYS = (
    "schema_version",
    "run_id",
    "anchor",
    "mode",
    "status",
    "failure",
    "git_commit",
    "git_dirty",
    "code_tree_sha256",
    "image_digest",
    "platform",
    "inputs",
    "outputs",
    "metrics",
    "steps",
    "started_at_utc",
    "finished_at_utc",
    "total_seconds",
    "driver_max_rss_gb",
    "max_process_rss_gb",
)


@dataclass(frozen=True)
class FileRecord:
    """One input or output file of a tick.

    Attributes:
        role: What the file is for the tick, e.g. "transactions" or
            "candidates/r4_itemcf". Unique among the inputs, and among the outputs.
        path: Where the file is, relative to the root the pipeline reads and
            writes under (the repository on the laptop, the bucket in the cloud).
        size_bytes: Size of the file.
        sha256: sha256 of the file's bytes.
    """

    role: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class MetricRecord:
    """One metric value of a tick.

    Attributes:
        name: E.g. "recall/r1_popularity", "union_recall", "map@12".
        week: The anchor of the week the value DESCRIBES. For a labelled metric
            this is the week before the tick's anchor (docs/mlops.md §6.1).
        value: The value as float64 (metric values are float64, AGENTS.md §4).
    """

    name: str
    week: date
    value: float


@dataclass(frozen=True)
class StepRecord:
    """What one step of a tick did.

    Attributes:
        name: Step name, e.g. "O1_recall_r4_itemcf".
        status: "done", "skipped" (the output existed: a resumed tick) or "failed".
        seconds: Wall time of the step's process (0 when skipped).
        process_max_rss_gb: TRUE peak RSS of the step's process (ru_maxrss; 0
            when skipped). Each step is one process, so this is the step's peak.
        platform: What the step's process ran on; None when skipped.
        detail: Short free text.
    """

    name: str
    status: str
    seconds: float
    process_max_rss_gb: float
    platform: PlatformFields | None
    detail: str


@dataclass(frozen=True)
class RunCode:
    """The code a tick ran.

    Attributes:
        git_commit: HEAD (inside an image: the commit the image was built from).
        git_dirty: Uncommitted changes outside results/ (inside an image: False).
        code_tree_sha256: The verified code-tree hash of the image, or "not an image".
        image_digest: The registry digest of the image ("sha256:..."), or None
            when the runner did not pass it.
    """

    git_commit: str
    git_dirty: bool
    code_tree_sha256: str
    image_digest: str | None


@dataclass(frozen=True)
class RunManifest:
    """The record of one tick (see the module docstring).

    Attributes:
        run_id: Unique name of the tick's run; also its folder name.
        anchor: First day of the week the tick opens.
        mode: "eval" or "serve".
        status: "succeeded" or "failed".
        failure: The reason of a failed tick; "" when it succeeded.
        code: Commit, dirty flag, code-tree hash and image digest.
        platform: Platform fields of the driver process.
        inputs: Files the tick read.
        outputs: Files the tick wrote.
        metrics: Metric values, each with the week it describes.
        steps: The steps in the order they ran.
        started_at_utc: Start of the tick (UTC, to the second).
        finished_at_utc: End of the tick (UTC, to the second).
        total_seconds: Wall time of the whole tick.
        driver_max_rss_gb: True peak RSS of the driver process.
    """

    run_id: str
    anchor: date
    mode: str
    status: str
    failure: str
    code: RunCode
    platform: PlatformFields
    inputs: tuple[FileRecord, ...]
    outputs: tuple[FileRecord, ...]
    metrics: tuple[MetricRecord, ...]
    steps: tuple[StepRecord, ...]
    started_at_utc: datetime
    finished_at_utc: datetime
    total_seconds: float
    driver_max_rss_gb: float

    @property
    def max_process_rss_gb(self) -> float:
        """The largest true peak RSS of any process of the tick (driver and steps)."""
        return max([self.driver_max_rss_gb, *(step.process_max_rss_gb for step in self.steps)])


# ---------------------------------------------------------------- reading the process


def image_digest(environment: Mapping[str, str]) -> str | None:
    """The image digest the runner passed, or None.

    Args:
        environment: The process environment (os.environ).

    Raises:
        ValueError: If the variable is set to something that is not "sha256:"
            followed by 64 hexadecimal characters. A wrong digest in a record is
            worse than none.
    """
    value = environment.get(IMAGE_DIGEST_VARIABLE)
    if value is None or value == "":
        return None
    if IMAGE_DIGEST.fullmatch(value) is None:
        raise ValueError(
            f"{IMAGE_DIGEST_VARIABLE} must be sha256:<64 hex characters>, got {value!r}"
        )
    return value


def read_run_code(repo_root: Path, results_root: Path) -> RunCode:
    """The code state of THIS process (hmrec.git_state), with the image digest if known.

    Args:
        repo_root: Root of the code.
        results_root: Folder of committed result tables (left out of "dirty").
    """
    commit, dirty = git_state(repo_root, results_root)
    return RunCode(
        git_commit=commit,
        git_dirty=dirty,
        code_tree_sha256=code_tree_of_run(repo_root),
        image_digest=image_digest(os.environ),
    )


def step_record_of_this_process(
    name: str, seconds: float, detail: str, repo_root: Path
) -> StepRecord:
    """The record of a step that ran in THIS process; call it as the step's last action.

    The true peak and the loaded libraries are read from the process itself
    (hmrec.memory.read_process_peak_memory, hmrec.platform_fields), so they
    describe what the step really used.

    Args:
        name: Step name.
        seconds: Wall time of the step.
        detail: Short free text.
        repo_root: Root of the code.
    """
    return StepRecord(
        name=name,
        status=STEP_DONE,
        seconds=seconds,
        process_max_rss_gb=read_process_peak_memory().process_max_rss_gb,
        platform=read_platform_fields(repo_root),
        detail=detail,
    )


def file_record(role: str, root: Path, relative_path: str) -> FileRecord:
    """The record of a file under `root` (size and sha256 are read from the file).

    Args:
        role: What the file is for the tick.
        root: The root the pipeline reads and writes under.
        relative_path: Path of the file relative to `root`, with "/".
    """
    size, sha256, _ = hash_file(root / relative_path)
    return FileRecord(role=role, path=relative_path, size_bytes=size, sha256=sha256)


# ---------------------------------------------------------------- validation


def _require(condition: bool, message: str) -> None:
    """Raise ValueError(message) unless the condition holds."""
    if not condition:
        raise ValueError(f"run manifest: {message}")


def _is_number(value: Any) -> bool:
    """True for a finite int or float (not bool)."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _check_files(files: Sequence[FileRecord], what: str) -> None:
    """Roles are unique; every size and sha256 is well-formed."""
    roles = [file.role for file in files]
    _require(len(set(roles)) == len(roles), f"a role repeats among the {what}: {sorted(roles)}")
    for file in files:
        _require(file.role != "" and file.path != "", f"{what}: empty role or path")
        _require(
            isinstance(file.size_bytes, int) and file.size_bytes >= 0,
            f"{what} {file.role}: size_bytes must be an integer >= 0",
        )
        _require(SHA256.fullmatch(file.sha256) is not None, f"{what} {file.role}: bad sha256")


def _check_steps(steps: Sequence[StepRecord]) -> None:
    """Step names are unique; a skipped step has no platform, any other step has one."""
    names = [step.name for step in steps]
    _require(len(set(names)) == len(names), f"a step name repeats: {names}")
    for step in steps:
        _require(step.status in STEP_STATUSES, f"step {step.name}: status {step.status!r}")
        _require(_is_number(step.seconds) and step.seconds >= 0, f"step {step.name}: bad seconds")
        _require(
            _is_number(step.process_max_rss_gb) and step.process_max_rss_gb >= 0,
            f"step {step.name}: bad process_max_rss_gb",
        )
        _require(
            (step.platform is None) == (step.status == STEP_SKIPPED),
            f"step {step.name}: a skipped step has no platform fields, every other step has them",
        )


def _check_code(code: RunCode) -> None:
    """Commit, code-tree hash and image digest are well-formed."""
    _require(COMMIT.fullmatch(code.git_commit) is not None, "git_commit is not a full commit hash")
    _require(isinstance(code.git_dirty, bool), "git_dirty must be true or false")
    tree = code.code_tree_sha256
    _require(
        tree == NOT_AN_IMAGE or SHA256.fullmatch(tree) is not None,
        f"code_tree_sha256 must be a sha256 or {NOT_AN_IMAGE!r}",
    )
    digest = code.image_digest
    _require(
        digest is None or IMAGE_DIGEST.fullmatch(digest) is not None,
        "image_digest must be null or sha256:<64 hex characters>",
    )


def validate_run_manifest(manifest: RunManifest) -> None:
    """Fail unless the manifest is well-formed (called by the writer and the reader).

    Raises:
        ValueError: With the first rule that is broken.
    """
    _require(RUN_ID.fullmatch(manifest.run_id) is not None, f"bad run_id {manifest.run_id!r}")
    _require(manifest.mode in MODES, f"mode must be one of {MODES}, got {manifest.mode!r}")
    _require(manifest.status in RUN_STATUSES, f"status must be one of {RUN_STATUSES}")
    _require(
        (manifest.failure == "") == (manifest.status == SUCCEEDED),
        "failure is empty for a succeeded tick and says the reason of a failed one",
    )
    _check_code(manifest.code)
    _check_files(manifest.inputs, "inputs")
    _check_files(manifest.outputs, "outputs")
    keys = [(metric.name, metric.week) for metric in manifest.metrics]
    _require(len(set(keys)) == len(keys), "a (metric name, week) pair repeats")
    for metric in manifest.metrics:
        _require(metric.name != "", "a metric has no name")
        _require(
            isinstance(metric.value, float) and math.isfinite(metric.value),
            f"metric {metric.name}: the value must be a finite float",
        )
    _check_steps(manifest.steps)
    _require(
        manifest.started_at_utc.tzinfo is not None and manifest.finished_at_utc.tzinfo is not None,
        "started_at_utc and finished_at_utc must carry a time zone",
    )
    _require(manifest.finished_at_utc >= manifest.started_at_utc, "the tick ends before it starts")
    _require(
        _is_number(manifest.total_seconds) and manifest.total_seconds >= 0, "bad total_seconds"
    )
    _require(
        _is_number(manifest.driver_max_rss_gb) and manifest.driver_max_rss_gb >= 0,
        "bad driver_max_rss_gb",
    )


# ---------------------------------------------------------------- JSON


def _platform_to_json(platform: PlatformFields | None) -> dict[str, str] | None:
    """Platform fields as a JSON object (None stays null)."""
    return None if platform is None else platform.columns()


def manifest_to_json(manifest: RunManifest) -> dict[str, Any]:
    """The manifest as the JSON object that is written (keys in MANIFEST_KEYS order)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest.run_id,
        "anchor": manifest.anchor.isoformat(),
        "mode": manifest.mode,
        "status": manifest.status,
        "failure": manifest.failure,
        "git_commit": manifest.code.git_commit,
        "git_dirty": manifest.code.git_dirty,
        "code_tree_sha256": manifest.code.code_tree_sha256,
        "image_digest": manifest.code.image_digest,
        "platform": _platform_to_json(manifest.platform),
        "inputs": [_file_to_json(file) for file in manifest.inputs],
        "outputs": [_file_to_json(file) for file in manifest.outputs],
        "metrics": [
            {"name": metric.name, "week": metric.week.isoformat(), "value": metric.value}
            for metric in manifest.metrics
        ],
        "steps": [
            {
                "name": step.name,
                "status": step.status,
                "seconds": step.seconds,
                "process_max_rss_gb": step.process_max_rss_gb,
                "platform": _platform_to_json(step.platform),
                "detail": step.detail,
            }
            for step in manifest.steps
        ],
        "started_at_utc": manifest.started_at_utc.isoformat(timespec="seconds"),
        "finished_at_utc": manifest.finished_at_utc.isoformat(timespec="seconds"),
        "total_seconds": manifest.total_seconds,
        "driver_max_rss_gb": manifest.driver_max_rss_gb,
        "max_process_rss_gb": manifest.max_process_rss_gb,
    }


def _file_to_json(file: FileRecord) -> dict[str, Any]:
    """A file record as a JSON object (keys in FILE_KEYS order)."""
    return {
        "role": file.role,
        "path": file.path,
        "size_bytes": file.size_bytes,
        "sha256": file.sha256,
    }


def _keys(raw: Any, expected: Sequence[str], where: str) -> dict[str, Any]:
    """`raw` as a dict, after checking that it has exactly the expected keys."""
    _require(isinstance(raw, dict), f"{where}: expected an object")
    missing = [key for key in expected if key not in raw]
    unknown = [key for key in raw if key not in expected]
    _require(not missing and not unknown, f"{where}: missing keys {missing}, unknown {unknown}")
    return raw


def _platform_from_json(raw: Any, where: str) -> PlatformFields | None:
    """Platform fields from a JSON object (null stays None)."""
    if raw is None:
        return None
    values = _keys(raw, PLATFORM_COLUMNS, where)
    _require(all(isinstance(value, str) for value in values.values()), f"{where}: text only")
    return PlatformFields(**values)


def _file_from_json(raw: Any, where: str) -> FileRecord:
    """A file record from a JSON object."""
    values = _keys(raw, FILE_KEYS, where)
    return FileRecord(**values)


def _step_from_json(raw: Any, where: str) -> StepRecord:
    """A step record from a JSON object."""
    values = _keys(raw, STEP_KEYS, where)
    platform = _platform_from_json(values["platform"], f"{where}.platform")
    _require(_is_number(values["seconds"]), f"{where}: seconds must be a number")
    _require(_is_number(values["process_max_rss_gb"]), f"{where}: bad process_max_rss_gb")
    return StepRecord(
        name=values["name"],
        status=values["status"],
        seconds=float(values["seconds"]),
        process_max_rss_gb=float(values["process_max_rss_gb"]),
        platform=platform,
        detail=values["detail"],
    )


def _metric_from_json(raw: Any, where: str) -> MetricRecord:
    """A metric record from a JSON object."""
    values = _keys(raw, METRIC_KEYS, where)
    _require(_is_number(values["value"]), f"{where}: the value must be a finite number")
    return MetricRecord(
        name=values["name"], week=date.fromisoformat(values["week"]), value=float(values["value"])
    )


def manifest_from_json(raw: Any) -> RunManifest:
    """Build and validate a RunManifest from a parsed manifest file.

    Raises:
        ValueError: If the schema version is another one, a key is missing or
            unknown, a value is malformed, or max_process_rss_gb is not the
            largest peak of the recorded processes.
    """
    values = _keys(raw, MANIFEST_KEYS, "top level")
    _require(
        values["schema_version"] == SCHEMA_VERSION,
        f"schema_version {values['schema_version']!r}, expected {SCHEMA_VERSION}",
    )
    platform = _platform_from_json(values["platform"], "platform")
    if platform is None:
        raise ValueError("run manifest: platform must not be null")
    for key in ("inputs", "outputs", "metrics", "steps"):
        _require(isinstance(values[key], list), f"{key} must be a list")
    _require(_is_number(values["total_seconds"]), "total_seconds must be a number")
    _require(_is_number(values["driver_max_rss_gb"]), "driver_max_rss_gb must be a number")
    manifest = RunManifest(
        run_id=values["run_id"],
        anchor=date.fromisoformat(values["anchor"]),
        mode=values["mode"],
        status=values["status"],
        failure=values["failure"],
        code=RunCode(
            git_commit=values["git_commit"],
            git_dirty=values["git_dirty"],
            code_tree_sha256=values["code_tree_sha256"],
            image_digest=values["image_digest"],
        ),
        platform=platform,
        inputs=tuple(_file_from_json(f, f"inputs[{i}]") for i, f in enumerate(values["inputs"])),
        outputs=tuple(_file_from_json(f, f"outputs[{i}]") for i, f in enumerate(values["outputs"])),
        metrics=tuple(
            _metric_from_json(m, f"metrics[{i}]") for i, m in enumerate(values["metrics"])
        ),
        steps=tuple(_step_from_json(s, f"steps[{i}]") for i, s in enumerate(values["steps"])),
        started_at_utc=datetime.fromisoformat(values["started_at_utc"]),
        finished_at_utc=datetime.fromisoformat(values["finished_at_utc"]),
        total_seconds=float(values["total_seconds"]),
        driver_max_rss_gb=float(values["driver_max_rss_gb"]),
    )
    validate_run_manifest(manifest)
    _require(
        values["max_process_rss_gb"] == manifest.max_process_rss_gb,
        "max_process_rss_gb is not the largest peak of the driver and the steps",
    )
    return manifest


# ---------------------------------------------------------------- writer and reader


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    """Write a manifest file. It is written once and never replaced.

    The file is written under a temporary name and renamed, so a reader never
    sees a half-written manifest.

    Args:
        path: The manifest file (hmrec.paths.ProjectPaths.run_manifest_file).
        manifest: The record of the tick.

    Raises:
        ValueError: If the manifest is not well-formed (nothing is written).
        FileExistsError: If the file already exists.
    """
    validate_run_manifest(manifest)
    if path.exists():
        raise FileExistsError(f"{path} already exists; a run manifest is never replaced")
    # allow_nan=False: NaN and Infinity are not JSON, and validate_run_manifest refuses them.
    text = json.dumps(manifest_to_json(manifest), indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.rename(path)


def read_run_manifest(path: Path) -> RunManifest:
    """Read and validate a manifest file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: As manifest_from_json.
    """
    if not path.is_file():
        raise FileNotFoundError(f"run manifest not found: {path}")
    return manifest_from_json(json.loads(path.read_text(encoding="utf-8")))


def find_run_manifests(runs_dir: Path) -> list[Path]:
    """Every runs/{run_id}/manifest.json under a folder, sorted by run id.

    Args:
        runs_dir: The folder that holds one folder per run. It may not exist yet
            (no tick has run): the result is then empty.
    """
    if not runs_dir.is_dir():
        return []
    return sorted(runs_dir.glob("*/manifest.json"))
