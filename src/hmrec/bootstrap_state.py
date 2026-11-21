"""The bootstrap state: what the first tick of the replay starts from (docs/mlops.md §6.3).

The weekly pipeline keeps state from week to week: the R6 chain model, the
Ranker's training samples of the last weeks, and the history its monitoring
rules compare a week with. The first tick (V1) needs that state as it was at the
END of the week before (F4). Stage 1 produced all of it; this module COPIES it
into one folder, data/bootstrap/, which is tracked with DVC.

What the V1 tick reads, and why (traced through the stage-1 code):

    R6 predicts V1 with the chain model saved after the F4 fine-tune
    (hmrec.recall.chain_records.start_model_for). Loading any R6 model first
    reads the base checkpoint (twotower.infer.load_twotower_resources), and
    building the chain draws the alarm users from the base training array
    customer_idx.npy (hmrec.recall.locked.load_locked_alarm_users). The sha256
    of the three files are in the R6 lock and in results/m6_twotower_chain.csv.

    The Ranker that predicts V1 is trained on the window_weeks weeks before it
    (hmrec.ranker.window.training_folds: F1 F2 F3 F4). Each week is one sample
    file with its record (hmrec.ranker.sampling.verify_sample_file).

    Monitoring compares a week with the weeks before it
    (hmrec.eval.monitor.compare_with_previous): the rows of F1-F4 in
    results/m6_candidates.csv (recall of each channel, union recall, candidates
    per query) and in results/m6_features.csv (missing rate and mean of every
    feature). The chain table's rows of F1-F4 are the history of the R6 chain.

R1-R5, the features and the queries need no state: they are computed from the
processed tables, which DVC tracks separately.

Rules of this module:
    - a stage-1 file is only READ: it is copied, never moved or changed, and its
      sha256 is compared with its stage-1 record BEFORE the copy and with the
      copy AFTER it;
    - a table of results/ is not copied whole: only the header and the rows of
      the weeks before the first tick are kept, byte for byte, so the state holds
      nothing of a later week;
    - an existing data/bootstrap/ is never replaced.

This module reads lock files and result tables only. It imports neither torch
nor lightgbm, and no label is read.
"""

import csv
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hmrec.config import BootstrapConfig, load_versioning_config
from hmrec.data.folds import FoldManifest, load_fold_manifest
from hmrec.data_manifest import hash_file
from hmrec.locks import (
    BASE_CHECKPOINT,
    RankerLock,
    TwoTowerLock,
    load_ranker_lock,
    load_twotower_lock,
)
from hmrec.paths import ProjectPaths

SCHEMA_VERSION = 1
COPY = "copy"
ROWS = "rows"
KINDS = (COPY, ROWS)
# Folder of the state that holds the history rows of the result tables.
MONITORING_FOLDER = "monitoring"
# While the state is being built it has this suffix; the folder gets its real name last.
PARTIAL_SUFFIX = ".partial"
# Column of the result tables that names the week of a row (always the first column).
FOLD_COLUMN = "fold"
NO_RECORD = ""
# The array the alarm users are drawn from (hmrec.recall.locked.load_locked_alarm_users
# reads the same file; that module loads torch, so its constants are not imported here).
BASE_WINDOW_NAME = "base"
ALARM_SOURCE_FILE = "customer_idx.npy"
ENTRY_KEYS = (
    "path",
    "source",
    "kind",
    "folds",
    "size_bytes",
    "sha256",
    "source_sha256",
    "recorded_in",
    "purpose",
)
MANIFEST_KEYS = (
    "schema_version",
    "first_tick_fold",
    "closed_fold",
    "git_commit",
    "created_at_utc",
    "files",
)


@dataclass(frozen=True)
class BootstrapItem:
    """One file of the state, before it is built.

    Attributes:
        destination: Path inside data/bootstrap/, with "/".
        source: The stage-1 file, relative to the repository root, with "/".
        kind: "copy" (the whole file) or "rows" (the header and the rows of `folds`).
        folds: For "rows": the weeks whose rows are kept. Empty for "copy".
        expected_sha256: The sha256 stage 1 recorded for the source, or "" if
            stage 1 has no record of it.
        recorded_in: Where that record is, or "".
        purpose: Why the first tick needs the file.
    """

    destination: str
    source: str
    kind: str
    folds: tuple[str, ...]
    expected_sha256: str
    recorded_in: str
    purpose: str

    def __post_init__(self) -> None:
        """A "rows" item names its weeks; a "copy" item names none."""
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if (self.kind == ROWS) != bool(self.folds):
            raise ValueError(f"{self.destination}: only a {ROWS!r} item lists folds")


@dataclass(frozen=True)
class BootstrapEntry:
    """One file of the built state, as the manifest records it.

    Attributes:
        path: Path inside data/bootstrap/, with "/".
        source: The stage-1 file it came from, relative to the repository root.
        kind: "copy" or "rows".
        folds: For "rows": the weeks whose rows were kept.
        size_bytes: Size of the file in the state.
        sha256: sha256 of the file in the state.
        source_sha256: sha256 of the stage-1 file (equal to sha256 for a "copy").
        recorded_in: Where stage 1 recorded source_sha256, or "" if nowhere.
        purpose: Why the first tick needs the file.
    """

    path: str
    source: str
    kind: str
    folds: tuple[str, ...]
    size_bytes: int
    sha256: str
    source_sha256: str
    recorded_in: str
    purpose: str


@dataclass(frozen=True)
class BootstrapPlan:
    """Everything the state is built from.

    Attributes:
        first_tick_fold: The first tick of the replay ("V1").
        closed_fold: The week right before it, the last closed week ("F4").
        items: The files of the state, in the order they are written.
    """

    first_tick_fold: str
    closed_fold: str
    items: tuple[BootstrapItem, ...]


# ---------------------------------------------------------------- which weeks


def folds_before(manifest: FoldManifest, fold: str, n_weeks: int) -> tuple[str, ...]:
    """The `n_weeks` folds right before `fold`, in time order.

    Args:
        manifest: All folds in time order.
        fold: A fold name, e.g. "V1".
        n_weeks: How many folds to return.

    Returns:
        E.g. ("F1", "F2", "F3", "F4") for ("V1", 4).

    Raises:
        ValueError: If the fold is unknown or has fewer folds before it.
    """
    names = list(manifest.names)
    if fold not in names:
        raise ValueError(f"unknown fold {fold!r}; the manifest has {names}")
    position = names.index(fold)
    if n_weeks <= 0 or position < n_weeks:
        raise ValueError(f"fold {fold} has {position} folds before it, asked for {n_weeks}")
    return tuple(names[position - n_weeks : position])


# ---------------------------------------------------------------- the plan


def _relative(paths: ProjectPaths, path: Path) -> str:
    """A path relative to the repository root, with "/" (absolute if it lies outside)."""
    if path.is_relative_to(paths.project_root):
        return path.relative_to(paths.project_root).as_posix()
    return path.as_posix()


def _in_state(paths: ProjectPaths, artifact: Path) -> str:
    """Where an artifact goes inside the state: its path below artifacts/."""
    return artifact.relative_to(paths.artifacts_root).as_posix()


def read_table_value(table: Path, fold: str, column: str) -> str:
    """One value of an append-only result table: `column` of the row of `fold`.

    Raises:
        ValueError: If the table has not exactly one row of that fold.
    """
    with table.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get(FOLD_COLUMN) == fold]
    if len(rows) != 1:
        raise ValueError(f"{table}: expected one row of fold {fold}, found {len(rows)}")
    return rows[0][column]


def _chain_items(
    paths: ProjectPaths, lock: TwoTowerLock, closed_fold: str
) -> tuple[BootstrapItem, ...]:
    """The three files the R6 chain needs to predict the first tick."""
    base = paths.twotower_checkpoint_file(lock.channel.config["base_epoch"])
    chain = paths.twotower_locked_chain_checkpoint_file(closed_fold)
    alarm_source = paths.twotower_data_dir(BASE_WINDOW_NAME) / ALARM_SOURCE_FILE
    chain_table = paths.m6_twotower_chain_results_file
    lock_file = _relative(paths, paths.twotower_lock_file)
    return (
        BootstrapItem(
            destination=_in_state(paths, base),
            source=_relative(paths, base),
            kind=COPY,
            folds=(),
            expected_sha256=lock.checkpoints[BASE_CHECKPOINT].sha256,
            recorded_in=f"{lock_file} checkpoints.base",
            purpose="R6: the base checkpoint; read before any chain model is loaded",
        ),
        BootstrapItem(
            destination=_in_state(paths, chain),
            source=_relative(paths, chain),
            kind=COPY,
            folds=(),
            expected_sha256=read_table_value(chain_table, closed_fold, "sha256"),
            recorded_in=f"{_relative(paths, chain_table)} fold {closed_fold}",
            purpose=f"R6: the chain model after the {closed_fold} fine-tune; it predicts the "
            "first tick and is the start of the next fine-tune",
        ),
        BootstrapItem(
            destination=_in_state(paths, alarm_source),
            source=_relative(paths, alarm_source),
            kind=COPY,
            folds=(),
            expected_sha256=lock.alarm_users.source_sha256,
            recorded_in=f"{lock_file} alarm_users.source_sha256",
            purpose="R6: the array the alarm users of every fine-tune are drawn from",
        ),
    )


def _sample_items(
    paths: ProjectPaths, experiment: str, ratio: int, folds: Sequence[str]
) -> tuple[BootstrapItem, ...]:
    """The Ranker's sample file and its record, for every training week of the first tick."""
    items: list[BootstrapItem] = []
    for fold in folds:
        sample = paths.ranker_sample_file(experiment, fold, ratio)
        record = paths.ranker_sample_record_file(experiment, fold, ratio)
        if not record.is_file():
            raise FileNotFoundError(f"sample record not found: {record}")
        recorded: dict[str, Any] = json.loads(record.read_text(encoding="utf-8"))
        items.append(
            BootstrapItem(
                destination=_in_state(paths, sample),
                source=_relative(paths, sample),
                kind=COPY,
                folds=(),
                expected_sha256=str(recorded["sha256"]),
                recorded_in=_relative(paths, record),
                purpose=f"Ranker: the training sample of {fold} (1 : {ratio})",
            )
        )
        items.append(
            BootstrapItem(
                destination=_in_state(paths, record),
                source=_relative(paths, record),
                kind=COPY,
                folds=(),
                expected_sha256=NO_RECORD,
                recorded_in=NO_RECORD,
                purpose=f"Ranker: the record of the {fold} sample (its sha256, rows, seed)",
            )
        )
    return tuple(items)


def _history_items(paths: ProjectPaths, folds: Sequence[str]) -> tuple[BootstrapItem, ...]:
    """The history rows of the three result tables the pipeline continues."""
    tables = (
        (paths.m6_twotower_chain_results_file, "R6: the chain's history (one row per fine-tune)"),
        (
            paths.m6_candidates_results_file,
            "monitoring: recall of each channel, union recall and candidates per query",
        ),
        (
            paths.m6_features_results_file,
            "monitoring: missing rate, min, max and mean of every feature",
        ),
    )
    return tuple(
        BootstrapItem(
            destination=f"{MONITORING_FOLDER}/{table.name}",
            source=_relative(paths, table),
            kind=ROWS,
            folds=tuple(folds),
            expected_sha256=NO_RECORD,
            recorded_in=NO_RECORD,
            purpose=purpose,
        )
        for table, purpose in tables
    )


def bootstrap_plan(
    paths: ProjectPaths,
    config: BootstrapConfig,
    manifest: FoldManifest,
    ranker_lock: RankerLock,
    twotower_lock: TwoTowerLock,
) -> BootstrapPlan:
    """The files of the bootstrap state, derived from the lock files (module docstring).

    Nothing is copied here. The function reads the R6 chain table (the sha256 of
    the chain model) and the sample records (the sha256 of the sample files).

    Args:
        paths: Project paths (the stage-1 locations).
        config: The first tick, the monitoring history length, the Ranker experiment.
        manifest: All folds in time order.
        ranker_lock: Gives the training window and the sampling ratio.
        twotower_lock: Gives the base epoch, the alarm-user array and their sha256.

    Raises:
        ValueError: If the first tick has too few weeks before it, or the chain
            table has no row of the closed week.
        FileNotFoundError: If a sample record is missing.
    """
    ranker_folds = folds_before(manifest, config.first_tick_fold, ranker_lock.window_weeks)
    monitoring_folds = folds_before(manifest, config.first_tick_fold, config.monitoring_weeks)
    closed_fold = ranker_folds[-1]
    ratio = ranker_lock.sampling.negatives_per_positive
    items = (
        *_chain_items(paths, twotower_lock, closed_fold),
        *_sample_items(paths, config.ranker_experiment, ratio, ranker_folds),
        *_history_items(paths, monitoring_folds),
    )
    destinations = [item.destination for item in items]
    if len(set(destinations)) != len(destinations):
        raise ValueError(f"bootstrap plan: a destination repeats: {sorted(destinations)}")
    return BootstrapPlan(
        first_tick_fold=config.first_tick_fold, closed_fold=closed_fold, items=items
    )


# ---------------------------------------------------------------- building


def rows_of_folds(content: bytes, folds: Sequence[str], known_folds: Sequence[str]) -> bytes:
    """The header line and the lines of the given weeks of a result table, byte for byte.

    The tables this is used for have the week in their first column and no line
    break inside a field. Every line must therefore start with a known fold name
    and a comma; any other line stops the function.

    Args:
        content: The bytes of the table.
        folds: The weeks whose rows are kept.
        known_folds: Every fold name of the fold manifest.

    Returns:
        The header line followed by the kept lines, in their original order and
        with their original line ends.

    Raises:
        ValueError: If the first column is not `fold`, a line does not start
            with a known fold, or a wanted fold has no row.
    """
    lines = content.splitlines(keepends=True)
    if not lines or not lines[0].startswith(f"{FOLD_COLUMN},".encode()):
        raise ValueError(f"the table's first column must be {FOLD_COLUMN!r}")
    kept = [lines[0]]
    found: set[str] = set()
    for number, line in enumerate(lines[1:], start=2):
        fold = line.split(b",", 1)[0].decode("utf-8")
        if fold not in known_folds:
            raise ValueError(f"line {number} does not start with a fold name: {fold!r}")
        if fold in folds:
            kept.append(line)
            found.add(fold)
    missing = [fold for fold in folds if fold not in found]
    if missing:
        raise ValueError(f"the table has no row of fold(s) {missing}")
    return b"".join(kept)


def _check_source(item: BootstrapItem, source: Path) -> str:
    """sha256 of a source file, after comparing it with its stage-1 record (if it has one)."""
    _, sha256, _ = hash_file(source)
    if item.expected_sha256 != NO_RECORD and sha256 != item.expected_sha256:
        raise ValueError(
            f"{item.source}: sha256 is {sha256}, but {item.recorded_in} records "
            f"{item.expected_sha256}. This is not the stage-1 file; nothing was copied."
        )
    return sha256


def build_item(
    item: BootstrapItem, repo_root: Path, state_dir: Path, known_folds: Sequence[str]
) -> BootstrapEntry:
    """Write one file of the state and return its manifest entry.

    The source is read only. A "copy" must have the sha256 of its source.

    Args:
        item: The file to write.
        repo_root: The repository root (sources are relative to it).
        state_dir: The folder the state is written into.
        known_folds: Every fold name of the fold manifest (for "rows" items).

    Raises:
        FileNotFoundError: If the source does not exist.
        FileExistsError: If the destination already exists.
        ValueError: If the source is not the recorded file, or the copy differs.
    """
    source = repo_root / item.source
    destination = state_dir / item.destination
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; the state is never replaced")
    source_sha256 = _check_source(item, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if item.kind == COPY:
        shutil.copyfile(source, destination)
    else:
        destination.write_bytes(rows_of_folds(source.read_bytes(), item.folds, known_folds))
    size, sha256, _ = hash_file(destination)
    if item.kind == COPY and sha256 != source_sha256:
        raise ValueError(f"{destination}: the copy has sha256 {sha256}, the source {source_sha256}")
    return BootstrapEntry(
        path=item.destination,
        source=item.source,
        kind=item.kind,
        folds=item.folds,
        size_bytes=size,
        sha256=sha256,
        source_sha256=source_sha256,
        recorded_in=item.recorded_in,
        purpose=item.purpose,
    )


def manifest_text(plan: BootstrapPlan, entries: Sequence[BootstrapEntry], git_commit: str) -> str:
    """The content of data/bootstrap/manifest.json."""
    content = {
        "schema_version": SCHEMA_VERSION,
        "first_tick_fold": plan.first_tick_fold,
        "closed_fold": plan.closed_fold,
        "git_commit": git_commit,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": [{**asdict(entry), "folds": list(entry.folds)} for entry in entries],
    }
    return json.dumps(content, indent=2) + "\n"


def build_bootstrap_state(
    paths: ProjectPaths, plan: BootstrapPlan, known_folds: Sequence[str], git_commit: str
) -> tuple[BootstrapEntry, ...]:
    """Build data/bootstrap/ from the plan and write its manifest.

    The files are written into data/bootstrap.partial/ first; the folder gets
    its name only after every file and the manifest are written, so a folder
    named data/bootstrap/ is always a complete state.

    Args:
        paths: Project paths.
        plan: The files to write (bootstrap_plan).
        known_folds: Every fold name of the fold manifest.
        git_commit: HEAD of the code that builds the state (recorded in the manifest).

    Returns:
        The manifest entries, in the order of the plan.

    Raises:
        FileExistsError: If data/bootstrap/ or a left-over partial folder exists.
            Nothing is deleted automatically.
        ValueError / FileNotFoundError: As build_item.
    """
    state_dir = paths.bootstrap_dir
    partial_dir = state_dir.with_name(state_dir.name + PARTIAL_SUFFIX)
    for folder in (state_dir, partial_dir):
        if folder.exists():
            raise FileExistsError(f"{folder} already exists; the bootstrap state is never replaced")
    partial_dir.mkdir(parents=True)
    entries = tuple(
        build_item(item, paths.project_root, partial_dir, known_folds) for item in plan.items
    )
    manifest_file = partial_dir / paths.bootstrap_manifest_file.name
    manifest_file.write_text(manifest_text(plan, entries, git_commit), encoding="utf-8")
    partial_dir.rename(state_dir)
    return entries


# ---------------------------------------------------------------- reading and verifying


def _parse_entry(raw: Any, position: int) -> BootstrapEntry:
    """One entry of a manifest file, validated."""
    if not isinstance(raw, Mapping) or tuple(raw) != ENTRY_KEYS:
        raise ValueError(f"bootstrap manifest: files[{position}] must have the keys {ENTRY_KEYS}")
    return BootstrapEntry(**{**raw, "folds": tuple(raw["folds"])})


def read_bootstrap_manifest(path: Path) -> tuple[BootstrapEntry, ...]:
    """Read the entries of data/bootstrap/manifest.json.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the schema version is another one or a key is missing or unknown.
    """
    if not path.is_file():
        raise FileNotFoundError(f"bootstrap manifest not found: {path}")
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or tuple(content) != MANIFEST_KEYS:
        raise ValueError(f"{path}: expected the keys {MANIFEST_KEYS}")
    if content["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {content['schema_version']!r}, expected {SCHEMA_VERSION}"
        )
    return tuple(_parse_entry(raw, position) for position, raw in enumerate(content["files"]))


def verify_bootstrap_state(state_dir: Path, entries: Sequence[BootstrapEntry]) -> list[str]:
    """Compare the files of a state with its manifest entries.

    Args:
        state_dir: The folder of the state (data/bootstrap/, or a copy elsewhere).
        entries: The manifest entries (read_bootstrap_manifest).

    Returns:
        One line per file that is missing or differs; empty if all are as recorded.
    """
    problems: list[str] = []
    for entry in entries:
        file = state_dir / entry.path
        if not file.is_file():
            problems.append(f"{entry.path}: missing")
            continue
        size, sha256, _ = hash_file(file)
        if (size, sha256) != (entry.size_bytes, entry.sha256):
            problems.append(
                f"{entry.path}: size {size} sha256 {sha256}, recorded "
                f"{entry.size_bytes} {entry.sha256}"
            )
    return problems


# ---------------------------------------------------------------- from the repository


def plan_from_repository(paths: ProjectPaths) -> tuple[BootstrapPlan, tuple[str, ...]]:
    """The plan of this repository's bootstrap state, and every fold name.

    Reads configs/m11_versioning.yaml, the fold manifest and the two lock files
    (strict parsers of hmrec.locks); nothing is copied.

    Returns:
        (the plan, the fold names of the manifest in time order).
    """
    config = load_versioning_config(paths.versioning_config_file).bootstrap
    manifest = load_fold_manifest(paths.fold_manifest_file)
    plan = bootstrap_plan(
        paths,
        config,
        manifest,
        load_ranker_lock(paths.ranker_lock_file),
        load_twotower_lock(paths.twotower_lock_file),
    )
    return plan, manifest.names
