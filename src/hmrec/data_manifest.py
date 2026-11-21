"""The data manifest: path, size and sha256 of every file DVC tracks (docs/mlops.md §7.1).

DVC names a file by its md5 and keeps that in the pointer files (`*.dvc`). The
project records files by sha256 everywhere else (lock files, sample records, the
calibration config), and several inputs had no sha256 record at all in stage 1
(the raw CSV files, the id maps, the embedding files). The manifest closes that
gap: one entry per DVC-tracked file with

    path        relative to the repository root, with "/"
    size_bytes  size of the file
    sha256      of the file's bytes
    md5         of the file's bytes: the name DVC stores the file under

It is written once per change of the tracked data, from the files on disk, and is
committed to git (configs/data_manifest.json). After a `dvc pull` on another
machine, verify_entries proves that the files there are the recorded ones.

Which files are tracked is asked from DVC (`dvc ls --recursive --dvc-only
--json`), not guessed from the folders: a file DVC leaves out (.dvcignore) is
not in the manifest. The md5 DVC reports must equal the md5 computed here, so
an entry is written only for a file that IS what DVC tracks.

This module is at the bottom layer and imports only the standard library.
"""

import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
READ_BLOCK_BYTES = 1024 * 1024  # files are hashed 1 MB at a time, never read whole
SHA256_LENGTH = 64
MD5_LENGTH = 32
HEX_DIGITS = frozenset("0123456789abcdef")
ENTRY_KEYS = ("path", "size_bytes", "sha256", "md5")
# The command whose output parse_dvc_listing reads (run in the repository root).
DVC_LISTING_COMMAND: tuple[str, ...] = ("dvc", "ls", ".", "--recursive", "--dvc-only", "--json")


@dataclass(frozen=True)
class FileEntry:
    """One file of the manifest.

    Attributes:
        path: Path relative to the repository root, with "/" as separator.
        size_bytes: Size of the file in bytes.
        sha256: sha256 of the file's bytes (64 lower-case hexadecimal characters).
        md5: md5 of the file's bytes (32 characters): the name of the file in
            the DVC cache and in the DVC remote.
    """

    path: str
    size_bytes: int
    sha256: str
    md5: str


@dataclass(frozen=True)
class TrackedFile:
    """One file DVC tracks, as DVC lists it.

    Attributes:
        path: Path relative to the repository root, with "/".
        size_bytes: The size DVC recorded.
        md5: The md5 DVC recorded.
    """

    path: str
    size_bytes: int
    md5: str


def _is_hex(value: Any, length: int) -> bool:
    """True if `value` is a string of exactly `length` lower-case hexadecimal characters."""
    return isinstance(value, str) and len(value) == length and set(value) <= HEX_DIGITS


def hash_file(path: Path) -> tuple[int, str, str]:
    """Size, sha256 and md5 of a file, read once in blocks.

    Args:
        path: The file.

    Returns:
        (size in bytes, sha256 in hexadecimal, md5 in hexadecimal).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    sha256 = hashlib.sha256()
    # md5 is not used for security here: it is only the name DVC gives the file.
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK_BYTES), b""):
            sha256.update(block)
            md5.update(block)
            size += len(block)
    return size, sha256.hexdigest(), md5.hexdigest()


def file_entry(root: Path, relative_path: str) -> FileEntry:
    """The manifest entry of one file under `root`.

    Args:
        root: The repository root (or the root of a clone).
        relative_path: Path of the file relative to `root`, with "/".
    """
    size, sha256, md5 = hash_file(root / relative_path)
    return FileEntry(path=relative_path, size_bytes=size, sha256=sha256, md5=md5)


# ---------------------------------------------------------------- what DVC tracks


def parse_dvc_listing(text: str) -> tuple[TrackedFile, ...]:
    """The tracked files in the output of DVC_LISTING_COMMAND, sorted by path.

    Args:
        text: The JSON the command printed: a list with one object per file,
            holding at least "path", "size", "md5" and "isdir".

    Raises:
        ValueError: If the text is not such a list, an entry is a folder or has
            no md5 or size, or a path appears twice.
    """
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError(f"DVC listing: expected a list, got {type(parsed).__name__}")
    files: dict[str, TrackedFile] = {}
    for item in parsed:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError(f"DVC listing: not a file entry: {item!r}")
        path = item["path"]
        size = item.get("size")
        if item.get("isdir") or not _is_hex(item.get("md5"), MD5_LENGTH):
            raise ValueError(f"DVC listing: {path!r} is a folder or has no md5: {item!r}")
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError(f"DVC listing: {path!r} has no size: {item!r}")
        if path in files:
            raise ValueError(f"DVC listing: {path!r} is listed twice")
        files[path] = TrackedFile(path=path, size_bytes=size, md5=item["md5"])
    return tuple(files[path] for path in sorted(files))


def run_dvc_listing(repo_root: Path) -> str:
    """Ask DVC which files it tracks (DVC_LISTING_COMMAND, run in the repository root).

    DVC is a command of the dependency group "versioning"; it is found on PATH
    when the caller runs under `uv run --group versioning`.

    Returns:
        The JSON text the command printed (parse_dvc_listing reads it).

    Raises:
        RuntimeError: If the command fails; the message holds what DVC printed.
        FileNotFoundError: If `dvc` is not installed in this environment.
    """
    # DVC sends anonymous usage data unless told not to; this project sends none.
    environment = {**os.environ, "DVC_NO_ANALYTICS": "1"}
    completed = subprocess.run(
        list(DVC_LISTING_COMMAND),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{' '.join(DVC_LISTING_COMMAND)} failed: {completed.stderr.strip()}")
    return completed.stdout


def build_entries(root: Path, tracked: Sequence[TrackedFile]) -> tuple[FileEntry, ...]:
    """Hash every tracked file on disk and check it against what DVC recorded.

    Args:
        root: The repository root.
        tracked: The files DVC tracks (parse_dvc_listing).

    Returns:
        One entry per file, in the order of `tracked`.

    Raises:
        FileNotFoundError: If a tracked file is not on disk.
        ValueError: If a file's size or md5 is not the one DVC recorded: the file
            on disk is then not the file DVC would upload, and a sha256 of it
            would describe something else.
    """
    entries: list[FileEntry] = []
    for file in tracked:
        entry = file_entry(root, file.path)
        if (entry.size_bytes, entry.md5) != (file.size_bytes, file.md5):
            raise ValueError(
                f"{file.path}: on disk size {entry.size_bytes} md5 {entry.md5}, but DVC "
                f"tracks size {file.size_bytes} md5 {file.md5}; run `dvc status` first"
            )
        entries.append(entry)
    return tuple(entries)


# ---------------------------------------------------------------- the manifest file


def _check_entry(raw: Any, position: int) -> FileEntry:
    """One parsed entry of the manifest file, validated."""
    where = f"data manifest: files[{position}]"
    if not isinstance(raw, dict) or tuple(raw) != ENTRY_KEYS:
        raise ValueError(f"{where}: expected the keys {list(ENTRY_KEYS)}, got {raw!r}")
    size = raw["size_bytes"]
    if not isinstance(raw["path"], str) or raw["path"] == "":
        raise ValueError(f"{where}: path must be a non-empty string")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{where}: size_bytes must be an integer >= 0, got {size!r}")
    if not _is_hex(raw["sha256"], SHA256_LENGTH) or not _is_hex(raw["md5"], MD5_LENGTH):
        raise ValueError(f"{where}: sha256 or md5 is not a lower-case hexadecimal hash")
    return FileEntry(path=raw["path"], size_bytes=size, sha256=raw["sha256"], md5=raw["md5"])


def check_entries(entries: Sequence[FileEntry]) -> None:
    """Fail unless the entries are sorted by path and no path repeats.

    Raises:
        ValueError: Otherwise. A fixed order makes the file's diff in git readable.
    """
    paths = [entry.path for entry in entries]
    if paths != sorted(set(paths)):
        raise ValueError("data manifest: entries must be sorted by path, without duplicates")


def manifest_text(entries: Sequence[FileEntry]) -> str:
    """The content of the manifest file for these entries (JSON, one key per line)."""
    check_entries(entries)
    content = {
        "schema_version": SCHEMA_VERSION,
        "n_files": len(entries),
        "total_size_bytes": sum(entry.size_bytes for entry in entries),
        "files": [asdict(entry) for entry in entries],
    }
    return json.dumps(content, indent=2) + "\n"


def write_data_manifest(path: Path, entries: Sequence[FileEntry]) -> None:
    """Write the manifest file (it is replaced: git keeps its history)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest_text(entries), encoding="utf-8")


def read_data_manifest(path: Path) -> tuple[FileEntry, ...]:
    """Read and validate the manifest file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the schema version is another one, a key is missing or
            unknown, an entry is malformed, or the totals do not fit the entries.
    """
    if not path.is_file():
        raise FileNotFoundError(f"data manifest not found: {path}")
    content = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = ["schema_version", "n_files", "total_size_bytes", "files"]
    if not isinstance(content, dict) or list(content) != expected_keys:
        raise ValueError(f"{path}: expected the keys {expected_keys}")
    if content["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {content['schema_version']!r}, expected {SCHEMA_VERSION}"
        )
    if not isinstance(content["files"], list):
        raise ValueError(f"{path}: files must be a list")
    entries = tuple(_check_entry(raw, position) for position, raw in enumerate(content["files"]))
    check_entries(entries)
    total = sum(entry.size_bytes for entry in entries)
    if content["n_files"] != len(entries) or content["total_size_bytes"] != total:
        raise ValueError(f"{path}: n_files or total_size_bytes does not fit the entries")
    return entries


# ---------------------------------------------------------------- selecting and verifying


def select_entries(entries: Sequence[FileEntry], prefixes: Sequence[str]) -> tuple[FileEntry, ...]:
    """The entries at or under one of the given paths.

    Args:
        entries: All entries of the manifest.
        prefixes: Files or folders relative to the repository root, e.g.
            ("data/processed", "artifacts/features/category_codes.json").

    Raises:
        ValueError: If a prefix selects no entry (most likely a typo).
    """
    selected: list[FileEntry] = []
    for prefix in prefixes:
        clean = prefix.rstrip("/")
        matching = [e for e in entries if e.path == clean or e.path.startswith(f"{clean}/")]
        if not matching:
            raise ValueError(f"data manifest: no entry at or under {prefix!r}")
        selected.extend(matching)
    unique = {entry.path: entry for entry in selected}
    return tuple(unique[path] for path in sorted(unique))


def entry_problem(root: Path, entry: FileEntry) -> str | None:
    """What is wrong with the file of one entry under `root`, or None if it is the recorded file."""
    path = root / entry.path
    if not path.is_file():
        return f"{entry.path}: missing"
    size, sha256, _ = hash_file(path)
    if size != entry.size_bytes:
        return f"{entry.path}: size {size}, recorded {entry.size_bytes}"
    if sha256 != entry.sha256:
        return f"{entry.path}: sha256 {sha256}, recorded {entry.sha256}"
    return None


def verify_entries(root: Path, entries: Sequence[FileEntry]) -> list[str]:
    """Compare the files under `root` with their entries.

    Args:
        root: The repository root, or the root of a clone after `dvc pull`.
        entries: The entries to check (all of them, or select_entries).

    Returns:
        One line per file that is missing or differs; empty if every file is
        the recorded one.
    """
    problems = [entry_problem(root, entry) for entry in entries]
    return [problem for problem in problems if problem is not None]
