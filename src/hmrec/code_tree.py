"""One hash for the code tree of a container image (docs/mlops.md §8.1; decision of 2026-11-22).

Why: inside an image there is no `.git`, so git cannot say whether the code that
runs is the code of the image's commit. Until now "not dirty" rested only on how
the image is built. A folder mounted over part of the code (for example
`--mount src=./src,dst=/app/src`) was not seen.

How:

    build time   the Dockerfile runs scripts/code_tree_hash.py on the code it has
                 just copied and stores the result in a file of the image
                 (hmrec.git_state.IMAGE_CODE_TREE_FILE);
    run time     hmrec.git_state computes the hash again, compares it with the
                 stored one and refuses the run if they differ;
    records      the hash is written next to the commit.

What is hashed: every file under CODE_TREE_ENTRIES, which is the allow-list of
`.dockerignore` WITHOUT `results/`. results/ is left out for the same reason
hmrec.git_state.git_state leaves it out of "dirty": it holds the output of
runs, and a calibration run mounts its own results folder there. Byte-code
caches are skipped: they are written by the interpreter, not by a commit.

The hash is the sha256 of a manifest with one line per file, sorted by path:

    <sha256 of the file's bytes>  <path relative to the root, with "/">

followed by a newline character, so a changed, added, removed or renamed file each change it.

This module is at the bottom layer and imports only the standard library.
"""

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

# The allow-list of .dockerignore without results/ (tests/unit/test_code_tree.py
# compares the two, so a folder added to the image cannot be forgotten here).
CODE_TREE_ENTRIES: tuple[str, ...] = (
    "src",
    "scripts",
    "tests",
    "configs",
    "docs",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "Makefile",
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
)
# Written by the Python interpreter next to the sources; never part of a commit.
SKIPPED_DIRECTORY = "__pycache__"
SKIPPED_SUFFIX = ".pyc"
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
READ_BLOCK_BYTES = 1024 * 1024


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes, as 64 lower-case hexadecimal characters."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def is_skipped(relative_path: Path) -> bool:
    """True for a byte-code cache file (a `__pycache__` folder or a `.pyc` file)."""
    return SKIPPED_DIRECTORY in relative_path.parts or relative_path.suffix == SKIPPED_SUFFIX


def code_tree_files(root: Path, entries: Sequence[str] = CODE_TREE_ENTRIES) -> list[str]:
    """The files of the code tree, as paths relative to `root`, sorted.

    Args:
        root: The folder that holds the entries (the repository root; /app in an image).
        entries: Files and folders directly under `root`.

    Returns:
        Relative paths with "/" as separator, in sorted order. Folders are walked
        recursively; byte-code caches are skipped.

    Raises:
        FileNotFoundError: If an entry does not exist. A tree without one of its
            entries is not the tree an image was built from.
    """
    files: list[str] = []
    for entry in entries:
        path = root / entry
        if path.is_file():
            files.append(entry)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"code tree entry {entry!r} not found under {root}")
        for file in path.rglob("*"):
            relative = file.relative_to(root)
            if file.is_file() and not is_skipped(relative):
                files.append(relative.as_posix())
    return sorted(files)


def code_tree_manifest(root: Path, entries: Sequence[str] = CODE_TREE_ENTRIES) -> str:
    """The manifest the tree hash is computed from: one line per file (module docstring)."""
    lines = [f"{file_sha256(root / name)}  {name}\n" for name in code_tree_files(root, entries)]
    return "".join(lines)


def code_tree_sha256(root: Path, entries: Sequence[str] = CODE_TREE_ENTRIES) -> str:
    """The hash of the code tree under `root` (module docstring).

    Args:
        root: The folder that holds the entries.
        entries: Files and folders directly under `root`.

    Returns:
        64 lower-case hexadecimal characters.

    Raises:
        FileNotFoundError: If an entry does not exist.
    """
    manifest = code_tree_manifest(root, entries)
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def read_recorded_code_tree(path: Path) -> str:
    """Read a tree hash that was stored in a file (one line).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If its content is not a sha256 in hexadecimal.
    """
    if not path.is_file():
        raise FileNotFoundError(f"recorded code-tree hash not found: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if SHA256_HEX.fullmatch(value) is None:
        raise ValueError(f"{path} does not hold a sha256 (64 hexadecimal characters): {value!r}")
    return value
