"""Tests for hmrec.code_tree: the hash of the code tree of a container image.

Expected values are built by hand from the definition in the module docstring:
the tree hash is the sha256 of a manifest with one line per file,
"<sha256 of the file>  <relative path>" plus a newline character, sorted by path.
"""

import hashlib
from pathlib import Path

import pytest

from hmrec.code_tree import (
    CODE_TREE_ENTRIES,
    code_tree_files,
    code_tree_manifest,
    code_tree_sha256,
    file_sha256,
    read_recorded_code_tree,
)

REPO = Path(__file__).resolve().parents[2]
ENTRIES = ("src", "pyproject.toml")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A tree of three files: src/a.py, src/pkg/b.py and pyproject.toml."""
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "a.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "src" / "pkg" / "b.py").write_text("b = 2\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return tmp_path


def test_file_sha256_is_the_sha256_of_the_bytes(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"abc")
    # The well-known sha256 of "abc".
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_files_are_relative_sorted_and_without_bytecode_caches(tree: Path) -> None:
    (tree / "src" / "__pycache__").mkdir()
    (tree / "src" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00")
    (tree / "src" / "stray.pyc").write_bytes(b"\x00")
    # A file outside the entries is not part of the tree.
    (tree / "notes.txt").write_text("x\n", encoding="utf-8")
    assert code_tree_files(tree, ENTRIES) == ["pyproject.toml", "src/a.py", "src/pkg/b.py"]


def test_the_hash_is_the_sha256_of_the_manifest(tree: Path) -> None:
    manifest = (
        f"{_sha256('[project]\n')}  pyproject.toml\n"
        f"{_sha256('a = 1\n')}  src/a.py\n"
        f"{_sha256('b = 2\n')}  src/pkg/b.py\n"
    )
    assert code_tree_manifest(tree, ENTRIES) == manifest
    assert code_tree_sha256(tree, ENTRIES) == _sha256(manifest)


def test_every_kind_of_change_changes_the_hash(tree: Path) -> None:
    before = code_tree_sha256(tree, ENTRIES)
    (tree / "src" / "a.py").write_text("a = 2\n", encoding="utf-8")
    edited = code_tree_sha256(tree, ENTRIES)
    (tree / "src" / "c.py").write_text("", encoding="utf-8")
    added = code_tree_sha256(tree, ENTRIES)
    (tree / "src" / "pkg" / "b.py").rename(tree / "src" / "pkg" / "renamed.py")
    renamed = code_tree_sha256(tree, ENTRIES)
    assert len({before, edited, added, renamed}) == 4


def test_the_hash_does_not_depend_on_where_the_tree_lies(tree: Path, tmp_path: Path) -> None:
    # /app in an image and a checkout on the laptop give the same hash for the same files.
    copy = tmp_path / "elsewhere"
    (copy / "src" / "pkg").mkdir(parents=True)
    for name in ("src/a.py", "src/pkg/b.py", "pyproject.toml"):
        (copy / name).write_bytes((tree / name).read_bytes())
    assert code_tree_sha256(copy, ENTRIES) == code_tree_sha256(tree, ENTRIES)


def test_a_missing_entry_is_an_error(tree: Path) -> None:
    with pytest.raises(FileNotFoundError, match="code tree entry 'configs' not found"):
        code_tree_sha256(tree, (*ENTRIES, "configs"))


def test_read_recorded_code_tree(tmp_path: Path) -> None:
    path = tmp_path / "code-tree.sha256"
    with pytest.raises(FileNotFoundError, match="recorded code-tree hash not found"):
        read_recorded_code_tree(path)
    value = _sha256("x")
    path.write_text(value + "\n", encoding="utf-8")
    assert read_recorded_code_tree(path) == value
    path.write_text("not a hash\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not hold a sha256"):
        read_recorded_code_tree(path)


def test_the_entries_are_the_allow_list_of_dockerignore_without_results() -> None:
    # .dockerignore lists what may enter an image: lines "!name" or "!name/". If a folder
    # is added there, it must be added to the tree hash too (or left out on purpose here).
    allowed = []
    for line in (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines():
        if line.startswith("!"):
            allowed.append(line[1:].rstrip("/"))
    assert "results" in allowed
    assert sorted(CODE_TREE_ENTRIES) == sorted(name for name in allowed if name != "results")


def test_the_tree_of_this_repository_can_be_hashed() -> None:
    value = code_tree_sha256(REPO)
    assert len(value) == 64 and set(value) <= set("0123456789abcdef")
