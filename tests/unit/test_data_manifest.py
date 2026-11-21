"""Tests for hmrec.data_manifest: hashing, the DVC listing, the manifest file, verification.

Hand-checked values: the three bytes b"abc" have the well-known digests
    sha256  ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    md5     900150983cd24fb0d6963f7d28e17f72
"""

import json
from pathlib import Path

import pytest
import yaml

from hmrec.data_manifest import (
    FileEntry,
    TrackedFile,
    build_entries,
    file_entry,
    hash_file,
    manifest_text,
    parse_dvc_listing,
    read_data_manifest,
    select_entries,
    verify_entries,
    write_data_manifest,
)

REPO = Path(__file__).resolve().parents[2]
ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
ABC_MD5 = "900150983cd24fb0d6963f7d28e17f72"


def _write(root: Path, relative: str, content: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _entries(root: Path) -> tuple[FileEntry, ...]:
    """A small tree: one file directly tracked, two in a tracked folder."""
    _write(root, "artifacts/codes.json", b"{}")
    _write(root, "data/processed/a.parquet", b"abc")
    _write(root, "data/processed/sub/b.npy", b"abcd")
    names = ("artifacts/codes.json", "data/processed/a.parquet", "data/processed/sub/b.npy")
    return tuple(file_entry(root, name) for name in names)


def test_hash_file_gives_size_sha256_and_md5(tmp_path: Path) -> None:
    _write(tmp_path, "abc.bin", b"abc")
    assert hash_file(tmp_path / "abc.bin") == (3, ABC_SHA256, ABC_MD5)
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "missing.bin")


def test_parse_dvc_listing_sorts_and_rejects_bad_entries() -> None:
    listing = [
        {"isout": True, "isdir": False, "size": 4, "md5": "b" * 32, "path": "data/raw/b.csv"},
        {"isout": True, "isdir": False, "size": 3, "md5": ABC_MD5, "path": "data/raw/a.csv"},
    ]
    tracked = parse_dvc_listing(json.dumps(listing))
    assert tracked == (
        TrackedFile("data/raw/a.csv", 3, ABC_MD5),
        TrackedFile("data/raw/b.csv", 4, "b" * 32),
    )
    with pytest.raises(ValueError, match="listed twice"):
        parse_dvc_listing(json.dumps([listing[0], listing[0]]))
    with pytest.raises(ValueError, match="folder or has no md5"):
        parse_dvc_listing(json.dumps([{**listing[0], "isdir": True}]))
    with pytest.raises(ValueError, match="folder or has no md5"):
        parse_dvc_listing(json.dumps([{"path": "x", "size": 1, "isdir": False}]))
    with pytest.raises(ValueError, match="has no size"):
        parse_dvc_listing(json.dumps([{**listing[0], "size": None}]))
    with pytest.raises(ValueError, match="expected a list"):
        parse_dvc_listing(json.dumps({"path": "x"}))


def test_build_entries_requires_the_file_dvc_tracks(tmp_path: Path) -> None:
    _write(tmp_path, "data/raw/a.csv", b"abc")
    tracked = (TrackedFile("data/raw/a.csv", 3, ABC_MD5),)
    assert build_entries(tmp_path, tracked) == (
        FileEntry("data/raw/a.csv", 3, ABC_SHA256, ABC_MD5),
    )
    # The file on disk is not the one DVC recorded: no entry may be written for it.
    with pytest.raises(ValueError, match="DVC tracks"):
        build_entries(tmp_path, (TrackedFile("data/raw/a.csv", 3, "0" * 32),))
    with pytest.raises(FileNotFoundError):
        build_entries(tmp_path, (TrackedFile("data/raw/missing.csv", 3, ABC_MD5),))


def test_manifest_round_trip_and_totals(tmp_path: Path) -> None:
    entries = _entries(tmp_path)
    manifest_file = tmp_path / "configs" / "data_manifest.json"
    write_data_manifest(manifest_file, entries)
    assert read_data_manifest(manifest_file) == entries
    content = json.loads(manifest_file.read_text(encoding="utf-8"))
    # 2 + 3 + 4 bytes in 3 files.
    assert (content["n_files"], content["total_size_bytes"]) == (3, 9)
    assert content["schema_version"] == 1


def test_manifest_rejects_unsorted_duplicate_and_malformed(tmp_path: Path) -> None:
    entries = _entries(tmp_path)
    with pytest.raises(ValueError, match="sorted by path"):
        manifest_text(tuple(reversed(entries)))
    with pytest.raises(ValueError, match="sorted by path"):
        manifest_text((entries[0], entries[0]))
    manifest_file = tmp_path / "manifest.json"
    good = json.loads(manifest_text(entries))
    for change, message in (
        ({"schema_version": 2}, "schema_version"),
        ({"n_files": 4}, "does not fit"),
        ({"total_size_bytes": 10}, "does not fit"),
        ({"files": [{**good["files"][0], "sha256": "xyz"}]}, "hexadecimal"),
        ({"files": [{"path": "a"}]}, "expected the keys"),
    ):
        manifest_file.write_text(json.dumps({**good, **change}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            read_data_manifest(manifest_file)
    with pytest.raises(FileNotFoundError):
        read_data_manifest(tmp_path / "no_manifest.json")


def test_select_entries_by_file_or_folder(tmp_path: Path) -> None:
    entries = _entries(tmp_path)
    processed = select_entries(entries, ["data/processed"])
    assert [entry.path for entry in processed] == [
        "data/processed/a.parquet",
        "data/processed/sub/b.npy",
    ]
    both = select_entries(entries, ["data/processed/", "artifacts/codes.json"])
    assert [entry.path for entry in both] == [entry.path for entry in entries]
    # A prefix is a path, not a string prefix: "data/proc" is not "data/processed".
    with pytest.raises(ValueError, match="no entry"):
        select_entries(entries, ["data/proc"])


def test_verify_entries_reports_missing_and_changed_files(tmp_path: Path) -> None:
    entries = _entries(tmp_path)
    assert verify_entries(tmp_path, entries) == []
    # Same size, other bytes: only the sha256 can tell.
    _write(tmp_path, "data/processed/a.parquet", b"abd")
    _write(tmp_path, "data/processed/sub/b.npy", b"abcde")
    (tmp_path / "artifacts" / "codes.json").unlink()
    problems = verify_entries(tmp_path, entries)
    assert len(problems) == 3
    assert problems[0] == "artifacts/codes.json: missing"
    assert problems[1].startswith("data/processed/a.parquet: sha256 ")
    assert problems[2] == "data/processed/sub/b.npy: size 5, recorded 4"


def test_the_manifest_of_this_repository_fits_the_pointer_files() -> None:
    entries = read_data_manifest(REPO / "configs" / "data_manifest.json")
    assert entries, "the data manifest lists no file"
    pointers = sorted(REPO.glob("data/*.dvc")) + sorted(REPO.glob("artifacts/features/*.dvc"))
    if pointers:
        # A checkout (the laptop). Every pointer file names a tracked file or folder with
        # its size and number of files; the manifest must hold exactly those files.
        covered = 0
        for pointer in pointers:
            out = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]
            target = (pointer.parent / out["path"]).relative_to(REPO).as_posix()
            selected = select_entries(entries, [target])
            assert len(selected) == out.get("nfiles", 1), target
            assert sum(entry.size_bytes for entry in selected) == out["size"], target
            if "nfiles" not in out:
                assert selected[0].md5 == out["md5"], target
            covered += len(selected)
        assert covered == len(entries)
    else:
        # Inside an image: data/ and artifacts/ are not copied, so there is no pointer
        # file to compare with. The manifest itself is in the image (configs/) and was
        # validated by read_data_manifest above; every entry lies in one of the two folders.
        assert not (REPO / ".dvc").exists()
        assert all(entry.path.startswith(("data/", "artifacts/")) for entry in entries)
