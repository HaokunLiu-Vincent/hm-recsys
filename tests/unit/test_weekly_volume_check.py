"""Tests for weekly.volume_check: every file operation the pipeline uses, tried in a folder.

On a local disk every check passes; that is what the tests show, together with
what a missing or changed input looks like. The check on a mounted bucket is the
smoke test of M13 (scripts/m13_volume_check.py) and is run there.
"""

from pathlib import Path

import pytest

from hmrec.data_manifest import FileEntry, file_entry
from hmrec.weekly.volume_check import (
    CHECK_DIR,
    FAILED,
    OK,
    REPORTED,
    check_inputs,
    check_volume,
    run_check,
)

INPUT_FILES = {
    "data/processed/transactions.parquet": b"purchase lines",
    "data/bootstrap/manifest.json": b"{}",
    "artifacts/features/category_codes.json": b"[]",
    # Not an input of a tick: the check must not ask for it.
    "data/raw/articles.csv": b"raw",
}


def _inputs(root: Path) -> tuple[Path, list[FileEntry]]:
    """Write the small input files under root/inputs and return their manifest entries."""
    inputs_root = root / "inputs"
    for name, content in INPUT_FILES.items():
        path = inputs_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return inputs_root, [file_entry(inputs_root, name) for name in INPUT_FILES]


def _run(tmp_path: Path, hash_inputs: bool = True, run_name: str = "run1"):
    inputs_root, entries = _inputs(tmp_path)
    return check_volume(
        state_root=tmp_path / "pipeline",
        inputs_root=inputs_root,
        entries=entries,
        run_name=run_name,
        megabytes=8,
        hash_inputs=hash_inputs,
        seed=42,
    )


def test_every_check_passes_on_a_local_disk_and_the_folder_is_removed(tmp_path: Path) -> None:
    results = _run(tmp_path)
    by_name = {result.name: result for result in results}
    assert list(by_name) == [
        "sequential write and read",
        "parquet",
        "npy",
        "write at an offset",
        "append",
        "rename and replace a file",
        "list and exists",
        "the store's commit",
        "rename a folder",
        "inputs",
        "remove the check's folder",
    ]
    # A folder rename is reported and never decides; every other check is required.
    assert by_name["rename a folder"].status == REPORTED
    assert all(result.status == OK for name, result in by_name.items() if name != "rename a folder")
    assert "8 MB" in by_name["sequential write and read"].detail
    assert "3 files" in by_name["inputs"].detail and "sha256" in by_name["inputs"].detail
    # The check leaves nothing of its own behind, and the state folder itself stays.
    assert (tmp_path / "pipeline" / CHECK_DIR).is_dir()
    assert not (tmp_path / "pipeline" / CHECK_DIR / "run1").exists()
    assert all(result.line().startswith("volume check | ") for result in results)


def test_a_run_never_reuses_a_folder(tmp_path: Path) -> None:
    (tmp_path / "pipeline" / CHECK_DIR / "run1").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="never reuses"):
        _run(tmp_path)


def test_a_missing_or_changed_input_fails_the_check(tmp_path: Path) -> None:
    inputs_root, entries = _inputs(tmp_path)
    assert "every size equals" in check_inputs(inputs_root, entries, hash_inputs=False)
    # Same size, other bytes: only the sha256 comparison sees it.
    (inputs_root / "data/bootstrap/manifest.json").write_bytes(b"[]")
    assert "every size equals" in check_inputs(inputs_root, entries, hash_inputs=False)
    with pytest.raises(AssertionError, match=r"data/bootstrap/manifest.json: sha256"):
        check_inputs(inputs_root, entries, hash_inputs=True)
    (inputs_root / "data/processed/transactions.parquet").unlink()
    with pytest.raises(AssertionError, match=r"data/processed/transactions.parquet: missing"):
        check_inputs(inputs_root, entries, hash_inputs=False)
    # A file of data/raw/ is not an input of a tick: removing it changes nothing.
    (inputs_root / "data/raw/articles.csv").unlink()
    with pytest.raises(AssertionError) as error:
        check_inputs(inputs_root, entries, hash_inputs=False)
    assert "data/raw" not in str(error.value)


def test_a_failing_check_becomes_a_result_and_a_reported_one_never_fails() -> None:
    def broken() -> str:
        raise OSError("operation not supported")

    required = run_check("rename a folder", broken)
    assert required.status == FAILED and "OSError: operation not supported" in required.detail
    reported = run_check("rename a folder", broken, required=False)
    assert reported.status == REPORTED and "operation not supported" in reported.detail
    assert run_check("fine", lambda: "done").status == OK
