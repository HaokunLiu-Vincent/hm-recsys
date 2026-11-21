"""Tests for hmrec.platform_records, and for the record configs/platforms/linux-arm64.yaml.

The record was written by hand. The tests compare it with the files it
describes (Dockerfile, uv.lock, the Ranker lock, docs/decisions.md), so a record
that does not describe this repository fails here.
"""

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from hmrec.config import load_base_config, load_yaml
from hmrec.locks import file_sha256
from hmrec.paths import get_paths
from hmrec.platform_records import load_platform_record, parse_platform_record

REPO = Path(__file__).resolve().parents[2]
PATHS = get_paths()
RECORD_FILE = PATHS.platform_record_file("linux-arm64")


def _raw() -> dict[str, Any]:
    return load_yaml(RECORD_FILE)


def _locked_versions(package: str) -> list[str]:
    """Every version of a package in uv.lock (torch has one per platform)."""
    lock_text = (REPO / "uv.lock").read_text(encoding="utf-8")
    return re.findall(rf'name = "{package}"\nversion = "([^"]+)"', lock_text)


def test_the_record_lies_where_the_paths_say_and_names_its_platform() -> None:
    assert RECORD_FILE == REPO / "configs" / "platforms" / "linux-arm64.yaml"
    record = load_platform_record(RECORD_FILE)
    assert (record.platform, record.os, record.architecture) == ("linux-arm64", "linux", "aarch64")


def test_the_acceptance_is_the_decision_of_2026_11_22() -> None:
    acceptance = load_platform_record(RECORD_FILE).acceptance
    # At most 1.0% relative difference on four quantities.
    assert acceptance.max_relative_difference == 0.01
    assert acceptance.quantities == (
        "recall of a channel in a week",
        "union recall of a week",
        "MAP@12 of a single week",
        "pooled MAP@12 over the replayed weeks",
    )
    # Two structural checks stay; three things are only reported.
    assert len(acceptance.structural_checks) == 2 and len(acceptance.reported_only) == 3
    assert acceptance.bit_identical_repetition_required is False


def test_the_record_describes_the_image_of_this_repository() -> None:
    record = load_platform_record(RECORD_FILE)
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    # The base image line of the Dockerfile, digest included.
    assert f"FROM {record.base_image}\n" in dockerfile
    (libgomp_version,) = re.findall(r"ARG LIBGOMP_VERSION=(\S+)", dockerfile)
    assert record.openmp_version == f"libgomp1 {libgomp_version}"
    assert _locked_versions("lightgbm") == [record.lightgbm_version]
    # torch is locked twice: PyPI for macOS, the CPU-only index for Linux.
    assert record.torch_version in _locked_versions("torch")
    assert record.torch_version.endswith("+cpu")
    assert record.num_threads == load_base_config(PATHS.base_config_file).num_threads
    assert record.r6_fine_tune_setting == "deterministic"
    raw = _raw()
    assert raw["settings"]["r5_fit_scipy_blas_threads"] == 1
    for package in ("implicit", "numpy", "scipy", "polars"):
        assert _locked_versions(package) == [raw["recall_runtime"][f"{package}_version"]]


def test_the_record_was_made_for_the_ranker_lock_of_this_repository() -> None:
    record = load_platform_record(RECORD_FILE)
    assert REPO / record.ranker_lock_path == PATHS.ranker_lock_file
    assert record.ranker_lock_sha256 == file_sha256(PATHS.ranker_lock_file)


def test_the_decision_entry_exists() -> None:
    record = load_platform_record(RECORD_FILE)
    prefix = "docs/decisions.md "
    assert record.decision_record.startswith(prefix)
    title = "## " + record.decision_record.removeprefix(prefix)
    assert title in (REPO / "docs" / "decisions.md").read_text(encoding="utf-8").splitlines()


def test_the_calibration_basis_names_the_images_and_the_result_tables() -> None:
    calibration = _raw()["calibration"]
    assert calibration["week"] == "V4"
    for image in calibration["images"]:
        assert re.fullmatch(r"[0-9a-f]{40}", image["commit"])
        assert re.fullmatch(r"[0-9a-f]{64}", image["code_tree_sha256"])
    # The two images are the ones the calibration rows carry.
    recorded = {(image["commit"], image["code_tree_sha256"]) for image in calibration["images"]}
    runs = (REPO / "results" / "m10_calibration" / "runs.csv").read_text(encoding="utf-8")
    assert all(commit in runs and tree in runs for commit, tree in recorded)
    commit = calibration["result_tables_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert len(calibration["result_tables"]) == 4
    if (REPO / ".git").exists():
        # A checkout with history (the laptop): the recorded sha256 is the one of each
        # table AS IT WAS at that commit. The tables are append-only, so the files of
        # today may be longer; the record must not be edited when they grow.
        for path, recorded_sha256 in calibration["result_tables"].items():
            content = subprocess.run(
                ["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True, check=True
            ).stdout
            assert hashlib.sha256(content).hexdigest() == recorded_sha256, path
    else:
        # Inside an image there is no git history to read the old tables from. The
        # tables exist and the recorded values are sha256 values.
        for path, recorded_sha256 in calibration["result_tables"].items():
            assert (REPO / path).is_file(), path
            assert re.fullmatch(r"[0-9a-f]{64}", recorded_sha256)


def test_a_malformed_record_is_refused(tmp_path: Path) -> None:
    raw = _raw()
    raw["tolerance"] = 0.5  # a key the format does not know
    with pytest.raises(ValueError):
        parse_platform_record(raw, "test.yaml")
    raw = _raw()
    raw["acceptance"]["max_relative_difference"] = 1.5
    with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
        parse_platform_record(raw, "test.yaml")
    raw = _raw()
    raw["base_image"] = "python:3.12-slim"
    with pytest.raises(ValueError, match="must be pinned by digest"):
        parse_platform_record(raw, "test.yaml")
    raw = _raw()
    raw["acceptance"]["structural_checks"] = []
    with pytest.raises(ValueError, match="at least one quantity and one structural check"):
        parse_platform_record(raw, "test.yaml")
    # A record under another file name is not a record of that platform.
    copy = tmp_path / "linux-amd64.yaml"
    copy.write_text(yaml.safe_dump(_raw()), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 'linux-amd64'"):
        load_platform_record(copy)
    with pytest.raises(FileNotFoundError, match="platform record not found"):
        load_platform_record(tmp_path / "linux-riscv64.yaml")
