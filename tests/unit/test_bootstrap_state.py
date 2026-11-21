"""Tests for hmrec.bootstrap_state: which files, the copy rules, the manifest.

The lock files and the fold manifest are the repository's own (configs/ is in the
image too). Artifacts and result tables are small fakes in a temporary project,
and the locks' recorded sha256 are replaced by the fakes' sha256 where a test
needs a copy to succeed.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from hmrec.bootstrap_state import (
    COPY,
    ROWS,
    BootstrapItem,
    BootstrapPlan,
    bootstrap_plan,
    build_bootstrap_state,
    folds_before,
    read_bootstrap_manifest,
    rows_of_folds,
    verify_bootstrap_state,
)
from hmrec.config import BootstrapConfig
from hmrec.data.folds import FoldManifest, load_fold_manifest
from hmrec.locks import (
    BASE_CHECKPOINT,
    RankerLock,
    TwoTowerLock,
    load_ranker_lock,
    load_twotower_lock,
)
from hmrec.paths import ProjectPaths

REPO = Path(__file__).resolve().parents[2]
CONFIG = BootstrapConfig(first_tick_fold="V1", monitoring_weeks=4, ranker_experiment="full")
COMMIT = "a" * 40
STATE_FOLDS = ("F1", "F2", "F3", "F4")
# A result table with rows of weeks before, at and after the first tick. The rows of V1
# and T1 are the "poison": nothing of them may reach the state.
TABLE = (
    b"fold,scope,recall\r\n"
    b"D4,union,0.20\r\n"
    b"F1,union,0.21\r\n"
    b"F2,union,0.22\r\n"
    b"F3,union,0.23\r\n"
    b"F4,union,0.24\r\n"
    b"V1,union,0.99\r\n"
    b"T1,union,0.98\r\n"
)
KEPT_ROWS = (
    b"fold,scope,recall\r\nF1,union,0.21\r\nF2,union,0.22\r\nF3,union,0.23\r\nF4,union,0.24\r\n"
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _manifest() -> FoldManifest:
    return load_fold_manifest(REPO / "configs" / "fold_manifest.yaml")


def _locks() -> tuple[RankerLock, TwoTowerLock]:
    locks = REPO / "configs" / "locks"
    return load_ranker_lock(locks / "ranker.lock.yaml"), load_twotower_lock(
        locks / "twotower.lock.yaml"
    )


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _project(tmp_path: Path) -> tuple[ProjectPaths, TwoTowerLock, RankerLock]:
    """A temporary project with fake stage-1 files, and locks that record the fakes' sha256."""
    paths = ProjectPaths(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        results_root=tmp_path / "results",
    )
    ranker_lock, twotower_lock = _locks()
    base = paths.twotower_checkpoint_file(twotower_lock.channel.config["base_epoch"])
    chain = paths.twotower_locked_chain_checkpoint_file("F4")
    alarm = paths.twotower_data_dir("base") / "customer_idx.npy"
    _write(base, b"base model")
    _write(chain, b"chain model after F4")
    _write(alarm, b"customer idx")
    # The chain table says which sha256 the model after F4 has; V1's row is poison.
    chain_table = (
        "fold,sha256\r\n"
        f"F1,{'1' * 64}\r\nF2,{'2' * 64}\r\nF3,{'3' * 64}\r\n"
        f"F4,{_sha256(b'chain model after F4')}\r\nV1,{'9' * 64}\r\n"
    )
    _write(paths.m6_twotower_chain_results_file, chain_table.encode())
    _write(paths.m6_candidates_results_file, TABLE)
    _write(paths.m6_features_results_file, TABLE)
    ratio = ranker_lock.sampling.negatives_per_positive
    for fold in STATE_FOLDS:
        sample = f"sample of {fold}".encode()
        _write(paths.ranker_sample_file("full", fold, ratio), sample)
        record = json.dumps({"fold": fold, "sha256": _sha256(sample)}).encode()
        _write(paths.ranker_sample_record_file("full", fold, ratio), record)
    checkpoints = dict(twotower_lock.checkpoints)
    checkpoints[BASE_CHECKPOINT] = replace(
        checkpoints[BASE_CHECKPOINT], sha256=_sha256(b"base model")
    )
    alarm_users = replace(twotower_lock.alarm_users, source_sha256=_sha256(b"customer idx"))
    fake_lock = replace(twotower_lock, checkpoints=checkpoints, alarm_users=alarm_users)
    return paths, fake_lock, ranker_lock


def _plan(tmp_path: Path) -> tuple[ProjectPaths, BootstrapPlan]:
    paths, twotower_lock, ranker_lock = _project(tmp_path)
    return paths, bootstrap_plan(paths, CONFIG, _manifest(), ranker_lock, twotower_lock)


def _tree(root: Path) -> dict[str, bytes]:
    """Every file under a folder with its bytes."""
    return {
        file.relative_to(root).as_posix(): file.read_bytes()
        for file in sorted(root.rglob("*"))
        if file.is_file()
    }


def test_folds_before_the_first_tick() -> None:
    manifest = _manifest()
    assert folds_before(manifest, "V1", 4) == STATE_FOLDS
    assert folds_before(manifest, "T1", 2) == ("V3", "V4")
    with pytest.raises(ValueError, match="unknown fold"):
        folds_before(manifest, "X1", 4)
    # D3 has only two folds before it.
    with pytest.raises(ValueError, match="2 folds before"):
        folds_before(manifest, "D3", 4)
    with pytest.raises(ValueError):
        folds_before(manifest, "V1", 0)


def test_rows_of_folds_keeps_only_the_weeks_before_the_first_tick() -> None:
    known = _manifest().names
    # Byte for byte: the header and the four rows, with their CRLF line ends.
    assert rows_of_folds(TABLE, STATE_FOLDS, known) == KEPT_ROWS
    assert b"0.99" not in rows_of_folds(TABLE, STATE_FOLDS, known)
    with pytest.raises(ValueError, match="first column"):
        rows_of_folds(b"scope,fold\r\nunion,F1\r\n", STATE_FOLDS, known)
    # A line that does not start with a fold (e.g. the rest of a field with a line break).
    with pytest.raises(ValueError, match="line 3 does not start with a fold"):
        rows_of_folds(b"fold,detail\r\nF1,a\r\nrest of a field\r\n", ("F1",), known)
    with pytest.raises(ValueError, match="no row of fold"):
        rows_of_folds(b"fold,recall\r\nF1,0.2\r\n", ("F1", "F2"), known)


def test_plan_lists_what_the_first_tick_reads(tmp_path: Path) -> None:
    _, plan = _plan(tmp_path)
    assert (plan.first_tick_fold, plan.closed_fold) == ("V1", "F4")
    # 3 files of the R6 chain, a sample and its record for each of F1-F4 (ratio 1 : 40 and
    # the 4-week window come from the Ranker lock), 3 tables of history rows.
    assert [item.destination for item in plan.items] == [
        "models/twotower/base/epoch_4.pt",
        "models/twotower/chain_locked/F4.pt",
        "twotower/data/base/customer_idx.npy",
        "ranker/samples/full/F1/r40.parquet",
        "ranker/samples/full/F1/r40.json",
        "ranker/samples/full/F2/r40.parquet",
        "ranker/samples/full/F2/r40.json",
        "ranker/samples/full/F3/r40.parquet",
        "ranker/samples/full/F3/r40.json",
        "ranker/samples/full/F4/r40.parquet",
        "ranker/samples/full/F4/r40.json",
        "monitoring/m6_twotower_chain.csv",
        "monitoring/m6_candidates.csv",
        "monitoring/m6_features.csv",
    ]
    kinds = [item.kind for item in plan.items]
    assert kinds == [COPY] * 11 + [ROWS] * 3
    assert all(item.folds == STATE_FOLDS for item in plan.items if item.kind == ROWS)
    by_name = {item.destination: item for item in plan.items}
    chain = by_name["models/twotower/chain_locked/F4.pt"]
    assert chain.source == "artifacts/models/twotower/chain_locked/F4.pt"
    assert chain.expected_sha256 == hashlib.sha256(b"chain model after F4").hexdigest()
    assert chain.recorded_in == "results/m6_twotower_chain.csv fold F4"
    sample = by_name["ranker/samples/full/F2/r40.parquet"]
    assert sample.expected_sha256 == hashlib.sha256(b"sample of F2").hexdigest()
    assert sample.recorded_in == "artifacts/ranker/samples/full/F2/r40.json"


def test_plan_fails_without_a_sample_record(tmp_path: Path) -> None:
    paths, twotower_lock, ranker_lock = _project(tmp_path)
    ratio = ranker_lock.sampling.negatives_per_positive
    paths.ranker_sample_record_file("full", "F3", ratio).unlink()
    with pytest.raises(FileNotFoundError, match="sample record not found"):
        bootstrap_plan(paths, CONFIG, _manifest(), ranker_lock, twotower_lock)


def test_item_kinds_are_checked() -> None:
    with pytest.raises(ValueError, match="kind must be"):
        BootstrapItem("a", "b", "move", (), "", "", "")
    with pytest.raises(ValueError, match="lists folds"):
        BootstrapItem("a", "b", COPY, ("F1",), "", "", "")
    with pytest.raises(ValueError, match="lists folds"):
        BootstrapItem("a", "b", ROWS, (), "", "", "")


def test_build_copies_and_leaves_every_source_as_it_was(tmp_path: Path) -> None:
    paths, plan = _plan(tmp_path)
    sources_before = {**_tree(paths.artifacts_root), **_tree(paths.results_root)}
    entries = build_bootstrap_state(paths, plan, _manifest().names, COMMIT)

    # Copied, not moved or changed: every stage-1 file is still there with the same bytes.
    assert {**_tree(paths.artifacts_root), **_tree(paths.results_root)} == sources_before
    state = _tree(paths.bootstrap_dir)
    assert sorted(state) == sorted([*(item.destination for item in plan.items), "manifest.json"])
    assert state["models/twotower/chain_locked/F4.pt"] == b"chain model after F4"
    assert state["ranker/samples/full/F3/r40.parquet"] == b"sample of F3"
    # History tables: the header and the rows of F1-F4 only. Nothing of V1 or T1.
    assert state["monitoring/m6_candidates.csv"] == KEPT_ROWS
    assert b"V1" not in state["monitoring/m6_twotower_chain.csv"]
    assert not paths.bootstrap_dir.with_name("bootstrap.partial").exists()

    assert read_bootstrap_manifest(paths.bootstrap_manifest_file) == entries
    by_path = {entry.path: entry for entry in entries}
    copied = by_path["twotower/data/base/customer_idx.npy"]
    assert copied.sha256 == copied.source_sha256 == hashlib.sha256(b"customer idx").hexdigest()
    assert copied.size_bytes == len(b"customer idx")
    rows = by_path["monitoring/m6_candidates.csv"]
    assert rows.sha256 == hashlib.sha256(KEPT_ROWS).hexdigest()
    assert rows.source_sha256 == hashlib.sha256(TABLE).hexdigest()
    assert (rows.kind, rows.folds, rows.source) == (ROWS, STATE_FOLDS, "results/m6_candidates.csv")
    content = json.loads(paths.bootstrap_manifest_file.read_text(encoding="utf-8"))
    assert (content["first_tick_fold"], content["closed_fold"]) == ("V1", "F4")
    assert content["git_commit"] == COMMIT
    assert verify_bootstrap_state(paths.bootstrap_dir, entries) == []


def test_build_never_replaces_a_state(tmp_path: Path) -> None:
    paths, plan = _plan(tmp_path)
    build_bootstrap_state(paths, plan, _manifest().names, COMMIT)
    before = _tree(paths.bootstrap_dir)
    with pytest.raises(FileExistsError, match="never replaced"):
        build_bootstrap_state(paths, plan, _manifest().names, COMMIT)
    assert _tree(paths.bootstrap_dir) == before


def test_build_stops_at_a_left_over_partial_folder(tmp_path: Path) -> None:
    paths, plan = _plan(tmp_path)
    partial = paths.bootstrap_dir.with_name("bootstrap.partial")
    partial.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="never replaced"):
        build_bootstrap_state(paths, plan, _manifest().names, COMMIT)
    assert not paths.bootstrap_dir.exists()


def test_build_refuses_a_source_that_is_not_the_recorded_file(tmp_path: Path) -> None:
    paths, plan = _plan(tmp_path)
    # The chain model on disk is another file than the one results/m6_twotower_chain.csv records.
    paths.twotower_locked_chain_checkpoint_file("F4").write_bytes(b"another model")
    with pytest.raises(ValueError, match="not the stage-1 file"):
        build_bootstrap_state(paths, plan, _manifest().names, COMMIT)
    # No folder named data/bootstrap/ exists: a half-built state never looks finished.
    assert not paths.bootstrap_dir.exists()


def test_verify_finds_a_changed_and_a_missing_file(tmp_path: Path) -> None:
    paths, plan = _plan(tmp_path)
    entries = build_bootstrap_state(paths, plan, _manifest().names, COMMIT)
    (paths.bootstrap_dir / "ranker/samples/full/F1/r40.parquet").write_bytes(b"sample of F9")
    (paths.bootstrap_dir / "monitoring/m6_features.csv").unlink()
    problems = verify_bootstrap_state(paths.bootstrap_dir, entries)
    assert len(problems) == 2
    assert problems[0].startswith("ranker/samples/full/F1/r40.parquet: size 12 sha256 ")
    assert problems[1] == "monitoring/m6_features.csv: missing"


def test_manifest_reader_rejects_other_schemas(tmp_path: Path) -> None:
    paths, plan = _plan(tmp_path)
    build_bootstrap_state(paths, plan, _manifest().names, COMMIT)
    content = json.loads(paths.bootstrap_manifest_file.read_text(encoding="utf-8"))
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**content, "schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        read_bootstrap_manifest(other)
    del content["git_commit"]
    other.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="expected the keys"):
        read_bootstrap_manifest(other)
    with pytest.raises(FileNotFoundError):
        read_bootstrap_manifest(tmp_path / "missing.json")
