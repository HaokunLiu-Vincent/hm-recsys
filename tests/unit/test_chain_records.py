"""Tests for recall.chain_records: the chain may only move forward one recorded model at a time."""

import dataclasses
from pathlib import Path

import pytest

from hmrec.locks import file_sha256
from hmrec.recall.chain_records import (
    ChainRecord,
    append_chain_record,
    check_model_and_record_agree,
    read_chain_records,
    record_to_row,
    row_to_record,
    start_model_for,
)
from hmrec.recall.locked import ModelFile

FOLDS = ("F1", "F2", "F3")
F1_START = ModelFile(Path("locked/D4.pt"), "a" * 64)


def _record(fold: str, sha256: str) -> ChainRecord:
    return ChainRecord(
        fold=fold,
        anchor="2020-07-01",
        start_model="D4.pt",
        start_sha256="a" * 64,
        model=f"{fold}.pt",
        sha256=sha256,
        n_new_rows=300,
        n_replay_rows=300,
        epochs=2,
        steps=4,
        n_train_samples=600,
        n_alarm_samples=7,
        mean_train_loss=5.5 + 1e-13,  # needs repr() to survive the CSV
        alarm_loss=5.25,
        fine_tune_seconds=61.5,
        sampled_peak_rss_gb=3.4,
        git_commit="abc123",
        git_dirty=False,
        finished_at="2026-11-18T10:00:00",
    )


def _model(tmp_path: Path, fold: str, content: bytes) -> Path:
    path = tmp_path / f"{fold}.pt"
    path.write_bytes(content)
    return path


def test_record_survives_the_csv_exactly(tmp_path: Path) -> None:
    record = _record("F1", "b" * 64)
    assert row_to_record(record_to_row(record)) == record
    path = tmp_path / "m6_twotower_chain.csv"
    assert read_chain_records(path) == {}
    append_chain_record(path, record)
    assert read_chain_records(path) == {"F1": record}


def test_a_fold_gets_exactly_one_row(tmp_path: Path) -> None:
    path = tmp_path / "m6_twotower_chain.csv"
    append_chain_record(path, _record("F1", "b" * 64))
    with pytest.raises(ValueError, match="already has a row for fold F1"):
        append_chain_record(path, _record("F1", "c" * 64))


def test_first_fold_starts_from_the_locked_model(tmp_path: Path) -> None:
    start = start_model_for("F1", FOLDS, F1_START, {}, lambda fold: tmp_path / f"{fold}.pt")
    assert start == F1_START


def test_later_fold_starts_from_the_previous_folds_recorded_model(tmp_path: Path) -> None:
    f1_model = _model(tmp_path, "F1", b"weights after F1")
    records = {"F1": _record("F1", file_sha256(f1_model))}
    start = start_model_for("F2", FOLDS, F1_START, records, lambda fold: tmp_path / f"{fold}.pt")
    assert start == ModelFile(f1_model, file_sha256(f1_model))


def test_a_week_cannot_be_skipped(tmp_path: Path) -> None:
    # F1 is recorded, F2 is not: F3 may not start (it would need F2's model).
    f1_model = _model(tmp_path, "F1", b"weights after F1")
    records = {"F1": _record("F1", file_sha256(f1_model))}
    with pytest.raises(ValueError, match="F2 has no row"):
        start_model_for("F3", FOLDS, F1_START, records, lambda fold: tmp_path / f"{fold}.pt")
    with pytest.raises(ValueError, match="unknown fold"):
        start_model_for("V9", FOLDS, F1_START, records, lambda fold: tmp_path / f"{fold}.pt")


def test_a_replaced_or_missing_previous_model_is_refused(tmp_path: Path) -> None:
    f1_model = _model(tmp_path, "F1", b"weights after F1")
    records = {"F1": _record("F1", file_sha256(f1_model))}
    f1_model.write_bytes(b"weights from another run")  # replaced after it was recorded
    with pytest.raises(ValueError, match="This is not the locked file"):
        start_model_for("F2", FOLDS, F1_START, records, lambda fold: tmp_path / f"{fold}.pt")
    f1_model.unlink()
    with pytest.raises(FileNotFoundError):
        start_model_for("F2", FOLDS, F1_START, records, lambda fold: tmp_path / f"{fold}.pt")


def test_model_file_and_record_must_agree(tmp_path: Path) -> None:
    def checkpoint(fold: str) -> Path:
        return tmp_path / f"{fold}.pt"

    assert check_model_and_record_agree("F1", {}, checkpoint) is False  # neither: not fine-tuned
    model = _model(tmp_path, "F1", b"weights")
    with pytest.raises(ValueError, match="they must agree"):
        check_model_and_record_agree("F1", {}, checkpoint)  # a model nobody recorded
    records = {"F1": _record("F1", file_sha256(model))}
    assert check_model_and_record_agree("F1", records, checkpoint) is True
    model.unlink()
    with pytest.raises(ValueError, match="they must agree"):
        check_model_and_record_agree("F1", records, checkpoint)  # a record without its model


def test_git_dirty_reads_back_as_a_bool() -> None:
    dirty = dataclasses.replace(_record("F1", "b" * 64), git_dirty=True)
    assert row_to_record(record_to_row(dirty)).git_dirty is True
