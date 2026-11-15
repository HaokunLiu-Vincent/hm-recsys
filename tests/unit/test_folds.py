import copy
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from hmrec.config import load_yaml
from hmrec.data.folds import (
    Fold,
    load_fold_manifest,
    parse_fold_manifest,
    validate_fold_manifest,
)

MANIFEST_FILE = Path(__file__).resolve().parents[2] / "configs" / "fold_manifest.yaml"


@pytest.fixture
def raw_manifest() -> dict[str, Any]:
    """A fresh copy of the real manifest content, safe to mutate in a test."""
    return copy.deepcopy(load_yaml(MANIFEST_FILE))


def parse_and_validate(raw: dict[str, Any]) -> None:
    manifest, checks = parse_fold_manifest(raw)
    validate_fold_manifest(manifest, checks)


# ---------------------------------------------------------------- the real manifest


def test_real_manifest_matches_proposal_section_3() -> None:
    manifest = load_fold_manifest(MANIFEST_FILE)

    assert manifest.names == (
        "D1", "D2", "D3", "D4",
        "F1", "F2", "F3", "F4",
        "V1", "V2", "V3", "V4",
        "T1", "T2", "T3", "T4",
    )  # fmt: skip
    d1 = manifest.get("D1")
    assert d1.anchor == date(2020, 6, 3)
    assert d1.label_end == date(2020, 6, 9)
    assert manifest.get("F1").anchor == date(2020, 7, 1)
    assert manifest.get("V1").anchor == date(2020, 7, 29)
    assert manifest.get("T1").anchor == date(2020, 8, 26)
    assert manifest.get("T4").label_end == date(2020, 9, 22)
    assert [fold.name for fold in manifest.by_phase("V")] == ["V1", "V2", "V3", "V4"]


def test_fold_window_helpers() -> None:
    fold = Fold(name="D1", phase="D", anchor=date(2020, 6, 3), label_end=date(2020, 6, 9))
    assert fold.label_start == date(2020, 6, 3)
    # Half-open end: labels are anchor <= t_dat < 2020-06-10.
    assert fold.label_end_exclusive == date(2020, 6, 10)
    assert fold.label_days == 7


def test_unknown_fold_and_phase_raise_key_error() -> None:
    manifest = load_fold_manifest(MANIFEST_FILE)
    with pytest.raises(KeyError, match="unknown fold 'X9'"):
        manifest.get("X9")
    with pytest.raises(KeyError, match="no folds for phase 'Z'"):
        manifest.by_phase("Z")


# ---------------------------------------------------------------- each check fails on bad input


def test_week_with_8_days_is_rejected(raw_manifest: dict[str, Any]) -> None:
    raw_manifest["folds"][0]["label_end"] = date(2020, 6, 10)  # D1: 06-03..06-10 = 8 days
    with pytest.raises(ValueError, match="fold D1: .* has 8 days, expected 7"):
        parse_and_validate(raw_manifest)


def test_overlapping_weeks_are_rejected(raw_manifest: dict[str, Any]) -> None:
    # Shift D2 one day earlier, keeping 7 days: D2 = 06-09..06-15 overlaps D1 on 06-09.
    raw_manifest["folds"][1]["anchor"] = date(2020, 6, 9)
    raw_manifest["folds"][1]["label_end"] = date(2020, 6, 15)
    with pytest.raises(ValueError, match="folds D1 and D2 overlap"):
        parse_and_validate(raw_manifest)


def test_gap_between_weeks_is_rejected(raw_manifest: dict[str, Any]) -> None:
    # Shift D1 one day earlier: D1 = 06-02..06-08, so 06-09 is in no fold.
    raw_manifest["folds"][0]["anchor"] = date(2020, 6, 2)
    raw_manifest["folds"][0]["label_end"] = date(2020, 6, 8)
    with pytest.raises(ValueError, match="gap between folds D1 and D2"):
        parse_and_validate(raw_manifest)


def test_wrong_last_date_is_rejected(raw_manifest: dict[str, Any]) -> None:
    raw_manifest["checks"]["last_label_date"] = date(2020, 9, 21)
    with pytest.raises(ValueError, match="last fold T4 ends 2020-09-22, expected 2020-09-21"):
        parse_and_validate(raw_manifest)


def test_missing_fold_is_rejected(raw_manifest: dict[str, Any]) -> None:
    del raw_manifest["folds"][-1]  # drop T4 -> only 15 folds
    with pytest.raises(ValueError, match="fold \\(phase, name\\) list is"):
        parse_and_validate(raw_manifest)


def test_wrong_phase_label_is_rejected(raw_manifest: dict[str, Any]) -> None:
    raw_manifest["folds"][4]["phase"] = "D"  # F1 marked as phase D
    with pytest.raises(ValueError, match="fold \\(phase, name\\) list is"):
        parse_and_validate(raw_manifest)


def test_datetime_instead_of_date_is_rejected(raw_manifest: dict[str, Any]) -> None:
    raw_manifest["folds"][0]["anchor"] = datetime(2020, 6, 3, 12, 0)
    with pytest.raises(ValueError, match=r"folds\[0\]\.anchor: expected a date"):
        parse_and_validate(raw_manifest)


def test_unknown_key_in_fold_is_rejected(raw_manifest: dict[str, Any]) -> None:
    raw_manifest["folds"][0]["label_start"] = date(2020, 6, 3)
    with pytest.raises(ValueError, match=r"unknown keys \['label_start'\]"):
        parse_and_validate(raw_manifest)
