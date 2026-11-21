"""Tests for hmrec.calibration.run: one calibration item per process."""

import subprocess
import sys

import pytest

from hmrec.calibration.run import ITEMS, check_arguments, process_label


def test_process_label() -> None:
    assert process_label("recall", "r5_wals", None) == "recall_r5_wals"
    assert process_label("r6-predict", None, "recorded") == "r6_predict_recorded"
    assert process_label("ranker-dataset", None, None) == "ranker_dataset"


def test_arguments_must_fit_the_item() -> None:
    check_arguments("recall", "r1_popularity", None, None)
    check_arguments("r6-predict", None, "calibration", None)
    check_arguments("repeat-r6", None, None, "run3")
    for item in ("r6-fine-tune", "union", "ranker-dataset", "ranker", "e2e-features", "e2e-ranker"):
        check_arguments(item, None, None, None)
    with pytest.raises(ValueError, match="--item must be one of"):
        check_arguments("features", None, None, None)
    with pytest.raises(ValueError, match="--channel is required for --item recall"):
        check_arguments("recall", None, None, None)
    with pytest.raises(ValueError, match="--channel is required for --item recall"):
        check_arguments("union", "r1_popularity", None, None)
    with pytest.raises(ValueError, match="--start is required for --item r6-predict"):
        check_arguments("r6-predict", None, None, None)
    with pytest.raises(ValueError, match="--start is required for --item r6-predict"):
        check_arguments("ranker", None, "recorded", None)
    # A repeat item compares two runs and must be told the earlier one; no other item may.
    with pytest.raises(ValueError, match="--against is required for a repeat item"):
        check_arguments("repeat-ranker", None, None, None)
    with pytest.raises(ValueError, match="--against is required for a repeat item"):
        check_arguments("e2e-features", None, None, "run1")


def test_every_item_of_the_plan_has_a_name() -> None:
    assert ITEMS == (
        "recall",
        "r6-predict",
        "r6-fine-tune",
        "union",
        "ranker-dataset",
        "ranker",
        "e2e-features",
        "e2e-ranker",
        "repeat-r5",
        "repeat-r6",
        "repeat-ranker",
    )


def test_importing_the_entry_loads_neither_torch_nor_lightgbm() -> None:
    # The entry imports an item's module only when that item runs, so a recall process
    # never loads LightGBM and a Ranker process never loads torch.
    code = (
        "import sys, hmrec.calibration.run, hmrec.calibration.repeat, "
        "hmrec.calibration.feature_items; "
        "print('torch' in sys.modules, 'lightgbm' in sys.modules)"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert output.split() == ["False", "False"]
