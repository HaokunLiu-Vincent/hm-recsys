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
    check_arguments("recall", "r1_popularity", None)
    check_arguments("r6-predict", None, "calibration")
    for item in ("r6-fine-tune", "union", "ranker-dataset", "ranker", "repeat-r5"):
        check_arguments(item, None, None)
    with pytest.raises(ValueError, match="--item must be one of"):
        check_arguments("features", None, None)
    with pytest.raises(ValueError, match="--channel is required for --item recall"):
        check_arguments("recall", None, None)
    with pytest.raises(ValueError, match="--channel is required for --item recall"):
        check_arguments("union", "r1_popularity", None)
    with pytest.raises(ValueError, match="--start is required for --item r6-predict"):
        check_arguments("r6-predict", None, None)
    with pytest.raises(ValueError, match="--start is required for --item r6-predict"):
        check_arguments("ranker", None, "recorded")


def test_every_item_of_the_plan_has_a_name() -> None:
    assert ITEMS == (
        "recall",
        "r6-predict",
        "r6-fine-tune",
        "union",
        "ranker-dataset",
        "ranker",
        "repeat-r5",
        "repeat-r6",
        "repeat-ranker",
    )


def test_importing_the_entry_loads_neither_torch_nor_lightgbm() -> None:
    # The entry imports an item's module only when that item runs, so a recall process
    # never loads LightGBM and a Ranker process never loads torch.
    code = (
        "import sys, hmrec.calibration.run, hmrec.calibration.repeat; "
        "print('torch' in sys.modules, 'lightgbm' in sys.modules)"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert output.split() == ["False", "False"]
