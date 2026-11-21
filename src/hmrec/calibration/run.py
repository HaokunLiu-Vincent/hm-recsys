"""Run ONE calibration item in this process (the entry of scripts/m10_calibrate.py).

One item per process, as in stage 1: the process's ru_maxrss is then the true
memory peak of that item, and torch and LightGBM never share a process. The
module of an item is imported only when that item runs, so a recall process
never loads LightGBM and a Ranker process never loads torch (AGENTS.md §5).

Items (docs/m10_calibration_plan.md §4):

    recall --channel C        A (R1-R4) or B (R5): generate and compare one channel
    r6-predict --start S      C1 (recorded model) or the second half of C2 (the
                              run's own fine-tuned model)
    r6-fine-tune              the first half of C2
    union                     the run's own six channels, fused
    ranker-dataset, ranker    D: the binary Dataset; training, scoring, evaluation
    e2e-features, e2e-ranker  E: the features from the run's own candidates; scoring
                              them with the model of item D, and evaluation
    repeat-r5, repeat-r6,     two container runs compared bit by bit (no computation
    repeat-ranker             of the system; hmrec.calibration.repeat); --against
                              names the earlier run
"""

from hmrec.calibration.context import (
    CalibrationContext,
    ItemOutcome,
    finish_item,
    finish_repeat,
    start_calibration,
)
from hmrec.calibration.records import RepeatCheck
from hmrec.logging_utils import get_logger
from hmrec.threads import check_polars_thread_pool

LOGGER = get_logger("calibration.run")

ITEM_RECALL = "recall"
ITEM_R6_PREDICT = "r6-predict"
ITEM_R6_FINE_TUNE = "r6-fine-tune"
ITEM_UNION = "union"
ITEM_RANKER_DATASET = "ranker-dataset"
ITEM_RANKER = "ranker"
ITEM_E2E_FEATURES = "e2e-features"
ITEM_E2E_RANKER = "e2e-ranker"
ITEM_REPEAT_R5 = "repeat-r5"
ITEM_REPEAT_R6 = "repeat-r6"
ITEM_REPEAT_RANKER = "repeat-ranker"
REPEAT_ITEMS = (ITEM_REPEAT_R5, ITEM_REPEAT_R6, ITEM_REPEAT_RANKER)
ITEMS = (
    ITEM_RECALL,
    ITEM_R6_PREDICT,
    ITEM_R6_FINE_TUNE,
    ITEM_UNION,
    ITEM_RANKER_DATASET,
    ITEM_RANKER,
    ITEM_E2E_FEATURES,
    ITEM_E2E_RANKER,
    *REPEAT_ITEMS,
)
LIGHTGBM_ITEMS = (ITEM_RANKER_DATASET, ITEM_RANKER, ITEM_E2E_RANKER)


def process_label(item: str, channel: str | None, start: str | None) -> str:
    """The name of a process for its log file, e.g. "recall_r5_wals"."""
    parts = [item.replace("-", "_"), channel, start]
    return "_".join(part for part in parts if part)


def check_arguments(item: str, channel: str | None, start: str | None, against: str | None) -> None:
    """Fail on an argument that does not fit the item.

    Raises:
        ValueError: On an unknown item; if --channel is missing for "recall" or
            given for another item; if --start is missing for "r6-predict" or
            given for another item; if --against is missing for a repeat item
            or given for another item.
    """
    if item not in ITEMS:
        raise ValueError(f"--item must be one of {ITEMS}, got {item!r}")
    if (item == ITEM_RECALL) != (channel is not None):
        raise ValueError("--channel is required for --item recall and not allowed otherwise")
    if (item == ITEM_R6_PREDICT) != (start is not None):
        raise ValueError("--start is required for --item r6-predict and not allowed otherwise")
    if (item in REPEAT_ITEMS) != (against is not None):
        raise ValueError("--against is required for a repeat item and not allowed otherwise")


def run_torch_side_item(
    context: CalibrationContext, item: str, channel: str | None, start: str | None
) -> ItemOutcome:
    """Run a recall item. Imports the torch-side module here, not at the top (module docstring)."""
    from hmrec.calibration import recall_items

    if item == ITEM_RECALL and channel is not None:
        return recall_items.run_statistical_channel(context, channel)
    if item == ITEM_R6_PREDICT and start is not None:
        return recall_items.run_r6_predict(context, start)
    if item == ITEM_R6_FINE_TUNE:
        return recall_items.run_r6_fine_tune(context)
    return recall_items.run_union(context)


def run_lightgbm_side_item(context: CalibrationContext, item: str) -> ItemOutcome:
    """Run a Ranker item. Imports the LightGBM-side module here, not at the top."""
    from hmrec.calibration import ranker_items

    if item == ITEM_RANKER_DATASET:
        return ranker_items.run_ranker_dataset(context)
    if item == ITEM_E2E_RANKER:
        return ranker_items.run_end_to_end_ranker(context)
    return ranker_items.run_ranker(context)


def run_feature_item(context: CalibrationContext) -> ItemOutcome:
    """Run the feature build of item E. Its module imports neither torch nor LightGBM."""
    from hmrec.calibration import feature_items

    return feature_items.run_end_to_end_features(context)


def run_repeat_item(
    context: CalibrationContext, item: str, first_run: str, second_run: str
) -> list[RepeatCheck]:
    """Compare the outputs of two runs for one item."""
    from hmrec.calibration import repeat

    if item == ITEM_REPEAT_R5:
        return repeat.repeat_r5(context, first_run, second_run)
    if item == ITEM_REPEAT_R6:
        return repeat.repeat_r6(context, first_run, second_run)
    return repeat.repeat_ranker(context, first_run, second_run)


def run_calibration_item(
    item: str, run: str, channel: str | None, start: str | None, against: str | None
) -> None:
    """Run one calibration item and write its rows.

    Args:
        item: One of ITEMS.
        run: Name of this container run (configs/m10_calibration.yaml: runs).
        channel: The channel of --item recall.
        start: The start model of --item r6-predict: "recorded" or "calibration".
        against: For a repeat item: the EARLIER run that `run` is compared with.

    Raises:
        ValueError: On arguments that do not fit, or a repeat item whose two
            runs are the same or unknown.
    """
    check_arguments(item, channel, start, against)
    label = process_label(item, channel, start if against is None else f"against_{against}")
    context = start_calibration(run, label)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(context.base.num_threads))
    if against is not None:
        if against == run or against not in context.config.runs:
            raise ValueError(
                f"--against must be another run of {list(context.config.runs)}, got {against!r}"
            )
        finish_repeat(context, run_repeat_item(context, item, against, run), against, run)
        return
    if item in LIGHTGBM_ITEMS:
        outcome = run_lightgbm_side_item(context, item)
    elif item == ITEM_E2E_FEATURES:
        outcome = run_feature_item(context)
    else:
        outcome = run_torch_side_item(context, item, channel, start)
    finish_item(context, outcome)
