"""The steps of one tick of the weekly pipeline, in the order they run (docs/mlops.md §6.1).

A tick at anchor `a` closes the week before `a` and opens the week of `a`:

    close   C1 ingest            transactions before `a` (the tick's clock)
            C2 evaluate          last week's predictions against last week's purchases
            C3 sample            the Ranker training sample of last week
            C4 fine-tune R6      the Two-Tower chain, one week further
    open    O1 queries, recall   who is predicted for; the six channels
            O2 features          the 71 features
            O3 dataset, train    LambdaMART on the last 4 closed weeks
            O4 score             the Top-K of every customer
            O5 publish           the prediction set; then the `current` pointer

Every step is ONE process and writes ONE folder (Step.step_dir under the week's
folder). The driver (hmrec.weekly.driver) starts the processes; a step process
(scripts/m12_step.py) finds its Step again by name with step_by_name.

`serve` mode predicts for every customer, in shards of customers (§6.2, §8.4):
the features of a shard are built, scored and then removed, shard by shard, so
only one shard's features are on disk at a time.

Sides: torch and LightGBM never share a process (AGENTS.md §5). A step names
the side its code is on; "none" loads neither.

This module imports neither torch nor lightgbm.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from hmrec.locks import ALL_CHANNELS
from hmrec.paths import (
    DATASET_DIR,
    EVALUATION_DIR,
    FEATURES_DIR,
    FEATURES_SHARED_DIR,
    INGEST_DIR,
    MODEL_DIR,
    QUERIES_DIR,
    SAMPLE_DIR,
    SCORES_DIR,
    SHARD_TOP_DIR,
    TWOTOWER_DIR,
    StateLayout,
    recall_step_dir,
    shard_name,
    shard_step_dir,
)
from hmrec.run_manifest import EVAL_MODE, MODES, SERVE_MODE
from hmrec.weekly.calendar import Week

SIDE_NONE = "none"
SIDE_TORCH = "torch"
SIDE_LIGHTGBM = "lightgbm"

INGEST = "ingest"
EVALUATE = "evaluate"
SAMPLE = "sample"
FINE_TUNE = "fine_tune"
QUERIES = "queries"
RECALL = "recall"
FEATURES = "features"
FEATURES_SHARED = "features_shared"
SHARD_FEATURES = "shard_features"
DATASET = "dataset"
TRAIN = "train"
SCORE = "score"
SHARD_SCORE = "shard_score"
MERGE_SCORES = "merge_scores"
PUBLISH = "publish"

C1_INGEST = "C1_ingest"
C2_EVALUATE = "C2_evaluate"
C3_SAMPLE = "C3_sample"
C4_FINE_TUNE = "C4_fine_tune_r6"
O1_QUERIES = "O1_queries"
O1_RECALL_PREFIX = "O1_recall_"
O2_FEATURES = "O2_features"
O2_FEATURES_SHARED = "O2_features_shared"
O2_SHARD_PREFIX = "O2_features_shard_"
O3_DATASET = "O3_dataset"
O3_TRAIN = "O3_train"
O4_SCORE = "O4_score"
O4_SHARD_PREFIX = "O4_score_shard_"
O5_PUBLISH = "O5_publish"

SHARD_STEP_NAME = re.compile(r"(O2_features_shard_|O4_score_shard_)(\d{3})")
# The record every step process writes into its output folder as its last action.
STEP_RECORD_FILE = "step.json"


@dataclass(frozen=True)
class Step:
    """One step of a tick.

    Attributes:
        name: The step's name in the run manifest, e.g. "O1_recall_r4_itemcf".
        kind: What the step does (one of the kinds above); the step process
            dispatches on it.
        side: "torch", "lightgbm" or "none".
        week: The week whose folder holds the step's output: the closed week
            for C2-C4, the tick's week for everything else.
        step_dir: The output folder under the week's folder; "" for O5, whose
            output is the prediction set.
        channel: The recall channel of a recall step, else "".
        shard: The shard of a shard step, else None.
    """

    name: str
    kind: str
    side: str
    week: Week
    step_dir: str
    channel: str = ""
    shard: int | None = None


def check_mode(mode: str) -> None:
    """Fail unless `mode` is "eval" or "serve"."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def closing_steps(tick: Week, closing: Week) -> list[Step]:
    """C1 for the tick's week, then C2-C4 for the week that is closed."""
    return [
        Step(C1_INGEST, INGEST, SIDE_NONE, tick, INGEST_DIR),
        Step(C2_EVALUATE, EVALUATE, SIDE_LIGHTGBM, closing, EVALUATION_DIR),
        Step(C3_SAMPLE, SAMPLE, SIDE_LIGHTGBM, closing, SAMPLE_DIR),
        Step(C4_FINE_TUNE, FINE_TUNE, SIDE_TORCH, closing, TWOTOWER_DIR),
    ]


def recall_steps(tick: Week) -> list[Step]:
    """The query list, then one step per channel R1 ... R6."""
    steps = [Step(O1_QUERIES, QUERIES, SIDE_NONE, tick, QUERIES_DIR)]
    steps += [
        Step(
            f"{O1_RECALL_PREFIX}{channel}",
            RECALL,
            SIDE_TORCH,
            tick,
            recall_step_dir(channel),
            channel,
        )
        for channel in ALL_CHANNELS
    ]
    return steps


def ranker_training_steps(tick: Week) -> list[Step]:
    """The Dataset of the 4 closed weeks, then the model."""
    return [
        Step(O3_DATASET, DATASET, SIDE_LIGHTGBM, tick, DATASET_DIR),
        Step(O3_TRAIN, TRAIN, SIDE_LIGHTGBM, tick, MODEL_DIR),
    ]


def shard_features_step(tick: Week, shard: int) -> Step:
    """The features of one shard (`serve` mode)."""
    name = f"{O2_SHARD_PREFIX}{shard_name(shard)}"
    return Step(
        name, SHARD_FEATURES, SIDE_NONE, tick, shard_step_dir(shard, FEATURES_DIR), shard=shard
    )


def shard_score_step(tick: Week, shard: int) -> Step:
    """The Top-K of one shard (`serve` mode)."""
    name = f"{O4_SHARD_PREFIX}{shard_name(shard)}"
    return Step(
        name, SHARD_SCORE, SIDE_LIGHTGBM, tick, shard_step_dir(shard, SHARD_TOP_DIR), shard=shard
    )


def publish_step(tick: Week) -> Step:
    """O5: the prediction set (its folder is not under the week's folder)."""
    return Step(O5_PUBLISH, PUBLISH, SIDE_NONE, tick, "")


def steps_before_shards(tick: Week, closing: Week, mode: str) -> list[Step]:
    """Every step up to the point where `serve` mode needs the number of shards.

    eval:   C1-C4, O1, O2 features, O3, O4 score, O5 (the whole tick).
    serve:  C1-C4, O1, O2 shared feature tables, O3. The shard steps follow
            (steps_of_shards), once the query list says how many shards there are.
    """
    check_mode(mode)
    steps = closing_steps(tick, closing) + recall_steps(tick)
    if mode == EVAL_MODE:
        steps.append(Step(O2_FEATURES, FEATURES, SIDE_NONE, tick, FEATURES_DIR))
        steps += ranker_training_steps(tick)
        steps.append(Step(O4_SCORE, SCORE, SIDE_LIGHTGBM, tick, SCORES_DIR))
        steps.append(publish_step(tick))
        return steps
    steps.append(Step(O2_FEATURES_SHARED, FEATURES_SHARED, SIDE_NONE, tick, FEATURES_SHARED_DIR))
    steps += ranker_training_steps(tick)
    return steps


def steps_of_shards(tick: Week, n_shards: int) -> list[Step]:
    """`serve` mode: features and score of every shard in turn, the merge, then O5.

    Raises:
        ValueError: If n_shards <= 0.
    """
    if n_shards <= 0:
        raise ValueError(f"a serve tick has at least one shard, got {n_shards}")
    steps: list[Step] = []
    for shard in range(n_shards):
        steps.append(shard_features_step(tick, shard))
        steps.append(shard_score_step(tick, shard))
    steps.append(Step(O4_SCORE, MERGE_SCORES, SIDE_NONE, tick, SCORES_DIR))
    steps.append(publish_step(tick))
    return steps


def step_by_name(name: str, tick: Week, closing: Week, mode: str) -> Step:
    """Find a step of a tick by its name (used by the step process).

    Raises:
        ValueError: If the tick has no step of that name in that mode.
    """
    check_mode(mode)
    shard_match = SHARD_STEP_NAME.fullmatch(name)
    if shard_match is not None and mode == SERVE_MODE:
        shard = int(shard_match.group(2))
        if shard_match.group(1) == O2_SHARD_PREFIX:
            return shard_features_step(tick, shard)
        return shard_score_step(tick, shard)
    known = steps_before_shards(tick, closing, mode)
    if mode == SERVE_MODE:
        # One shard is enough to list the two steps after the shards (merge, publish).
        known += [step for step in steps_of_shards(tick, 1) if step.shard is None]
    for step in known:
        if step.name == name:
            return step
    raise ValueError(f"a {mode} tick has no step {name!r}; its steps are {[s.name for s in known]}")


def final_dir(layout: StateLayout, step: Step) -> Path:
    """Where a step's finished output lies."""
    if step.kind == PUBLISH:
        return layout.prediction_set_dir(step.week.anchor)
    return layout.step_dir(step.week.anchor, step.step_dir)


def staging_dir(layout: StateLayout, step: Step) -> Path:
    """Where a step writes while it runs."""
    if step.kind == PUBLISH:
        return layout.prediction_staging_dir(step.week.anchor)
    return layout.staging_dir(step.week.anchor, step.step_dir)
