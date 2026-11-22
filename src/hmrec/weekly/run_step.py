"""Run ONE step of a tick in this process (called by scripts/m12_step.py).

The step's code is imported INSIDE the branch that runs it: the torch-side
modules (hmrec.weekly.recall_steps) and the LightGBM-side module
(hmrec.weekly.ranker_steps) must never be imported into the same process
(AGENTS.md §5), and a step that needs neither loads neither.

If the step raises, the reason is written to runs/{run_id}/failures/{step}.txt
before the error goes on: the driver puts it into the run manifest.
"""

import time
from datetime import date

from hmrec.logging_utils import get_logger
from hmrec.run_manifest import MetricRecord
from hmrec.weekly.context import OutputFile, TickContext, finish_step, start_step
from hmrec.weekly.steps import (
    DATASET,
    EVALUATE,
    FEATURES,
    FEATURES_SHARED,
    FINE_TUNE,
    INGEST,
    MERGE_SCORES,
    PUBLISH,
    QUERIES,
    RECALL,
    SAMPLE,
    SCORE,
    SHARD_FEATURES,
    SHARD_SCORE,
    SIDE_LIGHTGBM,
    SIDE_TORCH,
    TRAIN,
    Step,
)

LOGGER = get_logger("weekly.run_step")

StepOutcome = tuple[str, list[MetricRecord], list[OutputFile]]


def run_plain_step(context: TickContext, step: Step) -> StepOutcome:
    """A step that loads neither torch nor LightGBM."""
    if step.kind == INGEST:
        from hmrec.weekly.data_steps import run_ingest

        return run_ingest(context)
    if step.kind == QUERIES:
        from hmrec.weekly.data_steps import run_queries

        return run_queries(context)
    if step.kind == FEATURES:
        from hmrec.weekly.feature_steps import run_features

        return run_features(context)
    if step.kind == FEATURES_SHARED:
        from hmrec.weekly.feature_steps import run_features_shared

        return run_features_shared(context)
    if step.kind == SHARD_FEATURES:
        from hmrec.weekly.feature_steps import run_shard_features

        return run_shard_features(context, step)
    if step.kind == MERGE_SCORES:
        from hmrec.weekly.outputs import run_merge_scores

        return run_merge_scores(context)
    if step.kind == PUBLISH:
        from hmrec.weekly.outputs import run_publish

        return run_publish(context)
    raise ValueError(f"step {step.name}: unknown kind {step.kind!r}")


def run_torch_step(context: TickContext, step: Step) -> StepOutcome:
    """A recall step (torch side)."""
    from hmrec.weekly.recall_steps import run_fine_tune, run_recall

    if step.kind == FINE_TUNE:
        return run_fine_tune(context)
    if step.kind == RECALL:
        return run_recall(context, step)
    raise ValueError(f"step {step.name}: {step.kind!r} is not a torch-side step")


def run_lightgbm_step(context: TickContext, step: Step) -> StepOutcome:
    """A Ranker step (LightGBM side)."""
    from hmrec.weekly import ranker_steps

    setup = ranker_steps.load_ranker_setup(context)
    if step.kind == EVALUATE:
        return ranker_steps.run_evaluate(context, setup)
    if step.kind == SAMPLE:
        return ranker_steps.run_sample(context, setup)
    if step.kind == DATASET:
        return ranker_steps.run_dataset(context, setup)
    if step.kind == TRAIN:
        return ranker_steps.run_train(context, setup)
    if step.kind == SCORE:
        return ranker_steps.run_score(context, setup)
    if step.kind == SHARD_SCORE:
        return ranker_steps.run_shard_score(context, setup, step)
    raise ValueError(f"step {step.name}: {step.kind!r} is not a LightGBM-side step")


def dispatch(context: TickContext, step: Step) -> StepOutcome:
    """Run the step's code on its side."""
    if step.side == SIDE_TORCH:
        return run_torch_step(context, step)
    if step.side == SIDE_LIGHTGBM:
        return run_lightgbm_step(context, step)
    return run_plain_step(context, step)


def run_one_step(anchor: date, mode: str, run_id: str, step_name: str) -> None:
    """Start, run and finish one step of a tick in this process.

    Args:
        anchor: The tick's anchor.
        mode: "eval" or "serve".
        run_id: The run of the tick.
        step_name: The step, e.g. "O1_recall_r4_itemcf".

    Raises:
        Exception: Whatever the step raised; the reason is written to the run's
            failures folder first.
    """
    started = time.perf_counter()
    context, step = start_step(anchor, mode, run_id, step_name)
    try:
        detail, metrics, files = dispatch(context, step)
        finish_step(context, step, started, detail, metrics, files)
    except Exception as error:
        failure_file = context.layout.step_failure_file(run_id, step_name)
        failure_file.parent.mkdir(parents=True, exist_ok=True)
        failure_file.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        LOGGER.exception("step %s failed", step_name)
        raise
