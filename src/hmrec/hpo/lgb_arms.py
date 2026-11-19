"""The two LightGBM arms of the M9 comparison: one training on one V fold (proposal §16.8).

    lambdarank   the locked Ranker exactly, with another seed. (The locked seed is
                 the M7 record in results/v_phase*.csv and is never trained again.)
    binary       the locked Ranker with ONE change: objective "binary" (pointwise
                 log loss on the 0/1 label) instead of "lambdarank". The metric of
                 the training log becomes binary_logloss. The ranking-only
                 parameters stay in the parameter list; LightGBM does not use them.

Everything else is the locked configuration, read through hmrec.ranker.locked
(the only entry to it): the same binary Dataset of the 4 weeks before the target
fold that M7 built (so the same training rows and the same bins), the same
number of threads, row-wise histograms, `deterministic`.

Seeds. The locked training passes ONE seed parameter to LightGBM, `seed`, and
LightGBM derives its sub-seeds (bagging_seed, feature_fraction_seed, ...) from
it. An arm's seed is set in exactly the same way: `seed = S` and nothing else.
What is random in the locked parameters is listed by random_parameters. Not
varied by the seed: the bin boundaries (the Dataset was built once, with the
locked seed) and the negative sampling (its own seed, the same for every arm).

Fold guard: the target must be a V fold and the training folds F / V folds
(hmrec.ranker.window, through hmrec.hpo.v_trial.plan_dataset), and the labels of
the target fold are read through load_m7_evaluator. A T fold raises before any
file is opened.

Outputs of a training (never replaced):
    artifacts/predictions/m9_lgb/{arm}_s{seed}/{fold}.parquet             Top-K
    artifacts/predictions/m9_lgb/{arm}_s{seed}/{fold}_per_query.parquet   per-query metrics
    artifacts/models/ranker/m9_lgb/{arm}_s{seed}/{fold}/model.txt
The result rows are written by the script afterwards (hmrec.hpo.lgb_arm_records).

This module is on the LightGBM side and must not import torch.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from hmrec.config import BaseConfig, RankerConfig, load_base_config
from hmrec.data.folds import FoldManifest, load_fold_manifest
from hmrec.eval.ranker_eval import load_m7_evaluator
from hmrec.hpo.lgb_arm_records import ARM_BINARY, ARM_LAMBDARANK, ARMS
from hmrec.hpo.v_trial import (
    TrialInputs,
    TrialOutcome,
    TrialSpec,
    plan_dataset,
    score_and_save,
    train_planned_model,
)
from hmrec.locks import FeaturesLock, load_features_lock
from hmrec.logging_utils import get_logger
from hmrec.paths import ProjectPaths
from hmrec.ranker.ablation import FULL
from hmrec.ranker.identity import model_config, ranker_config_id
from hmrec.ranker.locked import LockedRanker, load_locked_ranker
from hmrec.ranker.runtime import LightGBMRuntime
from hmrec.ranker.train import first_trees_sha256, load_model
from hmrec.ranker.window import require_m7_target_fold

LOGGER = get_logger("hpo.lgb_arms")

AREA = "m9_lgb"
OBJECTIVE = "objective"
METRIC = "metric"
BINARY_OBJECTIVE = "binary"
BINARY_METRIC = "binary_logloss"
M7_AREA = "v_phase"


@dataclass(frozen=True)
class ArmContext:
    """What a training of an arm works with.

    Attributes:
        paths: Project paths.
        base: configs/base.yaml (threads, memory limit).
        locked: The locked Ranker configuration (hmrec.ranker.locked).
        manifest: configs/fold_manifest.yaml.
        lock: configs/locks/features.lock.yaml.
        metrics: The metrics to compute, the selection metric first.
    """

    paths: ProjectPaths
    base: BaseConfig
    locked: LockedRanker
    manifest: FoldManifest
    lock: FeaturesLock
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class ArmSpec:
    """One training of an arm.

    Attributes:
        arm: "lambdarank" or "binary".
        seed: The LightGBM seed.
        target_fold: The V fold to predict.
        num_trees: Trees to train.
        evaluated_trees: The checkpoints to evaluate and save, ascending; each
            is a multiple of the locked checkpoint length and at most num_trees.
    """

    arm: str
    seed: int
    target_fold: str
    num_trees: int
    evaluated_trees: tuple[int, ...]


def load_arm_context(
    paths: ProjectPaths, runtime: LightGBMRuntime, metrics: tuple[str, ...]
) -> ArmContext:
    """Read the configs of an arm's training; the Ranker comes from its lock (and is checked)."""
    base = load_base_config(paths.base_config_file)
    return ArmContext(
        paths=paths,
        base=base,
        locked=load_locked_ranker(paths, base, runtime),
        manifest=load_fold_manifest(paths.fold_manifest_file),
        lock=load_features_lock(paths.features_lock_file),
        metrics=metrics,
    )


# ---------------------------------------------------------------- parameters


def random_parameters(train_params: dict[str, Any]) -> dict[str, Any]:
    """The locked LightGBM parameters that make training depend on the seed.

    LightGBM draws at random only if one of these is switched on:
        feature_fraction < 1          the features of every tree
        feature_fraction_bynode < 1   the features of every split
        bagging_fraction < 1 with bagging_freq > 0     the rows of every tree

    Returns:
        name -> value of the parameters that are switched on (empty = training
        is the same for every seed).
    """
    found: dict[str, Any] = {}
    for name in ("feature_fraction", "feature_fraction_bynode"):
        if float(train_params.get(name, 1.0)) < 1.0:
            found[name] = train_params[name]
    bagging = float(train_params.get("bagging_fraction", 1.0))
    if bagging < 1.0 and int(train_params.get("bagging_freq", 0)) > 0:
        found["bagging_fraction"] = train_params["bagging_fraction"]
        found["bagging_freq"] = train_params["bagging_freq"]
    return found


def arm_train_params(locked_train_params: dict[str, Any], arm: str) -> dict[str, Any]:
    """The locked train parameters of an arm: unchanged, or with the binary objective.

    Raises:
        ValueError: If the arm is unknown.
    """
    if arm == ARM_LAMBDARANK:
        return dict(locked_train_params)
    if arm == ARM_BINARY:
        # The ONLY change of the arm: the objective, and with it the metric of the
        # training log. Every other parameter keeps its locked value.
        return {**locked_train_params, OBJECTIVE: BINARY_OBJECTIVE, METRIC: BINARY_METRIC}
    raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")


def arm_ranker_config(locked: LockedRanker, arm: str, seed: int, num_trees: int) -> RankerConfig:
    """The locked RankerConfig with the arm's objective, seed and number of trees.

    Only `seed` is set (module docstring); the sampling seed stays the locked one.
    """
    return replace(
        locked.config,
        seed=seed,
        num_trees=num_trees,
        train_params=arm_train_params(locked.config.train_params, arm),
    )


def arm_config(context: ArmContext, spec: ArmSpec) -> dict[str, Any]:
    """Everything that defines the arm's model (hmrec.ranker.identity.model_config)."""
    locked = context.locked
    ranker = arm_ranker_config(locked, spec.arm, spec.seed, spec.num_trees)
    return model_config(ranker, locked.tuned, FULL, locked.histogram_mode)


def check_arm_spec(context: ArmContext, spec: ArmSpec) -> None:
    """Refuse a training the comparison does not allow, before anything is read.

    Raises:
        ValueError: If the target is not a V fold, the arm is unknown, the
            lambdarank arm is asked for the locked seed (that is the M7 record)
            or for another number of trees than the locked one, or a checkpoint
            is not valid.
    """
    require_m7_target_fold(context.manifest.get(spec.target_fold))
    locked = context.locked
    if spec.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {spec.arm!r}")
    if spec.seed <= 0 or spec.num_trees <= 0:
        raise ValueError("seed and num_trees must be > 0")
    if spec.arm == ARM_LAMBDARANK:
        if spec.seed == locked.config.seed:
            raise ValueError(
                f"lambdarank with seed {spec.seed} is the M7 record "
                "(results/v_phase*.csv); it is not trained again"
            )
        if spec.num_trees != locked.num_trees:
            raise ValueError(
                f"the lambdarank arm trains the locked {locked.num_trees} trees, "
                f"got {spec.num_trees}"
            )
    every = locked.config.checkpoint_every
    bad = [k for k in spec.evaluated_trees if k <= 0 or k % every != 0 or k > spec.num_trees]
    if (
        not spec.evaluated_trees
        or bad
        or list(spec.evaluated_trees) != sorted(set(spec.evaluated_trees))
    ):
        raise ValueError(
            f"evaluated_trees must be ascending multiples of {every} up to {spec.num_trees}, "
            f"got {spec.evaluated_trees}"
        )
    if spec.num_trees % every != 0:
        raise ValueError(f"num_trees must be a multiple of {every}, got {spec.num_trees}")


def arm_output_files(paths: ProjectPaths, spec: ArmSpec) -> tuple[Path, Path, Path]:
    """(predictions, per-query, model) files of a training; none of them may exist yet.

    Raises:
        FileExistsError: If one exists (an earlier run stopped before its row was written).
    """
    key = (spec.arm, spec.seed, spec.target_fold)
    files = (
        paths.m9_lgb_predictions_file(*key),
        paths.m9_lgb_per_query_file(*key),
        paths.m9_lgb_model_file(*key),
    )
    existing = [str(path) for path in files if path.exists()]
    if existing:
        raise FileExistsError(
            f"outputs of this training already exist: {existing}. An earlier run stopped before "
            "its row was written; move the files away by hand, then run again."
        )
    return files


# ---------------------------------------------------------------- one training


def differs_from_m7_model(context: ArmContext, spec: ArmSpec, model_file: Path) -> bool | None:
    """Whether the arm's first trees differ from the M7 model of the same fold (locked seed).

    A check that the seed really changes the trees (lambdarank arm). Returns None
    if the M7 model file is not there (e.g. in tests).
    """
    locked = context.locked
    m7_file = context.paths.ranker_model_file(M7_AREA, FULL, locked.config_id, spec.target_fold)
    if not m7_file.is_file():
        return None
    n_trees = min(spec.num_trees, locked.num_trees)
    m7_trees = first_trees_sha256(load_model(m7_file), n_trees)
    return first_trees_sha256(load_model(model_file), n_trees) != m7_trees


def run_arm_trial(context: ArmContext, spec: ArmSpec, git_commit: str) -> TrialOutcome:
    """Train one arm's model for one V fold, score the fold and evaluate the checkpoints.

    The Dataset is the one M7 built for the locked configuration: it is planned
    with the LOCKED config (so its record must match), and only the training
    parameters carry the arm's objective and seed.

    Args:
        context: Configuration.
        spec: The training.
        git_commit: HEAD of this run.

    Returns:
        The outcome (hmrec.hpo.v_trial.TrialOutcome); the caller writes the rows.
    """
    check_arm_spec(context, spec)  # the fold check comes first
    locked = context.locked
    files = arm_output_files(context.paths, spec)
    locked_inputs = TrialInputs(
        paths=context.paths,
        base=context.base,
        ranker=locked.config,
        manifest=context.manifest,
        lock=context.lock,
        metrics=context.metrics,
    )
    ratio = locked.tuned.negatives_per_positive
    plan = plan_dataset(locked_inputs, FULL, spec.target_fold, ratio, git_commit)
    arm_inputs = replace(
        locked_inputs, ranker=arm_ranker_config(locked, spec.arm, spec.seed, spec.num_trees)
    )
    trial = TrialSpec(
        area=AREA,
        experiment=FULL,
        step=spec.arm,
        target_fold=spec.target_fold,
        tuned=locked.tuned,
        num_trees=spec.num_trees,
        histogram_mode=locked.histogram_mode,
        evaluate=True,
    )
    booster, outcome, train_metric = train_planned_model(
        arm_inputs, trial, plan, arm_config(context, spec)
    )
    target = context.manifest.get(spec.target_fold)
    evaluator = load_m7_evaluator(target, context.paths.labels_file(spec.target_fold))
    outcome = score_and_save(
        arm_inputs,
        trial,
        plan.experiment,
        booster,
        outcome,
        train_metric,
        files,
        evaluator,
        spec.evaluated_trees,
    )
    if spec.arm == ARM_LAMBDARANK:
        LOGGER.info(
            "%s seed %d %s: first trees differ from the M7 model of the locked seed: %s",
            spec.arm,
            spec.seed,
            spec.target_fold,
            differs_from_m7_model(context, spec, files[2]),
        )
    return outcome


# ---------------------------------------------------------------- result rows


def metric_rows(
    spec: ArmSpec, outcome: TrialOutcome, metrics: tuple[str, ...], git_commit: str, at: str
) -> list[dict[str, str]]:
    """The rows of results/m9_lgb_metrics.csv of one training: one per evaluated checkpoint."""
    rows = []
    for row in outcome.metric_rows:
        train_metric = "" if row.train_metric is None else repr(row.train_metric)
        rows.append(
            {
                "arm": spec.arm,
                "seed": str(spec.seed),
                "config_id": outcome.config_id,
                "fold": row.fold,
                "num_trees": str(row.num_trees),
                "n_queries": str(row.n_queries),
                **{metric: repr(row.scores[metric]) for metric in metrics},
                "train_metric": train_metric,
                "git_commit": git_commit,
                "finished_at": at,
            }
        )
    return rows


def trial_row(
    context: ArmContext,
    spec: ArmSpec,
    outcome: TrialOutcome,
    runtime: LightGBMRuntime,
    measured: dict[str, float],
    stamp: dict[str, str],
) -> dict[str, str]:
    """The row of results/m9_lgb_trials.csv of one training.

    Args:
        context: Configuration.
        spec: The training.
        outcome: What it produced.
        runtime: LightGBM / OpenMP runtime of the process.
        measured: "total_seconds" and "process_max_rss_gb" of the process.
        stamp: "git_commit", "git_dirty", "finished_at".
    """
    return {
        "arm": spec.arm,
        "objective": str(outcome.config["train_params"][OBJECTIVE]),
        "seed": str(spec.seed),
        "config_id": outcome.config_id,
        "config_json": json.dumps(outcome.config, sort_keys=True),
        "fold": spec.target_fold,
        "train_folds": " ".join(outcome.train_folds),
        "num_trees": str(spec.num_trees),
        "evaluated_trees": " ".join(str(k) for k in spec.evaluated_trees),
        "n_train_rows": str(outcome.n_train_rows),
        "n_train_queries": str(outcome.n_train_queries),
        "n_scored_rows": str(outcome.n_scored_rows),
        "n_queries": str(outcome.n_queries),
        "load_seconds": repr(outcome.load_seconds),
        "train_seconds": repr(outcome.train_seconds),
        "seconds_per_tree": repr(outcome.seconds_per_tree),
        "score_seconds": repr(outcome.score_seconds),
        "total_seconds": repr(measured["total_seconds"]),
        "process_max_rss_gb": repr(measured["process_max_rss_gb"]),
        "model_sha256": outcome.model_sha256,
        "num_threads": str(context.locked.num_threads),
        **runtime.columns(),
        **stamp,
    }


def locked_config_id_of_arm(context: ArmContext, arm: str, seed: int) -> str:
    """The configuration id of an arm with a seed (for looking up its rows)."""
    spec = ArmSpec(arm, seed, "", context.locked.num_trees, ())
    return ranker_config_id(arm_config(context, spec))
