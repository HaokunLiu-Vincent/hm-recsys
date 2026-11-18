"""A tiny workspace for the tests of the final evaluation (M8); not a test module itself.

Everything lives under a temporary folder: configs, features, labels, candidates,
samples, models and result tables. Nothing of the project's data is read.

    write_configs         configs/ of the workspace: copies of the real fold manifest,
                          m8_final.yaml, m6_candidates.yaml and the three recall /
                          feature lock files, a tiny base.yaml and ranker.yaml, and a
                          tiny ranker.lock.yaml that is consistent with them (its
                          config_id, reference hashes and runtime are computed here).
    write_folds           learnable folds (tests/unit/ranker_example.py) for F4,
                          V1 ... V4 and the four T folds: wide table, labels, query
                          list and the six candidate files.
    write_m7_reference    what M7 and M6 left behind for the rehearsal fold V4, made
                          with the M7 / M6 code itself: sample files of F4 V1 V2 V3,
                          the V4 models (full: 20 trees; minus_r6: 10 trees), their
                          rows in results/v_phase*.csv, the RRF baseline row and the
                          monitoring rows in results/m6_candidates.csv.
    build_final_workspace all three, and the FinalContext of a process at commit
                          COMMIT with a clean tree.
    copy_workspace        a fresh copy of such a workspace. The workspace is built once
                          per test process (about a second) and copied for every test
                          (a few milliseconds), optionally after a passed rehearsal.

The tiny lock keeps 10 trees of a model M7 trained to 20 trees, like the real
lock keeps 500 of 1000.
"""

import atexit
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from ranker_example import (
    LOCK,
    MANIFEST,
    METRICS,
    REPO,
    learnable_fold,
    tiny_ranker_raw,
    write_features,
    write_labels,
)
from v_phase_example import parse_arguments, run_and_record_trial

from hmrec.config import load_base_config, load_yaml, parse_ranker_config
from hmrec.csv_records import append_row
from hmrec.eval.monitor import MONITOR_COLUMNS, load_label_evaluator, monitor_rows
from hmrec.final.context import FinalContext, load_final_context
from hmrec.final.monitor import read_fold_candidates
from hmrec.final.process import run_worker
from hmrec.final.run import StepRequest, run_rehearsal
from hmrec.fusion.union import RANK_COLUMNS
from hmrec.hpo.v_phase import sample_job
from hmrec.hpo.v_records import AREA_V_PHASE, append_metric_row, results_files
from hmrec.hpo.v_trial import (
    TrialInputs,
    TrialSpec,
    build_and_save_dataset,
    plan_dataset,
    rrf_baseline_row,
)
from hmrec.locks import ALL_CHANNELS, file_sha256
from hmrec.paths import ProjectPaths, build_paths
from hmrec.ranker.identity import model_config, ranker_config_id
from hmrec.ranker.runtime import read_lightgbm_runtime
from hmrec.ranker.sampling import sample_fold
from hmrec.run_records import RunStamp

COMMIT = "a" * 40  # the commit the tests' processes "run at"
OTHER_COMMIT = "b" * 40
NOW = "2026-11-18T12:00:00"
# First query id of every fold: far apart, growing with the fold's position in time.
FOLD_QUERY_IDS = {
    "F4": 4000,
    "V1": 5000,
    "V2": 6000,
    "V3": 7000,
    "V4": 8000,
    "T1": 9000,
    "T2": 10000,
    "T3": 11000,
    "T4": 12000,
}
REHEARSAL_FOLD = "V4"
T_FOLDS = ("T1", "T2", "T3", "T4")
N_QUERIES = 40
LOCKED_TREES = 10  # the tiny lock's number of trees; M7's tiny model has 20
COPIED_CONFIGS = (
    "fold_manifest.yaml",
    "m8_final.yaml",
    "m6_candidates.yaml",
    "locks/features.lock.yaml",
    "locks/channels.lock.yaml",
    "locks/twotower.lock.yaml",
)
TINY_BASE = {
    "paths": {"data_root": "data", "artifacts_root": "artifacts", "results_root": "results"},
    "seed": 0,
    "num_threads": 2,
    "memory": {"limit_gb": 8.0, "sample_interval_seconds": 0.1},
    "chunking": {
        "query_chunk_size": 10,
        "item_chunk_size": 10,
        "dense_score_query_chunk_size": 10,
        "twotower_score_query_chunk_size": 10,
        "pair_chunk_size": 10,
    },
}


def _dump(path: Path, raw: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")


def tiny_lock_raw(configs: Path) -> dict[str, Any]:
    """The raw content of a ranker.lock.yaml that fits the tiny ranker.yaml of `configs`.

    Starts from the real lock file and replaces: the tuned values (4 leaves,
    min_data_in_leaf 5, 2 negatives per positive), 10 trees in segments of 10,
    col_wise, the tiny LightGBM parameters, the runtime of THIS process, the
    hashes of the lock files in `configs`, and the config_id these settings give.
    """
    ranker = parse_ranker_config(load_yaml(configs / "ranker.yaml"))
    runtime = read_lightgbm_runtime()
    raw = load_yaml(REPO / "configs" / "locks" / "ranker.lock.yaml")
    raw["model"].update(
        num_leaves=ranker.tuned_defaults.num_leaves,
        min_data_in_leaf=ranker.tuned_defaults.min_data_in_leaf,
        num_trees=LOCKED_TREES,
        seed=ranker.seed,
        histogram_mode=ranker.histogram.mode,
        dataset_params=dict(ranker.dataset_params),
        train_params=dict(ranker.train_params),
    )
    raw["sampling"].update(
        seed=ranker.sampling.seed,
        negatives_per_positive=ranker.tuned_defaults.negatives_per_positive,
        row_group_rows=ranker.sampling.row_group_rows,
    )
    raw["scoring"]["checkpoint_every"] = ranker.checkpoint_every
    raw["runtime"] = {
        "lightgbm_version": runtime.lightgbm_version,
        "libomp_version": runtime.libomp_version,
        "num_threads": TINY_BASE["num_threads"],
    }
    for reference in raw["references"].values():
        reference["sha256"] = file_sha256(configs.parent / reference["path"])
    settings = model_config(ranker, ranker.tuned_defaults, "full", ranker.histogram.mode)
    raw["selection"]["config_id"] = ranker_config_id(settings)
    raw["selection"]["rule_selected_config_id"] = raw["selection"]["config_id"]
    return raw


def write_configs(root: Path) -> ProjectPaths:
    """Write the workspace's configs/ (module docstring) and return its paths."""
    configs = root / "configs"
    for name in COPIED_CONFIGS:
        (configs / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "configs" / name, configs / name)
    _dump(configs / "base.yaml", TINY_BASE)
    _dump(configs / "ranker.yaml", tiny_ranker_raw())
    _dump(configs / "locks" / "ranker.lock.yaml", tiny_lock_raw(configs))
    return build_paths(load_base_config(configs / "base.yaml"), root)


def write_candidates(paths: ProjectPaths, fold: str, wide: pl.DataFrame) -> None:
    """Write the fold's query list and six candidate files from its wide table.

    A channel returned a row if its rank column is not NaN. Its candidates get
    rank 1, 2, ... inside each query, in the order of that rank column.
    """
    queries = wide.select("query_id").unique().sort("query_id")
    queries = queries.with_columns(pl.lit(fold).alias("fold"))
    paths.queries_file(fold).parent.mkdir(parents=True, exist_ok=True)
    queries.write_parquet(paths.queries_file(fold))
    for channel, rank_column in zip(ALL_CHANNELS, RANK_COLUMNS, strict=True):
        returned = wide.filter(pl.col(rank_column).is_not_nan())
        rank = pl.col(rank_column).rank("ordinal").over("query_id").cast(pl.Int32).alias("rank")
        candidates = returned.select("query_id", "article_idx", rank).sort("query_id", "rank")
        paths.candidates_file(fold, channel).parent.mkdir(parents=True, exist_ok=True)
        candidates.write_parquet(paths.candidates_file(fold, channel))


def write_folds(paths: ProjectPaths, folds: Sequence[str] = tuple(FOLD_QUERY_IDS)) -> None:
    """Write every fold's features (two part files), labels, queries and candidates."""
    for seed, fold in enumerate(folds):
        first_query_id = FOLD_QUERY_IDS[fold]
        wide, labels = learnable_fold(first_query_id, N_QUERIES, seed)
        split_at = first_query_id + N_QUERIES // 2
        parts = [
            wide.filter(pl.col("query_id") < split_at),
            wide.filter(pl.col("query_id") >= split_at),
        ]
        write_features(paths.features_dir(fold), fold, parts)
        write_labels(paths.labels_file(fold), fold, labels)
        write_candidates(paths, fold, wide)


def m7_inputs(paths: ProjectPaths) -> TrialInputs:
    """The inputs M7 worked with in the workspace: configs/ranker.yaml itself (20 trees)."""
    return TrialInputs(
        paths=paths,
        base=load_base_config(paths.base_config_file),
        ranker=parse_ranker_config(load_yaml(paths.ranker_config_file)),
        manifest=MANIFEST,
        lock=LOCK,
        metrics=METRICS,
    )


def write_m7_reference(paths: ProjectPaths) -> None:
    """Leave behind what M7 and M6 recorded for the rehearsal fold (module docstring)."""
    inputs = m7_inputs(paths)
    ranker, tuned = inputs.ranker, inputs.ranker.tuned_defaults
    ratio, mode = tuned.negatives_per_positive, ranker.histogram.mode
    for fold in ("F4", "V1", "V2", "V3"):
        for experiment, ratios in (("full", ranker.sampling.ratios), ("minus_r6", (ratio,))):
            job = sample_job(inputs, fold, experiment, ratios, "m7")
            sample_fold(job, ranker.sampling)
    # M7 trained the full model to num_trees (20) and the ablation to the selected 10.
    for experiment, num_trees in (("full", ranker.num_trees), ("minus_r6", LOCKED_TREES)):
        plan = plan_dataset(inputs, experiment, REHEARSAL_FOLD, ratio, "m7")
        build_and_save_dataset(plan, ranker.sampling.row_group_rows)
        spec = TrialSpec(
            AREA_V_PHASE, experiment, "m7", REHEARSAL_FOLD, tuned, num_trees, mode, True
        )
        run_and_record_trial(inputs, spec)
    metrics_file = results_files(paths, AREA_V_PHASE)[1]
    baseline = rrf_baseline_row(inputs, REHEARSAL_FOLD)
    append_metric_row(metrics_file, baseline, METRICS, "m7", "t")


def write_m6_monitor_rows(context: FinalContext) -> None:
    """Append the rehearsal fold's monitoring rows as M6 recorded them (the M6 function)."""
    paths = context.paths
    evaluator = load_label_evaluator(REHEARSAL_FOLD, "V", paths.labels_file(REHEARSAL_FOLD))
    rows = monitor_rows(
        read_fold_candidates(context, REHEARSAL_FOLD),
        evaluator,
        [],
        context.monitor,
        context.base.chunking.query_chunk_size,
        RunStamp("m6", False),
    )
    for row in rows:
        append_row(paths.m6_candidates_results_file, MONITOR_COLUMNS, row)


def load_context(paths: ProjectPaths, commit: str = COMMIT, dirty: bool = False) -> FinalContext:
    """The FinalContext of a process at `commit`; the code "changed" for any other commit."""
    context = load_final_context(paths, RunStamp(commit, dirty), read_lightgbm_runtime())
    return replace(context, code_changed=lambda recorded: recorded != commit)


def build_final_workspace(root: Path) -> FinalContext:
    """Configs, folds and the M7 / M6 reference under `root`; returns the context at COMMIT."""
    paths = write_configs(root)
    write_folds(paths)
    write_m7_reference(paths)
    context = load_context(paths)
    write_m6_monitor_rows(context)
    return context


# Workspaces built once per test process: "built" and "rehearsed" -> folder.
_TEMPLATES: dict[str, Path] = {}


def _template(rehearsed: bool) -> Path:
    """The folder of the template workspace, built on first use and removed at exit."""
    key = "rehearsed" if rehearsed else "built"
    if key not in _TEMPLATES:
        folder = Path(tempfile.mkdtemp(prefix=f"hmrec_final_{key}_"))
        atexit.register(shutil.rmtree, folder, ignore_errors=True)
        context = build_final_workspace(folder)
        if rehearsed:
            run_rehearsal(context, InProcessWorker(context), folder, NOW)
        _TEMPLATES[key] = folder
    return _TEMPLATES[key]


def copy_workspace(root: Path, rehearsed: bool = False) -> FinalContext:
    """A fresh copy of the tiny workspace under `root`; returns its context at COMMIT.

    Args:
        root: An empty folder of the test (tmp_path).
        rehearsed: True = the copy already holds a PASSED rehearsal (its trials,
            models and the row in results/m8_rehearsal_check.csv). The formal run
            has not been started in it.

    Nothing in the workspace stores an absolute path, so a copy behaves exactly
    like a workspace built in place.
    """
    shutil.copytree(_template(rehearsed), root, dirs_exist_ok=True)
    base = load_base_config(root / "configs" / "base.yaml")
    return load_context(build_paths(base, root))


class InProcessWorker:
    """Stands in for hmrec.pipeline.run_script: runs a step of scripts/m8_step.py in this process.

    Attributes:
        context: The context every step runs with.
        calls: The steps that were started, in order.
    """

    def __init__(self, context: FinalContext) -> None:
        """Create a launcher whose steps run with `context`."""
        self.context = context
        self.calls: list[StepRequest] = []

    def __call__(self, script: Path, arguments: Sequence[str]) -> float:
        """Run one step with hmrec.final.process.run_worker."""
        args = parse_arguments(arguments)
        request = StepRequest(
            args["step"][0], args["mode"][0], args["fold"][0], args["experiment"][0]
        )
        self.calls.append(request)
        run_worker(self.context, request, time.perf_counter())
        return 0.0
