"""A tiny workspace for the tests of the M9 evaluation on T1-T4; not a test module itself.

Everything lives under a temporary folder and nothing of the project's data is
read (only the real fold manifest, the real features lock and the real configs).
Built with the helpers of deepfm_example:

    folds F4 V1 V2 V3 V4 T1 T2 T3 T4, each with 30 queries of 12 candidates: the
    wide table with its features.json, the label file and the query list (every
    customer has queries in several folds).
    Sample files with ratio 40 for F4 ... T3 (T4 trains no model), written with
    hmrec.ranker.sampling like M7 and M8 wrote theirs.
    What M7 and M8 left behind: the Dataset record of V4 from M7, the Dataset
    record of V4 from the M8 rehearsal, the four Dataset records of the M8 formal
    run, and results/t_phase_label_access.csv with the commit of that run.

The chosen configuration of the tests is the tiny configuration of deepfm_example
with the standardised input and the listwise loss, one epoch.
"""

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import polars as pl
from deepfm_example import (
    BASE,
    LOCK,
    MANIFEST,
    RATIO,
    make_wide,
    tiny_config,
    write_category_codes,
    write_fold,
)

from hmrec.config import DeepFMConfig, RankerSamplingConfig, TEvalConfig
from hmrec.csv_records import append_row
from hmrec.deepfm.process import record_trial
from hmrec.deepfm.records import deepfm_config_id
from hmrec.deepfm.trial import AREA_V_PHASE, TrialInputs, TrialResult, TrialSpec, run_trial
from hmrec.locks import file_sha256
from hmrec.memory import ProcessPeakMemory
from hmrec.paths import ProjectPaths, build_paths
from hmrec.ranker.ablation import build_experiment, wide_table_columns
from hmrec.ranker.sampling import SampleJob, sample_fold
from hmrec.run_records import RunStamp
from hmrec.t_eval.gate import PASSED, REHEARSAL_CHECK_COLUMNS, worker_permit
from hmrec.t_eval.plan import STAGE_FORMAL, TEvalContext, TrialKey
from hmrec.t_eval.trial import chosen_trial_inputs, run_t_eval_trial

COMMIT = "a" * 40  # the commit the tests' processes "run at"
OTHER_COMMIT = "b" * 40
M8_COMMIT = "c" * 40  # the commit of the M8 formal run in the workspace
NOW = "2026-11-20T12:00:00"
FOLDS = ("F4", "V1", "V2", "V3", "V4", "T1", "T2", "T3", "T4")
T_FOLDS = ("T1", "T2", "T3", "T4")
REHEARSAL_FOLD = "V4"
SEEDS = (42, 43)
N_CUSTOMERS = 20
M8_ACCESS_COLUMNS = (
    "event",
    "started_at",
    "git_commit",
    "git_dirty",
    "rehearsal_commit",
    "rehearsal_checked_at",
)
CHOSEN = {"numeric_input": "standardised", "loss": "listwise_softmax"}


def base_config() -> DeepFMConfig:
    """configs/deepfm.yaml made tiny, as the workspace's processes read it (one epoch)."""
    return tiny_config(epochs=1)


def chosen_config() -> DeepFMConfig:
    """The chosen configuration of the tests: the tiny one, standardised and listwise."""
    config = tiny_config(epochs=1, loss=CHOSEN["loss"])
    return replace(config, model=replace(config.model, numeric_input=CHOSEN["numeric_input"]))


def t_eval_config(**changes: object) -> TEvalConfig:
    """The evaluation's config for the tiny workspace; `changes` replace single values."""
    config = chosen_config()
    values = {
        "phase": "T",
        "target_folds": T_FOLDS,
        "seeds": SEEDS,
        "rehearsal_target_fold": REHEARSAL_FOLD,
        "arm": "deepfm_standardised_listwise",
        "numeric_input": config.model.numeric_input,
        "n_bins": config.n_bins,
        "loss": config.training.loss,
        "learning_rate": config.training.learning_rate,
        "embedding_dim": config.model.embedding_dim,
        "dropout": config.model.dropout,
        "components": config.model.components,
        "epochs": 1,
        "config_id": deepfm_config_id(config),
        "m8_record_arm": "lgb_lambdarank",
        "m8_record_seed": 42,
        "m8_record_num_trees": 500,
    }
    return TEvalConfig(**{**values, **changes})  # type: ignore[arg-type]  # test helper


def write_sample(paths: ProjectPaths, fold: str) -> None:
    """Draw a fold's sample file with the sampling code (ratio 40), as M7 / M8 did."""
    job = SampleJob(
        fold=MANIFEST.get(fold),
        features_dir=paths.features_dir(fold),
        labels_file=paths.labels_file(fold),
        experiment=build_experiment("full", LOCK),
        wide_columns=wide_table_columns(LOCK),
        sample_files={RATIO: paths.ranker_sample_file("full", fold, RATIO)},
        allowed_phases=("F", "V", "T"),
        git_commit=M8_COMMIT,
    )
    sample_fold(job, RankerSamplingConfig(seed=42, ratios=(RATIO,), row_group_rows=100))


def write_queries(paths: ProjectPaths, fold: str, wide: pl.DataFrame) -> None:
    """The fold's query list: query_id and a customer who also has queries in other folds."""
    query_ids = wide["query_id"].unique(maintain_order=True)
    queries = pl.DataFrame(
        {
            "query_id": query_ids,
            "customer_idx": (query_ids % N_CUSTOMERS).cast(pl.Int32),
        }
    )
    paths.queries_file(fold).parent.mkdir(parents=True, exist_ok=True)
    queries.write_parquet(paths.queries_file(fold))


def window_of(target: str) -> tuple[str, ...]:
    """The four folds right before `target` in the workspace."""
    position = FOLDS.index(target)
    return FOLDS[position - 4 : position]


def write_dataset_record(directory: Path, paths: ProjectPaths, target: str, commit: str) -> None:
    """A stand-in for a LightGBM Dataset record: everything the DeepFM code reads from it."""
    window = window_of(target)
    record = {
        "experiment": "full",
        "target_fold": target,
        "ratio": RATIO,
        "train_folds": list(window),
        "sample_sha256": {
            fold: file_sha256(paths.ranker_sample_file("full", fold, RATIO)) for fold in window
        },
        "git_commit": commit,
    }
    directory.mkdir(parents=True)
    (directory / "dataset.json").write_text(json.dumps(record), encoding="utf-8")


def write_m8_access(paths: ProjectPaths) -> None:
    """results/t_phase_label_access.csv with the row of the M8 formal run."""
    row = {
        "event": "first_t_label_read",
        "started_at": "2026-11-18T22:14:34",
        "git_commit": M8_COMMIT,
        "git_dirty": "False",
        "rehearsal_commit": M8_COMMIT,
        "rehearsal_checked_at": "2026-11-18T22:05:40",
    }
    append_row(paths.t_phase_label_access_file, M8_ACCESS_COLUMNS, row)


def build_t_eval_workspace(root: Path) -> TrialInputs:
    """Write the tiny workspace under `root`; returns the inputs a process reads from configs."""
    paths = build_paths(BASE, root)
    for seed, fold in enumerate(FOLDS):
        wide, labels = make_wide(fold, seed)
        write_fold(paths, fold, wide, labels)
        write_queries(paths, fold, wide)
        if fold != FOLDS[-1]:
            write_sample(paths, fold)
    write_category_codes(paths)
    # M7's Dataset record of V4 (the V-phase trial reads it) and the two kinds M8 left.
    write_dataset_record(paths.ranker_dataset_dir("full", "V4", RATIO), paths, "V4", "m7")
    write_dataset_record(
        paths.final_dataset_dir("m8_rehearsal", "full", "V4", RATIO), paths, "V4", M8_COMMIT
    )
    for target in T_FOLDS:
        write_dataset_record(
            paths.final_dataset_dir("t_phase", "full", target, RATIO), paths, target, M8_COMMIT
        )
    write_m8_access(paths)
    return TrialInputs(paths, BASE, base_config(), MANIFEST, LOCK)


def context_of(
    inputs: TrialInputs, commit: str = COMMIT, dirty: bool = False, **changes: object
) -> TEvalContext:
    """The context of a process at `commit`; `changes` replace values of the config."""
    return TEvalContext(inputs.paths, t_eval_config(**changes), MANIFEST, RunStamp(commit, dirty))


def record_rehearsal(context: TEvalContext, **changes: str) -> None:
    """Append a rehearsal check row; `changes` replace values of a passed row at COMMIT."""
    config = context.config
    row = {
        "status": PASSED,
        "target_fold": config.rehearsal_target_fold,
        "seeds": " ".join(str(seed) for seed in config.seeds),
        "config_id": config.config_id,
        "n_checks": "8",
        "failed_checks": "",
        "checks_json": "{}",
        "git_commit": COMMIT,
        "git_dirty": "False",
        "checked_at": "2026-11-20T11:00:00",
    }
    append_row(context.paths.m9_t_rehearsal_check_file, REHEARSAL_CHECK_COLUMNS, {**row, **changes})


# ---------------------------------------------------------------- trainings (torch side)

V_COMMIT = "e" * 40  # the commit the V-phase records of the workspace were "written at"
METRICS = ("map@12", "recall@12", "ndcg@12")
PEAK = ProcessPeakMemory(2 * 1024**3, 0)


def write_v_records(inputs: TrialInputs) -> None:
    """Train the chosen configuration for V4 with the V-phase code and record it, every seed.

    This is what the V study left behind for the rehearsal fold: the trial's files
    under artifacts/deepfm/v_phase/ and its rows in results/deepfm_*.csv.
    """
    config = t_eval_config()
    for seed in config.seeds:
        chosen = chosen_trial_inputs(inputs, config, seed)
        spec = TrialSpec(AREA_V_PHASE, REHEARSAL_FOLD, seed, config.epochs)
        result = run_trial(chosen, spec)
        record_trial(chosen, spec, result, 0.0, PEAK, RunStamp(V_COMMIT, False))


def run_and_record(inputs: TrialInputs, context: TEvalContext, key: TrialKey) -> TrialResult:
    """What scripts/m9_t_trial.py does for one training, in this process."""
    permit = worker_permit(context) if key.stage == STAGE_FORMAL else None
    chosen, spec, result = run_t_eval_trial(context, inputs, key, permit)
    paths = inputs.paths
    record_trial(
        chosen,
        spec,
        result,
        0.0,
        PEAK,
        context.stamp,
        paths.m9_t_trials_file,
        paths.m9_t_metrics_file,
    )
    return result


def in_process_launcher(
    inputs: TrialInputs, commit: str = COMMIT, calls: list[TrialKey] | None = None
) -> Callable[[TrialKey], None]:
    """A launcher that runs a training in this process, as a worker at `commit` would.

    Every launched key is appended to `calls` (if given) BEFORE it runs.
    """

    def launch(key: TrialKey) -> None:
        if calls is not None:
            calls.append(key)
        run_and_record(inputs, context_of(inputs, commit), key)

    return launch


def copy_workspace(inputs: TrialInputs, root: Path) -> TrialInputs:
    """A fresh copy of a workspace under `root` (which must not exist yet)."""
    shutil.copytree(inputs.paths.project_root, root)
    return replace(inputs, paths=build_paths(BASE, root))
