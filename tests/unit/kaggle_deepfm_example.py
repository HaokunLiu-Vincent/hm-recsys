"""A tiny workspace for the tests of the DeepFM Kaggle submission; not a test module itself.

Built on the workspace of t_eval_example (folds F4 ... T4 with sample files and the
Dataset records M8 left). Added here:

    T4's query list with ONE query per customer (customers 0 ... 29), as a Kaggle
    run has it, and the stage-2 record of T4 with seed 42 (a formal trial of the
    T evaluation: its saved Top-12 and its row in results/m9_t_metrics.csv).

    What the LightGBM submission left behind for its two runs:
      rehearsal    35 customers (the 30 of T4 plus 5 others) in 2 shards: the query
                   list, the wide table of every shard (the T4 customers have
                   exactly the rows of T4's wide table, under the run's own query
                   ids), and the Dataset record of V4 T1 T2 T3
      submission   40 customers in 2 shards: query list, wide tables, the sample
                   file of T4 that run drew itself, and the Dataset record of
                   T1 T2 T3 T4

Everything is under a temporary folder; nothing of the project's data is read.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
from deepfm_example import BASE, CATEGORICAL, LOCK, MANIFEST, RATIO, REPO
from t_eval_example import (
    COMMIT,
    M8_COMMIT,
    NOW,
    build_t_eval_workspace,
    context_of,
    record_rehearsal,
    run_and_record,
    t_eval_config,
)

from hmrec.config import (
    KaggleDeepFMConfig,
    RankerSamplingConfig,
    load_kaggle_config,
    load_preprocess_config,
)
from hmrec.deepfm.trial import TrialInputs
from hmrec.feature_files import FeatureFiles, describe_part, write_feature_files
from hmrec.kaggle.context import KaggleContext
from hmrec.kaggle.plan import (
    RUN_REHEARSAL,
    RUN_SUBMISSION,
    build_queries,
    target_week,
    write_queries,
)
from hmrec.kaggle_deepfm.context import DeepFMKaggleContext
from hmrec.locks import file_sha256
from hmrec.paths import ProjectPaths
from hmrec.ranker.ablation import build_experiment, wide_table_columns
from hmrec.ranker.sampling import SampleJob, sample_fold
from hmrec.run_records import RunStamp
from hmrec.t_eval.gate import begin_formal_run
from hmrec.t_eval.plan import STAGE_FORMAL, TrialKey

KAGGLE = load_kaggle_config(REPO / "configs" / "kaggle_submission.yaml")
PREPROCESS = load_preprocess_config(REPO / "configs" / "preprocess.yaml")
SHARD_CUSTOMERS = 20
N_FOLD_CUSTOMERS = 30  # the T4 queries of the workspace
N_REHEARSAL_CUSTOMERS = 35
N_SUBMISSION_CUSTOMERS = 40
QUERY_ID_BASE = KAGGLE.query_id_base
SEED = 42
REFERENCE_AREA = "m9_t"
SCRIPT = "scripts/deepfm_kaggle_step.py --step train"


def kaggle_context(
    inputs: TrialInputs, run: str, commit: str = COMMIT, dirty: bool = False
) -> DeepFMKaggleContext:
    """The context of a process of `run` on the tiny workspace."""
    config = replace(KAGGLE, shard_customers=SHARD_CUSTOMERS)
    kaggle = KaggleContext(
        run=run,
        paths=inputs.paths,
        base=BASE,
        config=config,
        manifest=MANIFEST,
        preprocess=PREPROCESS,
        target=target_week(run, config, MANIFEST),
        stamp=RunStamp(commit, dirty),
    )
    chosen = t_eval_config()
    settings = KaggleDeepFMConfig(
        seed=SEED,
        config_id=chosen.config_id,
        reference_area=REFERENCE_AREA,
        lightgbm_private_score=0.036,
        lightgbm_public_score=0.03548,
        same_below_relative_difference=0.01,
    )
    return DeepFMKaggleContext(kaggle, settings, chosen)


def fold_wide(paths: ProjectPaths, fold: str) -> pl.DataFrame:
    """A fold's whole wide table, sorted by query_id."""
    directory = paths.features_dir(fold)
    parts = sorted(directory.glob("part-*.parquet"))
    return pl.concat([pl.read_parquet(part) for part in parts]).sort(
        "query_id", maintain_order=True
    )


def renumber(wide: pl.DataFrame, first_customer: int, n_queries: int) -> pl.DataFrame:
    """The rows of the first `n_queries` queries of a wide table under a Kaggle run's query ids.

    The k-th query of the table becomes the query of customer first_customer + k:
    query_id = QUERY_ID_BASE + first_customer + k.
    """
    first_query = int(wide["query_id"][0])
    kept = wide.filter(pl.col("query_id") < first_query + n_queries)
    new_id = pl.col("query_id") - first_query + QUERY_ID_BASE + first_customer
    return kept.with_columns(new_id.cast(pl.Int32).alias("query_id"))


def write_wide(directory: Path, name: str, wide: pl.DataFrame) -> None:
    """Write a wide table as two part files with its features.json."""
    directory.mkdir(parents=True)
    middle = int(wide["query_id"].unique().sort()[wide["query_id"].n_unique() // 2])
    pieces = [wide.filter(pl.col("query_id") < middle), wide.filter(pl.col("query_id") >= middle)]
    parts = []
    for number, piece in enumerate(pieces):
        file = directory / f"part-{number:05d}.parquet"
        piece.write_parquet(file)
        parts.append(describe_part(file))
    record = FeatureFiles(
        name,
        tuple(wide.columns),
        CATEGORICAL,
        tuple(parts),
        wide.height,
        wide["query_id"].n_unique(),
        {},
    )
    write_feature_files(directory, record)


def write_run(paths: ProjectPaths, run: str, wide: pl.DataFrame, n_customers: int) -> None:
    """The query list and the shards' wide tables of a Kaggle run, as the LightGBM run left them."""
    config = replace(KAGGLE, shard_customers=SHARD_CUSTOMERS)
    week = target_week(run, config, MANIFEST)
    queries = build_queries(np.arange(n_customers, dtype=np.int32), week, QUERY_ID_BASE)
    write_queries(queries, paths.kaggle_queries_file(run))
    n_shards = -(-n_customers // SHARD_CUSTOMERS)
    for shard in range(n_shards):
        low = QUERY_ID_BASE + shard * SHARD_CUSTOMERS
        rows = wide.filter(pl.col("query_id").is_between(low, low + SHARD_CUSTOMERS - 1))
        write_wide(paths.kaggle_features_dir(run, shard), week.name, rows)


def write_dataset_record(directory: Path, target: str, files: dict[str, Path]) -> None:
    """A stand-in for the Dataset record of a run's LightGBM ranker."""
    record = {
        "experiment": "full",
        "target_fold": target,
        "ratio": RATIO,
        "train_folds": list(files),
        "sample_sha256": {fold: file_sha256(file) for fold, file in files.items()},
        "git_commit": M8_COMMIT,
    }
    directory.mkdir(parents=True)
    (directory / "dataset.json").write_text(json.dumps(record), encoding="utf-8")


def write_t4_sample(paths: ProjectPaths) -> Path:
    """The sample file of T4 that the LightGBM submission drew itself (its own folder)."""
    file = paths.kaggle_sample_file(RUN_SUBMISSION, "T4", RATIO)
    job = SampleJob(
        fold=MANIFEST.get("T4"),
        features_dir=paths.features_dir("T4"),
        labels_file=paths.labels_file("T4"),
        experiment=build_experiment("full", LOCK),
        wide_columns=wide_table_columns(LOCK),
        sample_files={RATIO: file},
        allowed_phases=("T",),
        git_commit=M8_COMMIT,
    )
    sample_fold(job, RankerSamplingConfig(seed=42, ratios=(RATIO,), row_group_rows=100))
    return file


def write_stage2_record(inputs: TrialInputs) -> None:
    """The stage-2 record of T4 with seed 42: a formal trial of the T evaluation."""
    context = context_of(inputs)
    record_rehearsal(context)
    begin_formal_run(context, NOW, "scripts/m9_t_eval.py")
    run_and_record(inputs, context, TrialKey(STAGE_FORMAL, SEED, "T4"))


def build_kaggle_deepfm_workspace(root: Path) -> TrialInputs:
    """Write the tiny workspace under `root`; returns the inputs a process reads from configs."""
    inputs = build_t_eval_workspace(root)
    paths = inputs.paths
    t4_wide, t3_wide = fold_wide(paths, "T4"), fold_wide(paths, "T3")
    # One query per customer for T4: customer k has the k-th query of the fold.
    t4_queries = pl.read_parquet(paths.queries_file("T4")).sort("query_id")
    one_per_customer = t4_queries.with_columns(
        pl.int_range(pl.len(), dtype=pl.Int32).alias("customer_idx")
    )
    one_per_customer.write_parquet(paths.queries_file("T4"))
    write_stage2_record(inputs)
    # Rehearsal: the 30 T4 customers with T4's own rows, then 5 other customers.
    rehearsal_wide = pl.concat(
        [renumber(t4_wide, 0, N_FOLD_CUSTOMERS), renumber(t3_wide, N_FOLD_CUSTOMERS, 5)]
    )
    write_run(paths, RUN_REHEARSAL, rehearsal_wide, N_REHEARSAL_CUSTOMERS)
    shared = {
        fold: paths.ranker_sample_file("full", fold, RATIO) for fold in ("V4", "T1", "T2", "T3")
    }
    write_dataset_record(paths.kaggle_dataset_dir(RUN_REHEARSAL), "T4", shared)
    # Submission: 40 customers; T1-T3 have M8's sample files, T4 the run's own.
    submission_wide = pl.concat(
        [renumber(t4_wide, 0, N_FOLD_CUSTOMERS), renumber(t3_wide, N_FOLD_CUSTOMERS, 10)]
    )
    write_run(paths, RUN_SUBMISSION, submission_wide, N_SUBMISSION_CUSTOMERS)
    files = {fold: shared[fold] for fold in ("T1", "T2", "T3")}
    files["T4"] = write_t4_sample(paths)
    write_dataset_record(paths.kaggle_dataset_dir(RUN_SUBMISSION), "S", files)
    return inputs
