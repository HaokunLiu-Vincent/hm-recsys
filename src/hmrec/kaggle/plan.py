"""What a Kaggle run predicts: the week, the customers, their query ids and the shards.

Two runs use the same code path (docs/decisions.md 2026-11-19):

    submission   predicts the week after the data end (configs/kaggle_submission.yaml,
                 anchor 2020-09-23) for EVERY customer. This week is not a fold of
                 configs/fold_manifest.yaml and has no labels.
    rehearsal    predicts the last fold of the manifest (T4) again, for that fold's
                 query customers plus some other customers, so that the result can
                 be compared with what M8 saved. It reads no label.

Queries: one per customer. The customers are sorted by customer_idx and numbered
0..n-1 (position); query_id = query_id_base + position. The base is above every
query_id of the 16 folds, so a table of a Kaggle run can never be joined to a
fold's table by accident.

Shards: shard k holds the queries with positions [k * shard_customers,
(k + 1) * shard_customers). A shard only partitions the customers: every model
and every statistic over all customers is fitted once per run and used for all
shards (hmrec.kaggle.recall, hmrec.kaggle.features).

This module reads no transactions and no labels, and imports neither torch nor
LightGBM, so every process of a run can use it.
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from hmrec.config import INT32_MAX, KaggleConfig
from hmrec.data.folds import Fold, FoldManifest
from hmrec.data.id_maps import CUSTOMER_IDX
from hmrec.data.queries import ANCHOR, FOLD, QUERIES_SCHEMA, QUERY_ID, QUERY_ID_DTYPE
from hmrec.ranker.window import training_folds

RUN_REHEARSAL = "rehearsal"
RUN_SUBMISSION = "submission"
RUNS = (RUN_REHEARSAL, RUN_SUBMISSION)
# The phase of the submission week. It is not one of D, F, V, T, so every guard
# that checks a fold's phase (labels, tuning, evaluation) refuses it.
SUBMISSION_PHASE = "S"
# The phase whose folds train the ranker of a Kaggle run and are predicted again
# by the rehearsal.
LAST_PHASE = "T"


def check_run(run: str) -> None:
    """Fail unless `run` is "rehearsal" or "submission"."""
    if run not in RUNS:
        raise ValueError(f"run must be one of {RUNS}, got {run!r}")


# ---------------------------------------------------------------- the week


def submission_week(config: KaggleConfig, manifest: FoldManifest) -> Fold:
    """The submission week as a Fold object (it is NOT in the manifest).

    Raises:
        ValueError: If its name is a fold name, it does not start on the day after
            the manifest's last label day, or it is not as long as the folds.
    """
    last_fold = manifest.folds[-1]
    week = Fold(config.week_name, SUBMISSION_PHASE, config.week_anchor, config.week_label_end)
    if week.name in manifest.names:
        raise ValueError(f"the submission week is called {week.name!r}, which is a fold name")
    if week.anchor != last_fold.label_end + timedelta(days=1):
        raise ValueError(
            f"the submission week starts {week.anchor}, expected the day after the last "
            f"label day {last_fold.label_end} of fold {last_fold.name}"
        )
    if week.label_days != last_fold.label_days:
        raise ValueError(
            f"the submission week has {week.label_days} days, the folds have {last_fold.label_days}"
        )
    return week


def target_week(run: str, config: KaggleConfig, manifest: FoldManifest) -> Fold:
    """The week a run predicts: the submission week, or the rehearsal's fold.

    Raises:
        ValueError: If the rehearsal fold is not the LAST fold of the manifest.
            Only then is the model that predicts it the last model of the locked
            R6 chain, and its ranker the last one M8 trained.
    """
    check_run(run)
    if run == RUN_SUBMISSION:
        return submission_week(config, manifest)
    fold = manifest.get(config.rehearsal.target_fold)
    if fold != manifest.folds[-1] or fold.phase != LAST_PHASE:
        raise ValueError(
            f"the rehearsal must predict the last fold {manifest.folds[-1].name} "
            f"(phase {LAST_PHASE}), got {fold.name}"
        )
    return fold


def ranker_training_folds(
    run: str, config: KaggleConfig, manifest: FoldManifest, window_weeks: int
) -> tuple[Fold, ...]:
    """The `window_weeks` folds right before the predicted week, in time order.

    submission: the last folds of the manifest (T1 T2 T3 T4).
    rehearsal:  the folds before the rehearsal fold (V4 T1 T2 T3), the window M8
                trained the T4 model on.
    """
    target = target_week(run, config, manifest)
    if run == RUN_REHEARSAL:
        return training_folds(manifest, target.name, window_weeks)
    if not 0 < window_weeks <= len(manifest.folds):
        raise ValueError(f"window_weeks must be in 1..{len(manifest.folds)}, got {window_weeks}")
    folds = manifest.folds[-window_weeks:]
    wrong = [fold.name for fold in folds if fold.phase != LAST_PHASE]
    if wrong:
        raise ValueError(f"the submission's ranker trains on {LAST_PHASE} folds only, got {wrong}")
    return folds


def fine_tune_fold(run: str, manifest: FoldManifest) -> Fold | None:
    """The fold whose week R6 is fine-tuned on before predicting; None for the rehearsal.

    submission: the last fold (T4). The locked chain ends with the model fine-tuned
    on T3 (which predicted T4); one more weekly fine-tune, on the T4 week, gives
    the model that predicts the week after T4.
    rehearsal: none. T4 is predicted by the chain model that M6 used for it.
    """
    check_run(run)
    return manifest.folds[-1] if run == RUN_SUBMISSION else None


# ---------------------------------------------------------------- customers and queries


def rehearsal_customers(
    fold_customers: np.ndarray, all_customers: np.ndarray, n_extra: int, seed: int
) -> np.ndarray:
    """The rehearsal's customers: the fold's query customers plus `n_extra` others.

    The others are drawn without replacement, with numpy's default_rng(seed),
    from the customers that are not query customers of the fold (taken in
    ascending customer_idx order, so the draw depends only on the seed and the
    two sets).

    Args:
        fold_customers: int32 [n], the fold's query customers.
        all_customers: int32 [m], every customer of the customer table.
        n_extra: How many other customers to add.
        seed: Seed of the draw.

    Returns:
        int32 array, sorted, without duplicates: n + n_extra customers.

    Raises:
        ValueError: If fewer than n_extra other customers exist.
    """
    others = np.setdiff1d(all_customers, fold_customers)
    if n_extra > others.size:
        raise ValueError(f"cannot draw {n_extra} extra customers from {others.size}")
    drawn = np.random.default_rng(seed).choice(others, size=n_extra, replace=False)
    return np.union1d(fold_customers, drawn).astype(np.int32)


def check_query_id_range(
    query_id_base: int, n_customers: int, manifest: FoldManifest, query_id_fold_offset: int
) -> None:
    """Fail unless the run's query ids are above every fold's ids and fit in int32."""
    first_free = len(manifest.folds) * query_id_fold_offset
    if query_id_base < first_free:
        raise ValueError(
            f"query_id_base {query_id_base} is below {first_free}, where the query ids of "
            f"the {len(manifest.folds)} folds end"
        )
    if query_id_base + n_customers - 1 > INT32_MAX:
        raise ValueError(
            f"{n_customers} customers from query_id_base {query_id_base} do not fit in int32"
        )


def build_queries(customer_idx: np.ndarray, week: Fold, query_id_base: int) -> pl.DataFrame:
    """One query per customer for the predicted week.

    Args:
        customer_idx: int32 [n], distinct customers (any order).
        week: The predicted week (its name goes to the fold column).
        query_id_base: query_id of the customer with the smallest customer_idx.

    Returns:
        One row per customer, sorted by customer_idx (= by query_id).
        Columns: query_id (int32), fold (str), customer_idx (int32), anchor (date).

    Raises:
        ValueError: If a customer repeats.
    """
    customers = np.sort(customer_idx.astype(np.int32))
    if customers.size > 1 and bool(np.any(np.diff(customers) == 0)):
        raise ValueError("a customer appears twice; a run has one query per customer")
    query_ids = query_id_base + np.arange(customers.size, dtype=np.int64)
    queries = pl.DataFrame(
        {QUERY_ID: query_ids, CUSTOMER_IDX: customers},
        schema={QUERY_ID: pl.Int64, CUSTOMER_IDX: pl.Int32},
    ).with_columns(
        pl.col(QUERY_ID).cast(QUERY_ID_DTYPE, strict=True),
        pl.lit(week.name).alias(FOLD),
        pl.lit(week.anchor, dtype=pl.Date).alias(ANCHOR),
    )
    return queries.select(QUERIES_SCHEMA.names())


def write_queries(queries: pl.DataFrame, path: Path) -> None:
    """Write a run's query list once; an existing file must hold exactly the same queries.

    The query list is the frozen definition of a run: every shard file refers
    to its query ids. A second call (a resumed run) only compares.

    Raises:
        ValueError: If the schema is wrong or the existing file differs.
    """
    if queries.schema != QUERIES_SCHEMA:
        raise ValueError(
            f"queries schema is {dict(queries.schema)}, expected {dict(QUERIES_SCHEMA)}"
        )
    if path.exists():
        if not pl.read_parquet(path).equals(queries):
            raise ValueError(f"{path} exists and holds other queries; it is never replaced")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    queries.write_parquet(path)


def read_queries(path: Path) -> pl.DataFrame:
    """Read a run's query list (written by write_queries), sorted by query_id."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run the prepare step first")
    return pl.read_parquet(path)


# ---------------------------------------------------------------- shards


def number_of_shards(n_queries: int, shard_customers: int) -> int:
    """How many shards n_queries customers need: ceil(n_queries / shard_customers)."""
    if shard_customers <= 0:
        raise ValueError(f"shard_customers must be > 0, got {shard_customers}")
    if n_queries <= 0:
        raise ValueError(f"a run needs at least one query, got {n_queries}")
    return -(-n_queries // shard_customers)


def shard_queries(queries: pl.DataFrame, shard: int, shard_customers: int) -> pl.DataFrame:
    """The queries of one shard: positions [shard * size, (shard + 1) * size).

    Args:
        queries: The run's query list, sorted by query_id.
        shard: 0-based shard number.
        shard_customers: Customers per shard.

    Raises:
        ValueError: If the shard does not exist.
    """
    n_shards = number_of_shards(queries.height, shard_customers)
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard must be in 0..{n_shards - 1}, got {shard}")
    return queries.slice(shard * shard_customers, shard_customers)
