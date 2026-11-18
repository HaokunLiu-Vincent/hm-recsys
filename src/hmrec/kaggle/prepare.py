"""The first step of a Kaggle run: free disk space, the customers, the query list.

    submission   every customer of the customer table. They must be exactly the
                 customers of data/raw/sample_submission.csv (checked by id).
    rehearsal    the query customers of the rehearsal's fold (artifacts/queries/
                 {fold}.parquet: WHO bought that week, not what) plus the
                 configured number of other customers, drawn with a fixed seed.

The query list is written once (hmrec.kaggle.plan.write_queries); every later
step reads it. The real run starts only after a passed rehearsal of the same code
(hmrec.kaggle.rehearsal.require_passed_rehearsal).

No label and no transaction is read here.
"""

import shutil

import numpy as np
import polars as pl

from hmrec.config import load_queries_config
from hmrec.data.id_maps import CUSTOMER_IDX
from hmrec.data.metadata import scan_processed_parquet
from hmrec.git_state import code_changed_since
from hmrec.kaggle.context import KaggleContext
from hmrec.kaggle.plan import (
    RUN_REHEARSAL,
    RUN_SUBMISSION,
    build_queries,
    check_query_id_range,
    number_of_shards,
    rehearsal_customers,
    write_queries,
)
from hmrec.kaggle.rehearsal import require_passed_rehearsal
from hmrec.kaggle.submission import CUSTOMER_ID
from hmrec.logging_utils import get_logger
from hmrec.memory import bytes_to_gb

LOGGER = get_logger("kaggle.prepare")


def required_free_gb(context: KaggleContext) -> float:
    """Free space a run needs before it starts: the estimate times the safety factor."""
    config = context.config
    estimate = (
        config.rehearsal_disk_gb if context.run == RUN_REHEARSAL else config.submission_disk_gb
    )
    return estimate * config.disk_safety_factor


def check_free_disk(context: KaggleContext) -> float:
    """Fail if the artifacts folder's disk has less free space than the run needs.

    Returns:
        The free space in GB.

    Raises:
        OSError: If it is below required_free_gb.
    """
    root = context.paths.artifacts_root
    free_gb = bytes_to_gb(shutil.disk_usage(root).free)
    needed_gb = required_free_gb(context)
    LOGGER.info("disk: %.1f GB free at %s, %.1f GB needed", free_gb, root, needed_gb)
    if free_gb < needed_gb:
        raise OSError(f"only {free_gb:.1f} GB free at {root}; the run needs {needed_gb:.1f} GB")
    return free_gb


def read_customer_ids(context: KaggleContext) -> pl.DataFrame:
    """customer_idx (int32) -> customer_id (the original string) of every customer."""
    paths = context.paths
    table = scan_processed_parquet(
        paths.customer_id_map_file, paths.processed_metadata_file, context.preprocess
    )
    return table.select(CUSTOMER_IDX, CUSTOMER_ID).collect()


def read_sample_customer_ids(context: KaggleContext) -> pl.Series:
    """The customer ids of data/raw/sample_submission.csv, in file order."""
    sample = pl.read_csv(
        context.paths.sample_submission_csv,
        columns=[CUSTOMER_ID],
        schema_overrides={CUSTOMER_ID: pl.String},
    )
    return sample[CUSTOMER_ID]


def check_sample_customers(customer_ids: pl.DataFrame, sample_ids: pl.Series) -> None:
    """Fail unless the sample file holds exactly the customers of the customer table, once each.

    Raises:
        ValueError: If the counts differ, a sample id repeats or is unknown.
    """
    if sample_ids.n_unique() != sample_ids.len():
        raise ValueError("sample_submission.csv repeats a customer_id")
    known = customer_ids[CUSTOMER_ID]
    unknown = sample_ids.filter(~sample_ids.is_in(known.implode())).len()
    if sample_ids.len() != known.len() or unknown > 0:
        raise ValueError(
            f"sample_submission.csv has {sample_ids.len()} customers ({unknown} unknown), the "
            f"customer table has {known.len()}"
        )


def run_customers(context: KaggleContext, all_customers: np.ndarray) -> np.ndarray:
    """The customers a run predicts for (module docstring), int32, sorted."""
    if context.run == RUN_SUBMISSION:
        return np.sort(all_customers.astype(np.int32))
    fold_queries = pl.read_parquet(
        context.paths.queries_file(context.target.name), columns=[CUSTOMER_IDX]
    )
    rehearsal = context.config.rehearsal
    return rehearsal_customers(
        fold_queries[CUSTOMER_IDX].to_numpy().astype(np.int32),
        all_customers.astype(np.int32),
        rehearsal.extra_customers,
        rehearsal.extra_customers_seed,
    )


def prepare_run(context: KaggleContext) -> pl.DataFrame:
    """Check the preconditions of a run and write its query list.

    Returns:
        The run's queries: query_id, fold, customer_idx, anchor.

    Raises:
        RuntimeError: For the real run without a passed rehearsal of this code.
        OSError: If there is not enough free disk space.
        ValueError: If the sample file's customers are not the customer table's.
    """
    paths = context.paths
    if context.run == RUN_SUBMISSION:
        commit = require_passed_rehearsal(
            paths, lambda recorded: code_changed_since(paths.project_root, recorded)
        )
        LOGGER.info("the rehearsal passed at commit %s with this code", commit)
    check_free_disk(context)
    customer_ids = read_customer_ids(context)
    check_sample_customers(customer_ids, read_sample_customer_ids(context))
    customers = run_customers(context, customer_ids[CUSTOMER_IDX].to_numpy())
    offset = load_queries_config(paths.queries_config_file).query_id_fold_offset
    base = context.config.query_id_base
    check_query_id_range(base, customers.size, context.manifest, offset)
    queries = build_queries(customers, context.target, base)
    write_queries(queries, paths.kaggle_queries_file(context.run))
    LOGGER.info(
        "%s: %d customers in %d shards of %d; query ids %d..%d",
        context.run,
        queries.height,
        number_of_shards(queries.height, context.config.shard_customers),
        context.config.shard_customers,
        base,
        base + queries.height - 1,
    )
    return queries
