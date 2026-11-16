"""CLI for M0: convert the raw CSV files to Parquet and align the embeddings.

Usage:
    uv run python scripts/m0_preprocess.py [--run-id m0_preprocess]

Reads data/raw/*.csv and the precomputed embeddings, writes everything to
data/processed/ (see hmrec.data.preprocess for the schemas), then writes
data/processed/metadata.json. Each step logs its duration and peak memory.
The log is also written to artifacts/runs/{run_id}/run.log.
"""

# ---------------------------------------------------------------------------
# Thread setup. This block MUST run before polars (or any hmrec module, which
# imports polars) is imported: polars reads POLARS_MAX_THREADS only once, at
# import time. That is also why the base.yaml path is built by hand here
# instead of with hmrec.paths, and why the imports below come after code.
# ---------------------------------------------------------------------------
import os
from pathlib import Path

import yaml

_BASE_CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
_NUM_THREADS = str(yaml.safe_load(_BASE_CONFIG_FILE.read_text(encoding="utf-8"))["num_threads"])
_preset_threads = os.environ.get("POLARS_MAX_THREADS")
if _preset_threads is not None and _preset_threads != _NUM_THREADS:
    raise RuntimeError(
        f"POLARS_MAX_THREADS={_preset_threads} is set in the environment, but base.yaml "
        f"num_threads={_NUM_THREADS}; unset it or make them equal"
    )
os.environ["POLARS_MAX_THREADS"] = _NUM_THREADS

import argparse  # noqa: E402  (after the thread setup above, see comment)
import logging  # noqa: E402
from collections.abc import Callable  # noqa: E402

import polars as pl  # noqa: E402

from hmrec.config import (  # noqa: E402
    BaseConfig,
    PreprocessConfig,
    load_base_config,
    load_preprocess_config,
)
from hmrec.data import id_maps, metadata, preprocess  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import MemoryTracker, bytes_to_gb  # noqa: E402
from hmrec.paths import ProjectPaths, get_paths  # noqa: E402

LOGGER = get_logger("m0_preprocess")
BYTES_PER_MB = 1024**2


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", default="m0_preprocess", help="log goes to artifacts/runs/{run_id}/run.log"
    )
    return parser.parse_args()


def check_thread_pool(base_config: BaseConfig) -> None:
    """Fail if polars did not pick up num_threads from base.yaml."""
    pool_size = pl.thread_pool_size()
    LOGGER.info("polars thread pool size = %d", pool_size)
    if pool_size != base_config.num_threads:
        raise RuntimeError(
            f"polars uses {pool_size} threads, expected num_threads={base_config.num_threads}"
        )


def run_step[StepResult](
    label: str, base_config: BaseConfig, peaks: dict[str, float], step: Callable[[], StepResult]
) -> StepResult:
    """Run one step inside a MemoryTracker and remember its peak RSS in GB."""
    with MemoryTracker(
        label,
        LOGGER,
        limit_gb=base_config.memory.limit_gb,
        sample_interval_seconds=base_config.memory.sample_interval_seconds,
    ) as tracker:
        result = step()
    peaks[label] = bytes_to_gb(tracker.peak_rss_bytes)
    return result


def build_id_maps(paths: ProjectPaths) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Step 1: build and write the customer and article ID maps."""
    customer_ids = preprocess.scan_all_string_csv(paths.customers_csv).select(id_maps.CUSTOMER_ID)
    article_ids = preprocess.scan_all_string_csv(paths.articles_csv).select(id_maps.ARTICLE_ID)
    customer_map = id_maps.build_customer_id_map(customer_ids.collect()[id_maps.CUSTOMER_ID])
    article_map = id_maps.build_article_id_map(article_ids.collect()[id_maps.ARTICLE_ID])
    id_maps.write_id_map(customer_map, paths.customer_id_map_file)
    id_maps.write_id_map(article_map, paths.article_id_map_file)
    LOGGER.info("wrote %s (%d rows)", paths.customer_id_map_file, customer_map.height)
    LOGGER.info("wrote %s (%d rows)", paths.article_id_map_file, article_map.height)
    return customer_map, article_map


def build_customers(
    paths: ProjectPaths, customer_map: pl.DataFrame, config: PreprocessConfig
) -> pl.DataFrame:
    """Step 2: build and write the customers table."""
    raw = preprocess.scan_all_string_csv(paths.customers_csv)
    customers = preprocess.build_customers(raw, customer_map, config)
    preprocess.write_table(customers, paths.customers_file)
    LOGGER.info("wrote %s (%d rows)", paths.customers_file, customers.height)
    return customers


def build_articles(paths: ProjectPaths, article_map: pl.DataFrame) -> pl.DataFrame:
    """Step 3: build and write the articles table."""
    raw = preprocess.scan_all_string_csv(paths.articles_csv)
    articles = preprocess.build_articles(raw, article_map)
    preprocess.write_table(articles, paths.articles_file)
    LOGGER.info("wrote %s (%d rows)", paths.articles_file, articles.height)
    return articles


def build_transactions(
    paths: ProjectPaths,
    customer_map: pl.DataFrame,
    article_map: pl.DataFrame,
    config: PreprocessConfig,
) -> preprocess.TransactionStats:
    """Step 4: stream transactions to Parquet, then validate the written file."""
    raw_row_count = preprocess.count_rows(preprocess.scan_transactions_csv(paths.transactions_csv))
    LOGGER.info("raw transactions_train.csv has %d rows", raw_row_count)
    raw = preprocess.scan_transactions_csv(paths.transactions_csv)
    transactions = preprocess.build_transactions(raw, customer_map, article_map, config.day_zero)
    preprocess.write_transactions(
        transactions, paths.transactions_file, config.transactions_row_group_size
    )
    LOGGER.info("wrote %s", paths.transactions_file)
    stats = preprocess.validate_transactions(
        pl.scan_parquet(paths.transactions_file), expected_rows=raw_row_count
    )
    LOGGER.info(
        "transactions valid: %d rows, t_dat_day %d..%d", stats.n_rows, stats.min_day, stats.max_day
    )
    return stats


def build_embeddings(
    paths: ProjectPaths, article_map: pl.DataFrame, config: PreprocessConfig
) -> dict[str, int]:
    """Step 5: align image and text embeddings to article_idx; return missing counts."""
    sources = {
        "image": (
            paths.image_embeddings_file,
            paths.image_article_ids_file,
            config.embeddings.image_dim,
            paths.image_embeddings_by_idx_file,
            paths.image_embedding_mask_file,
        ),
        "text": (
            paths.text_embeddings_file,
            paths.text_article_ids_file,
            config.embeddings.text_dim,
            paths.text_embeddings_by_idx_file,
            paths.text_embedding_mask_file,
        ),
    }
    missing_counts: dict[str, int] = {}
    for name, (vectors_file, ids_file, dim, out_vectors, out_mask) in sources.items():
        vectors, row_ids = preprocess.read_embedding_source(vectors_file, ids_file)
        aligned, mask = preprocess.align_embeddings(
            vectors, row_ids, article_map, dim, config.embeddings.norm_tolerance
        )
        preprocess.write_aligned_embeddings(aligned, mask, out_vectors, out_mask)
        missing_counts[name] = int((~mask).sum())
        LOGGER.info(
            "wrote %s %s; %d articles without a vector",
            out_vectors,
            aligned.shape,
            missing_counts[name],
        )
    return missing_counts


def log_report(
    paths: ProjectPaths,
    customers: pl.DataFrame,
    config: PreprocessConfig,
    missing_counts: dict[str, int],
    peaks: dict[str, float],
) -> None:
    """Log rows / file size / memory per output, age-bucket counts and memory peaks."""
    parquet_outputs = {
        "transactions": paths.transactions_file,
        "customers": paths.customers_file,
        "articles": paths.articles_file,
        "customer_id_map": paths.customer_id_map_file,
        "article_id_map": paths.article_id_map_file,
    }
    npy_outputs = {
        "image_by_article_idx": paths.image_embeddings_by_idx_file,
        "text_by_article_idx": paths.text_embeddings_by_idx_file,
    }
    summaries = {name: preprocess.summarize_parquet_table(p) for name, p in parquet_outputs.items()}
    summaries |= {name: preprocess.summarize_npy_array(p) for name, p in npy_outputs.items()}
    for name, summary in summaries.items():
        LOGGER.info(
            "report | %-20s | rows=%10d | file=%8.1f MB | memory=%8.1f MB",
            name,
            summary.n_rows,
            summary.file_bytes / BYTES_PER_MB,
            summary.memory_bytes / BYTES_PER_MB,
        )
    for bucket, count in preprocess.count_by_age_bucket(customers, config).iter_rows():
        LOGGER.info("report | age_bucket %-12s | %d customers", bucket, count)
    for name, count in missing_counts.items():
        LOGGER.info("report | articles without %s embedding: %d", name, count)
    for label, peak_gb in peaks.items():
        LOGGER.info("report | peak rss %-14s | %.2f GB", label, peak_gb)


def main() -> None:
    """Run all M0 preprocessing steps."""
    args = parse_args()
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.run_dir(args.run_id) / "run.log")
    base_config = load_base_config(paths.base_config_file)
    config = load_preprocess_config(paths.preprocess_config_file)
    check_thread_pool(base_config)

    peaks: dict[str, float] = {}
    with MemoryTracker(
        "m0 total",
        LOGGER,
        limit_gb=base_config.memory.limit_gb,
        sample_interval_seconds=base_config.memory.sample_interval_seconds,
    ) as total_tracker:
        customer_map, article_map = run_step(
            "1 id maps", base_config, peaks, lambda: build_id_maps(paths)
        )
        customers = run_step(
            "2 customers", base_config, peaks, lambda: build_customers(paths, customer_map, config)
        )
        articles = run_step(
            "3 articles", base_config, peaks, lambda: build_articles(paths, article_map)
        )
        stats = run_step(
            "4 transactions",
            base_config,
            peaks,
            lambda: build_transactions(paths, customer_map, article_map, config),
        )
        missing_counts = run_step(
            "5 embeddings", base_config, peaks, lambda: build_embeddings(paths, article_map, config)
        )
        row_counts = {
            "transactions": stats.n_rows,
            "customers": customers.height,
            "articles": articles.height,
        }
        metadata.write_processed_metadata(
            paths.processed_metadata_file, metadata.build_processed_metadata(config, row_counts)
        )
        LOGGER.info("wrote %s", paths.processed_metadata_file)
    peaks["m0 total"] = bytes_to_gb(total_tracker.peak_rss_bytes)

    log_report(paths, customers, config, missing_counts, peaks)


if __name__ == "__main__":
    main()
