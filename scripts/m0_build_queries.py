"""CLI for M0: build the query list and label sets of each fold (proposal §2.3, §14.2).

Usage:
    uv run python scripts/m0_build_queries.py [--folds D1 D2 ...] [--overwrite]
        [--data-root DIR] [--artifacts-root DIR] [--run-id m0_build_queries]

Reads data/processed/transactions.parquet and articles.parquet (through
metadata.scan_processed_parquet, which checks metadata.json first) and writes,
for every selected fold:
    artifacts/queries/{fold}.parquet
    artifacts/labels/{fold}.parquet
Existing files are not replaced unless --overwrite is given.

--data-root / --artifacts-root replace the roots from configs/base.yaml, so the
script can be tried on a small hand-made dataset. Configs are always read from
the repository's configs/ folder.
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
# hmrec.threads imports only the standard library and yaml. See its docstring.
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import dataclasses  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

from hmrec.config import (  # noqa: E402
    load_base_config,
    load_preprocess_config,
    load_queries_config,
)
from hmrec.data.catalog import build_catalog  # noqa: E402
from hmrec.data.folds import Fold, load_fold_manifest  # noqa: E402
from hmrec.data.metadata import scan_processed_parquet  # noqa: E402
from hmrec.data.queries import build_fold_queries, write_fold_queries  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import MemoryTracker  # noqa: E402
from hmrec.paths import ProjectPaths, get_paths  # noqa: E402

LOGGER = get_logger("m0_build_queries")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", help="fold names, e.g. D1 D2; default: all folds")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    parser.add_argument("--data-root", type=Path, help="override paths.data_root")
    parser.add_argument("--artifacts-root", type=Path, help="override paths.artifacts_root")
    parser.add_argument(
        "--run-id", default="m0_build_queries", help="log goes to artifacts/runs/{run_id}/run.log"
    )
    return parser.parse_args()


def resolve_paths(data_root: Path | None, artifacts_root: Path | None) -> ProjectPaths:
    """Return the project paths, with the data / artifacts roots replaced if given."""
    paths = get_paths()
    if data_root is not None:
        paths = dataclasses.replace(paths, data_root=data_root.resolve())
    if artifacts_root is not None:
        paths = dataclasses.replace(paths, artifacts_root=artifacts_root.resolve())
    return paths


def select_folds(all_folds: tuple[Fold, ...], names: list[str] | None) -> list[Fold]:
    """Return the requested folds in manifest order; all folds when `names` is None."""
    if names is None:
        return list(all_folds)
    known = {fold.name for fold in all_folds}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"unknown folds {unknown}; known folds: {sorted(known)}")
    return [fold for fold in all_folds if fold.name in names]


def main() -> None:
    """Build and write queries and labels for the selected folds."""
    args = parse_args()
    paths = resolve_paths(args.data_root, args.artifacts_root)
    setup_logging(logging.INFO, log_file=paths.run_dir(args.run_id) / "run.log")
    base_config = load_base_config(paths.base_config_file)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(base_config.num_threads))
    preprocess_config = load_preprocess_config(paths.preprocess_config_file)
    queries_config = load_queries_config(paths.queries_config_file)
    manifest = load_fold_manifest(paths.fold_manifest_file)

    metadata_file = paths.processed_metadata_file
    transactions = scan_processed_parquet(paths.transactions_file, metadata_file, preprocess_config)
    articles = scan_processed_parquet(paths.articles_file, metadata_file, preprocess_config)
    catalog = build_catalog(articles)
    LOGGER.info("catalog has %d articles", catalog.height)

    for fold in select_folds(manifest.folds, args.folds):
        with MemoryTracker(
            f"queries {fold.name}",
            LOGGER,
            limit_gb=base_config.memory.limit_gb,
            sample_interval_seconds=base_config.memory.sample_interval_seconds,
        ):
            fold_queries = build_fold_queries(
                transactions,
                catalog,
                fold,
                manifest,
                preprocess_config.day_zero,
                queries_config.query_id_fold_offset,
            )
            queries_path = paths.queries_file(fold.name)
            labels_path = paths.labels_file(fold.name)
            write_fold_queries(fold_queries, queries_path, labels_path, overwrite=args.overwrite)
        LOGGER.info(
            "fold %s (anchor %s): %d queries, %d labels -> %s, %s",
            fold.name,
            fold.anchor,
            fold_queries.queries.height,
            fold_queries.labels.height,
            queries_path,
            labels_path,
        )


if __name__ == "__main__":
    main()
