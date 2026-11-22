"""CLI: the local MLflow view over run manifests and result tables (docs/mlops.md §7.3).

Usage (the Makefile targets run these under the dependency group "tracking"):
    python scripts/m11_mlflow.py sync     # write missing records and model versions
    python scripts/m11_mlflow.py check    # every metric against its source; the registry
                                          # against the manifests, the files and the state
    python scripts/m11_mlflow.py ui       # serve the UI on this machine

The store is artifacts/mlflow/mlflow.db (SQLite, not in git). It is only a view:
the result tables and the run manifests stay the record. `sync` never changes a
run that is already in the store; delete the folder artifacts/mlflow/ to rebuild it.

`check` ends with exit code 1 if any metric differs from its source or a record
is missing.
"""

import argparse
import logging
import os
import sys

from hmrec.config import load_versioning_config
from hmrec.data.folds import load_fold_manifest
from hmrec.locks import load_ranker_lock
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.paths import get_paths
from hmrec.tracking.store import TELEMETRY_OFF, store_uri
from hmrec.tracking.view import (
    check_models,
    check_view,
    sync_models,
    sync_view,
    ui_command,
    view_models,
    view_runs,
)

LOGGER = get_logger("scripts.m11_mlflow")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="The local MLflow view.")
    parser.add_argument("action", choices=("sync", "check", "ui"))
    return parser.parse_args()


def main() -> None:
    """Sync, check or serve the view."""
    args = parse_args()
    setup_logging(logging.INFO)
    paths = get_paths()
    config = load_versioning_config(paths.versioning_config_file).mlflow
    if args.action == "ui":
        command = ui_command(paths, config)
        LOGGER.info("store: %s", store_uri(paths.mlflow_store_file))
        LOGGER.info("open http://%s:%d (stop with Ctrl-C)", config.ui_host, config.ui_port)
        # The UI is another program: this process becomes it.
        os.execvpe(command[0], command, {**os.environ, **TELEMETRY_OFF})
    manifest = load_fold_manifest(paths.fold_manifest_file)
    lock = load_ranker_lock(paths.ranker_lock_file)
    runs = view_runs(paths, config, manifest, lock)
    models, current = view_models(paths)
    if args.action == "sync":
        summary = sync_view(paths, runs)
        LOGGER.info(
            "store %s: %d record(s) written, %d already there",
            paths.mlflow_store_file,
            len(summary.logged),
            len(summary.skipped),
        )
        registry = sync_models(paths, models, current)
        LOGGER.info(
            "model registry (a mirror of the pipeline's state): %d version(s) created, %d "
            "already there; alias 'current' -> %s",
            len(registry.registered),
            len(registry.present),
            registry.aliases or "no registered model",
        )
        return
    result = check_view(paths, config, runs)
    registry_problems = check_models(paths, models, current)
    for problem in (*result.problems, *registry_problems):
        LOGGER.error("%s", problem)
    LOGGER.info(
        "checked %d run(s), %d metric value(s): %d difference(s); %d registered model "
        "version(s) against the manifests, the files and state/current.json: %d difference(s)",
        result.n_runs,
        result.n_metrics,
        len(result.problems),
        len(models),
        len(registry_problems),
    )
    if not result.passed or registry_problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
