"""What every process of a Kaggle run starts with: configs, the predicted week, the code state.

A process of a run does one step (one recall channel, the features of one
shard, the ranker training, ...), so that its process max RSS is the true
memory peak of that step (hmrec.memory). Every process:

    context = start_process(run, label)        # logging, configs, git state
    ... one step ...
    finish_process(context, step, ...)         # one row in results/kaggle_runs.csv

A run works on the full data, so start_process REFUSES a working tree with
uncommitted code or config changes (AGENTS.md §3.4): every file and every result
row of a run belongs to one commit.

No torch and no LightGBM is imported here.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime

from hmrec.config import (
    BaseConfig,
    KaggleConfig,
    PreprocessConfig,
    load_base_config,
    load_kaggle_config,
    load_preprocess_config,
)
from hmrec.csv_records import append_row
from hmrec.data.folds import Fold, FoldManifest, load_fold_manifest
from hmrec.git_state import git_state
from hmrec.kaggle.plan import check_run, target_week
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.memory import log_process_peak_memory
from hmrec.paths import ProjectPaths, get_paths
from hmrec.run_records import RunStamp, check_memory_limit
from hmrec.threads import check_polars_thread_pool

LOGGER = get_logger("kaggle.context")

RUN_COLUMNS: tuple[str, ...] = (
    "run",
    "step",
    "shard",
    "status",
    "n_rows",
    "seconds",
    "process_max_rss_gb",
    "children_max_rss_gb",
    "detail",
    "git_commit",
    "git_dirty",
    "finished_at",
)
ALL_SHARDS = "all"
DONE = "done"
SKIPPED = "skipped"


@dataclass(frozen=True)
class KaggleContext:
    """The configuration of one process of a Kaggle run.

    Attributes:
        run: "rehearsal" or "submission".
        paths: Project paths.
        base: configs/base.yaml (threads, memory limit, chunk sizes).
        config: configs/kaggle_submission.yaml.
        manifest: configs/fold_manifest.yaml (unchanged; the submission week is not in it).
        preprocess: configs/preprocess.yaml (day_zero).
        target: The predicted week (hmrec.kaggle.plan.target_week).
        stamp: HEAD and whether the working tree is dirty.
    """

    run: str
    paths: ProjectPaths
    base: BaseConfig
    config: KaggleConfig
    manifest: FoldManifest
    preprocess: PreprocessConfig
    target: Fold
    stamp: RunStamp


def load_context(run: str, paths: ProjectPaths, stamp: RunStamp) -> KaggleContext:
    """Read the configs of a run and work out the predicted week (no data is read)."""
    check_run(run)
    config = load_kaggle_config(paths.kaggle_config_file)
    manifest = load_fold_manifest(paths.fold_manifest_file)
    return KaggleContext(
        run=run,
        paths=paths,
        base=load_base_config(paths.base_config_file),
        config=config,
        manifest=manifest,
        preprocess=load_preprocess_config(paths.preprocess_config_file),
        target=target_week(run, config, manifest),
        stamp=stamp,
    )


def require_clean_tree(stamp: RunStamp) -> None:
    """Fail if the working tree has uncommitted code or config changes.

    Raises:
        RuntimeError: If it has. A run on the full data must come from one commit.
    """
    if stamp.git_dirty:
        raise RuntimeError(
            f"the working tree has uncommitted changes (HEAD {stamp.git_commit[:12]}): a Kaggle "
            "run works on the full data and starts only from committed code"
        )


def start_process(run: str, label: str) -> KaggleContext:
    """Set up logging, read the configs and refuse a dirty working tree.

    Args:
        run: "rehearsal" or "submission".
        label: Name of this process, e.g. "recall_r4_itemcf"; names its log file.
    """
    check_run(run)
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.kaggle_log_file(run, label))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    stamp = RunStamp(commit, dirty)
    LOGGER.info("kaggle %s %s | git HEAD %s, uncommitted changes: %s", run, label, commit, dirty)
    require_clean_tree(stamp)
    context = load_context(run, paths, stamp)
    threads = check_polars_thread_pool(context.base.num_threads)
    LOGGER.info("polars thread pool size = %d; predicted week %s", threads, context.target)
    return context


def finish_process(
    context: KaggleContext,
    step: str,
    shard: str,
    status: str,
    n_rows: int,
    started: float,
    detail: str,
) -> None:
    """Append this process's row to results/kaggle_runs.csv, then check the memory limit.

    Call it as the LAST action of a process: the true peak (ru_maxrss) is read here.

    Args:
        context: The process's context.
        step: What the process did, e.g. "recall_r4_itemcf" or "features".
        shard: The shard number as text, or "all".
        status: "done" or "skipped".
        n_rows: Rows of the step's main output.
        started: time.perf_counter() at the start of the process.
        detail: Short free text.

    Raises:
        MemoryError: If the process's true peak RSS is above memory.limit_gb. The
            row is written first, so the run is on record.
    """
    label = f"kaggle {context.run} {step} shard {shard}"
    limit_gb = context.base.memory.limit_gb
    peak = log_process_peak_memory(LOGGER, label, limit_gb)
    row = {
        "run": context.run,
        "step": step,
        "shard": shard,
        "status": status,
        "n_rows": str(n_rows),
        "seconds": repr(time.perf_counter() - started),
        "process_max_rss_gb": repr(peak.process_max_rss_gb),
        "children_max_rss_gb": repr(peak.children_max_rss_gb),
        "detail": detail,
        "git_commit": context.stamp.git_commit,
        "git_dirty": str(context.stamp.git_dirty),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    append_row(context.paths.kaggle_runs_file, RUN_COLUMNS, row)
    LOGGER.info("run row -> %s: %s", context.paths.kaggle_runs_file, row)
    check_memory_limit(peak, limit_gb, label)
