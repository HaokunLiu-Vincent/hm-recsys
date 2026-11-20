"""What every process of the DeepFM Kaggle submission starts with (docs/decisions.md 2026-11-20).

A process does one step (train, score, submission file, rehearsal check), so its
process max RSS is the true memory peak of that step:

    context = start_process(run, label)        # logging, configs, git state
    ... one step ...
    finish_process(context, step, ...)         # one row in results/kaggle_deepfm_runs.csv

The two runs are the ones of the LightGBM submission (hmrec.kaggle.plan):

    rehearsal    predicts T4 again for the T4 queries plus 50,000 other customers,
                 with a model trained on V4 T1 T2 T3
    submission   predicts the week after the data end for every customer, with a
                 model trained on T1 T2 T3 T4

The candidates, the features and the query list of a run are READ from
artifacts/kaggle/{run}/ (written by the LightGBM submission); everything this
package writes goes to artifacts/kaggle/deepfm/{run}/ and results/kaggle_deepfm_*.csv.

Labels: the training reads the label column of the sample files (T labels as
TRAINING labels, with the user's consent). record_label_access appends a row to
results/kaggle_deepfm_label_access.csv and must be called BEFORE such a read.

A run works on the full data, so start_process REFUSES a working tree with
uncommitted code or config changes.

No torch and no LightGBM is imported here.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime

from hmrec.config import (
    KaggleDeepFMConfig,
    TEvalConfig,
    load_kaggle_deepfm_config,
    load_t_eval_config,
)
from hmrec.csv_records import append_row
from hmrec.data.folds import Fold
from hmrec.git_state import git_state
from hmrec.kaggle.context import RUN_COLUMNS, KaggleContext, load_context, require_clean_tree
from hmrec.kaggle.plan import check_run, ranker_training_folds
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.memory import log_process_peak_memory
from hmrec.paths import ProjectPaths, get_paths
from hmrec.run_records import RunStamp, check_memory_limit
from hmrec.threads import check_polars_thread_pool

LOGGER = get_logger("kaggle_deepfm.context")

ALL_SHARDS = "all"
DONE = "done"
SKIPPED = "skipped"
LABEL_ACCESS_COLUMNS: tuple[str, ...] = (
    "recorded_at",
    "git_commit",
    "git_dirty",
    "run",
    "folds",
    "purpose",
    "script",
)


@dataclass(frozen=True)
class DeepFMKaggleContext:
    """The configuration of one process of the DeepFM Kaggle submission.

    Attributes:
        kaggle: The run's context of the LightGBM submission: run, paths, base
            config, configs/kaggle_submission.yaml, fold manifest, the predicted
            week and the code state.
        settings: configs/kaggle_deepfm.yaml (seed, configuration id, reading).
        chosen: configs/m9_t_eval.yaml: the settings of the DeepFM configuration
            chosen on V.
    """

    kaggle: KaggleContext
    settings: KaggleDeepFMConfig
    chosen: TEvalConfig

    @property
    def run(self) -> str:
        """The run: "rehearsal" or "submission"."""
        return self.kaggle.run

    @property
    def paths(self) -> ProjectPaths:
        """Project paths."""
        return self.kaggle.paths

    @property
    def stamp(self) -> RunStamp:
        """HEAD and whether the working tree is dirty."""
        return self.kaggle.stamp


def check_settings(settings: KaggleDeepFMConfig, chosen: TEvalConfig) -> None:
    """Fail unless the settings name the configuration chosen on V and one of its seeds.

    The seed must be a seed of the stage-2 evaluation: the rehearsal is compared
    with that evaluation's record of the same seed.

    Raises:
        ValueError: On any violation.
    """
    if settings.config_id != chosen.config_id:
        raise ValueError(
            f"kaggle_deepfm.config_id is {settings.config_id}, but the configuration chosen on "
            f"V (configs/m9_t_eval.yaml) has id {chosen.config_id}"
        )
    if settings.seed not in chosen.seeds:
        raise ValueError(
            f"kaggle_deepfm.seed {settings.seed} is not a seed of the stage-2 evaluation "
            f"{chosen.seeds}: the rehearsal would have no record to compare with"
        )


def load_deepfm_context(run: str, paths: ProjectPaths, stamp: RunStamp) -> DeepFMKaggleContext:
    """Read the configs of a run (no data is read).

    Raises:
        ValueError: If a config is invalid or the settings do not fit together.
    """
    check_run(run)
    settings = load_kaggle_deepfm_config(paths.kaggle_deepfm_config_file)
    chosen = load_t_eval_config(paths.t_eval_config_file)
    check_settings(settings, chosen)
    return DeepFMKaggleContext(load_context(run, paths, stamp), settings, chosen)


def start_process(run: str, label: str) -> DeepFMKaggleContext:
    """Set up logging, read the configs and refuse a dirty working tree.

    Args:
        run: "rehearsal" or "submission".
        label: Name of this process, e.g. "train"; names its log file.
    """
    check_run(run)
    paths = get_paths()
    setup_logging(logging.INFO, log_file=paths.kaggle_deepfm_log_file(run, label))
    commit, dirty = git_state(paths.project_root, paths.results_root)
    stamp = RunStamp(commit, dirty)
    LOGGER.info(
        "kaggle deepfm %s %s | git HEAD %s, uncommitted changes: %s", run, label, commit, dirty
    )
    require_clean_tree(stamp)
    context = load_deepfm_context(run, paths, stamp)
    threads = check_polars_thread_pool(context.kaggle.base.num_threads)
    LOGGER.info("polars thread pool size = %d; predicted week %s", threads, context.kaggle.target)
    return context


def finish_process(
    context: DeepFMKaggleContext, step: str, status: str, n_rows: int, started: float, detail: str
) -> None:
    """Append this process's row to results/kaggle_deepfm_runs.csv, then check the memory limit.

    Call it as the LAST action of a process: the true peak (ru_maxrss) is read here.

    Args:
        context: The process's context.
        step: What the process did, e.g. "train".
        status: "done" or "skipped".
        n_rows: Rows of the step's main output.
        started: time.perf_counter() at the start of the process.
        detail: Short free text.

    Raises:
        MemoryError: If the process's true peak RSS is above memory.limit_gb. The
            row is written first, so the run is on record.
    """
    label = f"kaggle deepfm {context.run} {step}"
    limit_gb = context.kaggle.base.memory.limit_gb
    peak = log_process_peak_memory(LOGGER, label, limit_gb)
    row = {
        "run": context.run,
        "step": step,
        "shard": ALL_SHARDS,
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
    append_row(context.paths.kaggle_deepfm_runs_file, RUN_COLUMNS, row)
    LOGGER.info("run row -> %s: %s", context.paths.kaggle_deepfm_runs_file, row)
    check_memory_limit(peak, limit_gb, label)


def training_folds(context: DeepFMKaggleContext, window_weeks: int) -> tuple[Fold, ...]:
    """The folds the run's model is trained on, in time order (hmrec.kaggle.plan).

    rehearsal: V4 T1 T2 T3 (the window of the T4 models of M8 and of stage 2).
    submission: T1 T2 T3 T4.
    """
    kaggle = context.kaggle
    return ranker_training_folds(kaggle.run, kaggle.config, kaggle.manifest, window_weeks)


def record_label_access(
    context: DeepFMKaggleContext, folds: tuple[str, ...], purpose: str, script: str
) -> None:
    """Append a row to results/kaggle_deepfm_label_access.csv BEFORE training labels are read.

    Args:
        context: The process's context (run and code state).
        folds: The folds whose labels the step is about to read.
        purpose: What they are read for, in a few words.
        script: The script that reads them.
    """
    row = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": context.stamp.git_commit,
        "git_dirty": str(context.stamp.git_dirty),
        "run": context.run,
        "folds": " ".join(folds),
        "purpose": purpose,
        "script": script,
    }
    append_row(context.paths.kaggle_deepfm_label_access_file, LABEL_ACCESS_COLUMNS, row)
    LOGGER.info("label access recorded: %s", row)
