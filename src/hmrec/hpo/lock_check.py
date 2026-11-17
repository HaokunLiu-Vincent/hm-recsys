"""M5 check: the lock files must reproduce the recorded D-phase scores (proposal §11 M5).

For ONE channel on ONE D fold (run by scripts/m5_verify_locks.py, one channel per
process so that /usr/bin/time gives the true memory peak of each channel):

  1. the channel is built ONLY from configs/locks/ (hmrec.recall.locked);
  2. its candidates are generated again and written to
     artifacts/m5/lock_check/{fold}/{channel}.parquet. No later step reads that
     folder, so the check can never be mistaken for the F / V / T candidates;
  3. the channel's selection metric on the fold must equal, bit for bit, the
     score of the locked trial in the results file named by the lock
     (results/d_phase.csv, or results/m4_trials.csv for R3). Otherwise the run fails.

Nothing is selected or tuned here: the recorded score is only what the
regenerated candidates are compared with. Labels stay inside eval/ (FoldEvaluator).
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hmrec.config import MemoryConfig
from hmrec.eval.metrics import macro_mean
from hmrec.hpo.grid import PhaseGrid
from hmrec.hpo.records import TrialRecord, read_records, result_columns
from hmrec.hpo.runner import FoldContext, make_trial_id
from hmrec.locks import LockedChannel
from hmrec.logging_utils import get_logger
from hmrec.memory import MemoryTracker, bytes_to_gb
from hmrec.paths import ProjectPaths
from hmrec.recall.base import RecallChannel, write_candidates

LOGGER = get_logger("hpo.lock_check")


@dataclass(frozen=True)
class LockCheckResult:
    """Outcome of the check of one channel on one fold.

    Attributes:
        channel: Channel name.
        fold: Fold name, e.g. "D1".
        trial_id: The locked trial.
        metric: The channel's selection metric, e.g. "recall@100".
        score: The metric of the regenerated candidates on the fold (float64).
        recorded_score: The fold score of the locked trial in the results file.
        n_rows: Candidate rows generated.
        generate_seconds: Wall time of channel.generate (fit + scoring; for R6
            the prediction, the model was loaded before).
        peak_rss_gb: Sampled peak RSS during generate (a lower bound; the true
            peak of the process comes from /usr/bin/time).
    """

    channel: str
    fold: str
    trial_id: str
    metric: str
    score: float
    recorded_score: float
    n_rows: int
    generate_seconds: float
    peak_rss_gb: float


def results_file_of(paths: ProjectPaths, source: str) -> Path:
    """The results file a lock names in d_phase_record.source.

    Raises:
        ValueError: If `source` is not one of the two files that hold locked trials.
    """
    files = {
        "results/d_phase.csv": paths.d_phase_results_file,
        "results/m4_trials.csv": paths.m4_trials_file,
    }
    if source not in files:
        raise ValueError(f"unknown results file {source!r}; expected one of {sorted(files)}")
    return files[source]


def read_recorded_trial(
    paths: ProjectPaths, locked: LockedChannel, folds: Sequence[str], grid: PhaseGrid
) -> TrialRecord:
    """The locked trial's row in the results file the lock names.

    The row must agree with the lock file: the locked config must hash to the
    row's trial id (the id is a hash of the parsed config, so equal ids mean
    equal configs), and metric, pooled score and per-fold scores must be the
    same. So the scores the check compares with are the recorded ones, and the
    lock file's copy of them is not stale.

    Args:
        paths: Project paths.
        locked: The channel's lock.
        folds: The fold names of the results file, e.g. ["D1", "D2", "D3", "D4"].
        grid: configs/hpo/d_phase.yaml (only the column layout of the results file).

    Raises:
        ValueError: If the trial is missing, appears more than once, or differs
            from the lock file.
    """
    groups = list(grid.fold_groups)
    columns = result_columns(folds, groups, grid.extra_metrics)
    path = results_file_of(paths, locked.d_phase_record.source)
    records = read_records(path, columns, folds, groups, grid.extra_metrics)
    matches = [
        record
        for record in records
        if record.channel == locked.channel and record.trial_id == locked.trial_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{locked.channel}: expected exactly one trial {locked.trial_id} in {path}, "
            f"found {len(matches)}"
        )
    record = matches[0]
    lock_score = locked.d_phase_record
    # The ids are compared instead of the two configs: the row's config went through
    # JSON (dates became text), while the trial id was computed from the parsed config.
    same_config = make_trial_id(locked.channel, locked.config) == record.trial_id
    same_scores = (
        record.metric == lock_score.metric
        and record.summary.pooled == lock_score.pooled
        and record.summary.fold_scores == lock_score.folds
    )
    if not (same_config and same_scores):
        raise ValueError(
            f"{locked.channel}: trial {locked.trial_id} in {path} differs from the lock file "
            f"(same config: {same_config}, same scores: {same_scores})"
        )
    return record


def check_locked_channel(
    recall_channel: RecallChannel,
    locked: LockedChannel,
    context: FoldContext,
    recorded_score: float,
    candidates_file: Path,
    memory: MemoryConfig,
) -> LockCheckResult:
    """Regenerate one channel's candidates on one fold and compare its score with the record.

    The candidates are written BEFORE the comparison, so they can be inspected
    when the scores differ. An existing file is never replaced.

    Args:
        recall_channel: The channel, built from the lock files.
        locked: The channel's lock (quota, trial id, metric).
        context: The fold (queries, inputs as of the anchor, label evaluator).
        recorded_score: The fold score of the locked trial in the results file.
        candidates_file: Where the candidates go (artifacts/m5/lock_check/...).
        memory: Memory limit and sampling interval.

    Raises:
        FileExistsError: If `candidates_file` already exists.
        ValueError: If the score is not exactly the recorded one.
    """
    fold = context.fold.name
    if candidates_file.exists():
        raise FileExistsError(f"{candidates_file} already exists; move it away to check again")
    with MemoryTracker(
        f"{locked.channel} {fold} generate",
        LOGGER,
        limit_gb=memory.limit_gb,
        sample_interval_seconds=memory.sample_interval_seconds,
    ) as tracker:
        started = time.perf_counter()
        candidates = recall_channel.generate(context.queries, context.inputs)
        generate_seconds = time.perf_counter() - started
    write_candidates(
        candidates, recall_channel.output_spec, locked.quota, candidates_file, overwrite=False
    )
    metric = locked.d_phase_record.metric
    score = macro_mean(context.evaluator.per_query(candidates, metric))
    result = LockCheckResult(
        channel=locked.channel,
        fold=fold,
        trial_id=locked.trial_id,
        metric=metric,
        score=score,
        recorded_score=recorded_score,
        n_rows=candidates.height,
        generate_seconds=generate_seconds,
        peak_rss_gb=bytes_to_gb(tracker.peak_rss_bytes),
    )
    # Exact comparison on purpose: the same code, config and data must give the
    # same float64 (metrics are summed in a fixed order, docs/decisions.md 2026-11-16).
    if score != recorded_score:
        raise ValueError(
            f"{locked.channel} {fold}: {metric} from the lock files is {score!r}, but trial "
            f"{locked.trial_id} recorded {recorded_score!r}. Candidates: {candidates_file}"
        )
    LOGGER.info(
        "%s %s: %s = %r, identical to the record (trial %s); %d rows, generate %.1f s",
        locked.channel,
        fold,
        metric,
        score,
        locked.trial_id,
        candidates.height,
        generate_seconds,
    )
    return result
