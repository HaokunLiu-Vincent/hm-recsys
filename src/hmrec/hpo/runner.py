"""Run the D-phase steps of one recall channel (proposal §4.3-4.4, §5).

For each channel:
    1. evaluate the default config (configs/recall/*.yaml) on every fold;
    2. for each step of configs/hpo/d_phase.yaml, in order: evaluate the step's
       candidates (built from the CURRENT default), apply the selection rule
       (hpo/selection.py), and make the winner the default of the next step.
Every trial is recomputed on every fold from that fold's own history (never fit
once and reused, proposal §5). Finished trials are appended to
results/d_phase.csv right away; on a re-run, a trial whose trial_id is already
in the file is read back instead of recomputed.

Only D (and later V) folds may reach this code: T folds are refused
(AGENTS.md §3.2: no tuning or selection may ever see T).

The runner never reads labels itself: each fold has an eval.report.FoldEvaluator.
"""

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl

from hmrec.config import (
    BaseConfig,
    SelectionRule,
    load_preprocess_config,
    parse_itemcf_config,
    parse_popularity_config,
    parse_repurchase_config,
    parse_sibling_config,
    parse_twotower_config,
    parse_wals_config,
)
from hmrec.data.folds import Fold
from hmrec.eval.report import IS_HIT, FoldEvaluator, MultiFoldSummary, summarize_folds
from hmrec.hpo.grid import (
    RawConfig,
    Step,
    is_on_boundary,
    ordered_candidate_configs,
    step_applies,
)
from hmrec.hpo.records import TrialRecord
from hmrec.hpo.selection import STOP, StepDecision, choose_winner, convergence_decision
from hmrec.logging_utils import get_logger
from hmrec.memory import MemoryTracker, bytes_to_gb
from hmrec.paths import get_paths
from hmrec.recall.base import ARTICLE_IDX, QUERY_ID, RecallChannel, RecallInputs
from hmrec.recall.build import build_statistical_channel, twotower_chunks
from hmrec.recall.r1_popularity import IS_FALLBACK, fallback_diagnostics
from hmrec.recall.twotower.chain import TwoTowerChainChannel
from hmrec.recall.twotower.data import read_training_array
from hmrec.recall.twotower.infer import TwoTowerChannel, load_twotower_resources
from hmrec.recall.twotower.train import choose_alarm_users

ALLOWED_PHASES = ("D", "V")
# R6 settings that only matter when the model is fine-tuned (e_update > 0).
TWOTOWER_CHAIN_KEYS = ("e_update", "replay", "chain_seed", "chain")
IN_ACTIVE_POOL = "in_active_pool"
POOL_FLAG_CHANNELS = ("r2_repurchase", "r3_sibling")  # channels not limited to A(a) (§2.4)

LOGGER = get_logger("hpo.runner")


# ---------------------------------------------------------------- channels


def build_channel(channel: str, raw_config: Mapping[str, Any], base: BaseConfig) -> RecallChannel:
    """Parse a raw config and build the recall channel (chunk sizes from configs/base.yaml).

    R1-R5 are built by hmrec.recall.build, the same code the locked path uses after M5.
    """
    if channel == "r6_twotower":
        return _build_twotower_channel(raw_config, base)
    return build_statistical_channel(channel, raw_config, base)


def _build_twotower_channel(
    raw_config: Mapping[str, Any], base: BaseConfig
) -> TwoTowerChannel | TwoTowerChainChannel:
    """R6: frozen (e_update = 0) or the weekly training chain (e_update > 0, §6.4).

    Both start from the base checkpoint config.base_epoch. The chain saves its
    weekly models under artifacts/models/twotower/chain/{trial_id}/ and keeps
    the base training's alarm users (drawn again from the base arrays) out.
    """
    config = parse_twotower_config(raw_config)
    paths = get_paths()
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    resources = load_twotower_resources(paths, preprocess, config)
    chunks = twotower_chunks(base)
    if config.e_update == 0:
        return TwoTowerChannel(config, resources, chunks)
    trial_id = make_trial_id("r6_twotower", raw_config)
    base_customers = read_training_array(paths.twotower_data_dir("base"), "customer_idx")
    alarm_customers = choose_alarm_users(
        base_customers, config.training.alarm_user_share, config.training.seed
    )
    metadata = {
        "trial_id": trial_id,
        "config": asdict(config),
        "model_identity": resources.identity,
        "start": f"base epoch {config.base_epoch}",
    }
    return TwoTowerChainChannel(
        config,
        resources,
        chunks,
        partial(paths.twotower_chain_checkpoint_file, trial_id),
        alarm_customers,
        base.memory,
        metadata,
    )


def parsed_config(channel: str, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """The channel's config after parsing and type conversion, as a plain dict."""
    parsers: dict[str, Callable[[Mapping[str, Any]], Any]] = {
        "r1_popularity": parse_popularity_config,
        "r2_repurchase": parse_repurchase_config,
        "r3_sibling": parse_sibling_config,
        "r4_itemcf": parse_itemcf_config,
        "r5_wals": parse_wals_config,
        "r6_twotower": parse_twotower_config,
    }
    if channel not in parsers:
        raise ValueError(f"unknown channel {channel!r}")
    parsed = asdict(parsers[channel](raw_config))
    if channel == "r3_sibling" and not parsed["active_pool_only"]:
        # The option did not exist in the D phase and "off" is exactly the old
        # behaviour, so it is not part of the identity: recorded R3 trial ids stay valid.
        del parsed["active_pool_only"]
    if channel == "r6_twotower" and parsed["e_update"] == 0:
        # Frozen (no fine-tune): the chain settings cannot change the result, so they
        # are not part of the identity. This keeps the E_base trial ids, recorded
        # before the chain settings existed, valid (docs/decisions.md 2026-11-17).
        for key in TWOTOWER_CHAIN_KEYS:
            del parsed[key]
    return parsed


def make_trial_id(channel: str, raw_config: Mapping[str, Any]) -> str:
    """12 hex characters identifying (channel, parsed config).

    The PARSED config is hashed, so 3 and 3.0 give the same id.
    """
    # default=str: R6 has a date (base_window.end_anchor); R1-R5 ids are unchanged.
    text = json.dumps(
        {"channel": channel, "config": parsed_config(channel, raw_config)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def changed_params(default: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Keys whose value differs from the channel's configs/recall/*.yaml."""
    return {key: value for key, value in config.items() if default.get(key) != value}


# ---------------------------------------------------------------- folds


@dataclass(frozen=True)
class FoldContext:
    """Everything needed to evaluate a channel on one fold.

    Attributes:
        fold: The fold (D1 ... D4).
        queries: One row per query; columns query_id, fold, customer_idx.
        inputs: The channel inputs as of the fold's anchor.
        evaluator: Scores candidates against the fold's labels.
    """

    fold: Fold
    queries: pl.DataFrame
    inputs: RecallInputs
    evaluator: FoldEvaluator


def check_tuning_folds(folds: Sequence[Fold]) -> None:
    """Refuse anything but D / V folds: T must never reach a tuning code path (AGENTS.md §3.2)."""
    if not folds:
        raise ValueError("no folds to evaluate")
    for fold in folds:
        if fold.phase not in ALLOWED_PHASES:
            raise ValueError(
                f"fold {fold.name} has phase {fold.phase}; HPO runs only on {list(ALLOWED_PHASES)}"
            )
    if len({fold.phase for fold in folds}) != 1:
        raise ValueError(f"folds mix phases: {[fold.name for fold in folds]}")


UNCOMMITTED_BANNER = "!" * 78


def warn_uncommitted_changes(commit: str) -> None:
    """Log a hard-to-miss warning: results will not match any commit exactly.

    Does not stop the run (AGENTS.md: the user commits). Rows record
    git_dirty=True, so they can be found later.
    """
    LOGGER.warning(UNCOMMITTED_BANNER)
    LOGGER.warning("!! UNCOMMITTED CHANGES in the working tree (HEAD = %s).", commit[:12])
    LOGGER.warning("!! Results are recorded with git_dirty=True: the code that produced them")
    LOGGER.warning("!! is HEAD plus uncommitted edits. Commit first for reproducible results.")
    LOGGER.warning(UNCOMMITTED_BANNER)


def git_state(repo_root: Path, results_root: Path) -> tuple[str, bool]:
    """HEAD commit, and whether the code or configs differ from HEAD.

    "Dirty" means any tracked change or untracked file in the working tree,
    EXCEPT under results_root. results/ holds this runner's own output: after
    one channel finishes, results/d_phase.csv is new or modified, and that must
    not mark the next channel's trials as git_dirty (the code is still HEAD).
    data/ and artifacts/ are in .gitignore, so git never reports them anyway.
    Everything else (src/, configs/, scripts/, docs/, pyproject.toml, ...) counts.

    Args:
        repo_root: Root of the git repository.
        results_root: Folder of committed result tables; must be inside repo_root.

    Returns:
        (HEAD commit hash, True if there are changes outside results_root).
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    # A git "exclude" pathspec: list changes in the whole tree ("."), minus results_root.
    results_relative = results_root.resolve().relative_to(repo_root.resolve()).as_posix()
    status_command = ["git", "status", "--porcelain", "--", ".", f":(exclude){results_relative}"]
    status = subprocess.run(
        status_command, cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    return commit, bool(status.strip())


# ---------------------------------------------------------------- one trial


def fold_diagnostics(
    channel: str, candidates: pl.DataFrame, context: FoldContext
) -> tuple[dict[str, Any], pl.DataFrame | None]:
    """Diagnostic numbers of one fold, plus the pool-flag table for R2 / R3.

    Returns:
        (numbers, flags). numbers always has n_rows, empty_query_share and
        mean_candidates_per_query; R1 adds fallback rows per age bucket; R2 / R3
        add rows, hits and hit rate inside / outside A(a). flags is
        (query_id, article_idx, rank, score, in_active_pool) for R2 / R3, else None.
    """
    n_queries = context.queries.height
    numbers: dict[str, Any] = {
        "n_rows": candidates.height,
        "empty_query_share": 1.0 - candidates[QUERY_ID].n_unique() / n_queries,
        "mean_candidates_per_query": candidates.height / n_queries,
    }
    if channel == "r1_popularity" and IS_FALLBACK in candidates.columns:
        per_bucket = fallback_diagnostics(candidates, context.queries, context.inputs.customers)
        numbers["fallback_rows_by_bucket"] = dict(
            zip(
                per_bucket["age_bucket"].to_list(),
                per_bucket["n_fallback_rows"].to_list(),
                strict=True,
            )
        )
    if channel not in POOL_FLAG_CHANNELS:
        return numbers, None
    pool = context.inputs.active_pool[ARTICLE_IDX].implode()
    flags = candidates.select(QUERY_ID, ARTICLE_IDX, "rank", "score").with_columns(
        pl.col(ARTICLE_IDX).is_in(pool).alias(IN_ACTIVE_POOL)
    )
    with_hits = context.evaluator.mark_hits(flags)
    for inside, name in ((True, "in_pool"), (False, "out_of_pool")):
        part = with_hits.filter(pl.col(IN_ACTIVE_POOL) == inside)
        hits = int(part[IS_HIT].sum())
        numbers[f"{name}_rows"] = part.height
        numbers[f"{name}_hits"] = hits
        numbers[f"{name}_hit_rate"] = hits / part.height if part.height else 0.0
    return numbers, flags


@dataclass(frozen=True)
class TrialSettings:
    """Fixed settings shared by every trial of a run.

    Attributes:
        phase: "D" or "V".
        metric: Selection metric of the channel, e.g. "recall@100".
        extra_metrics: Reported metrics, e.g. ("ndcg@12",).
        fold_groups: Diagnostic fold groups, e.g. {"D1_D2": ("D1", "D2")}.
        base: configs/base.yaml (chunk sizes, threads, memory limit).
        git_commit: HEAD when the run started.
        git_dirty: Uncommitted changes when the run started.
        flags_dir: (channel, trial_id) -> directory for the pool-flag files.
    """

    phase: str
    metric: str
    extra_metrics: tuple[str, ...]
    fold_groups: dict[str, tuple[str, ...]]
    base: BaseConfig
    git_commit: str
    git_dirty: bool
    flags_dir: Callable[[str, str], Path]


def _score_folds(
    channel: str,
    recall_channel: RecallChannel,
    contexts: Sequence[FoldContext],
    settings: TrialSettings,
    trial_id: str,
) -> tuple[dict[str, list[pl.DataFrame]], dict[str, Any]]:
    """Generate candidates fold by fold; return per-query metric tables and diagnostics."""
    metrics = (settings.metric, *settings.extra_metrics)
    per_query: dict[str, list[pl.DataFrame]] = {metric: [] for metric in metrics}
    diagnostics: dict[str, Any] = {}
    for context in contexts:
        candidates = recall_channel.generate(context.queries, context.inputs)
        # One check and one join with the labels for all metrics (not one per metric).
        fold_values = context.evaluator.per_query_many(candidates, metrics)
        for metric in metrics:
            per_query[metric].append(fold_values[metric])
        numbers, flags = fold_diagnostics(channel, candidates, context)
        if isinstance(recall_channel, TwoTowerChainChannel):
            # The chain fine-tunes inside generate(); keep its time / memory per fold.
            numbers["fine_tune"] = asdict(recall_channel.fine_tune_results[context.fold.name])
        diagnostics[context.fold.name] = numbers
        if flags is not None:
            directory = settings.flags_dir(channel, trial_id)
            directory.mkdir(parents=True, exist_ok=True)
            flags.write_parquet(directory / f"{context.fold.name}.parquet")
        LOGGER.info(
            "%s %s %s: %d candidate rows", channel, trial_id, context.fold.name, candidates.height
        )
    return per_query, diagnostics


def evaluate_trial(
    channel: str,
    step: str,
    raw_config: RawConfig,
    default_raw: Mapping[str, Any],
    contexts: Sequence[FoldContext],
    settings: TrialSettings,
) -> TrialRecord:
    """Run one configuration on every fold and summarise it."""
    trial_id = make_trial_id(channel, raw_config)
    recall_channel = build_channel(channel, raw_config, settings.base)
    start = time.perf_counter()
    with MemoryTracker(
        f"{channel} {trial_id}",
        LOGGER,
        limit_gb=settings.base.memory.limit_gb,
        sample_interval_seconds=settings.base.memory.sample_interval_seconds,
    ) as tracker:
        per_query, diagnostics = _score_folds(channel, recall_channel, contexts, settings, trial_id)
    seconds = time.perf_counter() - start
    summaries: dict[str, MultiFoldSummary] = {
        metric: summarize_folds(pl.concat(tables), settings.fold_groups)
        for metric, tables in per_query.items()
    }
    return TrialRecord(
        phase=settings.phase,
        channel=channel,
        step=step,
        trial_id=trial_id,
        config=dict(raw_config),
        params=changed_params(default_raw, raw_config),
        metric=settings.metric,
        summary=summaries[settings.metric],
        extra_pooled={metric: summaries[metric].pooled for metric in settings.extra_metrics},
        seconds=seconds,
        peak_rss_gb=bytes_to_gb(tracker.peak_rss_bytes),
        git_commit=settings.git_commit,
        git_dirty=settings.git_dirty,
        finished_at=datetime.now().isoformat(timespec="seconds"),
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------- steps


@dataclass(frozen=True)
class StepReport:
    """What happened in one step.

    Attributes:
        channel: Channel name.
        step: Step name.
        applied: False if the step's `requires` did not match (step skipped).
        default: The current default at the start of the step.
        candidates: The step's candidates in priority order.
        decision: The selection outcome (None if skipped).
        default_config: Raw config of `default`.
        candidate_configs: Raw configs of `candidates`, same order.
    """

    channel: str
    step: str
    applied: bool
    default: TrialRecord
    candidates: tuple[TrialRecord, ...]
    decision: StepDecision | None
    default_config: RawConfig | None = None
    candidate_configs: tuple[RawConfig, ...] = ()


EvaluateFn = Callable[[str, RawConfig], TrialRecord]


def _decide(
    step: Step,
    default: TrialRecord,
    candidates: list[TrialRecord],
    configs: list[RawConfig],
    current_raw: Mapping[str, Any],
    rule: SelectionRule,
) -> StepDecision:
    """Apply the selection rule (select) or the convergence check (R5 iterations)."""
    if step.kind == "select":
        return choose_winner(default.scores(), [record.scores() for record in candidates], rule)
    key = next(iter(step.candidates[0]))
    shorter = [r for r, c in zip(candidates, configs, strict=True) if c[key] < current_raw[key]]
    longer = [r for r, c in zip(candidates, configs, strict=True) if c[key] > current_raw[key]]
    if len(shorter) != 1 or len(longer) != 1:
        raise ValueError(f"{step.name}: a convergence check needs one shorter and one longer value")
    return convergence_decision(default.scores(), shorter[0].scores(), longer[0].scores(), rule)


def _apply_boundary_rule(
    step: Step, decision: StepDecision, configs: list[RawConfig], candidates: list[TrialRecord]
) -> StepDecision:
    """Handle a winner whose value is on the edge of a one-numeric-parameter step.

    The grid is never widened automatically (docs/decisions.md 2026-11-17):
      - stop_at_boundary steps (R4, R5): the "switch" becomes a "stop", so a human
        decides; the winner is still named in the step report;
      - other numeric steps (e.g. R3 gamma): the switch stands, and the reason
        notes that the value is on the boundary so the report can say so.
    """
    numeric = step.stop_at_boundary or step.order == "nearest_to_current"
    if not numeric or decision.action == STOP:
        return decision
    for record, config in zip(candidates, configs, strict=True):
        if record.trial_id != decision.winner_trial_id or not is_on_boundary(step, config):
            continue
        key = next(iter(step.candidates[0]))
        note = f"{decision.reason}; the selected {key}={config[key]} is on the grid boundary"
        if step.stop_at_boundary:
            reason = f"{note}: stop for a human decision (the range is not widened automatically)"
            return StepDecision(decision.winner_trial_id, STOP, decision.checks, reason)
        LOGGER.warning("%s: %s (range not widened)", step.name, note)
        return StepDecision(decision.winner_trial_id, decision.action, decision.checks, note)
    return decision


def run_channel(
    channel: str,
    steps: Sequence[Step],
    default_raw: Mapping[str, Any],
    evaluate: EvaluateFn,
    rule: SelectionRule,
) -> tuple[list[StepReport], RawConfig]:
    """Run every step of one channel in order (proposal §4.3).

    Args:
        channel: Channel name.
        steps: From configs/hpo/d_phase.yaml.
        default_raw: configs/recall/{channel}.yaml.
        evaluate: (step name, raw config) -> TrialRecord; reads finished trials
            from results/d_phase.csv instead of recomputing them.
        rule: configs/selection_rule.yaml.

    Returns:
        (one report per step, the selected config). Stops early after a
        "stop" decision (R5 convergence check, or a stop_at_boundary step whose
        winner is on the grid edge); the selected config is then the default of
        that step, and the step report names the winner.
    """
    current_raw: RawConfig = dict(default_raw)
    current = evaluate("default", current_raw)
    reports: list[StepReport] = []
    for step in steps:
        if not step_applies(step, current_raw):
            reports.append(StepReport(channel, step.name, False, current, (), None))
            continue
        configs = ordered_candidate_configs(step, current_raw)
        candidates = [evaluate(step.name, config) for config in configs]
        decision = _decide(step, current, candidates, configs, current_raw, rule)
        decision = _apply_boundary_rule(step, decision, configs, candidates)
        reports.append(
            StepReport(
                channel,
                step.name,
                True,
                current,
                tuple(candidates),
                decision,
                dict(current_raw),
                tuple(configs),
            )
        )
        LOGGER.info("%s step %s: %s (%s)", channel, step.name, decision.action, decision.reason)
        if decision.action == STOP:
            break
        for record, config in zip(candidates, configs, strict=True):
            if record.trial_id == decision.winner_trial_id:
                current, current_raw = record, config
    return reports, current_raw


def make_evaluate_fn(
    channel: str,
    finished: Mapping[str, TrialRecord],
    run_trial: Callable[[str, RawConfig], TrialRecord],
    save: Callable[[TrialRecord], None],
) -> EvaluateFn:
    """Wrap `run_trial` so finished trials are reused and new ones are saved at once.

    Args:
        channel: Channel name.
        finished: trial_id -> record already in results/d_phase.csv.
        run_trial: Actually evaluates a config (evaluate_trial with fixed arguments).
        save: Appends a new record to results/d_phase.csv.
    """
    cache = dict(finished)

    def evaluate(step: str, raw_config: RawConfig) -> TrialRecord:
        trial_id = make_trial_id(channel, raw_config)
        if trial_id in cache:
            LOGGER.info("%s %s: already in the results file, skipped", channel, trial_id)
            return cache[trial_id]
        record = run_trial(step, raw_config)
        save(record)
        cache[trial_id] = record
        return record

    return evaluate
