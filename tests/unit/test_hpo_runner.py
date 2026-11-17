"""Tests for the HPO runner: T-fold guard, trial ids, step flow, resume, records, diagnostics."""

import dataclasses
import subprocess
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from conftest import make_queries, make_recall_inputs

from hmrec.config import load_base_config, load_selection_rule, load_yaml
from hmrec.data.folds import Fold
from hmrec.eval.report import FoldEvaluator, MultiFoldSummary
from hmrec.hpo.grid import load_phase_grid, parse_step
from hmrec.hpo.records import (
    TrialRecord,
    append_record,
    read_records,
    record_to_row,
    result_columns,
    row_to_record,
)
from hmrec.hpo.runner import LOGGER as RUNNER_LOGGER
from hmrec.hpo.runner import (
    FoldContext,
    TrialSettings,
    check_tuning_folds,
    evaluate_trial,
    fold_diagnostics,
    git_state,
    make_evaluate_fn,
    make_trial_id,
    run_channel,
    warn_uncommitted_changes,
)
from hmrec.hpo.selection import STOP

REPO = Path(__file__).resolve().parents[2]
RULE = load_selection_rule(REPO / "configs" / "selection_rule.yaml")
GRID = load_phase_grid(REPO / "configs" / "hpo" / "d_phase.yaml")
FOLDS: tuple[str, ...] = ("D1", "D2", "D3", "D4")
GROUPS = {"D1_D2": ("D1", "D2"), "D3_D4": ("D3", "D4")}


def _fold(name: str, phase: str) -> Fold:
    return Fold(name=name, phase=phase, anchor=date(2020, 1, 1), label_end=date(2020, 1, 7))


def _record(
    channel: str, config: dict, fold_scores: tuple[float, ...], step: str = "x"
) -> TrialRecord:
    scores = dict(zip(FOLDS, fold_scores, strict=True))
    pooled = sum(fold_scores) / len(fold_scores)
    summary = MultiFoldSummary(
        scores, dict.fromkeys(FOLDS, 10), pooled, pooled, {"D1_D2": 0.0, "D3_D4": 0.0}
    )
    return TrialRecord(
        phase="D",
        channel=channel,
        step=step,
        trial_id=make_trial_id(channel, config),
        config=config,
        params={},
        metric="recall@100",
        summary=summary,
        extra_pooled={"ndcg@12": 0.1},
        seconds=1.0,
        peak_rss_gb=0.5,
        git_commit="abc",
        git_dirty=False,
        finished_at="2026-11-17T00:00:00",
        diagnostics={"D1": {"n_rows": 3}},
    )


# ---------------------------------------------------------------- guards and ids


def test_t_and_f_folds_are_refused() -> None:
    check_tuning_folds([_fold("D1", "D"), _fold("D2", "D")])
    with pytest.raises(ValueError, match="T1 has phase T"):
        check_tuning_folds([_fold("D1", "D"), _fold("T1", "T")])
    with pytest.raises(ValueError, match="F1 has phase F"):
        check_tuning_folds([_fold("F1", "F")])
    with pytest.raises(ValueError, match="mix phases"):
        check_tuning_folds([_fold("D1", "D"), _fold("V1", "V")])


def test_trial_id_uses_parsed_config() -> None:
    r1 = load_yaml(REPO / "configs" / "recall" / "r1_popularity.yaml")
    assert make_trial_id("r1_popularity", {**r1, "tau_days": 3}) == make_trial_id(
        "r1_popularity", {**r1, "tau_days": 3.0}
    )
    assert make_trial_id("r1_popularity", r1) != make_trial_id(
        "r1_popularity", {**r1, "tau_days": 2.0}
    )


# ---------------------------------------------------------------- step flow


def _r1_scores(config: dict) -> tuple[float, ...]:
    """Fake D-phase scores: G0 is worse; tau 2 passes the rule, tau 4 is about equal but farther."""
    if config["variant"] == "G0":
        return (0.05, 0.05, 0.05, 0.05)
    by_tau = {
        1.0: (0.100, 0.100, 0.100, 0.100),
        2.0: (0.104, 0.104, 0.104, 0.104),
        3.0: (0.100, 0.100, 0.100, 0.100),
        4.0: (0.105, 0.105, 0.105, 0.105),
    }
    return by_tau.get(config["tau_days"], (0.090, 0.090, 0.090, 0.090))


def test_run_channel_chains_steps_and_picks_by_priority() -> None:
    default = load_yaml(REPO / "configs" / "recall" / "r1_popularity.yaml")
    calls: list[tuple[str, dict]] = []

    def evaluate(step: str, config: dict) -> TrialRecord:
        calls.append((step, config))
        return _record("r1_popularity", config, _r1_scores(config), step)

    reports, selected = run_channel(
        "r1_popularity", GRID.channels["r1_popularity"], default, evaluate, RULE
    )
    # default once, G0 once, then 6 tau candidates (tau 3 is the default itself).
    assert [step for step, _ in calls] == ["default", "variant"] + ["tau_days"] * 6
    assert reports[0].decision is not None and reports[0].decision.action == "keep"
    # tau 2 (+4%) and tau 4 (+5%) both pass. 0.104 is 0.95% below 0.105, so they are
    # "about equal"; both are 1 away from the default 3, and the smaller (2) wins.
    assert selected["tau_days"] == 2.0
    assert selected["variant"] == "A1"


def test_requires_skips_tau_after_g0_wins() -> None:
    default = load_yaml(REPO / "configs" / "recall" / "r1_popularity.yaml")

    def evaluate(step: str, config: dict) -> TrialRecord:
        scores = (0.2, 0.2, 0.2, 0.2) if config["variant"] == "G0" else (0.1, 0.1, 0.1, 0.1)
        return _record("r1_popularity", config, scores, step)

    reports, selected = run_channel(
        "r1_popularity", GRID.channels["r1_popularity"], default, evaluate, RULE
    )
    assert selected["variant"] == "G0"
    assert reports[1].applied is False
    assert reports[1].decision is None


def test_convergence_stop_ends_the_channel() -> None:
    steps = (
        parse_step(
            {
                "name": "iterations",
                "kind": "convergence_check",
                "order": "listed",
                "candidates": [{"iterations": 10}, {"iterations": 15}, {"iterations": 25}],
            },
            "t",
        ),
        parse_step(
            {
                "name": "after",
                "kind": "select",
                "order": "listed",
                "candidates": [{"factors": 64}, {"factors": 128}],
            },
            "t",
        ),
    )
    default = load_yaml(REPO / "configs" / "recall" / "r5_wals.yaml")
    seen: list[str] = []

    def evaluate(step: str, config: dict) -> TrialRecord:
        seen.append(step)
        scores = {10: (0.11,) * 4, 15: (0.10,) * 4, 25: (0.10,) * 4}[config["iterations"]]
        return _record("r5_wals", config, scores, step)

    reports, selected = run_channel("r5_wals", steps, default, evaluate, RULE)
    assert reports[0].decision is not None and reports[0].decision.action == STOP
    assert len(reports) == 1  # the "after" step never ran
    assert "after" not in seen
    assert selected["iterations"] == 15


# ---------------------------------------------------------------- resume and records


def test_finished_trials_are_not_recomputed() -> None:
    default = load_yaml(REPO / "configs" / "recall" / "r2_repurchase.yaml")
    done = _record("r2_repurchase", default, (0.1,) * 4, "default")
    runs: list[dict] = []
    saved: list[TrialRecord] = []

    def run_trial(step: str, config: dict) -> TrialRecord:
        runs.append(config)
        return _record("r2_repurchase", config, (0.1,) * 4, step)

    evaluate = make_evaluate_fn("r2_repurchase", {done.trial_id: done}, run_trial, saved.append)
    assert evaluate("default", dict(default)) is done
    other = {**default, "tau_days": 14.0}
    evaluate("tau_days", other)
    evaluate("tau_days", other)  # second call comes from the cache
    assert runs == [other]
    assert [record.trial_id for record in saved] == [make_trial_id("r2_repurchase", other)]


def test_record_round_trip_and_csv(tmp_path: Path) -> None:
    columns = result_columns(FOLDS, list(GROUPS), ["ndcg@12"])
    config = load_yaml(REPO / "configs" / "recall" / "r3_sibling.yaml")
    record = _record("r3_sibling", config, (0.1, 0.2, 0.3, 1 / 3))
    assert row_to_record(record_to_row(record), FOLDS, list(GROUPS), ["ndcg@12"]) == record
    path = tmp_path / "d_phase.csv"
    append_record(path, record, columns)
    append_record(path, record, columns)
    assert read_records(path, columns, FOLDS, list(GROUPS), ["ndcg@12"]) == [record, record]
    with pytest.raises(ValueError, match="has columns"):
        read_records(path, columns[:-1], FOLDS, list(GROUPS), ["ndcg@12"])


# ---------------------------------------------------------------- one real trial on small data

ACTIVE = [20, 21]
ROWS = [(0, 20, 95), (0, 22, 96), (1, 21, 90), (2, 20, 50)]


def _context(name: str, labels: list[tuple[int, int]], offset: int = 0) -> FoldContext:
    """Queries offset+1 -> customer 0, offset+2 -> 1, offset+3 -> 5 (query ids are global)."""
    queries = make_queries([(offset + 1, 0), (offset + 2, 1), (offset + 3, 5)], fold=name)
    labels = [(offset + query, article) for query, article in labels]
    label_table = pl.DataFrame(
        {
            "query_id": [q for q, _ in labels],
            "fold": [name] * len(labels),
            "article_idx": [a for _, a in labels],
        },
        schema={"query_id": pl.Int32, "fold": pl.String, "article_idx": pl.Int32},
    )
    inputs = make_recall_inputs(ROWS, 100, active_pool=ACTIVE)
    return FoldContext(_fold(name, "D"), queries, inputs, FoldEvaluator(name, label_table))


def test_evaluate_trial_on_small_data_writes_pool_flags(tmp_path: Path) -> None:
    # R2 on 4 identical folds. Customer 0 bought 20 (active) and 22 (not active);
    # customer 1 bought 21 (active); customer 5 has nothing -> 1 of 3 queries empty.
    # Labels: q1 -> 22 (an out-of-pool hit), q2 -> 99 (miss), q3 -> 20 (miss).
    contexts = [
        _context(name, [(1, 22), (2, 99), (3, 20)], offset=100 * position)
        for position, name in enumerate(FOLDS)
    ]
    default = load_yaml(REPO / "configs" / "recall" / "r2_repurchase.yaml")
    base = load_base_config(REPO / "configs" / "base.yaml")
    settings = TrialSettings(
        phase="D",
        metric="recall@30",
        extra_metrics=("ndcg@12",),
        fold_groups=GROUPS,
        base=base,
        git_commit="abc",
        git_dirty=True,
        flags_dir=lambda channel, trial_id: tmp_path / channel / trial_id,
    )
    record = evaluate_trial("r2_repurchase", "default", dict(default), default, contexts, settings)
    # Recall@30 per query: q1 = 1, q2 = 0, q3 = 0 -> 1/3 on every fold.
    assert record.summary.fold_scores == dict.fromkeys(FOLDS, pytest.approx(1 / 3))
    assert record.summary.pooled == pytest.approx(1 / 3)
    d1 = record.diagnostics["D1"]
    assert d1["empty_query_share"] == pytest.approx(1 / 3)
    assert (d1["in_pool_rows"], d1["out_of_pool_rows"]) == (2, 1)
    assert (d1["in_pool_hits"], d1["out_of_pool_hits"]) == (0, 1)
    flags = pl.read_parquet(tmp_path / "r2_repurchase" / record.trial_id / "D1.parquet")
    assert flags.sort("article_idx")["in_active_pool"].to_list() == [True, True, False]
    assert "is_hit" not in flags.columns  # labels never leave eval/


def test_fold_diagnostics_r1_has_no_pool_flags() -> None:
    context = _context("D1", [(1, 20)])
    candidates = pl.DataFrame(
        {"query_id": [1], "article_idx": [20], "rank": [1], "score": [1.0], "pop_is_fallback": [0]},
        schema={
            "query_id": pl.Int32,
            "article_idx": pl.Int32,
            "rank": pl.Int32,
            "score": pl.Float32,
            "pop_is_fallback": pl.Int8,
        },
    )
    context = dataclasses.replace(
        context,
        inputs=make_recall_inputs(
            ROWS, 100, {0: "AGE_18_24", 1: "AGE_18_24", 5: "AGE_UNKNOWN"}, active_pool=ACTIVE
        ),
    )
    numbers, flags = fold_diagnostics("r1_popularity", candidates, context)
    assert flags is None
    assert numbers["fallback_rows_by_bucket"] == {"AGE_18_24": 0, "AGE_UNKNOWN": 0}


# ---------------------------------------------------------------- grid boundary


def test_boundary_winner_stops_r4_for_a_human_decision() -> None:
    # alpha 0.9 (the top of {0.5, 0.7, 0.9}) passes the rule -> stop, do not move on.
    default = load_yaml(REPO / "configs" / "recall" / "r4_itemcf.yaml")
    seen: list[str] = []

    def evaluate(step: str, config: dict) -> TrialRecord:
        seen.append(step)
        better = config["variant"] == "CF3" and config["alpha"] == 0.9
        return _record("r4_itemcf", config, (0.11,) * 4 if better else (0.10,) * 4, step)

    reports, selected = run_channel(
        "r4_itemcf", GRID.channels["r4_itemcf"], default, evaluate, RULE
    )
    alpha_report = reports[-1]
    assert alpha_report.step == "alpha"
    assert alpha_report.decision is not None and alpha_report.decision.action == STOP
    assert "grid boundary" in alpha_report.decision.reason
    winner = next(
        r for r in alpha_report.candidates if r.trial_id == alpha_report.decision.winner_trial_id
    )
    assert winner.config["alpha"] == 0.9  # the report names the boundary winner
    assert selected["alpha"] == 0.7  # nothing is changed automatically
    assert "seed_window_x_decay" not in seen


def test_boundary_winner_in_unflagged_step_continues_with_a_note() -> None:
    # R3 gamma 2.0 (top of the widened grid) wins: noted, but the tau step still runs.
    default = load_yaml(REPO / "configs" / "recall" / "r3_sibling.yaml")

    def evaluate(step: str, config: dict) -> TrialRecord:
        scores = (0.11,) * 4 if config["gamma"] == 2.0 else (0.10,) * 4
        return _record("r3_sibling", config, scores, step)

    reports, selected = run_channel(
        "r3_sibling", GRID.channels["r3_sibling"], default, evaluate, RULE
    )
    gamma_decision = reports[0].decision
    assert gamma_decision is not None and gamma_decision.action == "switch"
    assert "grid boundary" in gamma_decision.reason
    assert selected["gamma"] == 2.0
    assert reports[1].step == "tau_days" and reports[1].applied


def test_uncommitted_changes_warning_is_loud(caplog: pytest.LogCaptureFixture) -> None:
    # hmrec loggers may not propagate to the root logger; let caplog see this one.
    old_propagate = RUNNER_LOGGER.propagate
    RUNNER_LOGGER.propagate = True
    try:
        with caplog.at_level("WARNING", logger=RUNNER_LOGGER.name):
            warn_uncommitted_changes("d9ede739b9e42b0d21bfea5cd9928bb243e56056")
    finally:
        RUNNER_LOGGER.propagate = old_propagate
    assert "UNCOMMITTED CHANGES" in caplog.text
    assert "d9ede739b9e4" in caplog.text
    assert "git_dirty=True" in caplog.text


# ---------------------------------------------------------------- git_state


def _git(repo: Path, *args: str) -> str:
    """Run git in a throw-away test repo (identity and signing set per call)."""
    command = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com"]
    command += ["-c", "commit.gpgsign=false", *args]
    return subprocess.run(command, cwd=repo, capture_output=True, text=True, check=True).stdout


def _make_repo(tmp_path: Path) -> Path:
    """A committed repo with src/, configs/ and results/, like the project layout."""
    repo = tmp_path / "repo"
    for relative in ("src/model.py", "configs/model.yaml", "results/d_phase.csv"):
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_text("original\n")
    _git(repo, "init", "--quiet")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "initial")
    return repo


def test_git_state_clean_repo(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    commit, dirty = git_state(repo, repo / "results")
    assert commit == _git(repo, "rev-parse", "HEAD").strip()
    assert dirty is False


def test_git_state_ignores_results(tmp_path: Path) -> None:
    # This is what happens between two channels: the first channel modified
    # results/d_phase.csv and wrote a new, untracked step report.
    repo = _make_repo(tmp_path)
    (repo / "results" / "d_phase.csv").write_text("original\nnew trial\n")
    (repo / "results" / "d_phase_steps").mkdir()
    (repo / "results" / "d_phase_steps" / "r1_popularity.csv").write_text("step\n")
    assert git_state(repo, repo / "results")[1] is False


@pytest.mark.parametrize(
    ("relative", "tracked"),
    [("src/model.py", True), ("configs/model.yaml", True), ("src/new_module.py", False)],
)
def test_git_state_code_or_config_change_is_dirty(
    tmp_path: Path, relative: str, tracked: bool
) -> None:
    # tracked=True: an edit to a committed file; False: a new, untracked file.
    repo = _make_repo(tmp_path)
    assert (repo / relative).exists() is tracked
    (repo / relative).write_text("edited\n")
    # A change under results/ at the same time must not hide the code change.
    (repo / "results" / "d_phase.csv").write_text("new trial\n")
    assert git_state(repo, repo / "results")[1] is True


def test_r6_chain_settings_enter_the_trial_id_only_when_fine_tuning() -> None:
    raw = load_yaml(REPO / "configs" / "recall" / "r6_twotower.yaml")
    frozen = {**raw, "base_epoch": 4}
    # Frozen (e_update 0): seed / replay cannot change the result -> same id, and it is
    # the id recorded for E_base epoch 4 before the chain settings existed.
    assert make_trial_id("r6_twotower", frozen) == "ab2952a3cd9b"
    assert (
        make_trial_id("r6_twotower", {**frozen, "chain_seed": 7, "replay": 0.5}) == "ab2952a3cd9b"
    )
    chain = {**frozen, "e_update": 1}
    assert make_trial_id("r6_twotower", chain) != "ab2952a3cd9b"
    assert make_trial_id("r6_twotower", chain) != make_trial_id(
        "r6_twotower", {**chain, "chain_seed": 43}
    )


def test_r3_active_pool_option_enters_the_trial_id_only_when_on() -> None:
    raw = load_yaml(REPO / "configs" / "recall" / "r3_sibling.yaml")
    assert raw["active_pool_only"] is False
    # Ids recorded in results/d_phase.csv before the option existed: the default
    # config and the D-phase selection (gamma 1.0).
    assert make_trial_id("r3_sibling", raw) == "f50217180db2"
    assert make_trial_id("r3_sibling", {**raw, "gamma": 1.0}) == "3188358fa459"
    # A config stored without the key (old records) is the same trial.
    without_key = {key: value for key, value in raw.items() if key != "active_pool_only"}
    assert make_trial_id("r3_sibling", {**without_key, "gamma": 1.0}) == "3188358fa459"
    switched = make_trial_id("r3_sibling", {**raw, "gamma": 1.0, "active_pool_only": True})
    assert switched not in {"3188358fa459", "f50217180db2"}
