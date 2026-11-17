"""Tests for hpo.union_check: hit-based unions equal the direct computation, pool rule, R6 files.

The union example is the one of test_union_precheck (fold D1, channels a, b, c):
    labels: q1 -> {10, 11, 12}, q2 -> {20}, q3 -> {30}
    a: q1 -> 10, 11; q2 -> 21      b: q1 -> 11, 40; q2 -> 20      c: q1 -> 12

Active-pool example (fold D1, same labels), pool channel x with A(a) = {10, 21}:
    x: q1 -> 10 (in), 11 (out), 40 (out); q2 -> 21 (in)
    y: q1 -> 11; q2 -> 20            z: q1 -> 12, 40
    rows: 2 in, 2 out; hits: (q1, 10) in, (q1, 11) out.
    Both out-of-pool rows are also returned elsewhere ((q1, 11) by y, (q1, 40) by z), so
    limiting x to A(a) removes nothing from the union: union recall stays q1 = 1, q2 = 1, q3 = 0.
    Without y's (q1, 11): that row is "out-of-pool only" and a hit -> the union loses label 11:
    q1 = {10, 12} = 2/3, a drop of 1/3 on q1 = 1/9 on the fold mean.
"""

import dataclasses
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import polars as pl
import pytest
import torch
from conftest import make_queries, make_recall_inputs
from test_twotower_infer import CHUNKS, LINES, QUERIES, _resources
from test_twotower_infer import CONFIG as R6_CONFIG
from test_union_precheck import (
    D1_CANDIDATES,
    D1_CONTEXT,
    D2_CANDIDATES,
    D2_CONTEXT,
    R1_DEFAULT,
    _candidates,
    _record,
    _selected,
)

from hmrec.config import UnionCheckConfig, load_base_config, load_yaml
from hmrec.eval.metrics import macro_mean
from hmrec.eval.report import FoldEvaluator
from hmrec.hpo.grid import apply_overrides
from hmrec.hpo.records import TrialRecord
from hmrec.hpo.runner import EvaluateFn, FoldContext, build_channel, make_trial_id
from hmrec.hpo.union_check import LOGGER as UNION_LOGGER
from hmrec.hpo.union_check import (
    CandidateFiles,
    R6Source,
    UnionCheckRun,
    apply_variant,
    channel_hits,
    check_reproduces_record,
    count_candidates_chunked,
    flag_active_pool,
    fold_result,
    load_or_generate_r6,
    memory_stage_tracker,
    pool_diagnostics,
    r3_rule_switches,
    r6_fold_checkpoints,
    run_union_check,
    summarize_union_check,
    union_recall_tables_from_hits,
    variant_step,
)
from hmrec.hpo.union_precheck import (
    CandidatesFile,
    SelectedChannel,
    apply_manual_selection,
    count_candidates_per_query,
    precheck_scopes,
    union_recall_tables,
)
from hmrec.recall.twotower.infer import TwoTowerChannel
from hmrec.recall.twotower.train import save_checkpoint

REPO = Path(__file__).resolve().parents[2]
SETTINGS = UnionCheckConfig(
    union_recall_target=0.24, r3_max_union_recall_drop=0.001, manual_selection={}, variants={}
)
POOL = pl.DataFrame({"article_idx": [10, 21]}, schema={"article_idx": pl.Int32})


# ---------------------------------------------------------------- new algorithm = old algorithm


@pytest.mark.parametrize(
    ("candidates", "context"), [(D1_CANDIDATES, D1_CONTEXT), (D2_CANDIDATES, D2_CONTEXT)]
)
def test_unions_from_hits_equal_the_direct_computation(
    candidates: dict[str, pl.DataFrame], context: FoldContext
) -> None:
    direct = union_recall_tables(candidates, context)
    from_hits = union_recall_tables_from_hits(channel_hits(candidates, context), context)
    assert list(from_hits) == list(direct)
    for name, table in direct.items():
        assert from_hits[name].equals(table), name


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 100])
def test_chunked_candidate_counts_equal_the_direct_computation(chunk_size: int) -> None:
    direct = count_candidates_per_query(D1_CANDIDATES, D1_CONTEXT.queries)
    chunked = count_candidates_chunked(D1_CANDIDATES, D1_CONTEXT.queries, chunk_size)
    assert chunked.equals(direct)
    assert chunked["value"].to_list() == [4.0, 2.0, 0.0]


def test_hits_are_much_smaller_than_the_candidates() -> None:
    hits = channel_hits(D1_CANDIDATES, D1_CONTEXT)
    # a: (q1, 10), (q1, 11) hit, (q2, 21) does not.
    assert hits["a"].select("query_id", "article_idx").rows() == [(1, 10), (1, 11)]
    assert hits["c"].height == 1


# ---------------------------------------------------------------- active-pool diagnostics


def _pool_example(y_rows: list[tuple[int, int, int]]) -> dict[str, pl.DataFrame]:
    x = flag_active_pool(_candidates([(1, 10, 1), (1, 11, 2), (1, 40, 3), (2, 21, 1)]), POOL)
    return {"x": x, "y": _candidates(y_rows), "z": _candidates([(1, 12, 1), (1, 40, 2)])}


def test_pool_diagnostics_when_other_channels_cover_the_out_of_pool_rows() -> None:
    candidates = _pool_example([(1, 11, 1), (2, 20, 1)])
    assert candidates["x"]["in_active_pool"].to_list() == [True, False, False, True]
    hits = channel_hits(candidates, D1_CONTEXT)
    result = pool_diagnostics("x", candidates, hits, D1_CONTEXT)
    assert (result.rows_in, result.rows_out) == (2, 2)
    assert (result.hits_in, result.hits_out) == (1, 1)
    assert (result.rows_out_only, result.hits_out_only) == (0, 0)
    assert result.union_without_out["value"].to_list() == [1.0, 1.0, 0.0]


def test_pool_diagnostics_when_an_out_of_pool_hit_is_unique() -> None:
    candidates = _pool_example([(2, 20, 1)])  # y no longer returns (q1, 11)
    hits = channel_hits(candidates, D1_CONTEXT)
    result = pool_diagnostics("x", candidates, hits, D1_CONTEXT)
    assert (result.rows_out_only, result.hits_out_only) == (1, 1)  # (q1, 11); (q1, 40) is in z
    assert result.union_without_out["value"].to_list() == [pytest.approx(2 / 3), 1.0, 0.0]


def test_r3_rule_is_strictly_less_than_the_threshold() -> None:
    assert r3_rule_switches(0.00099, 0.001) is True
    assert r3_rule_switches(0.001, 0.001) is False  # exactly 0.1 pp: R3 stays
    assert r3_rule_switches(0.002, 0.001) is False
    assert r3_rule_switches(0.0, 0.001) is True


# ---------------------------------------------------------------- fold result and summary


def _named(candidates: dict[str, pl.DataFrame], context: FoldContext) -> dict[str, pl.DataFrame]:
    """The a / b / c example with channel a renamed to r3_sibling (it gets the pool flag)."""
    pool = pl.DataFrame({"article_idx": [10, 20, 50]}, schema={"article_idx": pl.Int32})
    return {
        "r3_sibling": flag_active_pool(candidates["a"], pool),
        "b": candidates["b"],
        "c": candidates["c"],
    }


def test_fold_result_and_summary_rows() -> None:
    metrics = {"r3_sibling": "recall@2", "b": "recall@2", "c": "recall@2"}
    results = [
        fold_result(_named(D1_CANDIDATES, D1_CONTEXT), D1_CONTEXT, metrics, 2),
        fold_result(_named(D2_CANDIDATES, D2_CONTEXT), D2_CONTEXT, metrics, 2),
    ]
    assert list(results[0].pools) == ["r3_sibling"]
    # Same union values as the direct computation (channel a under another name).
    direct = union_recall_tables(D1_CANDIDATES, D1_CONTEXT)
    assert results[0].tables.union_recall["all"].equals(direct["all"])
    assert results[0].tables.union_recall["without_r3_sibling"].equals(direct["without_a"])
    scopes = precheck_scopes(["D1", "D2"], {"D1_D2": ("D1", "D2")})
    selected = [_selected("r3_sibling", 0.5), _selected("b", 0.1), _selected("c", 0.1)]
    table = summarize_union_check(results, scopes, selected, SETTINGS)
    rows = {row["measure"]: row for row in table.iter_rows(named=True)}
    assert rows["union_recall_all"]["pooled"] == pytest.approx(3 / 4)
    assert rows["union_recall_target"]["pooled"] == 0.24
    # r3_sibling (= a): D1 rows (q1,10) in, (q1,11) out, (q2,21) out; D2 (q101,50) in.
    assert rows["r3_sibling_out_of_pool_row_share"]["D1"] == pytest.approx(2 / 3)
    assert rows["r3_sibling_out_of_pool_row_share"]["pooled"] == pytest.approx(2 / 4)
    assert rows["r3_sibling_in_pool_hit_rate"]["pooled"] == 1.0  # (q1,10) and (q101,50) hit
    assert rows["r3_sibling_out_of_pool_hit_rate"]["D1"] == pytest.approx(1 / 2)  # (q1,11)
    # (q1, 11) is also returned by b, so only (q2, 21) is out-of-pool only, and it is no hit:
    assert rows["r3_sibling_out_of_pool_only_hits"]["pooled"] == 0.0
    assert rows["union_recall_drop_without_r3_sibling_out_of_pool"]["pooled"] == 0.0
    assert rows["r3_rule_max_union_recall_drop"]["pooled"] == 0.001
    assert rows["r3_rule_switch_to_active_pool_only"]["pooled"] == 1.0  # drop 0 < 0.001
    assert rows["union_recall_target"]["recorded_pooled"] is None


def _choice_with_d1_score(score: float) -> SelectedChannel:
    record = _record("r1_popularity", dict(R1_DEFAULT), score)
    summary = dataclasses.replace(record.summary, fold_scores={"D1": score})
    return SelectedChannel("a", {}, dataclasses.replace(record, summary=summary))


def test_check_reproduces_record_needs_exact_equality() -> None:
    # Channel a's recall@2 on D1: q1 = 2/3, q2 = 0, q3 = 0 -> mean (2/3 + 0 + 0) / 3.
    per_query = D1_CONTEXT.evaluator.per_query(D1_CANDIDATES["a"], "recall@2")
    check_reproduces_record(_choice_with_d1_score((2 / 3 + 0 + 0) / 3), "D1", per_query)
    with pytest.raises(ValueError, match="results/d_phase.csv records"):
        check_reproduces_record(_choice_with_d1_score(0.2222), "D1", per_query)


# ---------------------------------------------------------------- manual selection


def test_manual_selection_needs_an_existing_trial() -> None:
    rule_choice = _selected("r1_popularity", 0.1)
    manual_config = apply_overrides(dict(R1_DEFAULT), {"tau_days": 5.0})
    manual_record = _record("r1_popularity", manual_config, 0.2)
    finished = {record.trial_id: record for record in (rule_choice.record, manual_record)}
    choice = apply_manual_selection(rule_choice, {"tau_days": 5.0}, finished)
    assert choice.record is manual_record
    assert choice.raw_config["tau_days"] == 5.0
    assert make_trial_id("r1_popularity", choice.raw_config) == manual_record.trial_id
    with pytest.raises(ValueError, match="not in results/d_phase.csv"):
        apply_manual_selection(rule_choice, {"tau_days": 6.0}, finished)


# ---------------------------------------------------------------- R6 from saved models


def test_each_fold_uses_the_model_saved_before_it() -> None:
    checkpoints = r6_fold_checkpoints(
        ["D1", "D2", "D3", "D4"], Path("base/epoch_4.pt"), lambda fold: Path(f"chain/{fold}.pt")
    )
    assert checkpoints == {
        "D1": Path("base/epoch_4.pt"),  # the base model predicted D1
        "D2": Path("chain/D1.pt"),  # the model fine-tuned on D1 predicted D2
        "D3": Path("chain/D2.pt"),
        "D4": Path("chain/D3.pt"),  # chain/D4.pt (M4) predicts F1, not D4
    }


def test_r6_candidates_use_the_folds_checkpoint_and_are_written_once(tmp_path: Path) -> None:
    inputs = make_recall_inputs(LINES, 10, active_pool=[0, 1, 3, 4])
    context = FoldContext(D1_CONTEXT.fold, QUERIES, inputs, D1_CONTEXT.evaluator)
    # Save a model whose weights differ from the freshly built one.
    saved = _resources()
    with torch.no_grad():
        saved.model.item_tower.article_embedding.weight[:6].normal_(0.0, 2.0)
    checkpoint = tmp_path / "D1_model.pt"
    save_checkpoint(saved.model, checkpoint, {"model_identity": {}})
    expected = TwoTowerChannel(R6_CONFIG, saved, CHUNKS).generate(QUERIES, inputs)
    source = R6Source(R6_CONFIG, _resources(), CHUNKS, {"D1": checkpoint})
    path = tmp_path / "m4" / "D1" / "r6.parquet"
    assert load_or_generate_r6(context, source, path).equals(expected)
    written_at = path.stat().st_mtime_ns
    # A second call reads the file back (the checkpoint is not even needed).
    source_without_model = dataclasses.replace(source, checkpoints={})
    assert load_or_generate_r6(context, source_without_model, path).equals(expected)
    assert path.stat().st_mtime_ns == written_at


# ---------------------------------------------------------------- one fold at a time (R2 + R3)
#
# Two tiny folds with the REAL R2 and R3 channels (default configs). Catalog: product 500 =
# articles 30, 31, 32; product 600 = 40, 41. Customer 0 bought 30 and 40, customer 1 bought 31.
# R2 returns what the customer bought, R3 the other colours of the same products.


def _real_fold(name: str, anchor_day: int, query_offset: int) -> FoldContext:
    rows = [(0, 30, anchor_day - 30), (0, 40, anchor_day - 1), (1, 31, anchor_day - 5)]
    inputs = make_recall_inputs(
        rows,
        anchor_day,
        product_codes={30: 500, 31: 500, 32: 500, 40: 600, 41: 600},
        active_pool=[30, 31, 40],
    )
    queries = make_queries([(query_offset + 1, 0), (query_offset + 2, 1)], fold=name)
    labels = pl.DataFrame(
        {
            "query_id": [query_offset + 1, query_offset + 1, query_offset + 2],
            "fold": [name] * 3,
            "article_idx": [31, 40, 99],
        },
        schema={"query_id": pl.Int32, "fold": pl.String, "article_idx": pl.Int32},
    )
    fold = dataclasses.replace(D1_CONTEXT.fold, name=name)
    return FoldContext(fold, queries, inputs, FoldEvaluator(name, labels))


REAL_FOLDS = {"D1": _real_fold("D1", 100, 0), "D2": _real_fold("D2", 107, 100)}
BASE = load_base_config(REPO / "configs" / "base.yaml")


def _real_selection(channel: str, metric: str, wrong_score: bool = False) -> SelectedChannel:
    """The channel's default config with a record holding its true fold scores."""
    raw = load_yaml(REPO / "configs" / "recall" / f"{channel}.yaml")
    scores = {}
    for name, context in REAL_FOLDS.items():
        candidates = build_channel(channel, raw, BASE).generate(context.queries, context.inputs)
        scores[name] = macro_mean(context.evaluator.per_query(candidates, metric))
    if wrong_score:
        scores["D2"] += 0.01
    record = _record(channel, dict(raw), 0.0, metric)
    summary = dataclasses.replace(record.summary, fold_scores=scores)
    return SelectedChannel(channel, dict(raw), dataclasses.replace(record, summary=summary))


def _real_run(tmp_path: Path, labels: list[str], wrong_score: bool = False) -> UnionCheckRun:
    selected = [
        _real_selection("r2_repurchase", "recall@30"),
        _real_selection("r3_sibling", "recall@50", wrong_score),
    ]

    def track(label: str) -> AbstractContextManager[object]:
        labels.append(label)
        return nullcontext()

    def file_in(folder: str) -> CandidatesFile:
        return lambda fold, channel, trial: tmp_path / folder / fold / f"{channel}_{trial}.parquet"

    files = CandidateFiles(reuse=file_in("precheck"), write=file_in("m4"))
    return UnionCheckRun(selected, BASE, files, None, track)


def test_run_checks_one_fold_at_a_time_and_is_repeatable(tmp_path: Path) -> None:
    labels: list[str] = []
    loaded: list[str] = []

    def load_context(fold: str) -> FoldContext:
        loaded.append(fold)
        return REAL_FOLDS[fold]

    groups = {"D1_D2": ("D1", "D2")}
    table = run_union_check(
        _real_run(tmp_path, labels), ["D1", "D2"], load_context, SETTINGS, groups
    )
    assert loaded == ["D1", "D2"]
    # Every stage of D1 is tracked before D2 starts: nothing of two folds overlaps.
    first_d2 = next(i for i, label in enumerate(labels) if label.startswith("D2"))
    assert all(label.startswith("D1") for label in labels[:first_d2])
    assert all(label.startswith("D2") for label in labels[first_d2:])
    assert "D1 | r3_sibling candidates" in labels and "D1 | candidates per query" in labels
    assert (tmp_path / "m4" / "D2").exists() and not (tmp_path / "precheck").exists()
    rows = {row["measure"]: row for row in table.iter_rows(named=True)}
    # q1: labels {31, 40}: R2 returns 40 (bought), R3 returns 31 (other colour) -> 2/2;
    # q2: label 99 is never returned -> 0. Mean 1/2 on both folds.
    assert rows["union_recall_all"]["D1"] == 0.5
    assert rows["union_recall_all"]["pooled"] == 0.5
    assert rows["union_recall_drop_without_r3_sibling"]["pooled"] == 0.25
    # A second run reads the candidate files back and gives exactly the same table.
    again = run_union_check(_real_run(tmp_path, []), ["D1", "D2"], load_context, SETTINGS, groups)
    assert again.equals(table)


def test_run_stops_when_a_fold_score_differs_from_the_record(tmp_path: Path) -> None:
    run = _real_run(tmp_path, [], wrong_score=True)
    with pytest.raises(ValueError, match="r3_sibling D2: recall@50"):
        run_union_check(run, ["D1", "D2"], REAL_FOLDS.__getitem__, SETTINGS, {})


def test_r3_rule_rows_are_left_out_once_r3_has_no_out_of_pool_rows() -> None:
    metrics = {"r3_sibling": "recall@2", "b": "recall@2", "c": "recall@2"}
    everything = pl.DataFrame({"article_idx": [10, 11, 21]}, schema={"article_idx": pl.Int32})
    candidates = {
        "r3_sibling": flag_active_pool(D1_CANDIDATES["a"], everything),
        "b": D1_CANDIDATES["b"],
        "c": D1_CANDIDATES["c"],
    }
    results = [fold_result(candidates, D1_CONTEXT, metrics, 2)]
    selected = [_selected("r3_sibling", 0.5), _selected("b", 0.1), _selected("c", 0.1)]
    table = summarize_union_check(results, precheck_scopes(["D1"], {}), selected, SETTINGS)
    measures = table["measure"].to_list()
    assert "r3_sibling_out_of_pool_row_share" in measures
    assert not [measure for measure in measures if measure.startswith("r3_rule_")]


# ---------------------------------------------------------------- variants and stage tracking


def test_apply_variant_replaces_only_the_named_channel() -> None:
    selected = [_selected("r1_popularity", 0.1), _selected("r2_repurchase", 0.2)]
    selected[1] = dataclasses.replace(selected[1], raw_config={"tau_days": 30.0, "quota": 30})
    evaluated: list[tuple[str, str, dict]] = []

    def evaluator_of(channel: str) -> EvaluateFn:
        def evaluate(step: str, raw_config: dict) -> TrialRecord:
            evaluated.append((channel, step, raw_config))
            return _record("r1_popularity", dict(R1_DEFAULT), 0.9)

        return evaluate

    result = apply_variant(selected, {"r2_repurchase": {"tau_days": 14.0}}, evaluator_of, "m4_x")
    assert result[0] is selected[0]
    assert evaluated == [("r2_repurchase", "m4_x", {"tau_days": 14.0, "quota": 30})]
    assert result[1].raw_config["tau_days"] == 14.0 and result[1].record.summary.pooled == 0.9
    assert variant_step("r3_active_pool") == "m4_r3_active_pool"
    with pytest.raises(ValueError, match="not selected"):
        apply_variant(selected, {"r9": {"x": 1}}, evaluator_of, "m4_x")


def test_memory_stage_tracker_logs_each_stage(caplog: pytest.LogCaptureFixture) -> None:
    memory = dataclasses.replace(BASE.memory, sample_interval_seconds=0.01)
    track = memory_stage_tracker(memory)
    with caplog.at_level("INFO", logger=UNION_LOGGER.name), track("D3 | candidates per query"):
        pass
    ends = [r.getMessage() for r in caplog.records if "end rss" in r.getMessage()]
    assert len(ends) == 1 and "stage D3 | candidates per query" in ends[0] and "peak rss" in ends[0]
