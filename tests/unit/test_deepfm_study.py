"""Tests for deepfm.study and deepfm.study_run: the rules on hand-made rows, then a tiny study.

Hand example for the scores (two folds, V1 with 100 queries and V2 with 300; one
configuration "c2" with 2 epochs, two seeds):
    seed 42: V1 0.20, V2 0.10 -> pooled (100 x 0.20 + 300 x 0.10) / 400 = 0.125
    seed 43: V1 0.24, V2 0.12 -> pooled (24 + 36) / 400 = 0.15
    configuration: pooled = (0.125 + 0.15) / 2 = 0.1375; V1 0.22, V2 0.11;
    seed range = (0.15 - 0.125) / 0.125 = 0.2.

Hand example for the epoch rule (the same settings with T = 1, 2, 3 epochs, seed 42):
    T = 1: pooled 0.125;  T = 2: pooled 0.145;  T = 3: pooled 0.142875 (1.47% below the best)
    with a 1% tolerance only T = 2 is "about equal" to the best -> T* = 2;
    with a 2% tolerance T = 2 and T = 3 are, and the smallest wins -> still 2;
    with a 20% tolerance T = 1 is too (13.8% below) -> T* = 1, on the edge of the grid.
"""

import csv
import dataclasses
import shutil
import time
from pathlib import Path

import pytest
import torch
from deepfm_example import REPO, STUDY_TARGETS, build_study_workspace

from hmrec.config import (
    DeepFMStudyStep,
    load_deepfm_config,
    load_deepfm_study_config,
    load_selection_rule,
)
from hmrec.csv_records import append_row, read_rows
from hmrec.deepfm.process import record_trial, with_seed
from hmrec.deepfm.records import TRIAL_COLUMNS, read_pooled_reference
from hmrec.deepfm.study import (
    WITHIN_SEED_NOISE,
    ConfigurationScore,
    StudyStop,
    Variant,
    ablation_variants,
    apply_variant,
    check_above_baseline,
    choose_schedule,
    decide_step,
    listwise_start,
    listwise_step,
    noise_mark,
    parameter_value,
    pooled_value,
    score_configuration,
    step_variants,
    tuning_seeds,
    variant_config_id,
    variant_of,
    with_parameter,
)
from hmrec.deepfm.study_run import (
    AREA_V_PHASE,
    STAGES,
    STUDY_COLUMNS,
    StudyContext,
    StudyResult,
    TrialRequest,
    ensure_trials,
    recorded_trials,
    run_study,
    study_rows,
    trial_arguments,
    write_study_table,
)
from hmrec.deepfm.trial import TrialInputs, TrialSpec, run_trial
from hmrec.locks import load_ranker_lock
from hmrec.memory import ProcessPeakMemory
from hmrec.run_records import RunStamp

RULE = load_selection_rule(REPO / "configs" / "selection_rule.yaml")
STUDY = load_deepfm_study_config(REPO / "configs" / "hpo" / "deepfm.yaml")
REAL_CONFIG = load_deepfm_config(REPO / "configs" / "deepfm.yaml")
FOLDS: tuple[str, ...] = ("V1", "V2", "V3", "V4")
DEFAULT = Variant(0.001, 16, 0.0, "full", "binned", "pointwise_bce", 3)
STAMP = RunStamp("abc123", False)
N_QUERIES = {"V1": 100, "V2": 300, "V3": 100, "V4": 300}


def _rows(config_id: str, seed: int, epochs: int, values: dict[str, float]):
    """Metric rows of one finished configuration and seed: fold -> MAP@12 of the final model."""
    return [
        {
            "area": "v_phase",
            "config_id": config_id,
            "seed": str(seed),
            "fold": fold,
            "epoch": str(epochs),
            "n_queries": str(N_QUERIES[fold]),
            "map@12": repr(value),
            "train_loss": repr(1.0 / epochs),
        }
        for fold, value in values.items()
    ]


TWO_SEEDS = _rows("c2", 42, 2, {"V1": 0.20, "V2": 0.10}) + _rows(
    "c2", 43, 2, {"V1": 0.24, "V2": 0.12}
)
SCHEDULE_ROWS = (
    _rows("t1", 42, 1, {"V1": 0.20, "V2": 0.10})
    + _rows("t2", 42, 2, {"V1": 0.22, "V2": 0.12})
    + _rows("t3", 42, 3, {"V1": 0.21, "V2": 0.1205})
)


def _score(rows, config_id: str, epochs: int, seeds=(42,), folds=("V1", "V2")):
    return score_configuration(rows, "v_phase", config_id, epochs, seeds, folds, "map@12")


# ---------------------------------------------------------------- variants


def test_the_real_configs_give_the_default_variant_and_the_recorded_rules() -> None:
    assert variant_of(REAL_CONFIG) == DEFAULT
    assert STUDY.schedule_epochs == (1, 2, 3) and STUDY.epoch_tolerance == 0.01
    assert (STUDY.noise_seeds, STUDY.max_seed_range, STUDY.selection_seed) == (
        (42, 43, 44),
        0.01,
        42,
    )
    assert [(step.parameter, step.candidates) for step in STUDY.steps] == [
        ("learning_rate", (0.001, 0.0003, 0.003)),
        ("embedding_dim", (16.0, 8.0, 32.0)),
        ("dropout", (0.0, 0.1, 0.3)),
    ]
    assert STUDY.listwise_learning_rates == (0.001, 0.0003, 0.003)
    assert STUDY.ablation_components == ("first_order", "fm", "deep", "full")
    assert STUDY.ablation_numeric_inputs == ("standardised",)
    assert STUDY.final_seeds == (42, 43, 44) and not STUDY.extend_grid_on_edge


def test_a_variant_changes_only_its_seven_settings() -> None:
    variant = Variant(0.003, 8, 0.1, "fm", "standardised", "listwise_softmax", 1)
    config = apply_variant(REAL_CONFIG, variant)
    assert variant_of(config) == variant
    assert config.training.batch_rows == REAL_CONFIG.training.batch_rows
    assert config.model.hidden_units == REAL_CONFIG.model.hidden_units
    assert apply_variant(REAL_CONFIG, DEFAULT) == REAL_CONFIG
    # Every setting is part of the configuration id; the epochs too (cosine schedule).
    ids = {variant_config_id(REAL_CONFIG, DEFAULT)}
    for name, value in dataclasses.asdict(variant).items():
        ids.add(variant_config_id(REAL_CONFIG, dataclasses.replace(DEFAULT, **{name: value})))
    assert len(ids) == 8


def test_with_parameter_sets_one_tuned_value() -> None:
    assert with_parameter(DEFAULT, "learning_rate", 0.003).learning_rate == 0.003
    assert with_parameter(DEFAULT, "embedding_dim", 32.0).embedding_dim == 32
    assert with_parameter(DEFAULT, "dropout", 0.3).dropout == 0.3
    assert parameter_value(DEFAULT, "embedding_dim") == 16.0
    with pytest.raises(ValueError, match="whole number"):
        with_parameter(DEFAULT, "embedding_dim", 12.5)
    with pytest.raises(ValueError, match="not a tuned parameter"):
        with_parameter(DEFAULT, "components", 1.0)
    with pytest.raises(ValueError, match="not a tuned parameter"):
        parameter_value(DEFAULT, "epochs")


# ---------------------------------------------------------------- scores


def test_scores_of_one_configuration_hand_computed() -> None:
    assert pooled_value({"V1": 0.20, "V2": 0.10}, N_QUERIES) == pytest.approx(0.125)
    score = _score(TWO_SEEDS, "c2", 2, seeds=(42, 43))
    assert [seed.seed for seed in score.seeds] == [42, 43] and score.epochs == 2
    assert [seed.pooled for seed in score.seeds] == pytest.approx([0.125, 0.15])
    assert score.pooled == pytest.approx(0.1375)
    assert score.fold_scores == pytest.approx({"V1": 0.22, "V2": 0.11})
    assert score.seed_range == pytest.approx(0.2)
    assert score.seeds[0].train_loss == pytest.approx(0.5)
    trial = score.trial_scores()
    assert (trial.trial_id, trial.pooled_score) == ("c2", pytest.approx(0.1375))
    # One seed: the means are that seed's values, and there is no range.
    single = score.only_seed(43)
    assert single.pooled == pytest.approx(0.15) and single.fold_scores["V1"] == 0.24
    with pytest.raises(ValueError, match="at least two"):
        _ = single.seed_range
    with pytest.raises(ValueError, match="not among its seeds"):
        score.only_seed(44)


def test_scores_need_every_fold_once_and_only_the_final_epoch() -> None:
    with pytest.raises(ValueError, match="recorded folds"):
        _score(TWO_SEEDS, "c2", 2, folds=("V1", "V2", "V3"))
    with pytest.raises(ValueError, match="recorded folds"):
        _score(TWO_SEEDS, "c2", 2, seeds=(44,))
    with pytest.raises(ValueError, match="recorded twice"):
        _score(TWO_SEEDS + _rows("c2", 42, 2, {"V1": 0.3}), "c2", 2)
    # A row of another epoch under this id means the id and the run do not belong together.
    with pytest.raises(ValueError, match="has 2 epochs"):
        _score(TWO_SEEDS + _rows("c2", 42, 1, {"V1": 0.3}), "c2", 2)
    with pytest.raises(ValueError, match="at least one seed"):
        _score(TWO_SEEDS, "c2", 2, seeds=())
    # Rows of other areas and other configurations are not this configuration's rows.
    other_area = [{**row, "area": "speed_test"} for row in _rows("c2", 42, 2, {"V1": 0.9})]
    assert _score(TWO_SEEDS + other_area, "c2", 2).pooled == pytest.approx(0.125)


def _schedule(tolerance: float):
    scores = {epochs: _score(SCHEDULE_ROWS, f"t{epochs}", epochs) for epochs in (1, 2, 3)}
    return choose_schedule(scores, tolerance)


def test_epoch_rule_takes_the_smallest_schedule_within_the_tolerance() -> None:
    choice = _schedule(0.01)
    assert [score.pooled for score in choice.scores.values()] == pytest.approx(
        [0.125, 0.145, 0.142875]
    )
    assert (choice.selected_epochs, choice.best_epochs, choice.on_grid_edge) == (2, 2, False)
    assert choice.selected.config_id == "t2"
    assert _schedule(0.02).selected_epochs == 2
    wide = _schedule(0.2)
    assert (wide.selected_epochs, wide.best_epochs, wide.on_grid_edge) == (1, 2, True)


def test_epoch_rule_uses_the_seed_mean() -> None:
    # T = 1: seeds 0.100 and 0.120 -> mean 0.110.  T = 2: seeds 0.118 and 0.104 -> mean 0.111.
    # Seed 42 alone would pick T = 2 (0.118 against 0.100); the mean is within 1% -> T = 1.
    rows = (
        _rows("a1", 42, 1, {"V1": 0.100, "V2": 0.100})
        + _rows("a1", 43, 1, {"V1": 0.120, "V2": 0.120})
        + _rows("a2", 42, 2, {"V1": 0.118, "V2": 0.118})
        + _rows("a2", 43, 2, {"V1": 0.104, "V2": 0.104})
    )
    both = {t: _score(rows, f"a{t}", t, seeds=(42, 43)) for t in (1, 2)}
    assert choose_schedule(both, 0.01).selected_epochs == 1
    only_42 = {t: _score(rows, f"a{t}", t) for t in (1, 2)}
    assert choose_schedule(only_42, 0.01).selected_epochs == 2


# ---------------------------------------------------------------- step 0


def test_noise_rule_decides_the_seeds_of_tuning_and_never_stops() -> None:
    assert tuning_seeds(STUDY, 0.0099) == (42,)
    # 1% or more: every candidate runs with all noise seeds (no stop).
    assert tuning_seeds(STUDY, 0.01) == (42, 43, 44)
    assert tuning_seeds(STUDY, 0.0185) == (42, 43, 44)


def test_default_below_the_baseline_stops_the_study() -> None:
    score = _score(TWO_SEEDS, "c2", 2, seeds=(42, 43))  # seed-mean pooled 0.1375
    check_above_baseline(score, 0.1375)
    check_above_baseline(score, 0.13)
    with pytest.raises(StudyStop, match="below the RRF baseline"):
        check_above_baseline(score, 0.14)


# ---------------------------------------------------------------- tuning

LEARNING_RATE_STEP = STUDY.steps[0]


def _step_scores() -> tuple[dict[float, Variant], dict[float, ConfigurationScore]]:
    """A learning-rate step whose candidates were selected at T = 2, 3 and 1.

    default 0.001 (T = 2): 0.100 in every fold                 -> pooled 0.100
    0.0003 (T = 3): 0.102, 0.102, 0.102, 0.099                  -> pooled 0.100875 (+0.875%)
    0.003  (T = 1): 0.103, 0.103, 0.103, 0.100                  -> pooled 0.101875 (+1.875%)
    (queries 100, 300, 100, 300)
    """
    values = {
        0.001: (2, {"V1": 0.100, "V2": 0.100, "V3": 0.100, "V4": 0.100}),
        0.0003: (3, {"V1": 0.102, "V2": 0.102, "V3": 0.102, "V4": 0.099}),
        0.003: (1, {"V1": 0.103, "V2": 0.103, "V3": 0.103, "V4": 0.100}),
    }
    variants, scores = {}, {}
    for rate, (epochs, by_fold) in values.items():
        variants[rate] = dataclasses.replace(DEFAULT, learning_rate=rate, epochs=epochs)
        config_id = f"lr{rate}"
        scores[rate] = _score(_rows(config_id, 42, epochs, by_fold), config_id, epochs, folds=FOLDS)
    return variants, scores


def test_step_selects_by_the_rule_and_carries_the_schedule_of_the_winner() -> None:
    variants, scores = _step_scores()
    assert [scores[rate].pooled for rate in (0.001, 0.0003, 0.003)] == pytest.approx(
        [0.100, 0.100875, 0.101875]
    )
    outcome = decide_step(LEARNING_RATE_STEP, DEFAULT, variants, scores, RULE)
    checks = {check.trial_id: check for check in outcome.decision.checks}
    # 0.0003: better in 3 folds but only +0.875% -> fails. 0.003: +1.875% and 3 folds -> passes.
    assert (checks["lr0.0003"].folds_better, checks["lr0.0003"].passes) == (3, False)
    assert (checks["lr0.003"].folds_better, checks["lr0.003"].passes) == (3, True)
    assert outcome.selected_value == 0.003 and outcome.on_grid_edge
    # The winner goes on WITH its own schedule length.
    assert outcome.selected == dataclasses.replace(DEFAULT, learning_rate=0.003, epochs=1)
    assert outcome.schedules == {}


def test_step_keeps_the_default_when_no_candidate_passes() -> None:
    variants, scores = _step_scores()
    weaker = _score(
        _rows("lr0.003", 42, 1, {"V1": 0.103, "V2": 0.100, "V3": 0.103, "V4": 0.100}),
        "lr0.003",
        1,
        folds=FOLDS,
    )  # better in 2 folds only
    outcome = decide_step(LEARNING_RATE_STEP, DEFAULT, variants, {**scores, 0.003: weaker}, RULE)
    assert outcome.selected_value == 0.001 and not outcome.on_grid_edge
    assert outcome.selected.epochs == 2  # the default's own schedule length
    with pytest.raises(ValueError, match="same configuration id"):
        decide_step(LEARNING_RATE_STEP, DEFAULT, variants, {**scores, 0.003: scores[0.001]}, RULE)


def test_a_step_starts_from_the_current_configuration() -> None:
    variants = step_variants(LEARNING_RATE_STEP, DEFAULT)
    assert list(variants) == [0.001, 0.0003, 0.003]
    assert variants[0.003] == dataclasses.replace(DEFAULT, learning_rate=0.003)
    moved = dataclasses.replace(DEFAULT, learning_rate=0.003)
    with pytest.raises(ValueError, match="is not the current value"):
        step_variants(LEARNING_RATE_STEP, moved)


def test_listwise_arm_starts_from_the_pointwise_selection() -> None:
    pointwise = Variant(0.0003, 32, 0.1, "full", "binned", "pointwise_bce", 2)
    start = listwise_start(STUDY, pointwise)
    # The loss changes, the learning rate restarts at the first value of its own grid;
    # the embedding dimension and dropout are those selected for the pointwise arm.
    assert (start.loss, start.learning_rate) == ("listwise_softmax", 0.001)
    assert (start.embedding_dim, start.dropout) == (32, 0.1)
    step = listwise_step(STUDY)
    assert (step.parameter, step.candidates) == ("learning_rate", (0.001, 0.0003, 0.003))
    assert list(step_variants(step, start)) == [0.001, 0.0003, 0.003]


def test_ablations_change_one_setting_and_keep_the_schedule() -> None:
    selected = Variant(0.003, 16, 0.1, "full", "binned", "pointwise_bce", 2)
    variants = ablation_variants(STUDY, selected)
    assert list(variants) == [
        "components=first_order",
        "components=fm",
        "components=deep",
        "numeric_input=standardised",
    ]
    assert all(
        (v.learning_rate, v.epochs, v.loss) == (0.003, 2, "pointwise_bce")
        for v in variants.values()
    )
    assert variants["components=fm"] == dataclasses.replace(selected, components="fm")
    assert variants["numeric_input=standardised"].components == "full"


def test_noise_mark_compares_the_change_with_the_seed_range() -> None:
    # Seed range 1.2%: a change of -1.0% is within the noise, -1.5% is not.
    assert noise_mark(0.0300, 0.0297, 0.012) == WITHIN_SEED_NOISE
    assert noise_mark(0.0300, 0.0303, 0.012) == WITHIN_SEED_NOISE
    assert noise_mark(0.0300, 0.02955, 0.012) == ""
    assert noise_mark(0.0300, 0.0200, 0.012) == ""


def test_trial_arguments_carry_the_whole_configuration() -> None:
    variant = dataclasses.replace(DEFAULT, dropout=0.1, epochs=2)
    assert trial_arguments(TrialRequest("v_phase", "V3", 44, variant)) == [
        "--area", "v_phase", "--target", "V3", "--seed", "44", "--epochs", "2",
        "--learning-rate", "0.001", "--embedding-dim", "16", "--dropout", "0.1",
        "--components", "full", "--numeric-input", "binned", "--loss", "pointwise_bce",
    ]  # fmt: skip


# ---------------------------------------------------------------- a tiny study, end to end


def _write_reference(inputs: TrialInputs) -> None:
    """The real ranker lock and RRF / LightGBM rows for V1 ... V4 (30 queries each)."""
    paths = inputs.paths
    paths.locks_dir.mkdir(parents=True)
    shutil.copyfile(REPO / "configs" / "locks" / "ranker.lock.yaml", paths.ranker_lock_file)
    lock = load_ranker_lock(paths.ranker_lock_file)
    header = ["experiment", "config_id", "fold", "num_trees", "n_queries", "map@12"]
    rows = [["rrf_baseline", "rrf_baseline", fold, "0", "30", "0.05"] for fold in STUDY_TARGETS]
    rows += [
        ["full", lock.selection.config_id, fold, str(lock.model.num_trees), "30", "0.5"]
        for fold in STUDY_TARGETS
    ]
    paths.results_root.mkdir(parents=True, exist_ok=True)
    with paths.v_phase_metrics_file.open("w", newline="") as handle:
        csv.writer(handle).writerows([header, *rows])


def _context(inputs: TrialInputs, calls: list[TrialRequest], max_seed_range: float) -> StudyContext:
    """A study on the tiny workspace whose trainings run in this process."""
    peak = ProcessPeakMemory(1024**3, 0)

    def launch(request: TrialRequest) -> None:
        calls.append(request)
        config = with_seed(apply_variant(inputs.config, request.variant), request.seed)
        trial_inputs = dataclasses.replace(inputs, config=config)
        spec = TrialSpec(request.area, request.target_fold, request.seed, request.variant.epochs)
        result = run_trial(trial_inputs, spec)
        record_trial(trial_inputs, spec, result, time.perf_counter(), peak, STAMP)

    study = dataclasses.replace(
        STUDY,
        noise_seeds=(42, 43),
        schedule_epochs=(1, 2),
        max_seed_range=max_seed_range,
        steps=(
            DeepFMStudyStep("learning_rate", "learning_rate", (0.001, 0.003)),
            DeepFMStudyStep("dropout", "dropout", (0.0, 0.1)),
        ),
        listwise_learning_rates=(0.001, 0.003),
        ablation_components=("fm", "full"),
        ablation_numeric_inputs=("standardised",),
        final_seeds=(42, 43),
    )
    return StudyContext(inputs.paths, inputs.config, study, RULE, launch)


def _keys(calls: list[TrialRequest]) -> set[tuple[Variant, int, str]]:
    return {(call.variant, call.seed, call.target_fold) for call in calls}


def _check_single_seed_study(
    result: StudyResult, calls: list[TrialRequest], default: Variant
) -> None:
    """The trainings and decisions of the tiny study when tuning uses the selection seed only."""
    step0 = result.step0
    assert result.tuning is not None and result.pointwise is not None
    assert result.listwise_step is not None and result.listwise is not None
    assert result.pointwise_variant is not None and result.listwise_variant is not None
    assert result.ablations is not None
    # ---- step 0: the default with 2 schedule lengths x 2 seeds x 4 folds
    assert list(step0.schedule.scores) == [1, 2] and step0.tuning_seeds == (42,)
    assert not step0.multi_seed and step0.baseline == pytest.approx(0.05)
    step0_calls = calls[:16]
    assert _keys(step0_calls) == {
        (dataclasses.replace(default, epochs=epochs), seed, fold)
        for epochs in (1, 2)
        for seed in (42, 43)
        for fold in STUDY_TARGETS
    }
    selected = step0.schedule.selected
    assert [seed.seed for seed in selected.seeds] == [42, 43]
    assert step0.seed_range == pytest.approx(selected.seed_range)
    # ---- learning rate: every candidate with both schedule lengths, seed 42 only
    rate_step, dropout_step = result.tuning
    assert sorted(rate_step.schedules) == [0.001, 0.003]
    for value, schedule in rate_step.schedules.items():
        assert list(schedule.scores) == [1, 2]
        assert rate_step.variants[value].epochs == schedule.selected_epochs
        assert rate_step.scores[value] == schedule.selected
        assert [seed.seed for seed in schedule.selected.seeds] == [42]
    faster = dataclasses.replace(default, learning_rate=0.003)
    rate_calls = calls[16:24]
    assert _keys(rate_calls) == {
        (dataclasses.replace(faster, epochs=epochs), 42, fold)
        for epochs in (1, 2)
        for fold in STUDY_TARGETS
    }
    # ---- dropout: at the schedule length that came with the learning rate, seed 42 only
    after_rate = rate_step.selected
    assert dropout_step.start == after_rate and dropout_step.schedules == {}
    assert {variant.epochs for variant in dropout_step.variants.values()} == {after_rate.epochs}
    assert _keys(calls[24:28]) == {
        (dataclasses.replace(after_rate, dropout=0.1), 42, fold) for fold in STUDY_TARGETS
    }
    assert result.pointwise_variant == dropout_step.selected
    # ---- the final seeds of the pointwise arm
    assert [seed.seed for seed in result.pointwise.seeds] == [42, 43]
    assert result.pointwise.epochs == result.pointwise_variant.epochs
    # ---- listwise: its own learning-rate step from the pointwise selection
    start = dataclasses.replace(
        result.pointwise_variant, loss="listwise_softmax", learning_rate=0.001
    )
    assert result.listwise_step.start == start
    assert sorted(result.listwise_step.schedules) == [0.001, 0.003]
    assert result.listwise_variant.loss == "listwise_softmax"
    assert result.listwise_variant.dropout == result.pointwise_variant.dropout
    assert [seed.seed for seed in result.listwise.seeds] == [42, 43]
    # ---- ablations: seed 42, the pointwise selection's schedule length
    assert list(result.ablations) == ["components=fm", "numeric_input=standardised"]
    for score in result.ablations.values():
        assert score.epochs == result.pointwise_variant.epochs
        assert [seed.seed for seed in score.seeds] == [42]
    # Every training ran once, in the area of the study.
    assert len(_keys(calls)) == len(calls) and {call.area for call in calls} == {AREA_V_PHASE}


def test_tiny_study_runs_every_stage_and_continues_where_it_stopped(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    inputs = build_study_workspace(tmp_path)
    _write_reference(inputs)
    calls: list[TrialRequest] = []
    # A huge limit: the tiny models are noisy, and this test is about the single-seed branch.
    context = _context(inputs, calls, max_seed_range=10.0)
    reference = read_pooled_reference(inputs.paths, STUDY_TARGETS, "map@12")
    assert (reference.rrf_baseline, reference.lightgbm) == pytest.approx((0.05, 0.5))

    # Stage by stage: a later stage does not repeat the trainings of an earlier one.
    only_step0 = run_study(context, "step0")
    assert len(calls) == 16 and only_step0.tuning is None and only_step0.pointwise is None
    result = run_study(context, STAGES[-1])
    _check_single_seed_study(result, calls, context.default)
    n_trainings = len(calls)
    assert len(recorded_trials(inputs.paths)) == n_trainings
    again = run_study(context, STAGES[-1])
    assert len(calls) == n_trainings and again == result
    with pytest.raises(ValueError, match="through must be one of"):
        run_study(context, "final")

    # ---- the study table
    assert result.pointwise is not None and result.ablations is not None
    with pytest.raises(ValueError, match="run it through ablations"):
        study_rows(only_step0, 42, reference.lightgbm)
    rows = study_rows(result, 42, reference.lightgbm)
    write_study_table(inputs.paths, rows, STAMP)
    written = read_rows(inputs.paths.deepfm_study_file, STUDY_COLUMNS)
    stages = [row["stage"] for row in written]
    assert stages == (
        ["step0"] * 2
        + ["tune:learning_rate:schedule"] * 4
        + ["tune:learning_rate"] * 2
        + ["tune:dropout"] * 2
        + ["final"]
        + ["tune:listwise_learning_rate:schedule"] * 4
        + ["tune:listwise_learning_rate"] * 2
        + ["final"]
        + ["ablation"] * 2
    )
    by_stage = {stage: [row for row in written if row["stage"] == stage] for stage in set(stages)}
    assert [row["selected"] for row in by_stage["step0"]].count("True") == 1
    assert [row["selected"] for row in by_stage["tune:learning_rate"]].count("True") == 1
    assert {row["arm"] for row in by_stage["final"]} == {"deepfm_pointwise", "deepfm_listwise"}
    assert all(row["seeds"] == "42 43" and row["seed_range"] for row in by_stage["final"])
    assert all(row["git_commit"] == "abc123" for row in written)
    # An ablation is compared with the full model of the SAME seed; the mark follows the range.
    full = result.pointwise.only_seed(42)
    for row in by_stage["ablation"]:
        change = float(row["relative_change"])
        ablation = result.ablations[row["name"]]
        assert change == pytest.approx(ablation.pooled / full.pooled - 1)
        marked = row["note"].startswith(WITHIN_SEED_NOISE)
        assert marked == (abs(change) < result.step0.seed_range)
        assert "relative to LightGBM" in row["note"] and row["reference"] == full.config_id
    with pytest.raises(FileExistsError, match="never replaced"):
        write_study_table(inputs.paths, rows, STAMP)

    # ---- the second branch of the noise rule, on the same records: with a limit below the
    # observed range every tuning candidate also runs with seed 43, and nothing stops.
    noisy = _context(inputs, calls, max_seed_range=1e-9)
    multi = run_study(noisy, STAGES[-1])
    assert multi.step0.multi_seed and multi.step0.tuning_seeds == (42, 43)
    assert multi.tuning is not None and multi.listwise_step is not None
    for outcome in (*multi.tuning, multi.listwise_step):
        for score in outcome.scores.values():
            assert [seed.seed for seed in score.seeds] == [42, 43]
    new_calls = calls[n_trainings:]
    assert new_calls and {call.seed for call in new_calls} <= {42, 43}
    # The ablations stay at the selection seed in both branches.
    assert multi.ablations is not None
    assert all([s.seed for s in score.seeds] == [42] for score in multi.ablations.values())


def test_a_training_recorded_on_another_device_does_not_count(tmp_path: Path) -> None:
    inputs = build_study_workspace(tmp_path)
    paths = inputs.paths
    variant = variant_of(inputs.config)
    config_id = variant_config_id(inputs.config, variant)
    row = dict.fromkeys(TRIAL_COLUMNS, "0")
    for fold in STUDY_TARGETS:
        row.update({"area": AREA_V_PHASE, "config_id": config_id, "seed": "42", "fold": fold})
        append_row(paths.deepfm_trials_file, TRIAL_COLUMNS, {**row, "device": "mps"})
    assert (AREA_V_PHASE, config_id, 42, "V1", "mps") in recorded_trials(paths)
    requested: list[TrialRequest] = []
    study = dataclasses.replace(STUDY, target_folds=STUDY_TARGETS)
    context = StudyContext(paths, inputs.config, study, RULE, requested.append)
    # The config says CPU: the four MPS rows do not satisfy the request.
    assert inputs.config.training.device == "cpu"
    assert ensure_trials(context, variant, [42]) == 4
    assert [request.target_fold for request in requested] == list(STUDY_TARGETS)
    for fold in STUDY_TARGETS:
        row.update({"area": AREA_V_PHASE, "config_id": config_id, "seed": "42", "fold": fold})
        append_row(paths.deepfm_trials_file, TRIAL_COLUMNS, {**row, "device": "cpu"})
    assert ensure_trials(context, variant, [42]) == 0
