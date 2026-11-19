"""Tests for deepfm.study and deepfm.study_run: the rules on hand-made rows, then a tiny study.

Hand example for the epoch rule (two folds, V1 with 100 queries and V2 with 300):
    epoch 1: V1 0.20, V2 0.10 -> pooled (100 x 0.20 + 300 x 0.10) / 400 = 0.125
    epoch 2: V1 0.22, V2 0.12 -> pooled 0.145
    epoch 3: V1 0.21, V2 0.1205 -> pooled 0.142875 (1.47% below the best)
    with a 1% tolerance only epoch 2 is "about equal" to the best -> e* = 2;
    with a 2% tolerance epochs 2 and 3 are, and the smallest wins -> still 2;
    with a 20% tolerance epoch 1 is too (13.8% below) -> e* = 1.
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
    load_deepfm_study_config,
    load_selection_rule,
)
from hmrec.csv_records import read_rows
from hmrec.deepfm.process import record_trial, with_seed
from hmrec.deepfm.records import read_pooled_reference
from hmrec.deepfm.study import (
    ConfigurationScore,
    StudyStop,
    Variant,
    ablation_variants,
    apply_variant,
    check_above_baseline,
    check_noise,
    decide_step,
    parameter_value,
    pooled_value,
    score_configuration,
    seed_range,
    step_variants,
    variant_config_id,
    variant_of,
    with_parameter,
)
from hmrec.deepfm.study_run import (
    AREA_V_EXTENDED,
    AREA_V_PHASE,
    STUDY_COLUMNS,
    StudyContext,
    TrialRequest,
    final_rows,
    recorded_trials,
    run_ablations,
    run_final,
    run_seeds,
    run_step0,
    run_tuning,
    selected_variant,
    step0_rows,
    trial_arguments,
    tuning_rows,
    write_study_table,
)
from hmrec.deepfm.trial import TrialInputs, TrialSpec, run_trial
from hmrec.locks import load_ranker_lock
from hmrec.memory import ProcessPeakMemory
from hmrec.run_records import RunStamp

RULE = load_selection_rule(REPO / "configs" / "selection_rule.yaml")
STUDY = load_deepfm_study_config(REPO / "configs" / "hpo" / "deepfm.yaml")
FOLDS: tuple[str, ...] = ("V1", "V2", "V3", "V4")
DEFAULT = Variant(0.001, 16, 0.0, "full", "binned", "pointwise_bce")
STAMP = RunStamp("abc123", False)


def _rows(config_id: str, seed: int, values: dict[int, dict[str, float]], area: str = "v_phase"):
    """Metric rows of one configuration: epoch -> fold -> MAP@12 (100 / 300 queries)."""
    n_queries = {"V1": 100, "V2": 300, "V3": 100, "V4": 300}
    return [
        {
            "area": area,
            "config_id": config_id,
            "seed": str(seed),
            "fold": fold,
            "epoch": str(epoch),
            "n_queries": str(n_queries[fold]),
            "map@12": repr(value),
            "train_loss": repr(1.0 / epoch),
        }
        for epoch, by_fold in values.items()
        for fold, value in by_fold.items()
    ]


CURVE = {
    1: {"V1": 0.20, "V2": 0.10},
    2: {"V1": 0.22, "V2": 0.12},
    3: {"V1": 0.21, "V2": 0.1205},
}


def _score(values=CURVE, tolerance: float = 0.01, folds=("V1", "V2")) -> ConfigurationScore:
    return score_configuration(
        _rows("c1", 42, values), "v_phase", "c1", 42, folds, "map@12", tolerance
    )


# ---------------------------------------------------------------- scores and the epoch rule


def test_pooled_value_is_weighted_by_queries() -> None:
    assert pooled_value({"V1": 0.20, "V2": 0.10}, {"V1": 100, "V2": 300}) == pytest.approx(0.125)


def test_epoch_rule_hand_computed() -> None:
    score = _score()
    assert score.curve == pytest.approx({1: 0.125, 2: 0.145, 3: 0.142875})
    assert (score.selected_epoch, score.best_epoch, score.epochs_trained) == (2, 2, 3)
    assert score.pooled == pytest.approx(0.145)
    # The fold scores are those of the SELECTED epoch.
    assert score.fold_scores == {"V1": 0.22, "V2": 0.12}
    assert not score.uses_last_epoch
    assert _score(tolerance=0.02).selected_epoch == 2
    assert _score(tolerance=0.20).selected_epoch == 1
    assert score.train_loss == pytest.approx({1: 1.0, 2: 0.5, 3: 1 / 3})
    trial = score.trial_scores()
    assert (trial.trial_id, trial.pooled_score) == ("c1", score.pooled)


def test_a_rising_curve_selects_the_last_epoch() -> None:
    rising = {1: {"V1": 0.1, "V2": 0.1}, 2: {"V1": 0.2, "V2": 0.2}, 3: {"V1": 0.3, "V2": 0.3}}
    score = _score(rising)
    assert score.selected_epoch == 3 and score.uses_last_epoch


def test_scores_need_every_fold_for_every_epoch_and_only_their_own_rows() -> None:
    rows = _rows("c1", 42, CURVE) + _rows("c1", 43, CURVE) + _rows("c2", 42, CURVE)
    rows += _rows("c1", 42, {1: {"V1": 0.9, "V2": 0.9}}, area="v_phase_extended")
    own = score_configuration(rows, "v_phase", "c1", 42, ("V1", "V2"), "map@12", 0.01)
    assert own.pooled == pytest.approx(0.145)  # other seeds, configurations and areas are ignored
    with pytest.raises(ValueError, match="not all recorded"):
        score_configuration(rows, "v_phase", "c1", 42, ("V1", "V2", "V3"), "map@12", 0.01)
    with pytest.raises(ValueError, match="not all recorded"):
        score_configuration(rows, "v_phase", "missing", 42, ("V1", "V2"), "map@12", 0.01)
    with pytest.raises(ValueError, match="recorded twice"):
        score_configuration(rows + rows[:1], "v_phase", "c1", 42, ("V1", "V2"), "map@12", 0.01)


# ---------------------------------------------------------------- step 0


def _pooled_score(pooled: float, seed: int = 42, config_id: str = "c") -> ConfigurationScore:
    folds = dict.fromkeys(FOLDS, pooled)
    return ConfigurationScore(config_id, seed, "v_phase", 5, 3, 3, pooled, folds, {3: pooled}, {})


def test_seed_range_is_max_minus_min_over_min() -> None:
    scores = [_pooled_score(0.0300, 42), _pooled_score(0.0303, 43), _pooled_score(0.0301, 44)]
    assert seed_range(scores) == pytest.approx(0.01)
    assert check_noise(scores, 0.0101) == pytest.approx(0.01)
    with pytest.raises(StudyStop, match="stops before any tuning"):
        check_noise(scores, 0.0099)
    with pytest.raises(ValueError, match="at least two"):
        seed_range(scores[:1])


def test_default_below_the_rrf_baseline_stops_the_study() -> None:
    check_above_baseline(_pooled_score(0.0195), 0.019419)
    with pytest.raises(StudyStop, match="below the RRF"):
        check_above_baseline(_pooled_score(0.0194), 0.019419)


# ---------------------------------------------------------------- tuning

LEARNING_RATE = DeepFMStudyStep("learning_rate", "learning_rate", (0.001, 0.0003, 0.003))


def _fold_score(config_id: str, folds: list[float]) -> ConfigurationScore:
    """A score whose four folds each have 100 queries (pooled = plain mean)."""
    pooled = sum(folds) / 4
    by_fold = dict(zip(FOLDS, folds, strict=True))
    return ConfigurationScore(config_id, 42, "v_phase", 5, 3, 3, pooled, by_fold, {3: pooled}, {})


def test_step_keeps_the_default_when_no_candidate_passes() -> None:
    scores = {
        0.001: _fold_score("default", [0.030, 0.030, 0.030, 0.030]),
        # +0.5%: better in all 4 weeks, but below the 1% gain.
        0.0003: _fold_score("slow", [0.03015, 0.03015, 0.03015, 0.03015]),
        # +5% pooled, but better in only 2 weeks.
        0.003: _fold_score("fast", [0.040, 0.038, 0.024, 0.024]),
    }
    outcome = decide_step(LEARNING_RATE, DEFAULT, scores, RULE)
    assert (outcome.selected_value, outcome.decision.action) == (0.001, "keep")
    assert [(c.folds_better, c.passes) for c in outcome.decision.checks] == [(4, False), (2, False)]
    assert outcome.selected == DEFAULT and not outcome.on_grid_edge  # 0.001 is inside its grid


def test_step_switches_to_the_passing_candidate_with_the_highest_priority() -> None:
    scores = {
        0.001: _fold_score("default", [0.030, 0.030, 0.030, 0.030]),
        0.0003: _fold_score("slow", [0.0310, 0.0310, 0.0310, 0.0295]),  # +2.1%, 3 weeks
        0.003: _fold_score("fast", [0.0312, 0.0312, 0.0312, 0.0312]),  # +4%, 4 weeks
    }
    outcome = decide_step(LEARNING_RATE, DEFAULT, scores, RULE)
    # "fast" is better, but "slow" comes first in priority order and is NOT within 1% of it.
    assert outcome.selected_value == 0.003
    close = {**scores, 0.003: _fold_score("fast", [0.0309, 0.0309, 0.0309, 0.0309])}
    outcome = decide_step(LEARNING_RATE, DEFAULT, close, RULE)
    # Now the two passing candidates are within 1% of each other: the first in priority wins.
    assert (outcome.selected_value, outcome.decision.action) == (0.0003, "switch")
    assert outcome.selected.learning_rate == 0.0003 and outcome.on_grid_edge  # the smallest value


def test_a_step_must_start_from_the_current_configuration() -> None:
    assert list(step_variants(LEARNING_RATE, DEFAULT)) == [0.001, 0.0003, 0.003]
    moved = dataclasses.replace(DEFAULT, learning_rate=0.003)
    with pytest.raises(ValueError, match="not the current value"):
        step_variants(LEARNING_RATE, moved)
    dimension = DeepFMStudyStep("embedding_dim", "embedding_dim", (16.0, 8.0, 32.0))
    variants = step_variants(dimension, moved)
    # The next step keeps the learning rate the previous step selected.
    assert [(v.embedding_dim, v.learning_rate) for v in variants.values()] == [
        (16, 0.003),
        (8, 0.003),
        (32, 0.003),
    ]
    assert isinstance(variants[8.0].embedding_dim, int)


def test_variants_change_only_the_named_setting(tmp_path: Path) -> None:
    config = build_study_workspace(tmp_path).config
    default = variant_of(config)
    assert apply_variant(config, default) == config
    faster = with_parameter(default, "learning_rate", 0.003)
    changed = apply_variant(config, faster)
    assert changed.training.learning_rate == 0.003 and changed.model == config.model
    assert variant_config_id(config, faster) != variant_config_id(config, default)
    assert parameter_value(with_parameter(default, "dropout", 0.3), "dropout") == 0.3
    with pytest.raises(ValueError, match="not a tuned parameter"):
        with_parameter(default, "batch_rows", 1)
    with pytest.raises(ValueError, match="whole number"):
        with_parameter(default, "embedding_dim", 8.5)


def test_ablations_change_one_setting_and_leave_out_the_selected_configuration() -> None:
    selected = dataclasses.replace(DEFAULT, learning_rate=0.003)
    variants = ablation_variants(STUDY, selected)
    assert list(variants) == [
        "components=first_order",
        "components=fm",
        "components=deep",
        "numeric_input=standardised",
        "loss=listwise_softmax",
    ]
    assert all(variant.learning_rate == 0.003 for variant in variants.values())
    assert variants["components=fm"] == dataclasses.replace(selected, components="fm")
    assert variants["loss=listwise_softmax"].components == "full"


def test_trial_arguments_carry_the_whole_configuration() -> None:
    request = TrialRequest(
        "v_phase_extended", "V3", 44, 10, dataclasses.replace(DEFAULT, dropout=0.1)
    )
    assert trial_arguments(request) == [
        "--area", "v_phase_extended", "--target", "V3", "--seed", "44", "--epochs", "10",
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


def _context(inputs: TrialInputs, calls: list[TrialRequest]) -> StudyContext:
    """A study on the tiny workspace whose trainings run in this process."""
    peak = ProcessPeakMemory(1024**3, 0)

    def launch(request: TrialRequest) -> None:
        calls.append(request)
        config = with_seed(apply_variant(inputs.config, request.variant), request.seed)
        trial_inputs = dataclasses.replace(inputs, config=config)
        spec = TrialSpec(request.area, request.target_fold, request.seed, request.epochs)
        result = run_trial(trial_inputs, spec)
        record_trial(trial_inputs, spec, result, time.perf_counter(), peak, STAMP)

    study = dataclasses.replace(
        STUDY,
        noise_seeds=(42, 43),
        max_seed_range=10.0,  # the tiny models are noisy; the stop rule is tested above
        steps=(DeepFMStudyStep("learning_rate", "learning_rate", (0.001, 0.003)),),
        ablation_components=("fm", "full"),
        ablation_numeric_inputs=(),
        ablation_losses=("listwise_softmax",),
        final_seeds=(42, 43),
        extension_epochs=2,
    )
    return StudyContext(inputs.paths, inputs.config, study, RULE, launch)


def test_tiny_study_runs_every_stage_and_extends_the_epochs_at_the_cap(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    inputs = build_study_workspace(tmp_path, epochs=1)  # 1 epoch: the rule must pick the cap
    _write_reference(inputs)
    calls: list[TrialRequest] = []
    context = _context(inputs, calls)
    reference = read_pooled_reference(inputs.paths, STUDY_TARGETS, "map@12")
    assert (reference.rrf_baseline, reference.lightgbm) == pytest.approx((0.05, 0.5))

    # ---- step 0: 2 seeds x 4 folds, one process each
    step0 = run_step0(context)
    assert [(c.seed, c.target_fold, c.epochs) for c in calls] == [
        (seed, fold, 1) for seed in (42, 43) for fold in STUDY_TARGETS
    ]
    assert [score.seed for score in step0.scores] == [42, 43] and step0.baseline == 0.05
    assert step0.seed_range == pytest.approx(seed_range(step0.scores))
    assert all(score.selected_epoch == 1 for score in step0.scores)

    # ---- tuning: the default is not trained again; the candidate is, with seed 42 only
    outcomes = run_tuning(context)
    new = calls[8:]
    assert [(c.variant.learning_rate, c.seed, c.target_fold) for c in new] == [
        (0.003, 42, fold) for fold in STUDY_TARGETS
    ]
    assert len(outcomes) == 1 and set(outcomes[0].scores) == {0.001, 0.003}
    selected = selected_variant(outcomes, context.default)
    assert selected.learning_rate == outcomes[0].selected_value

    # ---- final: epoch 1 of 1 is the cap -> the selected configuration is trained for 2 epochs
    final = run_final(context, selected)
    extension = calls[12:]
    assert final.extended and (final.area, final.epochs) == (AREA_V_EXTENDED, 2)
    assert [(c.area, c.epochs, c.seed) for c in extension] == [(AREA_V_EXTENDED, 2, 42)] * 4
    assert final.default_epochs_score.uses_last_epoch and final.score.epochs_trained == 2
    # On CPU the first epoch of the longer run repeats the 1-epoch run exactly.
    assert final.score.curve[1] == final.default_epochs_score.curve[1]

    # ---- ablations and seeds use the extended number of epochs too
    ablations = run_ablations(context, final)
    assert list(ablations) == ["components=fm", "loss=listwise_softmax"]
    seeds = run_seeds(context, final)
    later = calls[16:]
    assert all((c.area, c.epochs) == (AREA_V_EXTENDED, 2) for c in later)
    assert [score.seed for score in seeds] == [42, 43] and seeds[0] == final.score
    assert len(calls) == 16 + 2 * 4 + 4  # two ablations, and seed 43 of the selected one

    # ---- nothing is trained twice: every stage again starts no process
    n_calls = len(calls)
    run_step0(context)
    assert run_final(context, selected_variant(run_tuning(context), context.default)) == final
    run_ablations(context, final)
    run_seeds(context, final)
    assert len(calls) == n_calls == len(recorded_trials(inputs.paths))

    # ---- the study table
    rows = step0_rows(step0) + tuning_rows(outcomes)
    rows += final_rows(final, ablations, seeds, reference.lightgbm)
    write_study_table(inputs.paths, rows, STAMP)
    written = read_rows(inputs.paths.deepfm_study_file, STUDY_COLUMNS)
    assert [row["stage"] for row in written] == (
        ["step0"] * 2
        + ["tune:learning_rate"] * 2
        + ["final"] * 2
        + ["ablation"] * 2
        + ["seeds"] * 2
    )
    tuned = [row for row in written if row["stage"] == "tune:learning_rate"]
    assert [row["selected"] for row in tuned].count("True") == 1
    assert tuned[0]["relative_change"] == "0.0" and tuned[0]["passes"] == ""  # the step's start
    assert tuned[1]["passes"] in ("True", "False") and tuned[1]["folds_better"] in "01234"
    assert all(row["git_commit"] == "abc123" for row in written)
    with pytest.raises(FileExistsError, match="never replaced"):
        write_study_table(inputs.paths, rows, STAMP)


def test_no_extension_when_the_selected_epoch_is_below_the_cap(tmp_path: Path) -> None:
    inputs = build_study_workspace(tmp_path)
    paths = inputs.paths
    config_id = variant_config_id(inputs.config, variant_of(inputs.config))
    values = {
        1: dict.fromkeys(FOLDS, 0.30),
        2: dict.fromkeys(FOLDS, 0.40),
        3: dict.fromkeys(FOLDS, 0.401),
    }
    header = ["area", "config_id", "seed", "fold", "epoch", "n_queries", "map@12", "train_loss"]
    paths.results_root.mkdir(parents=True)
    with paths.deepfm_metrics_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(_rows(config_id, 42, values))

    def never(request: TrialRequest) -> None:
        raise AssertionError(f"no training is needed, got {request}")

    context = StudyContext(paths, inputs.config, STUDY, RULE, never)
    final = run_final(context, variant_of(inputs.config))
    # Epoch 2 is within 1% of the best (epoch 3): the cap is not reached, nothing is extended.
    assert (final.score.selected_epoch, final.extended, final.area) == (2, False, AREA_V_PHASE)
    assert final.epochs == 3 and final.score == final.default_epochs_score
