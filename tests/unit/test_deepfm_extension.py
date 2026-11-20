"""Tests for deepfm.extension: the runs of the input-representation extension (proposal §16.10).

The recorded ids below are those of the 2x2 study (results/deepfm_study.csv):
    3cb19c17461b   the default configuration (32 bins, pointwise) with T = 1
    19ba76cfd205   the listwise arm (32 bins) with T = 1
    08775a5d6efb   the ablation "numeric_input = standardised" (pointwise, T = 1)
"""

import dataclasses
import json
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from deepfm_example import REPO, STUDY_TARGETS, build_study_workspace

from hmrec.config import (
    DeepFMExtensionConfig,
    DeepFMExtensionRun,
    load_deepfm_config,
    load_deepfm_extension_config,
    load_deepfm_study_config,
    load_selection_rule,
)
from hmrec.csv_records import read_rows
from hmrec.deepfm.extension import (
    check_extension,
    n_bins_arguments,
    run_config,
    run_config_id,
    run_context,
    run_extension,
    run_variant,
)
from hmrec.deepfm.process import record_trial, with_n_bins, with_seed
from hmrec.deepfm.records import TRIAL_COLUMNS
from hmrec.deepfm.study import Variant, apply_variant, variant_config_id
from hmrec.deepfm.study_run import StudyContext, TrialRequest, recorded_trials, subprocess_launcher
from hmrec.deepfm.trial import TrialInputs, TrialSpec, run_trial
from hmrec.memory import ProcessPeakMemory
from hmrec.run_records import RunStamp

RULE = load_selection_rule(REPO / "configs" / "selection_rule.yaml")
STUDY = load_deepfm_study_config(REPO / "configs" / "hpo" / "deepfm.yaml")
REAL_CONFIG = load_deepfm_config(REPO / "configs" / "deepfm.yaml")
EXTENSION = load_deepfm_extension_config(REPO / "configs" / "hpo" / "deepfm_extension.yaml")
STAMP = RunStamp("abc123", False)


def _refuse(request: TrialRequest) -> None:
    raise AssertionError(f"the study context's own launcher was used: {request}")


def _real_context() -> StudyContext:
    return StudyContext(Path("unused"), REAL_CONFIG, STUDY, RULE, _refuse)  # type: ignore[arg-type]  # paths are not read


# ---------------------------------------------------------------- the real configs


def test_a_run_is_the_fixed_settings_plus_its_input_and_loss() -> None:
    e1, e2, e3, e4 = EXTENSION.runs
    assert run_variant(EXTENSION, e1) == Variant(
        0.001, 16, 0.0, "full", "binned", "pointwise_bce", 1
    )
    assert run_variant(EXTENSION, e4) == Variant(
        0.001, 16, 0.0, "full", "standardised", "listwise_softmax", 1
    )
    assert run_variant(EXTENSION, e2).loss == "listwise_softmax"
    # Only the number of bins of the config changes.
    assert run_config(REAL_CONFIG, e1) == dataclasses.replace(REAL_CONFIG, n_bins=255)
    assert run_config(REAL_CONFIG, e3) == REAL_CONFIG
    assert n_bins_arguments(e1) == ("--n-bins", "255")
    assert n_bins_arguments(e3) == ("--n-bins", "32")


def test_e3_is_the_recorded_ablation_and_the_other_runs_are_new_configurations() -> None:
    ids = {run.name: run_config_id(REAL_CONFIG, EXTENSION, run) for run in EXTENSION.runs}
    # E3 has the id of the ablation run of the 2x2 study, so its seed 42 is reused.
    assert ids["E3"] == "08775a5d6efb"
    assert len(set(ids.values())) == 4
    # With 32 bins the settings of E1 / E2 ARE the two arms of the 2x2 study ...
    pointwise, listwise = (run_variant(EXTENSION, run) for run in EXTENSION.runs[:2])
    assert variant_config_id(REAL_CONFIG, pointwise) == "3cb19c17461b"
    assert variant_config_id(REAL_CONFIG, listwise) == "19ba76cfd205"
    # ... and with 255 bins they are other configurations.
    assert not {ids["E1"], ids["E2"]} & {"3cb19c17461b", "19ba76cfd205"}


def test_the_extension_must_fit_the_study() -> None:
    check_extension(EXTENSION, STUDY, REAL_CONFIG)
    with pytest.raises(ValueError, match="are not the study's"):
        other_folds = dataclasses.replace(EXTENSION, target_folds=("V1", "V2"))
        check_extension(other_folds, STUDY, REAL_CONFIG)
    with pytest.raises(ValueError, match="are not the study's final seeds"):
        check_extension(dataclasses.replace(EXTENSION, seeds=(42,)), STUDY, REAL_CONFIG)
    first = EXTENSION.runs[0]
    taken = dataclasses.replace(first, arm="deepfm_pointwise")
    with pytest.raises(ValueError, match="belong to the 2x2 study"):
        check_extension(dataclasses.replace(EXTENSION, runs=(taken,)), STUDY, REAL_CONFIG)
    twin = dataclasses.replace(first, name="E9", arm="another_arm")
    with pytest.raises(ValueError, match="the same configuration"):
        check_extension(dataclasses.replace(EXTENSION, runs=(first, twin)), STUDY, REAL_CONFIG)
    unknown = (("deepfm_bins255_pointwise", "deepfm_bins64_pointwise"),)
    with pytest.raises(ValueError, match="unknown arms"):
        check_extension(dataclasses.replace(EXTENSION, contrasts=unknown), STUDY, REAL_CONFIG)


def test_a_run_context_carries_the_bins_in_the_config_and_in_the_launcher() -> None:
    given: list[Sequence[str]] = []

    def launcher_for(extra: Sequence[str]):
        given.append(extra)
        return _refuse

    own = run_context(_real_context(), EXTENSION.runs[0], launcher_for)
    assert own.config.n_bins == 255 and given == [("--n-bins", "255")]
    assert own.study == STUDY and own.launcher is _refuse


def test_the_subprocess_launcher_appends_the_extra_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def record(command: list[str], check: bool) -> None:
        assert check
        commands.append(command)

    monkeypatch.setattr("hmrec.deepfm.study_run.subprocess.run", record)
    request = TrialRequest("v_phase", "V2", 43, run_variant(EXTENSION, EXTENSION.runs[1]))
    subprocess_launcher(Path("trial.py"), ("--n-bins", "255"))(request)
    subprocess_launcher(Path("trial.py"))(request)
    with_bins, without = commands
    assert with_bins[-2:] == ["--n-bins", "255"] and with_bins[:-2] == without
    assert "--n-bins" not in without and without[1] == "trial.py"


# ---------------------------------------------------------------- a tiny extension, end to end


def _tiny_extension() -> DeepFMExtensionConfig:
    """Four runs on the tiny workspace (whose config has 4 bins and 4-dim embeddings)."""
    runs = (
        DeepFMExtensionRun("E1", "deepfm_bins6_pointwise", "binned", 6, "pointwise_bce"),
        DeepFMExtensionRun("E2", "deepfm_bins6_listwise", "binned", 6, "listwise_softmax"),
        DeepFMExtensionRun(
            "E3", "deepfm_standardised_pointwise", "standardised", 4, "pointwise_bce"
        ),
        DeepFMExtensionRun(
            "E4", "deepfm_standardised_listwise", "standardised", 4, "listwise_softmax"
        ),
    )
    contrasts = (("deepfm_bins6_pointwise", "deepfm_pointwise"),)
    return dataclasses.replace(
        EXTENSION, seeds=(42, 43), embedding_dim=4, runs=runs, contrasts=contrasts
    )


def _in_process_launchers(inputs: TrialInputs, calls: list[tuple[TrialRequest, tuple[str, ...]]]):
    """launcher_for of a study whose trainings run in this process, as the trial script does."""
    peak = ProcessPeakMemory(1024**3, 0)

    def launcher_for(extra: Sequence[str]):
        flag, n_bins = extra
        assert flag == "--n-bins"

        def launch(request: TrialRequest) -> None:
            calls.append((request, tuple(extra)))
            config = with_n_bins(apply_variant(inputs.config, request.variant), int(n_bins))
            trial_inputs = dataclasses.replace(inputs, config=with_seed(config, request.seed))
            spec = TrialSpec(
                request.area, request.target_fold, request.seed, request.variant.epochs
            )
            result = run_trial(trial_inputs, spec)
            record_trial(trial_inputs, spec, result, time.perf_counter(), peak, STAMP)

        return launch

    return launcher_for


def test_tiny_extension_trains_every_run_once_and_reuses_what_is_recorded(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    inputs = build_study_workspace(tmp_path)
    extension = _tiny_extension()
    study = dataclasses.replace(STUDY, target_folds=STUDY_TARGETS, final_seeds=(42, 43))
    context = StudyContext(inputs.paths, inputs.config, study, RULE, _refuse)
    calls: list[tuple[TrialRequest, tuple[str, ...]]] = []
    launcher_for = _in_process_launchers(inputs, calls)

    # E3 with seed 42 exists already (as the ablation run does in the real study).
    e3 = extension.runs[2]
    for fold in STUDY_TARGETS:
        launcher_for(n_bins_arguments(e3))(
            TrialRequest("v_phase", fold, 42, run_variant(extension, e3))
        )
    calls.clear()

    results = run_extension(context, extension, launcher_for)
    assert [result.run.name for result in results] == ["E1", "E2", "E3", "E4"]
    # 4 runs x 2 seeds x 4 folds, minus the 4 recorded trainings of E3 with seed 42.
    assert len(calls) == 4 * 2 * 4 - 4
    e3_calls = [request for request, _ in calls if request.variant == run_variant(extension, e3)]
    assert {request.seed for request in e3_calls} == {43} and len(e3_calls) == 4
    # In run order, and every training got its run's number of bins.
    order = [request.variant for request, _ in calls]
    assert order == sorted(order, key=[result.variant for result in results].index)
    for request, extra in calls:
        expected = "6" if request.variant.numeric_input == "binned" else "4"
        assert extra == ("--n-bins", expected)
    for result in results:
        assert result.score.config_id == run_config_id(inputs.config, extension, result.run)
        assert [seed.seed for seed in result.score.seeds] == [42, 43]
        assert list(result.score.seeds[0].fold_scores) == list(STUDY_TARGETS)
        assert result.score.epochs == 1 and 0 < result.score.pooled <= 1
    # The recorded configuration of a training says how many bins it used.
    recorded = read_rows(inputs.paths.deepfm_trials_file, TRIAL_COLUMNS)
    bins = {row["config_id"]: json.loads(row["config"])["n_bins"] for row in recorded}
    assert [bins[result.score.config_id] for result in results] == [6, 6, 4, 4]
    assert len(recorded_trials(inputs.paths)) == 32

    # Nothing is trained twice: a second call only reads the records.
    calls.clear()
    assert run_extension(context, extension, launcher_for) == results and calls == []
