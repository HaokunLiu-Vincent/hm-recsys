"""Tests for hmrec.t_eval.plan and the config of the evaluation on T1-T4 (proposal §16.12)."""

import json
from pathlib import Path

import pytest
from deepfm_example import MANIFEST, RATIO, REPO
from t_eval_example import (
    COMMIT,
    M8_COMMIT,
    T_FOLDS,
    build_t_eval_workspace,
    context_of,
    t_eval_config,
)

from hmrec.config import load_t_eval_config, load_yaml, parse_t_eval_config
from hmrec.deepfm.trial import TrialInputs
from hmrec.ranker.final_window import final_training_window
from hmrec.t_eval.plan import (
    STAGE_FORMAL,
    STAGE_REHEARSAL,
    TrialKey,
    area_of,
    check_config_folds,
    check_trial_key,
    formal_training_window,
    m8_dataset_dir,
    m8_formal_run_commit,
    training_window,
)

CONFIG_FILE = REPO / "configs" / "m9_t_eval.yaml"


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> TrialInputs:
    return build_t_eval_workspace(tmp_path_factory.mktemp("t_eval_plan"))


def _names(folds: tuple) -> list[str]:
    return [fold.name for fold in folds]


def test_the_recorded_config_is_the_configuration_chosen_on_v() -> None:
    config = load_t_eval_config(CONFIG_FILE)
    assert (config.phase, config.target_folds) == ("T", ("T1", "T2", "T3", "T4"))
    assert (config.seeds, config.rehearsal_target_fold) == ((42, 43, 44), "V4")
    # Standardised numeric input, listwise loss, the settings the V study kept, T = 1.
    assert (config.numeric_input, config.loss) == ("standardised", "listwise_softmax")
    assert (config.learning_rate, config.embedding_dim, config.dropout) == (0.001, 16, 0.0)
    assert (config.components, config.epochs, config.n_bins) == ("full", 1, 32)
    assert (config.arm, config.config_id) == ("deepfm_standardised_listwise", "f8811fb59b79")
    # LightGBM lambdarank is the M8 record: one seed, the locked number of trees.
    assert (config.m8_record_arm, config.m8_record_seed, config.m8_record_num_trees) == (
        "lgb_lambdarank",
        42,
        500,
    )
    check_config_folds(config, MANIFEST)


def test_config_errors_are_refused() -> None:
    raw = load_yaml(CONFIG_FILE)
    with pytest.raises(ValueError, match="phase must be 'T'"):
        parse_t_eval_config({**raw, "phase": "V"})
    with pytest.raises(ValueError, match="unknown keys"):
        parse_t_eval_config({**raw, "lightgbm": {}})
    with pytest.raises(ValueError, match="a seed repeats"):
        parse_t_eval_config({**raw, "seeds": [42, 42]})
    with pytest.raises(ValueError, match="is not one of the seeds"):
        parse_t_eval_config({**raw, "seeds": [43, 44]})
    with pytest.raises(ValueError, match="12 characters"):
        parse_t_eval_config({**raw, "deepfm": {**raw["deepfm"], "config_id": "abc"}})
    with pytest.raises(ValueError, match="numeric_input"):
        parse_t_eval_config({**raw, "deepfm": {**raw["deepfm"], "numeric_input": "raw"}})
    with pytest.raises(ValueError, match="is not a T fold"):
        check_config_folds(t_eval_config(target_folds=("V4", "T1")), MANIFEST)
    with pytest.raises(ValueError, match="time order"):
        check_config_folds(t_eval_config(target_folds=("T2", "T1")), MANIFEST)
    with pytest.raises(ValueError, match="is not a V fold"):
        check_config_folds(t_eval_config(rehearsal_target_fold="T1"), MANIFEST)


def test_formal_windows_are_the_windows_of_m8() -> None:
    # Hand-written from proposal §10.1, and equal to M8's own window function.
    expected = {
        "T1": ["V1", "V2", "V3", "V4"],
        "T2": ["V2", "V3", "V4", "T1"],
        "T3": ["V3", "V4", "T1", "T2"],
        "T4": ["V4", "T1", "T2", "T3"],
    }
    for target in T_FOLDS:
        window = formal_training_window(MANIFEST, target, 4)
        assert _names(window) == expected[target]
        assert window == final_training_window(MANIFEST, target, 4)


def test_formal_window_refuses_other_targets() -> None:
    for target in ("V4", "F1", "D2"):
        with pytest.raises(ValueError, match="predicts only T folds"):
            formal_training_window(MANIFEST, target, 4)


def test_rehearsal_window_is_the_m7_window_and_cannot_reach_a_t_fold(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    window = training_window(context, TrialKey(STAGE_REHEARSAL, 42, "V4"), 4)
    assert _names(window) == ["F4", "V1", "V2", "V3"]
    for target in ("T1", "V3"):
        with pytest.raises(ValueError, match="only the configured rehearsal fold V4"):
            training_window(context, TrialKey(STAGE_REHEARSAL, 42, target), 4)


def test_trainings_outside_the_evaluation_are_refused(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    check_trial_key(context, TrialKey(STAGE_FORMAL, 43, "T4"))
    with pytest.raises(ValueError, match="stage must be one of"):
        check_trial_key(context, TrialKey("tuning", 42, "T1"))
    with pytest.raises(ValueError, match="got seed 7"):
        check_trial_key(context, TrialKey(STAGE_FORMAL, 7, "T1"))
    with pytest.raises(ValueError, match="predicts only the folds"):
        check_trial_key(context, TrialKey(STAGE_FORMAL, 42, "V4"))
    assert _names(training_window(context, TrialKey(STAGE_FORMAL, 42, "T2"), 4)) == [
        "V2",
        "V3",
        "V4",
        "T1",
    ]


def test_areas_keep_a_rehearsal_with_its_commit() -> None:
    assert area_of(STAGE_FORMAL, COMMIT) == "m9_t"
    assert area_of(STAGE_REHEARSAL, COMMIT) == "m9_t_rehearsal/aaaaaaaaaaaa"
    with pytest.raises(ValueError, match="stage must be one of"):
        area_of("tuning", COMMIT)


def test_the_m8_dataset_record_is_found_and_checked(inputs: TrialInputs) -> None:
    context, paths = context_of(inputs), inputs.paths
    assert m8_formal_run_commit(paths) == M8_COMMIT
    formal = TrialKey(STAGE_FORMAL, 42, "T3")
    folds = training_window(context, formal, 4)
    assert m8_dataset_dir(context, formal, folds, RATIO) == paths.final_dataset_dir(
        "t_phase", "full", "T3", RATIO
    )
    rehearsal = TrialKey(STAGE_REHEARSAL, 42, "V4")
    rehearsal_folds = training_window(context, rehearsal, 4)
    assert m8_dataset_dir(context, rehearsal, rehearsal_folds, RATIO) == paths.final_dataset_dir(
        "m8_rehearsal", "full", "V4", RATIO
    )
    # Other training folds than the record's, or another ratio: refused.
    with pytest.raises(ValueError, match="train_folds"):
        m8_dataset_dir(context, formal, folds[1:], RATIO)
    with pytest.raises(FileNotFoundError, match="M8 built no Dataset"):
        m8_dataset_dir(context, formal, folds, 60)


def test_a_dataset_record_of_another_commit_is_refused(tmp_path: Path) -> None:
    inputs = build_t_eval_workspace(tmp_path)
    context = context_of(inputs)
    key = TrialKey(STAGE_FORMAL, 42, "T1")
    folds = training_window(context, key, 4)
    record_file = inputs.paths.final_dataset_dir("t_phase", "full", "T1", RATIO) / "dataset.json"
    record = json.loads(record_file.read_text())
    record_file.write_text(json.dumps({**record, "git_commit": "d" * 40}))
    with pytest.raises(ValueError, match="git_commit"):
        m8_dataset_dir(context, key, folds, RATIO)
    # Without the record of the M8 formal run there is no commit to compare with.
    inputs.paths.t_phase_label_access_file.unlink()
    with pytest.raises(ValueError, match="formal run of M8 is not recorded"):
        m8_formal_run_commit(inputs.paths)
