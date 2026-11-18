"""Tests for hmrec.final.trial: windows per mode, reused and new samples, the trial itself.

The workspace is tiny (tests/unit/final_example.py): 9 folds of 40 queries, a lock
with 10 trees, and what M7 left behind for the rehearsal fold V4.
"""

from pathlib import Path

import polars as pl
import pytest
from final_example import (
    COMMIT,
    LOCKED_TREES,
    NOW,
    copy_workspace,
)

from hmrec.eval.final_eval import FormalRunPermit
from hmrec.final.context import MODE_FORMAL, MODE_REHEARSAL, FinalContext
from hmrec.final.gate import FinalRunRefused, begin_formal_run
from hmrec.final.trial import (
    baseline_row,
    build_final_dataset,
    check_reused_samples,
    dataset_plan,
    evaluated_trees,
    experiment_config_id,
    new_sample_folds,
    reused_sample_folds,
    run_final_trial,
    run_sampling,
    sample_file,
    sample_job,
    target_evaluator,
    training_window,
    trial_spec,
)
from hmrec.locks import file_sha256
from hmrec.ranker.sampling import read_sample_record, record_file

PERMIT = FormalRunPermit(COMMIT, COMMIT, NOW, NOW)


def _rehearsed(tmp_path: Path) -> tuple[FinalContext, FormalRunPermit]:
    """A workspace whose rehearsal passed and whose formal run has been started."""
    context = copy_workspace(tmp_path, rehearsed=True)
    return context, begin_formal_run(context, NOW)


def _names(folds: tuple) -> list[str]:
    return [fold.name for fold in folds]


# ---------------------------------------------------------------- windows and identities


def test_each_mode_has_its_own_window_and_refuses_the_other_modes_folds(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    assert _names(training_window(context, MODE_FORMAL, "T1")) == ["V1", "V2", "V3", "V4"]
    assert _names(training_window(context, MODE_FORMAL, "T2")) == ["V2", "V3", "V4", "T1"]
    assert _names(training_window(context, MODE_FORMAL, "T3")) == ["V3", "V4", "T1", "T2"]
    assert _names(training_window(context, MODE_FORMAL, "T4")) == ["V4", "T1", "T2", "T3"]
    assert _names(training_window(context, MODE_REHEARSAL, "V4")) == ["F4", "V1", "V2", "V3"]
    with pytest.raises(ValueError, match="only the configured rehearsal fold"):
        training_window(context, MODE_REHEARSAL, "T1")
    with pytest.raises(ValueError, match="predicts only T folds"):
        training_window(context, MODE_FORMAL, "V4")
    with pytest.raises(ValueError, match="unknown mode"):
        training_window(context, "dry_run", "T1")


def test_the_trial_uses_the_locked_values_and_ids(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    spec = trial_spec(context, MODE_FORMAL, "full", "T1")
    assert (spec.area, spec.step, spec.target_fold) == ("t_phase", "formal", "T1")
    assert spec.tuned == context.locked.tuned and spec.num_trees == LOCKED_TREES
    assert spec.histogram_mode == context.locked.histogram_mode and spec.evaluate
    assert trial_spec(context, MODE_REHEARSAL, "full", "V4").area == "m8_rehearsal"
    assert experiment_config_id(context, "full") == context.locked.config_id
    # The ablation differs only in the experiment name, so it has another id.
    assert experiment_config_id(context, "minus_r6") != context.locked.config_id


def test_the_formal_run_evaluates_only_the_locked_number_of_trees(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    assert evaluated_trees(context, MODE_FORMAL) == (LOCKED_TREES,)
    assert evaluated_trees(context, MODE_REHEARSAL) is None  # every checkpoint, to compare


def test_datasets_of_the_two_modes_live_in_their_own_folders(tmp_path: Path) -> None:
    context, _ = _rehearsed(tmp_path)
    rehearsal = dataset_plan(context, MODE_REHEARSAL, "full", "V4")
    assert rehearsal.directory == context.paths.final_dataset_dir("m8_rehearsal", "full", "V4", 2)
    # Not the folder of M7's Dataset of the same fold: the rehearsal built its own.
    assert rehearsal.directory != context.paths.ranker_dataset_dir("full", "V4", 2)
    assert [source.fold for source in rehearsal.sources] == ["F4", "V1", "V2", "V3"]


# ---------------------------------------------------------------- samples


def test_v1_to_v3_are_reused_and_the_other_training_folds_are_new(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    assert reused_sample_folds(context) == ("V1", "V2", "V3")
    assert new_sample_folds(context) == ("V4", "T1", "T2", "T3")


def test_new_samples_use_the_locked_seed_and_ratio(tmp_path: Path) -> None:
    context, permit = _rehearsed(tmp_path)
    record = run_sampling(context, "full", "T1", permit)
    assert (record.fold, record.experiment, record.ratio) == ("T1", "full", 2)
    assert record.seed == context.locked.config.sampling.seed == 42
    # 40 queries, each with 1 positive and 11 negatives; 2 negatives per positive are kept.
    assert (record.n_queries, record.n_positives, record.n_rows) == (40, 40, 120)
    written = sample_file(context, "full", "T1")
    assert read_sample_record(record_file(written)).sha256 == file_sha256(written)
    # Only the locked ratio is written (M7 wrote three ratios; the lock keeps one).
    assert sorted(path.name for path in written.parent.iterdir()) == ["r2.json", "r2.parquet"]


def test_a_reused_fold_is_never_sampled_again(tmp_path: Path) -> None:
    context, permit = _rehearsed(tmp_path)
    before = file_sha256(sample_file(context, "full", "V2"))
    for fold in ("V1", "V2", "V3"):
        with pytest.raises(ValueError, match="is not sampled by the formal run"):
            sample_job(context, "full", fold, permit)
    # T4 trains no model and F4 trains no T model: not sampled either.
    for fold in ("T4", "F4"):
        with pytest.raises(ValueError, match="is not sampled by the formal run"):
            run_sampling(context, "full", fold, permit)
    assert file_sha256(sample_file(context, "full", "V2")) == before


def test_a_t_fold_is_not_sampled_without_a_permit(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    context.paths.labels_file("T1").unlink()  # a read would fail with FileNotFoundError
    with pytest.raises(TypeError, match="only with a FormalRunPermit"):
        sample_job(context, "full", "T1", None)  # type: ignore[arg-type]  # the test
    # V4 is a V fold: its job needs no permit and may read only V labels.
    assert sample_job(context, "full", "V4", PERMIT).allowed_phases == ("V",)
    assert sample_job(context, "full", "T2", PERMIT).allowed_phases == ("T",)


def test_a_reused_sample_with_another_sha256_is_refused(tmp_path: Path) -> None:
    context, permit = _rehearsed(tmp_path)
    run_sampling(context, "full", "V4", permit)
    check_reused_samples(context, "full", ["V1", "V2", "V3"])  # the M7 files: no error
    # Replace the V2 sample by another valid Parquet file (one row less) AFTER the rehearsal.
    path = sample_file(context, "full", "V2")
    pl.read_parquet(path).head(-1).write_parquet(path)
    with pytest.raises(FinalRunRefused, match="but the passed rehearsal trained on"):
        check_reused_samples(context, "full", ["V2"])
    # The Dataset of T1 (window V1 V2 V3 V4) is refused for that reason, and nothing is built.
    with pytest.raises(FinalRunRefused, match="This is not the M7 sample file"):
        build_final_dataset(context, MODE_FORMAL, "full", "T1")
    assert not context.paths.final_dataset_dir("t_phase", "full", "T1", 2).exists()


def test_reused_samples_need_a_recorded_hash_and_an_existing_file(tmp_path: Path) -> None:
    context, _ = _rehearsed(tmp_path)
    with pytest.raises(FinalRunRefused, match="recorded no sha256 for the full sample of V4"):
        check_reused_samples(context, "full", ["V4"])  # V4 was not in the rehearsal's window
    sample_file(context, "minus_r6", "V3").unlink()
    with pytest.raises(FileNotFoundError, match="reused sample file not found"):
        check_reused_samples(context, "minus_r6", ["V3"])


# ---------------------------------------------------------------- labels and the trial


def test_target_labels_come_through_the_modes_guarded_loader(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    assert target_evaluator(context, MODE_REHEARSAL, "V4", None).fold == "V4"
    assert target_evaluator(context, MODE_FORMAL, "T1", PERMIT).fold == "T1"
    with pytest.raises(TypeError, match="needs a FormalRunPermit"):
        target_evaluator(context, MODE_FORMAL, "T1", None)
    with pytest.raises(ValueError, match="its labels are not read"):
        target_evaluator(context, MODE_REHEARSAL, "T1", None)
    with pytest.raises(ValueError, match="evaluates only T folds"):
        target_evaluator(context, MODE_FORMAL, "V4", PERMIT)


def test_a_formal_trial_trains_the_locked_model_and_saves_only_its_top_12(tmp_path: Path) -> None:
    context, permit = _rehearsed(tmp_path)
    for fold in new_sample_folds(context):
        run_sampling(context, "full", fold, permit)
    build_final_dataset(context, MODE_FORMAL, "full", "T2")
    outcome = run_final_trial(context, MODE_FORMAL, "full", "T2", permit)
    assert outcome.config_id == context.locked.config_id
    assert outcome.train_folds == ("V2", "V3", "V4", "T1")
    assert (outcome.n_queries, outcome.n_scored_rows) == (40, 480)  # 40 queries x 12 candidates
    # One metric row, at the locked number of trees.
    assert [row.num_trees for row in outcome.metric_rows] == [LOCKED_TREES]
    assert outcome.metric_rows[0].fold == "T2"
    predictions = pl.read_parquet(
        context.paths.ranker_predictions_file("t_phase", "full", outcome.config_id, "T2")
    )
    assert predictions["num_trees"].unique().to_list() == [LOCKED_TREES]
    assert predictions.height == 480 and predictions["rank"].max() == 12
    # The learnable signal is found: the model is far better than the RRF baseline.
    baseline = baseline_row(context, MODE_FORMAL, "T2", permit)
    assert outcome.metric_rows[0].scores["map@12"] > baseline.scores["map@12"] + 0.3


def test_a_formal_trial_without_a_permit_reads_no_label(tmp_path: Path) -> None:
    context, permit = _rehearsed(tmp_path)
    for fold in new_sample_folds(context):
        run_sampling(context, "full", fold, permit)
    build_final_dataset(context, MODE_FORMAL, "full", "T4")
    context.paths.labels_file("T4").unlink()  # a read would fail with FileNotFoundError
    with pytest.raises(TypeError, match="needs a FormalRunPermit"):
        run_final_trial(context, MODE_FORMAL, "full", "T4", None)
    with pytest.raises(TypeError, match="needs a FormalRunPermit"):
        baseline_row(context, MODE_FORMAL, "T4", None)
