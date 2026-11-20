"""Tests for hmrec.t_eval.gate and hmrec.eval.t_eval_labels (proposal §16.12).

No T label without a passed rehearsal at HEAD and a recorded read. The gate
reads two small tables only, never data or labels; the workspace is the tiny
one of t_eval_example.
"""

from pathlib import Path

import pytest
from deepfm_example import MANIFEST
from t_eval_example import (
    COMMIT,
    NOW,
    OTHER_COMMIT,
    build_t_eval_workspace,
    context_of,
    record_rehearsal,
)

from hmrec.csv_records import read_rows
from hmrec.deepfm.trial import TrialInputs
from hmrec.eval.t_eval_labels import (
    TEvalPermit,
    load_t_eval_evaluator,
    load_t_eval_rehearsal_evaluator,
    require_t_eval_permit,
)
from hmrec.t_eval.gate import (
    ACCESS_COLUMNS,
    EVENT_FIRST,
    EVENT_RESUMED,
    FAILED,
    PURPOSE,
    TEvalRefused,
    begin_formal_run,
    passed_rehearsal,
    require_clean_tree,
    worker_permit,
)
from hmrec.t_eval.plan import TEvalContext

SCRIPT = "scripts/m9_t_eval.py"


@pytest.fixture
def inputs(tmp_path: Path) -> TrialInputs:
    return build_t_eval_workspace(tmp_path)


def _access_rows(context: TEvalContext) -> list[dict[str, str]]:
    return read_rows(context.paths.m9_t_label_access_file, ACCESS_COLUMNS)


# ---------------------------------------------------------------- the rehearsal


def test_no_rehearsal_no_permit_and_nothing_is_written(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    with pytest.raises(TEvalRefused, match="no rehearsal is recorded"):
        begin_formal_run(context, NOW, SCRIPT)
    with pytest.raises(TEvalRefused, match="no rehearsal is recorded"):
        worker_permit(context)
    assert not context.paths.m9_t_label_access_file.exists()


def test_a_failed_rehearsal_gives_no_permit(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    record_rehearsal(context, status=FAILED, failed_checks="seed 42: values")
    with pytest.raises(TEvalRefused, match="did NOT pass: seed 42: values"):
        begin_formal_run(context, NOW, SCRIPT)
    assert not context.paths.m9_t_label_access_file.exists()


def test_only_the_latest_rehearsal_counts(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    record_rehearsal(context)
    record_rehearsal(context, status=FAILED, failed_checks="seed 43: top_k")
    with pytest.raises(TEvalRefused, match="did NOT pass"):
        passed_rehearsal(context)
    record_rehearsal(context)
    assert passed_rehearsal(context)["status"] == "passed"


def test_a_rehearsal_of_another_commit_gives_no_permit(inputs: TrialInputs) -> None:
    record_rehearsal(context_of(inputs))  # passed at COMMIT
    other = context_of(inputs, commit=OTHER_COMMIT)
    with pytest.raises(TEvalRefused, match="only at the commit of a passed rehearsal"):
        begin_formal_run(other, NOW, SCRIPT)
    assert not other.paths.m9_t_label_access_file.exists()


def test_a_rehearsal_from_a_dirty_tree_gives_no_permit(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    record_rehearsal(context, git_dirty="True")
    with pytest.raises(TEvalRefused, match="dirty: True"):
        passed_rehearsal(context)


def test_a_dirty_working_tree_gives_no_permit(inputs: TrialInputs) -> None:
    record_rehearsal(context_of(inputs))
    dirty = context_of(inputs, dirty=True)
    with pytest.raises(TEvalRefused, match="needs a clean working tree"):
        begin_formal_run(dirty, NOW, SCRIPT)
    with pytest.raises(TEvalRefused, match="needs a clean working tree"):
        require_clean_tree(dirty, "the rehearsal")
    assert not dirty.paths.m9_t_label_access_file.exists()


def test_a_rehearsal_of_another_configuration_or_other_seeds_gives_no_permit(
    inputs: TrialInputs,
) -> None:
    context = context_of(inputs)
    record_rehearsal(context, config_id="0" * 12)
    with pytest.raises(TEvalRefused, match="configuration"):
        passed_rehearsal(context)
    record_rehearsal(context, seeds="42")
    with pytest.raises(TEvalRefused, match="configured ones"):
        passed_rehearsal(context)
    record_rehearsal(context, target_fold="V3")
    with pytest.raises(TEvalRefused, match="configured ones"):
        passed_rehearsal(context)


# ---------------------------------------------------------------- the record of the read


def test_the_driver_records_the_read_and_gets_a_permit(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    record_rehearsal(context)
    permit = begin_formal_run(context, NOW, SCRIPT)
    assert permit == TEvalPermit(COMMIT, "2026-11-20T11:00:00", NOW, PURPOSE, SCRIPT)
    assert _access_rows(context) == [
        {
            "event": EVENT_FIRST,
            "recorded_at": NOW,
            "purpose": PURPOSE,
            "script": SCRIPT,
            "git_commit": COMMIT,
            "git_dirty": "False",
            "rehearsal_commit": COMMIT,
            "rehearsal_checked_at": "2026-11-20T11:00:00",
        }
    ]


def test_a_worker_never_records_and_needs_the_drivers_row(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    record_rehearsal(context)
    with pytest.raises(TEvalRefused, match="is not recorded"):
        worker_permit(context)
    assert not context.paths.m9_t_label_access_file.exists()
    begin_formal_run(context, NOW, SCRIPT)
    permit = worker_permit(context)
    assert (permit.git_commit, permit.recorded_at, permit.script) == (COMMIT, NOW, SCRIPT)
    assert len(_access_rows(context)) == 1  # the worker appended nothing


def test_a_continued_run_appends_its_own_row(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    record_rehearsal(context)
    begin_formal_run(context, NOW, SCRIPT)
    later = "2026-11-20T13:00:00"
    permit = begin_formal_run(context, later, SCRIPT)
    assert permit.recorded_at == later
    assert [(row["event"], row["recorded_at"]) for row in _access_rows(context)] == [
        (EVENT_FIRST, NOW),
        (EVENT_RESUMED, later),
    ]
    # A worker gets the latest row.
    assert worker_permit(context).recorded_at == later


def test_a_started_run_is_not_continued_at_another_commit(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    record_rehearsal(context)
    begin_formal_run(context, NOW, SCRIPT)
    # Even a passed rehearsal of the new commit does not help: T labels were read by other code.
    other = context_of(inputs, commit=OTHER_COMMIT)
    record_rehearsal(other, git_commit=OTHER_COMMIT)
    with pytest.raises(TEvalRefused, match="not\\s+continued with other code"):
        begin_formal_run(other, NOW, SCRIPT)
    with pytest.raises(TEvalRefused, match="not\\s+continued with other code"):
        worker_permit(other)
    assert len(_access_rows(context)) == 1


# ---------------------------------------------------------------- the label door


def test_t_labels_need_a_permit_and_only_t_folds_are_read(inputs: TrialInputs) -> None:
    paths = inputs.paths
    permit = TEvalPermit(COMMIT, NOW, NOW, PURPOSE, SCRIPT)
    t1, v4 = MANIFEST.get("T1"), MANIFEST.get("V4")
    for not_a_permit in (None, "permit", object()):
        with pytest.raises(TypeError, match="only with a TEvalPermit"):
            load_t_eval_evaluator(t1, paths.labels_file("T1"), not_a_permit)  # type: ignore[arg-type]  # the test passes wrong types on purpose
        with pytest.raises(TypeError, match="only with a TEvalPermit"):
            require_t_eval_permit(not_a_permit)  # type: ignore[arg-type]  # as above
    # The checks come before the file: a missing file is not even looked for.
    with pytest.raises(TypeError):
        load_t_eval_evaluator(t1, paths.labels_file("nowhere"), None)
    with pytest.raises(ValueError, match="evaluates only T folds"):
        load_t_eval_evaluator(v4, paths.labels_file("V4"), permit)
    evaluator = load_t_eval_evaluator(t1, paths.labels_file("T1"), permit)
    assert (evaluator.fold, evaluator.n_queries) == ("T1", 30)


def test_the_rehearsal_loader_reads_the_rehearsal_fold_only(inputs: TrialInputs) -> None:
    paths = inputs.paths
    v4 = MANIFEST.get("V4")
    assert load_t_eval_rehearsal_evaluator(v4, paths.labels_file("V4"), "V4").fold == "V4"
    for name in ("T1", "V3"):
        fold = MANIFEST.get(name)
        with pytest.raises(ValueError, match="only the configured rehearsal fold V4"):
            load_t_eval_rehearsal_evaluator(fold, paths.labels_file(name), "V4")
    # A T fold configured as the rehearsal fold by mistake is refused too.
    t1 = MANIFEST.get("T1")
    with pytest.raises(ValueError, match="its labels are not read"):
        load_t_eval_rehearsal_evaluator(t1, paths.labels_file("T1"), "T1")
