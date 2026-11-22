"""Tests for weekly.driver: one tick end to end, resume, refusal, a failing check, serve shards.

The steps are played by weekly_example.FakeSteps: small synthetic files, nothing
computed, nothing of the project's data read. What is tested is the driver: the
order of the steps, staging and renaming, what a resumed tick skips, when a tick
refuses to start, and that `current` moves only when every step passed.

Calendar (the real fold manifest): the first tick is V1 (2020-07-29); the week
before it, F4, is closed in the bootstrap state. V2 is 2020-08-05, V3 2020-08-12.
"""

import json
import subprocess
from pathlib import Path

import pytest
from weekly_example import (
    STAMP,
    V1,
    V2,
    V3,
    WEEK_ROWS,
    FakeSteps,
    build_workspace,
    patch_driver_environment,
    tick_context,
)

from hmrec.csv_records import read_rows
from hmrec.paths import (
    PIPELINE_INPUTS_ROOT_VARIABLE,
    PIPELINE_STATE_ROOT_VARIABLE,
    ProjectPaths,
    StateLayout,
    tick_root_paths,
)
from hmrec.run_manifest import (
    EVAL_MODE,
    FAILED,
    SERVE_MODE,
    STEP_DONE,
    STEP_SKIPPED,
    read_run_manifest,
)
from hmrec.run_records import RunStamp
from hmrec.weekly.driver import (
    CALIBRATION_METRIC,
    LABEL_ACCESS_COLUMNS,
    StepCall,
    next_anchor,
    run_tick,
    subprocess_runner,
)
from hmrec.weekly.history import MONITORING_COLUMNS
from hmrec.weekly.state import read_current
from hmrec.weekly.steps import steps_before_shards
from hmrec.weekly.storage import LocalStateStore, MountedStateStore

EVAL_STEPS = [
    "C1_ingest",
    "O1_queries",
    "O1_recall_r1_popularity",
    "O1_recall_r2_repurchase",
    "O1_recall_r3_sibling",
    "O1_recall_r4_itemcf",
    "O1_recall_r5_wals",
    "O1_recall_r6_twotower",
    "O2_features",
    "O3_dataset",
    "O3_train",
    "O4_score",
    "O5_publish",
]
CLOSING = ["C2_evaluate", "C3_sample", "C4_fine_tune_r6"]


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectPaths:
    patch_driver_environment(monkeypatch)
    return build_workspace(tmp_path)


def _layout(paths: ProjectPaths) -> StateLayout:
    return StateLayout(paths.pipeline_state_dir)


def _current(paths: ProjectPaths):
    layout = _layout(paths)
    return read_current(LocalStateStore(layout.root), layout)


def _monitoring_rows(paths: ProjectPaths) -> list[dict[str, str]]:
    return read_rows(paths.replay_table_file("monitoring"), MONITORING_COLUMNS)


# ---------------------------------------------------------------- one tick end to end


def test_first_tick_opens_the_week_and_publishes(paths: ProjectPaths) -> None:
    steps = FakeSteps(paths)
    result = run_tick(paths, V1, EVAL_MODE, steps, STAMP)
    assert result.succeeded and result.manifest.run_id == "2020-07-29-eval-a1"
    # F4 is closed in the bootstrap state: the closing half has nothing to do but C1.
    assert steps.calls == EVAL_STEPS
    statuses = {step.name: step.status for step in result.manifest.steps}
    assert [statuses[name] for name in CLOSING] == [STEP_SKIPPED] * 3
    assert all(statuses[name] == STEP_DONE for name in EVAL_STEPS)
    layout = _layout(paths)
    current = _current(paths)
    assert current is not None
    assert (current.anchor, current.mode, current.closed_week.isoformat()) == (
        V1,
        EVAL_MODE,
        "2020-07-22",
    )
    assert current.prediction_set == "predictions/2020-07-29"
    assert {model.role for model in current.models} == {"ranker", "twotower_chain"}
    # The R6 model behind the first week's predictions is the bootstrap state's.
    chain = next(model for model in current.models if model.role == "twotower_chain")
    assert chain.path.endswith("data/bootstrap/models/twotower/chain_locked/F4.pt")
    assert layout.published_file(V1).is_file() and not layout.closed_file(V1).exists()
    # Every step's folder was renamed out of the staging folder.
    week = layout.week_dir(V1)
    assert (week / "features" / "features.json").is_file()
    assert not list((week / ".staging").iterdir())
    assert read_run_manifest(result.manifest_file) == result.manifest


def test_tick_records_metrics_outputs_and_same_week_findings(paths: ProjectPaths) -> None:
    result = run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    manifest = result.manifest
    names = {(metric.name, metric.week) for metric in manifest.metrics}
    assert ("output/distinct_articles", V1) in names
    assert ("monitor/flags_raised", V1) in names and ("monitor/alerts_raised", V1) in names
    roles = {file.role for file in manifest.outputs}
    assert {"queries", "candidates/r4_itemcf", "ranker_model", "prediction_set"} <= roles
    assert all(file.path.startswith("artifacts/pipeline/") for file in manifest.outputs)
    assert {"lock/ranker", "config/monitoring"} <= {file.role for file in manifest.inputs}
    # The findings are in the state and in the table: input data, 2 features x 2 rules,
    # 3 output signals. F1-F4 give 4 weeks of history for the features and for
    # candidates per query; the two signals that need a ranking are recorded only.
    rows = _monitoring_rows(paths)
    by_rule = {(row["rule"], row["subject"]): row for row in rows}
    assert by_rule[("input/transaction_rows", "")]["flag"] == "False"
    assert by_rule[("feature/missing_rate", "pop_rank")]["n_history"] == "4"
    assert by_rule[("output/signal", "candidates_per_query")]["n_history"] == "4"
    assert by_rule[("output/signal", "distinct_articles")]["limit"] == "recorded only"
    assert not any(row["alert"] == "True" for row in rows)
    assert _layout(paths).monitoring_file(V1).is_file()


def test_next_tick_closes_the_previous_week(paths: ProjectPaths) -> None:
    run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    steps = FakeSteps(paths)
    result = run_tick(paths, V2, EVAL_MODE, steps, STAMP)
    assert result.succeeded
    assert steps.calls == [EVAL_STEPS[0], *CLOSING, *EVAL_STEPS[1:]]
    layout = _layout(paths)
    assert layout.closed_file(V1).is_file()
    # C2-C4 wrote into the CLOSED week's folder, the rest into the tick's.
    assert (layout.week_dir(V1) / "evaluation" / "metrics.json").is_file()
    assert (layout.week_dir(V1) / "twotower" / "V1.pt").is_file()
    assert (layout.week_dir(V2) / "scores" / "top.parquet").is_file()
    current = _current(paths)
    assert current is not None and current.anchor == V2
    assert current.closed_week == V1
    chain = next(model for model in current.models if model.role == "twotower_chain")
    assert chain.path == "weeks/2020-07-29/twotower/V1.pt"
    # A labelled metric names the week it describes: the closed one.
    assert ("map@12", V1) in {(m.name, m.week) for m in result.manifest.metrics}
    labelled = [row for row in _monitoring_rows(paths) if row["layer"] == "labelled"]
    assert {row["week"] for row in labelled} == {"2020-07-29"}
    assert {"recall/channel", "recall/union", "ranker/above_rrf_baseline", "hit_value"} == {
        row["rule"] for row in labelled
    }


def test_label_reads_are_recorded_before_the_step(paths: ProjectPaths) -> None:
    run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    run_tick(paths, V2, EVAL_MODE, FakeSteps(paths), STAMP)
    rows = read_rows(paths.replay_table_file("label_access"), LABEL_ACCESS_COLUMNS)
    assert [(row["tick_anchor"], row["step"], row["week"]) for row in rows] == [
        ("2020-07-29", "O1_queries", "V1"),
        ("2020-08-05", "C2_evaluate", "V1"),
        ("2020-08-05", "C3_sample", "V1"),
        ("2020-08-05", "O1_queries", "V2"),
    ]


def test_a_quantity_two_steps_measure_is_recorded_once(paths: ProjectPaths) -> None:
    # The feature step and the scoring step both report the candidates per query (300).
    # A run manifest holds a metric name once per week: an equal value is recorded once.
    result = run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    assert result.succeeded
    values = [m.value for m in result.manifest.metrics if m.name == "output/candidates_per_query"]
    assert values == [300.0]


def test_two_steps_that_disagree_about_a_quantity_stop_the_tick(paths: ProjectPaths) -> None:
    steps = FakeSteps(paths, candidates_per_query=299.0)
    result = run_tick(paths, V1, EVAL_MODE, steps, STAMP)
    assert result.manifest.status == FAILED
    assert (
        "O4_score" in result.manifest.failure
        and "an earlier step recorded" in result.manifest.failure
    )
    # The tick stopped before O5: nothing was published.
    assert steps.calls[-1] == "O4_score" and _current(paths) is None


# ---------------------------------------------------------------- refusal


def test_tick_refuses_to_start_when_the_previous_week_is_not_closed(paths: ProjectPaths) -> None:
    steps = FakeSteps(paths)
    with pytest.raises(RuntimeError, match="not closed and has no published predictions"):
        run_tick(paths, V2, EVAL_MODE, steps, STAMP)
    # The tick did not start: no step ran, no manifest and no pointer was written.
    assert steps.calls == []
    assert not _layout(paths).runs_dir.exists() and _current(paths) is None


def test_tick_refuses_a_week_before_the_first_tick_and_uncommitted_code(
    paths: ProjectPaths,
) -> None:
    from datetime import date

    with pytest.raises(RuntimeError, match="before the first tick"):
        run_tick(paths, date(2020, 7, 22), EVAL_MODE, FakeSteps(paths), STAMP)
    dirty = RunStamp(STAMP.git_commit, True)
    with pytest.raises(RuntimeError, match="differs from commit"):
        run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), dirty)


def test_a_tick_is_not_continued_in_another_mode(paths: ProjectPaths) -> None:
    run_tick(paths, V1, EVAL_MODE, FakeSteps(paths, fail={"O2_features"}), STAMP)
    with pytest.raises(RuntimeError, match="was started in mode 'eval'"):
        run_tick(paths, V1, SERVE_MODE, FakeSteps(paths), STAMP)


# ---------------------------------------------------------------- a failed step, then resume


def test_failed_step_stops_the_tick_and_leaves_current_unset(paths: ProjectPaths) -> None:
    steps = FakeSteps(paths, fail={"O2_features"})
    result = run_tick(paths, V1, EVAL_MODE, steps, STAMP)
    manifest = result.manifest
    assert manifest.status == FAILED and not result.succeeded
    # The reason is what the step's process wrote before it ended.
    assert (
        "O2_features" in manifest.failure and "a check of the fake step failed" in manifest.failure
    )
    assert steps.calls[-1] == "O2_features"  # nothing ran after it
    layout = _layout(paths)
    assert _current(paths) is None and not layout.published_file(V1).exists()
    # The half-written folder is kept aside, not deleted and not in place.
    assert not (layout.week_dir(V1) / "features").exists()
    assert (layout.failed_dir(V1) / "O2_features.1" / "partial.txt").is_file()
    alerts = [row for row in _monitoring_rows(paths) if row["alert"] == "True"]
    assert [(row["rule"], row["subject"]) for row in alerts] == [
        ("pipeline/step_failed", "O2_features")
    ]


def test_resumed_tick_skips_finished_steps_and_writes_a_new_manifest(paths: ProjectPaths) -> None:
    first = run_tick(paths, V1, EVAL_MODE, FakeSteps(paths, fail={"O2_features"}), STAMP)
    steps = FakeSteps(paths)
    second = run_tick(paths, V1, EVAL_MODE, steps, STAMP)
    assert second.succeeded and second.manifest.run_id == "2020-07-29-eval-a2"
    # Only the failed step and the ones after it ran again.
    assert steps.calls == EVAL_STEPS[EVAL_STEPS.index("O2_features") :]
    statuses = {step.name: step.status for step in second.manifest.steps}
    assert statuses["O1_recall_r5_wals"] == STEP_SKIPPED and statuses["O2_features"] == STEP_DONE
    # The first manifest is still there, unchanged; the skipped steps' outputs are in the new one.
    assert read_run_manifest(first.manifest_file).status == FAILED
    assert "candidates/r5_wals" in {file.role for file in second.manifest.outputs}
    current = _current(paths)
    assert current is not None and current.run_id == "2020-07-29-eval-a2"


def test_a_finished_tick_run_again_does_nothing(paths: ProjectPaths) -> None:
    run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    before = _layout(paths).current_file.read_text(encoding="utf-8")
    steps = FakeSteps(paths)
    again = run_tick(paths, V1, EVAL_MODE, steps, STAMP)
    assert again.succeeded and steps.calls == []
    assert _layout(paths).current_file.read_text(encoding="utf-8") == before
    # The findings of the tick are recorded once.
    assert len([row for row in _monitoring_rows(paths) if row["rule"] == "input/date_range"]) == 1


# ---------------------------------------------------------------- a failing check


def test_failing_input_check_stops_the_tick_and_leaves_current_unchanged(
    paths: ProjectPaths,
) -> None:
    run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    before = _layout(paths).current_file.read_text(encoding="utf-8")
    # The new week has 40% of the rows of the weeks before it: below the 50% band.
    steps = FakeSteps(paths, new_week_rows=int(0.4 * WEEK_ROWS))
    result = run_tick(paths, V2, EVAL_MODE, steps, STAMP)
    assert result.manifest.status == FAILED
    assert "input data check failed" in result.manifest.failure
    assert steps.calls == ["C1_ingest"]  # the tick stopped at the check
    layout = _layout(paths)
    assert layout.current_file.read_text(encoding="utf-8") == before
    current = _current(paths)
    assert current is not None and current.anchor == V1
    assert not (layout.week_dir(V2) / "ingest").exists()
    assert (layout.failed_dir(V2) / "C1_ingest.1" / "ingest.json").is_file()
    alerts = {
        (row["rule"], row["subject"]) for row in _monitoring_rows(paths) if row["alert"] == "True"
    }
    assert alerts == {("input/transaction_rows", ""), ("pipeline/step_failed", "C1_ingest")}
    names = {metric.name for metric in result.manifest.metrics}
    assert "monitor/alert/input/transaction_rows" in names


def test_low_union_recall_is_flagged_and_becomes_an_alert_after_three_weeks(
    paths: ProjectPaths,
) -> None:
    from datetime import date

    anchors = [V1, V2, V3, date(2020, 8, 19), date(2020, 8, 26)]
    for anchor in anchors:
        result = run_tick(paths, anchor, EVAL_MODE, FakeSteps(paths, union_recall=0.20), STAMP)
        assert result.succeeded  # a flag or an alert never stops a tick or changes anything
    union = [row for row in _monitoring_rows(paths) if row["rule"] == "recall/union"]
    # Weeks V1, V2, V3, V4 were closed: flagged every week, an alert from the third on.
    assert [row["week"] for row in union] == [
        "2020-07-29",
        "2020-08-05",
        "2020-08-12",
        "2020-08-19",
    ]
    assert [row["flag"] for row in union] == ["True"] * 4
    assert [row["consecutive_weeks"] for row in union] == ["1", "2", "3", "4"]
    assert [row["alert"] for row in union] == ["False", "False", "True", "True"]


# ---------------------------------------------------------------- serve mode


def test_serve_tick_scores_shard_by_shard_and_removes_each_shards_features(
    paths: ProjectPaths,
) -> None:
    run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    steps = FakeSteps(paths)
    result = run_tick(paths, V2, SERVE_MODE, steps, STAMP)
    assert result.succeeded
    # 5 customers in shards of 2: three shards, each built and scored before the next.
    shard_calls = [name for name in steps.calls if "shard" in name]
    assert shard_calls == [
        "O2_features_shard_000",
        "O4_score_shard_000",
        "O2_features_shard_001",
        "O4_score_shard_001",
        "O2_features_shard_002",
        "O4_score_shard_002",
    ]
    assert steps.calls.index("O3_train") < steps.calls.index("O2_features_shard_000")
    assert steps.calls[-2:] == ["O4_score", "O5_publish"]
    week = _layout(paths).week_dir(V2)
    for shard in ("000", "001", "002"):
        assert not (week / "shards" / shard / "features").exists()
        assert (week / "shards" / shard / "top" / "top.parquet").is_file()
    current = _current(paths)
    assert current is not None and (current.anchor, current.mode) == (V2, SERVE_MODE)
    # Feature statistics of a serve week are merged from the shards; with no serve week
    # before, every same-week value is recorded only.
    serve_rows = [row for row in _monitoring_rows(paths) if row["tick_anchor"] == "2020-08-05"]
    features = [row for row in serve_rows if row["rule"] == "feature/missing_rate"]
    assert len(features) == 2 and {row["limit"] for row in features} == {"recorded only"}


def test_resumed_serve_tick_does_not_rebuild_a_scored_shard(paths: ProjectPaths) -> None:
    run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    run_tick(paths, V2, SERVE_MODE, FakeSteps(paths, fail={"O4_score_shard_001"}), STAMP)
    steps = FakeSteps(paths)
    result = run_tick(paths, V2, SERVE_MODE, steps, STAMP)
    assert result.succeeded
    # Shard 0 is scored (its features are gone): neither step runs again. Shard 1's
    # features were finished; only its score runs.
    assert [name for name in steps.calls if "shard" in name] == [
        "O4_score_shard_001",
        "O2_features_shard_002",
        "O4_score_shard_002",
    ]


def test_a_serve_week_cannot_be_closed(paths: ProjectPaths) -> None:
    run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    run_tick(paths, V2, SERVE_MODE, FakeSteps(paths), STAMP)
    with pytest.raises(RuntimeError, match="was opened in serve mode"):
        run_tick(paths, V3, EVAL_MODE, FakeSteps(paths), STAMP)


def test_state_files_are_valid_json(paths: ProjectPaths) -> None:
    run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    layout = _layout(paths)
    for file in (layout.current_file, layout.published_file(V1), layout.monitoring_file(V1)):
        json.loads(file.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- M13: roots, next tick


def _mounted(paths: ProjectPaths, tmp_path: Path) -> ProjectPaths:
    """The same workspace with inputs and state OUTSIDE the repository, as on a mounted bucket.

    The workspace's data/ is moved to {mount}/inputs/data/, and the state is to
    be {mount}/pipeline/. Returns the paths the two environment variables give.
    """
    mount = tmp_path.parent / f"{tmp_path.name}-mount"
    (mount / "inputs").mkdir(parents=True)
    paths.data_root.rename(mount / "inputs" / "data")
    environment = {
        PIPELINE_INPUTS_ROOT_VARIABLE: str(mount / "inputs"),
        PIPELINE_STATE_ROOT_VARIABLE: str(mount / "pipeline"),
    }
    return tick_root_paths(paths, environment, calibration=False)


def test_a_tick_runs_with_inputs_and_state_outside_the_repository(
    paths: ProjectPaths, tmp_path: Path
) -> None:
    mounted = _mounted(paths, tmp_path)
    state = mounted.pipeline_state_dir
    assert state.name == "pipeline" and not state.is_relative_to(paths.project_root)
    steps = FakeSteps(mounted)
    result = run_tick(mounted, V1, EVAL_MODE, steps, STAMP)
    assert result.succeeded and steps.calls == EVAL_STEPS
    # State, manifests and result tables are all under the state root; the repository
    # got nothing: no artifacts/pipeline/, no results/m12_replay/.
    assert result.manifest_file == state / "runs" / "2020-07-29-eval-a1" / "manifest.json"
    assert (state / "weeks" / "2020-07-29" / "features" / "features.json").is_file()
    assert (state / "results" / "steps.csv").is_file()
    assert (state / "results" / "monitoring.csv").is_file()
    assert not paths.pipeline_state_dir.exists() and not paths.replay_results_dir.exists()
    # A file is recorded by its key below the mount: "pipeline/..." and "inputs/...";
    # a file of the code (a lock, a config) by its path in the repository.
    output_paths = [file.path for file in result.manifest.outputs]
    assert "pipeline/weeks/2020-07-29/ingest/transactions.parquet" in output_paths
    assert any(path.startswith("pipeline/predictions/2020-07-29/") for path in output_paths)
    assert all(path.startswith("pipeline/") for path in output_paths)
    inputs = {file.role: file.path for file in result.manifest.inputs}
    assert inputs["lock/ranker"] == "configs/locks/ranker.lock.yaml"
    current = _current(mounted)
    assert current is not None and current.prediction_set == "predictions/2020-07-29"
    chain = next(model for model in current.models if model.role == "twotower_chain")
    assert chain.path == "inputs/data/bootstrap/models/twotower/chain_locked/F4.pt"
    # The next tick finds the state there and closes V1.
    assert run_tick(mounted, V2, EVAL_MODE, FakeSteps(mounted), STAMP).succeeded
    assert (state / "weeks" / "2020-07-29" / "closed.json").is_file()


class _CommitDiesOnce(MountedStateStore):
    """A mounted store whose commit of ONE step stops after its first file, once."""

    def __init__(self, root: Path, step_folder: str) -> None:
        super().__init__(root)
        self.step_folder = step_folder
        self.died = False

    def commit_dir(self, staging: Path, final: Path) -> None:
        if final.name == self.step_folder and not self.died:
            self.died = True
            first = sorted(path for path in staging.rglob("*") if path.is_file())[0]
            final.mkdir(parents=True)
            first.rename(final / first.name)
            raise OSError("the connection to the bucket was lost")
        super().commit_dir(staging, final)


def test_a_commit_that_did_not_end_is_completed_without_running_the_step_again(
    paths: ProjectPaths, tmp_path: Path
) -> None:
    mounted = _mounted(paths, tmp_path)
    layout = _layout(mounted)
    store = _CommitDiesOnce(layout.root, "features")
    steps = FakeSteps(mounted)
    stopped = run_tick(mounted, V1, EVAL_MODE, steps, STAMP, store)
    assert not stopped.succeeded and "O2_features" in stopped.manifest.failure
    final, staging = layout.step_dir(V1, "features"), layout.staging_dir(V1, "features")
    # The step finished; half of its files are in place, the record is still in staging.
    # Nothing was moved to .failed/: this is not a failed attempt.
    assert final.is_dir() and not (final / "step.json").exists()
    assert (staging / "step.json").is_file()
    assert not layout.failed_dir(V1).exists()
    assert _current(mounted) is None
    resumed = run_tick(mounted, V1, EVAL_MODE, steps, STAMP, store)
    assert resumed.succeeded and resumed.manifest.run_id == "2020-07-29-eval-a2"
    # The feature step ran ONCE; the second run only completed its commit.
    assert steps.calls.count("O2_features") == 1
    assert (final / "step.json").is_file() and (final / "features.json").is_file()
    assert not staging.exists()
    statuses = {step.name: step.status for step in resumed.manifest.steps}
    assert statuses["O2_features"] == STEP_SKIPPED and statuses["O3_train"] == STEP_DONE


def test_the_next_tick_is_the_week_after_the_current_prediction_set(paths: ProjectPaths) -> None:
    # What a scheduler runs every week: the same arguments, the anchor from the state.
    assert next_anchor(paths) == V1
    assert run_tick(paths, next_anchor(paths), EVAL_MODE, FakeSteps(paths), STAMP).succeeded
    assert next_anchor(paths) == V2
    # A tick that stops leaves the pointer where it was: the same tick is due again.
    failing = FakeSteps(paths, fail={"O2_features"})
    assert not run_tick(paths, next_anchor(paths), EVAL_MODE, failing, STAMP).succeeded
    assert next_anchor(paths) == V2
    assert run_tick(paths, next_anchor(paths), EVAL_MODE, FakeSteps(paths), STAMP).succeeded
    assert next_anchor(paths) == V3


def test_a_calibration_tick_writes_only_into_its_own_state(paths: ProjectPaths) -> None:
    calibration = tick_root_paths(paths, {}, calibration=True)
    state = calibration.pipeline_state_dir
    assert state == paths.artifacts_root / "pipeline-calibration"
    seen: list[bool] = []
    steps = FakeSteps(calibration)

    def runner(call: StepCall) -> None:
        seen.append(call.calibration)
        steps(call)

    result = run_tick(calibration, V1, EVAL_MODE, runner, STAMP)
    assert result.succeeded
    # Every step process is told that it belongs to a calibration tick.
    assert seen and all(seen)
    assert result.manifest_file.is_relative_to(state)
    assert (state / "results" / "steps.csv").is_file()
    # The pipeline's own state and the M12 result tables were not touched.
    assert not paths.pipeline_state_dir.exists() and not paths.replay_results_dir.exists()
    marks = [metric for metric in result.manifest.metrics if metric.name == CALIBRATION_METRIC]
    assert [(metric.week, metric.value) for metric in marks] == [(V1, 1.0)]
    # A tick of the pipeline itself is not marked, and starts from the bootstrap state.
    plain = run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    assert plain.succeeded
    assert all(metric.name != CALIBRATION_METRIC for metric in plain.manifest.metrics)


def test_the_step_process_is_started_with_the_calibration_flag(
    paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = subprocess_runner(Path("/app/scripts"))
    context = tick_context(paths, V1)
    step = steps_before_shards(context.tick, context.closing, EVAL_MODE)[0]
    runner(StepCall(V1, EVAL_MODE, "2020-07-29-eval-a1", step))
    runner(StepCall(V1, EVAL_MODE, "2020-07-29-eval-a1", step, calibration=True))
    assert "--calibration" not in commands[0] and commands[1][-1] == "--calibration"
    assert commands[0][1:] == [
        "/app/scripts/m12_step.py",
        "--anchor",
        "2020-07-29",
        "--mode",
        "eval",
        "--run-id",
        "2020-07-29-eval-a1",
        "--step",
        "C1_ingest",
    ]
