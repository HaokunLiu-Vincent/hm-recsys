"""Tests for hmrec.tracking.store: the local MLflow store.

MLflow is in the dependency group "tracking". The laptop installs it for
`make mlflow-sync`; the pipeline image does not. Both cases are tested, each
with its own assertions.
"""

from pathlib import Path

import pytest
from tracking_example import weekly_run

from hmrec.tracking.check import check_logged_runs
from hmrec.tracking.store import (
    load_mlflow,
    mlflow_is_installed,
    open_client,
    read_experiment_runs,
    store_uri,
    sync_runs,
)


def test_store_uri_is_a_sqlite_file(tmp_path: Path) -> None:
    store_file = tmp_path / "mlflow" / "mlflow.db"
    # An absolute path follows the three slashes of the scheme: four in a row.
    assert store_uri(store_file) == f"sqlite:///{store_file.resolve().as_posix()}"
    assert store_uri(store_file).startswith("sqlite:////")


def test_sync_then_read_back(tmp_path: Path) -> None:
    run = weekly_run(tmp_path)
    store_file = tmp_path / "mlflow" / "mlflow.db"
    artifacts_dir = tmp_path / "mlflow" / "artifacts"
    if mlflow_is_installed():
        # The laptop with the "tracking" group: a real SQLite store in a temporary folder.
        client = open_client(store_file)
        summary = sync_runs(client, [run], artifacts_dir)
        assert (summary.logged, summary.skipped) == (("stage1/V1",), ())
        logged = read_experiment_runs(client, run.experiment)
        assert len(logged) == 1
        # The store gives back the same float64 values and every tag.
        assert logged[0].metrics == {metric.key: metric.value for metric in run.metrics}
        assert logged[0].record_id == "stage1/V1"
        for key, value in run.all_tags().items():
            assert logged[0].tags[key] == value
        assert check_logged_runs(logged, [run], tmp_path).problems == ()
        # A second sync changes nothing and adds nothing.
        again = sync_runs(open_client(store_file), [run], artifacts_dir)
        assert (again.logged, again.skipped) == ((), ("stage1/V1",))
        assert len(read_experiment_runs(client, run.experiment)) == 1
        assert read_experiment_runs(client, "no such experiment") == []
        with pytest.raises(ValueError, match="same record id"):
            sync_runs(client, [run, run], artifacts_dir)
    else:
        # The pipeline image: MLflow is not installed, and the code says how to get it
        # instead of failing with an import error. Nothing is written.
        with pytest.raises(RuntimeError, match="dependency group 'tracking'"):
            load_mlflow()
        with pytest.raises(RuntimeError, match="uv sync --group tracking"):
            open_client(store_file)
        assert not store_file.parent.exists()
