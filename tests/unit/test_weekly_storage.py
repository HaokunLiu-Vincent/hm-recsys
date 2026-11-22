"""Tests for weekly.storage.MountedStateStore: a commit without renaming a folder.

A bucket that is mounted as a volume has no folders, so "renaming a folder" is
not one step there. The store moves the files one by one and the step's record
(step.json) last; a commit that died in the middle is completed, not repeated.
The tests run on a local disk: the store uses only single-file operations, which
work everywhere. The local store's own tests are in test_weekly_state.py.
"""

from pathlib import Path

import pytest

from hmrec.weekly.steps import STEP_RECORD_FILE
from hmrec.weekly.storage import COMMIT_RECORD_FILE, LocalStateStore, MountedStateStore


@pytest.fixture
def store(tmp_path: Path) -> MountedStateStore:
    return MountedStateStore(tmp_path / "pipeline")


def _staged_step(store: MountedStateStore, with_record: bool = True) -> tuple[Path, Path]:
    """A staging folder with two outputs (one in a subfolder) and, if asked, the record."""
    staging = store.root / "weeks" / "2020-07-29" / ".staging" / "features"
    final = store.root / "weeks" / "2020-07-29" / "features"
    (staging / "shards" / "000").mkdir(parents=True)
    (staging / "shards" / "000" / "part.parquet").write_text("shard rows")
    (staging / "part-00000.parquet").write_text("rows")
    if with_record:
        (staging / COMMIT_RECORD_FILE).write_text("{}")
    return staging, final


def _files(folder: Path) -> list[str]:
    return sorted(
        path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()
    )


class _DiesAfter(MountedStateStore):
    """A store whose commit stops with an error after `moves` files were moved."""

    def __init__(self, root: Path, moves: int) -> None:
        super().__init__(root)
        self.moves_left = moves

    def _move_files(self, source: Path, target: Path, last: str | None) -> None:
        names = sorted(path for path in source.rglob("*") if path.is_file())
        ordered = [n for n in names if n.name != last] + [n for n in names if n.name == last]
        for origin in ordered:
            if self.moves_left == 0:
                raise OSError("the connection to the bucket was lost")
            destination = target / origin.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            origin.rename(destination)
            self.moves_left -= 1


def test_the_step_record_has_one_name() -> None:
    # The store moves the file the step writes last; both modules mean the same file.
    assert COMMIT_RECORD_FILE == STEP_RECORD_FILE == "step.json"


def test_a_commit_moves_every_file_and_removes_the_staging_folder(store: MountedStateStore) -> None:
    staging, final = _staged_step(store)
    assert not store.is_committed(final) and not store.commit_interrupted(staging, final)
    store.commit_dir(staging, final)
    assert _files(final) == ["part-00000.parquet", "shards/000/part.parquet", "step.json"]
    assert (final / "shards" / "000" / "part.parquet").read_text() == "shard rows"
    assert store.is_committed(final) and not staging.exists()


def test_an_output_is_complete_only_with_its_record(store: MountedStateStore) -> None:
    # A folder that exists without step.json is NOT a finished step (on a local disk a
    # folder that exists is: LocalStateStore renames it in one step).
    staging, final = _staged_step(store)
    final.mkdir(parents=True)
    (final / "part-00000.parquet").write_text("rows")
    assert not store.is_committed(final)
    assert LocalStateStore(store.root).is_committed(final)


def test_a_step_that_did_not_finish_is_never_committed(store: MountedStateStore) -> None:
    staging, final = _staged_step(store, with_record=False)
    with pytest.raises(FileNotFoundError, match="the step did not finish"):
        store.commit_dir(staging, final)
    assert not final.exists() and (staging / "part-00000.parquet").is_file()


def test_a_complete_output_is_never_overwritten(store: MountedStateStore) -> None:
    staging, final = _staged_step(store)
    store.commit_dir(staging, final)
    again, _ = _staged_step(store)
    with pytest.raises(FileExistsError, match="never overwritten"):
        store.commit_dir(again, final)


def test_a_commit_that_died_is_completed_by_the_next_call(tmp_path: Path) -> None:
    dying = _DiesAfter(tmp_path / "pipeline", moves=1)
    staging, final = _staged_step(dying)
    with pytest.raises(OSError, match="connection"):
        dying.commit_dir(staging, final)
    # One file is under the final name, the rest and the record are still in staging.
    assert _files(final) == ["part-00000.parquet"]
    assert _files(staging) == ["shards/000/part.parquet", "step.json"]
    store = MountedStateStore(tmp_path / "pipeline")
    assert not store.is_committed(final) and store.commit_interrupted(staging, final)
    store.commit_dir(staging, final)
    assert _files(final) == ["part-00000.parquet", "shards/000/part.parquet", "step.json"]
    assert store.is_committed(final) and not store.commit_interrupted(staging, final)
    assert not staging.exists()


def test_a_file_copied_but_not_removed_is_finished_and_a_different_one_refused(
    store: MountedStateStore,
) -> None:
    # A rename on a bucket is a copy and a delete. Dying between the two leaves the same
    # file under both names: the source is then removed, nothing is written twice.
    staging, final = _staged_step(store)
    final.mkdir(parents=True)
    (final / "part-00000.parquet").write_text("rows")
    store.commit_dir(staging, final)
    assert (final / "part-00000.parquet").read_text() == "rows" and store.is_committed(final)
    # Another size under the final name is another file: refused, nothing is replaced.
    other_staging = store.root / "weeks" / "2020-07-29" / ".staging" / "queries"
    other_final = store.root / "weeks" / "2020-07-29" / "queries"
    other_staging.mkdir(parents=True)
    (other_staging / "queries.parquet").write_text("the new attempt")
    (other_staging / COMMIT_RECORD_FILE).write_text("{}")
    other_final.mkdir()
    (other_final / "queries.parquet").write_text("old")
    with pytest.raises(FileExistsError, match="another size"):
        store.commit_dir(other_staging, other_final)
    assert (other_final / "queries.parquet").read_text() == "old"


def test_failed_attempts_are_set_aside_file_by_file(store: MountedStateStore) -> None:
    failed_dir = store.root / "weeks" / "2020-07-29" / ".failed"
    for attempt in (1, 2):
        staging, _ = _staged_step(store, with_record=False)
        moved = store.set_aside(staging, failed_dir, "O2_features")
        assert moved == failed_dir / f"O2_features.{attempt}"
        assert _files(moved) == ["part-00000.parquet", "shards/000/part.parquet"]
        assert not staging.exists()


def test_the_mounted_store_touches_nothing_outside_its_root(
    store: MountedStateStore, tmp_path: Path
) -> None:
    outside = tmp_path / "elsewhere"
    with pytest.raises(ValueError, match="not inside the pipeline state"):
        store.is_committed(outside)
    with pytest.raises(ValueError, match="not inside the pipeline state"):
        store.commit_interrupted(outside, store.root / "weeks")
