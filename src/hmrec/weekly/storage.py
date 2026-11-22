"""The storage interface of the weekly pipeline's state (docs/mlops.md §6.4, §7.2).

The driver never touches the state's files directly: it goes through a
StateStore. There are two implementations:

    LocalStateStore     a folder on a local disk (artifacts/pipeline/ on the
                        laptop, M12). Renaming a folder is ONE step there.
    MountedStateStore   a folder of a bucket that is mounted as a volume (M13;
                        Cloud Storage FUSE on Cloud Run). The keys are the same.
                        A bucket has no folders: "renaming a folder" is a copy
                        and a delete per file, so a process that dies in the
                        middle leaves half of the files under each name. This
                        store therefore moves the files one by one and the
                        step's record LAST (see MountedStateStore).

The rules of §6.4 are the store's methods:

    commit_dir          a step writes under a temporary (staging) folder; when its
                        checks passed the folder is renamed. An output that
                        exists is NEVER overwritten.
    write_text_once     a record file is written under a temporary name and
                        renamed; it is never replaced.
    replace_text        the one file that is replaced: state/current.json (step O5).
    set_aside           the staging folder of a failed attempt is moved to a
                        ".failed" folder. Nothing of a failed attempt is deleted
                        or reused.
    discard_dir         removes an intermediate the design says to remove: the
                        features of a shard after the shard was scored (§8.4).
    is_committed        True if a step's output is complete.
    commit_interrupted  True if a commit began and did not end (possible only
                        where a folder rename is not one step).

This module imports only the standard library.
"""

import shutil
from pathlib import Path
from typing import Protocol

TEMPORARY_SUFFIX = ".tmp"
# The record a step writes as its LAST action (hmrec.weekly.context.finish_step). A
# mounted store moves it last, so its presence in the final folder says "complete".
COMMIT_RECORD_FILE = "step.json"


class StateStore(Protocol):
    """What the driver needs from the place the state lives in."""

    @property
    def root(self) -> Path:
        """The folder (or the local mirror of the prefix) that holds the state."""
        ...

    def exists(self, path: Path) -> bool:
        """True if a file or folder of the state exists."""
        ...

    def read_text(self, path: Path) -> str:
        """The text of a file of the state."""
        ...

    def write_text_once(self, path: Path, text: str) -> None:
        """Write a new file; fails if it exists."""
        ...

    def replace_text(self, path: Path, text: str) -> None:
        """Write a file, replacing an existing one in one step."""
        ...

    def commit_dir(self, staging: Path, final: Path) -> None:
        """Make a finished staging folder the final output; fails if the output exists."""
        ...

    def set_aside(self, staging: Path, failed_dir: Path, name: str) -> Path:
        """Move the staging folder of a failed attempt away; returns where it went."""
        ...

    def discard_dir(self, path: Path) -> None:
        """Remove a folder of the state (only for intermediates the design removes)."""
        ...

    def is_committed(self, final: Path) -> bool:
        """True if the output folder of a step is complete."""
        ...

    def commit_interrupted(self, staging: Path, final: Path) -> bool:
        """True if commit_dir(staging, final) began earlier and did not end."""
        ...


class LocalStateStore:
    """The state in a folder of this machine."""

    def __init__(self, root: Path) -> None:
        """Remember the root; nothing is created yet.

        Args:
            root: The state folder (hmrec.paths.ProjectPaths.pipeline_state_dir).
        """
        self._root = root

    @property
    def root(self) -> Path:
        """The state folder."""
        return self._root

    def _inside(self, path: Path) -> Path:
        """`path` after checking that it lies inside the state folder.

        Raises:
            ValueError: If it does not. The store never reads or writes elsewhere.
        """
        if path != self._root and self._root not in path.parents:
            raise ValueError(f"{path} is not inside the pipeline state {self._root}")
        return path

    def exists(self, path: Path) -> bool:
        """True if the file or folder exists."""
        return self._inside(path).exists()

    def read_text(self, path: Path) -> str:
        """The text of a file (UTF-8)."""
        return self._inside(path).read_text(encoding="utf-8")

    def write_text_once(self, path: Path, text: str) -> None:
        """Write a new file under a temporary name and rename it.

        Raises:
            FileExistsError: If the file exists (it is never replaced).
        """
        target = self._inside(path)
        if target.exists():
            raise FileExistsError(f"{target} already exists; it is never replaced")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + TEMPORARY_SUFFIX)
        temporary.write_text(text, encoding="utf-8")
        temporary.rename(target)

    def replace_text(self, path: Path, text: str) -> None:
        """Write a file under a temporary name and put it in place of the old one.

        Path.replace is one step for the reader: it sees the old file or the new
        one, never half of one.
        """
        target = self._inside(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + TEMPORARY_SUFFIX)
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(target)

    def commit_dir(self, staging: Path, final: Path) -> None:
        """Rename a staging folder to its final name.

        Raises:
            FileNotFoundError: If the staging folder does not exist.
            FileExistsError: If the final folder exists (an output is never overwritten).
        """
        source, target = self._inside(staging), self._inside(final)
        if not source.is_dir():
            raise FileNotFoundError(f"staging folder {source} does not exist")
        if target.exists():
            raise FileExistsError(f"{target} already exists; an output is never overwritten")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)

    def set_aside(self, staging: Path, failed_dir: Path, name: str) -> Path:
        """Move a staging folder into `failed_dir` under a name that is not taken yet.

        Args:
            staging: The staging folder of the failed attempt.
            failed_dir: The week's ".failed" folder.
            name: The step's name; the folder becomes "{name}.{n}" with the
                first n that is free.

        Returns:
            The new place, e.g. .failed/O1_recall_r5_wals.2 for the second failed attempt.
        """
        source = self._inside(staging)
        folder = self._inside(failed_dir)
        folder.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (folder / f"{name}.{attempt}").exists():
            attempt += 1
        target = folder / f"{name}.{attempt}"
        source.rename(target)
        return target

    def discard_dir(self, path: Path) -> None:
        """Remove a folder and what is in it (a missing folder is not an error)."""
        target = self._inside(path)
        if target == self._root:
            raise ValueError("the state folder itself is never removed")
        if target.exists():
            shutil.rmtree(target)

    def is_committed(self, final: Path) -> bool:
        """True if the folder exists: a local rename puts all of it in place at once."""
        return self._inside(final).exists()

    def commit_interrupted(self, staging: Path, final: Path) -> bool:
        """Always False: a local rename happens or does not happen."""
        self._inside(staging)
        self._inside(final)
        return False


def files_below(folder: Path) -> list[Path]:
    """Every file below a folder, as paths relative to it, in sorted order."""
    return sorted(path.relative_to(folder) for path in folder.rglob("*") if path.is_file())


def remove_empty_folders(folder: Path) -> None:
    """Remove `folder` and the folders below it; all of them must be empty of files.

    Raises:
        OSError: If a file is left (os.rmdir refuses a folder that is not empty).
    """
    below = sorted((path for path in folder.rglob("*") if path.is_dir()), reverse=True)
    for path in [*below, folder]:
        path.rmdir()


class MountedStateStore(LocalStateStore):
    """The state in a folder of a bucket that is mounted as a volume (M13).

    Reading, writing a file and renaming ONE file work as on a disk. A folder
    is not an object of a bucket, so this store never renames a folder:

        commit_dir   moves the files of the staging folder one by one, in sorted
                     order, and the step's record (step.json) LAST. The output
                     is complete if and only if its record is there
                     (is_committed).
        resume       a process that died during a commit leaves files under both
                     names. The step itself had finished (its record exists, in
                     the staging folder), so commit_dir called again moves the
                     rest; the step is not computed again (commit_interrupted).
        set_aside    moves the files one by one as well.

    Every method works on a local disk too (the tests run there); it is only
    slower than one rename.
    """

    def is_committed(self, final: Path) -> bool:
        """True if the output folder holds the step's record, which is moved last."""
        return (self._inside(final) / COMMIT_RECORD_FILE).is_file()

    def commit_interrupted(self, staging: Path, final: Path) -> bool:
        """True if a commit began and did not end.

        The final folder exists without its record, and the staging folder
        still holds the record: the step finished, the move did not.
        """
        source, target = self._inside(staging), self._inside(final)
        return (
            target.exists()
            and not (target / COMMIT_RECORD_FILE).is_file()
            and (source / COMMIT_RECORD_FILE).is_file()
        )

    def _move_files(self, source: Path, target: Path, last: str | None) -> None:
        """Move every file below `source` to the same place below `target`.

        Args:
            source: The folder to empty.
            target: The folder to fill.
            last: A file name directly in `source` that is moved after all others.

        Raises:
            FileExistsError: If a file exists under both names with different
                sizes. Equal sizes mean that an earlier move copied the file and
                died before it removed the source: the source is then removed.
        """
        names = files_below(source)
        if last is not None:
            names = [name for name in names if name != Path(last)] + [Path(last)]
        for name in names:
            origin, destination = source / name, target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.stat().st_size != origin.stat().st_size:
                    raise FileExistsError(
                        f"{destination} exists with another size than {origin}; "
                        "an output is never overwritten"
                    )
                origin.unlink()
            else:
                origin.rename(destination)
        remove_empty_folders(source)

    def commit_dir(self, staging: Path, final: Path) -> None:
        """Move a finished staging folder file by file; the step's record goes last.

        Raises:
            FileNotFoundError: If the staging folder does not exist or holds no
                step record (a step that did not finish is never committed).
            FileExistsError: If the output is complete already, or exists
                without this being the rest of an interrupted commit.
        """
        source, target = self._inside(staging), self._inside(final)
        if not (source / COMMIT_RECORD_FILE).is_file():
            raise FileNotFoundError(
                f"staging folder {source} holds no {COMMIT_RECORD_FILE}: the step did not finish"
            )
        if self.is_committed(target):
            raise FileExistsError(f"{target} already exists; an output is never overwritten")
        self._move_files(source, target, COMMIT_RECORD_FILE)

    def set_aside(self, staging: Path, failed_dir: Path, name: str) -> Path:
        """Move what a failed attempt left into `failed_dir`, file by file.

        Returns:
            The new place, "{failed_dir}/{name}.{n}" with the first free n.
        """
        source = self._inside(staging)
        folder = self._inside(failed_dir)
        folder.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while (folder / f"{name}.{attempt}").exists():
            attempt += 1
        target = folder / f"{name}.{attempt}"
        target.mkdir()
        self._move_files(source, target, None)
        return target
