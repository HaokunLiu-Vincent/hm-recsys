"""The storage interface of the weekly pipeline's state (docs/mlops.md §6.4, §7.2).

The driver never touches the state's files directly: it goes through a
StateStore. In M12 there is one implementation, LocalStateStore (a folder on
this machine, artifacts/pipeline/). M13 adds one that keeps the SAME relative
keys in a Cloud Storage bucket; step processes keep working on local files and
the store moves them.

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

This module imports only the standard library.
"""

import shutil
from pathlib import Path
from typing import Protocol

TEMPORARY_SUFFIX = ".tmp"


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
