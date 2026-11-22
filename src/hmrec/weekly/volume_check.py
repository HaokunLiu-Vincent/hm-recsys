"""Check that a folder behaves as the weekly pipeline needs (docs/mlops.md §8.4, M13).

On Cloud Run the state of the pipeline is a folder of a bucket, mounted as a
volume (Cloud Storage FUSE). Such a mount is not a POSIX file system: an object
is written when its file is closed, a rename is a copy and a delete, and a
folder is not an object at all. Before a tick is trusted to it, this module
tries every kind of file operation the pipeline uses, in a folder of its own
inside the state, and reports each one:

    sequential write and read     a large file in blocks, with its sha256 and the
                                  throughput (candidates, features, the Dataset)
    parquet                       polars writes a table and reads back a column
                                  of the rows that pass a filter (a ranged read)
    npy                           numpy saves and loads an array (vectors)
    write at an offset            a byte in the middle of an existing file is
                                  replaced (writers that go back to patch a header)
    append                        rows are appended to an existing text file
                                  (the result tables)
    rename and replace a file     the two steps of write_text_once / replace_text
    list and exists               right after a write and right after a delete
    the store's commit            MountedStateStore: staging to final, file by
                                  file, the record last; set aside; discard
    rename a folder               REPORTED ONLY: the pipeline does not need it
    inputs                        every pipeline input of configs/data_manifest.json
                                  exists under the inputs root with its size;
                                  with `hash_inputs` also with its sha256

Nothing of the pipeline is computed and no label is read. The check removes its
own folder at the end. This module imports neither torch nor lightgbm.
"""

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from hmrec.data_manifest import FileEntry, hash_file, select_entries, verify_entries
from hmrec.weekly.storage import COMMIT_RECORD_FILE, MountedStateStore

CHECK_DIR = ".volume-check"
BLOCK_BYTES = 8 * 1024 * 1024
BYTES_PER_MEGABYTE = 1024 * 1024
# The prefixes of configs/data_manifest.json a tick reads (data/raw/ is not an input).
PIPELINE_INPUT_PREFIXES = ("data/processed", "data/bootstrap", "artifacts/features")

OK = "ok"
FAILED = "FAILED"
REPORTED = "reported"


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check.

    Attributes:
        name: What was tried.
        status: "ok", "FAILED", or "reported" for a check that never fails the run.
        seconds: How long it took.
        detail: What was measured, or the error.
    """

    name: str
    status: str
    seconds: float
    detail: str

    def line(self) -> str:
        """One line for the log."""
        return f"volume check | {self.name} | {self.status} | {self.seconds:.2f} s | {self.detail}"


def expect(condition: bool, message: str) -> None:
    """Raise if a check's expectation does not hold.

    Raises:
        AssertionError: With the message. (`raise`, not `assert`: python -O
            removes assert statements.)
    """
    if not condition:
        raise AssertionError(message)


def run_check(name: str, check: Callable[[], str], required: bool = True) -> CheckResult:
    """Run one check and turn an error into a result (the other checks still run).

    Args:
        name: The check's name.
        check: Does the work; returns the detail text; raises on a failure.
        required: False for a check that is reported and never fails the run.
    """
    started = time.perf_counter()
    try:
        detail = check()
        status = OK if required else REPORTED
    except (OSError, AssertionError, ValueError) as error:
        detail = f"{type(error).__name__}: {error}"
        status = FAILED if required else REPORTED
    return CheckResult(name, status, time.perf_counter() - started, detail)


def megabytes_per_second(n_bytes: int, seconds: float) -> float:
    """Throughput in MB/s (0.0 if no time passed)."""
    return 0.0 if seconds <= 0 else n_bytes / BYTES_PER_MEGABYTE / seconds


def check_sequential(folder: Path, megabytes: int, seed: int) -> str:
    """Write a file of `megabytes` MB in blocks, read it back, compare the sha256."""
    target = folder / "sequential.bin"
    generator = np.random.default_rng(seed)
    block = generator.integers(0, 256, size=BLOCK_BYTES, dtype=np.uint8).tobytes()
    n_blocks = max(1, megabytes * BYTES_PER_MEGABYTE // BLOCK_BYTES)
    written = hashlib.sha256()
    started = time.perf_counter()
    with target.open("wb") as handle:
        for _ in range(n_blocks):
            handle.write(block)
            written.update(block)
    write_seconds = time.perf_counter() - started
    started = time.perf_counter()
    size, sha256, _ = hash_file(target)
    read_seconds = time.perf_counter() - started
    expect(size == n_blocks * BLOCK_BYTES, f"size {size}, expected {n_blocks * BLOCK_BYTES}")
    expect(sha256 == written.hexdigest(), "the sha256 read back is not the one written")
    return (
        f"{size / BYTES_PER_MEGABYTE:.0f} MB; write {megabytes_per_second(size, write_seconds):.0f}"
        f" MB/s; read {megabytes_per_second(size, read_seconds):.0f} MB/s"
    )


def check_parquet(folder: Path, n_rows: int) -> str:
    """Polars writes a table; one column of the rows that pass a filter is read back."""
    target = folder / "table.parquet"
    table = pl.DataFrame(
        {
            "query_id": pl.arange(0, n_rows, eager=True).cast(pl.Int32),
            "score": pl.arange(0, n_rows, eager=True).cast(pl.Float32) / 7.0,
        }
    )
    table.write_parquet(target)
    kept = (
        pl.scan_parquet(target).filter(pl.col("query_id") >= n_rows // 2).select("score").collect()
    )
    expected = table.filter(pl.col("query_id") >= n_rows // 2).select("score")
    expect(kept.equals(expected), "the rows read back differ from the rows written")
    return f"{n_rows} rows written, {kept.height} read back through a filter"


def check_npy(folder: Path, n_rows: int, seed: int) -> str:
    """Numpy saves a float32 array and loads it again."""
    target = folder / "vectors.npy"
    vectors = np.random.default_rng(seed).random((n_rows, 64), dtype=np.float32)
    np.save(target, vectors)
    loaded = np.load(target)
    expect(loaded.dtype == np.float32 and np.array_equal(loaded, vectors), "the array differs")
    return f"array of shape {vectors.shape} float32"


def check_write_at_offset(folder: Path) -> str:
    """Replace bytes in the middle of an existing file (not a sequential write)."""
    target = folder / "patched.bin"
    target.write_bytes(b"a" * 4096)
    with target.open("r+b") as handle:
        handle.seek(1024)
        handle.write(b"patch")
    content = target.read_bytes()
    expect(len(content) == 4096, f"size {len(content)} after the patch, expected 4096")
    expect(content[1024:1029] == b"patch", "the patched bytes are not there")
    expect(content[:1024] == b"a" * 1024, "bytes before the patch changed")
    return "5 bytes replaced at offset 1024 of a 4096-byte file"


def check_append(folder: Path) -> str:
    """Append rows to an existing text file, as the result tables are written."""
    target = folder / "table.csv"
    target.write_text("name,value\n", encoding="utf-8")
    for row in range(3):
        with target.open("a", encoding="utf-8") as handle:
            handle.write(f"row{row},{row}\n")
    lines = target.read_text(encoding="utf-8").splitlines()
    expect(lines == ["name,value", "row0,0", "row1,1", "row2,2"], f"content {lines}")
    return "3 rows appended in 3 opens"


def check_rename_and_replace(folder: Path) -> str:
    """Rename a file to a new name, then put a file in place of an existing one."""
    first, second = folder / "first.json", folder / "second.json"
    first.write_text("one", encoding="utf-8")
    first.rename(second)
    expect(not first.exists() and second.read_text(encoding="utf-8") == "one", "rename failed")
    newer = folder / "newer.json"
    newer.write_text("two", encoding="utf-8")
    newer.replace(second)
    expect(not newer.exists() and second.read_text(encoding="utf-8") == "two", "replace failed")
    return "a rename to a new name and a replace of an existing file"


def check_list_and_exists(folder: Path) -> str:
    """A file is listed and found right after it is written, and gone right after its delete."""
    nested = folder / "listing" / "deep"
    nested.mkdir(parents=True)
    names = [f"part-{number:05d}.parquet" for number in range(5)]
    for name in names:
        (nested / name).write_bytes(b"x")
    listed = sorted(path.name for path in nested.iterdir())
    expect(listed == names, f"listed {listed}, expected {names}")
    below = sorted(path.name for path in (folder / "listing").rglob("*") if path.is_file())
    expect(below == names, f"recursive listing gave {below}")
    (nested / names[0]).unlink()
    expect(not (nested / names[0]).exists(), "a deleted file still exists")
    expect(len(list(nested.iterdir())) == 4, "a deleted file is still listed")
    expect(not (folder / "listing" / "never-written").exists(), "a missing file exists")
    return "5 files in a nested folder: listed, found, one deleted"


def check_store_commit(folder: Path) -> str:
    """MountedStateStore: text files, a commit file by file, set aside, discard."""
    store = MountedStateStore(folder)
    marker = folder / "store" / "marker.json"
    store.write_text_once(marker, "first")
    store.replace_text(marker, "second")
    expect(store.read_text(marker) == "second", "replace_text did not replace")
    staging, final = folder / "store" / ".staging" / "step", folder / "store" / "step"
    (staging / "shards" / "000").mkdir(parents=True)
    (staging / "shards" / "000" / "candidates.parquet").write_bytes(b"c" * 1000)
    (staging / "output.parquet").write_bytes(b"o" * 1000)
    (staging / COMMIT_RECORD_FILE).write_text("{}", encoding="utf-8")
    expect(not store.is_committed(final), "an output that was never committed is complete")
    store.commit_dir(staging, final)
    expect(store.is_committed(final), "the committed output is not complete")
    expect(not staging.exists(), "the staging folder is still there after the commit")
    moved = sorted(
        path.relative_to(final).as_posix() for path in final.rglob("*") if path.is_file()
    )
    expected = sorted(["output.parquet", "shards/000/candidates.parquet", COMMIT_RECORD_FILE])
    expect(moved == expected, f"the output holds {moved}, expected {expected}")
    failed = folder / "store" / ".staging" / "failed-step"
    failed.mkdir(parents=True)
    (failed / "half.parquet").write_bytes(b"h")
    aside = store.set_aside(failed, folder / "store" / ".failed", "failed-step")
    expect((aside / "half.parquet").is_file() and not failed.exists(), "set_aside did not move")
    store.discard_dir(final)
    expect(not final.exists(), "a discarded folder still exists")
    return "write once, replace, commit (3 files, record last), set aside, discard"


def check_folder_rename(folder: Path) -> str:
    """Rename a folder that holds a file. Reported only: the pipeline does not rely on it."""
    source, target = folder / "folder-a", folder / "folder-b"
    source.mkdir()
    (source / "file.txt").write_text("x", encoding="utf-8")
    source.rename(target)
    expect((target / "file.txt").is_file() and not source.exists(), "the folder did not move")
    return "a folder with one file was renamed"


def check_inputs(inputs_root: Path, entries: Sequence[FileEntry], hash_inputs: bool) -> str:
    """Every pipeline input exists under the inputs root with its recorded size (and sha256).

    Args:
        inputs_root: The folder that holds data/ and artifacts/.
        entries: The entries of configs/data_manifest.json.
        hash_inputs: Also read every file and compare its sha256.

    Raises:
        AssertionError: If a file is missing or differs; the message lists them.
    """
    wanted = select_entries(entries, PIPELINE_INPUT_PREFIXES)
    total = sum(entry.size_bytes for entry in wanted)
    if hash_inputs:
        started = time.perf_counter()
        problems = verify_entries(inputs_root, wanted)
        seconds = time.perf_counter() - started
        expect(problems == [], "; ".join(problems))
        return (
            f"{len(wanted)} files, {total / BYTES_PER_MEGABYTE:.0f} MB: size and sha256 equal the "
            f"data manifest; read {megabytes_per_second(total, seconds):.0f} MB/s"
        )
    problems = []
    for entry in wanted:
        path = inputs_root / entry.path
        if not path.is_file():
            problems.append(f"{entry.path}: missing")
        elif path.stat().st_size != entry.size_bytes:
            problems.append(f"{entry.path}: size {path.stat().st_size}, not {entry.size_bytes}")
    expect(problems == [], "; ".join(problems))
    return (
        f"{len(wanted)} files, {total / BYTES_PER_MEGABYTE:.0f} MB: every size equals the manifest"
    )


def check_volume(
    state_root: Path,
    inputs_root: Path,
    entries: Sequence[FileEntry],
    run_name: str,
    megabytes: int,
    hash_inputs: bool,
    seed: int,
) -> list[CheckResult]:
    """Run every check in {state_root}/.volume-check/{run_name}/ and remove the folder.

    Args:
        state_root: The folder the pipeline's state would live in.
        inputs_root: The folder that holds the pipeline's inputs (data/, artifacts/).
        entries: The entries of configs/data_manifest.json.
        run_name: Names this run's folder; it must not exist yet.
        megabytes: Size of the file of the sequential check.
        hash_inputs: Also compare the sha256 of every input (reads all of them).
        seed: Seed of the generated bytes and arrays.

    Returns:
        One result per check, in the order they ran.

    Raises:
        FileExistsError: If the run's folder exists.
    """
    folder = state_root / CHECK_DIR / run_name
    if folder.exists():
        raise FileExistsError(f"{folder} exists; a volume check never reuses a folder")
    folder.mkdir(parents=True)
    results = [
        run_check("sequential write and read", lambda: check_sequential(folder, megabytes, seed)),
        run_check("parquet", lambda: check_parquet(folder, 200_000)),
        run_check("npy", lambda: check_npy(folder, 10_000, seed)),
        run_check("write at an offset", lambda: check_write_at_offset(folder)),
        run_check("append", lambda: check_append(folder)),
        run_check("rename and replace a file", lambda: check_rename_and_replace(folder)),
        run_check("list and exists", lambda: check_list_and_exists(folder)),
        run_check("the store's commit", lambda: check_store_commit(folder)),
        run_check("rename a folder", lambda: check_folder_rename(folder), required=False),
        run_check("inputs", lambda: check_inputs(inputs_root, entries, hash_inputs)),
    ]
    results.append(
        run_check("remove the check's folder", lambda: remove_folder(state_root, folder))
    )
    return results


def remove_folder(state_root: Path, folder: Path) -> str:
    """Remove the check's own folder through the store (it refuses anything outside the state)."""
    MountedStateStore(state_root).discard_dir(folder)
    expect(not folder.exists(), f"{folder} still exists")
    return "removed"
