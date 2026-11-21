"""Tests for hmrec.run_manifest: the schema, the writer, the reader and their checks."""

import json
import re
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from run_manifest_example import (
    ANCHOR,
    CLOSED_WEEK,
    COMMIT,
    DIGEST,
    LIGHTGBM_PLATFORM,
    example_manifest,
)

from hmrec.git_state import NOT_AN_IMAGE
from hmrec.run_manifest import (
    IMAGE_DIGEST_VARIABLE,
    MANIFEST_KEYS,
    STEP_DONE,
    STEP_SKIPPED,
    FileRecord,
    MetricRecord,
    RunManifest,
    StepRecord,
    file_record,
    find_run_manifests,
    image_digest,
    manifest_from_json,
    manifest_to_json,
    read_run_code,
    read_run_manifest,
    step_record_of_this_process,
    validate_run_manifest,
    write_run_manifest,
)

REPO = Path(__file__).resolve().parents[2]


def test_the_manifest_holds_every_field_of_a_tick() -> None:
    content = manifest_to_json(example_manifest())
    assert tuple(content) == MANIFEST_KEYS
    assert content["schema_version"] == 1
    assert (content["run_id"], content["anchor"], content["mode"]) == (
        "2020-08-05-eval-0123456",
        "2020-08-05",
        "eval",
    )
    assert (content["git_commit"], content["git_dirty"]) == (COMMIT, False)
    assert content["image_digest"] == DIGEST
    assert content["platform"]["architecture"] == "aarch64"
    assert content["inputs"][0] == {
        "role": "transactions",
        "path": "data/processed/transactions.parquet",
        "size_bytes": 1000,
        "sha256": "c" * 64,
    }
    # A labelled metric names the week it describes: the week before the anchor.
    assert content["metrics"][0] == {
        "name": "map@12",
        "week": "2020-07-29",
        "value": 0.030112271332103004,
    }
    # Every step has its own platform fields: the OpenMP runtime differs between processes.
    assert content["steps"][1]["platform"]["openmp_version"] == "libgomp1 14.2.0-19"
    assert content["steps"][2]["platform"] is None
    # The largest true peak of the driver (0.5) and the steps (4.25, 6.5, 0.0).
    assert content["max_process_rss_gb"] == 6.5
    assert content["started_at_utc"] == "2026-11-23T01:00:00+00:00"


def test_write_then_read_gives_the_same_manifest(tmp_path: Path) -> None:
    manifest = example_manifest()
    path = tmp_path / "runs" / manifest.run_id / "manifest.json"
    write_run_manifest(path, manifest)
    assert read_run_manifest(path) == manifest
    # Floats come back as the same float64 (JSON holds their shortest exact text).
    assert read_run_manifest(path).metrics[1].value == 0.2543785197141435
    # No temporary file is left next to the manifest.
    assert [file.name for file in path.parent.iterdir()] == ["manifest.json"]


def test_a_manifest_is_never_replaced(tmp_path: Path) -> None:
    manifest = example_manifest()
    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="never replaced"):
        write_run_manifest(path, replace(manifest, total_seconds=1.0))
    assert path.read_bytes() == before


def test_a_failed_tick_says_why(tmp_path: Path) -> None:
    failed = replace(example_manifest(), status="failed", failure="O2: 3 duplicate keys")
    path = tmp_path / "manifest.json"
    write_run_manifest(path, failed)
    assert read_run_manifest(path).failure == "O2: 3 duplicate keys"


def _with_step(manifest: RunManifest, **changes: object) -> RunManifest:
    step = replace(manifest.steps[0], **changes)
    return replace(manifest, steps=(step, *manifest.steps[1:]))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"run_id": "has space"}, "bad run_id"),
        ({"run_id": ""}, "bad run_id"),
        ({"mode": "train"}, "mode must be one of"),
        ({"status": "done"}, "status must be one of"),
        ({"failure": "something"}, "failure is empty for a succeeded tick"),
        ({"status": "failed"}, "failure is empty for a succeeded tick"),
        ({"total_seconds": -1.0}, "bad total_seconds"),
        ({"driver_max_rss_gb": float("nan")}, "bad driver_max_rss_gb"),
    ],
)
def test_validation_of_the_top_level(change: dict[str, object], message: str) -> None:
    manifest = replace(example_manifest(), **change)
    with pytest.raises(ValueError, match=message):
        validate_run_manifest(manifest)


def test_validation_of_code_files_metrics_and_steps() -> None:
    manifest = example_manifest()
    cases: list[tuple[RunManifest, str]] = [
        (replace(manifest, code=replace(manifest.code, git_commit="abc123")), "full commit hash"),
        (replace(manifest, code=replace(manifest.code, code_tree_sha256="tree")), "code_tree"),
        (replace(manifest, code=replace(manifest.code, image_digest="b" * 64)), "image_digest"),
        (replace(manifest, outputs=(manifest.outputs[0], manifest.outputs[0])), "role repeats"),
        (replace(manifest, inputs=(FileRecord("a", "p", 1, "short"),)), "bad sha256"),
        (replace(manifest, inputs=(FileRecord("a", "p", -1, "c" * 64),)), "size_bytes"),
        (replace(manifest, metrics=(manifest.metrics[0], manifest.metrics[0])), "pair repeats"),
        (replace(manifest, metrics=(MetricRecord("m", ANCHOR, float("inf")),)), "finite float"),
        (replace(manifest, steps=(manifest.steps[0], manifest.steps[0])), "step name repeats"),
        (_with_step(manifest, status="running"), "status 'running'"),
        (_with_step(manifest, platform=None), "skipped step has no platform"),
        (_with_step(manifest, status=STEP_SKIPPED), "skipped step has no platform"),
        (_with_step(manifest, seconds=-0.5), "bad seconds"),
        (
            replace(manifest, finished_at_utc=manifest.started_at_utc.replace(year=2025)),
            "ends before it starts",
        ),
        (
            replace(manifest, started_at_utc=manifest.started_at_utc.replace(tzinfo=None)),
            "time zone",
        ),
    ]
    for broken, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_run_manifest(broken)
    # The same metric name for the closed and for the opened week is allowed.
    two_weeks = (MetricRecord("rows", CLOSED_WEEK, 1.0), MetricRecord("rows", ANCHOR, 2.0))
    validate_run_manifest(replace(manifest, metrics=two_weeks))
    # On the laptop there is no image: the code-tree field says so and the digest is unknown.
    laptop = replace(manifest.code, code_tree_sha256=NOT_AN_IMAGE, image_digest=None)
    validate_run_manifest(replace(manifest, code=laptop))


def test_writer_refuses_a_malformed_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="mode must be one of"):
        write_run_manifest(path, replace(example_manifest(), mode="train"))
    assert not path.exists()


def test_reader_rejects_unknown_missing_and_inconsistent_content() -> None:
    good = manifest_to_json(example_manifest())
    assert manifest_from_json(json.loads(json.dumps(good))) == example_manifest()
    with pytest.raises(ValueError, match="unknown \\['extra'\\]"):
        manifest_from_json({**good, "extra": 1})
    without_metrics = {key: value for key, value in good.items() if key != "metrics"}
    with pytest.raises(ValueError, match="missing keys \\['metrics'\\]"):
        manifest_from_json(without_metrics)
    with pytest.raises(ValueError, match="schema_version 2"):
        manifest_from_json({**good, "schema_version": 2})
    with pytest.raises(ValueError, match="platform must not be null"):
        manifest_from_json({**good, "platform": None})
    # The recorded maximum must be the maximum of the recorded processes.
    with pytest.raises(ValueError, match="largest peak"):
        manifest_from_json({**good, "max_process_rss_gb": 7.0})
    bad_step = {**good["steps"][0], "threads": 6}
    with pytest.raises(ValueError, match="steps\\[0\\]"):
        manifest_from_json({**good, "steps": [bad_step]})
    with pytest.raises(ValueError, match="finite number"):
        manifest_from_json({**good, "metrics": [{"name": "m", "week": "2020-08-05", "value": "x"}]})
    with pytest.raises(ValueError, match="expected an object"):
        manifest_from_json([good])


def test_read_run_manifest_needs_the_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_run_manifest(tmp_path / "manifest.json")


def test_image_digest_is_read_from_the_environment() -> None:
    assert image_digest({}) is None
    assert image_digest({IMAGE_DIGEST_VARIABLE: ""}) is None
    assert image_digest({IMAGE_DIGEST_VARIABLE: DIGEST}) == DIGEST
    # A tag or a bare hash is not a digest.
    for value in ("hmrec:dev", "b" * 64, "sha256:xyz"):
        with pytest.raises(ValueError, match=IMAGE_DIGEST_VARIABLE):
            image_digest({IMAGE_DIGEST_VARIABLE: value})


def test_file_record_reads_size_and_sha256(tmp_path: Path) -> None:
    (tmp_path / "weeks").mkdir()
    (tmp_path / "weeks" / "out.bin").write_bytes(b"abc")
    record = file_record("output", tmp_path, "weeks/out.bin")
    # sha256 of b"abc" (the well-known test vector).
    assert record == FileRecord(
        "output",
        "weeks/out.bin",
        3,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    )


def test_records_of_this_process() -> None:
    # The code of this very process: a commit on the laptop (git), the image's commit in
    # an image. Both are full hashes; the code-tree field is a hash or "not an image".
    code = read_run_code(REPO, REPO / "results")
    assert re.fullmatch(r"[0-9a-f]{40}", code.git_commit)
    assert code.code_tree_sha256 == NOT_AN_IMAGE or re.fullmatch(
        r"[0-9a-f]{64}", code.code_tree_sha256
    )
    step = step_record_of_this_process("C1_ingest", 1.5, "3 rows", REPO)
    assert (step.name, step.status, step.seconds, step.detail) == (
        "C1_ingest",
        STEP_DONE,
        1.5,
        "3 rows",
    )
    # ru_maxrss of a running Python process is never zero.
    assert step.process_max_rss_gb > 0
    assert step.platform is not None and step.platform.code_tree_sha256 == code.code_tree_sha256
    manifest = replace(example_manifest(), code=replace(code, git_dirty=False), steps=(step,))
    validate_run_manifest(manifest)


def test_find_run_manifests(tmp_path: Path) -> None:
    assert find_run_manifests(tmp_path / "runs") == []
    for run_id in ("b-run", "a-run"):
        write_run_manifest(
            tmp_path / "runs" / run_id / "manifest.json",
            replace(example_manifest(), run_id=run_id),
        )
    (tmp_path / "runs" / "c-run").mkdir()  # a tick that has not written its manifest yet
    found = find_run_manifests(tmp_path / "runs")
    assert [file.parent.name for file in found] == ["a-run", "b-run"]


def test_step_of_another_platform_keeps_its_fields(tmp_path: Path) -> None:
    step = StepRecord("O4_score", STEP_DONE, 12.0, 3.0, LIGHTGBM_PLATFORM, "")
    manifest = replace(example_manifest(), steps=(step,), anchor=date(2020, 9, 23))
    path = tmp_path / "manifest.json"
    write_run_manifest(path, manifest)
    assert read_run_manifest(path).steps[0].platform == LIGHTGBM_PLATFORM
