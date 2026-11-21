"""A small run manifest shared by the tests of hmrec.run_manifest and hmrec.tracking.

The tick: anchor 2020-08-05 (V2) in "eval" mode. It closes the week of
2020-07-29 (V1), whose labelled metrics it reports, and opens the week of its
anchor. Three steps: one ran on torch's OpenMP runtime, one on the system
libgomp, one was skipped (its output existed: a resumed tick).
"""

from datetime import UTC, date, datetime

from hmrec.platform_fields import PlatformFields
from hmrec.run_manifest import (
    EVAL_MODE,
    STEP_DONE,
    STEP_SKIPPED,
    SUCCEEDED,
    FileRecord,
    MetricRecord,
    RunCode,
    RunManifest,
    StepRecord,
)

ANCHOR = date(2020, 8, 5)
CLOSED_WEEK = date(2020, 7, 29)
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TREE = "a" * 64
DIGEST = "sha256:" + "b" * 64


def platform(openmp_library: str, openmp_version: str) -> PlatformFields:
    """Platform fields of a process in the arm64 image."""
    return PlatformFields(
        os="linux",
        architecture="aarch64",
        openmp_library=openmp_library,
        openmp_version=openmp_version,
        blas_libraries="not loaded",
        image_commit=COMMIT,
        code_tree_sha256=TREE,
    )


DRIVER_PLATFORM = platform("not loaded", "unknown")
TORCH_PLATFORM = platform(
    "/app/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1", "unknown"
)
LIGHTGBM_PLATFORM = platform("/usr/lib/aarch64-linux-gnu/libgomp.so.1", "libgomp1 14.2.0-19")


def example_manifest() -> RunManifest:
    """The example tick (see the module docstring)."""
    return RunManifest(
        run_id="2020-08-05-eval-0123456",
        anchor=ANCHOR,
        mode=EVAL_MODE,
        status=SUCCEEDED,
        failure="",
        code=RunCode(
            git_commit=COMMIT, git_dirty=False, code_tree_sha256=TREE, image_digest=DIGEST
        ),
        platform=DRIVER_PLATFORM,
        inputs=(FileRecord("transactions", "data/processed/transactions.parquet", 1000, "c" * 64),),
        outputs=(
            FileRecord("candidates/r4_itemcf", "weeks/2020-08-05/r4_itemcf.parquet", 50, "d" * 64),
            FileRecord("model/ranker", "weeks/2020-08-05/model.txt", 70, "e" * 64),
        ),
        metrics=(
            # Labelled: they describe the week this tick CLOSED.
            MetricRecord("map@12", CLOSED_WEEK, 0.030112271332103004),
            MetricRecord("union_recall", CLOSED_WEEK, 0.2543785197141435),
            # Same-week signal of the week this tick OPENED.
            MetricRecord("candidates_per_query", ANCHOR, 306.5),
        ),
        steps=(
            StepRecord("O1_recall_r4_itemcf", STEP_DONE, 31.5, 4.25, TORCH_PLATFORM, "1000 rows"),
            StepRecord("O3_train_ranker", STEP_DONE, 278.0, 6.5, LIGHTGBM_PLATFORM, "500 trees"),
            StepRecord("O2_features", STEP_SKIPPED, 0.0, 0.0, None, "output existed"),
        ),
        started_at_utc=datetime(2026, 11, 23, 1, 0, 0, tzinfo=UTC),
        finished_at_utc=datetime(2026, 11, 23, 1, 13, 20, tzinfo=UTC),
        total_seconds=800.0,
        driver_max_rss_gb=0.5,
    )
