"""CLI for M13: what this process runs on, and whether the state folder behaves as needed.

    python scripts/m13_volume_check.py --megabytes 512 --hash-inputs

The smoke test of a new place to run the weekly pipeline (docs/mlops.md §8.4),
for example a Cloud Run job whose state is a folder of a mounted bucket:

    1. the image and platform report of both sides (scripts/image_info.py, each
       side in its own process: torch and LightGBM never share one);
    2. the roots this process was given (HMREC_PIPELINE_INPUTS_ROOT,
       HMREC_PIPELINE_STATE_ROOT) and the image digest (HMREC_IMAGE_DIGEST);
    3. every kind of file operation the pipeline uses, tried in a folder of its
       own inside the state (hmrec.weekly.volume_check), and the pipeline's
       inputs against configs/data_manifest.json.

It computes nothing of the system and reads no label. The check's folder is
removed at the end.

Exit code 0: every required check passed. 1: at least one failed.
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from hmrec.config import load_base_config
from hmrec.data_manifest import read_data_manifest
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.paths import (
    PIPELINE_INPUTS_ROOT_VARIABLE,
    PIPELINE_STATE_ROOT_VARIABLE,
    get_paths,
    tick_root_paths,
)
from hmrec.platform_report import SIDES
from hmrec.run_manifest import IMAGE_DIGEST_VARIABLE, image_digest
from hmrec.weekly.volume_check import FAILED, check_volume

LOGGER = get_logger("scripts.m13_volume_check")
IMAGE_INFO_SCRIPT = "image_info.py"
NOT_SET = "not set"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Platform report and volume check.")
    parser.add_argument(
        "--megabytes", type=int, default=512, help="size of the sequential test file"
    )
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="also read every pipeline input and compare its sha256 with the data manifest",
    )
    return parser.parse_args()


def report_platform(scripts_dir: Path) -> None:
    """Log the image and platform report of both sides, each from its own process."""
    for side in SIDES:
        command = [sys.executable, str(scripts_dir / IMAGE_INFO_SCRIPT), "--side", side]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        for line in completed.stdout.splitlines():
            LOGGER.info("platform (%s side) | %s", side, line)


def main() -> None:
    """Report the platform, run the checks, and exit with 1 if one failed."""
    args = parse_args()
    setup_logging(logging.INFO)
    report_platform(Path(__file__).resolve().parent)
    paths = tick_root_paths(get_paths(), os.environ, calibration=False)
    for variable in (PIPELINE_INPUTS_ROOT_VARIABLE, PIPELINE_STATE_ROOT_VARIABLE):
        LOGGER.info("env %s: %s", variable, os.environ.get(variable, NOT_SET))
    LOGGER.info("env %s: %s", IMAGE_DIGEST_VARIABLE, image_digest(os.environ) or NOT_SET)
    # The inputs root holds data/ and artifacts/: it is the parent of the data root.
    inputs_root = paths.data_root.parent
    state_root = paths.pipeline_state_dir
    LOGGER.info("inputs root %s | state root %s", inputs_root, state_root)
    results = check_volume(
        state_root=state_root,
        inputs_root=inputs_root,
        entries=read_data_manifest(paths.data_manifest_file),
        run_name=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        megabytes=args.megabytes,
        hash_inputs=args.hash_inputs,
        seed=load_base_config(paths.base_config_file).seed,
    )
    for result in results:
        LOGGER.info(result.line())
    failed = [result for result in results if result.status == FAILED]
    LOGGER.info("volume check: %d check(s), %d failed", len(results), len(failed))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
