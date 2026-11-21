"""CLI: write or verify the data manifest, configs/data_manifest.json (docs/mlops.md §7.1).

Usage (the Makefile targets run these under the dependency group "versioning"):
    python scripts/m11_data_manifest.py write
        Ask DVC which files it tracks, hash each one on disk (sha256 and md5),
        check the md5 against what DVC recorded, and write the manifest.
    python scripts/m11_data_manifest.py verify
        Compare every tracked file on disk with the manifest.
    python scripts/m11_data_manifest.py verify --root /tmp/clone --only data/processed
        Compare only some files, in another checkout (after a partial `dvc pull`).
        The manifest is read from that checkout too.

`verify` ends with exit code 1 if a file is missing or differs.
"""

import argparse
import logging
import sys
from pathlib import Path

from hmrec.data_manifest import (
    build_entries,
    parse_dvc_listing,
    read_data_manifest,
    run_dvc_listing,
    select_entries,
    verify_entries,
    write_data_manifest,
)
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.paths import get_paths

LOGGER = get_logger("scripts.m11_data_manifest")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Write or verify the data manifest.")
    parser.add_argument("action", choices=("write", "verify"))
    parser.add_argument("--root", type=Path, default=None, help="verify: another checkout")
    parser.add_argument("--only", nargs="+", default=None, help="verify: only these paths")
    return parser.parse_args()


def main() -> None:
    """Write the manifest from the files on disk, or compare files with it."""
    args = parse_args()
    setup_logging(logging.INFO)
    paths = get_paths()
    if args.action == "write":
        tracked = parse_dvc_listing(run_dvc_listing(paths.project_root))
        LOGGER.info("DVC tracks %d file(s); hashing them", len(tracked))
        entries = build_entries(paths.project_root, tracked)
        write_data_manifest(paths.data_manifest_file, entries)
        total_bytes = sum(entry.size_bytes for entry in entries)
        LOGGER.info(
            "wrote %s: %d file(s), %d bytes", paths.data_manifest_file, len(entries), total_bytes
        )
        return
    root = paths.project_root if args.root is None else args.root.resolve()
    manifest_file = root / paths.data_manifest_file.relative_to(paths.project_root)
    entries = read_data_manifest(manifest_file)
    if args.only is not None:
        entries = select_entries(entries, args.only)
    problems = verify_entries(root, entries)
    for problem in problems:
        LOGGER.error("%s", problem)
    total_bytes = sum(entry.size_bytes for entry in entries)
    LOGGER.info(
        "%s: %d file(s), %d bytes compared with %s: %d problem(s)",
        root,
        len(entries),
        total_bytes,
        manifest_file,
        len(problems),
    )
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
