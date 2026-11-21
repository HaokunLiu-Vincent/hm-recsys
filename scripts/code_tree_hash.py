"""CLI: print the hash of the code tree (hmrec.code_tree), one line.

Usage:
    python scripts/code_tree_hash.py            # the tree this script belongs to
    python scripts/code_tree_hash.py --root /app

The Dockerfile runs it at build time and stores the line in the image; at run
time hmrec.git_state computes the hash again and refuses a run whose code is
not the code the image was built with (docs/mlops.md §8.1).
"""

import argparse
import sys
from pathlib import Path

from hmrec.code_tree import code_tree_sha256

# scripts/code_tree_hash.py -> parents[1] = the repository root (/app in an image).
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Print the hash of the code tree.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="root of the code tree")
    return parser.parse_args()


def main() -> None:
    """Write the hash to stdout."""
    sys.stdout.write(code_tree_sha256(parse_args().root) + "\n")


if __name__ == "__main__":
    main()
