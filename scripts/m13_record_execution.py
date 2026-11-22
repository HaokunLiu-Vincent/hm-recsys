"""CLI for M13: append one execution of the Cloud Run job to results/m13_cloud/executions.csv.

    uv run python scripts/m13_record_execution.py --label V1 --kind eval \
        --execution <name> --started-by operator --image-commit <commit> \
        --started 2026-11-22T09:00:00Z --finished 2026-11-22T09:15:00Z \
        --max-memory-fraction 0.31 --state-bytes 2600000000 --outcome succeeded

The start and the end come from `bash infra/operate.sh describe <execution>`, the
memory from `bash infra/operate.sh memory ...`, the size from
`bash infra/operate.sh state-size`. The cost is computed from the run time and the
prices of configs/m13_cloud.yaml (hmrec.cloud_costs): an estimate, not the bill.
An execution is recorded once.
"""

import argparse
from datetime import datetime

from hmrec.cloud_costs import EXECUTION_COLUMNS, execution_row, load_cloud_config
from hmrec.csv_records import append_row, read_rows
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.paths import get_paths

LOGGER = get_logger("scripts.m13_record_execution")
KINDS = ("smoke", "calibration", "eval", "serve")


def parse_time(text: str) -> datetime:
    """An ISO time with a timezone ("Z" is accepted)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Record one execution of the Cloud Run job.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--started-by", required=True, choices=("operator", "scheduler"))
    parser.add_argument("--image-commit", required=True)
    parser.add_argument("--started", required=True, type=parse_time)
    parser.add_argument("--finished", required=True, type=parse_time)
    parser.add_argument("--max-memory-fraction", type=float, default=None)
    parser.add_argument("--state-bytes", type=int, default=None)
    parser.add_argument("--outcome", required=True, choices=("succeeded", "failed"))
    return parser.parse_args()


def main() -> None:
    """Append the row; refuse an execution that is recorded already."""
    args = parse_args()
    setup_logging()
    paths = get_paths()
    table = paths.cloud_table_file("executions")
    recorded = {row["execution"] for row in read_rows(table, EXECUTION_COLUMNS)}
    if args.execution in recorded:
        raise ValueError(f"execution {args.execution} is recorded already in {table}")
    row = execution_row(
        label=args.label,
        kind=args.kind,
        execution=args.execution,
        started_by=args.started_by,
        image_commit=args.image_commit,
        started=args.started,
        finished=args.finished,
        max_memory_fraction=args.max_memory_fraction,
        state_bytes=args.state_bytes,
        outcome=args.outcome,
        config=load_cloud_config(paths.cloud_config_file),
    )
    table.parent.mkdir(parents=True, exist_ok=True)
    append_row(table, EXECUTION_COLUMNS, row)
    LOGGER.info(
        "%s (%s): %.0f s, %.0f vCPU-s, %.0f GiB-s, %.4f USD; %s",
        args.label,
        args.execution,
        float(row["wall_seconds"]),
        float(row["vcpu_seconds"]),
        float(row["gib_seconds"]),
        float(row["cost_usd"]),
        table,
    )


if __name__ == "__main__":
    main()
