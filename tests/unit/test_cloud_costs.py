"""Tests for hmrec.cloud_costs: the cost of an execution from its run time.

The expected values are computed by hand from the prices of configs/m13_cloud.yaml:
one second at 6 vCPU and 24 GiB costs 6 x 0.000018 + 24 x 0.000002 = 0.000156 USD.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from hmrec.cloud_costs import (
    EXECUTION_COLUMNS,
    CloudConfig,
    execution_cost,
    execution_row,
    load_cloud_config,
    parse_cloud_config,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = CloudConfig(6, 24, 0.000018, 0.000002)
START = datetime(2026, 11, 22, 9, 0, 0, tzinfo=UTC)


def test_the_config_file_holds_the_shape_of_decision_d10_and_the_tier_1_prices() -> None:
    assert load_cloud_config(REPO / "configs" / "m13_cloud.yaml") == CONFIG


def test_cost_of_a_quarter_of_an_hour() -> None:
    # 900 s: 5,400 vCPU-s and 21,600 GiB-s; 5,400 x 0.000018 + 21,600 x 0.000002 = 0.1404.
    cost = execution_cost(START, START + timedelta(seconds=900), CONFIG)
    assert (cost.wall_seconds, cost.vcpu_seconds, cost.gib_seconds) == (900.0, 5400.0, 21600.0)
    assert cost.cost_usd == pytest.approx(0.1404)
    # Nothing ran, nothing is billed.
    assert execution_cost(START, START, CONFIG).cost_usd == 0.0


def test_times_need_a_timezone_and_an_order() -> None:
    with pytest.raises(ValueError, match="need a timezone"):
        execution_cost(datetime(2026, 11, 22, 9, 0, 0), START, CONFIG)
    with pytest.raises(ValueError, match="before it started"):
        execution_cost(START, START - timedelta(seconds=1), CONFIG)


def test_a_row_has_the_tables_columns_and_the_memory_in_gib() -> None:
    row = execution_row(
        label="V1",
        kind="eval",
        execution="job-abcde",
        started_by="scheduler",
        image_commit="a" * 40,
        started=START,
        finished=START + timedelta(seconds=1000),
        max_memory_fraction=0.25,
        state_bytes=2_600_000_000,
        outcome="succeeded",
        config=CONFIG,
    )
    assert tuple(row) == EXECUTION_COLUMNS
    assert row["started_at_utc"] == "2026-11-22T09:00:00+00:00"
    assert float(row["wall_seconds"]) == 1000.0
    assert float(row["cost_usd"]) == pytest.approx(0.156)
    # A quarter of 24 GiB.
    assert (row["max_memory_fraction"], row["max_memory_gib"]) == ("0.25", "6.0")
    # What was not measured stays empty; it is not written as zero.
    unmeasured = execution_row(
        "smoke test",
        "smoke",
        "job-fghij",
        "operator",
        "a" * 40,
        START,
        START,
        None,
        None,
        "succeeded",
        CONFIG,
    )
    assert unmeasured["max_memory_gib"] == "" and unmeasured["state_bytes"] == ""


def test_a_config_with_an_unknown_key_or_a_zero_is_refused() -> None:
    good = {
        "job": {"cpu": 6, "memory_gib": 24},
        "prices": {"usd_per_vcpu_second": 0.000018, "usd_per_gib_second": 0.000002},
    }
    assert parse_cloud_config(good) == CONFIG
    with pytest.raises(ValueError):
        parse_cloud_config({**good, "region": "somewhere"})
    with pytest.raises(ValueError, match="must be > 0"):
        parse_cloud_config({**good, "job": {"cpu": 0, "memory_gib": 24}})
