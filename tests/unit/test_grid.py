"""Tests for the D-phase grid: reading, candidate priority order, requires."""

from pathlib import Path

import pytest

from hmrec.config import load_yaml
from hmrec.hpo.grid import (
    apply_overrides,
    is_on_boundary,
    load_phase_grid,
    ordered_candidate_configs,
    parse_step,
    step_applies,
)

REPO = Path(__file__).resolve().parents[2]
GRID = load_phase_grid(REPO / "configs" / "hpo" / "d_phase.yaml")


def _default(channel: str) -> dict:
    return load_yaml(REPO / "configs" / "recall" / f"{channel}.yaml")


def _step(channel: str, name: str):
    return next(step for step in GRID.channels[channel] if step.name == name)


def test_configuration_counts_match_proposal() -> None:
    # proposal §5: R1 8, R2 5, R3 7, R4 9, R5 9 configurations (default included).
    # Counted with every step starting from the repo default (no winner changes).
    expected = {
        "r1_popularity": 8,
        "r2_repurchase": 5,
        "r3_sibling": 7,
        "r4_itemcf": 9,
        "r5_wals": 9,
    }
    for channel, count in expected.items():
        default = _default(channel)
        n_candidates = sum(
            len(ordered_candidate_configs(s, default)) for s in GRID.channels[channel]
        )
        assert 1 + n_candidates == count, channel


def test_nearest_to_current_order_ties_smaller_first() -> None:
    # R1 tau from the default 3: distance 1 -> 2, 4; distance 2 -> 1, 5; then 6, 7.
    configs = ordered_candidate_configs(
        _step("r1_popularity", "tau_days"), _default("r1_popularity")
    )
    assert [config["tau_days"] for config in configs] == [2.0, 4.0, 1.0, 5.0, 6.0, 7.0]


def test_nearest_order_follows_the_current_value_not_the_file_default() -> None:
    # After CF0 wins the R4 ablation, alpha is 0.5: order 0.7, then 0.9.
    current = {**_default("r4_itemcf"), "variant": "CF0", "alpha": 0.5}
    configs = ordered_candidate_configs(_step("r4_itemcf", "alpha"), current)
    assert [config["alpha"] for config in configs] == [0.7, 0.9]


def test_gamma_ties_prefer_smaller_value() -> None:
    configs = ordered_candidate_configs(_step("r3_sibling", "gamma"), _default("r3_sibling"))
    # From 0.5: 0.0 and 1.0 (distance 0.5, smaller first), then 1.5, then 2.0.
    assert [config["gamma"] for config in configs] == [0.0, 1.0, 1.5, 2.0]


def test_r4_seed_grid_listed_order() -> None:
    configs = ordered_candidate_configs(
        _step("r4_itemcf", "seed_window_x_decay"), _default("r4_itemcf")
    )
    cells = [(config["seed_window_days"], config["seed_decay"]) for config in configs]
    assert cells == [(90, True), (180, False), (180, True), (365, False), (365, True)]


def test_listed_order_and_default_is_left_out() -> None:
    configs = ordered_candidate_configs(_step("r4_itemcf", "variant"), _default("r4_itemcf"))
    assert [(config["variant"], config["alpha"]) for config in configs] == [("CF0", 0.5)]


def test_requires_skips_tau_for_variants_without_decay() -> None:
    tau = _step("r1_popularity", "tau_days")
    assert step_applies(tau, _default("r1_popularity"))
    assert not step_applies(tau, {**_default("r1_popularity"), "variant": "G0"})
    r2_tau = _step("r2_repurchase", "tau_days")
    assert not step_applies(r2_tau, {**_default("r2_repurchase"), "variant": "recency_only"})


def test_override_with_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="not config keys"):
        apply_overrides({"tau_days": 3.0}, {"tau": 2.0})


def test_parse_step_errors() -> None:
    good = {"name": "x", "kind": "select", "order": "listed", "candidates": [{"a": 1}, {"a": 2}]}
    parse_step(good, "t")
    with pytest.raises(ValueError, match="kind"):
        parse_step({**good, "kind": "grid"}, "t")
    with pytest.raises(ValueError, match="at least 2"):
        parse_step({**good, "candidates": [{"a": 1}]}, "t")
    with pytest.raises(ValueError, match="one and the same key"):
        parse_step({**good, "order": "nearest_to_current", "candidates": [{"a": 1}, {"b": 2}]}, "t")
    with pytest.raises(ValueError, match="numeric"):
        parse_step(
            {**good, "order": "nearest_to_current", "candidates": [{"a": "x"}, {"a": "y"}]}, "t"
        )
    with pytest.raises(ValueError, match="unknown keys"):
        parse_step({**good, "extra": 1}, "t")


def test_report_section() -> None:
    assert GRID.fold_groups == {"D1_D2": ("D1", "D2"), "D3_D4": ("D3", "D4")}
    assert GRID.extra_metrics == ("recall@12", "recall@20", "ndcg@12", "ndcg@20", "map@12")


def test_r2_lookback_step_after_tau() -> None:
    names = [step.name for step in GRID.channels["r2_repurchase"]]
    assert names == ["variant", "tau_days", "lookback_days"]
    configs = ordered_candidate_configs(
        _step("r2_repurchase", "lookback_days"), _default("r2_repurchase")
    )
    assert [config["lookback_days"] for config in configs] == [365]


def test_stop_at_boundary_flags_and_edges() -> None:
    flagged = {
        (channel, step.name)
        for channel, steps in GRID.channels.items()
        for step in steps
        if step.stop_at_boundary
    }
    assert flagged == {
        ("r4_itemcf", "alpha"),
        ("r5_wals", "alpha_conf"),
        ("r5_wals", "regularization"),
    }
    alpha = _step("r4_itemcf", "alpha")
    assert is_on_boundary(alpha, {"alpha": 0.9})
    assert is_on_boundary(alpha, {"alpha": 0.5})
    assert not is_on_boundary(alpha, {"alpha": 0.7})


def test_stop_at_boundary_needs_one_numeric_key() -> None:
    raw = {
        "name": "x",
        "kind": "select",
        "order": "listed",
        "stop_at_boundary": True,
        "candidates": [{"a": 1, "b": 2}, {"a": 2, "b": 3}],
    }
    with pytest.raises(ValueError, match="stop_at_boundary needs one and the same key"):
        parse_step(raw, "t")
    with pytest.raises(ValueError, match="must be true or false"):
        parse_step({**raw, "stop_at_boundary": "yes", "candidates": [{"a": 1}, {"a": 2}]}, "t")
