"""Read the D-phase grid and order each step's candidates (proposal §4.2-4.3).

Input: configs/hpo/d_phase.yaml. For each channel, an ordered list of steps.
A step lists candidate overrides; each override is applied to the CURRENT
default config (the winner of the previous step) to get a candidate config.

Priority order of the candidates (part of the selection rule, decided before
any run, see docs/decisions.md 2026-11-17):
    order: listed             -> as written in the YAML (simpler / cheaper first)
    order: nearest_to_current -> one numeric parameter; candidates closer to the
                                 current value come first, ties -> smaller value

This module only orders configurations. Scores, the §4.2 rule and the decision
live in hmrec.hpo.selection; running trials lives in hmrec.hpo.runner.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hmrec.config import load_yaml

STEP_KINDS = ("select", "convergence_check")
ORDERS = ("listed", "nearest_to_current")
STEP_KEYS = {"name", "kind", "order", "candidates", "requires", "stop_at_boundary"}
REQUIRED_STEP_KEYS = {"name", "kind", "order", "candidates"}

RawConfig = dict[str, Any]


@dataclass(frozen=True)
class Step:
    """One step of a channel's D-phase plan.

    Attributes:
        name: Step name, e.g. "tau_days".
        kind: "select" (the §4.2 multi-candidate rule) or "convergence_check" (R5 iterations).
        order: "listed" or "nearest_to_current" (see the module docstring).
        candidates: Overrides, each a mapping {config key: value}, as written.
        requires: {config key: allowed values}; the step is skipped when the
            current config has another value. Empty = always runs.
        stop_at_boundary: If True (one numeric parameter only), a selected value
            equal to the smallest or largest value of the step stops the channel
            for a human decision instead of moving on (see is_on_boundary).
    """

    name: str
    kind: str
    order: str
    candidates: tuple[dict[str, Any], ...]
    requires: dict[str, tuple[Any, ...]]
    stop_at_boundary: bool = False


# ---------------------------------------------------------------- reading


def _parse_requires(raw: Any, where: str) -> dict[str, tuple[Any, ...]]:
    """`requires` maps a config key to a list of allowed values."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}.requires must be a mapping, got {raw!r}")
    requires: dict[str, tuple[Any, ...]] = {}
    for key, values in raw.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"{where}.requires.{key} must be a non-empty list")
        requires[str(key)] = tuple(values)
    return requires


def _check_one_numeric_key(
    candidates: tuple[dict[str, Any], ...], where: str, feature: str
) -> None:
    """`feature` (nearest_to_current / stop_at_boundary) needs one numeric key per candidate."""
    keys = {tuple(candidate) for candidate in candidates}
    if len(keys) != 1 or len(next(iter(keys))) != 1:
        raise ValueError(f"{where}: {feature} needs one and the same key in every candidate")
    for candidate in candidates:
        value = next(iter(candidate.values()))
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{where}: {feature} needs numeric values, got {value!r}")


def parse_step(raw: Any, where: str) -> Step:
    """Build and validate one Step from its YAML mapping."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a mapping, got {raw!r}")
    unknown = set(raw) - STEP_KEYS
    missing = REQUIRED_STEP_KEYS - set(raw)
    if unknown or missing:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    kind, order = raw["kind"], raw["order"]
    if kind not in STEP_KINDS:
        raise ValueError(f"{where}.kind must be one of {list(STEP_KINDS)}, got {kind!r}")
    if order not in ORDERS:
        raise ValueError(f"{where}.order must be one of {list(ORDERS)}, got {order!r}")
    raw_candidates = raw["candidates"]
    if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
        raise ValueError(f"{where}.candidates must list at least 2 configurations")
    candidates = tuple(dict(candidate) for candidate in raw_candidates)
    if any(not candidate for candidate in candidates):
        raise ValueError(f"{where}.candidates: an override must set at least one key")
    if order == "nearest_to_current":
        _check_one_numeric_key(candidates, where, "nearest_to_current")
    stop_at_boundary = raw.get("stop_at_boundary", False)
    if not isinstance(stop_at_boundary, bool):
        raise ValueError(f"{where}.stop_at_boundary must be true or false")
    if stop_at_boundary:
        _check_one_numeric_key(candidates, where, "stop_at_boundary")
    return Step(
        name=str(raw["name"]),
        kind=kind,
        order=order,
        candidates=candidates,
        requires=_parse_requires(raw.get("requires"), where),
        stop_at_boundary=stop_at_boundary,
    )


@dataclass(frozen=True)
class PhaseGrid:
    """The whole configs/hpo/d_phase.yaml.

    Attributes:
        fold_groups: Group name -> fold names; diagnostic group scores (proposal §4.1).
        extra_metrics: Metrics reported next to the selection metric, e.g. "ndcg@12".
        channels: Channel name (e.g. "r1_popularity") -> its steps in run order.
    """

    fold_groups: dict[str, tuple[str, ...]]
    extra_metrics: tuple[str, ...]
    channels: dict[str, tuple[Step, ...]]


def _parse_report(raw: Any) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """The `report` section: fold_groups and extra_metrics."""
    if not isinstance(raw, Mapping) or set(raw) != {"fold_groups", "extra_metrics"}:
        raise ValueError("report must be a mapping with keys fold_groups and extra_metrics")
    fold_groups = {str(name): tuple(folds) for name, folds in raw["fold_groups"].items()}
    return fold_groups, tuple(str(metric) for metric in raw["extra_metrics"])


def parse_phase_grid(raw: Mapping[str, Any]) -> PhaseGrid:
    """Build a PhaseGrid from the parsed YAML (top-level keys: report, channels)."""
    if set(raw) != {"report", "channels"}:
        raise ValueError(f"top-level keys must be report and channels, got {sorted(raw)}")
    fold_groups, extra_metrics = _parse_report(raw["report"])
    channels: dict[str, tuple[Step, ...]] = {}
    for channel, raw_steps in raw["channels"].items():
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(f"{channel} must be a non-empty list of steps")
        steps = tuple(
            parse_step(raw_step, f"{channel}[{position}]")
            for position, raw_step in enumerate(raw_steps)
        )
        names = [step.name for step in steps]
        if len(set(names)) != len(names):
            raise ValueError(f"{channel}: step names repeat: {names}")
        channels[str(channel)] = steps
    return PhaseGrid(fold_groups, extra_metrics, channels)


def load_phase_grid(path: Path) -> PhaseGrid:
    """Read and validate configs/hpo/d_phase.yaml."""
    return parse_phase_grid(load_yaml(path))


# ---------------------------------------------------------------- ordering


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> RawConfig:
    """Return a copy of `config` with `overrides` applied.

    Raises:
        ValueError: If an override key is not a key of `config` (a typo would
            otherwise silently add an unused setting).
    """
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(f"override keys {sorted(unknown)} are not config keys {sorted(config)}")
    result = copy.deepcopy(dict(config))
    result.update(overrides)
    return result


def step_applies(step: Step, current: Mapping[str, Any]) -> bool:
    """True if the current config satisfies the step's `requires`."""
    return all(current[key] in allowed for key, allowed in step.requires.items())


def _nearest_first(step: Step, current: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Candidates sorted by |value - current value|, ties -> smaller value first."""
    key = next(iter(step.candidates[0]))
    current_value = float(current[key])

    def priority(candidate: dict[str, Any]) -> tuple[float, float]:
        value = float(candidate[key])
        return (abs(value - current_value), value)

    return sorted(step.candidates, key=priority)


def ordered_candidate_configs(step: Step, current: Mapping[str, Any]) -> list[RawConfig]:
    """The step's candidate configs in priority order, without the current config itself.

    Args:
        step: The step.
        current: The current default config (raw YAML mapping).

    Returns:
        Full configs (current + override), highest priority first. A candidate
        whose config equals `current` is the default, not a candidate, and is
        left out; duplicates are kept once.
    """
    if step.order == "nearest_to_current":
        overrides = _nearest_first(step, current)
    else:
        overrides = list(step.candidates)
    configs: list[RawConfig] = []
    for override in overrides:
        config = apply_overrides(current, override)
        if config != dict(current) and config not in configs:
            configs.append(config)
    return configs


def is_on_boundary(step: Step, config: Mapping[str, Any]) -> bool:
    """True if `config`'s value of the step's parameter is the smallest or largest step value.

    Only for one-numeric-parameter steps (stop_at_boundary steps are checked for
    that when the grid is read).
    """
    key = next(iter(step.candidates[0]))
    values = [float(candidate[key]) for candidate in step.candidates]
    value = float(config[key])
    return value in (min(values), max(values))
