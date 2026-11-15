"""Read YAML config files into typed, frozen dataclasses.

Code should pass these dataclasses around, never raw dicts (AGENTS.md §6.4).

Loaders in this module:
    - load_base_config: configs/base.yaml -> BaseConfig
    - load_selection_rule: configs/selection_rule.yaml -> SelectionRule

The fold manifest is parsed in hmrec.data.folds, which reuses load_yaml and the
small field helpers defined here.

Every loader is strict: a missing key, an unknown key (usually a typo), or a
value of the wrong type raises ValueError with a message that names the key.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class PathsConfig:
    """Root folders. Relative values are resolved later by hmrec.paths.

    Attributes:
        data_root: Folder with raw/ and processed/ data.
        artifacts_root: Folder for regenerable outputs (candidates, features, models).
        results_root: Folder for small summary tables that are committed to git.
    """

    data_root: str
    artifacts_root: str
    results_root: str


@dataclass(frozen=True)
class MemoryConfig:
    """Memory budget settings (proposal §1.3).

    Attributes:
        limit_gb: Peak resident memory allowed for one task, in GB (1 GB = 1024**3 bytes).
        sample_interval_seconds: How often MemoryTracker samples process memory.
    """

    limit_gb: float
    sample_interval_seconds: float


@dataclass(frozen=True)
class ChunkingConfig:
    """Default chunk sizes for data that may not fit in memory (proposal §1.3).

    Attributes:
        query_chunk_size: Number of queries per chunk (fusion, features, ranker scoring).
        item_chunk_size: Number of seed items per chunk (R4 ItemCF co-occurrence).
    """

    query_chunk_size: int
    item_chunk_size: int


@dataclass(frozen=True)
class BaseConfig:
    """Global settings from configs/base.yaml.

    Attributes:
        paths: Root folders.
        seed: Global default random seed.
        num_threads: CPU threads for Polars, LightGBM and implicit.
        memory: Memory budget settings.
        chunking: Default chunk sizes.
    """

    paths: PathsConfig
    seed: int
    num_threads: int
    memory: MemoryConfig
    chunking: ChunkingConfig


@dataclass(frozen=True)
class SelectionRule:
    """The §4 selection rule from configs/selection_rule.yaml.

    A candidate replaces the default only if its query-weighted score improves
    by at least `min_relative_improvement` (relative) AND it is strictly better
    on at least `min_folds_better` of `n_folds` folds.

    Attributes:
        primary_aggregation: How fold scores are combined for selection ("query_weighted").
        also_report: Other aggregations that are reported but not used to select.
        min_relative_improvement: Minimum (candidate - default) / default, e.g. 0.01 = 1%.
        min_folds_better: Minimum number of folds where the candidate must be better.
        n_folds: Number of folds in one selection phase.
        tie_breaker: What to prefer when candidates are about equal.
        selection_metric: Component name -> metric name, e.g. "r2_repurchase" -> "recall@30".
    """

    primary_aggregation: str
    also_report: tuple[str, ...]
    min_relative_improvement: float
    min_folds_better: int
    n_folds: int
    tie_breaker: str
    selection_metric: dict[str, str]


# ---------------------------------------------------------------- YAML + field helpers


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file whose top level is a mapping.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed top-level mapping. Dates written as YYYY-MM-DD are parsed
        into datetime.date by PyYAML.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the top level is not a mapping.
    """
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        parsed = yaml.safe_load(file)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(parsed).__name__}")
    return parsed


def check_keys(raw: Mapping[str, Any], expected: set[str], where: str) -> None:
    """Fail if `raw` is missing an expected key or has an unknown key.

    Unknown keys are rejected on purpose: they are almost always typos, and a
    typo would otherwise silently fall back to some other value.

    Args:
        raw: The mapping to check.
        expected: The exact set of keys that must be present.
        where: Name of the section, used in the error message.

    Raises:
        ValueError: If keys are missing or unknown.
    """
    actual = set(raw.keys())
    missing = expected - actual
    unknown = actual - expected
    if missing:
        raise ValueError(f"{where}: missing keys {sorted(missing)}")
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; expected {sorted(expected)}")


def get_mapping(raw: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    """Return raw[key] and check that it is a mapping."""
    value = raw[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}.{key}: expected a mapping, got {type(value).__name__}")
    return value


def get_str(raw: Mapping[str, Any], key: str, where: str) -> str:
    """Return raw[key] and check that it is a non-empty string."""
    value = raw[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{where}.{key}: expected a non-empty string, got {value!r}")
    return value


def get_int(raw: Mapping[str, Any], key: str, where: str) -> int:
    """Return raw[key] and check that it is an int.

    bool is rejected explicitly, because in Python `True` is also an int.
    """
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}.{key}: expected an integer, got {value!r}")
    return value


def get_float(raw: Mapping[str, Any], key: str, where: str) -> float:
    """Return raw[key] as a float. Integers such as `6` are accepted."""
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}.{key}: expected a number, got {value!r}")
    return float(value)


def get_str_list(raw: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    """Return raw[key] as a tuple of non-empty strings."""
    value = raw[key]
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{where}.{key}: expected a list of strings, got {value!r}")
    return tuple(value)


def require_positive(value: float, name: str) -> None:
    """Fail if `value` is not strictly greater than zero."""
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


# ---------------------------------------------------------------- base.yaml


def parse_base_config(raw: Mapping[str, Any]) -> BaseConfig:
    """Build a BaseConfig from the parsed content of configs/base.yaml.

    Args:
        raw: Top-level mapping with keys paths, seed, num_threads, memory, chunking.

    Returns:
        A validated BaseConfig.

    Raises:
        ValueError: If a key is missing or unknown, a type is wrong, or a
            number that must be positive is not.
    """
    check_keys(raw, {"paths", "seed", "num_threads", "memory", "chunking"}, "base")

    raw_paths = get_mapping(raw, "paths", "base")
    check_keys(raw_paths, {"data_root", "artifacts_root", "results_root"}, "base.paths")
    paths = PathsConfig(
        data_root=get_str(raw_paths, "data_root", "base.paths"),
        artifacts_root=get_str(raw_paths, "artifacts_root", "base.paths"),
        results_root=get_str(raw_paths, "results_root", "base.paths"),
    )

    raw_memory = get_mapping(raw, "memory", "base")
    check_keys(raw_memory, {"limit_gb", "sample_interval_seconds"}, "base.memory")
    memory = MemoryConfig(
        limit_gb=get_float(raw_memory, "limit_gb", "base.memory"),
        sample_interval_seconds=get_float(raw_memory, "sample_interval_seconds", "base.memory"),
    )

    raw_chunking = get_mapping(raw, "chunking", "base")
    check_keys(raw_chunking, {"query_chunk_size", "item_chunk_size"}, "base.chunking")
    chunking = ChunkingConfig(
        query_chunk_size=get_int(raw_chunking, "query_chunk_size", "base.chunking"),
        item_chunk_size=get_int(raw_chunking, "item_chunk_size", "base.chunking"),
    )

    config = BaseConfig(
        paths=paths,
        seed=get_int(raw, "seed", "base"),
        num_threads=get_int(raw, "num_threads", "base"),
        memory=memory,
        chunking=chunking,
    )
    _validate_base_config(config)
    return config


def _validate_base_config(config: BaseConfig) -> None:
    """Check value ranges that the type checks above cannot express."""
    if config.seed < 0:
        raise ValueError(f"base.seed must be >= 0, got {config.seed}")
    require_positive(config.num_threads, "base.num_threads")
    require_positive(config.memory.limit_gb, "base.memory.limit_gb")
    require_positive(config.memory.sample_interval_seconds, "base.memory.sample_interval_seconds")
    require_positive(config.chunking.query_chunk_size, "base.chunking.query_chunk_size")
    require_positive(config.chunking.item_chunk_size, "base.chunking.item_chunk_size")


def load_base_config(path: Path) -> BaseConfig:
    """Read and validate configs/base.yaml.

    Args:
        path: Path to base.yaml.

    Returns:
        A validated BaseConfig.
    """
    return parse_base_config(load_yaml(path))


# ---------------------------------------------------------------- selection_rule.yaml

# The only aggregation the proposal allows for selection (§4.1).
_ALLOWED_PRIMARY_AGGREGATIONS = {"query_weighted"}


def parse_selection_rule(raw: Mapping[str, Any]) -> SelectionRule:
    """Build a SelectionRule from the parsed content of configs/selection_rule.yaml.

    Args:
        raw: Top-level mapping with keys aggregation, acceptance, selection_metric.

    Returns:
        A validated SelectionRule.

    Raises:
        ValueError: If a key is missing or unknown, a type is wrong, or the
            numbers are inconsistent (e.g. min_folds_better > n_folds).
    """
    check_keys(raw, {"aggregation", "acceptance", "selection_metric"}, "selection_rule")

    raw_aggregation = get_mapping(raw, "aggregation", "selection_rule")
    check_keys(raw_aggregation, {"primary", "also_report"}, "selection_rule.aggregation")

    raw_acceptance = get_mapping(raw, "acceptance", "selection_rule")
    acceptance_keys = {"min_relative_improvement", "min_folds_better", "n_folds", "tie_breaker"}
    check_keys(raw_acceptance, acceptance_keys, "selection_rule.acceptance")

    raw_metrics = get_mapping(raw, "selection_metric", "selection_rule")
    selection_metric = {
        component: get_str(raw_metrics, component, "selection_rule.selection_metric")
        for component in raw_metrics
    }

    where = "selection_rule.acceptance"
    rule = SelectionRule(
        primary_aggregation=get_str(raw_aggregation, "primary", "selection_rule.aggregation"),
        also_report=get_str_list(raw_aggregation, "also_report", "selection_rule.aggregation"),
        min_relative_improvement=get_float(raw_acceptance, "min_relative_improvement", where),
        min_folds_better=get_int(raw_acceptance, "min_folds_better", where),
        n_folds=get_int(raw_acceptance, "n_folds", where),
        tie_breaker=get_str(raw_acceptance, "tie_breaker", where),
        selection_metric=selection_metric,
    )
    _validate_selection_rule(rule)
    return rule


def _validate_selection_rule(rule: SelectionRule) -> None:
    """Check value ranges and consistency of a SelectionRule."""
    if rule.primary_aggregation not in _ALLOWED_PRIMARY_AGGREGATIONS:
        raise ValueError(
            f"selection_rule.aggregation.primary must be one of "
            f"{sorted(_ALLOWED_PRIMARY_AGGREGATIONS)}, got {rule.primary_aggregation!r}"
        )
    if rule.min_relative_improvement < 0:
        raise ValueError(
            f"selection_rule.acceptance.min_relative_improvement must be >= 0, "
            f"got {rule.min_relative_improvement}"
        )
    require_positive(rule.n_folds, "selection_rule.acceptance.n_folds")
    if not 1 <= rule.min_folds_better <= rule.n_folds:
        raise ValueError(
            f"selection_rule.acceptance.min_folds_better must be in [1, n_folds={rule.n_folds}], "
            f"got {rule.min_folds_better}"
        )
    if not rule.selection_metric:
        raise ValueError("selection_rule.selection_metric must list at least one component")


def load_selection_rule(path: Path) -> SelectionRule:
    """Read and validate configs/selection_rule.yaml.

    Args:
        path: Path to selection_rule.yaml.

    Returns:
        A validated SelectionRule.
    """
    return parse_selection_rule(load_yaml(path))
