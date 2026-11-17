"""Read YAML config files into typed, frozen dataclasses.

Code should pass these dataclasses around, never raw dicts (AGENTS.md §6.4).

Loaders in this module:
    - load_base_config: configs/base.yaml -> BaseConfig
    - load_selection_rule: configs/selection_rule.yaml -> SelectionRule
    - load_preprocess_config: configs/preprocess.yaml -> PreprocessConfig
    - load_catalog_config: configs/catalog.yaml -> CatalogConfig
    - load_queries_config: configs/queries.yaml -> QueriesConfig
    - load_popularity_config / load_repurchase_config / load_sibling_config:
      configs/recall/r{1,2,3}_*.yaml -> PopularityConfig / RepurchaseConfig / SiblingConfig

The fold manifest is parsed in hmrec.data.folds, which reuses load_yaml and the
small field helpers defined here.

Every loader is strict: a missing key, an unknown key (usually a typo), or a
value of the wrong type raises ValueError with a message that names the key.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime
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
        dense_score_query_chunk_size: Number of queries per chunk when every active
            article is scored with a matrix product (R5 WALS; later R6).
        pair_chunk_size: Number of (customer, article) pairs per chunk when
            computing vector dot products (hmrec.vectors.dot_pairs).
    """

    query_chunk_size: int
    item_chunk_size: int
    dense_score_query_chunk_size: int
    pair_chunk_size: int


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


def get_bool(raw: Mapping[str, Any], key: str, where: str) -> bool:
    """Return raw[key] and check that it is a bool (YAML true / false, not 0 / 1)."""
    value = raw[key]
    if not isinstance(value, bool):
        raise ValueError(f"{where}.{key}: expected true or false, got {value!r}")
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


def get_date(raw: Mapping[str, Any], key: str, where: str) -> date:
    """Return raw[key] as a date. PyYAML already turns YYYY-MM-DD into a date."""
    value = raw[key]
    # datetime is a subclass of date, so reject it explicitly: a time of day
    # would make "<" comparisons with plain dates fail or behave unexpectedly.
    if not isinstance(value, date) or isinstance(value, datetime):
        raise ValueError(f"{where}.{key}: expected a date YYYY-MM-DD, got {value!r}")
    return value


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
    check_keys(
        raw_chunking,
        {"query_chunk_size", "item_chunk_size", "dense_score_query_chunk_size", "pair_chunk_size"},
        "base.chunking",
    )
    chunking = ChunkingConfig(
        query_chunk_size=get_int(raw_chunking, "query_chunk_size", "base.chunking"),
        item_chunk_size=get_int(raw_chunking, "item_chunk_size", "base.chunking"),
        dense_score_query_chunk_size=get_int(
            raw_chunking, "dense_score_query_chunk_size", "base.chunking"
        ),
        pair_chunk_size=get_int(raw_chunking, "pair_chunk_size", "base.chunking"),
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
    require_positive(
        config.chunking.dense_score_query_chunk_size, "base.chunking.dense_score_query_chunk_size"
    )
    require_positive(config.chunking.pair_chunk_size, "base.chunking.pair_chunk_size")


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


# ---------------------------------------------------------------- preprocess.yaml


@dataclass(frozen=True)
class AgeBucket:
    """One age bucket for R1 popularity (proposal §5 R1).

    Attributes:
        name: Bucket label stored in customers.age_bucket, e.g. "AGE_18_24".
        min_age: Smallest age in the bucket, INCLUSIVE.
        max_age: Largest age in the bucket, INCLUSIVE.
    """

    name: str
    min_age: int
    max_age: int


@dataclass(frozen=True)
class EmbeddingConfig:
    """Expected shape and normalization of the precomputed embeddings (proposal §2.1).

    Attributes:
        image_dim: Width of the CLIP image vectors (512).
        text_dim: Width of the MiniLM text vectors (384).
        norm_tolerance: Every present vector must have L2 norm in [1 - tol, 1 + tol].
    """

    image_dim: int
    text_dim: int
    norm_tolerance: float


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings for M0 preprocessing from configs/preprocess.yaml.

    Attributes:
        day_zero: Day 0 of the integer day index: t_dat_day = (t_dat - day_zero).days.
        age_buckets: Valid-age buckets, sorted and contiguous.
        invalid_age_bucket: Bucket for ages outside every bucket (< 18 or > 99).
        unknown_age_bucket: Bucket for a missing age.
        embeddings: Expected embedding widths and norm tolerance.
        transactions_row_group_size: Rows per Parquet row group in transactions.parquet.
    """

    day_zero: date
    age_buckets: tuple[AgeBucket, ...]
    invalid_age_bucket: str
    unknown_age_bucket: str
    embeddings: EmbeddingConfig
    transactions_row_group_size: int

    @property
    def all_age_bucket_names(self) -> tuple[str, ...]:
        """Every possible age_bucket value: valid buckets, then invalid, then unknown."""
        valid_names = tuple(bucket.name for bucket in self.age_buckets)
        return (*valid_names, self.invalid_age_bucket, self.unknown_age_bucket)


def _parse_age_bucket(raw: Any, position: int) -> AgeBucket:
    """Parse one entry of the age_buckets list."""
    where = f"preprocess.age_buckets[{position}]"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {raw!r}")
    check_keys(raw, {"name", "min_age", "max_age"}, where)
    return AgeBucket(
        name=get_str(raw, "name", where),
        min_age=get_int(raw, "min_age", where),
        max_age=get_int(raw, "max_age", where),
    )


def parse_preprocess_config(raw: Mapping[str, Any]) -> PreprocessConfig:
    """Build a PreprocessConfig from the parsed content of configs/preprocess.yaml.

    Args:
        raw: Top-level mapping with keys day_zero, age_buckets, invalid_age_bucket,
            unknown_age_bucket, embeddings, transactions_row_group_size.

    Returns:
        A validated PreprocessConfig.

    Raises:
        ValueError: If a key is missing or unknown, a type is wrong, or the age
            buckets overlap, leave a gap, or reuse a name.
    """
    expected_keys = {
        "day_zero",
        "age_buckets",
        "invalid_age_bucket",
        "unknown_age_bucket",
        "embeddings",
        "transactions_row_group_size",
    }
    check_keys(raw, expected_keys, "preprocess")

    raw_buckets = raw["age_buckets"]
    if not isinstance(raw_buckets, list) or not raw_buckets:
        raise ValueError("preprocess.age_buckets: expected a non-empty list")
    age_buckets = tuple(
        _parse_age_bucket(raw_bucket, position) for position, raw_bucket in enumerate(raw_buckets)
    )

    raw_embeddings = get_mapping(raw, "embeddings", "preprocess")
    check_keys(raw_embeddings, {"image_dim", "text_dim", "norm_tolerance"}, "preprocess.embeddings")
    embeddings = EmbeddingConfig(
        image_dim=get_int(raw_embeddings, "image_dim", "preprocess.embeddings"),
        text_dim=get_int(raw_embeddings, "text_dim", "preprocess.embeddings"),
        norm_tolerance=get_float(raw_embeddings, "norm_tolerance", "preprocess.embeddings"),
    )

    config = PreprocessConfig(
        day_zero=get_date(raw, "day_zero", "preprocess"),
        age_buckets=age_buckets,
        invalid_age_bucket=get_str(raw, "invalid_age_bucket", "preprocess"),
        unknown_age_bucket=get_str(raw, "unknown_age_bucket", "preprocess"),
        embeddings=embeddings,
        transactions_row_group_size=get_int(raw, "transactions_row_group_size", "preprocess"),
    )
    _validate_preprocess_config(config)
    return config


def _validate_age_buckets(age_buckets: tuple[AgeBucket, ...]) -> None:
    """Fail unless the buckets are non-empty ranges, sorted, and contiguous (no gap, no overlap)."""
    for bucket in age_buckets:
        if bucket.min_age > bucket.max_age:
            raise ValueError(
                f"age bucket {bucket.name}: min_age={bucket.min_age} > max_age={bucket.max_age}"
            )
    for previous, current in zip(age_buckets, age_buckets[1:], strict=False):
        # Both ends are inclusive, so the next bucket must start exactly one year later.
        if current.min_age <= previous.max_age:
            raise ValueError(
                f"age buckets {previous.name} and {current.name} overlap: "
                f"{previous.max_age} >= {current.min_age}"
            )
        if current.min_age != previous.max_age + 1:
            raise ValueError(
                f"age buckets {previous.name} and {current.name} leave a gap: "
                f"ages {previous.max_age + 1}..{current.min_age - 1} belong to no bucket"
            )


def _validate_preprocess_config(config: PreprocessConfig) -> None:
    """Check value ranges and consistency of a PreprocessConfig."""
    _validate_age_buckets(config.age_buckets)
    names = config.all_age_bucket_names
    if len(set(names)) != len(names):
        raise ValueError(f"preprocess: age bucket names must be unique, got {list(names)}")
    require_positive(config.embeddings.image_dim, "preprocess.embeddings.image_dim")
    require_positive(config.embeddings.text_dim, "preprocess.embeddings.text_dim")
    require_positive(config.embeddings.norm_tolerance, "preprocess.embeddings.norm_tolerance")
    require_positive(config.transactions_row_group_size, "preprocess.transactions_row_group_size")


def load_preprocess_config(path: Path) -> PreprocessConfig:
    """Read and validate configs/preprocess.yaml.

    Args:
        path: Path to preprocess.yaml.

    Returns:
        A validated PreprocessConfig.
    """
    return parse_preprocess_config(load_yaml(path))


def preprocess_config_hash(config: PreprocessConfig) -> str:
    """Return a sha256 hex digest that identifies the preprocessing settings.

    The hash is computed from the PARSED values (not the raw file bytes), so
    editing comments or whitespace in preprocess.yaml does not change it, but
    changing any value does. Keys are sorted so the result is stable.

    Args:
        config: Parsed preprocess config.

    Returns:
        64-character lowercase hex string.
    """
    # default=str turns the day_zero date into "YYYY-MM-DD".
    canonical_json = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- catalog.yaml

# Largest value of an int32 column. query_id and the *_idx columns are int32.
INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class ActivePoolConfig:
    """Rule for the active pool A(a) (proposal §2.4). Frozen in M0, not a hyperparameter.

    Attributes:
        window_days: Length of the history window [anchor - window_days, anchor).
        min_purchase_lines: Minimum number of purchase lines in the window.
            Duplicate lines count separately.
    """

    window_days: int
    min_purchase_lines: int


@dataclass(frozen=True)
class CatalogConfig:
    """Settings from configs/catalog.yaml.

    Attributes:
        active_pool: Rule for A(a).
    """

    active_pool: ActivePoolConfig


def parse_catalog_config(raw: Mapping[str, Any]) -> CatalogConfig:
    """Build a CatalogConfig from the parsed content of configs/catalog.yaml.

    Raises:
        ValueError: If a key is missing or unknown, or a value is not a positive integer.
    """
    check_keys(raw, {"active_pool"}, "catalog")
    raw_pool = get_mapping(raw, "active_pool", "catalog")
    where = "catalog.active_pool"
    check_keys(raw_pool, {"window_days", "min_purchase_lines"}, where)
    active_pool = ActivePoolConfig(
        window_days=get_int(raw_pool, "window_days", where),
        min_purchase_lines=get_int(raw_pool, "min_purchase_lines", where),
    )
    require_positive(active_pool.window_days, f"{where}.window_days")
    require_positive(active_pool.min_purchase_lines, f"{where}.min_purchase_lines")
    return CatalogConfig(active_pool=active_pool)


def load_catalog_config(path: Path) -> CatalogConfig:
    """Read and validate configs/catalog.yaml."""
    return parse_catalog_config(load_yaml(path))


# ---------------------------------------------------------------- queries.yaml


@dataclass(frozen=True)
class QueriesConfig:
    """Settings from configs/queries.yaml.

    Attributes:
        query_id_fold_offset: query_id = fold_ordinal * offset + position.
            Every fold must have fewer queries than this value.
    """

    query_id_fold_offset: int


def parse_queries_config(raw: Mapping[str, Any]) -> QueriesConfig:
    """Build a QueriesConfig from the parsed content of configs/queries.yaml.

    The check that (number of folds) * offset fits in int32 needs the fold
    manifest, so it is done in hmrec.data.queries, not here.

    Raises:
        ValueError: If the key is missing or unknown, or the offset is not a
            positive integer that fits in int32.
    """
    check_keys(raw, {"query_id_fold_offset"}, "queries")
    offset = get_int(raw, "query_id_fold_offset", "queries")
    require_positive(offset, "queries.query_id_fold_offset")
    if offset > INT32_MAX:
        raise ValueError(f"queries.query_id_fold_offset={offset} does not fit in int32")
    return QueriesConfig(query_id_fold_offset=offset)


def load_queries_config(path: Path) -> QueriesConfig:
    """Read and validate configs/queries.yaml."""
    return parse_queries_config(load_yaml(path))


# ---------------------------------------------------------------- recall/r1..r4

POPULARITY_VARIANTS = ("A1", "G0")
REPURCHASE_VARIANTS = ("decayed_frequency", "recency_only")
# CF0 = every user has weight 1; CF3 = IUF weight 1 / ln(1 + d_u) (proposal §5 R4).
ITEMCF_VARIANTS = ("CF0", "CF3")
# W0 = r_ui is 1 (bought or not); W2 = r_ui is the decayed number of purchase lines (§5 R5).
WALS_VARIANTS = ("W0", "W2")


def _get_choice(raw: Mapping[str, Any], key: str, where: str, choices: tuple[str, ...]) -> str:
    """Return raw[key] and check that it is one of `choices`."""
    value = get_str(raw, key, where)
    if value not in choices:
        raise ValueError(f"{where}.{key}: expected one of {list(choices)}, got {value!r}")
    return value


@dataclass(frozen=True)
class PopularityConfig:
    """R1 settings from configs/recall/r1_popularity.yaml (proposal §5 R1).

    Attributes:
        variant: "A1" (per age bucket, decayed) or "G0" (global raw count, no buckets).
        window_days: History window [anchor - window_days, anchor).
        tau_days: Decay time constant (not half-life); unused by G0.
        quota: Maximum candidates per query.
    """

    variant: str
    window_days: int
    tau_days: float
    quota: int


@dataclass(frozen=True)
class RepurchaseConfig:
    """R2 settings from configs/recall/r2_repurchase.yaml (proposal §5 R2).

    Attributes:
        variant: "decayed_frequency" or "recency_only".
        lookback_days: History window [anchor - lookback_days, anchor).
        tau_days: Decay time constant; unused by recency_only.
        quota: Maximum candidates per query (soft, never padded).
    """

    variant: str
    lookback_days: int
    tau_days: float
    quota: int


@dataclass(frozen=True)
class SiblingConfig:
    """R3 settings from configs/recall/r3_sibling.yaml (proposal §5 R3, formula S2).

    Attributes:
        style_lookback_days: Window for P(u, c) and the exclusion of bought articles.
        market_window_days: Window for the market count M(j).
        tau_days: Decay time constant of P(u, c).
        gamma: Exponent of (1 + M(j)); 0 means style preference only.
        quota: Maximum candidates per query (never padded).
    """

    style_lookback_days: int
    market_window_days: int
    tau_days: float
    gamma: float
    quota: int


@dataclass(frozen=True)
class ItemCFConfig:
    """R4 settings from configs/recall/r4_itemcf.yaml (proposal §5 R4).

    Attributes:
        variant: "CF0" (user weight 1) or "CF3" (IUF user weight 1 / ln(1 + d_u)).
        fit_window_days: Window [anchor - fit_window_days, anchor) for C_w and n_w.
        seed_window_days: Window [anchor - seed_window_days, anchor) for a user's seeds.
        alpha: Exponent in s(i -> j) = C_w(i, j) / (n_w(i)^alpha * n_w(j)^(1 - alpha)),
            in [0, 1]. CF0 with alpha 0.5 is the plain cosine C / sqrt(n(i) n(j)).
        neighbors_per_seed: K_n, neighbours kept per seed item.
        seed_decay: False -> seed weight q_ui = 1; True -> q_ui = exp(-age_ui / seed_tau_days),
            age_ui = days since u last bought seed i.
        seed_tau_days: Time constant of the seed decay. Fixed at 30 (not tuned);
            unused when seed_decay is False.
        quota: Maximum candidates per query (never padded).
    """

    variant: str
    fit_window_days: int
    seed_window_days: int
    alpha: float
    neighbors_per_seed: int
    seed_decay: bool
    seed_tau_days: float
    quota: int


@dataclass(frozen=True)
class WalsConfig:
    """R5 settings from configs/recall/r5_wals.yaml (proposal §5 R5).

    Attributes:
        variant: "W0" (r_ui = 1) or "W2" (r_ui = sum of exp(-age_days / decay_tau_days)).
        window_days: Fit window [anchor - window_days, anchor).
        decay_tau_days: Decay time constant of r_ui (fixed at 60); unused by W0.
        alpha_conf: Confidence c_ui = 1 + alpha_conf * r_ui.
        regularization: ALS L2 regularization (lambda).
        factors: Vector length.
        iterations: Number of ALS iterations (no early stopping).
        random_seed: Seed of the random initial factors.
        quota: Maximum candidates per query (never padded).
    """

    variant: str
    window_days: int
    decay_tau_days: float
    alpha_conf: float
    regularization: float
    factors: int
    iterations: int
    random_seed: int
    quota: int


def parse_popularity_config(raw: Mapping[str, Any]) -> PopularityConfig:
    """Build a PopularityConfig from the parsed r1_popularity.yaml."""
    where = "r1_popularity"
    check_keys(raw, {"variant", "window_days", "tau_days", "quota"}, where)
    config = PopularityConfig(
        variant=_get_choice(raw, "variant", where, POPULARITY_VARIANTS),
        window_days=get_int(raw, "window_days", where),
        tau_days=get_float(raw, "tau_days", where),
        quota=get_int(raw, "quota", where),
    )
    require_positive(config.window_days, f"{where}.window_days")
    require_positive(config.tau_days, f"{where}.tau_days")
    require_positive(config.quota, f"{where}.quota")
    return config


def parse_repurchase_config(raw: Mapping[str, Any]) -> RepurchaseConfig:
    """Build a RepurchaseConfig from the parsed r2_repurchase.yaml."""
    where = "r2_repurchase"
    check_keys(raw, {"variant", "lookback_days", "tau_days", "quota"}, where)
    config = RepurchaseConfig(
        variant=_get_choice(raw, "variant", where, REPURCHASE_VARIANTS),
        lookback_days=get_int(raw, "lookback_days", where),
        tau_days=get_float(raw, "tau_days", where),
        quota=get_int(raw, "quota", where),
    )
    require_positive(config.lookback_days, f"{where}.lookback_days")
    require_positive(config.tau_days, f"{where}.tau_days")
    require_positive(config.quota, f"{where}.quota")
    return config


def parse_sibling_config(raw: Mapping[str, Any]) -> SiblingConfig:
    """Build a SiblingConfig from the parsed r3_sibling.yaml.

    gamma may be 0 (style preference only) but not negative.
    """
    where = "r3_sibling"
    expected = {"style_lookback_days", "market_window_days", "tau_days", "gamma", "quota"}
    check_keys(raw, expected, where)
    config = SiblingConfig(
        style_lookback_days=get_int(raw, "style_lookback_days", where),
        market_window_days=get_int(raw, "market_window_days", where),
        tau_days=get_float(raw, "tau_days", where),
        gamma=get_float(raw, "gamma", where),
        quota=get_int(raw, "quota", where),
    )
    require_positive(config.style_lookback_days, f"{where}.style_lookback_days")
    require_positive(config.market_window_days, f"{where}.market_window_days")
    require_positive(config.tau_days, f"{where}.tau_days")
    require_positive(config.quota, f"{where}.quota")
    if config.gamma < 0:
        raise ValueError(f"{where}.gamma must be >= 0, got {config.gamma}")
    return config


def parse_itemcf_config(raw: Mapping[str, Any]) -> ItemCFConfig:
    """Build an ItemCFConfig from the parsed r4_itemcf.yaml.

    The seed window must fit inside the fit window: every seed must be a node
    of the co-occurrence graph, otherwise it would have no neighbours.
    """
    where = "r4_itemcf"
    expected = {
        "variant",
        "fit_window_days",
        "seed_window_days",
        "alpha",
        "neighbors_per_seed",
        "seed_decay",
        "seed_tau_days",
        "quota",
    }
    check_keys(raw, expected, where)
    config = ItemCFConfig(
        variant=_get_choice(raw, "variant", where, ITEMCF_VARIANTS),
        fit_window_days=get_int(raw, "fit_window_days", where),
        seed_window_days=get_int(raw, "seed_window_days", where),
        alpha=get_float(raw, "alpha", where),
        neighbors_per_seed=get_int(raw, "neighbors_per_seed", where),
        seed_decay=get_bool(raw, "seed_decay", where),
        seed_tau_days=get_float(raw, "seed_tau_days", where),
        quota=get_int(raw, "quota", where),
    )
    require_positive(config.fit_window_days, f"{where}.fit_window_days")
    require_positive(config.seed_window_days, f"{where}.seed_window_days")
    require_positive(config.neighbors_per_seed, f"{where}.neighbors_per_seed")
    require_positive(config.seed_tau_days, f"{where}.seed_tau_days")
    require_positive(config.quota, f"{where}.quota")
    if not 0.0 <= config.alpha <= 1.0:
        raise ValueError(f"{where}.alpha must be in [0, 1], got {config.alpha}")
    if config.seed_window_days > config.fit_window_days:
        raise ValueError(
            f"{where}.seed_window_days ({config.seed_window_days}) must be <= "
            f"fit_window_days ({config.fit_window_days})"
        )
    return config


def load_popularity_config(path: Path) -> PopularityConfig:
    """Read and validate configs/recall/r1_popularity.yaml."""
    return parse_popularity_config(load_yaml(path))


def load_repurchase_config(path: Path) -> RepurchaseConfig:
    """Read and validate configs/recall/r2_repurchase.yaml."""
    return parse_repurchase_config(load_yaml(path))


def load_sibling_config(path: Path) -> SiblingConfig:
    """Read and validate configs/recall/r3_sibling.yaml."""
    return parse_sibling_config(load_yaml(path))


def load_itemcf_config(path: Path) -> ItemCFConfig:
    """Read and validate configs/recall/r4_itemcf.yaml."""
    return parse_itemcf_config(load_yaml(path))


def parse_wals_config(raw: Mapping[str, Any]) -> WalsConfig:
    """Build a WalsConfig from the parsed r5_wals.yaml."""
    where = "r5_wals"
    expected = {
        "variant",
        "window_days",
        "decay_tau_days",
        "alpha_conf",
        "regularization",
        "factors",
        "iterations",
        "random_seed",
        "quota",
    }
    check_keys(raw, expected, where)
    config = WalsConfig(
        variant=_get_choice(raw, "variant", where, WALS_VARIANTS),
        window_days=get_int(raw, "window_days", where),
        decay_tau_days=get_float(raw, "decay_tau_days", where),
        alpha_conf=get_float(raw, "alpha_conf", where),
        regularization=get_float(raw, "regularization", where),
        factors=get_int(raw, "factors", where),
        iterations=get_int(raw, "iterations", where),
        random_seed=get_int(raw, "random_seed", where),
        quota=get_int(raw, "quota", where),
    )
    require_positive(config.window_days, f"{where}.window_days")
    require_positive(config.decay_tau_days, f"{where}.decay_tau_days")
    require_positive(config.alpha_conf, f"{where}.alpha_conf")
    require_positive(config.regularization, f"{where}.regularization")
    require_positive(config.factors, f"{where}.factors")
    require_positive(config.iterations, f"{where}.iterations")
    require_positive(config.quota, f"{where}.quota")
    if config.random_seed < 0:
        raise ValueError(f"{where}.random_seed must be >= 0, got {config.random_seed}")
    return config


def load_wals_config(path: Path) -> WalsConfig:
    """Read and validate configs/recall/r5_wals.yaml."""
    return parse_wals_config(load_yaml(path))


# ---------------------------------------------------------------- twotower.yaml


@dataclass(frozen=True)
class TrainingWindow:
    """A window of Two-Tower training samples: purchase days [end_anchor - window_days, end_anchor).

    Attributes:
        end_anchor: First day AFTER the window (an anchor; it is not in the window).
        window_days: Length of the window in days.
    """

    end_anchor: date
    window_days: int


@dataclass(frozen=True)
class TwoTowerConfig:
    """Settings from configs/twotower.yaml (proposal §5 R6, §6).

    Attributes:
        history_length: Number of past purchases in a user history (20).
        base_window: Training samples of the base model (§6.2).
    """

    history_length: int
    base_window: TrainingWindow


def parse_twotower_config(raw: Mapping[str, Any]) -> TwoTowerConfig:
    """Build a TwoTowerConfig from the parsed content of configs/twotower.yaml.

    Raises:
        ValueError: If a key is missing or unknown, or a number is not positive.
    """
    check_keys(raw, {"history_length", "base_window"}, "twotower")
    raw_window = get_mapping(raw, "base_window", "twotower")
    where = "twotower.base_window"
    check_keys(raw_window, {"end_anchor", "window_days"}, where)
    config = TwoTowerConfig(
        history_length=get_int(raw, "history_length", "twotower"),
        base_window=TrainingWindow(
            end_anchor=get_date(raw_window, "end_anchor", where),
            window_days=get_int(raw_window, "window_days", where),
        ),
    )
    require_positive(config.history_length, "twotower.history_length")
    require_positive(config.base_window.window_days, f"{where}.window_days")
    return config


def load_twotower_config(path: Path) -> TwoTowerConfig:
    """Read and validate configs/twotower.yaml."""
    return parse_twotower_config(load_yaml(path))
