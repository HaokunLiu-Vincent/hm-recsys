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
    - load_features_config: configs/features.yaml -> FeaturesConfig
    - load_ranker_config: configs/ranker.yaml -> RankerConfig

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
            article is scored with a matrix product (R5 WALS), and per model
            forward pass when R6 encodes users / articles.
        twotower_score_query_chunk_size: R6 only: queries per chunk of the
            user x active-article score matrix (memory only, same results).
        pair_chunk_size: Number of (customer, article) pairs per chunk when
            computing vector dot products (hmrec.vectors.dot_pairs).
    """

    query_chunk_size: int
    item_chunk_size: int
    dense_score_query_chunk_size: int
    twotower_score_query_chunk_size: int
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
        {
            "query_chunk_size",
            "item_chunk_size",
            "dense_score_query_chunk_size",
            "twotower_score_query_chunk_size",
            "pair_chunk_size",
        },
        "base.chunking",
    )
    chunking = ChunkingConfig(
        query_chunk_size=get_int(raw_chunking, "query_chunk_size", "base.chunking"),
        item_chunk_size=get_int(raw_chunking, "item_chunk_size", "base.chunking"),
        dense_score_query_chunk_size=get_int(
            raw_chunking, "dense_score_query_chunk_size", "base.chunking"
        ),
        twotower_score_query_chunk_size=get_int(
            raw_chunking, "twotower_score_query_chunk_size", "base.chunking"
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
    require_positive(
        config.chunking.twotower_score_query_chunk_size,
        "base.chunking.twotower_score_query_chunk_size",
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
        active_pool_only: If True, only articles in the active pool A(a) can be
            candidates; the filter is applied BEFORE the Top-quota cut, so the
            freed places go to in-pool siblings. Default False (the D-phase
            runs). Switched on once at M4 by the pre-registered rule (§2.4, §5.7).
    """

    style_lookback_days: int
    market_window_days: int
    tau_days: float
    gamma: float
    quota: int
    active_pool_only: bool = False


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
    # active_pool_only is optional (configs recorded before M4 do not have it): absent = False.
    required = {key: value for key, value in raw.items() if key != "active_pool_only"}
    check_keys(required, expected, where)
    active_pool_only = (
        get_bool(raw, "active_pool_only", where) if "active_pool_only" in raw else False
    )
    config = SiblingConfig(
        style_lookback_days=get_int(raw, "style_lookback_days", where),
        market_window_days=get_int(raw, "market_window_days", where),
        tau_days=get_float(raw, "tau_days", where),
        gamma=get_float(raw, "gamma", where),
        quota=get_int(raw, "quota", where),
        active_pool_only=active_pool_only,
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


# ---------------------------------------------------------------- recall/r6_twotower.yaml


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
class TwoTowerBenchmarkConfig:
    """Settings of the 200-step speed benchmark with a placeholder model (proposal §1.4).

    Attributes:
        steps: Training steps per device.
        warmup_steps: First steps reported separately (MPS compiles kernels then).
        batch_size: Samples per step.
        embedding_dim: Output width of both towers.
        learning_rate: Adam learning rate.
        temperature: Softmax temperature of the in-batch loss.
        seed: Seed of the sample order and the weight initialisation.
    """

    steps: int
    warmup_steps: int
    batch_size: int
    embedding_dim: int
    learning_rate: float
    temperature: float
    seed: int


TWOTOWER_DEVICES = ("cpu", "mps")


@dataclass(frozen=True)
class TwoTowerModelConfig:
    """Layer sizes of the two towers (proposal §5 R6). Fixed, not tuned.

    Attributes:
        output_dim: Width of the L2-normalised user and item vectors (128).
        image_projection_dim: Linear(512 -> this) on the CLIP vector.
        text_projection_dim: Linear(384 -> this) on the MiniLM vector.
        article_embedding_dim: article ID embedding width (initialised to 0).
        product_code_embedding_dim: product_code embedding width (initialised to 0).
        category_embedding_dim: Embedding width of a category field with more
            than one_hot_max_cardinality codes.
        one_hot_max_cardinality: A category field with at most this many codes
            (missing code 0 included) is one-hot coded instead of embedded.
        hidden_dim: Width of the hidden ReLU layer of both towers.
        embedding_init_std: Std of the N(0, std^2) init of category embeddings.
    """

    output_dim: int
    image_projection_dim: int
    text_projection_dim: int
    article_embedding_dim: int
    product_code_embedding_dim: int
    category_embedding_dim: int
    one_hot_max_cardinality: int
    hidden_dim: int
    embedding_init_std: float


@dataclass(frozen=True)
class TwoTowerTrainingConfig:
    """Training settings (proposal §5 R6, §6.2). Fixed, not tuned.

    Attributes:
        batch_size: Samples per step (512).
        temperature: tau of the in-batch softmax (logits = u.v / tau - log Q).
        learning_rate: Adam learning rate of the base model; constant (no decay).
        fine_tune_learning_rate: Weekly fine-tune learning rate (0.1 x base, §6.4).
        adam_beta1: Adam beta1.
        adam_beta2: Adam beta2.
        adam_epsilon: Adam epsilon.
        weight_decay: Adam weight decay (0).
        epochs: Base epochs; one checkpoint per epoch (E_base candidates 1..epochs).
        seed: Seed of the weights, the shuffles and the alarm-user split.
        alarm_user_share: Share of users kept out of training; their loss is
            only logged as an alarm (§6.2), never used to choose anything.
        log_every_steps: Progress log interval.
        device: "cpu" or "mps"; chosen once by the device benchmark (decisions.md).
    """

    batch_size: int
    temperature: float
    learning_rate: float
    fine_tune_learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    epochs: int
    seed: int
    alarm_user_share: float
    log_every_steps: int
    device: str


@dataclass(frozen=True)
class TwoTowerChainConfig:
    """Fixed settings of the weekly training chain (proposal §6.4).

    Attributes:
        week_days: Length of one fine-tune week (= a fold's label week, 7 days):
            the new data of the fold with anchor a are purchase days [a, a + week_days).
        replay_window_days: Replay rows come from purchase days [a - this, a)
            (the 4 weeks before the week).
    """

    week_days: int
    replay_window_days: int


@dataclass(frozen=True)
class TwoTowerConfig:
    """Settings from configs/recall/r6_twotower.yaml (proposal §5 R6, §6).

    Attributes:
        history_length: Number of past purchases in a user history (20).
        quota: Candidates per query (Top-100).
        base_epoch: Which base checkpoint (epoch) the channel predicts with;
            the D-phase step "base_epoch" selects it (E_base, §6.3).
        base_window: Training samples of the base model (§6.2).
        e_update: Fine-tune epochs per week in the training chain (§6.4); 0 =
            "frozen": the base checkpoint predicts every week with no update.
        replay: Share of each fine-tune set drawn from the replay window
            (0.5 -> as many old rows as new rows, 1 : 1); 0 = new data only.
        chain_seed: Seed of the fine-tune shuffles and the replay draws.
        chain: Fixed chain settings.
        model: Layer sizes.
        training: Optimiser, loss and device settings.
        benchmark: The pre-training speed benchmark (§1.4).
    """

    history_length: int
    quota: int
    base_epoch: int
    base_window: TrainingWindow
    e_update: int
    replay: float
    chain_seed: int
    chain: TwoTowerChainConfig
    model: TwoTowerModelConfig
    training: TwoTowerTrainingConfig
    benchmark: TwoTowerBenchmarkConfig


BENCHMARK_KEYS = {
    "steps",
    "warmup_steps",
    "batch_size",
    "embedding_dim",
    "learning_rate",
    "temperature",
    "seed",
}


def _parse_twotower_benchmark(raw: Mapping[str, Any]) -> TwoTowerBenchmarkConfig:
    """Parse and check the benchmark section."""
    where = "r6_twotower.benchmark"
    check_keys(raw, BENCHMARK_KEYS, where)
    config = TwoTowerBenchmarkConfig(
        steps=get_int(raw, "steps", where),
        warmup_steps=get_int(raw, "warmup_steps", where),
        batch_size=get_int(raw, "batch_size", where),
        embedding_dim=get_int(raw, "embedding_dim", where),
        learning_rate=get_float(raw, "learning_rate", where),
        temperature=get_float(raw, "temperature", where),
        seed=get_int(raw, "seed", where),
    )
    for key in ("steps", "batch_size", "embedding_dim", "learning_rate", "temperature"):
        require_positive(getattr(config, key), f"{where}.{key}")
    if not 0 <= config.warmup_steps < config.steps:
        raise ValueError(f"{where}.warmup_steps must be in [0, steps), got {config.warmup_steps}")
    return config


MODEL_INT_KEYS = (
    "output_dim",
    "image_projection_dim",
    "text_projection_dim",
    "article_embedding_dim",
    "product_code_embedding_dim",
    "category_embedding_dim",
    "one_hot_max_cardinality",
    "hidden_dim",
)


def _parse_twotower_model(raw: Mapping[str, Any]) -> TwoTowerModelConfig:
    """Parse the model section; every size must be a positive integer."""
    where = "r6_twotower.model"
    check_keys(raw, {*MODEL_INT_KEYS, "embedding_init_std"}, where)
    sizes = {key: get_int(raw, key, where) for key in MODEL_INT_KEYS}
    for key, value in sizes.items():
        require_positive(value, f"{where}.{key}")
    init_std = get_float(raw, "embedding_init_std", where)
    require_positive(init_std, f"{where}.embedding_init_std")
    return TwoTowerModelConfig(
        output_dim=sizes["output_dim"],
        image_projection_dim=sizes["image_projection_dim"],
        text_projection_dim=sizes["text_projection_dim"],
        article_embedding_dim=sizes["article_embedding_dim"],
        product_code_embedding_dim=sizes["product_code_embedding_dim"],
        category_embedding_dim=sizes["category_embedding_dim"],
        one_hot_max_cardinality=sizes["one_hot_max_cardinality"],
        hidden_dim=sizes["hidden_dim"],
        embedding_init_std=init_std,
    )


TRAINING_KEYS = {
    "batch_size",
    "temperature",
    "learning_rate",
    "fine_tune_learning_rate",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "weight_decay",
    "epochs",
    "seed",
    "alarm_user_share",
    "log_every_steps",
    "device",
}


def _check_twotower_training(config: TwoTowerTrainingConfig, where: str) -> None:
    """Range checks of the training section."""
    for key in ("batch_size", "temperature", "learning_rate", "fine_tune_learning_rate"):
        require_positive(getattr(config, key), f"{where}.{key}")
    for key in ("adam_epsilon", "epochs", "log_every_steps"):
        require_positive(getattr(config, key), f"{where}.{key}")
    for key in ("adam_beta1", "adam_beta2", "alarm_user_share"):
        if not 0 <= getattr(config, key) < 1:
            raise ValueError(f"{where}.{key} must be in [0, 1), got {getattr(config, key)}")
    if config.weight_decay < 0:
        raise ValueError(f"{where}.weight_decay must be >= 0, got {config.weight_decay}")
    if config.device not in TWOTOWER_DEVICES:
        raise ValueError(f"{where}.device must be one of {TWOTOWER_DEVICES}, got {config.device!r}")


def _parse_twotower_training(raw: Mapping[str, Any]) -> TwoTowerTrainingConfig:
    """Parse and check the training section."""
    where = "r6_twotower.training"
    check_keys(raw, TRAINING_KEYS, where)
    config = TwoTowerTrainingConfig(
        batch_size=get_int(raw, "batch_size", where),
        temperature=get_float(raw, "temperature", where),
        learning_rate=get_float(raw, "learning_rate", where),
        fine_tune_learning_rate=get_float(raw, "fine_tune_learning_rate", where),
        adam_beta1=get_float(raw, "adam_beta1", where),
        adam_beta2=get_float(raw, "adam_beta2", where),
        adam_epsilon=get_float(raw, "adam_epsilon", where),
        weight_decay=get_float(raw, "weight_decay", where),
        epochs=get_int(raw, "epochs", where),
        seed=get_int(raw, "seed", where),
        alarm_user_share=get_float(raw, "alarm_user_share", where),
        log_every_steps=get_int(raw, "log_every_steps", where),
        device=get_str(raw, "device", where),
    )
    _check_twotower_training(config, where)
    return config


TWOTOWER_KEYS = {
    "history_length",
    "quota",
    "base_epoch",
    "base_window",
    "e_update",
    "replay",
    "chain_seed",
    "chain",
    "model",
    "training",
    "benchmark",
}


def _parse_twotower_chain(raw: Mapping[str, Any]) -> TwoTowerChainConfig:
    """Parse the fixed chain section; both lengths must be positive."""
    where = "r6_twotower.chain"
    check_keys(raw, {"week_days", "replay_window_days"}, where)
    config = TwoTowerChainConfig(
        week_days=get_int(raw, "week_days", where),
        replay_window_days=get_int(raw, "replay_window_days", where),
    )
    require_positive(config.week_days, f"{where}.week_days")
    require_positive(config.replay_window_days, f"{where}.replay_window_days")
    return config


def _check_twotower_chain_values(config: TwoTowerConfig) -> None:
    """e_update >= 0 and replay in [0, 1) (replay 1 would leave no new data)."""
    if config.e_update < 0:
        raise ValueError(f"r6_twotower.e_update must be >= 0, got {config.e_update}")
    if not 0 <= config.replay < 1:
        raise ValueError(f"r6_twotower.replay must be in [0, 1), got {config.replay}")


def parse_twotower_config(raw: Mapping[str, Any]) -> TwoTowerConfig:
    """Build a TwoTowerConfig from the parsed content of configs/recall/r6_twotower.yaml.

    Raises:
        ValueError: If a key is missing or unknown, or a number is out of range.
    """
    where = "r6_twotower"
    check_keys(raw, TWOTOWER_KEYS, where)
    raw_window = get_mapping(raw, "base_window", where)
    check_keys(raw_window, {"end_anchor", "window_days"}, f"{where}.base_window")
    config = TwoTowerConfig(
        history_length=get_int(raw, "history_length", where),
        quota=get_int(raw, "quota", where),
        base_epoch=get_int(raw, "base_epoch", where),
        base_window=TrainingWindow(
            end_anchor=get_date(raw_window, "end_anchor", f"{where}.base_window"),
            window_days=get_int(raw_window, "window_days", f"{where}.base_window"),
        ),
        e_update=get_int(raw, "e_update", where),
        replay=get_float(raw, "replay", where),
        chain_seed=get_int(raw, "chain_seed", where),
        chain=_parse_twotower_chain(get_mapping(raw, "chain", where)),
        model=_parse_twotower_model(get_mapping(raw, "model", where)),
        training=_parse_twotower_training(get_mapping(raw, "training", where)),
        benchmark=_parse_twotower_benchmark(get_mapping(raw, "benchmark", where)),
    )
    for key in ("history_length", "quota", "base_epoch"):
        require_positive(getattr(config, key), f"{where}.{key}")
    require_positive(config.base_window.window_days, f"{where}.base_window.window_days")
    _check_twotower_chain_values(config)
    if config.base_epoch > config.training.epochs:
        raise ValueError(
            f"{where}.base_epoch={config.base_epoch} is larger than training.epochs="
            f"{config.training.epochs} (no such checkpoint)"
        )
    return config


def load_twotower_config(path: Path) -> TwoTowerConfig:
    """Read and validate configs/recall/r6_twotower.yaml."""
    return parse_twotower_config(load_yaml(path))


# ---------------------------------------------------------------- m4_union_check.yaml


@dataclass(frozen=True)
class UnionCheckConfig:
    """Settings from configs/m4_union_check.yaml (proposal §5.7).

    Attributes:
        union_recall_target: G2, the target union recall of the six channels (0.24).
        r3_max_union_recall_drop: Pre-registered R3 rule: if removing R3's
            candidates outside A(a) lowers the pooled union recall by less than
            this (absolute), R3 becomes "A(a) only".
        manual_selection: Channel -> config keys that replace the rule's
            D-phase selection (each has a docs/decisions.md record).
        variants: Variant name -> {channel -> config keys}: further changes on
            top of the selections, checked with `--variant NAME` and written
            to their own results file (e.g. R3 limited to A(a) after the rule).
    """

    union_recall_target: float
    r3_max_union_recall_drop: float
    manual_selection: dict[str, dict[str, Any]]
    variants: dict[str, dict[str, dict[str, Any]]]


def _channel_overrides(raw: Any, where: str) -> dict[str, dict[str, Any]]:
    """Parse {channel: {config key: value}}; every channel needs at least one key."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a mapping of channel -> config keys, got {raw!r}")
    overrides: dict[str, dict[str, Any]] = {}
    for channel, keys in raw.items():
        if not isinstance(keys, Mapping) or not keys:
            raise ValueError(f"{where}.{channel} must be a non-empty mapping")
        overrides[str(channel)] = dict(keys)
    return overrides


def parse_union_check_config(raw: Mapping[str, Any]) -> UnionCheckConfig:
    """Build a UnionCheckConfig from the parsed content of configs/m4_union_check.yaml.

    Raises:
        ValueError: If a key is missing or unknown, a number is out of (0, 1),
            or a manual_selection entry is not a non-empty mapping.
    """
    where = "m4_union_check"
    expected = {"union_recall_target", "r3_active_pool_rule", "manual_selection", "variants"}
    check_keys(raw, expected, where)
    raw_rule = get_mapping(raw, "r3_active_pool_rule", where)
    check_keys(raw_rule, {"max_union_recall_drop"}, f"{where}.r3_active_pool_rule")
    manual = _channel_overrides(
        get_mapping(raw, "manual_selection", where), f"{where}.manual_selection"
    )
    variants = {
        str(name): _channel_overrides(overrides, f"{where}.variants.{name}")
        for name, overrides in get_mapping(raw, "variants", where).items()
    }
    config = UnionCheckConfig(
        union_recall_target=get_float(raw, "union_recall_target", where),
        r3_max_union_recall_drop=get_float(
            raw_rule, "max_union_recall_drop", f"{where}.r3_active_pool_rule"
        ),
        manual_selection=manual,
        variants=variants,
    )
    for name in ("union_recall_target", "r3_max_union_recall_drop"):
        if not 0 < getattr(config, name) < 1:
            raise ValueError(f"{where}: {name} must be in (0, 1), got {getattr(config, name)}")
    return config


def load_union_check_config(path: Path) -> UnionCheckConfig:
    """Read and validate configs/m4_union_check.yaml."""
    return parse_union_check_config(load_yaml(path))


# ---------------------------------------------------------------- m6_candidates.yaml


@dataclass(frozen=True)
class CandidateRunConfig:
    """Settings from configs/m6_candidates.yaml (proposal §6.6; candidate generation, M6).

    Nothing here changes a candidate: these are monitoring and stop settings only.

    Attributes:
        previous_folds: A channel's recall is compared with the mean of this
            many folds before it (4).
        max_relative_recall_drop: A fold is FLAGGED (reported, nothing is
            changed) if its recall is more than this share below that mean (0.10).
        max_fine_tune_loss: The R6 chain stops for a human if a weekly
            fine-tune's training loss or alarm-user loss is not finite or above this.
    """

    previous_folds: int
    max_relative_recall_drop: float
    max_fine_tune_loss: float


def parse_candidate_run_config(raw: Mapping[str, Any]) -> CandidateRunConfig:
    """Build a CandidateRunConfig from the parsed content of configs/m6_candidates.yaml.

    Raises:
        ValueError: If a key is missing or unknown, or a number is out of range.
    """
    where = "m6_candidates"
    check_keys(raw, {"monitoring", "twotower_chain"}, where)
    monitoring = get_mapping(raw, "monitoring", where)
    check_keys(monitoring, {"previous_folds", "max_relative_recall_drop"}, f"{where}.monitoring")
    chain = get_mapping(raw, "twotower_chain", where)
    check_keys(chain, {"max_fine_tune_loss"}, f"{where}.twotower_chain")
    config = CandidateRunConfig(
        previous_folds=get_int(monitoring, "previous_folds", f"{where}.monitoring"),
        max_relative_recall_drop=get_float(
            monitoring, "max_relative_recall_drop", f"{where}.monitoring"
        ),
        max_fine_tune_loss=get_float(chain, "max_fine_tune_loss", f"{where}.twotower_chain"),
    )
    require_positive(config.previous_folds, f"{where}.monitoring.previous_folds")
    require_positive(config.max_fine_tune_loss, f"{where}.twotower_chain.max_fine_tune_loss")
    if not 0 < config.max_relative_recall_drop < 1:
        raise ValueError(
            f"{where}.monitoring.max_relative_recall_drop must be in (0, 1), "
            f"got {config.max_relative_recall_drop}"
        )
    return config


def load_candidate_run_config(path: Path) -> CandidateRunConfig:
    """Read and validate configs/m6_candidates.yaml."""
    return parse_candidate_run_config(load_yaml(path))


# ---------------------------------------------------------------- features.yaml

FEATURE_DTYPES = ("float32", "int16")
# The 71 features of proposal §8.2-§8.6, by group and in the proposal's order
# (A 14, B 7, C 17, D 12, E 21). configs/features.yaml must list exactly these
# names in exactly this order.
PROPOSAL_FEATURES: dict[str, tuple[str, ...]] = {
    "A": (
        "pop_rank",
        "rep_rank",
        "sib_rank",
        "icf_rank",
        "wals_rank",
        "tt_rank",
        "pop_score",
        "rep_score",
        "sib_score",
        "icf_score",
        "pop_is_fallback",
        "n_channels",
        "best_rank",
        "rrf_score",
    ),
    "B": (
        "tt_dot",
        "wals_dot",
        "pop_bucket_score",
        "img_sim_hist",
        "txt_sim_hist",
        "tt_sim_max_hist",
        "tt_sim_top5_hist",
    ),
    "C": (
        "item_cnt_1w",
        "item_cnt_prev_1w",
        "item_cnt_4w",
        "item_uniq_buyers_4w",
        "item_days_since_first_sale",
        "item_days_since_last_sale",
        "item_price_mean_4w",
        "item_online_share_4w",
        "item_buyer_age_mean_4w",
        "item_repurchase_ratio_52w",
        "item_discount_ratio",
        "product_type_no",
        "product_group_name",
        "garment_group_no",
        "index_group_no",
        "section_no",
        "colour_group_code",
    ),
    "D": (
        "age",
        "club_member_status",
        "fashion_news_frequency",
        "fn",
        "active",
        "user_cnt_4w",
        "user_cnt_52w",
        "user_days_since_last_purchase",
        "user_price_mean_52w",
        "user_online_share_52w",
        "user_uniq_items_52w",
        "user_purchase_days_52w",
    ),
    "E": (
        "ui_cnt_item",
        "ui_days_since_item",
        "ui_cnt_pcode",
        "ui_days_since_pcode",
        "ui_share_ptype_52w",
        "ui_share_garment_52w",
        "ui_share_index_52w",
        "ui_share_colour_52w",
        "ui_price_ratio",
        "ui_age_gap",
        "ui_decay_ptype",
        "ui_decay_garment",
        "ui_decay_section",
        "ui_decay_colour",
        "ui_days_since_ptype",
        "ui_days_since_section",
        "ui_share_section_52w",
        "ui_share_pgroup_52w",
        "ui_lift_ptype_52w",
        "ui_channel_gap",
        "ui_attr_match_cnt",
    ),
}
N_PROPOSAL_FEATURES = sum(len(names) for names in PROPOSAL_FEATURES.values())
# Category fields (LightGBM categorical features, §8.1): the six article fields of
# §8.4 and the two customer fields of §8.5.
PROPOSAL_CATEGORICAL_FEATURES = (
    "product_type_no",
    "product_group_name",
    "garment_group_no",
    "index_group_no",
    "section_no",
    "colour_group_code",
    "club_member_status",
    "fashion_news_frequency",
)


@dataclass(frozen=True)
class FeatureSpec:
    """One column of the wide table.

    Attributes:
        name: Feature name, as in proposal §8.
        group: "A" ... "E" (proposal §8.2-§8.6).
        dtype: "float32" (a number, missing = NaN) or "int16" (a category code,
            0 = missing).
        categorical: True if LightGBM treats it as a categorical feature.
    """

    name: str
    group: str
    dtype: str
    categorical: bool


@dataclass(frozen=True)
class FeaturesConfig:
    """Settings from configs/features.yaml (proposal §8).

    Attributes:
        history_length: Past purchases used by the *_hist features (20).
        history_similarity_top_k: tt_sim_top5_hist averages this many of the
            largest similarities (5).
        rrf_k: The constant of rrf_score = sum of 1 / (rrf_k + rank) (60).
        decay_tau_days: Time constant of the ui_decay_* features (30; fixed, not tuned).
        online_sales_channel_id: sales_channel_id of the online channel (2).
        week_days: Length of the 1w and prev_1w windows (7).
        four_week_days: Length of the 4w window (28).
        year_days: Length of the 52w window (364).
        embedding_pair_chunk_size: Pairs per chunk for the image / text cosine.
        features: The 71 features in wide-table column order.
    """

    history_length: int
    history_similarity_top_k: int
    rrf_k: int
    decay_tau_days: float
    online_sales_channel_id: int
    week_days: int
    four_week_days: int
    year_days: int
    embedding_pair_chunk_size: int
    features: tuple[FeatureSpec, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Feature names in wide-table column order."""
        return tuple(feature.name for feature in self.features)

    @property
    def categorical_names(self) -> tuple[str, ...]:
        """Names of the categorical features, in column order."""
        return tuple(feature.name for feature in self.features if feature.categorical)

    def group_names(self, group: str) -> tuple[str, ...]:
        """Names of one group's features, in column order."""
        return tuple(feature.name for feature in self.features if feature.group == group)


def _parse_feature_spec(raw: Any, position: int) -> FeatureSpec:
    """Parse one entry of `features`; a category code must be int16 and vice versa."""
    where = f"features.features[{position}]"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {raw!r}")
    check_keys(raw, {"name", "group", "dtype", "categorical"}, where)
    spec = FeatureSpec(
        name=get_str(raw, "name", where),
        group=get_str(raw, "group", where),
        dtype=get_str(raw, "dtype", where),
        categorical=get_bool(raw, "categorical", where),
    )
    if spec.dtype not in FEATURE_DTYPES:
        raise ValueError(f"{where}.dtype must be one of {FEATURE_DTYPES}, got {spec.dtype!r}")
    if spec.categorical != (spec.dtype == "int16"):
        raise ValueError(
            f"{where} ({spec.name}): categorical features are int16 codes and numeric "
            f"features are float32; got dtype {spec.dtype}, categorical {spec.categorical}"
        )
    return spec


def _check_features_match_proposal(features: tuple[FeatureSpec, ...]) -> None:
    """The feature list must be exactly proposal §8: same names, groups, order, categories."""
    listed = [(feature.group, feature.name) for feature in features]
    expected = [(group, name) for group, names in PROPOSAL_FEATURES.items() for name in names]
    if listed != expected:
        missing = sorted(set(expected) - set(listed))
        extra = sorted(set(listed) - set(expected))
        raise ValueError(
            f"configs/features.yaml must list exactly the {N_PROPOSAL_FEATURES} features of "
            f"proposal §8 in the proposal's order; missing {missing}, not in the proposal {extra}"
        )
    categorical = tuple(feature.name for feature in features if feature.categorical)
    if categorical != PROPOSAL_CATEGORICAL_FEATURES:
        raise ValueError(
            f"categorical features must be {PROPOSAL_CATEGORICAL_FEATURES}, got {categorical}"
        )


def parse_features_config(raw: Mapping[str, Any]) -> FeaturesConfig:
    """Build a FeaturesConfig from the parsed content of configs/features.yaml.

    Raises:
        ValueError: If a key is missing or unknown, a number is not positive, or
            the feature list differs from proposal §8 in any way.
    """
    where = "features"
    expected = {
        "history_length",
        "history_similarity_top_k",
        "rrf_k",
        "decay_tau_days",
        "online_sales_channel_id",
        "windows",
        "chunking",
        "features",
    }
    check_keys(raw, expected, where)
    windows = get_mapping(raw, "windows", where)
    check_keys(windows, {"week_days", "four_week_days", "year_days"}, f"{where}.windows")
    chunking = get_mapping(raw, "chunking", where)
    check_keys(chunking, {"embedding_pair_chunk_size"}, f"{where}.chunking")
    raw_features = raw["features"]
    if not isinstance(raw_features, list):
        raise ValueError(f"{where}.features: expected a list, got {type(raw_features).__name__}")
    features = tuple(_parse_feature_spec(item, i) for i, item in enumerate(raw_features))
    _check_features_match_proposal(features)
    config = FeaturesConfig(
        history_length=get_int(raw, "history_length", where),
        history_similarity_top_k=get_int(raw, "history_similarity_top_k", where),
        rrf_k=get_int(raw, "rrf_k", where),
        decay_tau_days=get_float(raw, "decay_tau_days", where),
        online_sales_channel_id=get_int(raw, "online_sales_channel_id", where),
        week_days=get_int(windows, "week_days", f"{where}.windows"),
        four_week_days=get_int(windows, "four_week_days", f"{where}.windows"),
        year_days=get_int(windows, "year_days", f"{where}.windows"),
        embedding_pair_chunk_size=get_int(
            chunking, "embedding_pair_chunk_size", f"{where}.chunking"
        ),
        features=features,
    )
    for name in (
        "history_length",
        "history_similarity_top_k",
        "rrf_k",
        "decay_tau_days",
        "week_days",
        "four_week_days",
        "year_days",
        "embedding_pair_chunk_size",
    ):
        require_positive(getattr(config, name), f"{where}.{name}")
    return config


def load_features_config(path: Path) -> FeaturesConfig:
    """Read and validate configs/features.yaml."""
    return parse_features_config(load_yaml(path))


# ---------------------------------------------------------------- ranker.yaml

HISTOGRAM_UNDECIDED = "undecided"
HISTOGRAM_COL_WISE = "col_wise"
HISTOGRAM_ROW_WISE = "row_wise"
HISTOGRAM_MODES = (HISTOGRAM_UNDECIDED, HISTOGRAM_COL_WISE, HISTOGRAM_ROW_WISE)

# The exact keys of dataset_params and train_params in configs/ranker.yaml. A key that
# is missing or unknown is an error: every LightGBM parameter that can change the
# result is written out, and nothing else is passed.
RANKER_DATASET_PARAM_KEYS = frozenset(
    {
        "max_bin",
        "min_data_in_bin",
        "bin_construct_sample_cnt",
        "use_missing",
        "zero_as_missing",
        "enable_bundle",
        "feature_pre_filter",
    }
)
RANKER_TRAIN_PARAM_KEYS = frozenset(
    {
        "objective",
        "boosting",
        "learning_rate",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "bagging_by_query",
        "feature_fraction_bynode",
        "max_depth",
        "min_sum_hessian_in_leaf",
        "min_gain_to_split",
        "lambda_l1",
        "lambda_l2",
        "lambdarank_truncation_level",
        "lambdarank_norm",
        "max_cat_threshold",
        "cat_l2",
        "cat_smooth",
        "max_cat_to_onehot",
        "min_data_per_group",
        "metric",
        "eval_at",
        "deterministic",
        "verbosity",
    }
)
RANKER_TUNED_KEYS = ("num_leaves", "min_data_in_leaf", "negatives_per_positive")


@dataclass(frozen=True)
class RankerTunedParams:
    """The three values the V phase tunes (proposal §9.4).

    Attributes:
        num_leaves: Leaves per tree.
        min_data_in_leaf: Minimum training rows in a leaf.
        negatives_per_positive: r, sampled negatives per positive (proposal §9.1).
    """

    num_leaves: int
    min_data_in_leaf: int
    negatives_per_positive: int


@dataclass(frozen=True)
class RankerSamplingConfig:
    """The `sampling` section of configs/ranker.yaml (proposal §9.1).

    Attributes:
        seed: Seed of the negative sampling.
        ratios: Negatives per positive of the sample files that are written.
        row_group_rows: Rows per Parquet row group of a sample file.
    """

    seed: int
    ratios: tuple[int, ...]
    row_group_rows: int


@dataclass(frozen=True)
class HistogramConfig:
    """The `histogram` section of configs/ranker.yaml: col-wise or row-wise histograms.

    Attributes:
        mode: "undecided", "col_wise" or "row_wise".
        comparison_trees: Trees each mode trains in the speed test.
        warmup_trees: First trees left out of the seconds-per-tree mean.
        min_row_wise_speedup: row_wise is chosen only if it saves at least this
            share of col_wise's seconds per tree (0.15 = 15%).
    """

    mode: str
    comparison_trees: int
    warmup_trees: int
    min_row_wise_speedup: float


@dataclass(frozen=True)
class SpeedTestConfig:
    """The `speed_test` section of configs/ranker.yaml.

    Attributes:
        target_fold: The fold the speed test predicts ("V1").
        num_trees: Trees of its evaluated model (300).
    """

    target_fold: str
    num_trees: int


@dataclass(frozen=True)
class RankerConfig:
    """Settings from configs/ranker.yaml (proposal §9).

    Attributes:
        window_weeks: Training weeks right before the predicted week (4).
        top_k: Rows kept per query when ranking (12).
        num_trees: Trees of a grid model (1000; 2000 until 2026-11-18).
        checkpoint_every: Evaluate after every this many trees (100).
        train_metric_every: Log the training-set metric every this many trees.
        scoring_query_chunk_size: Queries scored at a time (memory only).
        seed: LightGBM seed.
        sampling: Negative sampling settings.
        tuned_defaults: Default values of the tuned parameters.
        histogram: Histogram mode and the rule that decides it.
        speed_test: What the speed test runs before the grid.
        dataset_params: LightGBM parameters that define the bins.
        train_params: The fixed LightGBM training parameters.
    """

    window_weeks: int
    top_k: int
    num_trees: int
    checkpoint_every: int
    train_metric_every: int
    scoring_query_chunk_size: int
    seed: int
    sampling: RankerSamplingConfig
    tuned_defaults: RankerTunedParams
    histogram: HistogramConfig
    speed_test: SpeedTestConfig
    dataset_params: dict[str, Any]
    train_params: dict[str, Any]


def parse_ranker_tuned_params(raw: Mapping[str, Any], where: str) -> RankerTunedParams:
    """Build RankerTunedParams from a mapping with exactly the three tuned keys."""
    check_keys(raw, set(RANKER_TUNED_KEYS), where)
    tuned = RankerTunedParams(
        num_leaves=get_int(raw, "num_leaves", where),
        min_data_in_leaf=get_int(raw, "min_data_in_leaf", where),
        negatives_per_positive=get_int(raw, "negatives_per_positive", where),
    )
    for name in RANKER_TUNED_KEYS:
        require_positive(getattr(tuned, name), f"{where}.{name}")
    return tuned


def _parse_ranker_sampling(raw: Mapping[str, Any], where: str) -> RankerSamplingConfig:
    """The `sampling` section: seed, ratios (distinct, ascending), row_group_rows."""
    check_keys(raw, {"seed", "ratios", "row_group_rows"}, where)
    ratios = raw["ratios"]
    is_int_list = isinstance(ratios, list) and all(
        isinstance(ratio, int) and not isinstance(ratio, bool) and ratio > 0 for ratio in ratios
    )
    if not is_int_list or not ratios or sorted(set(ratios)) != ratios:
        raise ValueError(
            f"{where}.ratios: expected distinct positive integers in ascending order, "
            f"got {ratios!r}"
        )
    sampling = RankerSamplingConfig(
        seed=get_int(raw, "seed", where),
        ratios=tuple(ratios),
        row_group_rows=get_int(raw, "row_group_rows", where),
    )
    require_positive(sampling.row_group_rows, f"{where}.row_group_rows")
    if sampling.seed < 0:
        raise ValueError(f"{where}.seed must be >= 0, got {sampling.seed}")
    return sampling


def _parse_histogram(raw: Mapping[str, Any], where: str) -> HistogramConfig:
    """The `histogram` section: the mode and the numbers of the decision rule."""
    check_keys(raw, {"mode", "comparison_trees", "warmup_trees", "min_row_wise_speedup"}, where)
    histogram = HistogramConfig(
        mode=_get_choice(raw, "mode", where, HISTOGRAM_MODES),
        comparison_trees=get_int(raw, "comparison_trees", where),
        warmup_trees=get_int(raw, "warmup_trees", where),
        min_row_wise_speedup=get_float(raw, "min_row_wise_speedup", where),
    )
    if not 0 <= histogram.warmup_trees < histogram.comparison_trees:
        raise ValueError(f"{where}: need 0 <= warmup_trees < comparison_trees")
    if not 0 < histogram.min_row_wise_speedup < 1:
        raise ValueError(f"{where}.min_row_wise_speedup must be in (0, 1)")
    return histogram


def parse_lightgbm_params(
    raw: Mapping[str, Any], expected: frozenset[str], where: str
) -> dict[str, Any]:
    """A section of LightGBM parameters with exactly the expected keys and plain values."""
    check_keys(raw, set(expected), where)
    for key, value in raw.items():
        is_scalar = isinstance(value, bool | int | float | str)
        is_int_list = isinstance(value, list) and all(isinstance(item, int) for item in value)
        if not is_scalar and not is_int_list:
            raise ValueError(f"{where}.{key}: expected a number, text, bool or list of ints")
    return dict(raw)


def parse_ranker_config(raw: Mapping[str, Any]) -> RankerConfig:
    """Build a RankerConfig from the parsed content of configs/ranker.yaml.

    Raises:
        ValueError: If a key is missing or unknown, or a value is out of range.
    """
    where = "ranker"
    expected = {
        "window_weeks",
        "top_k",
        "num_trees",
        "checkpoint_every",
        "train_metric_every",
        "scoring_query_chunk_size",
        "seed",
        "sampling",
        "tuned_defaults",
        "histogram",
        "speed_test",
        "dataset_params",
        "train_params",
    }
    check_keys(raw, expected, where)
    speed_test = get_mapping(raw, "speed_test", where)
    check_keys(speed_test, {"target_fold", "num_trees"}, f"{where}.speed_test")
    config = RankerConfig(
        window_weeks=get_int(raw, "window_weeks", where),
        top_k=get_int(raw, "top_k", where),
        num_trees=get_int(raw, "num_trees", where),
        checkpoint_every=get_int(raw, "checkpoint_every", where),
        train_metric_every=get_int(raw, "train_metric_every", where),
        scoring_query_chunk_size=get_int(raw, "scoring_query_chunk_size", where),
        seed=get_int(raw, "seed", where),
        sampling=_parse_ranker_sampling(get_mapping(raw, "sampling", where), f"{where}.sampling"),
        tuned_defaults=parse_ranker_tuned_params(
            get_mapping(raw, "tuned_defaults", where), f"{where}.tuned_defaults"
        ),
        histogram=_parse_histogram(get_mapping(raw, "histogram", where), f"{where}.histogram"),
        speed_test=SpeedTestConfig(
            target_fold=get_str(speed_test, "target_fold", f"{where}.speed_test"),
            num_trees=get_int(speed_test, "num_trees", f"{where}.speed_test"),
        ),
        dataset_params=parse_lightgbm_params(
            get_mapping(raw, "dataset_params", where),
            RANKER_DATASET_PARAM_KEYS,
            f"{where}.dataset_params",
        ),
        train_params=parse_lightgbm_params(
            get_mapping(raw, "train_params", where),
            RANKER_TRAIN_PARAM_KEYS,
            f"{where}.train_params",
        ),
    )
    _validate_ranker_config(config, where)
    return config


def _validate_ranker_config(config: RankerConfig, where: str) -> None:
    """Range checks that involve more than one field."""
    positive = (
        "window_weeks",
        "top_k",
        "num_trees",
        "checkpoint_every",
        "train_metric_every",
        "scoring_query_chunk_size",
    )
    for name in positive:
        require_positive(getattr(config, name), f"{where}.{name}")
    if config.seed < 0:
        raise ValueError(f"{where}.seed must be >= 0, got {config.seed}")
    if config.num_trees % config.checkpoint_every != 0:
        raise ValueError(
            f"{where}.num_trees ({config.num_trees}) must be a multiple of checkpoint_every "
            f"({config.checkpoint_every})"
        )
    speed_trees = config.speed_test.num_trees
    if speed_trees <= 0 or speed_trees % config.checkpoint_every != 0:
        raise ValueError(
            f"{where}.speed_test.num_trees ({speed_trees}) must be a positive multiple of "
            f"checkpoint_every ({config.checkpoint_every})"
        )
    if config.tuned_defaults.negatives_per_positive not in config.sampling.ratios:
        raise ValueError(
            f"{where}.tuned_defaults.negatives_per_positive must be one of sampling.ratios "
            f"{list(config.sampling.ratios)}"
        )


def load_ranker_config(path: Path) -> RankerConfig:
    """Read and validate configs/ranker.yaml."""
    return parse_ranker_config(load_yaml(path))


# ---------------------------------------------------------------- m8_final.yaml

FINAL_PHASE = "T"
# The pre-registered secondary analyses of the final evaluation (reported only).
FINAL_SECONDARY_EXPERIMENTS = ("minus_r6",)


@dataclass(frozen=True)
class FinalGoals:
    """The acceptance criteria of the final evaluation (proposal §0.1, §10).

    Written to configs/m8_final.yaml BEFORE any label of a T fold was read and
    never changed afterwards.

    Attributes:
        g1_min_pooled_map: G1 is met if the pooled (query-weighted) MAP@12 of the
            target folds is >= this (0.032).
        g2_min_union_recall: G2 is met if the union recall of the six channels is
            >= this on EVERY target fold (0.24).
        g3_min_folds_above_baseline: G3 is met if the Ranker's MAP@12 is strictly
            above the RRF baseline's on at least this many target folds (3).
    """

    g1_min_pooled_map: float
    g2_min_union_recall: float
    g3_min_folds_above_baseline: int


@dataclass(frozen=True)
class FinalEvalConfig:
    """Settings from configs/m8_final.yaml: the final evaluation on the T folds (M8).

    Attributes:
        phase: "T".
        target_folds: The folds to predict and report, in time order.
        metric: The main metric, "map@12".
        extra_metrics: Reported next to it, e.g. ("recall@12", "ndcg@12").
        rehearsal_target_fold: The V fold the rehearsal predicts with the M8 code
            path; its result must equal the recorded V-phase result exactly.
        goals: The acceptance criteria G1-G3.
        secondary_experiments: Pre-registered secondary analyses (report only).
    """

    phase: str
    target_folds: tuple[str, ...]
    metric: str
    extra_metrics: tuple[str, ...]
    rehearsal_target_fold: str
    goals: FinalGoals
    secondary_experiments: tuple[str, ...]

    @property
    def metrics(self) -> tuple[str, ...]:
        """All metrics, the main metric first."""
        return (self.metric, *self.extra_metrics)


def _parse_final_goals(raw: Mapping[str, Any], n_folds: int, where: str) -> FinalGoals:
    """The `goals` section; thresholds must be shares in (0, 1) and a fold count in 1..n."""
    keys = {"g1_min_pooled_map", "g2_min_union_recall", "g3_min_folds_above_baseline"}
    check_keys(raw, keys, where)
    goals = FinalGoals(
        g1_min_pooled_map=get_float(raw, "g1_min_pooled_map", where),
        g2_min_union_recall=get_float(raw, "g2_min_union_recall", where),
        g3_min_folds_above_baseline=get_int(raw, "g3_min_folds_above_baseline", where),
    )
    for name in ("g1_min_pooled_map", "g2_min_union_recall"):
        if not 0 < getattr(goals, name) < 1:
            raise ValueError(f"{where}.{name} must be in (0, 1), got {getattr(goals, name)}")
    if not 1 <= goals.g3_min_folds_above_baseline <= n_folds:
        raise ValueError(
            f"{where}.g3_min_folds_above_baseline must be in 1..{n_folds}, "
            f"got {goals.g3_min_folds_above_baseline}"
        )
    return goals


def parse_final_eval_config(raw: Mapping[str, Any]) -> FinalEvalConfig:
    """Build a FinalEvalConfig from the parsed content of configs/m8_final.yaml.

    Raises:
        ValueError: If a key is missing or unknown, the phase is not "T", the
            folds repeat, a threshold is out of range or a secondary experiment
            is not a pre-registered one.
    """
    where = "m8_final"
    keys = {
        "phase",
        "target_folds",
        "metric",
        "extra_metrics",
        "rehearsal",
        "goals",
        "secondary_experiments",
    }
    check_keys(raw, keys, where)
    rehearsal = get_mapping(raw, "rehearsal", where)
    check_keys(rehearsal, {"target_fold"}, f"{where}.rehearsal")
    target_folds = get_str_list(raw, "target_folds", where)
    if not target_folds or len(set(target_folds)) != len(target_folds):
        raise ValueError(f"{where}.target_folds must be distinct folds, got {target_folds}")
    config = FinalEvalConfig(
        phase=get_str(raw, "phase", where),
        target_folds=target_folds,
        metric=get_str(raw, "metric", where),
        extra_metrics=get_str_list(raw, "extra_metrics", where),
        rehearsal_target_fold=get_str(rehearsal, "target_fold", f"{where}.rehearsal"),
        goals=_parse_final_goals(
            get_mapping(raw, "goals", where), len(target_folds), f"{where}.goals"
        ),
        secondary_experiments=get_str_list(raw, "secondary_experiments", where),
    )
    if config.phase != FINAL_PHASE:
        raise ValueError(f"{where}.phase must be {FINAL_PHASE!r}, got {config.phase!r}")
    if len(set(config.metrics)) != len(config.metrics):
        raise ValueError(f"{where}: metrics repeat: {config.metrics}")
    unknown = sorted(set(config.secondary_experiments) - set(FINAL_SECONDARY_EXPERIMENTS))
    if unknown or len(set(config.secondary_experiments)) != len(config.secondary_experiments):
        raise ValueError(
            f"{where}.secondary_experiments must be distinct names from "
            f"{FINAL_SECONDARY_EXPERIMENTS}, got {config.secondary_experiments}"
        )
    return config


def load_final_eval_config(path: Path) -> FinalEvalConfig:
    """Read and validate configs/m8_final.yaml."""
    return parse_final_eval_config(load_yaml(path))


# ---------------------------------------------------------------- m8_diagnostics.yaml

# The phases whose folds have recorded Ranker / baseline / union-recall numbers
# (results/v_phase*.csv and results/m6_candidates.csv; results/t_phase*.csv).
DIAGNOSTIC_RESULT_PHASES = ("V", "T")
DIAGNOSTIC_LABEL_PHASES = ("D", "F", "V", "T")


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Settings from configs/m8_diagnostics.yaml: descriptive diagnostics after M8.

    Nothing here selects or tunes anything; the values only say what is described.

    Attributes:
        decomposition_phases: Phases of table A (one row per fold).
        concentration_phases: Phases of table B (one row per fold).
        query_group_phases: Phases of table C (one row per fold and query group).
        bootstrap_phases: Phases of table D (the folds of a phase are pooled).
        top_article_counts: Table B: the N of "the N most bought articles", ascending.
        new_article_days: Table B: an article is new if its first sale is in
            [anchor - new_article_days, anchor).
        recent_purchase_days: Table C: a customer bought recently if the last
            purchase is at most this many days before the anchor.
        bootstrap_replicates: Table D: number of bootstrap replicates.
        bootstrap_seed: Table D: seed of the random draws.
        confidence_level: Table D: e.g. 0.95 for the 2.5% and 97.5% quantiles.
    """

    decomposition_phases: tuple[str, ...]
    concentration_phases: tuple[str, ...]
    query_group_phases: tuple[str, ...]
    bootstrap_phases: tuple[str, ...]
    top_article_counts: tuple[int, ...]
    new_article_days: int
    recent_purchase_days: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float


def _get_phases(
    raw: Mapping[str, Any], key: str, where: str, allowed: tuple[str, ...]
) -> tuple[str, ...]:
    """A non-empty list of distinct phases, each one of `allowed`."""
    phases = get_str_list(raw, key, where)
    if not phases or len(set(phases)) != len(phases) or not set(phases) <= set(allowed):
        raise ValueError(f"{where}.{key} must be distinct phases from {allowed}, got {phases}")
    return phases


def _get_top_article_counts(raw: Mapping[str, Any], where: str) -> tuple[int, ...]:
    """`top_article_counts`: a non-empty, strictly ascending list of positive integers."""
    value = raw["top_article_counts"]
    is_int_list = isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    )
    if not is_int_list or not value or value[0] <= 0 or sorted(set(value)) != value:
        raise ValueError(
            f"{where}.top_article_counts must be ascending positive integers, got {value!r}"
        )
    return tuple(value)


def parse_diagnostics_config(raw: Mapping[str, Any]) -> DiagnosticsConfig:
    """Build a DiagnosticsConfig from the parsed content of configs/m8_diagnostics.yaml.

    Raises:
        ValueError: If a key is missing or unknown, a phase list is empty, repeats
            or names a phase without the needed records, or a number is out of range.
    """
    where = "m8_diagnostics"
    keys = {
        "decomposition_phases",
        "concentration_phases",
        "concentration",
        "query_group_phases",
        "query_groups",
        "bootstrap_phases",
        "bootstrap",
    }
    check_keys(raw, keys, where)
    concentration = get_mapping(raw, "concentration", where)
    check_keys(concentration, {"top_article_counts", "new_article_days"}, f"{where}.concentration")
    query_groups = get_mapping(raw, "query_groups", where)
    check_keys(query_groups, {"recent_purchase_days"}, f"{where}.query_groups")
    bootstrap = get_mapping(raw, "bootstrap", where)
    check_keys(bootstrap, {"n_replicates", "seed", "confidence_level"}, f"{where}.bootstrap")
    result_phases = DIAGNOSTIC_RESULT_PHASES
    config = DiagnosticsConfig(
        decomposition_phases=_get_phases(raw, "decomposition_phases", where, result_phases),
        concentration_phases=_get_phases(
            raw, "concentration_phases", where, DIAGNOSTIC_LABEL_PHASES
        ),
        query_group_phases=_get_phases(raw, "query_group_phases", where, result_phases),
        bootstrap_phases=_get_phases(raw, "bootstrap_phases", where, result_phases),
        top_article_counts=_get_top_article_counts(concentration, f"{where}.concentration"),
        new_article_days=get_int(concentration, "new_article_days", f"{where}.concentration"),
        recent_purchase_days=get_int(query_groups, "recent_purchase_days", f"{where}.query_groups"),
        bootstrap_replicates=get_int(bootstrap, "n_replicates", f"{where}.bootstrap"),
        bootstrap_seed=get_int(bootstrap, "seed", f"{where}.bootstrap"),
        confidence_level=get_float(bootstrap, "confidence_level", f"{where}.bootstrap"),
    )
    require_positive(config.new_article_days, f"{where}.concentration.new_article_days")
    require_positive(config.recent_purchase_days, f"{where}.query_groups.recent_purchase_days")
    require_positive(config.bootstrap_replicates, f"{where}.bootstrap.n_replicates")
    if config.bootstrap_seed < 0:
        raise ValueError(f"{where}.bootstrap.seed must be >= 0, got {config.bootstrap_seed}")
    if not 0 < config.confidence_level < 1:
        raise ValueError(
            f"{where}.bootstrap.confidence_level must be in (0, 1), got {config.confidence_level}"
        )
    return config


def load_diagnostics_config(path: Path) -> DiagnosticsConfig:
    """Read and validate configs/m8_diagnostics.yaml."""
    return parse_diagnostics_config(load_yaml(path))


# ---------------------------------------------------------------- deepfm.yaml

DEEPFM_COMPONENTS = ("first_order", "fm", "deep", "full")
DEEPFM_NUMERIC_INPUTS = ("binned", "standardised")
DEEPFM_LOSSES = ("pointwise_bce", "listwise_softmax")
DEEPFM_LR_COSINE = "cosine"
DEEPFM_LR_CONSTANT = "constant"
DEEPFM_LR_SCHEDULES = (DEEPFM_LR_COSINE, DEEPFM_LR_CONSTANT)
DEEPFM_UNDECIDED_DEVICE = "undecided"
DEEPFM_DEVICES = ("cpu", "mps", DEEPFM_UNDECIDED_DEVICE)


@dataclass(frozen=True)
class DeepFMModelConfig:
    """The `model` section of configs/deepfm.yaml (proposal §16.2, §16.3).

    Attributes:
        embedding_dim: Length of every embedding (shared by the FM and the deep part).
        hidden_units: Units of the MLP's hidden layers, e.g. (256, 128, 64).
        dropout: Dropout probability after every hidden layer, in [0, 1).
        components: "first_order", "fm", "deep" or "full".
        numeric_input: "binned" or "standardised".
        embedding_init_std: Standard deviation of the embeddings' initial values.
    """

    embedding_dim: int
    hidden_units: tuple[int, ...]
    dropout: float
    components: str
    numeric_input: str
    embedding_init_std: float


@dataclass(frozen=True)
class DeepFMTrainingConfig:
    """The `training` section of configs/deepfm.yaml.

    Attributes:
        loss: "pointwise_bce" or "listwise_softmax".
        learning_rate: Adam learning rate (with the cosine schedule: of the first step).
        lr_schedule: "cosine" (the rate falls to learning_rate * min_lr_ratio over
            all steps of the run) or "constant".
        min_lr_ratio: Lowest rate of the cosine schedule as a share of learning_rate.
        batch_rows: Rows per batch (listwise: at most this many rows of whole queries).
        weight_decay: Adam weight decay (0 = none).
        epochs: Passes over the training rows = the length of the schedule. The
            model is evaluated once, after the last epoch.
        seed: Fixes the initial weights, the batch order and the dropout.
        device: "cpu", "mps", or "undecided" until the device benchmark has run.
    """

    loss: str
    learning_rate: float
    lr_schedule: str
    min_lr_ratio: float
    batch_rows: int
    weight_decay: float
    epochs: int
    seed: int
    device: str


@dataclass(frozen=True)
class DeepFMSpeedTestConfig:
    """The `speed_test` section of configs/deepfm.yaml (proposal §16.6).

    Attributes:
        target_fold: The V fold of the speed test's training.
        benchmark_steps: Training steps of the device benchmark.
        benchmark_discard_steps: First steps that are not timed.
        benchmark_scoring_rows: Rows of the scoring throughput measurement.
        epochs: Epochs of the speed test's training.
    """

    target_fold: str
    benchmark_steps: int
    benchmark_discard_steps: int
    benchmark_scoring_rows: int
    epochs: int


@dataclass(frozen=True)
class DeepFMConfig:
    """Settings from configs/deepfm.yaml: DeepFM as an alternative ranker (proposal §16).

    Attributes:
        experiment: "full" (the sample files and features of all six channels).
        negatives_per_positive: Ratio of the M7 sample files that are reused.
        window_weeks: Training folds before the target week.
        top_k: Candidates kept per query.
        metric: The main metric, "map@12".
        extra_metrics: Reported next to it.
        model: The model settings.
        n_bins: Quantile bins of a numeric feature (missing values get one more).
        clip_low_percentile: Lower clipping percentile of the standardised input.
        clip_high_percentile: Upper clipping percentile of the standardised input.
        training: The training settings.
        scoring_rows_per_chunk: Candidate rows per forward pass when scoring.
        scoring_rows_per_block: Rows per block inside a forward pass, for building
            the field embeddings and the second-order term (memory only).
        cached_numeric_inputs: The numeric-input modes whose encoded inputs are
            cached on disk (speed only).
        min_mps_speedup: MPS is used only if it is at least this many times faster.
        speed_test: The speed test settings.
    """

    experiment: str
    negatives_per_positive: int
    window_weeks: int
    top_k: int
    metric: str
    extra_metrics: tuple[str, ...]
    model: DeepFMModelConfig
    n_bins: int
    clip_low_percentile: float
    clip_high_percentile: float
    training: DeepFMTrainingConfig
    scoring_rows_per_chunk: int
    scoring_rows_per_block: int
    cached_numeric_inputs: tuple[str, ...]
    min_mps_speedup: float
    speed_test: DeepFMSpeedTestConfig

    @property
    def metrics(self) -> tuple[str, ...]:
        """All metrics, the main metric first."""
        return (self.metric, *self.extra_metrics)


def _require_choice(value: str, choices: tuple[str, ...], name: str) -> None:
    """Fail unless `value` is one of `choices`."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def _get_positive_int_list(raw: Mapping[str, Any], key: str, where: str) -> tuple[int, ...]:
    """Return raw[key] as a non-empty tuple of positive integers."""
    value = raw[key]
    is_int_list = isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in value
    )
    if not is_int_list or not value:
        raise ValueError(f"{where}.{key}: expected a list of positive integers, got {value!r}")
    return tuple(value)


def _parse_deepfm_model(raw: Mapping[str, Any], where: str) -> DeepFMModelConfig:
    """The `model` section of configs/deepfm.yaml."""
    keys = {
        "embedding_dim",
        "hidden_units",
        "dropout",
        "components",
        "numeric_input",
        "embedding_init_std",
    }
    check_keys(raw, keys, where)
    model = DeepFMModelConfig(
        embedding_dim=get_int(raw, "embedding_dim", where),
        hidden_units=_get_positive_int_list(raw, "hidden_units", where),
        dropout=get_float(raw, "dropout", where),
        components=get_str(raw, "components", where),
        numeric_input=get_str(raw, "numeric_input", where),
        embedding_init_std=get_float(raw, "embedding_init_std", where),
    )
    require_positive(model.embedding_dim, f"{where}.embedding_dim")
    require_positive(model.embedding_init_std, f"{where}.embedding_init_std")
    if not 0 <= model.dropout < 1:
        raise ValueError(f"{where}.dropout must be in [0, 1), got {model.dropout}")
    _require_choice(model.components, DEEPFM_COMPONENTS, f"{where}.components")
    _require_choice(model.numeric_input, DEEPFM_NUMERIC_INPUTS, f"{where}.numeric_input")
    return model


def _parse_deepfm_training(raw: Mapping[str, Any], where: str) -> DeepFMTrainingConfig:
    """The `training` section of configs/deepfm.yaml."""
    keys = {
        "loss",
        "learning_rate",
        "lr_schedule",
        "min_lr_ratio",
        "batch_rows",
        "weight_decay",
        "epochs",
        "seed",
        "device",
    }
    check_keys(raw, keys, where)
    training = DeepFMTrainingConfig(
        loss=get_str(raw, "loss", where),
        learning_rate=get_float(raw, "learning_rate", where),
        lr_schedule=get_str(raw, "lr_schedule", where),
        min_lr_ratio=get_float(raw, "min_lr_ratio", where),
        batch_rows=get_int(raw, "batch_rows", where),
        weight_decay=get_float(raw, "weight_decay", where),
        epochs=get_int(raw, "epochs", where),
        seed=get_int(raw, "seed", where),
        device=get_str(raw, "device", where),
    )
    for name in ("learning_rate", "batch_rows", "epochs"):
        require_positive(getattr(training, name), f"{where}.{name}")
    if training.weight_decay < 0:
        raise ValueError(f"{where}.weight_decay must be >= 0, got {training.weight_decay}")
    if not 0 < training.min_lr_ratio <= 1:
        raise ValueError(f"{where}.min_lr_ratio must be in (0, 1], got {training.min_lr_ratio}")
    _require_choice(training.loss, DEEPFM_LOSSES, f"{where}.loss")
    _require_choice(training.lr_schedule, DEEPFM_LR_SCHEDULES, f"{where}.lr_schedule")
    _require_choice(training.device, DEEPFM_DEVICES, f"{where}.device")
    return training


def _parse_deepfm_speed_test(raw: Mapping[str, Any], where: str) -> DeepFMSpeedTestConfig:
    """The `speed_test` section of configs/deepfm.yaml."""
    keys = {
        "target_fold",
        "benchmark_steps",
        "benchmark_discard_steps",
        "benchmark_scoring_rows",
        "epochs",
    }
    check_keys(raw, keys, where)
    speed_test = DeepFMSpeedTestConfig(
        target_fold=get_str(raw, "target_fold", where),
        benchmark_steps=get_int(raw, "benchmark_steps", where),
        benchmark_discard_steps=get_int(raw, "benchmark_discard_steps", where),
        benchmark_scoring_rows=get_int(raw, "benchmark_scoring_rows", where),
        epochs=get_int(raw, "epochs", where),
    )
    for name in ("benchmark_steps", "benchmark_scoring_rows", "epochs"):
        require_positive(getattr(speed_test, name), f"{where}.{name}")
    if not 0 <= speed_test.benchmark_discard_steps < speed_test.benchmark_steps:
        raise ValueError(f"{where}.benchmark_discard_steps must be in 0..benchmark_steps - 1")
    return speed_test


def parse_deepfm_config(raw: Mapping[str, Any]) -> DeepFMConfig:
    """Build a DeepFMConfig from the parsed content of configs/deepfm.yaml.

    Raises:
        ValueError: If a key is missing or unknown, or a value is out of range.
    """
    where = "deepfm"
    keys = {
        "experiment",
        "negatives_per_positive",
        "window_weeks",
        "top_k",
        "metric",
        "extra_metrics",
        "model",
        "binning",
        "standardise",
        "training",
        "scoring",
        "input_cache",
        "device_rule",
        "speed_test",
    }
    check_keys(raw, keys, where)
    input_cache = get_mapping(raw, "input_cache", where)
    check_keys(input_cache, {"numeric_inputs"}, f"{where}.input_cache")
    binning = get_mapping(raw, "binning", where)
    check_keys(binning, {"n_bins"}, f"{where}.binning")
    standardise = get_mapping(raw, "standardise", where)
    check_keys(standardise, {"clip_low_percentile", "clip_high_percentile"}, f"{where}.standardise")
    scoring = get_mapping(raw, "scoring", where)
    check_keys(scoring, {"rows_per_chunk", "rows_per_block"}, f"{where}.scoring")
    device_rule = get_mapping(raw, "device_rule", where)
    check_keys(device_rule, {"min_mps_speedup"}, f"{where}.device_rule")
    config = DeepFMConfig(
        experiment=get_str(raw, "experiment", where),
        negatives_per_positive=get_int(raw, "negatives_per_positive", where),
        window_weeks=get_int(raw, "window_weeks", where),
        top_k=get_int(raw, "top_k", where),
        metric=get_str(raw, "metric", where),
        extra_metrics=get_str_list(raw, "extra_metrics", where),
        model=_parse_deepfm_model(get_mapping(raw, "model", where), f"{where}.model"),
        n_bins=get_int(binning, "n_bins", f"{where}.binning"),
        clip_low_percentile=get_float(standardise, "clip_low_percentile", f"{where}.standardise"),
        clip_high_percentile=get_float(standardise, "clip_high_percentile", f"{where}.standardise"),
        training=_parse_deepfm_training(get_mapping(raw, "training", where), f"{where}.training"),
        scoring_rows_per_chunk=get_int(scoring, "rows_per_chunk", f"{where}.scoring"),
        scoring_rows_per_block=get_int(scoring, "rows_per_block", f"{where}.scoring"),
        cached_numeric_inputs=get_str_list(input_cache, "numeric_inputs", f"{where}.input_cache"),
        min_mps_speedup=get_float(device_rule, "min_mps_speedup", f"{where}.device_rule"),
        speed_test=_parse_deepfm_speed_test(
            get_mapping(raw, "speed_test", where), f"{where}.speed_test"
        ),
    )
    positive = ("negatives_per_positive", "window_weeks", "top_k")
    for name in (*positive, "scoring_rows_per_chunk", "scoring_rows_per_block"):
        require_positive(getattr(config, name), f"{where}.{name}")
    # Bin 0 is the missing bin and a binned value is stored as uint8: 1 + n_bins values.
    if not 2 <= config.n_bins <= 255:
        raise ValueError(f"{where}.binning.n_bins must be in 2..255, got {config.n_bins}")
    if not 0 <= config.clip_low_percentile < config.clip_high_percentile <= 100:
        raise ValueError(f"{where}.standardise: need 0 <= low < high <= 100")
    if config.min_mps_speedup < 1:
        raise ValueError(f"{where}.device_rule.min_mps_speedup must be >= 1")
    for name in config.cached_numeric_inputs:
        _require_choice(name, DEEPFM_NUMERIC_INPUTS, f"{where}.input_cache.numeric_inputs")
    if len(set(config.metrics)) != len(config.metrics):
        raise ValueError(f"{where}: metrics repeat: {config.metrics}")
    return config


def load_deepfm_config(path: Path) -> DeepFMConfig:
    """Read and validate configs/deepfm.yaml."""
    return parse_deepfm_config(load_yaml(path))


# ---------------------------------------------------------------- hpo/deepfm.yaml


@dataclass(frozen=True)
class DeepFMStudyStep:
    """One tuning step of the DeepFM study: one parameter and its candidate values.

    Attributes:
        name: Step name.
        parameter: "learning_rate", "embedding_dim" or "dropout".
        candidates: The values in priority order; the first is the current value.
    """

    name: str
    parameter: str
    candidates: tuple[float, ...]


@dataclass(frozen=True)
class DeepFMStudyConfig:
    """Settings from configs/hpo/deepfm.yaml: the rules of the M9 study (proposal §16.8).

    Attributes:
        phase: "V".
        target_folds: The folds every configuration is trained for and evaluated on.
        metric: The selection metric, pooled over the folds.
        schedule_epochs: The schedule lengths T (epochs of a run), ascending.
        epoch_tolerance: Relative tolerance of the epoch rule (and of the
            number-of-trees rule of the binary LightGBM arm).
        noise_seeds: Seeds of step 0.
        max_seed_range: A step-0 seed range of at least this much makes every
            tuning candidate run with all noise seeds (there is no stop).
        selection_seed: The seed of tuning when the seed range is below the limit,
            and of the ablations.
        steps: The tuning steps of the pointwise arm in order; the first is the
            learning rate.
        listwise_learning_rates: Learning-rate candidates of the listwise arm, the
            current value first.
        ablation_components: Component variants run at the selected configuration.
        ablation_numeric_inputs: Numeric-input variants run there.
        final_seeds: Seeds of each arm's selected configuration in the final report.
        extend_grid_on_edge: Always False: a selected value on the edge of its
            grid is flagged, the grid is not extended.
        lightgbm_seeds: Seeds of the two LightGBM arms, the selection seed first.
        lightgbm_binary_num_trees: Trees the binary arm's first seed is trained to.
    """

    phase: str
    target_folds: tuple[str, ...]
    metric: str
    schedule_epochs: tuple[int, ...]
    epoch_tolerance: float
    noise_seeds: tuple[int, ...]
    max_seed_range: float
    selection_seed: int
    steps: tuple[DeepFMStudyStep, ...]
    listwise_learning_rates: tuple[float, ...]
    ablation_components: tuple[str, ...]
    ablation_numeric_inputs: tuple[str, ...]
    final_seeds: tuple[int, ...]
    extend_grid_on_edge: bool
    lightgbm_seeds: tuple[int, ...]
    lightgbm_binary_num_trees: int


DEEPFM_STUDY_PHASE = "V"
DEEPFM_LEARNING_RATE = "learning_rate"
DEEPFM_TUNED_PARAMETERS = (DEEPFM_LEARNING_RATE, "embedding_dim", "dropout")


def _distinct_numbers(value: Any, where: str) -> tuple[float, ...]:
    """A list of at least 2 distinct numbers, as floats."""
    is_numbers = isinstance(value, list) and all(
        isinstance(item, int | float) and not isinstance(item, bool) for item in value
    )
    if not is_numbers or len(value) < 2 or len(set(value)) != len(value):
        raise ValueError(f"{where}: expected at least 2 distinct numbers")
    return tuple(float(item) for item in value)


def _parse_deepfm_study_step(raw: Any, position: int) -> DeepFMStudyStep:
    """One entry of `steps`; candidates must be distinct numbers."""
    where = f"hpo/deepfm.steps[{position}]"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping")
    check_keys(raw, {"name", "parameter", "candidates"}, where)
    step = DeepFMStudyStep(
        name=get_str(raw, "name", where),
        parameter=get_str(raw, "parameter", where),
        candidates=_distinct_numbers(raw["candidates"], f"{where}.candidates"),
    )
    _require_choice(step.parameter, DEEPFM_TUNED_PARAMETERS, f"{where}.parameter")
    return step


def _check_deepfm_study(config: DeepFMStudyConfig, where: str) -> None:
    """The range and consistency checks of a parsed study config."""
    if config.extend_grid_on_edge:
        raise ValueError(f"{where}.extend_grid_on_edge must be false: a grid is never extended")
    if config.phase != DEEPFM_STUDY_PHASE:
        raise ValueError(f"{where}.phase must be {DEEPFM_STUDY_PHASE!r}, got {config.phase!r}")
    require_positive(config.epoch_tolerance, f"{where}.epoch_tolerance")
    require_positive(config.max_seed_range, f"{where}.noise.max_seed_range")
    require_positive(config.lightgbm_binary_num_trees, f"{where}.lightgbm.binary_num_trees")
    if list(config.schedule_epochs) != sorted(set(config.schedule_epochs)):
        raise ValueError(f"{where}.schedule_epochs must be distinct and ascending")
    for name in config.ablation_components:
        _require_choice(name, DEEPFM_COMPONENTS, f"{where}.ablations.components")
    for name in config.ablation_numeric_inputs:
        _require_choice(name, DEEPFM_NUMERIC_INPUTS, f"{where}.ablations.numeric_input")
    if config.selection_seed not in config.noise_seeds:
        raise ValueError(f"{where}.selection_seed must be one of the noise seeds")
    if config.selection_seed not in config.final_seeds:
        raise ValueError(f"{where}.selection_seed must be one of the final seeds")
    if config.lightgbm_seeds[0] != config.selection_seed:
        raise ValueError(f"{where}.lightgbm.seeds must start with the selection seed")
    # The schedule length is tuned together with the learning rate, so that step comes first.
    if config.steps[0].parameter != DEEPFM_LEARNING_RATE:
        raise ValueError(f"{where}.steps: the first step must tune {DEEPFM_LEARNING_RATE}")


def parse_deepfm_study_config(raw: Mapping[str, Any]) -> DeepFMStudyConfig:
    """Build a DeepFMStudyConfig from the parsed content of configs/hpo/deepfm.yaml.

    Raises:
        ValueError: If a key is missing or unknown, the phase is not "V", a
            tolerance is not positive, or a variant name is unknown.
    """
    where = "hpo/deepfm"
    keys = {
        "phase",
        "target_folds",
        "metric",
        "schedule_epochs",
        "epoch_tolerance",
        "noise",
        "selection_seed",
        "steps",
        "listwise",
        "ablations",
        "final_seeds",
        "extend_grid_on_edge",
        "lightgbm",
    }
    check_keys(raw, keys, where)
    noise = get_mapping(raw, "noise", where)
    check_keys(noise, {"seeds", "max_seed_range"}, f"{where}.noise")
    listwise = get_mapping(raw, "listwise", where)
    check_keys(listwise, {"learning_rate"}, f"{where}.listwise")
    ablations = get_mapping(raw, "ablations", where)
    check_keys(ablations, {"components", "numeric_input"}, f"{where}.ablations")
    lightgbm = get_mapping(raw, "lightgbm", where)
    check_keys(lightgbm, {"seeds", "binary_num_trees"}, f"{where}.lightgbm")
    raw_steps = raw["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"{where}.steps: expected a non-empty list")
    config = DeepFMStudyConfig(
        phase=get_str(raw, "phase", where),
        target_folds=get_str_list(raw, "target_folds", where),
        metric=get_str(raw, "metric", where),
        schedule_epochs=_get_positive_int_list(raw, "schedule_epochs", where),
        epoch_tolerance=get_float(raw, "epoch_tolerance", where),
        noise_seeds=_get_positive_int_list(noise, "seeds", f"{where}.noise"),
        max_seed_range=get_float(noise, "max_seed_range", f"{where}.noise"),
        selection_seed=get_int(raw, "selection_seed", where),
        steps=tuple(_parse_deepfm_study_step(step, i) for i, step in enumerate(raw_steps)),
        listwise_learning_rates=_distinct_numbers(
            listwise["learning_rate"], f"{where}.listwise.learning_rate"
        ),
        ablation_components=get_str_list(ablations, "components", f"{where}.ablations"),
        ablation_numeric_inputs=get_str_list(ablations, "numeric_input", f"{where}.ablations"),
        final_seeds=_get_positive_int_list(raw, "final_seeds", where),
        extend_grid_on_edge=get_bool(raw, "extend_grid_on_edge", where),
        lightgbm_seeds=_get_positive_int_list(lightgbm, "seeds", f"{where}.lightgbm"),
        lightgbm_binary_num_trees=get_int(lightgbm, "binary_num_trees", f"{where}.lightgbm"),
    )
    _check_deepfm_study(config, where)
    return config


def load_deepfm_study_config(path: Path) -> DeepFMStudyConfig:
    """Read and validate configs/hpo/deepfm.yaml."""
    return parse_deepfm_study_config(load_yaml(path))


# ---------------------------------------------------------------- hpo/deepfm_extension.yaml


@dataclass(frozen=True)
class DeepFMExtensionRun:
    """One run of the input-representation extension (proposal §16.10).

    Attributes:
        name: Short name of the run, e.g. "E1".
        arm: Name of the run's arm in the comparison tables.
        numeric_input: "binned" or "standardised".
        n_bins: Quantile bins of the configuration. The standardised input does
            not use it, but it is part of the configuration id.
        loss: "pointwise_bce" or "listwise_softmax".
    """

    name: str
    arm: str
    numeric_input: str
    n_bins: int
    loss: str


@dataclass(frozen=True)
class DeepFMExtensionConfig:
    """Settings from configs/hpo/deepfm_extension.yaml (proposal §16.10).

    The runs are trained and reported; nothing is tuned or selected.

    Attributes:
        phase: "V".
        target_folds: The folds every run is trained for and evaluated on.
        seeds: The training seeds of every run.
        learning_rate: Adam learning rate of every run.
        embedding_dim: Embedding length of every run.
        dropout: Dropout of every run.
        components: Model components of every run.
        epochs: T, the epochs (schedule length) of every run.
        runs: The runs, in the order they are trained.
        contrasts: (first arm, second arm) pairs: the bootstrap contrasts
            "first minus second" of the comparison.
    """

    phase: str
    target_folds: tuple[str, ...]
    seeds: tuple[int, ...]
    learning_rate: float
    embedding_dim: int
    dropout: float
    components: str
    epochs: int
    runs: tuple[DeepFMExtensionRun, ...]
    contrasts: tuple[tuple[str, str], ...]


def _parse_deepfm_extension_run(raw: Any, position: int) -> DeepFMExtensionRun:
    """One entry of `runs`."""
    where = f"hpo/deepfm_extension.runs[{position}]"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping")
    check_keys(raw, {"name", "arm", "numeric_input", "n_bins", "loss"}, where)
    run = DeepFMExtensionRun(
        name=get_str(raw, "name", where),
        arm=get_str(raw, "arm", where),
        numeric_input=get_str(raw, "numeric_input", where),
        n_bins=get_int(raw, "n_bins", where),
        loss=get_str(raw, "loss", where),
    )
    _require_choice(run.numeric_input, DEEPFM_NUMERIC_INPUTS, f"{where}.numeric_input")
    _require_choice(run.loss, DEEPFM_LOSSES, f"{where}.loss")
    # The same limits as binning.n_bins of configs/deepfm.yaml (a code is a uint8).
    if not 2 <= run.n_bins <= 255:
        raise ValueError(f"{where}.n_bins must be in 2..255, got {run.n_bins}")
    return run


def _parse_arm_pair(raw: Any, where: str) -> tuple[str, str]:
    """One contrast: a list of two different arm names."""
    is_pair = isinstance(raw, list) and len(raw) == 2 and all(isinstance(arm, str) for arm in raw)
    if not is_pair or raw[0] == raw[1]:
        raise ValueError(f"{where}: expected a list of two different arm names")
    return raw[0], raw[1]


def parse_deepfm_extension_config(raw: Mapping[str, Any]) -> DeepFMExtensionConfig:
    """Build a DeepFMExtensionConfig from the content of configs/hpo/deepfm_extension.yaml.

    Raises:
        ValueError: If a key is missing or unknown, the phase is not "V", a
            value is out of range, or run names or arm names repeat.
    """
    where = "hpo/deepfm_extension"
    keys = {
        "phase",
        "target_folds",
        "seeds",
        "learning_rate",
        "embedding_dim",
        "dropout",
        "components",
        "epochs",
        "runs",
        "contrasts",
    }
    check_keys(raw, keys, where)
    raw_runs, raw_contrasts = raw["runs"], raw["contrasts"]
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError(f"{where}.runs: expected a non-empty list")
    if not isinstance(raw_contrasts, list) or not raw_contrasts:
        raise ValueError(f"{where}.contrasts: expected a non-empty list")
    config = DeepFMExtensionConfig(
        phase=get_str(raw, "phase", where),
        target_folds=get_str_list(raw, "target_folds", where),
        seeds=_get_positive_int_list(raw, "seeds", where),
        learning_rate=get_float(raw, "learning_rate", where),
        embedding_dim=get_int(raw, "embedding_dim", where),
        dropout=get_float(raw, "dropout", where),
        components=get_str(raw, "components", where),
        epochs=get_int(raw, "epochs", where),
        runs=tuple(_parse_deepfm_extension_run(run, i) for i, run in enumerate(raw_runs)),
        contrasts=tuple(
            _parse_arm_pair(pair, f"{where}.contrasts[{i}]") for i, pair in enumerate(raw_contrasts)
        ),
    )
    if config.phase != DEEPFM_STUDY_PHASE:
        raise ValueError(f"{where}.phase must be {DEEPFM_STUDY_PHASE!r}, got {config.phase!r}")
    for name in ("learning_rate", "embedding_dim", "epochs"):
        require_positive(getattr(config, name), f"{where}.{name}")
    if not 0 <= config.dropout < 1:
        raise ValueError(f"{where}.dropout must be in [0, 1), got {config.dropout}")
    _require_choice(config.components, DEEPFM_COMPONENTS, f"{where}.components")
    for label, names in (
        ("name", [run.name for run in config.runs]),
        ("arm", [run.arm for run in config.runs]),
    ):
        if len(set(names)) != len(names):
            raise ValueError(f"{where}.runs: a run {label} repeats: {names}")
    return config


def load_deepfm_extension_config(path: Path) -> DeepFMExtensionConfig:
    """Read and validate configs/hpo/deepfm_extension.yaml."""
    return parse_deepfm_extension_config(load_yaml(path))


# ---------------------------------------------------------------- M9: the evaluation on T1-T4

T_EVAL_PHASE = "T"
T_EVAL_CONFIG_ID_LENGTH = 12


@dataclass(frozen=True)
class TEvalConfig:
    """Settings from configs/m9_t_eval.yaml (proposal §16.12).

    ONE configuration, chosen on V, is evaluated on the T folds once and reported
    only; nothing is tuned or selected.

    Attributes:
        phase: "T".
        target_folds: The T folds, in time order.
        seeds: The training seeds (also the seeds of the rehearsal).
        rehearsal_target_fold: The V fold the rehearsal predicts.
        arm: Name of the DeepFM arm in the result tables.
        numeric_input: "binned" or "standardised".
        n_bins: Quantile bins of the configuration (part of the configuration id
            also for the standardised input, which does not use it).
        loss: "pointwise_bce" or "listwise_softmax".
        learning_rate: Adam learning rate.
        embedding_dim: Embedding length.
        dropout: Dropout after every hidden layer.
        components: Model components.
        epochs: T, the epochs (schedule length).
        config_id: The configuration id the arm has on V; the code refuses to
            run a configuration with another id.
        m8_record_arm: Name of the arm that is the M8 record (LightGBM lambdarank).
        m8_record_seed: The seed of that record.
        m8_record_num_trees: Its number of trees.
    """

    phase: str
    target_folds: tuple[str, ...]
    seeds: tuple[int, ...]
    rehearsal_target_fold: str
    arm: str
    numeric_input: str
    n_bins: int
    loss: str
    learning_rate: float
    embedding_dim: int
    dropout: float
    components: str
    epochs: int
    config_id: str
    m8_record_arm: str
    m8_record_seed: int
    m8_record_num_trees: int


def _check_t_eval_config(config: TEvalConfig, where: str) -> None:
    """The value checks of parse_t_eval_config."""
    if config.phase != T_EVAL_PHASE:
        raise ValueError(f"{where}.phase must be {T_EVAL_PHASE!r}, got {config.phase!r}")
    if not config.target_folds or len(set(config.target_folds)) != len(config.target_folds):
        raise ValueError(f"{where}.target_folds must be non-empty and without repeats")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError(f"{where}.seeds: a seed repeats: {config.seeds}")
    _require_choice(config.numeric_input, DEEPFM_NUMERIC_INPUTS, f"{where}.deepfm.numeric_input")
    _require_choice(config.loss, DEEPFM_LOSSES, f"{where}.deepfm.loss")
    _require_choice(config.components, DEEPFM_COMPONENTS, f"{where}.deepfm.components")
    if not 2 <= config.n_bins <= 255:
        raise ValueError(f"{where}.deepfm.n_bins must be in 2..255, got {config.n_bins}")
    for name in ("learning_rate", "embedding_dim", "epochs", "m8_record_num_trees"):
        require_positive(getattr(config, name), f"{where}.{name}")
    if not 0 <= config.dropout < 1:
        raise ValueError(f"{where}.deepfm.dropout must be in [0, 1), got {config.dropout}")
    if len(config.config_id) != T_EVAL_CONFIG_ID_LENGTH:
        raise ValueError(
            f"{where}.deepfm.config_id must have {T_EVAL_CONFIG_ID_LENGTH} characters, "
            f"got {config.config_id!r}"
        )
    if config.arm == config.m8_record_arm:
        raise ValueError(f"{where}: the two arms have the same name {config.arm!r}")
    if config.m8_record_seed not in config.seeds:
        raise ValueError(
            f"{where}.m8_record.seed {config.m8_record_seed} is not one of the seeds {config.seeds}"
        )


def parse_t_eval_config(raw: Mapping[str, Any]) -> TEvalConfig:
    """Build a TEvalConfig from the content of configs/m9_t_eval.yaml.

    Raises:
        ValueError: If a key is missing or unknown, the phase is not "T", or a
            value is out of range.
    """
    where = "m9_t_eval"
    check_keys(raw, {"phase", "target_folds", "seeds", "rehearsal", "deepfm", "m8_record"}, where)
    rehearsal = get_mapping(raw, "rehearsal", where)
    check_keys(rehearsal, {"target_fold"}, f"{where}.rehearsal")
    m8_record = get_mapping(raw, "m8_record", where)
    check_keys(m8_record, {"arm", "seed", "num_trees"}, f"{where}.m8_record")
    deepfm = get_mapping(raw, "deepfm", where)
    deepfm_keys = {
        "arm",
        "numeric_input",
        "n_bins",
        "loss",
        "learning_rate",
        "embedding_dim",
        "dropout",
        "components",
        "epochs",
        "config_id",
    }
    check_keys(deepfm, deepfm_keys, f"{where}.deepfm")
    config = TEvalConfig(
        phase=get_str(raw, "phase", where),
        target_folds=get_str_list(raw, "target_folds", where),
        seeds=_get_positive_int_list(raw, "seeds", where),
        rehearsal_target_fold=get_str(rehearsal, "target_fold", f"{where}.rehearsal"),
        arm=get_str(deepfm, "arm", f"{where}.deepfm"),
        numeric_input=get_str(deepfm, "numeric_input", f"{where}.deepfm"),
        n_bins=get_int(deepfm, "n_bins", f"{where}.deepfm"),
        loss=get_str(deepfm, "loss", f"{where}.deepfm"),
        learning_rate=get_float(deepfm, "learning_rate", f"{where}.deepfm"),
        embedding_dim=get_int(deepfm, "embedding_dim", f"{where}.deepfm"),
        dropout=get_float(deepfm, "dropout", f"{where}.deepfm"),
        components=get_str(deepfm, "components", f"{where}.deepfm"),
        epochs=get_int(deepfm, "epochs", f"{where}.deepfm"),
        config_id=get_str(deepfm, "config_id", f"{where}.deepfm"),
        m8_record_arm=get_str(m8_record, "arm", f"{where}.m8_record"),
        m8_record_seed=get_int(m8_record, "seed", f"{where}.m8_record"),
        m8_record_num_trees=get_int(m8_record, "num_trees", f"{where}.m8_record"),
    )
    _check_t_eval_config(config, where)
    return config


def load_t_eval_config(path: Path) -> TEvalConfig:
    """Read and validate configs/m9_t_eval.yaml."""
    return parse_t_eval_config(load_yaml(path))


# ---------------------------------------------------------------- m10_calibration.yaml

SHA256_HEX_LENGTH = 64


@dataclass(frozen=True)
class CalibrationTolerance:
    """The provisional tolerance of docs/mlops.md §8.3 (not a decision; see the YAML file).

    Attributes:
        min_recorded_pair_share: R5 / R6: smallest share of the recorded
            (customer, article) pairs that must be present (0.99).
        max_count_relative_difference: Largest relative difference of the
            number of deduplicated candidates (0.01).
        max_recall_relative_difference: Largest relative difference of a
            channel's recall and of the union recall (0.01).
        max_map_relative_difference: Largest relative difference of MAP@12 of
            one week (0.01).
    """

    min_recorded_pair_share: float
    max_count_relative_difference: float
    max_recall_relative_difference: float
    max_map_relative_difference: float


@dataclass(frozen=True)
class CalibrationConfig:
    """Settings from configs/m10_calibration.yaml (docs/m10_calibration_plan.md).

    Nothing here changes a result of the system; see the YAML file.

    Attributes:
        fold: The week that is measured and compared, e.g. "V4".
        fine_tune_fold: The week of the R6 fine-tuning step, e.g. "V3".
        runs: Names of the container runs, e.g. ("run1", "run2").
        ranker_reference_area: The results area of the stage-1 Ranker record
            ("m8_rehearsal").
        metrics: The Ranker metrics that are compared, the main metric first.
        processed_sha256: File name under data/processed/ -> its sha256.
        artifact_sha256: Path under artifacts/ -> its sha256 (inputs).
        reference_sha256: Path under artifacts/ -> its sha256 (stage-1 outputs
            the container results are compared with).
        tolerance: The provisional tolerance.
    """

    fold: str
    fine_tune_fold: str
    runs: tuple[str, ...]
    ranker_reference_area: str
    metrics: tuple[str, ...]
    processed_sha256: dict[str, str]
    artifact_sha256: dict[str, str]
    reference_sha256: dict[str, str]
    tolerance: CalibrationTolerance


def _get_sha256_mapping(raw: Mapping[str, Any], key: str, where: str) -> dict[str, str]:
    """A mapping file path -> sha256 (64 lower-case hexadecimal characters); not empty."""
    mapping = get_mapping(raw, key, where)
    if not mapping:
        raise ValueError(f"{where}.{key} must list at least one file")
    result: dict[str, str] = {}
    for name in mapping:
        value = get_str(mapping, name, f"{where}.{key}")
        is_hex = set(value) <= set("0123456789abcdef")
        if len(value) != SHA256_HEX_LENGTH or not is_hex:
            raise ValueError(f"{where}.{key}.{name} must be a sha256 in hexadecimal, got {value!r}")
        result[str(name)] = value
    return result


def parse_calibration_config(raw: Mapping[str, Any]) -> CalibrationConfig:
    """Build a CalibrationConfig from the parsed content of configs/m10_calibration.yaml.

    Raises:
        ValueError: If a key is missing or unknown, a hash is not a sha256, a
            share is outside (0, 1), or the run names repeat.
    """
    where = "m10_calibration"
    top_keys = {
        "fold",
        "fine_tune_fold",
        "runs",
        "ranker_reference_area",
        "metrics",
        "inputs",
        "references",
        "provisional_tolerance",
    }
    check_keys(raw, top_keys, where)
    inputs = get_mapping(raw, "inputs", where)
    check_keys(inputs, {"processed", "artifacts"}, f"{where}.inputs")
    tolerance_where = f"{where}.provisional_tolerance"
    tolerance_raw = get_mapping(raw, "provisional_tolerance", where)
    tolerance_keys = {
        "min_recorded_pair_share",
        "max_count_relative_difference",
        "max_recall_relative_difference",
        "max_map_relative_difference",
    }
    check_keys(tolerance_raw, tolerance_keys, tolerance_where)
    tolerance = CalibrationTolerance(
        **{name: get_float(tolerance_raw, name, tolerance_where) for name in sorted(tolerance_keys)}
    )
    for name in sorted(tolerance_keys):
        value = getattr(tolerance, name)
        if not 0 < value < 1:
            raise ValueError(f"{tolerance_where}.{name} must be in (0, 1), got {value}")
    runs = get_str_list(raw, "runs", where)
    if not runs or len(set(runs)) != len(runs):
        raise ValueError(f"{where}.runs must be distinct names and not empty, got {runs}")
    return CalibrationConfig(
        fold=get_str(raw, "fold", where),
        fine_tune_fold=get_str(raw, "fine_tune_fold", where),
        runs=runs,
        ranker_reference_area=get_str(raw, "ranker_reference_area", where),
        metrics=get_str_list(raw, "metrics", where),
        processed_sha256=_get_sha256_mapping(inputs, "processed", f"{where}.inputs"),
        artifact_sha256=_get_sha256_mapping(inputs, "artifacts", f"{where}.inputs"),
        reference_sha256=_get_sha256_mapping(raw, "references", where),
        tolerance=tolerance,
    )


def load_calibration_config(path: Path) -> CalibrationConfig:
    """Read and validate configs/m10_calibration.yaml."""
    return parse_calibration_config(load_yaml(path))


# ---------------------------------------------------------------- m10_r6_repeatability.yaml


@dataclass(frozen=True)
class RepeatabilityTableSizes:
    """Sizes of the tables the reproducer's model is built on (the real tables' sizes).

    Attributes:
        n_articles: Number of articles (105,542).
        n_customers: Number of customers (1,371,980).
        image_dim: Width of the image vectors (512).
        text_dim: Width of the text vectors (384).
        article_code_values: Distinct values of product_code and of each category column.
        profile_values: Distinct values of each profile column.
    """

    n_articles: int
    n_customers: int
    image_dim: int
    text_dim: int
    article_code_values: tuple[int, ...]
    profile_values: tuple[int, ...]


@dataclass(frozen=True)
class RepeatabilityConfig:
    """Settings from configs/m10_r6_repeatability.yaml (see that file).

    Attributes:
        seed: Seed of the synthetic tables and samples.
        table_sizes: Sizes of the tables the model is built on.
        scales: Allowed numbers of training rows of a synthetic fine-tune set.
        alarm_samples: Rows of the alarm users (loss only).
        popularity_offset: Article k is drawn with probability proportional to
            1 / (k + popularity_offset).
        max_group_rows: A synthetic loss group has 1 .. max_group_rows rows.
    """

    seed: int
    table_sizes: RepeatabilityTableSizes
    scales: tuple[int, ...]
    alarm_samples: int
    popularity_offset: float
    max_group_rows: int


def _get_positive_ints(raw: Mapping[str, Any], key: str, where: str) -> tuple[int, ...]:
    """A non-empty list of positive integers."""
    values = raw.get(key)
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"{where}.{key} must be a non-empty list of positive integers")
    if not all(type(value) is int and value > 0 for value in values):
        raise ValueError(f"{where}.{key} must be a non-empty list of positive integers")
    return tuple(int(value) for value in values)


def parse_repeatability_config(raw: Mapping[str, Any]) -> RepeatabilityConfig:
    """Build a RepeatabilityConfig from the parsed content of configs/m10_r6_repeatability.yaml.

    Raises:
        ValueError: If a key is missing or unknown, or a number is not positive.
    """
    where = "m10_r6_repeatability"
    top_keys = {
        "seed",
        "table_sizes",
        "scales",
        "alarm_samples",
        "popularity_offset",
        "max_group_rows",
    }
    check_keys(raw, top_keys, where)
    sizes_where = f"{where}.table_sizes"
    sizes_raw = get_mapping(raw, "table_sizes", where)
    size_keys = {
        "n_articles",
        "n_customers",
        "image_dim",
        "text_dim",
        "article_code_values",
        "profile_values",
    }
    check_keys(sizes_raw, size_keys, sizes_where)
    sizes = RepeatabilityTableSizes(
        n_articles=get_int(sizes_raw, "n_articles", sizes_where),
        n_customers=get_int(sizes_raw, "n_customers", sizes_where),
        image_dim=get_int(sizes_raw, "image_dim", sizes_where),
        text_dim=get_int(sizes_raw, "text_dim", sizes_where),
        article_code_values=_get_positive_ints(sizes_raw, "article_code_values", sizes_where),
        profile_values=_get_positive_ints(sizes_raw, "profile_values", sizes_where),
    )
    config = RepeatabilityConfig(
        seed=get_int(raw, "seed", where),
        table_sizes=sizes,
        scales=_get_positive_ints(raw, "scales", where),
        alarm_samples=get_int(raw, "alarm_samples", where),
        popularity_offset=get_float(raw, "popularity_offset", where),
        max_group_rows=get_int(raw, "max_group_rows", where),
    )
    for name in ("n_articles", "n_customers", "image_dim", "text_dim"):
        require_positive(getattr(sizes, name), f"{sizes_where}.{name}")
    require_positive(config.alarm_samples, f"{where}.alarm_samples")
    require_positive(config.popularity_offset, f"{where}.popularity_offset")
    require_positive(config.max_group_rows, f"{where}.max_group_rows")
    return config


def load_repeatability_config(path: Path) -> RepeatabilityConfig:
    """Read and validate configs/m10_r6_repeatability.yaml."""
    return parse_repeatability_config(load_yaml(path))
