"""Shared recall interface, ranking and output schema check (proposal §5).

Every channel produces one table, one row per (query_id, article_idx):

    query_id (int32), article_idx (int32), channel (str), rank (int32), score (float32)
    + the channel's declared extra columns (e.g. R1: pop_is_fallback int8)

Rules (checked by `validate_candidates` before anything is written):
    - article_idx is unique within a query;
    - rows of a query are ordered by the channel's sort keys; the default is
      (score DESC, article_idx ASC). R1 declares (pop_is_fallback ASC, score DESC,
      article_idx ASC), because its fill-in rows use a different score scale;
    - rank is 1, 2, ..., n in that order;
    - at most `quota` rows per query, never padded;
    - exactly the base columns plus the declared extras, with the declared
      dtypes, and no nulls / NaN.

Each channel declares its output with an `OutputSpec` (channel name, extra
columns, sort keys) and builds its table with `rank_candidates`, so ranking
follows the same code everywhere.

Time: every channel reads history only through hmrec.data.asof. Ages are
age_days = anchor_day - t_dat_day (proposal §2.2), always >= 1;
`add_age_days` raises if a row has age_days < 1 (that would mean data from the
anchor day or later leaked into the history).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

import polars as pl

from hmrec.data.day_index import date_to_day_index

QUERY_ID = "query_id"
CUSTOMER_IDX = "customer_idx"
ARTICLE_IDX = "article_idx"
CHANNEL = "channel"
RANK = "rank"
SCORE = "score"
AGE_DAYS = "age_days"
TRANSACTION_DAY = "t_dat_day"

BASE_SCHEMA = pl.Schema(
    {
        QUERY_ID: pl.Int32,
        ARTICLE_IDX: pl.Int32,
        CHANNEL: pl.String,
        RANK: pl.Int32,
        SCORE: pl.Float32,
    }
)


# ---------------------------------------------------------------- output spec


@dataclass(frozen=True)
class SortKey:
    """One key of a channel's within-query order.

    Attributes:
        column: Column name (a base column or a declared extra column).
        descending: True for DESC, False for ASC.
    """

    column: str
    descending: bool


DEFAULT_SORT_KEYS = (SortKey(SCORE, descending=True), SortKey(ARTICLE_IDX, descending=False))


@dataclass(frozen=True)
class OutputSpec:
    """What one channel outputs: its name, extra columns and within-query order.

    Attributes:
        channel: Short channel name, e.g. "pop"; stored in the channel column.
        extra_columns: Extra column name -> dtype. Every name must start with
            "{channel}_" so columns from different channels never collide in the
            wide table (proposal §7.3).
        sort_keys: Within-query order. Must end with article_idx ASC, which is
            unique within a query, so the order is always total.
    """

    channel: str
    extra_columns: Mapping[str, pl.DataType] = field(default_factory=dict)
    sort_keys: tuple[SortKey, ...] = DEFAULT_SORT_KEYS

    def __post_init__(self) -> None:
        """Check the declaration itself (prefixes, sort keys)."""
        for name in self.extra_columns:
            if name in BASE_SCHEMA:
                raise ValueError(f"{self.channel}: extra column {name!r} is a base column")
            if not name.startswith(f"{self.channel}_"):
                raise ValueError(
                    f"{self.channel}: extra column {name!r} must start with {self.channel + '_'!r}"
                )
        known_columns = set(BASE_SCHEMA.names()) | set(self.extra_columns)
        for key in self.sort_keys:
            if key.column not in known_columns:
                raise ValueError(f"{self.channel}: sort key {key.column!r} is not an output column")
        if not self.sort_keys or self.sort_keys[-1] != SortKey(ARTICLE_IDX, descending=False):
            raise ValueError(f"{self.channel}: sort keys must end with article_idx ASC")

    @property
    def schema(self) -> pl.Schema:
        """Full output schema: base columns, then extra columns."""
        return pl.Schema({**dict(BASE_SCHEMA), **dict(self.extra_columns)})


# ---------------------------------------------------------------- inputs and interface


@dataclass(frozen=True)
class RecallInputs:
    """Everything a channel may read for one anchor.

    Attributes:
        anchor: First day of the label week; history is t_dat < anchor.
        day_zero: Day 0 of t_dat_day (PreprocessConfig.day_zero).
        transactions: Processed transactions (one row per purchase line, int16
            t_dat_day), read with metadata.scan_processed_parquet. Channels must
            cut it with hmrec.data.asof only.
        customers: One row per customer; columns customer_idx (int32), age_bucket (str).
        articles: One row per article; columns article_idx (int32), product_code (int32).
        active_pool: A(a) from catalog.build_active_pool; column article_idx (int32).
    """

    anchor: date
    day_zero: date
    transactions: pl.LazyFrame
    customers: pl.DataFrame
    articles: pl.DataFrame
    active_pool: pl.DataFrame

    @property
    def anchor_day(self) -> int:
        """The anchor as a t_dat_day index (via hmrec.data.day_index)."""
        return date_to_day_index(self.anchor, self.day_zero)


class RecallChannel(Protocol):
    """Interface of every recall channel.

    A channel is configured once (its config dataclass) and then called once
    per anchor with that week's query list.
    """

    @property
    def output_spec(self) -> OutputSpec:
        """The channel's output declaration."""
        ...

    def generate(self, queries: pl.DataFrame, inputs: RecallInputs) -> pl.DataFrame:
        """Return candidates for `queries`.

        Args:
            queries: One row per query; columns query_id (int32), customer_idx (int32).
            inputs: Data for this anchor.

        Returns:
            One row per (query_id, article_idx), valid for `output_spec`
            (see validate_candidates). Queries with no candidate have no rows.
        """
        ...


# ---------------------------------------------------------------- time helpers


def add_age_days(history: pl.DataFrame, anchor_day: int) -> pl.DataFrame:
    """Add age_days = anchor_day - t_dat_day (proposal §2.2): yesterday has age 1.

    Args:
        history: Purchase lines from hmrec.data.asof (column t_dat_day, int16).
        anchor_day: The anchor as a day index.

    Returns:
        `history` with an extra int32 column age_days.

    Raises:
        ValueError: If any age_days < 1, which means a row from the anchor day or
            later is in the history (a time leak).
    """
    with_age = history.with_columns(
        (pl.lit(anchor_day, dtype=pl.Int32) - pl.col(TRANSACTION_DAY).cast(pl.Int32)).alias(
            AGE_DAYS
        )
    )
    if with_age.height > 0:
        min_age = with_age[AGE_DAYS].min()
        if not isinstance(min_age, int) or min_age < 1:
            raise ValueError(
                f"history has age_days={min_age} (< 1): rows on or after the anchor leaked in"
            )
    return with_age


def decay_weight_expr(tau_days: float) -> pl.Expr:
    """Weight of one purchase line: exp(-age_days / tau) (proposal §2.2), as float64.

    tau is a time constant, not a half-life: the weight halves after tau * ln 2 days.
    Computed in float64; channels cast the final score to float32 in rank_candidates.
    """
    if tau_days <= 0 or math.isnan(tau_days):
        raise ValueError(f"tau_days must be > 0, got {tau_days}")
    return (-pl.col(AGE_DAYS).cast(pl.Float64) / tau_days).exp()


# ---------------------------------------------------------------- ranking


def rank_candidates(scored: pl.DataFrame, spec: OutputSpec, quota: int) -> pl.DataFrame:
    """Order, cut to quota, number and validate the candidates of every query.

    The score is cast to float32 BEFORE sorting, so the stored order always
    matches the stored scores: two scores that differ only beyond float32
    precision become equal and are then ordered by article_idx.

    Args:
        scored: One row per (query_id, article_idx) with column score (any float
            dtype) and the spec's extra columns.
        spec: The channel's output declaration.
        quota: Maximum rows kept per query (> 0).

    Returns:
        A table valid for `spec`, sorted by (query_id, rank).
    """
    if quota <= 0:
        raise ValueError(f"quota must be > 0, got {quota}")
    typed = scored.with_columns(
        pl.col(QUERY_ID).cast(pl.Int32),
        pl.col(ARTICLE_IDX).cast(pl.Int32),
        pl.col(SCORE).cast(pl.Float32),
        *[pl.col(name).cast(dtype) for name, dtype in spec.extra_columns.items()],
    )
    sort_columns = [QUERY_ID] + [key.column for key in spec.sort_keys]
    descending = [False] + [key.descending for key in spec.sort_keys]
    ordered = typed.sort(sort_columns, descending=descending)
    ranked = ordered.with_columns(
        pl.int_range(1, pl.len() + 1, dtype=pl.Int32).over(QUERY_ID).alias(RANK),
        pl.lit(spec.channel).alias(CHANNEL),
    ).filter(pl.col(RANK) <= quota)
    result = ranked.select(spec.schema.names())
    validate_candidates(result, spec, quota)
    return result


# ---------------------------------------------------------------- validation


def _check_columns(candidates: pl.DataFrame, spec: OutputSpec) -> None:
    """Fail on undeclared columns, missing columns, or wrong dtypes (both directions)."""
    expected = spec.schema
    undeclared = [name for name in candidates.columns if name not in expected]
    missing = [name for name in expected.names() if name not in candidates.columns]
    if undeclared:
        raise ValueError(f"{spec.channel}: undeclared columns {undeclared}")
    if missing:
        raise ValueError(f"{spec.channel}: missing columns {missing}")
    for name, dtype in expected.items():
        if candidates.schema[name] != dtype:
            raise ValueError(
                f"{spec.channel}: column {name!r} has dtype {candidates.schema[name]}, "
                f"expected {dtype}"
            )


def _check_values(candidates: pl.DataFrame, spec: OutputSpec, quota: int | None) -> None:
    """Fail on nulls, NaN/inf scores, a wrong channel value, duplicates, or > quota rows."""
    null_columns = [name for name in candidates.columns if candidates[name].null_count() > 0]
    if null_columns:
        raise ValueError(f"{spec.channel}: null values in columns {null_columns}")
    if not candidates[SCORE].is_finite().all():
        raise ValueError(f"{spec.channel}: score has NaN or infinite values")
    if not (candidates[CHANNEL] == spec.channel).all():
        raise ValueError(f"{spec.channel}: channel column must be {spec.channel!r} in every row")
    if candidates.select(QUERY_ID, ARTICLE_IDX).is_duplicated().any():
        raise ValueError(f"{spec.channel}: article_idx repeats within a query")
    if quota is not None and candidates.height > 0:
        largest = candidates.group_by(QUERY_ID).len()["len"].max()
        if isinstance(largest, int) and largest > quota:
            raise ValueError(f"{spec.channel}: a query has {largest} rows, quota is {quota}")


def _check_order(candidates: pl.DataFrame, spec: OutputSpec) -> None:
    """Fail unless rank is 1..n per query AND follows the declared sort keys.

    Sorting by (query_id, sort keys) and numbering 1..n per query gives the
    expected rank of every row; it must equal the stored rank. Because the
    keys end with article_idx (unique per query), the expected order is unique.
    """
    sort_columns = [QUERY_ID] + [key.column for key in spec.sort_keys]
    descending = [False] + [key.descending for key in spec.sort_keys]
    expected = candidates.sort(sort_columns, descending=descending).with_columns(
        pl.int_range(1, pl.len() + 1, dtype=pl.Int32).over(QUERY_ID).alias("expected_rank")
    )
    mismatched = expected.filter(pl.col(RANK) != pl.col("expected_rank"))
    if mismatched.height > 0:
        examples = mismatched[QUERY_ID].unique().sort().head(5).to_list()
        order = ", ".join(f"{k.column} {'DESC' if k.descending else 'ASC'}" for k in spec.sort_keys)
        raise ValueError(
            f"{spec.channel}: rank must be 1..n following ({order}); bad queries {examples}"
        )


def validate_candidates(
    candidates: pl.DataFrame, spec: OutputSpec, quota: int | None = None
) -> None:
    """Check a channel's output against its OutputSpec (see the module docstring).

    Args:
        candidates: The channel output.
        spec: The channel's output declaration.
        quota: If given, no query may have more rows.

    Raises:
        ValueError: On the first broken rule, naming the channel and the rule.
    """
    _check_columns(candidates, spec)
    _check_values(candidates, spec, quota)
    _check_order(candidates, spec)


def write_candidates(
    candidates: pl.DataFrame, spec: OutputSpec, quota: int, path: Path, overwrite: bool
) -> None:
    """Validate, then write a channel's candidates to Parquet.

    Candidates of a fold are frozen once written (AGENTS.md §3.1), so an
    existing file is replaced only with overwrite=True.

    Raises:
        FileExistsError: If `path` exists and overwrite is False.
        ValueError: If validation fails (nothing is written).
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")
    validate_candidates(candidates, spec, quota)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates.write_parquet(path)
