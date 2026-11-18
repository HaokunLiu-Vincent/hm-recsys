"""The rehearsal's comparison with M8, and the gate it puts before the real run.

The submission week has no labels, so the new code path is checked on the last
fold (T4), which M8 predicted with the old one (docs/decisions.md 2026-11-19).
Three checks, all needed to pass:

    1. trees     the ranker the rehearsal trained has the same first trees as the
                 model M8 trained for T4 (hmrec.kaggle.ranker).
    2. Top-K     for the T4 queries, the share of customers whose Top-K list is
                 IDENTICAL (same articles, same order) to the one M8 saved must be
                 at least min_identical_top12_share (compare_top_lists, below).
    3. format    the submission-format checks pass on the rehearsal's file
                 (hmrec.kaggle.submission).

The outcome is one row of results/kaggle_rehearsal.csv. The real run starts only
if the LAST row passed and the code has not changed since
(require_passed_rehearsal). Nothing here reads a label: M8's saved predictions
are compared, not scored.

No torch and no LightGBM is imported here.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from hmrec.csv_records import append_row, read_rows
from hmrec.paths import ProjectPaths
from hmrec.run_records import RunStamp

QUERY_ID = "query_id"
CUSTOMER_IDX = "customer_idx"
ARTICLE_IDX = "article_idx"
RANK = "rank"
NEW_LIST = "new_list"
OLD_LIST = "old_list"

REHEARSAL_COLUMNS: tuple[str, ...] = (
    "checked_at",
    "git_commit",
    "git_dirty",
    "n_customers",
    "n_shards",
    "trees_identical",
    "new_trees_sha256",
    "m8_trees_sha256",
    "n_fold_queries",
    "n_identical_top",
    "identical_top_share",
    "min_identical_top_share",
    "format_checks_passed",
    "passed",
)
# Reads a recorded commit and says whether the code at HEAD differs from it.
CodeChanged = Callable[[str], bool]


@dataclass(frozen=True)
class TopComparison:
    """How many of the fold's queries got an identical Top-K list.

    Attributes:
        n_queries: Queries of the fold (all of them are compared).
        n_identical: Queries whose new list equals the saved one: the same
            articles in the same order.
    """

    n_queries: int
    n_identical: int

    @property
    def share(self) -> float:
        """n_identical / n_queries."""
        return self.n_identical / self.n_queries


def _lists_by_customer(top: pl.DataFrame, queries: pl.DataFrame, name: str) -> pl.DataFrame:
    """customer_idx and its articles in rank order, as one list column called `name`."""
    with_customer = top.select(QUERY_ID, RANK, ARTICLE_IDX).join(
        queries.select(QUERY_ID, CUSTOMER_IDX), on=QUERY_ID, how="inner"
    )
    return with_customer.group_by(CUSTOMER_IDX).agg(pl.col(ARTICLE_IDX).sort_by(RANK).alias(name))


def compare_top_lists(
    new_top: pl.DataFrame,
    new_queries: pl.DataFrame,
    old_top: pl.DataFrame,
    old_queries: pl.DataFrame,
) -> TopComparison:
    """Compare the new Top-K lists with the saved ones, customer by customer.

    The two runs number their queries differently, so the lists are matched by
    customer_idx. A list is identical only if it holds the same articles in the
    same order. A fold query without a new list counts as not identical.

    Args:
        new_top: The rehearsal's Top-K: query_id, rank, article_idx (any customers).
        new_queries: The rehearsal's queries: query_id, customer_idx.
        old_top: The saved Top-K of the fold: query_id, rank, article_idx.
        old_queries: The fold's queries: query_id, customer_idx.

    Raises:
        ValueError: If the fold has no queries, or a fold query has no saved list.
    """
    old_lists = _lists_by_customer(old_top, old_queries, OLD_LIST)
    if old_queries.height == 0 or old_lists.height != old_queries.height:
        raise ValueError(
            f"the fold has {old_queries.height} queries but {old_lists.height} saved Top-K lists"
        )
    new_lists = _lists_by_customer(new_top, new_queries, NEW_LIST)
    both = old_lists.join(new_lists, on=CUSTOMER_IDX, how="left")
    # A missing new list is null, and null == list is null: counted as not identical.
    identical = (pl.col(NEW_LIST) == pl.col(OLD_LIST)).fill_null(False)
    return TopComparison(old_lists.height, both.filter(identical).height)


@dataclass(frozen=True)
class RehearsalOutcome:
    """The three checks of one rehearsal.

    Attributes:
        n_customers: Customers of the rehearsal.
        n_shards: Shards they were processed in.
        new_trees_sha256: sha256 of the trees of the rehearsal's model.
        m8_trees_sha256: sha256 of the trees of M8's model of the fold.
        comparison: The Top-K comparison.
        min_identical_share: The required share of identical lists.
        format_checks_passed: Whether every submission-format check passed.
    """

    n_customers: int
    n_shards: int
    new_trees_sha256: str
    m8_trees_sha256: str
    comparison: TopComparison
    min_identical_share: float
    format_checks_passed: bool

    @property
    def trees_identical(self) -> bool:
        """Check 1: the two models have the same trees."""
        return self.new_trees_sha256 == self.m8_trees_sha256

    @property
    def top_identical_enough(self) -> bool:
        """Check 2: enough of the fold's queries have an identical Top-K list."""
        return self.comparison.share >= self.min_identical_share

    @property
    def passed(self) -> bool:
        """All three checks passed."""
        return self.trees_identical and self.top_identical_enough and self.format_checks_passed


def record_rehearsal(paths: ProjectPaths, outcome: RehearsalOutcome, stamp: RunStamp) -> None:
    """Append the outcome of a rehearsal to results/kaggle_rehearsal.csv."""
    row = {
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": stamp.git_commit,
        "git_dirty": str(stamp.git_dirty),
        "n_customers": str(outcome.n_customers),
        "n_shards": str(outcome.n_shards),
        "trees_identical": str(outcome.trees_identical),
        "new_trees_sha256": outcome.new_trees_sha256,
        "m8_trees_sha256": outcome.m8_trees_sha256,
        "n_fold_queries": str(outcome.comparison.n_queries),
        "n_identical_top": str(outcome.comparison.n_identical),
        "identical_top_share": repr(outcome.comparison.share),
        "min_identical_top_share": repr(outcome.min_identical_share),
        "format_checks_passed": str(outcome.format_checks_passed),
        "passed": str(outcome.passed),
    }
    append_row(paths.kaggle_rehearsal_file, REHEARSAL_COLUMNS, row)


def require_passed_rehearsal(paths: ProjectPaths, code_changed: CodeChanged) -> str:
    """Fail unless the last recorded rehearsal passed with the code that is running now.

    Args:
        paths: Project paths.
        code_changed: Says whether the code at HEAD differs from a recorded commit
            (hmrec.git_state.code_changed_since: src, scripts, configs and the
            dependency files; result tables and documents do not count).

    Returns:
        The commit the passed rehearsal ran at.

    Raises:
        RuntimeError: If there is no rehearsal, the last one failed, was run from a
            dirty tree, or the code changed after it.
    """
    rows = read_rows(paths.kaggle_rehearsal_file, REHEARSAL_COLUMNS)
    if not rows:
        raise RuntimeError("no rehearsal is recorded: the real run needs a passed rehearsal first")
    last = rows[-1]
    if last["passed"] != "True" or last["git_dirty"] != "False":
        raise RuntimeError(
            f"the last rehearsal ({last['checked_at']}, commit {last['git_commit'][:12]}) did "
            "not pass from a clean working tree: the real run does not start"
        )
    if code_changed(last["git_commit"]):
        raise RuntimeError(
            f"the code changed since the passed rehearsal (commit {last['git_commit'][:12]}): "
            "run the rehearsal again before the real run"
        )
    return last["git_commit"]
