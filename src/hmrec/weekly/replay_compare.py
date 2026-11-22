"""Compare the weeks the replay closed with their stage-1 records (docs/mlops.md §6.3, §8.3).

For every `eval` week the pipeline has closed, one batch of rows goes to
results/m12_replay/comparison.csv:

    acceptance   recall of each channel, union recall and MAP@12 of the week
                 against the stage-1 record; relative difference
                 |value - record| / record; inside if it is at most the threshold
                 of the platform record (configs/platforms/linux-arm64.yaml: 1.0%)
    structural   feature rows minus the deduplicated candidates of the same tick
                 (must be 0) and the number of repeated keys (must be 0)
    reported     pair overlap of every channel's candidates with the stage-1
                 file, the share of common pairs with a bit-identical score, the
                 deduplicated candidate count, the RRF baseline. Never judged.

When all the weeks of the replay are closed: the pooled MAP@12 (query-weighted
mean over the weeks, hmrec.hpo.v_records.pooled_score's formula) against the same
pooling of the records (acceptance).

Where the numbers come from:
    the pipeline   weeks/{anchor}/evaluation/metrics.json (what step C2 measured)
                   and the step records of the state. NO label is read here.
    the records    results/m6_candidates.csv and results/v_phase_metrics.csv for
                   V weeks; results/t_phase_recall.csv and results/t_phase_metrics.csv
                   (the M8 tables) for T weeks. They are only read. The Ranker's
                   row is the one of the locked configuration and tree count.
    pair overlap   the stage-1 candidate files artifacts/candidates/{fold}/. This
                   is why the comparison runs on the laptop and not in a tick's
                   container: a tick cannot see a stage-1 artifact.

The `serve` tick's diagnostic (reported only): the Top-12 overlap with the saved
prediction set of the Kaggle late submission (artifacts/kaggle/submission/).

Nothing is selected or tuned with any of these numbers. A week is compared once:
its rows are appended and never replaced.

This module imports neither torch nor lightgbm.
"""

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from hmrec.csv_records import append_row, read_rows
from hmrec.data.folds import Fold
from hmrec.locks import ALL_CHANNELS, RankerLock
from hmrec.paths import ProjectPaths, StateLayout, recall_step_dir
from hmrec.weekly.context import read_step_output
from hmrec.weekly.outputs import WeekEvaluation, read_week_evaluation

ACCEPTANCE = "acceptance"
STRUCTURAL = "structural"
REPORTED = "reported"
UNION = "union"
FULL_EXPERIMENT = "full"
RRF_BASELINE = "rrf_baseline"
RRF_NUM_TREES = "0"
MAP_AT_12 = "map@12"
VALIDATION_PHASE = "V"
TEST_PHASE = "T"
PAIR = ["query_id", "article_idx"]
POOLED_WEEK = "pooled"
NOT_JUDGED = ""

COMPARISON_COLUMNS: tuple[str, ...] = (
    "week",
    "anchor",
    "kind",
    "quantity",
    "subject",
    "value",
    "record",
    "relative_difference",
    "threshold",
    "inside",
    "record_source",
    "git_commit",
    "recorded_at",
)
SERVE_OVERLAP_COLUMNS: tuple[str, ...] = (
    "anchor",
    "n_customers",
    "n_customers_in_reference",
    "identical_list_share",
    "same_articles_share",
    "mean_overlap",
    "first_article_identical_share",
    "reference",
    "git_commit",
    "recorded_at",
)


@dataclass(frozen=True)
class Comparison:
    """One compared quantity of one week.

    Attributes:
        kind: "acceptance", "structural" or "reported".
        quantity: What is compared, e.g. "recall" or "map@12".
        subject: A channel, "union", or "" if the quantity has one subject.
        value: The replay's value.
        record: The stage-1 record (for a structural check: the value it must have).
        source: Where the record comes from, e.g. "results/v_phase_metrics.csv".
    """

    kind: str
    quantity: str
    subject: str
    value: float
    record: float
    source: str

    @property
    def relative_difference(self) -> float:
        """|value - record| / |record|; 0 if both are 0, infinite if only the record is 0."""
        if self.record == 0:
            return 0.0 if self.value == 0 else math.inf
        return abs(self.value - self.record) / abs(self.record)


def is_inside(comparison: Comparison, threshold: float) -> bool | None:
    """Whether a comparison meets its criterion; None for a reported quantity.

    acceptance: the relative difference is at most the threshold.
    structural: the value equals the value it must have.
    """
    if comparison.kind == ACCEPTANCE:
        return comparison.relative_difference <= threshold
    if comparison.kind == STRUCTURAL:
        return comparison.value == comparison.record
    return None


# ---------------------------------------------------------------- the stage-1 records


@dataclass(frozen=True)
class WeekRecord:
    """The stage-1 record of one week.

    Attributes:
        channel_recall: Channel -> recall of the week.
        union_recall: Union recall of the six channels.
        deduplicated_candidates: Distinct (query, article) pairs of the union.
        map_at_12: MAP@12 of the locked Ranker.
        rrf_map_at_12: MAP@12 of the RRF baseline.
        n_queries: Queries of the week.
        recall_source: The recall table, relative to the repository root.
        metrics_source: The metric table, relative to the repository root.
    """

    channel_recall: dict[str, float]
    union_recall: float
    deduplicated_candidates: int
    map_at_12: float
    rrf_map_at_12: float
    n_queries: int
    recall_source: str
    metrics_source: str


def read_table(path: Path) -> list[dict[str, str]]:
    """The rows of a result table (header from the file)."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one_row(
    rows: Sequence[Mapping[str, str]], where: Mapping[str, str], table: Path
) -> Mapping[str, str]:
    """The one row whose columns have the given values.

    Raises:
        ValueError: If no row or more than one row matches.
    """
    matches = [row for row in rows if all(row[key] == value for key, value in where.items())]
    if len(matches) != 1:
        raise ValueError(f"{table}: {len(matches)} rows match {dict(where)}, expected 1")
    return matches[0]


def record_tables(paths: ProjectPaths, week: Fold) -> tuple[Path, Path]:
    """(recall table, metric table) that hold the stage-1 record of a week.

    Raises:
        ValueError: If the week is not a V or T fold: stage 1 recorded no MAP@12 for it.
    """
    if week.phase == VALIDATION_PHASE:
        return paths.m6_candidates_results_file, paths.v_phase_metrics_file
    if week.phase == TEST_PHASE:
        return paths.t_phase_recall_file, paths.t_phase_metrics_file
    raise ValueError(f"week {week.name} (phase {week.phase}) has no stage-1 record to compare with")


def read_week_record(paths: ProjectPaths, week: Fold, lock: RankerLock) -> WeekRecord:
    """Read the stage-1 record of a V or T week. Only recorded numbers are read; no label.

    Args:
        paths: The project paths (the stage-1 result tables).
        week: The week.
        lock: The Ranker lock: its config id and tree count select the Ranker's row.
    """
    recall_file, metrics_file = record_tables(paths, week)
    recall_rows, metric_rows = read_table(recall_file), read_table(metrics_file)
    union = one_row(recall_rows, {"fold": week.name, "scope": UNION}, recall_file)
    ranker_where = {
        "experiment": FULL_EXPERIMENT,
        "config_id": lock.selection.config_id,
        "fold": week.name,
        "num_trees": str(lock.model.num_trees),
    }
    baseline_where = {
        "experiment": RRF_BASELINE,
        "config_id": RRF_BASELINE,
        "fold": week.name,
        "num_trees": RRF_NUM_TREES,
    }
    ranker = one_row(metric_rows, ranker_where, metrics_file)
    baseline = one_row(metric_rows, baseline_where, metrics_file)
    root = paths.project_root
    return WeekRecord(
        channel_recall={
            channel: float(
                one_row(recall_rows, {"fold": week.name, "scope": channel}, recall_file)["recall"]
            )
            for channel in ALL_CHANNELS
        },
        union_recall=float(union["recall"]),
        deduplicated_candidates=int(union["n_rows"]),
        map_at_12=float(ranker[MAP_AT_12]),
        rrf_map_at_12=float(baseline[MAP_AT_12]),
        n_queries=int(ranker["n_queries"]),
        recall_source=recall_file.relative_to(root).as_posix(),
        metrics_source=metrics_file.relative_to(root).as_posix(),
    )


# ---------------------------------------------------------------- comparing one week


def metric_comparisons(
    evaluation: WeekEvaluation, record: WeekRecord, feature_rows: int
) -> list[Comparison]:
    """The acceptance quantities of a week and the two reported numbers that need no file.

    Args:
        evaluation: What step C2 measured for the week.
        record: The stage-1 record of the week.
        feature_rows: Feature rows of the week's tick (= its deduplicated candidates).
    """
    rows = [
        Comparison(
            ACCEPTANCE,
            "recall",
            channel,
            evaluation.channel_recall[channel],
            record.channel_recall[channel],
            record.recall_source,
        )
        for channel in ALL_CHANNELS
    ]
    rows.append(
        Comparison(
            ACCEPTANCE,
            "recall",
            UNION,
            evaluation.union_recall,
            record.union_recall,
            record.recall_source,
        )
    )
    rows.append(
        Comparison(
            ACCEPTANCE,
            MAP_AT_12,
            "",
            evaluation.ranker[MAP_AT_12],
            record.map_at_12,
            record.metrics_source,
        )
    )
    rows.append(
        Comparison(
            REPORTED,
            f"rrf_baseline_{MAP_AT_12}",
            "",
            evaluation.rrf_baseline[MAP_AT_12],
            record.rrf_map_at_12,
            record.metrics_source,
        )
    )
    rows.append(
        Comparison(
            REPORTED,
            "deduplicated_candidates",
            UNION,
            float(feature_rows),
            float(record.deduplicated_candidates),
            record.recall_source,
        )
    )
    rows.append(
        Comparison(
            REPORTED,
            "queries",
            "",
            float(evaluation.n_queries),
            float(record.n_queries),
            record.metrics_source,
        )
    )
    return rows


def structural_comparisons(metrics: Mapping[str, float]) -> list[Comparison]:
    """The two structural checks of docs/mlops.md §8.3, from the feature step's own record.

    Args:
        metrics: The metrics of step O2 of the week's tick: "structure/feature_rows",
            "structure/deduplicated_candidates", "structure/duplicate_keys".
    """
    source = "the feature step of the same tick (weeks/{anchor}/features/step.json)"
    difference = metrics["structure/feature_rows"] - metrics["structure/deduplicated_candidates"]
    return [
        Comparison(
            STRUCTURAL, "feature_rows_minus_deduplicated_candidates", "", difference, 0.0, source
        ),
        Comparison(
            STRUCTURAL, "duplicate_keys", "", metrics["structure/duplicate_keys"], 0.0, source
        ),
    ]


def pair_overlap(ours: pl.DataFrame, recorded: pl.DataFrame) -> tuple[float, float, int]:
    """How much of a channel's stage-1 candidates the replay's candidates hold.

    Args:
        ours: The replay's candidates: query_id, article_idx, score.
        recorded: The stage-1 candidates of the same week and channel.

    Returns:
        (share of the recorded (query, article) pairs that are also ours, share
        of those common pairs whose score is bit-identical, number of our pairs).
    """
    common = recorded.select(*PAIR, pl.col("score").alias("recorded_score")).join(
        ours.select(*PAIR, "score"), on=PAIR, how="inner"
    )
    share = common.height / recorded.height
    same_score = common.filter(pl.col("score") == pl.col("recorded_score")).height
    identical = same_score / common.height if common.height > 0 else 0.0
    return share, identical, ours.height


def candidate_comparisons(paths: ProjectPaths, layout: StateLayout, week: Fold) -> list[Comparison]:
    """Pair overlap and score equality of every channel with the stage-1 candidate files."""
    rows = []
    for channel in ALL_CHANNELS:
        recorded_file = paths.candidates_file(week.name, channel)
        ours_file = layout.step_dir(week.anchor, recall_step_dir(channel)) / "candidates.parquet"
        recorded = pl.read_parquet(recorded_file, columns=[*PAIR, "score"])
        ours = pl.read_parquet(ours_file, columns=[*PAIR, "score"])
        share, identical, n_ours = pair_overlap(ours, recorded)
        source = recorded_file.relative_to(paths.project_root).as_posix()
        rows.append(Comparison(REPORTED, "recorded_pairs_present", channel, share, 1.0, source))
        rows.append(
            Comparison(
                REPORTED, "common_pairs_with_identical_score", channel, identical, 1.0, source
            )
        )
        rows.append(
            Comparison(
                REPORTED, "candidate_rows", channel, float(n_ours), float(recorded.height), source
            )
        )
    return rows


def compare_week(
    paths: ProjectPaths, layout: StateLayout, week: Fold, lock: RankerLock
) -> list[Comparison]:
    """Every comparison of one closed week (see the module docstring).

    Raises:
        FileNotFoundError: If the week is not closed (step C2 wrote no metrics).
    """
    week_dir = layout.week_dir(week.anchor)
    metrics_file = week_dir / "evaluation" / "metrics.json"
    if not metrics_file.is_file():
        raise FileNotFoundError(f"{metrics_file} not found: week {week.name} is not closed")
    evaluation = read_week_evaluation(metrics_file)
    record = read_week_record(paths, week, lock)
    features = read_step_output(week_dir / "features")
    structure = {metric.name: metric.value for metric in features.metrics}
    feature_rows = int(structure["structure/feature_rows"])
    return [
        *metric_comparisons(evaluation, record, feature_rows),
        *structural_comparisons(structure),
        *candidate_comparisons(paths, layout, week),
    ]


# ---------------------------------------------------------------- the pooled MAP@12


def pooled(values: Sequence[tuple[int, float]]) -> float:
    """Query-weighted mean of weekly scores: sum(n x score) / sum(n) (proposal §4.1).

    Args:
        values: (queries of the week, the week's score) for every week.

    Raises:
        ValueError: If there is no query at all.
    """
    total = sum(n_queries for n_queries, _ in values)
    if total <= 0:
        raise ValueError("the weeks have no queries")
    return math.fsum(n_queries * score for n_queries, score in values) / total


def pooled_comparison(
    paths: ProjectPaths, layout: StateLayout, weeks: Sequence[Fold], lock: RankerLock
) -> Comparison:
    """The pooled MAP@12 over the given weeks against the same pooling of their records."""
    ours, recorded, sources = [], [], set()
    for week in weeks:
        evaluation = read_week_evaluation(
            layout.week_dir(week.anchor) / "evaluation" / "metrics.json"
        )
        record = read_week_record(paths, week, lock)
        ours.append((evaluation.n_queries, evaluation.ranker[MAP_AT_12]))
        recorded.append((record.n_queries, record.map_at_12))
        sources.add(record.metrics_source)
    subject = " ".join(week.name for week in weeks)
    return Comparison(
        ACCEPTANCE,
        f"pooled_{MAP_AT_12}",
        subject,
        pooled(ours),
        pooled(recorded),
        " + ".join(sorted(sources)),
    )


# ---------------------------------------------------------------- records


def comparison_row(
    week: str, anchor: str, comparison: Comparison, threshold: float, git_commit: str
) -> dict[str, str]:
    """A comparison as a row of results/m12_replay/comparison.csv (floats with repr())."""
    inside = is_inside(comparison, threshold)
    return {
        "week": week,
        "anchor": anchor,
        "kind": comparison.kind,
        "quantity": comparison.quantity,
        "subject": comparison.subject,
        "value": repr(comparison.value),
        "record": repr(comparison.record),
        "relative_difference": repr(comparison.relative_difference),
        "threshold": repr(threshold) if comparison.kind == ACCEPTANCE else NOT_JUDGED,
        "inside": NOT_JUDGED if inside is None else str(inside),
        "record_source": comparison.source,
        "git_commit": git_commit,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }


def compared_weeks(table: Path) -> set[str]:
    """The weeks that already have rows in the comparison table."""
    return {row["week"] for row in read_rows(table, COMPARISON_COLUMNS)}


def append_comparisons(
    table: Path,
    week: str,
    anchor: str,
    comparisons: Sequence[Comparison],
    threshold: float,
    git_commit: str,
) -> None:
    """Append the rows of one week (or of the pooled comparison).

    Raises:
        ValueError: If the week already has rows: a week is compared once.
    """
    if week in compared_weeks(table):
        raise ValueError(f"{table} already has rows for {week}; a week is compared once")
    for comparison in comparisons:
        append_row(
            table,
            COMPARISON_COLUMNS,
            comparison_row(week, anchor, comparison, threshold, git_commit),
        )


def outside(comparisons: Sequence[Comparison], threshold: float) -> list[Comparison]:
    """The acceptance and structural comparisons that do not meet their criterion."""
    return [comparison for comparison in comparisons if is_inside(comparison, threshold) is False]


# ---------------------------------------------------------------- the R6 chain and the serve tick


def copy_chain_records(layout: StateLayout, anchors: Sequence[date], table: Path) -> int:
    """Append the chain records of the given closed weeks to results/m12_replay/twotower_chain.csv.

    The state keeps one record file per week (weeks/{anchor}/twotower/chain_record.csv);
    this copies their rows into one committed table, each fold once.

    Returns:
        The number of rows appended.
    """
    appended = 0
    for anchor in anchors:
        source = layout.week_dir(anchor) / "twotower" / "chain_record.csv"
        if not source.is_file():
            continue
        rows = read_table(source)
        columns = tuple(rows[0]) if rows else ()
        known = {row["fold"] for row in read_rows(table, columns)} if rows else set()
        for row in rows:
            if row["fold"] not in known:
                append_row(table, columns, row)
                appended += 1
    return appended


def top_lists(top: pl.DataFrame) -> pl.DataFrame:
    """One row per query: its articles in rank order (query_id, articles)."""
    return (
        top.sort("query_id", "rank")
        .group_by("query_id", maintain_order=True)
        .agg(pl.col("article_idx").alias("articles"))
    )


def serve_overlap(ours: pl.DataFrame, reference: pl.DataFrame) -> dict[str, float]:
    """How close two Top-K sets of the same queries are.

    Args:
        ours: The serve tick's Top-K: query_id, rank, article_idx.
        reference: The reference Top-K of the same query ids (the Kaggle run's).

    Returns:
        n_customers, n_customers_in_reference, identical_list_share (the same
        articles in the same order), same_articles_share (the same set),
        mean_overlap (mean share of our K articles that are in the reference's
        list), first_article_identical_share. All shares are over the customers
        both sets hold.
    """
    joined = top_lists(ours).join(
        top_lists(reference).rename({"articles": "reference_articles"}), on="query_id", how="inner"
    )
    both = joined.with_columns(
        (pl.col("articles") == pl.col("reference_articles")).alias("identical"),
        pl.col("articles").list.set_intersection("reference_articles").list.len().alias("common"),
        pl.col("articles").list.len().alias("k"),
        (pl.col("articles").list.first() == pl.col("reference_articles").list.first()).alias(
            "first"
        ),
    )
    n = both.height
    if n == 0:
        raise ValueError("the two prediction sets have no query in common")
    return {
        "n_customers": float(ours["query_id"].n_unique()),
        "n_customers_in_reference": float(n),
        "identical_list_share": both.filter(pl.col("identical")).height / n,
        "same_articles_share": both.filter(pl.col("common") == pl.col("k")).height / n,
        "mean_overlap": float((both["common"] / both["k"]).to_numpy().mean()),
        "first_article_identical_share": both.filter(pl.col("first")).height / n,
    }


def read_reference_top(reference_dir: Path) -> pl.DataFrame:
    """The saved Top-K of the Kaggle late submission, all shards (query_id, rank, article_idx).

    Raises:
        FileNotFoundError: If the folder holds no shard.
    """
    files = sorted((reference_dir / "shards").glob("*/top.parquet"))
    if not files:
        raise FileNotFoundError(f"no shards/*/top.parquet under {reference_dir}")
    columns = ["query_id", "rank", "article_idx"]
    return pl.concat([pl.read_parquet(file, columns=columns) for file in files])


def serve_overlap_row(
    paths: ProjectPaths, layout: StateLayout, anchor: date, git_commit: str
) -> dict[str, str]:
    """The row of results/m12_replay/serve_overlap.csv for the serve tick at `anchor`."""
    ours = pl.read_parquet(
        layout.week_dir(anchor) / "scores" / "top.parquet",
        columns=["query_id", "rank", "article_idx"],
    )
    reference_dir = paths.kaggle_reference_dir
    values = serve_overlap(ours, read_reference_top(reference_dir))
    return {
        "anchor": anchor.isoformat(),
        "n_customers": str(int(values["n_customers"])),
        "n_customers_in_reference": str(int(values["n_customers_in_reference"])),
        "identical_list_share": repr(values["identical_list_share"]),
        "same_articles_share": repr(values["same_articles_share"]),
        "mean_overlap": repr(values["mean_overlap"]),
        "first_article_identical_share": repr(values["first_article_identical_share"]),
        "reference": reference_dir.relative_to(paths.project_root).as_posix(),
        "git_commit": git_commit,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }


def week_mode_of(layout: StateLayout, anchor: date) -> str | None:
    """The mode a week's tick was started in; None if it never started."""
    tick_file = layout.week_dir(anchor) / "tick.json"
    if not tick_file.is_file():
        return None
    return str(json.loads(tick_file.read_text(encoding="utf-8"))["mode"])


# ---------------------------------------------------------------- the whole replay


@dataclass(frozen=True)
class ReplayComparison:
    """What one run of the comparison did.

    Attributes:
        n_weeks: `eval` weeks of the replay (configs/m12_pipeline.yaml).
        closed: Names of the weeks the pipeline has closed so far.
        compared_now: Names of the weeks compared by this run ("pooled" included).
        failed: (week, comparison) of every acceptance quantity outside its
            threshold and every failed structural check, of this run.
        chain_records_copied: Rows appended to the chain table.
        serve_row: The serve tick's overlap row if it was written now, else None.
    """

    n_weeks: int
    closed: tuple[str, ...]
    compared_now: tuple[str, ...]
    failed: tuple[tuple[str, Comparison], ...]
    chain_records_copied: int
    serve_row: dict[str, str] | None


def compare_replay(
    paths: ProjectPaths,
    weeks: Sequence[Fold],
    serve_anchor: date,
    lock: RankerLock,
    threshold: float,
    git_commit: str,
) -> ReplayComparison:
    """Compare every closed, not yet compared week; then the pooled MAP@12 and the serve tick.

    Args:
        paths: The project paths (state, stage-1 records, the replay's tables).
        weeks: The `eval` weeks of the replay, in order.
        serve_anchor: The anchor of the replay's `serve` tick.
        lock: The Ranker lock (selects the Ranker's rows of the records).
        threshold: The largest accepted relative difference (the platform record's).
        git_commit: HEAD of this run, written into every row.

    Returns:
        What was compared and what is outside its criterion.
    """
    layout = StateLayout(paths.pipeline_state_dir)
    table = paths.replay_table_file("comparison")
    closed = [week for week in weeks if layout.closed_file(week.anchor).is_file()]
    compared_now: list[str] = []
    failed: list[tuple[str, Comparison]] = []
    for week in closed:
        if week.name in compared_weeks(table):
            continue
        comparisons = compare_week(paths, layout, week, lock)
        anchor = week.anchor.isoformat()
        append_comparisons(table, week.name, anchor, comparisons, threshold, git_commit)
        compared_now.append(week.name)
        failed += [(week.name, comparison) for comparison in outside(comparisons, threshold)]
    if len(closed) == len(weeks) and POOLED_WEEK not in compared_weeks(table):
        pooled_row = pooled_comparison(paths, layout, weeks, lock)
        append_comparisons(table, POOLED_WEEK, "", [pooled_row], threshold, git_commit)
        compared_now.append(POOLED_WEEK)
        failed += [(POOLED_WEEK, comparison) for comparison in outside([pooled_row], threshold)]
    chain_table = paths.replay_table_file("twotower_chain")
    copied = copy_chain_records(layout, [week.anchor for week in closed], chain_table)
    serve_row = None
    overlap_table = paths.replay_table_file("serve_overlap")
    served = layout.published_file(serve_anchor).is_file()
    if served and not read_rows(overlap_table, SERVE_OVERLAP_COLUMNS):
        serve_row = serve_overlap_row(paths, layout, serve_anchor, git_commit)
        append_row(overlap_table, SERVE_OVERLAP_COLUMNS, serve_row)
    return ReplayComparison(
        n_weeks=len(weeks),
        closed=tuple(week.name for week in closed),
        compared_now=tuple(compared_now),
        failed=tuple(failed),
        chain_records_copied=copied,
        serve_row=serve_row,
    )
