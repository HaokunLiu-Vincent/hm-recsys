"""The submission file of a Kaggle run: build it from the shards' Top-K, write it, check it.

File format (the competition's sample_submission.csv):

    customer_id,prediction
    <64-character customer id>,<12 article ids separated by one space>

    - one row per customer, the SAME customers in the SAME row order as the
      sample file (a rehearsal: the sample file's rows of its own customers);
    - an article id is the original 10-character string with its leading zero
      (article_idx is mapped back through data/processed/article_id_map.parquet);
    - exactly `k` (12) DISTINCT article ids per row, best first.

check_submission_file reads the WRITTEN file back and checks every rule; the run
stops if one fails. The gzip copy must unpack to exactly the bytes of the CSV.

Also here: the candidate statistics of a run (from the shards' query_stats files).

No label, no model: this module imports neither torch nor LightGBM.
"""

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from hmrec.locks import file_sha256
from hmrec.logging_utils import get_logger

LOGGER = get_logger("kaggle.submission")

CUSTOMER_ID = "customer_id"
ARTICLE_ID = "article_id"
PREDICTION = "prediction"
QUERY_ID = "query_id"
CUSTOMER_IDX = "customer_idx"
ARTICLE_IDX = "article_idx"
RANK = "rank"
HEADER = f"{CUSTOMER_ID},{PREDICTION}"
SEPARATOR = " "
# An H&M article id: 10 digits, the first one a zero (it is lost if the id is read as a number).
ARTICLE_ID_PATTERN = r"^0[0-9]{9}$"
# Bytes copied at a time when the gzip copy is written or read back.
COPY_BYTES = 1024 * 1024


# ---------------------------------------------------------------- building


def check_complete_top(top: pl.DataFrame, queries: pl.DataFrame, k: int) -> None:
    """Fail unless every query has exactly the ranks 1..k, each with a distinct article.

    Args:
        top: One row per (query_id, rank): query_id, rank, article_idx.
        queries: The run's queries (query_id).
        k: Predictions per customer.

    Raises:
        ValueError: If a query is missing, has another number of rows or ranks, or
            repeats an article.
    """
    per_query = top.group_by(QUERY_ID).agg(
        pl.len().alias("n"),
        pl.col(RANK).min().alias("first"),
        pl.col(RANK).max().alias("last"),
        pl.col(RANK).n_unique().alias("n_ranks"),
        pl.col(ARTICLE_IDX).n_unique().alias("n_articles"),
    )
    if (
        per_query.height != queries.height
        or per_query.join(queries, on=QUERY_ID, how="anti").height
    ):
        raise ValueError(
            f"the Top-{k} holds {per_query.height} queries, the run has {queries.height}"
        )
    complete = (
        (pl.col("n") == k)
        & (pl.col("first") == 1)
        & (pl.col("last") == k)
        & (pl.col("n_ranks") == k)
        & (pl.col("n_articles") == k)
    )
    broken = per_query.filter(~complete)
    if broken.height > 0:
        examples = broken[QUERY_ID].sort().head(5).to_list()
        raise ValueError(
            f"{broken.height} queries do not have exactly {k} distinct articles at ranks "
            f"1..{k}; e.g. query_id {examples}"
        )


def predictions_table(
    top: pl.DataFrame,
    queries: pl.DataFrame,
    customer_ids: pl.DataFrame,
    article_ids: pl.DataFrame,
    k: int,
) -> pl.DataFrame:
    """One row per customer: its id and its k article ids as one space-separated string.

    Args:
        top: The Top-k of every query of the run: query_id, rank, article_idx.
        queries: The run's queries: query_id, customer_idx.
        customer_ids: customer_idx -> customer_id (the original string).
        article_ids: article_idx -> article_id (the original 10-character string).
        k: Predictions per customer.

    Returns:
        Columns customer_id (str), prediction (str); one row per query, in no
        particular order (order_like_sample sorts them).

    Raises:
        ValueError: If the Top-k is not complete, or an id is missing from a map.
    """
    check_complete_top(top, queries, k)
    with_ids = top.select(QUERY_ID, RANK, ARTICLE_IDX).join(article_ids, on=ARTICLE_IDX, how="left")
    if with_ids[ARTICLE_ID].null_count() > 0:
        raise ValueError("an article_idx of the Top-k is not in the article id map")
    per_query = with_ids.group_by(QUERY_ID).agg(
        # Best first: the ids are joined in rank order.
        pl.col(ARTICLE_ID).sort_by(RANK).str.join(SEPARATOR).alias(PREDICTION)
    )
    with_customer = per_query.join(queries.select(QUERY_ID, CUSTOMER_IDX), on=QUERY_ID).join(
        customer_ids, on=CUSTOMER_IDX, how="left"
    )
    if with_customer[CUSTOMER_ID].null_count() > 0:
        raise ValueError("a customer_idx of the run is not in the customer id map")
    return with_customer.select(CUSTOMER_ID, PREDICTION)


def order_like_sample(predictions: pl.DataFrame, sample_customer_ids: pl.Series) -> pl.DataFrame:
    """Put the rows in the order of the given customer ids (the sample file's order).

    Args:
        predictions: customer_id, prediction; one row per customer.
        sample_customer_ids: The customer ids the file must hold, in file order.

    Raises:
        ValueError: If the two sets of customers differ or a customer repeats.
    """
    wanted = pl.DataFrame({CUSTOMER_ID: sample_customer_ids}).with_row_index("row")
    if wanted[CUSTOMER_ID].n_unique() != wanted.height:
        raise ValueError("the sample customer ids repeat")
    ordered = wanted.join(predictions, on=CUSTOMER_ID, how="left").sort("row")
    if ordered[PREDICTION].null_count() > 0 or predictions.height != wanted.height:
        raise ValueError(
            f"the run predicts {predictions.height} customers, the sample holds {wanted.height}; "
            f"{ordered[PREDICTION].null_count()} sample customers have no prediction"
        )
    return ordered.select(CUSTOMER_ID, PREDICTION)


# ---------------------------------------------------------------- writing


def write_submission(table: pl.DataFrame, csv_file: Path, gzip_file: Path) -> None:
    """Write the submission CSV and its gzip copy. Neither file is ever replaced.

    The gzip copy is written with a fixed header (no file name, no time), so the
    same CSV always gives the same compressed bytes.

    Raises:
        FileExistsError: If one of the files exists.
    """
    for path in (csv_file, gzip_file):
        if path.exists():
            raise FileExistsError(f"{path} already exists; a submission file is never replaced")
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    table.select(CUSTOMER_ID, PREDICTION).write_csv(csv_file, line_terminator="\n")
    with (
        csv_file.open("rb") as source,
        gzip_file.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as target,
    ):
        shutil.copyfileobj(source, target, COPY_BYTES)


# ---------------------------------------------------------------- checking


@dataclass(frozen=True)
class FileCheck:
    """The outcome of the format checks of one written submission file.

    Attributes:
        n_rows: Data rows of the file.
        csv_bytes: Size of the CSV.
        csv_sha256: sha256 of the CSV.
        gzip_bytes: Size of the gzip copy.
        gzip_sha256: sha256 of the gzip copy.
        checks: Check name -> passed, in the order they were made.
    """

    n_rows: int
    csv_bytes: int
    csv_sha256: str
    gzip_bytes: int
    gzip_sha256: str
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        """True if every check passed."""
        return all(self.checks.values())

    @property
    def failed(self) -> list[str]:
        """The names of the checks that failed."""
        return [name for name, ok in self.checks.items() if not ok]


def _first_line(path: Path) -> str:
    """The first line of a text file without its line end."""
    with path.open(encoding="utf-8") as handle:
        return handle.readline().rstrip("\n")


def _gzip_holds(gzip_file: Path, csv_file: Path) -> bool:
    """True if the gzip file unpacks to exactly the bytes of the CSV."""
    with gzip.open(gzip_file, "rb") as packed, csv_file.open("rb") as plain:
        while True:
            left, right = packed.read(COPY_BYTES), plain.read(COPY_BYTES)
            if left != right:
                return False
            if not left:
                return True


def check_submission_file(
    csv_file: Path,
    gzip_file: Path,
    expected_customer_ids: pl.Series,
    known_article_ids: pl.Series,
    k: int,
) -> FileCheck:
    """Read a written submission file back and check every format rule (module docstring).

    Args:
        csv_file: The written CSV.
        gzip_file: Its gzip copy.
        expected_customer_ids: The customer ids the file must hold, in row order.
        known_article_ids: Every article id of the catalog (original strings).
        k: Article ids per row.

    Returns:
        The outcome of every check; the caller stops if FileCheck.passed is False.
    """
    # Every column is read as text: an article id read as a number loses its zero.
    table = pl.read_csv(csv_file, schema_overrides={CUSTOMER_ID: pl.String, PREDICTION: pl.String})
    checks: dict[str, bool] = {}
    checks["header is customer_id,prediction"] = _first_line(csv_file) == HEADER
    checks["same number of rows as the sample"] = table.height == expected_customer_ids.len()
    checks["same customers in the same row order"] = bool(
        checks["same number of rows as the sample"]
        and table[CUSTOMER_ID].equals(expected_customer_ids.rename(CUSTOMER_ID))
    )
    checks["no empty prediction"] = table[PREDICTION].null_count() == 0
    tokens = table.select(pl.col(PREDICTION).fill_null("").str.split(SEPARATOR).alias("ids"))
    checks[f"exactly {k} ids per row"] = bool((tokens["ids"].list.len() == k).all())
    checks[f"{k} distinct ids per row"] = bool((tokens["ids"].list.n_unique() == k).all())
    flat = tokens["ids"].explode()
    checks["every id is 10 digits with a leading zero"] = bool(
        flat.str.contains(ARTICLE_ID_PATTERN).all()
    )
    checks["every id is an article of the catalog"] = bool(
        flat.is_in(known_article_ids.implode()).all()
    )
    checks["the gzip copy unpacks to the same bytes"] = _gzip_holds(gzip_file, csv_file)
    result = FileCheck(
        n_rows=table.height,
        csv_bytes=csv_file.stat().st_size,
        csv_sha256=file_sha256(csv_file),
        gzip_bytes=gzip_file.stat().st_size,
        gzip_sha256=file_sha256(gzip_file),
        checks=checks,
    )
    for name, ok in checks.items():
        LOGGER.info("format check | %s | %s", "ok" if ok else "FAILED", name)
    return result


# ---------------------------------------------------------------- candidate statistics


@dataclass(frozen=True)
class CandidateStatistics:
    """Candidate counts of a run, from the shards' query_stats files.

    Attributes:
        n_customers: Queries (= customers) of the run.
        n_candidate_rows: Deduplicated candidate rows over all customers.
        per_customer_mean: Mean candidates per customer.
        per_customer_p50: Median candidates per customer.
        per_customer_p90: 90th percentile of candidates per customer.
        no_history_share: Share of customers with no purchase before the anchor.
    """

    n_customers: int
    n_candidate_rows: int
    per_customer_mean: float
    per_customer_p50: float
    per_customer_p90: float
    no_history_share: float


def candidate_statistics(query_stats: pl.DataFrame) -> CandidateStatistics:
    """Summarise the per-query statistics of all shards.

    Args:
        query_stats: One row per query of the run: n_candidates (int), has_history (bool).

    Raises:
        ValueError: If the table is empty.
    """
    if query_stats.height == 0:
        raise ValueError("no query statistics: the run has no finished shard")
    counts = query_stats["n_candidates"].to_numpy().astype(np.float64)
    has_history = query_stats["has_history"].to_numpy().astype(np.bool_)
    return CandidateStatistics(
        n_customers=query_stats.height,
        n_candidate_rows=int(counts.sum()),
        per_customer_mean=float(counts.mean()),
        per_customer_p50=float(np.quantile(counts, 0.5)),
        per_customer_p90=float(np.quantile(counts, 0.9)),
        no_history_share=float(1.0 - has_history.mean()),
    )
