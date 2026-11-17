"""Independent cross-check of artifacts/features/{fold}/ (see README.md).

Usage:
    uv run python scripts/checks/check_features.py --fold F1 [--n-rows 1000] [--seed 0]

Nothing from hmrec.features, hmrec.fusion, hmrec.data.asof, hmrec.data.day_index,
hmrec.data.histories or hmrec.vectors is used, so a bug there cannot hide itself.
Dates are re-derived with plain datetime arithmetic (the documented exception of
scripts/checks/), and every feature is recomputed in the simplest way: plain
Python loops over the purchase lines of one customer or one article.

Whole table (key columns only):
    1. rows are sorted by (query_id, article_idx); no pair repeats; a query is
       never split over two part files;
    2. the row count equals the number of distinct (query_id, article_idx) over
       the six candidate files, counted here.
Random sample of --n-rows rows: 25 features are recomputed (at least 2 per group;
6 of them are among the 15 features added on 2026-11-17, marked +)
    A: pop_rank, rep_score, n_channels, best_rank, rrf_score
    B: tt_dot, pop_bucket_score, txt_sim_hist, +tt_sim_max_hist
    C: item_cnt_1w, item_uniq_buyers_4w, item_days_since_last_sale, item_price_mean_4w,
       +item_discount_ratio
    D: age, user_cnt_52w, user_days_since_last_purchase, user_uniq_items_52w,
       +user_purchase_days_52w
    E: ui_cnt_item, ui_days_since_item, ui_share_ptype_52w, +ui_decay_ptype,
       +ui_days_since_section, +ui_attr_match_cnt
and compared with the wide table. Counts, days and ranks must be equal; values
that involve float arithmetic must agree to a relative 1e-5 (the table stores
float32; this script computes in float64). Missing must be missing on both sides.

Any failure raises ValueError (non-zero exit). No label is read.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import date, timedelta  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import yaml  # noqa: E402

from hmrec.config import load_base_config, load_preprocess_config  # noqa: E402
from hmrec.data.metadata import scan_processed_parquet  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import ProjectPaths, get_paths  # noqa: E402

LOGGER = get_logger("check_features")
KEYS = ["query_id", "article_idx"]
# Written out again here on purpose (not read from configs/features.yaml): proposal §8.
CHANNELS = {
    "r1_popularity": "pop",
    "r2_repurchase": "rep",
    "r3_sibling": "sib",
    "r4_itemcf": "icf",
    "r5_wals": "wals",
    "r6_twotower": "tt",
}
RRF_K = 60
HISTORY_LENGTH = 20
WEEK, FOUR_WEEKS, YEAR = 7, 28, 364
EXACT = ("pop_rank", "n_channels", "best_rank", "item_cnt_1w", "item_uniq_buyers_4w")
EXACT += ("item_days_since_last_sale", "age", "user_cnt_52w", "user_days_since_last_purchase")
EXACT += ("user_uniq_items_52w", "ui_cnt_item", "ui_days_since_item")
CLOSE = ("rep_score", "rrf_score", "tt_dot", "pop_bucket_score", "txt_sim_hist")
CLOSE += ("item_price_mean_4w", "ui_share_ptype_52w")
# Six of the features added on 2026-11-17.
EXACT += ("user_purchase_days_52w", "ui_days_since_section", "ui_attr_match_cnt")
CLOSE += ("tt_sim_max_hist", "item_discount_ratio", "ui_decay_ptype")
DECAY_TAU_DAYS = 30.0
CATEGORY_FIELDS = (
    "product_type_no",
    "product_group_name",
    "garment_group_no",
    "index_group_no",
    "section_no",
    "colour_group_code",
)
RELATIVE_TOLERANCE = 1e-5
ABSOLUTE_TOLERANCE = 1e-6

Line = tuple[date, int, int, float]  # (purchase date, customer_idx, article_idx, price)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--n-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_anchor(paths: ProjectPaths, fold: str) -> date:
    """The fold's anchor, read directly from the YAML (not via hmrec.data.folds)."""
    manifest = yaml.safe_load(paths.fold_manifest_file.read_text(encoding="utf-8"))
    for entry in manifest["folds"]:
        if entry["name"] == fold:
            return entry["anchor"]
    raise ValueError(f"fold {fold!r} is not in {paths.fold_manifest_file}")


# ---------------------------------------------------------------- whole-table checks


def check_keys(paths: ProjectPaths, fold: str) -> pl.DataFrame:
    """Checks 1 and 2 of the module docstring; returns every key pair in file order."""
    part_files = sorted(paths.features_dir(fold).glob("part-*.parquet"))
    if not part_files:
        raise ValueError(f"no part files in {paths.features_dir(fold)}")
    parts = [pl.read_parquet(path, columns=KEYS) for path in part_files]
    previous_last_query = -1
    for path, part in zip(part_files, parts, strict=True):
        if not part.equals(part.sort(KEYS)):
            raise ValueError(f"{path.name} is not sorted by (query_id, article_idx)")
        first_query, last_query = part["query_id"][0], part["query_id"][-1]
        if first_query <= previous_last_query:
            raise ValueError(
                f"{path.name} starts with query {first_query}: a query spans two files"
            )
        previous_last_query = last_query
    keys = pl.concat(parts)
    if keys.is_duplicated().any():
        raise ValueError("a (query_id, article_idx) pair repeats in the wide table")
    candidate_pairs = pl.concat(
        [pl.read_parquet(paths.candidates_file(fold, c), columns=KEYS) for c in CHANNELS]
    ).unique()
    if keys.height != candidate_pairs.height:
        raise ValueError(
            f"wide table has {keys.height} rows, the six candidate files have "
            f"{candidate_pairs.height} distinct pairs"
        )
    LOGGER.info(
        "%s: %d part files, %d rows = distinct candidate pairs; sorted, no repeats, no "
        "query split over files",
        fold,
        len(part_files),
        keys.height,
    )
    return keys


# ---------------------------------------------------------------- data for the sample


def sample_rows(
    paths: ProjectPaths, fold: str, keys: pl.DataFrame, n_rows: int, seed: int
) -> pl.DataFrame:
    """`n_rows` random rows of the wide table (all columns) with their customer_idx."""
    chosen = np.sort(np.random.default_rng(seed).choice(keys.height, size=n_rows, replace=False))
    wanted = keys[chosen]
    table = pl.scan_parquet(str(paths.features_dir(fold) / "part-*.parquet"))
    rows = table.join(wanted.lazy(), on=KEYS, how="semi").collect()
    queries = pl.read_parquet(paths.queries_file(fold), columns=["query_id", "customer_idx"])
    return rows.join(queries, on="query_id", how="left").sort(KEYS)


def load_lines(
    paths: ProjectPaths, day_zero: date, anchor: date, column: str, wanted: list[int]
) -> dict[int, list[Line]]:
    """Purchase lines BEFORE the anchor of the wanted customers (or articles), keyed by that id.

    The date of a line is day_zero + t_dat_day days, computed here with timedelta.
    """
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    scan = scan_processed_parquet(
        paths.transactions_file, paths.processed_metadata_file, preprocess
    )
    table = scan.filter(pl.col(column).is_in(wanted)).collect()
    lines: dict[int, list[Line]] = defaultdict(list)
    for customer, article, day, price, _channel in table.select(
        "customer_idx", "article_idx", "t_dat_day", "price", "sales_channel_id"
    ).iter_rows():
        purchase_date = day_zero + timedelta(days=int(day))
        if purchase_date < anchor:  # the anchor day itself belongs to the label week
            key = customer if column == "customer_idx" else article
            lines[key].append((purchase_date, customer, article, float(price)))
    return lines


def in_window(line: Line, anchor: date, days: int) -> bool:
    """True if the line's date is in [anchor - days, anchor)."""
    return anchor - timedelta(days=days) <= line[0] < anchor


# ---------------------------------------------------------------- naive features


def source_features(
    paths: ProjectPaths, fold: str, sample: pl.DataFrame
) -> dict[tuple[int, int], dict[str, float]]:
    """A group, from the six candidate files: ranks, rep_score, n_channels, best_rank, rrf_score."""
    result: dict[tuple[int, int], dict[str, float]] = {}
    ranks: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    rep_score: dict[tuple[int, int], float] = {}
    for channel, prefix in CHANNELS.items():
        rows = pl.read_parquet(paths.candidates_file(fold, channel)).join(
            sample.select(KEYS), on=KEYS, how="semi"
        )
        for query, article, rank, score in rows.select(*KEYS, "rank", "score").iter_rows():
            ranks[(query, article)][prefix] = rank
            if prefix == "rep":
                rep_score[(query, article)] = score
    for query, article in sample.select(KEYS).iter_rows():
        own = ranks[(query, article)]
        result[(query, article)] = {
            "pop_rank": own.get("pop", math.nan),
            "rep_score": rep_score.get((query, article), math.nan),
            "n_channels": len(own),
            "best_rank": min(own.values()),
            "rrf_score": sum(1.0 / (RRF_K + rank) for rank in own.values()),
        }
    return result


def last_purchases(lines: list[Line]) -> list[int]:
    """The customer's last 20 purchases: one per (day, article), newest day first, then article."""
    distinct = sorted(
        {(line[0], line[2]) for line in lines}, key=lambda pair: (-pair[0].toordinal(), pair[1])
    )
    return [article for _, article in distinct[:HISTORY_LENGTH]]


def unit(vector: np.ndarray) -> np.ndarray:
    """The vector divided by its length, in float64."""
    vector = vector.astype(np.float64)
    return vector / np.linalg.norm(vector)


class Sources:
    """Files read once: R6 vectors, text embeddings, R1 bucket scores, customers, articles."""

    def __init__(self, paths: ProjectPaths, fold: str) -> None:
        """Load everything with np.load / pl.read_parquet (no hmrec.vectors)."""
        vectors = paths.twotower_vectors_dir(fold)
        self.tt_users = np.load(vectors / "user_customer_idx.npy")
        self.tt_user_vectors = np.load(vectors / "user_vectors.npy")
        self.tt_items = np.load(vectors / "item_article_idx.npy")
        self.tt_item_vectors = np.load(vectors / "item_vectors.npy")
        self.text = np.load(paths.text_embeddings_by_idx_file)
        buckets = pl.read_parquet(paths.r1_bucket_scores_file(fold))
        self.bucket_scores = {
            (bucket, article): score for bucket, article, score in buckets.iter_rows()
        }
        preprocess = load_preprocess_config(paths.preprocess_config_file)
        metadata = paths.processed_metadata_file
        customers = scan_processed_parquet(paths.customers_file, metadata, preprocess).collect()
        self.age = dict(zip(customers["customer_idx"], customers["age"], strict=True))
        self.age_bucket = dict(zip(customers["customer_idx"], customers["age_bucket"], strict=True))
        articles = scan_processed_parquet(paths.articles_file, metadata, preprocess).collect()
        # Field name -> {article_idx -> value}, for the six category fields.
        self.fields = {
            name: dict(zip(articles["article_idx"], articles[name], strict=True))
            for name in CATEGORY_FIELDS
        }
        self.product_type = self.fields["product_type_no"]

    def tt_dot(self, customer: int, article: int) -> float:
        """Dot product of the saved R6 user and article vectors."""
        user = self.tt_user_vectors[int(np.searchsorted(self.tt_users, customer))]
        item = self.tt_item_vectors[int(np.searchsorted(self.tt_items, article))]
        return float(np.dot(user.astype(np.float64), item.astype(np.float64)))

    def tt_sim_max_hist(self, history: list[int], article: int) -> float:
        """Largest dot product of the article's R6 vector with a history article's R6 vector."""
        if not history:
            return math.nan
        candidate = self.tt_item_vectors[int(np.searchsorted(self.tt_items, article))]
        similarities = [
            float(
                np.dot(
                    candidate.astype(np.float64),
                    self.tt_item_vectors[int(np.searchsorted(self.tt_items, past))].astype(
                        np.float64
                    ),
                )
            )
            for past in history
        ]
        return max(similarities)

    def attr_match_cnt(self, history: list[int], article: int) -> float:
        """For how many of the six category fields the article's value occurs in the history."""
        if not history:
            return math.nan
        matches = 0
        for values in self.fields.values():
            if values[article] in {values[past] for past in history}:
                matches += 1
        return matches

    def txt_sim_hist(self, history: list[int], article: int) -> float:
        """Cosine of the article's text vector with the mean text vector of the history."""
        if not history:
            return math.nan
        mean = np.mean(self.text[history].astype(np.float64), axis=0)
        return float(np.dot(unit(mean), unit(self.text[article])))


def naive_row(
    customer: int,
    article: int,
    anchor: date,
    user_lines: list[Line],
    item_lines: list[Line],
    sources: Sources,
) -> dict[str, float]:
    """B, C, D and E features of one (customer, article) with plain loops."""
    age = sources.age[customer]
    item_week = [line for line in item_lines if in_window(line, anchor, WEEK)]
    item_four = [line for line in item_lines if in_window(line, anchor, FOUR_WEEKS)]
    user_year = [line for line in user_lines if in_window(line, anchor, YEAR)]
    same_item = [line for line in user_lines if line[2] == article]
    same_type = [
        line for line in user_year if sources.product_type[line[2]] == sources.product_type[article]
    ]
    return {
        "tt_dot": sources.tt_dot(customer, article),
        "pop_bucket_score": sources.bucket_scores.get((sources.age_bucket[customer], article), 0.0),
        "txt_sim_hist": sources.txt_sim_hist(last_purchases(user_lines), article),
        "item_cnt_1w": len(item_week),
        "item_uniq_buyers_4w": len({line[1] for line in item_four}),
        "item_days_since_last_sale": (anchor - max(line[0] for line in item_lines)).days
        if item_lines
        else math.nan,
        "item_price_mean_4w": sum(line[3] for line in item_four) / len(item_four)
        if item_four
        else math.nan,
        "age": math.nan if age is None else age,
        "user_cnt_52w": len(user_year),
        "user_days_since_last_purchase": (anchor - max(line[0] for line in user_lines)).days
        if user_lines
        else math.nan,
        "user_uniq_items_52w": len({line[2] for line in user_year}),
        "ui_cnt_item": len(same_item),
        "ui_days_since_item": (anchor - max(line[0] for line in same_item)).days
        if same_item
        else math.nan,
        "ui_share_ptype_52w": len(same_type) / len(user_year) if user_year else math.nan,
    }


def naive_added_features(
    article: int,
    anchor: date,
    user_lines: list[Line],
    item_lines: list[Line],
    sources: Sources,
) -> dict[str, float]:
    """Six of the features added on 2026-11-17, with plain loops."""
    history = last_purchases(user_lines)
    item_week = [line[3] for line in item_lines if in_window(line, anchor, WEEK)]
    item_year = [line[3] for line in item_lines if in_window(line, anchor, YEAR)]
    user_year = [line for line in user_lines if in_window(line, anchor, YEAR)]
    section = sources.fields["section_no"]
    same_section = [line for line in user_lines if section[line[2]] == section[article]]
    same_type = [
        line
        for line in user_lines
        if sources.product_type[line[2]] == sources.product_type[article]
    ]
    discount = math.nan
    if item_week and item_year:
        discount = (sum(item_week) / len(item_week)) / (sum(item_year) / len(item_year))
    return {
        "tt_sim_max_hist": sources.tt_sim_max_hist(history, article),
        "item_discount_ratio": discount,
        "user_purchase_days_52w": len({line[0] for line in user_year}),
        # exp(-age_days / 30) summed over ALL the user's lines of the same product type.
        "ui_decay_ptype": sum(
            math.exp(-(anchor - line[0]).days / DECAY_TAU_DAYS) for line in same_type
        ),
        "ui_days_since_section": (anchor - max(line[0] for line in same_section)).days
        if same_section
        else math.nan,
        "ui_attr_match_cnt": sources.attr_match_cnt(history, article),
    }


# ---------------------------------------------------------------- comparison


def same_value(name: str, stored: float, naive: float) -> bool:
    """True if the wide table's value agrees with the naive one (see the module docstring)."""
    if math.isnan(stored) or math.isnan(naive):
        return math.isnan(stored) and math.isnan(naive)
    if name in EXACT:
        return stored == naive
    return math.isclose(stored, naive, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE)


def main() -> None:
    """Run the whole-table checks, then recompute the sampled rows and compare."""
    args = parse_args()
    paths = get_paths()
    setup_logging(logging.INFO)
    base = load_base_config(paths.base_config_file)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(base.num_threads))
    day_zero = load_preprocess_config(paths.preprocess_config_file).day_zero
    anchor = read_anchor(paths, args.fold)
    keys = check_keys(paths, args.fold)
    sample = sample_rows(paths, args.fold, keys, args.n_rows, args.seed)
    customers = sample["customer_idx"].unique().to_list()
    articles = sample["article_idx"].unique().to_list()
    user_lines = load_lines(paths, day_zero, anchor, "customer_idx", customers)
    item_lines = load_lines(paths, day_zero, anchor, "article_idx", articles)
    sources = Sources(paths, args.fold)
    from_candidates = source_features(paths, args.fold, sample)
    mismatches: dict[str, int] = dict.fromkeys((*EXACT, *CLOSE), 0)
    for row in sample.iter_rows(named=True):
        customer, article = row["customer_idx"], row["article_idx"]
        naive = naive_row(
            customer, article, anchor, user_lines[customer], item_lines[article], sources
        )
        naive.update(from_candidates[(row["query_id"], article)])
        naive.update(
            naive_added_features(
                article, anchor, user_lines[customer], item_lines[article], sources
            )
        )
        for name in mismatches:
            if not same_value(name, float(row[name]), float(naive[name])):
                mismatches[name] += 1
                LOGGER.error(
                    "query %d article %d %s: table %r, naive %r",
                    row["query_id"],
                    article,
                    name,
                    row[name],
                    naive[name],
                )
    LOGGER.info(
        "%s: %d sampled rows x %d features compared", args.fold, sample.height, len(mismatches)
    )
    failed = {name: count for name, count in mismatches.items() if count > 0}
    if failed:
        raise ValueError(f"features differ from the naive recomputation: {failed}")
    LOGGER.info("all %d features agree on all %d rows", len(mismatches), sample.height)


if __name__ == "__main__":
    main()
