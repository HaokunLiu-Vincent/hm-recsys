"""Two-Tower training arrays and device tensors (proposal §5 R6, §6.1).

Training samples
    One positive per (customer_idx, purchase day, article_idx). Buying the same
    article several times on the same day is ONE positive (the extra purchase
    lines are dropped). Samples are sorted by (t_dat_day, customer_idx, article_idx).

Groups (for the multi-positive loss, §5 R6)
    All samples of one customer on one day share a group_id (0, 1, 2, ...).
    Because of the sort order, a group is a block of consecutive rows.

User history
    For a sample on day d: the last `history_length` purchases of that customer
    with t_dat_day < d (the purchase day itself is NOT included, so the answer
    never leaks into the input). There is no lower time limit.
    Purchases are counted the same way as samples: one per (customer, day, article).
    Order: most recent day first; several articles on the same day in ascending
    article_idx, so the result never depends on file order.
    Shorter histories are padded at the END with the padding index n_articles
    and history_mask = False. n_articles is not a real article (real ones are
    0..n_articles-1), so a padded slot can never pass for a purchase, even if
    some code forgets the mask. Arrays: history int32 [N, L], history_mask bool [N, L].

Profile and item features
    Category columns are coded as integers: code 0 = missing (null), codes
    1..K = the distinct values in sorted order. The item feature table holds the
    CLIP image (float32 [n_articles + 1, 512]) and MiniLM text (float32
    [n_articles + 1, 384]) vectors with their masks, and the product_code and §8.4
    category codes (int32 [n_articles + 1, 7]). The LAST row (index n_articles)
    is the padding row: vectors all 0, masks False, codes all 0. The model's ID
    and category embeddings use it as padding_idx (ID: n_articles; categories: 0,
    which no real article has in the item table).

Batches
    Everything is moved to the device once (mps or cpu). A batch is fetched by
    indexing those tensors with sample indices on the device (get_batch); no
    DataLoader, no copy through the CPU.

Dates are filtered only through hmrec.data.asof; processed tables are read
through hmrec.data.metadata.scan_processed_parquet.
"""

import json
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch

from hmrec.config import PreprocessConfig, TrainingWindow
from hmrec.data.asof import TRANSACTION_DAY, history_window
from hmrec.data.day_index import date_to_day_index
from hmrec.data.id_maps import ARTICLE_IDX, CUSTOMER_IDX
from hmrec.data.metadata import check_processed_metadata
from hmrec.paths import ProjectPaths

GROUP_ID = "group_id"
# Key for sorting/searching (customer, day) pairs as one int64:
# key = customer_idx * DAY_KEY_SPAN + day. t_dat_day is int16 (0..32767), so
# 2**16 is larger than any day and keys of different customers never overlap.
DAY_KEY_SPAN = 2**16

PROFILE_COLUMNS = ("age_bucket", "club_member_status", "fashion_news_frequency")
# product_code, then the six categorical columns of proposal §8.4.
ARTICLE_CODE_COLUMNS = (
    "product_code",
    "product_type_no",
    "product_group_name",
    "garment_group_no",
    "index_group_no",
    "section_no",
    "colour_group_code",
)
MISSING_CODE = 0


# ---------------------------------------------------------------- reading purchase lines (IO)


def load_window_lines(
    transactions: pl.LazyFrame, end_anchor: date, day_zero: date, window_days: int
) -> pl.DataFrame:
    """Purchase lines with t_dat_day in [end_anchor - window_days, end_anchor) (the sample window).

    Returns:
        One row per purchase line; columns customer_idx (int32), article_idx
        (int32), t_dat_day (int16).
    """
    return history_window(
        transactions, end_anchor, day_zero, window_days, columns=[CUSTOMER_IDX, ARTICLE_IDX]
    )


def load_all_lines_before(
    transactions: pl.LazyFrame, end_anchor: date, day_zero: date
) -> pl.DataFrame:
    """Every purchase line before end_anchor: t_dat_day in [0, end_anchor) (the history source).

    The window starts at day 0 (day_zero, the first day of the data), so a
    history may reach back as far as the data goes (proposal §5 R6: no time limit).
    """
    days_since_day_zero = date_to_day_index(end_anchor, day_zero)
    if days_since_day_zero <= 0:
        raise ValueError(f"end_anchor {end_anchor} must be after day_zero {day_zero}")
    return history_window(
        transactions,
        end_anchor,
        day_zero,
        days_since_day_zero,
        columns=[CUSTOMER_IDX, ARTICLE_IDX],
    )


# ---------------------------------------------------------------- samples and groups


def daily_purchases(lines: pl.DataFrame) -> pl.DataFrame:
    """Collapse purchase lines to one row per (customer_idx, t_dat_day, article_idx).

    Several lines of the same article by the same customer on the same day
    (e.g. two pieces of one item) become one row: for the Two-Tower model they
    are one positive, not two (proposal §6.1).

    Returns:
        Columns customer_idx (int32), t_dat_day (int16), article_idx (int32);
        unsorted.
    """
    return lines.select(CUSTOMER_IDX, TRANSACTION_DAY, ARTICLE_IDX).unique()


def build_samples(window_lines: pl.DataFrame) -> pl.DataFrame:
    """Training samples with their group_id.

    Args:
        window_lines: Purchase lines of the sample window; columns
            customer_idx, t_dat_day, article_idx.

    Returns:
        One row per (customer_idx, t_dat_day, article_idx), sorted by
        (t_dat_day, customer_idx, article_idx). Extra column group_id (int32):
        0, 1, 2, ... one value per (customer_idx, t_dat_day); rows of a group
        are consecutive.
    """
    samples = daily_purchases(window_lines).sort(TRANSACTION_DAY, CUSTOMER_IDX, ARTICLE_IDX)
    day, customer = pl.col(TRANSACTION_DAY), pl.col(CUSTOMER_IDX)
    # A new group starts where (day, customer) differs from the previous row.
    # The first row has no previous row (shift gives null), so it starts group 0.
    starts_group = ((day != day.shift(1)) | (customer != customer.shift(1))).fill_null(True)
    group_id = (starts_group.cast(pl.Int32).cum_sum() - 1).cast(pl.Int32).alias(GROUP_ID)
    return samples.with_columns(group_id)


# ---------------------------------------------------------------- histories


def _event_keys(customer_idx: np.ndarray, day: np.ndarray) -> np.ndarray:
    """int64 key that sorts by customer, then day (see DAY_KEY_SPAN)."""
    if day.size and (day.min() < 0 or day.max() >= DAY_KEY_SPAN):
        raise ValueError(f"t_dat_day must be in [0, {DAY_KEY_SPAN}); got {day.min()}..{day.max()}")
    return customer_idx.astype(np.int64) * DAY_KEY_SPAN + day.astype(np.int64)


def _sorted_events(events: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Events as numpy arrays (keys int64, article_idx int32, day int32).

    Sorted by customer, day ascending, and article_idx DESCENDING within a day.
    Reading this order BACKWARDS from a position gives: most recent day first,
    and ascending article_idx within a day, which is the history order we want.
    """
    ordered = events.sort(
        [CUSTOMER_IDX, TRANSACTION_DAY, ARTICLE_IDX], descending=[False, False, True]
    )
    customer = ordered[CUSTOMER_IDX].to_numpy()
    day = ordered[TRANSACTION_DAY].to_numpy().astype(np.int32)
    article = ordered[ARTICLE_IDX].to_numpy().astype(np.int32)
    return _event_keys(customer, day), article, day


def _check_history_before_query(
    history_day: np.ndarray, history_mask: np.ndarray, query_day: np.ndarray
) -> None:
    """Fail if a filled history slot is not strictly before its query day (a leak)."""
    leaked = history_mask & (history_day >= query_day[:, None])
    if leaked.any():
        row = int(np.argwhere(leaked)[0, 0])
        raise ValueError(
            f"history of query row {row} (day {query_day[row]}) contains a purchase on day "
            f"{history_day[row][leaked[row]].max()}, which is not before the query day"
        )


def _check_below_padding(name: str, article_idx: np.ndarray, padding_idx: int) -> None:
    """Fail unless every article_idx is in [0, padding_idx): only padding may equal padding_idx."""
    if article_idx.size and (article_idx.min() < 0 or article_idx.max() >= padding_idx):
        raise ValueError(
            f"{name}: article_idx must be in [0, {padding_idx}) (padding_idx = n_articles), "
            f"got {article_idx.min()}..{article_idx.max()}"
        )


def build_histories(
    events: pl.DataFrame,
    query_customer: np.ndarray,
    query_day: np.ndarray,
    history_length: int,
    padding_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each query (customer, day): the last `history_length` purchases before that day.

    Args:
        events: Daily purchases (daily_purchases) to take histories from; must
            contain every purchase before the latest query day.
        query_customer: int32 [Q], customer_idx of each query.
        query_day: int16/int32 [Q], t_dat_day of each query.
        history_length: L, e.g. 20.
        padding_idx: Value of empty slots: n_articles, which is not a real article.

    Returns:
        (history int32 [Q, L], history_mask bool [Q, L]). Slot 0 is the most
        recent purchase; empty slots hold padding_idx with mask False.

    Raises:
        ValueError: If an event's article_idx is not below padding_idx (it
            would be confused with padding), or a history leaks (see
            _check_history_before_query).
    """
    keys, articles, days = _sorted_events(events)
    _check_below_padding("events", articles, padding_idx)
    query_day = query_day.astype(np.int32)
    # Events of this customer before the query day are exactly keys[start:end]:
    #   start = first event of this customer (key >= customer * SPAN + 0),
    #   end   = first event of this customer ON or AFTER the query day.
    start = np.searchsorted(keys, _event_keys(query_customer, np.zeros_like(query_day)), "left")
    end = np.searchsorted(keys, _event_keys(query_customer, query_day), "left")
    n_queries = query_customer.shape[0]
    history = np.full((n_queries, history_length), padding_idx, dtype=np.int32)
    history_day = np.full((n_queries, history_length), -1, dtype=np.int32)
    history_mask = np.zeros((n_queries, history_length), dtype=np.bool_)
    for slot in range(history_length):
        # Slot k holds the (k+1)-th most recent event: position end - 1 - k.
        position = end - 1 - slot
        filled = position >= start
        safe_position = np.where(filled, position, 0)  # any valid index; masked anyway
        history[:, slot] = np.where(filled, articles[safe_position], padding_idx)
        history_day[:, slot] = np.where(filled, days[safe_position], -1)
        history_mask[:, slot] = filled
    _check_history_before_query(history_day, history_mask, query_day)
    return history, history_mask


# ---------------------------------------------------------------- training arrays


@dataclass(frozen=True)
class TrainingArrays:
    """Training samples of one window, N rows, in build_samples order.

    Attributes:
        customer_idx: int32 [N].
        t_dat_day: int16 [N], the purchase day.
        article_idx: int32 [N], the positive article.
        group_id: int32 [N], one value per (customer_idx, t_dat_day); consecutive rows.
        history: int32 [N, L], see build_histories; padded slots = n_articles.
        history_mask: bool [N, L].
    """

    customer_idx: np.ndarray
    t_dat_day: np.ndarray
    article_idx: np.ndarray
    group_id: np.ndarray
    history: np.ndarray
    history_mask: np.ndarray

    @property
    def n_samples(self) -> int:
        """Number of samples N."""
        return int(self.article_idx.shape[0])


def build_training_arrays(
    window_lines: pl.DataFrame, all_lines: pl.DataFrame, history_length: int, n_articles: int
) -> TrainingArrays:
    """Samples of the window plus their user histories.

    Args:
        window_lines: Purchase lines of the sample window (load_window_lines).
        all_lines: Every purchase line before the window's end (load_all_lines_before).
        history_length: L.
        n_articles: Number of real articles; also the padding index of the history.

    Returns:
        The arrays; the history is computed once per group (customer, day) and
        copied to each sample of the group, since it depends only on the group.
    """
    samples = build_samples(window_lines)
    _check_below_padding("samples", samples[ARTICLE_IDX].to_numpy(), n_articles)
    groups = samples.unique(subset=GROUP_ID, keep="first", maintain_order=True)
    group_history, group_mask = build_histories(
        daily_purchases(all_lines),
        groups[CUSTOMER_IDX].to_numpy(),
        groups[TRANSACTION_DAY].to_numpy(),
        history_length,
        padding_idx=n_articles,
    )
    group_id = samples[GROUP_ID].to_numpy()
    return TrainingArrays(
        customer_idx=samples[CUSTOMER_IDX].to_numpy(),
        t_dat_day=samples[TRANSACTION_DAY].to_numpy(),
        article_idx=samples[ARTICLE_IDX].to_numpy(),
        group_id=group_id,
        history=group_history[group_id],
        history_mask=group_mask[group_id],
    )


def write_training_arrays(arrays: TrainingArrays, directory: Path, metadata: dict) -> None:
    """Save every array as {name}.npy plus metadata.json; never into a non-empty directory.

    Raises:
        FileExistsError: If `directory` already holds files (artifacts are never overwritten).
    """
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"{directory} is not empty; move it away to rebuild the arrays")
    directory.mkdir(parents=True, exist_ok=True)
    for field in fields(arrays):
        np.save(directory / f"{field.name}.npy", getattr(arrays, field.name))
    with (directory / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=str)


def read_training_arrays(directory: Path) -> TrainingArrays:
    """Load the arrays written by write_training_arrays."""
    loaded = {
        field.name: np.load(directory / f"{field.name}.npy") for field in fields(TrainingArrays)
    }
    return TrainingArrays(**loaded)


def check_window_before_tuning(window: TrainingWindow, first_tuning_anchor: date) -> None:
    """Fail if a training window reaches into the first tuning fold (proposal §6.2).

    The base model must never see D1-D4, so its window must end on or before
    the D1 anchor (end_anchor is exclusive).
    """
    if window.end_anchor > first_tuning_anchor:
        raise ValueError(
            f"training window ends at {window.end_anchor}, after the first tuning anchor "
            f"{first_tuning_anchor}: the model would see tuning weeks"
        )


def build_window_arrays(
    transactions: pl.LazyFrame,
    window: TrainingWindow,
    day_zero: date,
    history_length: int,
    n_articles: int,
) -> TrainingArrays:
    """Read the purchase lines of a window (through asof) and build its training arrays.

    Args:
        transactions: data/processed/transactions.parquet (scan_processed_parquet).
        window: Sample window [end_anchor - window_days, end_anchor).
        day_zero: PreprocessConfig.day_zero.
        history_length: L (20).
        n_articles: Number of real articles (the padding index).
    """
    window_lines = load_window_lines(transactions, window.end_anchor, day_zero, window.window_days)
    all_lines = load_all_lines_before(transactions, window.end_anchor, day_zero)
    return build_training_arrays(window_lines, all_lines, history_length, n_articles)


@dataclass(frozen=True)
class TrainingArraysSummary:
    """Numbers reported after building a window's arrays.

    Attributes:
        n_samples: Positives N.
        n_groups: Distinct (customer, day) groups.
        n_users: Distinct customers.
        n_articles: Distinct positive articles.
        first_day: Smallest t_dat_day.
        last_day: Largest t_dat_day.
        mean_history_length: Mean number of filled history slots per sample.
        empty_history_share: Share of samples with no purchase before their day.
        array_bytes: Total size of the arrays in memory.
    """

    n_samples: int
    n_groups: int
    n_users: int
    n_articles: int
    first_day: int
    last_day: int
    mean_history_length: float
    empty_history_share: float
    array_bytes: int


def summarize_training_arrays(arrays: TrainingArrays) -> TrainingArraysSummary:
    """Count samples, groups, users and articles, and describe the histories."""
    filled_slots = arrays.history_mask.sum(axis=1)
    return TrainingArraysSummary(
        n_samples=arrays.n_samples,
        n_groups=int(np.unique(arrays.group_id).size),
        n_users=int(np.unique(arrays.customer_idx).size),
        n_articles=int(np.unique(arrays.article_idx).size),
        first_day=int(arrays.t_dat_day.min()),
        last_day=int(arrays.t_dat_day.max()),
        mean_history_length=float(filled_slots.mean()),
        empty_history_share=float((filled_slots == 0).mean()),
        array_bytes=sum(int(getattr(arrays, f.name).nbytes) for f in fields(arrays)),
    )


# ---------------------------------------------------------------- category codes


@dataclass(frozen=True)
class CategoryCodes:
    """Integer codes of several category columns, one row per customer or article.

    Attributes:
        codes: int32 [n_rows, n_columns]; row i belongs to customer_idx / article_idx i.
        columns: Column names, in the column order of `codes`.
        vocabularies: Column -> its values as text; value vocabularies[c][k] has code k + 1.
    """

    codes: np.ndarray
    columns: tuple[str, ...]
    vocabularies: dict[str, tuple[str, ...]]

    @property
    def cardinalities(self) -> tuple[int, ...]:
        """Number of codes per column, including the missing code 0 (embedding table sizes)."""
        return tuple(len(self.vocabularies[column]) + 1 for column in self.columns)


def encode_category(values: pl.Series) -> tuple[np.ndarray, tuple[str, ...]]:
    """Code one column: 0 = null, 1..K = distinct non-null values in sorted order.

    Returns:
        (codes int32 [n], vocabulary as text, where vocabulary[k] has code k + 1).
    """
    distinct = values.drop_nulls().unique().sort()
    codes = values.replace_strict(
        distinct, pl.int_range(1, distinct.len() + 1, eager=True), return_dtype=pl.Int32
    )
    codes = codes.fill_null(MISSING_CODE)
    return codes.to_numpy(), tuple(str(value) for value in distinct.to_list())


def encode_table(table: pl.DataFrame, idx_column: str, columns: tuple[str, ...]) -> CategoryCodes:
    """Code several columns of a table whose idx_column is exactly 0..n-1 (row i = idx i)."""
    ordered = table.sort(idx_column)
    expected = np.arange(ordered.height, dtype=np.int64)
    if not np.array_equal(ordered[idx_column].to_numpy().astype(np.int64), expected):
        raise ValueError(f"{idx_column} must be exactly 0..{ordered.height - 1}")
    code_columns: list[np.ndarray] = []
    vocabularies: dict[str, tuple[str, ...]] = {}
    for column in columns:
        codes, vocabulary = encode_category(ordered[column])
        code_columns.append(codes)
        vocabularies[column] = vocabulary
    return CategoryCodes(np.stack(code_columns, axis=1).astype(np.int32), columns, vocabularies)


def build_customer_profiles(customers: pl.DataFrame) -> CategoryCodes:
    """User profile codes: age_bucket, club_member_status, fashion_news_frequency (§5 R6)."""
    return encode_table(customers, CUSTOMER_IDX, PROFILE_COLUMNS)


def build_article_codes(articles: pl.DataFrame) -> CategoryCodes:
    """Article codes: product_code and the §8.4 category columns."""
    return encode_table(articles, ARTICLE_IDX, ARTICLE_CODE_COLUMNS)


# ---------------------------------------------------------------- item features


@dataclass(frozen=True)
class ItemFeatureArrays:
    """Everything the item tower reads: row i = article_idx i, plus ONE padding row at the end.

    Rows 0..n_articles-1 are the real articles; row n_articles (= padding_idx)
    is the padding row used by padded history slots: vectors 0, masks False,
    codes 0.

    Attributes:
        image: float32 [n_articles + 1, 512] CLIP vectors (zeros where missing).
        image_mask: bool [n_articles + 1]; False = no image (442 articles + padding).
        text: float32 [n_articles + 1, 384] MiniLM vectors.
        text_mask: bool [n_articles + 1].
        codes: product_code and §8.4 category codes, int32 [n_articles + 1, 7].
    """

    image: np.ndarray
    image_mask: np.ndarray
    text: np.ndarray
    text_mask: np.ndarray
    codes: CategoryCodes

    @property
    def padding_idx(self) -> int:
        """Index of the padding row = n_articles (the number of real articles)."""
        return int(self.image.shape[0]) - 1


def _check_embedding(name: str, vectors: np.ndarray, mask: np.ndarray, n: int, dim: int) -> None:
    """Fail unless vectors is float32 [n, dim] and mask is bool [n]."""
    if vectors.dtype != np.float32 or vectors.shape != (n, dim):
        raise ValueError(
            f"{name}: expected float32 [{n}, {dim}], got {vectors.dtype} {vectors.shape}"
        )
    if mask.dtype != np.bool_ or mask.shape != (n,):
        raise ValueError(f"{name} mask: expected bool [{n}], got {mask.dtype} {mask.shape}")


def _with_padding_row(array: np.ndarray) -> np.ndarray:
    """Append one row of zeros (False for bool arrays) to an array of shape [n, ...] or [n]."""
    padding_row = np.zeros((1, *array.shape[1:]), dtype=array.dtype)
    return np.concatenate([array, padding_row], axis=0)


def build_item_feature_arrays(
    embeddings: dict[str, np.ndarray], articles: pl.DataFrame, image_dim: int, text_dim: int
) -> ItemFeatureArrays:
    """Check the aligned embeddings, code the article columns and add the padding row.

    Args:
        embeddings: "image" float32 [n, image_dim], "image_mask" bool [n],
            "text" float32 [n, text_dim], "text_mask" bool [n]; row i = article_idx i.
        articles: One row per article; article_idx exactly 0..n-1.
        image_dim: Expected image width (512).
        text_dim: Expected text width (384).

    Returns:
        Arrays with n + 1 rows; the last one is the padding row.
    """
    n_articles = articles.height
    _check_embedding("image", embeddings["image"], embeddings["image_mask"], n_articles, image_dim)
    _check_embedding("text", embeddings["text"], embeddings["text_mask"], n_articles, text_dim)
    codes = build_article_codes(articles)
    padded_codes = CategoryCodes(_with_padding_row(codes.codes), codes.columns, codes.vocabularies)
    return ItemFeatureArrays(
        image=_with_padding_row(embeddings["image"]),
        image_mask=_with_padding_row(embeddings["image_mask"]),
        text=_with_padding_row(embeddings["text"]),
        text_mask=_with_padding_row(embeddings["text_mask"]),
        codes=padded_codes,
    )


def load_item_feature_arrays(
    paths: ProjectPaths, preprocess: PreprocessConfig, articles: pl.DataFrame
) -> ItemFeatureArrays:
    """Read the aligned embeddings (after the metadata check) and build the item feature table.

    Args:
        paths: Project paths (the aligned .npy files in data/processed/embeddings/).
        preprocess: Current preprocess config; checked against metadata.json
            before any file is read, and gives the expected widths (512, 384).
        articles: data/processed/articles.parquet (read with scan_processed_parquet).

    Returns:
        See build_item_feature_arrays (n_articles + 1 rows, padding row last).
    """
    check_processed_metadata(paths.processed_metadata_file, preprocess)
    embeddings = {
        "image": np.load(paths.image_embeddings_by_idx_file),
        "image_mask": np.load(paths.image_embedding_mask_file),
        "text": np.load(paths.text_embeddings_by_idx_file),
        "text_mask": np.load(paths.text_embedding_mask_file),
    }
    dims = preprocess.embeddings
    return build_item_feature_arrays(embeddings, articles, dims.image_dim, dims.text_dim)


# ---------------------------------------------------------------- device tensors and batches


@dataclass(frozen=True)
class DeviceSamples:
    """TrainingArrays on the device. IDs stay int32 to save memory (history is the largest).

    Attributes:
        article_idx, group_id: int32 [N].
        customer_idx: int32 [N].
        history: int32 [N, L].
        history_mask: bool [N, L].
    """

    customer_idx: torch.Tensor
    article_idx: torch.Tensor
    group_id: torch.Tensor
    history: torch.Tensor
    history_mask: torch.Tensor


@dataclass(frozen=True)
class DeviceItemFeatures:
    """ItemFeatureArrays on the device: image / text float32, masks bool, codes int32."""

    image: torch.Tensor
    image_mask: torch.Tensor
    text: torch.Tensor
    text_mask: torch.Tensor
    codes: torch.Tensor


@dataclass(frozen=True)
class Batch:
    """One training batch of B samples; index tensors are int64 (what nn.Embedding expects).

    Attributes:
        article_idx: int64 [B], the positive articles.
        group_id: int64 [B]; two samples with the same group_id are positives
            of each other and must not be used as negatives (§5 R6 multi-positive).
        history: int64 [B, L].
        history_mask: bool [B, L].
        profile: int64 [B, 3], profile codes of each sample's customer.
    """

    article_idx: torch.Tensor
    group_id: torch.Tensor
    history: torch.Tensor
    history_mask: torch.Tensor
    profile: torch.Tensor


def to_device_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """Move a numpy array to the device without changing its dtype.

    np.require copies the array only when it is read-only or not contiguous
    (e.g. a zero-copy view from Polars): torch must not share read-only memory.
    """
    writable = np.require(array, requirements=["C_CONTIGUOUS", "WRITEABLE"])
    return torch.from_numpy(writable).to(device)


def samples_to_device(arrays: TrainingArrays, device: torch.device) -> DeviceSamples:
    """Copy the sample arrays to the device once (int32 / bool, no float64)."""
    return DeviceSamples(
        customer_idx=to_device_tensor(arrays.customer_idx.astype(np.int32, copy=False), device),
        article_idx=to_device_tensor(arrays.article_idx.astype(np.int32, copy=False), device),
        group_id=to_device_tensor(arrays.group_id.astype(np.int32, copy=False), device),
        history=to_device_tensor(arrays.history.astype(np.int32, copy=False), device),
        history_mask=to_device_tensor(arrays.history_mask, device),
    )


def items_to_device(features: ItemFeatureArrays, device: torch.device) -> DeviceItemFeatures:
    """Copy the item feature table (padding row included) to the device once (~0.4 GB)."""
    return DeviceItemFeatures(
        image=to_device_tensor(features.image, device),
        image_mask=to_device_tensor(features.image_mask, device),
        text=to_device_tensor(features.text, device),
        text_mask=to_device_tensor(features.text_mask, device),
        codes=to_device_tensor(features.codes.codes, device),
    )


def profiles_to_device(profiles: CategoryCodes, device: torch.device) -> torch.Tensor:
    """Profile codes int32 [n_customers, 3] on the device; row i = customer_idx i."""
    return to_device_tensor(profiles.codes, device)


def get_batch(samples: DeviceSamples, profiles: torch.Tensor, indices: torch.Tensor) -> Batch:
    """Gather one batch by sample index, entirely on the device.

    Args:
        samples: From samples_to_device.
        profiles: From profiles_to_device (same device).
        indices: int64 [B] sample indices, on the same device.
    """
    customer_idx = samples.customer_idx[indices].long()
    return Batch(
        article_idx=samples.article_idx[indices].long(),
        group_id=samples.group_id[indices].long(),
        history=samples.history[indices].long(),
        history_mask=samples.history_mask[indices],
        profile=profiles[customer_idx].long(),
    )
