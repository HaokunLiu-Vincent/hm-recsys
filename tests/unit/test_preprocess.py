"""Tests for hmrec.data.preprocess, using the hand-made CSVs in tests/fixtures/preprocess/.

See tests/fixtures/preprocess/README.md for how the expected indexes are derived:
customer_idx: cust_a..cust_j -> 0..9; article_idx: 0108775015 -> 0, 0108775044 -> 1,
0110065001 -> 2, 0951111001 -> 3; day 0 = 2018-09-20.
"""

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from hmrec.config import PreprocessConfig, load_preprocess_config
from hmrec.data.id_maps import build_article_id_map, build_customer_id_map
from hmrec.data.preprocess import (
    TRANSACTIONS_SCHEMA,
    align_embeddings,
    assign_age_bucket_expr,
    build_articles,
    build_customers,
    build_transactions,
    check_product_code_matches_article_id,
    count_by_age_bucket,
    count_rows,
    scan_all_string_csv,
    scan_transactions_csv,
    validate_transactions,
    write_transactions,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "preprocess"
REPO_PREPROCESS_YAML = Path(__file__).resolve().parents[2] / "configs" / "preprocess.yaml"
DAY_ZERO = date(2018, 9, 20)
NORM_TOLERANCE = 1e-3


@pytest.fixture
def config() -> PreprocessConfig:
    return load_preprocess_config(REPO_PREPROCESS_YAML)


@pytest.fixture
def customer_map() -> pl.DataFrame:
    ids = scan_all_string_csv(FIXTURES / "customers.csv").collect()["customer_id"]
    return build_customer_id_map(ids)


@pytest.fixture
def article_map() -> pl.DataFrame:
    ids = scan_all_string_csv(FIXTURES / "articles.csv").collect()["article_id"]
    return build_article_id_map(ids)


def build_fixture_transactions(
    customer_map: pl.DataFrame, article_map: pl.DataFrame, day_zero: date = DAY_ZERO
) -> pl.DataFrame:
    raw = scan_transactions_csv(FIXTURES / "transactions.csv")
    return build_transactions(raw, customer_map, article_map, day_zero).collect()


# ---------------------------------------------------------------- transactions


def test_transactions_schema_rows_and_order(
    customer_map: pl.DataFrame, article_map: pl.DataFrame
) -> None:
    transactions = build_fixture_transactions(customer_map, article_map)
    assert dict(transactions.schema) == TRANSACTIONS_SCHEMA
    # Sorted by (t_dat_day, customer_idx); ties keep CSV order. Worked out by hand:
    #   day 0: cust_a (CSV line 3), cust_c (lines 2 and 5, an exact duplicate)
    #   day 1: cust_a (line 4), cust_b (lines 1 then 7)
    #   day 2: cust_j (line 6)
    expected = [
        (0, 0, 0, 0.0169, 2),
        (2, 3, 0, 0.0508, 1),
        (2, 3, 0, 0.0508, 1),
        (0, 2, 1, 0.0101, 2),
        (1, 1, 1, 0.0338, 2),
        (1, 0, 1, 0.0200, 2),
        (9, 0, 2, 0.0169, 1),
    ]
    assert len(transactions) == len(expected)
    for actual_row, expected_row in zip(transactions.rows(), expected, strict=True):
        assert actual_row[:3] == expected_row[:3]
        assert actual_row[3] == pytest.approx(expected_row[3], rel=1e-6)
        assert actual_row[4] == expected_row[4]


def test_exact_duplicate_lines_are_kept(
    customer_map: pl.DataFrame, article_map: pl.DataFrame
) -> None:
    transactions = build_fixture_transactions(customer_map, article_map)
    raw_rows = count_rows(scan_transactions_csv(FIXTURES / "transactions.csv"))
    assert raw_rows == 7
    assert transactions.height == raw_rows
    assert transactions.is_duplicated().sum() == 2  # the pair from CSV lines 2 and 5


def test_write_validate_and_row_groups(
    tmp_path: Path, customer_map: pl.DataFrame, article_map: pl.DataFrame
) -> None:
    raw = scan_transactions_csv(FIXTURES / "transactions.csv")
    path = tmp_path / "processed" / "transactions.parquet"
    write_transactions(
        build_transactions(raw, customer_map, article_map, DAY_ZERO), path, row_group_size=2
    )

    stats = validate_transactions(pl.scan_parquet(path), expected_rows=7)
    assert (stats.n_rows, stats.min_day, stats.max_day) == (7, 0, 2)
    # 7 rows in groups of 2 -> at least 4 row groups, each with min/max statistics.
    metadata = pq.ParquetFile(path).metadata
    assert metadata.num_row_groups >= 4
    assert metadata.row_group(0).column(2).statistics.has_min_max


def test_validation_rejects_wrong_row_count(
    customer_map: pl.DataFrame, article_map: pl.DataFrame
) -> None:
    transactions = build_fixture_transactions(customer_map, article_map)
    with pytest.raises(ValueError, match="7 rows, expected 8"):
        validate_transactions(transactions.lazy(), expected_rows=8)


def test_first_date_must_equal_day_zero(
    customer_map: pl.DataFrame, article_map: pl.DataFrame
) -> None:
    # One day earlier than the real first date: the earliest t_dat_day becomes 1.
    transactions = build_fixture_transactions(customer_map, article_map, date(2018, 9, 19))
    with pytest.raises(ValueError, match="earliest t_dat_day is 1, expected 0"):
        validate_transactions(transactions.lazy(), expected_rows=7)


def test_unsorted_file_is_rejected(customer_map: pl.DataFrame, article_map: pl.DataFrame) -> None:
    transactions = build_fixture_transactions(customer_map, article_map)
    reversed_rows = transactions.reverse()
    with pytest.raises(ValueError, match="t_dat_day decreases"):
        validate_transactions(reversed_rows.lazy(), expected_rows=7)


def test_unknown_customer_is_rejected(
    tmp_path: Path, customer_map: pl.DataFrame, article_map: pl.DataFrame
) -> None:
    csv_path = tmp_path / "transactions.csv"
    csv_path.write_text(
        "t_dat,customer_id,article_id,price,sales_channel_id\n"
        "2018-09-20,cust_a,0108775015,0.01,2\n"
        "2018-09-20,cust_unknown,0108775015,0.01,2\n"
    )
    raw = scan_transactions_csv(csv_path)
    transactions = build_transactions(raw, customer_map, article_map, DAY_ZERO)
    with pytest.raises(ValueError, match="1 rows have customer_idx = NULL"):
        validate_transactions(transactions, expected_rows=2)


# ---------------------------------------------------------------- customers


@pytest.mark.parametrize(
    ("age", "bucket"),
    [
        (None, "AGE_UNKNOWN"),
        (16, "AGE_INVALID"),
        (17, "AGE_INVALID"),
        (18, "AGE_18_24"),
        (24, "AGE_18_24"),
        (25, "AGE_25_34"),
        (44, "AGE_35_44"),
        (45, "AGE_45_54"),
        (55, "AGE_55_PLUS"),
        (99, "AGE_55_PLUS"),
        (100, "AGE_INVALID"),
    ],
)
def test_age_bucket_rules(age: int | None, bucket: str, config: PreprocessConfig) -> None:
    frame = pl.DataFrame({"age": pl.Series([age], dtype=pl.Int16)})
    result = frame.select(assign_age_bucket_expr(pl.col("age"), config).alias("bucket"))
    assert result["bucket"].to_list() == [bucket]


def test_build_customers(customer_map: pl.DataFrame, config: PreprocessConfig) -> None:
    raw = scan_all_string_csv(FIXTURES / "customers.csv")
    customers = build_customers(raw, customer_map, config)

    assert customers.schema == {
        "customer_idx": pl.Int32,
        "age": pl.Int8,
        "age_bucket": pl.String,
        "fn": pl.Int8,
        "active": pl.Int8,
        "club_member_status": pl.String,
        "fashion_news_frequency": pl.String,
    }
    assert "postal_code" not in customers.columns
    assert customers["customer_idx"].to_list() == list(range(10))
    # Rows are cust_a..cust_j after sorting by customer_idx.
    assert customers["age"].to_list() == [18, 24, 17, None, 25, 54, 55, 99, 100, 35]
    assert customers["age_bucket"].to_list() == [
        "AGE_18_24",
        "AGE_18_24",
        "AGE_INVALID",
        "AGE_UNKNOWN",
        "AGE_25_34",
        "AGE_45_54",
        "AGE_55_PLUS",
        "AGE_55_PLUS",
        "AGE_INVALID",
        "AGE_35_44",
    ]
    # Empty FN / Active means "not set" -> 0.
    assert customers["fn"].to_list() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert customers["active"].to_list() == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # cust_b had "None", folded into "NONE"; cust_d's missing values stay NULL.
    assert customers["fashion_news_frequency"][1] == "NONE"
    assert customers["fashion_news_frequency"][3] is None
    assert customers["club_member_status"][3] is None


def test_count_by_age_bucket(customer_map: pl.DataFrame, config: PreprocessConfig) -> None:
    raw = scan_all_string_csv(FIXTURES / "customers.csv")
    counts = count_by_age_bucket(build_customers(raw, customer_map, config), config)
    assert counts.rows() == [
        ("AGE_18_24", 2),
        ("AGE_25_34", 1),
        ("AGE_35_44", 1),
        ("AGE_45_54", 1),
        ("AGE_55_PLUS", 2),
        ("AGE_INVALID", 2),
        ("AGE_UNKNOWN", 1),
    ]


def test_unexpected_flag_value_raises(config: PreprocessConfig, tmp_path: Path) -> None:
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text(
        "customer_id,FN,Active,club_member_status,fashion_news_frequency,age,postal_code\n"
        "cust_a,2.0,,ACTIVE,NONE,30,p\n"
    )
    customer_map = build_customer_id_map(pl.Series(["cust_a"]))
    with pytest.raises(ValueError, match=r"customers\.fn: expected only 0/1"):
        build_customers(scan_all_string_csv(csv_path), customer_map, config)


# ---------------------------------------------------------------- articles


def test_build_articles(article_map: pl.DataFrame) -> None:
    articles = build_articles(scan_all_string_csv(FIXTURES / "articles.csv"), article_map)
    assert articles.schema == {
        "article_idx": pl.Int32,
        "product_code": pl.Int32,
        "product_type_no": pl.Int16,
        "product_group_name": pl.String,
        "garment_group_no": pl.Int16,
        "index_group_no": pl.Int8,
        "section_no": pl.Int8,
        "colour_group_code": pl.Int8,
    }
    assert articles["article_idx"].to_list() == [0, 1, 2, 3]
    # "0108775" -> 108775: the two colour variants share one product_code.
    assert articles["product_code"].to_list() == [108775, 108775, 110065, 951111]
    assert articles["colour_group_code"].to_list() == [9, 10, -1, 93]
    assert articles["product_group_name"][3] == "Garment Full body"


def test_product_code_mismatch_raises() -> None:
    raw = pl.DataFrame(
        {"article_id": ["0108775015", "0108775044"], "product_code": ["0108775", "0108776"]}
    )
    with pytest.raises(ValueError, match=r"1 rows have product_code != article_id\[:7\]"):
        check_product_code_matches_article_id(raw)


# ---------------------------------------------------------------- embeddings


def make_embedding_source() -> tuple[np.ndarray, pl.DataFrame]:
    """3 unit vectors (dim 2) in a shuffled order; article 0108775044 (idx 1) has none."""
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32)
    row_ids = pl.DataFrame(
        {
            "embedding_row": [0, 1, 2],
            "article_id": ["0951111001", "0108775015", "0110065001"],
        }
    )
    return vectors, row_ids


def test_align_embeddings(article_map: pl.DataFrame) -> None:
    vectors, row_ids = make_embedding_source()
    aligned, mask = align_embeddings(vectors, row_ids, article_map, 2, NORM_TOLERANCE)

    assert aligned.dtype == np.float32
    assert aligned.shape == (4, 2)
    expected = np.array([[0.0, 1.0], [0.0, 0.0], [0.6, 0.8], [1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(aligned, expected)
    assert mask.tolist() == [True, False, True, True]


def test_align_embeddings_rejects_non_unit_vector(article_map: pl.DataFrame) -> None:
    vectors, row_ids = make_embedding_source()
    vectors[2] = [0.606, 0.808]  # norm 1.01
    with pytest.raises(ValueError, match=r"1 vectors have L2 norm .* worst is article_idx 2"):
        align_embeddings(vectors, row_ids, article_map, 2, NORM_TOLERANCE)


def test_align_embeddings_rejects_wrong_dim(article_map: pl.DataFrame) -> None:
    vectors, row_ids = make_embedding_source()
    with pytest.raises(ValueError, match=r"expected shape \[n, 3\]"):
        align_embeddings(vectors, row_ids, article_map, 3, NORM_TOLERANCE)


def test_align_embeddings_rejects_bad_row_ids(article_map: pl.DataFrame) -> None:
    vectors, row_ids = make_embedding_source()
    gap = row_ids.with_columns(pl.Series("embedding_row", [0, 1, 5]))
    with pytest.raises(ValueError, match=r"embedding_row must be exactly 0\.\.2"):
        align_embeddings(vectors, gap, article_map, 2, NORM_TOLERANCE)
    duplicated = row_ids.with_columns(
        pl.Series("article_id", ["0951111001", "0951111001", "0110065001"])
    )
    with pytest.raises(ValueError, match="duplicated article_id"):
        align_embeddings(vectors, duplicated, article_map, 2, NORM_TOLERANCE)


def test_align_embeddings_rejects_unknown_article(article_map: pl.DataFrame) -> None:
    vectors, row_ids = make_embedding_source()
    unknown = row_ids.with_columns(
        pl.Series("article_id", ["0951111001", "0999999999", "0110065001"])
    )
    with pytest.raises(ValueError, match="not in the ID map"):
        align_embeddings(vectors, unknown, article_map, 2, NORM_TOLERANCE)
