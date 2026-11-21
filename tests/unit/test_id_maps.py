from pathlib import Path

import polars as pl
import pytest

from hmrec.data.id_maps import (
    article_ids_to_idx,
    article_idx_to_ids,
    build_article_id_map,
    build_customer_id_map,
    customer_ids_to_idx,
    customer_idx_to_ids,
    read_id_map,
    write_id_map,
)

# Deliberately unsorted. Sorted order: 0100000015, 0100000044, 0951111001.
ARTICLE_IDS = ["0951111001", "0100000015", "0100000044"]


def test_article_map_numbers_sorted_ids_and_keeps_leading_zeros() -> None:
    article_map = build_article_id_map(pl.Series(ARTICLE_IDS))
    assert article_map.schema == {"article_idx": pl.Int32, "article_id": pl.String}
    assert article_map.rows() == [(0, "0100000015"), (1, "0100000044"), (2, "0951111001")]


def test_customer_map_ignores_input_order() -> None:
    forward = build_customer_id_map(pl.Series(["b", "a", "c"]))
    backward = build_customer_id_map(pl.Series(["c", "b", "a"]))
    assert forward.equals(backward)
    assert forward["customer_id"].to_list() == ["a", "b", "c"]


def test_duplicate_ids_raise() -> None:
    with pytest.raises(ValueError, match="duplicated IDs"):
        build_customer_id_map(pl.Series(["a", "b", "a"]))


def test_null_ids_raise() -> None:
    with pytest.raises(ValueError, match="null IDs"):
        build_customer_id_map(pl.Series(["a", None]))


@pytest.mark.parametrize("bad_id", ["100000015", "01000000150", "01000000x5"])
def test_article_id_must_be_ten_digits(bad_id: str) -> None:
    # "100000015" is what you get when the CSV is read as an integer.
    with pytest.raises(ValueError, match="10-digit string"):
        build_article_id_map(pl.Series(["0100000015", bad_id]))


def test_round_trip_keeps_row_order_and_duplicates() -> None:
    article_map = build_article_id_map(pl.Series(ARTICLE_IDS))
    frame = pl.DataFrame({"article_id": ["0951111001", "0100000015", "0951111001"], "x": [1, 2, 3]})

    encoded = article_ids_to_idx(frame, article_map)
    assert encoded.columns == ["article_idx", "x"]
    assert encoded["article_idx"].to_list() == [2, 0, 2]
    assert encoded["article_idx"].dtype == pl.Int32

    decoded = article_idx_to_ids(encoded, article_map)
    assert decoded.equals(frame)


def test_customer_round_trip() -> None:
    customer_map = build_customer_id_map(pl.Series(["b", "a"]))
    frame = pl.DataFrame({"customer_id": ["b", "a", "b"]})
    encoded = customer_ids_to_idx(frame, customer_map)
    assert encoded["customer_idx"].to_list() == [1, 0, 1]
    assert customer_idx_to_ids(encoded, customer_map).equals(frame)


def test_unknown_id_raises() -> None:
    article_map = build_article_id_map(pl.Series(ARTICLE_IDS))
    frame = pl.DataFrame({"article_id": ["0100000015", "0999999999"]})
    with pytest.raises(ValueError, match=r"1 article_id values are not in the ID map.*0999999999"):
        article_ids_to_idx(frame, article_map)


def test_unknown_idx_raises() -> None:
    article_map = build_article_id_map(pl.Series(ARTICLE_IDS))
    frame = pl.DataFrame({"article_idx": [0, 7]})
    with pytest.raises(ValueError, match="article_idx values are not in the ID map"):
        article_idx_to_ids(frame, article_map)


def test_write_and_read_id_map(tmp_path: Path) -> None:
    article_map = build_article_id_map(pl.Series(ARTICLE_IDS))
    path = tmp_path / "maps" / "article_id_map.parquet"
    write_id_map(article_map, path)
    assert read_id_map(path, "article_id", "article_idx").equals(article_map)


def test_read_id_map_rejects_gaps(tmp_path: Path) -> None:
    broken = pl.DataFrame(
        {"article_idx": pl.Series([0, 2], dtype=pl.Int32), "article_id": ["a", "b"]}
    )
    path = tmp_path / "broken.parquet"
    broken.write_parquet(path)
    with pytest.raises(ValueError, match=r"exactly 0\.\.1"):
        read_id_map(path, "article_id", "article_idx")
