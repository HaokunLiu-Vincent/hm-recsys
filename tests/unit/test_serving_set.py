"""Tests for hmrec.serving_set: writing, loading and checking a serving set.

The example set is in serving_example.py (3 customers, Top-2, 5 articles).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from serving_example import (
    ALICE,
    ARTICLES,
    BOB,
    CAROL,
    FALLBACK_IDS,
    UNKNOWN,
    example_arrays,
    example_description,
    write_example_set,
)

from hmrec.serving_set import (
    ARRAY_FILES,
    CUSTOMER_IDS_FILE,
    SERVING_RECORD_FILE,
    SOURCE_FALLBACK,
    SOURCE_RANKER,
    TOP_ARTICLES_FILE,
    Fallback,
    ServingSetError,
    load_serving_set,
    read_serving_record,
    write_serving_set,
)


def test_a_written_set_loads_back_and_answers_with_original_ids(tmp_path: Path) -> None:
    record = write_example_set(tmp_path / "serving")
    assert (record.n_customers, record.top_k, record.n_articles) == (3, 2, 5)
    # The model version is the first 12 characters of the Ranker's sha256.
    assert record.model_version == "e" * 12 and record.set_id == "2020-09-23/" + "e" * 12
    serving = load_serving_set(tmp_path / "serving")
    assert serving.record == record
    assert serving.recommend(ALICE) == ((ARTICLES[0], ARTICLES[1]), SOURCE_RANKER)
    assert serving.recommend(BOB) == ((ARTICLES[2], ARTICLES[3]), SOURCE_RANKER)
    assert serving.recommend(CAROL) == ((ARTICLES[4], ARTICLES[0]), SOURCE_RANKER)


@pytest.mark.parametrize(
    "customer_id",
    [UNKNOWN, "", "not-a-customer", ALICE[:-1], ALICE + "0", ALICE.upper(), "é" * 64, "\x00" * 64],
)
def test_anyone_else_gets_the_fallback_list(tmp_path: Path, customer_id: str) -> None:
    write_example_set(tmp_path / "serving")
    serving = load_serving_set(tmp_path / "serving")
    assert serving.recommend(customer_id) == (FALLBACK_IDS, SOURCE_FALLBACK)


def test_the_record_is_written_last_and_never_replaced(tmp_path: Path) -> None:
    folder = tmp_path / "serving"
    write_example_set(folder)
    assert sorted(path.name for path in folder.iterdir()) == sorted(
        [*ARRAY_FILES, SERVING_RECORD_FILE]
    )
    with pytest.raises(FileExistsError, match="never replaced"):
        write_example_set(folder)
    # Without its record a set is not complete, whatever files are there.
    (folder / SERVING_RECORD_FILE).unlink()
    with pytest.raises(ServingSetError, match="not complete"):
        load_serving_set(folder)


def test_a_set_that_breaks_a_rule_is_not_written(tmp_path: Path) -> None:
    customer_ids, top_articles, article_ids = example_arrays()
    description = example_description()
    cases = [
        (customer_ids[::-1].copy(), top_articles, "not sorted strictly ascending"),
        (np.array([ALICE, ALICE, BOB], dtype=np.bytes_), top_articles, "not sorted strictly"),
        (customer_ids, top_articles.astype(np.int64), "2-D int32"),
        (customer_ids, top_articles[:2], "expected 3 rows"),
        (customer_ids, np.where(top_articles == 4, 5, top_articles), "outside 0..4"),
    ]
    for ids, top, message in cases:
        with pytest.raises(ServingSetError, match=message):
            write_serving_set(tmp_path / "serving", description, ids, top, article_ids)
    for listed, message in [
        ((ARTICLES[3],), "2 distinct article ids"),
        ((ARTICLES[3], ARTICLES[3]), "2 distinct article ids"),
        ((ARTICLES[3], "0000000999"), "unknown article ids"),
    ]:
        fallback = Fallback("AGE_UNKNOWN", listed, "weeks/x/candidates.parquet", "f" * 64)
        broken = example_description()
        broken = type(broken)(**{**broken.__dict__, "fallback": fallback})
        with pytest.raises(ServingSetError, match=message):
            write_serving_set(tmp_path / "serving", broken, customer_ids, top_articles, article_ids)
    assert not (tmp_path / "serving").exists()


def test_a_changed_file_fails_its_checksum(tmp_path: Path) -> None:
    folder = tmp_path / "serving"
    write_example_set(folder)
    # Another Top-K with the same shape and dtype: only the sha256 can tell.
    _, top_articles, _ = example_arrays()
    np.save(folder / TOP_ARTICLES_FILE, top_articles[::-1].copy(), allow_pickle=False)
    with pytest.raises(ServingSetError, match="sha256 .* is not the recorded"):
        load_serving_set(folder)


def test_a_missing_or_truncated_file_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "serving"
    write_example_set(folder)
    data = (folder / CUSTOMER_IDS_FILE).read_bytes()
    (folder / CUSTOMER_IDS_FILE).write_bytes(data[: len(data) // 2])
    with pytest.raises(ServingSetError, match="sha256"):
        load_serving_set(folder)
    (folder / CUSTOMER_IDS_FILE).unlink()
    with pytest.raises(ServingSetError, match="cannot be read"):
        load_serving_set(folder)


def test_a_record_with_other_keys_or_another_version_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "serving"
    write_example_set(folder)
    record_file = folder / SERVING_RECORD_FILE
    raw = json.loads(record_file.read_text(encoding="utf-8"))
    for changed, message in [
        ({**raw, "schema_version": 2}, "schema_version 2"),
        ({**raw, "extra": 1}, "exactly the keys"),
        ({key: value for key, value in raw.items() if key != "fallback"}, "exactly the keys"),
        (
            {**raw, "files": {CUSTOMER_IDS_FILE: raw["files"][CUSTOMER_IDS_FILE]}},
            "exactly the files",
        ),
        ({**raw, "fallback": {"age_bucket": "AGE_UNKNOWN"}}, "not a serving record"),
    ]:
        record_file.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ServingSetError, match=message):
            read_serving_record(folder)
    record_file.write_text("not json", encoding="utf-8")
    with pytest.raises(ServingSetError, match="cannot be read"):
        read_serving_record(folder)
    # A record whose numbers do not describe the files is refused by the loader.
    record_file.write_text(json.dumps({**raw, "n_customers": 4}), encoding="utf-8")
    with pytest.raises(ServingSetError, match="the record says"):
        load_serving_set(folder)
    record_file.write_text(json.dumps({**raw, "model_version": "0" * 12}), encoding="utf-8")
    with pytest.raises(ServingSetError, match="model version"):
        load_serving_set(folder)
