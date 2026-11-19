"""Tests for recall.twotower.data: samples, groups, histories, codes, device batches.

Hand example (days are t_dat_day on FIXTURE_DAY_ZERO; lines are (customer, article, day)):

    customer 1: day 5: 30, 10, 10 (article 10 bought twice on day 5)
                day 7: 20
                day 9: 40, 15, 40 (article 40 bought twice on day 9)
                day 12: 77            (after the anchor: poison)
    customer 2: day 9: 50
                day 10: 88            (on the anchor day: poison)

Sample window: end_anchor = day 10, window_days = 3 -> purchase days [7, 10).
Samples, sorted by (day, customer, article), with group_id:
    (c1, d7, 20) g0 | (c1, d9, 15) g1 | (c1, d9, 40) g1 | (c2, d9, 50) g2
    -> 4 samples: the second line of article 40 on day 9 is not a new sample.
Histories (L = 4, most recent day first, same day in ascending article_idx;
PAD = n_articles = 100, which is not a real article: real ones are 0..99):
    g0 (c1, day 7): day 5 -> [10, 30]            -> [10, 30, PAD, PAD], mask T T F F
    g1 (c1, day 9): day 7, then day 5 -> [20, 10, 30] -> mask T T T F
    g2 (c2, day 9): nothing before day 9         -> all PAD, mask F F F F
Article 10 appears once in a history although it was bought twice on day 5.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from conftest import FIXTURE_DAY_ZERO, make_transactions
from hmrec.config import TrainingWindow
from hmrec.data.day_index import day_index_to_date
from hmrec.recall.twotower.data import (
    TrainingArrays,
    build_article_codes,
    build_customer_profiles,
    build_histories,
    build_item_feature_arrays,
    build_samples,
    build_training_arrays,
    build_window_arrays,
    check_window_before_tuning,
    daily_purchases,
    encode_category,
    get_batch,
    load_all_lines_before,
    load_window_lines,
    profiles_to_device,
    read_training_arrays,
    samples_to_device,
    summarize_training_arrays,
    write_training_arrays,
)

N_ARTICLES = 100
PAD = N_ARTICLES  # the padding index is n_articles
LINES = [
    (1, 30, 5),
    (1, 10, 5),
    (1, 10, 5),
    (1, 20, 7),
    (1, 40, 9),
    (1, 15, 9),
    (1, 40, 9),
    (2, 50, 9),
]
POISON = [(1, 77, 12), (2, 88, 10)]
END_ANCHOR = day_index_to_date(10, FIXTURE_DAY_ZERO)


def _arrays(lines: list[tuple[int, int, int]], history_length: int = 4) -> TrainingArrays:
    transactions = make_transactions(lines).lazy()
    window = load_window_lines(transactions, END_ANCHOR, FIXTURE_DAY_ZERO, 3)
    everything_before = load_all_lines_before(transactions, END_ANCHOR, FIXTURE_DAY_ZERO)
    return build_training_arrays(window, everything_before, history_length, N_ARTICLES)


# ---------------------------------------------------------------- samples and groups


def test_same_day_repeat_purchase_is_one_sample() -> None:
    arrays = _arrays(LINES)
    assert arrays.n_samples == 4
    rows = list(zip(arrays.customer_idx, arrays.t_dat_day, arrays.article_idx, strict=True))
    assert rows == [(1, 7, 20), (1, 9, 15), (1, 9, 40), (2, 9, 50)]


def test_same_day_positives_share_a_group() -> None:
    arrays = _arrays(LINES)
    assert arrays.group_id.tolist() == [0, 1, 1, 2]
    assert arrays.group_id.dtype == np.int32


def test_build_samples_groups_are_consecutive_blocks() -> None:
    lines = make_transactions([(3, 1, 0), (2, 5, 0), (2, 4, 0), (3, 1, 1)])
    samples = build_samples(lines)
    # Sorted by (day, customer, article): (0,2,4) (0,2,5) (0,3,1) (1,3,1).
    assert samples["article_idx"].to_list() == [4, 5, 1, 1]
    assert samples["group_id"].to_list() == [0, 0, 1, 2]


# ---------------------------------------------------------------- histories


def test_history_order_padding_and_mask() -> None:
    arrays = _arrays(LINES)
    assert arrays.history.tolist() == [
        [10, 30, PAD, PAD],
        [20, 10, 30, PAD],
        [20, 10, 30, PAD],
        [PAD, PAD, PAD, PAD],
    ]
    assert arrays.history_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, False],
        [False, False, False, False],
    ]
    assert arrays.history.dtype == np.int32 and arrays.history.shape == (4, 4)
    assert arrays.history_mask.dtype == np.bool_


def test_padding_slots_are_n_articles_and_real_slots_are_below() -> None:
    arrays = _arrays(LINES)
    assert (arrays.history[~arrays.history_mask] == N_ARTICLES).all()
    assert (arrays.history[arrays.history_mask] < N_ARTICLES).all()
    assert (arrays.history[arrays.history_mask] >= 0).all()
    assert (arrays.article_idx < N_ARTICLES).all()


def test_an_article_equal_to_the_padding_index_is_rejected() -> None:
    # Article 100 would be indistinguishable from padding when n_articles = 100.
    with pytest.raises(ValueError, match=r"must be in \[0, 100\)"):
        _arrays([*LINES, (1, 100, 5)])


def test_history_is_cut_to_the_most_recent_purchases() -> None:
    # L = 2: g1 keeps day 7 (20) and the smaller article of day 5 (10); 30 is dropped.
    arrays = _arrays(LINES, history_length=2)
    assert arrays.history[1].tolist() == [20, 10]
    assert arrays.history_mask[1].tolist() == [True, True]


def test_history_excludes_the_purchase_day_and_later() -> None:
    # Query (c1, day 9) with events on day 9 (same day) and day 12 (later):
    # neither may appear; only days 5 and 7 do.
    events = daily_purchases(make_transactions(LINES + POISON))
    history, mask = build_histories(
        events, np.array([1], dtype=np.int32), np.array([9], dtype=np.int32), 5, N_ARTICLES
    )
    assert history[0][mask[0]].tolist() == [20, 10, 30]
    assert not set(history[0][mask[0]].tolist()) & {15, 40, 77}


def test_poison_on_and_after_the_anchor_changes_nothing() -> None:
    clean = _arrays(LINES)
    poisoned = _arrays(LINES + POISON)
    for name in ("customer_idx", "t_dat_day", "article_idx", "group_id", "history"):
        assert np.array_equal(getattr(clean, name), getattr(poisoned, name)), name
    assert np.array_equal(clean.history_mask, poisoned.history_mask)


def test_build_histories_rejects_negative_days() -> None:
    events = daily_purchases(make_transactions(LINES))
    with pytest.raises(ValueError, match="t_dat_day must be in"):
        build_histories(
            events, np.array([1], dtype=np.int32), np.array([-1], dtype=np.int32), 2, N_ARTICLES
        )


def test_build_window_arrays_matches_the_step_by_step_build() -> None:
    window = TrainingWindow(end_anchor=END_ANCHOR, window_days=3)
    arrays = build_window_arrays(
        make_transactions(LINES + POISON).lazy(), window, FIXTURE_DAY_ZERO, 4, N_ARTICLES
    )
    assert np.array_equal(arrays.history, _arrays(LINES).history)
    assert arrays.article_idx.tolist() == [20, 15, 40, 50]


def test_window_may_not_reach_into_the_tuning_folds() -> None:
    window = TrainingWindow(end_anchor=END_ANCHOR, window_days=3)
    check_window_before_tuning(window, END_ANCHOR)  # ends exactly at the anchor: fine
    with pytest.raises(ValueError, match="would see tuning weeks"):
        check_window_before_tuning(window, day_index_to_date(9, FIXTURE_DAY_ZERO))


def test_summarize_training_arrays_hand_computed() -> None:
    summary = summarize_training_arrays(_arrays(LINES))
    assert (summary.n_samples, summary.n_groups, summary.n_users) == (4, 3, 2)
    assert summary.n_articles == 4
    assert (summary.first_day, summary.last_day) == (7, 9)
    # Filled slots per sample: 2, 3, 3, 0 -> mean 2; one of four is empty.
    assert summary.mean_history_length == 2.0
    assert summary.empty_history_share == 0.25


# ---------------------------------------------------------------- files


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    arrays = _arrays(LINES)
    directory = tmp_path / "base"
    write_training_arrays(arrays, directory, {"window": "test"})
    loaded = read_training_arrays(directory)
    assert np.array_equal(loaded.history, arrays.history)
    assert np.array_equal(loaded.history_mask, arrays.history_mask)
    assert loaded.t_dat_day.dtype == arrays.t_dat_day.dtype
    with pytest.raises(FileExistsError, match="not empty"):
        write_training_arrays(arrays, directory, {})


# ---------------------------------------------------------------- category codes


def test_encode_category_null_is_zero_and_values_are_sorted() -> None:
    codes, vocabulary = encode_category(pl.Series(["PRE-CREATE", None, "ACTIVE", "ACTIVE"]))
    assert vocabulary == ("ACTIVE", "PRE-CREATE")
    assert codes.tolist() == [2, 0, 1, 1]


def test_customer_profiles_are_indexed_by_customer_idx() -> None:
    customers = pl.DataFrame(
        {
            "customer_idx": [1, 0],
            "age_bucket": ["AGE_25_34", "AGE_UNKNOWN"],
            "club_member_status": [None, "ACTIVE"],
            "fashion_news_frequency": ["NONE", None],
        },
        schema_overrides={"customer_idx": pl.Int32},
    )
    profiles = build_customer_profiles(customers)
    # Row 0 = customer 0: AGE_UNKNOWN (2nd of 2), ACTIVE (1st of 1), null -> 0.
    assert profiles.codes.tolist() == [[2, 1, 0], [1, 0, 1]]
    assert profiles.cardinalities == (3, 2, 2)
    assert profiles.codes.dtype == np.int32


def test_profiles_need_every_customer_idx() -> None:
    customers = pl.DataFrame(
        {
            "customer_idx": [0, 2],
            "age_bucket": ["a", "b"],
            "club_member_status": ["x", "y"],
            "fashion_news_frequency": ["n", "n"],
        }
    )
    with pytest.raises(ValueError, match="must be exactly 0..1"):
        build_customer_profiles(customers)


def test_article_codes_cover_product_code_and_categories() -> None:
    articles = pl.DataFrame(
        {
            "article_idx": [0, 1],
            "product_code": [100000, 100000],
            "product_type_no": [-1, 901],
            "product_group_name": ["Group B", "Group A"],
            "garment_group_no": [1101, 1101],
            "index_group_no": [1, 1],
            "section_no": [11, 11],
            "colour_group_code": [7, -1],
        }
    )
    codes = build_article_codes(articles)
    # Same product_code -> same code; -1 sorts first, so it gets code 1.
    assert codes.codes.tolist() == [[1, 1, 2, 1, 1, 1, 2], [1, 2, 1, 1, 1, 1, 1]]
    assert codes.vocabularies["product_type_no"] == ("-1", "901")


def _articles(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "article_idx": list(range(n)),
            "product_code": [7, 7, 8][:n],
            "product_type_no": [1, 2, 3][:n],
            "product_group_name": ["a", "b", "c"][:n],
            "garment_group_no": [1, 1, 1][:n],
            "index_group_no": [1, 1, 1][:n],
            "section_no": [1, 1, 1][:n],
            "colour_group_code": [1, 1, 1][:n],
        }
    )


def test_item_features_end_with_a_zero_padding_row() -> None:
    # 3 real articles; article 1 has no image (zero vector, mask False).
    image = np.ones((3, 4), dtype=np.float32)
    image[1] = 0.0
    embeddings = {
        "image": image,
        "image_mask": np.array([True, False, True]),
        "text": np.full((3, 2), 0.5, dtype=np.float32),
        "text_mask": np.array([True, True, True]),
    }
    features = build_item_feature_arrays(embeddings, _articles(3), image_dim=4, text_dim=2)
    assert features.padding_idx == 3  # = n_articles
    assert features.image.shape == (4, 4) and features.text.shape == (4, 2)
    assert features.codes.codes.shape == (4, 7)
    # The padding row: vectors 0, masks False, codes 0.
    assert (features.image[3] == 0).all() and (features.text[3] == 0).all()
    assert not features.image_mask[3] and not features.text_mask[3]
    assert (features.codes.codes[3] == 0).all()
    # Real rows keep their values, and no real article has code 0 (no nulls here).
    assert features.image_mask[:3].tolist() == [True, False, True]
    assert (features.text[:3] == 0.5).all()
    assert (features.codes.codes[:3] > 0).all()
    assert features.image.dtype == np.float32 and features.codes.codes.dtype == np.int32


def test_item_features_reject_a_wrong_width() -> None:
    embeddings = {
        "image": np.ones((3, 5), dtype=np.float32),
        "image_mask": np.ones(3, dtype=np.bool_),
        "text": np.ones((3, 2), dtype=np.float32),
        "text_mask": np.ones(3, dtype=np.bool_),
    }
    with pytest.raises(ValueError, match=r"image: expected float32 \[3, 4\]"):
        build_item_feature_arrays(embeddings, _articles(3), image_dim=4, text_dim=2)


# ---------------------------------------------------------------- device batches


def test_get_batch_indexes_on_the_device() -> None:
    arrays = _arrays(LINES)
    device = torch.device("cpu")
    samples = samples_to_device(arrays, device)
    profiles = profiles_to_device(
        build_customer_profiles(
            pl.DataFrame(
                {
                    "customer_idx": [0, 1, 2],
                    "age_bucket": ["a", "b", "c"],
                    "club_member_status": ["a", "b", "c"],
                    "fashion_news_frequency": ["a", "b", "c"],
                }
            )
        ),
        device,
    )
    batch = get_batch(samples, profiles, torch.tensor([3, 1], dtype=torch.int64))
    assert batch.article_idx.tolist() == [50, 15]
    assert batch.group_id.tolist() == [2, 1]
    assert batch.history.tolist() == [[PAD] * 4, [20, 10, 30, PAD]]
    assert batch.history_mask.tolist() == [[False] * 4, [True, True, True, False]]
    # Customer 2 -> codes (3, 3, 3); customer 1 -> (2, 2, 2) (values a < b < c -> 1, 2, 3).
    assert batch.profile.tolist() == [[3, 3, 3], [2, 2, 2]]
    assert batch.history.dtype == torch.int64 and batch.profile.dtype == torch.int64
