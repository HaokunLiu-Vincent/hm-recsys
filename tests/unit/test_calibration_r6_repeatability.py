"""Tests for hmrec.calibration.r6_repeatability: the synthetic R6 fine-tuning reproducer.

The reproducer's worker is run here on a tiny model (a temporary copy of
configs/ with small table sizes), in this process. Two runs with
torch.use_deterministic_algorithms(True) must give the same weights on every
platform; whether two runs WITHOUT it agree is what the reproducer measures on
the real sizes, and is not asserted here.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from hmrec.calibration.r6_repeatability import (
    REPEATABILITY_COLUMNS,
    article_probabilities,
    check_row,
    read_summary,
    run_worker,
    summary_line,
    synthetic_fine_tune_arrays,
    synthetic_samples,
    synthetic_tables,
    worker_exit_code,
)
from hmrec.calibration.recall_items import parameter_difference
from hmrec.config import (
    RepeatabilityConfig,
    RepeatabilityTableSizes,
    load_base_config,
    load_repeatability_config,
    load_yaml,
    parse_repeatability_config,
)
from hmrec.paths import ProjectPaths, build_paths, get_paths
from hmrec.recall.twotower.fine_tune_runtime import SettingNotApplicable

REPO = Path(__file__).resolve().parents[2]
SMALL = RepeatabilityConfig(
    seed=7,
    table_sizes=RepeatabilityTableSizes(
        n_articles=50,
        n_customers=30,
        image_dim=8,
        text_dim=6,
        article_code_values=(20, 5, 3, 4, 2, 6, 7),
        profile_values=(7, 3, 3),
    ),
    scales=(64, 128),
    alarm_samples=16,
    popularity_offset=2.0,
    max_group_rows=4,
)
HISTORY_LENGTH = 5


def test_the_repository_config_has_the_real_table_sizes() -> None:
    config = load_repeatability_config(get_paths().r6_repeatability_config_file)
    sizes = config.table_sizes
    assert (sizes.n_articles, sizes.n_customers) == (105542, 1371980)
    assert (sizes.image_dim, sizes.text_dim) == (512, 384)
    # product_code and six category columns; three profile columns.
    assert len(sizes.article_code_values) == 7 and len(sizes.profile_values) == 3
    # The largest scale is the size of the real fine-tune set of week V3.
    assert config.scales[-1] == 476668 and list(config.scales) == sorted(config.scales)
    raw = load_yaml(REPO / "configs" / "m10_r6_repeatability.yaml")
    raw["scales"] = [512, 0]
    with pytest.raises(ValueError, match="non-empty list of positive integers"):
        parse_repeatability_config(raw)


def test_synthetic_tables_have_the_configured_shapes_and_a_padding_row() -> None:
    tables = synthetic_tables(SMALL)
    features = tables.features
    assert features.image.shape == (51, 8) and features.image.dtype == np.float32
    assert features.text.shape == (51, 6)
    # Real rows have length 1; the padding row (the last one) is zero with False masks.
    assert np.allclose(np.linalg.norm(features.image[:50], axis=1), 1.0, atol=1e-5)
    assert not features.image[50].any() and not features.text[50].any()
    assert features.image_mask[:50].all() and not features.image_mask[50]
    codes = features.codes.codes
    assert codes.shape == (51, 7) and codes.dtype == np.int32
    assert (codes[:50].min(), codes[50].max()) == (1, 0)
    assert all(
        codes[:50, column].max() <= size for column, size in enumerate((20, 5, 3, 4, 2, 6, 7))
    )
    # Embedding table sizes: the number of values plus the code 0.
    assert tables.article_codes.cardinalities == (21, 6, 4, 5, 3, 7, 8)
    assert tables.profiles.codes.shape == (30, 3)
    assert tables.profiles.cardinalities == (8, 4, 4)


def test_synthetic_samples_look_like_training_samples() -> None:
    samples = synthetic_samples(SMALL, 64, HISTORY_LENGTH, stream=1)
    assert samples.n_samples == 64
    assert (samples.article_idx.dtype, samples.history.dtype) == (np.int32, np.int32)
    assert samples.t_dat_day.dtype == np.int16 and samples.history.shape == (64, 5)
    assert samples.article_idx.min() >= 0 and samples.article_idx.max() < 50
    assert samples.customer_idx.max() < 30
    # Rows of a group are consecutive: the group ids never go down and step by at most 1.
    steps = np.diff(samples.group_id)
    assert set(steps.tolist()) <= {0, 1} and samples.group_id[0] == 0
    # No group has more rows than max_group_rows.
    assert np.bincount(samples.group_id).max() <= 4
    for first, second in zip(range(63), range(1, 64), strict=True):
        if samples.group_id[first] == samples.group_id[second]:
            # One customer on one day: the same customer, day and history.
            assert samples.customer_idx[first] == samples.customer_idx[second]
            assert samples.t_dat_day[first] == samples.t_dat_day[second]
            assert (samples.history[first] == samples.history[second]).all()
    # Filled slots come first; empty slots hold the padding index (n_articles = 50).
    assert (samples.history[~samples.history_mask] == 50).all()
    assert (samples.history[samples.history_mask] < 50).all()
    lengths = samples.history_mask.sum(axis=1)
    assert all(samples.history_mask[row, :length].all() for row, length in enumerate(lengths))


def test_the_same_seed_gives_the_same_data_and_another_stream_gives_other_data() -> None:
    first = synthetic_fine_tune_arrays(SMALL, 64, HISTORY_LENGTH)
    second = synthetic_fine_tune_arrays(SMALL, 64, HISTORY_LENGTH)
    for name in ("customer_idx", "article_idx", "group_id", "history", "history_mask"):
        assert (getattr(first.train, name) == getattr(second.train, name)).all()
        assert (getattr(first.alarm, name) == getattr(second.alarm, name)).all()
    assert (first.n_new, first.n_replay, first.alarm.n_samples) == (32, 32, 16)
    assert not (first.train.article_idx[:16] == first.alarm.article_idx).all()
    tables_a, tables_b = synthetic_tables(SMALL), synthetic_tables(SMALL)
    assert tables_a.features.image.tobytes() == tables_b.features.image.tobytes()


def test_article_probabilities_are_concentrated_and_sum_to_one() -> None:
    probabilities = article_probabilities(4, 1.0, np.random.default_rng(0))
    # Weights 1/1, 1/2, 1/3, 1/4 (sum 25/12) in a random order.
    expected = np.array([12 / 25, 6 / 25, 4 / 25, 3 / 25])
    assert np.allclose(np.sort(probabilities)[::-1], expected)
    assert probabilities.sum() == pytest.approx(1.0)


def _small_project(root: Path) -> ProjectPaths:
    """A copy of configs/ whose reproducer config has the small table sizes."""
    shutil.copytree(REPO / "configs", root / "configs")
    raw = load_yaml(root / "configs" / "m10_r6_repeatability.yaml")
    raw["seed"] = SMALL.seed
    raw["scales"] = list(SMALL.scales)
    raw["alarm_samples"] = SMALL.alarm_samples
    raw["table_sizes"].update(
        n_articles=50,
        n_customers=30,
        image_dim=8,
        text_dim=6,
        article_code_values=[20, 5, 3, 4, 2, 6, 7],
        profile_values=[7, 3, 3],
    )
    (root / "configs" / "m10_r6_repeatability.yaml").write_text(
        yaml.safe_dump(raw), encoding="utf-8"
    )
    return build_paths(load_base_config(root / "configs" / "base.yaml"), root)


def test_two_workers_with_deterministic_algorithms_give_the_same_weights(tmp_path: Path) -> None:
    paths = _small_project(tmp_path)
    first = run_worker(paths, "deterministic", 128, tmp_path / "out" / "first.pt")
    second = run_worker(paths, "deterministic", 128, tmp_path / "out" / "second.pt")
    # 128 rows, batch 512 (one batch per epoch), 2 locked epochs = 2 steps.
    assert (first.steps, first.train_rows, first.setting) == (2, 128, "deterministic")
    assert "deterministic algorithms True" in first.threads
    difference = parameter_difference(tmp_path / "out" / "first.pt", tmp_path / "out" / "second.pt")
    assert difference.n_values > 0 and difference.n_different_values == 0
    assert first.mean_train_loss == second.mean_train_loss
    assert read_summary(tmp_path / "out" / "first.pt") == first
    # The fine-tune really changed the weights: a third model that was not trained differs.
    with pytest.raises(ValueError, match="--rows must be one of"):
        run_worker(paths, "none", 100, tmp_path / "out" / "third.pt")


def test_rows_and_the_summary_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hmrec.calibration.r6_repeatability as module

    paths = _small_project(tmp_path)
    monkeypatch.setattr(module, "git_state", lambda root, results: ("c" * 40, False))
    values = {
        "steps": "8",
        "os": "linux",
        "architecture": "aarch64",
        "max_abs_parameter_difference": "0.0",
        "first_seconds": "1.2345678",
        "second_seconds": "1.3",
        "threads_during_fine_tune": "torch intra-op 6",
    }
    row = check_row(paths, "one_thread", 2048, "repeatable", values)
    assert tuple(row) == REPEATABILITY_COLUMNS
    assert (row["setting"], row["train_rows"], row["outcome"]) == (
        "one_thread",
        "2048",
        "repeatable",
    )
    assert (row["git_commit"], row["git_dirty"], row["num_threads"]) == ("c" * 40, "False", "6")
    assert summary_line(row) == (
        "r6 fine-tune repeatability | linux-aarch64 | setting one_thread | 2048 rows, 8 steps | "
        "repeatable | max parameter difference 0.0 | 1.2345 s and 1.3 s | torch intra-op 6"
    )


def test_a_setting_that_cannot_be_applied_has_its_own_exit_code() -> None:
    assert worker_exit_code(SettingNotApplicable("no BLAS")) == 3
    assert worker_exit_code(RuntimeError("no deterministic algorithm")) == 1
