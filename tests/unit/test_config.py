import copy
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from hmrec.config import (
    HISTOGRAM_UNDECIDED,
    N_PROPOSAL_FEATURES,
    PROPOSAL_CATEGORICAL_FEATURES,
    PROPOSAL_FEATURES,
    RANKER_DATASET_PARAM_KEYS,
    RANKER_TRAIN_PARAM_KEYS,
    ActivePoolConfig,
    RankerTunedParams,
    load_base_config,
    load_candidate_run_config,
    load_catalog_config,
    load_features_config,
    load_itemcf_config,
    load_popularity_config,
    load_preprocess_config,
    load_queries_config,
    load_ranker_config,
    load_repurchase_config,
    load_selection_rule,
    load_sibling_config,
    load_twotower_config,
    load_union_check_config,
    load_wals_config,
    load_yaml,
    parse_base_config,
    parse_candidate_run_config,
    parse_catalog_config,
    parse_features_config,
    parse_itemcf_config,
    parse_popularity_config,
    parse_preprocess_config,
    parse_queries_config,
    parse_ranker_config,
    parse_selection_rule,
    parse_sibling_config,
    parse_twotower_config,
    parse_union_check_config,
    parse_wals_config,
    preprocess_config_hash,
)

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

VALID_BASE: dict[str, Any] = {
    "paths": {"data_root": "data", "artifacts_root": "artifacts", "results_root": "results"},
    "seed": 7,
    "num_threads": 6,
    "memory": {"limit_gb": 6, "sample_interval_seconds": 0.5},
    "chunking": {
        "query_chunk_size": 10,
        "item_chunk_size": 3,
        "dense_score_query_chunk_size": 4,
        "twotower_score_query_chunk_size": 2,
        "pair_chunk_size": 5,
    },
}

VALID_RULE: dict[str, Any] = {
    "aggregation": {"primary": "query_weighted", "also_report": ["equal_fold", "per_fold"]},
    "acceptance": {
        "min_relative_improvement": 0.01,
        "min_folds_better": 3,
        "n_folds": 4,
        "tie_breaker": "prefer_simpler_cheaper",
    },
    "selection_metric": {"r2_repurchase": "recall@30", "ranker": "map@12"},
}


# ---------------------------------------------------------------- base.yaml


def test_parse_base_config_reads_all_fields() -> None:
    config = parse_base_config(VALID_BASE)
    assert config.paths.data_root == "data"
    assert config.seed == 7
    assert config.num_threads == 6
    # An int in YAML is accepted for a float field and converted.
    assert config.memory.limit_gb == 6.0
    assert isinstance(config.memory.limit_gb, float)
    assert config.chunking.item_chunk_size == 3


def test_repo_base_yaml_matches_agents_md_limits() -> None:
    config = load_base_config(REPO_CONFIGS / "base.yaml")
    # AGENTS.md §4: 6 threads; the true peak of one process is at most 8 GB, the
    # project's memory budget (6 GB, then 7 GB, until 2026-11-18; see docs/decisions.md).
    assert config.num_threads == 6
    assert config.memory.limit_gb == 8.0
    assert config.paths.data_root == "data"
    assert config.paths.artifacts_root == "artifacts"


def test_missing_key_is_reported() -> None:
    raw = copy.deepcopy(VALID_BASE)
    del raw["memory"]["limit_gb"]
    with pytest.raises(ValueError, match=r"base\.memory: missing keys \['limit_gb'\]"):
        parse_base_config(raw)


def test_unknown_key_is_reported_as_typo() -> None:
    raw = copy.deepcopy(VALID_BASE)
    raw["num_thread"] = 6  # typo
    with pytest.raises(ValueError, match=r"unknown keys \['num_thread'\]"):
        parse_base_config(raw)


@pytest.mark.parametrize("bad_value", ["6", 6.5, True, None])
def test_num_threads_must_be_int(bad_value: object) -> None:
    raw = copy.deepcopy(VALID_BASE)
    raw["num_threads"] = bad_value
    with pytest.raises(ValueError, match=r"base\.num_threads: expected an integer"):
        parse_base_config(raw)


@pytest.mark.parametrize(
    ("section", "key"),
    [(None, "num_threads"), ("memory", "limit_gb"), ("chunking", "query_chunk_size")],
)
def test_non_positive_numbers_are_rejected(section: str | None, key: str) -> None:
    raw = copy.deepcopy(VALID_BASE)
    target = raw if section is None else raw[section]
    target[key] = 0
    with pytest.raises(ValueError, match="must be > 0"):
        parse_base_config(raw)


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="top level must be a mapping"):
        load_yaml(path)


def test_load_yaml_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")


# ---------------------------------------------------------------- selection_rule.yaml


def test_repo_selection_rule_matches_proposal_section_4() -> None:
    rule = load_selection_rule(REPO_CONFIGS / "selection_rule.yaml")
    # Proposal §4.1: query-weighted primary score, equal-fold and per-fold also reported.
    assert rule.primary_aggregation == "query_weighted"
    assert set(rule.also_report) == {"equal_fold", "per_fold"}
    # Proposal §4.2: >= 1% relative improvement AND better on >= 3 of 4 folds.
    assert rule.min_relative_improvement == pytest.approx(0.01)
    assert rule.min_folds_better == 3
    assert rule.n_folds == 4
    # Proposal §4.1 table of selection metrics.
    assert rule.selection_metric == {
        "r1_popularity": "recall@100",
        "r2_repurchase": "recall@30",
        "r3_sibling": "recall@50",
        "r4_itemcf": "recall@100",
        "r5_wals": "recall@100",
        "r6_twotower": "recall@100",
        "ranker": "map@12",
    }


def test_selection_rule_rejects_equal_fold_as_primary() -> None:
    raw = copy.deepcopy(VALID_RULE)
    raw["aggregation"]["primary"] = "equal_fold"
    with pytest.raises(ValueError, match="primary must be one of"):
        parse_selection_rule(raw)


def test_selection_rule_rejects_more_folds_than_exist() -> None:
    raw = copy.deepcopy(VALID_RULE)
    raw["acceptance"]["min_folds_better"] = 5
    with pytest.raises(ValueError, match="min_folds_better must be in"):
        parse_selection_rule(raw)


def test_selection_rule_rejects_negative_improvement() -> None:
    raw = copy.deepcopy(VALID_RULE)
    raw["acceptance"]["min_relative_improvement"] = -0.01
    with pytest.raises(ValueError, match="min_relative_improvement must be >= 0"):
        parse_selection_rule(raw)


# ---------------------------------------------------------------- preprocess.yaml

VALID_PREPROCESS: dict[str, Any] = {
    "day_zero": date(2018, 9, 20),
    "age_buckets": [
        {"name": "YOUNG", "min_age": 18, "max_age": 29},
        {"name": "OLD", "min_age": 30, "max_age": 99},
    ],
    "invalid_age_bucket": "INVALID",
    "unknown_age_bucket": "UNKNOWN",
    "embeddings": {"image_dim": 4, "text_dim": 3, "norm_tolerance": 0.001},
    "transactions_row_group_size": 10,
}


def test_repo_preprocess_yaml_matches_proposal() -> None:
    config = load_preprocess_config(REPO_CONFIGS / "preprocess.yaml")
    # Proposal §2.1: transactions start on 2018-09-20.
    assert config.day_zero == date(2018, 9, 20)
    # Proposal §5 R1 age buckets (frozen), both ends inclusive.
    assert [(b.name, b.min_age, b.max_age) for b in config.age_buckets] == [
        ("AGE_18_24", 18, 24),
        ("AGE_25_34", 25, 34),
        ("AGE_35_44", 35, 44),
        ("AGE_45_54", 45, 54),
        ("AGE_55_PLUS", 55, 99),
    ]
    assert config.invalid_age_bucket == "AGE_INVALID"
    assert config.unknown_age_bucket == "AGE_UNKNOWN"
    # Proposal §2.1 embedding widths.
    assert (config.embeddings.image_dim, config.embeddings.text_dim) == (512, 384)


def test_parse_preprocess_config() -> None:
    config = parse_preprocess_config(VALID_PREPROCESS)
    assert config.all_age_bucket_names == ("YOUNG", "OLD", "INVALID", "UNKNOWN")
    assert config.embeddings.norm_tolerance == pytest.approx(0.001)


def test_overlapping_age_buckets_raise() -> None:
    raw = copy.deepcopy(VALID_PREPROCESS)
    raw["age_buckets"][1]["min_age"] = 29
    with pytest.raises(ValueError, match="YOUNG and OLD overlap"):
        parse_preprocess_config(raw)


def test_age_bucket_gap_raises() -> None:
    raw = copy.deepcopy(VALID_PREPROCESS)
    raw["age_buckets"][1]["min_age"] = 31
    with pytest.raises(ValueError, match=r"leave a gap: ages 30\.\.30"):
        parse_preprocess_config(raw)


def test_duplicate_bucket_names_raise() -> None:
    raw = copy.deepcopy(VALID_PREPROCESS)
    raw["unknown_age_bucket"] = "INVALID"
    with pytest.raises(ValueError, match="names must be unique"):
        parse_preprocess_config(raw)


def test_preprocess_unknown_key_raises() -> None:
    raw = copy.deepcopy(VALID_PREPROCESS)
    raw["day_zeroo"] = date(2018, 9, 20)
    with pytest.raises(ValueError, match=r"unknown keys \['day_zeroo'\]"):
        parse_preprocess_config(raw)


def test_day_zero_must_be_a_date() -> None:
    raw = copy.deepcopy(VALID_PREPROCESS)
    raw["day_zero"] = "2018-09-20"
    with pytest.raises(ValueError, match="expected a date"):
        parse_preprocess_config(raw)


def test_preprocess_hash_is_stable_and_value_sensitive() -> None:
    first = preprocess_config_hash(parse_preprocess_config(VALID_PREPROCESS))
    second = preprocess_config_hash(parse_preprocess_config(copy.deepcopy(VALID_PREPROCESS)))
    assert first == second
    raw = copy.deepcopy(VALID_PREPROCESS)
    raw["transactions_row_group_size"] = 11
    assert preprocess_config_hash(parse_preprocess_config(raw)) != first


# ---------------------------------------------------------------- catalog.yaml / queries.yaml


def test_repo_catalog_and_queries_yaml() -> None:
    catalog = load_catalog_config(REPO_CONFIGS / "catalog.yaml")
    assert catalog.active_pool == ActivePoolConfig(window_days=28, min_purchase_lines=3)
    queries = load_queries_config(REPO_CONFIGS / "queries.yaml")
    assert queries.query_id_fold_offset == 10_000_000


def test_catalog_config_rejects_typo_and_non_positive() -> None:
    # A typo shows up as a missing key (checked before unknown keys).
    with pytest.raises(ValueError, match=r"missing keys \['window_days'\]"):
        parse_catalog_config({"active_pool": {"window_day": 28, "min_purchase_lines": 3}})
    with pytest.raises(ValueError, match="must be > 0"):
        parse_catalog_config({"active_pool": {"window_days": 28, "min_purchase_lines": 0}})


def test_queries_config_rejects_offset_outside_int32() -> None:
    with pytest.raises(ValueError, match="int32"):
        parse_queries_config({"query_id_fold_offset": 2**31})
    with pytest.raises(ValueError, match="must be > 0"):
        parse_queries_config({"query_id_fold_offset": 0})


# ---------------------------------------------------------------- recall/r1..r4


def test_repo_recall_configs_match_proposal_defaults() -> None:
    r1 = load_popularity_config(REPO_CONFIGS / "recall" / "r1_popularity.yaml")
    assert (r1.variant, r1.window_days, r1.tau_days, r1.quota) == ("A1", 7, 3.0, 100)
    r2 = load_repurchase_config(REPO_CONFIGS / "recall" / "r2_repurchase.yaml")
    assert (r2.variant, r2.lookback_days, r2.tau_days, r2.quota) == (
        "decayed_frequency",
        180,
        30.0,
        30,
    )
    r3 = load_sibling_config(REPO_CONFIGS / "recall" / "r3_sibling.yaml")
    assert (r3.style_lookback_days, r3.market_window_days, r3.tau_days, r3.gamma, r3.quota) == (
        180,
        7,
        30.0,
        0.5,
        50,
    )


def test_recall_config_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        parse_popularity_config({"variant": "A2", "window_days": 7, "tau_days": 3, "quota": 100})


def test_sibling_gamma_zero_allowed_negative_rejected() -> None:
    raw = {
        "style_lookback_days": 180,
        "market_window_days": 7,
        "tau_days": 30,
        "gamma": 0,
        "quota": 50,
    }
    assert parse_sibling_config(raw).gamma == 0.0
    with pytest.raises(ValueError, match="gamma must be >= 0"):
        parse_sibling_config({**raw, "gamma": -0.5})


VALID_ITEMCF: dict[str, Any] = {
    "variant": "CF3",
    "fit_window_days": 365,
    "seed_window_days": 90,
    "alpha": 0.7,
    "neighbors_per_seed": 100,
    "seed_decay": False,
    "seed_tau_days": 30.0,
    "quota": 100,
}


def test_repo_itemcf_config_matches_proposal_defaults() -> None:
    r4 = load_itemcf_config(REPO_CONFIGS / "recall" / "r4_itemcf.yaml")
    assert r4 == parse_itemcf_config(VALID_ITEMCF)


def test_itemcf_alpha_must_be_in_unit_interval() -> None:
    assert parse_itemcf_config({**VALID_ITEMCF, "alpha": 0}).alpha == 0.0
    assert parse_itemcf_config({**VALID_ITEMCF, "alpha": 1}).alpha == 1.0
    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
        parse_itemcf_config({**VALID_ITEMCF, "alpha": 1.2})


def test_itemcf_seed_window_must_fit_in_fit_window() -> None:
    with pytest.raises(ValueError, match="seed_window_days"):
        parse_itemcf_config({**VALID_ITEMCF, "seed_window_days": 400})


def test_itemcf_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        parse_itemcf_config({**VALID_ITEMCF, "variant": "CF1"})


def test_itemcf_seed_decay_must_be_a_bool() -> None:
    assert parse_itemcf_config({**VALID_ITEMCF, "seed_decay": True}).seed_decay is True
    with pytest.raises(ValueError, match="expected true or false"):
        parse_itemcf_config({**VALID_ITEMCF, "seed_decay": 1})


VALID_WALS: dict[str, Any] = {
    "variant": "W2",
    "window_days": 180,
    "decay_tau_days": 60.0,
    "alpha_conf": 20.0,
    "regularization": 0.1,
    "factors": 64,
    "iterations": 15,
    "random_seed": 42,
    "quota": 100,
}


def test_repo_wals_config_matches_proposal_defaults() -> None:
    assert load_wals_config(REPO_CONFIGS / "recall" / "r5_wals.yaml") == parse_wals_config(
        VALID_WALS
    )


def test_wals_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        parse_wals_config({**VALID_WALS, "variant": "W1"})
    with pytest.raises(ValueError, match="alpha_conf"):
        parse_wals_config({**VALID_WALS, "alpha_conf": 0})
    with pytest.raises(ValueError, match="random_seed"):
        parse_wals_config({**VALID_WALS, "random_seed": -1})


# ---------------------------------------------------------------- recall/r6_twotower.yaml

VALID_TWOTOWER: dict[str, Any] = load_yaml(REPO_CONFIGS / "recall" / "r6_twotower.yaml")


def test_repo_twotower_yaml_matches_proposal() -> None:
    config = load_twotower_config(REPO_CONFIGS / "recall" / "r6_twotower.yaml")
    assert config.history_length == 20
    # Purchase days 2019-12-03 .. 2020-06-02 = [2020-06-03 - 182 days, 2020-06-03) (§6.2).
    assert config.base_window.end_anchor == date(2020, 6, 3)
    assert config.base_window.window_days == 182
    # §1.4: 200 steps; §5 R6: batch 512, embedding 128.
    assert (config.benchmark.steps, config.benchmark.batch_size) == (200, 512)
    assert config.benchmark.embedding_dim == 128


def test_repo_twotower_model_and_training_values() -> None:
    config = load_twotower_config(REPO_CONFIGS / "recall" / "r6_twotower.yaml")
    model, training = config.model, config.training
    assert (model.output_dim, model.hidden_dim) == (128, 256)
    assert (model.image_projection_dim, model.text_projection_dim) == (128, 128)
    assert (model.article_embedding_dim, model.product_code_embedding_dim) == (64, 32)
    assert (model.category_embedding_dim, model.one_hot_max_cardinality) == (16, 20)
    assert (training.batch_size, training.temperature, training.learning_rate) == (512, 0.05, 1e-3)
    assert (training.epochs, training.seed, training.weight_decay) == (6, 42, 0.0)
    assert training.fine_tune_learning_rate == pytest.approx(0.1 * training.learning_rate)
    assert (config.quota, config.base_epoch) == (100, 1)


def test_twotower_config_rejects_typo_and_bad_numbers() -> None:
    assert parse_twotower_config(VALID_TWOTOWER).benchmark.warmup_steps == 10
    with pytest.raises(ValueError, match="history_lenght"):
        parse_twotower_config({**VALID_TWOTOWER, "history_lenght": 20})
    bad_window = copy.deepcopy(VALID_TWOTOWER)
    bad_window["base_window"]["window_days"] = 0
    with pytest.raises(ValueError, match="window_days"):
        parse_twotower_config(bad_window)
    bad_warmup = copy.deepcopy(VALID_TWOTOWER)
    bad_warmup["benchmark"]["warmup_steps"] = 200
    with pytest.raises(ValueError, match="warmup_steps"):
        parse_twotower_config(bad_warmup)
    with pytest.raises(ValueError, match="no such checkpoint"):
        parse_twotower_config({**VALID_TWOTOWER, "base_epoch": 7})
    bad_device = copy.deepcopy(VALID_TWOTOWER)
    bad_device["training"]["device"] = "cuda"
    with pytest.raises(ValueError, match="device"):
        parse_twotower_config(bad_device)
    with pytest.raises(ValueError, match="replay must be in"):
        parse_twotower_config({**VALID_TWOTOWER, "replay": 1.0})
    with pytest.raises(ValueError, match="e_update must be"):
        parse_twotower_config({**VALID_TWOTOWER, "e_update": -1})


def test_repo_twotower_chain_defaults() -> None:
    config = load_twotower_config(REPO_CONFIGS / "recall" / "r6_twotower.yaml")
    # Default = frozen (no update); the chain step overrides e_update / replay (§6.4).
    assert (config.e_update, config.replay, config.chain_seed) == (0, 0.0, 42)
    assert (config.chain.week_days, config.chain.replay_window_days) == (7, 28)


# ---------------------------------------------------------------- m4_union_check.yaml


def test_repo_union_check_config() -> None:
    config = load_union_check_config(REPO_CONFIGS / "m4_union_check.yaml")
    assert config.union_recall_target == 0.24  # G2
    # Pre-registered 2026-11-16: 0.1 percentage points, absolute. Must never change.
    assert config.r3_max_union_recall_drop == 0.001
    assert config.manual_selection == {"r6_twotower": {"replay": 0.5}}
    assert config.variants == {"r3_active_pool": {"r3_sibling": {"active_pool_only": True}}}


def test_union_check_config_errors() -> None:
    valid = load_yaml(REPO_CONFIGS / "m4_union_check.yaml")
    with pytest.raises(ValueError, match="must be in"):
        parse_union_check_config({**valid, "union_recall_target": 24})
    with pytest.raises(ValueError, match="non-empty mapping"):
        parse_union_check_config({**valid, "manual_selection": {"r6_twotower": {}}})
    with pytest.raises(ValueError, match="r3_active_pool_rule"):  # wrong key name
        parse_union_check_config({**valid, "r3_active_pool_rule": {"max_drop": 0.001}})


def test_sibling_active_pool_only_is_optional_and_a_bool() -> None:
    raw = load_yaml(REPO_CONFIGS / "recall" / "r3_sibling.yaml")
    assert parse_sibling_config(raw).active_pool_only is False
    without_key = {key: value for key, value in raw.items() if key != "active_pool_only"}
    assert parse_sibling_config(without_key).active_pool_only is False
    assert parse_sibling_config({**raw, "active_pool_only": True}).active_pool_only is True
    with pytest.raises(ValueError, match="active_pool_only"):
        parse_sibling_config({**raw, "active_pool_only": "yes"})


def test_repo_candidate_run_config() -> None:
    config = load_candidate_run_config(REPO_CONFIGS / "m6_candidates.yaml")
    # Proposal §6.6: flagged if more than 10% below the mean of the previous 4 weeks.
    assert (config.previous_folds, config.max_relative_recall_drop) == (4, 0.10)
    assert config.max_fine_tune_loss == 7.0


def test_candidate_run_config_errors() -> None:
    raw = {
        "monitoring": {"previous_folds": 4, "max_relative_recall_drop": 0.1},
        "twotower_chain": {"max_fine_tune_loss": 7.0},
    }
    assert parse_candidate_run_config(raw).previous_folds == 4
    with pytest.raises(ValueError, match="unknown keys"):
        parse_candidate_run_config({**raw, "channels": {}})
    for drop in (0.0, 1.0):
        bad = {**raw, "monitoring": {"previous_folds": 4, "max_relative_recall_drop": drop}}
        with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
            parse_candidate_run_config(bad)
    with pytest.raises(ValueError, match="must be > 0"):
        parse_candidate_run_config({**raw, "twotower_chain": {"max_fine_tune_loss": 0.0}})


def test_repo_features_config_is_exactly_the_71_features_of_the_proposal() -> None:
    config = load_features_config(REPO_CONFIGS / "features.yaml")
    assert len(config.features) == 71 and len(set(config.names)) == 71
    assert N_PROPOSAL_FEATURES == 71
    assert [len(config.group_names(group)) for group in "ABCDE"] == [14, 7, 17, 12, 21]
    assert {group: config.group_names(group) for group in "ABCDE"} == PROPOSAL_FEATURES
    assert config.categorical_names == PROPOSAL_CATEGORICAL_FEATURES
    assert len(config.categorical_names) == 8
    assert (config.history_length, config.rrf_k, config.online_sales_channel_id) == (20, 60, 2)
    # Fixed values, not hyperparameters: the decay tau of ui_decay_* and the 5 of tt_sim_top5_hist.
    assert (config.decay_tau_days, config.history_similarity_top_k) == (30.0, 5)
    assert (config.week_days, config.four_week_days, config.year_days) == (7, 28, 364)
    for feature in config.features:
        assert feature.dtype == ("int16" if feature.categorical else "float32")


def test_features_config_refuses_any_change_of_the_feature_list() -> None:
    raw = load_yaml(REPO_CONFIGS / "features.yaml")

    def with_features(features: list) -> dict:
        return {**raw, "features": features}

    extra = {"name": "item_cnt_2w", "group": "C", "dtype": "float32", "categorical": False}
    with pytest.raises(ValueError, match="exactly the 71 features"):
        parse_features_config(with_features([*raw["features"], extra]))
    with pytest.raises(ValueError, match="exactly the 71 features"):
        parse_features_config(with_features(raw["features"][:-1]))
    swapped = [raw["features"][1], raw["features"][0], *raw["features"][2:]]
    with pytest.raises(ValueError, match="exactly the 71 features"):
        parse_features_config(with_features(swapped))
    as_number = [dict(item) for item in raw["features"]]
    section = next(item for item in as_number if item["name"] == "section_no")
    section.update(dtype="float32", categorical=False)
    with pytest.raises(ValueError, match="categorical features must be"):
        parse_features_config(with_features(as_number))
    mismatch = [dict(item) for item in raw["features"]]
    mismatch[0]["dtype"] = "int16"  # a numeric feature declared as a category code
    with pytest.raises(ValueError, match="categorical features are int16 codes"):
        parse_features_config(with_features(mismatch))
    with pytest.raises(ValueError, match="must be > 0"):
        parse_features_config({**raw, "rrf_k": 0})


# ---------------------------------------------------------------- ranker.yaml


def _raw_ranker() -> dict[str, Any]:
    return load_yaml(REPO_CONFIGS / "ranker.yaml")


def test_repo_ranker_yaml_matches_the_proposal() -> None:
    config = load_ranker_config(REPO_CONFIGS / "ranker.yaml")
    # proposal §9.2-9.4
    assert (config.window_weeks, config.top_k) == (4, 12)
    assert (config.num_trees, config.checkpoint_every) == (2000, 100)
    assert config.sampling.ratios == (40, 60, 80) and config.sampling.seed == 42
    assert config.tuned_defaults == RankerTunedParams(31, 500, 40)
    # proposal §9.3 fixed parameters
    train = config.train_params
    assert train["objective"] == "lambdarank" and train["learning_rate"] == 0.05
    assert (train["feature_fraction"], train["bagging_fraction"], train["bagging_freq"]) == (
        0.8,
        0.8,
        1,
    )
    assert config.dataset_params["max_bin"] == 255
    assert train["metric"] == "ndcg" and train["eval_at"] == [12]
    # Reproducibility, and the one dataset parameter that is not LightGBM's default.
    assert train["deterministic"] is True and config.seed == 42
    assert config.dataset_params["feature_pre_filter"] is False
    # LightGBM 4.7.0 defaults that are written out because they change the result.
    assert train["lambdarank_truncation_level"] == 30 and train["bagging_by_query"] is False
    assert set(train) == RANKER_TRAIN_PARAM_KEYS
    assert set(config.dataset_params) == RANKER_DATASET_PARAM_KEYS


def test_histogram_mode_is_undecided_until_the_speed_test() -> None:
    # The rule is committed before the speed test; the mode is filled in after it.
    histogram = load_ranker_config(REPO_CONFIGS / "ranker.yaml").histogram
    assert histogram.mode == HISTOGRAM_UNDECIDED
    assert (histogram.comparison_trees, histogram.warmup_trees) == (50, 10)
    assert histogram.min_row_wise_speedup == 0.15


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (None, "num_trees", 2050, "multiple of checkpoint_every"),
        (None, "top_k", 0, "must be > 0"),
        ("sampling", "ratios", [60, 40], "ascending"),
        ("sampling", "ratios", [60, 80], "must be one of sampling.ratios"),
        ("sampling", "row_group_rows", 0, "must be > 0"),
        ("histogram", "mode", "auto", "expected one of"),
        ("histogram", "warmup_trees", 50, "warmup_trees < comparison_trees"),
        ("histogram", "min_row_wise_speedup", 1.5, "must be in"),
        ("speed_test", "num_trees", 250, "multiple of"),
        ("tuned_defaults", "num_leaves", 0, "must be > 0"),
    ],
)
def test_ranker_config_range_checks(
    section: str | None, key: str, value: object, message: str
) -> None:
    raw = _raw_ranker()
    target = raw if section is None else raw[section]
    target[key] = value
    with pytest.raises(ValueError, match=message):
        parse_ranker_config(raw)


def test_ranker_config_refuses_missing_and_unknown_lightgbm_parameters() -> None:
    raw = _raw_ranker()
    raw["train_params"]["num_leaves"] = 31  # a tuned value does not belong to the fixed ones
    with pytest.raises(ValueError, match="unknown keys"):
        parse_ranker_config(raw)
    raw = _raw_ranker()
    del raw["train_params"]["lambdarank_truncation_level"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_ranker_config(raw)
    raw = _raw_ranker()
    del raw["dataset_params"]["feature_pre_filter"]
    with pytest.raises(ValueError, match="missing keys"):
        parse_ranker_config(raw)
    raw = _raw_ranker()
    raw["tuned_defaults"]["learning_rate"] = 0.1
    with pytest.raises(ValueError, match="unknown keys"):
        parse_ranker_config(raw)
