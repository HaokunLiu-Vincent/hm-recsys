import copy
from pathlib import Path
from typing import Any

import pytest

from hmrec.config import (
    load_base_config,
    load_selection_rule,
    load_yaml,
    parse_base_config,
    parse_selection_rule,
)

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"

VALID_BASE: dict[str, Any] = {
    "paths": {"data_root": "data", "artifacts_root": "artifacts", "results_root": "results"},
    "seed": 7,
    "num_threads": 6,
    "memory": {"limit_gb": 6, "sample_interval_seconds": 0.5},
    "chunking": {"query_chunk_size": 10, "item_chunk_size": 3},
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
    # AGENTS.md §4: 6 threads, ~6 GB peak per task.
    assert config.num_threads == 6
    assert config.memory.limit_gb == 6.0
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
