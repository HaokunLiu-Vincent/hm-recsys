"""Calibration item E, first process: the wide table from the container's own candidates.

Items A-D each isolate one step. Item E runs the steps one after the other, as a
weekly tick does: the container's own candidates of the week are deduplicated and
turned into the 71 locked features (this module); the Ranker model of item D then
scores them and the result is evaluated (hmrec.calibration.ranker_items).

NO new computation happens here. The wide table is built by the locked feature
code, hmrec.features.build (load_fold_data, load_history, load_candidates,
load_dense_sources, build_fold_features), exactly as scripts/m6_build_features.py
calls it. Two things differ from stage 1:

    where the candidates come from   the calibration trees (R1-R5 of one run, R6
                                     of the run item E is started for), through
                                     hmrec.paths.EndToEndInputPaths
    where the features go            the calibration tree of the run; no stage-1
                                     file is written

The feature code checks while it builds that the table has exactly the expected
number of rows, that no (query_id, article_idx) repeats, and that tt_dot and
wals_dot equal the R6 and R5 scores bit for bit. The expected number of rows is
counted here from the candidates of THIS run, independently of the feature code.

No label is read. This module imports neither torch nor lightgbm.
"""

import polars as pl

from hmrec.calibration.context import CalibrationContext, ItemOutcome, timed
from hmrec.calibration.records import Measurement
from hmrec.calibration.references import stage1_run_row, verify_files
from hmrec.config import load_features_config, load_preprocess_config
from hmrec.data.folds import load_fold_manifest
from hmrec.features.build import (
    BuildResult,
    FoldInputs,
    build_fold_features,
    load_candidates,
    load_dense_sources,
    load_fold_data,
    load_history,
)
from hmrec.locks import (
    check_features_match_lock,
    file_sha256,
    load_features_lock,
    load_twotower_lock,
)
from hmrec.logging_utils import get_logger
from hmrec.memory import read_process_peak_memory
from hmrec.paths import EndToEndInputPaths, end_to_end_input_paths

LOGGER = get_logger("calibration.feature_items")

ITEM_E = "E"
FEATURES_SCOPE = "features"
FEATURES_STEP = "features"
KEYS = ["query_id", "article_idx"]
M6_RUNS = "results/m6_runs.csv"


def input_paths(context: CalibrationContext) -> EndToEndInputPaths:
    """Where item E reads the six channels from: R6 of this run, R1-R5 of the configured run."""
    return end_to_end_input_paths(
        context.paths, context.platform_name, context.run, context.config.statistical_run
    )


def deduplicated_candidate_count(candidates: dict[str, pl.DataFrame]) -> int:
    """The number of distinct (query_id, article_idx) pairs over all channels.

    Args:
        candidates: Channel -> its candidates; columns query_id, article_idx.

    Returns:
        The row count the wide table must have: one row per distinct pair.
    """
    pairs = pl.concat([table.select(KEYS) for table in candidates.values()])
    return pairs.unique().height


def written_feature_keys(context: CalibrationContext) -> tuple[int, int]:
    """(rows, distinct (query_id, article_idx) pairs) of the feature files that were written.

    Read back from disk after the build, part by part, so the answer is about
    the files and not about tables in memory.
    """
    directory = context.main_paths.features_dir(context.config.fold)
    parts = sorted(directory.glob("part-*.parquet"))
    keys = pl.concat([pl.read_parquet(part, columns=KEYS) for part in parts])
    return keys.height, keys.unique().height


def run_end_to_end_features(context: CalibrationContext) -> ItemOutcome:
    """Item E, first process: build the 71 features of the week from this run's candidates.

    Raises:
        ValueError: If configs/features.yaml is not the locked feature list, an
            input has another sha256, the wide table does not have the expected
            number of rows, a key repeats, or tt_dot / wals_dot differ from the scores.
        FileExistsError: If the run already has features of the week.
    """
    paths, config, base = context.paths, context.config, context.base
    fold_name = config.fold
    sources = input_paths(context)
    verify_files(paths.processed_dir, config.processed_sha256, list(config.processed_sha256))
    verify_files(paths.processed_dir, config.embedding_sha256, list(config.embedding_sha256))
    queries_name = paths.queries_file(fold_name).relative_to(paths.artifacts_root).as_posix()
    verify_files(paths.artifacts_root, config.artifact_sha256, [queries_name])
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    features_config = load_features_config(paths.features_config_file)
    # The same two checks as scripts/m6_build_features.py: the features are locked, and
    # the feature history is the one R6 was locked with.
    check_features_match_lock(features_config, load_features_lock(paths.features_lock_file))
    locked_history = load_twotower_lock(paths.twotower_lock_file).channel.config["history_length"]
    if features_config.history_length != locked_history:
        raise ValueError(
            f"features.history_length is {features_config.history_length}, but the locked R6 "
            f"uses {locked_history}"
        )
    output = context.main_paths
    if output.features_dir(fold_name).exists():
        raise FileExistsError(
            f"{output.features_dir(fold_name)} already exists. A calibration run never reuses "
            "or replaces an output; look at the earlier attempt first."
        )
    fold = load_fold_manifest(paths.fold_manifest_file).get(fold_name)

    def build() -> tuple[BuildResult, int]:
        data = load_fold_data(paths, preprocess, fold)
        history = load_history(data, features_config)
        candidates = load_candidates(sources, fold_name)
        expected_rows = deduplicated_candidate_count(candidates)
        inputs = FoldInputs(
            data=data,
            history=history,
            candidates=candidates,
            dense=load_dense_sources(sources, preprocess, data, history, features_config, base),
            expected_rows=expected_rows,
        )
        result = build_fold_features(
            output, inputs, features_config, base.chunking.query_chunk_size, context.stamp
        )
        return result, expected_rows

    (result, expected_rows), seconds, sampled_peak = timed(context, f"{fold_name} features", build)
    item_peak = read_process_peak_memory()
    written_rows, written_keys = written_feature_keys(context)
    stage1 = stage1_run_row(paths, fold_name, FEATURES_STEP)
    codes_name = paths.category_codes_file.relative_to(paths.artifacts_root).as_posix()
    same_codes = file_sha256(output.category_codes_file) == config.artifact_sha256[codes_name]
    consistent = written_rows == expected_rows and written_keys == written_rows
    where = f"this run's own candidates (R1-R5 of {config.statistical_run}, R6 of {context.run})"

    def row(
        quantity: str, value: float | int | str, reference: float | int | str, source: str
    ) -> Measurement:
        return Measurement(ITEM_E, FEATURES_SCOPE, quantity, value, reference, source=source)

    rows = [
        Measurement(
            ITEM_E,
            FEATURES_SCOPE,
            "feature_rows_minus_deduplicated_candidates",
            written_rows - expected_rows,
            0,
            "feature rows equal the deduplicated candidates of the same run; no duplicate key",
            consistent,
            where,
        ),
        row("deduplicated_candidates_of_this_run", expected_rows, int(stage1["n_rows"]), M6_RUNS),
        row("feature_rows", written_rows, int(stage1["n_rows"]), M6_RUNS),
        row("duplicate_keys", written_rows - written_keys, 0, where),
        row("queries", result.n_queries, result.n_queries, where),
        row(
            "tt_dot_rows_equal_to_r6_score",
            result.tt_dot_rows_checked,
            result.tt_dot_rows_checked,
            where,
        ),
        row(
            "wals_dot_rows_equal_to_r5_score",
            result.wals_dot_rows_checked,
            result.wals_dot_rows_checked,
            where,
        ),
        row("category_codes_sha256_equal", same_codes, True, f"artifacts/{codes_name}"),
        row("seconds", result.seconds, float(stage1["generate_seconds"]), M6_RUNS),
        row(
            "max_rss_gb", item_peak.process_max_rss_gb, float(stage1["process_max_rss_gb"]), M6_RUNS
        ),
    ]
    LOGGER.info(
        "E features: %d rows in %d parts, %d queries",
        written_rows,
        result.n_parts,
        result.n_queries,
    )
    return ItemOutcome(
        item=ITEM_E,
        scope=FEATURES_SCOPE,
        item_seconds=seconds,
        sampled_peak_rss_gb=sampled_peak,
        item_peak=item_peak,
        measurements=rows,
        detail=(
            f"{written_rows} rows in {result.n_parts} parts; R1-R5 of {config.statistical_run}, "
            f"R6 of {context.run}; tt_sim step {result.history_similarity_seconds:.1f} s"
        ),
    )
