"""CLI for M4: union check of the six channels on D1-D4 (proposal §5.7).

Usage:
    /usr/bin/time -l caffeinate -i uv run python scripts/m4_union_check.py
        [--variant r3_active_pool] [--memory-sample-interval 0.05] [--run-id m4_union_check]

Uses each channel's D-phase selection (replayed from results/d_phase.csv, no new
trials) with the manual choices of configs/m4_union_check.yaml (R6). R1-R5
candidates are read from the union pre-check files when they exist; R6 is
predicted with the models the training chain saved (nothing is retrained).
Candidates go to artifacts/m4/candidates/, the table to results/m4_union_check.csv
(never replaced). Nothing is locked.

--variant NAME applies the variant's config changes on top (e.g. R3 limited to
A(a) after the pre-registered rule). A variant config without a trial is first
evaluated on D1-D4 and recorded in results/m4_trials.csv (it is a rule outcome,
never part of a selection); the table goes to results/m4_union_check_{NAME}.csv.

Folds are checked one at a time. Every stage logs its memory peak;
--memory-sample-interval sets how often RSS is sampled (default: configs/base.yaml).
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import dataclasses  # noqa: E402
import logging  # noqa: E402

import polars as pl  # noqa: E402
import torch  # noqa: E402

from hmrec.config import (  # noqa: E402
    load_base_config,
    load_catalog_config,
    load_preprocess_config,
    load_selection_rule,
    load_union_check_config,
)
from hmrec.data.folds import load_fold_manifest  # noqa: E402
from hmrec.hpo.d_phase import PHASE, load_fold_contexts  # noqa: E402
from hmrec.hpo.grid import load_phase_grid  # noqa: E402
from hmrec.hpo.runner import FoldContext, git_state, warn_uncommitted_changes  # noqa: E402
from hmrec.hpo.union_check import (  # noqa: E402
    R6_CHANNEL,
    CandidateFiles,
    UnionCheckRun,
    VariantEvaluation,
    apply_variant,
    build_r6_source,
    memory_stage_tracker,
    run_union_check,
    variant_step,
)
from hmrec.hpo.union_precheck import select_channels, write_precheck_table  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import MemoryTracker  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402

CHANNELS = (
    "r1_popularity",
    "r2_repurchase",
    "r3_sibling",
    "r4_itemcf",
    "r5_wals",
    R6_CHANNEL,
)
LOGGER = get_logger("m4_union_check")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default=None, help="a name under `variants` in the config")
    parser.add_argument(
        "--memory-sample-interval",
        type=float,
        default=None,
        help="seconds between RSS samples (default: memory.sample_interval_seconds)",
    )
    parser.add_argument("--run-id", default=None, help="log goes to artifacts/runs/{run_id}/")
    return parser.parse_args()


def main() -> None:
    """Run the union check (of the selections or of one variant) and write the table."""
    args = parse_args()
    paths = get_paths()
    run_id = args.run_id or ("m4_union_check" + (f"_{args.variant}" if args.variant else ""))
    setup_logging(logging.INFO, log_file=paths.run_dir(run_id) / "run.log")
    base = load_base_config(paths.base_config_file)
    memory = base.memory
    if args.memory_sample_interval is not None:
        memory = dataclasses.replace(memory, sample_interval_seconds=args.memory_sample_interval)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(base.num_threads))
    LOGGER.info("memory is sampled every %.3f s", memory.sample_interval_seconds)
    torch.set_num_threads(base.num_threads)
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("git HEAD %s, uncommitted code/config changes: %s", commit, dirty)
    if dirty:
        warn_uncommitted_changes(commit)
    settings = load_union_check_config(paths.union_check_config_file)
    if args.variant is not None and args.variant not in settings.variants:
        raise ValueError(f"unknown variant {args.variant!r}; known: {sorted(settings.variants)}")
    results_file = (
        paths.m4_union_check_variant_file(args.variant)
        if args.variant
        else paths.m4_union_check_results_file
    )
    if results_file.exists():
        raise FileExistsError(f"{results_file} already exists")
    grid = load_phase_grid(paths.d_phase_grid_file)
    rule = load_selection_rule(paths.selection_rule_file)
    folds = load_fold_manifest(paths.fold_manifest_file).by_phase(PHASE)
    fold_names = [fold.name for fold in folds]
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    catalog = load_catalog_config(paths.catalog_config_file)
    selected = select_channels(paths, CHANNELS, grid, rule, fold_names, settings.manual_selection)
    if args.variant is not None:
        # A variant trial is evaluated on all D folds like any trial; the contexts
        # are light (queries, labels, active pool) and are released right after.
        contexts = load_fold_contexts(paths, folds, preprocess, catalog)
        evaluation = VariantEvaluation(paths, grid, rule, base, contexts, commit, dirty)
        overrides = settings.variants[args.variant]
        selected = apply_variant(
            selected, overrides, evaluation.evaluator_of, variant_step(args.variant)
        )
        del contexts, evaluation
    r6_choice = next(choice for choice in selected if choice.channel == R6_CHANNEL)
    run = UnionCheckRun(
        selected=selected,
        base=base,
        files=CandidateFiles(
            reuse=paths.union_precheck_candidates_file, write=paths.m4_candidates_file
        ),
        r6=build_r6_source(paths, preprocess, base, r6_choice, fold_names),
        track=memory_stage_tracker(memory),
    )
    folds_by_name = {fold.name: fold for fold in folds}

    def load_context(fold_name: str) -> FoldContext:
        return load_fold_contexts(paths, [folds_by_name[fold_name]], preprocess, catalog)[0]

    with MemoryTracker(
        "M4 union check",
        LOGGER,
        limit_gb=memory.limit_gb,
        sample_interval_seconds=memory.sample_interval_seconds,
    ):
        table = run_union_check(run, fold_names, load_context, settings, grid.fold_groups)
    write_precheck_table(table, results_file)
    # Print every row and column (polars shortens long tables by default).
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=5, fmt_str_lengths=60):
        LOGGER.info("M4 table -> %s\n%s", results_file, table)


if __name__ == "__main__":
    main()
