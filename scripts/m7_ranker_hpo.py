"""CLI for M7: the Ranker's V phase on V1-V4 (proposal §9.4, §9.5).

Usage (each stage can be run again; finished trials are skipped):
    caffeinate -i uv run python scripts/m7_ranker_hpo.py --stage baseline
    caffeinate -i uv run python scripts/m7_ranker_hpo.py --stage grid [--last-step NAME]
    caffeinate -i uv run python scripts/m7_ranker_hpo.py --stage minus_r6
    caffeinate -i uv run python scripts/m7_ranker_hpo.py --stage importance
    uv run python scripts/m7_ranker_hpo.py --stage report

Stages:
    baseline    the RRF baseline of every target fold;
    grid        the steps of configs/hpo/v_phase.yaml in order: every missing trial is
                run in its own process (sampling and the binary Dataset first, if they
                are missing), then the step is decided and written to
                results/v_phase_steps.csv. Stops after a step whose winner needs all trees;
    minus_r6    the final configuration without R6 (needs a finished grid);
    importance  gain importance of the final configuration's models (analysis only);
    report      G1 / G3 and the three systems side by side; starts nothing.

This script only starts other scripts and applies the selection rule; it reads no
feature and no label itself. The grid refuses to start while histogram.mode in
configs/ranker.yaml is "undecided" (run scripts/m7_speed_test.py first).
Run it on a clean working tree (AGENTS.md §3.4).
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402

from hmrec.hpo.v_phase import (  # noqa: E402
    compare_systems,
    final_winner,
    goal_lines,
    refuse_to_launch,
    run_baseline,
    run_grid,
    run_minus_r6,
    start_m7_run,
    write_importance,
)
from hmrec.hpo.v_records import AREA_V_PHASE, read_metric_rows, results_files  # noqa: E402
from hmrec.logging_utils import get_logger  # noqa: E402
from hmrec.pipeline import run_script  # noqa: E402

LOGGER = get_logger("m7_ranker_hpo")
STAGES = ("baseline", "grid", "minus_r6", "importance", "report")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--last-step", help="grid only: stop after this step (default: all steps)")
    return parser.parse_args()


def main() -> None:
    """Run one stage."""
    args = parse_args()
    launch = refuse_to_launch if args.stage == "report" else run_script
    context, _ = start_m7_run(f"m7_ranker_hpo_{args.stage}", launch)
    grid = context.grid
    if args.stage == "baseline":
        run_baseline(context, AREA_V_PHASE, grid.target_folds)
        return
    if args.stage == "grid":
        run_grid(context, args.last_step)
        return
    winner = final_winner(context)
    if args.stage == "minus_r6":
        run_minus_r6(context, winner)
    if args.stage == "importance":
        write_importance(context, winner)
    _, metrics_file = results_files(context.inputs.paths, AREA_V_PHASE)
    metric_rows = read_metric_rows(metrics_file, grid.metrics)
    comparison = compare_systems(context, winner, metric_rows)
    for line in goal_lines(comparison, context.rule, grid.g1_min_pooled_score):
        LOGGER.info(line)


if __name__ == "__main__":
    main()
