"""CLI for M7: the speed test that runs once before the Ranker grid.

Usage:
    caffeinate -i uv run python scripts/m7_speed_test.py

Steps (hmrec.hpo.speed_test; every part in its own process, finished parts are skipped):
    1. sample the training folds of speed_test.target_fold (V1 <- F1 ... F4) and build
       the binary Dataset of the default configuration;
    2. train 50 trees with col-wise and with row-wise histograms, without evaluation, and
       apply the rule of configs/ranker.yaml (speed and memory only);
    3. train the default configuration to speed_test.num_trees (300) trees in the chosen
       mode and evaluate the target fold after every 100 trees;
    4. compute the RRF baseline of the target fold.
Then it logs: seconds per tree, scoring time, true memory peaks, the training-data
checks (positive share, rows per query), and MAP@12 / Recall@12 / NDCG@12 at every
checkpoint next to the RRF baseline. If the Ranker is below the baseline at the last
checkpoint it says STOP: investigate first, do not extrapolate the time.

Results go to results/m7_speed_test.csv, results/m7_speed_test_metrics.csv and
results/m7_runs.csv, never to results/v_phase*.csv. configs/ranker.yaml is not changed:
the chosen histogram mode is written into it by hand afterwards and committed.
Run it on a clean working tree (AGENTS.md §3.4).
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

from hmrec.hpo.speed_test import report_lines, run_speed_test, training_records  # noqa: E402
from hmrec.hpo.v_phase import start_m7_run  # noqa: E402
from hmrec.logging_utils import get_logger  # noqa: E402
from hmrec.pipeline import run_script  # noqa: E402

LOGGER = get_logger("m7_speed_test")


def main() -> None:
    """Run the speed test and log its report."""
    context, _ = start_m7_run("m7_speed_test", run_script)
    summary = run_speed_test(context)
    for line in report_lines(summary, training_records(context), context.grid.metrics):
        LOGGER.info(line)


if __name__ == "__main__":
    main()
