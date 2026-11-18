"""CLI for M8: the final evaluation on T1-T4 with all locked configs (proposal §10).

Usage, in this order:
    caffeinate -i uv run python scripts/m8_final_eval.py --stage rehearsal
    caffeinate -i uv run python scripts/m8_final_eval.py --stage formal
    uv run python scripts/m8_final_eval.py --stage report

rehearsal   The M8 code path with the locked configuration predicts the rehearsal
            fold (configs/m8_final.yaml: V4). No label of a T fold is read. Every
            number must equal what M7 recorded for the fold, exactly; the outcome
            goes to results/m8_rehearsal_check.csv.
formal      THE formal run, done ONCE. It starts only if the latest rehearsal passed
            with the same code, and it records the time and commit of the first read
            of T labels (results/t_phase_label_access.csv) before any step runs.
            Results: results/t_phase.csv, t_phase_metrics.csv, t_phase_recall.csv,
            t_phase_goals.csv, m8_runs.csv. If it is interrupted, the same command
            continues it (finished steps are skipped); with changed code it refuses.
report      Prints the recorded numbers and the verdict on G1-G3. Reads result
            tables only.

Every step runs in its own process (scripts/m8_step.py).
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
from datetime import datetime  # noqa: E402

from hmrec.final.process import start_final_process  # noqa: E402
from hmrec.final.report import final_verdicts, summary_lines  # noqa: E402
from hmrec.final.run import run_formal, run_rehearsal  # noqa: E402
from hmrec.logging_utils import get_logger  # noqa: E402
from hmrec.pipeline import run_script  # noqa: E402

LOGGER = get_logger("m8_final_eval")
STAGE_REHEARSAL = "rehearsal"
STAGE_FORMAL = "formal"
STAGE_REPORT = "report"
STAGES = (STAGE_REHEARSAL, STAGE_FORMAL, STAGE_REPORT)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=STAGES)
    return parser.parse_args()


def main() -> None:
    """Run one stage."""
    stage = parse_args().stage
    context = start_final_process(f"m8_final_eval_{stage}")
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(context.base.num_threads))
    scripts_dir = context.paths.project_root / "scripts"
    now = datetime.now().isoformat(timespec="seconds")
    if stage == STAGE_REHEARSAL:
        run_rehearsal(context, run_script, scripts_dir, now)
    elif stage == STAGE_FORMAL:
        run_formal(context, run_script, scripts_dir, now)
    else:
        for line in summary_lines(context, final_verdicts(context)):
            LOGGER.info(line)


if __name__ == "__main__":
    main()
