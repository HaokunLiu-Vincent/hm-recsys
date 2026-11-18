"""CLI for M8: ONE step of the final evaluation for one fold, in its own process.

Started by scripts/m8_final_eval.py (the driver); not meant to be run by hand.

    uv run python scripts/m8_step.py --step trial --mode formal --fold T1 --experiment full

Steps: monitor, baseline, sample, dataset, trial (hmrec.final.process).
Modes: rehearsal (the configured V fold; reads no T label) or formal (T folds).
A formal step passes the gate first (hmrec.final.gate): without a passed
rehearsal of this code and a recorded start of the formal run it stops before
it reads anything.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402

from hmrec.final.context import MODES  # noqa: E402
from hmrec.final.process import run_worker, start_final_process, step_label  # noqa: E402
from hmrec.final.run import STEPS, StepRequest  # noqa: E402
from hmrec.logging_utils import get_logger  # noqa: E402
from hmrec.ranker.ablation import EXPERIMENTS, FULL  # noqa: E402

LOGGER = get_logger("m8_step")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True, choices=STEPS)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument(
        "--fold", required=True, help="the target fold (sample: the fold to sample)"
    )
    parser.add_argument("--experiment", default=FULL, choices=EXPERIMENTS)
    return parser.parse_args()


def main() -> None:
    """Run one step and record it."""
    started = time.perf_counter()
    args = parse_args()
    request = StepRequest(args.step, args.mode, args.fold, args.experiment)
    context = start_final_process(f"m8_{args.fold}_{step_label(request)}")
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(context.base.num_threads))
    run_worker(context, request, started)


if __name__ == "__main__":
    main()
