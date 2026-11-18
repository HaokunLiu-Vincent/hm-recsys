"""A fake process launcher for the V-phase driver tests (not a test module itself).

The real driver starts scripts/m7_*.py in new processes. FakeLauncher stands in
for hmrec.pipeline.run_script: it records every call and writes what the real
script would leave behind (sample records, a Dataset record, result rows), with
scores taken from a function the test supplies. No model is trained.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

from hmrec.config import RankerTunedParams, load_selection_rule
from hmrec.hpo.v_phase import BASELINE_SCRIPT, DATASET_SCRIPT, SAMPLE_SCRIPT, TRIAL_SCRIPT
from hmrec.hpo.v_records import (
    RRF_BASELINE,
    MetricRow,
    RankerTrialRecord,
    append_metric_row,
    append_trial,
    results_files,
)
from hmrec.hpo.v_trial import (
    TrialInputs,
    TrialOutcome,
    TrialSpec,
    model_config,
    ranker_config_id,
    run_trial,
)
from hmrec.ranker.dataset import RECORD_FILE
from hmrec.ranker.predict import checkpoints
from hmrec.ranker.runtime import LightGBMRuntime, read_lightgbm_runtime
from hmrec.ranker.sampling import record_file

N_QUERIES = 100
REPO = Path(__file__).resolve().parents[2]
RULE = load_selection_rule(REPO / "configs" / "selection_rule.yaml")

# A small grid with the shape of configs/hpo/v_phase.yaml, for the tiny Ranker config
# of ranker_example (defaults 4 / 5 / 2; ratios 2 and 4; checkpoints 10 and 20).
TINY_GRID_RAW = {
    "phase": "V",
    "target_folds": ["V1", "V2", "V3", "V4"],
    "metric": "map@12",
    "extra_metrics": ["recall@12", "ndcg@12"],
    "stop_if_winner_uses_all_trees": True,
    "g1_min_pooled_score": 0.03,
    "steps": [
        {
            "name": "leaves_x_min_data",
            "kind": "select",
            "order": "listed",
            "candidates": [
                {"num_leaves": 4, "min_data_in_leaf": 5},
                {"num_leaves": 4, "min_data_in_leaf": 2},
                {"num_leaves": 8, "min_data_in_leaf": 5},
            ],
        },
        {
            "name": "negatives_per_positive",
            "kind": "select",
            "order": "listed",
            "candidates": [{"negatives_per_positive": 2}, {"negatives_per_positive": 4}],
        },
    ],
    "experiments": ["rrf_baseline", "minus_r6"],
}
# (tuned parameters, fold, number of trees) -> MAP@12 of that fold.
ScoreFunction = Callable[[RankerTunedParams, str, int], float]


def parse_arguments(arguments: Sequence[str]) -> dict[str, list[str]]:
    """["--fold", "F1", "--ratios", "2", "4", "--no-eval"] -> {"fold": ["F1"], ...}."""
    parsed: dict[str, list[str]] = {}
    name = ""
    for argument in arguments:
        if argument.startswith("--"):
            name = argument[2:]
            parsed[name] = []
        else:
            parsed[name].append(argument)
    return parsed


class FakeLauncher:
    """Stands in for run_script; see the module docstring.

    Attributes:
        calls: (script name, parsed arguments) of every call, in order.
        seconds_per_tree: histogram mode -> the value written to trial rows.
        peak_gb: histogram mode -> the true memory peak written to trial rows.
    """

    def __init__(self, inputs: TrialInputs, score: ScoreFunction, baseline: float = 0.02) -> None:
        """Create a launcher that writes under `inputs.paths` with scores from `score`."""
        self.inputs = inputs
        self.score = score
        self.baseline = baseline
        self.calls: list[tuple[str, dict[str, list[str]]]] = []
        self.seconds_per_tree = {"col_wise": 1.0, "row_wise": 1.0}
        self.peak_gb = {"col_wise": 2.0, "row_wise": 3.0}

    def __call__(self, script: Path, arguments: Sequence[str]) -> float:
        """Record the call and leave the script's outputs behind."""
        parsed = parse_arguments(arguments)
        self.calls.append((script.name, parsed))
        handlers = {
            SAMPLE_SCRIPT: self._sample,
            DATASET_SCRIPT: self._dataset,
            TRIAL_SCRIPT: self._trial,
            BASELINE_SCRIPT: self._baseline,
        }
        handlers[script.name](parsed)
        return 0.0

    def count(self, script_name: str) -> int:
        """How often a script was started."""
        return sum(1 for name, _ in self.calls if name == script_name)

    def _sample(self, args: dict[str, list[str]]) -> None:
        for ratio in args["ratios"]:
            sample_file = self.inputs.paths.ranker_sample_file(
                args["experiment"][0], args["fold"][0], int(ratio)
            )
            sample_file.parent.mkdir(parents=True, exist_ok=True)
            record_file(sample_file).write_text("{}")

    def _dataset(self, args: dict[str, list[str]]) -> None:
        directory = self.inputs.paths.ranker_dataset_dir(
            args["experiment"][0], args["target"][0], int(args["ratio"][0])
        )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / RECORD_FILE).write_text("{}")

    def _metric_row(self, experiment: str, config_id: str, fold: str, k: int, value: float) -> None:
        _, metrics_file = results_files(self.inputs.paths, self._area)
        scores = dict.fromkeys(self.inputs.metrics, value)
        row = MetricRow(experiment, config_id, fold, k, N_QUERIES, scores, 0.5 + k / 10000)
        append_metric_row(metrics_file, row, self.inputs.metrics, "fake", "t")

    def _trial(self, args: dict[str, list[str]]) -> None:
        self._area = args["area"][0]
        ranker = self.inputs.ranker
        tuned = RankerTunedParams(
            int(args["num-leaves"][0]), int(args["min-data-in-leaf"][0]), int(args["ratio"][0])
        )
        experiment, fold = args["experiment"][0], args["target"][0]
        mode, num_trees = args["histogram-mode"][0], int(args["num-trees"][0])
        config = model_config(ranker, tuned, experiment, mode)
        config_id = ranker_config_id(config)
        evaluated = "no-eval" not in args
        if evaluated:
            for k in checkpoints(num_trees, ranker.checkpoint_every):
                self._metric_row(experiment, config_id, fold, k, self.score(tuned, fold, k))
        trial_file, _ = results_files(self.inputs.paths, self._area)
        append_trial(
            trial_file,
            RankerTrialRecord(
                phase="V",
                experiment=experiment,
                step=args["step"][0],
                config_id=config_id,
                fold=fold,
                train_folds=("F1",),
                params=json.loads(json.dumps(tuned.__dict__)),
                config=config,
                histogram_mode=mode,
                num_trees=num_trees,
                evaluated=evaluated,
                n_train_rows=1000,
                n_train_queries=100,
                n_scored_rows=1200 if evaluated else 0,
                n_queries=N_QUERIES if evaluated else 0,
                load_seconds=0.1,
                train_seconds=self.seconds_per_tree[mode] * num_trees,
                seconds_per_tree=self.seconds_per_tree[mode],
                score_seconds=5.0 if evaluated else 0.0,
                total_seconds=10.0,
                sampled_peak_rss_gb=1.0,
                process_max_rss_gb=self.peak_gb[mode],
                model_sha256="",
                runtime=LightGBMRuntime(
                    "4.7.0", "/opt/homebrew/Cellar/libomp/22.1.4/lib/libomp.dylib", "22.1.4"
                ),
                git_commit="fake",
                git_dirty=False,
                finished_at="t",
            ),
        )

    def _baseline(self, args: dict[str, list[str]]) -> None:
        self._area = args["area"][0]
        self._metric_row(RRF_BASELINE, RRF_BASELINE, args["fold"][0], 0, self.baseline)


def run_and_record_trial(inputs: TrialInputs, spec: TrialSpec) -> TrialOutcome:
    """Run a REAL (tiny) trial in this process and write its rows, like scripts/m7_train_eval.py."""
    outcome = run_trial(inputs, spec, "it")
    trial_file, metrics_file = results_files(inputs.paths, spec.area)
    for metric_row in outcome.metric_rows:
        append_metric_row(metrics_file, metric_row, inputs.metrics, "it", "t")
    append_trial(
        trial_file,
        RankerTrialRecord(
            phase="V",
            experiment=spec.experiment,
            step=spec.step,
            config_id=outcome.config_id,
            fold=spec.target_fold,
            train_folds=outcome.train_folds,
            params=asdict(spec.tuned),
            config=outcome.config,
            histogram_mode=spec.histogram_mode,
            num_trees=spec.num_trees,
            evaluated=spec.evaluate,
            n_train_rows=outcome.n_train_rows,
            n_train_queries=outcome.n_train_queries,
            n_scored_rows=outcome.n_scored_rows,
            n_queries=outcome.n_queries,
            load_seconds=outcome.load_seconds,
            train_seconds=outcome.train_seconds,
            seconds_per_tree=outcome.seconds_per_tree,
            score_seconds=outcome.score_seconds,
            total_seconds=outcome.train_seconds + outcome.score_seconds,
            sampled_peak_rss_gb=0.5,
            process_max_rss_gb=0.5,
            model_sha256=outcome.model_sha256,
            runtime=read_lightgbm_runtime(),
            git_commit="it",
            git_dirty=False,
            finished_at="t",
        ),
    )
    return outcome
