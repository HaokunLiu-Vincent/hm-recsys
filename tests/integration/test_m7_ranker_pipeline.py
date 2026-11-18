"""Integration test of M7 on tiny data, from sampling to selection.

sampling -> Dataset -> training -> scoring -> evaluation -> selection, plus the speed
test, the RRF baseline, the run without R6 and the feature importance.

The real driver (hmrec.hpo.v_phase, hmrec.hpo.speed_test) starts scripts/m7_*.py in
new processes. Those scripts read the project's real paths, so here an in-process
launcher does what each script does with the same library calls, under a temporary
folder. Everything else is the real code: real sample files, real binary Datasets,
real (tiny) LightGBM models.

Data (tests/unit/ranker_example.py): 8 folds F1 ... F4, V1 ... V4, 60 queries each with
12 candidates. The only label of a query is its candidate with the largest
`ui_decay_ptype`; rrf_score is unrelated to it. So a trained Ranker must beat the
RRF baseline clearly on every fold.
"""

import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType

import polars as pl

from hmrec.config import RankerTunedParams
from hmrec.csv_records import read_rows
from hmrec.hpo.speed_test import run_speed_test
from hmrec.hpo.v_phase import (
    BASELINE_SCRIPT,
    DATASET_SCRIPT,
    IMPORTANCE_COLUMNS,
    SAMPLE_SCRIPT,
    TRIAL_SCRIPT,
    GridContext,
    compare_systems,
    final_winner,
    parse_v_phase_grid,
    run_baseline,
    run_grid,
    run_minus_r6,
    sample_job,
    step_columns,
    write_importance,
)
from hmrec.hpo.v_records import (
    AREA_V_PHASE,
    RankerTrialRecord,
    append_metric_row,
    append_trial,
    read_metric_rows,
    read_trial_rows,
    results_files,
)
from hmrec.hpo.v_trial import (
    TrialInputs,
    TrialSpec,
    build_and_save_dataset,
    plan_dataset,
    rrf_baseline_row,
    run_trial,
)
from hmrec.ranker.runtime import read_lightgbm_runtime
from hmrec.ranker.sampling import sample_fold
from hmrec.ranker.train import first_trees_sha256, load_model

UNIT_TESTS = Path(__file__).resolve().parents[1] / "unit"


def _load(name: str) -> ModuleType:
    """Import a helper module of tests/unit by file (it is not a package)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, UNIT_TESTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ranker_example = _load("ranker_example")
v_phase_example = _load("v_phase_example")


class InProcessLauncher:
    """Does in this process what each scripts/m7_*.py does in its own process."""

    def __init__(self, inputs: TrialInputs) -> None:
        """Create a launcher that works under `inputs.paths`."""
        self.inputs = inputs
        self.calls: list[str] = []

    def __call__(self, script: Path, arguments: Sequence[str]) -> float:
        """Run the library calls of one script."""
        args = v_phase_example.parse_arguments(arguments)
        self.calls.append(script.name)
        inputs = self.inputs
        if script.name == SAMPLE_SCRIPT:
            ratios = [int(ratio) for ratio in args["ratios"]]
            job = sample_job(inputs, args["fold"][0], args["experiment"][0], ratios, "it")
            sample_fold(job, inputs.ranker.sampling)
        elif script.name == DATASET_SCRIPT:
            plan = plan_dataset(
                inputs, args["experiment"][0], args["target"][0], int(args["ratio"][0]), "it"
            )
            build_and_save_dataset(plan, inputs.ranker.sampling.row_group_rows)
        elif script.name == TRIAL_SCRIPT:
            self._trial(args)
        elif script.name == BASELINE_SCRIPT:
            _, metrics_file = results_files(inputs.paths, args["area"][0])
            row = rrf_baseline_row(inputs, args["fold"][0])
            append_metric_row(metrics_file, row, inputs.metrics, "it", "t")
        return 0.0

    def _trial(self, args: dict[str, list[str]]) -> None:
        inputs = self.inputs
        tuned = RankerTunedParams(
            int(args["num-leaves"][0]), int(args["min-data-in-leaf"][0]), int(args["ratio"][0])
        )
        spec = TrialSpec(
            area=args["area"][0],
            experiment=args["experiment"][0],
            step=args["step"][0],
            target_fold=args["target"][0],
            tuned=tuned,
            num_trees=int(args["num-trees"][0]),
            histogram_mode=args["histogram-mode"][0],
            evaluate="no-eval" not in args,
        )
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
                params=asdict(tuned),
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


def _context(tmp_path: Path) -> tuple[GridContext, InProcessLauncher]:
    inputs: TrialInputs = ranker_example.tiny_inputs(tmp_path)
    ranker_example.write_learnable_folds(inputs.paths)
    for seed, (fold, first_query_id) in enumerate({"V2": 6000, "V3": 7000, "V4": 8000}.items()):
        ranker_example.write_learnable_fold(inputs.paths, fold, first_query_id, 60, 100 + seed)
    launcher = InProcessLauncher(inputs)
    # On this tiny data the curve can still rise at the last checkpoint (20 trees); the
    # stop rule itself is tested in tests/unit/test_v_phase.py, so it is switched off here.
    grid = replace(
        parse_v_phase_grid(v_phase_example.TINY_GRID_RAW), stop_if_winner_uses_all_trees=False
    )
    return GridContext(
        inputs, grid, v_phase_example.RULE, Path("scripts"), launcher, "it"
    ), launcher


def test_m7_from_sampling_to_selection(tmp_path: Path) -> None:
    context, launcher = _context(tmp_path)
    inputs, grid = context.inputs, context.grid
    folds = grid.target_folds

    # ---- the speed test: V1 only, into its own result files
    summary = run_speed_test(context)
    assert summary.timing.chosen_mode in ("col_wise", "row_wise")
    assert [row.num_trees for row in summary.checkpoints] == [10, 20]
    assert summary.below_baseline is False
    assert summary.checkpoints[-1].scores["map@12"] > 0.8 > summary.baseline.scores["map@12"]
    assert not inputs.paths.v_phase_results_file.exists()
    speed_trial = summary.trial

    # ---- the grid: 3 configurations in step 1, 1 new one in step 2, on 4 folds
    results = run_grid(context, None)
    assert [result.step for result in results] == ["leaves_x_min_data", "negatives_per_positive"]
    winner = final_winner(context)
    trial_rows = read_trial_rows(inputs.paths.v_phase_results_file)
    assert len(trial_rows) == (3 + 1) * 4
    assert {row["num_trees"] for row in trial_rows} == {"20"}
    assert {row["train_folds"] for row in trial_rows if row["fold"] == "V3"} == {"F3 F4 V1 V2"}
    # Samples: every training fold of V1 ... V4 once (the speed test already drew F1 ... F4).
    assert launcher.calls.count(SAMPLE_SCRIPT) == 7
    step_rows = read_rows(inputs.paths.v_phase_steps_file, step_columns(folds))
    assert sum(row["selected"] == "True" for row in step_rows) == 2  # one winner per step
    assert winner.choice.num_trees in (10, 20)

    # ---- training is reproducible: the grid's default V1 model = the speed test's model
    default_id = results[0].default.config_id
    if speed_trial["histogram_mode"] == inputs.ranker.histogram.mode:
        assert speed_trial["config_id"] == default_id
        grid_model = load_model(
            inputs.paths.ranker_model_file(AREA_V_PHASE, "full", default_id, "V1")
        )
        speed_model = load_model(
            inputs.paths.ranker_model_file("speed_test", "full", default_id, "V1")
        )
        assert first_trees_sha256(grid_model, 20) == first_trees_sha256(speed_model, 20)

    # ---- baseline, the system without R6, goals
    run_baseline(context, AREA_V_PHASE, folds)
    run_minus_r6(context, winner)
    metric_rows = read_metric_rows(inputs.paths.v_phase_metrics_file, grid.metrics)
    comparison = compare_systems(context, winner, metric_rows)
    assert comparison.folds_above_baseline == 4
    assert comparison.full.pooled_score > 0.8 > comparison.baseline.pooled_score
    assert comparison.minus_r6 is not None
    # R6 carries no signal in this data, so the system without it is just as good.
    assert comparison.minus_r6.pooled_score > 0.8
    minus_rows = [row for row in read_trial_rows(inputs.paths.v_phase_results_file)
                  if row["experiment"] == "minus_r6"]  # fmt: skip
    assert len(minus_rows) == 4
    assert {row["num_trees"] for row in minus_rows} == {str(winner.choice.num_trees)}

    # ---- the saved Top-12 of a trial reproduces its recorded metric
    predictions = pl.read_parquet(
        inputs.paths.ranker_predictions_file(AREA_V_PHASE, "full", winner.config_id, "V2")
    ).filter(pl.col("num_trees") == winner.choice.num_trees)
    labels = pl.read_parquet(inputs.paths.labels_file("V2"))
    hits = predictions.join(labels, on=["query_id", "article_idx"], how="inner")
    # One label per query: AP@12 = 1 / rank of the label (0 if it is not in the Top-12).
    reciprocal_ranks = [1.0 / rank for rank in hits["rank"].to_list()]
    by_hand = sum(reciprocal_ranks) / labels["query_id"].n_unique()
    assert abs(by_hand - winner.scores.fold_scores["V2"]) < 1e-12

    # ---- feature importance of the final configuration (analysis only)
    write_importance(context, winner)
    write_importance(context, winner)  # written once
    importance = read_rows(inputs.paths.v_phase_feature_importance_file, IMPORTANCE_COLUMNS)
    assert len(importance) == 71 * 4
    top = max(
        (row for row in importance if row["fold"] == "V1"), key=lambda row: float(row["gain"])
    )
    assert top["feature"] == ranker_example.SIGNAL_FEATURE

    # ---- running everything again starts nothing
    calls_before = len(launcher.calls)
    run_speed_test(context)
    run_grid(context, None)
    run_baseline(context, AREA_V_PHASE, folds)
    run_minus_r6(context, winner)
    assert len(launcher.calls) == calls_before
