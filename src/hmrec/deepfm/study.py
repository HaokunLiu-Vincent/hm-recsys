"""The DeepFM study on V1-V4: its rules, applied to the recorded metric rows (proposal §16.5).

Nothing here trains or reads data: the functions take rows of
results/deepfm_metrics.csv and say what the rules decide. Training happens in
one process per (configuration, seed, fold); hmrec.deepfm.study_run starts them.

Scores. For a configuration with a seed:
    pooled(e) = sum_f n_f * MAP_f(e) / sum_f n_f      (query-weighted over the folds)
    epoch rule: e* = the smallest epoch whose pooled value is within
                `epoch_tolerance` of the best one (hmrec.hpo.selection.choose_num_trees)
    the configuration's score is pooled(e*); its fold scores are MAP_f(e*).

Step 0 (noise): the default configuration with several seeds;
    range = (max - min) / min of their pooled scores.

Tuning, one factor at a time: hmrec.hpo.selection.choose_winner, the rule of
proposal §4.2 (at least 1% relative gain of the pooled score AND strictly better
in at least 3 of 4 folds; among passing candidates within 1% of the best, the
one with the highest priority).

A configuration is described by a Variant: the six settings that the study
changes. Everything else stays as in configs/deepfm.yaml.

This module imports neither torch nor LightGBM.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from hmrec.config import DeepFMConfig, DeepFMStudyConfig, DeepFMStudyStep, SelectionRule
from hmrec.deepfm.records import deepfm_config_id
from hmrec.hpo.selection import StepDecision, TrialScores, choose_num_trees, choose_winner

LEARNING_RATE = "learning_rate"
EMBEDDING_DIM = "embedding_dim"
DROPOUT = "dropout"


@dataclass(frozen=True)
class Variant:
    """The settings of a configuration that the study changes.

    Attributes:
        learning_rate: Adam learning rate.
        embedding_dim: Length of every embedding.
        dropout: Dropout after every hidden layer.
        components: "first_order", "fm", "deep" or "full".
        numeric_input: "binned" or "standardised".
        loss: "pointwise_bce" or "listwise_softmax".
    """

    learning_rate: float
    embedding_dim: int
    dropout: float
    components: str
    numeric_input: str
    loss: str


def variant_of(config: DeepFMConfig) -> Variant:
    """The six study settings of a configuration."""
    return Variant(
        learning_rate=config.training.learning_rate,
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout,
        components=config.model.components,
        numeric_input=config.model.numeric_input,
        loss=config.training.loss,
    )


def apply_variant(config: DeepFMConfig, variant: Variant) -> DeepFMConfig:
    """`config` with the six study settings replaced by those of `variant`."""
    model = replace(
        config.model,
        embedding_dim=variant.embedding_dim,
        dropout=variant.dropout,
        components=variant.components,
        numeric_input=variant.numeric_input,
    )
    training = replace(config.training, learning_rate=variant.learning_rate, loss=variant.loss)
    return replace(config, model=model, training=training)


def with_parameter(variant: Variant, parameter: str, value: float) -> Variant:
    """A variant with one tuned parameter set to `value` (the dimension as an integer).

    Raises:
        ValueError: If `parameter` is not one of the three tuned parameters.
    """
    if parameter == LEARNING_RATE:
        return replace(variant, learning_rate=float(value))
    if parameter == EMBEDDING_DIM:
        if float(value) != int(value):
            raise ValueError(f"embedding_dim must be a whole number, got {value}")
        return replace(variant, embedding_dim=int(value))
    if parameter == DROPOUT:
        return replace(variant, dropout=float(value))
    raise ValueError(f"{parameter!r} is not a tuned parameter")


def parameter_value(variant: Variant, parameter: str) -> float:
    """The value of one tuned parameter of a variant."""
    values = {
        LEARNING_RATE: variant.learning_rate,
        EMBEDDING_DIM: float(variant.embedding_dim),
        DROPOUT: variant.dropout,
    }
    if parameter not in values:
        raise ValueError(f"{parameter!r} is not a tuned parameter")
    return values[parameter]


def variant_config_id(config: DeepFMConfig, variant: Variant) -> str:
    """The configuration id of `config` with `variant` applied."""
    return deepfm_config_id(apply_variant(config, variant))


# ---------------------------------------------------------------- scores from metric rows


@dataclass(frozen=True)
class ConfigurationScore:
    """A configuration's result with one seed, after the epoch rule.

    Attributes:
        config_id: The configuration.
        seed: The training seed.
        area: The area of its rows ("v_phase" or "v_phase_extended").
        epochs_trained: Epochs every fold was trained for.
        selected_epoch: e*, the epoch the rule selects.
        best_epoch: The smallest epoch with the highest pooled value.
        pooled: The pooled value at e*.
        fold_scores: Fold -> its value at e*.
        curve: Epoch -> pooled value (all epochs).
        train_loss: Epoch -> mean training loss, averaged over the folds (equal weights).
    """

    config_id: str
    seed: int
    area: str
    epochs_trained: int
    selected_epoch: int
    best_epoch: int
    pooled: float
    fold_scores: dict[str, float]
    curve: dict[int, float]
    train_loss: dict[int, float]

    @property
    def uses_last_epoch(self) -> bool:
        """True if the rule selects the last trained epoch (the curve may still be rising)."""
        return self.selected_epoch == self.epochs_trained

    def trial_scores(self) -> TrialScores:
        """The same result in the shape the selection rule takes."""
        return TrialScores(self.config_id, dict(self.fold_scores), self.pooled)


def pooled_value(values: Mapping[str, float], n_queries: Mapping[str, int]) -> float:
    """Query-weighted mean over the folds: sum_f n_f * v_f / sum_f n_f (math.fsum)."""
    total = sum(n_queries[fold] for fold in values)
    if total <= 0:
        raise ValueError("the folds have no queries")
    return math.fsum(n_queries[fold] * value for fold, value in values.items()) / total


def score_configuration(
    rows: Sequence[Mapping[str, str]],
    area: str,
    config_id: str,
    seed: int,
    folds: Sequence[str],
    metric: str,
    tolerance: float,
) -> ConfigurationScore:
    """Apply the epoch rule to the recorded rows of one configuration and seed.

    Args:
        rows: Rows of results/deepfm_metrics.csv (any configurations).
        area: "v_phase" or "v_phase_extended".
        config_id: The configuration.
        seed: The seed.
        folds: The folds that must all be there (V1 ... V4).
        metric: The selection metric column, "map@12".
        tolerance: Relative tolerance of the epoch rule.

    Raises:
        ValueError: If a fold is missing, a (fold, epoch) repeats, or the folds
            were trained for different numbers of epochs.
    """
    own = [
        row
        for row in rows
        if row["area"] == area and row["config_id"] == config_id and int(row["seed"]) == seed
    ]
    values: dict[int, dict[str, float]] = {}
    losses: dict[int, dict[str, float]] = {}
    n_queries: dict[str, int] = {}
    for row in own:
        epoch, fold = int(row["epoch"]), row["fold"]
        if fold in values.setdefault(epoch, {}):
            raise ValueError(
                f"{config_id} seed {seed}: fold {fold} epoch {epoch} is recorded twice"
            )
        values[epoch][fold] = float(row[metric])
        losses.setdefault(epoch, {})[fold] = float(row["train_loss"])
        n_queries[fold] = int(row["n_queries"])
    incomplete = [epoch for epoch, by_fold in values.items() if sorted(by_fold) != sorted(folds)]
    if not values or incomplete:
        raise ValueError(
            f"{config_id} seed {seed} ({area}): the folds {list(folds)} are not all recorded "
            f"for every epoch (epochs {sorted(values)}, incomplete {sorted(incomplete)})"
        )
    curve = {epoch: pooled_value(values[epoch], n_queries) for epoch in sorted(values)}
    choice = choose_num_trees(curve, tolerance)
    selected = choice.num_trees
    mean_loss = {
        epoch: math.fsum(losses[epoch].values()) / len(losses[epoch]) for epoch in sorted(losses)
    }
    return ConfigurationScore(
        config_id=config_id,
        seed=seed,
        area=area,
        epochs_trained=max(curve),
        selected_epoch=selected,
        best_epoch=choice.best_num_trees,
        pooled=curve[selected],
        fold_scores={fold: values[selected][fold] for fold in folds},
        curve=curve,
        train_loss=mean_loss,
    )


# ---------------------------------------------------------------- step 0


def seed_range(scores: Sequence[ConfigurationScore]) -> float:
    """(max - min) / min of the pooled scores of several seeds."""
    pooled = [score.pooled for score in scores]
    if len(pooled) < 2 or min(pooled) <= 0:
        raise ValueError("the range needs at least two positive pooled scores")
    return (max(pooled) - min(pooled)) / min(pooled)


class StudyStop(Exception):
    """A stop condition of the study is met: stop and report, do not work around it."""


def check_noise(scores: Sequence[ConfigurationScore], max_seed_range: float) -> float:
    """Step 0: the seed-to-seed range of the default configuration; stops if it is too large.

    Returns:
        The range.

    Raises:
        StudyStop: If the range is above `max_seed_range`.
    """
    observed = seed_range(scores)
    if observed > max_seed_range:
        raise StudyStop(
            f"step 0: the pooled score ranges by {observed:.4%} over the seeds, which is "
            f"above {max_seed_range:.2%}. The study stops before any tuning."
        )
    return observed


def check_above_baseline(score: ConfigurationScore, baseline: float) -> None:
    """Stop if the default configuration's pooled score is below the RRF baseline.

    Raises:
        StudyStop: If it is.
    """
    if score.pooled < baseline:
        raise StudyStop(
            f"the default configuration's pooled score {score.pooled:.6f} is below the RRF "
            f"baseline {baseline:.6f}. The study stops."
        )


# ---------------------------------------------------------------- tuning


@dataclass(frozen=True)
class StepOutcome:
    """What one tuning step decided.

    Attributes:
        step: The step (parameter and candidate values in priority order).
        start: The configuration the step started from (its first candidate).
        variants: Candidate value -> the configuration with that value.
        scores: Candidate value -> its score (seed of the selection).
        decision: The outcome of the selection rule.
        selected_value: The value that goes on to the next step.
        on_grid_edge: True if the selected value is the smallest or largest of the grid.
    """

    step: DeepFMStudyStep
    start: Variant
    variants: dict[float, Variant]
    scores: dict[float, ConfigurationScore]
    decision: StepDecision
    selected_value: float
    on_grid_edge: bool

    @property
    def selected(self) -> Variant:
        """The configuration that goes on to the next step."""
        return self.variants[self.selected_value]


def step_variants(step: DeepFMStudyStep, current: Variant) -> dict[float, Variant]:
    """The step's configurations: `current` with the tuned parameter set to each candidate.

    Raises:
        ValueError: If the step's first candidate is not the current value (the
            step would not start from the configuration the study has reached).
    """
    if parameter_value(current, step.parameter) != step.candidates[0]:
        raise ValueError(
            f"step {step.name}: its first candidate {step.candidates[0]} is not the current "
            f"value {parameter_value(current, step.parameter)}"
        )
    return {value: with_parameter(current, step.parameter, value) for value in step.candidates}


def decide_step(
    step: DeepFMStudyStep,
    current: Variant,
    scores: Mapping[float, ConfigurationScore],
    rule: SelectionRule,
) -> StepOutcome:
    """Apply the selection rule to one tuning step.

    The first candidate is the current configuration (the default of the step);
    the others are compared with it, in priority order.

    Args:
        step: The step.
        current: The configuration the step starts from.
        scores: Candidate value -> its score; every candidate must be there.
        rule: configs/selection_rule.yaml.
    """
    variants = step_variants(step, current)
    default_value, *other_values = step.candidates
    default = scores[default_value].trial_scores()
    candidates = [scores[value].trial_scores() for value in other_values]
    decision = choose_winner(default, candidates, rule)
    by_trial = {scores[value].config_id: value for value in step.candidates}
    selected_value = by_trial[decision.winner_trial_id]
    return StepOutcome(
        step=step,
        start=current,
        variants=variants,
        scores=dict(scores),
        decision=decision,
        selected_value=selected_value,
        on_grid_edge=selected_value in (min(step.candidates), max(step.candidates)),
    )


# ---------------------------------------------------------------- ablations


def ablation_variants(study: DeepFMStudyConfig, selected: Variant) -> dict[str, Variant]:
    """The ablations of the selected configuration: name -> configuration.

    One setting is changed at a time; the variant equal to the selected
    configuration itself (components "full") is left out.
    """
    variants: dict[str, Variant] = {}
    for components in study.ablation_components:
        if components != selected.components:
            variants[f"components={components}"] = replace(selected, components=components)
    for numeric_input in study.ablation_numeric_inputs:
        if numeric_input != selected.numeric_input:
            variants[f"numeric_input={numeric_input}"] = replace(
                selected, numeric_input=numeric_input
            )
    for loss in study.ablation_losses:
        if loss != selected.loss:
            variants[f"loss={loss}"] = replace(selected, loss=loss)
    return variants
