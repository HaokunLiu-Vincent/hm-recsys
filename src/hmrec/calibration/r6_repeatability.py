"""The R6 fine-tuning repeatability reproducer (M10 part 3; torch side).

Question it answers: do two fine-tunes of the same model on the same samples,
each in a FRESH process, end with the same weights, bit for bit? That is the
only criterion. Loss values are recorded and never used to choose anything.

What runs: the REAL fine-tuning function, hmrec.recall.twotower.train.fine_tune,
with the model class, layer sizes, batch size, optimizer settings, number of
epochs and seeds of configs/locks/twotower.lock.yaml, and with tables of the
real sizes (configs/m10_r6_repeatability.yaml), so every tensor has its real
shape.

What is synthetic: the data. Item vectors, category codes, customer profiles
and purchase rows are drawn from a seed. No competition data is read here and
nothing is derived from its content. The model starts from freshly seeded
weights (a base model is trained on real data, so it cannot be used here).

A worker process builds everything, fine-tunes once under a runtime setting
(hmrec.recall.twotower.fine_tune_runtime) and saves its weights; the caller
starts two workers one after the other and compares the two files.

This module is on the torch side; it must not import lightgbm.
"""

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from hmrec.calibration.recall_items import parameter_difference
from hmrec.calibration.records import IDENTITY_COLUMNS, now_text
from hmrec.config import (
    RepeatabilityConfig,
    TwoTowerConfig,
    load_base_config,
    load_repeatability_config,
    parse_twotower_config,
)
from hmrec.csv_records import append_row
from hmrec.git_state import git_state
from hmrec.logging_utils import get_logger
from hmrec.paths import PROJECT_ROOT, ProjectPaths
from hmrec.platform_fields import read_platform_fields
from hmrec.platform_report import read_machine_info
from hmrec.recall.locked import load_locked_recall
from hmrec.recall.twotower.data import (
    CategoryCodes,
    ItemFeatureArrays,
    TrainingArrays,
    items_to_device,
    profiles_to_device,
)
from hmrec.recall.twotower.fine_tune_runtime import (
    SettingNotApplicable,
    fine_tune_setting,
    thread_report,
)
from hmrec.recall.twotower.model import build_two_tower_model
from hmrec.recall.twotower.train import (
    DeviceTables,
    FineTuneArrays,
    FineTuneSchedule,
    fine_tune,
)

LOGGER = get_logger("calibration.r6_repeatability")

# The anchor day of the synthetic week: only a seed of the epoch shuffles (any value works).
SYNTHETIC_WEEK_DAY = 700
WEEK_DAYS = 7
ARTICLE_CODE_COLUMNS = (
    "product_code",
    "product_type_no",
    "product_group_name",
    "garment_group_no",
    "index_group_no",
    "section_no",
    "colour_group_code",
)
PROFILE_COLUMNS = ("age_bucket", "club_member_status", "fashion_news_frequency")
NOT_APPLICABLE = "not applicable"
FAILED = "failed"

REPEATABILITY_COLUMNS: tuple[str, ...] = (
    "setting",
    "train_rows",
    "steps",
    "outcome",
    "bit_identical",
    "max_abs_parameter_difference",
    "different_parameter_values",
    "parameter_values",
    "first_seconds",
    "second_seconds",
    "first_train_loss",
    "second_train_loss",
    "threads_during_fine_tune",
    "num_threads",
    "usable_cpus",
    "detail",
    *[name for name in IDENTITY_COLUMNS if name != "run"],
    "finished_at",
)


# ---------------------------------------------------------------- synthetic data


def _codes(rng: np.random.Generator, n_rows: int, values: tuple[int, ...]) -> np.ndarray:
    """Random category codes int32 [n_rows, n_columns]; column c holds 1 .. values[c]."""
    columns = [rng.integers(1, size + 1, size=n_rows, dtype=np.int32) for size in values]
    return np.stack(columns, axis=1)


def _category_codes(
    codes: np.ndarray, columns: tuple[str, ...], values: tuple[int, ...]
) -> CategoryCodes:
    """CategoryCodes whose vocabularies have the given sizes (the values are placeholders)."""
    vocabularies = {
        column: tuple(f"value_{index}" for index in range(size))
        for column, size in zip(columns, values, strict=True)
    }
    return CategoryCodes(codes=codes, columns=columns, vocabularies=vocabularies)


def _unit_rows(rng: np.random.Generator, n_rows: int, width: int) -> np.ndarray:
    """Random float32 [n_rows, width] with rows of length 1, plus ONE zero padding row."""
    vectors = rng.standard_normal((n_rows, width), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.concatenate([vectors, np.zeros((1, width), dtype=np.float32)])


@dataclass(frozen=True)
class SyntheticTables:
    """The tables a fine-tune reads besides its samples.

    Attributes:
        features: Item feature table with the padding row; image float32
            [n_articles + 1, image_dim], text float32 [n_articles + 1, text_dim].
        article_codes: The codes inside `features`, with vocabularies of the configured sizes.
        profiles: Profile codes int32 [n_customers, 3].
    """

    features: ItemFeatureArrays
    article_codes: CategoryCodes
    profiles: CategoryCodes


def synthetic_tables(config: RepeatabilityConfig) -> SyntheticTables:
    """Draw the item feature table and the customer profiles from the seed.

    Every article has an image and a text vector (masks True); the padding row
    has zero vectors, False masks and code 0, as in the real table.
    """
    sizes = config.table_sizes
    rng = np.random.default_rng([config.seed, 1])
    real_rows = np.ones(sizes.n_articles, dtype=bool)
    mask = np.concatenate([real_rows, np.zeros(1, dtype=bool)])
    codes = _codes(rng, sizes.n_articles, sizes.article_code_values)
    codes = np.concatenate([codes, np.zeros((1, codes.shape[1]), dtype=np.int32)])
    article_codes = _category_codes(codes, ARTICLE_CODE_COLUMNS, sizes.article_code_values)
    features = ItemFeatureArrays(
        image=_unit_rows(rng, sizes.n_articles, sizes.image_dim),
        image_mask=mask,
        text=_unit_rows(rng, sizes.n_articles, sizes.text_dim),
        text_mask=mask.copy(),
        codes=article_codes,
    )
    profile_codes = _codes(rng, sizes.n_customers, sizes.profile_values)
    profiles = _category_codes(profile_codes, PROFILE_COLUMNS, sizes.profile_values)
    return SyntheticTables(features, article_codes, profiles)


def article_probabilities(n_articles: int, offset: float, rng: np.random.Generator) -> np.ndarray:
    """Probability of each article: proportional to 1 / (k + offset) for a random order k.

    Returns:
        float64 [n_articles], sums to 1. A few articles are drawn very often,
        most rarely, as in a real week.
    """
    weights = 1.0 / (np.arange(n_articles, dtype=np.float64) + offset)
    probabilities = weights / weights.sum()
    return probabilities[rng.permutation(n_articles)]


def synthetic_samples(
    config: RepeatabilityConfig, n_samples: int, history_length: int, stream: int
) -> TrainingArrays:
    """Draw `n_samples` purchase rows in the shape of real training samples.

    Rows come in groups (one customer on one day, 1 .. max_group_rows rows);
    the rows of a group are consecutive, share their group_id and their
    history, and each has its own target article. A history holds 0 ..
    history_length articles in its first slots; empty slots hold the padding
    index n_articles with mask False.

    Args:
        config: The reproducer's settings.
        n_samples: Rows to draw (N).
        history_length: L of the locked model.
        stream: Makes the training set and the alarm set different draws.

    Returns:
        TrainingArrays with N rows: int32 ids, int16 day, history int32 [N, L],
        history_mask bool [N, L].
    """
    sizes = config.table_sizes
    rng = np.random.default_rng([config.seed, 2, stream])
    probabilities = article_probabilities(sizes.n_articles, config.popularity_offset, rng)
    # At least one row per group, so n_samples groups are always enough.
    group_rows = rng.integers(1, config.max_group_rows + 1, size=n_samples)
    row_group = np.repeat(np.arange(n_samples), group_rows)[:n_samples]
    n_groups = int(row_group[-1]) + 1
    group_customer = rng.integers(0, sizes.n_customers, size=n_groups, dtype=np.int32)
    group_day = SYNTHETIC_WEEK_DAY + rng.integers(0, WEEK_DAYS, size=n_groups)
    group_history_length = rng.integers(0, history_length + 1, size=n_groups)
    group_history = rng.choice(
        sizes.n_articles, size=(n_groups, history_length), p=probabilities
    ).astype(np.int32)
    group_mask = np.arange(history_length)[None, :] < group_history_length[:, None]
    group_history[~group_mask] = sizes.n_articles
    return TrainingArrays(
        customer_idx=group_customer[row_group],
        t_dat_day=group_day[row_group].astype(np.int16),
        article_idx=rng.choice(sizes.n_articles, size=n_samples, p=probabilities).astype(np.int32),
        group_id=row_group.astype(np.int32),
        history=group_history[row_group],
        history_mask=group_mask[row_group],
    )


def synthetic_fine_tune_arrays(
    config: RepeatabilityConfig, n_samples: int, history_length: int
) -> FineTuneArrays:
    """The fine-tune set (n_samples rows) and the alarm rows of one synthetic week.

    Half of the rows are counted as new and half as replay, as with the locked
    1 : 1 replay; the split is only a count here (the rows are drawn the same way).
    """
    train = synthetic_samples(config, n_samples, history_length, stream=1)
    alarm = synthetic_samples(config, config.alarm_samples, history_length, stream=2)
    n_new = (n_samples + 1) // 2
    return FineTuneArrays(train=train, alarm=alarm, n_new=n_new, n_replay=n_samples - n_new)


# ---------------------------------------------------------------- one worker


@dataclass(frozen=True)
class WorkerSummary:
    """What one worker process did.

    Attributes:
        setting: The runtime setting the fine-tune ran under.
        train_rows: Rows of the synthetic fine-tune set.
        steps: Optimizer steps.
        seconds: Wall time of the update steps.
        mean_train_loss: Mean loss of the last epoch.
        alarm_loss: Mean loss of the alarm rows after the update.
        threads: The thread counts in force during the fine-tune.
        platform: The platform fields of the worker process (table columns).
    """

    setting: str
    train_rows: int
    steps: int
    seconds: float
    mean_train_loss: float
    alarm_loss: float
    threads: str
    platform: dict[str, str]


def locked_twotower_config(paths: ProjectPaths) -> TwoTowerConfig:
    """The locked R6 settings (configs/locks/twotower.lock.yaml)."""
    return parse_twotower_config(load_locked_recall(paths).twotower.channel.config)


def run_worker(
    paths: ProjectPaths, setting: str, n_samples: int, state_file: Path
) -> WorkerSummary:
    """Fine-tune once on synthetic data under `setting`; save the weights and a summary.

    Args:
        paths: Project paths (configs only; no data file is read).
        setting: One of hmrec.recall.twotower.fine_tune_runtime.SETTINGS.
        n_samples: Rows of the fine-tune set; must be one of the configured scales.
        state_file: Where the weights go (torch.save of name -> tensor); the
            summary goes next to it with the suffix ".json".

    Raises:
        ValueError: If n_samples is not a configured scale.
        SettingNotApplicable: If the setting cannot be applied in this process.
    """
    config = load_repeatability_config(paths.r6_repeatability_config_file)
    if n_samples not in config.scales:
        raise ValueError(f"--rows must be one of {list(config.scales)}, got {n_samples}")
    base = load_base_config(paths.base_config_file)
    # As every stage-1 script that trains with torch does.
    torch.set_num_threads(base.num_threads)
    twotower = locked_twotower_config(paths)
    tables = synthetic_tables(config)
    sets = synthetic_fine_tune_arrays(config, n_samples, twotower.history_length)
    sizes = config.table_sizes
    model = build_two_tower_model(
        sizes.n_articles,
        sizes.image_dim,
        sizes.text_dim,
        tables.article_codes,
        tables.profiles,
        twotower.history_length,
        twotower.model,
        twotower.training.seed,
    )
    device = torch.device(twotower.training.device)
    device_tables = DeviceTables(
        profiles_to_device(tables.profiles, device), items_to_device(tables.features, device)
    )
    schedule = FineTuneSchedule(twotower.e_update, twotower.chain_seed, SYNTHETIC_WEEK_DAY)
    with fine_tune_setting(setting):
        threads = thread_report()
        result = fine_tune(model, sets, device_tables, twotower.training, schedule, base.memory)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    state_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "metadata": {}}, state_file)
    summary = WorkerSummary(
        setting=setting,
        train_rows=n_samples,
        steps=result.steps,
        seconds=result.seconds,
        mean_train_loss=result.mean_train_loss,
        alarm_loss=result.alarm_loss,
        threads=threads,
        # The code that runs is the code of this package (in an image: /app, whose tree hash
        # is verified), whatever folder `paths` takes its configs from.
        platform=read_platform_fields(PROJECT_ROOT).columns(),
    )
    state_file.with_suffix(".json").write_text(json.dumps(asdict(summary)), encoding="utf-8")
    LOGGER.info(
        "worker: %s, %d rows, %d steps, %.1f s, train loss %r | %s",
        setting,
        n_samples,
        result.steps,
        result.seconds,
        result.mean_train_loss,
        threads,
    )
    return summary


# ---------------------------------------------------------------- two workers, compared

WORKER_NOT_APPLICABLE_EXIT = 3


def start_worker(script: Path, setting: str, n_samples: int, state_file: Path) -> int:
    """Run one worker in a fresh Python process; returns its exit code."""
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--setting",
        setting,
        "--rows",
        str(n_samples),
        "--state-file",
        str(state_file),
    ]
    return subprocess.run(command, check=False).returncode


def read_summary(state_file: Path) -> WorkerSummary:
    """The summary a worker wrote next to its weights."""
    return WorkerSummary(**json.loads(state_file.with_suffix(".json").read_text(encoding="utf-8")))


def check_row(
    paths: ProjectPaths, setting: str, n_samples: int, outcome: str, values: dict[str, str]
) -> dict[str, str]:
    """One row of results/m10_calibration/r6_repeatability.csv."""
    commit, dirty = git_state(paths.project_root, paths.results_root)
    machine = read_machine_info()
    usable = machine.cpu_quota if machine.cpu_quota is not None else machine.usable_cpus
    row = {name: "" for name in REPEATABILITY_COLUMNS}
    row.update(
        {
            "setting": setting,
            "train_rows": str(n_samples),
            "outcome": outcome,
            "num_threads": str(load_base_config(paths.base_config_file).num_threads),
            "usable_cpus": repr(float(usable)),
            "git_commit": commit,
            "git_dirty": str(dirty),
            "finished_at": now_text(),
        }
    )
    row.update(values)
    return row


def run_check(
    paths: ProjectPaths, script: Path, setting: str, n_samples: int, record: bool
) -> dict[str, str]:
    """Fine-tune twice in fresh processes under one setting and compare the weights.

    Args:
        paths: Project paths.
        script: scripts/m10_r6_repeatability.py (started again as the worker).
        setting: The runtime setting of both workers.
        n_samples: Rows of the synthetic fine-tune set.
        record: Append the row to results/m10_calibration/r6_repeatability.csv.

    Returns:
        The row: outcome "repeatable", "not repeatable", "not applicable" (the
        setting cannot be applied on this platform) or "failed" (a worker stopped
        with an error, e.g. an operation without a deterministic algorithm).
    """
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="r6_repeatability_") as folder:
        files = [Path(folder) / "first.pt", Path(folder) / "second.pt"]
        codes = [start_worker(script, setting, n_samples, file) for file in files[:1]]
        if codes[0] == 0:
            codes.append(start_worker(script, setting, n_samples, files[1]))
        if any(code != 0 for code in codes):
            outcome = NOT_APPLICABLE if WORKER_NOT_APPLICABLE_EXIT in codes else FAILED
            row = check_row(
                paths, setting, n_samples, outcome, {"detail": f"worker exit codes {codes}"}
            )
        else:
            first, second = read_summary(files[0]), read_summary(files[1])
            difference = parameter_difference(files[0], files[1])
            identical = difference.n_different_values == 0
            values = {
                "steps": str(first.steps),
                "bit_identical": str(identical),
                "max_abs_parameter_difference": repr(difference.max_abs_difference),
                "different_parameter_values": str(difference.n_different_values),
                "parameter_values": str(difference.n_values),
                "first_seconds": repr(first.seconds),
                "second_seconds": repr(second.seconds),
                "first_train_loss": repr(first.mean_train_loss),
                "second_train_loss": repr(second.mean_train_loss),
                "threads_during_fine_tune": first.threads,
                "detail": f"two fresh processes; total {time.perf_counter() - started:.1f} s",
                **first.platform,
            }
            outcome = "repeatable" if identical else "not repeatable"
            row = check_row(paths, setting, n_samples, outcome, values)
    if record:
        append_row(paths.r6_repeatability_file, REPEATABILITY_COLUMNS, row)
    return row


def summary_line(row: dict[str, str]) -> str:
    """One line for a log or a CI notice."""
    return (
        f"r6 fine-tune repeatability | {row['os']}-{row['architecture']} | setting "
        f"{row['setting']} | {row['train_rows']} rows, {row['steps']} steps | "
        f"{row['outcome']} | max parameter difference {row['max_abs_parameter_difference']} | "
        f"{row['first_seconds'][:6]} s and {row['second_seconds'][:6]} s | "
        f"{row['threads_during_fine_tune']}"
    )


def worker_exit_code(error: BaseException) -> int:
    """The exit code a worker ends with after `error` (3 = the setting is not applicable)."""
    return WORKER_NOT_APPLICABLE_EXIT if isinstance(error, SettingNotApplicable) else 1
