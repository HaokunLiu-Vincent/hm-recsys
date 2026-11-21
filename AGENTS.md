# AGENTS.md

This file binds every coding agent working in this repository (Claude Code and others). **The rules here take precedence over your default habits.** Where it conflicts with `docs/proposal.md`, `docs/proposal.md` prevails, and you must point out the conflict in your reply.

---

## 1. What the project is

Personalized fashion recommendation for H&M (Kaggle H&M dataset). Goal: predict the 12 articles each customer will buy in the following week; the main metric is MAP@12.

System structure:

```
6 recall channels (R1 Popularity, R2 Repurchase, R3 Sibling, R4 ItemCF, R5 WALS, R6 Two-Tower)
  → union + dedup (one row per (query, article))
  → wide-table features
  → LightGBM LambdaMART
  → Top-12
```

**The only design reference is `docs/proposal.md`.** Read the relevant sections before writing code. Do not change the design from memory or because of "common practice".

## 2. Before starting any task

1. Read the sections of `docs/proposal.md` that are relevant to the task.
2. Read the modules you are going to modify, and their tests.
3. If more than 1 file will change, **give a short plan first** (which files, why, how it will be tested), then start.
4. Where the proposal is not clear, **ask first; do not guess**.

---

## 3. Hard rules (must never be broken)

### 3.1 Preventing temporal leakage

- Any computation for an anchor `a` (recall, features, model training) **may use only data with `t_dat < a`**. The anchor day belongs to the label week.
- Transactions are filtered by date **only through `src/hmrec/data/asof.py`**, for both the history (`history_window`, `[anchor − N, anchor)`) and the label week (`label_week_transactions`, `[anchor, label_end]`). Do not write your own date filter anywhere else.
- **The only exception**: the independent check scripts under `scripts/checks/` derive the date windows themselves with `datetime`, without asof / queries / day_index, so that they do not repeat a bug of the code they check (see `scripts/checks/README.md`). This code must not be copied into `src/`.
- `label_week_transactions` reads the answers and **may be called only by `src/hmrec/data/queries.py`**. `tests/unit/test_asof.py` scans `src/hmrec/` and `scripts/`; the test fails if any other file references it.
- Every as-of function must check that the dates of the data it returns are inside the window (the latest date of a history is before the anchor) and raise an error otherwise (use `raise`, not `assert`, which `python -O` removes).
- Tables under `data/processed/` **must be read through `hmrec.data.metadata.scan_processed_parquet`**. It first checks `day_zero`, the age buckets and the hash of the preprocessing config against `data/processed/metadata.json`, and raises an error on any mismatch. Do not call `pl.scan_parquet` / `pl.read_parquet` directly on processed files.
- Dates in the processed data are the int16 column `t_dat_day` (days since `day_zero`). Conversion between dates and day numbers **must call `src/hmrec/data/day_index.py`**; do not write formulas such as `(d - day_zero).days` yourself.
- **Recall and feature code never reads labels.** Labels are read only in `src/hmrec/eval/` and `src/hmrec/ranker/sampling.py`.
- The candidates and features of a fold are frozen once generated. When they are later used for training, **regenerating them with a newer model is not allowed**.

### 3.2 Phase rules

| Phase | Allowed | Not allowed |
|---|---|---|
| D1–D4 | Ablation and HPO of recall | Training the ranker on data of the D weeks |
| F1–F4 | Generating candidates with the locked configuration | Any tuning |
| V1–V4 | HPO of the ranker | Changing the recall configuration |
| T1–T4 | Final evaluation with all locked configurations | **Any tuning, any selection**; no T fold may appear in the HPO code path |

**T1–T4 has been used once** (the formal run of M8, 2026-11-18, results in `results/t_phase*.csv`). From now on:

- **M8 is not re-run.** Do not retrain the models of the T weeks, and do not recompute the metrics of the T weeks to replace the existing records. `results/t_phase*.csv` and `results/m8_*.csv` are the original records and are not modified.
- **Any later change (candidates, features, ranker, any parameter) may only be compared and selected on V1–V4.** Results of the T weeks may not be used to pick, to tune, or to decide whether to adopt a change.
- Descriptive diagnostics of the T-week results are allowed (describe only; select nothing, modify nothing); reading the labels of the T weeks again must be recorded, and needs the user's consent first.

**The descriptive diagnostics after M8 have a second entry point for reading T labels** (2026-11-19, with the user's consent; details in proposal §10.2):

- The entry point is `load_diagnostic_evaluator` in `src/hmrec/eval/diagnostic_labels.py`. Reading the labels of a T week requires a `DiagnosticAccess`, which is issued only by `src/hmrec/diagnostics/access.py`. `eval/final_eval.py` (the entry point of the M8 formal run) is not touched.
- Conditions for issuing it: `results/t_phase_goals.csv` already holds the three verdicts G1–G3; the working tree is clean; the content of the M8 files and of the files under `configs/` that existed at the commit of the formal run has not changed (new files are allowed); and a row is appended to `results/t_diag_label_access.csv` **first** (time, commit, purpose, script), one row per run.
- **Only the diagnostics code may use this entry point**: `src/hmrec/diagnostics/` and `scripts/m8_diagnostics.py`. No other code (any module or script of M0–M8) may import `hmrec.diagnostics` or `hmrec.eval.diagnostic_labels`. The diagnostics code may not import code that trains, samples or scores with a model, nor the M8 trial, the gate or `eval/final_eval.py`; it writes only `results/t_diag_*.csv`. `tests/unit/test_diag_code_rules.py` checks these; **a newly written diagnostics module must be added to its file list**.
- The results of the diagnostics (`results/t_diag_*.csv`) are for description only; **they may not be used to select, to tune, or to decide whether to adopt a change**.
- **Once the diagnostics code is committed, `scripts/checks/check_ranker_eval.py --area t_phase` can no longer run at the current HEAD**: to read T labels it has to pass the M8 gate, and the gate requires the code to equal the commit of the formal run. To re-run it, first check out tag `m8-run-v1` (`git checkout m8-run-v1`), run it, then switch back. This check was already run once after the formal run; the result is in `docs/decisions.md`.

**The M9 evaluation on T1–T4 has a third entry point for reading T labels** (2026-11-20, with the user's consent; details in proposal §16.12):

- It evaluates **one** configuration, the DeepFM configuration chosen on V (`configs/m9_t_eval.yaml`), once; the result is reported only. **T1–T4 is the final test set, not a second validation set: no other arm is run on T, and nothing may be changed after the T result is seen.**
- The entry point is `load_t_eval_evaluator` in `src/hmrec/eval/t_eval_labels.py`. Reading the labels of a T week requires a `TEvalPermit`, which is issued only by `src/hmrec/t_eval/gate.py`. The M8 entry point and the diagnostics entry point are not touched.
- Conditions for issuing it: the latest row of `results/m9_t_rehearsal_check.csv` says `passed` and was written **at the commit of HEAD** from a clean working tree; the working tree is clean; and a row was appended to `results/m9_t_label_access.csv` **first** (time, commit, purpose, script) by the driver. A formal run that was started is continued only at the same commit.
- **Only the code of this evaluation may use this entry point**: `src/hmrec/t_eval/` and `scripts/m9_t_*.py`; no other code may import `hmrec.t_eval` or `hmrec.eval.t_eval_labels`. This code may not import the M8 code, the diagnostics, their label entry points or LightGBM. `tests/unit/test_t_eval_code_rules.py` checks these; **a newly written module of the evaluation must be added to its file lists**.

### 3.3 Things that must not be changed

- The dates in `configs/fold_manifest.yaml`.
- Any file under `configs/locks/`. Files may only be added, never modified or deleted.
- `configs/selection_rule.yaml` (the selection rule was committed before any score was seen).
- `configs/m8_final.yaml` (the target weeks of M8, the rehearsal week and the criteria of G1–G3, committed before any label of a T week was read).
- After the M8 formal run has started: the existing rows of `results/t_phase*.csv` and `results/m8_*.csv`, and the logic of the M8 code (`src/hmrec/final/`, `eval/final_eval.py`, `ranker/final_window.py`, `scripts/m8_final_eval.py`, `scripts/m8_step.py`; `scripts/m8_diagnostics.py` is the descriptive diagnostics after M8 and is not part of the M8 code). The formal run is done once; a bug that is found is only recorded, not fixed; stop and let the user decide.
- The history of `docs/decisions.md`. Entries may only be appended.

When any of the above needs to change, stop, explain why, and wait for the user's confirmation.

**Rules of M8 (the final evaluation on T1–T4)** (details in proposal §10.1):

- The labels of the T weeks may be read only through `src/hmrec/eval/final_eval.py`, and only with a `FormalRunPermit` issued by `src/hmrec/final/gate.py`. The protection against T folds in the entry points of M6 and M7 stays unchanged. Do not construct a permit anywhere else, and do not bypass it by opening the label files of the T weeks directly. (The descriptive diagnostics after M8 have another entry point, see §3.2.)
- A rehearsal must come before the formal run (`scripts/m8_final_eval.py --stage rehearsal`, target week V4, reads no T label), and its result must be bit-identical to the M7 records. If the rehearsal has not passed, if the code or configuration changed after the rehearsal, or if the working tree is not clean, the code refuses the formal run. **Do not modify or delete the rehearsal records in order to let the formal run start.**
- The formal run (`--stage formal`) is done once. After an interruption it can be resumed with the same command; it cannot be resumed once the code or configuration has changed.
- What "no T fold may appear in the HPO code path" means in M8: no T fold may appear in the modules that tune and select (`hpo/v_phase.py`, `hpo/v_lock.py`, `hpo/speed_test.py`, `hpo/runner.py`, `hpo/d_phase.py`), and M8 may not import them either. The steps in `hpo/v_trial.py` that do not depend on which week is predicted (training a model, scoring, saving) and the reading and writing of result tables in `hpo/v_records.py` are shared by M7 and M8; they do no fold check and make no selection, and the fold checks are in the respective entry points.

**After M5, the configuration of the six recall channels may be obtained only through `src/hmrec/recall/locked.py`** (it reads only `configs/locks/`; the lock files are parsed in `src/hmrec/locks.py`). Code of M6 and later may no longer read the defaults in `configs/recall/`, nor obtain a configuration by replaying the selection over `results/d_phase.csv`. The model files of R6 must be loaded through this entry point: it checks the sha256 before loading the weights and raises an error on a mismatch. `tests/unit/test_locked.py` scans the scripts of M6–M8 and `fusion/`, `features/`, `ranker/`.

**After M7, the configuration of the ranker may be obtained only through `load_locked_ranker` in `src/hmrec/ranker/locked.py`** (it reads `configs/locks/ranker.lock.yaml`; `configs/ranker.yaml` supplies only two values that affect only memory and logging). The M8 code may no longer read the tuning defaults and fixed parameters of `configs/ranker.yaml`, nor `configs/hpo/v_phase.yaml`, nor obtain a configuration by replaying the selection over `results/v_phase*.csv` (`src/hmrec/hpo/v_lock.py` is only a check used by tests; M8 does not call it). Before returning a configuration, this entry point checks that: the `config_id` computed from the settings in the lock file equals the recorded one; the sha256 of the referenced features / channels / twotower lock files match; and the LightGBM version, the libomp version and `num_threads` of the running process equal the recorded ones; on any mismatch it raises an error. `tests/unit/test_final_code_rules.py` scans the M8 code; **a newly written M8 module must be added to its `M8_FILES`**.

### 3.4 Other prohibitions

- Do not delete files under `data/`, `artifacts/` or `results/`.
- Do not add dependencies (modify `pyproject.toml`) without the user's consent.
- Do not run heavy tasks on the full data (full recall, Two-Tower training, ranker training) without the user's consent. Development and tests use only the small data in `tests/fixtures/`.
- **After writing or modifying code, you must stop and wait for the user to commit; only when the working tree is clean may anything run on the full data.** Stop even if this run has already been authorised. Reason: a result must correspond to one definite commit (the `git_commit` in the run record, with `git_dirty=False`); a result produced by uncommitted code cannot be reproduced, and it cannot be checked afterwards which code produced it.
- Do not leave `print` statements used for debugging in the code.
- Do not change the expected value of a test to make the test pass, unless you can show that the original expected value was wrong.
- After modifying `.gitignore`, you must run two checks and report the results in your reply:
  1. Use `git check-ignore` to confirm that no file under `src/` or `tests/` is ignored. The following command must produce **no output at all**:
     `find src tests -type f -not -path '*/__pycache__/*' | git check-ignore --stdin`
  2. Confirm that the number of files ruff checks has not gone down: compare the file count printed by `uv run ruff format --check src tests scripts` before and after the change.

  Reason: ruff skips files listed in `.gitignore`. Once, the line `data/` also ignored `src/hmrec/data/`, so that package neither went into git nor was checked by ruff, while `make lint` still reported success.

---

## 4. Runtime environment constraints

| Resource | Limit |
|---|---|
| Memory | **The project's memory budget is 12 GB, shared by CPU and GPU** (a budget the project sets for itself; the machine has 32 GB of physical memory). The limit on the true peak of a single process is **12 GB**, equal to the budget. The limit is an alert threshold and affects no computed result; chunk sizes are not relaxed with it |
| CPU | 6 threads: `num_threads=6`, `n_jobs=6`, `OMP_NUM_THREADS=6` |
| GPU | Apple M1 Max, PyTorch **MPS** (`device="mps"`). No CUDA |
| Python | **3.12** (`.python-version`); the environment is managed by **uv** (`.venv` + `uv.lock`) |

When writing code:

- **IDs are always int32**, and floating-point numbers are always **float32** (MPS does not support float64). float32 applies to large arrays such as features, model inputs and embeddings.
  - **Exception: metric values and aggregated results use float64** (`src/hmrec/eval/metrics.py`). Reason: the data are small, computed on CPU only, and the "strictly better" comparison of §4.2 is sensitive to rounding.
- Data that may exceed 1 GB **must be processed in chunks**, with chunk sizes read from `configs/base.yaml`.
- Prefer the lazy API of Polars (`scan_parquet`), and read only the columns that are needed.
- Write large intermediate results to disk (Parquet); do not keep them all in memory.
- Record the memory peak at the start and at the end of a long task (use `src/hmrec/memory.py`).
- Do not use `faiss-gpu` or any library that depends on CUDA. `implicit` uses the CPU version, and so does LightGBM.

---

## 5. Directories and layering

The full directory tree is in `docs/proposal.md` §14. The key rules:

```
scripts/  →  src/hmrec/{recall, fusion, features, ranker, hpo, final, diagnostics, deepfm, t_eval, pipeline}  →  src/hmrec/{data, eval, vectors, locks, csv_records, run_records, feature_files, git_state}  →  config / paths
```

- An upper layer may import a lower layer; **a lower layer may not import an upper layer**.
- `recall/`, `features/` and `ranker/` **do not import one another**; they hand over through files in `artifacts/`.
- `fusion/` (union + dedup, the expressions of `n_channels` / `best_rank` / `rrf_score`, the ordering of the RRF baseline) is outside the restriction of the previous rule: **both `features/` and `ranker/` may import `fusion/`**, and `fusion/` imports none of the three. `ranker/` uses it to recompute the three aggregate features when R6 is removed (`ranker/ablation.py`), and `hpo/` uses it to compute the RRF baseline, so the wide table, the ablation and the baseline use the same set of expressions.
- `src/hmrec/vectors.py` (user / item vectors, batched dot products, saving / loading) belongs to the bottom layer; both `recall/` and `features/` may import it (see proposal §14.1). The recall scores of R5 / R6 and the `wals_dot` / `tt_dot` of §8.3 are computed only through it.
- `src/hmrec/locks.py` (reading and validating the lock files in `configs/locks/`, checking the sha256 of files) also belongs to the bottom layer, on the same level as `vectors.py`, and depends only on `config.py` and the standard library; `recall/`, `features/` and `ranker/` may all import it. **Building** the ranker configuration from the lock file is in `src/hmrec/ranker/locked.py` (see §3.3); **building** the channels from the lock files is in `src/hmrec/recall/locked.py` (see §3.3).
- `src/hmrec/csv_records.py` (reading and writing the append-only result tables `results/*.csv` with a fixed header) and `src/hmrec/run_records.py` (run records with one row per process: run time, sampled peak, true peak) also belong to the bottom layer and depend only on the standard library and `memory.py`; any upper-layer module and `eval/` may import them.
- `src/hmrec/feature_files.py` (the record `features.json` of each fold's feature files, and the verification function) also belongs to the bottom layer and depends only on the standard library, polars and `locks.py`: `features/` writes the record, and `ranker/` calls `verify_feature_files` to verify it before reading features; the two do not import each other.
- `src/hmrec/git_state.py` (reading the HEAD commit and whether the working tree has uncommitted changes) also belongs to the bottom layer and depends only on the standard library and `logging_utils.py`. `hpo/runner.py` still re-exports these two functions, so the scripts of M2–M6 need no change.
- `src/hmrec/final/` (the final evaluation of M8: gate, trial, rehearsal comparison, recall monitoring, the G1–G3 verdict, driver) belongs to the upper layer, on the same level as `hpo/`. It may import `eval/`, `ranker/`, `fusion/`, the bottom-layer modules, and `hpo/v_records.py` and `hpo/v_trial.py` (see the M8 rules in §3.3); it may not import `recall/`, torch, or the `hpo` modules that make selections. The code of M6 and M7 may not import `final/`, `eval/final_eval.py` or `ranker/final_window.py`. `tests/unit/test_final_code_rules.py` checks these; **a newly written M8 module must be added to its `M8_FILES`**.
- `src/hmrec/diagnostics/` (the descriptive diagnostics after M8: recording and checking the read of T labels, computing the four descriptive tables, driver) belongs to the upper layer, on the same level as `final/` and `hpo/`. It may import `eval/`, `fusion/`, `data/`, the bottom-layer modules, `hpo/v_records.py` (reading result tables), `hpo/selection.py` (only the tolerance constant), `ranker/locked.py`, `ranker/runtime.py`, `ranker/ablation.py` (only to get the id of the locked configuration, the number of trees, the Top-K and the column names of the wide table) and `final/goals.py` (only the column names of the verdict table). It may not import code that trains, samples or scores, the other modules of `final/`, `eval/final_eval.py`, `recall/`, `features/` or torch; no other code may import it (see §3.2).
- **The M7 code may not import torch** (neither may the M8 code): `ranker/`, `hpo/v_records.py`, `hpo/v_trial.py`, `hpo/v_phase.py`, `hpo/speed_test.py`, `eval/ranker_eval.py` and `scripts/m7_*.py` may not import torch, nor any module that loads torch (`recall/`, `hpo/runner.py`, `hpo/d_phase.py` and so on). Reason: torch and LightGBM each load their own copy of the OpenMP runtime (libomp), and two copies in one process exit immediately (`OMP: Error #15`), hang or segfault. `tests/unit/test_m7_code_rules.py` checks this rule.
- `src/hmrec/deepfm/` (stage 2, M9: DeepFM as an alternative ranker, proposal §16) belongs to the upper layer and is on the **torch side**. It may import `eval/`, `ranker/window.py`, `ranker/sampling.py`, `ranker/ablation.py` (none of which loads LightGBM), `hpo/selection.py` and the bottom-layer modules. **It may not import lightgbm or any module that does** (`ranker/dataset.py`, `ranker/train.py`, `ranker/predict.py`, `ranker/locked.py`, `hpo/v_*.py`, `final/` and so on), and no code of M0–M8 imports it. It works with V folds only: its entry points use the M7 fold guard and the M7 evaluator, so a T fold is refused. `tests/unit/test_deepfm_code_rules.py` checks these; **a newly written DeepFM module must be added to its `MODULES`**. Its test files start with `test_deepfm_` and run on the torch side. It may also import `hpo/lgb_arm_records.py` (the result tables of the LightGBM arms; it loads neither LightGBM nor torch). **From 2026-11-20 a DeepFM run of T epochs is evaluated once, after its last epoch** (cosine schedule over the whole run, proposal §16.8); the encoded inputs of a target week are cached under `artifacts/deepfm/cache/` and verified by sha256 before reuse (`deepfm/cache.py`). **The input-representation extension of 2026-11-20** (proposal §16.10: 255 bins and the standardised input, `deepfm/extension.py`, `configs/hpo/deepfm_extension.yaml`) trains and reports only: it tunes nothing and selects nothing. **When scoring, the scoring chunk (`scoring.rows_per_chunk`) is not a memory knob**: the last bits of a matrix product depend on the number of rows in the call, so the MLP and the first-order term always run once per chunk; only the field embeddings are built in blocks (`scoring.rows_per_block`, bit-identical for every block size). `deepfm/rescore.py` checks a finished trial against a new scoring, bit for bit.
- `src/hmrec/t_eval/` (stage 2, M9 step 4: the single evaluation on T1–T4 of the DeepFM configuration chosen on V, proposal §16.12) belongs to the upper layer and is on the **torch side** (`t_eval/trial.py`; `plan`, `gate`, `rehearsal`, `run` and `compare` load neither torch nor LightGBM). It may import `deepfm/`, `eval/t_eval_labels.py`, `ranker/window.py`, `hpo/selection.py` (only the tolerance constant) and the bottom-layer modules. It trains through `deepfm/trial.py::run_trial_through` with its own door (training folds, the Dataset record of M8 that names the sample files, the label loader); the V-phase entry point `run_trial` still uses the M7 fold guard and refuses every T fold. It may not import lightgbm or a module that loads it, `final/`, `diagnostics/`, `eval/final_eval.py` or `eval/diagnostic_labels.py`, and no other code imports it (see §3.2). It writes only `results/m9_t_*.csv` and files under `artifacts/deepfm/m9_t*/`. Its test files start with `test_t_eval_` and run on the torch side.
- `src/hmrec/hpo/lgb_arms.py`, `hpo/lgb_arms_run.py` and `hpo/lgb_arm_records.py` (stage 2, M9: the two LightGBM arms of the 2×2 comparison, proposal §16.8) belong to the upper layer and are on the **LightGBM side**. `lgb_arms.py` gets the Ranker configuration **only through `ranker/locked.py`**, changes nothing but the objective (arm `binary`) and the `seed`, reuses the M7 binary Datasets, the M7 fold guard and the M7 evaluator (**a T fold is refused**), and writes only `results/m9_lgb_*.csv` and files under `m9_lgb/`; it never writes `results/v_phase*.csv`. It may import `eval/`, `ranker/`, `hpo/v_trial.py`, `hpo/selection.py` and the bottom-layer modules; it may not import torch, `recall/`, `deepfm/`, `final/` or `diagnostics/`. `tests/unit/test_lgb_arms.py` checks these; its test files start with `test_lgb_arm` and run on the LightGBM side.
- `src/hmrec/pipeline.py` (running the steps of one fold one after another, each in its own process, and writing a summary per fold) belongs to the upper layer, on the same level as `recall/` and `hpo/`: it may import `eval/` and the bottom-layer modules, but `recall/`, `features/` and `ranker/` may not import it.
- `scripts/` does only three things: parse command-line arguments, call functions in `src/`, write results. **No business logic in scripts.**
- `notebooks/` is only for exploration and plotting; `src/` may not depend on any code in a notebook.
- All file paths come from `src/hmrec/paths.py`; do not hand-write path strings in the code.
- All numeric parameters are read from `configs/`; do not hard-code them.

---

## 6. Code conventions

The reader of this code is a **junior MLE**. Readability comes before brevity and cleverness.

### 6.1 Basic requirements

- Python 3.12, with type annotations everywhere.
- Formatted with `ruff format`; `ruff check` reports no errors; `pyright` (standard mode) reports no errors.
- Do not hide type errors with `# type: ignore` / `# pyright: ignore`; when ignoring is really necessary, state the reason on the same line.
- Each function does one thing and is normally no more than 50 lines. Split it if it is longer.
- Names are complete and plain: `item_purchase_count_7d`, not `ipc7`; `anchor_date`, not `ad`.
- Do not use hard-to-read constructs (multi-level nested comprehensions, metaprogramming, show-off method chaining). Write step by step, and give intermediate variables meaningful names.
- Separate computation from IO: a computation function receives and returns DataFrames or arrays and does not read or write files inside.

### 6.2 Docstring

Every public function and class must have a docstring (Google style, English, simple words). It must state clearly:

- what the function does;
- the **grain** of the input and output tables (what one row stands for) and the key columns;
- the **time window** used (open or closed interval);
- the **shape** and dtype of arrays.

Example:

```python
def compute_repurchase_scores(
    transactions: pl.DataFrame,
    anchor_date: date,
    lookback_days: int,
    decay_tau_days: float,
) -> pl.DataFrame:
    """Score articles that each customer bought before, for R2 Repurchase.

    Only purchases in the window [anchor_date - lookback_days, anchor_date)
    are used. The anchor date itself is NOT included, because it belongs to
    the label week.

    Each purchase line adds exp(-age_days / decay_tau_days) to the score,
    where age_days = (anchor_date - t_dat).days. So yesterday has age 1.
    Duplicate purchase lines are counted separately on purpose: buying the
    same item twice is a stronger repurchase signal.

    Args:
        transactions: One row per purchase line.
            Columns: customer_idx (int32), article_idx (int32), t_dat (date).
        anchor_date: First day of the week we predict.
        lookback_days: Length of the history window in days.
        decay_tau_days: Time constant of the exponential decay (not half-life).

    Returns:
        One row per (customer_idx, article_idx).
        Columns: customer_idx (int32), article_idx (int32), score (float32).
    """
```

### 6.3 Comments

- **Write "why", not only "what".** For example:

```python
# Bad:  filter dates
# Good: Keep only rows before the anchor. Rows on the anchor date belong to
#       the label week, so including them would leak the answer.
history = transactions.filter(pl.col("t_dat") < anchor_date)
```

- A comment is **mandatory** in these cases: time-window boundaries, dtype choices, the reason for chunking, the grain of a deduplication, any code that "looks redundant but has to be this way".
- Formulas are written out in mathematical form in a comment, with the section of `docs/proposal.md` they correspond to, for example `# See proposal §5 R4: s(i->j) = C_w / (n_w(i)^alpha * n_w(j)^(1-alpha))`.

### 6.4 Configuration, logging, randomness

- Config files are read into typed dataclasses (`src/hmrec/config.py`); do not pass dicts around the code.
- Use `logging` (`src/hmrec/logging_utils.py`). A long task records: start, progress, run time, memory peak, output file paths.
- Every place that uses random numbers must take a `seed` parameter, fixed in the config (NumPy, PyTorch, LightGBM, sampling).

### 6.5 Error handling

- When an input is not what is expected, **raise an error as early as possible**; the message must say what is wrong and what was expected. For example: `raise ValueError(f"history contains t_dat={max_date}, which is not before anchor {anchor_date}")`.
- Do not swallow errors with `try/except`.
- Before a recall output is written to disk, use the schema validation function in `src/hmrec/recall/base.py` to check the column names, the dtypes, that `article_idx` does not repeat within a query, and that `rank` starts at 1 and is consecutive.

---

## 7. Testing requirements

- Every `src/` module must have a corresponding test file under `tests/unit/`.
- Test data go in `tests/fixtures/` and are **small enough to compute by hand** (a few dozen rows). The test states how the expected values were computed.
- The following tests are **mandatory**:
  1. **Time cut-off test**: add "poison" data on the anchor day and after it; the result must be exactly unchanged.
  2. **Metric tests**: verify Recall@K, AP@12 and NDCG@K with small hand-computed examples, including edge cases such as "fewer than K outputs" and "empty label set".
  3. **Recall output schema test**: deduplication, ordering, tie-break (`score DESC, article_idx ASC`).
  4. **Negative sampling test**: all positives are kept, the number of negatives per query is correct, and the result is reproducible with a fixed seed.
  5. **Integration test**: run one fold on small data from candidate generation to evaluation.
- Tests must be fast: the whole of `make test-fast` should finish within 1 minute.
- **Tests run in two processes** (for the same reason as "the M7 code may not import torch" in §5): `uv run pytest tests/unit` runs only the torch-side tests, and `uv run pytest tests/unit --lightgbm-side` runs only the LightGBM-side tests (file names starting with `test_ranker_`, `test_v_`, `test_speed_test`, `test_m7_`, `test_final_`, `test_diag_`, `test_lgb_arm`, see `tests/conftest.py`; `test_lgb_arm` is the two LightGBM arms of M9; `test_diag_` is the diagnostics after M8, which train no model, but the modules that read the locked configuration and the result tables load lightgbm). `make test-fast` and `make test` run both. A newly written test that trains a LightGBM model must use one of these prefixes in its file name and may not import torch; when both torch and lightgbm appear in one test process, `tests/conftest.py` fails the run.

---

## 8. Common commands

The environment is managed by [uv](https://docs.astral.sh/uv/): the Python version is in `.python-version`, dependencies are written in `pyproject.toml`, and exact versions are locked in `uv.lock` (in git). All commands in the Makefile run through `uv run`.

```bash
make setup        # uv sync: create / update .venv from uv.lock (including the dev group)
make format       # uv run ruff format + ruff check --fix
make lint         # uv run ruff format --check, ruff check, pyright
make test-fast    # uv run pytest tests/unit, then once more with --lightgbm-side (two processes, see §7)
make test         # uv run pytest tests (unit + integration)
```

- Run any ad-hoc Python command with `uv run ...` too; do not call the system Python or `pip install` directly.
- **Adding a dependency needs the user's consent first** (see §3.4); after consent, use `uv add <package>` (for development tools, `uv add --group dev <package>`), so that `pyproject.toml` and `uv.lock` are updated together. Do not edit `uv.lock` by hand.

**After any code change, you must run `make lint` and `make test-fast` and report the results in your reply.**

---

## 9. Definition of Done

A task is done only when all of the following hold:

- [ ] The code follows the conventions of section 6, and `make lint` passes.
- [ ] New code has corresponding tests, and `make test-fast` passes.
- [ ] No hard rule of section 3 is broken.
- [ ] If the design changed (not just an implementation detail), an entry has been appended at the end of `docs/decisions.md`: date, decision, reason.
- [ ] The reply states: which files changed, why, how it was verified, and what is still not done or uncertain.

---

## 10. Git

- **The agent does not commit on its own** (nor push, nor tag). After finishing a task, list in the reply:
  1. the files suggested for the commit (if a task should be split into several commits, list them grouped by commit);
  2. the commit message of each commit (format in the next rule).

  The user commits. The agent may commit only when the user explicitly asks it to in the conversation.
- **Branches**: one branch per milestone, named `m<number>-<content>`, for example `m1-recall-r1-r3`. The agent may create and switch branches, but **does not commit and does not merge**. When a milestone is finished, the user fast-forward merges it into main (`git merge --ff-only <branch>`).
- Each commit does one thing; commit messages are in English, in the format `<scope>: <what changed>`, for example `recall/r2: add decayed-frequency scoring`.
- A tag is created when a configuration is locked (at points such as M5 and M7), for example `lock-channels-v1` (created by the user; the agent only suggests the tag name).
- Do not commit files under `data/` or `artifacts/`.
- Do not push unless the user asks.

---

## 11. Documentation language

**The repository keeps English documents only (from 2026-11-21; English had been the primary language since 2026-11-19).**

- **Write in English.** All documents, entries in `docs/decisions.md`, commit messages and code comments (including docstrings) are written in English.
- **No translations.** No Chinese version, or version in any other language, of a document is written or maintained. The translations and Chinese archives that existed before 2026-11-21 were deleted; they remain in the git history up to tag `v0.2`.
- **`docs/decisions.zh-CN.md` is a frozen archive, kept only for the lock-file references.** The `decision_record` fields of the lock files under `configs/locks/` quote entry titles of the decision log in Chinese, as it was at tag `v0.1`, and lock files may not be modified (§3.3). A quoted title is found in this archive, and the entry at the same position of `docs/decisions.md` is its English version; the table of every quoted title is in `docs/decisions.md` (2026-11-21, "English-only documents"), and `tests/unit/test_locks.py` checks the ranker lock's pointer in this way. **Do not edit this archive and do not delete it.**
- `docs/decisions.md` stays append-only: new entries are in English, and earlier entries are not edited.
- This rule is about files in the repository. It does not decide the language of replies in a conversation: reply in the language the user writes in, unless the user asks for another.

---

## 12. Stage 3: MLOps on GCP (M10–M15)

**The design of stage 3 is `docs/mlops.md`** (approved by the user on 2026-11-21; `docs/proposal.md` §17 points to it). Read its relevant sections before any stage-3 task. Stage 3 changes how the system runs, not what it computes; everything in §1–§11 above still holds.

- **The invariants of `docs/mlops.md` §4 are binding**, all eight of them. In short: only data from before the anchor; lock files, existing result tables and the history of `docs/decisions.md` are not modified, and new records go to new tables; **T1–T4 labels are read only by the replay, for monitoring and verification, never to select or tune anything, with thresholds fixed and recorded before the first replay tick**; code names no cloud project, bucket or region (they come from configuration); no token, key or credential file goes into the repository, an image, a log or a document; full-data runs start from committed code and record their commit and image digest; a budget alert exists before any billable resource is created.
- **Milestone branches**: `m10-…` to `m15-…`, one per milestone (§10), for example `m10-container-ci`.
- **The laptop stays the reference for bit-identical checks** (Apple silicon, macOS, Homebrew libomp). **Results of a container or of the cloud are accepted by the tolerance of `docs/mlops.md` §8.3** and are never described as bit-identical to the laptop. A difference above the tolerance stops the milestone and is investigated; the tolerance is not widened after a replay result is seen.
- **A test or a run-time check is never loosened, skipped or deleted to make a container or CI run pass.** A platform assumption (a path, Homebrew, a `.git` directory) is made platform-aware explicitly, with a test for the Linux branch; a bit-exact number that differs on Linux and a test that needs data the container does not have are listed for the user's decision.
- **New dependencies need the user's explicit approval in the prompt of each milestone** (§3.4; `docs/mlops.md` D6).
