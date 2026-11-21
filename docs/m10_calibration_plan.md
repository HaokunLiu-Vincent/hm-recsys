# M10 calibration plan: week V4 in the arm64 container

> **Status.** Written on 2026-11-21 as step 7 of M10 part 1. **Nothing in this plan has been run.** It is the plan for M10 part 2, which measures the calibration items of `docs/mlops.md` §8.3 in the arm64 container on the laptop. Part 2 starts only after the user has read this plan and decided the open points of §9.
> Run times and memory peaks marked "expected" are estimates from the stage-1 records of the laptop; the container has not been timed.

## 1. What is measured and why

`docs/mlops.md` §8.3 accepts container and cloud results by a tolerance instead of bit-identity, because Linux uses another OpenMP runtime and another BLAS than the laptop. Before the tolerance is relied on, it is measured once on week V4, item by item, so that a difference can be traced to the first step that produces it.

| # | Item | What it isolates |
|---|---|---|
| A | R1, R2, R3, R4 candidates of V4 | Counting and sorting in Polars. Expected: identical |
| B | One R5 (WALS) fit as of V4, with its candidates | `implicit` on Linux: multi-threaded, OpenBLAS (§8) |
| C1 | R6 candidates of V4 from the **recorded** model `V3.pt` | Encoding and dot products only, no training |
| C2 | One R6 fine-tuning step from the **recorded** model `V2.pt` on the week of V3, then the R6 candidates of V4 from the resulting model | What one tick of the weekly pipeline does (steps C4 and O1 of `docs/mlops.md` §6.1) |
| D | LightGBM: training on the frozen samples of F4, V1, V2, V3 and scoring of the frozen features of V4, then evaluation | LightGBM on GNU libgomp instead of Homebrew libomp |

C1 is not required by `docs/mlops.md`; it is added because it costs about 20 seconds and separates "the model differs" from "the same model is applied differently" if C2 is outside the tolerance.

Only V4 is touched. **No label of a T week is read**: the evaluators used are those of M6 and M7, whose fold guards refuse a T fold.

## 2. The image and the container

- **Image**: built with `make image-build` from the clean, committed tree of part 2 (the calibration driver is new code, so the commit is not the one of part 1). Recorded with every run: the commit (`HMREC_IMAGE_GIT_COMMIT`), the image id, and the output of `make image-info`.
- **Resources the container runtime must be given**: Docker Desktop with **at least 6 CPUs and 14 GiB** of memory (today: 6 CPUs, 15.6 GiB). Each container runs with `--cpus 6 --memory 12g`: 6 threads is `num_threads` of `configs/base.yaml`, and 12 GiB is the project's memory budget, so a process above the budget is stopped by the kernel instead of only logging a warning.
- **One item at a time**, each in its own container and process, as in stage 1 (torch and LightGBM never share a process).

### Mounts

Inputs are mounted **read-only**. The only writable mounts are two new, empty folders. A stage-1 artifact or result therefore cannot be overwritten, whatever the code does: the kernel refuses the write.

```bash
RUN_ROOT="$PWD"                       # the repository on the laptop
mkdir -p artifacts/m10_calibration/linux-arm64 results/m10_calibration

CALIBRATE="docker run --rm --cpus 6 --memory 12g \
  --mount type=bind,src=$RUN_ROOT/data/processed,dst=/app/data/processed,readonly \
  --mount type=bind,src=$RUN_ROOT/artifacts,dst=/app/artifacts,readonly \
  --mount type=bind,src=$RUN_ROOT/artifacts/m10_calibration,dst=/app/artifacts/m10_calibration \
  --mount type=bind,src=$RUN_ROOT/results/m10_calibration,dst=/app/results/m10_calibration \
  hmrec:dev"
```

| Mount (inside the container) | Mode | Content |
|---|---|---|
| `/app/data/processed` | read-only | Processed transactions, articles, customers, embeddings |
| `/app/artifacts` | read-only | Stage-1 queries, labels of V4, candidates, models, samples, features (inputs and references) |
| `/app/artifacts/m10_calibration` | read-write | **New.** Everything the calibration generates, under `linux-arm64/` (M13 adds `linux-amd64/`) |
| `/app/results/m10_calibration` | read-write | **New.** The calibration's own result tables |

`/app/results` (the committed stage-1 tables), `/app/configs` and the code are inside the image and belong to root; the process runs as user `hmrec` and cannot change them.

### Where outputs go

| Output | Location |
|---|---|
| Candidates, vectors, the fine-tuned R6 model, the LightGBM model, Top-12 | `artifacts/m10_calibration/linux-arm64/…`, mirroring the stage-1 layout below that folder (not in git) |
| One row per comparison | `results/m10_calibration/comparisons.csv` (new table, committed) |
| One row per process: run time, sampled and true memory peak, image commit and id, platform, OpenMP runtime, BLAS | `results/m10_calibration/runs.csv` (new table, committed) |
| The R6 fine-tuning record of the calibration (start model, sha256, losses) | `results/m10_calibration/twotower_chain.csv` (new table; `results/m6_twotower_chain.csv` is not written) |

### The driver (new code of part 2)

The stage-1 scripts cannot be used as they are, for three reasons: they write to the stage-1 locations and skip a step whose output exists; `scripts/m6_generate_candidates.py` appends to `results/m6_runs.csv` and `results/m6_twotower_chain.csv`; and the LightGBM entry points go through `load_locked_ranker`, which refuses on Linux (§8).

Part 2 therefore adds one script, `scripts/m10_calibrate.py`, and a module `src/hmrec/calibration/`. They contain **no new computation**: each item calls the same functions the stage-1 scripts call (`hmrec.recall.generate`, `hmrec.recall.locked`, `hmrec.ranker.train`, `hmrec.ranker.predict`, the evaluators of `hmrec.eval`), with a second `ProjectPaths` whose artifacts root is the calibration folder. New code is only: the output root, the comparisons of §5, and the two tables. The commands below are the ones part 2 will provide; their names may change in review, their mounts and inputs may not.

## 3. Inputs and their sha256

The sha256 values in the last column marked "recorded" are in a stage-1 record. The others were computed on the laptop on 2026-11-21 for this plan, because stage 1 recorded no sha256 for them; the driver recomputes every one of them inside the container before an item starts and stops on a mismatch.

| Input | Used by | sha256 | Source |
|---|---|---|---|
| `data/processed/transactions.parquet` | A, B, C | `67391a1e9f7f6ac6902a358c192b5ee4004a2b0261710535f280b72f73e75315` | computed 2026-11-21 |
| `data/processed/articles.parquet` | A, B, C | `515d85638a2ae0f6d430ef9be540f934ba579abb2c0f67362e7c699c546c77df` | computed 2026-11-21 |
| `data/processed/customers.parquet` | A, B, C | `8995d79e2a2c86499acd934d0af0ccb79db5ae7fe85e430ae1b8db5414aefa39` | computed 2026-11-21 |
| `data/processed/metadata.json` | A, B, C | `da0a6023af36d02014a4df2feb7ee369bcb50745311bb9550b854cf3e4ec017c` | computed 2026-11-21; its `config_hash` `9a82333b…` is checked by `scan_processed_parquet` |
| `artifacts/queries/V4.parquet` (72,035 queries) | all | `8dc76f9100391f887dba2964cd302f9b90752a9d001f7aab3e87d70a3b2fd104` | computed 2026-11-21 |
| `artifacts/labels/V4.parquet` | evaluation only | `32af63ff461ccd93d6f392bdc9583b259beb4ba3e15f37d5b7bd5827981c58b6` | computed 2026-11-21 |
| R6 base checkpoint and alarm users | C | the values in `configs/locks/twotower.lock.yaml` | recorded (lock file; checked by `hmrec.recall.locked`) |
| `artifacts/models/twotower/chain_locked/V2.pt` | C2 | `8cbd33f961fa97c1f2f8cd1ecda7da3020f167f9083fe7c16d0353f541c22caf` | recorded: `results/m6_twotower_chain.csv`, fold V2; the file on disk matches |
| `artifacts/models/twotower/chain_locked/V3.pt` | C1 | `21983ac5fb092b2e743313d29df455a8b5dbd442af487a27764ad383c92f6dea` | recorded: `results/m6_twotower_chain.csv`, fold V3; the file on disk matches |
| Ranker samples `artifacts/ranker/samples/full/{F4,V1,V2,V3}/r40.parquet` | D | F4 `98f98abf212c751de3eaa5cdb2331b6d2adc486acde3abed878d6bcd961b2a47`; V1 `914e7ffa8272e09a03135334330cb8217019aed2ada3f29eaa0a652c4f725d79`; V2 `b68b799c9dd5ea0346d25d285cedc25c44554fc9156fbfafc36d3ad8b62041cb`; V3 `b6662a1a1bd2f5f94fa3ee5111d99a64f9c9494eecabeb909164d4d029ad0284` | recorded: `results/m8_rehearsal_check.csv` and each sample's `r40.json` |
| Features of V4: `artifacts/features/V4/part-0000{0..7}.parquet` (22,293,247 rows) | D | one sha256 per part inside `features.json`; `features.json` itself `f5c769eb58016a799f7062d756f229b1a972a395bb98d023735c2545195ee1f1` | recorded per part (`verify_feature_files`); the record's own hash computed 2026-11-21 |
| `artifacts/features/category_codes.json` | D | `049accdf849d05b9bf5c6726225c50a9e2197d2923cffcf6db219ac1a2484dff` | computed 2026-11-21 |
| The four lock files | all | referenced by sha256 in `configs/locks/ranker.lock.yaml` | recorded |

**References** (stage-1 outputs of V4 that the container results are compared with; read-only):

| Reference | sha256 (computed 2026-11-21) |
|---|---|
| `artifacts/candidates/V4/r1_popularity.parquet` | `aea2e4f234960cd53c07b2baab8d74fd1c37c00a66155a64e6810d9d7cad32b0` |
| `artifacts/candidates/V4/r2_repurchase.parquet` | `f23e3ea4c0c5fe3a1543e89c36ba50267c52ffb74d15c78dd0ce7a6bdf4e9721` |
| `artifacts/candidates/V4/r3_sibling.parquet` | `28a5db10b34e7b8e9ad7e89485cbbeb16278d38e0d2e340ee3a56a8df4b3e1a7` |
| `artifacts/candidates/V4/r4_itemcf.parquet` | `8a5a629186f8d4c27a8fb143703fc1739969bfe76ec9f9bf62ebb9a1be4b9d3b` |
| `artifacts/candidates/V4/r5_wals.parquet` | `cc047f0cac98d70d69a726aa257c4dc1a093bb5a1981b4252d66bbd1f625f1f9` |
| `artifacts/candidates/V4/r6_twotower.parquet` | `f243ee8a329d4d45b5a4447af4b32e203016e905dd3b8db3ddd751f0d4f18c1d` |
| The LightGBM model of V4 saved by M7 / M8 (500 trees, configuration `ca1a3cdfce49`) and its saved Top-12 | sha256 of the first 500 trees as checked by the M8 rehearsal (`results/m8_rehearsal_check.csv`, "full model: first 500 trees") |

## 4. The items

Stage-1 numbers are those of the laptop for V4 (`results/m6_runs.csv`, `results/m6_twotower_chain.csv`, `results/v_phase.csv`, `results/m7_runs.csv`). "Expected in the container" is an estimate: between the laptop's time and about three times it.

### A. R1–R4 candidates

```bash
$CALIBRATE python scripts/m10_calibrate.py --item recall --fold V4 --channel r1_popularity
$CALIBRATE python scripts/m10_calibrate.py --item recall --fold V4 --channel r2_repurchase
$CALIBRATE python scripts/m10_calibrate.py --item recall --fold V4 --channel r3_sibling
$CALIBRATE python scripts/m10_calibrate.py --item recall --fold V4 --channel r4_itemcf
```

| | R1 | R2 | R3 | R4 |
|---|---|---|---|---|
| Stage-1 rows | 7,203,500 | 1,219,040 | 1,982,031 | 6,513,892 |
| Stage-1 recall (`results/m6_candidates.csv`) | 0.109989 | 0.041286 | 0.033426 | 0.088194 |
| Stage-1 time / true peak | 1.4 s / 1.97 GB | 0.5 s / 0.96 GB | 0.6 s / 1.12 GB | 23.6 s / 4.09 GB |
| Expected in the container | under 10 s / 2 GB | under 10 s / 1 GB | under 10 s / 1.2 GB | 25–75 s / 4–5 GB |

- **Comparison**: content identical to the reference (§5); if not, the share of recorded pairs present and the recall.
- **Tolerance of `docs/mlops.md` §8.3**: identical.
- **Risk**: R1, R3 and R4 store float32 scores. A score that differs in its last bit can swap two articles whose scores are almost tied, although the candidates are "counting and sorting". If that happens it is reported with the rows concerned; the tolerance is not changed by this plan.

### B. One R5 fit with its candidates

```bash
$CALIBRATE python scripts/m10_calibrate.py --item recall --fold V4 --channel r5_wals
```

- **Stage 1**: 6,072,000 rows, recall 0.094975, 179.1 s, true peak 3.77 GB, fitted on **one thread** (the macOS wheel of `implicit` has no OpenMP).
- **Expected in the container**: 60–540 s, 4–6 GB. On Linux `implicit` runs on `num_threads = 6` OpenMP threads, and it warns that OpenBLAS also uses 6 threads, which "can lead to severe performance issues"; so the fit may be faster or much slower than on the laptop. The behaviour is **not** changed for the calibration: it is measured as it is (§9, decision 3).
- **Comparison**: content identical or not; share of the 6,072,000 recorded pairs present; recall against 0.094975; also the largest absolute difference of the saved user and item vectors, as information.
- **Tolerance**: at least 99% of the recorded pairs; recall within 1.0% relative.

### C. R6

```bash
# C1: candidates of V4 from the recorded model V3.pt (no training)
$CALIBRATE python scripts/m10_calibrate.py --item r6-predict --fold V4 --start recorded

# C2: fine-tune the recorded V2.pt on the week of V3, then the candidates of V4 from the result
$CALIBRATE python scripts/m10_calibrate.py --item r6-fine-tune --fold V3
$CALIBRATE python scripts/m10_calibrate.py --item r6-predict --fold V4 --start calibration
```

- **Stage 1, the fine-tuning step of V3**: 238,334 new and 238,334 replay rows, 2 epochs, 1,862 steps, mean training loss 5.536406, alarm loss 5.616730, 55.8 s, sampled peak 3.49 GB. Device `cpu` (locked).
- **Stage 1, the candidates of V4**: 7,203,500 rows, recall 0.160109; prediction 18.3 s; true peak of the whole step 4.38 GB.
- **Expected in the container**: C1 20–60 s; C2 fine-tuning 60–180 s, prediction 20–60 s; 4–6 GB.
- **Comparison**: for C1 and C2 separately: content identical or not; share of the 7,203,500 recorded pairs present; recall against 0.160109. For the fine-tuning step: mean training loss and alarm loss against the record, and whether the sha256 of the model equals `21983ac5…` (expected: not equal).
- **Tolerance**: at least 99% of the recorded pairs; recall within 1.0% relative.
- **The week of V3 is used as training data** by the fine-tuning step, exactly as in stage 1 (M6 fine-tuned on it after predicting V3). It is a V week; nothing is selected with it.

After A, B and C, the union of the six calibration channels is formed with `hmrec.fusion`:

```bash
$CALIBRATE python scripts/m10_calibrate.py --item union --fold V4
```

- **Comparison**: deduplicated rows against 22,293,247 (tolerance: within 1%); union recall against 0.276163 (within 1.0% relative). Run time under 30 s, peak under 3 GB (stage 1: the monitoring step, 2.7 s, 2.0 GB).

### D. LightGBM training and evaluation from the frozen features

```bash
$CALIBRATE python scripts/m10_calibrate.py --item ranker --fold V4
```

- **What it does**: builds the LightGBM Dataset from the four recorded sample files, trains the locked configuration to 500 trees, scores the frozen features of V4 (22,293,247 rows) and evaluates with the M7 evaluator. Features and samples are the stage-1 files, so only LightGBM and its runtime differ.
- **Stage 1**: 10,453,387 training rows in 150,955 queries; Dataset 6.5 s, 1.91 GB; training 276.7 s for 1,000 trees (about 140 s for 500); scoring 156.4 s for all checkpoints up to 1,000 trees; true peak 2.83 GB. **MAP@12 of V4 at 500 trees: 0.034849701577104486**; Recall@12 0.077972; NDCG@12 0.053913 (`results/v_phase_metrics.csv`).
- **Expected in the container**: 4–15 minutes in total, 3–5 GB.
- **Comparison**: MAP@12, Recall@12 and NDCG@12 against the record; the model text tree by tree against the stage-1 model: **the index of the first tree that differs**, or "none"; the share of queries with an identical Top-12 and the mean overlap of the Top-12 sets.
- **Tolerance**: MAP@12 of the week within 1.0% relative.
- **This item cannot run through the locked entry point as it is** (§8, and decision 1 of §9).

## 5. How two results are compared

| Comparison | Definition |
|---|---|
| Identical | The two tables have the same rows: the same `(query_id, article_idx, rank)` and bit-equal float32 `score`. Equality of the files' sha256 is also reported, as information only (a Parquet file can differ in bytes with equal content) |
| Share of recorded pairs present | Number of recorded `(customer, article)` pairs of the week that are also in the container's candidates, divided by the number of recorded pairs. One query per customer and week, so this is the share of recorded `(query_id, article_idx)` |
| Recall | The same function and labels as `scripts/m6_monitor_candidates.py`; relative difference = (container − record) / record |
| MAP@12 | The M7 evaluator (`hmrec.eval.ranker_eval`), float64; relative difference as above |
| First tree that differs | The model text is split into trees; trees are compared as text in order; the first index at which they differ |

Every comparison is one row of `results/m10_calibration/comparisons.csv`: item, quantity, record, container value, difference, tolerance of `docs/mlops.md` §8.3, inside or outside.

## 6. Order, total cost, stop conditions

1. `make image-build`, `make image-info`, `make image-lint`, `make image-test` (all green).
2. Items in the order A, B, C1, C2, union, D. Expected total: **15–45 minutes**, one process at a time, highest expected peak 4–6 GB.
3. **Stop conditions**: an input sha256 mismatch; a check of the code fails (schema validation, scores equal to dot products of the saved vectors, the fine-tuning loss above the recorded stop limit); a true memory peak above 12 GB; a loss that is NaN. A result outside the tolerance does **not** stop the remaining items: all of them are measured, and the result goes to the user.
4. **After the measurement** the thresholds of `docs/mlops.md` §8.3 are confirmed or changed by the user and recorded in `docs/decisions.md`. Nothing downstream (M12, M13) starts before that.

## 7. What the calibration does not show

- One week, one run. It does not show how a difference grows over the eight chained R6 updates of the replay (`docs/mlops.md` §15); the replay of M12 compares every week with its record.
- The arm64 container only. The x86-64 calibration is the first step of M13.
- Feature building is not a calibration item of `docs/mlops.md` §8.3. Its acceptance in the replay is internal consistency (row count and keys), not equality with stage 1.
- Whether the container's results are "as good": matching a record shows that the pipeline reproduces the system, nothing more (`docs/mlops.md` §16).

## 8. How the locked entry points and the run-time checks behave in the container

Measured on 2026-11-21 with the image of M10 part 1 (`make image-info`, the unit tests inside the image) or read from the code; nothing was run on the data.

| Entry or check | In the arm64 container |
|---|---|
| `hmrec.recall.locked.load_locked_recall` (R1–R5) | Reads `configs/locks/channels.lock.yaml`. It checks no library version, so it **passes** |
| `hmrec.recall.locked.load_locked_twotower_chain` (R6) | Checks the sha256 of the base checkpoint, of the start model and of the alarm users before loading. **Passes** for the recorded files. In C2 the start model of the prediction is the calibration's own model; its sha256 is checked against the calibration's own chain table, never against `results/m6_twotower_chain.csv` |
| `scan_processed_parquet` | Checks `day_zero`, age buckets and the preprocessing hash against `metadata.json`. **Passes** (same files) |
| `feature_files.verify_feature_files` | Checks the sha256 of every feature part against `features.json`. **Passes** (same files, read-only) |
| `hmrec.git_state.git_state` | Returns the commit of the image and `git_dirty=False`; refuses if a `.git` is mounted over the code |
| `hmrec.ranker.runtime.read_lightgbm_runtime` | Records `/usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0` and version `libgomp1 14.2.0-19`; refuses a process in which torch is imported |
| **`hmrec.ranker.locked.load_locked_ranker`** | Check 1 (the lock's settings give its `config_id`) passes. Check 2 (sha256 of the features, channels and twotower locks) passes. **Check 3 fails**: LightGBM version `4.7.0` and `num_threads` 6 match, but the OpenMP runtime is `libgomp1 14.2.0-19` and the lock records Homebrew libomp `22.1.4`. The entry raises `ValueError` and returns no configuration. **This is intended**: the lock says which runtime the V phase ran on, and a Linux process is not that runtime |
| The memory limit | `memory.limit_gb` logs a warning above 12 GB as on the laptop; the container limit of 12 GiB additionally stops the process |

Two facts about threads, measured in the image and not changed (`make image-info`):

- **`implicit` (R5) is multi-threaded on Linux and single-threaded on the laptop.** The Linux wheel links the system libgomp; the macOS wheel links no OpenMP runtime. The R5 code passes `num_threads` of `configs/base.yaml` (6) to `AlternatingLeastSquares` on both platforms; on the laptop the value has no effect.
- **torch**: `scripts/m6_generate_candidates.py` (R6 prediction and fine-tuning) and the other scripts of M3–M6 and M9 that train with torch call `torch.set_num_threads(num_threads)` (6); the call carries over unchanged. Without that call torch uses 6 threads in the container (the CPUs it is given) and 8 on the laptop. In the arm64 image torch and `implicit` share one libgomp in a process; which file it is (the system's or the copy in the torch wheel) depends on which of the two is imported first.

## 9. Decisions needed before part 2 runs

1. **How the LightGBM item gets the locked configuration on Linux.** `load_locked_ranker` refuses, by design. Proposal, which changes no lock file and leaves the laptop entry exactly as it is: a **second entry for tolerance-accepted platforms**, `load_locked_ranker_for_platform`, which performs checks 1 and 2 unchanged, requires the LightGBM version and `num_threads` of the lock unchanged, and compares the OpenMP runtime not with the lock but with a **platform record** — a new file (for example `configs/platforms/linux-arm64.yaml`: the libgomp package version, the base image digest, the decision that accepted it). Results obtained through this entry carry the platform in their records and are never compared bit by bit with the laptop. For the calibration itself, which runs before any platform is accepted, the driver uses the same entry in a `calibration` mode that records the runtime it finds instead of comparing it. The alternative is to keep LightGBM training on the laptop only, which ends stage 3 at the container.
2. **Whether C1 is wanted** (it is an addition to `docs/mlops.md` §8.3), and whether the names and locations of §2 are acceptable (`artifacts/m10_calibration/`, `results/m10_calibration/`, `scripts/m10_calibrate.py`, `src/hmrec/calibration/`).
3. **R5 threads in the container.** Measure as it is (6 OpenMP threads, OpenBLAS with its own 6 threads, as the image runs today), which is what this plan does. If the fit turns out to be much slower than on the laptop, setting `OPENBLAS_NUM_THREADS=1` for R5, as `implicit` recommends, would be a change of the run-time environment and is proposed then, not made silently.
4. **Reading the labels of V4** for the recall and MAP@12 comparisons: V weeks may be evaluated (`AGENTS.md` §3.2), so no further consent is needed; it is listed here so that it is not a surprise.
