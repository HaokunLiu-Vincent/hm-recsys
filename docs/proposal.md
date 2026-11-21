# H&M Personalized Fashion Recommendation System — Project Proposal v0.1.2

> Date: 2026-11-15 (v0.1.1: O1–O5 decided; §14 directory structure and §15 code conventions added)
> v0.1.2 (2026-11-15, from the outcome of the project skeleton and the toolchain migration): §1.2 / §15 changed to Python 3.12 + uv (dependencies locked by `uv.lock`) + Pyright; §1.3 notes that memory peaks are sampled values; §2.1 records the measured facts about the precomputed embeddings; §4.2 states that "better" means strictly better; §5 R6 explains that articles without an image embedding use a mask; §8.3 specifies when `img_sim_hist` / `txt_sim_hist` are NaN; §14 updates the `data/` directory tree.
> Status: the first version of the plan after starting over. Results of the old project serve only as reference numbers; no old code, configuration or conclusion is carried over.
> System structure: **6 recall channels → candidate union + dedup → 1 LightGBM LambdaMART ranker → Top-12**
> Principle: **run as little as possible**. Each channel gets only 1–2 ablations plus a small HPO grid. One factor is changed at a time (one-factor-at-a-time, OAT); no full factorial.

---

## 0. One-page summary

### 0.1 Goals

| ID | Metric | Target | Where it is checked |
|---|---|---|---|
| G1 | End-to-end macro **MAP@12** (LambdaMART) | **≥ 0.032** (minimum target) | V1–V4 select the model, T1–T4 give the final report |
| G2 | Macro **union recall** of the 6 channels merged | **≥ 24%** (measured on D1–D4: **25.24%**, the configuration locked at M5; see the note below the next table) | D1–D4 (before recall is locked) and every week of F/V/T |
| G3 | LambdaMART better than the RRF baseline | MAP@12 higher in at least 3 of the 4 weeks | V1–V4, T1–T4 |

Reference values of Recall@quota per channel (from the old project, **doubtful; used only as a sanity check, not as acceptance criteria**):

| Channel | Quota | Reference Recall@quota | Measured in the D phase (the configuration locked at M5, the four weeks D1–D4 pooled) |
|---|---|---|---|
| R1 Popularity | 100 | ~7% | **7.39%** |
| R2 Repurchase | ≤30 | ~4% | **4.27%** |
| R3 Sibling | ≤50 | ~3.6% | **3.24%** (outputs only `A(a)`; 3.26% before the switch) |
| R4 ItemCF | 100 | ~8% | **9.33%** |
| R5 WALS | 100 | ~10% | **10.81%** |
| R6 Two-Tower | 100 | ~10% | **14.63%** (training chain, each week predicted first and fine-tuned afterwards; 11.84% for the base without updates) |

The measured column comes from the trial recorded for each channel in `configs/locks/` (`results/d_phase.csv`; for R3, `results/m4_trials.csv`). **The D-phase scores are optimistically biased** (the configurations were selected on these very 4 weeks, see §12 item 4); they serve only for comparison with the reference values and do not represent performance on the F / V / T weeks. No channel is more than half below its reference value.

If the measured result of a channel is more than half below its reference value, first look for bugs (time cut-off, ID mapping, deduplication), and only then discuss the model itself.

### 0.2 Schedule

| Phase | Weeks | Dates (2020, inclusive) | What happens | What is selected here |
|---|---|---|---|---|
| Base | — | 2019-12-03 – 06-02 | Training of the Two-Tower base model | — |
| **D** | D1–D4 | 06-03 – 06-30 | **Ablation + HPO** of the 6 recall channels, then all recall configurations are frozen | Hyperparameters of R1–R6 |
| **F** | F1–F4 | 07-01 – 07-28 | Candidates generated with the frozen configuration, as **ranker training data** | Nothing |
| **V** | V1–V4 | 07-29 – 08-25 | **Rolling retraining + HPO** of the ranker, system-level ablation | Hyperparameters of the ranker |
| **T** | T1–T4 | 08-26 – 09-22 | Everything locked, final evaluation | Nothing |

### 0.3 Three iron rules

1. **Each week is predicted using only data with `t_dat < anchor`**. This holds for model weights, statistics and features alike.
2. **Each phase selects only what it is meant to select**: D selects recall, V selects the ranker, F and T only execute.
3. **Hyperparameters (including the number of epochs and the number of trees) are selected once over 4 weeks and then locked**. Afterwards only model parameters (weights or statistics) are updated each week; nothing is selected again.

---

## 1. Experiment environment and resource budget

### 1.1 Hardware

| Item | Specification |
|---|---|
| Machine | MacBook Pro, Apple **M1 Max** (10 CPU cores + 32 GPU cores) |
| CPU available to the project | **6 cores** |
| GPU available to the project | **28 cores** (PyTorch **MPS** backend) |
| Memory available to the project | **12 GB of unified memory, shared by CPU and GPU** (8 GB before 2026-11-18, see §1.3). This is a budget the project sets for itself; the machine has 32 GB of physical memory |
| Cloud | Not used. Colab is considered only when something cannot run on this machine |

### 1.2 Software conventions

| Component | Device | Settings |
|---|---|---|
| Data processing | CPU | Polars only (lazy API / streaming preferred), no pandas; all intermediate results are stored as Parquet |
| R1–R4 | CPU | Vectorized computation, chunked where needed |
| R5 WALS | CPU | `implicit`, CPU version, `num_threads=6` (the GPU version supports CUDA only) |
| R6 Two-Tower | **CPU** | PyTorch, `device="cpu"` (`training.device` in `configs/recall/r6_twotower.yaml`), `torch.set_num_threads(6)`, float32 throughout. The original plan was MPS; in the 200-step benchmark of the real model CPU takes 30.4 ms per step and MPS 45.2 ms, which does not satisfy the rule "use MPS only if it is more than 1.3 times faster", and CPU results are bit-reproducible (`docs/decisions.md` 2026-11-17) |
| Ranker | CPU | LightGBM, `num_threads=6` |
| Random seeds | all | A fixed seed for every configuration, written in the config |
| Python environment | — | **Python 3.12** (`.python-version`), managed by **uv** (`.venv`); dependencies are written in `pyproject.toml`, and exact versions are locked by **`uv.lock`** (in git) |
| Development tools | — | `ruff` for formatting and lint; **Pyright** (`standard` mode) for type checking; `pytest` for tests. Consistent with §15 |

### 1.3 Constraints of the memory budget (12 GB) and how they are met

Memory is the main limitation. The budget is set by the project itself (§1.1; the machine has 32 GB of physical memory) and is now 12 GB. **The limit on the true peak of a single process is 12 GB** (`memory.limit_gb` in `configs/base.yaml`), equal to the budget. **Only one heavy task runs at a time**; nothing runs in parallel.

The limit is only an alert threshold: when it is exceeded, the process raises an error after writing its run record and a person decides; it takes part in no computation and affects neither results nor speed. **The chunk settings (`chunking` in `configs/base.yaml`, `scoring_query_chunk_size` in `configs/ranker.yaml` and so on) are not relaxed with the limit**, and the true peak of every process is recorded as usual. The highest true peak measured up to 2026-11-18 is the 7.25 GB of feature generation for V2.

Changes of the limit (all on 2026-11-18, see `docs/decisions.md`): at first it was "about 6 GB"; the true peak of F1 feature generation was 6.10 GB, and the user decided not to change the code and raised the limit to 7 GB ("leave 1 GB of headroom"); then the true peak of V1 was 7.002 GB while the machine has 32 GB of physical memory, so the reason "leave headroom to prevent swapping" did not hold, and the limit was set equal to the budget of the time, 8 GB; after M7 step 1 and before step 2, the budget and the limit were both changed to 12 GB. The "6 GB limit", "within 6 GB" and "8 GB" written earlier in the table below, in the notes below the table and in later sections refer to the limit of their time and are kept as a record, not rewritten.

| Task | Estimated memory | Measured | How it is handled |
|---|---|---|---|
| Transaction table held in memory | ~0.5 GB | **0.44 GB** (454.7 MB, 31,788,324 rows; the Parquet file is 160 MB) | All IDs mapped to int32; dates stored as int16 (days since the first day); price as float32; channel as int8 |
| M0 preprocessing (streaming CSV to Parquet, aligning the embeddings) | — | **Peak 2.25 GB** (of which the transaction-table step peaks at 2.10 GB); about 6 seconds overall | Polars lazy scan + streaming `sink_parquet`; string IDs are converted to int32 before sorting |
| M0 query / label generation (16 folds) | — | **Peak 0.36 GB** (sampled by MemoryTracker; maximum RSS from `/usr/bin/time`: 0.38 GB); about 1.9 seconds for the 16 folds (about 0.1 seconds per fold) | Each fold reads only two columns of the label week (`customer_idx, article_idx`); about 69,000–120,000 queries and about 210,000–490,000 labels per week |
| Candidate generation of R1–R3 (D1, default configuration) | — | **Overall peak 1.77 GB** (maximum RSS from `/usr/bin/time`, the whole script: reading queries / labels, computing the active pool, running 3 channels, evaluating); run time R1 0.6 seconds, R2 0.2 seconds, R3 0.5 seconds, 2.2 seconds overall. A single channel runs for less than the 0.5-second sampling interval, so **single-channel peaks are not reliable** and are not listed separately | Each channel reads only the columns it needs; the histories of R2 / R3 are first semi-joined on the query users |
| R4 ItemCF co-occurrence matrix (365 days) | Possibly far above 8 GB without chunking | **D1 (CF3, α=0.7): overall peak 2.95 GB** (maximum RSS from `/usr/bin/time`, including reading the data; 2.62 GB sampled by MemoryTracker); run time 17.2 seconds (building the graph + neighbours 15.5 seconds, scoring 1.8 seconds). Size of the graph: 15.4 million purchase lines → 13.25 million (user, article) pairs, 988,000 users, 72,000 articles, `Σ d_u²` ≈ 510 million; 28,110 seeds in 15 chunks. **D phase (the four weeks D1–D4, 9 configurations): the heaviest trial peaks at 4.76 GB** (sampled by MemoryTracker, the whole process, including the loaded data of 4 weeks; the configuration with a 365-day seed window and decay off, 115 seconds for the 4 weeks); the longer the seed window, the more seeds need neighbours and the higher the peak (3.9–4.6 GB for the other trials) | **Chunked by seed item**: each chunk keeps only the Top-`K_n` neighbours when done and is then released. The D phase used 2,000 items per chunk; 4.76 GB is too close to the 6 GB limit, so **M5 changed `chunking.item_chunk_size` from 2000 to 1000** (chunking affects only memory and speed, not results: chunked and unchunked `C_w` are bit-identical, tested). **Measured with the new chunk size (1000)** (the configuration locked at M5: 365-day seed window, decay on; D1): generate 25.8 seconds, true peak 3.72 GB at the end of generate and 4.20 GB for the whole process; the candidate file is byte-identical to the file generated with a chunk size of 2000 (see "Single-channel measurements of the M5 locked configuration on D1" below the table) |
| R5 WALS (180 days) | ~0.5–1 GB | **D1 (W2, default configuration): overall peak 3.10 GB** (maximum RSS from `/usr/bin/time`; sampled by MemoryTracker: fit 0.96 GB, scoring 2.11 GB); run time 63.8 seconds (fit 53.0 seconds, scoring 10.8 seconds). Matrix of 711,000 users × 52,000 articles with 5.9 million non-zero values. **The peak comes from scoring**, not from the fit: scoring is chunked by `chunking.dense_score_query_chunk_size` = 5,000 queries, and each chunk has a float32 score matrix of 5,000 × 20,764 active articles, about 0.41 GB, plus the copy made by `np.partition` (with a chunk of 10,000 the overall peak is 4.05 GB; Recall@100 is 11.36% with both chunk sizes, with the same number of candidate rows). The current macOS wheel of `implicit` is not linked against OpenMP, so the fit is in fact **single-threaded** (`num_threads=6` has no effect) | Sparse CSR; no problem |
| R6 Two-Tower training | ~1.5–2 GB | **Generating the training arrays (base window, 2026-11-17): peak 4.70 GB** (the more conservative value: two runs were sampled by MemoryTracker at 3.35 GB and 4.70 GB, and the task lasts only about 4.5 seconds, so the 0.5-second sampling interval has a large error; maximum RSS from `/usr/bin/time`: 4.15 GB, including reading the data), run time 4.6 seconds; 6.16 million samples, arrays of 670 MB in total (`history` int32 470 MB, `history_mask` 118 MB). **200-step benchmark (placeholder model, batch 512)**: maximum RSS of the MPS process 2.00 GB, MPS tensors 1.17 GB (the driver holds 2.02 GB in total; unified memory, partly overlapping with RSS); maximum RSS of the CPU process 2.34 GB. The real model (a full training run) is still to be measured | User histories are stored in advance as an int32 array `[N, 20]`; the item feature table (~105,000 × 896 dimensions, about 380 MB) is put into MPS memory as a whole; a batch is fetched directly by index, without a DataLoader |
| Candidate generation (per week) | At most about 39.4 million rows before merging, about 0.73 GB if all are kept in memory (about 1.07 GB for D3); at most about 0.15 GB for a single channel; about **25.2 million rows** after merging and deduplication (307 per query on average; calculation in ① below the table) | **Measured at M4 (D1–D4, 2026-11-17, the configuration locked at M5, i.e. R3 outputs only `A(a)`)**: before merging, D1 has 32.19 million rows and D3 49.47 million rows (about 0.36 / 0.55 GB when only the three columns `query_id`, `article_idx`, `rank` are kept); after deduplication each query has **307** candidates on average (P50 320, P90 364; at most 480), D1 about 24.40 million rows and D3 about 36.61 million rows. Before the switch of R3: 313 on average (P50 327, P90 373), with 32.68 million rows for D1 and 50.15 million rows for D3 before merging. **Memory**: the true peak of the whole process in the first M4 union check (four weeks processed together) was 6.66 GB (`/usr/bin/time`; the 0.5-second sampling saw only 5.22 GB), above 6 GB; after changing to one fold at a time (the re-run after the switch of R3) it fell to **4.78 GB** (source: the terminal output of `/usr/bin/time -l`, **not written to the run log**; the largest value sampled in the log is 4.72 GB). **This 4.78 GB does not include on-the-spot prediction by R6**: in that run the candidates of R1, R2, R4, R5 and R6 were all read back from existing files, and only R3 was generated on the spot. **The true peak of a single channel** (one process per channel, including on-the-spot prediction by R6) was measured in the D1 check of M5, see "Single-channel measurements of the M5 locked configuration on D1" below the table: the highest is the 4.20 GB of R4. **Measured at M6 (the 12 weeks F1–T4, 2026-11-17, `results/m6_runs.csv`)**: one process per (fold, channel), so there is no "six channels in one process" peak; the highest true peak of any process over the 12 weeks is **4.70 GB** (R6 in F2, including prediction, all item vectors and the fine-tuning of that week); ranges per channel: R1 1.9–2.1, R2 0.9–1.1, R3 1.0–1.1, R4 3.8–4.3, R5 3.6–4.2, R6 4.3–4.7 GB. The monitoring process (reads back the candidates of the six channels and counts the deduplicated candidates; for F and V weeks also computes recall) peaks at 2.12 GB at most. The deduplicated candidates per week average 304–310 (P50 317–324, P90 359–364), in line with the 307 of the D phase. The peak of merging into the wide table (`fusion/union.py`) is left to the second half of M6 | Each channel writes its own Parquet, then the channels are merged and deduplicated in chunks of queries |
| Wide feature table (per week, §8) | About **6.9 GB** for a whole week in memory (25.2 million rows × 292 B), which is not feasible; about **0.83 GB** per chunk of 10,000 queries, and about 0.99 GB for a P90 chunk (calculation in ④ below the table). The feature Parquet of 12 weeks (F, V, T) is about **77 GB** uncompressed, an estimated 23–46 GB compressed (calculation in ⑤ below the table). All of this is for 71 features (extended from 56 to 71 on 2026-11-17) | **Measured on F1 (2026-11-18, `results/m6_runs.csv`)**: 85,648 queries, 26,373,844 rows, 9 part files; run time 61 seconds (3.5 seconds for the three kinds of tables of the fold, 6.0 seconds for the `tt_sim_*` step). **True peak 6.10 GB, 0.10 GB above the 6 GB limit** (5.76 GB sampled; in agreement with `/usr/bin/time -l`). The RSS at the end of each chunk rises chunk by chunk from 3.80 GB to 4.95 GB, i.e. memory is not fully released between chunks; the candidates of the six channels for the whole week and the three kinds of tables of the fold stay in memory for the whole process, and how much each takes was not measured separately. The files were written completely and passed all checks; the user decided to leave the code unchanged (`docs/decisions.md` 2026-11-18). Disk **1.67 GB** (about 209 MB per part), see ⑤ below the table. **Measured on the 12 weeks (F1–T4, 2026-11-18, `results/m6_runs.csv`)**: 281,750,956 rows in total (20.98 million–26.37 million per week), 7–9 part files per week; run time 48–61 seconds per week, about 11 minutes for the 12 weeks together; true peak **6.10–7.25 GB** (highest: the 7.25 GB of V2), unrelated to the row count (F1, with the most rows, has the lowest peak), within the 8 GB limit; **17.8 GB** on disk in total. The feature-generation code has not changed since F1 | **Computed in chunks of queries** (10,000 queries per chunk); each chunk is written as one Parquet row group / part file as soon as it is computed, and the whole week is never held in memory |
| Ranker training (4 weeks) | About 11 million rows with negative sampling at 1:40 and about 21.7 million rows at 1:80 (an upper bound); raw float32 features about 2.9–5.7 GB, about 0.7–1.4 GB after binning (calculation in ② below the table; 71 features) | **Measured rows (2026-11-18, a read-only count, see §9.1)**: for a 4-week window, 10.45 million–10.71 million rows at r = 40, 15.30 million–15.69 million rows at r = 60 and 19.76 million–20.28 million rows at r = 80, with about 150,000 queries per window. **Measured in the speed test (2026-11-18, the window F1–F4 of V1, r = 40, 10.71 million rows, 154,815 queries; `results/m7_speed_test.csv`, `results/m7_runs.csv`)**: sampling 21–24 seconds per fold (r = 40 / 60 / 80 written in one go), true peak 3.92–4.13 GB, the sample files of the 4 folds 4.0 GB in total (about 91 B per row at r = 40); building the Dataset 6.6 seconds, true peak 2.00 GB, binary 644 MB; training 50 trees (not evaluated): col-wise 0.682 seconds/tree with a peak of 1.47 GB, row-wise 0.368 seconds/tree with a peak of 1.99 GB; training 300 trees row-wise takes 103.5 seconds in total (0.342 seconds/tree; 37.5, 34.8 and 31.2 seconds for each successive 100 trees) | **Stream loading**: the features of each week are stored as Parquet by row group and read in batches through LightGBM's `lgb.Sequence` to build the `Dataset`, so the raw features never have to be in memory all at once; the binned data after construction (1 byte per value) take about 0.5–1 GB and are stored as a LightGBM binary file for reuse |
| Ranker inference (per week, no sampling) | About **6.7 GB** all in memory at once; about **0.81 GB** per chunk of 10,000 queries, and about 0.96 GB for a P90 chunk (calculation in ③ below the table; 71 features) | **Measured in the speed test (2026-11-18, V1, 25.23 million rows × 300 trees, 31 leaves, one part scored at a time)**: scoring plus evaluation 86.4 seconds, about 10 seconds per part (about 3.05 million rows); the true peak of the whole trial (training plus scoring) is **5.69 GB**, occurring in the scoring stage (training itself takes about 2 GB). Afterwards scoring was changed to chunks of 2,500 queries (see the right column); the new peak is to be measured in the grid of step 1 | **Scored in chunks of queries**: after a part file (10,000 queries) is read in, it is scored in chunks of 2,500 queries (`scoring_query_chunk_size` in `configs/ranker.yaml`; made smaller than "one part per chunk" after the speed test of 2026-11-18). This affects memory only: the score of a row does not depend on the other rows of the chunk, a query is never split, and tests confirm that the scores and the Top-12 are bit-identical |

**How the rows candidate generation, ranker training, ranker inference and wide feature table were estimated** (recomputed on 2026-11-16 with the measured numbers of queries; recomputed on 2026-11-17 with the numbers of candidates measured at M4, and again the same day at the M5 lock with the 307 candidates after the switch of R3; 1 GB = 1024³ B). Common inputs: **82,000** queries per week on average (at most 120,000, in D3; 69,000–86,000 in the F, V and T weeks), about **3.4** labels per query on average (after deduplication; 3.2–3.6 in the F and V weeks); the quotas of the 6 channels add up to 100 + 30 + 50 + 100 + 100 + 100 = **480**; after merging and deduplication each query has **307** candidates on average, P50 320, P90 364 (measured at M4 on D1–D4, the configuration locked at M5, i.e. R3 outputs only `A(a)`; 313 / 327 / 373 before the switch of R3, and 250 assumed at first); **71** features (56 at first, 15 added on 2026-11-17, see §8). The measured queries of the 12 weeks F, V and T are 69,000–86,000 per week, **917,354** in total, 76,400 on average.

- ① **Candidate generation**: before merging, at most 82,000 × 480 ≈ 39.4 million rows (D3: 120,000 × 480 ≈ 57.6 million rows). About 20 B per row (`query_id`, `article_idx`, `rank` as int32, 4 B each; `score` as float32, 4 B; `channel` as Categorical, about 4 B) → 39.4 million × 20 B ≈ 0.73 GB (D3 ≈ 1.07 GB). Each channel is written to disk separately, and a single channel has at most 82,000 × 100 × 20 B ≈ 0.15 GB. After merging and deduplication, 82,000 × 307 ≈ **25.2 million rows** (25.7 million before the switch of R3, 20.5 million in the first estimate; D3: 120,000 × 305 ≈ 36.6 million rows).
- ② **Ranker training**: positives per week ≈ 82,000 queries × 3.4 labels × 24% (the union recall target of G2; M4 measured 25.2% on D1–D4, so the positives and the rows below are about 5% higher accordingly) ≈ 67,000. Each positive is paired with r negatives: at r = 40, 67,000 × 41 ≈ 2.74 million rows/week, 4 weeks ≈ **11 million rows**; at r = 80, 67,000 × 81 ≈ 5.42 million rows/week, 4 weeks ≈ **21.7 million rows** (when there are not enough candidates all negatives are taken, so this is an upper bound). Raw features 71 × 4 B = 284 B/row → 11 million × 284 B ≈ 2.9 GB and 21.7 million × 284 B ≈ 5.7 GB; after binning, 1 B per value → 11 million × 71 B ≈ 0.73 GB and 21.7 million × 71 B ≈ 1.4 GB. At r = 80 the raw features are already close to the 6 GB limit, so stream loading is a must. (With 56 features these were 2.3 / 4.5 GB and 0.57 / 1.1 GB.)
- ③ **Ranker inference**: per week 82,000 × 307 = 25.2 million rows × 284 B ≈ 7.16 × 10⁹ B ≈ **6.7 GB** (5.3 GB with 56 features, 4.3 GB in the first estimate; the V and T weeks have at most 83,000 queries, about the same). In chunks of 10,000 queries: 10,000 × 307 × 284 B ≈ 8.7 × 10⁸ B ≈ **0.81 GB/chunk** (0.64 GB with 56 features); for a chunk with many candidates, taking P90: 10,000 × 364 × 284 B ≈ 0.96 GB/chunk.
- ④ **Wide feature table**: each row has 71 features at 284 B as float32, plus the two int32 columns `query_id` and `article_idx`, 8 B together → **292 B/row** (slightly less as actually stored: the 8 categorical features are int16, 276 B per row). A week of 25.2 million rows × 292 B ≈ 7.36 × 10⁹ B ≈ 6.9 GB exceeds the 6 GB limit, so a whole week cannot be held in memory. In chunks of 10,000 queries: 10,000 × 307 × 292 B ≈ 9.0 × 10⁸ B ≈ **0.83 GB/chunk**; taking P90: 10,000 × 364 × 292 B ≈ 0.99 GB/chunk. (With 56 features: 232 B/row, 5.4 GB, 0.66 / 0.79 GB.)
- ⑤ **Disk usage of the feature Parquet of 12 weeks**: F1–F4, V1–V4 and T1–T4 have 917,354 queries × 307 ≈ **282 million rows**; uncompressed, 282 million × 292 B ≈ 8.23 × 10¹⁰ B ≈ **77 GB** (about 6.4 GB per week on average; 61 GB with 56 features). The Parquet compression ratio can only be estimated for now: the candidate files (IDs, rank, score) compress to about 18% of the raw size (the 125 million rows of 5 channels and 4 weeks in the pre-check take 449 MB), and floating-point features compress worse; at 30%–60% the estimate is **about 23–46 GB**. After M6 generates F1, the size of one week is measured first and then extrapolated. The disk currently has about 520 GB free, which is enough. **Measured on F1 (2026-11-18)**: 26,373,844 rows and 1.67 GB of Parquet in total, i.e. about 68 B per row, about 23%–25% of the uncompressed size (26,373,844 × 276 B ≈ 6.8 GB), a better compression than the 30%–60% estimated above. Extrapolated to 282 million rows, the 12 weeks take about **18 GB**. **Measured on the 12 weeks**: 281,750,956 rows and 19,136,477,799 bytes ≈ **17.8 GB** in total, about 68 B per row, in line with the extrapolation.
- Note: 307 is the value measured when the union check was re-run after R3 was changed to output only `A(a)` (`results/m4_union_check_r3_active_pool.csv`, 307.1 for the 4 weeks pooled). Before the switch it was 313: the candidates of R3 outside `A(a)` were about 23% of R3's rows (about 7 rows per query on average), and after the switch part of these slots are filled by same-style articles inside the pool, so the rows of R3 per query fall from 29.8 to 24.1 on average, and there are 5.8 fewer per query after merging and deduplication.

Everything above except the "measured" column is an estimate; the "measured" column continues to be filled in as each phase is measured. The M0 query / label generation was measured on 2026-11-16 (`scripts/m0_build_queries.py`). Measurements of the M0 preprocessing (2026-11-15): the memory held is the Polars `estimated_size()` after reading the Parquet back; the peak is the RSS sampled by `MemoryTracker` (interval 0.5 seconds; a lower bound, see the next paragraph). Memory held by the other M0 outputs: customers 37 MB, articles 3 MB, customer ID map 89 MB, the image / text embeddings aligned by `article_idx` 206 MB / 155 MB.

**Single-channel measurements of the M5 locked configuration on D1** (2026-11-17, `scripts/m5_verify_locks.py`, HEAD `0ce824a`, clean working tree; one process per channel; `results/m5_lock_check.csv`). D1 has 78,193 queries and an active pool of 20,764 articles. The scores of the six channels are all bit-identical to the records, and the candidate files are byte-identical to the files generated in the D phase / M4.

| Channel | Candidate rows | generate time | Whole process time | True peak: up to the end of generate | True peak: whole process | Sampled peak (generate) |
|---|---|---|---|---|---|---|
| R1 Popularity | 7,819,300 | 0.5 seconds | 3.8 seconds | 1.98 GB | **1.98 GB** | 1.05 GB |
| R2 Repurchase | 1,292,785 | 0.3 seconds | 3.1 seconds | 0.99 GB | **0.99 GB** | 0.77 GB |
| R3 Sibling | 1,860,095 | 0.4 seconds | 3.2 seconds | 1.06 GB | **1.06 GB** | 0.76 GB |
| R4 ItemCF (`item_chunk_size` 1000) | 7,001,322 | 25.8 seconds | 29.1 seconds | 3.72 GB | **4.20 GB** | 3.72 GB |
| R5 WALS (factors 128) | 6,398,300 | 169.6 seconds | 173.1 seconds | 2.72 GB | **3.78 GB** | 2.61 GB |
| R6 Two-Tower (base epoch 4, predicting on the spot, no training) | 7,819,300 | 13.6 seconds | 17.4 seconds | 2.33 GB | **3.53 GB** | 2.33 GB |

- The "true peak" is `ru_maxrss` (`process_max_rss_gb`). "Up to the end of generate" is the peak from process start (including reading queries / labels, computing the active pool, and loading the model for R6) until `generate` returns; "whole process" also includes the validation, the Parquet write and the evaluation that follow. The "whole process" column of the six channels is **byte for byte equal** to the maximum resident set size of `/usr/bin/time -l`.
- All six channels are within 6 GB; the highest is the 4.20 GB of R4.
- For R4, R5 and R6 the peak rises by another 0.48 / 1.06 / 1.20 GB **after** `generate` returns, from validating, writing and evaluating the candidates (the three steps were not measured separately this time). In M6 every channel validates and writes in the same way, so the "whole process" column, not "up to the end of generate", should be used to estimate the single-channel peak of M6.
- The sampled peak misses the spikes of short tasks: R1 is sampled at 1.05 GB while the true value is 1.98 GB.
- R4 has no true peak of the same configuration and the same fold with a chunk size of 2000 to compare with (the earlier 2.95 GB of D1 is the default configuration, and the 3.9–4.8 GB of the D phase are sampled values with four weeks in one process), so no number for "how much halving the chunk saved" is given here.

**How memory peaks are measured**: `src/hmrec/memory.py` records two kinds of peak, with different field names, which answer different questions:

- **Sampled peak** (`MemoryTracker`, `peak_rss` / `sampled_peak_rss_gb`): a background thread samples the RSS at intervals with psutil (interval in `memory.sample_interval_seconds` of `configs/base.yaml`) and takes the maximum of all samples and of the readings at the start and at the end. It can be measured **per stage** (one tracker per stage) and is used to locate where memory goes; but a transient peak shorter than the sampling interval is not seen, so it is only a **lower bound**.
- **True peak** (`read_process_peak_memory`, `process_max_rss_gb`): the standard library's `resource.getrusage(RUSAGE_SELF).ru_maxrss`, the maximum RSS since the process started as recorded by the operating system; it does not miss transient peaks and reads the same counter as `/usr/bin/time -l`. But it is **one number for the whole process** and cannot be reset, so it cannot say in which stage the peak occurred. The unit depends on the system (bytes on macOS, KB on Linux) and is converted by `ru_maxrss_to_bytes`. Child processes do not count towards the RSS of the process itself; `RUSAGE_CHILDREN` is recorded separately (the largest finished child process, not the sum).

Whether the limit (12 GB; 6 GB before 2026-11-18, changed that day to 7 GB, 8 GB and 12 GB in turn) is exceeded is **judged by the true peak**; the sampled peak is used to find which stage caused it. Every `MemoryTracker` writes both numbers to the log when it ends (if the true peak grows between the logs of two adjacent stages, the spike occurred in the later stage); a script calls `log_process_peak_memory` as its last step and writes the true peak into the result record of the run (the lock check of M5 writes to `results/m5_lock_check.csv`). **Numbers from before 2026-11-17 (M5)**: those in the table above marked "sampled" or MemoryTracker are sampled peaks (lower bounds); those marked `/usr/bin/time` are true peaks, but they appear only in terminal output and were not written to the run log. The unified memory taken by MPS tensors is not necessarily all counted in RSS, which affects both kinds of peak.

### 1.4 Speed estimate for Two-Tower (M1 Max)

| Step | Estimated time | Measured |
|---|---|---|
| 1 epoch of the base (26 weeks, about 7–8 million samples, batch 512) | About 2–5 minutes | Placeholder model: about 9.2 minutes on MPS, about 7.5 minutes on CPU. **Real model (2026-11-17, 200 steps): CPU 30.4 ms per step (P90 31.1) → about 6.1 minutes / epoch, about 37 minutes for 6 epochs; MPS 45.2 ms per step (P90 54.4) → about 9.1 minutes / epoch** (6.16 million samples = 12,033 steps; about 11,900 steps in practice without the 1% alarm users). CPU is chosen |
| Weekly fine-tuning (about 300,000 samples) | About 10 seconds – 1 minute | **Measured at M6 (the 11 weeks F1–T3, CPU, E_update 2, replay 50%): 52–71 seconds** (230,000–310,000 new samples per week, plus as many replay samples, 460,000–610,000 rows in total, 1,792–2,396 steps). The whole R6 process (checking the sha256, loading the model, predicting, saving the vectors, fine-tuning) takes 74–95 seconds per week, of which predicting and saving the vectors take 15–19 seconds; T4 only predicts, 18 seconds |
| Weekly inference (about 82,000 queries on average, at most 120,000 (D3) × about 100,000 articles, taking the Top-100) | A few seconds to a dozen or so (chunked matrix multiplication + `torch.topk`) | **D1 (2026-11-17, M5 lock check, CPU): 13.6 seconds** (78,193 queries × 20,764 active articles, including user histories, encoding by both towers and taking the Top-100); plus 0.4 seconds for checking the sha256 and loading the model and tables. True peak 2.33 GB at the end of generate |

**Benchmark before starting**: run 200 steps each on `mps` and on `cpu`, recording the time for fetching data and the time for forward + backward separately (calling `torch.mps.synchronize()` before and after timing). If fetching data takes more than half of the time, optimize the data pipeline first.

**Measured benchmark (2026-11-17, `scripts/m0_benchmark_twotower.py`, torch 2.15.0)**: with a placeholder model (`src/hmrec/recall/twotower/benchmark.py`: image / text projections + ID and 7 category embeddings, a user tower with a masked mean over 20 history items plus profile embeddings, multi-positive in-batch softmax, no LogQ; not the real R6 model), the arrays of the base window, batch 512, 200 steps, a separate process per device.

| Device | Fetching a batch (per step, first 10 steps dropped) | forward + backward + update (per step) | Share of fetching a batch | First 10 steps in total | 200 steps in total |
|---|---|---|---|---|---|
| MPS | 0.80 ms | 45.0 ms (median 44.5, P90 47.8) | **1.7%** | 1.84 seconds | 10.5 seconds |
| CPU (6 threads) | 0.26 ms | 37.1 ms (median 37.1, P90 37.8) | **0.7%** | 0.43 seconds | 7.5 seconds |

- Fetching a batch is far below half, so **the data pipeline needs no optimization**.
- The loss of the two devices agrees step by step (7.860 → 6.388), which shows the computation is the same.
- The placeholder model is **about 20% slower on MPS than on CPU**. Its matrices per step are small, and the kernel start-up and the backward pass of embeddings (scatter) on MPS dominate the cost; with a larger real model the conclusion could differ. Before the real training, the two devices were compared again with the real model: **CPU is still faster** (30.4 ms per step for the real model vs 45.2 ms on MPS, with fetching a batch at 0.9% / 1.2%), and R6 trains on CPU (§1.2).

---

## 2. Data and general conventions

### 2.1 Raw data

| File | Content |
|---|---|
| `transactions_train.csv` | `t_dat, customer_id, article_id, price, sales_channel_id`; 2018-09-20 to 2020-09-22. Fully identical duplicate rows mean several purchases and are **kept** |
| `customers.csv` | `customer_id, FN, Active, club_member_status, fashion_news_frequency, age, postal_code` |
| `articles.csv` | `article_id, product_code` and metadata such as the category levels, colours and descriptions |
| Precomputed embeddings | For every article, a CLIP ViT-B/32 image vector (512 dimensions) and an all-MiniLM-L6-v2 text vector (384 dimensions). Read-only; not extracted again. See "Precomputed embeddings" below |

`article_id` keeps its leading zeros, is handled throughout as a 10-character string, and is mapped internally to int32.

**Precomputed embeddings** (result of a read-only inspection on 2026-11-15, see `docs/decisions.md`):

| | Image | Text |
|---|---|---|
| Model | CLIP ViT-B/32 | all-MiniLM-L6-v2 |
| Vector file | `data/processed/image_features/image_embeddings.npy` | `data/processed/text_features/text_embeddings.npy` |
| shape / dtype | `(105100, 512)` float32 | `(105542, 384)` float32 |
| id table | `image_article_ids.parquet` (`embedding_row, article_id, image_path`) | `text_article_ids.parquet` (`embedding_row, article_id, text, token_count, was_truncated`) |
| Coverage | 105,100 / 105,542 articles; **442 articles have no image vector** | All 105,542 articles |

- **Correspondence**: row `i` of the `.npy` corresponds to the `article_id` (10-character string) with `embedding_row = i` in the id table. M0 then maps it to `article_idx` (int32). It may not be assumed that the row number of the `.npy` equals `article_idx`.
- **Already L2-normalized**: the dot product of two vectors is their cosine similarity, and no further normalization is needed when using them.
- The `*_faiss.index` files in the directory are **not used** (the project does not depend on faiss). The manifest json in the directory records the model version and sha256 and is for reference only.

### 2.2 Time conventions

- The anchor `a` is the first day of the predicted week. **Inputs**: `t_dat < a`. **Labels**: `t_dat ∈ [a, a+7d)`.
- Event age: `age_days = (a − t_dat).days`, so the day before the anchor has age 1.
- Every exponential decay is written as `exp(−age_days / τ)`, where τ is a **time constant**, not a half-life (half-life = τ × ln2).
- **The date column of the processed data**: `data/processed/transactions.parquet` stores no date but the int16 column **`t_dat_day`** = the number of days since `day_zero` (`configs/preprocess.yaml`, 2018-09-20), in the range 0..733. With `anchor_day = date_to_day_index(a, day_zero)`, the `t_dat < a` above is written in code as `t_dat_day < anchor_day`, and `age_days` as `anchor_day − t_dat_day`. Conversion between dates and day numbers may **only** call `src/hmrec/data/day_index.py` (`date_to_day_index` / `day_index_to_date`, and the corresponding Polars expression versions `*_expr`); the formula may not be written anywhere else.

### 2.3 Queries and labels

- **Query** = `(customer_id, anchor)`, provided the user bought at least one catalog article in the label week. There are **69,000–120,000** queries per week, **82,000** on average (measured on the 16 folds on 2026-11-16, see §1.3; D3 and D4 are larger, see §12).
- **query_id** (int32, unique across all folds) = `fold_ordinal × query_id_fold_offset + position`. `fold_ordinal` is the position of the fold in `fold_manifest.yaml` (D1 = 0 … T4 = 15), `position` is the index of the query within the fold in ascending `customer_idx` (starting at 0), and `query_id_fold_offset = 10,000,000` (`configs/queries.yaml`). Tables of several folds can be joined on `query_id` alone without mismatching. The reverse lookup is `hmrec.data.queries.decode_query_id`.
- **Label set `G_q`** = the **deduplicated** set of `article_id` the user bought in the label week.
- If a channel outputs nothing for a query, this counts as a miss of that channel; **the query may not be removed from the denominator**.

### 2.4 Article scope

- **Catalog `C`**: all articles in `articles.csv`. Labels are not filtered by any article pool.
- **Active pool `A(a)`**: the articles with at least 3 purchase records in `[a−28d, a)`. It approximates "currently on sale"; it is not real inventory.
- **Which channels are filtered by the active pool**: the outputs of R1, R3, R4, R5 and R6 are limited to `A(a)`; R2 (repurchase) is not filtered, because it is based on the user's own history anyway. **R3 (same style, other colour) was not filtered at first either; on 2026-11-17, at M4, it was changed under a pre-registered rule to output only articles inside `A(a)`** (a one-off change, see the next sentence and `docs/decisions.md`): removing the R3 candidates outside `A(a)` lowers the 4-week pooled six-channel union recall by only 0.045 percentage points (< 0.1 percentage points). The filter comes before taking the Top-50 (as for R4), and the freed slots are filled by same-style articles inside the active pool; γ and τ are not re-tuned, and the D-phase selection (γ 1.0, τ 30) is kept. The D-phase HPO of R3 and the R3 records in `results/d_phase.csv` were all obtained without the filter. This setting was frozen at M0 and is **not tuned as a hyperparameter**. The only exception: whether R3 is changed to output only articles inside `A(a)` is decided once at M4 by the pre-registered criterion of §5.7 item 4 (a system-level decision, not D-phase HPO).
- Articles a user has bought before **may be recommended again** (repurchase is common at H&M).

### 2.5 Metrics

All are "the score of each query first, then the mean" (macro).

| Metric | Definition |
|---|---|
| Recall@K | `|Top-K ∩ G_q| / |G_q|`. The denominator is the complete `|G_q|`, **not cut off at K**, so for a query with more than K labels Recall@K is at most `K / |G_q|` |
| AP@K / MAP@12 | Let position r among the first K be a hit, and `P(r)` = the number of hits in the first r positions ÷ r. `AP@K = Σ P(r)` (summed over every hit position r among the first K) `÷ min(K, |G_q|)`. With K = 12 it is the definition of the Kaggle H&M competition; MAP@12 is the mean of AP@12 over queries |
| NDCG@K | Binary relevance. `DCG@K = Σ 1 / log2(r + 1)`, summed only over the hit positions r among the first K. **The number of hits of the ideal DCG is `min(K, |G_q|)`**: `IDCG@K = Σ_{r=1..min(K, |G_q|)} 1 / log2(r + 1)`. `NDCG@K = DCG@K / IDCG@K`, and a perfect ranking (all of the first `min(K, |G_q|)` positions are hits) scores 1.0 |
| Union recall | The share of `G_q` covered by all candidates after the 6 channels are merged and deduplicated: `|candidates ∩ G_q| / |G_q|`, with no cut-off; an article recalled by several channels counts once |

Conventions that hold for all of these metrics (`src/hmrec/eval/metrics.py`):

- **`G_q` is everything the user actually bought in the label week (deduplicated, §2.3), not only the recalled part.** A label that no channel recalled still counts in `|G_q|`, so it lowers Recall, the denominator term of AP and the ideal DCG of NDCG. The NDCG row used to say "the ideal DCG is computed from the complete `G_q`", which meant exactly this; the cut-off is implied by "@K". The former wording could be read as "the ideal DCG is not cut off at K" and was changed to the wording above on 2026-11-18 (`docs/decisions.md`); neither the code nor the values recorded before changed.
- The queries averaged over are all queries of the label table. A query with no prediction, or with no hit, scores 0 and **stays in the denominator** (§2.3).
- When fewer than K items are output, the missing positions count as "no hit" and are not padded.
- The position is determined by `rank` (1 = best): the `rank` of each query must be 1, 2, …, n, with no repeated article within a query; a query in the predictions that is not in the label table is an error.
- The 4-week pooled score is the mean over the queries of the 4 weeks together (§4.1).

---

## 3. Calendar (fold manifest)

| Phase | Fold | Anchor | Label dates (inclusive) |
|---|---|---|---|
| Recall HPO | D1 | 2020-06-03 | 06-03 – 06-09 |
| Recall HPO | D2 | 2020-06-10 | 06-10 – 06-16 |
| Recall HPO | D3 | 2020-06-17 | 06-17 – 06-23 |
| Recall HPO | D4 | 2020-06-24 | 06-24 – 06-30 |
| Ranker Train | F1 | 2020-07-01 | 07-01 – 07-07 |
| Ranker Train | F2 | 2020-07-08 | 07-08 – 07-14 |
| Ranker Train | F3 | 2020-07-15 | 07-15 – 07-21 |
| Ranker Train | F4 | 2020-07-22 | 07-22 – 07-28 |
| Validation | V1 | 2020-07-29 | 07-29 – 08-04 |
| Validation | V2 | 2020-08-05 | 08-05 – 08-11 |
| Validation | V3 | 2020-08-12 | 08-12 – 08-18 |
| Validation | V4 | 2020-08-19 | 08-19 – 08-25 |
| Test | T1 | 2020-08-26 | 08-26 – 09-01 |
| Test | T2 | 2020-09-02 | 09-02 – 09-08 |
| Test | T3 | 2020-09-09 | 09-09 – 09-15 |
| Test | T4 | 2020-09-16 | 09-16 – 09-22 |

Stored as `configs/fold_manifest.yaml` and validated by a script: 7 days per week, adjacent, non-overlapping, and the last day is 2020-09-22.

**D1–D4 are not used to train the ranker**: the candidates of these weeks are generated with "the configuration selected on these very weeks" and carry selection bias.

---

## 4. General selection rule (all HPO and ablations)

### 4.1 One score from 4 weeks

- Main metric: **query-weighted** (the mean over the queries of the 4 weeks together).
- The equal-fold score (the plain mean of the 4 weekly scores) and the weekly scores are reported as well.
- **Grouped diagnostic of the D phase**: in the D-phase reports, besides the 4-week pooled score, the scores of **D1–D2** and of **D3–D4** are reported separately (the mean over the queries of the two weeks together, i.e. query-weighted). **A diagnostic only, not used for selection**; the selection rule is still §4.2 and `configs/selection_rule.yaml`. Reason: D3 and D4 look like promotion weeks and carry about 60% of the weight in the 4-week pooled score (see §12); the group scores serve to spot configurations that are "better only in promotion weeks" or "better only in ordinary weeks".

Selection metric of each component:

| Component | Selection metric |
|---|---|
| R1, R4, R5, R6 | Recall@100 |
| R2 | Recall@30 |
| R3 | Recall@50 |
| Ranker | MAP@12 |

Other metrics (Recall@12/20, NDCG@12/20, MAP@12) are reported as usual but not used for selection.

### 4.2 When a change is accepted

The default is replaced by a new variant or hyperparameter only if both conditions below hold; otherwise **the default is kept**:

1. the 4-week pooled score **improves by ≥ 1% relative**;
2. it is better in **at least 3 of the 4 weeks**. "Better" here means **strictly better** (new score > default score); **equal scores do not count as better**.

When two candidates are about the same, the **simpler and cheaper** one is chosen. This rule is written into `configs/selection_rule.yaml` and **committed before any score is seen**.

**When a step has several candidates** (for example, the τ of R1 has 6 non-default values), the implementation is in `src/hmrec/hpo/selection.py` (`selection_rule.yaml` is not changed):

1. Each candidate is compared with the **current default** (the configuration selected by the previous step) under the two conditions above, giving "passes / does not pass".
2. If no candidate passes → the default is kept.
3. If some pass → among the passing candidates, those whose **relative gap** to the highest pooled score is **< 1%** (the same threshold as in condition 1) count as "about the same", and the one that comes **first in the priority order** is chosen. So if the highest-scoring candidate does not pass while another candidate does, the passing one is chosen.
4. The priority order is written in `configs/hpo/d_phase.yaml` before the run and may not be modified after it:
   - Where candidates differ in simplicity / cost (variants, windows, switches, dimensions): the listed order, simpler / cheaper / closer to the default first. For example G0 < A1, recency-only < decayed frequency, CF0 < CF3, W0 < W2, factors 64 < 128; the seed grid of R4 is (90, off) [default] < (90, on) < (180, off) < (180, on) < (365, off) < (365, on).
   - Purely numeric parameters with no difference in cost (τ, γ, α, λ, α_conf): ordered by distance from the **current default**, the smaller value first at equal distance. That is, among the "about the same" candidates the one closest to the current default is chosen, not simply the one with the highest score.
5. **The iterations {10, 15, 25} of R5 do not use this rule** and are handled by the convergence check of §5 R5: it changes to 25 only if 25 passes conditions 1 and 2 (against 15). If the score drops clearly after 15 (the pooled score of 25 is ≥ 1% below that of 15 and at least 3 weeks are strictly worse), or if 10 passes conditions 1 and 2 (against 15), λ may be too small: **iterations are not changed automatically; stop and report**, and a person decides whether to go back to the λ step.
6. **Grid boundaries**: grids are **never extended automatically**. If, in a step of R4 or R5 with a single numeric parameter (the α of R4; the λ of R5), the selected value falls on the minimum or maximum of the step's range, the runner stops that channel (`stop_at_boundary`) and a person decides whether to widen the range; the report states which value was selected. Other numeric steps (the τ of R1 and R2; the γ and τ of R3) continue as usual when a boundary value is selected and only note it in the report. The α_conf of R5 has already been extended once ({10, 20, 40} → {10, 20, 40, 60, 80}), and **each parameter is extended at most once**, so it does not stop any more; the factors of R5 have only the two values {64, 128}, and if 128 passes it is accepted, without extending to 256 and without this check (docs/decisions.md 2026-11-17). `stop_at_boundary` triggers only when a step **switches to** a boundary value and does not support multi-parameter steps (such as the seed window × decay of R4); these cases are checked by hand after each channel's run.

The configuration selected by one step is the default of the next, so the runner applies this rule automatically in the order of §4.3; the final selection is confirmed by a person before it is written to `configs/locks/`. The report of every step lists all candidates: whether each passes, its score in each of the 4 weeks, the pooled score, the group scores D1–D2 / D3–D4, which one is finally selected and why.

### 4.3 Order

Every component does the ablation first (selecting the variant) and then HPO on the selected variant. HPO changes one parameter at a time, with the other parameters kept at their current best values.

### 4.4 Records

Every trial records: component, variant, hyperparameters, the score of each of the 4 weeks, the pooled score, the number of queries, run time, memory peak, code version (git commit). All results are collected in `results/d_phase.csv` and `results/v_phase.csv`.

---

## 5. The six recall channels: definitions, ablations, HPO

**Every trial is recomputed in each week of D1–D4**: statistics are recomputed, or models refitted, from the data before that week's anchor. Fitting once and then evaluating 4 weeks is not allowed.

**Final configuration (M5, locked on 2026-11-17)**: the final configurations of the six channels are written in `configs/locks/channels.lock.yaml` (R1–R5) and `configs/locks/twotower.lock.yaml` (R6), and given in the last row ("locked at M5") of each channel's table below. The scores in the tables are the 4-week pooled score of that trial on D1–D4, **for the record only**, and optimistically biased (§12 item 4). **From M6 on, the six channels may be built only from the lock files through `src/hmrec/recall/locked.py`**; the defaults of `configs/recall/` and `results/d_phase.csv` are no longer read (a test scans the code of M6 and later).

Unified output format (one table per channel): `query_id, article_idx, channel, rank, score`. Within a query `article_idx` does not repeat; rows are ordered by `score DESC, article_idx ASC`; `rank` is numbered consecutively from 1; nothing is padded when the quota is not reached.

**Output spec of each channel** (`OutputSpec` in `src/hmrec/recall/base.py`): each channel declares its extra columns (their names must start with the channel's prefix, such as `pop_`, and nulls are not allowed) and its sort key (default `score DESC, article_idx ASC`). The validation before writing to disk goes both ways: an undeclared extra column, a missing declared column and a dtype mismatch all raise an error; and `rank` is checked against the declared sort key.

**The ordering exception of R1**: R1 has the extra column `pop_is_fallback` (int8), and its sort key is `pop_is_fallback ASC, score DESC, article_idx ASC`. Reason: rows from the age bucket use the bucket score `S(g, i)` and fallback rows use the global score `S(i)`; the two are on different scales, and the score of a fallback row may be larger than that of a bucket row; ordering by `score DESC` would put fallback rows before bucket rows. So the `score` of R1 (i.e. the `pop_score` feature) mixes two scales, told apart by `pop_is_fallback`; `pop_bucket_score` of group B (§8.3) gives every candidate the single scale `S(g, i)`.

`article_idx` (int32) is the number of each `article_id` after all `article_id` of `articles.csv` are sorted as strings (0..N−1, see `src/hmrec/data/id_maps.py`), so ordering by `article_idx ASC` is exactly equivalent to ordering by `article_id ASC`, and the result of the tie-break is unchanged.

### R1 — Age-Bucketed Decayed Popularity

**Definition**: take all purchase lines in `[a−7d, a)`. For an age bucket `g` and an article `i`:

`S(g, i) = Σ exp(−age_days / τ)`, summed over every purchase line in which a user of age bucket `g` buys `i`.

- Age buckets (frozen; **not a hyperparameter**):

| Bucket | Rule |
|---|---|
| `AGE_18_24` … `AGE_45_54` | 18–24, 25–34, 35–44, 45–54 |
| `AGE_55_PLUS` | 55–99 |
| `AGE_INVALID` | age < 18 or > 99 (under 18 and outliers in one bucket of their own) |
| `AGE_UNKNOWN` | age missing |

- Each user gets the Top-100 of their own age bucket; when there are fewer than 100, the list is filled in order from the global decayed popularity, skipping the articles already chosen, with `pop_is_fallback=1`. The global decayed popularity is `S(i) = Σ exp(−age_days / τ)`, summed over the purchase lines of **all** users buying `i` in `[a−7d, a)`, with the same τ as in the buckets; fallback rows come after bucket rows (see the ordering exception at the start of this section). The number of fallback rows per fold and per age bucket is written to the log and the results as a diagnostic.
- **G0**: one global list for all users, scored by the raw number of purchase lines in `[a−7d, a)` (no decay, no buckets), with no fallback rows.
- Default: τ = 3 days, quota 100, limited to the active pool `A(a)`.

| Type | Content | Configurations |
|---|---|---|
| Ablation | **G0** (global, no buckets, raw 7-day count) vs **A1** (τ=3) | 2 |
| HPO | `τ ∈ {1, 2, 3, 4, 5, 6, 7}` | +6 |
| **Locked at M5** | **A1, τ = 3** (no candidate passed the rule, the default configuration is kept), window 7 days, quota 100. Selected by the rule; trial `d2094d455698`; Recall@100 = 7.388% | — |

**8 configurations in total, a matter of minutes.**

### R2 — Decayed-Frequency Repurchase

**Definition**: recommend only the same `article_id` the user bought in `[a−180d, a)`:

`S(u, i) = Σ exp(−age_days / τ)`, summed over every purchase line in which user `u` buys `i` (duplicate lines each count once).

- Default: τ = 30 days, quota at most 30 (soft, not padded). The output is empty for users without purchase history.

| Type | Content | Configurations |
|---|---|---|
| Ablation | **Recency-only** (ordered by the time of the most recent purchase) vs **Decayed frequency** (τ=30) | 2 |
| HPO step 1 | `τ ∈ {14, 30, 60}` | +2 |
| HPO step 2 | History window `lookback_days ∈ {180, 365}` (default 180; priority order 180 < 365, shorter is simpler). **Added after seeing the D-phase results of R1–R3**, see `docs/decisions.md` | +1 |
| **Locked at M5** | **Decayed frequency, τ = 30, history window 365 days**, quota 30. Selected by the rule; trial `f562d598b0d6`; Recall@30 = 4.266% | — |

The score of Recency-only = `1 / age_days`, where `age_days` is the age of the user's most recent purchase of the article (yesterday = 1.0, the day before = 0.5); earlier purchases and the number of purchases take no part. Several articles bought on the same day score the same and are ordered only by `article_idx ASC`, with no other information used to break ties, so that it stays a pure control.

**5 configurations in total.**

### R3 — Sibling (same style, other colour; S2)

**Definition**:
- Style preference: `P(u, c) = Σ exp(−age_days / τ)`, summed over every purchase line of user `u` in `[a−180d, a)` with `product_code = c`.
- Market popularity: `M(j)` = the number of site-wide purchase lines of article `j` in `[a−7d, a)`.
- Score: `S(u, j) = P(u, c(j)) × (1 + M(j))^γ`.
- Candidates: the same `product_code` as an article the user bought within 180 days, **excluding** the very `article_id` the user bought within 180 days.
- Default: τ = 30, γ = 0.5, quota at most 50 (not padded).

| Type | Content | Configurations |
|---|---|---|
| Ablation | `γ ∈ {0, 0.5, 1.0, 1.5, 2.0}` (γ=0 means style preference only, to test whether market popularity helps; 1.5 and 2.0 were **added after seeing the D-phase results of R1–R3**, see `docs/decisions.md`; the priority order is by distance from the default 0.5) | 5 |
| HPO | With the best γ, tune `τ ∈ {14, 30, 60}` | +2 |
| **Locked at M5** | **γ = 1.0, τ = 30, outputs only articles inside `A(a)`** (`active_pool_only: true`), style window 180 days, market window 7 days, quota 50. Execution of a rule: the D-phase rule selected γ = 1.0, τ = 30 (trial `3188358fa459`, Recall@50 = 3.257%), and the pre-registered M4 rule then changed it to output only `A(a)`; what is locked is the trial after the switch, `06025cc3d129` (`results/m4_trials.csv`), Recall@50 = 3.240% | — |

**7 configurations in total.** If γ = 2.0 (the boundary of the new grid) is selected, the grid is not extended further and this is only noted in the report.

**Active pool** (2026-11-17, M4): R3 outputs only articles inside `A(a)` (`active_pool_only` in `configs/recall/r3_sibling.yaml`, turned on in `variants.r3_active_pool` of `configs/m4_union_check.yaml`; M5 locks it as `true` in `configs/locks/channels.lock.yaml`). The filter comes before taking the Top-50, and the freed slots are filled by same-style articles inside the pool; `P(u, c)`, `M(j)` and the exclusion of bought articles still use all articles. The ablation and the HPO of this section were all done without the filter, and γ and τ are not re-tuned (§2.4, §5.7 item 4).

### R4 — ItemCF

**Definition** (fit window `[a−365d, a)`; interactions between users and articles count only as 0/1):
- User weight (IUF): `w_u = 1 / ln(1 + d_u)`, where `d_u` = the number of distinct articles the user bought in the window.
- Weighted co-occurrence: `C_w(i, j) = Σ_u w_u · R_ui · R_uj`; weighted marginal: `n_w(i) = Σ_u w_u · R_ui`.
- Similarity (from seed `i` to candidate `j`): `s(i→j) = C_w(i, j) / (n_w(i)^α · n_w(j)^(1−α))`. `i = j` is removed.
- Each seed keeps only its Top-`K_n` (=100) neighbours. **The neighbours `j` are restricted to `A(a)`, and the filter comes before taking the Top-`K_n`**: what each seed keeps are its best `K_n` **active** neighbours. `C_w`, `n_w` and `d_u` are computed from all articles of the fit window and are not affected by `A(a)`; the seed itself need not be in `A(a)` (inactive seeds contribute to the score all the same). Reason: on D1 the graph of the fit window has 71,713 articles, of which only 20,764 (29%) are in `A(a)`; if the Top-100 were taken first and filtered afterwards, a measured **45.4%** of the neighbour slots would fall on inactive articles and be wasted (D1, CF3, α=0.7).
- User score: `S(u, j) = Σ_i q_ui · s(i→j)`, summed over every distinct seed `i` the user bought in the seed window `[a−L_seed, a)`.
- **Seed weight** `q_ui`: with decay off, `q_ui = 1`; with it on, `q_ui = exp(−age_ui / τ_seed)`, where `age_ui` is the number of days from the user's **most recent** purchase of seed `i` to the anchor (yesterday = 1; a seed bought several times takes only the most recent purchase, with no accumulation). `τ_seed` is fixed at 30 days (the same as the default τ of R2 and R3) and **not tuned**. Building the graph (`C_w`, `n_w`, `d_u`) stays without decay.
- Default: CF3 (IUF + α=0.7), `L_seed` = 90 days, seed decay off, quota 100, articles the user already bought are not excluded.

| Type | Content | Configurations |
|---|---|---|
| Ablation | **CF0** (unweighted, α = 0.5, i.e. plain cosine `s = C / sqrt(n(i)·n(j))`) vs **CF3** | 2 |
| HPO step 1 | `α ∈ {0.5, 0.7, 0.9}` | +2 |
| HPO step 2 | A small 3×2 grid: `L_seed ∈ {90, 180, 365}` × seed decay {off, on (τ_seed = 30)}; (90, off) is the current configuration | +5 |
| **Locked at M5** | **CF3, α = 0.7, `L_seed` = 365 days, seed decay on** (τ_seed = 30), graph window 365 days, `K_n` = 100, quota 100. Selected by the rule; trial `df281d91d7c1`; Recall@100 = 9.332% | — |

**Why HPO step 2 uses a grid instead of "one factor at a time"**: the two factors clearly interact. The longer the seed window, the more users are covered (on D1, 32.2% of the queries get an empty output because they have no purchase within 90 days), but without decay an old purchase from months ago weighs the same as yesterday's, and the noise grows with the window; decay is exactly what offsets this noise. Tuning the window alone (decay off) would underrate long windows, and tuning decay alone (90 days) would not show its effect on long windows. The grid has only 6 cells, each taking only a dozen seconds or so on D1 (§1.3), so the cost is negligible.

Different α can reuse the `C_w` and `n_w` of the same week; only the normalization and the Top-`K_n` have to be redone. **Memory**: computed in chunks of seeds (see §1.3).

**Implementation** (`src/hmrec/recall/r4_itemcf.py`): `C_w` is not stored as a whole. The `C_w(i, ·)` of each chunk of seeds (`chunking.item_chunk_size`) is computed by scipy sparse matrix multiplication, normalized and cut to the Top-`K_n` once per α, and then released; so one pass gives the neighbour tables of several α at once (`generate_for_alphas`). Every element of `C_w` is summed in the same order of users, so the chunked and the unchunked results are bit-identical (tested). Only the neighbours of the seeds used by this week's queries are computed. User scoring is a sparse matrix multiplication as well (the seed 0/1 matrix × the neighbour similarity matrix), chunked by `chunking.query_chunk_size`. CF0 and CF3 differ only in the user weights, and α is a separate config item; the CF0 ablation uses α=0.5 (i.e. the `C / sqrt(n(i)·n(j))` of the table above). The seed window and the seed decay affect only user scoring, not the graph; `L_seed` may not exceed the fit window (365 days), which guarantees that every seed is in the graph.

**9 configurations in total.**

### R5 — WALS (implicit ALS with time decay)

**Definition** (fit window `[a−180d, a)`):
- Interaction strength: `r_ui = Σ exp(−age_days / 60)`, summed over every purchase line.
- Confidence: `c_ui = 1 + α_conf · r_ui`, passed to `implicit` as the values of the CSR matrix, with the library parameter `alpha=1.0` to avoid amplifying twice.
- Default: W2, `α_conf=20`, `factors=64`, `λ=0.1`, `iterations=15`, fixed seed.
- Inference: the fitted user vector is dotted with the item vectors, and the Top-100 inside the active pool are taken. Bought articles are not excluded. The output is empty for users with no interaction in the window.
- **Implementation** (`src/hmrec/recall/r5_wals.py`): only users and articles with a purchase in the window have a row / column (and hence a vector). First a float32 matrix multiplication (chunked by `chunking.query_chunk_size` queries) finds, for each user, the articles scoring no lower than the 100th; then `hmrec.vectors.dot_pairs` recomputes the scores of these candidates and orders them, which guarantees that `score` is bit-identical to the `wals_dot` of §8.3. The two computations may differ in the last digit, so when scores are nearly tied around the 100th place, the set kept may differ from the exact ordering by its last article; the saved scores and order are always exact. `implicit` uses the CPU version, `num_threads` comes from `configs/base.yaml`, and `random_seed` is fixed in `configs/recall/r5_wals.yaml`.

| Type | Content | Configurations |
|---|---|---|
| Ablation | **W0** (`r_ui = 1`, only whether there was a purchase) vs **W2** | 2 |
| HPO step 1 | `α_conf ∈ {10, 20, 40, 60, 80}` (60 and 80 were added after seeing D-phase results, see docs/decisions.md 2026-11-17) | +4 |
| HPO step 2 | `λ ∈ {0.01, 0.1, 1.0}` | +2 |
| HPO step 3 | `factors ∈ {64, 128}` | +1 |
| Convergence check | Run `iterations ∈ {10, 15, 25}` on the final configuration, only to confirm that 15 iterations are already on the plateau | +2 |
| **Locked at M5** | **W2, `α_conf` = 60, `λ` = 0.1, `factors` = 128, `iterations` = 15**, window 180 days, decay τ = 60, seed 42, quota 100. Selected by the rule; trial `3261309f8e8f`; Recall@100 = 10.807% | — |

**No early stopping on iterations**: every round of ALS is a closed-form solution and converges quickly; overfitting is controlled by λ and factors. If 25 iterations are clearly better than 15, change to 25; if the score drops clearly after 15 iterations, λ is too small, and the remedy is to go back to step 2 and increase λ, not to reduce iterations.

**About 11 configurations × 4 weeks ≈ 44 ALS fits in total.**

### R6 — Two-Tower (training described in §6)

**Model** (all values fixed and not tuned; written in `configs/recall/r6_twotower.yaml`, implemented in `src/hmrec/recall/twotower/`; for the reasons see `docs/decisions.md` 2026-11-17 "Two-Tower model structure"):

**Encoding rule for categorical features** (for both towers): a field with ≤ 20 values (including the missing code 0) is one-hot; a field with > 20 values uses an embedding of dimension 16, `padding_idx = 0`, initialized N(0, 0.02²). Article ID and product_code are exceptions (see the table below). With the value counts of the actual data:

| Field | Values (including 0) | Encoding | Dimension |
|---|---|---|---|
| product_type_no | 133 | embedding | 16 |
| product_group_name | 20 | one-hot | 20 |
| garment_group_no | 22 | embedding | 16 |
| index_group_no | 6 | one-hot | 6 |
| section_no | 58 | embedding | 16 |
| colour_group_code | 51 | embedding | 16 |
| age_bucket | 8 | one-hot | 8 |
| club_member_status | 4 | one-hot | 4 |
| fashion_news_frequency | 4 | one-hot | 4 |

- **Item tower** (history items and target items share the same item tower, with fully shared parameters): the parts below are **concatenated**:

  | Part | Structure | Dimension |
  |---|---|---|
  | Image | `Linear(512 → 128)`, with the output multiplied by `image_mask` (without an image this part is all 0, bias included) | 128 |
  | Text | `Linear(384 → 128)`, with the output multiplied by `text_mask` | 128 |
  | Article ID | `Embedding(n_articles + 1, 64, padding_idx = n_articles)`, initialized to 0 | 64 |
  | product_code | `Embedding(47,225, 32, padding_idx = 0)`, initialized to 0 | 32 |
  | The 6 category fields | By the rule above: 16 + 20 + 16 + 6 + 16 + 16 | 90 |
  | **Total** | | **442** |

  After concatenation → `Linear(442 → 256)` → ReLU → `Linear(256 → 128)` → L2 normalization.
  - The 442 articles without an image vector (see §2.1): the image part is 0 after multiplication by the mask (a 0 vector is not fed in as if it were a real input), and the item vector comes from the text, ID and category parts. These articles **take part in training and recall as usual** and are not removed from the catalog or the active pool.
  - Article ID and product_code are initialized to 0: for an article (or a style) not seen in training, this part stays 0, and the vector is determined by content and categories only, with no random noise added.
- **User tower**: the user's **20 most recent purchases** before the anchor (before the purchase date in training), with no time window (for deduplication, order and padding rules see §6.1), are encoded by the same item tower (padding positions are not encoded). **Position-weighted pooling**: one scalar `w_j` is learned for each of the 20 positions (starting at 0, i.e. equal to a masked mean at the start), `pooled = Σ_j a_j · h_j`, with `a = softmax(w)` computed only over positions where the mask is True; with an empty history, `pooled = 0`. Profile: the three fields are all one-hot, 16 dimensions together. `[pooled 128 ; profile 16]` (144 dimensions) → `Linear(144 → 256)` → ReLU → `Linear(256 → 128)` → L2 normalization.
- **Loss**: a multi-positive, duplicate-aware in-batch softmax with LogQ correction:
  - columns = the **deduplicated** target articles of the batch (an article that appears several times in the batch is one column and is not treated as two negatives);
  - `logit[i, k] = u_i · v_k / τ − log Q(k)`, **τ = 0.05**, with the LogQ correction applied to all columns (including the target column);
  - `Q(k)` = the share of the samples of the current training set whose target is k: for the base it is computed over all base training samples (excluding the alarm users), and for each weekly fine-tuning over the training set of that run (the new week + replay);
  - the columns of the other positives of the same group (the same user on the same day) are set to −∞ in that row (neither target nor negative); loss = the mean over rows of the cross-entropy with respect to the row's target column.
- **Training**: batch 512; for every epoch a shuffled order is generated on CPU from `(seed, epoch)`, and the last incomplete batch is kept; Adam (β = 0.9, 0.999; ε = 1e-8), lr **1e-3 fixed with no decay** (1e-4 for fine-tuning), weight decay 0, no gradient clipping; **6 epochs** of training, with a checkpoint saved after each epoch (`artifacts/models/twotower/base/epoch_{k}.pt`); seed 42; 1% of the users are held out and only their loss is recorded as an alarm, used for no selection. In every step each distinct article of the batch (targets + histories, padding removed) is encoded once and then fetched by index.
- **Device**: the real model runs 200 steps each on mps and on cpu (one process each, the first 10 steps not timed); MPS is used only if its mean time per step is more than 1.3 times faster than CPU, otherwise CPU (the backward pass of embeddings on MPS is not deterministic, so the result cannot be reproduced exactly). The result is recorded in `docs/decisions.md`.
- **Inference**: R6 is an ordinary `RecallChannel` (`recall/twotower/infer.py`). The history is the 20 most recent items before the anchor (the same rule as in training; the data are first cut down to the query users of the week); a user without history uses a vector from the profile alone. Vectors are computed only for the articles of the active pool `A(a)`, and the Top-100 is taken in chunks of `chunking.dense_score_query_chunk_size` (sharing `recall/vector_scoring.py` with R5); `score` is recomputed with `vectors.dot_pairs`, which guarantees it is bit-identical to the `tt_dot` of §8.3; ordering `score DESC, article_idx ASC`.
- **Selection of E_base**: default epoch 1, candidates epoch 2–6, fewer epochs first (the `r6_twotower.base_epoch` step of `configs/hpo/d_phase.yaml`), selected by the multi-candidate rule of §4.2; each checkpoint predicts D1–D4 directly with no update, and the results are written to `results/d_phase.csv` and `results/d_phase_steps/r6_twotower.csv`. If epoch 6 is selected, the run stops for a person to decide.
- Fixed (not tuned): all of the above, including batch 512, output 128, temperature 0.05, the LR, history length 20, and 6 epochs.

| Type | Content | Cost |
|---|---|---|
| HPO step 1 | **E_base**: the base model saves a checkpoint after each epoch (1–6), and each checkpoint predicts D1–D4 directly **with no update** | 1 base training |
| Ablation | **Frozen base** (no update afterwards) vs **weekly fine-tuning** | included in step 2 |
| HPO step 2 | `E_update ∈ {1, 2}` × `replay ∈ {0, 50%}`, 4 rolling chains in total | 4 chains × 4 weeks |
| Robustness | The top 2 configurations are each run with 2 more seeds | about 4 chains |
| Optional | ① a content ablation with IDs only, without CLIP / MiniLM; ② a base length of 52 weeks | if time allows |
| Not done | A control retrained from scratch every week (full retrain) | — |
| **Locked at M5** | **E_base = 4, E_update = 2, replay = 50%**, fine-tune lr 1e-4, chain seed 42, the rest at the fixed values above. **A manual decision**: the rule selected (E_update = 2, replay = 0) (trial `ef62019e6c73`, 14.576%), and (2, 50%) was adopted by hand (trial `835ef201ee4d`, Recall@100 = 14.634%); for the reasons see `docs/decisions.md`. Model files and sha256 in §6.5 | — |

### 5.7 Closing the D phase: the union check

With the final configuration of each of the 6 channels, compute on D1–D4:

1. **Union recall** (target ≥ 24%, i.e. G2);
2. **Leave-one-out union recall**: how much union recall drops when each channel is removed, to judge whether the channel brings candidates the other channels do not have;
3. the distribution of candidates per query (mean, P50, P90);
4. **Active-pool diagnostic of R3**: on D1–D4, the share of R3 candidates that are not inside `A(a)` and their hit rate (compared with the candidates inside `A(a)`), and the change in union recall when those of them "not recalled by any other channel" are removed. **Pre-registered decision criterion**: if union recall drops by **< 0.1 percentage points** (absolute), R3 is changed to output only articles inside `A(a)`; otherwise it stays as it is (§2.4). The same statistics are computed for R2, **as a diagnostic only, with no change**. The criterion was set after seeing R3's results on D1, see `docs/decisions.md` (2026-11-16 "R3 candidates outside the active pool").

If union recall is clearly below 24%, **recall is changed first, and the F phase is not entered**. Whether a channel is dropped is decided by the system-level ablation of the V phase (MAP@12), not here.

**Implementation** (`src/hmrec/hpo/union_check.py`, `scripts/m4_union_check.py`, settings in `configs/m4_union_check.yaml`):
- The six channels use the D-phase selections (replaying the rule over `results/d_phase.csv`, with no new trial run), plus the manual selection in `manual_selection` (R6 adopts (E_update = 2, replay = 50%), see `docs/decisions.md`).
- The candidates of R1–R5 preferably reuse the files of the pre-check (`artifacts/hpo/D/union_precheck/`; the same trial_id means the same configuration); R6 is **not retrained**, and each week is predicted with the model the training chain saved **before** fine-tuning on that week (D1 with the E_base checkpoint of the base, D2 with `chain/{trial_id}/D1.pt`, and so on). Candidates are written to `artifacts/m4/candidates/` and results to `results/m4_union_check.csv`.
- The selection metric of every channel in every week must be **bit-identical** to the record in `results/d_phase.csv`, otherwise an error is raised and the run stops.
- Memory: the union is computed from **the candidates that hit a label** only (each channel's candidates are first semi-joined with the labels, and the hit rows are about 1/400); union recall and leave-one-out are both computed on these small tables; the candidates per query are counted in chunks of queries. The result is exactly the same as computing directly on all candidates (tested).
- The decision criterion of item 4 uses the **4-week pooled (query-weighted) union recall**, with the threshold 0.001 (0.1 percentage points) written in `configs/m4_union_check.yaml`; the change is made only if the drop is strictly smaller. The M4 run only computes and reports the conclusion of the rule; if the conclusion is "change", the configuration change of R3 and the regeneration of its candidates are done separately before the lock (M5).

### 5.8 Summary of the D-phase workload

| Channel | Configurations | Cost |
|---|---|---|
| R1 | 8 | very low |
| R2 | 5 | very low |
| R3 | 7 | low |
| R4 | 9 | medium (the graph is built every week and needs chunking) |
| R5 | ~11 | medium (about 44 ALS fits) |
| R6 | 1 base + about 8 chains | high (GPU) |

---

## 6. The Two-Tower training chain (the project has exactly one)

### 6.1 Training samples

Each (user, purchase date, article) is one positive sample: `(the user's 20 most recent purchases before the purchase date + the user profile) → the article bought`. The same user buying the same article several times on the same day counts as one sample; the positives of the same user on the same day share one group_id (for the multi-positive loss). The history takes only purchases **before** the purchase date (the day itself excluded), deduplicated by the same (user, day, article), ordered from most recent to oldest, and in ascending `article_idx` within a day; histories shorter than 20 are padded at the end and carry a mask; **the padding index is `n_articles`** (it corresponds to no real article). The item feature table (image and text vectors with their masks, product_code and category codes) has a dedicated padding row at the end (vectors all 0, masks False, codes all 0), and the ID / category embeddings of the model point to it with `padding_idx`. User histories are computed in advance and stored as an int32 array `[N, 20]` (implementation: `src/hmrec/recall/twotower/data.py`).

### 6.2 Step 0: the base model

- Training samples: all purchases with a label date in **2019-12-03 – 2020-06-02** (26 weeks).
- A **fixed learning rate** is used (no decay), and a checkpoint is saved after each epoch. The k-th checkpoint is thus the model "trained for k epochs", and one training run covers E_base = 1–6.
- **No validation set is held out and there is no early stopping.** 1% of the users may be held out at random as an alarm only (to check whether the loss blows up or there is a bug); **it is used for no selection**.

### 6.3 Step 1: selecting E_base

Each checkpoint predicts D1, D2, D3 and D4 directly, **with no update at all**. The model has seen none of these weeks, so this is a legitimate validation. E_base is selected by the rule of §4.

### 6.4 Step 2: selecting the update scheme (rolling chain over D1–D4)

Starting from the selected base, each update configuration runs one chain:

| Time | Operation |
|---|---|
| 06-03 | The base predicts D1 → scored → fine-tuned for `E_update` epochs on the purchases of week D1 (plus replay) → M1 |
| 06-10 | M1 predicts D2 → scored → fine-tuned on D2 → M2 |
| 06-17 | M2 predicts D3 → scored → fine-tuned on D3 → M3 |
| 06-24 | M3 predicts D4 → scored → fine-tuned on D4 → **M4** |

- **Every score is obtained "predict first, train afterwards"** (progressive validation): the model has not yet seen the week it predicts. In the implementation (`src/hmrec/recall/twotower/chain.py`), each week the current model first generates the candidates and is then fine-tuned; the folds must be processed in time order, otherwise an error is raised.
- **New data**: for the week with anchor `a`, the purchases with a purchase date in `[a, a + 7)`; the samples, histories and deduplication rules are exactly as in base training (§6.1).
- **Definition of replay**: `replay` is the share of old data in the fine-tuning set. The old data are drawn at random without replacement from the samples of **the 4 weeks before the week begins** (purchase dates `[a − 28, a)`), in the number `n_replay = round(n_new × replay / (1 − replay))`: with `replay = 50%`, `n_replay = n_new` (new : old = 1 : 1), and with `replay = 0` only the new data are used. Sampling uses a fixed seed (`chain_seed` and the anchor of the week).
- **Alarm users**: the 1% of alarm users of base training remain excluded (the same users are drawn again from the base arrays and seed), from both the new data and the replay; after each weekly fine-tuning their loss on that week is recorded, as an alarm only, used for no selection.
- **Optimizer**: **Adam is re-initialized** for every weekly fine-tuning (lr 1e-4 = 0.1 times the base LR, other settings as for the base), without inheriting the optimizer state of the previous week. Every epoch is shuffled with `(chain_seed, anchor, epoch)`. The `Q` of LogQ is computed from the fine-tuning set of that run (new data + replay).
- **Saving**: the model after each week's fine-tuning is saved to `artifacts/models/twotower/chain/{trial_id}/{fold}.pt` (existing files are never overwritten); after D4 has been predicted, the model is also fine-tuned once on D4, giving **M4** (`D4.pt`), from which F1 starts.
- **Default configuration and candidates** (the `r6_twotower.update` step of `configs/hpo/d_phase.yaml`): the default is **frozen** (`e_update = 0`: the base is not updated and predicts D1–D4 directly, which is the result of the E_base selected in step 1, reused as it is). Candidates in priority order: `(E_update = 1, replay = 0) < (1, 50%) < (2, 0) < (2, 50%)` (fewer epochs and no replay are cheaper and come first), selected by the multi-candidate rule of §4.2.
- All chains score the same on D1 (all are predictions of the base), and what really separates the configurations is D2–D4: "strictly better in at least 3 of the 4 weeks" of §4.2 in effect requires D2, D3 and D4 all to be better, which is the intended behaviour.
- **Robustness**: the top 2 configurations by the rule of §4.2 (the 1st is the one the rule selects; the 2nd is the one the rule would select if the 1st were removed; ordered by pooled score if no candidate passes) are each run again with seeds 43 and 44 (which affect only the shuffling order of fine-tuning and the replay sampling), and the mean and the variation are reported (`results/d_phase_steps/r6_twotower_seeds.csv`). **The selection is still based on the result of seed 42**; the extra seeds are for reporting only.
- Memory: the chain processes one fold at a time, and each week's fine-tuning arrays are released once used; the history reads only the purchases of the users involved.
- **D-phase result** (2026-11-17): the rule selected (E_update = 2, replay = 0) (pooled Recall@100 14.576%, against 11.843% for frozen); **(2, 50%) was adopted by hand** (14.634%), which is a decision that deviates from the rule; for the reasons see `docs/decisions.md`.

### 6.5 Step 3: the lock

Before 2020-07-01, `base_window, E_base, E_update, replay, fine_tune_lr, seed` are written into `configs/locks/twotower.lock.yaml`. **The M4 produced by the selected configuration is the model used from then on; the base is not retrained.**

**What is locked (M5, 2026-11-17)**: `configs/locks/twotower.lock.yaml` records:

- the full configuration of R6: the base window (purchase dates 2019-12-03 – 2020-06-02), all values of the model structure and the training settings, `E_base = 4`, `E_update = 2`, `replay = 0.5`, fine-tune lr 1e-4, chain seed 42;
- the basis of the selection: **a manual decision**. The rule selected (2, 0) (trial `ef62019e6c73`), what is locked is (2, 50%) (trial `835ef201ee4d`), with a pointer to the entry in `docs/decisions.md`;
- the settings of the alarm users: share 1%, seed 42, and the sha256 of the source of the draw (`customer_idx.npy` of the base training arrays) with the number of people drawn (7,141);
- the paths and **sha256** of two model files: `base` (`artifacts/models/twotower/base/epoch_4.pt`, which predicted D1 in the training chain) and `f1_start` (`artifacts/models/twotower/chain/835ef201ee4d/D4.pt`, i.e. M4, which predicts F1).

`artifacts/` is not in git, so sha256 confirms that the same file is used later: `src/hmrec/recall/locked.py` checks the sha256 **before** loading the model weights and, on a mismatch, raises an error and does not load.

### 6.6 Step 4: F → V → T, the same every week

| Time | Operation |
|---|---|
| 07-01 | M4 predicts F1 → outputs candidates → fine-tuned on F1 → M5 |
| 07-08 | M5 predicts F2 → outputs candidates → fine-tuned on F2 → M6 |
| … | The same every week, up to T4 |

The Recall@100 of every week is recorded as usual. If a week is more than 10% below the mean of the previous 4 weeks, it is flagged in the report, but **the configuration is not changed**.

**Implementation (M6, `src/hmrec/recall/generate.py`, `scripts/m6_generate_candidates.py`)**:

- **Built from the lock files only**: all six channels are obtained through `src/hmrec/recall/locked.py`. R6 uses `load_locked_twotower_chain`, which returns the very class of the D-phase training chain (`TwoTowerChainChannel` in `recall/twotower/chain.py`), so the weekly fine-tuning runs the same code: the purchases of the week `[a, a + 7)` plus 1 : 1 replay, the locked alarm users excluded, Adam re-initialized every week, the locked `E_update = 2` and fine-tune lr 1e-4.
- **Order within each fold** (R6): ① take "the model that predicts this week" and check its sha256; ② predict, and save the vectors and the candidates; ③ then fine-tune on this week, save the model `artifacts/models/twotower/chain_locked/{fold}.pt`, and append the sha256 and the rows, loss, alarm-user loss and run time of the fine-tuning to `results/m6_twotower_chain.csv`. **T4 only predicts and is not fine-tuned**, so it has neither a model nor a record.
- **The order is enforced by code** (`src/hmrec/recall/chain_records.py`): F1 uses `f1_start` of the lock file (i.e. M4); every later fold uses the model saved by the previous fold, which must exist, have a record in `results/m6_twotower_chain.csv`, and match the sha256 of the record, otherwise an error is raised. So a week cannot be skipped, and a model left by another run cannot be used.
- **Files saved for the features** (`features/` may not import `recall/` and can only read these files, see §14.2): the vectors of R5 (for users, only the query users of the fold are saved); the vectors of R6 (the users are the query users of the fold, and the items are **all articles of the catalog**, not only `A(a)`, because the candidates of R2 may be outside the active pool; computed with **the model that predicts this week**, i.e. before this week's fine-tuning); the complete bucket score table `S(g, i)` of R1 (not limited by the Top-100 or the active pool). The item vectors of R6 inside `A(a)` are the ones used when the candidates were generated, and the articles outside the pool are encoded separately and merged in, so the candidates and scores are exactly the same as when only `A(a)` is encoded.
- **Consistency check** (done for every fold; on failure an error is raised and the candidates are not written): the saved files are read back, and the `score` of every candidate of R5 and R6 must be bit-identical to the dot product computed from the saved vectors; the `score` of every non-fallback candidate of R1 must be bit-identical to the value in the score table.
- **One separate process per (fold, channel)**, which appends its run time and true memory peak (`ru_maxrss`) to `results/m6_runs.csv` when it ends; `scripts/m6_run_folds.py` drives the folds one by one in time order. A candidate file is never overwritten once it exists; a re-run skips it with a notice.
- **Monitoring** (`src/hmrec/eval/monitor.py`, `scripts/m6_monitor_candidates.py`, results appended to `results/m6_candidates.csv`; recorded only, with no selection made): for the F and V weeks it records the Recall@quota of each channel, the six-channel union recall, the deduplicated candidates per query, and the share of queries no channel hits; the recall of each channel is compared with its mean over the previous 4 weeks (for F1 the previous 4 weeks are D1–D4, with scores taken from the lock files), and when it is more than 10% lower, this is flagged in the column `low_recall_flag`. **For the T weeks no label at all is read in M6**; only the number of candidate rows and the candidates per query are recorded, and the metrics of the T weeks are left to M8. The code refuses to build an evaluator for a T fold (tested).
- **When the run stops for a person to decide** (settings in `configs/m6_candidates.yaml`): the true memory peak of any process exceeds `memory.limit_gb`; the schema validation or the consistency check fails; the sha256 of a model does not match its record; the training loss or the alarm-user loss of the R6 fine-tuning is not finite, or exceeds `twotower_chain.max_fine_tune_loss` (7.0; in the D-phase training chain the weekly training loss was 5.49–5.56 and the alarm-user loss 5.25–5.69).

### 6.7 Post-hoc diagnostic (reported only; changes no selection)

In the V and T phases, **HP drift regret** is computed additionally: how much more recall could at most be gained if `E_update` and `replay` were re-picked with the labels of the current week. A small gap means the assumption "hyperparameters stay stable over a few months" holds.

---

## 7. Candidate fusion: fuse first, then train

### 7.1 Three options

| Option | How | Verdict |
|---|---|---|
| A. Each channel goes into the wide table separately | An article recalled by 3 channels appears as 3 rows | **Not used** |
| B. Truncate after RRF fusion | Score with RRF and keep only the top N for the ranker | **Not used** |
| C. **Union + dedup, then train** | One row per `(query, article)`, with the rank / score of every channel as columns of that row | **Used** |

### 7.2 Why C

I agree with your preference for "fuse first", but the fusion here has to be **union + dedup, not RRF truncation**:

- **Not A**: the same `(user, article)` appears in several rows, and the label is counted repeatedly; popular articles recalled by several channels have more rows, and the ranker would wrongly take them to be more important; with duplicate articles in a LambdaMART group the ranking itself is meaningless, and the final Top-12 could contain duplicates too.
- **Not B**: RRF looks only at ranks, and the ranker never sees the articles it cuts off, which directly lowers the recall ceiling of the system.
- **C**: all candidates are kept, and "how many channels recalled it and at which rank in each" all become features for LambdaMART to learn from. The RRF score is kept as a feature column too.

### 7.3 Fusion procedure (once per anchor)

1. Read the output tables of the 6 channels (one Parquet file per channel).
2. In chunks of queries, pivot the results of the 6 channels into the wide table: one row per `(query_id, article_id)`, one column each for the rank and the score of every channel, filled with NaN where the channel did not recall the article.
3. Compute the aggregate columns: `n_channels`, `best_rank`, `rrf_score`.
4. Join the other features of §8.
5. Store as `artifacts/features/{fold}/part-*.parquet` (chunked by query, one file per chunk). **Frozen once generated, and not regenerated when later used for training.**

---

## 8. Ranker wide-table features (71 in total)

> 2026-11-17: 15 features were added to the original 56 (2 in group B, 1 in group C, 1 in group D, 11 in group E), 71 in total. The additions are a design change made **before the feature lock, with no ranker result in existence**, based on the emphasis that the first-place solution of the H&M competition puts on user–item interaction features and on multiple time windows (`docs/decisions.md`). The added features are marked **(new)** in the tables below.

### 8.1 Principles

- Each row = one `(query, article)` candidate.
- **All features use only data with `t_dat < anchor`**, and every time window is counted backwards from the anchor.
- Raw IDs (`customer_id`, `article_id`, `product_code`) are not used as features.
- Category fields are LightGBM categorical features.
- Missing values stay NaN; LightGBM handles them itself.

**Conventions as implemented (M6, `configs/features.yaml`, `src/hmrec/features/`)**:

- **Columns of the wide table**: `query_id`, `article_idx` (int32, keys only, not features), then the 71 features below, in the order of `configs/features.yaml`. There is no `customer_idx` and **no label column** (the label is joined only in `ranker/sampling.py` of M7).
- **dtype**: numeric features are always float32, with NaN for missing; the 8 categorical features (the 6 article categories of §8.4, and `club_member_status` and `fashion_news_frequency` of §8.5) are int16 integer codes: 0 = missing, 1..K = the index of the value among all sorted values of the field. The code table is built from all rows of `articles` and `customers`, is shared by all folds, and is saved in `artifacts/features/category_codes.json`; when a later fold builds it again, it must be exactly the same.
- **"Number of purchase lines / purchases" always counts purchase lines** (two lines of the same article by the same user on the same day count as 2), consistent with how R1 and R2 count. The only exception is `item_repurchase_ratio_52w` (see §8.4).
- **"Days since"** = `anchor_day − t_dat_day`; yesterday = 1.
- **Windows**: 1w = `[a−7, a)`, prev_1w = `[a−14, a−7)`, 4w = `[a−28, a)`, 52w = `[a−364, a)`; a feature with no window stated uses the whole history `t_dat < a`. Every window is read through `data/asof.py`.
- **Missing-value rules**: count features are 0 when there are no data; "days since", means, shares and ratios are NaN when they cannot be defined (never bought, no data in the window, denominator 0, either side missing).
- **Output**: `artifacts/features/{fold}/part-{i}.parquet`, one file per `chunking.query_chunk_size` queries, sorted by `(query_id, article_idx)`, with the rows of a query contiguous and never spanning two files; `features.json` is written last, and only its existence means the fold is complete. Nothing is overwritten after generation.

### 8.2 Group A: recall source (14)

| Feature | Definition |
|---|---|
| `pop_rank`, `rep_rank`, `sib_rank`, `icf_rank`, `wals_rank`, `tt_rank` | The rank in each channel (1 = best); NaN when the channel did not recall the article |
| `pop_score`, `rep_score`, `sib_score`, `icf_score` | The raw score in each channel; NaN when not recalled |
| `pop_is_fallback` | Whether this candidate of R1 comes from the global fill-in |
| `n_channels` | How many channels recalled the article (1–6) |
| `best_rank` | The best rank over the 6 channels |
| `rrf_score` | `Σ 1 / (60 + rank)`, summed over the channels that recalled it |

The raw scores of WALS and Two-Tower are in group B, because group B is computed for all candidates and has wider coverage.

Implementation (`src/hmrec/fusion/union.py`, `fusion/rrf.py`, `features/source.py`): the six channels are merged and deduplicated on `(query_id, article_idx)`, one row per candidate. `pop_is_fallback` is 0 / 1 on the rows R1 recalled and **NaN on the rows R1 did not recall** (like the other source columns). `n_channels` and `best_rank` are stored as float32. `rrf_score` is summed in float64 in a fixed order (pop, rep, sib, icf, wals, tt) and then stored as float32. Ordering of the RRF baseline (§9.5): `rrf_score DESC`, then `best_rank ASC`, then `article_idx ASC`.

### 8.3 Group B: dense scores computed for all candidates (7)

The features of this group are computed whichever channel a candidate comes from. They are usually the most useful group.

| Feature | Definition |
|---|---|
| `tt_dot` | Two-Tower user vector · item vector (the model of the current week's anchor) |
| `wals_dot` | WALS user vector · item vector; NaN when the user or the article has no vector |
| `pop_bucket_score` | The R1 decayed popularity score of the article in the user's age bucket (not limited by the Top-100) |
| `img_sim_hist` | Cosine similarity between the article's CLIP vector and the mean CLIP vector of the user's 20 most recent purchases. The mean vector uses only those of the 20 that have an image vector. **NaN when the article itself has no image vector, or when none of the user's 20 most recent purchases has one** |
| `txt_sim_hist` | The same with the MiniLM text vector. Text covers all articles, so it is **NaN only when the user has no purchase history (no purchase with `t_dat < a`)** |
| `tt_sim_max_hist` **(new)** | The **maximum** of the similarities between the candidate article and the user's 20 most recent purchases (the same history as for `img_sim_hist`). Similarity = the dot product of the R6 item vectors saved for the fold (the vectors are L2-normalized). NaN when the user has no history |
| `tt_sim_top5_hist` **(new)** | As above, the mean of the 5 largest similarities; all of them when the history has fewer than 5 items. NaN when the user has no history |

Implementation (`src/hmrec/features/dense.py`):

- `tt_dot` and `wals_dot` are computed from the vectors saved in M6 with `vectors.dot_pairs` (the same function as the `score` of R5 and R6). `tt_dot` uses the model that predicts this week (before this week's fine-tuning) and has a value for every candidate. **Checked in full for every fold**: on the rows recalled by R6, `tt_dot` is bit-identical to the `score` of the R6 candidates; on the rows recalled by R5, `wals_dot` is bit-identical to the `score` of R5.
- `pop_bucket_score` reads the bucket score table saved by R1; an (age bucket, article) not in the table is recorded as **0** (nobody of that age bucket bought it within the window, so the decayed popularity is 0).
- `img_sim_hist`, `txt_sim_hist`: "the 20 most recent purchases" come from **the same function** as for Two-Tower (`recent_purchase_histories` in `src/hmrec/data/histories.py`: the same article on the same day counts once, the same article bought on different days counts several times, most recent first). The mean vector = the simple mean of those of the 20 that have a vector (an article counts as often as it appears), **L2-normalized after averaging**; the vector of the candidate article is re-normalized too, and the dot product of the two is the cosine.
- `tt_sim_max_hist`, `tt_sim_top5_hist`: unlike the two "mean vector" features above, they look at the similarity of **single** history items to the candidate and are high as soon as one item is very similar. An article that appears twice among the 20 counts twice. When the vectors are read, the norm of the R6 item vectors is checked to be 1 (±1e-3). They are computed query by query (the candidates of a query are contiguous in the wide table), so memory holds only a small "candidates of the query × history items" matrix; the run time of this step is recorded separately in `features.json` and `results/m6_runs.csv`.

### 8.4 Group C: article features (17)

| Feature | Definition |
|---|---|
| `item_cnt_1w` | Number of purchase lines in `[a−7d, a)` |
| `item_cnt_prev_1w` | Number of purchase lines in `[a−14d, a−7d)` (together with the previous feature, for the trend) |
| `item_cnt_4w` | Number of purchase lines in `[a−28d, a)` |
| `item_uniq_buyers_4w` | Number of distinct buyers in `[a−28d, a)` |
| `item_days_since_first_sale` | Days since it was first purchased (to recognize new articles) |
| `item_days_since_last_sale` | Days since it was last purchased |
| `item_price_mean_4w` | Mean sale price in `[a−28d, a)` |
| `item_online_share_4w` | Share of the online channel (`sales_channel_id=2`) in `[a−28d, a)` |
| `item_buyer_age_mean_4w` | Mean age of the buyers in `[a−28d, a)` |
| `item_repurchase_ratio_52w` | Among the users who bought it in `[a−364d, a)`, the share who bought it 2 or more times |
| `item_discount_ratio` **(new)** | Mean sale price in `[a−7d, a)` ÷ mean sale price in `[a−364d, a)` (below 1 means it sold cheaper last week than over the year); NaN when either window has no sale |
| `product_type_no` | Category (categorical) |
| `product_group_name` | Category (categorical) |
| `garment_group_no` | Category (categorical) |
| `index_group_no` | Category (categorical) |
| `section_no` | Category (categorical) |
| `colour_group_code` | Colour (categorical) |

Implementation (`src/hmrec/features/item.py`; for each fold one row is computed for every article of the catalog):

- `item_days_since_first_sale`: "first purchased" means the first purchase line in the data (which start on 2018-09-20), so for an article already on sale before the data begin this value is at most about 650 days. For an article never sold, both "days since" are NaN.
- `item_price_mean_4w`, `item_online_share_4w`: the mean / share over the purchase lines within 4w; NaN when there is no purchase line within 4w.
- `item_buyer_age_mean_4w`: the mean over the **distinct buyers** within 4w (each buyer counted once), using only buyers of known age; NaN when there is no such buyer.
- `item_repurchase_ratio_52w`: "bought 2 or more times" means bought on **2 or more different dates**; several lines on the same day (several pieces in one purchase) do not count as a repurchase. NaN when nobody bought it within 52w.
- **Mean prices** (`item_price_mean_4w`, the two means of `item_discount_ratio`, `user_price_mean_52w`, and `ui_price_ratio`, which is derived from them): `price` is cast to Float64 before aggregation and back to float32 afterwards. The summation order of a multi-threaded group_by is not fixed, but in float64 this affects only about the 16th digit, far below the precision of float32; a test guarantees that the float32 results of these features are exactly the same after the rows of the same data are shuffled.

### 8.5 Group D: user features (12)

| Feature | Definition |
|---|---|
| `age` | Age (NaN when missing) |
| `club_member_status` | categorical |
| `fashion_news_frequency` | categorical |
| `fn`, `active` | 0/1 |
| `user_cnt_4w` | Number of purchase lines in `[a−28d, a)` |
| `user_cnt_52w` | Number of purchase lines in `[a−364d, a)` |
| `user_days_since_last_purchase` | Days since the most recent purchase (NaN if never bought) |
| `user_price_mean_52w` | Mean sale price in `[a−364d, a)` |
| `user_online_share_52w` | Online share in `[a−364d, a)` |
| `user_uniq_items_52w` | Number of distinct articles bought in `[a−364d, a)` |
| `user_purchase_days_52w` **(new)** | Number of distinct dates with a purchase in `[a−364d, a)`; 0 when there are none |

Implementation (`src/hmrec/features/user.py`; computed only for the query users of the fold): `age` uses the raw value of the table (ages 16 and 17 are kept as they are, not clipped), with NaN for unknown; `fn` and `active` are float32 0 / 1. `user_price_mean_52w` and `user_online_share_52w` are NaN when there is no purchase within 52w.

### 8.6 Group E: user–article cross features (21)

| Feature | Definition |
|---|---|
| `ui_cnt_item` | Number of times the user bought this `article_id` in the past (`t_dat < a`) |
| `ui_days_since_item` | Days since the user last bought this article (NaN if never bought) |
| `ui_cnt_pcode` | Number of times the user bought articles of the same `product_code` |
| `ui_days_since_pcode` | Days since the user last bought an article of the same `product_code` |
| `ui_share_ptype_52w` | Among the user's purchases within 52 weeks, the share with the same `product_type_no` as this article |
| `ui_share_garment_52w` | As above, by `garment_group_no` |
| `ui_share_index_52w` | As above, by `index_group_no` |
| `ui_share_colour_52w` | As above, by `colour_group_code` |
| `ui_price_ratio` | `item_price_mean_4w / user_price_mean_52w` |
| `ui_age_gap` | `|age − item_buyer_age_mean_4w|` |
| `ui_decay_ptype`, `ui_decay_garment`, `ui_decay_section`, `ui_decay_colour` **(new)** | Over the user's whole history with `t_dat < a`, every purchase line with the same `product_type_no` / `garment_group_no` / `section_no` / `colour_group_code` as the candidate article, weighted by `exp(−age_days / 30)` and summed; 0 when there are none. τ is fixed at 30 (`decay_tau_days` in `configs/features.yaml`) and is **not a hyperparameter** |
| `ui_days_since_ptype`, `ui_days_since_section` **(new)** | Days since the user last bought an article of the same `product_type_no` / `section_no`; NaN if never bought |
| `ui_share_section_52w`, `ui_share_pgroup_52w` **(new)** | The same definition as `ui_share_*_52w`, with the attributes `section_no` and `product_group_name` |
| `ui_lift_ptype_52w` **(new)** | `ui_share_ptype_52w` ÷ the site-wide share of the purchase lines of that `product_type_no` within 52w; NaN when the user has no purchase within 52w or the site-wide share is 0 |
| `ui_channel_gap` **(new)** | `|user_online_share_52w − item_online_share_4w|`; NaN when either side is missing |
| `ui_attr_match_cnt` **(new)** | Of the 6 category attributes, the number whose value for the candidate article appears among the user's 20 most recent purchases (0–6); NaN when the user has no history |

Implementation (`src/hmrec/features/user_item.py`):

- "Articles of the same `product_code`" in `ui_cnt_pcode` and `ui_days_since_pcode` **include the candidate article itself**.
- `ui_share_*_52w` = the user's purchase lines within 52w in the same category as the candidate / all purchase lines of the user within 52w. NaN (0 / 0) when the user has no purchase within 52w; 0 when there are purchases but none in the same category.
- `ui_price_ratio`: NaN when either side is missing or the user's mean price is 0. `ui_age_gap`: NaN when the age is unknown or the article has no buyer of known age within 4w.

### 8.7 Freezing and checking the features

- The feature list and dtypes are written in `configs/features.yaml` (checked item by item on load: they must be exactly the 71 of §8.2–§8.6, and neither order, nor groups, nor the categorical features may change). **The features are locked** (2026-11-18, after the features of F1 were generated and verified and before any ranker result appeared; the original plan was "before M6"): `configs/locks/features.lock.yaml` records the names, order and dtypes of the 71 features, the list of the 8 categorical features, two fixed values (`decay_tau_days` 30, `history_similarity_top_k` 5), and four further settings that determine feature values (`history_length` 20, `rrf_k` 60, `online_sales_channel_id` 2, the three window lengths 7 / 28 / 364). The code still reads `configs/features.yaml`; `hmrec.locks.check_features_match_lock` compares the two files item by item and raises an error on any mismatch; `scripts/m6_build_features.py` calls it before generating each fold, and so does `tests/unit/test_features_lock.py`. Only `embedding_pair_chunk_size` of `configs/features.yaml` is not in the lock file (it affects only memory, not results). The category code table (`artifacts/features/category_codes.json`, not in git) is not in the lock file either; each fold checks at generation that it agrees with the existing file. **No feature is tuned** in the V phase.
- **Automatic checks after each fold is generated** (`src/hmrec/features/build.py`): the number of rows equals the number of deduplicated candidates (the union row of `results/m6_candidates.csv`); no duplicate `(query_id, article_idx)`; `tt_dot` / `wals_dot` are bit-identical to the `score` of R6 / R5; the missing rate, minimum, maximum and mean of every feature are written to `results/m6_features.csv`.
- **Independent verification** (`scripts/checks/check_features.py`): using no code of `features/`, `fusion/`, `asof` or `vectors`, it recomputes 25 features (at least 2 per group, 6 of them new) for 1,000 random rows with the plainest loops and compares them with the wide table.
- **Recording and verifying the feature files** (`src/hmrec/feature_files.py`): the `features.json` of each fold records the sha256 and the row count of every part file, and the column list. `verify_feature_files` verifies: the column list is as expected, the part files on disk are exactly the recorded ones, and the sha256 / rows / columns of every file agree. **M7 must call it before reading the features of any fold.**
- Every feature must have a time cut-off test: delete all transactions on the anchor day and after it and recompute; the result must be exactly the same.
- After training, feature importance (gain) is output, for analysis only, not for adding or dropping features.

---

## 9. Ranker training, HPO and baseline

### 9.1 Training samples and negative sampling

- **Only queries with at least 1 positive are used for training.** LambdaMART has zero gradient on a group without positives, and keeping such groups only takes memory.
- **All positives** of each query are kept.
- **Negatives**: sampled **uniformly at random** among the candidates of the same query, r negatives per positive (all of them when there are not enough candidates). `r ∈ {40, 60, 80}` is an HPO parameter, with a default of 40. Empirically at least 40 are needed for MAP@12 to reach 0.032 or more. **Only random negative sampling is used, no hard negatives.**
- Negative sampling is for training only. **Evaluation on V and T uses all candidates, with no sampling.**
- The sampling seed is fixed and written in the config.

Estimate (consistent with §1.3 ②): positives per week ≈ 82,000 queries × 3.4 labels × 24% (the union recall target of G2) ≈ 67,000; at r = 40, 67,000 × 41 ≈ 2.74 million rows per week, about 11 million rows for 4 weeks; at r = 80, 67,000 × 81 ≈ 5.42 million rows per week, about 21.7 million rows for 4 weeks (when there are not enough candidates all negatives are taken, so this is an upper bound). The calculation and the memory handling are in §1.3.

Measured at M4 on D1–D4 (2026-11-17, §5.7, the configuration locked at M5): the six-channel union recall is 25.2%, so the positives are about 5% more than estimated above; **49.5% of the queries have at least 1 positive** (for 50.5% of the queries no channel hits, and they do not enter training); after deduplication each query has 307 candidates on average (P50 320, P90 364; 313 / 327 / 373 before R3 was changed to output only `A(a)`), so at r = 80 the great majority of positives can be given their full number of negatives (a query with 2 positives needs 160 negatives). The scores of these weeks are optimistically biased (the configurations of the channels were selected on these 4 weeks); for the F, V and T weeks the measured values count.

### 9.2 Rolling retraining: a fixed 4-week window

| Predicted | The 4 training weeks |
|---|---|
| V1 | F1 F2 F3 F4 |
| V2 | F2 F3 F4 V1 |
| V3 | F3 F4 V1 V2 |
| V4 | F4 V1 V2 V3 |
| T1 | V1 V2 V3 V4 |
| T2 | V2 V3 V4 T1 |
| T3 | V3 V4 T1 T2 |
| T4 | V4 T1 T2 T3 |

- When V2 is predicted (anchor 08-05), the labels of V1 (07-29 – 08-04) are already complete and can be used directly.
- Training starts from scratch every time; `init_model` is not used to continue training.
- The window is fixed at 4 weeks, so every training run has about the same amount of data, which is what lets the selected number of trees carry over between weeks.

### 9.3 Fixed parameters

| Parameter | Value |
|---|---|
| objective | `lambdarank` |
| learning_rate | 0.05 |
| feature_fraction | 0.8 |
| bagging_fraction / bagging_freq | 0.8 / 1 |
| max_bin | 255 |
| eval_at | 12 (for the training log only) |
| num_threads | 6 |

### 9.4 HPO (V1–V4)

| Step | Parameter | Values | Training runs |
|---|---|---|---|
| 1 | `num_leaves` × `min_data_in_leaf` | {31, 63, 127} × {100, 500} | 6 × 4 = 24 |
| 2 | Negative sampling ratio r | {40, 60, 80} (training data are stream-loaded, see §1.3) | +2 × 4 = 8 |
| Throughout | `num_trees` | Every model is trained to **1000** trees and evaluated at k = 100, 200, …, 1000 trees, so one training gives the whole curve (2000 trees originally; changed to 1000 on 2026-11-18, after seeing the speed-test results of V1, as a compute-budget decision, see the note below the table). "The score with k trees" is defined by `staged_raw_scores` of `ranker/predict.py`: a segment of raw scores is computed per 100 trees and accumulated in order, with every tree computed once (see §9.7) | no increase |

Under the rule of §4, `(num_leaves, min_data_in_leaf, negative sampling ratio, num_trees)` is selected by the MAP@12 of the 4 weeks.

**Fix the number of trees first, then compare parameters** (2026-11-18, written into `configs/hpo/v_phase.yaml` before the run, `docs/decisions.md`):

1. Each parameter set first fixes its own number of trees. Let `P(k)` be the 4-week pooled (query-weighted) MAP@12; `k* = min{k : (max P − P(k)) / max P < 1%}`: the smallest k among those whose relative gap to the maximum is below 1%; a gap of exactly 1% does not count.
2. The parameter sets are then compared by the multi-candidate rule of §4.2, each with the scores of the 4 weeks and the pooled score at its own `k*`.
3. Priority order of step 1: (31, 500) [default] < (31, 100) < (63, 500) < (63, 100) < (127, 500) < (127, 100). Step 2: r = 40 [default, i.e. the winner of step 1, not re-run] < 60 < 80, with every r first fixing its own k.
4. If the `k*` of a step's winner equals the cap (the last checkpoint; now 1000), the curve may still be rising: the run stops after this step and reports, and a person decides whether to add trees. For an unselected candidate whose `k*` equals the cap, this is only noted in the report.

**Cap on the number of trees lowered from 2000 to 1000** (2026-11-18, `docs/decisions.md`). This is a compute-budget decision made **after seeing the speed-test results of V1**; it was not registered beforehand: for the default configuration on V1, the gain in MAP@12 from 100 → 200 trees is +2.5% and from 200 → 300 trees +1.4%, with the gain per 100 trees shrinking to about six tenths of the previous one; extrapolated at this rate, the gain from 900 → 1000 trees is about 0.03%, far below the 1% "about the same" criterion. The extrapolation comes from one fold and one parameter set only; the cap rule of item 4 above is kept, and the run still stops if the winner's `k*` equals 1000. The selection rule itself (`configs/hpo/v_phase.yaml`) was not changed.

**No per-fit early stopping**: like the number of Two-Tower epochs, the number of trees is a hyperparameter selected over 4 weeks and then locked. Every training run can thus use the data of the latest week, and the models of all weeks are trained to the same degree.

### 9.5 Baseline and system-level ablation (V1–V4)

| Experiment | Required / optional | Description |
|---|---|---|
| **RRF baseline** | Required | The same candidate set, ordered by `rrf_score` (k=60), Top-12; ties are broken by `best_rank`, then by `article_id`; nothing is trained and k is not tuned |
| Full (6 channels + LambdaMART) | Required | The main system |
| −R6 (without Two-Tower) | Required | Two-Tower is the most expensive, so its contribution to MAP@12 has to be shown |
| −R1 … −R5 | Not done | The value of R1–R5 is judged only by the leave-one-out union recall of the D phase |

How "removing a channel" is done: drop the candidate rows **recalled only by this channel**, drop the feature columns of the channel (including the corresponding dense scores of group B), then retrain in the rolling manner of §9.2.

**Implementation of −R6** (`src/hmrec/ranker/ablation.py`; sampling and scoring share the same function; candidates and features are not regenerated): (1) drop the rows whose ranks in the other five channels are all NaN; (2) drop `tt_rank`, `tt_dot`, `tt_sim_max_hist` and `tt_sim_top5_hist`, leaving 67 features in the same order; (3) **recompute** `n_channels`, `best_rank` and `rrf_score` from the other five rank columns, with the same expressions of `fusion/rrf.py` and the locked `rrf_k` (otherwise these three columns would still carry the rank of R6); (4) resample (only the selected r), train with the selected parameters to the selected `k*` trees (training is deterministic, and the first k trees equal the first k trees of a model trained to the cap (1000 trees)), and evaluate on V1–V4. The result is only reported and triggers no selection.

### 9.6 The lock

After V4 and before T1, the ranker configuration is written into `configs/locks/ranker.lock.yaml`.

**Locked (2026-11-18)**: `num_leaves` 31, `min_data_in_leaf` 500, negative sampling ratio 40, 500 trees (the V-phase results are at the end of §9.7). The lock file states:

- `model`: two of the three tuned items, the number of trees, `seed`, the histogram mode (row-wise), and all parameters passed to LightGBM (7 binning parameters and 24 training parameters);
- `sampling`: the method of negative sampling, `seed`, ratio 40, and the number of rows per row group of the sample files;
- `training_window`: 4 weeks, trained from scratch each time; `scoring`: the segment length of staged scoring (segments of 100 trees, accumulated in float64) and Top-12;
- `runtime`: LightGBM 4.7.0, libomp 22.1.4, `num_threads` 6;
- `references`: the paths and sha256 of `features.lock.yaml`, `channels.lock.yaml` and `twotower.lock.yaml` (the candidates of R6 and the four `tt_` features depend on the twotower lock);
- `selection`: the basis of the selection (selected by the rule, the default kept), the corresponding `config_id` (`ca1a3cdfce49`), the entry in `docs/decisions.md`, and the scores on V (for the record only);
- the `lock` section (the date and the HEAD of the time).

Not written into the lock file: the settings that affect only memory and logging (`scoring_query_chunk_size`, `train_metric_every`, the memory limit and the chunking of `configs/base.yaml`).

**The only entry point**: M8 may obtain the ranker configuration only through `load_locked_ranker` of `src/hmrec/ranker/locked.py`; it may no longer read the defaults and fixed parameters of `configs/ranker.yaml`, nor `configs/hpo/v_phase.yaml`, nor replay the selection over `results/v_phase*.csv`. Before returning a configuration the entry point checks three things and raises an error if any one fails: (1) the `config_id` recomputed from the settings in the lock file equals the one recorded in the file; (2) the sha256 of the three referenced lock files (features, channels, twotower) match the records; (3) the LightGBM version, the libomp version and `num_threads` of the running process equal the records (libomp is a system dependency outside `uv.lock`, and after an upgrade M8 stops here).

**Check against the records** (`src/hmrec/hpo/v_lock.py`, called only in tests): without reading the lock file, it replays the selection rule of the two steps over `results/v_phase_metrics.csv`, adds the fixed configuration of the V-phase run and the versions recorded by the winning trials, obtains what the lock file should contain, and compares item by item. `tests/unit/test_v_lock.py` also changes every value of the lock file one by one and confirms that every change is rejected.

### 9.7 Conventions as implemented (M7, 2026-11-18)

The configuration is in `configs/ranker.yaml` (fixed parameters, the defaults of the three tuned items, sampling settings, the rule for the histogram mode, speed-test settings) and `configs/hpo/v_phase.yaml` (steps, priority order, the selection rule; committed before any ranker result existed).

**Scope**: M7 involves only F1–F4 and V1–V4. The predicted target must be a V week and the training weeks must be F or V weeks (`ranker/window.py`); every entry point checks this as its first step, and a T (or D) fold raises an error before any file is opened. The labels for training are read only in `ranker/sampling.py`, and those for evaluation only through `load_m7_evaluator` of `eval/ranker_eval.py`. Features are read only from `artifacts/features/`, after calling `verify_feature_files`, with the column list taken from `features.lock.yaml`.

**Negative sampling** (§9.1): each query gets its own random generator from `(seed, query_id)`, the negatives of the query (sorted by `article_idx`) are permuted at random, and the first `min(number of negatives, r × number of positives)` are taken. So the sample of a query depends only on the seed, the query_id and its candidates, not on chunking or processing order; the sample of r = 40 is a subset of that of r = 60, and r = 60 a subset of r = 80. One file per (fold, r), `artifacts/ranker/samples/{full|minus_r6}/{fold}/r{r}.parquet`: `query_id`, `article_idx`, `label` (int8) and the feature columns, sorted by `(query_id, article_idx)`, with 100,000 rows per row group; the `.json` next to it (rows, queries, positives, the distribution of rows per query, sha256, the sha256 of the source `features.json`) is written last. Written once, never overwritten. M7 samples only F1–F4 and V1–V3 (V4 trains only the models that predict T weeks and is left to M8).

Measured sample sizes (2026-11-18, a read-only count; in each fold 47.9%–51.0% of the queries have at least 1 positive):

| fold | Positives | Queries with a positive | Rows at r = 40 | Rows at r = 60 | Rows at r = 80 | Queries not fully supplied (r = 40 / r = 80) |
|---|---|---|---|---|---|---|
| F1 | 72,385 | 41,420 | 2,951,046 | 4,313,058 | 5,559,664 | 282 / 2,908 |
| F2 | 61,660 | 36,679 | 2,515,389 | 3,686,828 | 4,768,474 | 182 / 2,208 |
| F3 | 63,370 | 37,628 | 2,585,402 | 3,787,205 | 4,896,656 | 198 / 2,261 |
| F4 | 65,210 | 39,088 | 2,662,389 | 3,901,694 | 5,052,137 | 185 / 2,281 |
| V1 | 66,559 | 39,673 | 2,716,842 | 3,982,707 | 5,151,132 | 165 / 2,406 |
| V2 | 65,586 | 38,166 | 2,676,158 | 3,916,942 | 5,054,204 | 210 / 2,597 |
| V3 | 58,920 | 34,028 | 2,397,998 | 3,496,447 | 4,502,500 | 250 / 2,271 |

**Dataset** (`ranker/dataset.py`): one `lgb.Sequence` per sample file, with only one row group in memory at a time; the 4 files are passed to `lgb.Dataset` in time order. `query_id` increases with the order of the folds, so the group sizes are the run lengths of `query_id`, in the same order as the rows. After construction it is stored as a binary: `artifacts/lgb_datasets/{experiment}/{target week}_r{r}/train.bin`, with a `dataset.json` next to it that records the sha256 of the 4 sample files and the binning parameters; a trial checks it before loading. Three facts about LightGBM 4.7.0 (verified on synthetic data): single-row access of a `Sequence` must return float64; building twice through `Sequence` gives byte-identical binary files, but the binning differs from "passing a numpy array directly", so every ranker goes through this path; `feature_pre_filter` must be off for the same binary to be reused with different `min_data_in_leaf`.

**Parameters** (everything beyond §9.3 is written in `configs/ranker.yaml`, with the LightGBM 4.7.0 defaults marked as such): `seed` 42, `deterministic` true; `lambdarank_truncation_level` 30, `lambdarank_norm` true, `bagging_by_query` false (all defaults, not in the first round of the grid); the 8 categorical features are passed as integer codes, and code 0 (missing) is an ordinary category. **The histogram mode** (col-wise / row-wise) may give different trees and has to be fixed before the grid, and M8 keeps it: the rule is written into `configs/ranker.yaml` before the speed test (`histogram`: each mode trains 50 trees without evaluation; row-wise is chosen only if it is at least 15% faster per tree and its true peak is within 8 GB, otherwise col-wise), and `mode` is filled in after the speed test; while `mode` is `undecided` the grid refuses to run. **The result of the speed test (2026-11-18) is row-wise**: 0.368 seconds per tree against 0.682 seconds for col-wise, 46% faster; the true peak of the row-wise process is 1.99 GB. `mode: row_wise` has been written into the config.

**trial**: a trial = one training plus evaluation of one parameter set on one V week, in a separate process (`scripts/m7_train_eval.py`), so the `ru_maxrss` of the process is the true peak of this trial itself. Training starts from scratch with no early stopping; the training-set NDCG@12 is recorded every 100 trees (for the log only). Evaluation scores all candidates of the predicted week (no sampling), one part file at a time.

**The score with k trees** (`staged_raw_scores` of `ranker/predict.py`, shared by M7 and M8): `predict(start_iteration=s, num_iteration=100, raw_score=True)` computes one segment per 100 trees, accumulated in order in float64. It is not bit-identical to `predict(num_iteration=k)` (the order of floating-point additions differs; the maximum difference on synthetic data is 1.4e-14), so this function is the definition. At every checkpoint the Top-12 is taken by `query_id ASC, score DESC, article_idx ASC`.

**Records** (all in `results/`): `v_phase.csv` has one row per trial (configuration, fold, training weeks, rows, training and scoring time, sampled peak `sampled_peak_rss_gb`, true peak `process_max_rss_gb`, git commit); `v_phase_metrics.csv` has one row per (configuration, fold, k) (queries, MAP@12, Recall@12, NDCG@12, training-set NDCG@12); `v_phase_steps.csv` has, for every configuration of every step, its `k*`, scores, whether it passes, which one is selected and why. The V phase uses a new record class (`hpo/v_records.py`); `TrialRecord` of the D phase and `results/d_phase.csv` are not touched. **The selection is computed from `v_phase_metrics.csv` alone**: pooled score = `Σ(n_f × mean_f) / Σ n_f` (`math.fsum`), which differs from "the mean over all queries together" only by floating-point rounding (about 1e-16); the benefit is that the selection can be replayed from the files in git alone. Every trial also saves the Top-12 of every checkpoint and the metrics of every query (`artifacts/predictions/`), and `scripts/checks/check_ranker_eval.py` recomputes the metrics from the Top-12 with plain loops.

**Speed test** (`scripts/m7_speed_test.py`, done once before the grid): sample F1–F4 and build the Dataset of V1 / r = 40; each of the two histogram modes trains 50 trees (not evaluated) and the mode is chosen by the rule; the default configuration trains 300 trees and is evaluated on V1 at 100, 200 and 300 trees, with the RRF baseline of V1 computed as well; reported are the training time per tree, the scoring time, the true peak, and checks of the training data (the share of positives of each fold, the distribution of rows per query, whether the training-set NDCG@12 rises). **If MAP@12 at 300 trees is below the RRF baseline, stop and investigate, without extrapolating times.** The results are written to `results/m7_speed_test.csv` and `results/m7_speed_test_metrics.csv`, not into `v_phase*.csv`.

Results of the speed test (2026-11-18, HEAD `25cacf2`, clean working tree; LightGBM 4.7.0, libomp 22.1.4):

| V1 (82,802 queries) | MAP@12 | Recall@12 | NDCG@12 | Training-set NDCG@12 |
|---|---|---|---|---|
| RRF baseline | 0.018511 | 0.048789 | 0.031607 | — |
| Default configuration (31, 500, r = 40), 100 trees | 0.028597 | 0.064804 | 0.044900 | 0.336789 |
| 200 trees | 0.029305 | 0.066029 | 0.045978 | 0.349325 |
| 300 trees | 0.029719 | 0.066720 | 0.046538 | 0.357690 |

At 300 trees MAP@12 is 61% above the RRF baseline, so "stop and investigate" was not triggered. The training-set NDCG@12 rises each time; it cannot be compared directly with the NDCG of V1 (the training set has only queries with a positive, and each positive is paired with only 40 negatives). Checks of the training data: the share of positives of F1–F4 is 2.449%–2.453% (slightly above 1/41, because a few queries are not fully supplied), and the rows per query are 68.1–71.2 on average, with P50 41 and P90 123. Run time and memory are in §1.3. This is the result of one fold, one parameter set and 300 trees; it serves only to confirm the pipeline and to estimate times and is not a conclusion about model performance.

**Automatic check in the grid** (`check_default_against_speed_test` of `hpo/v_phase.py`): as soon as the grid has trained the model of the default configuration on V1, its first 300 trees are compared with the model of the speed test (each of the two model files is first checked against its sha256, then the text of the first 300 trees is compared); on a mismatch the run stops and no later trial is run. Reason: training is deterministic, so the two must be the same; a difference means the environment (LightGBM, libomp), the Dataset or the parameters changed.

**torch and LightGBM cannot be in the same process**: torch ships its own libomp, and LightGBM uses Homebrew's libomp (`/opt/homebrew/Cellar/libomp/22.1.4`, not locked by `uv.lock`); two copies in one process exit at once, hang or segfault. The M7 code does not import torch (for this, `git_state` moved from `hpo/runner.py` to `src/hmrec/git_state.py`), and tests run in two processes (`tests/conftest.py`, AGENTS.md §7). libomp is a system dependency outside `uv.lock` (`brew install libomp`, see the README): the result rows of every M7 process record `lightgbm_version`, `libomp_path` and `libomp_version` (`ranker/runtime.py`, taken from the library actually loaded in the process); after upgrading libomp, the check "two trainings give the same model" has to be done again.

**V-phase results (2026-11-18, the final results of M7)**. Result tables: `results/v_phase.csv` (36 trials), `results/v_phase_metrics.csv`, `results/v_phase_steps.csv`, `results/v_phase_feature_importance.csv`; the item-by-item record is in `docs/decisions.md`.

Step 1 (`num_leaves` × `min_data_in_leaf`, r = 40) and step 2 (the negative sampling ratio); each configuration first fixes its own k and is then compared with the default under the rule of §4.2. No candidate passed in either step, so **the default is kept**:

| Step | Configuration | Selected k | V1–V4 pooled MAP@12 | k of the best score | Best score | Relative to default |
|---|---|---|---|---|---|---|
| 1, 2 | **(31, 500), r = 40 (default, final configuration)** | **500** | **0.032713** | 1000 | 0.032865 | — |
| 1 | (31, 100) | 500 | 0.032619 | 1000 | 0.032849 | −0.29% |
| 1 | (63, 500) | 300 | 0.032645 | 600 | 0.032804 | −0.21% |
| 1 | (63, 100) | 300 | 0.032555 | 600 | 0.032755 | −0.48% |
| 1 | (127, 500) | 200 | 0.032536 | 500 | 0.032766 | −0.54% |
| 1 | (127, 100) | 200 | 0.032528 | 400 | 0.032653 | −0.56% |
| 2 | (31, 500), r = 60 | 400 | 0.032664 | 900 | 0.032917 | −0.15% |
| 2 | (31, 500), r = 80 | 400 | 0.032698 | 800 | 0.032952 | −0.04% |

- An effect of "each fixes its own k": the larger r, the higher the best score, and the rule gives r = 60 and 80 a smaller k (400), which is why the differences in the table are negative. At the same k = 500, r = 80 scores 0.032821, 0.33% above r = 40; comparing best scores it is 0.27% higher. Both are far below the 1% threshold, and the conclusion is unchanged.
- The best score of the default configuration falls on the cap of 1000 trees, but the selected k is 500, so "stop if the winner's k equals the cap" was not triggered; 900 → 1000 trees adds only 0.03%.
- **These parameters of the ranker are on a plateau**: the 8 configurations all have a pooled MAP@12 between 0.0325 and 0.0330. The headroom is in candidates and features. The list of alternatives (`lambdarank_truncation_level`, converting clearly drifting score-type features into within-query ranks) was not evaluated and is left to the next version; the next version may also consider dropping features with a contribution close to 0.

The final configuration (k = 500) against the RRF baseline and the system without R6:

| MAP@12 | V1 | V2 | V3 | V4 | Pooled |
|---|---|---|---|---|---|
| RRF baseline | 0.018511 | 0.019064 | 0.020604 | 0.019662 | 0.019419 |
| Full system | 0.030112 | 0.031332 | 0.035029 | 0.034850 | **0.032713** |
| Without R6 | 0.028727 | 0.029881 | 0.033603 | 0.033013 | 0.031193 |
| Relative drop without R6 | −4.60% | −4.63% | −4.07% | −5.27% | −4.64% |

- **G1 (pooled MAP@12 ≥ 0.032) is met on V**, and G3 (above the RRF baseline in at least 3 of the 4 weeks) is met (higher in all 4 weeks). **The margin is only 2.2%, and the scores on V are optimistically biased** (k and the parameters were selected on these very four weeks), so **T may fall below 0.032**; M8 decides.
- Without R6, pooled MAP@12 falls by 4.64%, below G1; Recall@12 falls by 7.00% (0.072987 → 0.067875) and NDCG@12 by 6.05%.

Feature importance (the 4 models of the final configuration, first 500 trees, the four-week mean of the share of gain; for analysis only). Top 20:

| Rank | Feature | Group | Share | Rank | Feature | Group | Share |
|---|---|---|---|---|---|---|---|
| 1 | `best_rank` | A | 12.01% | 11 | `section_no` | C | 2.85% |
| 2 | `rrf_score` | A | 10.80% | 12 | `item_days_since_first_sale` | C | 2.02% |
| 3 | `tt_dot` | B | 9.37% | 13 | `tt_sim_max_hist` | B | 1.87% |
| 4 | `ui_days_since_pcode` | E | 9.28% | 14 | `rep_score` | A | 1.79% |
| 5 | `tt_rank` | A | 7.57% | 15 | `pop_bucket_score` | B | 1.78% |
| 6 | `product_type_no` | C | 5.39% | 16 | `ui_cnt_pcode` | E | 1.68% |
| 7 | `ui_days_since_item` | E | 4.15% | 17 | `ui_channel_gap` | E | 1.62% |
| 8 | `item_cnt_1w` | C | 3.21% | 18 | `wals_dot` | B | 1.40% |
| 9 | `colour_group_code` | C | 3.07% | 19 | `item_discount_ratio` | C | 1.19% |
| 10 | `user_days_since_last_purchase` | D | 2.97% | 20 | `item_online_share_4w` | C | 1.06% |

The top 5 account for 49.0% and the top 20 for 85.1%; the 4 R6 features account for about 19.0% together. The last 10:

| Rank | Feature | Group | Share | Rank | Feature | Group | Share |
|---|---|---|---|---|---|---|---|
| 62 | `ui_share_index_52w` | E | 0.0499% | 67 | `index_group_no` | C | 0.0040% |
| 63 | `ui_share_pgroup_52w` | E | 0.0419% | 68 | `pop_is_fallback` | A | 0.0028% |
| 64 | `product_group_name` | C | 0.0327% | 69 | `fashion_news_frequency` | D | 0.0017% |
| 65 | `n_channels` | A | 0.0245% | 70 | `active` | D | 0.0013% |
| 66 | `club_member_status` | D | 0.0045% | 71 | `fn` | D | 0.0008% |

Share of each of the five groups:

| Group | Features | Share (four-week mean) | Range over the four weeks |
|---|---|---|---|
| A (recall source) | 14 | 33.85% | 33.09–35.14% |
| B (dense scores) | 7 | 15.02% | 14.27–15.98% |
| C (article features) | 17 | 24.22% | 23.82–24.68% |
| D (user features) | 12 | 5.27% | 4.93–5.82% |
| E (user–article cross features) | 21 | 21.64% | 21.23–22.68% |

Run time and memory: a trial of 1000 trees takes 433–707 seconds (training 0.27–0.53 seconds/tree, scoring plus evaluation 148–286 seconds), with a true peak of 2.63–3.84 GB; the maximum over the 36 trials is 3.84 GB.

---

## 10. T phase (T1–T4)

- The 6 recall channels: with the locked configuration, each week is recomputed or fine-tuned from the data with `t_dat < anchor`, and then outputs its candidates (generated and frozen in M6).
- Ranker: retrained every week on the latest 4 weeks as in §9.2, with the configuration unchanged.
- Results are reported per week and pooled over the 4 weeks: **MAP@12 (the main metric)**, Recall@12, NDCG@12, union recall, the share of zero-hit queries; the RRF baseline is reported as well.
- Whether G1–G3 are met is stated.

### 10.1 Conventions as implemented (M8, 2026-11-18, written before any label of a T week was read)

**Run once.** The labels of T1–T4 were never read before M8 (the entry points of M6 and M7 all reject T folds, and these protections stay unchanged). The formal run of M8 is done once; after the T results are seen, no configuration, code logic or selection is changed. The order is: write code and tests (on small data only) → commit → rehearsal → formal run.

**Configuration and inputs** (`configs/m8_final.yaml`, `src/hmrec/final/`):

- The ranker configuration is obtained only through `load_locked_ranker` of `ranker/locked.py` (§9.6); candidates and features are the files frozen in M6, read after calling `verify_feature_files`.
- Training windows (`ranker/final_window.py`): T1 ← V1–V4, T2 ← V2, V3, V4, T1, T3 ← V3, V4, T1, T2, T4 ← V4, T1, T2, T3; trained from scratch each time to the locked 500 trees. So the labels of T1–T3 are also training labels of the later weeks; every training week ends before the predicted week begins.
- Sampling: V1–V3 reuse the r = 40 sample files of M7 and are **not resampled**; the formal run checks, file by file, that their sha256 equals "the files the passed rehearsal trained on" (recorded in `results/m8_rehearsal_check.csv`). V4, T1, T2 and T3 are sampled anew, with the method, seed and ratio from the lock file. T4 trains no model and is not sampled.
- **The only entry point for the labels of the T weeks** (the descriptive diagnostics after M8 have a second entry point, see §10.2) is `eval/final_eval.py`: evaluation uses `load_final_evaluator` and sampling uses `final_training_label_phases`, both of which require a `FormalRunPermit`; the permit is issued only by `final/gate.py` (see below).

**Rehearsal** (`scripts/m8_final_eval.py --stage rehearsal`; reads no label of a T week): the M8 code path with the locked configuration runs once with V4 as the target week (training weeks F4, V1, V2, V3, using the window function of M7, so T folds are out of reach). The result must be **bit-identical** to the records of M7 and M6 (`final/rehearsal.py`):

- MAP@12, Recall@12, NDCG@12 and the number of queries of the full system at 100, 200, …, 500 trees equal the record of the locked configuration on V4 in `results/v_phase_metrics.csv` (float64 `==`);
- the RRF baseline and the system without R6 are compared in the same way;
- the first 500 trees of the model equal the M7 model (each of the two model files is first checked against its recorded sha256, then the text of the trees is compared);
- the saved Top-12 (500 trees) is the same row by row as the one M7 saved (including the float64 scores);
- the recall monitoring rows of V4 equal those M6 recorded in `results/m6_candidates.csv`;
- the rehearsal trials were run with the current code on a clean working tree.

The outcome of the comparison is written to `results/m8_rehearsal_check.csv` (one row per comparison: `passed` / `failed`, the details of every item, the sha256 of the sample files the rehearsal trained on, the git commit).

**Gate** (`final/gate.py`, enforced in code):

1. The latest rehearsal must be `passed`, run on a clean working tree, and the code and configuration at the current HEAD (`src/`, `scripts/`, `configs/`, `pyproject.toml`, `uv.lock`) must equal the commit of that rehearsal; otherwise the formal run refuses to start and reads no file.
2. When the formal run starts, **before** any label of a T week is read, one row "first read of T-week labels" is recorded in `results/t_phase_label_access.csv`: the time and the git commit.
3. Once this row exists, every process must run the same code: it refuses if the code or configuration at HEAD differs from the recorded commit. So an interrupted formal run can be resumed (recorded steps are skipped), but not with different code.
4. Every process of M8 requires a clean working tree (M7 only warns).

**Formal run** (`--stage formal`; one separate process per step, `scripts/m8_step.py`):

1. **Main results**: MAP@12, Recall@12 and NDCG@12 per week and pooled over the 4 weeks (query-weighted) for T1–T4, with the equal-weight mean of the four weeks as well. T weeks are evaluated only at the locked 500 trees; metrics at other tree counts are not computed.
2. **RRF baseline**: the same metrics.
3. **Monitoring of the recall layer** (the part M6 left to M8): the Recall@quota of each channel, the six-channel union recall, the candidates per query, the share of queries no channel hits; computed with the same function as in M6 and written to `results/t_phase_recall.csv`.
4. **Pre-registered secondary analysis**: the ablation without R6 on T, done as in M7 (§9.5). For reporting only; it affects no verdict.
5. **Criteria for the goals** (written in `configs/m8_final.yaml`, not to be changed after the run; judged by `final/goals.py`, with the verdict written to `results/t_phase_goals.csv`):
   - **G1**: pooled MAP@12 on T1–T4 ≥ 0.032;
   - **G2**: six-channel union recall ≥ 24% in **every week** of T1–T4;
   - **G3**: the MAP@12 of LambdaMART is **strictly above** the RRF baseline in at least 3 of the 4 weeks (a tie does not count).
6. **Records**: `results/t_phase.csv` (one row per trial: run time, sampled peak, true peak, the LightGBM and libomp versions, git commit), `results/t_phase_metrics.csv`, `results/m8_runs.csv` (the processes of sampling, Dataset, baseline and monitoring); the Top-12 of every week is saved in `artifacts/predictions/t_phase/`.

**One difference from the original text of this section**: the original also listed Recall@20 and NDCG@20. The locked ranker keeps only the Top-12 of each query (`scoring.top_k` of `ranker.lock.yaml`), so M8 reports only the three @12 metrics; a metric with K above 12 in `configs/m8_final.yaml` is rejected.

**Independent verification**: `scripts/checks/check_ranker_eval.py --area t_phase` recomputes the three metrics of one T-week trial from the saved Top-12 with plain loops. It also has to pass the gate first to read the labels of T weeks, so it cannot become the first program to read T labels.

---

### 10.2 Descriptive diagnostics after M8 (2026-11-19)

The results on T are higher than on V (§11), and the cause was not investigated. The diagnostics of this section only **describe** results that are already recorded, to help understand this difference. They train no model, do not re-run M8, change no configuration and no lock file, and are **not used to select or change anything**; any later change may still only be compared and selected on V1–V4.

**Code**: `scripts/m8_diagnostics.py`, `src/hmrec/diagnostics/`, with the config in `configs/m8_diagnostics.yaml`. Results are written to `results/t_diag_*.csv` (append-only; every row carries `git_commit` and `run_started_at`). `results/t_phase*.csv` and `results/m8_*.csv` are read, never written.

**A second entry point for reading T labels.** The entry point of M8 (`eval/final_eval.py` + `final/gate.py`) requires the code to be exactly the commit of the formal run; the diagnostics code is new code and so cannot pass that gate, which is how it should behave. The diagnostics use another entry point, `eval/diagnostic_labels.py`: reading the labels of a T week requires a `DiagnosticAccess`, issued only by `diagnostics/access.py`, on these conditions:

1. The formal run is finished: `results/t_phase_goals.csv` holds the three verdicts G1–G3 (reading the labels again cannot change the verdict).
2. The working tree is clean.
3. The content of the M8 files (`src/hmrec/final/`, `eval/final_eval.py`, `ranker/final_window.py`, `scripts/m8_final_eval.py`, `scripts/m8_step.py`) and of the files under `configs/` that existed at the commit of the formal run has not changed; new files are allowed.
4. **Before** any T label file is opened, a row is appended to `results/t_diag_label_access.csv`: time, git commit, purpose (`descriptive diagnostics`), script name. One row per run.

The labels of the D, F and V weeks have been read many times before and need no such object. Only the diagnostics code may use this entry point, and the diagnostics code does not import training, sampling, scoring or the trial code of M8 (`tests/unit/test_diag_code_rules.py`).

**A consequence**: once the diagnostics code is committed, `scripts/checks/check_ranker_eval.py --area t_phase` can no longer run at the current HEAD (it has to pass the M8 gate). To re-run it, first check out tag `m8-run-v1`.

**The four tables**:

| Table | File | Scope | Content | Data source |
|---|---|---|---|---|
| A decomposition | `t_diag_decomposition.csv` | V1–T4, one row per week | union recall, Recall@12, Recall@12 ÷ union recall, MAP@12; one set each for LambdaMART and the RRF baseline | result tables only, no label read |
| B concentration | `t_diag_concentration.csv` | D1–T4, one row per week | the share of the week's purchases on the 100 and on the 1000 best-selling articles; for the articles first sold in `[anchor − 28, anchor)` and the articles with no sale before the anchor, their number and their share of the week's purchases | label files; transactions before the anchor (through `data/asof.py`) |
| C groups | `t_diag_query_groups.csv` | V1–T4, three rows per week | queries split by history before the anchor into "no history / bought in the last 30 days / other"; for each group the number and share of queries, union recall, and the MAP@12 of the two systems | wide table, candidate files, saved per-query metrics, labels |
| D intervals | `t_diag_bootstrap.csv` | T pooled and V pooled, three rows each | the 95% interval of the pooled MAP@12 (LambdaMART, the RRF baseline, their difference) | the per-query values of C |

- **A "purchase" in B** is a deduplicated (customer, article) pair of the label file, not a purchase line. "Articles first sold on the anchor day or later" are defined as: present in the label file of the week and absent from the transactions with `t_dat < anchor`; the transaction table after the anchor is not read. This is only a statistic of the label week itself and **enters no model and no feature**.
- **The groups of C** use `user_days_since_last_purchase` as saved in the wide table (one value per query): NaN is "no history", 1–30 is "bought in the last 30 days" (yesterday is 1, and the anchor day is not history), and the rest is "other".
- **Recomputation and checks**: the per-query AP of LambdaMART is the saved one; the per-query AP of the RRF baseline and the per-query union recall were not saved at the time and are recomputed from the frozen wide table, the candidate files and the labels (with the project's own functions). The recomputed mean of every week must equal the record in the result tables (tolerance 1e-12, `SCORE_EQUALITY_TOLERANCE` of `hpo/selection.py`), and the queries of the three groups must add up to the recorded number of queries; if anything does not match, an error is raised and none of the four tables is written.
- **How D is done (a cluster bootstrap over customers)**: the 4 weeks of a phase are pooled and customers are drawn with replacement; a drawn customer brings all of their queries in these 4 weeks; pooled MAP@12 = the sum of the AP of the drawn queries ÷ the number of queries. 1000 times, with the seed written in the config, taking the 2.5% and 97.5% quantiles. "LambdaMART minus baseline" uses the same draws (paired). It is done once each for T and V. **This interval reflects only the uncertainty from sampling customers, not the differences between weeks**: every resample contains the same 4 weeks.

## 11. Implementation order (milestones)

The memory budget is 12 GB (§1.3), and **only one heavy task runs at a time**.

| # | Content | Output | Completion criterion | Status |
|---|---|---|---|---|
| M0 | Build the repository skeleton of §14, `AGENTS.md`, the lint / test configuration; data cleaning and compression (IDs to int32 and so on); `fold_manifest`; age buckets; the active-pool rule; `selection_rule.yaml`; the 200-step benchmark of Two-Tower; memory measurements | Processed Parquet, configs | The validation script passes; the rule is committed before any score is seen | **Done** |
| M1 | Implement the evaluation module (Recall / MAP / NDCG / union recall) and R1–R5 | Code + unit tests | The time cut-off tests, the tie-break tests and the hand-computed small examples all pass | **Done** |
| M2 | Ablation + HPO of R1–R5 on D1–D4 | `results/d_phase.csv` | The configuration of each channel locked | **Done** (the selection was finished in M2; everything is locked together at M5) |
| M3 | Two-Tower: data preparation → base → training chain over D1–D4 | checkpoints, M4, result tables | `twotower.lock.yaml` | **Done** (the lock file is written at M5) |
| M4 | Union check of the D phase (§5.7) | union recall table | **G2 met**, otherwise back to M2 / M3 | **Done** (25.24% for the 4 weeks pooled, G2 met; R3 changed to output only `A(a)` under the pre-registered rule) |
| M5 | **Freeze all recall configurations** (with a git tag) | `channels.lock.yaml`, `twotower.lock.yaml` | Not modified afterwards | **Done** (lock files written on 2026-11-17; for the result of regenerating the candidates on D1 after the lock and checking them bit by bit against the records, and for the tag, see `docs/decisions.md`) |
| M6 | Generate the candidates and wide-table features of F1–T4 week by week (Two-Tower advances along the training chain week by week) | `artifacts/candidates/`, `artifacts/features/` | Every week passes the time cut-off tests | The candidates (2026-11-17) and the features (2026-11-18) of the 12 weeks have all been generated and passed the checks; the features are locked (`features.lock.yaml`, tag `lock-features-v1`). Awaiting the user's confirmation |
| M7 | Rolling HPO of the ranker on V1–V4 + baseline + system ablation | `results/v_phase.csv` | `ranker.lock.yaml`; **G1 and G3 met on V** | Code and tests written (2026-11-18, §9.7); the selection rule written into `configs/hpo/v_phase.yaml`. The speed test is done (2026-11-18, §9.7): row-wise chosen, and the default configuration with 300 trees has MAP@12 0.0297 on V1 (RRF baseline 0.0185). The cap on trees changed to 1000. **Step 1 is done** (2026-11-18, 24 trials, results in `results/v_phase*.csv`, record in `docs/decisions.md`): by the rule the default configuration is kept (31 leaves, `min_data_in_leaf` 500, r = 40), k = 500, V1–V4 pooled MAP@12 **0.032713**; the five candidates are all below the default at their own k (−0.21% to −0.56%). RRF baseline pooled 0.019419 (four weeks 0.018511 / 0.019064 / 0.020604 / 0.019662). **G1 is currently met**, but only about 2.2% above 0.032, and the scores on V are optimistically biased (k and the parameters were selected on these very four weeks), so whether it is met on T cannot be known now; **G3 is currently met** (above the baseline in all 4 weeks). **Step 2, the run without R6 and feature importance are done** (2026-11-18, end of §9.7): neither r = 60 nor 80 passed the rule, and r = 40 is kept; without R6, pooled MAP@12 falls to 0.031193 (−4.64%). **The final configuration is locked** (`configs/locks/ranker.lock.yaml`, §9.6): 31 leaves, `min_data_in_leaf` 500, r = 40, 500 trees. **Done, awaiting the user's confirmation and tag**. The margin on V is only 2.2%, and T may fall below 0.032 |
| M8 | Final evaluation on T1–T4 | Final report | Report G1–G3 | **The formal run is complete** (2026-11-18, done this one time only; HEAD `18d957f`, clean working tree; results in `results/t_phase*.csv` and `results/m8_*.csv`, commit `6a0ee92`; record in `docs/decisions.md`). The rehearsal passed: 18 comparisons on V4 are bit-identical to the records of M7 and M6. First read of T-week labels: 2026-11-18T22:14:34. **T1–T4 pooled MAP@12 0.039194** (four weeks 0.034960 / 0.040270 / 0.039673 / 0.042435, equal-weight mean 0.039335); Recall@12 pooled 0.087135, NDCG@12 pooled 0.059852. RRF baseline pooled MAP@12 0.021154 (four weeks 0.019151 / 0.021471 / 0.021848 / 0.022411). Six-channel union recall of the four weeks 27.88% / 29.08% / 29.09% / 30.69%. **G1 met** (0.039194 ≥ 0.032), **G2 met** (the lowest week 27.88% ≥ 24%), **G3 met** (strictly above the baseline in all 4 weeks). Secondary analysis: without R6, pooled MAP@12 is 0.036724 (−6.30%). Only @12 metrics are reported (the lock file keeps only the Top-12). **T is higher than V (V pooled 0.032713), contrary to the prior expectation that "V is optimistically biased and T may fall below 0.032"; the cause is to be diagnosed**; the diagnostics only describe and are not used to select or change anything. T1–T4 has been used once, and M8 is not re-run; any later change may only be compared and selected on V1–V4. The final report has not been written yet |

The cheap R1–R3 of M2 can run at the same time as the Two-Tower training of M3; building the R4 graph and fitting R5 should not run at the same time as Two-Tower.

The directory structure is in §14 and the code conventions in §15.
---

## 12. Known limitations

1. **Assumption of hyperparameter stability**: we assume that the hyperparameters selected in June still apply in July–September. Model parameters are updated every week to follow changes in the data, but the hyperparameters are locked. This is tested with the regret diagnostic of §6.7, with the result written into the report.
2. **Seasonal gap**: the D phase (June) and the T phase (September) are about 3 months apart. The hyperparameters of the statistics-based recall channels (for example the decay τ) should not be very sensitive to the season; Two-Tower relies on weekly fine-tuning to keep up.
3. **The ranker's training window is only 4 weeks**: top Kaggle solutions often use 6–18 weeks. We are limited by the length of the F phase and the 8 GB of memory, and use 4 weeks for now.
4. **The scores of the D weeks are optimistically biased**: they took part in selecting recall and can only be used to compare configurations, not reported as model performance.
5. **The scores of the V weeks are optimistically biased**: they took part in selecting the ranker. System performance is judged on the T phase.
6. **The T phase is not completely unseen data**: the old project had looked at weeks around T3 and T4. The results of the T phase are reported as a retrospective test.
7. **The active pool is not real inventory**: it is only an approximation from recent sales.
8. **D3 and D4 look like promotion weeks**: D3 and D4 (2020-06-17 to 06-30) have over 40% more buyers (i.e. queries) than the 16-week mean (D3 +46.9%, D4 +41.8%), and more labels per query (4.0–4.1, against 3.1–3.6 in the other weeks); they look like promotion weeks (not verified). In the query-weighted pooling of the D phase, D3 and D4 carry about **60%** of the weight (236,182 / 391,681 = 60.3%), so the recall configurations selected in the D phase lean towards the purchase patterns of promotion weeks. How it is handled: the selection rule is not changed, and the D-phase reports only gain the grouped diagnostic D1–D2 / D3–D4 (see §4.1).

---


## 13. Decided questions

| ID | Question | Decision |
|---|---|---|
| O1 | A control that retrains Two-Tower from scratch every week | **Not done** |
| O2 | System-level ablations −R1 … −R5 (remove one channel at a time, regenerate candidates and features, retrain the ranker, and see how much MAP@12 falls) | **Not done**. Only −R6 is kept; R1–R5 are judged by the leave-one-out union recall of the D phase |
| O3 | The plausible range of age | Under 18 and outliers (> 99) form one bucket of their own, `AGE_INVALID`; missing is `AGE_UNKNOWN` |
| O4 | Hard negatives | **Not done**; only random negative sampling |
| O5 | Negative sampling ratio and memory | HPO tries `r ∈ {40, 60, 80}`; training data are stream-loaded and need not all fit in memory |

---

## 14. Directory structure

```
hm-recsys/
├── AGENTS.md                    # Rules for coding agents (Claude Code and others); mandatory
├── CLAUDE.md                    # A single line "@AGENTS.md", so that Claude Code reads AGENTS.md
├── README.md                    # Installation, where the data go, how to run each milestone
├── .python-version              # 3.12; uv picks the Python from it
├── pyproject.toml               # Dependencies; ruff / pyright / pytest configuration
├── uv.lock                      # Exact dependency versions generated by uv; in git
├── Makefile                     # Common commands: make lint / test / preprocess ...
├── .gitignore                   # Ignores data/ and artifacts/
│
├── docs/
│   ├── proposal.md              # This file, the only design reference (source of truth)
│   └── decisions.md             # Decision log: date, decision, reason (append-only, history not changed)
│
├── configs/
│   ├── base.yaml                # Paths, global seed, number of threads, memory limit, chunk sizes
│   ├── fold_manifest.yaml       # Anchor and dates of the 16 folds D1–T4
│   ├── selection_rule.yaml      # The selection rule of §4 (committed before scores were seen)
│   ├── preprocess.yaml          # M0: day_zero, age-bucket boundaries, embedding dimensions, row-group size of the transaction table
│   ├── catalog.yaml             # Active pool A(a): window_days=28, min_purchase_lines=3 (§2.4, frozen, not tuned)
│   ├── queries.yaml             # query_id_fold_offset (the numbering rule of query_id, §2.3)
│   ├── m4_union_check.yaml      # M4: union recall target, threshold of the R3 active-pool rule (pre-registered), manual selection
│   ├── m6_candidates.yaml       # M6: monitoring and stop conditions (comparison with the previous 4 weeks, fine-tune loss limit); affects no candidate
│   ├── m8_final.yaml            # M8: target weeks, rehearsal week, criteria of G1–G3 (committed before T labels were read; not to be changed afterwards)
│   ├── m8_diagnostics.yaml      # Descriptive diagnostics after M8: which phases each of the four tables covers, Top-N, days, bootstrap replicates and seed
│   ├── deepfm.yaml              # Stage 2 (§16): the default DeepFM model, its training, the variants, the device rule and the speed test
│   ├── recall/                  # Default configuration of each channel
│   │   ├── r1_popularity.yaml
│   │   ├── r2_repurchase.yaml
│   │   ├── r3_sibling.yaml
│   │   ├── r4_itemcf.yaml
│   │   ├── r5_wals.yaml
│   │   └── r6_twotower.yaml
│   ├── hpo/                     # Search grids of the D phase and the V phase
│   │   ├── d_phase.yaml
│   │   ├── v_phase.yaml         # M7: steps, priority order and selection rule of the ranker (committed before scores were seen)
│   │   └── deepfm.yaml          # Stage 2 (§16.5): the rules of the DeepFM study (committed before any DeepFM result)
│   ├── features.yaml            # The 71 features of the wide table, dtypes, which are categorical, and settings such as windows and decay τ (§8)
│   ├── ranker.yaml              # Fixed LightGBM parameters (§9.3), defaults of the tuned items, sampling settings, rule for the histogram mode, speed-test settings
│   └── locks/                   # Locked configurations; files may only be added, never modified
│       ├── channels.lock.yaml   # M5: full configuration of R1–R5, quotas, trial_id, D-phase scores (for the record only), basis of the selection
│       ├── twotower.lock.yaml   # M5: full configuration of R6, alarm-user settings, paths and sha256 of the two model files
│       ├── features.lock.yaml   # M6: names, order and dtypes of the 71 features, list of categorical features, two fixed values and four settings
│       └── ranker.lock.yaml
│
├── src/hmrec/                   # All reusable logic lives here
│   ├── __init__.py
│   ├── config.py                # Reads YAML → typed dataclasses
│   ├── paths.py                 # Every file path comes from here; paths are not assembled anywhere else
│   ├── logging_utils.py         # Uniform logger setup
│   ├── memory.py                # Memory monitoring: sampled peak per stage (a lower bound) and true peak of the whole process (ru_maxrss)
│   ├── threads.py               # Sets POLARS_MAX_THREADS from base.yaml before polars is imported (depends only on the standard library and yaml)
│   ├── vectors.py               # User / item vectors (shared by WALS and Two-Tower): batched dot products for (customer_idx, article_idx), saving / loading
│   ├── locks.py                 # Reads and validates the lock files of configs/locks/; sha256 check of files (bottom layer; recall / features / ranker may all use it)
│   ├── csv_records.py           # Reading and writing the append-only result tables with a fixed header (results/*.csv)
│   ├── feature_files.py         # Record of each fold's feature files (features.json: sha256 and rows of every part, column list) and its verification; M7 calls it before reading features
│   ├── git_state.py             # HEAD commit and whether the working tree is clean (git_commit / git_dirty of result records); does not depend on torch
│   ├── run_records.py           # results/m6_runs.csv: one row per process (run time, sampled peak, true peak); stops when the true peak exceeds the limit
│   ├── pipeline.py              # M6: runs the steps of one fold one after another, each in its own process; one summary line per fold
│   │
│   ├── data/
│   │   ├── preprocess.py        # Raw CSV → Parquet; IDs mapped to int32; compressed dtypes
│   │   ├── id_maps.py           # Conversion between customer_id / article_id and integer indexes
│   │   ├── day_index.py         # The only implementation of date ↔ t_dat_day (days since day_zero)
│   │   ├── metadata.py          # Reads and writes data/processed/metadata.json; scan_processed_parquet validates before reading a table
│   │   ├── folds.py             # Fold dataclass; reads and validates fold_manifest
│   │   ├── asof.py              # The only place that filters by date: history window and label week (the only entry point against leakage)
│   │   ├── histories.py         # The only implementation of "a user's N most recent purchases before a day"; shared by Two-Tower and the features (img / txt_sim_hist)
│   │   ├── catalog.py           # Catalog C, active pool A(a)
│   │   └── queries.py           # Weekly query list and label sets G_q
│   │
│   ├── recall/
│   │   ├── base.py              # RecallChannel interface; schema validation of the output table
│   │   ├── r1_popularity.py
│   │   ├── r2_repurchase.py
│   │   ├── r3_sibling.py
│   │   ├── r4_itemcf.py
│   │   ├── r5_wals.py
│   │   ├── vector_scoring.py    # Shared by R5 / R6: Top-quota by vector dot product, score from vectors.dot_pairs
│   │   ├── build.py             # Builds R1–R5 from a raw configuration (chunk sizes from base.yaml); shared by HPO and the locked entry point
│   │   ├── locked.py            # The only entry point for building the six channels after M5: reads only configs/locks/; R6 checks the sha256 before loading a model; the training chain of the locked configuration
│   │   ├── inputs.py            # Channel inputs of one fold (tables, active pool, query list); shared by the D phase and M6; reads no label
│   │   ├── chain_records.py     # results/m6_twotower_chain.csv; forces the training chain to advance week by week (the previous week's model exists and its sha256 matches)
│   │   ├── generate.py          # M6: generates one channel's candidates for one fold, saves the files the features need, runs the consistency check, writes to disk (no overwriting)
│   │   └── twotower/
│   │       ├── data.py          # Training samples, user history array [N, 20], item feature table (padding row at the end), fetching batches on the device
│   │       ├── benchmark.py     # The 200-step timing of §1.4 (placeholder model; also used for the device choice of the real model)
│   │       ├── model.py         # ItemTower, UserTower
│   │       ├── loss.py          # Multi-positive, duplicate-aware in-batch softmax + LogQ
│   │       ├── train.py         # Base training, weekly fine-tuning
│   │       ├── infer.py         # Generates user / item vectors, takes the Top-100
│   │       └── chain.py         # The training chain of §6: predict first, then fine-tune
│   │
│   ├── fusion/
│   │   ├── union.py             # Union + dedup of the 6 channels: one row per (query, article), rank / score of each channel as columns
│   │   └── rrf.py               # rrf_score, n_channels, best_rank; ordering of the RRF baseline
│   │
│   ├── features/                # Does not import recall/, reads no label; reads only files in artifacts/
│   │   ├── base.py              # FoldData; purchase lines by window (through asof) with age_days added; missing-value rules
│   │   ├── categories.py        # Integer code table of the categorical features (shared by all folds, saved and verified)
│   │   ├── source.py            # Group A: recall-source features
│   │   ├── dense.py             # Group B: tt_dot, wals_dot, pop_bucket_score, img / txt_sim_hist
│   │   ├── item.py              # Group C: one item table per fold
│   │   ├── user.py              # Group D: one table of query users per fold
│   │   ├── user_item.py         # Group E: small aggregate tables for user–article / product_code / category, then joined to the candidates
│   │   ├── stats.py             # Missing rate, minimum, maximum and mean of every feature (results/m6_features.csv)
│   │   └── build.py             # One fold: read, merge and compute in chunks of queries, automatic checks, write part files
│   │
│   ├── ranker/                  # Does not import recall/ or features/, does not import torch; may import fusion/
│   │   ├── window.py            # 4-week training window; fold check of M7 (target V, training F / V, T raises an error)
│   │   ├── sampling.py          # Random negative sampling within a query (one of the two places that read labels); sample files and records
│   │   ├── ablation.py          # Experiments full / minus_r6: drops the rows and features of R6, recomputes the three aggregate features
│   │   ├── dataset.py           # Streaming read through lgb.Sequence, builds the LightGBM Dataset, stores it as a binary and verifies it
│   │   ├── train.py             # LightGBM parameters, rule for the histogram mode, deterministic training, model files
│   │   ├── runtime.py           # LightGBM version and the path and version of the libomp actually used by the process; raises an error on two copies of OpenMP
│   │   ├── predict.py           # "The score with k trees" (segment accumulation), Top-12 and tie-break
│   │   ├── identity.py          # model_config and the 12-character config_id (shared by V-phase trials and the locked entry point)
│   │   ├── locked.py            # The only entry point for the ranker configuration after the lock (used by M8); checks config_id, the referenced lock files and the runtime environment
│   │   └── final_window.py      # M8: 4-week training window of the T weeks (training weeks V / T); the rehearsal uses the window of M7
│   │
│   ├── eval/
│   │   ├── metrics.py           # Recall@K, AP@12, NDCG@K, union recall
│   │   ├── ranker_eval.py       # M7: gives labels of V weeks only; MAP / Recall / NDCG of a Top-12 table
│   │   ├── final_eval.py        # M8: the only entry point for the labels of T weeks, requires a FormalRunPermit; the rehearsal gets the configured V week only
│   │   ├── diagnostic_labels.py # Entry point for labels of the descriptive diagnostics after M8; a T week requires a DiagnosticAccess (§10.2)
│   │   ├── monitor.py           # M6 monitoring: candidate counts (no label needed); recall / union recall of the F and V weeks; refuses to read the labels of T weeks
│   │   └── report.py            # Summarizes weekly / 4-week results, writes results/*.csv
│   │
│   └── hpo/
│       ├── grid.py              # Reads the search grids, generates the candidates of each step in priority order (§4.2, §4.3)
│       ├── selection.py         # Implements the selection rule of §4 (including the multi-candidate rule and the convergence check of R5)
│       ├── records.py           # The result of one trial; reading and writing results/d_phase.csv (resumable)
│       ├── runner.py            # Runs each trial on each fold, records it, and applies the selection rule step by step
│       ├── d_phase.py           # D phase: loads D1–D4, runs by channel, writes the results and the report of each step
│       ├── robustness.py        # Runs the top configurations of a step with extra seeds (reported only, not used for selection)
│       ├── union_precheck.py    # Replays the rule to get each channel's selection; union pre-check of R1–R5
│       ├── union_check.py       # M4: six-channel union check (computed from hit candidates only), the R3 active-pool rule
│       ├── lock_check.py        # M5: regenerates the candidates of one fold from the lock files only; the scores must be bit-identical to the records
│       ├── v_records.py         # M7: the three result tables of the V phase (trials, metrics, steps); pooled score
│       ├── v_trial.py           # M7: one trial (training + full scoring + evaluation); the Dataset plan; RRF baseline
│       ├── v_phase.py           # M7: reads v_phase.yaml; fixes k first, then compares parameters; starts the processes step by step (resumable); −R6, importance
│       ├── speed_test.py        # M7: the speed test before the grid; choice of the histogram mode; stops when below the RRF baseline
│       └── v_lock.py            # M7: replays the selection over the result tables and checks ranker.lock.yaml item by item (a check only, not a source of configuration)
│   │
│   ├── final/                   # M8: the final evaluation. Does not import torch, recall/ or the selection modules of hpo
│   │   ├── context.py           # Configuration and the locked ranker; the two modes rehearsal / formal
│   │   ├── gate.py              # Gate: formal run only after a passed rehearsal; records "first read of T labels"; run once; issues the FormalRunPermit
│   │   ├── trial.py             # Reused / new sampling, Dataset, one trial, RRF baseline (the same code path for both modes)
│   │   ├── monitor.py           # Recall monitoring of the T weeks (the same function as in M6)
│   │   ├── rehearsal.py         # Bit-by-bit comparison of the rehearsal with the records of M7 / M6
│   │   ├── goals.py             # Verdict on G1–G3 (pure functions)
│   │   ├── report.py            # Weekly / pooled / equal-weight summary; the verdict is written to results/t_phase_goals.csv
│   │   ├── run.py               # Driver: one process per step, resumable
│   │   └── process.py           # One step in one process: pass the gate first, then run, then record
│   ├── diagnostics/             # Descriptive diagnostics after M8 (§10.2): access (records and issues), computing the four tables, inputs (reads files), run (driver)
│   ├── deepfm/                  # Stage 2, M9 (§16): DeepFM as an alternative ranker. Torch side; V folds only; inputs, cache, model, loss, train, score, data, trial, benchmark, records, process, study, study_run, compare
│
├── scripts/                     # Command-line entry points; they only "read arguments → call src → write results"
│   ├── m0_preprocess.py
│   ├── m0_build_queries.py      # Query list and labels of each fold → artifacts/queries/, artifacts/labels/
│   ├── m0_benchmark_twotower.py # --device mps / cpu: 200 steps of the placeholder model, fetching batches and computing timed separately (§1.4)
│   ├── m2_recall_hpo.py         # --channel r1 ... r5
│   ├── m3_build_twotower_data.py   # Training arrays of the base window → artifacts/twotower/data/base/
│   ├── m3_twotower_base.py
│   ├── m4_union_check.py
│   ├── m5_verify_locks.py          # --channel r1_popularity ...: the lock files reproduce the recorded scores on D1 (one process per channel)
│   ├── m6_generate_candidates.py   # --fold F1 [--channel ...]: generates the candidates of one fold with the locked configuration (reads no label)
│   ├── m6_monitor_candidates.py    # --fold F1: monitors the candidates of one fold (reads labels for F and V weeks, not for T weeks)
│   ├── m6_run_folds.py             # --first-fold F1 --last-fold T4: runs the folds one by one in time order, one process per step
│   ├── m6_build_features.py        # --fold F1: generates the wide-table features of one fold (one process per fold, reads no label)
│   ├── m7_sample.py                # --fold F1: negative sampling of one training week (one file each for r = 40 / 60 / 80)
│   ├── m7_build_dataset.py         # --target V1 --ratio 40: the binary Dataset of one target week
│   ├── m7_train_eval.py            # One trial: one parameter set trained and evaluated on one V week (one process per trial)
│   ├── m7_rrf_baseline.py          # --fold V1: RRF baseline
│   ├── m7_speed_test.py            # The speed test before the grid (V1, default configuration, 300 trees)
│   ├── m7_ranker_hpo.py            # --stage baseline / grid / minus_r6 / importance / report
│   ├── m8_final_eval.py            # --stage rehearsal / formal / report
│   └── m8_step.py                  # One step of M8 (monitor / baseline / sample / dataset / trial), one process
│   (there is also m8_diagnostics.py: the descriptive diagnostics after M8, not part of the M8 run, see §10.2)
│   (stage 2, §16: m9_deepfm_benchmark.py --device cpu / mps, and m9_deepfm_trial.py: one DeepFM training for one V fold)
│
├── tests/
│   ├── conftest.py              # Tests run in two processes: the torch side and the LightGBM side (--lightgbm-side)
│   ├── fixtures/                # Hand-built small data (a few dozen rows), with results that can be computed by hand
│   ├── unit/                    # Unit tests of every module
│   └── integration/             # Runs one fold from start to end on small data
│
├── notebooks/                   # Exploration and plotting only; no logic that src/ depends on
│
├── data/                        # Not in git
│   ├── raw/                     # Raw CSV: articles / customers / transactions_train / sample_submission
│   └── processed/               # Files generated by M0; and the two directories of precomputed embeddings (read-only)
│       ├── transactions.parquet           # One row per purchase record: customer_idx, article_idx (int32), t_dat_day (int16), price (float32), sales_channel_id (int8); sorted by (t_dat_day, customer_idx), row groups of 1 million rows
│       ├── customers.parquet              # One row per user: customer_idx, age (int8), age_bucket, fn, active (int8 0/1), club_member_status, fashion_news_frequency
│       ├── articles.parquet               # One row per article: article_idx, product_code and the category columns of §8.4
│       ├── customer_id_map.parquet        # customer_idx (int32) ↔ customer_id (64-character string)
│       ├── article_id_map.parquet         # article_idx (int32) ↔ article_id (10-character string)
│       ├── embeddings/                    # Embeddings reordered by article_idx (generated by M0)
│       │   ├── image_by_article_idx.npy   # float32 [105542, 512]; rows without a vector are 0
│       │   ├── image_mask.npy             # bool [105542], True = has an image vector (False for 442)
│       │   ├── text_by_article_idx.npy    # float32 [105542, 384]
│       │   └── text_mask.npy              # bool [105542], all True
│       ├── metadata.json                  # day_zero, age buckets, hash of the preprocess config, row counts of the tables; validated by downstream readers
│       ├── image_features/      # CLIP ViT-B/32, covering 105,100 / 105,542 articles (442 missing)
│       │   ├── image_embeddings.npy           # float32 [105100, 512], already L2-normalized
│       │   ├── image_article_ids.parquet      # embedding_row (int64, 0..N-1) → article_id (10-character string)
│       │   ├── image_embedding_failures.parquet  # Empty table (0 rows)
│       │   ├── image_features_manifest.json   # Model version, sha256, verification record
│       │   └── image_faiss.index              # FAISS IndexFlatIP, not used by this project
│       └── text_features/       # all-MiniLM-L6-v2, covering all 105,542 articles
│           ├── text_embeddings.npy            # float32 [105542, 384], already L2-normalized
│           ├── text_article_ids.parquet       # embedding_row → article_id (plus text, token_count, was_truncated)
│           ├── text_features_manifest.json
│           └── text_faiss.index               # FAISS IndexFlatIP, not used by this project
│
├── artifacts/                   # Not in git; everything can be regenerated by code
│   ├── queries/{fold}.parquet   # query_id, fold, customer_idx, anchor (generated by M0, frozen once generated)
│   ├── labels/{fold}.parquet    # query_id, fold, article_idx (only eval/ and ranker/sampling.py may read it)
│   ├── candidates/{fold}/{channel}.parquet        # channel = r1_popularity ... r6_twotower; frozen once generated, never overwritten
│   ├── candidates/{fold}/r1_bucket_scores.parquet # The complete bucket score table of R1 (age_bucket, article_idx, score), for the features
│   ├── features/{fold}/part-*.parquet             # Wide table: query_id, article_idx + 71 features; one file per chunk of queries
│   ├── features/{fold}/features.json              # Written last and marks the fold as complete: sha256 and rows of every part, column list, rows checked
│   ├── features/category_codes.json               # Code table of the categorical features, shared by all folds
│   ├── ranker/samples/{experiment}/{fold}/r{r}.parquet   # Training rows after negative sampling (+ a .json record of the same name); experiment = full / minus_r6
│   ├── lgb_datasets/{experiment}/{target week}_r{r}/           # train.bin (LightGBM binary dataset) and dataset.json
│   ├── models/wals/{fold}/                        # R5 vectors (format of vectors.py): the query users of the fold, all articles that have a vector
│   ├── models/twotower/vectors/{fold}/            # R6 vectors: the query users of the fold, all articles of the catalog; the model before fine-tuning
│   ├── models/twotower/chain_locked/{fold}.pt     # Training chain of the locked configuration: the model after fine-tuning on the fold (it predicts the next fold)
│   ├── models/twotower/{base,chain}/              # The epochs of the base; the models of the D-phase training chains
│   ├── models/ranker/{area}/{experiment}/{config_id}/{fold}/model.txt   # area = v_phase / speed_test
│   ├── predictions/{area}/{experiment}/{config_id}/{fold}.parquet       # Top-12 of every checkpoint; also {fold}_per_query.parquet
│   ├── hpo/D/{channel}/{trial_id}/{fold}.parquet  # R2 / R3 candidates + the in_active_pool flag (used by §5.7 item 4 in M4)
│   ├── m5/lock_check/{fold}/{channel}.parquet     # Candidates generated by the M5 lock check; later steps do not read this directory
│   └── runs/{run_id}/           # Config snapshot, logs and metrics of every run
│
└── results/                     # In git: small summary tables
    ├── d_phase.csv
    ├── d_phase_steps/{channel}.csv   # All candidates of every step, whether they pass, weekly / pooled / group scores, the outcome of the selection
    ├── m5_lock_check.csv             # M5 lock check: one row per run (the two scores, whether they agree, run time, sampled peak and true peak)
    ├── m6_runs.csv                   # M6: one row per process (fold, channel or monitor, run time, sampled peak, true peak)
    ├── m6_candidates.csv             # M6 monitoring: one row per fold for each channel and for the six-channel union (T weeks have no label-based columns)
    ├── m6_twotower_chain.csv         # M6: sha256 of every model of the training chain and the rows, loss and run time of its fine-tuning
    ├── m6_features.csv               # M6: missing rate, minimum, maximum and mean of every feature of every fold
    ├── v_phase.csv                   # M7: one row per trial (run time, sampled peak, true peak, LightGBM version, libomp path and version)
    ├── v_phase_metrics.csv           # M7: one row per (configuration, fold, number of trees); the rows of the RRF baseline are here too
    ├── v_phase_steps.csv             # M7: k, scores, whether it passes and the outcome of the selection for every configuration of every step
    ├── v_phase_feature_importance.csv  # M7: gain of the selected configuration (for analysis only)
    ├── m7_runs.csv                   # M7: run time, true peak, LightGBM version, libomp path and version of the sampling, Dataset and baseline processes
    ├── m7_speed_test.csv             # Trial rows of the M7 speed test (same columns as v_phase.csv)
    ├── m7_speed_test_metrics.csv     # Metric rows of the M7 speed test
    ├── m8_rehearsal.csv              # Trial rows of the M8 rehearsal (same columns as v_phase.csv)
    ├── m8_rehearsal_metrics.csv      # Metric rows of the M8 rehearsal
    ├── m8_rehearsal_recall.csv       # Recall monitoring rows of the M8 rehearsal
    ├── m8_rehearsal_check.csv        # M8: the outcome of each comparison of a rehearsal with M7 (passed / failed) and the sha256 of the sample files
    ├── m8_runs.csv                   # M8: run time and true peak of the sampling, Dataset, baseline and monitoring processes
    ├── t_diag_label_access.csv       # Diagnostics: time, git commit and purpose of every re-read of T-week labels
    ├── t_diag_*.csv                  # The four tables of the diagnostics: decomposition / concentration / query_groups / bootstrap
    ├── t_phase_label_access.csv      # M8: time and git commit of the first read of T-week labels
    ├── t_phase.csv                   # M8: one row per trial of the T weeks
    ├── t_phase_metrics.csv           # M8: metrics of every (system, T week)
    ├── t_phase_recall.csv            # M8: Recall@quota of each channel, union recall and the zero-hit share of the T weeks
    └── t_phase_goals.csv             # M8: the verdict on G1–G3
```

### 14.1 Layering rules

```
scripts/  →  src/hmrec/{recall, fusion, features, ranker, hpo, final, diagnostics, deepfm, pipeline}  →  src/hmrec/{data, eval, vectors, locks, csv_records, run_records, feature_files, git_state}  →  config / paths
```

- An upper layer may call a lower layer; **a lower layer may not import an upper layer**. For example, `data/` may not import `recall/`.
- `final/` (M8) is on the same level as `hpo/`. It may import `eval/`, `ranker/`, `fusion/` and the bottom-layer modules, as well as `hpo/v_records.py` (reading and writing result tables) and the steps of `hpo/v_trial.py` that **do not depend on which week is predicted** (`plan_dataset_for_folds`, `train_planned_model`, `score_and_save`, `rrf_baseline_metric_row`: the training weeks or the evaluator are passed in by the caller, and these functions do no fold check and make no selection themselves). It may not import the modules that tune and select (`hpo/v_phase.py`, `hpo/v_lock.py`, `hpo/speed_test.py`, `hpo/runner.py`, `hpo/d_phase.py`), `recall/` or torch. Conversely, the code of M6 and M7 does not import `final/`, `eval/final_eval.py` or `ranker/final_window.py` (checked by `tests/unit/test_final_code_rules.py`).
- `diagnostics/` (the descriptive diagnostics after M8, §10.2) is on the same level as `hpo/` too. It may import `eval/`, `fusion/`, `data/`, the bottom-layer modules, `hpo/v_records.py`, `hpo/selection.py` (the tolerance constant), `ranker/locked.py`, `ranker/runtime.py`, `ranker/ablation.py` and `final/goals.py` (the column names of the verdict table); it may not import code that trains, samples or scores, the other modules of `final/`, `eval/final_eval.py`, `recall/`, `features/` or torch. No other code imports it (checked by `tests/unit/test_diag_code_rules.py`).
- `deepfm/` (stage 2, M9; §16) is on the same level as `hpo/` and on the torch side. It may import `eval/`, `ranker/window.py`, `ranker/sampling.py`, `ranker/ablation.py`, `hpo/selection.py` and the bottom-layer modules; it may not import lightgbm or any module that does, and no code of M0–M8 imports it (checked by `tests/unit/test_deepfm_code_rules.py`).
- `recall/`, `features/` and `ranker/` do not import one another and hand over through files in `artifacts/`.
- `fusion/` is outside the restriction of the previous rule: both `features/` and `ranker/` may import `fusion/`, and `fusion/` imports none of the three. The wide table, the recomputation for −R6 and the RRF baseline therefore use the same set of expressions.
- `git_state.py` (HEAD commit, whether the working tree is clean) belongs to the bottom layer and depends only on the standard library and `logging_utils.py`.
- The code of M7 (`ranker/`, `hpo/v_*.py`, `hpo/speed_test.py`, `eval/ranker_eval.py`, `scripts/m7_*.py`) may not import torch, nor any module that loads torch (`recall/`, `hpo/runner.py`): the two copies of the OpenMP runtime of torch and LightGBM cannot be in the same process (§9.7).
- `locks.py` belongs to the bottom layer (depending only on `config.py` and the standard library): it only reads and validates the lock files of `configs/locks/`. Both `recall/` and `features/` may import it (when a feature needs a locked parameter of some channel it takes it from here, not through `recall/`). `ranker/` may import it too. **Building** the channels from the lock files is in `recall/locked.py`; **building the ranker configuration** is in `ranker/locked.py` (the only entry point through which M8 obtains the ranker configuration, §9.6). `ranker/identity.py` (`model_config` and `config_id`) is in `ranker/`, and `hpo/` imports from it, because `ranker/` may not import `hpo/`.
- `csv_records.py` (reading and writing the append-only result tables with a fixed header) and `run_records.py` (run records with one row per process: run time, sampled peak, true peak) belong to the bottom layer and depend only on the standard library and `memory.py`.
- `feature_files.py` (the record of the feature files and its verification) belongs to the bottom layer and depends only on the standard library, polars and `locks.py`: `features/` writes the record and `ranker/` verifies it, and the two do not import each other.
- `pipeline.py` (runs the steps of one fold one after another, each in its own process; the summary of each fold) belongs to the upper layer, on the same level as `recall/` and `hpo/`: it may import `eval/` and the bottom-layer modules, and `recall/`, `features/` and `ranker/` may not import it.
- `vectors.py` belongs to the bottom layer (depending only on numpy and the standard library), and both `recall/` and `features/` may import it: after training, R5 / R6 use it to save the vectors to `artifacts/models/{wals,twotower}/{fold}/`, and `wals_dot` and `tt_dot` of §8.3 use it to read them and compute the dot products. Recall scores and dot-product features are computed by the same function, so the R5 `score` and the `wals_dot` of a (user, article) are exactly the same.
- Every operation that reads historical data by time must go through `data/asof.py`.

### 14.2 Data handed over between modules

| Producer | File | Main columns | Consumer |
|---|---|---|---|
| `data/queries.py` | `artifacts/queries/{fold}.parquet` | `query_id, fold, customer_idx, anchor` | all |
| `data/queries.py` | `artifacts/labels/{fold}.parquet` | `query_id, fold, article_idx` | `eval/`, `ranker/sampling.py` |
| `recall/generate.py` (six channels) | `artifacts/candidates/{fold}/{channel}.parquet`, with `channel` = `r1_popularity` … `r6_twotower` | `query_id, article_idx, channel, rank, score` (R1 also has `pop_is_fallback`); validated against each channel's `OutputSpec` before writing, never overwritten once generated | `fusion/union.py`, `eval/monitor.py` |
| `recall/generate.py` (R1) | `artifacts/candidates/{fold}/r1_bucket_scores.parquet` | `age_bucket (str), article_idx (int32), score (float32)`: the `S(g, i)` of every (age bucket, article) in `[a − 7d, a)`, **not limited by the Top-100 or the active pool**; the `score` of the non-fallback candidates of R1 is bit-identical to it | `features/dense.py` (`pop_bucket_score`) |
| `recall/generate.py` (R5, through `vectors.py`) | `artifacts/models/wals/{fold}/`: `user_customer_idx.npy`, `user_vectors.npy`, `item_article_idx.npy`, `item_vectors.npy`, `vectors.json` (source, fold) | User vectors are saved only for the query users of the fold (those without a vector do not appear), and all item vectors are saved; the `score` of the R5 candidates is bit-identical to the dot product of these vectors | `features/dense.py` (`wals_dot`, through `vectors.py`) |
| `recall/generate.py` (R6, through `vectors.py`) | `artifacts/models/twotower/vectors/{fold}/` (the same 5 files) | User vectors: the query users of the fold; item vectors: **all articles of the catalog** (the candidates of R2 may be outside `A(a)`); computed with the model that predicts this week (**before** this week's fine-tuning); the `score` of the R6 candidates is bit-identical to the dot product of these vectors | `features/dense.py` (`tt_dot`, through `vectors.py`) |
| `recall/generate.py` (R6 training chain) | `artifacts/models/twotower/chain_locked/{fold}.pt`, with the sha256 recorded in `results/m6_twotower_chain.csv` | The model after fine-tuning on the fold; T4 has none | R6 of the next fold (sha256 checked before loading) |
| `fusion/union.py` + `features/build.py` | `artifacts/features/{fold}/part-*.parquet`, `features.json` | `query_id, article_idx` (int32) and the 71 features of §8 (float32; the 8 categorical features as int16 codes), **with no label column**; sorted by `(query_id, article_idx)`, a query never spanning two files; `features.json` records the sha256 and rows of every part and the column list | `ranker/` (verified first with `feature_files.verify_feature_files`, then read; the label is joined in `ranker/sampling.py`) |
| `features/categories.py` | `artifacts/features/category_codes.json` | The list of values of every categorical field (position + 1 = code, 0 = missing), shared by all folds | `ranker/` (when interpreting the category codes) |
| `ranker/sampling.py` | `artifacts/ranker/samples/{experiment}/{fold}/r{r}.parquet` (+ `.json`) | `query_id, article_idx, label` (int8) and the feature columns of the experiment (71, or the 67 of −R6); only queries with at least 1 positive; sorted by `(query_id, article_idx)` | `ranker/dataset.py` (sha256 checked first) |
| `ranker/dataset.py` | `artifacts/lgb_datasets/{experiment}/{target week}_r{r}/train.bin`, `dataset.json` | Binned data, labels and groups of the 4 training weeks; `dataset.json` records the sha256 of the 4 sample files and the binning parameters | `hpo/v_trial.py` (verified first, then trained) |
| `hpo/v_trial.py` (using `ranker/predict.py`) | `artifacts/predictions/{area}/{experiment}/{config_id}/{fold}.parquet`, `{fold}_per_query.parquet` | Top-12: `num_trees, query_id, rank, article_idx, score`; metrics of every query: `query_id, fold, map@12, recall@12, ndcg@12, num_trees` | `scripts/checks/check_ranker_eval.py`, analysis |

Labels are read only in `eval/` and `ranker/sampling.py`; **recall and feature code never reads labels**. (The descriptive diagnostics after M8 read labels through `eval/diagnostic_labels.py`, and a re-read of the labels of a T week has to be recorded first, see §10.2.) The labels of the T weeks have one more layer: they can be read only through `eval/final_eval.py`, with the permit issued by `final/gate.py` (§10.1).

---

## 15. Code conventions

The reader is a junior MLE, so **readability comes before cleverness**. The full rules are in `AGENTS.md`; the main points are listed here:

1. **Language and tools**: Python 3.12, with `uv` managing the environment and locking the dependencies (`uv.lock`); Polars for tables, NumPy / SciPy for matrices, PyTorch (MPS) for Two-Tower, LightGBM for the ranker; `ruff` for formatting and lint, `pyright` (`standard` mode) for type checking, `pytest` for tests. Consistent with §1.2.
2. **Each function does one thing** and is normally no more than 50 lines; each file corresponds to one clear component.
3. **Type annotations everywhere**; every public function has a docstring that states the inputs, the outputs, **the grain of the tables (what one row stands for)** and **the time window**.
4. **Comments say "why"**, not only "what". In particular: whether a time window is open or closed, why this dtype is used, why something is chunked.
5. **All numbers go into the config**; no unexplained constant (magic number) appears in the code.
6. **Computation and IO are separate**: a computation function receives DataFrames / arrays and returns DataFrames / arrays; files are read and written only in `scripts/` and a few IO functions. This way computation functions can be tested directly on small data.
7. **Leakage prevention is written as assertions**: every as-of function checks that the latest date of its input data is before the anchor, and raises an error otherwise.
8. **Use a logger, not print**; a long task prints progress, run time and memory peak.
9. **Every module has unit tests**, with test data small enough to compute by hand.
10. Code comments and docstrings are in English, with words as simple as possible; the design documents (`docs/`) were in Chinese up to tag `v0.1` and are in English from 2026-11-19 on (`AGENTS.md` §11).

---

## 16. DeepFM as an alternative ranker (stage 2, M9)

**Purpose.** To understand the difference between a pointwise deep model and a listwise GBDT on the same inputs. **Revised on 2026-11-20 (§16.8)**: the comparison is now 2×2, {LightGBM, DeepFM} × {listwise, pointwise}, so that the model family and the loss are not mixed up; where §16.2–§16.5 and §16.8 differ, §16.8 holds. **Replacing the LightGBM ranker is not the goal**: the locked system of §9 and its results stay as they are, and nothing of this section changes a lock file, an existing result or an existing artifact.

### 16.1 The controlled comparison (fixed)

| Item | Value |
|---|---|
| Candidates | The frozen candidates of M6 (the same as for LightGBM) |
| Features | The 71 locked features of §8, read from the frozen wide tables after `feature_files.verify_feature_files` |
| Training window | The 4 folds before the target week (§9.2) |
| Training rows | **The same rows as LightGBM**: the M7 sample files with 40 negatives per positive (`artifacts/ranker/samples/full/{fold}/r40.parquet`). Each file is used only if its sha256 equals both its own record and the one in the record of the LightGBM Dataset of that target week (`artifacts/lgb_datasets/full/{target}_r40/dataset.json`). Nothing is sampled again |
| Target weeks | V1–V4 only. Every entry point checks the folds with the M7 window function and reads labels through `eval/ranker_eval.load_m7_evaluator`: **a T fold is refused** |
| Scoring | All candidates of the target week (no sampling); Top-12 by `score DESC, article_idx ASC` |
| Main metric | MAP@12 on the Top-12, with the existing evaluation code (`eval/ranker_eval.ranking_scores`); Recall@12 and NDCG@12 are reported too |
| Reference | The recorded LightGBM and RRF results on V (`results/v_phase_metrics.csv`); they are not recomputed |

### 16.2 The default model

- **Fields.** Every feature is one field: 71 fields. The 8 categorical features use their integer codes as they are (0 = missing; the vocabulary of a field is the shared code table `artifacts/features/category_codes.json`).
- **Numeric features: quantile bins.** For each of the 63 numeric features, the bin edges are the quantiles `k / 32` (k = 1 … 31) of its non-missing values **in the training rows of the window** (the 4 sample files together; numpy's linear interpolation, in float64). Equal edges are merged, so a feature with many equal values has fewer bins. Bin of a value x: `1 + (number of edges <= x)`, i.e. 1 … 32; **a missing value (NaN) is bin 0**. The edges are computed from the training rows only and then applied to the target week unchanged. Binned inputs are stored as uint8.
- **Embeddings.** Dimension 16, one vector per (field, value), shared by the FM part and the deep part. Initial values: normal with standard deviation 0.01; first-order weights and the bias start at 0.
- **FM part.** `b + Σ_f w_f[x_f] + ½ Σ_k [ (Σ_f e_{f,k})² − Σ_f e_{f,k}² ]`, where `e_f` is the embedding of field f's value: first-order weights plus the second-order interactions between all pairs of fields.
- **Deep part.** The 71 embeddings concatenated (1,136 values) → 256 → 128 → 64 with ReLU → 1 (dropout after each hidden layer; 0 by default).
- **Output and loss.** `logit = FM logit + deep logit`; pointwise binary cross-entropy on the label (1 = bought in the label week), mean over the rows of a batch.
- **Training.** Adam (default betas), learning rate 1e-3, batches of 4,096 rows in a new random order every epoch, no weight decay, 5 epochs, seed 42. After every epoch the model is saved, the target week is scored and evaluated. The seed fixes the initial weights, the row order and the dropout. **Withdrawn on 2026-11-20 (§16.8)**: the constant learning rate, the 5 epochs and the evaluation after every epoch; step 0 ran with them and its records stay. The training is now the cosine schedule of §16.8.
- **Device.** The pre-registered rule: MPS is used only if a training step on MPS is at least 1.5 times faster than on CPU (6 threads); otherwise CPU. **Deviation (a decision by the user, 2026-11-19): the study runs on CPU**, although the rule gives MPS (1.92 times faster, §16.6), because results on MPS cannot be reproduced from a fixed seed. The decision was made after the device benchmark and the 1-epoch V1 number were seen; it selects no model.

### 16.3 Variants (switchable by config)

- **Components**: `first_order` (bias and first-order weights only), `fm` (first plus second order), `deep` (the MLP only), `full` (the default).
- **Numeric input without binning (`standardised`).** **Redefined on 2026-11-20, before this variant was ever run (§16.8): `g(x) = sign(x) · log(1 + |x|)` is applied first, and everything below is computed on `g(x)` instead of `x`.** For each numeric feature, from the non-missing training values of the window: `lo` and `hi` = the 0.1st and 99.9th percentiles; a value is clipped to `[lo, hi]`; `mean` and `std` are those of the clipped training values; `z = (clip(x) − mean) / std` (`z = 0` if `std = 0`). A missing value has `z = 0` and the indicator `m = 1` (otherwise `m = 0`). The field's embedding is `z · v_f + m · u_f` and its first-order term `w_f · z + c_f · m`, with one learned vector `v_f`, `u_f` and one weight `w_f`, `c_f` per feature. The categorical fields are unchanged. The statistics come from the training rows only.
- **Listwise loss (`listwise_softmax`)** (from 2026-11-20 an arm of the comparison with its own learning-rate step, not an ablation; §16.8). A batch is made of whole queries: the queries are put in a new random order every epoch and packed into batches of at most 4,096 rows (a larger query is a batch of its own). Loss of a query: `− (1 / |P_q|) Σ_{i ∈ P_q} log softmax(s_q)_i`, with `s_q` the logits of the query's rows and `P_q` its positive rows; the batch loss is the mean over its queries. Every query of a sample file has at least one positive.

### 16.4 Outputs

- `artifacts/deepfm/`: the input transform of a training (bin edges or standardisation statistics), the model after every epoch, the Top-12 and the per-query metric values of every epoch.
- `results/deepfm_benchmark.csv` (device benchmark), `results/deepfm_trials.csv` (one row per training: time, true memory peak, device, torch version, commit), `results/deepfm_metrics.csv` (one row per configuration, seed, fold and epoch).
- The true memory peak of every process is recorded and must stay under 12 GB (§1.3).

### 16.5 Rules of the study (recorded before any result existed)

All comparisons and selections are on V1–V4, with the pooled MAP@12 (weighted by queries, as in M7).

**Revised on 2026-11-20 (§16.8), after the step-0 results on V1–V4 were seen and without any T data.** Items 1 (the epoch rule over the checkpoints of one run), 2 (the stop at a seed range above 1%) and 7 (the 10-epoch extension) are **withdrawn**; items 3–5 are replaced by the rules of §16.8; items 6 and 8 stay, with the additions of §16.8. The original text is kept below as the record of what was fixed before step 0.

1. **Epoch rule.** For each configuration: the smallest epoch whose pooled MAP@12 is within 1% of the configuration's best epoch (the number-of-trees rule of §9.4, `hpo/selection.choose_num_trees`).
2. **Step 0, noise.** The default configuration with seeds 42, 43, 44. For each seed the epoch rule gives its pooled MAP@12. If `(max − min) / min` over the three seeds is above 1%, **stop and report before any tuning**.
3. **Tuning, one factor at a time**, with the existing selection rule (§4.2: at least 1% relative gain of the pooled MAP@12 and strictly better in at least 3 of the 4 weeks): learning rate {3e-4, 1e-3, 3e-3}, then embedding dimension {8, 16, 32}, then dropout {0, 0.1, 0.3}. Priority: the value closer to the default first, the smaller value first at equal distance (learning rate and dimension: distance on a log scale). Selection uses seed 42.
4. **Ablations at the selected configuration**, without separate tuning, reported only: the four component variants, the standardised numeric input, the listwise loss.
5. **Final report.** The selected configuration with 3 seeds, and a 95% interval of DeepFM minus LightGBM on V with the customer-level bootstrap of §10.2.
6. **One evaluation on T1–T4** after the configuration is locked on V: run once, reported only, never used to select anything. It reads the labels of the T weeks again; the user has consented, and the read is recorded when it happens (AGENTS.md §3.2). It is not implemented yet.
7. **Epoch cap (added 2026-11-19, before any run of the study).** If the epoch rule selects epoch 5 (the last one) for the final configuration, that configuration is trained for 10 epochs and the epoch rule is applied again. This is done once and only for the final configuration; the ablations and the extra seeds then use 10 epochs too.
8. **Grid edges (added 2026-11-19, before any run of the study).** If a selected value lies on the edge of its grid, the grid is not extended; the value is flagged in the report.

### 16.6 Speed test (before the study)

- **Device benchmark**: the default model at batch 4,096, on CPU (6 threads) and on MPS, each in its own process; 300 training steps, the first 20 not timed. Also: whether two runs with the same seed give identical losses on each device, and the scoring throughput on 1 million rows.
- **One training** on the chosen device: the default configuration, target V1, 1 epoch, evaluated on V1. If its MAP@12 is below the RRF baseline of V1, stop and report; the model is not changed to investigate.
- Reported: time per epoch, scoring time, true peaks, and the projected total of the study (about 60 trainings of 5 epochs). If the projection is above 12 hours, the dropout step is dropped.

**Results of the speed test (2026-11-19; `results/deepfm_benchmark.csv`, `results/deepfm_trials.csv`, `results/deepfm_metrics.csv`).**

| Device | Median training step | Same seed twice: identical losses | Scoring, 1 million rows | True peak (RSS) |
|---|---|---|---|---|
| CPU, 6 threads | 19.50 ms | yes | 727,851 rows/s | 7.71 GB |
| MPS | 10.13 ms | no (largest difference 9.06e-06 over 300 steps) | 566,279 rows/s | 5.48 GB |

- **Device**: MPS is 1.92 times faster per training step, so the device rule gives MPS (`configs/deepfm.yaml`). On MPS a training does not repeat bit for bit from the same seed; on CPU it does.
- **V1, default configuration, 1 epoch, seed 42**: MAP@12 0.027923 (RRF baseline 0.018511, LightGBM 0.030112), Recall@12 0.063982, NDCG@12 0.044275. Above the RRF baseline, so the study may start.
- **Time**: preparing the training rows 34.2 s, one epoch 29.9 s (2,616 steps), scoring and evaluating V1 with one model 40.3 s; true peak 5.34 GB. Projected: at most about 390 s per training of 5 epochs, about 6.5 hours for 60 trainings. The dropout step is kept.

### 16.7 Code

- `src/hmrec/deepfm/` (torch side): `inputs.py` (bin edges, standardisation, encoding), `model.py`, `loss.py`, `train.py`, `score.py`, `data.py` (sample files and wide tables), `trial.py` (one training with its evaluations), `benchmark.py`, `records.py`; scripts `scripts/m9_deepfm_*.py`.
- The package may import `eval/`, `ranker/window.py`, `ranker/sampling.py`, `ranker/ablation.py` (none of which loads LightGBM), `hpo/selection.py` and the bottom-layer modules. **It must not import lightgbm or any module that does** (two OpenMP runtimes in one process abort or hang, §9.7); `tests/unit/test_deepfm_code_rules.py` checks this. Its tests run on the torch side.

### 16.8 Revision of 2026-11-20: the 2×2 comparison (recorded before any run of the revised study)

**Why and when.** Step 0 of §16.5 ran on 2026-11-19 (default configuration, 5 epochs at a constant learning rate, seeds 42, 43, 44; `docs/decisions.md`): all three seeds were best after epoch 1 (pooled MAP@12 0.030704 / 0.030222 / 0.030147) and fell in every later epoch (to about 0.0286 after epoch 5); the seed range at epoch 1 was 1.85%, above the 1% limit, and the study stopped by its rule. The user then revised the study. **The revision was made after seeing these step-0 results on V1–V4 and involved no T data.** It withdraws four earlier rules: 5 epochs at a constant learning rate, the epoch rule over checkpoints, the 10-epoch extension, and the stop at a range above 1%. The step-0 records stay as they are.

**A hypothesis that the new step 0 tests** (not an established cause): the model overfits after the first epoch at a constant learning rate of 1e-3, and the seed range is above 1% only at epoch 1 (it was 0.89%, 0.81%, 0.91%, 0.64% at epochs 2–5) because Adam still takes full steps at the end of that epoch; a learning rate that decays within the run would remove both.

**The four arms.** All arms use the controlled inputs of §16.1 (the same candidates, features, window and `r40` sample files), V1–V4 only.

| | Listwise | Pointwise |
|---|---|---|
| **LightGBM** (CPU, 6 threads, row-wise, learning rate 0.05, no decay) | **Arm 1**: `lambdarank`, the locked configuration, 500 trees. Seed 42 is the M7 record (pooled 0.032713); seeds 43, 44 are run | **Arm 2**: `binary`, the locked configuration with only the objective changed. Seed 42 to 1,000 trees, k* by the number-of-trees rule; seeds 43, 44 to k* trees |
| **DeepFM** (CPU, Adam, cosine schedule) | **Arm 3**: `listwise_softmax` (§16.3), its own learning-rate step, then seeds 42, 43, 44 | **Arm 4**: `pointwise_bce` (§16.2), step 0, tuning, then seeds 42, 43, 44 |

**DeepFM training.** Adam with a step-level cosine schedule: with N the number of optimizer steps of the whole run and t = 0 … N − 1, `lr(t) = lr_min + (lr − lr_min) · (1 + cos(π t / N)) / 2` and `lr_min = 0.01 · lr`. **The schedule length T (epochs) is a hyperparameter, T ∈ {1, 2, 3}.** A run with T epochs has its own schedule over exactly its T epochs and is evaluated once, at the end; no mid-schedule checkpoint is used. For the listwise loss N is counted from the whole-query batches of the T epochs before training starts.

**Rules** (all on V1–V4, pooled MAP@12 weighted by queries):

1. **Epoch rule.** The smallest T whose pooled MAP@12 is within 1% of the best T (`hpo/selection.choose_num_trees` on the curve T → pooled MAP@12).
2. **Step 0 (Arm 4).** The default configuration with seeds 42, 43, 44 and T ∈ {1, 2, 3}; T is chosen on the seed mean. Stop only if the seed-mean pooled MAP@12 at that T is below the RRF baseline (0.019419).
3. **Noise.** The 3-seed range `(max − min) / min` of the default configuration at its selected T. Below 1%: tuning uses seed 42. 1% or more: **no stop**; every tuning candidate of both DeepFM arms is run with seeds 42, 43, 44 and compared on seed means (the mean of the seeds' pooled MAP@12, and per week the mean of the seeds' MAP@12).
4. **Tuning of Arm 4**, one factor at a time, with the selection rule of §4.2: learning rate {1e-3, 3e-4, 3e-3}, then embedding dimension {16, 8, 32}, then dropout {0, 0.1, 0.3} (the first value is the current one, then in priority order). In the learning-rate step every candidate is run for T ∈ {1, 2, 3} and gets its own T by rule 1; the candidates are compared at their own T. The later steps, the ablations and the extra seeds use the T selected together with the learning rate. **Limitation: T is not tuned again after the learning-rate step.**
5. **Arm 3.** The embedding dimension and dropout selected for Arm 4; its own learning-rate step {1e-3, 3e-4, 3e-3} with T ∈ {1, 2, 3} per candidate, as in rule 4; then seeds 42, 43, 44. **Limitation: its embedding dimension and dropout are the ones tuned for Arm 4.**
6. **Ablations** at the selected Arm 4 configuration and T, seed 42, reported only: components `first_order`, `fm`, `deep`; numeric input `standardised` (with the `g(x)` of §16.3). A difference smaller than the default configuration's 3-seed range is marked "within seed noise".
7. **LightGBM seeds.** In the locked parameters two things are random: `feature_fraction: 0.8` and `bagging_fraction: 0.8` with `bagging_freq: 1`. A seed is set as the locked training sets it: the parameter `seed` only (LightGBM derives its sub-seeds from it). The bin boundaries (the M7 binary Datasets) and the negative sampling (seed 42) are the same for every seed and every arm, so **the seed variation excludes sampling**.
8. **Grid edges** (§16.5 item 8): a selected value on the edge of its grid, T and k* included, is flagged and the grid is not extended.
9. **Comparison (report only).** Per arm: pooled MAP@12 per seed, mean, sample standard deviation and range; Recall@12 and NDCG@12 as seed means. The customer-level paired bootstrap of §10.2 (1,000 replicates, seed 42, 95%), on seed 42 and on every query's 3-seed mean AP@12, for: DeepFM-pointwise − LGB-lambdarank, DeepFM-listwise − LGB-lambdarank, LGB-lambdarank − LGB-binary, DeepFM-listwise − DeepFM-pointwise. MAP@12 of the four arms in the three query groups. An independent naive recomputation of MAP@12 for the selected Arm 4 configuration, seed 42.
10. **T phase (recorded now, implemented later).** After the V study is finished: one evaluation on T1–T4 of the three new arms at their selected configurations, seeds 42, 43, 44, reported only and never used to select anything. LGB-lambdarank with seed 42 on T is the M8 record and is not rerun. Until it is implemented, every entry point of M9 refuses T folds. *(Superseded on 2026-11-20 by §16.12: only the configuration chosen on V is evaluated on T.)*

**Speed only.** The encoded inputs (training rows and the target week's candidates) of the binned input are cached per target week under `artifacts/deepfm/cache/` and verified by sha256 before reuse. The device stays CPU.

**Outputs added.** `results/m9_lgb_trials.csv`, `results/m9_lgb_metrics.csv` (LightGBM arms; nothing is appended to `results/v_phase*.csv`); `results/deepfm_study.csv` (what the rules decided); `results/m9_arms.csv`, `results/m9_contrasts.csv`, `results/m9_groups.csv` (the comparison). A cosine run has its own configuration id (it includes the schedule and T), so its rows do not mix with the step-0 rows of 2026-11-19.

**Code added.** `src/hmrec/hpo/lgb_arms.py` with `scripts/m9_lgb_trial.py` and `scripts/m9_lgb_comparison.py` (LightGBM side: the configuration comes from `ranker/locked.py`, the fold guard and the evaluator are M7's); `src/hmrec/deepfm/cache.py`, `study.py`, `study_run.py`, `compare.py` (torch side).

### 16.9 Results of the 2×2 study on V1–V4 (2026-11-20; report only)

Tables: `results/m9_arms.csv`, `results/m9_contrasts.csv`, `results/m9_groups.csv`, `results/deepfm_study.csv`, `results/m9_lgb_*.csv`; the full record is the entry of 2026-11-20 in `docs/decisions.md`. Nothing here changes the locked system, and no T data was read.

| Arm | Selected | Pooled MAP@12 (seeds 42 / 43 / 44) | Mean | Std | Range |
|---|---|---|---|---|---|
| LightGBM lambdarank (Arm 1) | 500 trees (locked) | 0.032713 / 0.032691 / 0.032701 | 0.032701 | 0.000011 | 0.0675% |
| LightGBM binary (Arm 2) | k\* = 500 | 0.032183 / 0.032270 / 0.032256 | 0.032236 | 0.000047 | 0.2702% |
| DeepFM listwise (Arm 3) | 1e-3, T = 1 | 0.031563 / 0.031557 / 0.031580 | 0.031566 | 0.000012 | 0.0728% |
| DeepFM pointwise (Arm 4) | default, T = 1 | 0.031164 / 0.031215 / 0.031043 | 0.031141 | 0.000088 | 0.5526% |

- Step 0: seed-mean pooled MAP@12 0.031141 (T = 1), 0.030921 (T = 2), 0.030556 (T = 3); T = 1 selected; 3-seed range 0.5526%, so tuning used seed 42.
- No tuning candidate of either DeepFM arm passed the selection rule of §4.2; both arms keep learning rate 1e-3, embedding dimension 16, dropout 0. Values on a grid edge (flagged, not extended): T = 1 and dropout 0.
- Paired bootstrap on the seed mean (95%): DeepFM pointwise − LightGBM lambdarank −0.001561 [−0.001750, −0.001399]; DeepFM listwise − LightGBM lambdarank −0.001135 [−0.001313, −0.000972]; LightGBM lambdarank − LightGBM binary +0.000465 [+0.000330, +0.000596]; DeepFM listwise − DeepFM pointwise +0.000426 [+0.000300, +0.000566].
- Ablations at the Arm 4 configuration (seed 42, relative to 0.031164): `first_order` −8.36%, `fm` −2.57%, `deep` −2.37%, numeric input `standardised` +3.59%; none is within the 3-seed range.
- The T-phase evaluation of rule 10 of §16.8 is not implemented.

### 16.10 Extension of 2026-11-20: the input representation (recorded before any run)

**Why and when.** Added after the 2×2 results (§16.9) were seen, because of the seed-42 ablation with the standardised numeric input (+3.59%). V1–V4 only; it selects nothing for the main system and leaves the 2×2 records as they are.

**Hypothesis under test** (not an established cause): the 32-bin encoding gives DeepFM a coarser view of the numeric features than LightGBM has. DeepFM default: 32 quantile bins plus a missing bin. Locked LightGBM (`configs/locks/ranker.lock.yaml`): `max_bin: 255`, `min_data_in_bin: 3`, `bin_construct_sample_cnt: 200000`, `use_missing: true`, `zero_as_missing: false`, `enable_bundle: true`, `feature_pre_filter: false`.

**Runs** (`configs/hpo/deepfm_extension.yaml`): learning rate 1e-3, embedding dimension 16, dropout 0, T = 1 with its full cosine schedule, CPU, seeds 42, 43, 44, V1–V4.

| Run | Numeric input | Loss | Arm |
|---|---|---|---|
| E1 | binned, 255 bins | pointwise | `deepfm_bins255_pointwise` |
| E2 | binned, 255 bins | listwise | `deepfm_bins255_listwise` |
| E3 | standardised (§16.3) | pointwise | `deepfm_standardised_pointwise` (seed 42 = the ablation run of §16.9) |
| E4 | standardised (§16.3) | listwise | `deepfm_standardised_listwise` |

No tuning, no selection. **Limitation: T is fixed at 1; the learning rate, embedding dimension and dropout are those of the 32-bin pointwise arm.**

**Comparison (report only).** Eight arms: the two LightGBM arms and DeepFM with {32 bins, 255 bins, standardised} × {pointwise, listwise}; per arm the seeds, mean, sample standard deviation and range. Paired bootstrap of §10.2 on seed means: standardised − 32 bins and 255 bins − 32 bins for each loss; each standardised arm − the LightGBM arm with the same loss. The three query groups. Tables: `results/m9_ext_arms.csv`, `results/m9_ext_contrasts.csv`, `results/m9_ext_groups.csv`.

**Memory (no result changes).** The standardised path peaked at 11.66 GB; it is lowered before the runs (aim: under 10 GB), with a test on small data that shows identical outputs.

How: the peak was the scoring of one chunk of 262,144 rows (the [rows, fields, dim] tensors of the field embeddings). The chunk size is **not** changed, because the last bits of a matrix product depend on the number of rows in the call (another chunk size moves standardised scores by about 1e-7); instead the field embeddings and the second-order term are built for `scoring.rows_per_block` = 32,768 rows at a time, and the MLP and the first-order term still run once per chunk. The training rows are standardised in place. Before the runs, `scripts/m9_deepfm_rescore.py` scores V1–V4 again with the saved seed-42 model of the ablation and must reproduce its saved Top-12 bit for bit.

**Code added.** `src/hmrec/deepfm/extension.py` with `scripts/m9_deepfm_extension.py` (the runs); `scripts/m9_deepfm_compare.py --extension` (the eight-arm tables); `src/hmrec/deepfm/rescore.py` with `scripts/m9_deepfm_rescore.py` (the check above); `--n-bins` of `scripts/m9_deepfm_trial.py`.

**T phase (replaces rule 10 of §16.8; implemented later).** One evaluation on T1–T4 of LightGBM binary and all six DeepFM arms, seeds 42, 43, 44, reported only and never used to select anything. LightGBM lambdarank with seed 42 on T is the M8 record and is not rerun. *(Superseded on 2026-11-20 by §16.12: only the configuration chosen on V is evaluated on T.)*

### 16.11 Results of the extension on V1–V4 (2026-11-20; report only)

Tables: `results/m9_ext_arms.csv`, `results/m9_ext_contrasts.csv`, `results/m9_ext_groups.csv`; the full record is the results entry of 2026-11-20 in `docs/decisions.md`. Nothing was tuned or selected, the locked system is unchanged, and no T data was read. The numbers are recorded without an interpretation of their cause.

| Arm | Numeric input | Loss | Pooled MAP@12 (seeds 42 / 43 / 44) | Mean | Std | Range |
|---|---|---|---|---|---|---|
| LightGBM lambdarank | LightGBM bins | listwise | 0.032713 / 0.032691 / 0.032701 | 0.032701 | 0.000011 | 0.0675% |
| LightGBM binary | LightGBM bins | pointwise | 0.032183 / 0.032270 / 0.032256 | 0.032236 | 0.000047 | 0.2702% |
| DeepFM listwise | 32 bins | listwise | 0.031563 / 0.031557 / 0.031580 | 0.031566 | 0.000012 | 0.0728% |
| DeepFM pointwise | 32 bins | pointwise | 0.031164 / 0.031215 / 0.031043 | 0.031141 | 0.000088 | 0.5526% |
| E2 | 255 bins | listwise | 0.030540 / 0.030297 / 0.030381 | 0.030406 | 0.000123 | 0.8022% |
| E1 | 255 bins | pointwise | 0.029945 / 0.029947 / 0.029888 | 0.029927 | 0.000034 | 0.1986% |
| E4 | standardised | listwise | 0.032738 / 0.032739 / 0.032818 | 0.032765 | 0.000046 | 0.2450% |
| E3 | standardised | pointwise | 0.032283 / 0.032257 / 0.032189 | 0.032243 | 0.000048 | 0.2912% |

- Paired bootstrap on the seed mean (95%): standardised − 32 bins: pointwise +0.001103 [+0.000952, +0.001261], listwise +0.001199 [+0.001035, +0.001367]; 255 bins − 32 bins: pointwise −0.001214 [−0.001385, −0.001047], listwise −0.001160 [−0.001336, −0.000989]; DeepFM standardised pointwise − LightGBM binary +0.000007 [−0.000142, +0.000165]; DeepFM standardised listwise − LightGBM lambdarank +0.000064 [−0.000104, +0.000238].
- E3 with seed 42 is the ablation run of §16.9 and was not repeated; the other 44 trainings ran at T = 1 without a NaN loss.
- Memory of the standardised path with the changes of §16.10: true peak 9.08–11.36 GB over 20 trainings (median 10.75 GB), against 11.38–11.66 GB before. The aim of under 10 GB was not reached; the limit of 12 GB was not exceeded. The rescore check reproduced all saved Top-12 rows of V1–V4 bit for bit.
- Limitation, as recorded: T is fixed at 1 and nothing is tuned for the new inputs. The T-phase evaluation of §16.10 is not implemented.

### 16.12 The single evaluation on T1–T4 (2026-11-20; recorded before any run)

**This section replaces its first version** (commit `c2abcb7`), which planned seven arms on T: LightGBM binary and six DeepFM arms. The scope was narrowed by the user **before any T label was read for this step**; the earlier T-phase notes (rule 10 of §16.8, the T-phase paragraph of §16.10) are superseded too. **Reason: T1–T4 is the final test set, not a second validation set. Only the configuration chosen on V is tested on T; the other arms would add looks at T that no conclusion needs.** The full record is the entry of 2026-11-20 in `docs/decisions.md`.

**Run once, reported only; nothing may be changed after the T result is seen.** The user consented to this read of the T labels (AGENTS.md §3.2). Config: `configs/m9_t_eval.yaml`.

**The chosen configuration**: DeepFM with the standardised numeric input and the listwise loss, learning rate 1e-3, embedding dimension 16, dropout 0, schedule length T = 1; configuration id `f8811fb59b79` (arm `deepfm_standardised_listwise` of §16.11). It was chosen on V as the DeepFM arm with the highest seed-mean pooled MAP@12 among the six arms (0.032765), after the input-representation extension. **It is a choice made on V, not an outcome of the one-factor selection rule of §16.5 / §16.8.**

**Not tested on T**: the effects of the input representation and of the loss; they remain findings on V only. The explanation for why binning does worse (§16.10) is a hypothesis and was not tested.

**What is run**: the chosen configuration with seeds 42, 43, 44 on T1–T4, 12 trainings. Training windows as in M8 (§10.1): T1 ← V1–V4, T2 ← V2 V3 V4 T1, T3 ← V3 V4 T1 T2, T4 ← V4 T1 T2 T3. No new sampling: the sample files of M8 are reused after their sha256 checks. LightGBM lambdarank is the M8 record (500 trees, seed 42), read only. No other arm is run on T.

**Order.** Code and tests on small data → commit → rehearsal → formal run → comparison.

1. **Rehearsal** (`scripts/m9_t_eval.py --stage rehearsal`; reads no T label): the new code path with V4 as the target week, seeds 42, 43, 44. Each result must be bit-identical to the V4 record of the configuration and seed (the three metrics, the number of queries and the training loss; the saved Top-12 with its scores; the per-query values). Outcome: `results/m9_t_rehearsal_check.csv`.
2. **Gate** (`src/hmrec/t_eval/gate.py`): T labels are read only if the latest rehearsal passed at the commit of HEAD on a clean working tree. Before any T label is opened a row is appended to `results/m9_t_label_access.csv` (time, commit, purpose, script). An interrupted formal run is continued with the same command and refused at another commit.
3. **Formal run** (`scripts/m9_t_eval.py --stage formal`): 12 trainings, one process each; `results/m9_t_trials.csv`, `results/m9_t_metrics.csv`.
4. **Comparison** (`scripts/m9_t_compare.py`; saved per-query values only): MAP@12 per week and pooled for every seed with mean, standard deviation and range, next to the M8 record; the paired customer-level bootstrap of §10.2 for DeepFM (seed mean) minus LightGBM lambdarank; V and T side by side for the two models. `results/m9_t_arms.csv`, `results/m9_t_contrasts.csv`, `results/m9_t_v_and_t.csv`.

**A third entry point for T labels** (besides §10.1 and §10.2): `src/hmrec/eval/t_eval_labels.py` with a `TEvalPermit` that only `src/hmrec/t_eval/gate.py` creates; only `src/hmrec/t_eval/` and `scripts/m9_t_*.py` may import it (`tests/unit/test_t_eval_code_rules.py`). The other entry points and every fold guard of M6, M7 and of the V-phase code of M9 are unchanged.

**Stop conditions.** The rehearsal fails; a loss becomes NaN or a training diverges; a true memory peak above 12 GB; any check or test fails.

### 16.13 Result of the evaluation on T1–T4 (2026-11-20; report only)

Run once at commit `db8e9c7`; tables `results/m9_t_*.csv`; the full record is the results entry of 2026-11-20 in `docs/decisions.md`. Nothing is selected or changed with these numbers. They are recorded without an interpretation of their cause.

- Rehearsal on V4 (seeds 42, 43, 44): passed, 12 of 12 checks bit-identical to the V4 records. T labels were read once, recorded at 2026-11-20T15:55:13 (`results/m9_t_label_access.csv`).

| Model | Seed | MAP@12 T1 | MAP@12 T2 | MAP@12 T3 | MAP@12 T4 | MAP@12 pooled |
|---|---|---|---|---|---|---|
| DeepFM standardised listwise | 42 | 0.035494 | 0.040229 | 0.039568 | 0.042256 | 0.039260 |
| DeepFM standardised listwise | 43 | 0.035674 | 0.040465 | 0.039577 | 0.042591 | 0.039449 |
| DeepFM standardised listwise | 44 | 0.035238 | 0.040335 | 0.039720 | 0.042505 | 0.039313 |
| DeepFM standardised listwise | mean | 0.035468 | 0.040343 | 0.039622 | 0.042451 | 0.039341 |
| LightGBM lambdarank (M8 record) | 42 | 0.034960 | 0.040270 | 0.039673 | 0.042435 | 0.039194 |

- DeepFM pooled MAP@12 over the seeds: standard deviation 0.000097, range 0.4807%; Recall@12 0.087564 and NDCG@12 0.060095 (seed means). M8 record: Recall@12 0.087135, NDCG@12 0.059852.
- Paired customer-level bootstrap (§10.2; 95%): DeepFM (seed mean) − LightGBM lambdarank on T, pooled MAP@12: +0.000147 [−0.000059, +0.000334].
- V and T, pooled MAP@12: DeepFM (seed mean) 0.032765 → 0.039341; LightGBM lambdarank (seed 42) 0.032713 → 0.039194.
- Highest true memory peak 10.72 GB; no NaN loss; no stop condition applied.
- The effects of the input representation and of the loss were not tested on T (§16.12).

## 17. Stage 3: MLOps on GCP (M10–M15)

**The design of stage 3 is `docs/mlops.md`** (approved by the user on 2026-11-21; decisions D1–D12 are in its §14 and in `docs/decisions.md`). This section is only a pointer: where stage 3 is concerned, `docs/mlops.md` is the reference, and this proposal stays the reference for what the system computes.

- **Stage 3 changes how the system runs, not what it computes.** §5–§10 and every lock file stay as they are at tag `v0.2.1`. The weekly pipeline refits model parameters with the locked hyperparameters; it selects and tunes nothing.
- **Milestones**: M10 container and CI, M11 versioning, M12 the weekly pipeline (local), M13 cloud, M14 serving, monitoring and deployment, M15 close (`docs/mlops.md` §13).
- **Files added by M10** (the directory tree of §14 describes stages 1 and 2 and is not rewritten): `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`, the `image-*` targets of the `Makefile`, `src/hmrec/native_libraries.py`, `src/hmrec/platform_report.py`, `scripts/image_info.py`, and `docs/m10_calibration_plan.md` (the plan for measuring the calibration items of `docs/mlops.md` §8.3 on week V4; not run yet).
- **The invariants of `docs/mlops.md` §4 are binding.** In particular: T1–T4 labels are read only by the replay, for monitoring and verification, never to select or tune anything, with thresholds fixed before the first replay tick; the laptop remains the reference for bit-identical checks, and container and cloud results are accepted by the tolerance of `docs/mlops.md` §8.3.
