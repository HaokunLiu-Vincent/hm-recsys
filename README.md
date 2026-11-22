# hmrec — H&M personalized fashion recommendation

**Full report:** [`docs/final_report.md`](docs/final_report.md). This page is the summary (§1 of
the report).

## Summary

**System.** Predict the 12 articles each customer buys next week (Kaggle H&M dataset,
transactions from 2018-09-20 to 2020-09-22); the main metric is MAP@12. Two stages: six recall
channels each return candidates, which are merged into one row per (customer, article); then
71 features and one LightGBM LambdaMART model rank them, and the top 12 are kept.

```
6 recall channels (R1 Popularity, R2 Repurchase, R3 Sibling, R4 ItemCF, R5 WALS, R6 Two-Tower)
  → union + dedup (one row per (query, article))
  → wide feature table (71 features)
  → LightGBM LambdaMART
  → Top-12
```

**Evaluation.** The last 16 weeks of the data are 16 folds in four phases, and each phase selects
only what it is meant to select:

| Phase | Weeks | What happens |
|---|---|---|
| D | D1–D4 | the configuration of the six recall channels is selected, then locked |
| F | F1–F4 | candidates are generated with the locked configuration; training data of the ranker only, nothing is selected |
| V | V1–V4 | the configuration of the ranker is selected, then locked |
| T | T1–T4 | final evaluation with everything locked, **run once** |

Every week is predicted from data before its first day (the anchor) only. The labels of the T
weeks were never read before the final evaluation; the code enforces this.

**Final result** (T1–T4, 2020-08-26 to 2020-09-22, 297,078 queries):

| | T1 | T2 | T3 | T4 | 4 weeks pooled |
|---|---|---|---|---|---|
| **LambdaMART MAP@12** | 0.034960 | 0.040270 | 0.039673 | 0.042435 | **0.039194** |
| RRF baseline MAP@12 (no trained model) | 0.019151 | 0.021471 | 0.021848 | 0.022411 | 0.021154 |
| Union recall of the six channels | 27.88% | 29.08% | 29.09% | 30.69% | — |

Sources: `results/t_phase_metrics.csv`, `results/t_phase_recall.csv`. "Pooled" averages over all
queries of the 4 weeks. The 95% interval of the pooled MAP@12 is 0.038672–0.039721
(`results/t_diag_bootstrap.csv`); it covers the sampling of customers only.

**The three goals** (the criteria were committed in `configs/m8_final.yaml` before any T label was
read; the verdict was written by code to `results/t_phase_goals.csv`):

| Goal | Criterion | Result | Verdict |
|---|---|---|---|
| G1 | pooled MAP@12 on T1–T4 ≥ 0.032 | 0.039194 | **met** |
| G2 | union recall of the six channels ≥ 24% on every one of T1–T4 | lowest week (T1) 27.88% | **met** |
| G3 | LambdaMART MAP@12 strictly above the RRF baseline on at least 3 of the 4 weeks | above on all 4 | **met** |

**Kaggle late submission** (one submission, made on 2026-11-19 after the T1–T4 evaluation; the
procedure was recorded before the run and the score is only reported): Public 0.03548, Private
0.03600 (§4.6 of the report; branch `kaggle-late-submission`). The Private score would rank about
4th of the 2,954 teams on the final leaderboard. **This is not an official rank**: the submission
was made after the deadline, with the public winning solutions available, and the feature design
drew on the first-place solution.

**Stage 2: DeepFM as an alternative ranker** (§10 of the report; the ranker of the system is still
LambdaMART). With the same candidates, features and training rows, the DeepFM configuration chosen
on V and LambdaMART cannot be told apart: pooled MAP@12 0.032765 against 0.032701 on V1–V4 and
0.039341 against 0.039194 on T1–T4 (both 95% intervals of the difference include zero), and Kaggle
Private 0.03588 against 0.03600 (−0.33%, inside the band of 1% fixed before the run; one late
submission, seed 42). "Cannot be told apart" does not mean that the two are equal.

**Stage 3 (in progress): MLOps on GCP.** The same system as a weekly pipeline that runs without a
person: in a container, on a schedule, with versioned inputs, recorded runs, a serving endpoint and
alerts. It changes how the system runs, not what it computes. Design:
[`docs/mlops.md`](docs/mlops.md).

- **M10, container and CI: done (2026-11-22).** One image builds from the repository alone and
  carries its commit and the hash of its code tree; CI builds the x86-64 image on every push and
  runs lint and both test suites inside it. Week V4 was run in the arm64 container and compared
  with the stage-1 records: the largest relative difference on recall or MAP@12 was 0.048%, and
  end to end (the container's own candidates, features and ranking) MAP@12 differed by +0.0054%.
  A container result is accepted when recall and MAP@12 are within 1.0% of the stage-1 record
  (`docs/mlops.md` §8.3, `configs/platforms/linux-arm64.yaml`); the environments differ, so
  bit-level reproduction is not the goal. One week and one platform were measured.
- **M11, versioning: done (2026-11-22).** The pipeline inputs (raw CSV files, processed tables and
  embeddings, the category code table) and the state the replay starts from (the R6 chain model
  after F4, the ranker samples of F1–F4, four weeks of monitoring history) are tracked with DVC
  in a private Cloud Storage bucket: 38 files, 6.16 GB. git holds four pointer files and
  `configs/data_manifest.json` with the sha256 of every file. A clean clone pulled the processed
  data and their sha256 equal the manifest. Every tick of the weekly pipeline will write a run
  manifest (`hmrec.run_manifest`); a local MLflow store shows the recorded runs, and a check
  proves that every metric in it equals its source table.
- **M12, the weekly pipeline (local): done (2026-11-22).** One command runs one week
  (`scripts/weekly.py --anchor DATE --mode eval|serve`): it closes the week before the anchor and
  opens the week of the anchor, each step in its own process, with the anchor date as the only
  clock. Eight weeks (V1–V4, T1–T4) were replayed in the arm64 container, then one `serve` tick
  for all 1,371,980 customers. Against the stage-1 records every week is inside 1.0%: channel and
  union recall within 0.04%, MAP@12 of a week within 0.74% (T4), the pooled MAP@12 of the eight
  weeks 0.035948 against 0.035933 (+0.042%); feature rows equal the deduplicated candidates and
  no key repeats, in every tick. An `eval` tick takes 14 to 15 minutes (peak 7.25 GB), the
  `serve` tick 86 minutes (peak 6.15 GB). The monitoring rules were fixed before the first tick
  and raised 5 flags and no alert. The nine runs are in the local MLflow store, the model
  registry mirrors the weekly models, and the check reports 0 differences. T1–T4 labels were
  read by the replay only, to verify the pipeline; nothing was selected or tuned, and these
  numbers are not a new evaluation. Measurements: `docs/mlops.md` §8.5.
- **M13–M15** (cloud, serving and monitoring, close): not started.

**Three things to know when reading these numbers:**

1. **T is 19.8% higher than V** (V1–V4 pooled: 0.032713, `results/v_phase_metrics.csv`), contrary
   to the prior expectation that "V is optimistically biased and T may fall below 0.032". §5 of
   the report uses four descriptive tables to give an explanation consistent with the numbers; it
   is a correlation, not a verified cause. Single-week MAP@12 over V1–T4 ranges from 0.030112 to
   0.042435, and the results carry a seasonal component (§4.4, §7.2 of the report).
2. **One run, one seed, offline evaluation only.** There is no online A/B test, and no repeated
   runs to give a range of variation (§7.2 of the report).
3. **T1–T4 is used data.** Its labels were read for the final evaluation and on 4 more occasions
   afterwards; every occasion is recorded (§7.2 item 5 of the report). Any later change may only be
   compared and selected on V1–V4 (`AGENTS.md` §3.2).

## How this project was built

The author designed the system, set the selection rules, reviewed every result and made every
decision recorded in [`docs/decisions.md`](docs/decisions.md). Most of the code was written by
Claude Code, working under the rules in [`AGENTS.md`](AGENTS.md). Commits that Claude made itself
carry a `Co-Authored-By` trailer; commits made by the author carry none, also where they contain
code that Claude wrote.

## Documents

The only design reference is [`docs/proposal.md`](docs/proposal.md). Design changes are logged in
[`docs/decisions.md`](docs/decisions.md). Results and their limitations are in the final report,
[`docs/final_report.md`](docs/final_report.md). The design of stage 3 (MLOps on GCP, in progress)
is [`docs/mlops.md`](docs/mlops.md). Rules for coding agents are in [`AGENTS.md`](AGENTS.md).

The documents are in English only; no translation is maintained (AGENTS.md §11). The proposal, the
decision log and `AGENTS.md` were written in Chinese up to tag `v0.1` and were then translated.
One Chinese file is kept, `docs/decisions.zh-CN.md`: a frozen archive of the decision log as it
was at `v0.1`, because the lock files under `configs/locks/` quote its entry titles and lock files
are never modified (`docs/decisions.md`, 2026-11-21, gives the English entry for each title).

## Install

Target machine: macOS on Apple Silicon (PyTorch MPS; no CUDA).

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), for example:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create the environment. uv reads `.python-version` (3.12), downloads that Python if needed,
   and installs the exact versions from `uv.lock` into `.venv`:

   ```bash
   make setup
   ```

3. Install the OpenMP runtime that LightGBM needs (the ranker, M7 and later):

   ```bash
   brew install libomp
   ```

   The LightGBM wheel does not contain an OpenMP runtime; on macOS it loads Homebrew's
   `libomp.dylib` (`/opt/homebrew/opt/libomp/lib`). **This is a system dependency outside
   `uv.lock`**: `make setup` neither installs nor pins it, and `brew upgrade` can change it
   without any change in this repository. Version in use when the ranker was built:
   **libomp 22.1.4** (`brew list --versions libomp`). Every M7 result row records the
   LightGBM version and the path and version of the libomp that the process really loaded
   (columns `lightgbm_version`, `libomp_path`, `libomp_version`). After upgrading libomp,
   repeat the "two trainings give the same model" check before producing new results
   (`docs/decisions.md`, 2026-11-18).

   torch brings its own copy of libomp, and two OpenMP runtimes in one process abort, hang
   or crash. So ranker code never imports torch, and the tests run in two processes.

Other commands (all run through `uv run`):

```bash
make lint         # ruff format --check, ruff check, pyright
make test-fast    # unit tests (two processes: the torch side, then the LightGBM side)
make test         # unit + integration tests (also two processes)
make format       # auto-format
```

**Container image (stage 3, `docs/mlops.md` §8).** The same code in a Linux image, built only
from a clean, committed tree; the image carries its commit and holds no data. Results computed in
a container are not bit-identical to the laptop (another OpenMP runtime and BLAS) and are accepted
by the tolerance of `docs/mlops.md` §8.3.

```bash
make image-build  # build hmrec:dev from HEAD (refuses uncommitted changes)
make image-info   # commit, Python, libraries, OpenMP, BLAS, CPUs and memory of the image
make image-lint   # `make lint` inside the image
make image-test   # `make test` inside the image (two processes)
make image-r6-repeatability ARGS="--setting implemented --rows 8192"
                  # two R6 fine-tunes of synthetic data in fresh processes, compared bit by bit
make image-calibrate ARGS="--run run1 --item recall --channel r1_popularity"
                  # one item of the M10 calibration of week V4 (docs/m10_calibration_plan.md);
                  # inputs are mounted read-only, outputs go to artifacts/m10_calibration/
                  # and results/m10_calibration/
```

GitHub Actions (`.github/workflows/ci.yml`) builds the linux/amd64 image on every push and pull
request and runs lint and both test suites inside it.

**The weekly pipeline (stage 3, `docs/mlops.md` §6).** One command runs one week: it closes the
week before the anchor (ingest, evaluate last week's predictions, sample, fine-tune the Two-Tower
chain) and opens the week of the anchor (recall, features, train the ranker, score, publish,
record). The anchor date is the only clock: the first step cuts the transactions at the anchor and
every later step reads only that cut.

```bash
make image-build                                            # once, from a clean committed tree
make image-weekly ARGS="--anchor 2020-07-29 --mode eval"    # one tick in the container
make image-weekly ARGS="--anchor 2020-09-23 --mode serve"   # all 1,371,980 customers, in shards
make image-replay-compare                                   # the closed weeks against their stage-1 records
```

- `eval` mode predicts for the customers who buy in the week (only then can a week be compared
  with the stage-1 records); `serve` mode predicts for every customer and removes a shard's
  features after it is scored.
- A tick stops at the first failed step or check and says why in its run manifest; the same command
  again skips the finished steps. A tick refuses to start if the tick before it has not finished.
- The container sees only the DVC-tracked inputs (`data/processed`, `data/bootstrap`, the category
  code table), read-only. State: `artifacts/pipeline/` (`state/current.json`, `weeks/<anchor>/`,
  `predictions/<anchor>/`, `runs/<run-id>/manifest.json`). Result tables: `results/m12_replay/`.
- **The replay** is nine ticks in order: `eval` at 2020-07-29, 08-05, 08-12, 08-19 (V1–V4) and
  2020-08-26, 09-02, 09-09, 09-16 (T1–T4), then `serve` at 2020-09-23. Keep the laptop awake
  (`caffeinate -i make image-weekly ...`); a tick takes about a quarter of an hour, the `serve`
  tick longer (measured times: `docs/mlops.md` §8.5).
- The monitoring rules are `configs/m12_monitoring.yaml`, fixed before the first replay tick. A flag
  or an alert is a row of `results/m12_replay/monitoring.csv`; it changes nothing.

**Versioning and run tracking (stage 3, `docs/mlops.md` §7).** DVC and MLflow are not runtime
dependencies. They are in dependency groups of their own and in a second environment,
`.venv-tools`, so that `.venv` stays exactly the set of packages every result was computed with.

```bash
make setup-tools        # create / update .venv-tools (DVC with Cloud Storage support, MLflow)
make dvc ARGS="status"  # any DVC command; see "Data" below for pull
make data-manifest      # write configs/data_manifest.json from the DVC-tracked files
make data-verify        # sha256 of the tracked files on disk against the manifest
make mlflow-sync        # copy result tables, run manifests and the pipeline's model versions into the local MLflow store
make mlflow-check       # every metric against its source; the model registry against the manifests, the files and the state
make mlflow-ui          # open http://127.0.0.1:5001 (stop with Ctrl-C)
make test-tools         # the tracking tests with MLflow installed (a real SQLite store)
```

The MLflow store (`artifacts/mlflow/`, not in git) is a view: the result tables and the run
manifests are the record. Runs marked "(imported)" were made before the weekly pipeline existed;
the experiment `weekly-pipeline` holds one run per run of a tick. The model registry mirrors the
weekly ranker and the Two-Tower chain models by path and sha256, with the alias `current` on the
models behind the current prediction set; nothing reads the registry to deploy.

Do not `pip install` into `.venv`. New dependencies are added with `uv add <package>`
(dev tools: `uv add --group dev <package>`), only after agreement (see AGENTS.md §3.4, §8).

## Data

The H&M competition data are **not included** in this repository and are **not public**: they
may not be redistributed. The files under `tests/fixtures/` are made up and hold no row of the
competition data.

**Without access to the project's storage:** download the data from the Kaggle competition
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
under the competition rules, put the files where "Data layout" below says, and run the
milestones from M0 on. The bootstrap state of stage 3 (`data/bootstrap/`) is made of stage-1
outputs; without access it exists only after stage 1 was run
(`scripts/m11_bootstrap_state.py` then builds it).

**With access (stage 3, `docs/mlops.md` §7):** the pipeline inputs are versioned with
[DVC](https://dvc.org) in a private Cloud Storage bucket. git holds the small pointer files
(`data/raw.dvc`, `data/processed.dvc`, `data/bootstrap.dvc`,
`artifacts/features/category_codes.json.dvc`) and `configs/data_manifest.json`, which lists
path, size and sha256 of every tracked file. The bucket's name is not in this repository;
whoever gives you access gives you the name.

```bash
make setup-tools            # DVC and MLflow in an environment of their own, .venv-tools
gcloud auth application-default login          # or impersonate the service account you were given
make dvc ARGS="remote add --local storage gs://<bucket>/dvc"
                            # written to .dvc/config.local, which git ignores
make dvc ARGS="pull"        # everything (about 6 GB); or only a part, for example:
make dvc ARGS="pull data/processed artifacts/features/category_codes.json"
make data-verify            # sha256 of every tracked file against configs/data_manifest.json
make data-verify ARGS="--only data/processed artifacts/features/category_codes.json"
make dvc ARGS="status --cloud"                 # is everything of this commit in the remote?
```

Downloading from Cloud Storage is charged to the bucket's project; pull only what you need.

## Data layout

The files under `data/` are not tracked by git (only the DVC pointer files are). Put the files here:

```
data/
├── raw/
│   ├── articles.csv
│   ├── customers.csv
│   ├── transactions_train.csv
│   └── sample_submission.csv    # not tracked with DVC: no step of the pipeline reads it
├── bootstrap/                   # stage 3: the state at the end of F4 (docs/mlops.md §6.3)
└── processed/
    ├── image_features/          # precomputed CLIP ViT-B/32 image vectors (read-only)
    │   ├── image_embeddings.npy           float32 [105100, 512], L2-normalized
    │   ├── image_article_ids.parquet      embedding_row -> article_id
    │   └── ...
    └── text_features/           # precomputed all-MiniLM-L6-v2 text vectors (read-only)
        ├── text_embeddings.npy            float32 [105542, 384], L2-normalized
        ├── text_article_ids.parquet       embedding_row -> article_id
        └── ...
```

The data and artifact roots are set in `configs/base.yaml` (`paths.data_root`, `paths.artifacts_root`).
All code gets paths from `src/hmrec/paths.py`.

## Milestones and scripts

The main entry point of each milestone. See proposal §11 for the milestones and §14 for every script.

| Milestone | What | Script |
|---|---|---|
| M0 | Preprocess CSV → Parquet, int32 IDs | `scripts/m0_preprocess.py` |
| M0 | Two-Tower 200-step MPS/CPU benchmark | `scripts/m0_benchmark_twotower.py --device mps` |
| M1 | Metrics + R1–R5 | unit tests only |
| M2 | R1–R5 ablation + HPO on D1–D4 | `scripts/m2_recall_hpo.py --channels r1_popularity` |
| M3 | Two-Tower base model | `scripts/m3_twotower_base.py` |
| M4 | Union recall check | `scripts/m4_union_check.py` |
| M5 | Freeze recall configs | `configs/locks/`; check with `scripts/m5_verify_locks.py --channel ...`; git tag `lock-channels-v1` |
| M6 | Candidates per fold | `scripts/m6_generate_candidates.py --fold F1` |
| M6 | Features per fold | `scripts/m6_build_features.py --fold F1` |
| M7 | Ranker HPO on V1–V4 | `scripts/m7_ranker_hpo.py --stage ...` |
| M8 | Final evaluation on T1–T4 | `scripts/m8_final_eval.py --stage ...` |
| M10 | Container image, CI, calibration of V4 in the container | `make image-build`, `scripts/m10_calibrate.py` |
| M11 | Data versioning, bootstrap state, run manifests, MLflow view | `make dvc ARGS="..."`, `scripts/m11_mlflow.py` |
| M12 | The weekly pipeline and its replay | `scripts/weekly.py --anchor DATE --mode eval\|serve`, `scripts/m12_replay_compare.py` |

Copyright (c) 2026 Haokun Liu. All rights reserved.
