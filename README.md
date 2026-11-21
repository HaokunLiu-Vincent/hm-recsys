# hmrec — H&M personalized fashion recommendation

**Full report:** [`docs/final_report.md`](docs/final_report.md) (English, the original and the
authoritative version) and [`docs/final_report.zh-CN.md`](docs/final_report.zh-CN.md) (Chinese
translation). The numbers in the two are identical; where they differ, the English version
prevails. This page is the summary (§1 of the report).

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

**Three things to know when reading these numbers:**

1. **T is 19.8% higher than V** (V1–V4 pooled: 0.032713, `results/v_phase_metrics.csv`), contrary
   to the prior expectation that "V is optimistically biased and T may fall below 0.032". §5 of
   the report uses four descriptive tables to give an explanation consistent with the numbers; it
   is a correlation, not a verified cause. Single-week MAP@12 over V1–T4 ranges from 0.030112 to
   0.042435, and the results carry a seasonal component (§4.4, §7.2 of the report).
2. **One run, one seed, offline evaluation only.** There is no online A/B test, and no repeated
   runs to give a range of variation (§7.2 of the report).
3. **T1–T4 has been used once.** Any later change may only be compared and selected on V1–V4
   (`AGENTS.md` §3.2).

## How this project was built

The author designed the system, set the selection rules, reviewed every result and made every
decision recorded in [`docs/decisions.md`](docs/decisions.md). Most of the code was written by
Claude Code, working under the rules in [`AGENTS.md`](AGENTS.md). Commits that Claude made itself
carry a `Co-Authored-By` trailer; commits made by the author carry none, also where they contain
code that Claude wrote.

## Documents

The only design reference is [`docs/proposal.md`](docs/proposal.md). Design changes are logged in
[`docs/decisions.md`](docs/decisions.md). Results and their limitations are in the final report:
[`docs/final_report.md`](docs/final_report.md) (English, authoritative) and
[`docs/final_report.zh-CN.md`](docs/final_report.zh-CN.md) (Chinese translation). Rules for coding
agents are in [`AGENTS.md`](AGENTS.md).

English is the primary language of this project: new documents are written in English first, and
a Chinese version, when there is one, is a translation saved as `*.zh-CN.md` (AGENTS.md §11).
The proposal, the decision log and `AGENTS.md` were written in Chinese up to tag `v0.1`; those
originals are kept as frozen archives (`docs/proposal.zh-CN.md`, `docs/decisions.zh-CN.md`,
`AGENTS.zh-CN.md`) and are no longer maintained. The English files are authoritative.

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

Do not `pip install` into `.venv`. New dependencies are added with `uv add <package>`
(dev tools: `uv add --group dev <package>`), only after agreement (see AGENTS.md §3.4, §8).

## Data

The H&M competition data are **not included** in this repository. Download them from the Kaggle
competition [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
under the competition rules. The files under `tests/fixtures/` are made up and hold no row of the
competition data.

## Data layout

`data/` is not tracked by git. Put the files here:

```
data/
├── raw/
│   ├── articles.csv
│   ├── customers.csv
│   ├── transactions_train.csv
│   └── sample_submission.csv
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

Copyright (c) 2026 Haokun Liu. All rights reserved.
