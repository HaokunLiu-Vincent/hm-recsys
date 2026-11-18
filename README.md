# hmrec — H&M personalized fashion recommendation

Predict the 12 articles each customer buys next week (Kaggle H&M dataset), main metric MAP@12.

```
6 recall channels (R1 Popularity, R2 Repurchase, R3 Sibling, R4 ItemCF, R5 WALS, R6 Two-Tower)
  → union + dedup (one row per (query, article))
  → wide feature table
  → LightGBM LambdaMART
  → Top-12
```

The only design reference is [`docs/proposal.md`](docs/proposal.md). Design changes are logged in
[`docs/decisions.md`](docs/decisions.md). Rules for coding agents are in [`AGENTS.md`](AGENTS.md).

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

3. Install the OpenMP runtime that LightGBM needs (the Ranker, M7 and later):

   ```bash
   brew install libomp
   ```

   The LightGBM wheel does not contain an OpenMP runtime; on macOS it loads Homebrew's
   `libomp.dylib` (`/opt/homebrew/opt/libomp/lib`). **This is a system dependency outside
   `uv.lock`**: `make setup` neither installs nor pins it, and `brew upgrade` can change it
   without any change in this repository. Version in use when the Ranker was built:
   **libomp 22.1.4** (`brew list --versions libomp`). Every M7 result row records the
   LightGBM version and the path and version of the libomp that the process really loaded
   (columns `lightgbm_version`, `libomp_path`, `libomp_version`). After upgrading libomp,
   repeat the "two trainings give the same model" check before producing new results
   (`docs/decisions.md`, 2026-11-18).

   torch brings its own copy of libomp, and two OpenMP runtimes in one process abort, hang
   or crash. So Ranker code never imports torch, and the tests run in two processes.

Other commands (all run through `uv run`):

```bash
make lint         # ruff format --check, ruff check, pyright
make test-fast    # unit tests (two processes: the torch side, then the LightGBM side)
make test         # unit + integration tests (also two processes)
make format       # auto-format
```

Do not `pip install` into `.venv`. New dependencies are added with `uv add <package>`
(dev tools: `uv add --group dev <package>`), only after agreement (see AGENTS.md §3.4, §8).

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

Placeholders for now; filled in as each milestone is implemented. See proposal §11.

| Milestone | What | Script |
|---|---|---|
| M0 | Preprocess CSV → Parquet, int32 IDs | `scripts/m0_preprocess.py` (TODO) |
| M0 | Validate fold manifest | `scripts/m0_check_folds.py` (TODO) |
| M0 | Two-Tower 200-step MPS/CPU benchmark | `scripts/m0_benchmark_twotower.py` (TODO) |
| M1 | Metrics + R1–R5 | unit tests only |
| M2 | R1–R5 ablation + HPO on D1–D4 | `scripts/m2_recall_hpo.py --channel r1` (TODO) |
| M3 | Two-Tower base model | `scripts/m3_twotower_base.py` (TODO) |
| M3 | Two-Tower D1–D4 chain | `scripts/m3_twotower_chain.py` (TODO) |
| M4 | Union recall check | `scripts/m4_union_check.py` (TODO) |
| M5 | Freeze recall configs | `configs/locks/`; check with `scripts/m5_verify_locks.py --channel ...`; git tag `lock-channels-v1` |
| M6 | Candidates per fold | `scripts/m6_generate_candidates.py --fold F1` (TODO) |
| M6 | Features per fold | `scripts/m6_build_features.py --fold F1` (TODO) |
| M7 | Ranker HPO on V1–V4 | `scripts/m7_ranker_hpo.py` (TODO) |
| M8 | Final evaluation on T1–T4 | `scripts/m8_final_eval.py` (TODO) |
