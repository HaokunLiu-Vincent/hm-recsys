# scripts/checks/

Independent cross-checks of generated data. Each script recomputes a result from
the processed tables with its own, deliberately simple code and compares it with
what the project code wrote.

**Why these scripts do not import the project logic they check:** if a check
reused `hmrec.data.asof`, `hmrec.data.queries` or `hmrec.data.day_index`, a bug
in those modules would be repeated in the check, and the check would still pass.
So these scripts re-derive date windows with plain `datetime` arithmetic. This is
the one deliberate exception to the AGENTS.md rules "date filtering only in
asof.py" and "day/date conversion only in day_index.py". The exception covers
only `scripts/checks/`; never copy this code into `src/`.

They still read processed tables through `hmrec.data.metadata.scan_processed_parquet`
(which only checks metadata.json and does no computation), and use `hmrec.paths`,
`hmrec.config` and `hmrec.logging_utils` for plumbing.

Each script raises an error on the first failed check and exits non-zero.

| Script | Checks | When to run |
|---|---|---|
| `check_queries.py` | `artifacts/queries/` and `artifacts/labels/` against `data/processed/transactions.parquet`: query_id range and int32 fit, fold/anchor columns, labels equal the distinct (customer, article) pairs bought in [anchor, anchor + 7d), query count equals the number of distinct buyers that week, last label day of the last fold. Also logs per-fold counts and folds whose query count is more than 30% from the average. | After `scripts/m0_build_queries.py` |
| `check_features.py` | `artifacts/features/{fold}/` against the candidate files and the processed tables: rows sorted by (query_id, article_idx), no repeated pair, no query split over part files, row count = distinct candidate pairs; then 25 features (at least 2 of every group A-E; 6 of the 15 features added on 2026-11-17) recomputed with plain Python loops for 1,000 random rows and compared with the wide table. Uses nothing from `hmrec.features`, `hmrec.fusion`, `hmrec.data.asof`, `hmrec.data.histories` or `hmrec.vectors`. | After `scripts/m6_build_features.py --fold F1` |
| `check_ranker_eval.py` | One Ranker trial at one number of trees: reads the saved Top-12 (`artifacts/predictions/...`) and recomputes MAP@12, Recall@12 and NDCG@12 with plain Python loops; checks that every query has ranks 1..n (n <= 12), no repeated article, and the order score DESC then article_idx ASC; compares the fold means with the row in the results table (difference < 1e-9). Uses nothing from `hmrec.eval.metrics` or `hmrec.ranker`. The label table of a V fold (areas `speed_test`, `v_phase`, `m8_rehearsal`) comes from `hmrec.eval.ranker_eval.load_m7_evaluator`, which refuses every fold that is not a V fold. For a trial of the final evaluation (`--area t_phase`, a T fold) it comes from `hmrec.eval.final_eval.load_final_evaluator` with a permit from `hmrec.final.gate`: the gate refuses unless the rehearsal passed and the formal run was started with the same code, so this script cannot be the first reader of a T label. | After a trial of `scripts/m7_speed_test.py`, `scripts/m7_ranker_hpo.py` or `scripts/m8_final_eval.py` |

Run:

```bash
uv run python scripts/checks/check_queries.py
uv run python scripts/checks/check_features.py --fold F1
uv run python scripts/checks/check_ranker_eval.py --area speed_test --experiment full --config-id <id> --fold V1 --num-trees 300
```
