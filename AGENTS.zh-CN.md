> **Archive of the v0.1 state; no longer maintained.** The English file `AGENTS.md` is authoritative. （本文件是 v0.1 时的中文存档，不再维护；以英文版 `AGENTS.md` 为准。）

# AGENTS.md

本文件约束所有在本仓库工作的 coding agent（Claude Code 等）。**这里的规则优先于你的默认习惯。** 和 `docs/proposal.md` 冲突时，以 `docs/proposal.md` 为准，并在回复中指出冲突。

---

## 1. 项目是什么

H&M 个性化时尚推荐（Kaggle H&M 数据集）。目标：预测每个用户下一周会买的 12 件商品，主指标 MAP@12。

系统结构：

```
6 路 recall（R1 Popularity, R2 Repurchase, R3 Sibling, R4 ItemCF, R5 WALS, R6 Two-Tower）
  → 合并去重（union + dedup，每个 (query, article) 一行）
  → 宽表特征
  → LightGBM LambdaMART
  → Top-12
```

**唯一的设计依据是 `docs/proposal.md`。** 写代码之前先读相关章节。不要凭记忆或"常见做法"改变设计。

## 2. 开始任何任务之前

1. 读 `docs/proposal.md` 中与任务相关的章节。
2. 读将要修改的模块及其测试。
3. 如果要改动超过 1 个文件，**先给出简短计划**（改哪些文件、为什么、怎么测试），再动手。
4. proposal 没写清楚的地方，**先问，不要猜**。

---

## 3. 硬性规则（绝对不能违反）

### 3.1 防止时间泄漏

- 对 anchor `a` 的任何计算（recall、特征、模型训练），**只能使用 `t_dat < a` 的数据**。anchor 当天属于 label 周。
- 按日期过滤交易，**只能通过 `src/hmrec/data/asof.py`**，包括 history（`history_window`，`[anchor − N, anchor)`）和 label 周（`label_week_transactions`，`[anchor, label_end]`）。不要在别处自己写日期过滤。
- **唯一例外**：`scripts/checks/` 下的独立核对脚本为了不重复被检查代码的 bug，自己用 `datetime` 推导日期窗口，不使用 asof / queries / day_index（见 `scripts/checks/README.md`）。这些代码不能复制到 `src/`。
- `label_week_transactions` 读的是答案，**只能由 `src/hmrec/data/queries.py` 调用**。`tests/unit/test_asof.py` 会扫描 `src/hmrec/` 和 `scripts/`，其他文件引用它测试就失败。
- 每个 as-of 函数都要检查返回数据的日期在窗口内（history 的最大日期小于 anchor），超出就报错（用 `raise`，不用会被 `python -O` 去掉的 `assert`）。
- 读取 `data/processed/` 下的表，**必须通过 `hmrec.data.metadata.scan_processed_parquet`**。它先用 `data/processed/metadata.json` 核对 `day_zero`、年龄桶和预处理配置 hash，不一致就报错。不要直接对 processed 文件调用 `pl.scan_parquet` / `pl.read_parquet`。
- processed 数据中的日期是 int16 列 `t_dat_day`（距 `day_zero` 的天数）。日期与天数的换算**必须调用 `src/hmrec/data/day_index.py`**，不能自己写 `(d - day_zero).days` 之类的公式。
- **Recall 和特征代码永远不读 label。** Label 只在 `src/hmrec/eval/` 和 `src/hmrec/ranker/sampling.py` 中读取。
- 某个 fold 的候选和特征一旦生成就冻结。之后用它们训练时，**不允许用更新的模型重新生成**。

### 3.2 阶段规则

| 阶段 | 可以做 | 不可以做 |
|---|---|---|
| D1–D4 | Recall 的 ablation 和 HPO | 用 D 周数据训练 Ranker |
| F1–F4 | 用锁定配置生成候选 | 任何调参 |
| V1–V4 | Ranker 的 HPO | 修改 recall 配置 |
| T1–T4 | 用全部锁定配置做最终评估 | **任何调参、任何选择**；HPO 代码路径里不能出现 T fold |

**T1–T4 已经用过一次**（M8 的正式运行，2026-11-18，结果在 `results/t_phase*.csv`）。从现在起：

- **M8 不重跑。** 不重新训练 T 周的模型，不重新计算 T 周的指标来替换已有的记录。`results/t_phase*.csv` 和 `results/m8_*.csv` 是原始记录，不修改。
- **之后的任何改动（候选、特征、Ranker、任何参数）只能在 V1–V4 上比较和选择。** T 周的结果不能用来挑选、调参或决定是否采用某个改动。
- 对 T 周的结果可以做描述性的诊断（只描述，不选择、不修改任何东西）；再次读取 T 周的 label 必须有记录，并且先得到用户同意。

**M8 之后的描述性诊断有第二个读 T label 的入口**（2026-11-19，用户同意；细节见 proposal §10.2）：

- 入口是 `src/hmrec/eval/diagnostic_labels.py` 的 `load_diagnostic_evaluator`。读 T 周的 label 要带 `DiagnosticAccess`，它只由 `src/hmrec/diagnostics/access.py` 签发。`eval/final_eval.py`（M8 正式运行的入口）不动。
- 签发的条件：`results/t_phase_goals.csv` 里已有 G1–G3 三条判定；工作区干净；正式运行那个 commit 时已有的 M8 文件和 `configs/` 下已有文件的内容没有变（允许新增文件）；并且**先**在 `results/t_diag_label_access.csv` 追加一行（时间、commit、用途、脚本），每运行一次一行。
- **只有诊断代码能用这个入口**：`src/hmrec/diagnostics/`、`scripts/m8_diagnostics.py`。其他代码（M0–M8 的任何模块和脚本）不能 import `hmrec.diagnostics` 和 `hmrec.eval.diagnostic_labels`。诊断代码不能 import 训练、采样、用模型打分的代码，也不能 import M8 的 trial、闸门和 `eval/final_eval.py`；只写 `results/t_diag_*.csv`。`tests/unit/test_diag_code_rules.py` 检查这些，**新写的诊断模块要加进它的文件清单**。
- 诊断的结果（`results/t_diag_*.csv`）只用来描述，**不能用来选择、调参或决定是否采用某个改动**。
- **诊断代码提交之后，`scripts/checks/check_ranker_eval.py --area t_phase` 在当前的 HEAD 上不能再运行**：它读 T label 要过 M8 的闸门，闸门要求代码与正式运行的 commit 相同。需要重跑时，先切回 tag `m8-run-v1`（`git checkout m8-run-v1`），跑完再切回来。这个核对在正式运行之后已经跑过一次，结果在 `docs/decisions.md`。

### 3.3 不能改的东西

- `configs/fold_manifest.yaml` 里的日期。
- `configs/locks/` 下的任何文件。只能新增，不能修改或删除。
- `configs/selection_rule.yaml`（选择规则在看分数之前就已提交）。
- `configs/m8_final.yaml`（M8 的目标周、彩排周和 G1–G3 的判定标准，在读取任何 T 周 label 之前提交）。
- M8 正式运行开始之后：`results/t_phase*.csv`、`results/m8_*.csv` 里已有的行，以及 M8 的代码逻辑（`src/hmrec/final/`、`eval/final_eval.py`、`ranker/final_window.py`、`scripts/m8_final_eval.py`、`scripts/m8_step.py`；`scripts/m8_diagnostics.py` 是 M8 之后的描述性诊断，不属于 M8 的代码）。正式运行只做一次；发现 bug 只记录、不修复，停下由用户决定。
- `docs/decisions.md` 的历史记录。只能追加。

需要改动以上内容时，停下来说明原因，等用户确认。

**M8（T1–T4 的最终评估）的规则**（细节见 proposal §10.1）：

- T 周的 label 只能通过 `src/hmrec/eval/final_eval.py` 读取，并且要带 `src/hmrec/final/gate.py` 签发的 `FormalRunPermit`。M6、M7 的入口对 T fold 的保护保持不变。不要在别处构造许可，也不要绕过它直接打开 T 周的 label 文件。（M8 之后的描述性诊断另有一个入口，见 §3.2。）
- 正式运行之前必须先彩排（`scripts/m8_final_eval.py --stage rehearsal`，目标周 V4，不读 T label），结果要与 M7 的记录逐位相同。彩排没有通过、彩排之后改过代码或配置、或者工作区不干净，正式运行都会被代码拒绝。**不要为了让正式运行开始而修改或删除彩排的记录。**
- 正式运行（`--stage formal`）只做一次。它中断后可以用同一条命令续跑；代码或配置变了就不能续跑。
- "HPO 代码路径里不能出现 T fold"在 M8 的含义：做调参和选择的模块（`hpo/v_phase.py`、`hpo/v_lock.py`、`hpo/speed_test.py`、`hpo/runner.py`、`hpo/d_phase.py`）里不能出现 T fold，M8 也不能 import 它们。`hpo/v_trial.py` 里与预测哪一周无关的步骤（训练一个模型、打分、保存）和 `hpo/v_records.py` 的结果表读写，M7 和 M8 共用；它们不做 fold 检查，也不做选择，fold 检查在各自的入口。

**M5 之后，六路 recall 的配置只能通过 `src/hmrec/recall/locked.py` 获取**（它只读 `configs/locks/`；锁定文件的解析在 `src/hmrec/locks.py`）。M6 及之后的代码不能再读 `configs/recall/` 的默认值，也不能从 `results/d_phase.csv` 重放选择来得到配置。R6 的模型文件必须通过这个入口加载：它在加载权重之前核对 sha256，不一致就报错。`tests/unit/test_locked.py` 会扫描 M6–M8 的脚本和 `fusion/`、`features/`、`ranker/`。

**M7 之后，Ranker 的配置只能通过 `src/hmrec/ranker/locked.py` 的 `load_locked_ranker` 获取**（它读 `configs/locks/ranker.lock.yaml`；`configs/ranker.yaml` 只提供两个只影响内存和日志的值）。M8 的代码不能再读 `configs/ranker.yaml` 的调参默认值和固定参数、`configs/hpo/v_phase.yaml`，也不能在 `results/v_phase*.csv` 上重放选择来得到配置（`src/hmrec/hpo/v_lock.py` 只是给测试用的核对，M8 不调用）。这个入口在返回配置之前核对：锁定文件里的设置算出的 `config_id` 与记录一致、引用的 features / channels / twotower 锁定文件的 sha256 一致、本进程的 LightGBM 版本、libomp 版本和 `num_threads` 与记录一致；不一致就报错。`tests/unit/test_final_code_rules.py` 会扫描 M8 的代码，**新写的 M8 模块要加进它的 `M8_FILES`**。

### 3.4 其他禁止事项

- 不要删除 `data/`、`artifacts/`、`results/` 下的文件。
- 未经用户同意，不要新增依赖（修改 `pyproject.toml`）。
- 未经用户同意，不要在完整数据上运行重任务（全量 recall、Two-Tower 训练、Ranker 训练）。开发和测试只用 `tests/fixtures/` 的小数据。
- **新写或修改代码之后，必须先停下等用户 commit；工作区干净之后才能在全量数据上运行。**即使这次运行已经获得许可，也要先停下。原因：结果要能对应到一个确定的 commit（运行记录里的 `git_commit`，且 `git_dirty=False`）；在未提交的代码上跑出的结果无法复现，也无法事后核对是哪份代码产生的。
- 不要用 `print` 调试后留在代码里。
- 不要为了让测试通过而修改测试的预期值，除非你能说明原来的预期值是错的。
- 修改 `.gitignore` 后，必须做两项检查，并在回复中报告结果：
  1. 用 `git check-ignore` 确认 `src/` 和 `tests/` 下的文件没有被忽略。下面的命令应当**没有任何输出**：
     `find src tests -type f -not -path '*/__pycache__/*' | git check-ignore --stdin`
  2. 确认 ruff 检查的文件数没有减少：比较修改前后 `uv run ruff format --check src tests scripts` 输出的文件数。

  原因：ruff 会跳过 `.gitignore` 中的文件。曾经 `data/` 这一行同时忽略了 `src/hmrec/data/`，导致这个包既不会进 git，也没有被 ruff 检查，而 `make lint` 仍然显示通过。

---

## 4. 运行环境约束

| 资源 | 限制 |
|---|---|
| 内存 | **项目的内存预算是 12 GB，CPU 和 GPU 共用**（项目自己定的预算；本机物理内存 32 GB）。单个进程的真实峰值上限为 **12 GB**，等于预算。上限是警戒线，不影响任何计算结果；分块设置不随它放宽 |
| CPU | 6 线程：`num_threads=6`，`n_jobs=6`，`OMP_NUM_THREADS=6` |
| GPU | Apple M1 Max，PyTorch **MPS**（`device="mps"`）。没有 CUDA |
| Python | **3.12**（`.python-version`），环境由 **uv** 管理（`.venv` + `uv.lock`） |

写代码时：

- **ID 一律用 int32**，浮点数一律用 **float32**（MPS 不支持 float64）。float32 适用于特征、模型输入、embedding 等大数组。
  - **例外：指标值和聚合结果用 float64**（`src/hmrec/eval/metrics.py`）。原因：数据量小、只在 CPU 上算、§4.2 的"严格更好"比较对舍入敏感。
- 可能超过 1 GB 的数据**必须分块处理**，分块大小从 `configs/base.yaml` 读取。
- 优先使用 Polars 的 lazy API（`scan_parquet`），只读取需要的列。
- 大的中间结果写到磁盘（Parquet），不要都留在内存里。
- 长任务开始和结束时记录内存峰值（使用 `src/hmrec/memory.py`）。
- 不要使用 `faiss-gpu` 或任何依赖 CUDA 的库。`implicit` 用 CPU 版，LightGBM 用 CPU 版。

---

## 5. 目录和分层

完整目录见 `docs/proposal.md` §14。关键规则：

```
scripts/  →  src/hmrec/{recall, fusion, features, ranker, hpo, final, diagnostics, pipeline}  →  src/hmrec/{data, eval, vectors, locks, csv_records, run_records, feature_files, git_state}  →  config / paths
```

- 上层可以 import 下层，**下层不能 import 上层**。
- `recall/`、`features/`、`ranker/` 之间**互不 import**，通过 `artifacts/` 里的文件交接。
- `fusion/`（合并去重、`n_channels` / `best_rank` / `rrf_score` 的表达式、RRF baseline 的排序）不在上一条的限制里：**`features/` 和 `ranker/` 都可以 import `fusion/`**，`fusion/` 不 import 这三者。`ranker/` 用它在去掉 R6 时重算三个聚合特征（`ranker/ablation.py`），`hpo/` 用它计算 RRF baseline，所以宽表、ablation 和 baseline 用的是同一组表达式。
- `src/hmrec/vectors.py`（用户 / 商品向量、批量点积、保存 / 读取）属于底层，`recall/` 和 `features/` 都可以 import 它（见 proposal §14.1）。R5 / R6 的召回分数和 §8.3 的 `wals_dot` / `tt_dot` 都只通过它计算。
- `src/hmrec/locks.py`（读取并校验 `configs/locks/` 的锁定文件、核对文件 sha256）也属于底层，与 `vectors.py` 同层，只依赖 `config.py` 和标准库；`recall/`、`features/` 和 `ranker/` 都可以 import 它。从锁定文件**构建** Ranker 的配置在 `src/hmrec/ranker/locked.py`（见 §3.3）；从锁定文件**构建** channel 在 `src/hmrec/recall/locked.py`（见 §3.3）。
- `src/hmrec/csv_records.py`（只追加、表头固定的结果表 `results/*.csv` 的读写）和 `src/hmrec/run_records.py`（每个进程一行的运行记录：耗时、采样峰值、真实峰值）也属于底层，只依赖标准库和 `memory.py`；任何上层模块和 `eval/` 都可以 import 它们。
- `src/hmrec/feature_files.py`（每个 fold 特征文件的记录 `features.json` 和核对函数）也属于底层，只依赖标准库、polars 和 `locks.py`：`features/` 写记录，`ranker/` 在读取特征之前调用 `verify_feature_files` 核对，两者互不 import。
- `src/hmrec/git_state.py`（读取 HEAD commit 和工作区是否有未提交的改动）也属于底层，只依赖标准库和 `logging_utils.py`。`hpo/runner.py` 仍然转出这两个函数，M2–M6 的脚本不用改。
- `src/hmrec/final/`（M8 的最终评估：闸门、trial、彩排比较、recall 监控、G1–G3 判定、驱动）属于上层，与 `hpo/` 同层。它可以 import `eval/`、`ranker/`、`fusion/`、底层模块，以及 `hpo/v_records.py` 和 `hpo/v_trial.py`（见 §3.3 的 M8 规则）；不能 import `recall/`、torch 和做选择的 `hpo` 模块。M6、M7 的代码不能 import `final/`、`eval/final_eval.py`、`ranker/final_window.py`。`tests/unit/test_final_code_rules.py` 检查这些，**新写的 M8 模块要加进它的 `M8_FILES`**。
- `src/hmrec/diagnostics/`（M8 之后的描述性诊断：读取 T label 的记录和检查、四张描述表的计算、驱动）属于上层，与 `final/`、`hpo/` 同层。它可以 import `eval/`、`fusion/`、`data/`、底层模块、`hpo/v_records.py`（读结果表）、`hpo/selection.py`（只用容差常数）、`ranker/locked.py`、`ranker/runtime.py`、`ranker/ablation.py`（只取锁定配置的 id、树数、Top-K 和宽表的列名）和 `final/goals.py`（只用判定表的列名）。它不能 import 训练、采样、打分的代码，`final/` 的其他模块，`eval/final_eval.py`，`recall/`、`features/` 和 torch；其他代码不能 import 它（见 §3.2）。
- **M7 的代码不能 import torch**（M8 的代码同样不能）：`ranker/`、`hpo/v_records.py`、`hpo/v_trial.py`、`hpo/v_phase.py`、`hpo/speed_test.py`、`eval/ranker_eval.py` 和 `scripts/m7_*.py`，不能 import torch，也不能 import 会加载 torch 的模块（`recall/`、`hpo/runner.py`、`hpo/d_phase.py` 等）。原因：torch 和 LightGBM 各自加载一份 OpenMP 运行库（libomp），两份在同一个进程里会直接退出（`OMP: Error #15`）、卡死或段错误。`tests/unit/test_m7_code_rules.py` 检查这一条。
- `src/hmrec/pipeline.py`（把一个 fold 的各步各用一个独立进程依次运行，并写每个 fold 的摘要）属于上层，与 `recall/`、`hpo/` 同层：它可以 import `eval/` 和底层模块，但 `recall/`、`features/`、`ranker/` 不能 import 它。
- `scripts/` 只做三件事：解析命令行参数、调用 `src/` 里的函数、写结果。**不在 scripts 里写业务逻辑。**
- `notebooks/` 只做探索和画图，`src/` 不能依赖 notebook 中的任何代码。
- 所有文件路径都从 `src/hmrec/paths.py` 获取，不要在代码里手写路径字符串。
- 所有数值参数都从 `configs/` 读取，不要写死在代码里。

---

## 6. 代码规范

读这份代码的人是 **junior MLE**。可读性优先于简洁和技巧。

### 6.1 基本要求

- Python 3.12，全部加类型标注。
- `ruff format` 格式化，`ruff check` 无报错，`pyright`（standard 模式）无报错。
- 不要用 `# type: ignore` / `# pyright: ignore` 掩盖类型错误；确实需要忽略时，在同一行写清楚原因。
- 每个函数只做一件事，一般不超过 50 行。超过就拆分。
- 命名要完整、直白：`item_purchase_count_7d`，不要 `ipc7`；`anchor_date`，不要 `ad`。
- 不使用难懂的写法（多层嵌套推导式、元编程、炫技的链式调用）。一步一步写，中间变量起有意义的名字。
- 计算和 IO 分开：计算函数接收并返回 DataFrame 或数组，不在内部读写文件。

### 6.2 Docstring

所有公开函数和类都要有 docstring（Google 风格，英文，用词简单）。必须写清楚：

- 这个函数做什么；
- 输入和输出**表的粒度**（一行代表什么）和关键列；
- 用到的**时间窗口**（开区间还是闭区间）；
- 数组的 **shape** 和 dtype。

示例：

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

### 6.3 注释

- **写"为什么"，不只写"做了什么"。** 例如：

```python
# Bad:  filter dates
# Good: Keep only rows before the anchor. Rows on the anchor date belong to
#       the label week, so including them would leak the answer.
history = transactions.filter(pl.col("t_dat") < anchor_date)
```

- 以下情况**必须**写注释：时间窗口边界、dtype 选择、分块原因、去重的粒度、任何"看起来多余但必须这样做"的代码。
- 公式要在注释中写出数学形式，并注明对应 `docs/proposal.md` 的哪一节，例如 `# See proposal §5 R4: s(i->j) = C_w / (n_w(i)^alpha * n_w(j)^(1-alpha))`。

### 6.4 配置、日志、随机性

- 配置文件读取成带类型的 dataclass（`src/hmrec/config.py`），不要在代码中到处传 dict。
- 使用 `logging`（`src/hmrec/logging_utils.py`）。长任务记录：开始、进度、耗时、内存峰值、输出文件路径。
- 每个会用到随机数的地方都要接收 `seed` 参数，并在 config 里固定（NumPy、PyTorch、LightGBM、采样）。

### 6.5 错误处理

- 输入不符合预期时，**尽早报错**，错误信息要说明哪里不对、期望什么。例如：`raise ValueError(f"history contains t_dat={max_date}, which is not before anchor {anchor_date}")`。
- 不要用 `try/except` 吞掉错误。
- Recall 输出写盘之前，用 `src/hmrec/recall/base.py` 的 schema 校验函数检查列名、dtype、每个 query 内 `article_idx` 不重复、`rank` 从 1 开始连续。

---

## 7. 测试要求

- 每个 `src/` 模块都要在 `tests/unit/` 下有对应的测试文件。
- 测试数据放在 `tests/fixtures/`，**小到能手算**（几十行）。测试里写清楚预期值是怎么算出来的。
- 以下测试是**必须的**：
  1. **时间截断测试**：在 anchor 当天及之后加入"有毒"数据，结果必须完全不变。
  2. **指标测试**：Recall@K、AP@12、NDCG@K 用手算的小例子验证，包括"输出不足 K 个""label 为空"等边界情况。
  3. **Recall 输出 schema 测试**：去重、排序、tie-break（`score DESC, article_idx ASC`）。
  4. **负采样测试**：所有正样本都保留，每个 query 的负样本数正确，固定 seed 时结果可复现。
  5. **集成测试**：用小数据把一个 fold 从候选生成跑到评估。
- 测试要快：整个 `make test-fast` 应在 1 分钟内完成。
- **测试分两个进程运行**（原因同 §5 的"M7 的代码不能 import torch"）：`uv run pytest tests/unit` 只运行 torch 一侧的测试，`uv run pytest tests/unit --lightgbm-side` 只运行 LightGBM 一侧的测试（文件名以 `test_ranker_`、`test_v_`、`test_speed_test`、`test_m7_`、`test_final_`、`test_diag_` 开头，见 `tests/conftest.py`；`test_diag_` 是 M8 之后的诊断，它不训练模型，但读锁定配置和结果表的模块会加载 lightgbm）。`make test-fast` 和 `make test` 两个都跑。新写的测试如果要训练 LightGBM 模型，文件名必须用这些前缀之一，并且不能 import torch；一个测试进程里同时出现 torch 和 lightgbm 时，`tests/conftest.py` 会让这次运行失败。

---

## 8. 常用命令

环境由 [uv](https://docs.astral.sh/uv/) 管理：Python 版本见 `.python-version`，依赖写在 `pyproject.toml`，精确版本锁在 `uv.lock`（进 git）。Makefile 中的命令都通过 `uv run` 执行。

```bash
make setup        # uv sync：按 uv.lock 创建 / 更新 .venv（含 dev 组）
make format       # uv run ruff format + ruff check --fix
make lint         # uv run ruff format --check、ruff check、pyright
make test-fast    # uv run pytest tests/unit，再加一次 --lightgbm-side（两个进程，见 §7）
make test         # uv run pytest tests（单元 + 集成）
```

- 临时运行任何 Python 命令也用 `uv run ...`，不要直接调用系统 Python 或 `pip install`。
- **新增依赖必须先征得用户同意**（见 §3.4），同意后用 `uv add <包>`（开发工具用 `uv add --group dev <包>`），这样 `pyproject.toml` 和 `uv.lock` 会一起更新。不要手工编辑 `uv.lock`。

**完成任何代码修改后，必须运行 `make lint` 和 `make test-fast`，并在回复中报告结果。**

---

## 9. 完成标准（Definition of Done）

一个任务只有满足以下全部条件才算完成：

- [ ] 代码符合第 6 节的规范，`make lint` 通过。
- [ ] 新代码有对应测试，`make test-fast` 通过。
- [ ] 没有违反第 3 节的任何硬性规则。
- [ ] 如果改变了设计（不只是实现细节），已在 `docs/decisions.md` 末尾追加一条记录：日期、决定、原因。
- [ ] 回复中说明：改了哪些文件、为什么、怎么验证的，以及还有哪些没做或不确定的地方。

---

## 10. Git

- **Agent 不自行 commit**（也不 push、不打 tag）。完成任务后，在回复中列出：
  1. 建议 commit 的文件（如果一个任务应该拆成多个 commit，按 commit 分组列出）；
  2. 每个 commit 的 commit message（格式见下一条）。

  由用户自己 commit。用户在对话中明确要求 agent commit 时才可以 commit。
- **分支**：每个 milestone 一个分支，命名为 `m<编号>-<内容>`，例如 `m1-recall-r1-r3`。agent 可以创建、切换分支，但**不 commit、不合并**。milestone 完成后由用户 fast-forward 合并到 main（`git merge --ff-only <分支>`）。
- 每次 commit 只做一件事，commit message 用英文，格式：`<scope>: <what changed>`，例如 `recall/r2: add decayed-frequency scoring`。
- 锁定配置（M5、M7 等节点）时打 tag，例如 `lock-channels-v1`（由用户打，agent 只给出建议的 tag 名）。
- 不要提交 `data/`、`artifacts/` 下的文件。
- 未经用户要求，不要 push。

---

## 11. Documentation language

**English is the primary language of this project (from 2026-11-19).** This section is written in English for that reason.

- **Write in English first.** All new documents, new entries in `docs/decisions.md`, commit messages and code comments (including docstrings) are written in English.
- **A Chinese version is a translation.** When a Chinese version of a document is needed, save it next to the original as `*.zh-CN.md` (for example `docs/final_report.md` and `docs/final_report.zh-CN.md`). It has the same structure, sections and numbers as the English original, says at the top that it is a translation, and where the two differ the English version prevails. When the English original changes, update the translation in the same task or say in the reply that it is now out of date.
- **Existing Chinese documents stay as they are for now**: `docs/proposal.md`, the existing entries of `docs/decisions.md`, and this file (sections 1–10). They will be translated in a separate task after tag `v0.1`. Do not translate or rewrite them as part of another task. `docs/decisions.md` stays append-only: new entries are in English, earlier entries are not edited.
- This rule is about files in the repository. It does not decide the language of replies in a conversation: reply in the language the user writes in, unless the user asks for another.
