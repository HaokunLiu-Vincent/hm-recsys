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
- 按时间截取历史数据，**只能通过 `src/hmrec/data/asof.py`**，不要在别处自己写日期过滤。
- 每个 as-of 函数都要用断言检查输入的最大日期小于 anchor。
- **Recall 和特征代码永远不读 label。** Label 只在 `src/hmrec/eval/` 和 `src/hmrec/ranker/sampling.py` 中读取。
- 某个 fold 的候选和特征一旦生成就冻结。之后用它们训练时，**不允许用更新的模型重新生成**。

### 3.2 阶段规则

| 阶段 | 可以做 | 不可以做 |
|---|---|---|
| D1–D4 | Recall 的 ablation 和 HPO | 用 D 周数据训练 Ranker |
| F1–F4 | 用锁定配置生成候选 | 任何调参 |
| V1–V4 | Ranker 的 HPO | 修改 recall 配置 |
| T1–T4 | 用全部锁定配置做最终评估 | **任何调参、任何选择**；HPO 代码路径里不能出现 T fold |

### 3.3 不能改的东西

- `configs/fold_manifest.yaml` 里的日期。
- `configs/locks/` 下的任何文件。只能新增，不能修改或删除。
- `configs/selection_rule.yaml`（选择规则在看分数之前就已提交）。
- `docs/decisions.md` 的历史记录。只能追加。

需要改动以上内容时，停下来说明原因，等用户确认。

### 3.4 其他禁止事项

- 不要删除 `data/`、`artifacts/`、`results/` 下的文件。
- 未经用户同意，不要新增依赖（修改 `pyproject.toml`）。
- 未经用户同意，不要在完整数据上运行重任务（全量 recall、Two-Tower 训练、Ranker 训练）。开发和测试只用 `tests/fixtures/` 的小数据。
- 不要用 `print` 调试后留在代码里。
- 不要为了让测试通过而修改测试的预期值，除非你能说明原来的预期值是错的。

---

## 4. 运行环境约束

| 资源 | 限制 |
|---|---|
| 内存 | **8 GB，CPU 和 GPU 共用**。单个任务峰值不超过约 6 GB |
| CPU | 6 线程：`num_threads=6`，`n_jobs=6`，`OMP_NUM_THREADS=6` |
| GPU | Apple M1 Max，PyTorch **MPS**（`device="mps"`）。没有 CUDA |
| Python | **3.12**（`.python-version`），环境由 **uv** 管理（`.venv` + `uv.lock`） |

写代码时：

- **ID 一律用 int32**，浮点数一律用 **float32**（MPS 不支持 float64）。
- 可能超过 1 GB 的数据**必须分块处理**，分块大小从 `configs/base.yaml` 读取。
- 优先使用 Polars 的 lazy API（`scan_parquet`），只读取需要的列。
- 大的中间结果写到磁盘（Parquet），不要都留在内存里。
- 长任务开始和结束时记录内存峰值（使用 `src/hmrec/memory.py`）。
- 不要使用 `faiss-gpu` 或任何依赖 CUDA 的库。`implicit` 用 CPU 版，LightGBM 用 CPU 版。

---

## 5. 目录和分层

完整目录见 `docs/proposal.md` §14。关键规则：

```
scripts/  →  src/hmrec/{recall, fusion, features, ranker, hpo}  →  src/hmrec/{data, eval}  →  config / paths
```

- 上层可以 import 下层，**下层不能 import 上层**。
- `recall/`、`features/`、`ranker/` 之间**互不 import**，通过 `artifacts/` 里的文件交接。
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

---

## 8. 常用命令

环境由 [uv](https://docs.astral.sh/uv/) 管理：Python 版本见 `.python-version`，依赖写在 `pyproject.toml`，精确版本锁在 `uv.lock`（进 git）。Makefile 中的命令都通过 `uv run` 执行。

```bash
make setup        # uv sync：按 uv.lock 创建 / 更新 .venv（含 dev 组）
make format       # uv run ruff format + ruff check --fix
make lint         # uv run ruff format --check、ruff check、pyright
make test-fast    # uv run pytest tests/unit
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

- 每次 commit 只做一件事，commit message 用英文，格式：`<scope>: <what changed>`，例如 `recall/r2: add decayed-frequency scoring`。
- 锁定配置（M5、M7 等节点）时打 tag，例如 `lock-channels-v1`。
- 不要提交 `data/`、`artifacts/` 下的文件。
- 未经用户要求，不要 push。
