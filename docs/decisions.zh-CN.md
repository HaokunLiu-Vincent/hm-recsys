> **Archive of the v0.1 state; no longer maintained.** The English file `docs/decisions.md` is authoritative. （本文件是 v0.1 时的中文存档，不再维护；以英文版 `docs/decisions.md` 为准。）

# 决策记录

**只追加，不修改历史。** 每条记录：日期、决定、原因。

---

## 2026-11-15 采用 proposal v0.1.1 作为设计依据

- **决定**：`docs/proposal.md`（v0.1.1，O1–O5 已决定，含 §14 目录结构、§15 代码规范）是本项目唯一的设计依据。
- **原因**：项目重新开始，旧项目的代码、配置和结论都不沿用；需要一个明确的 source of truth。

## 2026-11-15 预计算 embedding 的实际位置和格式

- **决定**：
  1. 图像和文本 embedding 放在 `data/processed/image_features/` 和 `data/processed/text_features/`，而不是 §14 原来写的 `data/raw/`。已更新 §14 的目录树。
  2. 只使用 `*_embeddings.npy`（float32，已 L2 归一化）和 `*_article_ids.parquet`。
     对应关系：`.npy` 的第 `embedding_row` 行 ↔ parquet 中同一 `embedding_row` 的 `article_id`（10 位字符串）。
     M0 中把它映射成 `article_idx`（int32）。
  3. 目录里的 `*_faiss.index` 不使用（项目不依赖 faiss）。
  4. 图像向量只覆盖 105,100 / 105,542 个商品，缺 442 个；Two-Tower 的 item tower 对这些商品使用 mask（§5 R6 已经要求缺失时用 mask）。文本向量覆盖全部商品。
- **原因**：实际数据已经按这个结构放好，并附带 manifest（模型版本、sha256、校验记录）。只读检查结果：
  - image：`(105100, 512)` float32，C-order；`embedding_row` 为 0..105099 连续，`article_id` 唯一且已排序；failures 表 0 行。
  - text：`(105542, 384)` float32，C-order；`embedding_row` 为 0..105541 连续，`article_id` 集合与顺序都和 `articles.csv` 一致。
  - 抽查前 1,000 行，L2 范数都是 1.0。

## 2026-11-15 改用 Python 3.12 + uv + Pyright

- **决定**：开发环境从 Python 3.11 + pip/venv + mypy 改为 **Python 3.12 + uv + Ruff + Pyright + pytest**。
  - Python 版本写在 `.python-version`（3.12），`requires-python = ">=3.12,<3.13"`。
  - 运行依赖仍在 `[project] dependencies`；开发工具（pytest、ruff、pyright）放在 `[dependency-groups] dev`。
  - `uv sync` 创建环境，`uv.lock` 进 git；Makefile 的命令都通过 `uv run` 执行。
  - 类型检查用 Pyright，`typeCheckingMode = "standard"`，删除 mypy 及其配置。
  - 新增依赖用 `uv add`，且必须先征得用户同意（AGENTS.md §3.4、§8）。
- **原因**：`uv.lock` 让所有依赖的精确版本可以锁定、可以复现；Pyright 和 VS Code（Pylance）用同一个类型检查器，编辑器里看到的报错和 `make lint` 一致。

## 2026-11-15 proposal 升到 v0.1.2；agent 不自行 commit；修改 .gitignore 后必须检查

- **决定**：
  1. `docs/proposal.md` 升到 v0.1.2，把项目骨架和工具链迁移的结果写回设计文档：
     - §1.2 / §15：Python 3.12 + uv（`uv.lock` 锁定依赖）+ Pyright（standard 模式）。
     - §1.3：内存峰值由后台线程采样得到，是下限；6 GB 上限要留余量。
     - §2.1：写入预计算 embedding 的实测事实（文件、shape、行号对应关系、442 个商品缺图像向量、已 L2 归一化、不用 FAISS index）。
     - §4.2："4 周中至少 3 周更好"指严格更好，分数相同不算，与 `configs/selection_rule.yaml` 一致。
     - §5 R6：缺图像向量的商品用 mask 遮掉图像输入，照常参与训练和召回。
     - §8.3：`img_sim_hist` 在商品没有图像向量、或用户最近 20 件购买都没有图像向量时为 NaN；`txt_sim_hist` 在用户没有历史购买时为 NaN。
  2. AGENTS.md §10：agent 不自行 commit / push / 打 tag，完成任务后列出建议 commit 的文件和 commit message，由用户 commit。
  3. AGENTS.md §3.4：修改 `.gitignore` 后，必须用 `git check-ignore` 确认 `src/`、`tests/` 下的文件没有被忽略，并确认 ruff 检查的文件数没有减少。
- **原因**：
  - 设计文档要和已经确认的数据事实、工具链保持一致，后续实现（Two-Tower、B 组特征、选择规则）直接以文档为准，不再靠对话记录。
  - commit 由用户把关，保证每个 commit 的内容经过人工确认。
  - 骨架阶段 `.gitignore` 中的 `data/` 误忽略了 `src/hmrec/data/`：这个包既不会进 git，也被 ruff 跳过，而 `make lint` 仍然显示通过。需要一个固定的检查步骤防止再次发生。

## 2026-11-15 M0 预处理的数据格式

- **决定**：
  1. **ID 映射**：`customer_id` / `article_id` 按字符串排序后编号 0..N-1，存成 int32（`customer_idx` / `article_idx`）。编号只取决于 ID 集合，不取决于 CSV 行序。用户范围是 `customers.csv` 的全部用户，商品范围是 `articles.csv` 的全部商品；交易表中出现映射表里没有的 ID 时直接报错。`article_id` 必须是 10 位数字字符串。
  2. **日期**：交易表的日期存成 int16 列 **`t_dat_day`** = `(t_dat − day_zero).days`，`day_zero = 2018-09-20` 写在新增的 `configs/preprocess.yaml`。日期和天数的互相转换**只在** `src/hmrec/data/day_index.py` 中实现，其他模块都调用它。预处理时断言最早日期等于 `day_zero`、最大天数不超过 int16 上限。
  3. **交易表排序和 row group**：`transactions.parquet` 按 `(t_dat_day, customer_idx)` 排序（相同键保持原始行序，结果确定），每个 row group 100 万行（配置项 `transactions_row_group_size`），带 min/max 统计，后续按日期过滤时可以跳过 row group。校验 `t_dat_day` 单调不减。完全相同的重复行保留。
  4. **customers 清洗**：删除 `postal_code`（没有任何 channel 或特征使用）；`FN` / `Active` 转成 int8 0/1（空值 = 0）；`fashion_news_frequency` 中的 `None`（2 行）合并到 `NONE`；`club_member_status` / `fashion_news_frequency` 的其他缺失值保持 null，整数编码留给特征层。`age` 存 int8（缺失为 null）。
  5. **年龄桶**：按 §5 R1 加 `age_bucket` 列，桶边界（两端都包含）写在 `configs/preprocess.yaml`，加载时检查有序、连续、不重叠。
  6. **articles**：保留 `product_code`（int32）和 §8.4 的 6 个类目列；断言 `product_code` 等于 `article_id` 前 7 位。
  7. **embedding**：图像 / 文本向量按 `article_idx` 重新排列，存到 `data/processed/embeddings/`（float32 `[n_articles, dim]` + bool mask）；没有向量的商品填 0 并且 mask=False（0 只是占位，使用时必须看 mask）。检查**所有**有向量的行 L2 范数在 1 ± 1e-3 以内。原始 embedding 目录不修改。
  8. **metadata.json**：预处理结束时写 `data/processed/metadata.json`（day_zero、年龄桶、`configs/preprocess.yaml` 解析后取值的 sha256、各表行数）。下游读取 processed 数据时通过 `hmrec.data.metadata.scan_processed_parquet` / `check_processed_metadata` 检查，与当前配置不一致就报错。
- **原因**：int32 / int16 / int8 / float32 的类型压缩是 §1.3 内存计划的前提；排序加小 row group 让按 anchor 截取历史时少读数据；metadata 检查防止改了 `day_zero` 或年龄桶之后误用旧的 processed 数据（那会让所有天数错位，破坏时间截断）。

## 2026-11-16 文档与 M0 实现对齐：article_idx、t_dat_day、只用 Polars、processed 读取入口

- **决定**：
  1. `docs/proposal.md` §5：recall 统一输出格式中的 `article_id` 改为 `article_idx`，排序和 tie-break 改为 `score DESC, article_idx ASC`。因为 `article_idx` 是 `article_id` 字符串排序后的编号，这与原来的 `article_id ASC` 完全等价，不改变任何结果。
  2. §2.2：写明 processed 数据中的日期列是 int16 的 `t_dat_day`（距 `day_zero` 的天数），时间截断写成 `t_dat_day < date_to_day_index(a, day_zero)`；换算只能用 `src/hmrec/data/day_index.py`。
  3. §1.2：数据处理只用 Polars，不再写"或 pandas + pyarrow"。
  4. §14 目录树：补上 `configs/preprocess.yaml`、`src/hmrec/data/day_index.py`、`src/hmrec/data/metadata.py`，以及 M0 在 `data/processed/` 下生成的文件（交易 / 用户 / 商品 Parquet、两张 ID 映射表、`embeddings/`、`metadata.json`）。
  5. AGENTS.md §3.1 新增两条硬性规则：读取 `data/processed/` 下的表必须通过 `hmrec.data.metadata.scan_processed_parquet`；日期与天数的换算必须调用 `day_index.py`。
- **原因**：M0 实现（见 2026-11-15 "M0 预处理的数据格式"）已经确定了 int32 索引、int16 天数和 metadata 校验。设计文档和 agent 规则要和实现一致，否则后续 recall / asof 代码可能按旧文档用字符串 ID 或自己写日期换算。processed 表的天数依赖 `day_zero`，绕过 metadata 校验或自写换算公式，一旦配置变化会让所有天数错位，破坏 §3.1 的时间截断。

## 2026-11-16 通用组件：asof、活跃池、query_id、指标精度

- **决定**：
  1. **活跃池配置**：新建 `configs/catalog.yaml`（`window_days: 28`、`min_purchase_lines: 3`），由 `hmrec.data.catalog` 读取，R1/R4/R5/R6 共用。不放进 `preprocess.yaml`：活跃池按每个 anchor 现场计算，不影响 processed 数据，也不应改变 processed 数据的配置 hash。按 §2.4 这两个值已冻结，不是超参数，修改必须记入本文件。购买记录按行数计，重复行分别计数。
  2. **query_id 全局唯一**：`query_id = fold_ordinal × 10,000,000 + position`（int32）。`fold_ordinal` 按 `fold_manifest.yaml` 顺序（D1 = 0 … T4 = 15），`position` 是 fold 内按 `customer_idx` 升序的序号。偏移量写在新增的 `configs/queries.yaml`，断言每个 fold 的 query 数小于偏移量、`fold 数 × 偏移量 − 1` 不超过 int32 上限。提供反向解析函数 `decode_query_id`。queries 和 labels 表保留 `fold` 列。
  3. **日期过滤全部在 `asof.py`**：包括 history 窗口和 label 周。`label_week_transactions` 只能由 `data/queries.py` 调用，由 `tests/unit/test_asof.py` 扫描 `src/hmrec/` 和 `scripts/` 强制检查。窗口检查用 `raise`，不用 `assert`（`python -O` 会去掉 assert）。AGENTS.md §3.1 已同步。
  4. **指标用 float64**：特征、模型输入、embedding 等大数组仍用 float32；指标值和聚合结果用 float64。AGENTS.md §4 写明了这个例外。指标结果完全可复现：每个 query 内按 rank 顺序求和；跨 query 先按 `query_id` 排序，再用 `math.fsum`（精确舍入，与顺序无关）求和；打乱输入行序后结果逐位相同（有测试）。
  5. **selection 的相等阈值**：两个分数之差的绝对值 < 1e-12 视为相同，只为消除浮点误差，不改变 §4.2"严格更好"的含义。`configs/selection_rule.yaml` 已提交、不能修改，所以这个阈值是 `src/hmrec/hpo/selection.py` 中带注释的常量 `SCORE_EQUALITY_TOLERANCE`。
  6. **queries / labels 生成后冻结**：`scripts/m0_build_queries.py` 默认不覆盖已有文件，需要显式 `--overwrite`。
  7. **线程设置抽成 `hmrec.threads`**：只依赖标准库和 yaml，脚本在 import polars 之前调用 `configure_polars_threads()`；`m0_preprocess.py` 同步改用它。
- **原因**：
  - 多个 fold 合并（D 阶段 4 周合并、Ranker 用 F1–F4 训练）时，per-fold 编号会重复，只按 `query_id` join 会静默接错；全局唯一编号从根上避免这个问题。
  - label 周的过滤也是日期过滤，放在 `asof.py` 才能保证"日期过滤只有一个地方"；但它读的是答案，所以用测试而不是只靠文档限制调用方。
  - §4.2 要求"严格更好"，float32 的舍入误差（约 1e-7）可能让相同的分数看起来不同；float64 加固定求和顺序让同样的输入一定得到同样的分数。

## 2026-11-16 D3、D4 疑似促销周：不改选择规则，增加分组诊断

- **背景**：16 个 fold 的 query 实测（`scripts/m0_build_queries.py`，由 `scripts/checks/check_queries.py` 独立核对）：每周 6.9–12.0 万个 query，平均 8.2 万；D3（+46.9%）、D4（+41.8%）明显偏多，每个 query 的 label 数也更多（4.0–4.1，其他周 3.1–3.6），疑似促销周。D 阶段 query-weighted 合并中 D3、D4 占 236,182 / 391,681 = **60.3%** 的权重（此前对话中口头报告的"约 54%"算错了，以 60.3% 为准）。
- **决定**：
  1. **不修改 `configs/selection_rule.yaml`**（它在看到任何分数之前已提交，AGENTS.md §3.3 也不允许改）。
  2. D 阶段报告中，除 4 周合并分数外，额外分别报告 **D1–D2** 和 **D3–D4** 的 query-weighted 分数，**只作诊断，不参与选择**（写入 proposal §4.1）。
  3. proposal §12 新增已知局限第 8 条（D3、D4 疑似促销周）；§2.3 的每周 query 数改为实测值；§1.3 候选生成、Ranker 训练、Ranker 推理三行按每周 8.2 万 query、每个 query 约 3.4 个 label 重新估算，并写出计算过程。
  4. 独立核对脚本放在 `scripts/checks/`。为了核对有意义，它们**不使用** `hmrec.data.asof`、`hmrec.data.queries`、`hmrec.data.day_index`，自己用 `datetime` 重新推导日期窗口。这是 AGENTS.md"日期过滤只在 asof.py""日期换算只用 day_index.py"两条规则的唯一例外，只适用于 `scripts/checks/`，AGENTS.md §3.1 已注明。
- **原因**：
  - §4.2 的第二个条件"4 周中至少 3 周严格更好"是**按周计票**的，每周一票，与该周 query 数无关。只在 D3、D4 两周更好的配置最多拿到 2 票，无法通过；所以 D3、D4 虽然在合并分数（第一个条件）中占 60% 权重，也不能单独决定选择结果。规则本身已经能防止"只适合促销周"的配置被选中，没有必要改规则。
  - 反过来，规则无法告诉我们一个被接受的配置的提升主要来自哪两周；分组诊断补上这一点，便于解释 D 阶段的结果和 §6.7 / §12 第 1、2 条的季节性讨论。

## 2026-11-16 Recall 输出规格；R1 补齐行的排序；R2 recency-only 的分数

- **决定**：
  1. **每路声明输出规格**（`OutputSpec`，`src/hmrec/recall/base.py`）：额外列及其 dtype、排序键放在同一份声明里。默认排序键 `score DESC, article_idx ASC`；排序键必须以 `article_idx ASC` 结尾（每个 query 内唯一，保证顺序唯一）。额外列名必须以该路前缀开头（如 `pop_`），避免合并宽表时冲突；额外列不允许 null。
  2. **双向校验**：多了未声明的列、缺少已声明的列、dtype 不一致、null / NaN、`channel` 值不对、query 内商品重复、超过 quota、`rank` 不是 1..n 或与声明的排序键不一致，都报错。每种错误都有单元测试。所有 channel 都通过同一个 `rank_candidates` 排序编号；`score` 在排序**之前**转成 float32，保证存下来的分数和顺序一致。
  3. **R1 的排序例外**：排序键为 `pop_is_fallback ASC, score DESC, article_idx ASC`。补齐行用与桶内相同的 τ 计算全局衰减分数 `S(i)`，并跳过桶内已选的商品，保证同一 query 内不重复。因此 R1 的 `score`（`pop_score`）**混合两种量级**：桶内行是 `S(g, i)`，补齐行是全局 `S(i)`，由 `pop_is_fallback` 区分；B 组的 `pop_bucket_score` 为所有候选提供统一的 `S(g, i)` 量级。每个 fold、每个年龄桶的补齐行数由 `fallback_diagnostics` 计算并写进日志（M2 的 HPO 脚本再写进结果文件）。
  4. **R1 只输出活跃池 A(a) 内的商品**，G0 也一样（§2.4），这是代码中的固定规则，不是配置项。
  5. **R2 recency-only 的分数 = `1 / age_days`**（最近一次购买）。同日购买同分时只按 `article_idx ASC`，不用购买次数等信息打破平局。所有 channel 计算 `age_days = anchor_day − t_dat_day` 时断言 `age_days ≥ 1`，出现 0 说明 anchor 当天的数据进入了 history，直接报错。
  6. **R3 的排除范围**：只排除用户在 180 天风格窗口内买过的同一个 article；更早买过的不排除（与 §5 R3 的定义一致）。市场热度 `M(j)` 用全站所有用户的购买行；没有在市场窗口内卖出的商品 `M(j) = 0`。
- **原因**：R1 的桶内分数和全局分数量级不同，强行按 `score DESC` 排会把补齐行排到桶内行前面，违背"先用桶内、不足再补"的定义；把排序键做成每路声明的规格，比给 R1 写特殊分支更容易检查。额外列加前缀，是为了 §7.3 把 6 路 pivot 成宽表时列名不冲突。

## 2026-11-16 R3 活跃池外候选：在 M4 按预先登记的标准决定

- **背景**：D1 全量实测（默认配置，quota 50），R3 的 2,352,909 行候选中有 679,893 行（**28.9%**）不在活跃池 `A(a)` 内。它们只命中 88 次，命中率 **0.013%**；`A(a)` 内的候选命中率为 **0.48%**（约 37 倍）。`A(a)` 外的候选只占 R3 命中的 **1.1%**。
- **决定**：
  1. 现在不改 R3（§2.4 仍然是 R2、R3 不按活跃池过滤）。
  2. 在 M4 的 union 检查（proposal §5.7 第 4 项）中，在 D1–D4 上统计 R3 候选中 `A(a)` 外的比例、命中率，以及去掉其中"没有被其他任何一路召回"的候选后 union recall 的变化。
  3. **预先登记的标准**：union recall 下降 **< 0.1 个百分点**（绝对值）→ R3 改为只输出 `A(a)` 内的商品；否则保持不变。
  4. R2 做同样的统计，只做诊断，不改动。
- **原因**：D 阶段 R3 的配置按 Recall@50 选择。去掉 `A(a)` 外的候选几乎不影响 R3 自己的召回，但减少无效行（合并后的候选数、特征计算和 Ranker 推理的行数）的好处在 Recall@50 上体现不出来，所以不适合作为 R3 的 ablation，而是在 M4 作为系统层面的决定。
- **如实说明**：这个标准是在**看过 D1 的结果之后**制定的，不是事先登记的。为了尽量减少这一点的影响，阈值在看到 D2–D4 和 union recall 的结果之前写定，之后不再修改。

## 2026-11-17 R4 ItemCF：活跃池过滤在 Top-K_n 之前；分块和多个 α 的实现

- **决定**：
  1. **R4 的邻居 `j` 限定在 `A(a)` 内，过滤在取 Top-`K_n` 之前**（proposal §5 R4 原文只写了"输出限制在 `A(a)` 内"，没写过滤位置）。`C_w`、`n_w`、`d_u` 用拟合窗口内的全部商品计算，不受 `A(a)` 影响；seed 不要求活跃。有测试保证：先过滤再取 Top-`K_n` 与"先算完整的 seed × 全部商品相似度矩阵、删掉不活跃商品、再取 Top-`K_n`"结果完全一致；改变 `A(a)` 不改变 `C_w` 和 `n_w`；不活跃的 seed 仍然贡献分数。
  2. **分块和多个 α**：`C_w` 不整块存储，按 seed 分块（`chunking.item_chunk_size`）用 scipy 稀疏矩阵乘法计算，每块对每个 α 各取一次 Top-`K_n` 后释放。这样同一周、同一个变体下的多个 α 共用一次 `C_w` / `n_w` 计算，而不用把 `C_w` 存下来（D1 上 `Σ d_u²` ≈ 5.1 亿个 (i, j) 组合，整块存放会超出内存）。用 scipy 而不是 Polars join + group_by：稀疏乘法中每个 `C_w` 元素按固定的用户顺序求和，分块和不分块的结果逐位相同；Polars 多线程 group_by 的浮点求和顺序不固定。
  3. **只计算本周 query 用到的 seed**（D1：28,110 个），输出与计算全部商品再查表相同。
  4. **CF0 / CF3 只决定用户权重**（1 或 IUF），α 是独立的配置项；CF0 的 ablation 用 α=0.5。
- **原因**：
  - D1 拟合窗口的图有 71,713 个商品，只有 20,764 个（29%）在 `A(a)` 内。按商品数估算约 71% 的邻居名额可能落在不活跃商品上；**实测**先取 Top-100 再过滤时，**45.4%** 的邻居名额是不活跃商品（D1，CF3，α=0.7；热门商品多为活跃商品，所以低于 71%）。先过滤再取 Top-`K_n` 后，每个 seed 平均保留 99.05 个活跃邻居。

## 2026-11-17 R4 增加 seed 窗口和 seed 衰减（3×2 网格，D 阶段 HPO 开始之前加入）

- **决定**：
  1. R4 的用户分数改为 `S(u, j) = Σ_i q_ui · s(i→j)`。seed 衰减打开时 `q_ui = exp(−age_ui / τ_seed)`，`age_ui` 取用户最近一次购买 seed `i` 的天数，`τ_seed` 固定为 30（与 R2、R3 一致），不调；关闭时 `q_ui = 1`。建图（`C_w`、`n_w`、`d_u`）不衰减。
  2. R4 的 HPO 在 CF0/CF3 和 α 之后，新增 3×2 网格：`L_seed ∈ {90, 180, 365}` × seed 衰减 {关, 开}，(90, 关) 是默认配置。R4 合计从 4 个配置增加到 **9 个**（proposal §5 R4、§5.8 已同步）。
  3. 这一步不按"一次只改一个因素"，用网格：窗口越长，不衰减时旧购买的噪声越大，两个因素明显相互影响；每个配置只需十几秒。
  4. CF0 ablation 使用 α = 0.5（普通 cosine）。
  5. 默认配置（90 天、衰减关）的结果与之前完全一致（有测试）。
- **原因**：D1 实测，R4 有 **32.2%** 的 query（25,201 / 78,193）因为 90 天内没有购买而输出为空。更长的 seed 窗口可以覆盖这些用户，但需要衰减来控制旧购买的噪声。
- **时间点**：这些配置是在**看过 D1 上 R4 默认配置的结果之后、D 阶段 HPO 开始之前**加入的。只看了默认配置的 D1 结果（Recall@100 和空 query 比例），没有看过任何新配置的分数。

## 2026-11-17 新增底层模块 vectors.py（WALS 和 Two-Tower 共用的向量）

- **决定**：
  1. 新建 `src/hmrec/vectors.py`，属于底层（与 `data/`、`eval/` 同层，只依赖 numpy 和标准库），`recall/` 和 `features/` 都可以 import。内容：
     - `UserItemVectors` dataclass：用户向量、商品向量（float32），以及与 `customer_idx` / `article_idx` 的对应关系（int32，严格递增）；
     - `dot_pairs`：按 (customer_idx, article_idx) 对批量计算点积（按 `chunking.pair_chunk_size` 分块，新增到 `configs/base.yaml`），用户或商品没有向量时返回 NaN；
     - `save_vectors` / `load_vectors`：存到 `artifacts/models/{wals,twotower}/{fold}/`，`vectors.json` 记录来源（wals / twotower）和 fold，读取时核对。可以只保存部分用户（M6 只存该 fold query 中的用户），商品全部保存；没有保存的用户按"没有向量"处理（NaN）。
  2. R5 的 `score` 由 `dot_pairs` 计算（先用矩阵乘法选出候选，再用 `dot_pairs` 重新计算分数），所以同一个 (用户, 商品) 的 R5 `score` 与 §8.3 的 `wals_dot` 逐位相同（有测试）。
  3. proposal §14 目录树、§14.1 分层规则、§14.2 交接表已同步；R5 的 D 阶段运行不写向量文件，M6 才写。
- **原因**：§8.3 的 `wals_dot` 和 `tt_dot` 需要和 R5 / R6 训练出的向量做点积，但 AGENTS.md §5 规定 `features/` 不能 import `recall/`。放在底层模块里，两边共用同一个实现，既不违反分层，也避免两份点积代码产生不一致。

## 2026-11-17 多候选时的选择规则；R5 打分分块；AGENTS.md 分层图补上 vectors.py

- **决定**：
  1. **一步有多个候选时**（proposal §4.2 新增段落，实现在 `src/hmrec/hpo/selection.py`，`configs/selection_rule.yaml` 不改）：每个候选分别与当前默认值按 §4.2 两条比较；没有候选通过就保留默认值；有候选通过时，与最高合并分数相对差距 < 1% 的通过候选视为"差不多"，从中选优先顺序最靠前的。最高分没通过而另一个通过时，选通过的那个。
  2. **优先顺序**写在 `configs/hpo/d_phase.yaml`，在运行前提交，运行后不改：有简单 / 成本差别的按列出顺序（更简单 / 更便宜 / 更接近默认的在前）；纯数值参数（τ、γ、α、λ、α_conf）按与当前默认值的距离，距离相同时较小的值在前。R4 seed 网格：(90, 关) [默认] < (90, 开) < (180, 关) < (180, 开) < (365, 关) < (365, 开)。
  3. **R5 iterations 是收敛检查**，不用多候选规则：只有 25 相对 15 通过 §4.2 才改为 25；25 相对 15 明显下降（合并分数下降 ≥ 1% 且至少 3 周严格更差），或 10 相对 15 通过 §4.2 时，不自动选，停下来由人决定是否回到 λ 那一步。
  4. runner 按 §4.3 的顺序在步与步之间自动应用这条规则（下一步从上一步的结果开始）；最终配置写入 `configs/locks/` 之前由人确认。
  5. R5 打分的 query 分块从 1 万改为 5,000，写成 `configs/base.yaml` 的新配置项 `chunking.dense_score_query_chunk_size`（R6 推理以后也用它；`query_chunk_size` 仍为 1 万，供 fusion / features / ranker 使用）。D1 重跑：Recall@100 仍为 11.36%，候选行数相同（6,398,300），整体峰值从 4.05 GB 降到 3.10 GB。
  6. AGENTS.md §5 的分层图补上 `vectors.py`（底层，`recall/` 和 `features/` 都可以 import），与 proposal §14.1 一致。
- **原因**：§4.2 的规则是"一个候选对默认值"，而 §5 的很多步有多个候选，需要在看到分数之前定下怎么选，避免看到结果后再挑规则。用同一个 1% 阈值定义"差不多"，与 §4.2 第 1 条一致；纯数值参数没有简单 / 便宜之分，所以选离当前默认值最近的，倾向于少改动。

## 2026-11-17 D 阶段 R1–R3 跑完之后：R2 加窗口步骤、R3 扩大 γ 网格、R4 / R5 网格边界停下

- **决定**：
  1. **R2 新增 HPO 第 2 步**：历史窗口 `lookback_days ∈ {180, 365}`（默认 180），放在 τ 那一步之后；优先顺序 180 < 365（短的更简单）。R2 合计 4 → 5 个配置。
  2. **R3 的 γ 步骤增加 1.5 和 2.0**：`γ ∈ {0, 0.5, 1.0, 1.5, 2.0}`，优先顺序按与当前默认值的距离（距离相同时较小的值在前）。R3 合计 5 → 7 个配置。如果最终选中 2.0，不再继续扩展，只在报告中说明。
  3. **R4、R5 网格边界**：单个数值参数的步骤（R4 的 α；R5 的 α_conf、λ）在 `configs/hpo/d_phase.yaml` 中标记 `stop_at_boundary: true`。选中值落在该步取值范围的边界上时，runner 停下这一路，由人决定是否扩展；网格从不自动扩展。其他数值步骤选中边界值时只在报告和日志中注明。R5 的 factors 只有两个值（都是边界），不加这条检查。
  4. runner 在 git 工作区有未提交修改时，在运行开始和结束时各打印一次醒目的警告（不中断运行）；结果行照常记录 `git_dirty=True`。
  5. proposal §4.2（新增第 6 条）、§5 R2、§5 R3 的 HPO 表和 §5.8 的配置数已同步。
- **如实说明**：R2 的 365 天窗口和 R3 的 γ 1.5 / 2.0 都是**在看到 D 阶段 R1–R3 结果之后**加入的，不是事先登记的：
  - R2：默认配置下每个非空 query 平均只有约 12.6 个候选，远少于 quota 30，名额填不满；四个配置的 Recall@30 几乎相同（3.965%–3.968%）。在这种情况下排序方式和 τ 基本不影响 Recall@30，配置只能通过窗口长度（候选集合的大小）影响 recall，所以加入更长的窗口。
  - R3：γ 的最优值 1.0 落在原网格 {0, 0.5, 1.0} 的边界上（+2.34%，4/4 周更好），所以向上扩展两个值。
  这两处新增都只在已经看过的 D1–D4 上评估，存在一定的选择偏差；之后的 V、T 阶段不受影响，因为这些配置在 M5 锁定后不再改动。

## 2026-11-17 D 阶段 R4 跑完、R5 跑到 α_conf 之后：接受 R4 seed 窗口 365 天；R5 α_conf 扩展一次；factors 不做边界停止；stop_at_boundary 的局限

- **如实说明**：以下四项都是**在看到 D 阶段结果之后**决定的（R4 全部 9 个配置；R5 的 variant 和 α_conf {10, 20, 40}），不是事先登记的。只在 D1–D4 上评估，存在一定的选择偏差；V、T 阶段不受影响，因为配置在 M5 锁定后不再改动。
- **决定**：
  1. **接受 R4 的 seed 窗口 365 天，不扩展网格。** R4 选中 (365 天, 衰减开)，合并 Recall@100 9.332%（相对默认 +14.52%，4/4 周更好），365 是 seed 窗口网格 {90, 180, 365} 的最大值。不再向外扩，原因：
     - 365 天等于建图窗口（`fit_window_days`）。proposal §5 R4 规定 `L_seed` 不能超过建图窗口，往外扩必须同时改建图窗口，那是另一个实验；
     - τ_seed = 30 时，一年以前的 seed 权重约为 e^(−365/30) ≈ e^(−12) ≈ 6×10⁻⁶，几乎只对一年内没有任何购买的用户起作用；
     - 这部分用户约占 query 的 10%（365 天窗口下 R4 空 query 比例 9.9%–10.5%），由 R1 覆盖。
  2. **R5 的 α_conf 扩展一次，到 {10, 20, 40, 60, 80}。** 原网格中 40 被选中（合并 Recall@100 10.312%，+3.10%，4/4 周更好），落在边界上，runner 按 `stop_at_boundary` 停下。扩展原因：
     - α_conf 没有结构上的上限（不像 R4 的 seed 窗口受建图窗口限制）；
     - 10 → 20 → 40 每周都一致上升（合并 9.583% → 10.002% → 10.312%），但涨幅变小（+4.4%、+3.1%），说明峰值可能就在附近，所以在 40 之后加密（60、80），而不是翻倍到 160；
     - **每个参数最多扩一次**：这一步的 `stop_at_boundary` 改为 false，即使 80 被选中也接受，不再继续扩。
     - 优先顺序仍按与当前默认值 20 的距离：10 < 40 < 60 < 80。runner 复用 10、20、40 的已有 trial，只新跑 60 和 80。
  3. **R5 的 factors 不做边界停止**（`stop_at_boundary: false`）。{64, 128} 只有两个值，两个都是边界，这条检查没有意义；如果 128 通过就接受，不扩展到 256，因为 256 会让 ALS 拟合内存和打分时间再翻一倍。λ 保留 `stop_at_boundary: true`；iterations 仍按 proposal §5 R5 的收敛检查处理（15 次之后明显下降时停下，由人决定是否回到 λ 那一步）。
  4. **`stop_at_boundary` 的已知局限**（不改代码，由人工检查补上）：
     - 只在这一步**切换到**边界值时触发；默认值本身在边界上而被保留时不触发；
     - 只支持单个数值参数的步骤，不能用于多参数步骤，例如 R4 的 seed 窗口 × 衰减网格。R4 这次的边界停下是人工检查发现的，不是 runner 触发的。
     以后每一路跑完，除了 runner 的停止之外，还要人工检查每个数值参数的选中值是否在边界上。
- **其他**：proposal §4.2 第 6 条、§5 R5 的 HPO 表和配置数、§5.8 的 R5 配置数已同步；§1.3 补上 R4 D 阶段最重 trial 的内存峰值。

## 2026-11-17 D 阶段 R1–R5 最终选择（尚未锁定）

- **决定**：按 §4.2 的选择规则（含多候选规则、R5 iterations 收敛检查，以及上面两条记录中看到结果之后做的扩展），R1–R5 的 D 阶段选择如下。分数是各路的选择指标（Recall@quota），4 周合并 = 4 周所有 query 合并后取平均（§4.1）；"相对默认值"是相对 `configs/recall/*.yaml` 默认配置的合并分数。

  | Channel | 选中配置（与默认不同的项加粗） | trial_id | D1 / D2 / D3 / D4 | 4 周合并 | 相对默认值 |
  |---|---|---|---|---|---|
  | R1 Popularity（Recall@100） | A1，window 7，τ 3，quota 100（即默认配置） | d2094d455698 | 7.497 / 8.333 / 7.029 / 7.058 | **7.388%** | 0（默认 7.388%） |
  | R2 Repurchase（Recall@30） | decayed_frequency，**lookback 365**，τ 30，quota 30 | f562d598b0d6 | 5.336 / 4.770 / 3.491 / 4.011 | **4.266%** | +7.51%（默认 3.968%） |
  | R3 Sibling（Recall@50） | style_lookback 180，market_window 7，τ 30，**γ 1.0**，quota 50 | 3188358fa459 | 3.535 / 3.589 / 2.908 / 3.210 | **3.257%** | +2.34%（默认 3.182%） |
  | R4 ItemCF（Recall@100） | **CF3**，fit_window 365，**α 0.7**，neighbors 100，**seed_window 365，seed_decay 开**，τ_seed 30，quota 100 | df281d91d7c1 | 10.544 / 10.135 / 8.336 / 9.012 | **9.332%** | +14.52%（默认 8.149%） |
  | R5 WALS（Recall@100） | W2，window 180，τ 60，**α_conf 60**，λ 0.1，**factors 128**，iterations 15，seed 42，quota 100 | 3261309f8e8f | 12.295 / 11.760 / 9.576 / 10.444 | **10.807%** | +8.05%（默认 10.002%） |

  完整的逐步记录在 `results/d_phase.csv` 和 `results/d_phase_steps/{channel}.csv`；40 个 trial 全部 `git_dirty=False`。
- **尚未锁定**：以上配置**还没有写入 `configs/locks/`**。按计划在 M4 之后、M5 统一锁定，因为 R3 是否只输出活跃池 `A(a)` 内的商品要在 M4 按 §5.7 第 4 项的预先登记标准决定（需要 R6 加入之后的 union）。M4 之前不再调 R1–R5。
- **原因**：记录 D 阶段每一路的结论，供 M4 union 检查和 M5 锁定引用；锁定推迟是为了让 R3 的活跃池决定在锁定之前完成，避免锁定后再改。

## 2026-11-17 D 阶段 R1–R5 union 预检查：结果记录，暂不修改 recall

- **结果**（`scripts/m2_union_precheck.py`，HEAD 5b5d14e，工作区干净；只做诊断，没有运行新的 trial，也没有做任何选择；表格见 `results/d_phase_union_precheck.csv`，候选文件在 `artifacts/hpo/D/union_precheck/`，与 M4 的正式结果分开）：
  - R1–R5 五路合并去重后的 union recall：D1 22.06%、D2 22.06%、D3 18.54%、D4 19.80%，**4 周合并 20.31%**。G2 的目标 24% 针对六路（含 R6），这里只有五路。
  - **58.2%** 的 query（4 周合并）没有任何一路命中任何 label。
  - Leave-one-out（去掉一路后 union recall 下降，4 周合并，百分点）：R1 **5.43**、R5 **2.23**、R4 **1.73**、R2 **0.58**、R3 **0.57**。
  - 每个 query 去重后的候选数（4 周合并）：平均 257.9，P50 276，P90 311（上限 100 + 30 + 50 + 100 + 100 = 380）。
  - 用重新生成的候选计算的各路选择指标与 `results/d_phase.csv` 中选中 trial 的合并分数逐位相同。
- **决定**：
  1. **现在不修改 R1–R5 的任何配置**（包括名额）。等 R6 加入后，在 M4 按 proposal §5.7 用六路统一判断；如果六路 union recall 仍明显低于 24%，按 §5.7 先改 recall，再进入 F 阶段。
  2. **union 预检查的内存峰值 6.71 GB**，超过单任务约 6 GB 的目标，但在 8 GB 以内；本次不修复。候选生成本身的峰值为 3.5–4.6 GB，超出部分来自每周 6 次 union 去重（每周最多约 3,800 万行候选）。**M4 六路时候选更多，届时 union 计算必须改为**：先过滤出命中 label 的候选，再在这个小表上算 union recall 和 leave-one-out；每个 query 的候选数按 query 分块统计。
- **原因**：D 阶段 R1–R5 到此结束。union recall 是系统层面的判断，R6 的贡献还不知道，现在改单路配置没有依据；内存问题只在 union 计算上，M4 必须重写这部分，届时一并处理。

## 2026-11-17 Two-Tower 训练样本和用户历史的定义（M3 第 7 步）

- **决定**：
  1. **正样本**：每个 (customer_idx, 购买日, article_idx) 一条。同一用户同一天重复购买同一商品（多条相同的购买行）只算一条正样本。proposal §6.1 原文是"每一次购买对应一条正样本"，现改为按天去重。
  2. **分组**：同一用户同一天的所有正样本共用一个 `group_id`（样本按 (t_dat_day, customer_idx, article_idx) 排序，同组的行连续），供 §5 R6 的 multi-positive loss 识别"互为正样本、不能当负样本"。
  3. **用户历史**：购买日**之前**（`t_dat_day` < 购买日，不含当天）最近 20 件购买，不限时间窗口（最早到数据第一天）。历史中的购买与正样本**用同样的方式去重**（同一天同一商品只算一件）；不同天买同一商品算多件。顺序：从近到远；同一天的多件按 `article_idx` 升序，结果与文件行序无关。不足 20 件在末尾补 padding，`history_mask = False`。**padding 编号为 `n_articles`**（真实商品是 0..n_articles−1，所以 padding 不对应任何真实商品；以后某处漏用 mask，也不会把真实商品悄悄混进用户历史）。
  4. **商品特征表末尾加一行 padding 专用行**（下标 `n_articles`）：图像、文本向量全 0，两个 mask 为 False，product_code 和类目编码全为 0。表的行数为 `n_articles + 1`。模型的商品 ID embedding 用 `padding_idx = n_articles`；类目 embedding 用 `padding_idx = 0`（商品表里只有 padding 行的编码是 0，真实商品的这些列没有空值）。
  5. **Profile 和类目编码**：age_bucket、club_member_status、fashion_news_frequency，以及 product_code 和 §8.4 的 6 个类目列，都编码成整数：0 = 空值，1..K = 去重后排序的取值。
  6. **存储**：样本数组（`customer_idx`、`t_dat_day`、`article_idx`、`group_id` [N]，`history` int32 [N, 20]，`history_mask` bool [N, 20]）存成 `.npy`，放在 `artifacts/twotower/data/{窗口名}/`，不覆盖已有目录。Base 窗口写在 `configs/twotower.yaml`：购买日 [2020-06-03 − 182 天, 2020-06-03)，即 2019-12-03 – 2020-06-02。
- **原因**：
  - 同一天的重复购买行通常是一次购买多件，对"下一件买什么"没有额外信息；按天去重后，正样本和历史使用同一种计数，避免同一商品在历史里占多个位置。
  - 第 3 项"历史也按天去重"是 agent 在实现时提出的，用户在 commit 前确认。
  - padding 最初用 `article_idx = 0`；0 是真实商品，漏用 mask 时会无声出错，所以在 commit 前按用户要求改为 `n_articles`，并给商品特征表加 padding 行。
- **proposal §6.1 已同步。**

## 2026-11-17 Two-Tower 配置文件位置更正；base 窗口数组和 200 步 benchmark

- **更正**：上面"Two-Tower 训练样本和用户历史的定义"一条写的配置文件 `configs/twotower.yaml` 与 proposal §14 的目录不一致，已移到 **`configs/recall/r6_twotower.yaml`**（与 R1–R5 的配置放在一起），并加入 benchmark 设置（200 步、batch 512、embedding 128；占位模型的学习率和温度只用于计时）。
- **新增**：`src/hmrec/recall/twotower/benchmark.py`（§1.4 的占位模型，不是 R6 正式模型）和 `scripts/m3_build_twotower_data.py`；proposal §14 的目录已同步。
- **实测**（填入 proposal §1.3、§1.4）：
  - Base 窗口（购买日 2019-12-03 – 2020-06-02）：6,160,524 个样本，1,960,337 个 (用户, 日) 组，714,091 个用户，52,538 个商品；平均历史长度 15.18；9.0% 的样本历史为空。数组 670 MB；生成耗时 4.6 秒，峰值 3.35 GB（`/usr/bin/time` 4.15 GB）。
  - 200 步 benchmark：取 batch 占每步时间的 1.7%（MPS）和 0.7%（CPU），数据管道不需要优化。占位模型在 MPS 上每步 45.0 ms，**比 CPU（37.1 ms）慢约 20%**；按此外推 1 个 epoch 约 9.2 分钟（MPS）/ 7.5 分钟（CPU），高于 §1.4 原估算的 2–5 分钟。
- **未决定**：R6 用哪个设备训练。§1.2 仍写 MPS；用真实模型重新比较后再定。
- **说明**：数组生成和 benchmark 运行时，Part 3 的新代码尚未 commit（`metadata.json` 记录 `git_dirty: true`，HEAD 9e34ba1）；数据构建的核心逻辑（`data.py` 的样本、历史、padding）在 9e34ba1 中已 commit。

## 2026-11-17 Two-Tower 模型结构、loss、训练设置和 E_base 选择（M3 第 8 步）

完整结构和全部固定数值见 proposal §5 R6 和 `configs/recall/r6_twotower.yaml`。以下是相对 proposal 原文新增或具体化的决定，都在**看到任何 R6 分数之前**做出。

- **决定与理由**：
  1. **Item tower 各部分拼接，而不是相加**：图像 128、文本 128、article ID 64、product_code 32、6 个类目字段 90（按下一条规则），共 442 维。理由：相加要求每个字段都是 128 维，取值很少的字段也要占 128 维；拼接后每个字段可以用合适的维度，由后面的层决定怎么组合。
  2. **类别特征编码规则**（两个塔都适用）：取值数（含缺失编码 0）≤ 20 的字段用 one-hot，> 20 的用 16 维 embedding（`padding_idx = 0`，初始化 N(0, 0.02²)）；article ID（64 维，`padding_idx = n_articles`）和 product_code（32 维）例外，都初始化为 0。按实际数据：product_group_name（20）、index_group_no（6）、age_bucket（8）、club_member_status（4）、fashion_news_frequency（4）用 one-hot；product_type_no（133）、garment_group_no（22）、section_no（58）、colour_group_code（51）用 embedding。理由：取值很少的字段，embedding 维度超过取值数没有意义，后面的 Linear 会把多余的维度合并掉，只会增加参数和噪声。ID 初始化为 0：训练中没出现过的商品，这一部分保持 0，不引入随机噪声。
  3. **两个塔都加一层非线性**：拼接后 → `Linear(→ 256)` → ReLU → `Linear(256 → 128)` → L2 归一化。理由：拼接后各部分只经过线性组合就无法表达字段之间的交互（例如某个类目下图像的重要性不同）。
  4. **按位置加权的 pooling**：20 个历史位置各学一个标量权重，只在有购买的位置上做 softmax（初始全为 0，即从 masked mean 开始）；历史为空时 pooled = 0。理由：历史按从近到远排序；R4 加入 seed 时间衰减后提升了 14.5%，说明新旧很重要，而 masked mean 对所有位置一视同仁。
  5. **Duplicate-aware**（**proposal 原文没有，这是新增的要求**）：loss 的列是 batch 中去重后的目标商品，同一商品出现多次只算一列。理由：热门商品在同一 batch 中常被多个用户购买；不去重时，它对其他用户会被当作多个负样本，惩罚被重复计算。同组的其他正样本所在列置为 −∞（multi-positive，proposal 原文已有）。τ = 0.05；LogQ 的 `Q(k)` 只用当前训练集统计（base：全部 base 训练样本，不含报警用户；fine-tune：该次的新一周 + replay）。
  6. **设备选择规则**：真实模型在 mps 和 cpu 上各跑 200 步（各自一个进程，前 10 步不计时），只有 MPS 的平均每步时间比 CPU 快 **1.3 倍以上**时才用 MPS，否则用 CPU。理由：MPS 上 embedding 的反向传播不是确定性的，训练结果无法精确复现；CPU 上的结果可以逐位复现（有测试）。`configs/recall/r6_twotower.yaml` 的 `training.device` 先写为 `cpu`，benchmark 之后按此规则确定，结果另记一条。
  7. **E_base 的选择方式**（proposal §6.3 只写了"按 §4 规则"）：默认 epoch 1，候选 epoch 2–6，epoch 少的优先（训练更少 = 更便宜），按 §4.2 的多候选规则选择，写在 `configs/hpo/d_phase.yaml` 的 `r6_twotower.base_epoch`。选中 epoch 6（最后一个 checkpoint）时停下，由人决定（`stop_at_boundary`）。
  8. **训练设置**：Adam（β = 0.9, 0.999，ε = 1e-8），lr 1e-3 固定不衰减（fine-tune 1e-4），weight decay 0（输出 L2 归一化，权重大小不影响分数，decay 只会让少见商品的 ID 行变小），不做梯度裁剪；batch 512，每个 epoch 用 `(seed, epoch)` 在 CPU 上生成打乱顺序，保留最后一个不满的 batch；6 个 epoch；seed 42；1% 的用户留作报警，只记录 loss。
  9. **实现**：R5 的"按向量点积取 Top-quota、score 用 `dot_pairs` 重新计算"移到 `src/hmrec/recall/vector_scoring.py`，R5 和 R6 共用（R5 的结果不变，R5 测试全部通过）；`make_trial_id` 和 `results/d_phase.csv` 的 JSON 序列化加上 `default=str`（R6 配置中有日期；R1–R5 的 trial_id 不变）。
- **proposal 已同步**：§5 R6（完整结构和数值）、§1.3（base 训练数组的内存峰值改为更保守的 4.70 GB）、§14（`vector_scoring.py`、`benchmark.py` 的说明）。

## 2026-11-17 Two-Tower 训练设备：CPU（真实模型 200 步 benchmark）

- **结果**（`scripts/m0_benchmark_twotower.py --model real`，HEAD e91b5cb，工作区干净；base 窗口数组，batch 512，200 步，每个设备单独一个进程，去掉前 10 步）：

  | 设备 | 每步平均 | 每步 P90 | 取 batch 占比 | 前 10 步合计 | 内存峰值 |
  |---|---|---|---|---|---|
  | CPU（6 线程） | **30.4 ms** | 31.1 ms | 0.9% | 0.34 秒 | RSS 2.18 GB（`/usr/bin/time` 2.18 GB） |
  | MPS | 45.2 ms | 54.4 ms | 1.2% | 2.94 秒 | RSS 2.15 GB；MPS 张量 1.15 GB（驱动共持有 2.03 GB） |

- **决定**：按上一条记录的规则（MPS 快 1.3 倍以上才用 MPS），**R6 用 CPU 训练和推理**。MPS 的速度只有 CPU 的 0.67 倍。`configs/recall/r6_twotower.yaml` 的 `training.device` 本来就是 `cpu`，不用改；proposal §1.2、§1.4 已同步。
- **推算**：每个 epoch 约 1.19 万步（去掉 1% 报警用户）× 30.4 ms ≈ **6.0 分钟**，6 个 epoch 约 **36–37 分钟**（另加每个 epoch 末尾的报警用户 loss 和 checkpoint 保存，各几秒）。内存峰值约 2.2–2.5 GB。
- **附带核对**：R5 的打分代码移到 `recall/vector_scoring.py` 之后，用 R5 在 D 阶段选中的配置（trial 3261309f8e8f）在 D1 上重新运行一次：Recall@100 = 0.12295079519167419、候选 6,398,300 行，与 `results/d_phase.csv` 的记录逐位相同（没有写入 results，也没有写候选文件）。

## 2026-11-17 Two-Tower 训练链（§6.4）的定义：replay、Adam、默认配置和候选顺序

- **时间点**：以下定义是在**看到 E_base 结果（选中 epoch 4，4 周合并 Recall@100 11.84%）之后、运行任何训练链之前**确定的。没有看过任何 fine-tune 的分数。
- **决定**：
  1. **训练链**：从 base 的 epoch 4 出发，按 D1 → D2 → D3 → D4 的顺序，每周先用当前模型预测（记分），再用这一周（购买日 `[a, a + 7)`）的数据 fine-tune；D4 预测完之后也 fine-tune 一次，得到 M4（F1 用）。每一步的模型存到 `artifacts/models/twotower/chain/{trial_id}/{fold}.pt`，不覆盖。
  2. **Replay = 50% 的含义**：从这一周开始之前 4 周（`[a − 28, a)`）的样本中不放回抽取，数量与新数据相同（新 : 旧 = 1 : 1）；一般公式 `n_replay = round(n_new × replay / (1 − replay))`；replay = 0 时只用新数据。抽样用固定 seed。
  3. **报警用户**：继续排除 base 训练中那 1% 的用户（用 base 数组和 base seed 重新抽出同一批），并记录他们每周的 loss。
  4. **每周重新初始化 Adam**（lr 1e-4，其余同 base），不继承上一周的优化器状态。理由：每周的数据和 LogQ 分布不同，上一周的动量和二阶矩不再对应当前的梯度；每周从同样的初始状态开始也让各周的设置一致、容易复现。
  5. **LogQ 的 Q** 用这一次 fine-tune 集（新数据 + replay）统计。
  6. **默认配置是 frozen**（不更新，直接复用 E_base epoch 4 的结果）。为了让这个 trial 被复用，`hmrec.hpo.runner.parsed_config` 规定：`e_update = 0` 时，`replay`、`chain_seed`、`chain` 这些设置不影响结果，不计入 trial_id，所以 E_base 的 6 个 trial_id 保持不变（有测试）。
  7. **候选和优先顺序**：(E_update = 1, replay = 0) < (1, 50%) < (2, 0) < (2, 50%)，按 §4.2 的多候选规则选择（epoch 少、不加 replay 的更便宜，排在前面）。所有配置在 D1 上分数相同，所以"4 周中至少 3 周严格更好"实际要求 D2、D3、D4 都更好，这是预期的。
  8. **稳健性**：按规则排名前 2 的配置（第 2 名 = 去掉第 1 名后规则会选中的；没有候选通过时按合并分数排）各再用 seed 43、44 跑一遍，只影响 fine-tune 的打乱顺序和 replay 抽样；汇报均值、标准差、最小值、最大值。选择仍以 seed 42 的结果为准，额外 seed 只用于汇报。写在 `configs/hpo/d_phase.yaml` 新增的 `robustness` 部分。
  9. **内存**：训练链按 fold 逐个处理，每周的 fine-tune 数组在这一周用完后释放；fine-tune 的历史只读取涉及的用户的购买（与推理相同）。预测的每个阶段（历史、向量、候选）记录内存，用来查明 E_base 评估中第一个 trial 6.27 GB 峰值的原因（结果在运行后另记）。
- **proposal §6.4 已同步。**

## 2026-11-17 R6 训练链：采用 (E_update = 2, replay = 50%)，偏离规则选中的 (2, 0)（人工决定）

- **如实说明**：这是一次**偏离预先登记规则的人工决定**，是在**看到训练链全部结果（4 个候选 + 4 个额外 seed）之后**做的。按 §4.2 的多候选规则，选中的是 (E_update = 2, replay = 0)：4 个候选都通过，最高合并分数是 (2, 50%) 的 14.634%，(2, 0) 为 14.576%（低 0.40%，在 1% 的"差不多"范围内），而 (2, 0) 在预先写定的优先顺序中更靠前。`configs/selection_rule.yaml`、`src/hmrec/hpo/selection.py` 和 `configs/hpo/d_phase.yaml` 都**不修改**；`results/d_phase.csv` 和 `results/d_phase_steps/r6_twotower.csv` 中规则的结论（选中 (2, 0)）保持原样。
- **决定**：R6 采用 **(E_update = 2, replay = 50%)**（seed 42 的 trial `835ef201ee4d`，M4 = `artifacts/models/twotower/chain/835ef201ee4d/D4.pt`）。M4 的 union 检查和之后的 F / V / T 阶段都用它。尚未写入 `configs/locks/`，锁定在 M5 统一进行。
- **理由**：
  1. 训练链在 D 阶段只验证了 4 周，但实际要**连续运行 16 周**（到 T4）；规则只比较这 4 周的合并分数，看不到更长时间的累积效应。
  2. 不加 replay 时已经出现**遗忘的迹象**：(2, 0) 的 D4（13.98%）低于它自己的 D3（14.23%）；(2, 50%) 的 D4 为 14.58%，D4/D1 为 1.007（(2, 0) 为 0.966）。
  3. 3 个 seed 的平均合并分数 (2, 50%) 更高（14.630% vs 14.563%），标准差小约 4 倍（0.004 vs 0.016 个百分点）；MAP@12 高 4.4%（1.458% vs 1.396%，只报告、不参与选择的指标）。
  4. 代价只是每周 fine-tune 多约 40 秒（D 阶段每周 58–107 秒 vs 29–54 秒）。
- **影响**：选择偏差只限于 D1–D4（这 4 周不进入 Ranker 训练）；V、T 阶段不受影响，因为配置在 M5 锁定后不再改动。人工选择写在 `configs/m4_union_check.yaml` 的 `manual_selection` 中（在规则选中的配置上覆盖 `replay`），M5 锁定时以它为准。

## 2026-11-17 M4 准备：R6 打分分块、一次 join 算多个指标、只用命中的候选算 union

- **背景**：R1–R5 union 预检查峰值 6.71 GB；R6 的 E_base 评估第一个 trial 采样到 6.27 GB，训练链整个进程的真实峰值（`/usr/bin/time`）5.76 GB，而 0.5 秒采样的各阶段读数只有 1.3–4.4 GB，说明峰值是预测 / 评估中短于采样间隔的瞬时峰值。
- **决定**（都只影响内存和耗时，不改变任何结果，有测试）：
  1. **R6 打分分块单独配置**：`configs/base.yaml` 新增 `chunking.twotower_score_query_chunk_size = 2500`（R5 仍用 `dense_score_query_chunk_size = 5000`）。只用于"用户 × 活跃商品"的分数矩阵；用户 / 商品向量的编码仍按 5,000 分块，与 D 阶段已记录的运行完全相同，保存的 `score` 来自 `dot_pairs`，与分块无关。放在 `base.yaml` 而不是 `r6_twotower.yaml`，是因为后者的任何新键都会改变 R6 已记录 trial 的 trial_id。
  2. **评估时一次 join 算多个指标**：`eval.metrics.ranked_hits` 对候选做一次输入检查、与 label 做一次 join，得到所有命中的 (query, rank)；Recall / NDCG / MAP 的各个 K 都从这张小表计算（`metric_from_hits`，与原来的单指标函数共用同一段计算代码，结果逐位相同）。runner 改用 `FoldEvaluator.per_query_many`。原来 6 个指标各自检查并 join 一次。
  3. **Union 只用命中的候选计算**（预检查那条记录中预告的方案）：每一路的候选先与 label 做一次 semi join（命中的行约占 1/400），union recall、leave-one-out 和 R2 / R3 的活跃池诊断都在这些小表上做；每个 query 的候选数按 `chunking.query_chunk_size` 个 query 分块统计。预检查的直接算法保留在 `union_precheck.py` 中作为对照，测试保证两种算法结果完全一致。
  4. **M4 的候选来源**：R1–R5 复用预检查的文件；R6 不重新训练，用训练链保存的模型逐周预测（D1 用 base epoch 4，D2 用 D1 之后保存的模型，依此类推），并要求每一路每一周的选择指标与 `results/d_phase.csv` 的记录逐位相同，否则停下。
  5. **人工选择的记录方式**：`configs/m4_union_check.yaml` 的 `manual_selection` 在规则选中的配置上覆盖指定的键（R6：`replay: 0.5`），对应的 trial 必须已经在 `results/d_phase.csv` 中。规则本身和它的结论不改。
  6. **R3 活跃池规则的执行方式**：阈值 0.001（2026-11-16 预先登记的 0.1 个百分点）写在 `configs/m4_union_check.yaml`，用 4 周合并的 union recall 判断，严格小于才改。M4 的运行只计算并报告结论（`results/m4_union_check.csv` 的 `r3_rule_switch_to_active_pool_only`）；如果结论是"改"，R3 的配置修改和候选重新生成在 M5 锁定之前另行完成。
- **原因**：M4 是六路（候选更多），M6 之后 R6 每周都要运行，瞬时峰值必须留出余量。

## 2026-11-17 M4 union 检查结果；R3 按预先登记的规则改为只输出 A(a) 内的商品

- **M4 结果**（`scripts/m4_union_check.py`，HEAD 90fe260，工作区干净；`results/m4_union_check.csv`）。六路使用 D 阶段的选择（R6 为人工选择的 (E_update = 2, replay = 50%)）；R1–R5 的候选复用预检查的文件，R6 用训练链保存的模型逐周预测；六路 × 4 周共 24 个分数与 `results/d_phase.csv` 的记录全部逐位相同。
  - **六路 union recall**：D1 26.53%、D2 27.18%、D3 23.48%、D4 25.00%，**4 周合并 25.27%**（D1–D2 26.86%，D3–D4 24.23%）。**G2（≥ 24%）达标**。D3 单周 23.48% 低于 24%，不处理。相对五路预检查（20.31%）提高 4.96 个百分点，全部来自 R6。
  - **Leave-one-out**（去掉一路后 4 周合并 union recall 的下降，百分点）：R6 **4.96**、R1 3.03、R5 1.42、R4 1.16、R3 0.51、R2 0.50。R6 的独特贡献最大；加入 R6 之后其他路的独特贡献都变小（R1 5.43 → 3.03，R5 2.23 → 1.42，R4 1.73 → 1.16）。
  - **没有任何一路命中的 query**：50.5%（五路时 58.2%）。
  - **每个 query 去重后的候选数**：平均 312.9，P50 327，P90 373（五路时 257.9 / 276 / 311；上限 480）。
  - **内存**：`/usr/bin/time` 的真实峰值 **6.66 GB**（0.5 秒采样只看到 5.22 GB），超过 6 GB、低于 8 GB；全程 98 秒。尖峰所在的阶段当时无法从日志判断。
- **R3 规则的判定和执行**（2026-11-16 预先登记：去掉 R3 在 `A(a)` 外、且没有被其他任何一路召回的候选后，4 周合并 union recall 下降 < 0.1 个百分点 → R3 改为只输出 `A(a)` 内的商品）：
  - 实测：R3 的候选中 23.4% 在 `A(a)` 外，命中率 0.018%（`A(a)` 内 0.456%，约 25 倍）；`A(a)` 外的候选几乎都没有被其他路召回（占 R3 行数的 23.3%），4 周共 494 次命中只来自它们。去掉后 union recall 下降 **0.045 个百分点**（每周 0.037–0.050），小于 0.1。
  - **判定：切换。**用户确认后执行：R3 新增配置项 `active_pool_only`（`configs/recall/r3_sibling.yaml`，默认 false）。过滤放在取 Top-50 **之前**（与 R4 一致），空出的名额由活跃池内的同款商品补上；`P(u, c)`、`M(j)` 和"已购买"的排除仍用全部商品，分数本身不变。**不重新调 γ 和 τ**，沿用 D 阶段的选择（γ 1.0、τ 30）。
  - 选项为 false 时不计入 trial_id（`hmrec.hpo.runner.parsed_config`），所以 `results/d_phase.csv` 中 7 个 R3 trial 的 trial_id 都不变（有测试）。
  - 切换后的 R3 写在 `configs/m4_union_check.yaml` 的 `variants.r3_active_pool` 中。它在 D1–D4 上的分数作为**规则执行的结果**单独记录到 `results/m4_trials.csv`（与 `d_phase.csv` 同样的列），**不参与任何选择**；用它重跑的六路 union 检查写到 `results/m4_union_check_r3_active_pool.csv`，不覆盖原来的结果。运行结果另记一条。
  - proposal §2.4 已同步（R3 改为只输出 `A(a)`，注明是 M4 按预先登记规则做的一次性修改）。
- **R2 只做诊断，保持不变**（预先登记时就是这样规定的）：R2 的候选中 22.4% 在 `A(a)` 外，命中率 0.019%（`A(a)` 内 0.917%），去掉后 union recall 下降 0.025 个百分点。
- **提醒：D 阶段的分数偏乐观。**D1–D4 上的各路分数和 union recall 都带有选择偏差：六路的配置（包括看到结果之后扩展的网格、R6 的人工选择和这次 R3 的切换）都是在这 4 周上决定的，R6 的 base 模型也紧接在这 4 周之前训练。F、V、T 周的 union recall 可能低于 25.3%，以后各周照常记录；G2 在 F / V / T 每周的检查以实测为准。D1–D4 不进入 Ranker 训练（§3）。
- **M4 之后的代码改动**（只影响内存和日志，不改变结果，有测试）：union 检查改为一次只处理一个 fold（fold 的 queries、labels、活跃池和候选在这个 fold 结束时释放）；每个阶段单独记录内存峰值，采样间隔可用 `--memory-sample-interval` 指定（定位尖峰时用 0.05 秒）。
- **proposal §1.3 已按实测更新**：每个 query 平均 313 个候选（原假设 250）→ 合并去重后每周约 2,570 万行；特征宽表每块（1 万 query）约 0.68 GB；Ranker 推理每块约 0.65 GB；12 周特征 Parquet 未压缩约 62 GB，压缩后估计 20–40 GB。

## 2026-11-17 R3 改为只输出 A(a) 之后的结果（规则执行的结果，不参与任何选择）

- **来源**：`scripts/m4_union_check.py --variant r3_active_pool`（HEAD 270998b，工作区干净）。R3 切换后的 trial `06025cc3d129` 记录在 `results/m4_trials.csv`，六路 union 检查记录在 `results/m4_union_check_r3_active_pool.csv`；原来的 `results/m4_union_check.csv` 不变。其余五路的候选没有变化，各路各周的分数与 `results/d_phase.csv` 逐位相同。
- **R3 自己**（Recall@50，D1 / D2 / D3 / D4）：3.526 / 3.574 / 2.890 / 3.188，4 周合并 **3.240%**；切换前（trial `3188358fa459`）为 3.257%，下降 0.016 个百分点（相对 −0.51%）。候选行数 4 周合计从 1,169 万降到 945 万（平均每个 query 29.8 → 24.1 行），全部在 `A(a)` 内。
- **六路 union recall**：D1 26.51%、D2 27.15%、D3 23.46%、D4 24.97%，**4 周合并 25.24%**（切换前 25.27%，下降 **0.025 个百分点**）。**G2（≥ 24%）仍然达标。**规则判定时估计的下降是 0.045 个百分点；实际更小，因为空出的名额由活跃池内的同款商品补上，补回了一部分命中。
- **Leave-one-out**（去掉一路后 4 周合并 union recall 的下降，百分点）：R6 4.96、R1 3.03、R5 1.41、R4 1.15、R2 0.50、R3 **0.48**（切换前 0.51）。
- **没有任何一路命中的 query**：50.55%（切换前 50.53%）。
- **每个 query 去重后的候选数**：平均 **307.1**，P50 320，P90 **364**（切换前 312.9 / 327 / 373；上限 480）。合并前 D1 3,219 万行、D3 4,947 万行；去重后 D1 约 2,440 万行、D3 约 3,661 万行。
- **内存**：这次运行一次只处理一个 fold，整个进程的真实峰值 **4.78 GB**（`/usr/bin/time`；采样到的最大值是 D4 整个 fold 的 4.72 GB），第一次 M4 运行（四周一起处理）是 6.66 GB。**这个 4.78 GB 不包含 R6 现场预测**：R1、R2、R4、R5、R6 的候选都是从已有文件读回的，只有 R3 现场生成。R6 现场预测、以及"六路候选生成 + 合并"在一个进程里的峰值，留到 M6 生成 F1 时实测；单独一路的峰值先在 M5 的 D1 核对中实测（见下一条）。
- **说明**：R3 切换是预先登记规则的执行结果，这个 trial 没有和任何其他配置比较过；γ、τ 没有重新调。

## 2026-11-17 M5：锁定全部 recall 配置

- **决定**：六路 recall 的配置写入 `configs/locks/channels.lock.yaml`（R1–R5）和 `configs/locks/twotower.lock.yaml`（R6），此后不再修改（AGENTS.md §3.3：`configs/locks/` 只能新增）。两个文件都记录写入时的 git commit（HEAD `db801c6`；添加锁定文件的 commit 是它的子 commit）和日期。

  | Channel | 锁定的配置 | 名额 | trial_id | D1–D4 合并分数（只作记录） | 选择依据 |
  |---|---|---|---|---|---|
  | R1 Popularity | A1，窗口 7 天，τ 3（默认配置） | 100 | `d2094d455698` | Recall@100 7.388% | 规则选择 |
  | R2 Repurchase | decayed_frequency，lookback 365 天，τ 30 | 30 | `f562d598b0d6` | Recall@30 4.266% | 规则选择 |
  | R3 Sibling | style_lookback 180，market_window 7，τ 30，γ 1.0，**active_pool_only: true** | 50 | `06025cc3d129` | Recall@50 3.240% | **规则执行**（D 阶段规则选中 `3188358fa459`；M4 预先登记的规则改为只输出 `A(a)`） |
  | R4 ItemCF | CF3，fit_window 365，α 0.7，neighbors 100，seed_window 365，seed_decay 开，τ_seed 30 | 100 | `df281d91d7c1` | Recall@100 9.332% | 规则选择 |
  | R5 WALS | W2，window 180，τ 60，α_conf 60，λ 0.1，factors 128，iterations 15，seed 42 | 100 | `3261309f8e8f` | Recall@100 10.807% | 规则选择 |
  | R6 Two-Tower | base 窗口 2019-12-03 – 2020-06-02，E_base 4，E_update 2，replay 0.5，fine-tune lr 1e-4，chain seed 42，模型和训练设置同 proposal §5 R6 | 100 | `835ef201ee4d` | Recall@100 14.634% | **人工决定**（规则选中 (2, 0) = `ef62019e6c73`；见上面 "R6 训练链：采用 (E_update = 2, replay = 50%)" 一条） |

  "规则选择"包含此前记录的、看到结果之后才加入的网格扩展（R2 的 365 天窗口、R3 的 γ 1.5 / 2.0、R4 的 seed 网格、R5 的 α_conf 60 / 80）；这些记录不变。
- **R6 的模型文件**（`artifacts/` 不进 git，用 sha256 确认以后用的是同一个文件）：
  - `base` = `artifacts/models/twotower/base/epoch_4.pt`，sha256 `0abcec18093362584826806b3a0a5968eb5d106e7684984bb7e8fecf0f59d3ab`（在训练链中预测了 D1）；
  - `f1_start` = `artifacts/models/twotower/chain/835ef201ee4d/D4.pt`，sha256 `2a17e7c21899dc024237fb57a0fe3e8af8e2f6145a04f9c857110d4fc14ea1cf`（M4，预测 F1，F / V / T 的训练链从它继续）；
  - 报警用户：比例 1%、seed 42，从 `artifacts/twotower/data/base/customer_idx.npy`（sha256 `1e8003eb…f0f9`）抽出 7,141 人。锁定文件记录这个数组的 sha256，因为数组一旦重新生成，被排除的用户就可能不同。
- **锁定文件怎么写的、怎么核对的**：两个文件是手写的（数值从 `results/d_phase.csv`、`results/m4_trials.csv` 和 `shasum` 的输出复制）。`tests/unit/test_locks_match_records.py` 不读锁定文件，独立推出每一路应有的配置（在 `results/d_phase.csv` 上重放选择规则 + `configs/m4_union_check.yaml` 中 R6 的人工决定 + R3 的规则执行），再与锁定文件比较：配置完全一致；锁定的 trial_id 等于锁定配置的 hash，也等于记录中的 trial；锁定文件抄录的分数与结果文件逐位相同。
- **代码（设计上新增的部分）**：
  1. **新增底层模块 `src/hmrec/locks.py`**（只依赖 `config.py` 和标准库，与 `vectors.py` 同层）：读取并严格校验两个锁定文件，提供文件 sha256 核对。放在底层是因为 `features/` 以后也需要某些路的锁定参数（例如 §8.3 的 `pop_bucket_score` 要用 R1 的 τ），而 `features/` 不能 import `recall/`。锁定的配置必须写全每一项，不能依赖解析器的默认值（R3 的 `active_pool_only` 缺省时等于 false，锁定文件里必须写出）。
  2. **`src/hmrec/recall/locked.py` 是 M5 之后构建六路的唯一入口**（`load_locked_recall`、`build_locked_channel` / `build_locked_channels`）。M6 之后的代码不再读 `configs/recall/` 的默认值或 `results/d_phase.csv`；`tests/unit/test_locked.py` 扫描 M6–M8 的脚本和 `fusion/`、`features/`、`ranker/`，出现这些读取就失败。
  3. **R6 加载模型前先核对 sha256**，不一致就报错、不加载；路径由 `paths.py` 按锁定的 `base_epoch` / `trial_id` 得到，并要求与锁定文件里写的路径是同一个文件。入口返回的 R6 是"只预测"的 channel，从 `base` 或 `f1_start` 开始；**M6 每周预测之后的 fine-tune（训练链的延续）还没有接到这个入口上**，M6 实现时补上，报警用户通过 `load_locked_alarm_users` 取得。
  4. R1–R5 的构建代码从 `hpo/runner.py` 移到 `src/hmrec/recall/build.py`，HPO 和锁定入口共用同一个函数（分块大小的接法只有一处）。D 阶段的行为不变（有测试）。
- **`configs/base.yaml`：R4 的 `chunking.item_chunk_size` 从 2000 改为 1000。**原因见 2026-11-17 "D 阶段 R4 跑完……" 一条和 proposal §1.3：D 阶段最重的 R4 trial 峰值 4.76 GB，离 6 GB 太近。分块只影响内存和速度，不影响结果（分块与不分块的 `C_w` 逐位相同，有测试）；分块大小、线程数不属于锁定内容。
- **锁定后的核对**（`scripts/m5_verify_locks.py`，每路一个进程；在本次 commit 之后运行，结果另记一条）：只通过锁定文件构建六路，在 D1 上重新生成候选（R6 用 `base` 预测，不训练），每一路的分数必须与记录逐位相同（R1、R2、R4、R5、R6 对 `results/d_phase.csv`，R3 对 `results/m4_trials.csv`）。候选写到 `artifacts/m5/lock_check/`，后续步骤不读取这个目录。同时记录每一路的耗时和 `/usr/bin/time` 的真实内存峰值（R4 是新的分块大小，R6 是现场预测）。
- **提醒**：D1–D4 上的分数偏乐观（各路配置都在这 4 周上决定），只作记录；F / V / T 的各路 recall 和 union recall 以实测为准。
- **proposal 已同步**：§0.1（参考值旁边的 D 阶段实测值）、§1.3（每个 query 307 个候选、P90 364，以及由此重算的各项估算；R4 分块；4.78 GB 的说明）、§5（每一路表格的"M5 锁定"行）、§6.5（锁定内容）、§9.1、§11（M0–M5 标为完成）、§14（新模块）。§5 R2 的表格原来被一段说明文字从中间断开，这次把那段文字移到了表格之后，内容没有改。

## 2026-11-17 真实内存峰值由 memory.py 记录（ru_maxrss）；4.78 GB 的来源

- **4.78 GB 的来源**：上面两条记录（"R3 改为只输出 A(a) 之后的结果"、"M5：锁定全部 recall 配置"）和 proposal §1.3 中的 **4.78 GB** 来自 `/usr/bin/time -l` 的**终端输出**，**没有写入运行日志**（`artifacts/runs/m4_union_check_r3_active_pool/run.log` 里只有采样值，最大 4.72 GB）。所以这个数字现在无法从仓库或日志中核对，只能以当时的终端输出为准。此前记录中标为 `/usr/bin/time` 的其他数字（例如第一次 M4 运行的 6.66 GB）也是同样的来源。proposal §1.3 已注明。
- **决定**：今后真实峰值由 `src/hmrec/memory.py` 自己记录，不再依赖终端输出。
  1. 新增 `read_process_peak_memory` / `log_process_peak_memory`：用标准库 `resource.getrusage(RUSAGE_SELF).ru_maxrss` 读取进程启动以来的真实最大 RSS（与 `/usr/bin/time -l` 是同一个计数器）；单位由 `ru_maxrss_to_bytes` 处理（macOS 是字节，Linux 是 KB）。同时读取 `RUSAGE_CHILDREN`（已结束的子进程中最大的一个，不是总和），有子进程结束时才写进日志。
  2. **两种峰值用不同的名字**：采样峰值（`MemoryTracker.peak_rss_bytes`，日志中的 `peak rss=… (sampled)`，结果中的 `sampled_peak_rss_gb`）可以分阶段，但只是下限；真实峰值（`process_max_rss_bytes` / `process_max_rss_gb`，日志中的 `process max rss=`）是真实值，但整个进程只有一个数，说不出出在哪个阶段。是否超过 6 GB 以真实峰值为准。
  3. 每个 `MemoryTracker` 结束时两种数字都写进日志；脚本在最后一步调用 `log_process_peak_memory`（超过 `memory.limit_gb` 时记 WARNING），并把真实峰值写进这次运行的结果记录。
  4. M5 的锁定核对每次运行向新文件 `results/m5_lock_check.csv` 追加一行：两个分数、是否逐位一致、耗时、`sampled_peak_rss_gb`、`process_max_rss_gb`、`children_max_rss_gb`、git commit。分数不一致的运行也照常记录，然后才报错。
- **没有改的**：`results/d_phase.csv` / `results/m4_trials.csv` 的列（其中 `peak_rss_gb` 是采样峰值）保持不变，因为这两个文件的表头是固定的，D 阶段也已经结束；Two-Tower 每个 epoch / 每周 fine-tune 记录里的 `peak_rss_gb` 同样仍是采样峰值。M0–M4 的脚本没有加"最后一步"的调用，它们通过各自的 `MemoryTracker` 在日志里得到真实峰值。M6 之后新写的脚本和结果记录按第 3 点执行。
- **原因**：采样峰值会漏掉短于采样间隔的尖峰（第一次 M4 运行采样只看到 5.22 GB，真实值是 6.66 GB），而真实值此前只存在于终端输出里，事后无法核对。
- **同时更新的规则（AGENTS.md）**：§5 分层图加入 `locks.py`（底层，与 `vectors.py` 同层）；§3.3 写明 M5 之后六路 recall 的配置只能通过 `src/hmrec/recall/locked.py` 获取；§3.4 新增"新写或修改代码之后必须先停下等用户 commit，工作区干净后才能在全量数据上运行，即使这次运行已经获得许可"。

## 2026-11-17 M5 锁定核对通过；V 阶段的 trial 记录要带真实峰值（M7 之前实现）

- **锁定核对的结果**（`scripts/m5_verify_locks.py`，HEAD `0ce824a`，工作区干净，`git_dirty=False`；每路一个进程；记录在 `results/m5_lock_check.csv`）：只通过锁定文件（`hmrec.recall.locked`）构建六路，在 D1 上重新生成候选。R6 用锁定的 `base`（base epoch 4，sha256 核对通过）现场预测，没有训练。

  | Channel | trial_id | D1 分数（重新生成） | 与记录 | generate 耗时 | 真实峰值（整个进程） |
  |---|---|---|---|---|---|
  | R1 Popularity | `d2094d455698` | Recall@100 0.07497200996179279 | 逐位相同（`results/d_phase.csv`） | 0.5 秒 | 1.98 GB |
  | R2 Repurchase | `f562d598b0d6` | Recall@30 0.05336155656120723 | 逐位相同（`results/d_phase.csv`） | 0.3 秒 | 0.99 GB |
  | R3 Sibling | `06025cc3d129` | Recall@50 0.03526219660601794 | 逐位相同（`results/m4_trials.csv`） | 0.4 秒 | 1.06 GB |
  | R4 ItemCF | `df281d91d7c1` | Recall@100 0.10543883432792314 | 逐位相同（`results/d_phase.csv`） | 25.8 秒 | 4.20 GB |
  | R5 WALS | `3261309f8e8f` | Recall@100 0.12295079519167419 | 逐位相同（`results/d_phase.csv`） | 169.6 秒 | 3.78 GB |
  | R6 Two-Tower | `835ef201ee4d` | Recall@100 0.14477651846610332 | 逐位相同（`results/d_phase.csv` 中 (2, 50%) trial 的 D1） | 13.6 秒 | 3.53 GB |

  - **候选文件逐字节相同**：`artifacts/m5/lock_check/D1/` 下的六个文件与此前的文件（R1、R2、R4、R5：`artifacts/hpo/D/union_precheck/D1/`；R3、R6：`artifacts/m4/candidates/D1/`）用 `cmp` 比较完全相同。这比分数相同更强：R4 用新的 `item_chunk_size = 1000` 生成的候选，与分块 2000 时的文件一字节不差。
  - **真实峰值的记录是对的**：六路的 `process_max_rss_gb`（`ru_maxrss`）与 `/usr/bin/time -l` 的 maximum resident set size 逐字节相等（差 0 字节）。
  - **R4（新分块）**：generate 25.8 秒；真实峰值到 generate 结束时 3.72 GB，整个进程 4.20 GB。没有同一配置、同一 fold、分块 2000 的真实峰值可比，所以不给出"省了多少"。
  - **R6 现场预测**：核对 sha256、加载模型和表 0.4 秒，预测 13.6 秒；真实峰值到 generate 结束时 2.33 GB，整个进程 3.53 GB。
  - **generate 之后峰值还会升高**：R4、R5、R6 在 `generate` 返回之后又升高 0.48 / 1.06 / 1.20 GB，来自候选的校验、写盘和评估（没有分开测）。M6 估算单路峰值时看"整个进程"的数字。六路在一个流程里生成再合并的峰值仍留到 M6 的 F1 实测。
  - 核对生成的候选在 `artifacts/m5/lock_check/`，后续步骤不读取；`artifacts/candidates/` 仍然不存在。
- **结论**：锁定文件能重现记录的结果，M5 完成。建议的 tag：`lock-channels-v1`（由用户打）。
- **决定（V 阶段的 trial 记录）**：V 阶段的 trial 记录要带真实峰值（`ru_maxrss`），字段名与采样峰值分开；**在 M7 之前修改 `TrialRecord`**。D 阶段和 M4 的结果文件（`results/d_phase.csv`、`results/m4_trials.csv`）**不动**，它们的 `peak_rss_gb` 仍是采样峰值。
- **注意（M7 实现时处理，现在只记录）**：`ru_maxrss` 只升不降，是整个进程的峰值。如果多个 trial 在同一个进程里依次运行，后面的 trial 会带上前面 trial 的峰值，记录下来的数字就不是这个 trial 自己的。所以 **M7 的 Ranker HPO 要每个 trial 一个独立进程**。
- **其他确认**：M0–M4 的脚本不补"最后一步"的 `log_process_peak_memory` 调用；`results/m5_lock_check.csv` 保留。
- **proposal 已同步**：§1.3（新增"M5 锁定配置在 D1 上的单路实测"一表；R4 一行和候选生成一行指向它）、§1.4（每周推理的实测）。

## 2026-11-17 M6 前半（候选生成）的实现决定

以下都是在**运行任何 F / V / T fold 之前**确定的；没有看过 D 阶段之后任何一周的分数。六路的配置没有任何改动（全部来自 `configs/locks/`）。

- **决定**：
  1. **每个 (fold, channel) 一个独立进程**（`scripts/m6_generate_candidates.py --fold … --channel …`），监控是另一个进程（`scripts/m6_monitor_candidates.py`），`scripts/m6_run_folds.py` 按时间顺序驱动。原因：真实峰值 `ru_maxrss` 只升不降，一个进程只做一步，它的峰值才是这一步自己的；每一步也能用到全部内存预算。
  2. **三个结果文件，分工不同**：`results/m6_runs.csv`（每个进程一行：耗时、采样峰值、真实峰值；由进程在最后一步写入）、`results/m6_candidates.csv`（监控：每个 fold 每一路和 union 各一行）、`results/m6_twotower_chain.csv`（训练链每个模型的 sha256 和 fine-tune 数字）。候选生成的进程不读 label，所以 recall 不能和耗时写在同一行里，这是把运行记录和监控分成两个文件的原因。
  3. **R6 训练链接入锁定入口**：`recall/locked.py` 新增 `load_locked_twotower_chain`，返回 D 阶段的 `TwoTowerChainChannel`（配置、报警用户都来自锁定文件），每周 fine-tune 调用的就是 D 阶段的 `fine_tune_on_week`，没有另写一份。模型存到 `artifacts/models/twotower/chain_locked/{fold}.pt`，与 D 阶段各条链的 `chain/{trial_id}/` 分开。
  4. **训练链的顺序规则**（`recall/chain_records.py`）：F1 的起点是锁定的 `f1_start`；之后每个 fold 的起点是上一个 fold 的模型，必须存在、有记录、sha256 一致。模型文件和记录必须同时存在或同时不存在，只有一个时报错（不自动修复）。**T4 只预测，不 fine-tune。**
  5. **R6 保存全部商品的向量，但不改变候选**：`A(a)` 内的商品向量沿用生成候选时的那一份（与 D 阶段完全相同的计算），池外的商品另外编码后合并。所以 R6 的候选和分数与只编码 `A(a)` 时相同（有测试），同时 `tt_dot` 对活跃池外的候选（R2）也有值。
  6. **R1 的分数表与候选来自同一次计算**：`PopularityChannel.generate_with_bucket_scores` 只算一次 `S(g, i)`，候选和分数表都从它得到，再用同样的 float64 → float32 转换。原因：Polars 多线程 group_by 的浮点求和顺序不固定，分两次算可能在最后一位不同。分数表存 float32，粒度是 (age_bucket, article_idx)。
  7. **一致性检查在每个 fold 运行时都做**，不只是单元测试：保存的向量 / 分数表先读回来，再与候选的 `score` 逐位比较；不一致就报错，候选文件不写。候选文件最后写，它的存在表示这一步完成；只有附带文件、没有候选文件，视为没做完的一步，报错，不自动清理。
  8. **监控的"前 4 周"**：F1 的前 4 周是 D1–D4，分数取自锁定文件里记录的 D 阶段分数；之后依次用已经写入 `results/m6_candidates.csv` 的周。只对单路的 recall 做这个比较（proposal §6.6 说的是每一路）；union recall 不设标记。标记只是结果表里的一列，不触发任何动作。
  9. **T 周不读 label**：监控脚本只在 fold 的阶段是 F 或 V 时才通过 `eval.monitor.load_label_evaluator` 取评估器；这个函数先检查阶段，T（以及 D）直接报错，不打开文件。`monitor_rows` 收到 T fold 的评估器也报错。T 周的行只有候选行数和每个 query 的候选数。
  10. **新增配置 `configs/m6_candidates.yaml`**：前 4 周对比的设置（4 周、10%，即 proposal §6.6 的数字），以及 fine-tune loss 的停止上限 **7.0**。7.0 是在没有任何 F 周结果时定的：D 阶段训练链每周的训练 loss 为 5.49–5.56，报警用户 loss 为 5.25–5.69，7.0 明显高于两者，达到它说明模型已经发散，而不是某一周略有波动。它只决定"停下来由人看"，不选择、不改变任何东西。
  11. **`hpo/d_phase.py` 读取 fold 输入的代码移到 `recall/inputs.py`**，D 阶段和 M6 共用（表、活跃池、query 名单的读取方式只有一处）。`count_candidates_chunked` 从 `hpo/union_check.py` 移到 `eval/monitor.py`（M6 的监控不属于 HPO 代码路径，T fold 不能进入 HPO 代码）。两处都只是移动，行为不变，原有测试全部通过。
- **新增模块在分层中的位置**：`csv_records.py`、`run_records.py` 属于底层（只依赖标准库和 `memory.py`）；`pipeline.py` 属于上层（它 import `eval/monitor.py` 的常量来写每个 fold 的摘要）；`recall/generate.py`、`recall/chain_records.py`、`recall/inputs.py` 在 `recall/` 内，不 import `eval/`，不读 label（有扫描测试）。
- **没有做的**：`fusion/union.py`（合并去重成宽表）和特征是 M6 后半，不在这次范围内；`load_shared_tables`（读真实的 processed 表）没有单元测试，它会在 F1 的运行中第一次被执行。
- **proposal 已同步**：§6.6（实现）、§14（目录）、§14.2（交接的文件）。

## 2026-11-17 M6 运行之前：分层图补全；重构后先做一次锁定核对的回归检查

- **AGENTS.md §5 和 proposal §14.1 的分层图**加入 `csv_records.py`、`run_records.py`（底层）和 `pipeline.py`（上层），两处一致。
- **回归检查**：M6 前半把 `hpo/d_phase.py` 读取 fold 输入的代码移到了 `recall/inputs.py`，并调整了 R1（`generate_with_bucket_scores`）和 R6 推理（`candidates_from_query_vectors`）的内部结构。这些改动按设计不改变任何结果，但原有的验证只有单元测试。所以在运行 F1 之前，用重构后的代码重跑一次 M5 的锁定核对（六路在 D1 上，分数必须与记录逐位相同）；有任何不一致就不运行 F1。
- **为此新增的一个参数**：`scripts/m5_verify_locks.py --run-name NAME` 把候选写到 `artifacts/m5/lock_check_{NAME}/D1/`，日志写到 `artifacts/runs/m5_verify_locks_{channel}_{NAME}/`，不碰第一次核对留下的 `artifacts/m5/lock_check/D1/`；`results/m5_lock_check.csv` 照常追加新行（用 `git_commit` 和 `finished_at` 区分两次核对）。不带这个参数时行为不变。

## 2026-11-17 M6 前半运行结果：F1–T4 共 12 周的六路候选（只记录，没有任何选择或改动）

- **运行**：先做回归检查，再 F1（HEAD `3cdbf36`），用户确认后 F2–T4（HEAD `7fd6572`，与前者只差结果文件）；工作区都是干净的（`git_dirty=False`）。六路配置全部来自 `configs/locks/`。12 周 × 7 步共 84 个进程全部正常结束，没有触发任何停止条件，也没有跳过的步骤。记录在 `results/m6_runs.csv`、`results/m6_candidates.csv`、`results/m6_twotower_chain.csv`。
- **回归检查**（重构后的代码重跑 M5 锁定核对，`--run-name after_m6_refactor`）：六路在 D1 上的分数与记录逐位相同，候选文件与第一次核对的文件逐字节相同；`results/m5_lock_check.csv` 追加 6 行。
- **F、V 各周的监控**（%，各路为 Recall@quota；"无命中"是没有任何一路命中的 query 比例）：

  | Fold | R1 | R2 | R3 | R4 | R5 | R6 | union | 无命中 | 标记 |
  |---|---|---|---|---|---|---|---|---|---|
  | F1 | 6.55 | 4.90 | 3.49 | 9.90 | 11.52 | 14.52 | 25.61 | 51.64 | R1（比 D1–D4 均值 7.48 低 12.4%） |
  | F2 | 7.33 | 4.49 | 3.48 | 9.56 | 10.96 | 14.77 | 25.89 | 51.74 | |
  | F3 | 7.67 | 4.33 | 3.56 | 9.61 | 10.86 | 15.43 | 26.33 | 51.31 | |
  | F4 | 8.25 | 4.20 | 3.62 | 9.59 | 10.62 | 15.24 | 26.13 | 51.50 | |
  | V1 | 7.81 | 3.87 | 3.41 | 9.07 | 10.15 | 15.53 | 25.44 | 52.09 | R2（比 F1–F4 均值 4.48 低 13.6%） |
  | V2 | 9.16 | 4.00 | 3.41 | 9.14 | 10.53 | 16.68 | 27.54 | 49.00 | |
  | V3 | 8.15 | 4.29 | 3.48 | 9.15 | 10.22 | 16.25 | 26.38 | 52.14 | |
  | V4 | 11.00 | 4.13 | 3.34 | 8.82 | 9.50 | 16.01 | 27.62 | 50.70 | |

  - **G2（union recall ≥ 24%）在 F、V 的 8 周每周都达标**（25.44%–27.62%）。
  - **两次偏低标记**（F1 的 R1、V1 的 R2）按规则只记录，没有做任何处理，配置没有改。其余最接近 10% 的是 V4 的 R5（比前 4 周低 8.5%），没有被标记。
  - 观察（只是记录）：F1 → V4，R4 从 9.90% 降到 8.82%，R5 从 11.52% 降到 9.50%；R1 从 6.55% 升到 11.00%，R6 从 14.52% 升到 16.01%。
- **训练链有没有退化**：没有看到退化的迹象。
  - R6 / R5 的 Recall@100 之比逐周上升：1.26、1.35、1.42、1.44（F1–F4），1.53、1.58、1.59、1.69（V1–V4）。
  - fine-tune 的训练 loss 平稳并缓慢下降：F1 5.573 → T3 5.503（D 阶段 5.49–5.56）。
  - 报警用户 loss 在 4.56–5.67 之间波动（每周只有约 2,000 个样本）；F2 的 4.56 明显低于其他周（其余为 5.05–5.67），没有出现向上的趋势。都远低于 7.0 的停止上限。
  - 11 个模型（F1–T3）的 sha256 都与 `results/m6_twotower_chain.csv` 的记录一致；T4 只预测，没有模型。
- **T1–T4**：监控行的 `labels_read` 全部为 False，所有基于 label 的列为空；只有行数和每个 query 的候选数。去重后每个 query 的候选数：T1 306.4、T2 305.8、T3 308.3、T4 304.2（P50 320–324，P90 360–364）；query 数 80,253 / 75,822 / 72,019 / 68,984。
- **一致性检查**：12 周全部通过。R5、R6 每一行候选的 `score` 都与保存的向量的点积逐位相同；R1 的非补齐候选都与桶内分数表逐位相同（12 周都没有出现补齐行）。
- **内存、耗时、磁盘**：任何进程的真实峰值最高 4.70 GB（F2 的 R6），每个 fold 的最高值都来自 R6（4.29–4.70 GB）。F1 用时 323 秒，F2–T4 共 3,251 秒，合计约 60 分钟。磁盘：候选 1.5 GB（每周 110–140 MB）、R5 向量 676 MB、R6 向量 1.0 GB、训练链模型 369 MB，共约 3.6 GB。
- **提醒**：以上 F、V 的分数只是监控，没有用来选择任何东西；T 周的指标留到 M8。
- **proposal 已同步**：§1.3（候选生成一行的实测）、§1.4（每周 fine-tune 的实测）。

## 2026-11-17 M6 后半（宽表特征）的实现决定，以及 proposal 没写清楚之处的解释（待用户确认）

以下都是在**生成任何一周的特征之前**确定的。这一条写的是 proposal §8 第一版的 56 个特征；在 commit 之前又新增了 15 个（共 71 个），见下一条。六路候选和 recall 配置没有任何改动。

- **需要用户确认的解释**（proposal §8 的定义没有写到这一层；每一条都已写进 §8 对应小节的"落实"说明）：
  1. **`pop_is_fallback`**：R1 没有召回的行取 **NaN**，而不是 0。理由：与其他来源列一致（"没被这一路召回则为 NaN"）；0 会和"来自桶内列表"混在一起。
  2. **"购买行数 / 次数"一律按购买行计**（`item_cnt_*`、`user_cnt_*`、`ui_cnt_item`、`ui_cnt_pcode`，以及 `ui_share_*` 的分子分母）：同一用户同一天买同一商品两行算 2。理由：与 R1、R2 的计数方式一致。
  3. **`item_repurchase_ratio_52w` 是第 2 条的例外**："买过 2 次及以上"按 **2 个或更多不同的日期**计。理由：同一天的多行通常是一次买多件，不是复购；按行计会把"一次买两件"算成复购。
  4. **`item_buyer_age_mean_4w`**：对 4w 内的**不同买家**求平均（每个买家一次），只用年龄已知的买家。另一种读法是按购买行加权。
  5. **`age`**：用表中的原值，16、17 岁（R1 中归入 `AGE_INVALID` 桶）照常保留，不截断、不置为缺失。
  6. **`item_days_since_first_sale`**："第一次被购买"是数据中（2018-09-20 起）的第一条购买行，不是商品真实的上架日期。
  7. **`ui_cnt_pcode` / `ui_days_since_pcode`**："同 `product_code` 商品"包括候选商品本身。
  8. **`ui_share_*_52w`**：分母是用户 52w 内的全部购买行数；用户 52w 内没有购买时为 NaN，有购买但没有同类目的为 0。
  9. **`ui_price_ratio` / `ui_age_gap`**：任一侧缺失为 NaN；用户均价为 0 时比值也为 NaN。
  10. **`img_sim_hist` / `txt_sim_hist` 的平均向量**：对最近 20 件中有向量的那些做简单平均，同一商品在 20 件中出现几次就计几次（不同天买同一商品算多件，这是 Two-Tower 历史的定义）；平均后 L2 归一化；候选商品的向量也重新归一化（原始向量的范数只保证在 1 ± 1e-3 以内）。
  11. **类别特征的存储**：int16 编码，0 = 缺失，1..K = 排序后的取值序号；`fn`、`active`、`n_channels`、`best_rank` 存为 float32。宽表只有 `query_id`、`article_idx` 和 56 个特征，**不带 `customer_idx`**。
  12. **`configs/locks/features.lock.yaml` 暂不写**（proposal §8.7 原文是"M6 之前写入"）。按用户要求，等 F1 的特征确认后再锁定。
- **其他实现决定**：
  1. **"最近 20 件购买"只有一份实现**：`recall/twotower/data.py` 中建历史的函数（`daily_purchases`、`build_histories`、`load_customer_lines_before`、`load_all_lines_before`）移到 `src/hmrec/data/histories.py`，新增 `recent_purchase_histories`；R6 推理的 `query_histories` 和特征代码都调用它。只是移动，行为不变（原有测试全部通过）；按用户要求，在生成特征之前用 M5 的锁定核对对 R6 做一次回归检查。
  2. **所有窗口都通过 `data/asof.py` 取数**，包括 prev_1w：它是"在 anchor 之前 7 天结束、长 7 天"的窗口（`history_window(anchor − 7d, 7)`），而不是先取 14 天再自己按日期切。
  3. **每个 fold 先算三类表，再按 query 分块 join**：商品表（商品目录全部商品）、query 用户表、用户-商品 / product_code / 类目的聚合小表（只算 query 用户）。分块大小用 `chunking.query_chunk_size`；512 维 / 384 维的 cosine 另按 `configs/features.yaml` 的 `embedding_pair_chunk_size` 分块。分块只影响内存，不影响结果（有测试）。
  4. **特征代码不 import `recall/`，不读 label**（扫描测试覆盖 `features/`、`fusion/`、`data/histories.py` 和 `scripts/m6_build_features.py`）。R1 分数表、R5 / R6 向量、六路候选都按文件读取。
  5. **完成标记和不覆盖**：`features.json` 最后写入；已有它的 fold 跳过；只有 part 文件没有它，视为没做完，报错，不自动清理。自动检查（行数、无重复、两项逐位相同）不通过时不写 `features.json`。
  6. **均值类特征的可复现性**：`item_price_mean_4w` 等用 Polars 的 group_by 求平均，多线程下 float64 的求和顺序不固定，转成 float32 后极少数值可能在最后一位不同。这不影响任何检查（逐位相同的两项检查用的是 `dot_pairs`，不经过 group_by），但两次运行的特征文件不保证逐字节相同。
  7. **独立核对脚本** `scripts/checks/check_features.py`：容差为相对 1e-5（宽表存 float32，脚本用 float64 计算）；计数、天数、排名要求完全相等。
- **没有做的**：`load_fold_data`、`load_candidates`、`load_dense_sources` 三个读盘函数没有单元测试，会在 F1 的运行中第一次执行；`ranker/` 还没有开始。
- **proposal 已同步**：§8.1–§8.7（落实后的定义）、§14、§14.2。

## 2026-11-17 特征从 56 个增加到 71 个（锁定之前的设计修改）；12 条解释已确认；特征文件的 sha256 记录

- **时间点和性质**：这是在**特征锁定之前、没有生成任何一周的特征、也没有任何 Ranker 结果**的情况下做的设计修改，不是根据分数做的选择。依据是 H&M 比赛第一名方案对用户–商品交互特征和多时间窗口的强调。`configs/locks/features.lock.yaml` 仍未写入。
- **上一条中的 12 条解释，用户已全部确认**，第 3 条（`item_repurchase_ratio_52w` 按不同日期计）和第 4 条（`item_buyer_age_mean_4w` 按不同买家平均）按当时的读法执行。
- **新增 15 个特征**（定义见 proposal §8，表中标"新增"；`configs/features.yaml` 更新为 71 个，加载时的数量和名称检查同步更新）：
  - **B 组 +2**：`tt_sim_max_hist`、`tt_sim_top5_hist`。候选与用户最近 20 件购买（与 `img_sim_hist` 同一份历史）之间，R6 商品向量点积的最大值、最大 5 个的平均（不足 5 件取全部）；没有历史为 NaN。
  - **C 组 +1**：`item_discount_ratio` = `[a−7, a)` 成交均价 ÷ `[a−364, a)` 成交均价；任一窗口没有销售为 NaN。
  - **D 组 +1**：`user_purchase_days_52w` = 52w 内有购买的不同日期数；没有为 0。
  - **E 组 +11**：`ui_decay_ptype` / `garment` / `section` / `colour`（全部历史中同属性购买行按 `exp(−age_days / 30)` 加权求和，没有为 0）；`ui_days_since_ptype` / `section`；`ui_share_section_52w`、`ui_share_pgroup_52w`；`ui_lift_ptype_52w`（用户占比 ÷ 全站 52w 占比）；`ui_channel_gap`；`ui_attr_match_cnt`（6 个类目属性中，候选的取值出现在最近 20 件购买里的个数）。
  - 各组数量：A 14、B 7、C 17、D 12、E 21，共 71。类别特征仍是 8 个。
- **两个固定值写进配置，不是超参数**：`decay_tau_days: 30`（`ui_decay_*` 的 τ，与 R2 / R3 / R4 seed 衰减的 τ 相同）、`history_similarity_top_k: 5`。
- **实现上的解释**（新增特征中 proposal / 用户说明没有写到的细节）：
  1. `ui_decay_*`、`ui_days_since_*` 按购买行计，"同属性"包括候选商品本身；衰减在 float64 中求和后存为 float32。
  2. `ui_lift_ptype_52w` 的"全站占比"= 全部用户（不只是 query 用户）52w 内该 `product_type_no` 的购买行数 ÷ 全部购买行数；该类型 52w 内没有任何购买时占比为 0，lift 为 NaN。
  3. `tt_sim_*`：同一商品在最近 20 件中出现两次（不同天各买一次）就计两次；读取 R6 向量时检查商品向量的范数为 1（±1e-3），否则报错。按 query 逐个计算，不需要额外的分块参数（一个 query 只有"候选数 × 历史件数"的小矩阵）；这一步的耗时单独累计，写进 `features.json` 和 `results/m6_runs.csv` 的 detail，内存在每个分块之后记入日志。
  4. `ui_attr_match_cnt` 用的"最近 20 件购买"与 `img_sim_hist`、R6 是同一份（`data/histories.py`），所以 52w 之外的旧购买也算在内。
  5. `item_discount_ratio`：年窗口包含上周，所以上周有销售的商品一定有年均价；只有上周没有销售时为 NaN。
- **价格均值类特征**（`item_price_mean_4w`、`user_price_mean_52w`、`item_discount_ratio` 的两个均值，以及由它们得到的 `ui_price_ratio`）：聚合前把 `price` 转成 Float64，算完再转 float32。新增测试：4,000 行随机价格的数据打乱行序 5 次，这些特征的 float32 结果完全相同。（最初的实现已经是先转 Float64；这次补上说明和测试。）
- **特征文件的记录和核对**：新增底层模块 `src/hmrec/feature_files.py`。每个 fold 的 `features.json` 记录每个 part 文件的 sha256、行数，以及列清单；`verify_feature_files` 核对列清单、文件集合、每个文件的 sha256 / 行数 / 列。**M7 读取特征之前必须调用它。**放在底层是因为 `features/` 写、`ranker/` 读，两者不能互相 import。AGENTS.md §5 和 proposal §14.1 的分层图已加入这个模块。
- **按 71 个特征重算的估算**（proposal §1.3 已更新）：宽表每行 292 B（实际存储 276 B）；一周约 6.9 GB，每块（1 万 query）约 0.83 GB；Ranker 推理每块约 0.81 GB；12 周特征 Parquet 未压缩约 77 GB，压缩后估计 23–46 GB；Ranker 训练的原始特征 r = 80 时约 5.7 GB，贴近 6 GB 上限（本来就要求流式加载）。
- **备选特征清单（未实现）**：
  1. 用户的购买间隔，以及"距上次购买天数 ÷ 平均间隔"；
  2. 商品的典型复购间隔；
  3. 用户的折扣偏好；
  4. 商品买家年龄的分布宽度；
  5. 商品在类目内的热度排名；
  6. ItemCF 对全部候选的分数（现在只有被 R4 召回的行才有 `icf_score`）。

  这些留到第一版 Ranker 的 feature importance 出来之后，按 §4 的规则在 V1–V4 上**作为一个整体候选**来评估（不逐个挑选）。`postal_code` 相关的特征留给下一版，因为 M0 预处理时删掉了这一列，需要重跑预处理。
- **测试**：每个新增特征都有手算例子和时间截断测试（anchor 当天及之后的"有毒"交易不改变结果）；独立核对脚本覆盖其中 6 个（`tt_sim_max_hist`、`item_discount_ratio`、`user_purchase_days_52w`、`ui_decay_ptype`、`ui_days_since_section`、`ui_attr_match_cnt`），共核对 25 个特征。

## 2026-11-18 F1 特征的实测；特征生成的真实内存峰值 6.10 GB 超过 6 GB 上限，用户决定代码保持不变

- **运行**：HEAD `c3e808c`，工作区干净。先做 R6 的回归检查，再只生成 F1 的特征，然后运行独立核对脚本。没有运行其他 fold，没有写 `configs/locks/`。
- **R6 回归检查**（"最近 20 件购买"的函数移到 `data/histories.py` 之后）：D1 的 Recall@100 与记录逐位相同（0.14477651846610332），候选文件与 `artifacts/m5/lock_check/D1/r6_twotower.parquet` 逐字节相同。输出在 `artifacts/m5/lock_check_after_histories_move/D1/`，`results/m5_lock_check.csv` 追加 1 行。
- **F1 特征**（`results/m6_runs.csv` 的 `features` 行，`results/m6_features.csv` 的 71 行）：85,648 个 query，26,373,844 行，73 列，9 个 part 文件，共 1.67 GB；耗时 61 秒。
- **检查全部通过**：行数等于去重后的候选数（与 `results/m6_candidates.csv` 一致）；没有重复的 `(query_id, article_idx)`；`tt_dot` 与 R6 的 `score` 在 8,564,800 行上逐位相同，`wals_dot` 与 R5 的 `score` 在 7,124,000 行上逐位相同；独立核对脚本随机 1,000 行 × 25 个特征全部一致，行已排序、query 不跨文件；`verify_feature_files` 通过。
- **内存超限**：进程的真实峰值 **6.10 GB**（`ru_maxrss`，与 `/usr/bin/time -l` 的 6,550,552,576 字节一致；采样峰值 5.76 GB），超过 `configs/base.yaml` 的 `memory.limit_gb: 6.0`，也超过 proposal §1.3 和 AGENTS.md §4 的"单个任务峰值不超过约 6 GB"。脚本按设计在写完全部文件和记录**之后**报 `MemoryError` 退出（退出码非零），所以文件是完整的。每个分块结束时的 RSS 从 3.80 GB 逐块升到 4.95 GB。
- **决定（用户，2026-11-18）**：特征生成代码**保持不变**。当时提出、但**没有实现**的两处只影响内存的改动是：读候选文件时只读用到的列（现在连 `channel` 字符串列也读进内存）；把特征的 query 分块从 1 万降到 5,000。F1 的特征文件保留，不重新生成。
- **这个决定带来的、还没有定的事**：
  1. `memory.limit_gb` 仍是 6.0，所以 F2–T4 中峰值超过 6 GB 的 fold，脚本同样会在写完文件之后以非零退出码结束。F1 的行数（2,637 万）高于 12 周的平均（约 2,350 万），其余 fold 的峰值可能略低，也可能同样超限；没有测过。
  2. F2–T4 运行时，"真实峰值超过 6 GB"是否仍是停下的条件，等用户在运行前指定。
- **磁盘估算更新**（proposal §1.3 ⑤ 已同步）：F1 每行约 68 B，约为未压缩大小的 23%–25%；12 周约 2.82 亿行，估计约 **18 GB**（此前估计 23–46 GB）。
- **F1 上值得知道的三个特征**（只是记录，没有改动任何特征）：
  1. `pop_is_fallback` 在非缺失处恒为 0：F1 没有 R1 的补齐行（12 周的候选里都没有），它的信息与 `pop_rank` 是否缺失重复。
  2. `ui_lift_ptype_52w` 尾部很重：最大 129,900，135 万行（5.1%）大于 100，中位数 1.19；原因是冷门类型的全站占比极小。
  3. `ui_cnt_item`、`ui_cnt_pcode` 的最大值都是 100：对照原始购买行核对了计数最大的 4 行（100、95、77、73），全部一致，是真实的重复购买，不是截断。
- **proposal 已同步**：§1.3（特征宽表一行的实测、⑤ 的磁盘估算）。

## 2026-11-18 内存上限从约 6 GB 调到 7 GB；特征锁定（`configs/locks/features.lock.yaml`）

- **内存上限**：`configs/base.yaml` 的 `memory.limit_gb` 从 6.0 改为 **7.0**。规则的写法同步改为：本机可用内存 8 GB，单个进程的真实峰值（`ru_maxrss`）上限为 7 GB，留 1 GB 余量（AGENTS.md §4、proposal §1.3）。
  - **原因**：F1 特征生成的真实峰值是 6.10 GB，超过原来的 6 GB 上限（见上一条）。用户决定不改特征生成代码，改为调高上限。
  - **影响范围**：`memory.limit_gb` 是所有脚本共用的一个值，所以这次调整对 recall、特征和之后的 Ranker 都生效，不只是特征生成。从 M5 起有真实峰值记录的运行中，最高是 F1 特征的 6.10 GB，其次是 M6 候选生成的 4.70 GB，都在新上限之内。
  - **没有改写的历史数字**：proposal §1.3 的表和表下说明里，写于此前的"6 GB 上限""6 GB 以内"指当时的上限，保留原文，并在 §1.3 开头加了说明。`configs/base.yaml` 中 `item_chunk_size` 的注释（M5 把分块减半的原因）也保留原文。
  - **测试**：`tests/unit/test_config.py` 中核对 `base.yaml` 的那一条，预期值从 6.0 改为 7.0。这是配置本身按用户决定修改，不是为了让测试通过而改预期值。
- **特征锁定**：新增 `configs/locks/features.lock.yaml`（`locked_on` 2026-11-18，`git_commit` 是写入时的 HEAD `068956d`）。内容：71 个特征的名称、顺序、dtype；8 个类别特征的清单；两个固定值 `decay_tau_days` 30、`history_similarity_top_k` 5；四项设置 `history_length` 20、`rrf_k` 60、`online_sales_channel_id` 2、三个窗口长度 7 / 28 / 364。前三部分是用户最初指定的；四项设置是 agent 在提交前指出"它们同样决定特征取值，而锁定文件提交后不能修改"之后，用户同意加入的。
  - **时间点**：F1 的特征已生成并通过全部检查，F2–T4 的特征还没有生成，也没有任何 Ranker 结果。
  - **怎么生效**：代码仍然读 `configs/features.yaml`。`src/hmrec/locks.py` 新增 `load_features_lock` 和 `check_features_match_lock`，逐项比较 `features.yaml` 和锁定文件，不一致就报错。调用它的有两处：`scripts/m6_build_features.py`（生成每个 fold 之前）和 `tests/unit/test_features_lock.py`。脚本里的这次调用是用户最初的要求之外加的一行，用户已确认保留。
  - **锁定文件没有记录的**：`configs/features.yaml` 的 `embedding_pair_chunk_size`（只影响内存，不影响结果）；类别编码表 `artifacts/features/category_codes.json`（不进 git），由 `save_or_check_category_codes` 在每个 fold 核对是否与已有文件一致。
  - **测试**（`tests/unit/test_features_lock.py`）：真实的两个文件一致；8 个数值（2 个固定值、3 项设置、3 个窗口长度）中改任何一个都被拒绝；删特征、调顺序、改 dtype 被拒绝；`FeaturesConfig` 的字段除特征清单和 `embedding_pair_chunk_size` 外必须全部在锁定核对的清单里，所以以后给 `features.yaml` 加新的数值而不加进锁定，测试会失败。
- **proposal 已同步**：§1.3（上限的写法）、§8.7（特征已锁定）、§14（目录树）。

## 2026-11-18 内存上限从 7 GB 调到 8 GB（等于项目预算）；F2–F4、V1 的特征已生成，V1 保留不重跑

- **运行**（HEAD `ad082b0`，tag `lock-features-v1`，工作区干净）：先用 `verify_feature_files` 确认 F1 完整（9 个 part，26,373,844 行），再依次生成 F2、F3、F4、V1。V1 的真实峰值是 **7.002 GB**，超过当时的 7 GB 上限约 2 MB，脚本在写完全部文件和记录之后以非零退出码结束，运行按规则停下。V2–T4 当时没有生成。
- **五个 fold 的真实峰值**（`results/m6_runs.csv`）：F1 6.10 GB（2,637 万行）、F2 6.25 GB（2,346 万行）、F3 6.83 GB（2,381 万行）、F4 6.71 GB（2,471 万行）、V1 7.00 GB（2,523 万行）。峰值在 6.10–7.00 GB 之间浮动，**与行数无关**：行数最多的 F1 峰值最低。**推测**是分配器没有及时把内存归还给系统（**未验证**）。
- **事实更正**：本机的物理内存是 **32 GB**（`sysctl hw.memsize` = 34,359,738,368 字节）。8 GB 是项目自己定的预算，不是机器的物理限制。
- **决定（用户）**：`configs/base.yaml` 的 `memory.limit_gb` 从 7.0 改为 **8.0**，等于项目预算。特征生成代码不改。
  - **原因**：上一条记录里 7 GB 上限的理由是"留 1 GB 余量"，防止系统内存不足而交换。物理内存是 32 GB，这个理由不成立，所以上限不再低于预算。
  - **同步修改**：AGENTS.md §4、proposal §1.1 和 §1.3 的文字；`memory.py`、`run_records.py` 的 docstring；`tests/unit/test_config.py` 中核对 `base.yaml` 的预期值（7.0 → 8.0，配置本身按用户决定修改）。
  - **如果之后还有 fold 超过 8 GB**：到时候再做"读候选文件时只读用到的列"这一处只影响内存的改动（用户已指定）。
- **V1 的文件保留，不重跑**：V1 的 9 个 part 文件和 `features.json` 完整；生成时的自动检查全部通过（行数等于去重后的候选数、没有重复的 `(query_id, article_idx)`、`tt_dot` / `wals_dot` 与 R6 / R5 的 `score` 逐位相同）；`verify_feature_files` 通过；独立核对脚本随机 1,000 行 × 25 个特征全部一致。F2、F3、F4 同样全部通过。
- **proposal 已同步**：§1.1（8 GB 是预算，物理内存 32 GB）、§1.3（上限等于预算，以及上限三次取值的经过）。

## 2026-11-18 M6 后半：F1–T4 共 12 周的特征全部生成；类别编码表的 sha256

- **运行**：V2–T4 在 HEAD `ab8f537`（上限 8 GB）、工作区干净的状态下依次生成，每个 fold 一个进程，没有任何 fold 停下。F1 在 `c3e808c` 生成，F2–F4、V1 在 `ad082b0` 生成。三个 commit 之间特征生成的代码只多了一行锁定核对（`check_features_match_lock`），计算部分没有改动。
- **12 周的结果**（`results/m6_runs.csv` 的 `features` 行；文件大小是各 part 文件字节数之和）：

  | fold | 行数 | part 数 | 耗时（秒） | 真实峰值（GB） | 文件大小（GB） |
  |---|---|---|---|---|---|
  | F1 | 26,373,844 | 9 | 61.0 | 6.10 | 1.67 |
  | F2 | 23,460,027 | 8 | 55.2 | 6.25 | 1.49 |
  | F3 | 23,814,920 | 8 | 57.2 | 6.83 | 1.52 |
  | F4 | 24,714,244 | 9 | 58.6 | 6.71 | 1.57 |
  | V1 | 25,226,147 | 9 | 59.7 | 7.00 | 1.61 |
  | V2 | 22,935,793 | 8 | 54.2 | 7.25 | 1.45 |
  | V3 | 21,965,434 | 8 | 51.4 | 6.76 | 1.40 |
  | V4 | 22,293,247 | 8 | 51.7 | 6.88 | 1.41 |
  | T1 | 24,592,808 | 9 | 57.1 | 7.20 | 1.55 |
  | T2 | 23,186,146 | 8 | 53.6 | 7.13 | 1.46 |
  | T3 | 22,204,166 | 8 | 51.2 | 6.78 | 1.39 |
  | T4 | 20,984,180 | 7 | 48.1 | 6.76 | 1.31 |

  合计 281,750,956 行，约 11 分钟，磁盘 17.8 GB。真实峰值最高是 V2 的 7.25 GB，在 8 GB 上限之内；峰值与行数无关（见上一条）。
- **检查**：12 个 fold 生成时的自动检查全部通过（行数等于去重后的候选数、没有重复的 `(query_id, article_idx)`、`tt_dot` / `wals_dot` 与 R6 / R5 的 `score` 逐位相同）；`verify_feature_files` 对 12 个 fold 全部通过（列清单取自 `features.lock.yaml`）；独立核对脚本对 12 个 fold 都是随机 1,000 行 × 25 个特征全部一致，行已排序、无重复、query 不跨文件。独立核对脚本自身的峰值是 3.67–5.75 GB（F2–T4 的 11 次，`/usr/bin/time -l`，没有写入结果文件；F1 那次没有测）。
- **T1–T4**：特征代码不读 label，这四周没有计算任何基于 label 的指标。
- **类别编码表**：`artifacts/features/category_codes.json` 的 sha256 是 `049accdf849d05b9bf5c6726225c50a9e2197d2923cffcf6db219ac1a2484dff`。这个文件由 F1 的运行写入，之后每个 fold 核对自己的编码与它一致；F2 生成之前、V1 之后、T4 之后三次计算的 sha256 相同。文件在 `artifacts/` 下，不进 git，所以把 sha256 记在这里。12 个 fold 的类别特征编码范围相同（例如 `product_type_no` 1–132、`section_no` 1–57、`colour_group_code` 1–50）。
- **各特征在 12 周之间的变化**（`results/m6_features.csv`，只是记录，没有改动任何特征）：
  1. **缺失率没有跳变**：71 个特征中 41 个的缺失率在 12 周内变化不到 0.5 个百分点；相邻两周变化最大的是 `wals_dot`（V2 10.97% → V3 9.85%，1.12 个百分点）。V3 这一周没有历史的用户偏少（2.92%，其他周 3.2%–4.1%），所以与用户历史有关的特征在 V3 的缺失率都低一些。
  2. **缺失率的缓慢趋势**：`item_discount_ratio` 从 2.40% 升到 4.24%（上周没有销售的候选商品变多）；`ui_days_since_ptype` 从 30.69% 降到 29.2% 左右，`ui_days_since_section` 从 24.72% 降到 22.21%。
  3. **均值随时间漂移的特征**（F → V → T 方向一致，不是单周跳变）：`tt_dot` 0.441 → 0.379（逐周下降，共 14%）；`rep_score` 0.213 → 0.164；`icf_score` 0.0284 → 0.0217；`sib_score` 37.6 → 14.1（F1 最高，F2 起 12–21）；`ui_decay_ptype` 0.564 → 0.380，`ui_decay_garment`、`ui_decay_section` 类似；`user_cnt_4w` 4.19 → 2.88；`item_price_mean_4w` 0.0225 → 0.0300（V4 起上升，对应秋季商品），`ui_price_ratio` 相应从 0.91 升到 1.12。
  4. **销量类特征逐周波动大**：`item_cnt_1w` 在 F1 是 212.8，之后在 107–151 之间；`pop_score` F1 37.6，之后 21.7–31.1，相邻两周最大变化是 V4 → T1 的 +36%。F1 的历史窗口是 6 月最后一周。
  5. `pop_is_fallback` 在 12 周的非缺失处都恒为 0；`ui_lift_ptype_52w` 的最大值每周都很大（4.4 万–38.9 万）。

  第 3、4 点的含义：Ranker 在 F1–F4 上训练、在 V 和 T 上使用时，这些特征的取值分布与训练时不同。这里只记录事实，是否处理留到 M7 按 §4 的规则决定。
- **proposal 已同步**：§1.3（特征宽表一行和 ⑤ 的 12 周实测）、§11（M6 的状态）。

## 2026-11-18 M7（Ranker）：方案和 12 项选择（用户已确认）；选择规则在看到任何结果之前写入配置

- **时间点**：这条记录和 `configs/ranker.yaml`、`configs/hpo/v_phase.yaml` 写于任何 Ranker 结果出现之前。此前在全量数据上只做过一次只读统计（数 F1–F4、V1–V3 的正样本和采样后的行数，label 通过 `eval/` 读取，没有计算任何排序指标），结果见 proposal §9.7 的表。
- **范围和红线**：M7 只涉及 F1–F4 和 V1–V4。预测目标必须是 V 周，训练周必须是 F 或 V 周；`ranker/window.py` 的检查在每个入口的第一步执行，T fold 在打开任何文件之前报错。label 只在 `ranker/sampling.py`（训练）和 `eval/`（评估，`load_m7_evaluator` 只给 V 周）读取。特征只从 `artifacts/features/` 读取，读之前调用 `verify_feature_files`；不新增、不修改任何特征。
- **用户确认的 12 项选择**：
  1. 负采样：每个 query 用 `(seed, query_id)` 独立抽，三个 r 的样本互相嵌套（r = 40 ⊂ 60 ⊂ 80）。
  2. 每个 (fold, r) 一个带全部特征列的采样文件；M7 只采 F1–F4、V1–V3；一个进程读一遍 fold、同时写出三个 r。
  3. Dataset 存成 LightGBM binary 复用（按 proposal §1.3），旁边记录来源文件的 sha256，加载前核对。
  4. col-wise / row-wise：规则在测速之前写进 `configs/ranker.yaml` 并提交（用户补充的要求）。测速时两种方式各训练 50 棵、**不评估**（不读 V 周的 label，看不到任何分数）；每棵树的耗时取前 10 棵之后的平均；row-wise 快 15% 以上且真实峰值不超过 8 GB 才选 row-wise，否则 col-wise。选定的方式在测速之后写进配置并提交，M8 沿用；`mode` 为 `undecided` 时网格拒绝运行。
  5. 不在网格里的参数保持 LightGBM 4.7.0 的默认值并显式写出：`lambdarank_truncation_level` 30、`bagging_by_query` false 等。
  6. V 阶段用新的记录类和三张结果表（`hpo/v_records.py`），D 阶段的 `TrialRecord` 和 `results/d_phase.csv` 不动。这**改变了** 2026-11-17 记录里"在 M7 之前修改 `TrialRecord`"的做法：那条记录的目的（V 阶段的 trial 记录带真实峰值，字段名与采样峰值分开；每个 trial 一个进程）都已做到，只是没有改 D 阶段的类，原因是 V 阶段一个 trial 的形状不同（单个 fold、20 个 k）。选择只从 `results/v_phase_metrics.csv` 计算，合并分数 = `Σ(n_f × mean_f) / Σ n_f`。
  7. k 的规则：每组参数先定自己的 k（与最高合并 MAP@12 相对差距小于 1% 的 k 里取最小的），再按 §4.2 的多候选规则比较；第 2 步每个 r 各自定 k；每一步选出赢家后检查，赢家的 k 是 2000 就停下汇报；没被选中的候选 k 是 2000 时只标注。第 1 步通过后自动进入第 2 步。
  8. 去掉 R6：删掉只由 R6 召回的行和 4 个 R6 特征，由其余五路的 rank 重算 `n_channels`、`best_rank`、`rrf_score`，重新采样，只训练到选中的 k 棵；结果只汇报，不触发任何选择。
  9. 测速的结果单独记在 `results/m7_speed_test*.csv`，不进 `v_phase*.csv`；网格里的默认配置照常从头训到 2000 棵。
  10. 每个 trial 保存每个检查点的 Top-12；`scripts/checks/check_ranker_eval.py` 用朴素循环从 Top-12 和 label 重算一个 trial 的三个指标（这两项是 agent 建议加的，用户已同意）。核对脚本自己不读 label 文件，label 表通过 `eval/ranker_eval.py` 取得。
  11. 分层：`ranker/` 和 `features/` 可以 import `fusion/`，已写进 AGENTS.md §5 和 proposal §14.1。
  12. 训练集 NDCG@12 每 100 棵记一次，不是每棵都算。
- **测速增加的两项**（用户要求）：同时计算 V1 的 RRF baseline，与默认配置在 100、200、300 棵时的 MAP@12、Recall@12、NDCG@12 一起汇报，**300 棵时 MAP@12 低于 RRF baseline 就停下排查、不外推时间**；汇报训练数据的检查（F1–F4 每个 fold 的正样本比例、每个 query 的行数的平均 / P50 / P90、训练集 NDCG@12 在 100、200、300 棵时是否上升）。
- **备选清单**（现在不做；等基线结果出来之后，按 §4 的规则在 V1–V4 上评估）：
  1. `lambdarank_truncation_level`：现在是默认值 30。LightGBM 文档建议略高于评估的 k（例如 k + 3 = 15）；proposal 没有规定。
  2. 把漂移明显的分数类特征换算成 query 内的名次。依据是 M6 记录的特征漂移（上一条记录）：`tt_dot`、`rep_score`、`icf_score`、`sib_score`、`pop_score` 等的均值从 F 到 T 单向变化或逐周波动大，Ranker 在 F、V 周训练、在后面的周使用时，这些特征的取值分布与训练时不同；query 内的名次不受整体水平变化的影响。（agent 的说明：特征已经锁定，这一条如果要评估，需要先决定它与 `features.lock.yaml` 的关系，到时候再定。）
- **LightGBM 4.7.0 的行为**（合成数据上验证，写进了 `ranker/dataset.py` 和 `ranker/predict.py` 的说明）：`lgb.Sequence` 的单行访问必须返回 float64（否则报错），批量访问可以是 float32；经 `Sequence` 构建两次，binary 文件逐字节相同，但它与"直接传 numpy 数组"的分箱不同；分段累加的分数与 `predict(num_iteration=k)` 不逐位相同（300 棵时最大相差 1.4e-14），所以"k 棵树时的分数"以 `staged_raw_scores` 为准，M7 和 M8 都用它；`feature_pre_filter` 默认是开的，关掉之后同一个 binary 才能换 `min_data_in_leaf` 复用。
- **耗时的估计**（合成数据 400 万行按行数线性外推，真实数据可能慢到 1.5 倍；以测速为准）：r = 40 单次训练 2000 棵约 18–38 分钟，单个 fold 打分约 6–10 分钟，完整网格加 ablation 约 18–28 小时（row-wise 约 14–21 小时）。
- **proposal 已同步**：§1.3（训练行数的实测）、§9.4（分段累加；先定 k 再比参数）、§9.5（−R6 的落实）、§9.7（新增：M7 落实后的约定）、§11（M7 的状态）、§14（目录、分层、交接的数据）。

## 2026-11-18 M7 实现中发现的两件事：torch 和 LightGBM 不能在同一个进程；计时用的训练不要求棵数是 100 的倍数

- **torch 和 LightGBM 的 OpenMP 冲突**
  - **现象**：写完测试后，集成测试在 LightGBM 构建 Dataset 时段错误，完整的 `make test-fast` 卡住不结束（等了 10 分钟后手动停掉）；单独用 python 先后 import torch 和 lightgbm 再训练，进程直接退出，报 `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized`。
  - **原因**（用 `DYLD_PRINT_LIBRARIES` 确认）：torch 加载自带的 `.venv/.../torch/lib/libomp.dylib`，LightGBM 加载 Homebrew 的 `/opt/homebrew/Cellar/libomp/22.1.4/lib/libomp.dylib`，一个进程里有两份 OpenMP 运行库。此前没有暴露，是因为 M0–M6 没有用到 LightGBM。
  - **处理**：
    1. M7 的进程不加载 torch。`git_state` 和 `warn_uncommitted_changes` 从 `hpo/runner.py`（它 import `recall/`，进而 import torch）移到新的底层模块 `src/hmrec/git_state.py`；`hpo/runner.py` 仍然转出这两个函数，M2–M6 的脚本和 `tests/unit/test_hpo_runner.py` 不用改；日志里的名字保持 `hmrec.hpo.runner` 不变。`tests/unit/test_m7_code_rules.py` 在一个新进程里 import M7 用到的模块，确认 torch、implicit 和 `hmrec.recall` 都没有被加载。
    2. 测试分两个进程：`uv run pytest tests/unit` 只跑 torch 一侧，`--lightgbm-side` 只跑 LightGBM 一侧（文件名以 `test_ranker_`、`test_v_`、`test_speed_test`、`test_m7_` 开头）；另一侧的文件不会被 import。`make test-fast` 和 `make test` 两个都跑。一个测试进程里同时出现 torch 和 lightgbm 时，`tests/conftest.py` 让这次运行失败。
    3. AGENTS.md §5、§7、§8 写明这两条规则（§5 的 `fusion/` 一条是用户要求的；`git_state.py` 和"不能 import torch"、§7 的"测试分两个进程"是这次为了解决这个问题加的）。
  - **环境上的一个事实**：LightGBM 的 OpenMP 运行库来自 Homebrew（libomp 22.1.4），不在 `uv.lock` 里，所以它的版本不随仓库锁定。
  - **libomp 是 `uv.lock` 之外的系统依赖，升级它之后需要重新做一次"两次训练得到相同模型"的核对**（同一个 binary Dataset、同一组参数训练两次，模型文本必须相同；以及训到 k 棵的模型与训到更多棵的模型的前 k 棵相同），核对通过之后才能继续产生新的结果。README 的安装步骤写明了 `brew install libomp` 和当前的版本 22.1.4。
  - **每个 M7 进程都记录实际用到的库**（`src/hmrec/ranker/runtime.py`）：`results/v_phase.csv`、`results/m7_speed_test.csv`（每个 trial 一行）和 `results/m7_runs.csv`（采样、构建 Dataset、RRF baseline 各一行）都有三列 `lightgbm_version`、`libomp_path`、`libomp_version`。路径取自操作系统给出的"本进程已加载的库"（macOS 的 dyld 列表），不是按安装位置猜的；版本从 Homebrew 的路径 `.../Cellar/libomp/<版本>/...` 里读出，其他位置记为 `unknown`。两个驱动脚本（`m7_speed_test.py`、`m7_ranker_hpo.py`）自己不写结果行，只把这三项写进运行日志。进程里如果加载了两份 OpenMP 运行库，`read_lightgbm_runtime` 直接报错，每个 M7 脚本在启动时调用它。
- **计时用的训练不要求棵数是检查点的倍数**：最初的检查要求每个 trial 的棵数都是 `checkpoint_every`（100）的倍数，而测速里比较两种直方图方式的训练是 50 棵，会被这条检查拒绝。小数据的集成测试发现了这个问题（在全量数据上运行之前）。现在只有要评估的 trial 才要求棵数落在检查点上。
- **还没有做的**：`configs/ranker.yaml` 的 `histogram.mode` 仍是 `undecided`；还没有在全量数据上运行任何 M7 的步骤；`configs/locks/ranker.lock.yaml` 在 V4 结束后才写。

## 2026-11-18 M7 测速的结果；直方图方式选定 row-wise；树的上限从 2000 改为 1000（看到 V1 测速结果之后的决定）；打分改为更小的分块；前 300 棵的自动核对

- **运行**：`scripts/m7_speed_test.py`，HEAD `25cacf2`，工作区干净（各行 `git_dirty=False`）；每一部分一个进程；LightGBM 4.7.0，libomp 22.1.4（`/opt/homebrew/Cellar/libomp/22.1.4/lib/libomp.dylib`，每行都有记录）。结果在 `results/m7_speed_test.csv`、`results/m7_speed_test_metrics.csv`、`results/m7_runs.csv`（commit `eede656`）。
- **V1 上的指标**（默认配置 31 / 500 / r = 40，row-wise；82,802 个 query，对全部 25,226,147 个候选打分）：

  | | MAP@12 | Recall@12 | NDCG@12 | 训练集 NDCG@12 |
  |---|---|---|---|---|
  | RRF baseline | 0.018511 | 0.048789 | 0.031607 | — |
  | 100 棵 | 0.028597 | 0.064804 | 0.044900 | 0.336789 |
  | 200 棵 | 0.029305 | 0.066029 | 0.045978 | 0.349325 |
  | 300 棵 | 0.029719 | 0.066720 | 0.046538 | 0.357690 |

  300 棵时 MAP@12 比 RRF baseline 高 61%，没有触发"低于 baseline 就停下排查"。训练集 NDCG@12 逐次上升。这是一个 fold、一组参数、300 棵的结果，只用来确认流程和估算时间。
- **训练数据的检查**：F1–F4（r = 40）的行数 2,951,046 / 2,515,389 / 2,585,402 / 2,662,389，与写方案时只读统计的数字逐个相同；正样本比例 2.453% / 2.451% / 2.451% / 2.449%；每个 query 的行数平均 71.2 / 68.6 / 68.7 / 68.1，P50 都是 41，P90 都是 123。
- **耗时和内存**：采样每个 fold 21–24 秒，真实峰值 3.92–4.13 GB；构建 Dataset（10,714,226 行，154,815 个 query）6.6 秒，2.00 GB，binary 644 MB；训练 50 棵（不评估）col-wise 0.682 秒/棵、1.47 GB，row-wise 0.368 秒/棵、1.99 GB；默认配置 300 棵：训练 103.5 秒（0.342 秒/棵），打分加评估 86.4 秒，整个 trial 的真实峰值 **5.69 GB**；RRF baseline 1.7 秒。采样文件 4 个 fold 共 4.0 GB。
- **直方图方式：row-wise**（规则的结果，不是挑选）。按测速之前写进 `configs/ranker.yaml` 的规则：row-wise 每棵树比 col-wise 快 46%（要求 15% 以上），row-wise 进程的真实峰值 1.99 GB（要求不超过 8 GB）。两次计时都不评估，没有读 V1 的 label。`histogram.mode: row_wise` 已写进配置，M8 沿用。
- **树的上限从 2000 改为 1000**（用户的决定）。**如实记录：这是在看到 V1 的测速结果之后做的，是计算预算上的决定，不是事先登记的。**
  - 依据：默认配置在 V1 上，MAP@12 每 100 棵的增量是 +2.5%（100 → 200 棵，0.028597 → 0.029305）和 +1.4%（200 → 300 棵，→ 0.029719），约每次缩小到上一次的六成（1.41 / 2.48 = 0.57）。按这个速度推算，900 → 1000 棵的增量约 0.03%（1.41% × 0.57⁷），远小于 1% 的"差不多"标准，所以 1000 棵之后的树按规则本来也不会被选中。
  - 兜底：`configs/hpo/v_phase.yaml` 的规则"赢家的 k 等于上限就停下汇报"保留，上限现在是 1000；推算如果错了（赢家的曲线到 1000 棵还在上升），会在第 1 步结束时停下。
  - 局限（agent 的说明）：这个推算只来自一个 fold、一组参数（31 叶）、三个点；叶子更多或 `min_data_in_leaf` 更小的配置收敛的快慢可能不同。
  - 改动：`configs/ranker.yaml` 的 `num_trees` 2000 → 1000，检查点为 100, 200, …, 1000。选择规则本身没有改；`v_phase.yaml` 里只改了注释中写死的"2000"。
- **打分改为更小的分块**：测速里一个 part（1 万个 query，约 305 万行）一次打分，整个 trial 的真实峰值 5.69 GB，出在打分阶段；r = 80 的训练数据更大，如果训练占的内存没有归还，打分时可能接近 8 GB。现在一个 part 读入后按 2,500 个 query 一块打分（`configs/ranker.yaml` 的 `scoring_query_chunk_size`，`ranker/predict.py` 的 `query_chunks`）。只影响内存：一行的分数与同一块里的其他行无关，query 不会被切开。测试确认块大小为 1、7、整个 part 时，分数、名次和 Top-12 逐位相同（`tests/unit/test_ranker_predict.py`、`test_v_trial.py`）。新的峰值待第 1 步网格实测。
- **前 300 棵的自动核对**（`hpo/v_phase.py` 的 `check_default_against_speed_test`，接进了网格驱动）：网格的第一个 trial 就是默认配置在 V1 上的训练；它结束后立刻把模型的前 300 棵与测速的模型比较（两个模型文件先各自核对与记录的 sha256 一致，再比较前 300 棵树的文本）。不一致、测速没有对应的 trial、或模型文件不是记录的那个，都直接报错停下，不再运行后面的 trial。测速选中的方式（row-wise）与配置一致，所以两者的 config_id 相同（`ca1a3cdfce49`）。
- **独立核对脚本的一个错误**：`scripts/checks/check_ranker_eval.py` 第一次在测速结果上运行时报错（Top-12 文件有 5 列，脚本按 4 列解包）。这是脚本自己的错误，与结果无关；已修（commit `56096f5`），并补了一个在真实 trial 输出上运行它的测试。修好后的脚本要等这次的代码提交之后才在全量数据上运行，所以**写这条记录时，测速的指标还没有经过独立核对**；核对放在第 1 步网格之前。
- **按 1000 棵重新估算的耗时**（只有 31 叶、r = 40 是实测；63 / 127 叶和打分到 1000 棵是推算）：31 叶单次训练约 6 分钟、单个 fold 打分约 5 分钟；第 1 步（24 个 trial）约 5 小时。
- **下一步**（用户指定）：先用修好的独立核对脚本核对 V1 在 100、200、300 棵的指标；然后只运行第 1 步（6 组参数 × 4 周）和 V1–V4 的 RRF baseline，不进入第 2 步。停止条件：任何 trial 的真实峰值超过 8 GB；前 300 棵的核对不一致；任何检查失败。赢家的 k 等于 1000 时，跑完第 1 步后停下汇报。
- **proposal 已同步**：§1.3（训练和推理两行的实测、打分分块）、§9.4（上限 1000 及其说明）、§9.7（直方图方式的结果、测速结果、自动核对）、§11。

## 2026-11-18 NDCG@K 的定义写清楚（独立核对脚本发现 proposal 的表述有歧义；代码和记录的数值不变）

- **怎么发现的**：第 1 步网格之前，用独立核对脚本（`scripts/checks/check_ranker_eval.py`，朴素循环）从测速保存的 Top-12 重算 V1 在 100、200、300 棵的指标。MAP@12 和 Recall@12 与记录一致（相差小于 1e-9），**NDCG@12 不一致**：重算 0.044708 / 0.045780 / 0.046338，记录 0.044900 / 0.045978 / 0.046538。名次的检查（每个 query 不超过 12 行、名次连续、按 score DESC、article_idx ASC 排序）通过。按停止条件，当时没有运行 RRF baseline 和网格。
- **原因**：proposal §2.5 原来写的是"理想 DCG 用完整的 `G_q` 计算"，表述有歧义。核对脚本照字面读成"理想排序里有 `|G_q|` 个命中，不在 K 处截断"；`src/hmrec/eval/metrics.py`（M1 写的，一直在用）实现的是"理想排序里有 `min(K, |G_q|)` 个命中"。两者只在 label 多于 K 个的 query 上不同：V1 有 1,859 个 query 的 label 多于 12 个（占 2.2%，最多 65 个），整体相差约 0.43%。
- **决定（用户）**：保留 `eval/metrics.py` 的定义，理想 DCG 的命中数是 `min(K, |G_q|)`。那句话的本意是"理想排序用用户实际购买的全部 label，而不是只用被召回的那部分"，截断由 @K 隐含。
- **改了什么**：proposal §2.5 把 NDCG@K 写清楚（DCG 只看前 K 个位置；理想 DCG 的命中数是 `min(K, |G_q|)`；`G_q` 是 label 周实际购买的全部商品，不限于被召回的部分；完美排序得 1.0）；核对脚本的理想 DCG 改成同样的定义，并补了 label 多于 12 个的手算测试。
- **没有变的**：`eval/metrics.py` 和它的测试没有改；此前所有记录里的 NDCG（`results/d_phase.csv`、`results/m4_*.csv`、`results/m7_speed_test_metrics.csv`）都是这个定义，数值不变。**选择指标不受影响**：recall 各路用 Recall@K，Ranker 用 MAP@12，NDCG 从来只是报告项。
- **顺便逐条对照了 §2.5 的其他定义与 `eval/metrics.py`**，没有发现不一致；有四处原来没写出来，这次补进了 §2.5：
  1. Recall@K 的分母是完整的 `|G_q|`，不在 K 处截断（与 AP、NDCG 不同），所以 label 多于 K 个的 query，Recall@K 最高只有 `K / |G_q|`。
  2. AP@K 原来只写了分母 `min(12, |G_q|)`，没有写分子；补上了 `Σ P(r)`（对前 K 位中命中的位置求和）。
  3. 三个指标的 `G_q` 都是实际购买的全部商品；没有预测或没有命中的 query 得 0 分并留在分母里（§2.3 已有，这里重申）。
  4. 位置由 `rank` 决定，`rank` 必须是 1, 2, …, n，同一个 query 内商品不重复；预测里出现 label 表中没有的 query 是错误（代码一直这样检查）。
- **下一步**：代码提交之后，重新核对 V1 的 100、200、300 棵（三个指标都要一致），然后按原计划运行 V1–V4 的 RRF baseline 和第 1 步的 24 个 trial；停止条件不变。

## 2026-11-18 内存上限从 8 GB 调到 12 GB（警戒线）；`num_threads` 保持 6

- **决定（用户）**：`configs/base.yaml` 的 `memory.limit_gb` 从 8.0 改为 12.0，项目的内存预算随之写为 12 GB。时间点：M7 第 1 步结束之后、第 2 步开始之前。
- **依据和边界**（用户给出的）：
  1. 本机物理内存是 32 GB。
  2. 上限只是警戒线，不影响任何计算结果和速度。
  3. 目前实测的最高真实峰值是特征生成的 7.25 GB（V2，`results/m6_runs.csv` 记的是 7.245 GB）。写这条记录时重新查了所有带真实峰值的结果表，各表的最大值是：`m5_lock_check.csv` 4.20 GB（D1 的 R4）、`m6_runs.csv` 7.25 GB（V2 特征）、`m7_runs.csv` 4.13 GB（F4 采样）、`m7_speed_test.csv` 5.69 GB（V1，打分还没有按 query 分块时）、`v_phase.csv` 3.44 GB。
  4. **分块设置保持不变**（`configs/base.yaml` 的 `chunking`、`configs/ranker.yaml` 的 `scoring_query_chunk_size`、各路 recall 自己的分块）。上限放宽不是放宽分块的理由。
  5. 继续记录每个进程的真实峰值（`process_max_rss_gb`）。
- **上限在代码里起什么作用**（agent 的说明，核对过代码）：`run_records.check_memory_limit` 在运行记录写完之后，真实峰值超过上限就报错，让逐步运行的驱动停下；`MemoryTracker` 和 `log_process_peak_memory` 超过时只写一条 WARNING。它不参与任何数值计算。两处与它有关的既有结果不受影响：
  - 直方图方式的规则里有一条"row-wise 进程的真实峰值不超过 `memory.limit_gb`"。测速时上限是 8 GB，row-wise 的峰值是 1.99 GB，按 12 GB 结论相同；选定的 `row_wise` 不变。
  - `configs/base.yaml` 不进入任何已记录的 hash：`data/processed/metadata.json` 的 hash 只覆盖 `configs/preprocess.yaml`；Ranker 的 `config_id` 由实验名、调的参数、窗口、seed、直方图方式和 LightGBM 参数算出，不含内存上限。所以已有的 processed 数据、特征记录和 `config_id`（默认配置仍是 `ca1a3cdfce49`）都照常通过核对。
- **改了什么**：`configs/base.yaml`（数值和注释里的历史）；`tests/unit/test_config.py` 核对仓库配置的那个测试，预期值 8.0 → 12.0（预期值变了是因为用户改了配置，不是为了让测试通过）；`src/hmrec/memory.py` 和 `src/hmrec/run_records.py` 的 docstring 里写着的"8 GB"（只改说明文字，没有改代码）；AGENTS.md §4；proposal §1.1、§1.3（标题、第一段、新增"上限只是警戒线"一段、上限变化的历史、"以真实峰值为准"一句）和 §11 开头的一句。
- **没有改的**：proposal 其余各处和本文件此前记录里的"6 GB""7 GB""8 GB"，都是当时的上限或当时的估算，作为记录保留（§1.3 已加一句说明）。其中 proposal §12 的"Ranker 训练窗口只有 4 周……受 F 阶段长度和 8 GB 内存的限制"是当时做这个选择的理由，窗口长度已经定了，这句话没有改。`configs/ranker.yaml` 注释里的"(<= 8 GB)"是测速时的事实，也没有改。
- **`num_threads` 保持 6**（用户的决定）。原因：线程数会改变多线程浮点求和的顺序，破坏与已锁定结果的逐位一致。也就是说，换了线程数之后，重新生成的候选、特征和模型不能再与已经锁定、已经记录 sha256 的结果逐位核对。（agent 的说明：没有逐个模块实测过改线程数之后差多少；这里记录的是保持不变的理由，不是实测结论。）

## 2026-11-18 M7 第 1 步（`num_leaves` × `min_data_in_leaf`）的结果：保留默认配置 (31, 500)，k = 500

- **运行**：`scripts/m7_ranker_hpo.py --stage baseline`，然后 `--stage grid --last-step leaves_x_min_data`，都用 `caffeinate -i`。HEAD `5a65308`，工作区干净（24 行都是 `git_dirty=False`）；LightGBM 4.7.0，libomp 22.1.4；row-wise，每个 trial 训练 1000 棵，r = 40。13:17 开始，16:47 结束，约 3.5 小时（之前估计约 5 小时）。结果在 `results/v_phase.csv`（24 行）、`results/v_phase_metrics.csv`（244 行）、`results/v_phase_steps.csv`（6 行）、`results/m7_runs.csv`（新增 10 行），commit `084dc09`。
- **停止条件都没有触发**：24 个 trial 的真实峰值最大 3.44 GB，所有进程中最大的是采样的 4.08 GB（当时的上限是 8 GB）；网格里默认配置在 V1 上的前 300 棵与测速的模型相同（自动核对通过）；两个阶段的退出码都是 0；赢家的 k 是 500，不是上限 1000。
- **六组参数的 k 和四周合并 MAP@12**（合并 = 按 query 数加权；V1–V4 的 query 数是 82,802 / 74,833 / 71,094 / 72,035，共 300,764）：

  | 优先级 | (`num_leaves`, `min_data_in_leaf`) | `config_id` | 选中的 k | 合并 MAP@12 | 最高分所在的 k | 最高分 | 相对默认 | 更好的周数 |
  |---|---|---|---|---|---|---|---|---|
  | 0（默认） | (31, 500) | `ca1a3cdfce49` | 500 | 0.032713 | 1000 | 0.032865 | — | — |
  | 1 | (31, 100) | `7f150c722991` | 500 | 0.032619 | 1000 | 0.032849 | −0.29% | 1 |
  | 2 | (63, 500) | `df9104b7b30b` | 300 | 0.032645 | 600 | 0.032804 | −0.21% | 1 |
  | 3 | (63, 100) | `19bc6e9961a6` | 300 | 0.032555 | 600 | 0.032755 | −0.48% | 0 |
  | 4 | (127, 500) | `13a5d78c7479` | 200 | 0.032536 | 500 | 0.032766 | −0.54% | 1 |
  | 5 | (127, 100) | `79e2690d2cff` | 200 | 0.032528 | 400 | 0.032653 | −0.56% | 1 |

  k 的规则：每组参数在与自己最高合并分数相对差距小于 1% 的 k 里取最小的。默认配置的最高分是 1000 棵的 0.032865；400 棵是 0.032525，差约 1.03%，不够；500 棵是 0.032713，差约 0.46%，所以 k = 500。
- **选中的配置和原因**：保留默认 (31, 500, r = 40)，k = 500。五个候选在各自的 k 上，合并 MAP@12 都低于默认，相对默认的改进都是负的（−0.21% 到 −0.56%），没有一个通过 §4.2 的规则。用户已确认这个结果。
- **默认配置（k = 500）和 RRF baseline 的 MAP@12**：

  | | V1 | V2 | V3 | V4 | 合并 |
  |---|---|---|---|---|---|
  | RRF baseline | 0.018511 | 0.019064 | 0.020604 | 0.019662 | 0.019419 |
  | 默认配置，k = 500 | 0.030112 | 0.031332 | 0.035029 | 0.034850 | 0.032713 |

  合并分数比 baseline 高 68%。
- **G1 和 G3 目前的状态**（V 上、第 1 步之后；第 2 步还可能改变选中的配置）：
  - **G1（≥ 0.032）达标**：合并 0.032713，四周等权平均 0.032831。**余量只有 2.2%**（0.032713 / 0.032 = 1.022）；单看 V1（0.030112）和 V2（0.031332）低于 0.032。
  - **G3（4 周中至少 3 周高于 RRF baseline）达标**：4 周都更高。
  - **V 阶段的分数偏乐观**：k 和参数就是在这四周上选的，同一批数据既用来选又用来报分数。加上余量只有 2.2%，T1–T4 上 G1 是否达标现在不能确定，以 M8 的结果为准。
- **另外三点观察**：
  1. 默认配置和 (31, 100) 的最高合并分数都落在上限 1000 棵。规则看的是选中的 k（500），所以没有触发"停下汇报"。曲线末端几乎走平：默认配置 800 → 900 棵增加 0.10%，900 → 1000 棵增加 0.03%，与把上限改为 1000 时的推算（900 → 1000 棵约 0.03%）一致。
  2. 六组参数的差距很小，各自 k 上最多相差 0.56%。叶子越多，最高分出现得越早，之后下降：127 叶在 400–500 棵之后逐步变差，(127, 100) 到 1000 棵时降到 0.032236。
  3. V1、V2 的水平明显低于 V3、V4（默认配置 k = 500：约 0.030–0.031 对 0.035）。baseline 也是 V3 最高，但幅度小一些。原因没有查。
- **耗时和内存**：每个 trial 训练 275–361 秒（31 叶约 0.28 秒/棵，63 叶约 0.30，127 叶约 0.35），打分加评估 150–286 秒（10 个检查点；V1–V4 的候选是 25,226,147 / 22,935,793 / 21,965,434 / 22,293,247 行），合计 433–648 秒；24 个 trial 共 12,472 秒。真实峰值 2.63–3.44 GB；默认配置在 V1 上是 3.15 GB，测速时同一配置（打分还没有按 2,500 个 query 分块，只训练 300 棵）是 5.69 GB。驱动另外生成了 V1–V3 的采样文件（每个 20–23 秒，3.99–4.08 GB）和 V2–V4 的 r = 40 Dataset（每个约 6.5 秒，约 1.9 GB）。
- **独立核对**（`scripts/checks/check_ranker_eval.py`，选中配置的 4 个 trial，k = 500）：四周的名次顺序和三个指标都与记录一致。重算的 MAP@12 / Recall@12 / NDCG@12：V1 0.030112 / 0.067410 / 0.047097，V2 0.031332 / 0.071690 / 0.049311，V3 0.035029 / 0.075796 / 0.053727，V4 0.034850 / 0.077972 / 0.053913。
- **下一步**（用户指定）：运行第 2 步，负采样比例 60 和 80，各 4 周，共 8 个 trial，用第 1 步选中的参数。第 2 步结束后，如果按规则选中的是 80，或者赢家的 k 等于 1000，停下汇报，由用户决定；否则用选中的配置做去掉 R6 的 ablation（4 个 trial），并输出选中配置的 feature importance。停止条件：任何进程的真实峰值超过 12 GB；任何检查失败。`configs/locks/ranker.lock.yaml` 这一步不写。
- **proposal 已同步**：§11（M7 的状态）。

## 2026-11-18 M7 第 2 步、去掉 R6 和 feature importance 的结果；Ranker 配置锁定（`configs/locks/ranker.lock.yaml`）

- **最终配置（用户确认）**：`num_leaves` 31、`min_data_in_leaf` 500、负采样比例 40、k = 500。两步都是规则的结果（保留默认值）。**备选清单里的各项不再评估，直接锁定。**
- **运行**：`scripts/m7_ranker_hpo.py --stage grid`（18:45–20:09），然后 `--stage minus_r6` 和 `--stage importance`（20:09–20:25），都用 `caffeinate -i`；HEAD `a445eeb`，工作区干净（新行都是 `git_dirty=False`）；LightGBM 4.7.0，libomp 22.1.4。结果在 `results/v_phase*.csv`、`results/v_phase_feature_importance.csv`、`results/m7_runs.csv`，commit `1de3531`。第 1 步的 24 个 trial 被跳过（已有记录），前 300 棵的自动核对再次通过。
- **停止条件都没有触发**：36 个 trial 的真实峰值最大 3.84 GB，其余进程最大 3.55 GB（上限 12 GB）；按规则选中的不是 80，赢家的 k 不是 1000，所以按用户事先给的分支继续做了 ablation 和 importance。
- **第 2 步：三个负采样比例**（各自先定 k，再按 §4.2 的规则与默认比较）：

  | r | `config_id` | 选中的 k | 合并 MAP@12 | 最高分所在的 k | 最高分 | 相对默认 | 更好的周数 |
  |---|---|---|---|---|---|---|---|
  | 40（默认） | `ca1a3cdfce49` | 500 | 0.032713 | 1000 | 0.032865 | — | — |
  | 60 | `3f2285d99ba3` | 400 | 0.032664 | 900 | 0.032917 | −0.15% | 2 |
  | 80 | `7cfadca2dcbe` | 400 | 0.032698 | 800 | 0.032952 | −0.04% | 2 |

  - **选择和原因**：保留 r = 40。r = 60 和 r = 80 在各自的 k 上合并 MAP@12 都略低于默认，没有通过规则（规则要求相对提升至少 1%，且 4 周中至少 3 周更好）。
  - **"各自定 k"带来的现象（如实记录）**：r 越大，曲线的最高分越高（0.032865 → 0.032917 → 0.032952），于是"与最高分相差小于 1% 的最小 k"给 r = 60 和 80 选了更小的 k（400），它们在 400 棵时的分数低于默认在 500 棵时的分数，所以上表里的差距是负的。**在相同的 k = 500 上，r = 80 是 0.032821，比 r = 40 的 0.032713 高 0.33%**；r = 60 是 0.032752，高 0.12%。按各自的最高分比，r = 80 高 0.27%，r = 60 高 0.16%。无论哪种比法都远低于 1% 的门槛，结论不变。
  - 耗时和内存：r = 60 每个 trial 训练约 400–413 秒（约 1,530 万–1,570 万行，0.40 秒/棵），r = 80 约 493–534 秒（约 1,980 万–2,030 万行，0.49–0.53 秒/棵）；打分加评估 148–177 秒；真实峰值 3.03–3.67 GB。
- **去掉 R6**（`config_id` `2ef460aae224`；删掉只由 R6 召回的行和 4 个 R6 特征，由其余五路重算三个聚合特征，重新采样，4 个模型各训练到 500 棵）：

  | MAP@12 | V1 | V2 | V3 | V4 | 合并 |
  |---|---|---|---|---|---|
  | RRF baseline | 0.018511 | 0.019064 | 0.020604 | 0.019662 | 0.019419 |
  | 完整系统，k = 500 | 0.030112 | 0.031332 | 0.035029 | 0.034850 | 0.032713 |
  | 去掉 R6，k = 500 | 0.028727 | 0.029881 | 0.033603 | 0.033013 | 0.031193 |
  | 去掉 R6 的相对下降 | −4.60% | −4.63% | −4.07% | −5.27% | **−4.64%** |

  - **合并 MAP@12 从 0.032713 降到 0.031193（−4.64%），低于 G1 的 0.032**；四周都下降，幅度接近。
  - Recall@12 合并 −7.00%（0.072987 → 0.067875），NDCG@12 合并 −6.05%（0.050848 → 0.047773）。
  - 规模：V1 的候选从 25,226,147 行降到 20,764,686 行；V1 的训练窗口里有正样本的 query 从 154,815 个降到 129,874 个（正样本只由 R6 召回的 query 不再有正样本）。
  - 这个实验只汇报，不触发任何选择（§9.5）。
- **Feature importance**（最终配置的 4 个模型，各取前 500 棵，按 gain 的占比；下面是四周的平均；`results/v_phase_feature_importance.csv`）。前 20 个：

  | 名次 | 特征 | 组 | 占比 | 名次 | 特征 | 组 | 占比 |
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

  前 5 个合计 49.0%，前 20 个合计 85.1%。最后 10 个：

  | 名次 | 特征 | 组 | 占比 | 名次 | 特征 | 组 | 占比 |
  |---|---|---|---|---|---|---|---|
  | 62 | `ui_share_index_52w` | E | 0.0499% | 67 | `index_group_no` | C | 0.0040% |
  | 63 | `ui_share_pgroup_52w` | E | 0.0419% | 68 | `pop_is_fallback` | A | 0.0028% |
  | 64 | `product_group_name` | C | 0.0327% | 69 | `fashion_news_frequency` | D | 0.0017% |
  | 65 | `n_channels` | A | 0.0245% | 70 | `active` | D | 0.0013% |
  | 66 | `club_member_status` | D | 0.0045% | 71 | `fn` | D | 0.0008% |

  五组各自的占比：

  | 组 | 特征数 | 占比（四周平均） | 四周的范围 |
  |---|---|---|---|
  | A（召回来源） | 14 | 33.85% | 33.09–35.14% |
  | B（稠密分数） | 7 | 15.02% | 14.27–15.98% |
  | C（商品特征） | 17 | 24.22% | 23.82–24.68% |
  | D（用户特征） | 12 | 5.27% | 4.93–5.82% |
  | E（用户–商品交叉特征） | 21 | 21.64% | 21.23–22.68% |

  4 个 R6 特征合计约 19.0%（`tt_dot` 9.37%、`tt_rank` 7.57%、`tt_sim_max_hist` 1.87%、`tt_sim_top5_hist` 0.19%）。feature importance 只用于分析，不参与任何选择；特征已经锁定。
- **结论（用户）**：Ranker 的参数已经在平台期：8 组配置（6 组叶子数和叶子最小样本数、3 个负采样比例，默认重复一次）的合并 MAP@12 都在 0.0325–0.0330 之间。提升空间在候选和特征，不在这几个参数。备选清单的各项（`lambdarank_truncation_level`、把漂移明显的分数类特征换算成 query 内名次）留到下一版；下一版还可以考虑删掉贡献接近 0 的特征（上表最后几个）。
- **G1 和 G3（V 上，最终配置）**：G1 达标（合并 0.032713 ≥ 0.032，四周等权平均 0.032831），G3 达标（4 周都高于 RRF baseline）。**V 上的余量只有 2.2%，而且 V 阶段的分数偏乐观**（k 和参数就是在这四周上选的），**T 上有可能低于 0.032**。以 M8 的结果为准，到时如实报告。
- **独立核对**（`scripts/checks/check_ranker_eval.py`）：最终配置的 4 个 trial（k = 500）和去掉 R6 的 4 个 trial（k = 500），名次顺序和三个指标都与记录一致（后 4 个是 agent 加跑的）。
- **锁定文件 `configs/locks/ranker.lock.yaml`**（2026-11-18 写入；`lock.git_commit` 是写文件时的 HEAD `1de3531`）。内容：
  - `model`：`num_leaves` 31、`min_data_in_leaf` 500、`num_trees` 500、`seed` 42、`histogram_mode` row_wise，以及传给 LightGBM 的全部参数（7 个分箱参数、24 个训练参数，与 V 阶段运行时的 `configs/ranker.yaml` 逐项相同）；
  - `sampling`：方法（只取有正样本的 query，保留全部正样本，每个 query 用 `(seed, query_id)` 的随机数发生器打乱负样本，取前 `min(负样本数, r × 正样本数)` 个）、`seed` 42、比例 40、采样文件的 row group 行数 100,000；
  - `training_window`：4 周，每次从头训练；`scoring`：分段打分的步长 100、Top-12；
  - `runtime`：LightGBM 4.7.0、libomp 22.1.4、`num_threads` 6；
  - `references`：`features.lock.yaml`、`channels.lock.yaml` 和 `twotower.lock.yaml` 的路径和 sha256；
  - `selection`：`basis: rule_selection`、`default_kept: true`、`config_id` `ca1a3cdfce49`、指向本条记录、V 上的分数（只作记录）；
  - `lock` 段。
- **用户列出的项目之外，agent 加进锁定文件的三处**（都已在汇报里标出）：`sampling.row_group_rows`（它决定采样文件的字节和 LightGBM 构建 Dataset 时一次读入的块；没有验证过改变它不影响结果，所以一并锁定）；`scoring.top_k`（12）；`selection` 里的 `rule_selected_config_id`、`default_kept`、`v_phase_record`（沿用 `channels.lock.yaml` 的写法）。**`twotower.lock.yaml` 的引用**：agent 最初只按用户列出的写了 features 和 channels 两个并提出这个问题；用户决定把 `twotower.lock.yaml` 也加进 `references`，和另外两个一样记录并核对 sha256，原因是 R6 的候选和四个 `tt_` 特征都依赖它。这是在锁定文件提交之前改的，提交的文件从一开始就有三个引用。以上三处用户都已同意。
- **唯一入口 `src/hmrec/ranker/locked.py`**：`load_locked_ranker(paths, base, runtime)` 读锁定文件、核对、返回配置；M8 只能通过它取 Ranker 的配置（AGENTS.md §3.3）。会改变结果的值全部来自锁定文件；`configs/ranker.yaml` 只提供两个只影响内存和日志的值（`scoring_query_chunk_size`、`train_metric_every`）。核对三件事，任何一件不过就报错、不返回配置：
  1. 用锁定文件里的设置重新算 `config_id`，必须等于文件里记录的 `ca1a3cdfce49`（设置被改过就对不上）；
  2. 引用的三个锁定文件（features、channels、twotower）的 sha256 与记录一致；
  3. 本进程的 LightGBM 版本、libomp 版本和 `num_threads` 与记录一致。**后果**：libomp 是 `uv.lock` 之外的系统依赖，`brew upgrade` 之后 M8 的进程会在入口处报错，需要先决定怎么处理（装回 22.1.4，或者重新核对"两次训练得到相同模型"之后另行决定）。
- **与记录的核对 `src/hmrec/hpo/v_lock.py`**：不读锁定文件，在 `results/v_phase_metrics.csv` 上重放两步的选择规则，加上 `configs/ranker.yaml`、`configs/base.yaml` 的固定值和 `results/v_phase.csv` 里赢家 4 个 trial 记录的版本，得到"锁定文件应当写什么"，再逐项比较。它只是核对，不是配置的来源；M8 的代码不调用它。
- **测试**：`tests/unit/test_v_lock.py`（锁定文件与重放的结果逐项一致；锁定的设置与赢家 4 个 trial 行里记录的 `config_json` 完全相同；**把锁定文件的 67 个值逐个改掉，每一个都被拒绝**（其中 64 个由解析、`config_id`、sha256 或与记录的比较拒绝）；写入日期、commit 和指向本条记录的文字无从比较，由 `tests/unit/test_locks.py` 固定）；`tests/unit/test_ranker_locked.py`（入口、三项核对、锁定之后改 `configs/ranker.yaml` 不影响取到的配置、M8 的脚本不出现读默认配置和 V 阶段结果的名字）；`tests/unit/test_locks.py`（严格解析）；`tests/unit/test_ranker_identity.py`（默认配置的 id 固定为 `ca1a3cdfce49`）。
- **顺带的代码移动**（不改变任何结果）：`model_config` 和 `ranker_config_id` 从 `hpo/v_trial.py` 移到 `ranker/identity.py`，因为锁定入口在 `ranker/`，不能 import `hpo/`；`config.py` 的 `_parse_lightgbm_params` 改为公开的 `parse_lightgbm_params`，供 `locks.py` 解析锁定文件里的 LightGBM 参数。
- **没有做的**：没有运行任何数据任务；T1–T4 的任何文件都没有读；M8 的代码还没有写（`scripts/m8_final_eval.py` 仍是空壳）。建议的 tag：`lock-ranker-v1`（由用户打）。
- **proposal 已同步**：§9.6（锁定的内容、入口和核对）、§9.7（V 阶段的结果）、§11（M7 完成）、§14（目录和分层）。AGENTS.md §3.3、§5 写明 Ranker 配置的唯一入口。

## 2026-11-18 M8（T1–T4 的最终评估）：只运行一次；彩排；判定标准；预先登记的次要分析（写于读取任何 T 周 label 之前）

- **时间点**：这条记录、`configs/m8_final.yaml` 和 M8 的全部代码写于任何 T 周的 label 被读取之前。写的过程中只用了小数据（测试里随机生成的 9 个 fold，每个 40 个 query），没有在全量数据上运行任何东西，也没有打开 `artifacts/labels/` 下任何 T 周的文件。
- **最重要的规则（用户）**：T1–T4 的 label 至今没有被读取过。M8 的正式运行只做一次，看到 T 的结果之后不再修改任何配置、代码逻辑或选择。顺序：写代码和测试（只用小数据）→ 用户 commit → 彩排 → 正式运行。发现 bug 只记录、不修复，停下由用户决定。
- **配置和输入**：
  1. Ranker 的配置只通过 `ranker/locked.py` 的 `load_locked_ranker` 取得；候选和特征用 M6 冻结的文件，读取前调用 `verify_feature_files`。
  2. 训练窗口：T1 ← V1–V4，T2 ← V2、V3、V4、T1，T3 ← V3、V4、T1、T2，T4 ← V4、T1、T2、T3；每次从头训练到锁定的 500 棵。
  3. 采样：V1–V3 复用 M7 的 r = 40 采样文件并核对 sha256；V4、T1、T2、T3 新采样，方法、seed、比例来自锁定文件。
  4. T 周的 label 只允许 M8 的入口读取（`eval/final_eval.py`，要带许可）；M6、M7 对 T fold 的保护没有改。
- **彩排**（不读任何 T 周的 label）：用 M8 的代码路径和锁定配置把 V4 当作目标周跑一遍（训练周 F4、V1、V2、V3）。500 棵时的 MAP@12、Recall@12、NDCG@12 必须与 `results/v_phase_metrics.csv` 中选中配置在 V4 的记录逐位相同，模型的前 500 棵也要与 M7 的模型相同。彩排不通过，正式运行不能开始，用代码强制（`final/gate.py`）。
- **彩排里 agent 加的比较**（用户要求的是上面两项；下面几项用的是同一条代码路径上的其他部分，任何一项不同都算彩排不通过）：100–400 棵时的三个指标和 query 数；RRF baseline 的指标；去掉 R6 的系统的指标、模型前 500 棵和 Top-12；完整系统保存的 Top-12（500 棵）与 M7 保存的逐行相同；V4 的 recall 监控行与 M6 的记录相同；彩排的 trial 是用当前代码、在干净的工作区里跑的。原因：正式运行里 G2 靠监控的 union recall 判定，G3 靠 baseline 判定，次要分析靠去掉 R6 的路径，这三条路径也应当先在 V4 上被证明与 M7 / M6 一致。
- **正式运行的内容**：(1) 主结果：T1–T4 每周和合并（query-weighted）的 MAP@12、Recall@12、NDCG@12，以及四周等权平均；(2) RRF baseline 的同样指标；(3) recall 层在 T 周的监控（M6 留到现在的）：每一路的 Recall@quota、六路 union recall、每个 query 的候选数、没有任何一路命中的 query 比例；(4) 预先登记的次要分析：T 上去掉 R6 的 ablation，做法与 M7 相同，只用于报告，不影响任何结论的判定；(5) 判定标准见下；(6) 每个 trial 一个独立进程，记录耗时、真实内存峰值、LightGBM 和 libomp 的版本、git commit，结果写入 `results/t_phase.csv`、`results/t_phase_metrics.csv`，保存每周的 Top-12；(7) 正式运行开始时，在读任何 T label 之前，把"T 周 label 首次读取"的时间和 git commit 记入 `results/t_phase_label_access.csv`。
- **判定标准**（写在 `configs/m8_final.yaml`，运行后不能改）：
  - **G1**：T1–T4 合并 MAP@12 ≥ 0.032；
  - **G2**：T1–T4 **每一周**的六路 union recall ≥ 24%；
  - **G3**：LambdaMART 的 MAP@12 在 4 周中至少 3 周**严格高于** RRF baseline。
  判定由 `final/goals.py` 的纯函数完成，结论由代码写入 `results/t_phase_goals.csv`，只写一次；表里已有结论时，再算出不同的结论会报错，不会追加第二组。
- **闸门怎么强制"彩排通过才能正式运行"和"只运行一次"**（`final/gate.py`）：
  1. 最近一次彩排比较必须是 `passed`，在干净的工作区里运行，并且 HEAD 的代码和配置（`src/`、`scripts/`、`configs/`、`pyproject.toml`、`uv.lock`）与那次彩排的 commit 相同（`git diff --quiet`）。在两次运行之间提交结果表或文档不算改代码。
  2. 驱动在启动任何一步之前写下"首次读取"的记录；每个工作进程自己再过一次闸门，没有这条记录就拒绝（所以手工启动的一步不可能成为第一个读 T label 的程序）。
  3. 这条记录存在之后，HEAD 的代码或配置与它的 commit 不同，任何进程都拒绝。中断的正式运行可以用同一条命令续跑（已记录的步骤跳过）。
  4. M8 的每个进程都要求工作区干净（比 M7 严格：M7 只警告并记 `git_dirty=True`）。
  代码做不到的：不能阻止有人手工删除记录或绕开入口直接读文件。这一条写进了 AGENTS.md §3.3。
- **T 周只在 500 棵处评估**（agent 的做法，已在汇报里标出）：彩排评估 100–500 棵的每个检查点，为了与 M7 比较；正式运行只评估并保存锁定的 500 棵。T 周在其他棵数上的指标用不到，不算出来，事后也就没有可供挑选的东西。500 棵时的分数与是否保留其他检查点无关。
- **与 proposal §10 原文的冲突**：原文要求报告 Recall@12/20 和 NDCG@12/20；用户这次的要求只列了 @12 的三个指标。锁定文件规定每个 query 只保留 Top-12，算不出 @20。按用户这次的要求实现，§10 已改为只报告 @12，并在 §10.1 写明这一处不同；汇报里向用户指出，由用户在 commit 之前确认。
- **"HPO 代码路径里不能出现 T fold"在 M8 的落实**（需要用户确认，已在汇报里标出）：M8 没有 import 任何做调参和选择的模块（`hpo/v_phase.py`、`hpo/v_lock.py`、`hpo/speed_test.py`、`hpo/runner.py`、`hpo/d_phase.py`），测试会扫描。但它复用了 `hpo/v_trial.py` 里与预测哪一周无关的四个步骤（Dataset 的计划、训练、打分和保存、RRF baseline 的计算）和 `hpo/v_records.py` 的结果表读写，T fold 会经过这些函数。这样做是为了让彩排和正式运行走的就是 M7 走过的那几个函数；这些函数不做 fold 检查，也不做任何选择。为此 `hpo/v_trial.py` 做了不改变行为的拆分：`plan_dataset` 拆出 `plan_dataset_for_folds`，`_train` 改名为 `train_planned_model`，`_score_and_save` 改为接收评估器的 `score_and_save`，`rrf_baseline_row` 拆出 `rrf_baseline_metric_row`；M7 的入口照旧先做自己的检查再调用它们，M7 的全部测试通过。另一种做法是把这些通用函数移出 `hpo/` 包，改动面更大，没有做。
- **其他改动**：`eval/monitor.py` 把计算监控行的部分拆成 `build_monitor_rows`（不做阶段检查），M6 的 `monitor_rows` 照旧先检查再调用它；`git_state.py` 新增 `code_changed_since`；`hpo/v_records.py` 的 `results_files` 认识 `t_phase` 和 `m8_rehearsal` 两个新的区域（M7 的脚本不接受它们）；`scripts/checks/check_ranker_eval.py` 可以核对 T 周的 trial，读 T label 也要先过闸门。
- **测试**（只用小数据）：训练窗口的四种情况；T 周 label 只能从 M8 入口读取（没有许可时文件根本不会被打开；M6、M7 的入口仍然拒绝 T fold）；彩排不通过、没有彩排、彩排之后代码变了、工作区不干净时，正式运行被拒绝且没有任何一步被启动；复用的采样文件 sha256 不一致时报错、Dataset 不会被构建；G1–G3 判定函数的手算例子（恰好等于阈值、差一个浮点数、G2 平均达标但有一周不达标、G3 相等不算）；M8 的模块进了"只能通过锁定入口取配置"的扫描；一个小数据的集成测试走完彩排和正式运行，并确认再次运行不启动任何一步、不改任何结果表。小数据的工作区里，M7 的参照结果是用 M7 自己的代码生成的。
- **还没有做的**：彩排和正式运行都还没有运行；最终报告不在这一步写。
- **proposal 已同步**：§10（新增 §10.1）、§11（M8 的状态）、§14（目录、分层、交接的数据）。AGENTS.md §3.3、§5、§7 写明 M8 的规则。

## 2026-11-18 M8 的结果：彩排通过，正式运行完成，G1–G3 都达标；只报告 @12；两处时间戳记的是驱动启动的时间；T 高于 V，原因待诊断

- **只报告 @12 的指标（用户确认；这是在读取任何 T 周 label 之前定的）**：最初的 proposal §10 要求报告 @12 和 @20（Recall@12/20、NDCG@12/20）。`configs/locks/ranker.lock.yaml` 规定每个 query 只保留 Top-12（`scoring.top_k`），@20 的指标算不出来，所以 M8 只报告 MAP@12、Recall@12、NDCG@12。代码和 `configs/m8_final.yaml` 在正式运行之前（commit `18d957f` 及之前）就是这样写的；这条说明按用户的要求放在正式运行之后的文档 commit 里，运行之前没有修改任何被 git 跟踪的文件。
- **另一处确认（用户）**：M8 复用 `hpo/v_trial.py` 和 `hpo/v_records.py` 中与预测周无关的函数，不违反"HPO 代码路径里不能出现 T fold"这条规则的本意。
- **运行**：两次运行都在 HEAD `18d957f`、工作区干净的状态下进行（所有行 `git_dirty=False`），LightGBM 4.7.0，libomp 22.1.4，都用 `caffeinate -i`。结果在 `results/m8_rehearsal*.csv`、`results/m8_runs.csv`、`results/t_phase*.csv`，commit `6a0ee92`。没有触发停止条件：所有 M8 进程的真实峰值最大 3.75 GB（上限 12 GB），没有检查失败。
- **彩排**（`--stage rehearsal`，22:05–22:14，约 8.5 分钟；不读任何 T 周的 label）：**通过，18 项比较与 M7、M6 的记录逐位相同。**
  - 完整系统在 V4 上 100、200、300、400、500 棵的 MAP@12、Recall@12、NDCG@12 和 query 数（5 项）；500 棵时是 0.034849701577104486 / 0.07797187269574504 / 0.0539127774637354，与 `results/v_phase_metrics.csv` 的文本完全相同。
  - 去掉 R6 的系统在 100–500 棵的指标（5 项）；RRF baseline 的指标（1 项）。
  - 两个模型（完整、去掉 R6）的前 500 棵与 M7 的模型相同；两份保存的 Top-12 逐行相同；两个 trial 是用当前代码、在干净的工作区里跑的（6 项）。
  - V4 的 recall 监控行与 `results/m6_candidates.csv` 中 M6 的记录相同（1 项）。
  - 彩排期间没有产生任何 T 周的文件。彩排训练用的采样文件（F4、V1、V2、V3，完整和去掉 R6 各一组）的 sha256 记在 `results/m8_rehearsal_check.csv`；正式运行核对了 V1–V3 的文件与之相同。
- **正式运行**（`--stage formal`，只做了这一次）：22:14:34 开始，22:49:38 结束，约 35 分钟。"T 周 label 首次读取"记在 `results/t_phase_label_access.csv`：2026-11-18T22:14:34，commit `18d957f`。新采样 V4、T1、T2、T3（完整和去掉 R6 各一组），V1–V3 复用 M7 的文件，T4 没有采样。训练周：T1 ← V1 V2 V3 V4，T2 ← V2 V3 V4 T1，T3 ← V3 V4 T1 T2，T4 ← V4 T1 T2 T3（`results/t_phase.csv` 的 `train_folds` 一列）。
- **主结果和 RRF baseline**（T 周只在锁定的 500 棵处评估；合并 = 按 query 数加权）：

  | | T1 | T2 | T3 | T4 | 合并 | 等权平均 |
  |---|---|---|---|---|---|---|
  | query 数 | 80,253 | 75,822 | 72,019 | 68,984 | 297,078 | — |
  | **LambdaMART MAP@12** | 0.034960 | 0.040270 | 0.039673 | 0.042435 | **0.039194** | 0.039335 |
  | LambdaMART Recall@12 | 0.080965 | 0.088694 | 0.086791 | 0.092959 | 0.087135 | 0.087352 |
  | LambdaMART NDCG@12 | 0.054734 | 0.061135 | 0.060277 | 0.063951 | 0.059852 | 0.060024 |
  | RRF baseline MAP@12 | 0.019151 | 0.021471 | 0.021848 | 0.022411 | 0.021154 | 0.021220 |
  | RRF baseline Recall@12 | 0.049777 | 0.055978 | 0.057542 | 0.059756 | 0.055559 | 0.055763 |
  | RRF baseline NDCG@12 | 0.031942 | 0.035476 | 0.036235 | 0.037149 | 0.035094 | 0.035200 |
  | MAP@12 相对 baseline | +82.6% | +87.6% | +81.6% | +89.4% | — | — |

- **recall 层在 T 周的监控**（M6 留到 M8 的部分；`results/t_phase_recall.csv`）：

  | | T1 | T2 | T3 | T4 |
  |---|---|---|---|---|
  | R1 Popularity Recall@100 | 0.122187 | 0.124717 | 0.112166 | 0.127886 |
  | R2 Repurchase Recall@30 | 0.037431 | 0.042449 | 0.043885 | 0.044764 |
  | R3 Sibling Recall@50 | 0.029859 | 0.029862 | 0.031628 | 0.030573 |
  | R4 ItemCF Recall@100 | 0.080344 | 0.085801 | 0.087109 | 0.089116 |
  | R5 WALS Recall@100 | 0.084065 | 0.090588 | 0.094789 | 0.097827 |
  | R6 Two-Tower Recall@100 | 0.173262 | 0.189991 | 0.188963 | 0.204582 |
  | **六路 union recall** | 0.278773 | 0.290767 | 0.290926 | 0.306944 |
  | 每个 query 的候选数（平均 / P50 / P90） | 306.4 / 321 / 360 | 305.8 / 321 / 361 | 308.3 / 324 / 364 | 304.2 / 320 / 360 |
  | 没有任何一路命中的 query 比例 | 50.29% | 49.34% | 49.21% | 48.14% |

  T1 有三路被标出"比前 4 周（V1–V4）的平均低 10% 以上"：R3（−12.5%）、R4（−11.2%）、R5（−16.7%）；T2–T4 没有被标出的。同一周 R1 比前 4 周的平均高 35.3%，R6 高 7.5%。这只是报告项，不改任何配置。
- **预先登记的次要分析：去掉 R6**（只报告，不影响任何判定）：

  | | T1 | T2 | T3 | T4 | 合并 |
  |---|---|---|---|---|---|
  | 去掉 R6 MAP@12 | 0.033202 | 0.037679 | 0.037330 | 0.039141 | 0.036724 |
  | MAP@12 相对完整系统 | −5.03% | −6.43% | −5.91% | −7.76% | −6.30% |
  | Recall@12 相对完整系统 | −6.93% | −8.21% | −8.53% | −9.84% | −8.37% |
  | NDCG@12 相对完整系统 | −6.15% | −7.53% | −7.39% | −8.72% | −7.45% |

- **G1–G3 的判定**（标准在 `configs/m8_final.yaml`，读取 T label 之前提交；结论由代码写入 `results/t_phase_goals.csv`）：
  - **G1 达标**：T1–T4 合并 MAP@12 = 0.039193596845611765 ≥ 0.032。
  - **G2 达标**：每一周的六路 union recall 都 ≥ 24%；最低的一周是 T1 的 27.88%。
  - **G3 达标**：LambdaMART 的 MAP@12 在 4 周都严格高于 RRF baseline（要求至少 3 周）。
- **耗时和内存**：完整系统的 4 个 trial 各 267–279 秒（训练 154–161 秒，0.305–0.319 秒/棵；打分加评估 106–124 秒），真实峰值 2.94–3.14 GB；去掉 R6 的 4 个 trial 各 199–230 秒，3.33–3.57 GB；监控每周约 3 秒、1.8–1.9 GB；baseline 每周约 1.6 秒、0.5 GB；新采样每个 6–8 秒、3.4–3.7 GB；Dataset 每个 5–7 秒、1.5–1.8 GB。
- **独立核对**（`scripts/checks/check_ranker_eval.py --area t_phase`，正式运行之后、任何代码改动之前运行）：T1–T4 完整系统的 4 个 trial（500 棵），名次顺序和三个指标都与记录一致；去掉 R6 的 4 个 trial 也一致（后 4 个是 agent 加跑的）。
- **发现的问题：两处时间戳写的是驱动启动的时间，不是事件发生的时间。只记录，不改结果文件，不改 M8 的驱动代码（用户的决定）。**
  - `results/t_phase_goals.csv` 的 `decided_at` 是 2026-11-18T22:14:34。**实际判定的时间是 22:49:38。** 依据：`results/t_phase.csv` 里最后一个 trial（去掉 R6，T4）的 `finished_at` 是 2026-11-18T22:49:38，判定在它之后才计算；运行日志 `artifacts/runs/m8_final_eval_formal/run.log` 里 "G1 met / G2 met / G3 met" 三行的时间是 22:49:38。
  - `results/m8_rehearsal_check.csv` 的 `checked_at` 是 2026-11-18T22:05:40。**实际比较完成的时间是 22:14:04。** 依据：`results/m8_rehearsal.csv` 里最后一个彩排 trial（去掉 R6，V4）的 `finished_at` 是 2026-11-18T22:14:03，比较在它之后才做；运行日志 `artifacts/runs/m8_final_eval_rehearsal/run.log` 里 "rehearsal PASSED" 一行的时间是 22:14:04。
  - 原因：驱动（`scripts/m8_final_eval.py`）在启动时取了一次时间，把同一个值传给了后面写这两条记录的函数。
  - 影响：只有这两个时间列不准，它们不参与任何数值和判定。`results/t_phase_label_access.csv` 的 `started_at`（22:14:34）是对的，它记的本来就是正式运行开始的时间；`rehearsal_checked_at` 一列是从彩排记录里抄来的，同样是 22:05:40。
- **T 高于 V，与事先的预期相反，原因待诊断。** 正式运行之前的判断是"V 上的分数偏乐观（k 和参数是在这四周上选的），余量只有 2.2%，T 上有可能低于 0.032"。实际是：

  | 周 | query 数 | LambdaMART MAP@12 | RRF baseline MAP@12 | union recall |
  |---|---|---|---|---|
  | V1 | 82,802 | 0.030112 | 0.018511 | 0.254379 |
  | V2 | 74,833 | 0.031332 | 0.019064 | 0.275415 |
  | V3 | 71,094 | 0.035029 | 0.020604 | 0.263840 |
  | V4 | 72,035 | 0.034850 | 0.019662 | 0.276163 |
  | T1 | 80,253 | 0.034960 | 0.019151 | 0.278773 |
  | T2 | 75,822 | 0.040270 | 0.021471 | 0.290767 |
  | T3 | 72,019 | 0.039673 | 0.021848 | 0.290926 |
  | T4 | 68,984 | 0.042435 | 0.022411 | 0.306944 |
  | V 合并 | 300,764 | 0.032713 | 0.019419 | — |
  | T 合并 | 297,078 | 0.039194 | 0.021154 | — |

  LambdaMART 的合并 MAP@12 在 T 上比 V 高 19.8%；RRF baseline 高 8.9%；T 每一周的 union recall 都高于 V 的任何一周；LambdaMART 相对 baseline 的优势也变大了（V 上 +68%，T 上 +82% 到 +89%）。**原因没有查，这里不下结论。** recall 层和 baseline 同时变高，说明至少一部分来自 T 这几周本身更容易召回，但这只是从这张表能读出来的，不是诊断的结论。已知的局限照旧适用：T 阶段不是完全没见过的数据（proposal §12 第 6 条），结果按 retrospective test 报告。
- **之后的规则（用户）**：不重跑 M8，不训练任何模型，不改任何配置和锁定文件；`results/t_phase*.csv` 和 `results/m8_*.csv` 是原始记录，不修改；接下来的诊断只做描述，不用来选择或修改任何东西。T1–T4 已经用过一次，之后的任何改动只能在 V1–V4 上比较和选择（已写进 AGENTS.md §3.2）。
- **还没有做的**：描述性诊断（`scripts/m8_diagnostics.py`）还没有写；最终报告还没有写。
- **proposal 已同步**：§11（M8 的结果）。AGENTS.md §3.2 加了"T1–T4 已经用过一次"的规则。

## 2026-11-19 M8 之后的描述性诊断：第二个读 T label 的入口；四张表的定义；区间的含义（代码已写，还没有在全量数据上运行）

- **背景**：T 上的合并 MAP@12 比 V 高 19.8%，与事先的预期相反，原因没有查（上一条记录）。用户要求做描述性诊断。**规则（用户）**：不重跑 M8，不训练任何模型，不改任何配置和锁定文件；`results/t_phase*.csv` 和 `results/m8_*.csv` 是原始记录，不修改；诊断只做描述，不用来选择或修改任何东西。
- **现有的单一入口不允许诊断读 T label**。`final/gate.py` 要求 HEAD 的代码和配置（`src/`、`scripts/`、`configs/`、`pyproject.toml`、`uv.lock`）与彩排、正式运行的 commit `18d957f` 完全相同；诊断代码是新代码，一提交就不同了。这是闸门应有的行为。按用户的要求先停下说明了方案，用户同意后才写代码。
- **决定：加第二个入口，只给描述性诊断用；M8 的代码一行不改。**
  - 入口：`src/hmrec/eval/diagnostic_labels.py` 的 `load_diagnostic_evaluator`。读 T 周的 label 必须带 `DiagnosticAccess`；D、F、V 周不需要（它们的 label 之前读过很多次）。
  - `DiagnosticAccess` 只由 `src/hmrec/diagnostics/access.py` 签发，条件是：
    1. `results/t_phase_goals.csv` 里有 G1、G2、G3 各一条判定（正式运行已结束，再读 label 改变不了结论）。
    2. 工作区干净。
    3. 正式运行那个 commit（从判定表的 `git_commit` 读出，`18d957f`，tag `m8-run-v1`）时已有的 M8 文件和 `configs/` 下已有文件的内容没有变；**允许新增文件**（用户的决定：诊断的配置放在新文件 `configs/m8_diagnostics.yaml`）。M8 文件指 `src/hmrec/final/`、`eval/final_eval.py`、`ranker/final_window.py`、`scripts/m8_final_eval.py`、`scripts/m8_step.py`。比较用 `git diff --name-status --no-renames`（新函数 `git_state.changed_existing_files`）：状态不是"新增"的文件都算改动，改名算删除。
    4. 在打开任何 T label 文件之前，先在 `results/t_diag_label_access.csv` 追加一行：时间、git commit、用途、脚本名。每运行一次追加一行。
  - `eval/final_eval.py` 不动（用户的决定）。它的说明里写着"唯一入口"，从今天起不再准确；AGENTS.md §3.2 和 proposal §10.2 写明了有第二个入口。
  - 用测试限制（`tests/unit/test_diag_code_rules.py`）：只有诊断代码 import 这个入口；`DiagnosticAccess` 只在一处创建，而且在记录写入之后；诊断代码不 import 训练、采样、打分的代码和 M8 的 trial、闸门、`eval/final_eval.py`；只写 `results/t_diag_*.csv`；其他代码不 import 诊断代码。
- **一个后果：诊断代码提交之后，`scripts/checks/check_ranker_eval.py --area t_phase` 在当前的 HEAD 上不能再运行。** 它读 T label 要过 M8 的闸门。需要重跑时先切回 tag `m8-run-v1`。这个核对在正式运行之后、任何代码改动之前已经跑过，结果在上一条记录里。
- **四张表**（`scripts/m8_diagnostics.py`，配置 `configs/m8_diagnostics.yaml`，结果 `results/t_diag_*.csv`，只追加，每行带 `git_commit` 和 `run_started_at`）：
  - **A 分解**（V1–T4 每周一行）：union recall、Recall@12、Recall@12 ÷ union recall、MAP@12，LambdaMART 和 RRF baseline 各一组。只用结果表，不读 label。
  - **B 集中度**（D1–T4 共 16 周，每周一行）：购买量最高的 100 个、1000 个商品各占该周购买的比例；首次销售在 `[anchor − 28, anchor)` 的商品、anchor 之前没有任何销售的商品，各自的个数和占该周购买的比例。
    - "购买"按 label 文件里去重后的（顾客, 商品）对统计，不是购买行（用户的决定）。
    - 首次销售日从 `t_dat < anchor` 的交易算出（通过 `data/asof.py` 的 `history_window`，每一周单独读一次）。这超出了最初"只用已保存的 Top-12、候选和 label 文件"的要求：这三类文件里没有商品的首次销售日（宽表里的 `item_days_since_first_sale` 只覆盖 F–T 周的候选商品，D 周没有宽表）。用户同意。
    - "anchor 当天及之后首次销售的商品"定义为：出现在该周的 label 文件里，而 anchor 之前的交易里没有它。不读 anchor 之后的交易表。这只是对 label 周本身的统计，不进入任何模型或特征。
  - **C 分组**（V1–T4 每周三行）：按宽表里已保存的 `user_days_since_last_purchase` 分组：NaN 是"无历史"，1–30 是"最近 30 天内买过"，其余是"其他"。每组报告 query 数和占比、union recall、两个系统的 MAP@12。
  - **D 区间**（T 合并、V 合并）：合并 MAP@12 的 95% 区间，LambdaMART、RRF baseline 和两者之差各一行。
- **重算和核对**：LambdaMART 的逐 query AP 用保存下来的文件（`artifacts/predictions/.../{fold}_per_query.parquet`，取锁定的 500 棵那一组）。RRF baseline 的逐 query AP 和逐 query 的 union recall 当时没有保存，用冻结的宽表（先核对 sha256）、候选文件和 label 重新算，用的是项目自己的函数（`fusion/rrf.py` 的 `rrf_baseline`、`FoldEvaluator`）。每一周重算出的平均值必须与结果表里的记录相同（容差 1e-12，沿用 `hpo/selection.py` 的 `SCORE_EQUALITY_TOLERANCE`）；三组的 query 数之和必须等于结果表里记录的 query 数；合并值也核对。任何一项不符就报错，四张表都不写（读取记录那一行会留下，因为 label 确实读过了）。
- **D 的做法：按顾客的 cluster bootstrap（用户的决定，替换了最初"按 query 重抽"的要求）。** 把一个阶段的 4 周合在一起，以顾客（`customer_idx`）为单位有放回地抽，抽的个数等于顾客数；被抽中的顾客带上他在这 4 周里的全部 query；合并 MAP@12 = 被抽中 query 的 AP 之和 ÷ query 数。1000 次，seed 42（写在配置里），取 2.5% 和 97.5% 分位（numpy 默认的线性插值）。"LambdaMART 减 baseline"用同一批抽样（配对）。T 和 V 各做一次，各用同一个 seed 重新开始。原因：同一个顾客在几周里的 query 不是独立的，按 query 重抽会把区间算窄。
  - **这个区间只反映顾客抽样的不确定性，不反映周与周之间的差别。** 每一次重抽里都是同样的 4 周；它回答的是"换一批顾客，这 4 周的合并分数会差多少"，不回答"换 4 周会差多少"。所以它不能用来判断 T 与 V 的差别是不是偶然。
- **几处实现上的选择（agent 定的，用户可以改）**：
  - 用途一栏写的是英文 `descriptive diagnostics`（项目的记录都是英文），不是中文"描述性诊断"。
  - 诊断的全部计算做完、所有核对通过之后才写四张表；写表之前先写读取记录。重复运行会追加新的行（用 `run_started_at` 区分），不覆盖旧的行。
  - 四张表各覆盖哪些周，配置里写的是阶段（`[V, T]` 等），不是 fold 名。A、C、D 只允许 V 和 T（只有这两个阶段有 Ranker 的记录）。
  - 诊断进程会加载 lightgbm（读锁定配置和结果表的模块 import 了它），但不训练也不加载任何模型。它的测试文件以 `test_diag_` 开头，归在 LightGBM 一侧（`tests/conftest.py`）。
  - `tests/unit/test_final_code_rules.py` 里 M8 脚本的清单从 `scripts/m8_*.py` 的通配改成了两个脚本名（`m8_final_eval.py`、`m8_step.py`）。原来的通配会把 `scripts/m8_diagnostics.py` 也当成 M8 的代码，这是不对的：它是正式运行之后写的诊断，不是 M8 的一部分。M8 文件的预期清单没有变。AGENTS.md §3.3 里 M8 代码的范围也相应写成了这两个脚本名。
- **没有做的**：没有在全量数据上运行（等用户 commit）；没有读任何 T 周的 label（`results/t_diag_label_access.csv` 还不存在）；最终报告没有写。
- **proposal 已同步**：新增 §10.2；§10.1、§14、§14.1、§14.2 加了指向。AGENTS.md：§3.2（第二个入口、`check_ranker_eval.py` 要切回 tag）、§3.3、§5、§7。
