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
