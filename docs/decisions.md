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
