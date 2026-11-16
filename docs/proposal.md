# H&M 个性化时尚推荐系统 — 项目方案 v0.1.2

> 日期：2026-11-15（v0.1.1：O1–O5 已决定，新增 §14 目录结构、§15 代码规范）
> v0.1.2（2026-11-15，根据项目骨架和工具链迁移的结果）：§1.2 / §15 改为 Python 3.12 + uv（`uv.lock` 锁定依赖）+ Pyright；§1.3 注明内存峰值是采样值；§2.1 写入预计算 embedding 的实测事实；§4.2 明确"更好"指严格更好；§5 R6 说明缺图像向量的商品用 mask；§8.3 规定 `img_sim_hist` / `txt_sim_hist` 何时为 NaN；§14 更新 `data/` 目录树。
> 状态：重新开始的第一版方案。旧项目的结果只作为参考数字，不沿用任何旧代码、旧配置或旧结论。
> 系统结构：**6 路 recall → 候选合并去重（union + dedup）→ 1 个 LightGBM LambdaMART ranker → Top-12**
> 原则：**能少跑就少跑**。每路只做 1–2 个 ablation 加一个小的 HPO 网格。一次只改一个因素（one-factor-at-a-time, OAT），不做全组合。

---

## 0. 一页总结

### 0.1 目标

| 编号 | 指标 | 目标 | 在哪里检查 |
|---|---|---|---|
| G1 | 端到端 macro **MAP@12**（LambdaMART） | **≥ 0.032**（最低目标） | V1–V4 选模型，T1–T4 最终报告 |
| G2 | 6 路合并后的 macro **union recall** | **≥ 24%** | D1–D4（锁定 recall 前）和 F/V/T 每周 |
| G3 | LambdaMART 优于 RRF baseline | 4 周中至少 3 周 MAP@12 更高 | V1–V4、T1–T4 |

各路 Recall@quota 的参考值（来自旧项目，**存疑，只作为 sanity check，不作为验收标准**）：

| Channel | Quota | 参考 Recall@quota |
|---|---|---|
| R1 Popularity | 100 | ~7% |
| R2 Repurchase | ≤30 | ~4% |
| R3 Sibling | ≤50 | ~3.6% |
| R4 ItemCF | 100 | ~8% |
| R5 WALS | 100 | ~10% |
| R6 Two-Tower | 100 | ~10% |

如果某一路实测结果比参考值低一半以上，先排查 bug（时间截断、ID 映射、去重），再讨论模型本身。

### 0.2 时间安排

| 阶段 | 周 | 日期（2020，含首尾） | 做什么 | 在这里被选择的东西 |
|---|---|---|---|---|
| Base | — | 2019-12-03 – 06-02 | Two-Tower base model 训练 | — |
| **D** | D1–D4 | 06-03 – 06-30 | 6 路 recall 的 **ablation + HPO**，然后冻结全部 recall 配置 | R1–R6 超参数 |
| **F** | F1–F4 | 07-01 – 07-28 | 用冻结配置生成候选，作为 **Ranker 训练数据** | 不选任何东西 |
| **V** | V1–V4 | 07-29 – 08-25 | Ranker **滚动重训 + HPO**，系统级 ablation | Ranker 超参数 |
| **T** | T1–T4 | 08-26 – 09-22 | 全部锁定，最终评估 | 不选任何东西 |

### 0.3 三条铁律

1. **每一周的预测只用 `t_dat < anchor` 的数据**。模型权重、统计量、特征都一样。
2. **每个阶段只选它该选的东西**：D 选 recall，V 选 Ranker，F 和 T 只执行。
3. **超参数（包括 epoch 数和 tree 数）在 4 周上一次选定，然后锁住**。之后每周只更新模型参数（权重或统计量），不再重新选。

---

## 1. 实验环境与资源预算

### 1.1 硬件

| 项目 | 规格 |
|---|---|
| 机器 | MacBook Pro，Apple **M1 Max**（10 核 CPU + 32 核 GPU） |
| 项目可用 CPU | **6 核** |
| 项目可用 GPU | **28 核**（PyTorch **MPS** 后端） |
| 项目可用内存 | **8 GB 统一内存（unified memory），CPU 和 GPU 共用** |
| 云端 | 不使用。只有本机跑不了时才考虑 Colab |

### 1.2 软件约定

| 组件 | 设备 | 设置 |
|---|---|---|
| 数据处理 | CPU | 只用 Polars（优先 lazy API / streaming），不用 pandas；中间结果一律存 Parquet |
| R1–R4 | CPU | 向量化计算，按需分块 |
| R5 WALS | CPU | `implicit` CPU 版，`num_threads=6`（GPU 版只支持 CUDA） |
| R6 Two-Tower | GPU（MPS） | PyTorch，`device="mps"`，全部 float32（MPS 不支持 float64） |
| Ranker | CPU | LightGBM，`num_threads=6` |
| 随机种子 | 全部 | 每个配置固定 seed，写进 config |
| Python 环境 | — | **Python 3.12**（`.python-version`），由 **uv** 管理（`.venv`）；依赖写在 `pyproject.toml`，精确版本由 **`uv.lock`** 锁定（进 git） |
| 开发工具 | — | `ruff` 格式化和 lint；**Pyright**（`standard` 模式）类型检查；`pytest` 测试。与 §15 一致 |

### 1.3 8 GB 内存的约束和对策

8 GB 是最主要的限制。**任何一个任务的内存峰值不超过约 6 GB**，给系统留余量。**重任务一次只跑一个**，不并行。

| 任务 | 估算内存 | 实测 | 对策 |
|---|---|---|---|
| 交易表常驻内存 | ~0.5 GB | **0.44 GB**（454.7 MB，31,788,324 行；Parquet 文件 160 MB） | ID 全部映射成 int32；日期存 int16（距起始日天数）；price 用 float32；channel 用 int8 |
| M0 预处理（CSV 流式转 Parquet、对齐 embedding） | — | **峰值 2.25 GB**（其中交易表一步峰值 2.10 GB）；全程约 6 秒 | Polars lazy 扫描 + streaming `sink_parquet`；字符串 ID 先换成 int32 再排序 |
| R4 ItemCF 共现矩阵（365 天） | 不分块可能远超 8 GB | 待测 | **按 seed item 分块**（每块约 2,000 个 item），每块算完只保留 Top-`K_n` 邻居就释放 |
| R5 WALS（180 天） | ~0.5–1 GB | 待测 | 稀疏 CSR，没问题 |
| R6 Two-Tower 训练 | ~1.5–2 GB | 待测 | 用户历史预先存成 int32 数组 `[N, 20]`；商品特征表（~10.5 万 × 896 维，约 380 MB）整块放进 MPS 显存；batch 直接按下标取，不用 DataLoader |
| 候选生成（每周） | 每周约 7 万 query × 最多 480 条 | 待测 | 每个 channel 单独写 Parquet，再按 query 分块合并去重 |
| Ranker 训练（4 周） | 负采样 1:40 约 900 万行，1:80 约 1,800 万行；原始 float32 特征 2–4 GB | 待测 | **流式加载（stream loading）**：每周特征按 row group 存 Parquet，用 LightGBM 的 `lgb.Sequence` 分批读入构建 `Dataset`，原始特征不必一次性放进内存；构建后的分箱数据（每个值 1 字节）约 0.5–1 GB，存成 LightGBM binary 文件复用 |
| Ranker 推理（每周，不采样） | 约 7 万 × 250 × 56 × 4 B ≈ 4 GB | 待测 | **按 query 分块打分**（每块 1 万个 query） |

以上除"实测"列外都是估算值，后续各阶段实测后继续填写"实测"列。M0 实测（2026-11-15）：常驻内存是读回 Parquet 后的 Polars `estimated_size()`；峰值是 `MemoryTracker` 的采样 RSS（间隔 0.5 秒，是下限，见下一段）。其他 M0 产物的常驻内存：customers 37 MB，articles 3 MB，customer ID 映射 89 MB，按 `article_idx` 对齐的图像 / 文本 embedding 206 MB / 155 MB。

**内存峰值怎么测**：macOS 上 psutil 拿不到进程的历史峰值，所以 `src/hmrec/memory.py` 用后台线程定时采样 RSS（间隔见 `configs/base.yaml` 的 `memory.sample_interval_seconds`），峰值取所有采样值和开始、结束读数中的最大值。比采样间隔还短的瞬时峰值可能测不到；MPS 张量占用的统一内存也不一定全部计入 RSS。因此实测峰值是**下限**，6 GB 上限要留余量：实测峰值接近 6 GB 时就应该当作超限处理。

### 1.4 Two-Tower 速度估算（M1 Max，待实测）

| 步骤 | 估算耗时 |
|---|---|
| Base 1 个 epoch（26 周，约 700–800 万样本，batch 512） | 约 2–5 分钟 |
| 每周 fine-tune（约 30 万样本） | 约 10 秒 – 1 分钟 |
| 每周推理（约 7 万 query × 约 10 万商品，取 Top-100） | 几秒到十几秒（分块矩阵乘法 + `torch.topk`） |

**开工前的 benchmark**：在 `mps` 和 `cpu` 上各跑 200 步，分别记录取数据的时间和 forward + backward 的时间（计时前后调用 `torch.mps.synchronize()`）。如果取数据占一半以上时间，先优化数据管道。

---

## 2. 数据与通用约定

### 2.1 原始数据

| 文件 | 内容 |
|---|---|
| `transactions_train.csv` | `t_dat, customer_id, article_id, price, sales_channel_id`；2018-09-20 至 2020-09-22。完全相同的重复行表示多次购买，**保留** |
| `customers.csv` | `customer_id, FN, Active, club_member_status, fashion_news_frequency, age, postal_code` |
| `articles.csv` | `article_id, product_code` 和各级类目、颜色、描述等元数据 |
| 预计算 embedding | 每个商品的 CLIP ViT-B/32 图像向量（512 维）和 all-MiniLM-L6-v2 文本向量（384 维）。只读，不重新提取。详见下面的"预计算 embedding" |

`article_id` 保留前导零，统一作为 10 位字符串处理，内部映射成 int32。

**预计算 embedding**（2026-11-15 只读检查的结果，见 `docs/decisions.md`）：

| | 图像 | 文本 |
|---|---|---|
| 模型 | CLIP ViT-B/32 | all-MiniLM-L6-v2 |
| 向量文件 | `data/processed/image_features/image_embeddings.npy` | `data/processed/text_features/text_embeddings.npy` |
| shape / dtype | `(105100, 512)` float32 | `(105542, 384)` float32 |
| id 表 | `image_article_ids.parquet`（`embedding_row, article_id, image_path`） | `text_article_ids.parquet`（`embedding_row, article_id, text, token_count, was_truncated`） |
| 覆盖 | 105,100 / 105,542 个商品，**442 个商品没有图像向量** | 全部 105,542 个商品 |

- **对应关系**：`.npy` 的第 `i` 行，对应 id 表中 `embedding_row = i` 的 `article_id`（10 位字符串）。M0 中再映射成 `article_idx`（int32）。不能假设 `.npy` 的行号等于 `article_idx`。
- **已经 L2 归一化**：两个向量的点积就是 cosine 相似度，使用时不需要再归一化。
- 目录中的 `*_faiss.index` 文件**不使用**（项目不依赖 faiss）。目录中的 manifest json 记录了模型版本和 sha256，只作参考。

### 2.2 时间约定

- Anchor `a` 是预测周的第一天。**输入**：`t_dat < a`。**Label**：`t_dat ∈ [a, a+7d)`。
- 事件年龄：`age_days = (a − t_dat).days`，所以 anchor 前一天的年龄是 1。
- 所有指数衰减都写成 `exp(−age_days / τ)`，τ 是**时间常数**，不是半衰期（半衰期 = τ × ln2）。
- **processed 数据中的日期列**：`data/processed/transactions.parquet` 不存日期，存 int16 列 **`t_dat_day`** = 距 `day_zero`（`configs/preprocess.yaml`，2018-09-20）的天数，范围 0..733。设 `anchor_day = date_to_day_index(a, day_zero)`，上面的 `t_dat < a` 在代码里写成 `t_dat_day < anchor_day`，`age_days` 写成 `anchor_day − t_dat_day`。日期与天数的换算**只能**调用 `src/hmrec/data/day_index.py`（`date_to_day_index` / `day_index_to_date`，以及对应的 Polars 表达式版本 `*_expr`），不能在别处自己写公式。

### 2.3 Query 和 Label

- **Query** = `(customer_id, anchor)`，前提是该用户在 label 周至少买了一件商品目录中的商品。每周约 7 万个 query（估计值）。
- **Label 集合 `G_q`** = 该用户在 label 周买过的**去重后**的 `article_id` 集合。
- 某一路没有给某个 query 输出结果，算这一路的漏召回（miss），**query 不能从分母里删掉**。

### 2.4 商品范围

- **商品目录 `C`**：`articles.csv` 中所有商品。Label 不按任何商品池过滤。
- **活跃池 `A(a)`**：在 `[a−28d, a)` 内至少有 3 条购买记录的商品。这是"近期在卖"的近似，不是真实库存。
- **哪些 channel 用活跃池过滤**：R1、R4、R5、R6 的输出限制在 `A(a)` 内；R2（复购）和 R3（同款不同色）不过滤，因为它们本来就基于用户自己的历史。这个设定在 M0 冻结，**不当作超参数调**。
- 用户以前买过的商品**允许再推荐**（复购在 H&M 很常见）。

### 2.5 指标

全部是"先算每个 query 的分数，再取平均"（macro）。

| 指标 | 定义 |
|---|---|
| Recall@K | `|Top-K ∩ G_q| / |G_q|` |
| AP@12 / MAP@12 | 分母为 `min(12, |G_q|)`，和 Kaggle H&M 比赛的定义一致 |
| NDCG@K | 二值相关性，理想 DCG 用完整的 `G_q` 计算 |
| Union recall | 6 路合并去重后，所有候选覆盖 `G_q` 的比例 |

输出不足 K 个时，缺的位置按"没命中"计，不补齐。

---

## 3. 日历（Fold manifest）

| 阶段 | Fold | Anchor | Label 日期（含首尾） |
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

存成 `configs/fold_manifest.yaml`，用脚本校验：每周 7 天、相邻、不重叠、最后一天是 2020-09-22。

**D1–D4 不进入 Ranker 训练**：这几周的候选是用"在这几周上选出来的配置"生成的，带有选择偏差（selection bias）。

---

## 4. 通用选择规则（所有 HPO 和 ablation）

### 4.1 4 周合成一个分数

- 主指标：**query-weighted**（把 4 周的 query 放在一起平均）。
- 同时报告 equal-fold（4 个周分数直接平均）和每周分数。

各组件的选择指标：

| 组件 | 选择指标 |
|---|---|
| R1、R4、R5、R6 | Recall@100 |
| R2 | Recall@30 |
| R3 | Recall@50 |
| Ranker | MAP@12 |

其他指标（Recall@12/20、NDCG@12/20、MAP@12）照常报告，但不用来做选择。

### 4.2 什么时候接受一个改动

只有同时满足下面两条，才把默认值换成新的变体或超参数，否则**保留默认值**：

1. 4 周合并分数**相对提升 ≥ 1%**；
2. **4 周中至少 3 周**更好。这里的"更好"指**严格更好**（新分数 > 默认分数），**分数相同不算更好**。

两个候选差不多时，选**更简单、更便宜**的那个。这条规则写进 `configs/selection_rule.yaml`，**在看到任何分数之前提交**。

### 4.3 顺序

每个组件都是先做 ablation（选变体），再在选中的变体上做 HPO。HPO 一次只动一个参数，其他参数保持当前最优值。

### 4.4 记录

每个 trial 记录：组件、变体、超参数、4 周各自分数、合并分数、query 数、运行时间、内存峰值、代码版本（git commit）。所有结果汇总到 `results/d_phase.csv` 和 `results/v_phase.csv`。

---

## 5. 六路 Recall：定义、Ablation、HPO

**每个 trial 都要在 D1–D4 每周重新计算**：用该周 anchor 之前的数据重新统计或重新 fit。不能 fit 一次然后评 4 周。

统一输出格式（每路一张表）：`query_id, article_idx, channel, rank, score`。每个 query 内 `article_idx` 不重复；按 `score DESC, article_idx ASC` 排序；`rank` 从 1 开始连续编号；不足 quota 时不补齐。

`article_idx`（int32）是把 `articles.csv` 中全部 `article_id` 按字符串排序后的编号（0..N−1，见 `src/hmrec/data/id_maps.py`），所以按 `article_idx ASC` 排序与按 `article_id ASC` 排序完全等价，tie-break 的结果不变。

### R1 — 分年龄桶的时间衰减热度（Age-Bucketed Decayed Popularity）

**定义**：取 `[a−7d, a)` 内的所有购买行。对年龄桶 `g` 和商品 `i`：

`S(g, i) = Σ exp(−age_days / τ)`，求和范围是年龄桶 `g` 的用户买 `i` 的每一条购买行。

- 年龄桶（已冻结，**不是超参数**）：

| 桶 | 规则 |
|---|---|
| `AGE_18_24` … `AGE_45_54` | 18–24、25–34、35–44、45–54 |
| `AGE_55_PLUS` | 55–99 |
| `AGE_INVALID` | 年龄 < 18 或 > 99（18 岁以下和异常值单独一个桶） |
| `AGE_UNKNOWN` | 年龄缺失 |

- 每个用户拿自己年龄桶的 Top-100；不足 100 时，用全局衰减热度按顺序补齐，跳过已选商品，并标记 `pop_is_fallback=1`。
- 默认：τ = 3 天，quota 100，限制在活跃池 `A(a)` 内。

| 类型 | 内容 | 配置数 |
|---|---|---|
| Ablation | **G0**（全局、不分桶、7 天原始计数）vs **A1**（τ=3） | 2 |
| HPO | `τ ∈ {1, 2, 3, 4, 5, 6, 7}` | +6 |

**合计 8 个配置，分钟级。**

### R2 — 时间衰减复购（Decayed-Frequency Repurchase）

**定义**：只推荐用户在 `[a−180d, a)` 内买过的同一个 `article_id`：

`S(u, i) = Σ exp(−age_days / τ)`，对用户 `u` 买 `i` 的每一条购买行求和（重复行各算一次）。

- 默认：τ = 30 天，quota 最多 30（soft，不补齐）。没有购买历史的用户输出为空。

| 类型 | 内容 | 配置数 |
|---|---|---|
| Ablation | **Recency-only**（按最近一次购买时间排序）vs **Decayed frequency**（τ=30） | 2 |
| HPO | `τ ∈ {14, 30, 60}` | +2 |

**合计 4 个配置。**

### R3 — 同款不同色（Sibling，S2）

**定义**：
- 风格偏好：`P(u, c) = Σ exp(−age_days / τ)`，对用户 `u` 在 `[a−180d, a)` 内买过的、`product_code = c` 的每一条购买行求和。
- 市场热度：`M(j)` = 商品 `j` 在 `[a−7d, a)` 内的全站购买行数。
- 分数：`S(u, j) = P(u, c(j)) × (1 + M(j))^γ`。
- 候选：与用户 180 天内买过的商品同 `product_code`，但**排除**用户 180 天内买过的同一个 `article_id`。
- 默认：τ = 30，γ = 0.5，quota 最多 50（不补齐）。

| 类型 | 内容 | 配置数 |
|---|---|---|
| Ablation | `γ ∈ {0, 0.5, 1.0}`（γ=0 表示只看风格偏好，用来检验市场热度有没有用） | 3 |
| HPO | 在最佳 γ 上调 `τ ∈ {14, 30, 60}` | +2 |

**合计 5 个配置。**

### R4 — ItemCF

**定义**（拟合窗口 `[a−365d, a)`，用户和商品的交互只记 0/1）：
- 用户权重（IUF）：`w_u = 1 / ln(1 + d_u)`，`d_u` = 用户在窗口内买过的不同商品数。
- 加权共现：`C_w(i, j) = Σ_u w_u · R_ui · R_uj`；加权边际：`n_w(i) = Σ_u w_u · R_ui`。
- 相似度（从 seed `i` 到候选 `j`）：`s(i→j) = C_w(i, j) / (n_w(i)^α · n_w(j)^(1−α))`。去掉 `i = j`。
- 每个 seed 只保留 Top-`K_n`（=100）个邻居。
- 用户分数：`S(u, j) = Σ s(i→j)`，对用户在 `[a−90d, a)` 内买过的每个不同 seed 求和。
- 默认：CF3（IUF + α=0.7），quota 100，不排除用户已买的商品。

| 类型 | 内容 | 配置数 |
|---|---|---|
| Ablation | **CF0**（不加权，`s = C / sqrt(n(i)·n(j))`）vs **CF3** | 2 |
| HPO | `α ∈ {0.5, 0.7, 0.9}` | +2 |

不同 α 可以复用同一周的 `C_w` 和 `n_w`，只需重新归一化、重新取 Top-`K_n`。**内存**：按 seed 分块计算（见 §1.3）。

**合计 4 个配置。**

### R5 — WALS（时间衰减的隐式 ALS）

**定义**（拟合窗口 `[a−180d, a)`）：
- 交互强度：`r_ui = Σ exp(−age_days / 60)`，对每条购买行求和。
- 置信度：`c_ui = 1 + α_conf · r_ui`，作为 CSR 矩阵的值传入 `implicit`，并设置库参数 `alpha=1.0`，避免重复放大。
- 默认：W2，`α_conf=20`，`factors=64`，`λ=0.1`，`iterations=15`，固定 seed。
- 推理：拟合得到的用户向量点乘商品向量，取活跃池内 Top-100。不排除已购商品。窗口内没有交互的用户输出为空。

| 类型 | 内容 | 配置数 |
|---|---|---|
| Ablation | **W0**（`r_ui = 1`，只看有没有买过）vs **W2** | 2 |
| HPO 第 1 步 | `α_conf ∈ {10, 20, 40}` | +2 |
| HPO 第 2 步 | `λ ∈ {0.01, 0.1, 1.0}` | +2 |
| HPO 第 3 步 | `factors ∈ {64, 128}` | +1 |
| 收敛检查 | 在最终配置上跑 `iterations ∈ {10, 15, 25}`，只确认 15 次时已经进入平台期 | +2 |

**Iterations 不做 early stopping**：ALS 每一轮都是闭式解（closed-form），很快收敛；过拟合由 λ 和 factors 控制。如果 25 次明显比 15 次好，就改成 25；如果 15 次之后分数明显下降，说明 λ 太小，回到第 2 步加大 λ，而不是减少 iterations。

**合计约 9 个配置 × 4 周 ≈ 36 次 ALS 拟合。**

### R6 — Two-Tower（训练方式详见 §6）

**模型**：
- **Item tower**：冻结的 CLIP（512 维）和 MiniLM（384 维）向量，经过可训练的投影层；加上可训练的 `article_id`、`product_code` 和类目 embedding；有缺失时用 mask。
  - 没有图像向量的 442 个商品（见 §2.1）：图像输入用 mask 遮掉（不填 0 向量当作真实输入），商品向量由文本、ID 和类目部分得到。这些商品**照常参与训练和召回**，不从商品目录或活跃池中删除。
- **User tower**：用户在 anchor 之前**最近 20 件购买**（不限时间窗口）经过 item tower 编码后做 pooling，再加上用户 profile（年龄桶、会员状态、news 频率）。
- 两边输出 128 维向量，L2 归一化。
- **Loss**：in-batch softmax，加 LogQ 修正（按商品被采样的频率扣分）。同一用户同一天买的多个商品，在同一个 batch 里都算正样本（multi-positive），不能互相当作负样本。
- 推理：用户向量点乘活跃池内所有商品向量，取 Top-100。
- 固定（不调）：batch 512、embedding 128、温度、LR、history 长度 20。

| 类型 | 内容 | 成本 |
|---|---|---|
| HPO 第 1 步 | **E_base**：base model 每个 epoch 存一个 checkpoint（1–6），每个 checkpoint **不做更新**直接预测 D1–D4 | 1 次 base 训练 |
| Ablation | **Frozen base**（之后不更新）vs **每周 fine-tune** | 包含在第 2 步 |
| HPO 第 2 步 | `E_update ∈ {1, 2}` × `replay ∈ {0, 50%}`，共 4 条滚动链 | 4 条链 × 4 周 |
| 稳健性 | 前 2 名的配置各再跑 2 个 seed | 约 4 条链 |
| 可选 | ① 只用 ID、不用 CLIP / MiniLM 的 content ablation；② base 长度改为 52 周 | 时间允许再做 |
| 不做 | 每周从零重训（full retrain）的对照 | — |

### 5.7 D 阶段的收尾：Union 检查

用 6 路各自的最终配置，在 D1–D4 上计算：

1. **Union recall**（目标 ≥ 24%，即 G2）；
2. **Leave-one-out union recall**：去掉每一路后，union recall 下降多少，用来判断这一路有没有带来其他路没有的候选；
3. 每个 query 的候选数分布（平均值、P50、P90）。

如果 union recall 明显低于 24%，**先改 recall，不进入 F 阶段**。是否删掉某一路，由 V 阶段的系统级 ablation（MAP@12）决定，不在这里决定。

### 5.8 D 阶段工作量汇总

| Channel | 配置数 | 成本 |
|---|---|---|
| R1 | 8 | 极低 |
| R2 | 4 | 极低 |
| R3 | 5 | 低 |
| R4 | 4 | 中（每周建图，需要分块） |
| R5 | ~9 | 中（约 36 次 ALS） |
| R6 | 1 次 base + 约 8 条链 | 高（GPU） |

---

## 6. Two-Tower 训练链（全项目只有一条）

### 6.1 训练样本

每一次购买对应一条正样本：`(用户在购买当天之前最近 20 件购买 + 用户 profile) → 买的商品`。用户历史预先算好，存成 int32 数组 `[N, 20]`。

### 6.2 第 0 步：Base model

- 训练样本：label 日期在 **2019-12-03 – 2020-06-02**（26 周）内的所有购买。
- 使用**固定 learning rate**（不做 decay），每个 epoch 存一个 checkpoint。这样第 k 个 checkpoint 就等于"训了 k 个 epoch"的模型，一次训练就覆盖 E_base = 1–6。
- **不留 validation 集，不做 early stopping。** 可以随机留出 1% 的用户不参与训练，只当报警器（检查 loss 有没有爆、有没有 bug），**不用它做任何选择**。

### 6.3 第 1 步：选 E_base

每个 checkpoint **不做任何更新**，直接预测 D1、D2、D3、D4。这几周模型都没见过，所以是合法的验证。按 §4 规则选 E_base。

### 6.4 第 2 步：选更新方式（D1–D4 滚动链）

从选中的 base 出发，每个更新配置跑一条链：

| 时间 | 操作 |
|---|---|
| 06-03 | base 预测 D1 → 记分 → 用 D1 这周的购买 fine-tune `E_update` 个 epoch（加 replay）→ M1 |
| 06-10 | M1 预测 D2 → 记分 → 用 D2 fine-tune → M2 |
| 06-17 | M2 预测 D3 → 记分 → 用 D3 fine-tune → M3 |
| 06-24 | M3 预测 D4 → 记分 → 用 D4 fine-tune → **M4** |

- **每个分数都是"先预测、后训练"得到的**（progressive validation）：被预测的那一周，模型还没见过。
- Replay：fine-tune 时额外混入的旧数据比例，从前 4 周的购买中随机抽取。
- Fine-tune 的 learning rate 固定为 base LR 的 0.1 倍，不调。
- "Frozen base"那条链就是第 1 步的结果，直接拿来对比。
- 所有链在 D1 上的分数相同（都是 base 的预测），真正区分配置的是 D2–D4。

### 6.5 第 3 步：锁定

在 2020-07-01 之前，把 `base_window, E_base, E_update, replay, fine_tune_lr, seed` 写进 `configs/locks/twotower.lock.yaml`。**选中配置跑出来的 M4 就是之后使用的模型，不重新训练 base。**

### 6.6 第 4 步：F → V → T，每周照做

| 时间 | 操作 |
|---|---|
| 07-01 | M4 预测 F1 → 输出候选 → 用 F1 fine-tune → M5 |
| 07-08 | M5 预测 F2 → 输出候选 → 用 F2 fine-tune → M6 |
| … | 每周相同，一直到 T4 |

每周的 Recall@100 照常记录。如果某周比前 4 周的平均值低 10% 以上，在报告中标出，但**不改配置**。

### 6.7 事后诊断（只报告，不改选择）

在 V 和 T 阶段，额外计算 **HP drift regret**：如果用当周的 label 重新挑选 `E_update` 和 `replay`，最多能多拿多少 Recall。差距小，说明"超参数在几个月内保持稳定"这个假设成立。

---

## 7. 候选合并：先 Fusion，再训练

### 7.1 三种做法

| 做法 | 怎么做 | 结论 |
|---|---|---|
| A. 每路单独进宽表 | 同一个商品被 3 路召回，就出现 3 行 | **不采用** |
| B. RRF 合并后截断 | 用 RRF 打分，只留前 N 个给 Ranker | **不采用** |
| C. **合并去重（union + dedup）后训练** | 每个 `(query, article)` 只有一行，每一路的 rank / score 作为这一行的列 | **采用** |

### 7.2 为什么选 C

我同意你倾向的"先 fusion"，但这里的 fusion 必须是**合并去重，不是 RRF 截断**：

- **不选 A**：同一个 `(用户, 商品)` 出现多行，label 被重复计算；被多路召回的热门商品行数更多，Ranker 会误以为它们更重要；LambdaMART 的 group 里有重复商品，排序本身就没有意义，最后输出的 Top-12 也可能出现重复。
- **不选 B**：RRF 只看排名，被它截掉的商品 Ranker 永远看不到，直接降低系统的召回上限。
- **选 C**：所有候选都保留，"被几路召回、在每一路排第几"全部变成特征，交给 LambdaMART 去学。RRF 分数也作为一列特征保留。

### 7.3 合并流程（每个 anchor 一次）

1. 读取 6 路的输出表（每路一个 Parquet 文件）。
2. 按 query 分块，把 6 路结果 pivot 成宽表：每个 `(query_id, article_id)` 一行，每路的 rank 和 score 各占一列，没被这一路召回的填 NaN。
3. 计算聚合列：`n_channels`、`best_rank`、`rrf_score`。
4. 拼接 §8 的其余特征。
5. 存成 `artifacts/features/{fold}/part-*.parquet`（按 query 分块，每块一个文件）。**生成之后冻结，之后被拿去训练时不再重新生成。**

---

## 8. Ranker 宽表特征（初版，共约 56 个）

### 8.1 原则

- 每一行 = 一个 `(query, article)` 候选。
- **所有特征只用 `t_dat < anchor` 的数据**，时间窗口都从 anchor 往前算。
- 不使用原始 ID（`customer_id`、`article_id`、`product_code`）作为特征。
- 类目字段作为 LightGBM categorical feature。
- 缺失值保留为 NaN，LightGBM 会自己处理。

### 8.2 A 组：召回来源（14 个）

| 特征 | 定义 |
|---|---|
| `pop_rank`、`rep_rank`、`sib_rank`、`icf_rank`、`wals_rank`、`tt_rank` | 各路的排名（1 = 最好）；没被这一路召回则为 NaN |
| `pop_score`、`rep_score`、`sib_score`、`icf_score` | 各路的原始分数；没被召回则为 NaN |
| `pop_is_fallback` | R1 的这个候选是否来自全局补齐 |
| `n_channels` | 有几路召回了这个商品（1–6） |
| `best_rank` | 6 路中最好的排名 |
| `rrf_score` | `Σ 1 / (60 + rank)`，对召回了它的各路求和 |

WALS 和 Two-Tower 的原始分数放在 B 组，因为 B 组对所有候选都计算，覆盖面更大。

### 8.3 B 组：对所有候选都计算的稠密分数（5 个）

这组特征不管候选来自哪一路，都会计算。通常是最有用的一组。

| 特征 | 定义 |
|---|---|
| `tt_dot` | Two-Tower 用户向量 · 商品向量（当周 anchor 的模型） |
| `wals_dot` | WALS 用户向量 · 商品向量；用户或商品没有向量时为 NaN |
| `pop_bucket_score` | 该商品在用户年龄桶里的 R1 衰减热度分数（不受 Top-100 限制） |
| `img_sim_hist` | 商品 CLIP 向量与用户最近 20 件购买的 CLIP 平均向量的 cosine 相似度。平均向量只用这 20 件中有图像向量的商品。**商品本身没有图像向量，或用户最近 20 件购买都没有图像向量时，取 NaN** |
| `txt_sim_hist` | 同上，换成 MiniLM 文本向量。文本覆盖全部商品，所以只有在**用户没有历史购买（`t_dat < a` 内没有购买）时取 NaN** |

### 8.4 C 组：商品特征（16 个）

| 特征 | 定义 |
|---|---|
| `item_cnt_1w` | `[a−7d, a)` 内的购买行数 |
| `item_cnt_prev_1w` | `[a−14d, a−7d)` 内的购买行数（和上一个特征配合，看趋势） |
| `item_cnt_4w` | `[a−28d, a)` 内的购买行数 |
| `item_uniq_buyers_4w` | `[a−28d, a)` 内的不同买家数 |
| `item_days_since_first_sale` | 距第一次被购买的天数（新品识别） |
| `item_days_since_last_sale` | 距最近一次被购买的天数 |
| `item_price_mean_4w` | `[a−28d, a)` 内的平均成交价 |
| `item_online_share_4w` | `[a−28d, a)` 内线上渠道（`sales_channel_id=2`）的占比 |
| `item_buyer_age_mean_4w` | `[a−28d, a)` 内买家的平均年龄 |
| `item_repurchase_ratio_52w` | `[a−364d, a)` 内买过它的用户中，买过 2 次及以上的比例 |
| `product_type_no` | 类目（categorical） |
| `product_group_name` | 类目（categorical） |
| `garment_group_no` | 类目（categorical） |
| `index_group_no` | 类目（categorical） |
| `section_no` | 类目（categorical） |
| `colour_group_code` | 颜色（categorical） |

### 8.5 D 组：用户特征（11 个）

| 特征 | 定义 |
|---|---|
| `age` | 年龄（缺失为 NaN） |
| `club_member_status` | categorical |
| `fashion_news_frequency` | categorical |
| `fn`、`active` | 0/1 |
| `user_cnt_4w` | `[a−28d, a)` 内的购买行数 |
| `user_cnt_52w` | `[a−364d, a)` 内的购买行数 |
| `user_days_since_last_purchase` | 距最近一次购买的天数（从没买过则为 NaN） |
| `user_price_mean_52w` | `[a−364d, a)` 内的平均成交价 |
| `user_online_share_52w` | `[a−364d, a)` 内的线上占比 |
| `user_uniq_items_52w` | `[a−364d, a)` 内买过的不同商品数 |

### 8.6 E 组：用户–商品交叉特征（10 个）

| 特征 | 定义 |
|---|---|
| `ui_cnt_item` | 用户历史上（`t_dat < a`）买过这个 `article_id` 的次数 |
| `ui_days_since_item` | 距用户最近一次买这个商品的天数（没买过为 NaN） |
| `ui_cnt_pcode` | 用户买过同 `product_code` 商品的次数 |
| `ui_days_since_pcode` | 距用户最近一次买同 `product_code` 商品的天数 |
| `ui_share_ptype_52w` | 用户 52 周内的购买中，与该商品同 `product_type_no` 的占比 |
| `ui_share_garment_52w` | 同上，按 `garment_group_no` |
| `ui_share_index_52w` | 同上，按 `index_group_no` |
| `ui_share_colour_52w` | 同上，按 `colour_group_code` |
| `ui_price_ratio` | `item_price_mean_4w / user_price_mean_52w` |
| `ui_age_gap` | `|age − item_buyer_age_mean_4w|` |

### 8.7 特征的冻结和检查

- 特征列表、类型、窗口在 M6 之前写进 `configs/locks/features.lock.yaml`，V 阶段**不调特征**。
- 每个特征都要有一个时间截断测试：把 anchor 当天及之后的交易全部删掉，重新计算，结果必须完全一致。
- 训练结束后输出 feature importance（gain），只用来分析，不用来增删特征。

---

## 9. Ranker 训练、HPO 和 Baseline

### 9.1 训练样本和负采样

- **只用至少有 1 个正样本的 query 训练**。LambdaMART 在没有正样本的 group 上梯度为零，留着只占内存。
- 保留每个 query 的**所有正样本**。
- **负样本**：在同一个 query 的候选里**均匀随机采样**，每个正样本对应 r 个负样本（候选不够时全取）。`r ∈ {40, 60, 80}` 是 HPO 参数，默认 40。经验上至少需要 40 才能让 MAP@12 达到 0.032 以上。**只用随机负采样，不用 hard negative。**
- 负采样只用于训练。**V 和 T 的评估使用全部候选，不采样。**
- 采样 seed 固定，写进 config。

估算：每周约 5 万个命中的正样本 × 41 ≈ 每周 200 万行，4 周约 900 万行。内存处理见 §1.3。

### 9.2 滚动重训：固定 4 周窗口

| 预测 | 训练用的 4 周 |
|---|---|
| V1 | F1 F2 F3 F4 |
| V2 | F2 F3 F4 V1 |
| V3 | F3 F4 V1 V2 |
| V4 | F4 V1 V2 V3 |
| T1 | V1 V2 V3 V4 |
| T2 | V2 V3 V4 T1 |
| T3 | V3 V4 T1 T2 |
| T4 | V4 T1 T2 T3 |

- 预测 V2 时（anchor 08-05），V1 的 label（07-29 – 08-04）已经完整，可以直接使用。
- 每次都从头训练，不用 `init_model` 续训。
- 窗口固定为 4 周，每次训练数据量差不多，这样选出的 tree 数才能在各周之间通用。

### 9.3 固定参数

| 参数 | 值 |
|---|---|
| objective | `lambdarank` |
| learning_rate | 0.05 |
| feature_fraction | 0.8 |
| bagging_fraction / bagging_freq | 0.8 / 1 |
| max_bin | 255 |
| eval_at | 12（只用于训练日志） |
| num_threads | 6 |

### 9.4 HPO（V1–V4）

| 步骤 | 参数 | 取值 | 训练次数 |
|---|---|---|---|
| 1 | `num_leaves` × `min_data_in_leaf` | {31, 63, 127} × {100, 500} | 6 × 4 = 24 |
| 2 | 负采样比例 r | {40, 60, 80}（训练数据流式加载，见 §1.3） | +2 × 4 = 8 |
| 贯穿 | `num_trees` | 每个模型训到 2000 棵，用 `num_iteration = 100, 200, …, 2000` 分别预测，一次训练得到整条曲线 | 不增加 |

按 §4 的规则，用 4 周的 MAP@12 选 `(num_leaves, min_data_in_leaf, 负采样比例, num_trees)`。

**不做 per-fit early stopping**：tree 数和 Two-Tower 的 epoch 一样，作为超参数在 4 周上选定后锁住。这样每次训练都能用上最新的一周数据，各周模型的训练程度也一致。

### 9.5 Baseline 和系统级 Ablation（V1–V4）

| 实验 | 必做 / 可选 | 说明 |
|---|---|---|
| **RRF baseline** | 必做 | 同一个候选集，按 `rrf_score`（k=60）排序取 Top-12；分数相同时按 `best_rank` 再按 `article_id`；不训练、不调 k |
| Full（6 路 + LambdaMART） | 必做 | 主系统 |
| −R6（去掉 Two-Tower） | 必做 | Two-Tower 成本最高，必须证明它对 MAP@12 有贡献 |
| −R1 … −R5 | 不做 | R1–R5 的价值只用 D 阶段的 leave-one-out union recall 来看 |

"去掉某一路"的做法：删掉**只由这一路召回**的候选行，删掉这一路的特征列（包括 B 组对应的稠密分数），然后按 §9.2 重新滚动训练。

### 9.6 锁定

V4 结束后、T1 之前，把 Ranker 配置写进 `configs/locks/ranker.lock.yaml`。

---

## 10. T 阶段（T1–T4）

- 6 路 recall：按锁定配置，每周用 `t_dat < anchor` 的数据重新计算或 fine-tune，然后输出候选。
- Ranker：按 §9.2 每周用最近 4 周重训，配置不变。
- 报告每周和 4 周合并的结果：**MAP@12（主指标）**、Recall@12/20、NDCG@12/20、union recall、零命中 query 比例；同时报告 RRF baseline。
- 对照 G1–G3 给出是否达标。

---

## 11. 实施顺序（Milestones）

内存只有 8 GB，**重任务一次只跑一个**。

| # | 内容 | 产出 | 完成标准 |
|---|---|---|---|
| M0 | 按 §14 建好仓库骨架、`AGENTS.md`、lint / test 配置；数据清洗与压缩（ID 转 int32 等）；`fold_manifest`；年龄桶；活跃池规则；`selection_rule.yaml`；Two-Tower 200 步 benchmark；内存实测 | 处理好的 Parquet、configs | 校验脚本通过；规则在看到任何分数之前提交 |
| M1 | 实现评估模块（Recall / MAP / NDCG / union recall）和 R1–R5 | 代码 + 单元测试 | 时间截断测试、tie-break 测试、手算小例子全部通过 |
| M2 | R1–R5 在 D1–D4 上做 ablation + HPO | `results/d_phase.csv` | 每路配置锁定 |
| M3 | Two-Tower：数据准备 → base → D1–D4 训练链 | checkpoints、M4、结果表 | `twotower.lock.yaml` |
| M4 | D 阶段 union 检查（§5.7） | union recall 表 | **G2 达标**，否则回到 M2 / M3 |
| M5 | **冻结全部 recall 配置**（打 git tag） | `channels.lock.yaml` | 之后不再修改 |
| M6 | 逐周生成 F1–T4 的候选和宽表特征（Two-Tower 沿训练链逐周推进） | `artifacts/candidates/`、`artifacts/features/` | 每周通过时间截断测试 |
| M7 | Ranker 在 V1–V4 上滚动 HPO + baseline + 系统 ablation | `results/v_phase.csv` | `ranker.lock.yaml`；**G1、G3 在 V 上达标** |
| M8 | T1–T4 最终评估 | 最终报告 | 报告 G1–G3 |

M2 中计算量小的 R1–R3 可以和 M3 的 Two-Tower 训练同时跑；R4 建图和 R5 拟合不要和 Two-Tower 同时跑。

目录结构见 §14，代码规范见 §15。
---

## 12. 已知局限

1. **超参数稳定性假设**：我们假设 6 月选出的超参数在 7–9 月仍然适用。模型参数每周更新来跟上数据变化，但超参数是锁住的。用 §6.7 的 regret 诊断来检验，结果写进报告。
2. **季节差距**：D 阶段（6 月）和 T 阶段（9 月）相隔约 3 个月。统计类 recall 的超参数（例如衰减 τ）对季节应该不太敏感；Two-Tower 依靠每周 fine-tune 跟上变化。
3. **Ranker 训练窗口只有 4 周**：Kaggle 前排方案常用 6–18 周。我们受 F 阶段长度和 8 GB 内存的限制，先用 4 周。
4. **D 周的分数偏乐观**：它们参与了 recall 的选择，只能用来比较配置，不能当作模型表现来报告。
5. **V 周的分数偏乐观**：它们参与了 Ranker 的选择。系统表现以 T 阶段为准。
6. **T 阶段不是完全没见过的数据**：旧项目曾经看过 T3、T4 附近的周。T 阶段的结果按 retrospective test 报告。
7. **活跃池不是真实库存**：只是近期销量的近似。

---


## 13. 已决定事项

| 编号 | 问题 | 决定 |
|---|---|---|
| O1 | Two-Tower 每周从零重训的对照 | **不做** |
| O2 | −R1 … −R5 的系统级 ablation（每次去掉一路，重新生成候选和特征，重训 Ranker，看 MAP@12 下降多少） | **不做**。只保留 −R6；R1–R5 用 D 阶段的 leave-one-out union recall 判断 |
| O3 | 年龄的合理范围 | 18 岁以下和异常值（> 99）单独一个桶 `AGE_INVALID`；缺失为 `AGE_UNKNOWN` |
| O4 | Hard negative | **不做**，只用随机负采样 |
| O5 | 负采样比例和内存 | HPO 试 `r ∈ {40, 60, 80}`；训练数据流式加载，不要求全部放进内存 |

---

## 14. 目录结构

```
hm-recsys/
├── AGENTS.md                    # 给 coding agent（Claude Code 等）的规则，必须遵守
├── CLAUDE.md                    # 只有一行 "@AGENTS.md"，让 Claude Code 读取 AGENTS.md
├── README.md                    # 安装、数据放置、如何运行每个 milestone
├── .python-version              # 3.12，uv 据此选择 Python
├── pyproject.toml               # 依赖、ruff / pyright / pytest 配置
├── uv.lock                      # uv 生成的精确依赖版本，进 git
├── Makefile                     # 常用命令：make lint / test / preprocess ...
├── .gitignore                   # 忽略 data/、artifacts/
│
├── docs/
│   ├── proposal.md              # 本文件，唯一的设计依据（source of truth）
│   └── decisions.md             # 决策记录：日期、决定、原因（只追加，不改历史）
│
├── configs/
│   ├── base.yaml                # 路径、全局 seed、线程数、内存上限、分块大小
│   ├── fold_manifest.yaml       # D1–T4 共 16 个 fold 的 anchor 和日期
│   ├── selection_rule.yaml      # §4 的选择规则（看分数之前提交）
│   ├── preprocess.yaml          # M0：day_zero、年龄桶边界、embedding 维数、交易表 row group 大小
│   ├── recall/                  # 每路的默认配置
│   │   ├── r1_popularity.yaml
│   │   ├── r2_repurchase.yaml
│   │   ├── r3_sibling.yaml
│   │   ├── r4_itemcf.yaml
│   │   ├── r5_wals.yaml
│   │   └── r6_twotower.yaml
│   ├── hpo/                     # D 阶段和 V 阶段的搜索网格
│   │   ├── d_phase.yaml
│   │   └── v_phase.yaml
│   ├── features.yaml            # 宽表特征列表（§8）
│   ├── ranker.yaml              # LightGBM 固定参数（§9.3）
│   └── locks/                   # 锁定后的配置，只能新增，不能修改
│       ├── channels.lock.yaml
│       ├── twotower.lock.yaml
│       ├── features.lock.yaml
│       └── ranker.lock.yaml
│
├── src/hmrec/                   # 所有可复用逻辑都在这里
│   ├── __init__.py
│   ├── config.py                # 读 YAML → 带类型的 dataclass
│   ├── paths.py                 # 所有文件路径都从这里拿，不在别处拼路径
│   ├── logging_utils.py         # 统一的 logger 设置
│   ├── memory.py                # 内存监控、分块迭代工具
│   │
│   ├── data/
│   │   ├── preprocess.py        # 原始 CSV → Parquet；ID 映射成 int32；压缩 dtype
│   │   ├── id_maps.py           # customer_id / article_id 与整数索引互转
│   │   ├── day_index.py         # 日期 ↔ t_dat_day（距 day_zero 的天数）换算的唯一实现
│   │   ├── metadata.py          # 读写 data/processed/metadata.json；scan_processed_parquet 先校验再读表
│   │   ├── folds.py             # Fold dataclass；读取并校验 fold_manifest
│   │   ├── asof.py              # 按 anchor 截取历史数据（防泄漏的唯一入口）
│   │   ├── catalog.py           # 商品目录 C、活跃池 A(a)
│   │   └── queries.py           # 每周的 query 名单和 label 集合 G_q
│   │
│   ├── recall/
│   │   ├── base.py              # RecallChannel 接口；输出表 schema 校验
│   │   ├── r1_popularity.py
│   │   ├── r2_repurchase.py
│   │   ├── r3_sibling.py
│   │   ├── r4_itemcf.py
│   │   ├── r5_wals.py
│   │   └── twotower/
│   │       ├── data.py          # 训练样本、用户历史数组 [N, 20]
│   │       ├── model.py         # ItemTower、UserTower
│   │       ├── loss.py          # multi-positive in-batch softmax + LogQ
│   │       ├── train.py         # base 训练、每周 fine-tune
│   │       ├── infer.py         # 生成用户 / 商品向量，取 Top-100
│   │       └── chain.py         # §6 的训练链：先预测、再 fine-tune
│   │
│   ├── fusion/
│   │   ├── union.py             # 6 路合并去重，pivot 成宽表的来源列
│   │   └── rrf.py               # rrf_score 计算；RRF baseline 排序
│   │
│   ├── features/
│   │   ├── base.py              # FeatureGroup 接口：输入 anchor + 候选，输出特征列
│   │   ├── source.py            # A 组：召回来源特征
│   │   ├── dense.py             # B 组：tt_dot、wals_dot 等稠密分数
│   │   ├── item.py              # C 组
│   │   ├── user.py              # D 组
│   │   ├── user_item.py         # E 组
│   │   └── build.py             # 按 query 分块拼接所有特征组，写 Parquet
│   │
│   ├── ranker/
│   │   ├── sampling.py          # query 内随机负采样
│   │   ├── dataset.py           # lgb.Sequence 流式读取，构建 LightGBM Dataset
│   │   ├── train.py             # 4 周滑动窗口训练
│   │   └── predict.py           # 分块打分，输出 Top-12 / Top-20
│   │
│   ├── eval/
│   │   ├── metrics.py           # Recall@K、AP@12、NDCG@K、union recall
│   │   └── report.py            # 汇总每周 / 4 周结果，写 results/*.csv
│   │
│   └── hpo/
│       ├── grid.py              # 读取搜索网格，生成 trial 列表（OAT 顺序）
│       ├── selection.py         # 实现 §4 的选择规则
│       └── runner.py            # 逐个 trial 运行并记录
│
├── scripts/                     # 命令行入口，只做"读参数 → 调用 src → 写结果"
│   ├── m0_preprocess.py
│   ├── m0_check_folds.py
│   ├── m0_benchmark_twotower.py
│   ├── m2_recall_hpo.py         # --channel r1 ... r5
│   ├── m3_twotower_base.py
│   ├── m3_twotower_chain.py
│   ├── m4_union_check.py
│   ├── m6_generate_candidates.py   # --fold F1 ...
│   ├── m6_build_features.py        # --fold F1 ...
│   ├── m7_ranker_hpo.py
│   └── m8_final_eval.py
│
├── tests/
│   ├── fixtures/                # 手工构造的小数据（几十行），能手算结果
│   ├── unit/                    # 每个模块的单元测试
│   └── integration/             # 用小数据把一个 fold 从头跑到尾
│
├── notebooks/                   # 只做探索和画图；不放任何被 src/ 依赖的逻辑
│
├── data/                        # 不进 git
│   ├── raw/                     # 原始 CSV：articles / customers / transactions_train / sample_submission
│   └── processed/               # M0 生成的文件；以及两个预计算 embedding 目录（只读）
│       ├── transactions.parquet           # 每行一条购买记录：customer_idx, article_idx (int32), t_dat_day (int16), price (float32), sales_channel_id (int8)；按 (t_dat_day, customer_idx) 排序，row group 100 万行
│       ├── customers.parquet              # 每行一个用户：customer_idx, age (int8), age_bucket, fn, active (int8 0/1), club_member_status, fashion_news_frequency
│       ├── articles.parquet               # 每行一个商品：article_idx, product_code 和 §8.4 的类目列
│       ├── customer_id_map.parquet        # customer_idx (int32) ↔ customer_id (64 位字符串)
│       ├── article_id_map.parquet         # article_idx (int32) ↔ article_id (10 位字符串)
│       ├── embeddings/                    # 按 article_idx 重新排列的 embedding（M0 生成）
│       │   ├── image_by_article_idx.npy   # float32 [105542, 512]，没有向量的行是 0
│       │   ├── image_mask.npy             # bool [105542]，True = 有图像向量（442 个为 False）
│       │   ├── text_by_article_idx.npy    # float32 [105542, 384]
│       │   └── text_mask.npy              # bool [105542]，全部为 True
│       ├── metadata.json                  # day_zero、年龄桶、preprocess 配置 hash、各表行数；下游读取时校验
│       ├── image_features/      # CLIP ViT-B/32，覆盖 105,100 / 105,542 个商品（缺 442 个）
│       │   ├── image_embeddings.npy           # float32 [105100, 512]，已 L2 归一化
│       │   ├── image_article_ids.parquet      # embedding_row (int64, 0..N-1) → article_id (10 位字符串)
│       │   ├── image_embedding_failures.parquet  # 空表（0 行）
│       │   ├── image_features_manifest.json   # 模型版本、sha256、校验记录
│       │   └── image_faiss.index              # FAISS IndexFlatIP，本项目不使用
│       └── text_features/       # all-MiniLM-L6-v2，覆盖全部 105,542 个商品
│           ├── text_embeddings.npy            # float32 [105542, 384]，已 L2 归一化
│           ├── text_article_ids.parquet       # embedding_row → article_id（另含 text, token_count, was_truncated）
│           ├── text_features_manifest.json
│           └── text_faiss.index               # FAISS IndexFlatIP，本项目不使用
│
├── artifacts/                   # 不进 git，全部可以由代码重新生成
│   ├── candidates/{fold}/{channel}.parquet
│   ├── features/{fold}/part-*.parquet
│   ├── lgb_datasets/{run_id}/   # LightGBM binary dataset
│   ├── models/{wals,twotower,ranker}/{fold}/
│   └── runs/{run_id}/           # 每次运行的 config 快照、日志、指标
│
└── results/                     # 进 git：小的汇总表
    ├── d_phase.csv
    ├── v_phase.csv
    └── t_phase.csv
```

### 14.1 分层规则

```
scripts/  →  src/hmrec/{recall, fusion, features, ranker, hpo}  →  src/hmrec/{data, eval}  →  config / paths
```

- 上层可以调用下层，**下层不能 import 上层**。例如 `data/` 不能 import `recall/`。
- `recall/`、`features/`、`ranker/` 之间互不 import，通过 `artifacts/` 里的文件交接。
- 所有按时间取历史数据的操作都必须经过 `data/asof.py`。

### 14.2 模块之间交接的数据

| 产出方 | 文件 | 主要列 | 使用方 |
|---|---|---|---|
| `data/queries.py` | `artifacts/queries/{fold}.parquet` | `query_id, customer_idx, anchor` | 全部 |
| `data/queries.py` | `artifacts/labels/{fold}.parquet` | `query_id, article_idx` | `eval/`、`ranker/sampling.py` |
| `recall/*` | `artifacts/candidates/{fold}/{channel}.parquet` | `query_id, article_idx, channel, rank, score` | `fusion/union.py` |
| `fusion/union.py` + `features/build.py` | `artifacts/features/{fold}/part-*.parquet` | `query_id, article_idx`，§8 的全部特征列 | `ranker/` |
| `ranker/predict.py` | `artifacts/predictions/{run_id}/{fold}.parquet` | `query_id, article_idx, score, rank` | `eval/` |

Label 只在 `eval/` 和 `ranker/sampling.py` 中读取，**recall 和特征代码永远不读 label**。

---

## 15. 代码规范

读者是 junior MLE，所以**可读性优先于技巧**。完整规则写在 `AGENTS.md`，这里列出要点：

1. **语言和工具**：Python 3.12，`uv` 管理环境和锁定依赖（`uv.lock`）；Polars 做表格处理，NumPy / SciPy 做矩阵，PyTorch（MPS）做 Two-Tower，LightGBM 做 Ranker；`ruff` 负责格式化和 lint，`pyright`（`standard` 模式）做类型检查，`pytest` 做测试。与 §1.2 一致。
2. **每个函数只做一件事**，一般不超过 50 行；每个文件对应一个清楚的组件。
3. **全部加类型标注**；公开函数都要有 docstring，写清楚输入、输出、**表的粒度（一行代表什么）**和**时间窗口**。
4. **注释写"为什么"**，而不只是"做了什么"。尤其要说明：时间窗口是开区间还是闭区间、为什么用这个 dtype、为什么要分块。
5. **数字都放进 config**，代码里不出现没有解释的常数（magic number）。
6. **计算和读写分开**：计算函数接收 DataFrame / 数组，返回 DataFrame / 数组；读写文件只在 `scripts/` 和少数 IO 函数里做。这样计算函数可以直接用小数据测试。
7. **防泄漏写成断言**：每个 as-of 函数都检查输入数据的最大日期小于 anchor，不满足就报错。
8. **用 logger，不用 print**；长任务要打印进度、耗时和内存峰值。
9. **每个模块都有单元测试**，测试数据小到能手算。
10. 代码注释和 docstring 用英文，用词尽量简单；设计文档（`docs/`）用中文。
