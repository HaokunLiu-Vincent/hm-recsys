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
| G2 | 6 路合并后的 macro **union recall** | **≥ 24%**（D1–D4 实测 **25.24%**，M5 锁定的配置；见下表说明） | D1–D4（锁定 recall 前）和 F/V/T 每周 |
| G3 | LambdaMART 优于 RRF baseline | 4 周中至少 3 周 MAP@12 更高 | V1–V4、T1–T4 |

各路 Recall@quota 的参考值（来自旧项目，**存疑，只作为 sanity check，不作为验收标准**）：

| Channel | Quota | 参考 Recall@quota | D 阶段实测（M5 锁定的配置，D1–D4 四周合并） |
|---|---|---|---|
| R1 Popularity | 100 | ~7% | **7.39%** |
| R2 Repurchase | ≤30 | ~4% | **4.27%** |
| R3 Sibling | ≤50 | ~3.6% | **3.24%**（只输出 `A(a)`；切换前 3.26%） |
| R4 ItemCF | 100 | ~8% | **9.33%** |
| R5 WALS | 100 | ~10% | **10.81%** |
| R6 Two-Tower | 100 | ~10% | **14.63%**（训练链，每周先预测后 fine-tune；不更新的 base 为 11.84%） |

实测列来自 `configs/locks/` 中每一路记录的 trial（`results/d_phase.csv`，R3 为 `results/m4_trials.csv`）。**D 阶段的分数偏乐观**（配置就是在这 4 周上选出来的，见 §12 第 4 条），只用来和参考值对照，不代表 F / V / T 周的表现。没有哪一路比参考值低一半以上。

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
| 项目可用内存 | **12 GB 统一内存（unified memory），CPU 和 GPU 共用**（2026-11-18 之前是 8 GB，见 §1.3）。这是项目自己定的预算；本机的物理内存是 32 GB |
| 云端 | 不使用。只有本机跑不了时才考虑 Colab |

### 1.2 软件约定

| 组件 | 设备 | 设置 |
|---|---|---|
| 数据处理 | CPU | 只用 Polars（优先 lazy API / streaming），不用 pandas；中间结果一律存 Parquet |
| R1–R4 | CPU | 向量化计算，按需分块 |
| R5 WALS | CPU | `implicit` CPU 版，`num_threads=6`（GPU 版只支持 CUDA） |
| R6 Two-Tower | **CPU** | PyTorch，`device="cpu"`（`configs/recall/r6_twotower.yaml` 的 `training.device`），`torch.set_num_threads(6)`，全部 float32。原计划用 MPS；真实模型的 200 步 benchmark 中 CPU 每步 30.4 ms、MPS 45.2 ms，不满足"MPS 快 1.3 倍以上才用 MPS"的规则，而且 CPU 结果可以逐位复现（`docs/decisions.md` 2026-11-17） |
| Ranker | CPU | LightGBM，`num_threads=6` |
| 随机种子 | 全部 | 每个配置固定 seed，写进 config |
| Python 环境 | — | **Python 3.12**（`.python-version`），由 **uv** 管理（`.venv`）；依赖写在 `pyproject.toml`，精确版本由 **`uv.lock`** 锁定（进 git） |
| 开发工具 | — | `ruff` 格式化和 lint；**Pyright**（`standard` 模式）类型检查；`pytest` 测试。与 §15 一致 |

### 1.3 内存预算（12 GB）的约束和对策

内存是最主要的限制。预算是项目自己定的（§1.1；本机物理内存 32 GB），现在是 12 GB。**单个进程的真实峰值上限为 12 GB**（`configs/base.yaml` 的 `memory.limit_gb`），等于预算。**重任务一次只跑一个**，不并行。

上限只是警戒线：超过时进程在写完运行记录之后报错，让人来决定；它不参与任何计算，不影响结果和速度。**分块设置（`configs/base.yaml` 的 `chunking`、`configs/ranker.yaml` 的 `scoring_query_chunk_size` 等）不随上限放宽**，每个进程的真实峰值照常记录。到 2026-11-18 为止实测的最高真实峰值是 V2 特征生成的 7.25 GB。

上限的变化（都在 2026-11-18，见 `docs/decisions.md`）：最初是"约 6 GB"；F1 特征生成的真实峰值是 6.10 GB，用户决定不改代码，把上限调到 7 GB（"留 1 GB 余量"）；随后 V1 的真实峰值是 7.002 GB，而本机物理内存是 32 GB，"留余量防止交换"的理由不成立，上限改为等于当时预算的 8 GB；M7 第 1 步之后、第 2 步之前，预算和上限一起改为 12 GB。下表、表下说明和后面各节中写于此前的"6 GB 上限""6 GB 以内""8 GB"指当时的上限，作为记录保留，没有改写。

| 任务 | 估算内存 | 实测 | 对策 |
|---|---|---|---|
| 交易表常驻内存 | ~0.5 GB | **0.44 GB**（454.7 MB，31,788,324 行；Parquet 文件 160 MB） | ID 全部映射成 int32；日期存 int16（距起始日天数）；price 用 float32；channel 用 int8 |
| M0 预处理（CSV 流式转 Parquet、对齐 embedding） | — | **峰值 2.25 GB**（其中交易表一步峰值 2.10 GB）；全程约 6 秒 | Polars lazy 扫描 + streaming `sink_parquet`；字符串 ID 先换成 int32 再排序 |
| M0 query / label 生成（16 个 fold） | — | **峰值 0.36 GB**（MemoryTracker 采样；`/usr/bin/time` 最大 RSS 0.38 GB）；16 个 fold 共约 1.9 秒（每个 fold 约 0.1 秒） | 每个 fold 只读 label 周的两列（`customer_idx, article_idx`）；每周约 6.9 万–12.0 万个 query，约 21 万–49 万条 label |
| R1–R3 候选生成（D1，默认配置） | — | **整体峰值 1.77 GB**（`/usr/bin/time` 最大 RSS，整个脚本：读 queries / labels、算活跃池、跑 3 路、评估）；耗时 R1 0.6 秒、R2 0.2 秒、R3 0.5 秒，全程 2.2 秒。单个 channel 的运行时间短于 0.5 秒的采样间隔，**单路峰值不可靠**，不单独列出 | 每路只读需要的列；R2 / R3 的历史先按 query 用户 semi-join |
| R4 ItemCF 共现矩阵（365 天） | 不分块可能远超 8 GB | **D1（CF3，α=0.7）：整体峰值 2.95 GB**（`/usr/bin/time` 最大 RSS，含读数据；MemoryTracker 采样 2.62 GB）；耗时 17.2 秒（建图 + 邻居 15.5 秒，打分 1.8 秒）。图规模：1,540 万条购买行 → 1,325 万个 (用户, 商品) 对，98.8 万用户，7.2 万商品，`Σ d_u²` ≈ 5.1 亿；28,110 个 seed，分 15 块。**D 阶段（D1–D4 四周，9 个配置）：最重的 trial 峰值 4.76 GB**（MemoryTracker 采样，整个进程，含已加载的 4 周数据；配置 seed 窗口 365 天、衰减关，4 周共 115 秒）；seed 窗口越长，要算邻居的 seed 越多，峰值越高（其余 trial 3.9–4.6 GB） | **按 seed item 分块**，每块算完只保留 Top-`K_n` 邻居就释放。D 阶段每块 2,000 个 item；4.76 GB 离 6 GB 上限太近，所以 **M5 把 `chunking.item_chunk_size` 从 2000 改为 1000**（分块只影响内存和速度，不影响结果：分块和不分块的 `C_w` 逐位相同，有测试）。**新分块（1000）实测**（M5 锁定的配置：seed 窗口 365 天、衰减开；D1）：generate 25.8 秒，真实峰值 generate 结束时 3.72 GB、整个进程 4.20 GB；候选文件与分块 2000 时生成的文件逐字节相同（见表下"M5 锁定配置在 D1 上的单路实测"） |
| R5 WALS（180 天） | ~0.5–1 GB | **D1（W2，默认配置）：整体峰值 3.10 GB**（`/usr/bin/time` 最大 RSS；MemoryTracker 采样：拟合 0.96 GB，打分 2.11 GB）；耗时 63.8 秒（拟合 53.0 秒，打分 10.8 秒）。矩阵 71.1 万用户 × 5.2 万商品，590 万个非零值。**峰值来自打分**而不是拟合：打分按 `chunking.dense_score_query_chunk_size` = 5,000 个 query 分块，每块 5,000 × 20,764 个活跃商品的 float32 分数矩阵约 0.41 GB，再加 `np.partition` 的副本（分块为 1 万时整体峰值 4.05 GB；两种分块的 Recall@100 都是 11.36%，候选行数相同）。当前 `implicit` 的 macOS wheel 没有链接 OpenMP，拟合实际是**单线程**（`num_threads=6` 不起作用） | 稀疏 CSR，没问题 |
| R6 Two-Tower 训练 | ~1.5–2 GB | **训练数组生成（base 窗口，2026-11-17）：峰值 4.70 GB**（取较保守的值：两次运行 MemoryTracker 采样分别为 3.35 GB 和 4.70 GB，任务只有约 4.5 秒，0.5 秒的采样间隔误差大；`/usr/bin/time` 最大 RSS 4.15 GB，含读数据），耗时 4.6 秒；616 万个样本，数组共 670 MB（`history` int32 470 MB、`history_mask` 118 MB）。**200 步 benchmark（占位模型，batch 512）**：MPS 进程最大 RSS 2.00 GB，MPS 张量占用 1.17 GB（驱动共持有 2.02 GB，统一内存，与 RSS 部分重叠）；CPU 进程最大 RSS 2.34 GB。真实模型（训练全程）待测 | 用户历史预先存成 int32 数组 `[N, 20]`；商品特征表（~10.5 万 × 896 维，约 380 MB）整块放进 MPS 显存；batch 直接按下标取，不用 DataLoader |
| 候选生成（每周） | 合并前最多约 3,940 万行，全部放在内存约 0.73 GB（D3 约 1.07 GB）；单个 channel 最多约 0.15 GB；合并去重后约 **2,520 万行**（每个 query 平均 307 个；计算见表下 ①） | **M4 实测（D1–D4，2026-11-17，M5 锁定的配置，即 R3 只输出 `A(a)`）**：合并前 D1 3,219 万行、D3 4,947 万行（只保留 `query_id`、`article_idx`、`rank` 三列时约 0.36 / 0.55 GB）；去重后每个 query 平均 **307** 个候选（P50 320，P90 364；上限 480），D1 约 2,440 万行、D3 约 3,661 万行。R3 切换之前：平均 313（P50 327，P90 373），合并前 D1 3,268 万行、D3 5,015 万行。**内存**：第一次 M4 union 检查（四周一起处理）整个进程的真实峰值 6.66 GB（`/usr/bin/time`；0.5 秒采样只看到 5.22 GB），超过 6 GB；改为一次只处理一个 fold 后（R3 切换后的重跑）降到 **4.78 GB**（来源：`/usr/bin/time -l` 的终端输出，**没有写入运行日志**；日志里采样到的最大值是 4.72 GB）。**这个 4.78 GB 不包含 R6 现场预测**：那次运行 R1、R2、R4、R5、R6 的候选都从已有文件读回，只有 R3 是现场生成的。**单路的真实峰值**（每路一个进程，含 R6 现场预测）已在 M5 的 D1 核对中测得，见表下"M5 锁定配置在 D1 上的单路实测"：最高是 R4 的 4.20 GB。**M6 实测（F1–T4 共 12 周，2026-11-17，`results/m6_runs.csv`）**：每个 (fold, channel) 一个进程，所以不存在"六路在一个进程里"的峰值；12 周中任何进程的真实峰值最高 **4.70 GB**（F2 的 R6，含预测、全部商品向量和这一周的 fine-tune），各路范围：R1 1.9–2.1、R2 0.9–1.1、R3 1.0–1.1、R4 3.8–4.3、R5 3.6–4.2、R6 4.3–4.7 GB。监控进程（读回六路候选、统计去重后的候选数；F、V 周另算 recall）最高 2.12 GB。每周去重后的候选数平均 304–310（P50 317–324，P90 359–364），与 D 阶段的 307 一致。合并成宽表（`fusion/union.py`）的峰值留到 M6 后半 | 每个 channel 单独写 Parquet，再按 query 分块合并去重 |
| 特征宽表（每周，§8） | 一周全放进内存约 **6.9 GB**（2,520 万行 × 292 B），不可行；按 1 万 query 分块后每块约 **0.83 GB**，P90 的块约 0.99 GB（计算见表下 ④）。12 周（F、V、T）的特征 Parquet 未压缩约 **77 GB**，压缩后估计 23–46 GB（计算见表下 ⑤）。以上按 71 个特征计（2026-11-17 从 56 个增加到 71 个） | **F1 实测（2026-11-18，`results/m6_runs.csv`）**：85,648 个 query，26,373,844 行，9 个 part 文件；耗时 61 秒（每 fold 的三类表 3.5 秒，`tt_sim_*` 一步 6.0 秒）。**真实峰值 6.10 GB，超过 6 GB 上限 0.10 GB**（采样 5.76 GB；与 `/usr/bin/time -l` 一致）。每个分块结束时的 RSS 从 3.80 GB 逐块升到 4.95 GB，即块与块之间内存没有完全释放；整周的六路候选和每 fold 的三类表在整个进程中常驻内存，各占多少没有分开测。文件已完整写出并通过全部检查；用户决定代码保持不变（`docs/decisions.md` 2026-11-18）。磁盘 **1.67 GB**（每个 part 约 209 MB），见表下 ⑤。**12 周实测（F1–T4，2026-11-18，`results/m6_runs.csv`）**：共 281,750,956 行（每周 2,098 万–2,637 万），每周 7–9 个 part 文件；每周耗时 48–61 秒，12 周合计约 11 分钟；真实峰值 **6.10–7.25 GB**（最高是 V2 的 7.25 GB），与行数无关（行数最多的 F1 峰值最低），在 8 GB 上限之内；磁盘合计 **17.8 GB**。特征生成的代码自 F1 起没有改动 | **按 query 分块计算**（每块 1 万个 query），每块算完直接写成一个 Parquet row group / part 文件，不在内存中保留整周 |
| Ranker 训练（4 周） | 负采样 1:40 约 1,100 万行，1:80 约 2,170 万行（上限）；原始 float32 特征约 2.9–5.7 GB，分箱后约 0.7–1.4 GB（计算见表下 ②；71 个特征） | **行数实测（2026-11-18，只读统计，见 §9.1）**：4 周窗口 r = 40 为 1,045 万–1,071 万行，r = 60 为 1,530 万–1,569 万行，r = 80 为 1,976 万–2,028 万行，每个窗口约 15 万个 query。**测速实测（2026-11-18，V1 的窗口 F1–F4，r = 40，1,071 万行，154,815 个 query；`results/m7_speed_test.csv`、`results/m7_runs.csv`）**：采样每个 fold 21–24 秒（一次写出 r = 40 / 60 / 80），真实峰值 3.92–4.13 GB，4 个 fold 的采样文件共 4.0 GB（r = 40 每行约 91 B）；构建 Dataset 6.6 秒，真实峰值 2.00 GB，binary 644 MB；训练 50 棵（不评估）col-wise 0.682 秒/棵、峰值 1.47 GB，row-wise 0.368 秒/棵、峰值 1.99 GB；row-wise 训练 300 棵共 103.5 秒（0.342 秒/棵，每 100 棵依次 37.5、34.8、31.2 秒） | **流式加载（stream loading）**：每周特征按 row group 存 Parquet，用 LightGBM 的 `lgb.Sequence` 分批读入构建 `Dataset`，原始特征不必一次性放进内存；构建后的分箱数据（每个值 1 字节）约 0.5–1 GB，存成 LightGBM binary 文件复用 |
| Ranker 推理（每周，不采样） | 一次全放进内存约 **6.7 GB**；按 1 万 query 分块后每块约 **0.81 GB**，P90 的块约 0.96 GB（计算见表下 ③；71 个特征） | **测速实测（2026-11-18，V1，2,523 万行 × 300 棵、31 叶，一个 part 一次打分）**：打分加评估 86.4 秒，每个 part（约 305 万行）约 10 秒；整个 trial（训练加打分）的真实峰值 **5.69 GB**，出在打分阶段（训练本身约 2 GB）。之后打分改为每 2,500 个 query 一块（见右栏），新的峰值待第 1 步网格实测 | **按 query 分块打分**：一个 part 文件（1 万个 query）读入后，再按 2,500 个 query 一块打分（`configs/ranker.yaml` 的 `scoring_query_chunk_size`；2026-11-18 测速后从"一个 part 一块"改小）。只影响内存：一行的分数与同一块里的其他行无关，query 不会被切开，测试确认分数和 Top-12 逐位相同 |

**候选生成、Ranker 训练、Ranker 推理、特征宽表几行的估算过程**（2026-11-16 按实测 query 数重算；2026-11-17 按 M4 实测的候选数重算，同日 M5 锁定时再按 R3 切换后的候选数 307 重算；1 GB = 1024³ B）。共同输入：每周平均 **8.2 万** query（D3 最多 12.0 万；F、V、T 周 6.9–8.6 万），每个 query 平均约 **3.4** 个 label（去重后；F、V 周 3.2–3.6）；6 路 quota 合计 100 + 30 + 50 + 100 + 100 + 100 = **480**；合并去重后每个 query 平均 **307** 个候选，P50 320，P90 364（M4 在 D1–D4 上实测，M5 锁定的配置，即 R3 只输出 `A(a)`；R3 切换之前为 313 / 327 / 373，最初假设 250）；特征 **71** 个（最初 56 个，2026-11-17 增加 15 个，见 §8）。F、V、T 共 12 周的 query 数实测为 6.9–8.6 万，合计 **917,354**，平均 7.64 万。

- ① **候选生成**：合并前最多 8.2 万 × 480 ≈ 3,940 万行（D3：12.0 万 × 480 ≈ 5,760 万行）。每行约 20 B（`query_id`、`article_idx`、`rank` 各 int32 4 B，`score` float32 4 B，`channel` 用 Categorical 约 4 B）→ 3,940 万 × 20 B ≈ 0.73 GB（D3 ≈ 1.07 GB）。每个 channel 单独写盘，单个 channel 最多 8.2 万 × 100 × 20 B ≈ 0.15 GB。合并去重后 8.2 万 × 307 ≈ **2,520 万行**（R3 切换之前 2,570 万，最初估算 2,050 万；D3：12.0 万 × 305 ≈ 3,660 万行）。
- ② **Ranker 训练**：每周正样本 ≈ 8.2 万 query × 3.4 label × 24%（G2 的 union recall 目标；M4 在 D1–D4 上实测 25.2%，正样本和下面的行数相应多约 5%）≈ 6.7 万。每个正样本配 r 个负样本：r = 40 时 6.7 万 × 41 ≈ 274 万行/周，4 周 ≈ **1,100 万行**；r = 80 时 6.7 万 × 81 ≈ 542 万行/周，4 周 ≈ **2,170 万行**（候选不够时负样本全取，所以这是上限）。原始特征 71 × 4 B = 284 B/行 → 1,100 万 × 284 B ≈ 2.9 GB，2,170 万 × 284 B ≈ 5.7 GB；分箱后每个值 1 B → 1,100 万 × 71 B ≈ 0.73 GB，2,170 万 × 71 B ≈ 1.4 GB。r = 80 时原始特征已经贴近 6 GB 上限，所以必须流式加载。（按 56 个特征时是 2.3 / 4.5 GB 和 0.57 / 1.1 GB。）
- ③ **Ranker 推理**：每周 8.2 万 × 307 = 2,520 万行 × 284 B ≈ 7.16 × 10⁹ B ≈ **6.7 GB**（56 个特征时 5.3 GB，最初估算 4.3 GB；V、T 周最多 8.3 万 query，差不多）。按 1 万 query 分块：1 万 × 307 × 284 B ≈ 8.7 × 10⁸ B ≈ **0.81 GB/块**（56 个特征时 0.64 GB）；候选多的块按 P90 算：1 万 × 364 × 284 B ≈ 0.96 GB/块。
- ④ **特征宽表**：每行 71 个特征按 float32 计 284 B，加上 `query_id`、`article_idx` 两个 int32 共 8 B → **292 B/行**（实际存储略小：8 个类别特征是 int16，每行 276 B）。一周 2,520 万行 × 292 B ≈ 7.36 × 10⁹ B ≈ 6.9 GB，超过 6 GB 上限，所以不能整周放进内存。按 1 万 query 分块：1 万 × 307 × 292 B ≈ 9.0 × 10⁸ B ≈ **0.83 GB/块**；按 P90：1 万 × 364 × 292 B ≈ 0.99 GB/块。（56 个特征时是 232 B/行、5.4 GB、0.66 / 0.79 GB。）
- ⑤ **12 周特征 Parquet 的磁盘占用**：F1–F4、V1–V4、T1–T4 共 917,354 个 query × 307 ≈ **2.82 亿行**；未压缩 2.82 亿 × 292 B ≈ 8.23 × 10¹⁰ B ≈ **77 GB**（平均每周约 6.4 GB；56 个特征时 61 GB）。Parquet 压缩比现在只能估计：候选文件（ID、rank、score）压缩后约为原始大小的 18%（预检查 5 路 4 周 1.25 亿行 449 MB），浮点特征压缩得差一些，按 30%–60% 估计为 **约 23–46 GB**。M6 生成 F1 后先实测一周的大小再外推。当前磁盘可用空间约 520 GB，足够。**F1 实测（2026-11-18）**：26,373,844 行，Parquet 共 1.67 GB，即每行约 68 B，约为未压缩大小（26,373,844 × 276 B ≈ 6.8 GB）的 23%–25%，比上面 30%–60% 的估计压缩得更好。按 2.82 亿行外推，12 周约 **18 GB**。**12 周实测**：281,750,956 行，共 19,136,477,799 字节 ≈ **17.8 GB**，每行约 68 B，与外推一致。
- 说明：307 是 R3 改为只输出 `A(a)` 之后重跑 union 检查的实测值（`results/m4_union_check_r3_active_pool.csv`，4 周合并 307.1）。切换之前是 313：R3 在 `A(a)` 外的候选约占 R3 行数的 23%（平均每个 query 约 7 行），切换后其中一部分名额由池内的同款商品补上，R3 平均每个 query 的行数从 29.8 降到 24.1，合并去重后平均少 5.8 个。

以上除"实测"列外都是估算值，后续各阶段实测后继续填写"实测"列。M0 query / label 生成实测于 2026-11-16（`scripts/m0_build_queries.py`）。M0 预处理实测（2026-11-15）：常驻内存是读回 Parquet 后的 Polars `estimated_size()`；峰值是 `MemoryTracker` 的采样 RSS（间隔 0.5 秒，是下限，见下一段）。其他 M0 产物的常驻内存：customers 37 MB，articles 3 MB，customer ID 映射 89 MB，按 `article_idx` 对齐的图像 / 文本 embedding 206 MB / 155 MB。

**M5 锁定配置在 D1 上的单路实测**（2026-11-17，`scripts/m5_verify_locks.py`，HEAD `0ce824a`，工作区干净；每路一个进程；`results/m5_lock_check.csv`）。D1 有 78,193 个 query，活跃池 20,764 个商品。六路的分数都与记录逐位相同，候选文件与 D 阶段 / M4 生成的文件逐字节相同。

| Channel | 候选行数 | generate 耗时 | 整个进程耗时 | 真实峰值：到 generate 结束 | 真实峰值：整个进程 | 采样峰值（generate） |
|---|---|---|---|---|---|---|
| R1 Popularity | 7,819,300 | 0.5 秒 | 3.8 秒 | 1.98 GB | **1.98 GB** | 1.05 GB |
| R2 Repurchase | 1,292,785 | 0.3 秒 | 3.1 秒 | 0.99 GB | **0.99 GB** | 0.77 GB |
| R3 Sibling | 1,860,095 | 0.4 秒 | 3.2 秒 | 1.06 GB | **1.06 GB** | 0.76 GB |
| R4 ItemCF（`item_chunk_size` 1000） | 7,001,322 | 25.8 秒 | 29.1 秒 | 3.72 GB | **4.20 GB** | 3.72 GB |
| R5 WALS（factors 128） | 6,398,300 | 169.6 秒 | 173.1 秒 | 2.72 GB | **3.78 GB** | 2.61 GB |
| R6 Two-Tower（base epoch 4，现场预测，不训练） | 7,819,300 | 13.6 秒 | 17.4 秒 | 2.33 GB | **3.53 GB** | 2.33 GB |

- "真实峰值"是 `ru_maxrss`（`process_max_rss_gb`）。"到 generate 结束"是从进程启动（含读 queries / labels、算活跃池、R6 加载模型）到 `generate` 返回时的峰值；"整个进程"还包括之后的校验、写 Parquet 和评估。六路的"整个进程"一栏与 `/usr/bin/time -l` 的 maximum resident set size **逐字节相等**。
- 六路都在 6 GB 以内，最高是 R4 的 4.20 GB。
- R4、R5、R6 在 `generate` 返回**之后**峰值又升高了 0.48 / 1.06 / 1.20 GB，来自候选的校验、写盘和评估（这次没有把三步分开测）。M6 每一路同样要校验和写盘，所以估算 M6 的单路峰值时应看"整个进程"一栏，而不是"到 generate 结束"。
- 采样峰值会漏掉短任务的尖峰：R1 采样到 1.05 GB，真实值是 1.98 GB。
- R4 没有同一配置、同一 fold、分块 2000 的真实峰值可以对比（此前 D1 的 2.95 GB 是默认配置，D 阶段的 3.9–4.8 GB 是四周在一个进程里的采样值），所以这里不给出"分块减半省了多少"的数字。

**内存峰值怎么测**：`src/hmrec/memory.py` 记录两种峰值，字段名不同，回答的问题也不同：

- **采样峰值**（`MemoryTracker`，`peak_rss` / `sampled_peak_rss_gb`）：后台线程用 psutil 定时采样 RSS（间隔见 `configs/base.yaml` 的 `memory.sample_interval_seconds`），取所有采样值和开始、结束读数中的最大值。可以**分阶段**测（每个阶段一个 tracker），用来定位内存用在哪里；但比采样间隔还短的瞬时峰值测不到，所以只是**下限**。
- **真实峰值**（`read_process_peak_memory`，`process_max_rss_gb`）：标准库 `resource.getrusage(RUSAGE_SELF).ru_maxrss`，操作系统记录的、进程启动以来的最大 RSS，不会漏掉瞬时峰值，与 `/usr/bin/time -l` 读的是同一个计数器。但它**只有整个进程的一个数**，不能清零，所以说不出峰值出在哪个阶段。单位随系统不同（macOS 是字节，Linux 是 KB），由 `ru_maxrss_to_bytes` 换算。子进程不计入本进程的 RSS，另外记录 `RUSAGE_CHILDREN`（已结束的子进程中最大的一个，不是总和）。

是否超过上限（12 GB；2026-11-18 之前是 6 GB，当天先后改为 7 GB、8 GB、12 GB）**以真实峰值为准**；采样峰值用来找出是哪个阶段造成的。每个 `MemoryTracker` 结束时两种数字都写进日志（真实峰值在相邻两个阶段的日志之间变大，说明尖峰出在后一个阶段）；脚本在最后一步调用 `log_process_peak_memory`，并把真实峰值写进这次运行的结果记录（M5 的锁定核对写到 `results/m5_lock_check.csv`）。**2026-11-17（M5）之前的数字**：上表中标为"采样"或 MemoryTracker 的都是采样峰值（下限）；标为 `/usr/bin/time` 的是真实峰值，但只出现在终端输出里，没有写入运行日志。MPS 张量占用的统一内存不一定全部计入 RSS，两种峰值都受此影响。

### 1.4 Two-Tower 速度估算（M1 Max）

| 步骤 | 估算耗时 | 实测 |
|---|---|---|
| Base 1 个 epoch（26 周，约 700–800 万样本，batch 512） | 约 2–5 分钟 | 占位模型：MPS 约 9.2 分钟，CPU 约 7.5 分钟。**真实模型（2026-11-17，200 步）：CPU 每步 30.4 ms（P90 31.1）→ 约 6.1 分钟 / epoch，6 个 epoch 约 37 分钟；MPS 每步 45.2 ms（P90 54.4）→ 约 9.1 分钟 / epoch**（616 万个样本 = 12,033 步；去掉 1% 报警用户后实际约 1.19 万步）。选用 CPU |
| 每周 fine-tune（约 30 万样本） | 约 10 秒 – 1 分钟 | **M6 实测（F1–T3 共 11 周，CPU，E_update 2，replay 50%）：52–71 秒**（每周新样本 23–31 万，加同样多的 replay，共 46–61 万行，1,792–2,396 步）。R6 整个进程（核对 sha256、加载模型、预测、保存向量、fine-tune）每周 74–95 秒，其中预测并保存向量 15–19 秒；T4 只预测，18 秒 |
| 每周推理（平均约 8.2 万 query，最多 12.0 万（D3）× 约 10 万商品，取 Top-100） | 几秒到十几秒（分块矩阵乘法 + `torch.topk`） | **D1（2026-11-17，M5 锁定核对，CPU）：13.6 秒**（78,193 个 query × 20,764 个活跃商品，含用户历史、两塔编码和取 Top-100）；另加核对 sha256、加载模型和表 0.4 秒。真实峰值 generate 结束时 2.33 GB |

**开工前的 benchmark**：在 `mps` 和 `cpu` 上各跑 200 步，分别记录取数据的时间和 forward + backward 的时间（计时前后调用 `torch.mps.synchronize()`）。如果取数据占一半以上时间，先优化数据管道。

**Benchmark 实测（2026-11-17，`scripts/m0_benchmark_twotower.py`，torch 2.15.0）**：用占位模型（`src/hmrec/recall/twotower/benchmark.py`：图像 / 文本投影 + ID 和 7 个类目 embedding，用户塔对 20 件历史做 masked mean 加 profile embedding，multi-positive in-batch softmax，没有 LogQ；不是 R6 的正式模型），base 窗口的数组，batch 512，200 步，每个设备单独一个进程。

| 设备 | 取 batch（每步，去掉前 10 步） | forward + backward + 更新（每步） | 取 batch 占比 | 前 10 步合计 | 200 步合计 |
|---|---|---|---|---|---|
| MPS | 0.80 ms | 45.0 ms（中位数 44.5，P90 47.8） | **1.7%** | 1.84 秒 | 10.5 秒 |
| CPU（6 线程） | 0.26 ms | 37.1 ms（中位数 37.1，P90 37.8） | **0.7%** | 0.43 秒 | 7.5 秒 |

- 取 batch 远低于一半，**数据管道不需要优化**。
- 两个设备的 loss 逐步一致（7.860 → 6.388），说明计算相同。
- 占位模型在 **MPS 上比 CPU 慢约 20%**。占位模型每步的矩阵很小，MPS 的 kernel 启动和 embedding 反向（scatter）开销占主要部分；真实模型更大时结论可能不同。正式训练前用真实模型重新比较了两个设备：**CPU 仍然更快**（真实模型每步 30.4 ms vs MPS 45.2 ms，取 batch 占比 0.9% / 1.2%），R6 用 CPU 训练（§1.2）。

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

- **Query** = `(customer_id, anchor)`，前提是该用户在 label 周至少买了一件商品目录中的商品。每周 **6.9–12.0 万**个 query，平均 **8.2 万**（2026-11-16 在 16 个 fold 上实测，见 §1.3；D3、D4 偏多，见 §12）。
- **query_id**（int32，全部 fold 内唯一）= `fold_ordinal × query_id_fold_offset + position`。`fold_ordinal` 是 fold 在 `fold_manifest.yaml` 中的位置（D1 = 0 … T4 = 15），`position` 是该 fold 的 query 按 `customer_idx` 升序的序号（从 0 开始），`query_id_fold_offset = 10,000,000`（`configs/queries.yaml`）。多个 fold 的表可以只按 `query_id` join，不会接错。反向解析用 `hmrec.data.queries.decode_query_id`。
- **Label 集合 `G_q`** = 该用户在 label 周买过的**去重后**的 `article_id` 集合。
- 某一路没有给某个 query 输出结果，算这一路的漏召回（miss），**query 不能从分母里删掉**。

### 2.4 商品范围

- **商品目录 `C`**：`articles.csv` 中所有商品。Label 不按任何商品池过滤。
- **活跃池 `A(a)`**：在 `[a−28d, a)` 内至少有 3 条购买记录的商品。这是"近期在卖"的近似，不是真实库存。
- **哪些 channel 用活跃池过滤**：R1、R3、R4、R5、R6 的输出限制在 `A(a)` 内；R2（复购）不过滤，因为它本来就基于用户自己的历史。**R3（同款不同色）原来也不过滤，2026-11-17 在 M4 按预先登记的规则改为只输出 `A(a)` 内的商品**（一次性修改，见下一句和 `docs/decisions.md`）：去掉 R3 在 `A(a)` 外的候选后，4 周合并的六路 union recall 只下降 0.045 个百分点（< 0.1 个百分点）。过滤放在取 Top-50 之前（与 R4 一致），空出的名额由活跃池内的同款商品补上；γ、τ 不重新调，沿用 D 阶段的选择（γ 1.0、τ 30）。D 阶段 R3 的 HPO 和 `results/d_phase.csv` 中的 R3 记录都是在不过滤的设置下得到的。这个设定在 M0 冻结，**不当作超参数调**。唯一的例外：R3 是否改为只输出 `A(a)` 内的商品，在 M4 按 §5.7 第 4 项预先登记的标准决定一次（系统层面的决定，不是 D 阶段 HPO）。
- 用户以前买过的商品**允许再推荐**（复购在 H&M 很常见）。

### 2.5 指标

全部是"先算每个 query 的分数，再取平均"（macro）。

| 指标 | 定义 |
|---|---|
| Recall@K | `|Top-K ∩ G_q| / |G_q|`。分母是完整的 `|G_q|`，**不在 K 处截断**，所以 label 多于 K 个的 query，Recall@K 最高只有 `K / |G_q|` |
| AP@K / MAP@12 | 设前 K 个位置中第 r 位命中，`P(r)` = 前 r 位的命中数 ÷ r。`AP@K = Σ P(r)`（对前 K 位中每个命中的位置 r 求和）`÷ min(K, |G_q|)`。K = 12 时和 Kaggle H&M 比赛的定义一致；MAP@12 是 AP@12 对 query 的平均 |
| NDCG@K | 二值相关性。`DCG@K = Σ 1 / log2(r + 1)`，只对前 K 个位置中命中的位置 r 求和。**理想 DCG 的命中数是 `min(K, |G_q|)`**：`IDCG@K = Σ_{r=1..min(K, |G_q|)} 1 / log2(r + 1)`。`NDCG@K = DCG@K / IDCG@K`，完美排序（前 `min(K, |G_q|)` 位全部命中）得 1.0 |
| Union recall | 6 路合并去重后，所有候选覆盖 `G_q` 的比例：`|候选 ∩ G_q| / |G_q|`，不截断；同一个商品被多路召回只算一次 |

对这几个指标都成立的约定（`src/hmrec/eval/metrics.py`）：

- **`G_q` 是用户在 label 周实际购买的全部商品（去重，§2.3），不限于被召回的那部分。**没有被任何一路召回的 label 仍然计入 `|G_q|`，所以它压低 Recall、AP 的分母项和 NDCG 的理想 DCG。NDCG 一行原来写的是"理想 DCG 用完整的 `G_q` 计算"，本意就是这一条；截断由"@K"隐含。原来的写法可以被读成"理想 DCG 不在 K 处截断"，2026-11-18 改成了上面的写法（`docs/decisions.md`）；代码和此前记录的数值都没有变。
- 参与平均的 query 是 label 表里的全部 query。某个 query 没有任何预测、或者没有命中，得 0 分，**留在分母里**（§2.3）。
- 输出不足 K 个时，缺的位置按"没命中"计，不补齐。
- 位置由 `rank` 决定（1 = 最好）：每个 query 的 `rank` 必须是 1, 2, …, n，同一个 query 内商品不重复；预测里出现 label 表中没有的 query 是错误。
- 4 周合并分数是把 4 周的 query 放在一起平均（§4.1）。

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
- **D 阶段分组诊断**：D 阶段的报告中，除 4 周合并分数外，再分别报告 **D1–D2** 和 **D3–D4** 的分数（各自两周的 query 放在一起平均，即 query-weighted）。**只作诊断，不参与选择**，选择规则仍是 §4.2 和 `configs/selection_rule.yaml`。原因：D3、D4 疑似促销周，在 4 周合并分数中约占 60% 权重（见 §12）；分组分数用来发现"只在促销周更好"或"只在普通周更好"的配置。

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

**一步有多个候选时**（如 R1 的 τ 有 6 个非默认值），实现在 `src/hmrec/hpo/selection.py`（`selection_rule.yaml` 不改）：

1. 每个候选分别与**当前默认值**（上一步选出的配置）按上面两条比较，得到"通过 / 不通过"。
2. 没有候选通过 → 保留默认值。
3. 有候选通过 → 在通过的候选中，与最高合并分数的**相对差距 < 1%**（与第 1 条同一个阈值）的都视为"差不多"，从中选**优先顺序最靠前**的那个。所以最高分的候选如果没通过，而另一个候选通过了，选通过的那个。
4. 优先顺序在运行前写在 `configs/hpo/d_phase.yaml` 中，运行后不得修改：
   - 有简单 / 成本差别的（变体、窗口、开关、维度）：按列出的顺序，更简单 / 更便宜 / 更接近默认的在前。例如 G0 < A1、recency-only < decayed frequency、CF0 < CF3、W0 < W2、factors 64 < 128；R4 的 seed 网格为 (90, 关) [默认] < (90, 开) < (180, 关) < (180, 开) < (365, 关) < (365, 开)。
   - 没有成本差别的纯数值参数（τ、γ、α、λ、α_conf）：按与**当前默认值**的距离排，距离相同时较小的值在前。也就是在"差不多"的候选中选离当前默认值最近的，而不是直接取最高分。
5. **R5 的 iterations {10, 15, 25} 不用这条规则**，按 §5 R5 的收敛检查处理：只有 25 按第 1、2 条通过（相对 15）时才改为 25。如果 15 之后分数明显下降（25 相对 15 的合并分数下降 ≥ 1%，且至少 3 周严格更差），或者 10 按第 1、2 条通过（相对 15），说明 λ 可能太小：**不自动改 iterations，停下来汇报**，由人决定是否回到 λ 那一步。
6. **网格边界**：网格**从不自动扩展**。R4、R5 的单个数值参数步骤（R4 的 α；R5 的 λ）如果选中值落在该步取值范围的最小或最大值上，runner 停下这一路（`stop_at_boundary`），由人决定是否扩大范围；报告中写明选中的是哪个值。其他数值步骤（R1、R2 的 τ，R3 的 γ、τ）选中边界值时照常继续，只在报告中注明。R5 的 α_conf 已经扩展过一次（{10, 20, 40} → {10, 20, 40, 60, 80}），**每个参数最多扩一次**，所以不再停下；R5 的 factors 只有 {64, 128} 两个值，128 通过就接受，不扩到 256，也不加这条检查（docs/decisions.md 2026-11-17）。`stop_at_boundary` 只在**切换到**边界值时触发，且不支持多参数步骤（如 R4 的 seed 窗口 × 衰减），这些情况每一路跑完后由人工检查。

前一步选出的配置是下一步的默认值，所以 runner 按 §4.3 的顺序自动执行这条规则；最终选择在写入 `configs/locks/` 之前由人确认。每一步的报告列出所有候选：是否通过、4 周各自的分数、合并分数、D1–D2 / D3–D4 分组分数、最终选中哪个以及原因。

### 4.3 顺序

每个组件都是先做 ablation（选变体），再在选中的变体上做 HPO。HPO 一次只动一个参数，其他参数保持当前最优值。

### 4.4 记录

每个 trial 记录：组件、变体、超参数、4 周各自分数、合并分数、query 数、运行时间、内存峰值、代码版本（git commit）。所有结果汇总到 `results/d_phase.csv` 和 `results/v_phase.csv`。

---

## 5. 六路 Recall：定义、Ablation、HPO

**每个 trial 都要在 D1–D4 每周重新计算**：用该周 anchor 之前的数据重新统计或重新 fit。不能 fit 一次然后评 4 周。

**最终配置（M5，2026-11-17 锁定）**：六路的最终配置写在 `configs/locks/channels.lock.yaml`（R1–R5）和 `configs/locks/twotower.lock.yaml`（R6），下面每一路表格的最后一行（"M5 锁定"）给出。表中的分数是该 trial 在 D1–D4 上的 4 周合并分数，**只作记录**，偏乐观（§12 第 4 条）。**M6 之后只能通过 `src/hmrec/recall/locked.py` 从锁定文件构建六路**，不再读取 `configs/recall/` 的默认值或 `results/d_phase.csv`（有测试扫描 M6 之后的代码）。

统一输出格式（每路一张表）：`query_id, article_idx, channel, rank, score`。每个 query 内 `article_idx` 不重复；按 `score DESC, article_idx ASC` 排序；`rank` 从 1 开始连续编号；不足 quota 时不补齐。

**每路的输出规格**（`src/hmrec/recall/base.py` 的 `OutputSpec`）：每一路声明自己的额外列（列名必须以该路前缀开头，如 `pop_`，不允许 null）和排序键（默认 `score DESC, article_idx ASC`）。写盘前的校验是双向的：多了未声明的列、缺少已声明的列、dtype 不一致都报错；并检查 `rank` 与声明的排序键一致。

**R1 的排序例外**：R1 有额外列 `pop_is_fallback`（int8），排序键为 `pop_is_fallback ASC, score DESC, article_idx ASC`。原因：年龄桶内的行用桶内分数 `S(g, i)`，补齐行用全局分数 `S(i)`，两者量级不同，补齐行的分数可能比桶内行大；按 `score DESC` 排会把补齐行排到桶内行前面。所以 R1 的 `score`（即 `pop_score` 特征）混合两种量级，由 `pop_is_fallback` 区分；B 组的 `pop_bucket_score`（§8.3）为所有候选提供统一的 `S(g, i)` 量级。

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

- 每个用户拿自己年龄桶的 Top-100；不足 100 时，用全局衰减热度按顺序补齐，跳过已选商品，并标记 `pop_is_fallback=1`。全局衰减热度 `S(i) = Σ exp(−age_days / τ)`，对 `[a−7d, a)` 内**所有**用户买 `i` 的购买行求和，τ 与桶内相同；补齐行排在桶内行之后（见本节开头的排序例外）。每个 fold、每个年龄桶的补齐行数作为诊断写进日志和结果。
- **G0**：所有用户同一个全局列表，分数是 `[a−7d, a)` 内的原始购买行数（不衰减、不分桶），没有补齐行。
- 默认：τ = 3 天，quota 100，限制在活跃池 `A(a)` 内。

| 类型 | 内容 | 配置数 |
|---|---|---|
| Ablation | **G0**（全局、不分桶、7 天原始计数）vs **A1**（τ=3） | 2 |
| HPO | `τ ∈ {1, 2, 3, 4, 5, 6, 7}` | +6 |
| **M5 锁定** | **A1，τ = 3**（没有候选通过规则，保留默认配置），窗口 7 天，quota 100。规则选择；trial `d2094d455698`；Recall@100 = 7.388% | — |

**合计 8 个配置，分钟级。**

### R2 — 时间衰减复购（Decayed-Frequency Repurchase）

**定义**：只推荐用户在 `[a−180d, a)` 内买过的同一个 `article_id`：

`S(u, i) = Σ exp(−age_days / τ)`，对用户 `u` 买 `i` 的每一条购买行求和（重复行各算一次）。

- 默认：τ = 30 天，quota 最多 30（soft，不补齐）。没有购买历史的用户输出为空。

| 类型 | 内容 | 配置数 |
|---|---|---|
| Ablation | **Recency-only**（按最近一次购买时间排序）vs **Decayed frequency**（τ=30） | 2 |
| HPO 第 1 步 | `τ ∈ {14, 30, 60}` | +2 |
| HPO 第 2 步 | 历史窗口 `lookback_days ∈ {180, 365}`（默认 180；优先顺序 180 < 365，短的更简单）。**在看到 D 阶段 R1–R3 结果之后加入**，见 `docs/decisions.md` | +1 |
| **M5 锁定** | **Decayed frequency，τ = 30，历史窗口 365 天**，quota 30。规则选择；trial `f562d598b0d6`；Recall@30 = 4.266% | — |

Recency-only 的分数 = `1 / age_days`，`age_days` 取用户最近一次购买该商品的年龄（昨天 = 1.0，前天 = 0.5）；更早的购买和购买次数都不参与。同一天买的多件商品同分，只按 `article_idx ASC` 排序，不用其他信息打破平局，保持它是纯粹的对照组。

**合计 5 个配置。**

### R3 — 同款不同色（Sibling，S2）

**定义**：
- 风格偏好：`P(u, c) = Σ exp(−age_days / τ)`，对用户 `u` 在 `[a−180d, a)` 内买过的、`product_code = c` 的每一条购买行求和。
- 市场热度：`M(j)` = 商品 `j` 在 `[a−7d, a)` 内的全站购买行数。
- 分数：`S(u, j) = P(u, c(j)) × (1 + M(j))^γ`。
- 候选：与用户 180 天内买过的商品同 `product_code`，但**排除**用户 180 天内买过的同一个 `article_id`。
- 默认：τ = 30，γ = 0.5，quota 最多 50（不补齐）。

| 类型 | 内容 | 配置数 |
|---|---|---|
| Ablation | `γ ∈ {0, 0.5, 1.0, 1.5, 2.0}`（γ=0 表示只看风格偏好，用来检验市场热度有没有用；1.5、2.0 **在看到 D 阶段 R1–R3 结果之后加入**，见 `docs/decisions.md`；优先顺序按与默认值 0.5 的距离） | 5 |
| HPO | 在最佳 γ 上调 `τ ∈ {14, 30, 60}` | +2 |
| **M5 锁定** | **γ = 1.0，τ = 30，只输出 `A(a)` 内的商品**（`active_pool_only: true`），风格窗口 180 天，市场窗口 7 天，quota 50。规则执行：D 阶段规则选中 γ = 1.0、τ = 30（trial `3188358fa459`，Recall@50 = 3.257%），M4 预先登记的规则再把它改为只输出 `A(a)`；锁定的是切换后的 trial `06025cc3d129`（`results/m4_trials.csv`），Recall@50 = 3.240% | — |

**合计 7 个配置。** 如果选中 γ = 2.0（新网格的边界），不再继续扩展，只在报告中说明。

**活跃池**（2026-11-17，M4）：R3 只输出 `A(a)` 内的商品（`configs/recall/r3_sibling.yaml` 的 `active_pool_only`，在 `configs/m4_union_check.yaml` 的 `variants.r3_active_pool` 中打开；M5 在 `configs/locks/channels.lock.yaml` 中锁定为 `true`）。过滤放在取 Top-50 之前，空出的名额由池内的同款商品补上；`P(u, c)`、`M(j)` 和已购买商品的排除仍用全部商品。本节的 ablation 和 HPO 都是在不过滤的设置下做的，γ、τ 不重新调（§2.4、§5.7 第 4 项）。

### R4 — ItemCF

**定义**（拟合窗口 `[a−365d, a)`，用户和商品的交互只记 0/1）：
- 用户权重（IUF）：`w_u = 1 / ln(1 + d_u)`，`d_u` = 用户在窗口内买过的不同商品数。
- 加权共现：`C_w(i, j) = Σ_u w_u · R_ui · R_uj`；加权边际：`n_w(i) = Σ_u w_u · R_ui`。
- 相似度（从 seed `i` 到候选 `j`）：`s(i→j) = C_w(i, j) / (n_w(i)^α · n_w(j)^(1−α))`。去掉 `i = j`。
- 每个 seed 只保留 Top-`K_n`（=100）个邻居。**邻居 `j` 限定在 `A(a)` 内，过滤在取 Top-`K_n` 之前**：每个 seed 保留的是它最好的 `K_n` 个**活跃**邻居。`C_w`、`n_w`、`d_u` 用拟合窗口内的全部商品计算，不受 `A(a)` 影响；seed 本身不要求在 `A(a)` 内（不活跃的 seed 照样贡献分数）。原因：D1 上拟合窗口的图有 71,713 个商品，其中只有 20,764 个（29%）在 `A(a)` 内；如果先取 Top-100 再过滤，实测 **45.4%** 的邻居名额会落在不活跃商品上被浪费（D1，CF3，α=0.7）。
- 用户分数：`S(u, j) = Σ_i q_ui · s(i→j)`，对用户在 seed 窗口 `[a−L_seed, a)` 内买过的每个不同 seed `i` 求和。
- **Seed 权重** `q_ui`：衰减关闭时 `q_ui = 1`；打开时 `q_ui = exp(−age_ui / τ_seed)`，`age_ui` 是用户**最近一次**购买 seed `i` 距 anchor 的天数（昨天 = 1；同一个 seed 买多次只取最近一次，不累加）。`τ_seed` 固定为 30 天（与 R2、R3 的 τ 默认值一致），**不调**。建图（`C_w`、`n_w`、`d_u`）保持不衰减。
- 默认：CF3（IUF + α=0.7），`L_seed` = 90 天，seed 衰减关闭，quota 100，不排除用户已买的商品。

| 类型 | 内容 | 配置数 |
|---|---|---|
| Ablation | **CF0**（不加权，α = 0.5，即普通 cosine `s = C / sqrt(n(i)·n(j))`）vs **CF3** | 2 |
| HPO 第 1 步 | `α ∈ {0.5, 0.7, 0.9}` | +2 |
| HPO 第 2 步 | 3×2 小网格：`L_seed ∈ {90, 180, 365}` × seed 衰减 {关, 开（τ_seed = 30）}；(90, 关) 就是当前配置 | +5 |
| **M5 锁定** | **CF3，α = 0.7，`L_seed` = 365 天，seed 衰减开**（τ_seed = 30），建图窗口 365 天，`K_n` = 100，quota 100。规则选择；trial `df281d91d7c1`；Recall@100 = 9.332% | — |

**HPO 第 2 步为什么用网格，而不是"一次只改一个因素"**：两个因素明显相互影响。seed 窗口越长，覆盖的用户越多（D1 上 32.2% 的 query 因 90 天内没有购买而输出为空），但不衰减时，几个月前的旧购买和昨天的购买权重相同，噪声随窗口变长而变大；衰减正是用来抵消这部分噪声的。单独调窗口（衰减关）会低估长窗口，单独调衰减（90 天）又看不到它在长窗口上的作用。网格只有 6 格，每格在 D1 上只需十几秒（§1.3），成本可以忽略。

不同 α 可以复用同一周的 `C_w` 和 `n_w`，只需重新归一化、重新取 Top-`K_n`。**内存**：按 seed 分块计算（见 §1.3）。

**实现**（`src/hmrec/recall/r4_itemcf.py`）：`C_w` 不整块存储。每块 seed（`chunking.item_chunk_size`）的 `C_w(i, ·)` 用 scipy 稀疏矩阵乘法算出，对每个 α 各做一次归一化和 Top-`K_n`，然后释放；所以一次扫描同时得到多个 α 的邻居表（`generate_for_alphas`）。每个 `C_w` 元素按同样的用户顺序求和，分块和不分块的结果逐位相同（有测试）。只计算本周 query 用到的 seed 的邻居。用户打分同样是稀疏矩阵乘法（seed 0/1 矩阵 × 邻居相似度矩阵），按 `chunking.query_chunk_size` 分块。CF0 和 CF3 只在用户权重上不同，α 是独立的配置项；CF0 的 ablation 用 α=0.5（即上表的 `C / sqrt(n(i)·n(j))`）。Seed 窗口和 seed 衰减只影响用户打分，不影响建图；`L_seed` 不能超过拟合窗口（365 天），保证每个 seed 都在图里。

**合计 9 个配置。**

### R5 — WALS（时间衰减的隐式 ALS）

**定义**（拟合窗口 `[a−180d, a)`）：
- 交互强度：`r_ui = Σ exp(−age_days / 60)`，对每条购买行求和。
- 置信度：`c_ui = 1 + α_conf · r_ui`，作为 CSR 矩阵的值传入 `implicit`，并设置库参数 `alpha=1.0`，避免重复放大。
- 默认：W2，`α_conf=20`，`factors=64`，`λ=0.1`，`iterations=15`，固定 seed。
- 推理：拟合得到的用户向量点乘商品向量，取活跃池内 Top-100。不排除已购商品。窗口内没有交互的用户输出为空。
- **实现**（`src/hmrec/recall/r5_wals.py`）：只有窗口内有购买的用户和商品才有行 / 列（才有向量）。先用 float32 矩阵乘法（按 `chunking.query_chunk_size` 个 query 分块）找出每个用户得分不低于第 100 名的商品，再用 `hmrec.vectors.dot_pairs` 重新计算这些候选的分数并排序，保证 `score` 与 §8.3 的 `wals_dot` 逐位相同。两种计算在最后一位可能不同，所以在第 100 名附近几乎同分时，保留的集合可能与精确排序差最后一个商品；保存的分数和顺序总是精确的。`implicit` 用 CPU 版，`num_threads` 取自 `configs/base.yaml`，`random_seed` 在 `configs/recall/r5_wals.yaml` 中固定。

| 类型 | 内容 | 配置数 |
|---|---|---|
| Ablation | **W0**（`r_ui = 1`，只看有没有买过）vs **W2** | 2 |
| HPO 第 1 步 | `α_conf ∈ {10, 20, 40, 60, 80}`（60、80 是看到 D 阶段结果后扩展的，见 docs/decisions.md 2026-11-17） | +4 |
| HPO 第 2 步 | `λ ∈ {0.01, 0.1, 1.0}` | +2 |
| HPO 第 3 步 | `factors ∈ {64, 128}` | +1 |
| 收敛检查 | 在最终配置上跑 `iterations ∈ {10, 15, 25}`，只确认 15 次时已经进入平台期 | +2 |
| **M5 锁定** | **W2，`α_conf` = 60，`λ` = 0.1，`factors` = 128，`iterations` = 15**，窗口 180 天，衰减 τ = 60，seed 42，quota 100。规则选择；trial `3261309f8e8f`；Recall@100 = 10.807% | — |

**Iterations 不做 early stopping**：ALS 每一轮都是闭式解（closed-form），很快收敛；过拟合由 λ 和 factors 控制。如果 25 次明显比 15 次好，就改成 25；如果 15 次之后分数明显下降，说明 λ 太小，回到第 2 步加大 λ，而不是减少 iterations。

**合计约 11 个配置 × 4 周 ≈ 44 次 ALS 拟合。**

### R6 — Two-Tower（训练方式详见 §6）

**模型**（全部数值固定、不调；写在 `configs/recall/r6_twotower.yaml`，实现在 `src/hmrec/recall/twotower/`；理由见 `docs/decisions.md` 2026-11-17 "Two-Tower 模型结构"）：

**类别特征的编码规则**（两个塔都适用）：取值数（含缺失编码 0）≤ 20 的字段用 one-hot；> 20 的字段用 embedding，维度 16，`padding_idx = 0`，初始化 N(0, 0.02²)。article ID 和 product_code 例外（见下表）。按实际数据的取值数：

| 字段 | 取值数（含 0） | 编码 | 维度 |
|---|---|---|---|
| product_type_no | 133 | embedding | 16 |
| product_group_name | 20 | one-hot | 20 |
| garment_group_no | 22 | embedding | 16 |
| index_group_no | 6 | one-hot | 6 |
| section_no | 58 | embedding | 16 |
| colour_group_code | 51 | embedding | 16 |
| age_bucket | 8 | one-hot | 8 |
| club_member_status | 4 | one-hot | 4 |
| fashion_news_frequency | 4 | one-hot | 4 |

- **Item tower**（历史商品和目标商品共用同一个 item tower，参数完全共享）：下列各部分**拼接**：

  | 部分 | 结构 | 维度 |
  |---|---|---|
  | 图像 | `Linear(512 → 128)`，输出乘以 `image_mask`（没有图像时这一部分包括 bias 全为 0） | 128 |
  | 文本 | `Linear(384 → 128)`，输出乘以 `text_mask` | 128 |
  | article ID | `Embedding(n_articles + 1, 64, padding_idx = n_articles)`，初始化为 0 | 64 |
  | product_code | `Embedding(47,225, 32, padding_idx = 0)`，初始化为 0 | 32 |
  | 6 个类目字段 | 按上面的规则：16 + 20 + 16 + 6 + 16 + 16 | 90 |
  | **合计** | | **442** |

  拼接后 → `Linear(442 → 256)` → ReLU → `Linear(256 → 128)` → L2 归一化。
  - 没有图像向量的 442 个商品（见 §2.1）：图像部分乘以 mask 后为 0（不填 0 向量当作真实输入），商品向量由文本、ID 和类目部分得到。这些商品**照常参与训练和召回**，不从商品目录或活跃池中删除。
  - article ID 和 product_code 初始化为 0：训练中没出现过的商品（或款式），这一部分保持 0，向量只由内容和类目决定，不引入随机噪声。
- **User tower**：用户在 anchor（训练时为购买日）之前**最近 20 件购买**（不限时间窗口；去重、排序、padding 规则见 §6.1）经过同一个 item tower 编码（padding 位置不参与编码）。**按位置加权 pooling**：为 20 个位置各学一个标量 `w_j`（初始为 0，即开始时等于 masked mean），`pooled = Σ_j a_j · h_j`，`a = softmax(w)` 只在 mask 为 True 的位置上计算；历史为空时 `pooled = 0`。Profile：三个字段都是 one-hot，共 16 维。`[pooled 128 ; profile 16]`（144 维）→ `Linear(144 → 256)` → ReLU → `Linear(256 → 128)` → L2 归一化。
- **Loss**：multi-positive、duplicate-aware 的 in-batch softmax 加 LogQ 修正：
  - 列 = batch 中**去重后**的目标商品（同一商品在 batch 中出现多次只算一列，不会被当成两个负样本）；
  - `logit[i, k] = u_i · v_k / τ − log Q(k)`，**τ = 0.05**，对所有列（包括目标列）都做 LogQ 修正；
  - `Q(k)` = 当前训练集中以 k 为目标的样本占比：base 用全部 base 训练样本（不含报警用户）统计，每周 fine-tune 用该次的训练集（新一周 + replay）统计；
  - 同一 group（同一用户同一天）的其他正样本所在的列，在该行置为 −∞（既不是目标也不是负样本）；loss = 每行对其目标列的交叉熵的平均。
- **训练**：batch 512；每个 epoch 用 `(seed, epoch)` 在 CPU 上生成随机打乱顺序，保留最后一个不满的 batch；Adam（β = 0.9, 0.999；ε = 1e-8），lr **1e-3 固定不衰减**（fine-tune 1e-4），weight decay 0，不做梯度裁剪；训练 **6 个 epoch**，每个 epoch 存一个 checkpoint（`artifacts/models/twotower/base/epoch_{k}.pt`）；seed 42；留出 1% 的用户只记录 loss 作为报警，不用于任何选择。每一步只对 batch 中出现的不同商品（目标 + 历史，去掉 padding）编码一次，再按下标取回。
- **设备**：真实模型在 mps 和 cpu 上各跑 200 步（各自一个进程，前 10 步不计时），只有 MPS 的平均每步时间比 CPU 快 1.3 倍以上时才用 MPS，否则用 CPU（MPS 上 embedding 反向传播不是确定性的，结果无法精确复现）。结果记入 `docs/decisions.md`。
- **推理**：R6 是普通的 `RecallChannel`（`recall/twotower/infer.py`）。历史为 anchor 之前最近 20 件（规则与训练相同，数据先裁剪到当周 query 用户）；没有历史的用户使用只含 profile 的向量。只对活跃池 `A(a)` 中的商品计算向量，按 `chunking.dense_score_query_chunk_size` 分块取 Top-100（与 R5 共用 `recall/vector_scoring.py`）；`score` 用 `vectors.dot_pairs` 重新计算，保证与 §8.3 的 `tt_dot` 逐位相同；排序 `score DESC, article_idx ASC`。
- **E_base 的选择**：默认 epoch 1，候选 epoch 2–6，epoch 少的优先（`configs/hpo/d_phase.yaml` 的 `r6_twotower.base_epoch` 步骤），按 §4.2 的多候选规则选择；每个 checkpoint 不做任何更新，直接预测 D1–D4，结果写入 `results/d_phase.csv` 和 `results/d_phase_steps/r6_twotower.csv`。选中 epoch 6 时停下，由人决定。
- 固定（不调）：以上全部，包括 batch 512、输出 128、温度 0.05、LR、history 长度 20、epoch 数 6。

| 类型 | 内容 | 成本 |
|---|---|---|
| HPO 第 1 步 | **E_base**：base model 每个 epoch 存一个 checkpoint（1–6），每个 checkpoint **不做更新**直接预测 D1–D4 | 1 次 base 训练 |
| Ablation | **Frozen base**（之后不更新）vs **每周 fine-tune** | 包含在第 2 步 |
| HPO 第 2 步 | `E_update ∈ {1, 2}` × `replay ∈ {0, 50%}`，共 4 条滚动链 | 4 条链 × 4 周 |
| 稳健性 | 前 2 名的配置各再跑 2 个 seed | 约 4 条链 |
| 可选 | ① 只用 ID、不用 CLIP / MiniLM 的 content ablation；② base 长度改为 52 周 | 时间允许再做 |
| 不做 | 每周从零重训（full retrain）的对照 | — |
| **M5 锁定** | **E_base = 4，E_update = 2，replay = 50%**，fine-tune lr 1e-4，chain seed 42，其余为上面的固定值。**人工决定**：规则选中的是 (E_update = 2, replay = 0)（trial `ef62019e6c73`，14.576%），人工采用 (2, 50%)（trial `835ef201ee4d`，Recall@100 = 14.634%），理由见 `docs/decisions.md`。模型文件和 sha256 见 §6.5 | — |

### 5.7 D 阶段的收尾：Union 检查

用 6 路各自的最终配置，在 D1–D4 上计算：

1. **Union recall**（目标 ≥ 24%，即 G2）；
2. **Leave-one-out union recall**：去掉每一路后，union recall 下降多少，用来判断这一路有没有带来其他路没有的候选；
3. 每个 query 的候选数分布（平均值、P50、P90）；
4. **R3 活跃池诊断**：在 D1–D4 上统计 R3 候选中不在 `A(a)` 内的比例和命中率（与 `A(a)` 内的候选对比），以及去掉其中"没有被其他任何一路召回"的候选后 union recall 的变化。**预先登记的决策标准**：如果 union recall 下降 **< 0.1 个百分点**（绝对值），则把 R3 改为只输出 `A(a)` 内的商品；否则保持不变（§2.4）。R2 做同样的统计，**只做诊断，不改动**。该标准在看过 D1 的 R3 结果之后制定，见 `docs/decisions.md`（2026-11-16 "R3 活跃池外候选"）。

如果 union recall 明显低于 24%，**先改 recall，不进入 F 阶段**。是否删掉某一路，由 V 阶段的系统级 ablation（MAP@12）决定，不在这里决定。

**实现**（`src/hmrec/hpo/union_check.py`、`scripts/m4_union_check.py`，设置在 `configs/m4_union_check.yaml`）：
- 六路使用 D 阶段的选择（从 `results/d_phase.csv` 重放规则，不运行新的 trial），加上 `manual_selection` 中的人工选择（R6 采用 (E_update = 2, replay = 50%)，见 `docs/decisions.md`）。
- R1–R5 的候选优先复用预检查的文件（`artifacts/hpo/D/union_precheck/`，trial_id 相同即配置相同）；R6 **不重新训练**，每一周用训练链在这一周 fine-tune **之前**保存的模型预测（D1 用 base 的 E_base checkpoint，D2 用 `chain/{trial_id}/D1.pt`，依此类推）。候选写到 `artifacts/m4/candidates/`，结果写到 `results/m4_union_check.csv`。
- 每一路每一周的选择指标必须与 `results/d_phase.csv` 中的记录**逐位相同**，否则报错停下。
- 内存：union 只用**命中 label 的候选**计算（先把每一路的候选与 label 做一次 semi join，命中的行约占 1/400），union recall 和 leave-one-out 都在这些小表上做；每个 query 的候选数按 query 分块统计。结果与直接在全部候选上计算完全一致（有测试）。
- 第 4 项的决策标准用 **4 周合并（query-weighted）的 union recall**，阈值 0.001（0.1 个百分点）写在 `configs/m4_union_check.yaml`，严格小于才改。M4 的运行只计算并报告规则的结论；如果结论是"改"，R3 的配置修改和候选重新生成在锁定（M5）之前另行完成。

### 5.8 D 阶段工作量汇总

| Channel | 配置数 | 成本 |
|---|---|---|
| R1 | 8 | 极低 |
| R2 | 5 | 极低 |
| R3 | 7 | 低 |
| R4 | 9 | 中（每周建图，需要分块） |
| R5 | ~11 | 中（约 44 次 ALS） |
| R6 | 1 次 base + 约 8 条链 | 高（GPU） |

---

## 6. Two-Tower 训练链（全项目只有一条）

### 6.1 训练样本

每个 (用户, 购买日, 商品) 对应一条正样本：`(用户在购买当天之前最近 20 件购买 + 用户 profile) → 买的商品`。同一用户同一天重复购买同一商品只算一条；同一用户同一天的多个正样本共用一个 group_id（multi-positive loss 用）。历史只取购买日**之前**（不含当天）的购买，按同样的 (用户, 日, 商品) 去重，从近到远排列，同一天按 `article_idx` 升序；不足 20 件在末尾补 padding 并带 mask；**padding 编号为 `n_articles`**（不对应任何真实商品）。商品特征表（图像、文本向量及 mask、product_code 和类目编码）末尾有一行 padding 专用行（向量全 0、mask 为 False、编码全 0），模型的 ID / 类目 embedding 用 `padding_idx` 指向它。用户历史预先算好，存成 int32 数组 `[N, 20]`（实现：`src/hmrec/recall/twotower/data.py`）。

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

- **每个分数都是"先预测、后训练"得到的**（progressive validation）：被预测的那一周，模型还没见过。实现上（`src/hmrec/recall/twotower/chain.py`）每一周先用当前模型生成候选，再 fine-tune；各 fold 必须按时间顺序处理，否则报错。
- **新数据**：anchor 为 `a` 的这一周，购买日在 `[a, a + 7)` 的购买；样本、历史、去重规则与 base 训练完全一致（§6.1）。
- **Replay 的定义**：`replay` 是 fine-tune 集中旧数据所占的比例。旧数据从**这一周开始之前 4 周**（购买日 `[a − 28, a)`）的样本中不放回随机抽取，数量 `n_replay = round(n_new × replay / (1 − replay))`：`replay = 50%` 时 `n_replay = n_new`（新 : 旧 = 1 : 1），`replay = 0` 时只用新数据。抽样用固定 seed（`chain_seed` 和这一周的 anchor）。
- **报警用户**：继续排除 base 训练中那 1% 的报警用户（用 base 的数组和 seed 重新抽出同一批用户），新数据和 replay 中都不含他们；每周 fine-tune 后记录他们在这一周的 loss，只作报警，不用于选择。
- **优化器**：每周 fine-tune 都**重新初始化 Adam**（lr 1e-4 = base LR 的 0.1 倍，其余设置同 base），不继承上一周的优化器状态。每个 epoch 用 `(chain_seed, anchor, epoch)` 打乱顺序。LogQ 的 `Q` 用这一次的 fine-tune 集（新数据 + replay）统计。
- **保存**：每一周 fine-tune 后的模型存到 `artifacts/models/twotower/chain/{trial_id}/{fold}.pt`（不覆盖已有文件）；D4 预测完之后也用 D4 fine-tune 一次，得到 **M4**（`D4.pt`），F1 从它开始。
- **默认配置和候选**（`configs/hpo/d_phase.yaml` 的 `r6_twotower.update` 步骤）：默认是 **frozen**（`e_update = 0`：base 不更新，直接预测 D1–D4，就是第 1 步选中的 E_base 的结果，直接复用）。候选按优先顺序：`(E_update = 1, replay = 0) < (1, 50%) < (2, 0) < (2, 50%)`（epoch 少、不加 replay 的更便宜，排在前面），按 §4.2 的多候选规则选择。
- 所有链在 D1 上的分数相同（都是 base 的预测），真正区分配置的是 D2–D4：§4.2 的"4 周中至少 3 周严格更好"实际要求 D2、D3、D4 都更好，这是预期的行为。
- **稳健性**：按 §4.2 规则排名前 2 的配置（第 1 名是规则选中的；第 2 名是去掉第 1 名后规则会选中的；没有候选通过时按合并分数排）各再用 seed 43、44 跑一遍（只影响 fine-tune 的打乱顺序和 replay 抽样），汇报均值和波动（`results/d_phase_steps/r6_twotower_seeds.csv`）。**选择仍以 seed 42 的结果为准**，额外的 seed 只用于汇报。
- 内存：训练链按 fold 逐个处理，每周的 fine-tune 数组用完就释放；历史只读取涉及的用户的购买。
- **D 阶段结果**（2026-11-17）：规则选中 (E_update = 2, replay = 0)（合并 Recall@100 14.576%，frozen 为 11.843%）；**人工采用 (2, 50%)**（14.634%），这是偏离规则的决定，理由见 `docs/decisions.md`。

### 6.5 第 3 步：锁定

在 2020-07-01 之前，把 `base_window, E_base, E_update, replay, fine_tune_lr, seed` 写进 `configs/locks/twotower.lock.yaml`。**选中配置跑出来的 M4 就是之后使用的模型，不重新训练 base。**

**锁定内容（M5，2026-11-17）**：`configs/locks/twotower.lock.yaml` 记录：

- 完整的 R6 配置：base 窗口（购买日 2019-12-03 – 2020-06-02）、模型结构和训练设置的全部数值、`E_base = 4`、`E_update = 2`、`replay = 0.5`、fine-tune lr 1e-4、chain seed 42；
- 选择依据：**人工决定**。规则选中的是 (2, 0)（trial `ef62019e6c73`），锁定的是 (2, 50%)（trial `835ef201ee4d`），并指向 `docs/decisions.md` 的记录；
- 报警用户的设置：比例 1%、seed 42，以及抽样来源（base 训练数组的 `customer_idx.npy`）的 sha256 和抽出的人数（7,141）；
- 两个模型文件的路径和 **sha256**：`base`（`artifacts/models/twotower/base/epoch_4.pt`，在训练链中预测了 D1）和 `f1_start`（`artifacts/models/twotower/chain/835ef201ee4d/D4.pt`，即 M4，用来预测 F1）。

`artifacts/` 不进 git，所以用 sha256 确认以后用的是同一个文件：`src/hmrec/recall/locked.py` 在加载模型权重**之前**先核对 sha256，不一致就报错，不加载。

### 6.6 第 4 步：F → V → T，每周照做

| 时间 | 操作 |
|---|---|
| 07-01 | M4 预测 F1 → 输出候选 → 用 F1 fine-tune → M5 |
| 07-08 | M5 预测 F2 → 输出候选 → 用 F2 fine-tune → M6 |
| … | 每周相同，一直到 T4 |

每周的 Recall@100 照常记录。如果某周比前 4 周的平均值低 10% 以上，在报告中标出，但**不改配置**。

**实现（M6，`src/hmrec/recall/generate.py`、`scripts/m6_generate_candidates.py`）**：

- **只从锁定文件构建**：六路都通过 `src/hmrec/recall/locked.py` 得到。R6 用 `load_locked_twotower_chain`，返回的就是 D 阶段训练链的那个类（`recall/twotower/chain.py` 的 `TwoTowerChainChannel`），所以每周 fine-tune 走的是同一段代码：这一周的购买 `[a, a + 7)` 加 1 : 1 的 replay、排除锁定的报警用户、每周重新初始化 Adam、锁定的 `E_update = 2` 和 fine-tune lr 1e-4。
- **每个 fold 内的顺序**（R6）：① 取"预测这一周的模型"并核对 sha256；② 预测，保存向量和候选；③ 再用这一周 fine-tune，保存模型 `artifacts/models/twotower/chain_locked/{fold}.pt`，把 sha256 和 fine-tune 的行数、loss、报警用户 loss、耗时追加到 `results/m6_twotower_chain.csv`。**T4 只预测，不 fine-tune**，所以没有模型也没有记录。
- **顺序由代码强制**（`src/hmrec/recall/chain_records.py`）：F1 用锁定文件中的 `f1_start`（即 M4）；之后每个 fold 用上一个 fold 保存的模型，这个模型必须存在、在 `results/m6_twotower_chain.csv` 中有记录、并且 sha256 与记录一致，否则报错。所以不能跳过一周，也不能用别的运行留下的模型。
- **为特征保存的文件**（`features/` 不能 import `recall/`，只能读这些文件，见 §14.2）：R5 的向量（用户只存该 fold 的 query 用户）；R6 的向量（用户是该 fold 的 query 用户，商品是**商品目录中的全部商品**，不只是 `A(a)`，因为 R2 的候选可能在活跃池外；用**预测这一周时的模型**计算，即这一周 fine-tune 之前）；R1 完整的桶内分数表 `S(g, i)`（不受 Top-100 和活跃池限制）。R6 在 `A(a)` 内的商品向量就是生成候选时用的那一份，池外的商品另外编码后合并，所以候选和分数与只编码 `A(a)` 时完全相同。
- **一致性检查**（每个 fold 都做，不通过就报错，候选不写盘）：把保存的文件读回来，R5、R6 每个候选的 `score` 必须与用保存的向量算出的点积逐位相同；R1 每个非补齐候选的 `score` 必须与分数表中的值逐位相同。
- **每个 (fold, channel) 一个独立进程**，结束时把耗时和真实内存峰值（`ru_maxrss`）追加到 `results/m6_runs.csv`；`scripts/m6_run_folds.py` 按时间顺序逐个 fold 驱动。候选文件一旦存在就不覆盖，重跑时跳过并提示。
- **监控**（`src/hmrec/eval/monitor.py`、`scripts/m6_monitor_candidates.py`，结果追加到 `results/m6_candidates.csv`；只记录，不做任何选择）：F、V 周记录每一路的 Recall@quota、六路 union recall、每个 query 去重后的候选数、没有任何一路命中的 query 比例；每一路的 recall 与它前 4 周的平均值比较（F1 的前 4 周是 D1–D4，分数取自锁定文件），低 10% 以上时在 `low_recall_flag` 一列标出。**T 周在 M6 中完全不读 label**，只记录候选行数和每个 query 的候选数；T 周的指标留到 M8。代码拒绝为 T fold 构建评估器（有测试）。
- **停下由人决定的情况**（设置在 `configs/m6_candidates.yaml`）：任何进程的真实内存峰值超过 `memory.limit_gb`；schema 校验或一致性检查失败；模型的 sha256 与记录不一致；R6 fine-tune 的训练 loss 或报警用户 loss 不是有限值、或超过 `twotower_chain.max_fine_tune_loss`（7.0；D 阶段训练链每周的训练 loss 为 5.49–5.56，报警用户 loss 为 5.25–5.69）。

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

## 8. Ranker 宽表特征（共 71 个）

> 2026-11-17：最初的 56 个之外新增 15 个（B 组 2、C 组 1、D 组 1、E 组 11），总数 71。新增是在**特征锁定之前、没有任何 Ranker 结果的情况下**做的设计修改，依据是 H&M 比赛第一名方案对用户–商品交互特征和多时间窗口的强调（`docs/decisions.md`）。下面各表中新增的特征标 **（新增）**。

### 8.1 原则

- 每一行 = 一个 `(query, article)` 候选。
- **所有特征只用 `t_dat < anchor` 的数据**，时间窗口都从 anchor 往前算。
- 不使用原始 ID（`customer_id`、`article_id`、`product_code`）作为特征。
- 类目字段作为 LightGBM categorical feature。
- 缺失值保留为 NaN，LightGBM 会自己处理。

**落实后的约定（M6，`configs/features.yaml`、`src/hmrec/features/`）**：

- **宽表的列**：`query_id`、`article_idx`（int32，只作键，不是特征），然后是下面 71 个特征，顺序与 `configs/features.yaml` 一致。没有 `customer_idx`，也**没有 label 列**（label 在 M7 的 `ranker/sampling.py` 中才拼接）。
- **dtype**：数值特征一律 float32，缺失为 NaN；8 个类别特征（§8.4 的 6 个商品类目、§8.5 的 `club_member_status` 和 `fashion_news_frequency`）是 int16 整数编码：0 = 缺失，1..K = 该字段全部取值排序后的序号。编码表由 `articles`、`customers` 的全部行生成，所有 fold 共用一份，保存在 `artifacts/features/category_codes.json`；后面的 fold 重新生成后必须与它完全相同。
- **"购买行数 / 次数"都按购买行计**（同一用户同一天买同一商品两行算 2），与 R1、R2 的计数方式一致。唯一的例外是 `item_repurchase_ratio_52w`（见 §8.4）。
- **"距今天数"** = `anchor_day − t_dat_day`，昨天 = 1。
- **窗口**：1w = `[a−7, a)`，prev_1w = `[a−14, a−7)`，4w = `[a−28, a)`，52w = `[a−364, a)`；没写窗口的特征用全部历史 `t_dat < a`。每个窗口都通过 `data/asof.py` 取数。
- **缺失值规则**：计数类特征没有数据时为 0；"距今天数"、均值、占比、比值在无法定义时为 NaN（从没买过、窗口内没有数据、分母为 0、任一侧缺失）。
- **输出**：`artifacts/features/{fold}/part-{i}.parquet`，每 `chunking.query_chunk_size` 个 query 一个文件，按 `(query_id, article_idx)` 排序，同一个 query 的行连续且不跨文件；最后写 `features.json`，它存在才表示这个 fold 完成。生成后不覆盖。

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

落实（`src/hmrec/fusion/union.py`、`fusion/rrf.py`、`features/source.py`）：六路按 `(query_id, article_idx)` 合并去重，每个候选一行。`pop_is_fallback` 在 R1 召回的行上是 0 / 1，**R1 没有召回的行为 NaN**（与其他来源列一致）。`n_channels`、`best_rank` 存为 float32。`rrf_score` 用 float64 按固定顺序（pop、rep、sib、icf、wals、tt）求和后存为 float32。RRF baseline（§9.5）的排序：`rrf_score DESC`，再 `best_rank ASC`，再 `article_idx ASC`。

### 8.3 B 组：对所有候选都计算的稠密分数（7 个）

这组特征不管候选来自哪一路，都会计算。通常是最有用的一组。

| 特征 | 定义 |
|---|---|
| `tt_dot` | Two-Tower 用户向量 · 商品向量（当周 anchor 的模型） |
| `wals_dot` | WALS 用户向量 · 商品向量；用户或商品没有向量时为 NaN |
| `pop_bucket_score` | 该商品在用户年龄桶里的 R1 衰减热度分数（不受 Top-100 限制） |
| `img_sim_hist` | 商品 CLIP 向量与用户最近 20 件购买的 CLIP 平均向量的 cosine 相似度。平均向量只用这 20 件中有图像向量的商品。**商品本身没有图像向量，或用户最近 20 件购买都没有图像向量时，取 NaN** |
| `txt_sim_hist` | 同上，换成 MiniLM 文本向量。文本覆盖全部商品，所以只有在**用户没有历史购买（`t_dat < a` 内没有购买）时取 NaN** |
| `tt_sim_max_hist` **（新增）** | 候选商品与用户最近 20 件购买（与 `img_sim_hist` 同一份历史）的相似度中的**最大值**。相似度 = 该 fold 保存的 R6 商品向量的点积（向量已 L2 归一化）。用户没有历史时为 NaN |
| `tt_sim_top5_hist` **（新增）** | 同上，取最大的 5 个相似度的平均；历史不足 5 件时取全部。用户没有历史时为 NaN |

落实（`src/hmrec/features/dense.py`）：

- `tt_dot`、`wals_dot` 用 M6 保存的向量和 `vectors.dot_pairs` 计算（与 R5、R6 的 `score` 是同一个函数）。`tt_dot` 用的是预测这一周时的模型（这一周 fine-tune 之前），对所有候选都有值。**每个 fold 全量检查**：被 R6 召回的行，`tt_dot` 与 R6 候选的 `score` 逐位相同；被 R5 召回的行，`wals_dot` 与 R5 的 `score` 逐位相同。
- `pop_bucket_score` 读 R1 保存的桶内分数表；表里没有的 (年龄桶, 商品) 记为 **0**（该年龄桶在窗口内没有人买过，衰减热度就是 0）。
- `img_sim_hist`、`txt_sim_hist`："最近 20 件购买"与 Two-Tower 用的是**同一个函数**（`src/hmrec/data/histories.py` 的 `recent_purchase_histories`：同一天同一商品算一件，不同天买同一商品算多件，从近到远）。平均向量 = 这 20 件中有向量的那些的简单平均（同一商品出现几次就计几次），**平均后做 L2 归一化**；候选商品的向量也重新归一化，两者的点积就是 cosine。
- `tt_sim_max_hist`、`tt_sim_top5_hist`：与上面两个"平均向量"特征不同，它们看的是**单件**历史商品与候选的相似度，只要有一件很像就会高。同一商品在 20 件中出现两次就计两次。读取向量时检查 R6 商品向量的范数为 1（±1e-3）。按 query 逐个计算（一个 query 的候选在宽表中连续），内存只有"该 query 的候选数 × 历史件数"的一个小矩阵；这一步的耗时单独记入 `features.json` 和 `results/m6_runs.csv`。

### 8.4 C 组：商品特征（17 个）

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
| `item_discount_ratio` **（新增）** | `[a−7d, a)` 的成交均价 ÷ `[a−364d, a)` 的成交均价（小于 1 表示上周卖得比全年便宜）；任一窗口没有销售为 NaN |
| `product_type_no` | 类目（categorical） |
| `product_group_name` | 类目（categorical） |
| `garment_group_no` | 类目（categorical） |
| `index_group_no` | 类目（categorical） |
| `section_no` | 类目（categorical） |
| `colour_group_code` | 颜色（categorical） |

落实（`src/hmrec/features/item.py`，每个 fold 对商品目录的全部商品各算一行）：

- `item_days_since_first_sale`："第一次被购买"指数据中（2018-09-20 起）的第一条购买行，所以数据开始之前就在卖的商品，这个值最多约 650 天。从没卖出过的商品，两个"距今天数"都是 NaN。
- `item_price_mean_4w`、`item_online_share_4w`：对 4w 内的购买行求平均 / 占比；4w 内没有购买行时为 NaN。
- `item_buyer_age_mean_4w`：对 4w 内的**不同买家**求平均（每个买家只算一次），只用年龄已知的买家；没有这样的买家时为 NaN。
- `item_repurchase_ratio_52w`："买过 2 次及以上"指在 **2 个或更多不同的日期**买过；同一天的多行（一次买多件）不算复购。52w 内没有人买过时为 NaN。
- **价格均值**（`item_price_mean_4w`、`item_discount_ratio` 的两个均值、`user_price_mean_52w`，以及由它们得到的 `ui_price_ratio`）：聚合前把 `price` 转成 Float64，算完再转成 float32。多线程 group_by 的求和顺序不固定，但在 float64 下只影响第 16 位左右，远小于 float32 的精度；有测试保证同一份数据打乱行序后这些特征的 float32 结果完全相同。

### 8.5 D 组：用户特征（12 个）

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
| `user_purchase_days_52w` **（新增）** | `[a−364d, a)` 内有购买的不同日期数；没有则为 0 |

落实（`src/hmrec/features/user.py`，只计算该 fold 的 query 用户）：`age` 用表中的原值（16、17 岁照常保留，不截断），未知为 NaN；`fn`、`active` 是 float32 的 0 / 1。`user_price_mean_52w`、`user_online_share_52w` 在 52w 内没有购买时为 NaN。

### 8.6 E 组：用户–商品交叉特征（21 个）

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
| `ui_decay_ptype`、`ui_decay_garment`、`ui_decay_section`、`ui_decay_colour` **（新增）** | 用户在 `t_dat < a` 的全部历史中，与候选商品同 `product_type_no` / `garment_group_no` / `section_no` / `colour_group_code` 的每一条购买行，按 `exp(−age_days / 30)` 加权求和；没有则为 0。τ 固定为 30（`configs/features.yaml` 的 `decay_tau_days`），**不是超参数** |
| `ui_days_since_ptype`、`ui_days_since_section` **（新增）** | 距用户最近一次购买同 `product_type_no` / `section_no` 商品的天数；没买过为 NaN |
| `ui_share_section_52w`、`ui_share_pgroup_52w` **（新增）** | 与 `ui_share_*_52w` 相同的定义，属性换成 `section_no` 和 `product_group_name` |
| `ui_lift_ptype_52w` **（新增）** | `ui_share_ptype_52w` ÷ 全站 52w 内该 `product_type_no` 的购买行占比；用户 52w 内没有购买，或全站占比为 0 时为 NaN |
| `ui_channel_gap` **（新增）** | `|user_online_share_52w − item_online_share_4w|`；任一侧缺失为 NaN |
| `ui_attr_match_cnt` **（新增）** | 6 个类目属性中，候选商品的取值出现在用户最近 20 件购买里的属性个数（0–6）；用户没有历史时为 NaN |

落实（`src/hmrec/features/user_item.py`）：

- `ui_cnt_pcode`、`ui_days_since_pcode` 的"同 `product_code` 商品"**包括候选商品本身**。
- `ui_share_*_52w` = 用户 52w 内与候选同类目的购买行数 / 用户 52w 内的全部购买行数。用户 52w 内没有购买时为 NaN（0 / 0）；有购买但没有同类目的为 0。
- `ui_price_ratio`：任一侧缺失、或用户均价为 0 时为 NaN。`ui_age_gap`：年龄未知、或商品 4w 内没有年龄已知的买家时为 NaN。

### 8.7 特征的冻结和检查

- 特征列表和 dtype 写在 `configs/features.yaml`（加载时逐项核对：必须正好是 §8.2–§8.6 的 71 个，顺序、分组、类别特征都不能变）。**特征已锁定**（2026-11-18，F1 的特征生成并核对之后、任何 Ranker 结果出现之前；原计划是"M6 之前"）：`configs/locks/features.lock.yaml` 记录 71 个特征的名称、顺序、dtype，8 个类别特征的清单，两个固定值（`decay_tau_days` 30、`history_similarity_top_k` 5），以及另外四项决定特征取值的设置（`history_length` 20、`rrf_k` 60、`online_sales_channel_id` 2、三个窗口长度 7 / 28 / 364）。代码仍然读 `configs/features.yaml`；`hmrec.locks.check_features_match_lock` 逐项比较两个文件，不一致就报错，`scripts/m6_build_features.py` 在生成每个 fold 之前调用它，`tests/unit/test_features_lock.py` 也调用它。`configs/features.yaml` 里只有 `embedding_pair_chunk_size` 不在锁定文件中（它只影响内存，不影响结果）。类别编码表（`artifacts/features/category_codes.json`，不进 git）也不在锁定文件中，由每个 fold 生成时核对是否与已有文件一致。V 阶段**不调特征**。
- **每个 fold 生成后自动检查**（`src/hmrec/features/build.py`）：行数等于去重后的候选数（`results/m6_candidates.csv` 的 union 行）；没有重复的 `(query_id, article_idx)`；`tt_dot` / `wals_dot` 与 R6 / R5 的 `score` 逐位相同；每个特征的缺失率、最小值、最大值、均值写进 `results/m6_features.csv`。
- **独立核对**（`scripts/checks/check_features.py`）：不使用 `features/`、`fusion/`、`asof`、`vectors` 的任何代码，用最朴素的循环对随机 1,000 行重新计算 25 个特征（每组至少 2 个，其中 6 个是新增的），与宽表比较。
- **特征文件的记录和核对**（`src/hmrec/feature_files.py`）：每个 fold 的 `features.json` 记录每个 part 文件的 sha256、行数，以及列清单。`verify_feature_files` 核对：列清单与预期一致、磁盘上的 part 文件正好是记录的那些、每个文件的 sha256 / 行数 / 列都一致。**M7 读取任何一个 fold 的特征之前必须先调用它。**
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

估算（与 §1.3 ② 一致）：每周正样本 ≈ 8.2 万 query × 3.4 label × 24%（G2 的 union recall 目标）≈ 6.7 万；r = 40 时 6.7 万 × 41 ≈ 每周 274 万行，4 周约 1,100 万行；r = 80 时 6.7 万 × 81 ≈ 每周 542 万行，4 周约 2,170 万行（候选不够时负样本全取，所以是上限）。计算过程和内存处理见 §1.3。

M4 在 D1–D4 上的实测（2026-11-17，§5.7，M5 锁定的配置）：六路 union recall 25.2%，所以正样本比上面的估算多约 5%；**49.5% 的 query 至少有 1 个正样本**（50.5% 的 query 没有任何一路命中，不进入训练）；每个 query 去重后平均 307 个候选（P50 320，P90 364；R3 改为只输出 `A(a)` 之前为 313 / 327 / 373），所以 r = 80 时绝大多数正样本都能配满负样本（一个有 2 个正样本的 query 需要 160 个负样本）。这几周的分数偏乐观（各路配置就是在这 4 周上选的），F、V、T 周以实测为准。

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
| 贯穿 | `num_trees` | 每个模型训到 **1000** 棵，在 k = 100, 200, …, 1000 棵处各评估一次，一次训练得到整条曲线（原定 2000 棵；2026-11-18 看到 V1 的测速结果之后，作为计算预算上的决定改为 1000，见表下说明）。"k 棵树时的分数"由 `ranker/predict.py` 的 `staged_raw_scores` 定义：每 100 棵算一段原始分数再按顺序累加，每棵树只算一次（见 §9.7） | 不增加 |

按 §4 的规则，用 4 周的 MAP@12 选 `(num_leaves, min_data_in_leaf, 负采样比例, num_trees)`。

**先定棵数、再比参数**（2026-11-18，运行之前写进 `configs/hpo/v_phase.yaml`，`docs/decisions.md`）：

1. 每组参数先定自己的棵数。设 `P(k)` 是 4 周合并（query-weighted）的 MAP@12，`k* = min{k : (max P − P(k)) / max P < 1%}`：与最高值相对差距小于 1% 的 k 里取最小的；差距正好 1% 不算。
2. 再按 §4.2 的多候选规则比较各组参数，每组用自己 `k*` 处的 4 周分数和合并分数。
3. 第 1 步的优先顺序：(31, 500) [默认] < (31, 100) < (63, 500) < (63, 100) < (127, 500) < (127, 100)。第 2 步：r = 40 [默认，即第 1 步的赢家，不重跑] < 60 < 80，每个 r 各自先定 k。
4. 一步的赢家如果 `k*` 等于上限（最后一个检查点；现在是 1000），曲线可能还在上升：这一步之后停下汇报，由人决定是否加树。没被选中的候选 `k*` 等于上限时只在报告里标注。

**树的上限从 2000 改为 1000**（2026-11-18，`docs/decisions.md`）。这是在**看到 V1 的测速结果之后**做的、计算预算上的决定，不是事先登记的：默认配置在 V1 上 100 → 200 棵的 MAP@12 增量是 +2.5%，200 → 300 棵是 +1.4%，每 100 棵的增量约缩小到上一次的六成；按这个速度推算，900 → 1000 棵的增量约 0.03%，远小于 1% 的"差不多"标准。推算只来自一个 fold、一组参数；上面第 4 条的上限规则保留，赢家的 `k*` 等于 1000 时照样停下。选择规则本身（`configs/hpo/v_phase.yaml`）没有改。

**不做 per-fit early stopping**：tree 数和 Two-Tower 的 epoch 一样，作为超参数在 4 周上选定后锁住。这样每次训练都能用上最新的一周数据，各周模型的训练程度也一致。

### 9.5 Baseline 和系统级 Ablation（V1–V4）

| 实验 | 必做 / 可选 | 说明 |
|---|---|---|
| **RRF baseline** | 必做 | 同一个候选集，按 `rrf_score`（k=60）排序取 Top-12；分数相同时按 `best_rank` 再按 `article_id`；不训练、不调 k |
| Full（6 路 + LambdaMART） | 必做 | 主系统 |
| −R6（去掉 Two-Tower） | 必做 | Two-Tower 成本最高，必须证明它对 MAP@12 有贡献 |
| −R1 … −R5 | 不做 | R1–R5 的价值只用 D 阶段的 leave-one-out union recall 来看 |

"去掉某一路"的做法：删掉**只由这一路召回**的候选行，删掉这一路的特征列（包括 B 组对应的稠密分数），然后按 §9.2 重新滚动训练。

**−R6 的落实**（`src/hmrec/ranker/ablation.py`，采样和打分共用同一个函数；不重新生成候选和特征）：(1) 删掉其余五路 rank 全为 NaN 的行；(2) 删掉 `tt_rank`、`tt_dot`、`tt_sim_max_hist`、`tt_sim_top5_hist`，剩 67 个特征，顺序不变；(3) 用 `fusion/rrf.py` 的同一组表达式和锁定的 `rrf_k`，由其余五个 rank 列**重新计算** `n_channels`、`best_rank`、`rrf_score`（否则这三列里仍然带着 R6 的排名）；(4) 重新采样（只采选中的 r），用选中的参数训练到选中的 `k*` 棵（训练是确定性的，前 k 棵与训到上限（1000 棵）的模型的前 k 棵相同），在 V1–V4 上评估。结果只汇报，不触发任何选择。

### 9.6 锁定

V4 结束后、T1 之前，把 Ranker 配置写进 `configs/locks/ranker.lock.yaml`。

### 9.7 落实后的约定（M7，2026-11-18）

配置在 `configs/ranker.yaml`（固定参数、三个调参项的默认值、采样设置、直方图方式的规则、测速设置）和 `configs/hpo/v_phase.yaml`（步骤、优先顺序、选择规则；在任何 Ranker 结果出现之前提交）。

**范围**：M7 只涉及 F1–F4 和 V1–V4。预测目标必须是 V 周，训练周必须是 F 或 V 周（`ranker/window.py`）；每个入口第一步检查，T（或 D）fold 在打开任何文件之前就报错。训练用的 label 只在 `ranker/sampling.py` 读，评估用的只通过 `eval/ranker_eval.py` 的 `load_m7_evaluator` 读。特征只从 `artifacts/features/` 读，读之前调用 `verify_feature_files`，列清单取自 `features.lock.yaml`。

**负采样**（§9.1）：每个 query 用 `(seed, query_id)` 单独建一个随机数发生器，把该 query 的负样本（按 `article_idx` 排序）随机排列，取前 `min(负样本数, r × 正样本数)` 个。所以一个 query 的样本只取决于 seed、query_id 和它的候选，与分块和处理顺序无关；r = 40 的样本是 r = 60 的子集，r = 60 是 r = 80 的子集。每个 (fold, r) 一个文件 `artifacts/ranker/samples/{full|minus_r6}/{fold}/r{r}.parquet`：`query_id`、`article_idx`、`label`（int8）和特征列，按 `(query_id, article_idx)` 排序，每 10 万行一个 row group；旁边的 `.json`（行数、query 数、正样本数、每个 query 的行数分布、sha256、来源 `features.json` 的 sha256）最后写入。写一次，不覆盖。M7 只采 F1–F4、V1–V3（V4 只训练预测 T 周的模型，留给 M8）。

实测的样本规模（2026-11-18，只读统计；每个 fold 有 47.9%–51.0% 的 query 至少有 1 个正样本）：

| fold | 正样本 | 有正样本的 query | r = 40 行数 | r = 60 行数 | r = 80 行数 | 配不满的 query（r = 40 / r = 80） |
|---|---|---|---|---|---|---|
| F1 | 72,385 | 41,420 | 2,951,046 | 4,313,058 | 5,559,664 | 282 / 2,908 |
| F2 | 61,660 | 36,679 | 2,515,389 | 3,686,828 | 4,768,474 | 182 / 2,208 |
| F3 | 63,370 | 37,628 | 2,585,402 | 3,787,205 | 4,896,656 | 198 / 2,261 |
| F4 | 65,210 | 39,088 | 2,662,389 | 3,901,694 | 5,052,137 | 185 / 2,281 |
| V1 | 66,559 | 39,673 | 2,716,842 | 3,982,707 | 5,151,132 | 165 / 2,406 |
| V2 | 65,586 | 38,166 | 2,676,158 | 3,916,942 | 5,054,204 | 210 / 2,597 |
| V3 | 58,920 | 34,028 | 2,397,998 | 3,496,447 | 4,502,500 | 250 / 2,271 |

**Dataset**（`ranker/dataset.py`）：每个采样文件一个 `lgb.Sequence`，一次只在内存里放一个 row group；4 个文件按时间顺序传给 `lgb.Dataset`。`query_id` 随 fold 的顺序递增，所以 group 大小就是 `query_id` 的游程长度，顺序与行顺序一致。构建后存成 binary：`artifacts/lgb_datasets/{实验}/{目标周}_r{r}/train.bin`，旁边的 `dataset.json` 记录 4 个采样文件的 sha256 和分箱参数；trial 加载前核对。LightGBM 4.7.0 的三个事实（合成数据上验证）：`Sequence` 的单行访问必须返回 float64；经 `Sequence` 构建两次，binary 文件逐字节相同，但它与"直接传 numpy 数组"的分箱不同，所以所有 Ranker 都走这条路径；`feature_pre_filter` 必须关掉，同一个 binary 才能被不同的 `min_data_in_leaf` 复用。

**参数**（§9.3 之外的全部写在 `configs/ranker.yaml`，标明哪些是 LightGBM 4.7.0 的默认值）：`seed` 42、`deterministic` true；`lambdarank_truncation_level` 30、`lambdarank_norm` true、`bagging_by_query` false（都是默认值，不在第一轮网格里）；8 个类别特征按整数编码传入，编码 0（缺失）是一个普通类别。**直方图方式**（col-wise / row-wise）两者可能给出不同的树，必须在网格之前定死，M8 沿用：规则在测速之前写进 `configs/ranker.yaml`（`histogram`：两种方式各训练 50 棵、不评估；row-wise 每棵树快 15% 以上且真实峰值在 8 GB 以内才选 row-wise，否则 col-wise），`mode` 在测速之后填入；`mode` 为 `undecided` 时网格拒绝运行。**测速的结果（2026-11-18）是 row-wise**：每棵树 0.368 秒，col-wise 0.682 秒，快 46%；row-wise 进程的真实峰值 1.99 GB。`mode: row_wise` 已写进配置。

**trial**：一个 trial = 一组参数在一个 V 周上的一次训练加评估，用一个独立进程（`scripts/m7_train_eval.py`），所以进程的 `ru_maxrss` 就是这个 trial 自己的真实峰值。训练从头开始，不做 early stopping；训练集 NDCG@12 每 100 棵记一次（只用于日志）。评估对预测周的全部候选打分（不采样），逐个 part 文件处理。

**k 棵树时的分数**（`ranker/predict.py` 的 `staged_raw_scores`，M7 和 M8 共用）：`predict(start_iteration=s, num_iteration=100, raw_score=True)` 每 100 棵算一段，用 float64 按顺序累加。它与 `predict(num_iteration=k)` 不是逐位相同（浮点加法的顺序不同，合成数据上最大相差 1.4e-14），所以以这个函数为准。每个检查点按 `query_id ASC, score DESC, article_idx ASC` 取 Top-12。

**记录**（都在 `results/`）：`v_phase.csv` 每个 trial 一行（配置、fold、训练周、行数、训练和打分耗时、采样峰值 `sampled_peak_rss_gb`、真实峰值 `process_max_rss_gb`、git commit）；`v_phase_metrics.csv` 每个 (配置, fold, k) 一行（query 数、MAP@12、Recall@12、NDCG@12、训练集 NDCG@12）；`v_phase_steps.csv` 每一步每个配置的 `k*`、分数、是否通过、选中谁和原因。V 阶段用新的记录类（`hpo/v_records.py`），D 阶段的 `TrialRecord` 和 `results/d_phase.csv` 不动。**选择只从 `v_phase_metrics.csv` 计算**：合并分数 = `Σ(n_f × mean_f) / Σ n_f`（`math.fsum`），与"全部 query 一起平均"只差浮点舍入（约 1e-16），好处是选择可以只凭 git 里的文件重放。每个 trial 另外保存每个检查点的 Top-12 和每个 query 的指标（`artifacts/predictions/`），`scripts/checks/check_ranker_eval.py` 用朴素循环从 Top-12 重算指标。

**测速**（`scripts/m7_speed_test.py`，网格之前只做一次）：采样 F1–F4，构建 V1 / r = 40 的 Dataset；两种直方图方式各训练 50 棵（不评估）并按规则选定方式；默认配置训练 300 棵，在 V1 上评估 100、200、300 棵，同时计算 V1 的 RRF baseline；汇报每棵树的训练耗时、打分耗时、真实峰值、训练数据的检查（各 fold 的正样本比例、每个 query 的行数分布、训练集 NDCG@12 是否上升）。**如果 300 棵时 MAP@12 低于 RRF baseline，停下排查，不外推时间。**结果写 `results/m7_speed_test.csv` 和 `results/m7_speed_test_metrics.csv`，不进 `v_phase*.csv`。

测速的结果（2026-11-18，HEAD `25cacf2`，工作区干净；LightGBM 4.7.0，libomp 22.1.4）：

| V1（82,802 个 query） | MAP@12 | Recall@12 | NDCG@12 | 训练集 NDCG@12 |
|---|---|---|---|---|
| RRF baseline | 0.018511 | 0.048789 | 0.031607 | — |
| 默认配置 (31, 500, r = 40)，100 棵 | 0.028597 | 0.064804 | 0.044900 | 0.336789 |
| 200 棵 | 0.029305 | 0.066029 | 0.045978 | 0.349325 |
| 300 棵 | 0.029719 | 0.066720 | 0.046538 | 0.357690 |

300 棵时 MAP@12 比 RRF baseline 高 61%，没有触发"停下排查"。训练集 NDCG@12 逐次上升；它与 V1 的 NDCG 不能直接比（训练集只有含正样本的 query，每个正样本只配 40 个负样本）。训练数据的检查：F1–F4 的正样本比例 2.449%–2.453%（略高于 1/41，因为少数 query 配不满），每个 query 的行数平均 68.1–71.2、P50 41、P90 123。耗时和内存见 §1.3。这是一个 fold、一组参数、300 棵的结果，只用来确认流程和估算时间，不是模型表现的结论。

**网格里的自动核对**（`hpo/v_phase.py` 的 `check_default_against_speed_test`）：网格训练完默认配置在 V1 上的模型之后，立刻比较它的前 300 棵与测速的模型（两个模型文件先各自核对 sha256，再比较前 300 棵树的文本）；不一致就停下，不再运行后面的 trial。原因：训练是确定性的，两者必须相同；不同说明环境（LightGBM、libomp）、Dataset 或参数变了。

**进程里不能同时有 torch 和 LightGBM**：torch 自带一份 libomp，LightGBM 用的是 Homebrew 的 libomp（`/opt/homebrew/Cellar/libomp/22.1.4`，不由 `uv.lock` 锁定），两份在一个进程里会直接退出、卡死或段错误。M7 的代码不 import torch（`git_state` 为此从 `hpo/runner.py` 移到 `src/hmrec/git_state.py`），测试分两个进程运行（`tests/conftest.py`，AGENTS.md §7）。libomp 是 `uv.lock` 之外的系统依赖（`brew install libomp`，见 README）：每个 M7 进程的结果行都记录 `lightgbm_version`、`libomp_path`、`libomp_version`（`ranker/runtime.py`，取自本进程实际加载的库）；升级 libomp 之后要重新做一次"两次训练得到相同模型"的核对。

---

## 10. T 阶段（T1–T4）

- 6 路 recall：按锁定配置，每周用 `t_dat < anchor` 的数据重新计算或 fine-tune，然后输出候选。
- Ranker：按 §9.2 每周用最近 4 周重训，配置不变。
- 报告每周和 4 周合并的结果：**MAP@12（主指标）**、Recall@12/20、NDCG@12/20、union recall、零命中 query 比例；同时报告 RRF baseline。
- 对照 G1–G3 给出是否达标。

---

## 11. 实施顺序（Milestones）

内存预算是 12 GB（§1.3），**重任务一次只跑一个**。

| # | 内容 | 产出 | 完成标准 | 状态 |
|---|---|---|---|---|
| M0 | 按 §14 建好仓库骨架、`AGENTS.md`、lint / test 配置；数据清洗与压缩（ID 转 int32 等）；`fold_manifest`；年龄桶；活跃池规则；`selection_rule.yaml`；Two-Tower 200 步 benchmark；内存实测 | 处理好的 Parquet、configs | 校验脚本通过；规则在看到任何分数之前提交 | **完成** |
| M1 | 实现评估模块（Recall / MAP / NDCG / union recall）和 R1–R5 | 代码 + 单元测试 | 时间截断测试、tie-break 测试、手算小例子全部通过 | **完成** |
| M2 | R1–R5 在 D1–D4 上做 ablation + HPO | `results/d_phase.csv` | 每路配置锁定 | **完成**（选择在 M2 完成，锁定在 M5 统一进行） |
| M3 | Two-Tower：数据准备 → base → D1–D4 训练链 | checkpoints、M4、结果表 | `twotower.lock.yaml` | **完成**（锁定文件在 M5 写入） |
| M4 | D 阶段 union 检查（§5.7） | union recall 表 | **G2 达标**，否则回到 M2 / M3 | **完成**（4 周合并 25.24%，G2 达标；R3 按预先登记的规则改为只输出 `A(a)`） |
| M5 | **冻结全部 recall 配置**（打 git tag） | `channels.lock.yaml`、`twotower.lock.yaml` | 之后不再修改 | **完成**（2026-11-17 写入锁定文件；锁定后在 D1 上重新生成候选、与记录逐位核对的结果，以及 tag，见 `docs/decisions.md`） |
| M6 | 逐周生成 F1–T4 的候选和宽表特征（Two-Tower 沿训练链逐周推进） | `artifacts/candidates/`、`artifacts/features/` | 每周通过时间截断测试 | 12 周的候选（2026-11-17）和特征（2026-11-18）都已生成并通过检查；特征已锁定（`features.lock.yaml`，tag `lock-features-v1`）。待用户确认 |
| M7 | Ranker 在 V1–V4 上滚动 HPO + baseline + 系统 ablation | `results/v_phase.csv` | `ranker.lock.yaml`；**G1、G3 在 V 上达标** | 代码和测试已写（2026-11-18，§9.7）；选择规则已写进 `configs/hpo/v_phase.yaml`。测速已完成（2026-11-18，§9.7）：选定 row-wise，默认配置 300 棵在 V1 上 MAP@12 0.0297（RRF baseline 0.0185）。树的上限改为 1000。**第 1 步已完成**（2026-11-18，24 个 trial，结果在 `results/v_phase*.csv`，记录见 `docs/decisions.md`）：按规则保留默认配置（31 叶、`min_data_in_leaf` 500、r = 40），k = 500，V1–V4 合并 MAP@12 **0.032713**；五个候选在各自的 k 上都低于默认（−0.21% 到 −0.56%）。RRF baseline 合并 0.019419（四周 0.018511 / 0.019064 / 0.020604 / 0.019662）。**G1 目前达标**，但只高出 0.032 约 2.2%，而且 V 上的分数偏乐观（k 和参数就是在这四周上选的），所以 T 上是否达标现在不能确定；**G3 目前达标**（4 周都高于 baseline）。下一步：第 2 步（r = 60、80，各 4 周）；选中的不是 80 且赢家的 k 不是 1000 时，继续做去掉 R6 的 ablation 和 feature importance。`ranker.lock.yaml` 还没有写 |
| M8 | T1–T4 最终评估 | 最终报告 | 报告 G1–G3 | |

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
8. **D3、D4 疑似促销周**：D3、D4（2020-06-17 至 06-30）的买家数（即 query 数）比 16 周平均多 40% 以上（D3 +46.9%，D4 +41.8%），每个 query 的 label 数也更多（4.0–4.1，其他周 3.1–3.6），疑似促销周（未核实）。在 D 阶段的 query-weighted 合并中，D3、D4 约占 **60%** 权重（236,182 / 391,681 = 60.3%），所以 D 阶段选出的 recall 配置会偏向促销周的购买模式。处理方式：不改选择规则，只在 D 阶段报告中增加 D1–D2 / D3–D4 分组诊断（见 §4.1）。

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
│   ├── catalog.yaml             # 活跃池 A(a)：window_days=28、min_purchase_lines=3（§2.4，冻结，不调）
│   ├── queries.yaml             # query_id_fold_offset（§2.3 query_id 的编号规则）
│   ├── m4_union_check.yaml      # M4：union recall 目标、R3 活跃池规则的阈值（预先登记）、人工选择
│   ├── m6_candidates.yaml       # M6：监控和停止条件（前 4 周对比、fine-tune loss 上限）；不影响任何候选
│   ├── recall/                  # 每路的默认配置
│   │   ├── r1_popularity.yaml
│   │   ├── r2_repurchase.yaml
│   │   ├── r3_sibling.yaml
│   │   ├── r4_itemcf.yaml
│   │   ├── r5_wals.yaml
│   │   └── r6_twotower.yaml
│   ├── hpo/                     # D 阶段和 V 阶段的搜索网格
│   │   ├── d_phase.yaml
│   │   └── v_phase.yaml         # M7：Ranker 的步骤、优先顺序、选择规则（看分数之前提交）
│   ├── features.yaml            # 宽表的 71 个特征、dtype、哪些是类别特征，以及窗口、衰减 τ 等设置（§8）
│   ├── ranker.yaml              # LightGBM 固定参数（§9.3）、调参项的默认值、采样设置、直方图方式的规则、测速设置
│   └── locks/                   # 锁定后的配置，只能新增，不能修改
│       ├── channels.lock.yaml   # M5：R1–R5 的完整配置、名额、trial_id、D 阶段分数（只作记录）、选择依据
│       ├── twotower.lock.yaml   # M5：R6 的完整配置、报警用户设置、两个模型文件的路径和 sha256
│       ├── features.lock.yaml   # M6：71 个特征的名称、顺序、dtype，类别特征清单，两个固定值和四项设置
│       └── ranker.lock.yaml
│
├── src/hmrec/                   # 所有可复用逻辑都在这里
│   ├── __init__.py
│   ├── config.py                # 读 YAML → 带类型的 dataclass
│   ├── paths.py                 # 所有文件路径都从这里拿，不在别处拼路径
│   ├── logging_utils.py         # 统一的 logger 设置
│   ├── memory.py                # 内存监控：分阶段的采样峰值（下限）和整个进程的真实峰值（ru_maxrss）
│   ├── threads.py               # 在 import polars 之前按 base.yaml 设置 POLARS_MAX_THREADS（只依赖标准库和 yaml）
│   ├── vectors.py               # 用户 / 商品向量（WALS、Two-Tower 共用）：按 (customer_idx, article_idx) 批量点积、保存 / 读取
│   ├── locks.py                 # 读取并校验 configs/locks/ 的锁定文件；文件 sha256 核对（底层，recall / features 都可以用）
│   ├── csv_records.py           # 只追加、表头固定的结果表（results/*.csv）的读写
│   ├── feature_files.py         # 每个 fold 特征文件的记录（features.json：各 part 的 sha256、行数、列清单）和核对；M7 读特征前调用
│   ├── git_state.py             # HEAD commit 和工作区是否干净（结果记录里的 git_commit / git_dirty）；不依赖 torch
│   ├── run_records.py           # results/m6_runs.csv：每个进程一行（耗时、采样峰值、真实峰值）；真实峰值超限时停下
│   ├── pipeline.py              # M6：一个 fold 的各步各用一个进程依次运行；每个 fold 的一行摘要
│   │
│   ├── data/
│   │   ├── preprocess.py        # 原始 CSV → Parquet；ID 映射成 int32；压缩 dtype
│   │   ├── id_maps.py           # customer_id / article_id 与整数索引互转
│   │   ├── day_index.py         # 日期 ↔ t_dat_day（距 day_zero 的天数）换算的唯一实现
│   │   ├── metadata.py          # 读写 data/processed/metadata.json；scan_processed_parquet 先校验再读表
│   │   ├── folds.py             # Fold dataclass；读取并校验 fold_manifest
│   │   ├── asof.py              # 唯一做日期过滤的地方：history 窗口和 label 周（防泄漏的唯一入口）
│   │   ├── histories.py         # "用户在某天之前最近 N 件购买"的唯一实现；Two-Tower 和特征（img / txt_sim_hist）共用
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
│   │   ├── vector_scoring.py    # R5 / R6 共用：按向量点积取 Top-quota，score 用 vectors.dot_pairs
│   │   ├── build.py             # 从原始配置构建 R1–R5（分块大小取自 base.yaml）；HPO 和锁定入口共用
│   │   ├── locked.py            # M5 之后构建六路的唯一入口：只读 configs/locks/；R6 加载模型前核对 sha256；锁定配置的训练链
│   │   ├── inputs.py            # 一个 fold 的 channel 输入（表、活跃池、query 名单）；D 阶段和 M6 共用，不读 label
│   │   ├── chain_records.py     # results/m6_twotower_chain.csv；强制训练链按周推进（上一周的模型存在且 sha256 一致）
│   │   ├── generate.py          # M6：为一个 fold 生成一路候选，保存特征要用的文件，做一致性检查，写盘（不覆盖）
│   │   └── twotower/
│   │       ├── data.py          # 训练样本、用户历史数组 [N, 20]、商品特征表（末尾 padding 行）、在设备上取 batch
│   │       ├── benchmark.py     # §1.4 的 200 步计时（占位模型；也用于真实模型的设备选择）
│   │       ├── model.py         # ItemTower、UserTower
│   │       ├── loss.py          # multi-positive、duplicate-aware 的 in-batch softmax + LogQ
│   │       ├── train.py         # base 训练、每周 fine-tune
│   │       ├── infer.py         # 生成用户 / 商品向量，取 Top-100
│   │       └── chain.py         # §6 的训练链：先预测、再 fine-tune
│   │
│   ├── fusion/
│   │   ├── union.py             # 6 路合并去重：每个 (query, article) 一行，各路的 rank / score 成列
│   │   └── rrf.py               # rrf_score、n_channels、best_rank；RRF baseline 排序
│   │
│   ├── features/                # 不 import recall/，不读 label；只读 artifacts/ 里的文件
│   │   ├── base.py              # FoldData；按窗口取购买行（经 asof）并加 age_days；缺失值规则
│   │   ├── categories.py        # 类别特征的整数编码表（所有 fold 共用，保存并核对）
│   │   ├── source.py            # A 组：召回来源特征
│   │   ├── dense.py             # B 组：tt_dot、wals_dot、pop_bucket_score、img / txt_sim_hist
│   │   ├── item.py              # C 组：每个 fold 一张商品表
│   │   ├── user.py              # D 组：每个 fold 一张 query 用户表
│   │   ├── user_item.py         # E 组：用户-商品 / product_code / 类目的聚合小表，再 join 到候选
│   │   ├── stats.py             # 每个特征的缺失率、最小值、最大值、均值（results/m6_features.csv）
│   │   └── build.py             # 一个 fold：读入、按 query 分块合并和计算、自动检查、写 part 文件
│   │
│   ├── ranker/                  # 不 import recall/、features/，不 import torch；可以 import fusion/
│   │   ├── window.py            # 4 周训练窗口；M7 的 fold 检查（目标 V，训练 F / V，T 直接报错）
│   │   ├── sampling.py          # query 内随机负采样（读 label 的两处之一）；采样文件和记录
│   │   ├── ablation.py          # 实验 full / minus_r6：去掉 R6 的行和特征，重算三个聚合特征
│   │   ├── dataset.py           # lgb.Sequence 流式读取，构建 LightGBM Dataset，存成 binary 并核对
│   │   ├── train.py             # LightGBM 参数、直方图方式的规则、确定性训练、模型文件
│   │   ├── runtime.py           # 本进程实际用到的 LightGBM 版本和 libomp 的路径、版本；两份 OpenMP 时报错
│   │   └── predict.py           # "k 棵树时的分数"（分段累加）、Top-12 和 tie-break
│   │
│   ├── eval/
│   │   ├── metrics.py           # Recall@K、AP@12、NDCG@K、union recall
│   │   ├── ranker_eval.py       # M7：只给 V 周的 label；Top-12 表的 MAP / Recall / NDCG
│   │   ├── monitor.py           # M6 监控：候选数（不需要 label）；F、V 周的 recall / union recall；拒绝读 T 周的 label
│   │   └── report.py            # 汇总每周 / 4 周结果，写 results/*.csv
│   │
│   └── hpo/
│       ├── grid.py              # 读取搜索网格，按优先顺序生成每一步的候选（§4.2、§4.3）
│       ├── selection.py         # 实现 §4 的选择规则（含多候选规则、R5 收敛检查）
│       ├── records.py           # 一个 trial 的结果；results/d_phase.csv 的读写（可断点续跑）
│       ├── runner.py            # 逐个 trial 在每个 fold 上运行、记录，并逐步应用选择规则
│       ├── d_phase.py           # D 阶段：加载 D1–D4，按 channel 运行，写结果和每一步的报告
│       ├── robustness.py        # 一步中排名靠前的配置再跑额外的 seed（只汇报，不参与选择）
│       ├── union_precheck.py    # 重放规则得到各路的选择；R1–R5 的 union 预检查
│       ├── union_check.py       # M4：六路 union 检查（只用命中的候选计算）、R3 活跃池规则
│       ├── lock_check.py        # M5：只用锁定文件重新生成一个 fold 的候选，分数必须与记录逐位相同
│       ├── v_records.py         # M7：V 阶段的三张结果表（trial、指标、步骤）；合并分数
│       ├── v_trial.py           # M7：一个 trial（训练 + 全量打分 + 评估）；Dataset 的计划；RRF baseline
│       ├── v_phase.py           # M7：读 v_phase.yaml；先定 k 再比参数；逐步启动各进程（可续跑）；−R6、importance
│       └── speed_test.py        # M7：网格之前的测速；直方图方式的选择；低于 RRF baseline 时停下
│
├── scripts/                     # 命令行入口，只做"读参数 → 调用 src → 写结果"
│   ├── m0_preprocess.py
│   ├── m0_build_queries.py      # 每个 fold 的 query 名单和 label → artifacts/queries/、artifacts/labels/
│   ├── m0_check_folds.py
│   ├── m0_benchmark_twotower.py # --device mps / cpu：占位模型 200 步，取 batch 与计算分开计时（§1.4）
│   ├── m2_recall_hpo.py         # --channel r1 ... r5
│   ├── m3_build_twotower_data.py   # base 窗口的训练数组 → artifacts/twotower/data/base/
│   ├── m3_twotower_base.py
│   ├── m3_twotower_chain.py
│   ├── m4_union_check.py
│   ├── m5_verify_locks.py          # --channel r1_popularity ...：锁定文件在 D1 上重现记录的分数（每路一个进程）
│   ├── m6_generate_candidates.py   # --fold F1 [--channel ...]：用锁定配置生成一个 fold 的候选（不读 label）
│   ├── m6_monitor_candidates.py    # --fold F1：监控一个 fold 的候选（F、V 周读 label，T 周不读）
│   ├── m6_run_folds.py             # --first-fold F1 --last-fold T4：按时间顺序逐个 fold 运行，每步一个进程
│   ├── m6_build_features.py        # --fold F1：生成一个 fold 的宽表特征（一个 fold 一个进程，不读 label）
│   ├── m7_sample.py                # --fold F1：一个训练周的负采样（r = 40 / 60 / 80 各一个文件）
│   ├── m7_build_dataset.py         # --target V1 --ratio 40：一个目标周的 binary Dataset
│   ├── m7_train_eval.py            # 一个 trial：一组参数在一个 V 周上训练加评估（一个 trial 一个进程）
│   ├── m7_rrf_baseline.py          # --fold V1：RRF baseline
│   ├── m7_speed_test.py            # 网格之前的测速（V1，默认配置 300 棵）
│   ├── m7_ranker_hpo.py            # --stage baseline / grid / minus_r6 / importance / report
│   └── m8_final_eval.py
│
├── tests/
│   ├── conftest.py              # 测试分两个进程：torch 一侧和 LightGBM 一侧（--lightgbm-side）
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
│   ├── queries/{fold}.parquet   # query_id, fold, customer_idx, anchor（M0 生成，生成后冻结）
│   ├── labels/{fold}.parquet    # query_id, fold, article_idx（只有 eval/ 和 ranker/sampling.py 能读）
│   ├── candidates/{fold}/{channel}.parquet        # channel = r1_popularity ... r6_twotower；生成后冻结，不覆盖
│   ├── candidates/{fold}/r1_bucket_scores.parquet # R1 完整的桶内分数表 (age_bucket, article_idx, score)，特征用
│   ├── features/{fold}/part-*.parquet             # 宽表：query_id, article_idx + 71 个特征；每块 query 一个文件
│   ├── features/{fold}/features.json              # 最后写入，表示这个 fold 完成：各 part 的 sha256 和行数、列清单、检查的行数
│   ├── features/category_codes.json               # 类别特征的编码表，所有 fold 共用
│   ├── ranker/samples/{experiment}/{fold}/r{r}.parquet   # 负采样后的训练行（+ 同名 .json 记录）；experiment = full / minus_r6
│   ├── lgb_datasets/{experiment}/{目标周}_r{r}/           # train.bin（LightGBM binary dataset）和 dataset.json
│   ├── models/wals/{fold}/                        # R5 向量（vectors.py 的格式）：该 fold 的 query 用户，全部有向量的商品
│   ├── models/twotower/vectors/{fold}/            # R6 向量：该 fold 的 query 用户，商品目录的全部商品；fine-tune 之前的模型
│   ├── models/twotower/chain_locked/{fold}.pt     # 锁定配置的训练链：在该 fold 上 fine-tune 之后的模型（预测下一个 fold）
│   ├── models/twotower/{base,chain}/              # base 的各个 epoch；D 阶段各条训练链的模型
│   ├── models/ranker/{area}/{experiment}/{config_id}/{fold}/model.txt   # area = v_phase / speed_test
│   ├── predictions/{area}/{experiment}/{config_id}/{fold}.parquet       # 每个检查点的 Top-12；另有 {fold}_per_query.parquet
│   ├── hpo/D/{channel}/{trial_id}/{fold}.parquet  # R2 / R3 候选 + in_active_pool 标记（M4 的 §5.7 第 4 项用）
│   ├── m5/lock_check/{fold}/{channel}.parquet     # M5 锁定核对生成的候选；后续步骤不读取这个目录
│   └── runs/{run_id}/           # 每次运行的 config 快照、日志、指标
│
└── results/                     # 进 git：小的汇总表
    ├── d_phase.csv
    ├── d_phase_steps/{channel}.csv   # 每一步的全部候选、是否通过、各周 / 合并 / 分组分数、选择结果
    ├── m5_lock_check.csv             # M5 锁定核对：每次运行一行（两个分数、是否一致、耗时、采样峰值和真实峰值）
    ├── m6_runs.csv                   # M6：每个进程一行（fold、channel 或 monitor、耗时、采样峰值、真实峰值）
    ├── m6_candidates.csv             # M6 监控：每个 fold 每一路和六路 union 各一行（T 周没有基于 label 的列）
    ├── m6_twotower_chain.csv         # M6：训练链每个模型的 sha256 和 fine-tune 的行数、loss、耗时
    ├── m6_features.csv               # M6：每个 fold 每个特征的缺失率、最小值、最大值、均值
    ├── v_phase.csv                   # M7：每个 trial 一行（耗时、采样峰值、真实峰值、LightGBM 版本、libomp 路径和版本）
    ├── v_phase_metrics.csv           # M7：每个 (配置, fold, 棵数) 一行；RRF baseline 的行也在这里
    ├── v_phase_steps.csv             # M7：每一步每个配置的 k、分数、是否通过、选择结果
    ├── v_phase_feature_importance.csv  # M7：选中配置的 gain（只作分析）
    ├── m7_runs.csv                   # M7：采样、构建 Dataset、baseline 各进程的耗时、真实峰值、LightGBM 版本、libomp 路径和版本
    ├── m7_speed_test.csv             # M7 测速的 trial 行（与 v_phase.csv 同列）
    ├── m7_speed_test_metrics.csv     # M7 测速的指标行
    └── t_phase.csv
```

### 14.1 分层规则

```
scripts/  →  src/hmrec/{recall, fusion, features, ranker, hpo, pipeline}  →  src/hmrec/{data, eval, vectors, locks, csv_records, run_records, feature_files, git_state}  →  config / paths
```

- 上层可以调用下层，**下层不能 import 上层**。例如 `data/` 不能 import `recall/`。
- `recall/`、`features/`、`ranker/` 之间互不 import，通过 `artifacts/` 里的文件交接。
- `fusion/` 不在上一条的限制里：`features/` 和 `ranker/` 都可以 import `fusion/`，`fusion/` 不 import 这三者。宽表、−R6 的重算和 RRF baseline 因此用同一组表达式。
- `git_state.py`（HEAD commit、工作区是否干净）属于底层，只依赖标准库和 `logging_utils.py`。
- M7 的代码（`ranker/`、`hpo/v_*.py`、`hpo/speed_test.py`、`eval/ranker_eval.py`、`scripts/m7_*.py`）不能 import torch，也不能 import 会加载 torch 的模块（`recall/`、`hpo/runner.py`）：torch 和 LightGBM 的两份 OpenMP 运行库不能在同一个进程里（§9.7）。
- `locks.py` 属于底层（只依赖 `config.py` 和标准库）：它只读取、校验 `configs/locks/` 的锁定文件。`recall/` 和 `features/` 都可以 import 它（特征需要某一路的锁定参数时从这里取，不经过 `recall/`）。从锁定文件**构建** channel 在 `recall/locked.py`。
- `csv_records.py`（只追加、表头固定的结果表的读写）和 `run_records.py`（每个进程一行的运行记录：耗时、采样峰值、真实峰值）属于底层，只依赖标准库和 `memory.py`。
- `feature_files.py`（特征文件的记录和核对）属于底层，只依赖标准库、polars 和 `locks.py`：`features/` 写记录，`ranker/` 核对，两者互不 import。
- `pipeline.py`（一个 fold 的各步各用一个独立进程依次运行；每个 fold 的摘要）属于上层，与 `recall/`、`hpo/` 同层：它可以 import `eval/` 和底层模块，`recall/`、`features/`、`ranker/` 不能 import 它。
- `vectors.py` 属于底层（只依赖 numpy 和标准库），`recall/` 和 `features/` 都可以 import 它：R5 / R6 训练后用它把向量存到 `artifacts/models/{wals,twotower}/{fold}/`，§8.3 的 `wals_dot`、`tt_dot` 用它读取并计算点积。召回分数和点积特征用同一个函数计算，所以同一个 (用户, 商品) 的 R5 `score` 与 `wals_dot` 完全相同。
- 所有按时间取历史数据的操作都必须经过 `data/asof.py`。

### 14.2 模块之间交接的数据

| 产出方 | 文件 | 主要列 | 使用方 |
|---|---|---|---|
| `data/queries.py` | `artifacts/queries/{fold}.parquet` | `query_id, fold, customer_idx, anchor` | 全部 |
| `data/queries.py` | `artifacts/labels/{fold}.parquet` | `query_id, fold, article_idx` | `eval/`、`ranker/sampling.py` |
| `recall/generate.py`（六路） | `artifacts/candidates/{fold}/{channel}.parquet`，`channel` = `r1_popularity` … `r6_twotower` | `query_id, article_idx, channel, rank, score`（R1 另有 `pop_is_fallback`）；写入前按各路的 `OutputSpec` 校验，生成后不覆盖 | `fusion/union.py`、`eval/monitor.py` |
| `recall/generate.py`（R1） | `artifacts/candidates/{fold}/r1_bucket_scores.parquet` | `age_bucket (str), article_idx (int32), score (float32)`：`[a − 7d, a)` 内每个 (年龄桶, 商品) 的 `S(g, i)`，**不受 Top-100 和活跃池限制**；R1 非补齐候选的 `score` 与它逐位相同 | `features/dense.py`（`pop_bucket_score`） |
| `recall/generate.py`（R5，通过 `vectors.py`） | `artifacts/models/wals/{fold}/`：`user_customer_idx.npy`、`user_vectors.npy`、`item_article_idx.npy`、`item_vectors.npy`、`vectors.json`（来源、fold） | 用户向量只存该 fold 的 query 用户（其中没有向量的不出现），商品向量全部保存；R5 候选的 `score` 与这些向量的点积逐位相同 | `features/dense.py`（`wals_dot`，通过 `vectors.py`） |
| `recall/generate.py`（R6，通过 `vectors.py`） | `artifacts/models/twotower/vectors/{fold}/`（同样的 5 个文件） | 用户向量：该 fold 的 query 用户；商品向量：**商品目录的全部商品**（R2 的候选可能在 `A(a)` 外）；用预测这一周时的模型计算（这一周 fine-tune **之前**）；R6 候选的 `score` 与这些向量的点积逐位相同 | `features/dense.py`（`tt_dot`，通过 `vectors.py`） |
| `recall/generate.py`（R6 训练链） | `artifacts/models/twotower/chain_locked/{fold}.pt`，sha256 记在 `results/m6_twotower_chain.csv` | 在该 fold 上 fine-tune 之后的模型；T4 没有 | 下一个 fold 的 R6（加载前核对 sha256） |
| `fusion/union.py` + `features/build.py` | `artifacts/features/{fold}/part-*.parquet`、`features.json` | `query_id, article_idx`（int32）和 §8 的 71 个特征（float32；8 个类别特征为 int16 编码），**没有 label 列**；按 `(query_id, article_idx)` 排序，同一个 query 不跨文件；`features.json` 记录各 part 的 sha256、行数和列清单 | `ranker/`（先用 `feature_files.verify_feature_files` 核对，再读取；label 在 `ranker/sampling.py` 中拼接） |
| `features/categories.py` | `artifacts/features/category_codes.json` | 每个类别字段的取值列表（位置 + 1 = 编码，0 = 缺失），所有 fold 共用 | `ranker/`（解释类别编码时） |
| `ranker/sampling.py` | `artifacts/ranker/samples/{experiment}/{fold}/r{r}.parquet`（+ `.json`） | `query_id, article_idx, label`（int8）和该实验的特征列（71 个，或 −R6 的 67 个）；只含至少有 1 个正样本的 query；按 `(query_id, article_idx)` 排序 | `ranker/dataset.py`（先核对 sha256） |
| `ranker/dataset.py` | `artifacts/lgb_datasets/{experiment}/{目标周}_r{r}/train.bin`、`dataset.json` | 4 个训练周的分箱数据、label、group；`dataset.json` 记录 4 个采样文件的 sha256 和分箱参数 | `hpo/v_trial.py`（先核对，再训练） |
| `hpo/v_trial.py`（用 `ranker/predict.py`） | `artifacts/predictions/{area}/{experiment}/{config_id}/{fold}.parquet`、`{fold}_per_query.parquet` | Top-12：`num_trees, query_id, rank, article_idx, score`；每个 query 的指标：`query_id, fold, map@12, recall@12, ndcg@12, num_trees` | `scripts/checks/check_ranker_eval.py`、分析 |

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
