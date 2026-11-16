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
