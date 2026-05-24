# Basic — Knowledge Distillation for Math Reasoning

## 工作概述

本目录实现一个知识蒸馏（Knowledge Distillation）流水线，目标是让一个较小的 **student model** 通过模仿大模型 **teacher model** 的推理过程，在数学推理任务上获得接近大模型的性能。

流水线共分为以下阶段：

| 阶段 | 说明 | 状态 |
|------|------|------|
| 0 — 数据集准备 | 从 GSM8K 和 MATH 原始数据中提取问题与答案，转为统一 JSONL 格式 | ✅ 已完成 |
| 1 — Teacher 生成 CoT | 用 teacher model 对训练集逐题生成带 CoT 的解答 | ✅ 已完成 |
| 2 — 答案过滤 | 从 CoT 中提取最终答案，比对 ground-truth，筛除答错的样本 | ✅ 已完成 |
| 3 — Student Fine-tune | 用过滤后的正确 CoT 数据 SFT 小模型 | 待实现 |
| 4 — 测试评估 | 在测试集上评估 student model 准确率 | 待实现 |

---

## 第一步：数据集加载（已完成）

将两个数学推理数据集统一处理为 `problem` / `answer` 格式的 JSONL 文件，供后续 Teacher 生成和评估阶段使用。

### 使用方法

```bash
# 1. 生成 GSM8K 数据集（直接从本地 clone 的 grade-school-math 仓库读取）
python load_grade_school_math_dataset.py

# 2. 生成 MATH 数据集（从 HuggingFace 自动下载）
python load_MATH_dataset.py
```

运行后会在 `data/` 目录下生成 4 个文件。

### 生成的数据文件

| 文件 | 样本数 | 来源 | 字段 |
|------|--------|------|------|
| `data/gsm8k_train.jsonl` | 7,473 | GSM8K train split | `problem`, `answer` |
| `data/gsm8k_test.jsonl` | 1,319 | GSM8K test split | `problem`, `answer` |
| `data/math_train.jsonl` | 11,248 | MATH（90% 随机划分） | `problem`, `answer`, `type`, `level` |
| `data/math_test.jsonl` | 1,250 | MATH（10% 随机划分） | `problem`, `answer`, `type`, `level` |

### 实现思路

两个脚本的核心逻辑相同：**从原始格式中读取题目，提取最终答案，写入统一 JSONL**。差异在于原始数据格式不同。

#### GSM8K (`load_grade_school_math_dataset.py`)

- **数据来源**：`../grade-school-math/grade_school_math/data/` 目录下的 `train.jsonl` 和 `test.jsonl`（本地 clone 的 [grade-school-math](https://github.com/openai/grade-school-math) 仓库）。
- **答案提取**：GSM8K 原始 `answer` 字段包含完整 CoT 推理过程，最终答案位于 `####` 标记之后。脚本通过 `rfind("####")` 定位最后一个 `####`，截取其后的数字串作为最终答案。
- **异常处理**：若某条数据找不到 `####` 标记，则跳过并计数，不会中断脚本。

#### MATH (`load_MATH_dataset.py`)

- **数据来源**：HuggingFace 数据集 `qwedsacf/competition_math`。该数据集仅有 `train` split，无预定义的 `test` split。
- **训练/测试划分**：按 9:1 比例随机划分（`random.seed(42)`），不打乱类别分布。
- **答案提取**：MATH 原始 `solution` 字段包含完整解答过程，最终答案包裹在 `\boxed{...}` 中。脚本实现了括号匹配算法来提取 `\boxed{}` 内的内容，支持 `\fbox{}` 变体及嵌套花括号。
- **额外字段**：保留了 MATH 数据集自带的 `type`（题目类别，如 algebra、geometry）和 `level`（难度等级 1-5），方便后续分类评估。

### 相关文件

```
basic/
├── data/
│   ├── gsm8k_train.jsonl      # GSM8K 训练集
│   ├── gsm8k_test.jsonl       # GSM8K 测试集
│   ├── math_train.jsonl       # MATH 训练集
│   └── math_test.jsonl        # MATH 测试集
├── load_grade_school_math_dataset.py   # GSM8K 数据预处理脚本
├── load_MATH_dataset.py                # MATH 数据预处理脚本
└── README.md                           # 本文件
```

### 输出格式

每行一条 JSON，字段说明：

```json
// GSM8K
{"problem": "Janet's ducks lay 16 eggs per day...", "answer": "18"}

// MATH（额外包含 type 和 level）
{"problem": "How many positive integers...", "answer": "48", "type": "Algebra", "level": "Level 3"}
```

---

## 第二步：Teacher 生成 CoT（已完成）

采用**单步生成**策略，用 teacher model（DeepSeek-R1-Distill-Qwen-1.5B）对训练集中每个 problem 生成完整的 CoT 解答。

### 生成流程

| 步骤 | 说明 |
|------|------|
| 生成 | 将 problem 原文发给模型，模型续写推理过程和答案 |
| 清洗 | 去除 `<think>...</think>` 标签及其之前的内容（模型内部思考），仅保留实际的 CoT 解答部分 |

DeepSeek-R1 系列模型会在输出中生成 `<think>...</think>` 包裹的内部推理过程，这部分内容对下游蒸馏没有帮助。脚本通过 `clean_deepseek_tags()` 定位 `</think>` 标记，截取其后的实际 CoT 解答内容。

最终输出格式为 `{"problem": ..., "answer": ...}` 的 JSONL。

### 使用方法

```bash
# 完整运行（两个数据集全部样本）
python generate_CoT.py

# 仅测试 GSM8K，限制 10 条
python generate_CoT.py --dataset gsm8k --max_samples 10

# 仅处理 MATH，限制 50 条，调整生成长度
python generate_CoT.py --dataset math --max_samples 50 --max_new_tokens 4096

# 使用其他 teacher model
python generate_CoT.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `both` | 选择数据集：`gsm8k`、`math` 或 `both` |
| `--max_samples` | `None`（全部） | 每个数据集最多处理 N 条，用于快速测试 |
| `--max_new_tokens` | `2048` | 生成最大 token 数 |
| `--model` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Teacher model 名称或路径 |

### 生成的输出文件

| 文件 | 字段 |
|------|------|
| `data/gsm8k_train_cot.jsonl` | `problem`, `answer` |
| `data/math_train_cot.jsonl` | `problem`, `answer`, `type`, `level` |

### 实现思路

#### Teacher Model

采用 DeepSeek-R1-Distill-Qwen-1.5B 作为 teacher model，这是一个基于 Qwen-1.5B 的推理蒸馏模型，原生支持 Chain-of-Thought 思考过程，模型规模适中（1.5B 参数），适合在单卡 GPU 甚至 CPU 环境下运行推理。默认使用 `device_map="auto"`，优先使用 GPU（CUDA），不可用时回退到 CPU。

#### Prompt 设计

Prompt 直接使用 problem 原文，通过 `tokenizer.apply_chat_template` 包装为 DeepSeek-R1 所需的 chat 格式（单条 user message）。

#### 输出清洗

DeepSeek-R1 模型的原始输出包含 `<think>...</think>` 标签包裹的内部推理过程。`clean_deepseek_tags()` 函数定位 `</think>` 标记并截取其后的实际 CoT 解答内容，丢弃模型内部思考部分。

#### 生成参数

- `temperature=0.6`：适度的随机性，鼓励模型展开多角度推理。
- `top_p=0.95`：nucleus sampling 截断低概率 token。
- `max_new_tokens=2048`：数学推理通常足够；MATH 难题可能需要 4096。

### 相关文件

```
basic/
├── data/
│   ├── gsm8k_train.jsonl          # GSM8K 训练集（输入）
│   ├── gsm8k_test.jsonl           # GSM8K 测试集
│   ├── math_train.jsonl           # MATH 训练集（输入）
│   ├── math_test.jsonl            # MATH 测试集
│   ├── gsm8k_train_cot.jsonl      # GSM8K CoT 生成结果（输出）
│   └── math_train_cot.jsonl       # MATH CoT 生成结果（输出）
├── load_grade_school_math_dataset.py   # GSM8K 数据预处理脚本
├── load_MATH_dataset.py                # MATH 数据预处理脚本
├── generate_CoT.py                     # Teacher CoT 生成脚本
└── README.md                           # 本文件
```

### 输出格式

每行一条 JSON：

```json
// GSM8K
{"problem": "Janet's ducks lay 16 eggs per day...", "answer": "Let's think step by step..."}

// MATH（额外包含 type 和 level）
{"problem": "How many positive integers...", "answer": "Let's think step by step...", "type": "Algebra", "level": "Level 3"}
```

---

## 第二步（vLLM 加速版）

vLLM 版本提供 10-20x 吞吐提升，通过 PagedAttention 和 continuous batching 替代逐条 inference。适合全量数据集跑批量生成。

### 优势

| 特性 | 原始 HF 版本 | vLLM 版本 |
|------|-------------|-----------|
| 推理模式 | 逐条生成 | 批量 continuous batching |
| KV cache 管理 | 每条独立分配 | PagedAttention 统一调度，接近零浪费 |
| 吞吐 | ~1 sample/s | ~10-20 samples/s |
| 代码复杂度 | 简单，易调试 | 同样简单，调用高层 API |

### 安装依赖

```bash
pip install vllm
```

> **注意**：vLLM 需要 CUDA 环境。如果安装遇到问题，参考 [官方文档](https://docs.vllm.ai/en/latest/getting_started/installation.html)。

### 使用方法

```bash
# 完整运行（两个数据集全部样本）
python generate_CoT_vllm.py

# 仅测试 GSM8K，限制 10 条
python generate_CoT_vllm.py --dataset gsm8k --max_samples 10

# 增大 batch size 以充分利用 GPU 显存
python generate_CoT_vllm.py --batch_size 64

# MATH 难题增加生成长度，降低显存占用
python generate_CoT_vllm.py --dataset math --max_tokens 4096 --gpu_memory_utilization 0.80

# 使用更大的 teacher model
python generate_CoT_vllm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --gpu_memory_utilization 0.85
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `both` | 选择数据集：`gsm8k`、`math` 或 `both` |
| `--max_samples` | `None`（全部） | 每个数据集最多处理 N 条 |
| `--max_tokens` | `2048` | 生成最大 token 数 |
| `--batch_size` | `32` | 每批处理样本数，GPU 显存充足时可调大 |
| `--model` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Teacher model |
| `--gpu_memory_utilization` | `0.90` | KV cache 显存占用比例，OOM 时降低 |

### 输出

输出格式和文件路径与原始版本完全一致，可直接替换使用：

| 文件 | 字段 |
|------|------|
| `data/gsm8k_train_cot.jsonl` | `problem`, `answer` |
| `data/math_train_cot.jsonl` | `problem`, `answer`, `type`, `level` |

### 相关文件

```
basic/
├── data/
│   ├── gsm8k_train.jsonl              # GSM8K 训练集（输入）
│   ├── math_train.jsonl               # MATH 训练集（输入）
│   ├── gsm8k_train_cot.jsonl          # GSM8K CoT 生成结果（输出）
│   └── math_train_cot.jsonl           # MATH CoT 生成结果（输出）
├── generate_CoT.py                    # 原始 HF 版本（逐条推理）
├── generate_CoT_vllm.py               # vLLM 加速版本（批量推理）
└── README.md
```

---

## 第三步：答案过滤（已完成）

过滤阶段分为：**Boxed 提取**、**答案清洗**、**不合格答案分离**、**答案比对**和**原因分析**，均已实现。

### 3.0 生成数据中存在的问题

在开始过滤之前，首先检查了 teacher model 生成的 CoT 数据质量，发现以下问题：

| 问题 | 说明 | 数据占比（GSM8K） |
|------|------|--------------------|
| **无 `\boxed{}` 标记** | 部分样本未按要求将最终答案放入 `\boxed{}`，而是使用完整自然语言回答（如 `**Answer:** Alexis paid $41 for the shoes.`），导致无法通过 `\boxed{}` 直接提取。已通过 `**Answer:**` 回退提取机制解决大部分情况。 | ~17.8%（1329/7473） |
| **答案包含单位/文本** | 部分 `\boxed{}` 内除数值外还包含 LaTeX 文本（如 `16 \\text{ hours}`、`400 \\, \\text{ml}`、`75\\%`），直接用于数值比对需要额外清洗。 | 少量 |
| **生成长度截断** | 部分样本因达到 `max_new_tokens=2048` 限制而截断，CoT 末尾不完整，可能缺失 `\boxed{}` 标记或导致推理不完整。 | 待统计 |

这些问题需要在过滤阶段逐步处理：当前 Boxed 提取步骤主要解决格式统一问题，后续答案比对步骤将进一步处理数值清洗和截断样本。

### 3.1 Boxed 提取

从 teacher model 生成的 CoT 解答中提取 `\boxed{...}` 包裹的最终答案。

- 采用**括号匹配算法**定位 `\boxed{...}` 边界，正确处理嵌套花括号。
- 若某条 CoT 中不存在 `\boxed{}`，则启用 `**Answer:**` **回退提取机制**（见下方）。

#### `**Answer:**` 回退提取

针对未使用 `\boxed{}` 的样本，从 `**Answer:**`（或 `**Answer**:`）标记处提取最终答案：

1. 定位 `**Answer:**` 标记，将文本分为标记前和标记后两部分。
2. 在标记后文本中找出所有数字。
3. 在标记前文本中找出所有数字，按位置倒序排列。
4. 选取**在标记前最晚出现、且同时存在于标记后数字集合中**的数字作为 `final_answer`。

例如，对于以下 CoT 结尾：

```
100,000 + 365,000 - 20,000 = 445,000 \text{ followers}
**Answer**: Denny will have 445,000 followers after 20,000 people unfollow him in a year.
```

标记后数字为 `445,000` 和 `20,000`，标记前最晚出现的交集数字为 `445,000`，因此 `final_answer` 取 `445000`（清洗后去逗号）。

- 回退提取的数字同样经过 `clean_final_answer()` 清洗（去逗号、LaTeX 符号等），与 `\boxed{}` 路径一致。
- 若 `**Answer:**` 中仅含英文数字（如 "Five"），则无法提取，`final_answer` 留空。

#### 末尾数字回退（last-resort）

若 `\boxed{}` 和 `**Answer:**` 均未提取到答案，且 CoT 文本以 `.` 结尾（说明推理未在中途截断），则取全文最后一次出现的数字作为 `final_answer`。

- 典型场景：模型使用 `**Final Answer:**`、`Answer:`、`**Conclusion:**` 等变体标记，或直接用自然语言收尾但未使用 `\boxed{}`。
- 该回退同样经过 `clean_final_answer()` 清洗。
- 若文本不以 `.` 结尾（截断样本），则不触发此回退，避免取到中间计算结果。

### 3.2 答案清洗

对提取出的 `final_answer` 进行清洗，去除单位和 LaTeX 格式，使答案尽可能为纯数值，便于后续与 ground-truth 比对。

#### 清洗内容

| 类别 | 原始示例 | 清洗后 |
|------|----------|--------|
| 单位文字 | `16 \text{ hours}` | `16` |
| 货币符号 | `\$990.00` | `990.00` |
| 百分号 | `50\%` | `50` |
| 度数 | `21^\circ` | `21` |
| 千分位逗号 | `1,\!000` | `1000` |
| LaTeX 间距 | `400 \, \text{ml}` | `400` |
| 循环小数上划线 | `2.\overline{6}` | `2.6` |
| 分数 | `\dfrac{35}{6}` | `35/6` |

实现通过 `clean_final_answer()` 函数，按顺序处理：`\text{...}` 块移除 → `\overline{...}` 提取内层 → `\dfrac` / `\frac` 转换 → 格式化符号移除 → 数字逗号移除 → 残余清理。

#### 使用方法

```bash
python filter.py -i data/gsm8k_train_cot.jsonl -o data/gsm8k_train_cot_filtered.jsonl
```

#### GSM8K 处理结果

| 统计项 | 数量 |
|--------|------|
| 总样本 | 7,473 |
| 提取到 `\boxed{}`（非空） | 6,142 |
| `**Answer:**` 回退提取（非空） | 1,108 |
| 末尾数字回退（非空） | 60 |
| 清洗后为纯数值 | 7,287 |
| 清洗后为分数（如 `35/6`） | 7 |
| 仍含非数值（表达式/多值等） | 16 |
| `final_answer` 仍为空 | 163 |

剩余 163 条空值主要为 CoT 截断（未以 `.` 结尾）或无任何数值的样本，以及 16 条非数值表达式。空值和非数值样本将在后续答案比对步骤中统一处理。

#### 输出格式

```json
{"problem": "Natalia sold clips...", "answer": "**Solution:**...", "final_answer": "72"}
```

### 3.3 不合格答案分离

将 Boxed 提取和清洗后仍然不合格的样本（`final_answer` 为空或非数值）从主数据集中分离出来，输出到独立文件，方便人工检查或后续处理。

#### 判断标准

| 分类 | 条件 | 数量 |
|------|------|------|
| 空值 | `final_answer == ""`（CoT 截断、无任何数值等） | 163 |
| 非数值 | `final_answer` 非纯数值、小数或分数（如 `3x`、`64 32`、`10 + (-6) + 4.67`） | 16 |

#### 使用方法

```bash
python filter_bad_answers.py -i data/gsm8k_train_cot_filtered.jsonl -o data/gsm8k_train_cot_bad.jsonl
```

#### 输出

输出文件保留原始全部字段（`problem`、`answer`、`final_answer`），便于逐条审查不合格原因。

### 3.4 答案比对

将 `final_answer` 与 ground-truth 答案进行比对，统计最终正确率。

#### 比对逻辑

1. 判断 `final_answer` 类别：空值 / 非数值 / 合法数值（整数、小数或分数如 `35/6`）
2. 对合法数值，去除千分位逗号后转为浮点数与 ground-truth 比对（容差 `1e-9`）
3. 空值和非数值直接计为错误

#### 使用方法

```bash
# 一步完成：Boxed 提取 + 答案清洗 + 比对统计 + 训练集输出
python filter.py -i data/gsm8k_train_cot.jsonl -o data/gsm8k_train_cot_filtered.jsonl \
  -g data/gsm8k_train.jsonl -r data/gsm8k_comparison_report.txt \
  -c data/gsm8k_train_cot_correct.jsonl -w data/gsm8k_train_cot_incorrect.jsonl
```

#### GSM8K 比对结果

| 统计项 | 数量 | 占比 |
|--------|------|------|
| 总样本 | 7,473 | 100% |
| `final_answer` 为空 | 163 | 2.2% |
| `final_answer` 非数值 | 16 | 0.2% |
| `final_answer` 合法数值 | 7,294 | 97.6% |
| **比对正确** | **6,298** | **84.3%** |

| 准确率指标 | 值 |
|-----------|------|
| 总准确率（correct / total） | 84.28% |
| 合法数值准确率（correct / valid） | 86.34% |

#### 输出文件

| 参数 | 输出文件 | 内容 | 数量 |
|------|----------|------|------|
| `-o` | `data/gsm8k_train_cot_filtered.jsonl` | 全部样本（追加 `final_answer` 字段） | 7,473 |
| `-c` | `data/gsm8k_train_cot_correct.jsonl` | **比对正确的样本（student 训练集）** | 6,298 |
| `-w` | `data/gsm8k_train_cot_incorrect.jsonl` | 合法数值但比对错误的样本 | 996 |
| `-r` | `data/gsm8k_comparison_report.txt` | 比对统计报告 | — |

### 3.5 原因分析

对 84.28% 总准确率背后的错误来源进行分析：

#### 错误分布

| 错误类别 | 数量 | 占比 | 说明 |
|----------|------|------|------|
| 合法数值但答案错误 | 996 | 13.3% | 推理过程有误，最终数值与 ground-truth 不符 |
| `final_answer` 为空 | 163 | 2.2% | 无法从 CoT 中提取到任何数值答案 |
| `final_answer` 非数值 | 16 | 0.2% | 提取到了内容但含字母/符号，非合法数值 |

#### 主要错误原因

**1. 题意理解偏差**

部分题目本身表述存在歧义，大模型容易产生理解偏差。典型情况：
- 题目要求计算两个量的**总数**，但模型将其理解为分别给出两个量的值，导致 `final_answer` 出现多值（如 `64 32`）
- 题目中的条件关系被模型反向解读，导致推理方向错误

此类问题既反映在 16 条非数值样本（多值输出）中，也是 996 条合法数值但答案错误的主要来源之一。

**2. Token 上限截断**

Teacher model 生成时 `max_new_tokens=2048`，但部分题目中模型在推理过程中出现反复验证、循环推敲（"反复横跳"），在触及 token 上限前未能完成推理并给出 `\boxed{}`，导致：
- CoT 末尾缺失 `\boxed{}` 标记
- 推理链不完整，无法通过 `**Answer:**` 回退提取
- 最终 `final_answer` 为空（163 条空值的主要来源）

**3. 大模型引入未知变量**

在部分需要设未知数列方程求解的题目中，模型习惯性地使用变量（如 `x`、`C`、`J`、`Y`、`L`、`F`）表示某个中间量，但最终未将变量具体数值代入 `\boxed{}`，而是将含字母的表达式直接作为答案输出，例如：
- `3x`、`6x - 3`（代数表达式）
- `3C + 15`、`Y + 46`、`3L + 60`（带变量的数值表达式）
- `2J`、`-7F`（变量与数值混合）

此类问题占 16 条非数值样本中的 9 条，是 `final_answer` 非数值的主要原因。

### 相关文件

```
basic/
├── data/
│   ├── gsm8k_train_cot.jsonl              # 输入：Teacher 生成的 CoT
│   ├── gsm8k_train_cot_filtered.jsonl     # 输出：追加 final_answer 字段（全部样本）
│   ├── gsm8k_train_cot_correct.jsonl      # 输出：比对正确的样本（student 训练集，6,298 条）
│   ├── gsm8k_train_cot_incorrect.jsonl    # 输出：合法数值但比对错误的样本
│   ├── gsm8k_train_cot_bad.jsonl          # 输出：final_answer 为空或非数值的样本
│   └── gsm8k_comparison_report.txt        # 输出：答案比对统计报告
├── filter.py                               # Boxed 提取 + 答案清洗 + 比对 + 训练集输出脚本
├── filter_bad_answers.py                   # 不合格答案分离脚本
└── README.md
```

---

## 后续工作

| 阶段 | 说明 |
|------|------|
| Student Fine-tune | 用 `data/gsm8k_train_cot_correct.jsonl`（6,298 条正确 CoT 数据）对较小的 student model 进行监督微调（SFT） |
| 测试评估 | 在 `data/gsm8k_test.jsonl` 和 `data/math_test.jsonl` 上评估微调后的 student model 准确率，并与 teacher / 未微调 baseline 对比 |
