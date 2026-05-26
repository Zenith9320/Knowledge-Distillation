# Basic — Knowledge Distillation for Math Reasoning

## 工作概述

本目录实现一个知识蒸馏（Knowledge Distillation）流水线，目标是让一个较小的 **student model** 通过模仿大模型 **teacher model** 的推理过程，在数学推理任务上获得接近大模型的性能。

流水线共分为以下阶段：

| 阶段 | 说明 | 状态 |
|------|------|------|
| 0 — 数据集准备 | 从 GSM8K 和 MATH 原始数据中提取问题与答案，转为统一 JSONL 格式 | ✅ 已完成 |
| 1 — Teacher 生成 CoT | 用 teacher model 对训练集逐题生成带 CoT 的解答 | ✅ 已完成 |
| 2 — 答案过滤 | 从 CoT 中提取最终答案，比对 ground-truth，筛除答错的样本 | ✅ 已完成 |
| 3 — 数据集合并 | 将 GSM8K 和 MATH 正确样本合并打乱，形成统一的 SFT 训练集 | ✅ 已完成 |
| 4 — Student Fine-tune | 用合并后的 CoT 数据 SFT 小模型 | 待实现 |
| 5 — 测试评估 | 在测试集上评估 student model 准确率 | 待实现 |

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

## 第二步补充：多温度答案去重

针对同一 problem 在不同 temperature（如 `temperature=0.6` 和 `temperature=0.9`）下生成的答案，部分 pair 语义高度相似——高 temperature 并未带来有意义的多样性，反而产生冗余数据。去重阶段通过 embedding + 余弦相似度识别这些近似重复的 pair，相似度过高的只保留其中一条，减少训练集冗余。

### 实现思路

| 步骤 | 说明 |
|------|------|
| 1. 加载数据 | 读取两个 JSONL 文件（同一 problem，不同 temperature 生成），逐一配对 |
| 2. 文本编码 | 用 `sentence-transformers` 的 `all-MiniLM-L6-v2` 模型将每条 answer 编码为 384 维归一化向量 |
| 3. 相似度计算 | 对每个 pair 的两个向量做点积，得到余弦相似度（向量已归一化） |
| 4. 阈值过滤 | 相似度 > 阈值 → 仅保留 file1 的答案（视为近似重复）；相似度 ≤ 阈值 → 两者都保留 |
| 5. 写入输出 | 将保留的答案逐行写入新的 JSONL 文件 |

**为什么用 embedding 而不是 TF-IDF**：TF-IDF 仅做词级匹配，无法识别同义表达（如 "Natalia sold 48 clips" vs "Natalia sold clips to 48 friends"）。Sentence-BERT 的语义 embedding 能捕获 paraphrase，去重判断更准确。

### 使用方法

```bash
# 默认阈值 0.92
python deduplicate_by_similarity.py data/gsm8k_train_cot.jsonl \
                                    data/gsm8k_train_cot_high_temp.jsonl \
                                    -o data/gsm8k_train_cot_dedup.jsonl

# 自定义阈值（更激进去重 → 设更低，更保守 → 设更高）
python deduplicate_by_similarity.py data/gsm8k_train_cot.jsonl \
                                    data/gsm8k_train_cot_high_temp.jsonl \
                                    -o data/gsm8k_train_cot_dedup.jsonl \
                                    -t 0.95

# 使用更大的嵌入模型以获得更高精度
python deduplicate_by_similarity.py data/gsm8k_train_cot.jsonl \
                                    data/gsm8k_train_cot_high_temp.jsonl \
                                    --model all-mpnet-base-v2
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `file1` | （必填） | 优先保留的 JSONL 文件（如低 temperature 版本） |
| `file2` | （必填） | 第二个 JSONL 文件 |
| `-o` / `--output` | `data/gsm8k_train_cot_dedup.jsonl` | 输出去重后的 JSONL |
| `-t` / `--threshold` | `0.92` | 余弦相似度阈值，超过即视为重复 |
| `--model` | `all-MiniLM-L6-v2` | SentenceTransformer 模型名称 |
| `--batch-size` | `64` | 编码时的 batch size |

### 输出文件

| 文件 | 内容 |
|------|------|
| `data/gsm8k_train_cot_dedup.jsonl` | 去重后的合并数据（`problem` + `answer`） |

输出始终包含 file1 的全部答案，file2 中仅保留与 file1 相似度 ≤ 阈值的答案。

### 相关文件

```
basic/
├── data/
│   ├── gsm8k_train_cot.jsonl              # 输入：低 temperature CoT（file1）
│   ├── gsm8k_train_cot_high_temp.jsonl    # 输入：高 temperature CoT（file2）
│   └── gsm8k_train_cot_dedup.jsonl        # 输出去重后数据
├── deduplicate_by_similarity.py            # 去重脚本
└── README.md
```

---

## 第二步补充（改进思路）：自适应采样生成

> **注意**：本节为改进思路，尚未实际执行。

同一 problem 并非都需要同样的采样策略——简单题低温一次即可得出正确答案，难题才需要高温多采样来探索不同的推理路径。自适应采样的目标是根据模型的生成过程自身体现的"确定程度"，按需分配采样算力。

### 实现思路

| 阶段 | 说明 |
|------|------|
| Round 1 | 全部 problem，低温（`T=0.3`）使用 `n_low=3` 生成 3 个候选答案 |
| 置信度评估 | 纯 logprob 驱动——计算每条生成序列的 per-token 平均 log 概率，若任意一条的均值 ≥ 阈值则高置信度 |
| Round 2+ | 迭代式：仅低置信度 problem，高温（`T=0.9`）每轮追加 `n_per_iter=2` 次采样，采样后重新评估，达标即退出，最多 `max_iters=4` 轮 |
| 输出合并 | 所有候选答案写入同一 JSONL，带 `temperature`、`round`、`sample_idx` 标签，后续由 filter.py 统一处理 |

**置信度模型 — 为什么用 logprob 而不是自洽性投票**：

1. **零额外开销**：vLLM 的 `SamplingParams(logprobs=1)` 在生成每个 token 的同时顺带返回其 log 概率，不增加任何计算量。
2. **与下游解耦**：`\boxed{}` 提取和答案比对是 `filter.py` 的职责，自适应采样不应越俎代庖。logprob 纯粹衡量"模型对自己输出文本的确定程度"，与答案正确性无关。
3. **更直接的信号**：低平均 logprob 意味着模型在生成时就在多个 token 选项之间摇摆，这正是"不确定性"的直接度量，比事后比对答案更早感知。

**迭代式 Round 2 的优势**：每个低置信度 problem 单独追踪置信度变化，一旦某轮追加采样后达到阈值就停止，避免对已经找到高置信度推理路径的 problem 继续浪费采样。例如 300 个低置信度 problem 可能第 1 轮就有 200 个达标，仅剩 100 个继续下一轮。

### 使用方法（预期）

```bash
# 默认参数运行
python generate_CoT_adaptive.py

# 仅测试 GSM8K，限制 20 条
python generate_CoT_adaptive.py --dataset gsm8k --max_samples 20

# 自定义迭代策略：低温保守、高温激进、最多迭代 6 轮
python generate_CoT_adaptive.py --t_low 0.2 --n_low 5 --t_high 1.0 --n_per_iter 3 --max_iters 6

# 调整 logprob 阈值（越接近 0 越严格，越负越宽松）
python generate_CoT_adaptive.py --logprob_threshold -1.5
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `both` | 选择数据集：`gsm8k`、`math` 或 `both` |
| `--max_samples` | `None`（全部） | 每个数据集最多处理 N 条 |
| `--max_tokens` | `2048` | 生成最大 token 数 |
| `--model` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Teacher model |
| `--t_low` | `0.3` | Round 1 低温采样温度 |
| `--t_high` | `0.9` | Round 2+ 迭代式高温采样温度 |
| `--n_low` | `3` | Round 1 每个 problem 的采样次数 |
| `--n_per_iter` | `2` | Round 2+ 每轮迭代每个 problem 追加的采样次数 |
| `--max_iters` | `4` | Round 2+ 最大迭代轮数 |
| `--logprob_threshold` | `-1.8` | 平均 logprob 阈值（越大越严格，如 `-1.0`；越小越宽松，如 `-3.0`） |
| `--gpu_memory_utilization` | `0.90` | KV cache 显存占用比例 |

### 输出格式

每行一条 JSON，代表一个候选答案。与 `generate_CoT_vllm.py` 的基础字段一致，额外附加采样元信息：

```json
{
  "problem": "Janet's ducks lay 16 eggs per day...",
  "answer": "**Solution:** ... \\boxed{72}",
  "temperature": 0.3,
  "round": 1,
  "sample_idx": 0
}
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `data/gsm8k_train_cot_adaptive.jsonl` | GSM8K 自适应采样结果（含所有 round 的全部候选） |
| `data/math_train_cot_adaptive.jsonl` | MATH 自适应采样结果（含所有 round 的全部候选） |

### 与原有流程的关系

```
generate_CoT_adaptive.py          → 自适应采样，输出多候选答案
    ↓
deduplicate_by_similarity.py      → embedding 去重，剔除高度相似的冗余答案
    ↓
filter.py                         → boxed 提取 + 答案比对，输出正确样本用于 SFT
```

### 相关文件

```
basic/
├── data/
│   ├── gsm8k_train_cot_adaptive.jsonl     # 输出：GSM8K 自适应采样结果
│   └── math_train_cot_adaptive.jsonl      # 输出：MATH 自适应采样结果
├── generate_CoT_adaptive.py                # 自适应采样脚本
└── README.md
```

---

## 第三步：答案过滤（已完成）

过滤阶段分为：**Boxed 提取**、**答案清洗**、**不合格答案分离**、**答案比对**和**原因分析**，均已实现。

### 3.0 生成数据中存在的问题

在开始过滤之前，首先检查了 teacher model 生成的 CoT 数据质量，发现以下问题：

| 问题 | 说明 | GSM8K | MATH |
|------|------|-------|------|
| **无 `\boxed{}` 标记** | 部分样本未按要求将最终答案放入 `\boxed{}`，而是使用完整自然语言回答或 `**Answer:**` 格式。已通过回退提取机制解决大部分情况。 | ~17.8%（1329/7473） | ~33.6%（3783/11248） |
| **答案包含单位/文本** | 部分 `\boxed{}` 内除数值外还包含 LaTeX 文本（如 `16 \\text{ hours}`、`400 \\, \\text{ml}`），直接用于数值比对需要额外清洗。 | 少量 | 少量 |
| **生成长度截断** | 部分样本因达到 `max_new_tokens` 限制（MATH: 4096, GSM8K: 2048）而截断，CoT 末尾不完整，可能缺失 `\boxed{}` 标记或导致推理不完整。通过 `check_token_limit.py` 统计。 | ~1.7%（130/7473） | ~26.8%（3018/11248） |
| **非数值答案** | MATH 答案包含代数表达式（`x^3+2x^2+x`）、LaTeX 分数（`\frac{3}{10}`）、日期（`June 20`）等非纯数值形式，需不同处理策略。 | 不适用 | ~21.3%（2395/11248） |

这些问题需要在过滤阶段逐步处理：GSM8K 和 MATH 采用不同的提取和比对策略（通过 `--math` 参数切换）。

### 3.1 Boxed 提取

从 teacher model 生成的 CoT 解答中提取 `\boxed{...}` 包裹的最终答案。

- 采用**括号匹配算法**定位 `\boxed{...}` 边界，正确处理嵌套花括号。
- 通过 `--math` 参数切换 GSM8K 和 MATH 两种处理模式（见下方）。

#### GSM8K 模式（默认）

若某条 CoT 中不存在 `\boxed{}`，则依次启用两级回退：

1. **`**Answer:**` 回退提取**（见下方）
2. **末尾数字回退**（last-resort）：若 `\boxed{}` 和 `**Answer:**` 均未提取到答案，且 CoT 以 `.` 结尾，取全文最后一次出现的数字。

#### MATH 模式（`--math`）

MATH 数据集的答案形式更多样（代数表达式、LaTeX 分数、日期等），采用更宽松的提取策略：

- **仅提取 `\boxed{}`** 内容，找不到则启用 **`**Answer:**` 回退**（不触发末尾数字回退）
- **日期识别**：检测 `**Answer:**` 后的月份名 + 数字（如 `June 20th` → `June 20`），处理 `st`/`nd`/`rd`/`th` 序数后缀
- **分数转换**：`**Answer:**` 后的分数 `4/5` 自动转为 `\frac{4}{5}` 以匹配 ground-truth LaTeX 格式
- 保留 LaTeX 命令（`\frac`、指数 `^2`、`^3` 等），不强制转为纯数值

#### `**Answer:**` 回退提取（GSM8K 模式）

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

#### 末尾数字回退（last-resort，仅 GSM8K 模式）

若 `\boxed{}` 和 `**Answer:**` 均未提取到答案，且 CoT 文本以 `.` 结尾（说明推理未在中途截断），则取全文最后一次出现的数字作为 `final_answer`。

- 典型场景：模型使用 `**Final Answer:**`、`Answer:`、`**Conclusion:**` 等变体标记，或直接用自然语言收尾但未使用 `\boxed{}`。
- 该回退同样经过 `clean_final_answer()` 清洗。
- 若文本不以 `.` 结尾（截断样本），则不触发此回退，避免取到中间计算结果。
- **MATH 模式不启用此回退**，因为 MATH 的答案不一定是数字，取末尾数字易误提取中间结果。

### 3.2 答案清洗

对提取出的 `final_answer` 进行清洗，去除单位和 LaTeX 格式。GSM8K 和 MATH 模式采用不同的清洗策略。

#### GSM8K 模式清洗

目标是使答案尽可能为纯数值，便于后续与 ground-truth 数值比对。

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

#### MATH 模式清洗

保留有意义的 LaTeX 结构和代数符号，仅移除无关格式化标记：

| 操作 | GSM8K | MATH |
|------|-------|------|
| `\text{...}` 移除 | 整体删除 | **保留内部内容**（如 `\text{42}` → `42`） |
| `\overline{...}` | 保留内部内容 | 保留内部内容 |
| `\frac` / `\dfrac` 转换 | 转为 `a/b` | **保留原样** |
| `\begin{}`/`\end{}` 移除 | 移除 | **保留** |
| 指数 `^2` / `^3` / `^4` | 删除 | **保留** |
| 时间格式标准化 | 是 | 跳过 |
| 反斜杠残余清理 | 全部移除 | 仅移除格式化命令，保留 LaTeX 命令 |

#### 使用方法

```bash
# GSM8K 模式（默认）
python filter.py -i data/gsm8k_train_cot.jsonl -o data/gsm8k_train_cot_filtered.jsonl

# MATH 模式
python filter.py --math -i data/math_train_cot.jsonl -o data/math_train_cot_filtered.jsonl
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

剩余 163 条空值主要为 CoT 截断（未以 `.` 结尾）或无任何数值的样本，以及 16 条非数值表达式。

#### MATH 处理结果

| 统计项 | 数量 |
|--------|------|
| 总样本 | 11,248 |
| 提取到 `\boxed{}`（非空） | 7,457 |
| `**Answer:**` 回退提取（非空） | 410 |
| 其中含 LaTeX 分数（`\frac`） | 91 |
| 其中含日期答案 | 3 |
| 含代数表达式等非纯数值 | 2,395 |
| `final_answer` 仍为空 | 3,381 |

MATH 空值比例（30.1%）远高于 GSM8K（2.2%），主要因为 teacher model 在 MATH 题目中更倾向于不使用 `\boxed{}` 标记答案。

#### 不合格答案自动分离

`--bad_output` / `-b` 参数可在过滤同时将 `final_answer` 为空的条目输出到独立文件：

```bash
python filter.py --math -i data/math_train_cot.jsonl -o data/math_train_cot_filtered.jsonl \
  -b data/math_train_cot_bad.jsonl
```

#### 输出格式

```json
// GSM8K
{"problem": "Natalia sold clips...", "answer": "**Solution:**...", "final_answer": "72"}

// MATH（额外包含 type 和 level）
{"problem": "Expand the product...", "answer": "To expand the product...", "final_answer": "x^3 + 2x^2 + x", "type": "Algebra", "level": "Level 3"}
```

### 3.3 不合格答案分离

将 Boxed 提取和清洗后 `final_answer` 为空的样本自动分离到独立文件，方便人工检查或后续处理。

`filter.py` 的 `--bad_output` / `-b` 参数在过滤过程中同时完成分离，无需额外脚本。

| 分类 | 条件 | GSM8K | MATH |
|------|------|-------|------|
| 空值 | `final_answer == ""` | 163 | 3,381 |

#### 使用方法

```bash
# GSM8K
python filter.py -i data/gsm8k_train_cot.jsonl -o data/gsm8k_train_cot_filtered.jsonl \
  -b data/gsm8k_train_cot_bad.jsonl

# MATH
python filter.py --math -i data/math_train_cot.jsonl -o data/math_train_cot_filtered.jsonl \
  -b data/math_train_cot_bad.jsonl
```

#### 输出

输出文件保留原始全部字段（`problem`、`answer`、`final_answer`、`type`、`level`），便于逐条审查不合格原因。

### 3.4 答案比对

将 `final_answer` 与 ground-truth 答案进行比对，统计最终正确率。GSM8K 和 MATH 采用不同的比对策略。

#### GSM8K 比对逻辑

1. 判断 `final_answer` 类别：空值 / 非数值 / 合法数值（整数、小数或分数如 `35/6`）
2. 对合法数值，去除千分位逗号后转为浮点数与 ground-truth 比对（容差 `1e-9`）
3. 空值和非数值直接计为错误

#### MATH 比对逻辑

1. 判断 `final_answer` 类别：空值 / 非空
2. 对非空答案，使用**字符串比对**（空白符归一化后），因为 MATH 答案包含 LaTeX 表达式、分数、日期等非纯数值形式
3. 空值计为错误

#### 使用方法

```bash
# gsm8k数据集
python filter.py -i data/gsm8k_train_cot.jsonl -o data/gsm8k_train_cot_filtered.jsonl \
  -g data/gsm8k_train.jsonl -r data/gsm8k_comparison_report.txt \
  -c data/gsm8k_train_cot_correct.jsonl -w data/gsm8k_train_cot_incorrect.jsonl

# math数据集
python filter.py --math -i data/math_train_cot.jsonl -o data/math_train_cot_filtered.jsonl \
  -g data/math_train.jsonl -r data/math_comparison_report.txt \
  -c data/math_train_cot_correct.jsonl -w data/math_train_cot_incorrect.jsonl
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

#### MATH 比对结果

| 统计项 | 数量 | 占比 |
|--------|------|------|
| 总样本 | 11,248 | 100% |
| `final_answer` 为空 | 3,381 | 30.1% |
| `final_answer` 非空（valid） | 7,867 | 69.9% |
| **比对正确** | **5,341** | **47.5%** |

| 准确率指标 | 值 |
|-----------|------|
| 总准确率（correct / total） | 47.48% |
| 非空答案准确率（correct / valid） | 67.89% |

MATH 总准确率显著低于 GSM8K，主要由两方面因素叠加：30.1% 的空答案率 + 非空答案中仍有 32.1% 比对错误。

#### 按题目类型与难度分类统计（MATH）

| 类型 | 总数 | 空 | 有效 | 正确 | 准确率 | 有效准确率 |
|------|------|-----|------|------|--------|------------|
| Algebra | 2,645 | 248 | 2,397 | 1,770 | 66.9% | 73.8% |
| Counting & Probability | 1,112 | 388 | 724 | 434 | 39.0% | 59.9% |
| Geometry | 1,228 | 575 | 653 | 410 | 33.4% | 62.8% |
| Intermediate Algebra | 1,951 | 976 | 975 | 575 | 29.5% | 59.0% |
| Number Theory | 1,256 | 420 | 836 | 725 | 57.7% | 86.7% |
| Prealgebra | 1,893 | 242 | 1,651 | 1,188 | 62.8% | 72.0% |
| Precalculus | 1,163 | 532 | 631 | 239 | 20.6% | 37.9% |

| 难度 | 总数 | 空 | 有效 | 正确 | 准确率 | 有效准确率 |
|------|------|-----|------|------|--------|------------|
| Level 1 | 914 | 67 | 847 | 661 | 72.3% | 78.0% |
| Level 2 | 2,038 | 253 | 1,785 | 1,256 | 61.6% | 70.4% |
| Level 3 | 2,427 | 489 | 1,938 | 1,293 | 53.3% | 66.7% |
| Level 4 | 2,603 | 783 | 1,820 | 1,200 | 46.1% | 65.9% |
| Level 5 | 3,264 | 1,789 | 1,475 | 929 | 28.5% | 63.0% |

准确率随难度递增递减，Level 5 空答案率高达 54.8%，说明难题上 teacher model 更倾向于不输出 `\boxed{}`。

#### 输出文件

| 参数 | 输出文件 | 内容 | GSM8K | MATH |
|------|----------|------|-------|------|
| `-o` | `*_filtered.jsonl` | 全部样本（追加 `final_answer` 字段） | 7,473 | 11,248 |
| `-c` | `*_correct.jsonl` | **比对正确的样本（student 训练集）** | 6,298 | 5,341 |
| `-w` | `*_incorrect.jsonl` | 非空但比对错误的样本 | 996 | 2,526 |
| `-b` | `*_bad.jsonl` | `final_answer` 为空的样本 | 163 | 3,381 |
| `-r` | `*_report.txt` | 比对统计报告 | — | — |

### 3.5 原因分析

#### GSM8K 错误分布

| 错误类别 | 数量 | 占比 | 说明 |
|----------|------|------|------|
| 合法数值但答案错误 | 996 | 13.3% | 推理过程有误，最终数值与 ground-truth 不符 |
| `final_answer` 为空 | 163 | 2.2% | 无法从 CoT 中提取到任何数值答案 |
| `final_answer` 非数值 | 16 | 0.2% | 提取到了内容但含字母/符号，非合法数值 |

#### MATH 错误分布

| 错误类别 | 数量 | 占比 | 说明 |
|----------|------|------|------|
| 非空但比对不匹配 | 2,526 | 22.5% | 提取到了答案但与 ground-truth 字符串不匹配 |
| `final_answer` 为空 | 3,381 | 30.1% | 无 `\boxed{}` 且 `**Answer:**` 回退失败 |
| 非空但比对正确 | 5,341 | 47.5% | 正确提取 |

#### 主要错误原因

**1. 题意理解偏差**

部分题目本身表述存在歧义，大模型容易产生理解偏差。典型情况：
- 题目要求计算两个量的**总数**，但模型将其理解为分别给出两个量的值，导致 `final_answer` 出现多值（如 `64 32`）
- 题目中的条件关系被模型反向解读，导致推理方向错误

此类问题既反映在 GSM8K 16 条非数值样本（多值输出）中，也是 996 条合法数值但答案错误的主要来源之一。

**2. Token 上限截断**

Teacher model 生成时 `max_new_tokens=2048`，但部分题目中模型在推理过程中出现反复验证、循环推敲（"反复横跳"），在触及 token 上限前未能完成推理并给出 `\boxed{}`，导致：
- CoT 末尾缺失 `\boxed{}` 标记
- 推理链不完整，无法通过 `**Answer:**` 回退提取
- 最终 `final_answer` 为空

GSM8K 数据集（`max_tokens=2048`）仅 **130 条（1.7%）** 达到上限，截断问题不严重。MATH 数据集（`max_tokens=4096`）则有 **3018 条（26.8%）** 达到上限。对比 MATH 空答案率 30.1%（3381 条），token 截断是空答案的主要原因——大部分截断样本无法输出完整的 `\boxed{}` 或 `**Answer:**` 标记，导致后续提取失败。剩余 ~3.3% 的空答案则源于模型不使用 `\boxed{}` 等其他原因。GSM8K 因截断比例低，空答案主要来自少数不使用 `\boxed{}` 的样本。

使用 `check_token_limit.py` 可对任意数据集进行 token 截断分析：

```bash
python check_token_limit.py --input data/math_train_cot.jsonl --max_tokens 4096
```

**3. 大模型引入未知变量**

在部分需要设未知数列方程求解的题目中，模型习惯性地使用变量（如 `x`、`C`、`J`）表示某个中间量，但最终未将变量具体数值代入 `\boxed{}`，而是将含字母的表达式直接作为答案输出。

**4. MATH 特有的格式问题**

- **不使用 `\boxed{}`**：MATH 空答案率（30.1%）远高于 GSM8K（2.2%），teacher model 在 MATH 题目中更倾向自然语言收尾
- **答案形式多样**：代数表达式、LaTeX 分数、坐标、日期等非纯数值形式，字符串比对要求精确匹配，细微格式差异即判错
- **推理能力不足**：Level 5 难题有效准确率仅 63.0%，模型在高级代数、微积分等领域的推理质量有限

### 相关文件

```
basic/
├── data/
│   ├── gsm8k_train_cot.jsonl              # 输入：Teacher 生成的 CoT（GSM8K）
│   ├── gsm8k_train_cot_filtered.jsonl     # 输出：追加 final_answer 字段（GSM8K 全部样本）
│   ├── gsm8k_train_cot_correct.jsonl      # 输出：GSM8K 比对正确的样本（6,298 条）
│   ├── gsm8k_train_cot_incorrect.jsonl    # 输出：GSM8K 合法数值但比对错误的样本
│   ├── gsm8k_train_cot_bad.jsonl          # 输出：GSM8K final_answer 为空的样本
│   ├── gsm8k_comparison_report.txt        # 输出：GSM8K 答案比对统计报告
│   ├── math_train_cot.jsonl               # 输入：Teacher 生成的 CoT（MATH）
│   ├── math_train_cot_filtered.jsonl      # 输出：追加 final_answer 字段（MATH 全部样本）
│   ├── math_train_cot_correct.jsonl       # 输出：MATH 比对正确的样本（5,341 条）
│   ├── math_train_cot_incorrect.jsonl     # 输出：MATH 非空但比对错误的样本
│   ├── math_train_cot_bad.jsonl           # 输出：MATH final_answer 为空的样本
│   ├── math_comparison_report.txt         # 输出：MATH 答案比对统计报告
│   └── math_empty_answer_problems.txt     # 输出：MATH 空答案题目详情
├── filter.py                               # Boxed 提取 + 答案清洗 + 比对 + 训练集输出脚本
├── check_token_limit.py                    # Token 截断分析：统计 answer token 数，定位超限样本
└── README.md
```

---

## 第四步：数据集合并（已完成）

将 GSM8K 和 MATH 两个数据集过滤后的正确样本合并为一个统一的训练集，打乱顺序后供 student model SFT 使用。两个数据集的字段不完全相同（MATH 比 GSM8K 多了 `type` 和 `level` 字段），合并时各自保留原有字段，缺失字段自然不存在。

### 使用方法

```bash
python merge_datasets.py data/gsm8k_train_cot_correct.jsonl \
                          data/math_train_cot_correct.jsonl \
                          -o data/train_cot_correct_merged.jsonl
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `files` | 一个或多个输入 JSONL 文件 |
| `-o` / `--output` | 输出文件路径 |
| `-s` / `--seed` | 随机种子（默认 `42`），用于可复现的 shuffle |

### 输出文件

| 文件 | GSM8K | MATH | 合计 |
|------|-------|------|------|
| `data/train_cot_correct_merged.jsonl` | 6,298 | 5,341 | 11,639 |

### 相关文件

```
basic/
├── data/
│   ├── gsm8k_train_cot_correct.jsonl        # 输入：GSM8K 正确样本
│   ├── math_train_cot_correct.jsonl         # 输入：MATH 正确样本
│   └── train_cot_correct_merged.jsonl       # 输出：合并打乱后的训练集
├── merge_datasets.py                         # 合并脚本
└── README.md
```

---

## 后续工作

| 阶段 | 说明 |
|------|------|
| Student Fine-tune | 用 `data/train_cot_correct_merged.jsonl`（11,639 条）对较小的 student model 进行监督微调（SFT） |
| 测试评估 | 在 `data/gsm8k_test.jsonl` 和 `data/math_test.jsonl` 上评估微调后的 student model 准确率，并与 teacher / 未微调 baseline 对比 |
| 准确率提升 | 探索改进 teacher model 生成质量（更好的 prompt、更大的模型、增加生成长度），以降低 MATH 30.1% 的空答案率和提升整体正确率 |
