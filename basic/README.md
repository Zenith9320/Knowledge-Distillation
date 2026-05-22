# Basic — Knowledge Distillation for Math Reasoning

## 工作概述

本目录实现一个知识蒸馏（Knowledge Distillation）流水线，目标是让一个较小的 **student model** 通过模仿大模型 **teacher model** 的推理过程，在数学推理任务上获得接近大模型的性能。

流水线共分为以下阶段：

| 阶段 | 说明 | 状态 |
|------|------|------|
| 0 — 数据集准备 | 从 GSM8K 和 MATH 原始数据中提取问题与答案，转为统一 JSONL 格式 | ✅ 已完成 |
| 1 — Teacher 生成 CoT | 用 teacher model 对训练集逐题生成带 CoT 的解答 | ✅ 已完成 |
| 2 — 答案过滤 | 比对生成答案与 ground-truth，筛除答错的样本 | 待实现 |
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

## 后续工作

剩余阶段待实现：

| 阶段 | 说明 |
|------|------|
| 答案过滤 | 从生成的 CoT 中提取最终答案，与 ground-truth 比对，仅保留回答正确的样本作为 student 训练数据 |
| Student Fine-tune | 用过滤后的正确 CoT 数据对较小的 student model 进行监督微调（SFT） |
| 测试评估 | 在 `data/gsm8k_test.jsonl` 和 `data/math_test.jsonl` 上评估微调后的 student model 准确率，并与 teacher / 未微调 baseline 对比 |
