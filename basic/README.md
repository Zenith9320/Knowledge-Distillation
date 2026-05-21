# Basic — Knowledge Distillation for Math Reasoning

## 工作概述

本目录实现一个知识蒸馏（Knowledge Distillation）流水线，目标是让一个较小的 **student model** 通过模仿大模型 **teacher model** 的推理过程，在数学推理任务上获得接近大模型的性能。

流水线共分为以下阶段：

| 阶段 | 说明 | 状态 |
|------|------|------|
| 0 — 数据集准备 | 从 GSM8K 和 MATH 原始数据中提取问题与答案，转为统一 JSONL 格式 | ✅ 已完成 |
| 1 — Teacher 生成 CoT | 用 teacher model 对训练集逐题生成带 CoT 的解答 | 待实现 |
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

## 后续工作

剩余阶段待实现：

| 阶段 | 说明 |
|------|------|
| Teacher 生成 CoT | 用 teacher model 对 `data/gsm8k_train.jsonl` 和 `data/math_train.jsonl` 中的每个 problem 生成带推理过程的解答 |
| 答案过滤 | 从生成的 CoT 中提取最终答案，与 ground-truth 比对，仅保留回答正确的样本作为 student 训练数据 |
| Student Fine-tune | 用过滤后的正确 CoT 数据对较小的 student model 进行监督微调（SFT） |
| 测试评估 | 在 `data/gsm8k_test.jsonl` 和 `data/math_test.jsonl` 上评估微调后的 student model 准确率，并与 teacher / 未微调 baseline 对比 |
