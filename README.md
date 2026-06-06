# Knowledge Distillation: Chain-of-Thought Reasoning from DeepSeek-R1 to Tiny Models

This project distills chain-of-thought (CoT) reasoning capabilities from **DeepSeek-R1-Distill-Qwen-1.5B** (teacher) into **Qwen2.5-0.5B** (student) on GSM8K and MATH benchmarks.

详细文档见 **[basic/README.md](basic/README.md)**。

## References

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](docs/DeepSeek-R1%20Incentivizing%20Reasoning%20Capability%20in%20LLMs%20via%20Reinforcement%20Learning.pdf)
- [Large Language Models Are Reasoning Teachers](docs/Large%20Language%20Models%20Are%20Reasoning%20Teachers.pdf)

## Pipeline

| 阶段 | 说明 | 详见 |
|------|------|------|
| 数据集准备 | GSM8K / MATH → 统一 JSONL | [第一步](basic/README.md#第一步数据集加载) |
| Teacher CoT 生成 | vLLM 批量 / 单温度 / 双温度 / 自适应采样 | [第二步](basic/README.md#第二步teacher-生成-cot) |
| 答案过滤 | `\boxed{}` 提取 + 比对 + 语义去重 | [第三步](basic/README.md#第三步答案过滤) |
| 数据集构建 | 合并、去重、打乱 → SFT 训练集 | [第四步](basic/README.md#第四步数据集合并) |
| Student 微调 | Instruct → Base，SFT + label masking | [第五步](basic/README.md#第五步student-sft-微调) |
| 测试评估 | GSM8K + MATH，含 baseline 对比 | [第六步](basic/README.md#第六步测试评估) |

## Results Summary

| 模型 | 基座 | 训练样本 | GSM8K | MATH |
|------|------|----------|-------|------|
| Baseline (未微调) | Instruct | — | 40.94% | 22.72% |
| `qwen-single_temp` | Instruct | 11,639 | 38.89% | 19.76% |
| `qwen-double_temp` | Instruct | 16,261 | 39.88% | 20.24% |
| `qwen-adaptive` | Instruct | 34,024 | 41.55% | 19.04% |
| Baseline (未微调) | Base | — | 13.80% | 13.04% |
| `qwen-base-adaptive` | Base | 17,231 | **42.76%** | **23.28%** |

> 详见 [所有模型汇总](basic/README.md#55-所有模型汇总)

## Key Findings

- **蒸馏税 (Distillation Tax)**：Instruct 模型微调后性能反而不如未微调，因为 CoT 风格与原生推理风格冲突。MATH 上三个微调模型全部不如 Baseline。
- **Base 模型纯净收益**：切换到 Base 模型后，GSM8K +28.96pp（13.80% → 42.76%），MATH +10.24pp（13.04% → 23.28%），收益可干净归因于蒸馏。
- **语义去重至关重要**：adaptive 采样中 52.6% 问题有近重复样本，去重后半数样本即可达到相同覆盖率。
- **覆盖率 > 样本数**：问题覆盖率是比样本数更好的数据集质量指标。

> 详见 [关键发现](basic/README.md#关键发现)

## Project Structure

```
Knowledge-Distillation/
├── README.md                              # This file
├── docs/                                  # Reference papers
├── paper/                                 # Mini-paper (NeurIPS 2023 format)
│   ├── main.tex
│   ├── neurips_2023.sty
│   └── ref.bib
├── basic/
│   ├── README.md                          # Detailed documentation
│   ├── data/                              # All generated JSONL data files
│   ├── results/                           # Evaluation predictions & reports
│   ├── models/                            # Fine-tuned student models
│   ├── load_grade_school_math_dataset.py  # Step 0: GSM8K preprocessing
│   ├── load_MATH_dataset.py               # Step 0: MATH preprocessing
│   ├── generate_CoT_vllm.py               # Step 1: vLLM batch generation
│   ├── generate_CoT_adaptive.py           # Step 1: Adaptive sampling
│   ├── generate_CoT.py                    # Step 1: [Legacy] HF native inference
│   ├── deduplicate_by_similarity.py       # Step 2: Semantic dedup
│   ├── filter.py                          # Step 2: Boxed extraction + comparison
│   ├── filter_adaptive.py                 # Step 2: Adaptive filter
│   ├── merge_datasets.py                  # Step 3: Merge + shuffle
│   ├── shuffle_jsonl.py                   # Step 3: Standalone shuffle tool
│   ├── finetune_student.py                # Step 4: Instruct SFT
│   ├── finetune_student_base.py           # Step 4: Base SFT
│   ├── evaluate.py                        # Step 5: Evaluation
│   └── evaluate_baseline.py               # Step 5: Baseline evaluation
```

## Environment

```bash
mamba create -n kd python=3.12 -y && mamba activate kd
mamba install "pytorch=2.5.1=*cuda*" torchvision torchaudio \
  pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge --override-channels -y
mamba install transformers accelerate datasets tqdm -c conda-forge -y
pip install vllm  # optional, for batch inference
```
