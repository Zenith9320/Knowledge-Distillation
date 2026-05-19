# Knowledge Distillation: Chain-of-Thought Reasoning from DeepSeek-R1 to Tiny Models

This project distills chain-of-thought (CoT) reasoning capabilities from **DeepSeek-R1-Distill-Qwen-1.5B** (teacher) into smaller student models (e.g. Qwen-0.5B, TinyLlama) using GSM8K and MATH datasets.

## References

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](docs/)
- [Large Language Models Are Reasoning Teachers](docs/)

## Project Structure

```
Knowledge-Distillation/
├── README.md
├── docs/                        # Reference papers
├── grade-school-math/           # GSM8K dataset
├── math/                        # MATH dataset
├── scripts/
│   ├── generate_cot.py          # Generate CoT traces from teacher
│   ├── train.py                 # Fine-tune student model
│   └── evaluate.py              # Evaluate accuracy on test sets
├── data/
│   ├── gsm8k_cot/               # Generated CoT traces for GSM8K
│   └── math_cot/                # Generated CoT traces for MATH
├── configs/
│   └── training_config.yaml     # Hyperparameters and settings
└── test.py                      # Quick test for teacher model inference
```

## Pipeline Overview

### 1. CoT Trace Generation (Teacher)

Use DeepSeek-R1-Distill-Qwen-1.5B to generate step-by-step reasoning traces on GSM8K/MATH subsets.

### 2. Fine-Tuning (Student)

Fine-tune a small student model on the generated CoT traces using standard language modeling (or sequence-to-sequence) objectives.

- **Student candidates**: Qwen-0.5B, TinyLlama-1.1B
- **Training strategy**: SFT on CoT-augmented data

### 3. Evaluation

Evaluate the fine-tuned student on GSM8K and MATH test splits. Metrics:

- Final answer accuracy (exact match)
- Reasoning trace quality (optional manual inspection)

## Quick Start

```bash
# 1. Generate CoT data from teacher
python scripts/generate_cot.py --dataset gsm8k --output data/gsm8k_cot/

# 2. Fine-tune student
python scripts/train.py --config configs/training_config.yaml

# 3. Evaluate
python scripts/evaluate.py --model outputs/student_model --dataset gsm8k
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers
- Datasets
- PEFT (optional, for LoRA)
- vLLM (optional, for fast inference)

## Key Design Decisions

| Aspect | Choice |
|--------|--------|
| Teacher | DeepSeek-R1-Distill-Qwen-1.5B |
| Students | Qwen-0.5B / TinyLlama |
| Datasets | GSM8K, MATH (subsets) |
| Distillation method | CoT trace SFT (reasoning teacher paradigm) |
| Evaluation | Exact-match accuracy on final answers |
