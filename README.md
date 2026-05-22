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
├── basic/
│   ├── generate_CoT.py          # Two-stage CoT generation from teacher
│   ├── load_grade_school_math_dataset.py  # GSM8K data preprocessing
│   └── load_MATH_dataset.py     # MATH data preprocessing
├── data/
│   ├── gsm8k_train_stage1.jsonl # Stage 1 CoT output (rationale + answer)
│   └── gsm8k_train_stage2.jsonl # Stage 2 refined answers
```

## Pipeline Overview

### 1. CoT Trace Generation (Teacher) — Two-Stage

**Stage 1**: Prompt the teacher with `Q: <problem>. A: Let's think step by step.`  
The model generates `<rationale> Therefore, the answer is <answer>`.  
Output: `{"problem", "rationale", "answer"}`

**Stage 2**: Re-prompt with `Q: <problem>. A: Let's think step by step. <rationale>. Therefore, the answer is`  
The model regenerates just the answer given its own reasoning.

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
# 1. Stage 1: Generate CoT reasoning from teacher
python basic/generate_CoT.py --dataset gsm8k --max_samples 100

# 2. Stage 2: Regenerate answer given the rationale
python basic/generate_CoT.py --stage 2 --dataset gsm8k --max_samples 100

# 3. Fine-tune student (WIP)
# python scripts/train.py --config configs/training_config.yaml

# 4. Evaluate (WIP)
# python scripts/evaluate.py --model outputs/student_model --dataset gsm8k
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
