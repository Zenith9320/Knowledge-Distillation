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

## Environment Setup

使用 mamba (miniforge) 创建独立环境:

```bash
# 1. 创建环境 (Python 3.12，锁定 PyTorch CUDA 版本)
mamba create -n kd python=3.12 -y
mamba activate kd

# 2. 安装 PyTorch CUDA 版本 (注意：必须匹配 CUDA 构建，不能装 CPU 版本)
mamba install "pytorch=2.5.1=*cuda*" torchvision torchaudio \
  pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge --override-channels -y

# 3. 安装 HuggingFace 生态
mamba install transformers accelerate datasets tqdm -c conda-forge -y

# 4. (可选) vLLM 用于批量推理加速
pip install vllm
```

验证环境:

```bash
python -c "import torch,transformers,accelerate,tqdm,datasets;print(f'PyTorch {torch.__version__}');print(f'CUDA:{torch.cuda.is_available()}');print(f'GPU:{torch.cuda.get_device_name(0)}');x=torch.randn(3,3).cuda();print(f'MATMUL OK:{(x@x.T).shape}');print(f'transformers {transformers.__version__}');print('OK')"
```

**注意事项:**
- 必须从 pytorch channel 安装带 `*cuda*` 构建的 PyTorch，conda-forge 默认为 CPU 版本
- `--override-channels` 防止 conda-forge 的 CPU 版本覆盖
- Python 3.13 暂不推荐，部分包兼容性不稳定

## Key Design Decisions

| Aspect | Choice |
|--------|--------|
| Teacher | DeepSeek-R1-Distill-Qwen-1.5B |
| Students | Qwen-0.5B / TinyLlama |
| Datasets | GSM8K, MATH (subsets) |
| Distillation method | CoT trace SFT (reasoning teacher paradigm) |
| Evaluation | Exact-match accuracy on final answers |
