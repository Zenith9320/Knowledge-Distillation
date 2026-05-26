# Knowledge Distillation: Chain-of-Thought Reasoning from DeepSeek-R1 to Tiny Models

This project distills chain-of-thought (CoT) reasoning capabilities from **DeepSeek-R1-Distill-Qwen-1.5B** (teacher) into smaller student models (e.g. Qwen2.5-0.5B, TinyLlama) using GSM8K and MATH datasets.

## References

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](docs/)
- [Large Language Models Are Reasoning Teachers](docs/)

## Project Structure

```
Knowledge-Distillation/
├── README.md                          # This file
├── docs/                              # Reference papers
├── basic/
│   ├── README.md                      # Detailed documentation (pipeline, methodology, results)
│   ├── data/                          # All generated JSONL data files
│   ├── load_grade_school_math_dataset.py  # Step 0: GSM8K data preprocessing
│   ├── load_MATH_dataset.py               # Step 0: MATH data preprocessing
│   ├── generate_CoT_vllm.py               # Step 1a: Teacher CoT generation (vLLM batch)
│   ├── generate_CoT_adaptive.py           # Step 1b: Teacher CoT generation (adaptive sampling)
│   ├── generate_CoT.py                    # Step 1: [Legacy] HF native inference, reference only
│   ├── deduplicate_by_similarity.py       # Step 2a: Multi-temperature answer dedup
│   ├── filter.py                          # Step 2b: Boxed extraction + answer comparison
│   ├── filter_adaptive.py                 # Step 2b: Adaptive filter (per-problem matching)
│   ├── merge_datasets.py                  # Step 3: Merge GSM8K + MATH correct samples
│   └── finetune_student.py                # Step 4: Student SFT fine-tuning
```

## Pipeline Overview

| Step | Description | Status |
|------|-------------|--------|
| 0 — Data Preparation | Extract problems & answers from GSM8K / MATH into unified JSONL | Done |
| 1 — Teacher CoT Generation | Generate CoT reasoning with DeepSeek-R1 (vLLM batch / adaptive sampling) | Done |
| 2 — Answer Filtering | Extract `\boxed{}` answers, compare with ground truth, filter correct samples | Done |
| 3 — Dataset Merging | Merge GSM8K + MATH correct samples into shuffled SFT training set | Done |
| 4 — Student Fine-tuning | SFT student model (Qwen2.5-0.5B / TinyLlama) on CoT traces | Script ready |
| 5 — Evaluation | Evaluate student model accuracy on test splits | TODO |

### Step 0: Data Preparation

Convert GSM8K and MATH raw data into unified `{"problem": ..., "answer": ...}` JSONL format.

| Dataset | Train | Test | Source |
|---------|-------|------|--------|
| GSM8K | 7,473 | 1,319 | Local `grade-school-math` repo |
| MATH | 11,248 | 1,250 | HuggingFace `qwedsacf/competition_math` (90/10 split) |

```bash
python basic/load_grade_school_math_dataset.py
python basic/load_MATH_dataset.py
```

### Step 1: Teacher CoT Generation

Teacher model: **DeepSeek-R1-Distill-Qwen-1.5B**. Two generation methods available:

**1a. vLLM Batch Generation** (`generate_CoT_vllm.py`) — Recommended. 10-20x throughput via PagedAttention and continuous batching. Generates at two temperatures (T=0.6 and T=0.9) per problem for multi-temperature dedup.

```bash
python basic/generate_CoT_vllm.py                          # Both datasets, all samples
python basic/generate_CoT_vllm.py --dataset gsm8k --max_samples 10  # Quick test
```

**1b. Adaptive Sampling** (`generate_CoT_adaptive.py`) — Dynamically adjusts sampling per problem. Round 1: low-T (0.3) with n=3. If any sample's avg logprob passes threshold, done; otherwise Round 2+ iterates with high-T (0.9), n=2 per iter, up to 4 rounds. GSM8K result: all 7,473 problems passed Round 1 (22,419 answers total).

```bash
python basic/generate_CoT_adaptive.py --dataset gsm8k
```

### Step 2: Answer Filtering

**2a. Multi-Temperature Dedup** (`deduplicate_by_similarity.py`) — For vLLM pipeline: remove near-duplicate answers between low-T and high-T generations using Sentence-BERT cosine similarity (threshold=0.92).

| Dataset | Low-T Input | High-T Input | After Dedup | Discarded | Retention |
|---------|------------|-------------|-------------|-----------|-----------|
| GSM8K | 7,473 | 7,473 | 11,568 | 3,378 | 77.4% |
| MATH | 11,248 | 5,087 | 14,700 | 1,635 | 90.0% |

**2b. Boxed Extraction & Comparison** (`filter.py` / `filter_adaptive.py`) — Extract `\boxed{}` content from CoT answers, clean formatting, compare with ground truth. `filter_adaptive.py` matches by problem text (not line index) for multi-answer adaptive output.

| Dataset | Pipeline | Total | Correct | Accuracy | ≥1 Correct per Problem |
|---------|----------|-------|---------|----------|------------------------|
| GSM8K | Single T=0.6 | 7,473 | 6,298 | 84.28% | — |
| GSM8K | Deduped (T=0.6+0.9) | 11,568 | 9,508 | 82.19% | 89.4% |
| GSM8K | Adaptive (T=0.3, n=3) | 22,419 | 18,916 | 84.37% | 92.8% |
| MATH | Single T=0.6 | 11,248 | 5,341 | 47.48% | — |
| MATH | Deduped (T=0.6+0.9) | 14,700 | 6,753 | 45.94% | 50.3% |

### Step 3: Dataset Merging

Merge GSM8K and MATH correct samples into a single shuffled training set for SFT.

```bash
python basic/merge_datasets.py \
  basic/data/gsm8k_train_cot_dedup_correct.jsonl \
  basic/data/math_train_cot_dedup_correct.jsonl \
  -o basic/data/train_cot_correct_merged.jsonl
```

### Step 4: Student Fine-Tuning

SFT student model (default Qwen2.5-0.5B-Instruct) on merged CoT traces. Supports full fine-tuning and LoRA.

```bash
python basic/finetune_student.py                           # Full fine-tuning
python basic/finetune_student.py --lora                     # LoRA (low VRAM)
python basic/finetune_student.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --lora
```

### Step 5: Evaluation (TODO)

Evaluate fine-tuned student on GSM8K and MATH test splits. Metrics: final answer exact-match accuracy.

## Environment Setup

Use mamba (miniforge) to create an isolated environment:

```bash
# 1. Create environment
mamba create -n kd python=3.12 -y
mamba activate kd

# 2. Install PyTorch with CUDA
mamba install "pytorch=2.5.1=*cuda*" torchvision torchaudio \
  pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge --override-channels -y

# 3. Install HuggingFace ecosystem
mamba install transformers accelerate datasets tqdm -c conda-forge -y

# 4. (Optional) vLLM for batch inference acceleration
pip install vllm
```

Verify:

```bash
python -c "import torch,transformers,accelerate,tqdm,datasets; \
  print(f'PyTorch {torch.__version__}'); \
  print(f'CUDA:{torch.cuda.is_available()}'); \
  print(f'GPU:{torch.cuda.get_device_name(0)}'); \
  x=torch.randn(3,3).cuda(); print(f'MATMUL OK:{(x@x.T).shape}'); \
  print(f'transformers {transformers.__version__}'); print('OK')"
```

**Notes:**
- PyTorch must be installed from the pytorch channel with `*cuda*` build tag; conda-forge defaults to CPU-only.
- `--override-channels` prevents conda-forge from overriding with CPU builds.
- Python 3.13 is not yet recommended due to package compatibility.

## Key Design Decisions

| Aspect | Choice |
|--------|--------|
| Teacher | DeepSeek-R1-Distill-Qwen-1.5B |
| Students | Qwen2.5-0.5B-Instruct / TinyLlama-1.1B-Chat-v1.0 |
| Datasets | GSM8K (7,473 train), MATH (11,248 train) |
| Generation | vLLM PagedAttention (batch) + Adaptive sampling (per-problem) |
| Answer extraction | `\boxed{}` with `**Answer:**` + last-number fallbacks |
| Dedup | Sentence-BERT cosine similarity (threshold 0.92) |
| Distillation method | CoT trace SFT (reasoning teacher paradigm) |
| Evaluation | Exact-match accuracy on final answers |
