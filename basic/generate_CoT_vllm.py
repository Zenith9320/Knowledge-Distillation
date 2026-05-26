"""
Generate Chain-of-Thought solutions using vLLM for high-throughput batch inference.

Compared to the base generate_CoT.py (single-sample HF loop), this version achieves
10-20x higher throughput via PagedAttention, continuous batching, and larger batch sizes.

Output:  {"problem": ..., "answer": "<cleaned CoT solution>"}

Usage:
    python generate_CoT_vllm.py                           # both datasets, full
    python generate_CoT_vllm.py --dataset gsm8k --max_samples 10
    python generate_CoT_vllm.py --dataset math --max_samples 50 --batch_size 64
"""

import argparse
import json
import os

# Fall back to PyTorch native sampler when CUDA toolkit (nvcc) is unavailable
# (e.g. WSL2 without full CUDA install).  Must be set before vLLM import.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from tqdm import tqdm
from vllm import LLM, SamplingParams

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DATASET_CONFIG = {
    "gsm8k": {
        "input_file": "gsm8k_train.jsonl",
        "output_file": "gsm8k_train_cot.jsonl",
    },
    "math": {
        "input_file": "math_train.jsonl",
        "output_file": "math_train_cot.jsonl",
    },
}


def clean_deepseek_tags(text: str) -> str:
    """Remove everything before and including </think> from generated text.

    DeepSeek-R1 wraps its internal reasoning in <think>...</think> tags.
    The content after </think> is the actual Chain-of-Thought answer that
    should be kept for distillation.
    """
    idx = text.find("</think>")
    if idx != -1:
        text = text[idx + len("</think>"):]
    return text.strip()


def chunked(seq, n):
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def load_model(model_name: str, gpu_memory_utilization: float = 0.90):
    """Load the teacher model via vLLM.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    gpu_memory_utilization : float
        Fraction of GPU memory to reserve for KV cache (0.0–1.0).
        Lower this if you hit OOM during generation.
    """
    print(f"Loading model with vLLM: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=4096,
    )
    print("Model loaded.")
    return llm


def process_dataset(
    llm,
    dataset_key: str,
    max_samples: int | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.6,
    batch_size: int = 32,
):
    """CoT generation for a single dataset using vLLM batched inference."""
    config = DATASET_CONFIG[dataset_key]
    input_path = os.path.join(DATA_DIR, config["input_file"])
    output_path = os.path.join(DATA_DIR, config["output_file"])

    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    print(f"\nProcessing {dataset_key}: {len(records)} problems -> {output_path}")

    with open(output_path, "w", encoding="utf-8") as f_out:
        batches = list(chunked(records, batch_size))
        for batch in tqdm(batches, desc=dataset_key):
            prompts = [
                [
                    {"role": "system", "content": "You are a math expert. Solve the problem step by step, showing your reasoning clearly. Put your final numeric answer in \\boxed{}."},
                    {"role": "user", "content": f"Problem: {record['problem']}"},
                ]
                for record in batch
            ]

            outputs = llm.chat(prompts, sampling_params, use_tqdm=False)

            for record, output in zip(batch, outputs):
                rationale_raw = output.outputs[0].text
                rationale = clean_deepseek_tags(rationale_raw)

                answer = rationale

                out = {"problem": record["problem"], "answer": answer}
                if "type" in record:
                    out["type"] = record["type"]
                if "level" in record:
                    out["level"] = record["level"]
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"  Done: {len(records)} generated")


def main():
    parser = argparse.ArgumentParser(
        description="Chain-of-Thought generation with vLLM (high-throughput)"
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "math", "both"],
        default="both",
        help="Which dataset to process (default: both)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit to N samples per dataset (for testing)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max tokens for generation (default: 2048)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="vLLM batch size (default: 32). Increase if GPU has spare memory.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Teacher model to use (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6). Higher = more diverse.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for KV cache (default: 0.90). Lower if OOM.",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    llm = load_model(
        args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    datasets = ["gsm8k", "math"] if args.dataset == "both" else [args.dataset]

    for ds_key in datasets:
        process_dataset(
            llm,
            ds_key,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
