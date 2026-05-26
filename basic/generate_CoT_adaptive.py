"""
Adaptive-sampling CoT generation with vLLM.

Confidence is estimated purely from per-token log-probabilities (logprobs=1
in vLLM SamplingParams — zero overhead).  No answer extraction or comparison
is done here; that belongs to filter.py downstream.

Strategy:
  Round 1 — low temperature (T_low), n_low samples per problem.
            If *any* sample's avg logprob >= threshold → high confidence, done.
  Round 2+ — iterative high-temperature (T_high) batches, n_per_iter samples
            per iteration, only for low-confidence problems.  Problems that
            reach the logprob threshold exit early.

Output:  {"problem": ..., "answer": ..., "temperature": ..., "round": ..., "sample_idx": ...}

Usage:
    python generate_CoT_adaptive.py
    python generate_CoT_adaptive.py --dataset gsm8k --max_samples 20
    python generate_CoT_adaptive.py --logprob_threshold -1.5
"""

import argparse
import json
import os
from typing import Any

# Fall back to PyTorch native sampler when CUDA toolkit (nvcc) is unavailable
# (e.g. WSL2 without full CUDA install).  Must be set before vLLM import.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DATASET_CONFIG = {
    "gsm8k": {
        "input_file": "gsm8k_train.jsonl",
        "output_file": "gsm8k_train_cot_adaptive.jsonl",
    },
    "math": {
        "input_file": "math_train.jsonl",
        "output_file": "math_train_cot_adaptive.jsonl",
    },
}


# ======================================================================
#  DeepSeek-R1 tag cleaning
# ======================================================================

def clean_deepseek_tags(text: str) -> str:
    """Remove everything before and including </think>."""
    idx = text.find("</think>")
    if idx != -1:
        text = text[idx + len("</think>"):]
    return text.strip()


# ======================================================================
#  Logprob helpers
# ======================================================================

def compute_avg_logprob(logprobs: list | None) -> float | None:
    """Average token log-probability over the whole generated sequence.

    Returns None if logprobs are unavailable.
    """
    if logprobs is None:
        return None
    values = []
    for item in logprobs:
        try:
            if isinstance(item, dict):
                lp_obj = next(iter(item.values()))
                values.append(float(lp_obj.logprob))
            elif hasattr(item, "logprob"):
                values.append(float(item.logprob))
            elif isinstance(item, (int, float)):
                values.append(float(item))
        except (StopIteration, TypeError, AttributeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


# ======================================================================
#  Confidence evaluation  (pure logprob-based)
# ======================================================================

def check_high_confidence(
    avg_logprobs: list[float],
    threshold: float,
) -> bool:
    """A problem is high-confidence if *any* accumulated sample exceeds the
    average-logprob threshold."""
    valid = [v for v in avg_logprobs if v is not None]
    if not valid:
        return False
    return max(valid) >= threshold


# ======================================================================
#  Model loading
# ======================================================================

def load_model(model_name: str, gpu_memory_utilization: float = 0.90):
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


# ======================================================================
#  Prompt construction
# ======================================================================

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are a math expert. Solve the problem step by step, "
        "showing your reasoning clearly. Put your final numeric answer in \\boxed{}."
    ),
}


def make_prompts(records: list[dict]) -> list:
    return [
        [
            SYSTEM_MSG,
            {"role": "user", "content": f"Problem: {record['problem']}"},
        ]
        for record in records
    ]


# ======================================================================
#  Main pipeline
# ======================================================================

def process_dataset(
    llm,
    dataset_key: str,
    max_samples: int | None = None,
    max_tokens: int = 2048,
    t_low: float = 0.3,
    t_high: float = 0.9,
    n_low: int = 3,
    n_per_iter: int = 2,
    max_iters: int = 4,
    logprob_threshold: float = -1.8,
):
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

    print(f"\n{'='*60}")
    print(f"Adaptive sampling: {dataset_key}  ({len(records)} problems)")
    print(f"  Round 1  — T={t_low}, n={n_low}")
    print(f"  Round 2+ — T={t_high}, n_per_iter={n_per_iter}, max_iters={max_iters}")
    print(f"  Logprob threshold: {logprob_threshold}")
    print(f"{'='*60}")

    all_outputs: list[dict] = []

    # ---- Round 1: low-temperature, every problem ----
    print("\n[Round 1] Generating with low temperature ...")
    params_r1 = SamplingParams(
        temperature=t_low, top_p=0.95, max_tokens=max_tokens,
        n=n_low, logprobs=1,
    )

    prompts = make_prompts(records)
    r1_results = llm.chat(prompts, params_r1, use_tqdm=True)

    # Per-problem accumulated average logprobs
    acc_logprobs: dict[int, list[float | None]] = {i: [] for i in range(len(records))}
    low_conf: set[int] = set()

    for idx, (record, out) in enumerate(zip(records, r1_results)):
        for sample_idx, completion in enumerate(out.outputs):
            text = clean_deepseek_tags(completion.text)
            avg_lp = compute_avg_logprob(getattr(completion, "logprobs", None))
            acc_logprobs[idx].append(avg_lp)

            all_outputs.append({
                "problem": record["problem"],
                "answer": text,
                "temperature": t_low,
                "round": 1,
                "sample_idx": sample_idx,
            })

        if not check_high_confidence(acc_logprobs[idx], logprob_threshold):
            low_conf.add(idx)

    n_high = len(records) - len(low_conf)
    print(f"  High confidence: {n_high}/{len(records)} ({n_high / len(records) * 100:.1f}%)")
    print(f"  Low confidence:  {len(low_conf)}/{len(records)} — will iterate in Round 2+")

    # ---- Iterative Round 2+: high-temperature, low-confidence only ----
    params_r2 = SamplingParams(
        temperature=t_high, top_p=0.95, max_tokens=max_tokens,
        n=n_per_iter, logprobs=1,
    )

    for it in range(1, max_iters + 1):
        if not low_conf:
            break

        print(f"\n[Round 2-{it}] {len(low_conf)} problems remaining ...")
        curr_indices = sorted(low_conf)
        curr_records = [records[i] for i in curr_indices]
        curr_prompts = make_prompts(curr_records)
        curr_results = llm.chat(curr_prompts, params_r2, use_tqdm=True)

        newly_high: set[int] = set()

        for orig_idx, out in zip(curr_indices, curr_results):
            for sample_idx, completion in enumerate(out.outputs):
                text = clean_deepseek_tags(completion.text)
                avg_lp = compute_avg_logprob(getattr(completion, "logprobs", None))
                acc_logprobs[orig_idx].append(avg_lp)

                all_outputs.append({
                    "problem": records[orig_idx]["problem"],
                    "answer": text,
                    "temperature": t_high,
                    "round": 1 + it,
                    "sample_idx": sample_idx,
                })

            if check_high_confidence(acc_logprobs[orig_idx], logprob_threshold):
                newly_high.add(orig_idx)

        low_conf -= newly_high
        print(f"  → {len(newly_high)} gained confidence, {len(low_conf)} still low")

    if low_conf:
        print(f"\n  {len(low_conf)} problems still low-confidence after {max_iters} iterations "
              f"(all accumulated samples kept)")

    # ---- Write output ----
    print(f"\nWriting {len(all_outputs)} records to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    avg_samples = len(all_outputs) / len(records)
    print(f"  Done.  Total: {len(all_outputs)} answers, "
          f"avg {avg_samples:.1f} samples/problem")
    print(f"  Output: {output_path}")


# ======================================================================
#  CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive-sampling CoT generation (logprob-based confidence)"
    )
    parser.add_argument(
        "--dataset", choices=["gsm8k", "math", "both"], default="both",
        help="Which dataset to process (default: both)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit to N samples per dataset (for testing)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2048,
        help="Max tokens for generation (default: 2048)",
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"Teacher model (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--t_low", type=float, default=0.3,
        help="Temperature for Round 1 (default: 0.3)",
    )
    parser.add_argument(
        "--t_high", type=float, default=0.9,
        help="Temperature for iterative Round 2+ (default: 0.9)",
    )
    parser.add_argument(
        "--n_low", type=int, default=3,
        help="Samples per problem in Round 1 (default: 3)",
    )
    parser.add_argument(
        "--n_per_iter", type=int, default=2,
        help="Samples per problem per iteration in Round 2+ (default: 2)",
    )
    parser.add_argument(
        "--max_iters", type=int, default=4,
        help="Max iterations for Round 2+ (default: 4)",
    )
    parser.add_argument(
        "--logprob_threshold", type=float, default=-1.8,
        help="Min average logprob to be considered high-confidence (default: -1.8). "
             "Higher (closer to 0) = stricter.",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.90,
        help="Fraction of GPU memory for KV cache (default: 0.90)",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    llm = load_model(args.model, gpu_memory_utilization=args.gpu_memory_utilization)

    datasets = ["gsm8k", "math"] if args.dataset == "both" else [args.dataset]
    for ds_key in datasets:
        process_dataset(
            llm, ds_key,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            t_low=args.t_low,
            t_high=args.t_high,
            n_low=args.n_low,
            n_per_iter=args.n_per_iter,
            max_iters=args.max_iters,
            logprob_threshold=args.logprob_threshold,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
