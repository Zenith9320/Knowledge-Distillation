#!/usr/bin/env python3
"""
Evaluate the base (non-fine-tuned) student model on GSM8K and MATH test sets.

This serves as the baseline for measuring how much knowledge distillation improves
the student model's mathematical reasoning accuracy.

Supports both the Instruct and Base variants of Qwen2.5-0.5B.

Usage:
    # Base model (recommended for clean distillation experiments)
    python evaluate_baseline.py --dataset gsm8k
    python evaluate_baseline.py --dataset both

    # Instruct model (for comparison with existing instruct fine-tuned models)
    python evaluate_baseline.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset both

    # Quick smoke test
    python evaluate_baseline.py --max_samples 50

    # Base model without system prompt
    python evaluate_baseline.py --no_system_prompt

Outputs:
    basic/results/baseline-predictions_{dataset}.jsonl   # per-sample predictions
    basic/results/baseline-report_{dataset}.txt           # accuracy summary
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from evaluate import evaluate  # type: ignore[import]

# ---------------------------------------------------------------------------
# Baseline configuration
# ---------------------------------------------------------------------------

# Default: Qwen2.5-0.5B base (no instruction tuning).
# The base model has near-zero native math reasoning, so any distillation
# gains are genuine rather than recovery from interference.
BASE_MODEL = "Qwen/Qwen2.5-0.5B"

# System prompt matched to finetune_student_base.py DEFAULT_SYSTEM_PROMPT.
# Consistency between training and inference is critical for base models.
DEFAULT_SYSTEM_PROMPT = (
    "Solve the problem step by step. "
    "Show your reasoning clearly. "
    "Put your final answer in \\boxed{}."
)

# Output paths
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PREDICTIONS_TEMPLATE = os.path.join(RESULTS_DIR, "baseline-predictions")
REPORT_TEMPLATE = os.path.join(RESULTS_DIR, "baseline-report")

# Evaluation parameters — matched to fine-tuned model evaluation for fair comparison.
EVAL_TEMPERATURE = 0.6
EVAL_TOP_P = 0.95
MAX_NEW_TOKENS = 4096


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate base Qwen2.5-0.5B (no fine-tuning) as baseline"
    )
    parser.add_argument(
        "--model", default=BASE_MODEL,
        help=f"Model to evaluate (default: {BASE_MODEL})",
    )
    parser.add_argument(
        "--dataset", default="both", choices=["gsm8k", "math", "both"],
        help="Which dataset(s) to evaluate on (default: both)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit to first N samples per dataset (default: all)",
    )
    parser.add_argument(
        "--temperature", type=float, default=EVAL_TEMPERATURE,
        help=f"Sampling temperature (default: {EVAL_TEMPERATURE})",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=MAX_NEW_TOKENS,
        help=f"Max tokens to generate (default: {MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--system_prompt", default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for generation. Default: minimal step-by-step prompt.",
    )
    parser.add_argument(
        "--no_system_prompt", action="store_true",
        help="Omit system prompt entirely.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-sample predictions",
    )
    parser.add_argument(
        "--repetition_max_repeats", type=int, default=None,
        help="Stop generation when a substring repeats this many times "
             "consecutively (e.g. 8-10 to catch \\nmoire\\nmoire... loops).",
    )
    parser.add_argument(
        "--repetition_min_len", type=int, default=4,
        help="Minimum pattern length for repetition detection (default: 4).",
    )
    args = parser.parse_args()

    is_instruct = "Instruct" in args.model or "instruct" in args.model.lower()
    model_label = "Qwen2.5-0.5B-Instruct" if is_instruct else "Qwen2.5-0.5B (base)"
    system_prompt = "" if args.no_system_prompt else args.system_prompt

    print("=" * 60)
    print(f"Baseline Evaluation: {model_label} (NO fine-tuning)")
    print("=" * 60)
    print(f"  Datasets:      {args.dataset}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Max tokens:    {args.max_new_tokens}")
    print(f"  System prompt: {'(none)' if args.no_system_prompt else args.system_prompt[:60] + '...'}")
    if args.max_samples:
        print(f"  Max samples:   {args.max_samples}")
    if args.repetition_max_repeats:
        print(f"  Repetition stop: {args.repetition_max_repeats} repeats (min_len={args.repetition_min_len})")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = evaluate(
        model_path=args.model,
        dataset=args.dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=EVAL_TOP_P,
        clean_tags=False,
        verbose=args.verbose,
        output=PREDICTIONS_TEMPLATE + ".jsonl",
        report_path=REPORT_TEMPLATE + ".txt",
        system_prompt=system_prompt,
        repetition_max_repeats=args.repetition_max_repeats,
        repetition_min_len=args.repetition_min_len,
    )

    # Print summary for paper-ready comparison
    if results:
        print("\n" + "=" * 60)
        print("Baseline Summary (for paper Table)")
        print("=" * 60)
        for ds, r in sorted(results.items()):
            print(
                f"  {ds.upper():6s}  "
                f"correct={r['correct']}/{r['total']}  "
                f"acc={r['accuracy']:.2f}%  "
                f"valid_acc={r['valid_accuracy']:.2f}%  "
                f"empty={r['empty']}"
            )
        print()


if __name__ == "__main__":
    main()
