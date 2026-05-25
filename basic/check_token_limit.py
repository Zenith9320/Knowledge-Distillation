"""Check how many answers exceed MAX_TOKEN and whether truncation explains empty results.

Usage:
    python check_token_limit.py
    python check_token_limit.py --input data/math_train_cot.jsonl --max_tokens 4096
    python check_token_limit.py --input data/gsm8k_train_cot.jsonl --max_tokens 1500
"""

import argparse
import json
import os
from transformers import AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def check_token_limit(input_path, max_tokens):
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(line.strip()) for line in f if line.strip()]

    over_limit = 0
    under_limit = 0
    over_limit_ids = []

    for i, record in enumerate(records):
        answer = record.get("answer", "")
        n_tokens = len(tokenizer.encode(answer))

        if n_tokens >= max_tokens:
            over_limit += 1
            over_limit_ids.append(i)
        else:
            under_limit += 1

    total = len(records)
    print(f"\nFile: {input_path}")
    print(f"Total entries: {total}")
    print(f"MAX_TOKEN = {max_tokens}")
    print(f"")
    print(f"  Under limit: {under_limit}  ({under_limit / total * 100:.1f}%)")
    print(f"  At/over limit: {over_limit}   ({over_limit / total * 100:.1f}%)")

    if over_limit_ids:
        print(f"\nEntries at/over token limit (first 10): {over_limit_ids[:10]}")
        if len(over_limit_ids) > 10:
            print(f"  ... and {len(over_limit_ids) - 10} more")

    return over_limit_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check how many answers have token count >= MAX_TOKEN"
    )
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(DATA_DIR, "math_train_cot.jsonl"),
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--max_tokens", "-m",
        type=int,
        default=4096,
        help="Token limit to check against (default: 4096)",
    )
    args = parser.parse_args()
    check_token_limit(args.input, args.max_tokens)
