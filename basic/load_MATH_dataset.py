"""
Load the MATH dataset (qwedsacf/competition_math) and convert to JSONL format.
Only keeps problem and the final boxed answer (extracted from \boxed{...} in the solution).

The dataset only provides a "train" split, so we randomly split it into train/test.

Usage:
    python load_MATH_dataset.py

Output:
    data/math_train.jsonl  — training split
    data/math_test.jsonl   — test split
"""

import json
import os
import random

from datasets import load_dataset

TEST_RATIO = 0.1   # fraction of data to use for test set
SEED = 42


def last_boxed_only_string(string):
    """Extract the last \\boxed{...} or \\fbox{...} from a string."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def extract_boxed_content(string):
    """Extract just the content inside the last \\boxed{...} or \\fbox{...}."""
    boxed = last_boxed_only_string(string)
    if boxed is None:
        return None
    # strip \boxed{ and trailing }
    start = boxed.find("{") + 1
    end = boxed.rfind("}")
    if start <= 0 or end < 0:
        return None
    return boxed[start:end].strip()


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    # Load full train split — the dataset has no pre-defined test split
    ds = load_dataset("qwedsacf/competition_math", split="train")

    # Extract boxed answers first, then shuffle and split
    records = []
    missing = 0
    for item in ds:
        answer = extract_boxed_content(item["solution"])
        if answer is None:
            missing += 1
            continue
        records.append({
            "problem": item["problem"],
            "answer": answer,
            "type": item["type"],
            "level": item["level"],
        })

    total = len(records)
    print(f"Total valid examples: {total}")
    if missing > 0:
        print(f"  ({missing} examples skipped — no \\boxed found)")

    random.seed(SEED)
    random.shuffle(records)

    split_idx = int(total * (1 - TEST_RATIO))
    train_records = records[:split_idx]
    test_records = records[split_idx:]

    output_files = {
        "train": ("math_train.jsonl", train_records),
        "test": ("math_test.jsonl", test_records),
    }

    for split_name, (filename, data) in output_files.items():
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{split_name}] {len(data)} examples saved to {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
