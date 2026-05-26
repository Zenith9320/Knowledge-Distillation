#!/usr/bin/env python3
"""
Merge multiple JSONL files into one, with shuffled order.

Usage:
    python merge_datasets.py data/gsm8k_train_cot_correct.jsonl \
                              data/math_train_cot_correct.jsonl \
                              -o data/train_cot_correct_merged.jsonl
"""

import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser(description="Merge and shuffle JSONL files")
    parser.add_argument("files", nargs="+", help="Input JSONL files to merge")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file path")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for shuffle (default: 42)")
    args = parser.parse_args()

    records = []
    for filepath in args.files:
        print(f"Reading {filepath} ...")
        with open(filepath, encoding="utf-8") as f:
            count = 0
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                    count += 1
            print(f"  {count} records")

    print(f"Total records: {len(records)}")

    random.seed(args.seed)
    random.shuffle(records)

    print(f"Writing {len(records)} shuffled records to {args.output} ...")
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
