import argparse
import random
import sys


def main():
    parser = argparse.ArgumentParser(description="Shuffle JSONL records")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = [line.strip() for line in f if line.strip()]

    rng = random.Random(args.seed)
    rng.shuffle(records)

    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record + "\n")

    print(f"Shuffled {len(records)} records. Input: {args.input}, Output: {args.output}, Seed: {args.seed}")


if __name__ == "__main__":
    main()
