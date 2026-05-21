"""
Load the Grade School Math (GSM8K) dataset and convert to JSONL format.
Drops the CoT (step-by-step solution) and keeps only the final answer
after the "####" marker.

Usage:
    python load_grade_school_math_dataset.py

Output:
    data/gsm8k_train.jsonl  — training split
    data/gsm8k_test.jsonl   — test split
"""

import json
import os

GSM8K_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "grade-school-math", "grade_school_math", "data",
)

splits = {
    "train": "train.jsonl",
    "test": "test.jsonl",
}

output_names = {
    "train": "gsm8k_train.jsonl",
    "test": "gsm8k_test.jsonl",
}


def extract_final_answer(raw_answer):
    """Extract the number after the last '####' marker, dropping all CoT."""
    idx = raw_answer.rfind("####")
    if idx < 0:
        return None
    final = raw_answer[idx + 4:].strip()
    return final


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    for split_name, src_filename in splits.items():
        src_path = os.path.join(GSM8K_DIR, src_filename)
        dst_filename = output_names[split_name]
        dst_path = os.path.join(output_dir, dst_filename)

        count = 0
        missing = 0
        with open(src_path, encoding="utf-8") as f_in, \
             open(dst_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                answer = extract_final_answer(item["answer"])
                if answer is None:
                    missing += 1
                    continue
                record = {
                    "problem": item["question"],
                    "answer": answer,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

        print(f"[{split_name}] {count} examples saved to {dst_path}")
        if missing > 0:
            print(f"  ⚠ {missing} examples skipped (no #### found)")

    print("Done.")


if __name__ == "__main__":
    main()
