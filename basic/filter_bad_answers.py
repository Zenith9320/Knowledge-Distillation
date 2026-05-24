import json
import re
import argparse


def is_valid_numeric(text):
    """Return True if text is a pure number, decimal, or simple fraction."""
    if not text:
        return False
    # Pure integer or decimal, optionally negative
    if re.match(r'^-?\d+(?:\.\d+)?$', text):
        return True
    # Simple fraction like 35/6 or -3/4
    if re.match(r'^-?\d+(?:\.\d+)?/\d+(?:\.\d+)?$', text):
        return True
    return False


def filter_bad(input_path, output_path):
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        count = 0
        for line in fin:
            data = json.loads(line.strip())
            if not is_valid_numeric(data.get('final_answer', '')):
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1
    print(f"Written {count} bad entries to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Filter out entries with non-numeric or empty final_answer.'
    )
    parser.add_argument(
        '--input', '-i',
        default='data/gsm8k_train_cot_filtered.jsonl',
        help='Input jsonl file path'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/gsm8k_train_cot_bad.jsonl',
        help='Output jsonl file path'
    )
    args = parser.parse_args()
    filter_bad(args.input, args.output)
