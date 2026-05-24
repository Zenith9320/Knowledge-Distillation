import json
import re
import argparse


def extract_boxed(text):
    """Extract content inside \\boxed{...} with balanced brace matching."""
    pattern = r'\\boxed\{'
    match = re.search(pattern, text)
    if not match:
        return ""

    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    return text[start:i - 1] if depth == 0 else ""


def _remove_balanced_blocks(text, marker, keep_content=False):
    """Remove `marker{...}` blocks with balanced brace matching.

    Args:
        text: The text to process.
        marker: The start marker including the opening brace (e.g. '\\text{').
        keep_content: If True, keep the inner content, only remove the wrapper.
    """
    result = []
    i = 0
    marker_len = len(marker)
    while i < len(text):
        if text[i:i + marker_len] == marker:
            depth = 1
            j = i + marker_len
            while j < len(text) and depth > 0:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            if keep_content:
                result.append(text[i + marker_len:j - 1])
            i = j
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def _extract_frac(text, marker):
    """Convert \\frac{a}{b} or \\dfrac{a}{b} to a/b representation.

    Handles both single-char and braced numerator/denominator.
    """
    result = []
    i = 0
    marker_len = len(marker)
    while i < len(text):
        if text[i:i + marker_len] == marker:
            rest = text[i + marker_len:]
            # Extract numerator (in braces)
            if rest.startswith('{'):
                num_start = 1
                depth = 1
                j = num_start
                while j < len(rest) and depth > 0:
                    if rest[j] == '{':
                        depth += 1
                    elif rest[j] == '}':
                        depth -= 1
                    j += 1
                num = rest[num_start:j - 1]
                # Extract denominator
                rest2 = rest[j:]
                if rest2.startswith('{'):
                    den_start = 1
                    depth = 1
                    k = den_start
                    while k < len(rest2) and depth > 0:
                        if rest2[k] == '{':
                            depth += 1
                        elif rest2[k] == '}':
                            depth -= 1
                        k += 1
                    den = rest2[den_start:k - 1]
                    result.append(f'{num}/{den}')
                    i += marker_len + j + k
                else:
                    result.append(rest[:j])
                    i += marker_len + j
            else:
                result.append(rest[:1])
                i += marker_len + 1
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def find_numbers_in_text(text):
    """Find all number-like substrings in text, returning list of (raw_string, start_pos)."""
    pattern = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    return [(m.group(), m.start()) for m in re.finditer(pattern, text)]


def _normalize_num(num_str):
    """Remove commas from a number string for comparison."""
    return num_str.replace(',', '')


def extract_answer_fallback(text):
    """Fallback: when no \\boxed{} is found, extract the answer from **Answer:** text.

    Finds all numbers after **Answer:**, then picks the number appearing latest
    before **Answer:** that also appears in the post-answer numbers.
    """
    marker_match = re.search(r'\*\*Answer:?\*\*\s*:?\s*', text)
    if not marker_match:
        return ""

    answer_text = text[marker_match.end():]
    text_before = text[:marker_match.start()]

    answer_numbers = find_numbers_in_text(answer_text)
    if not answer_numbers:
        return ""

    before_numbers = find_numbers_in_text(text_before)
    if not before_numbers:
        return ""

    answer_set = {_normalize_num(num) for num, _ in answer_numbers}

    # Pick the latest number before **Answer:** that also appears in the answer
    for raw_num, _pos in reversed(before_numbers):
        if _normalize_num(raw_num) in answer_set:
            return raw_num

    return ""


def clean_final_answer(text):
    """Remove units, formatting, and LaTeX artifacts from extracted boxed content.

    Steps (in order):
    1. Remove \\text{...} blocks (units like "hours", "minutes", "pounds")
    2. Remove \\overline{...} — keep inner content (repeating decimal notation)
    3. Convert \\dfrac{a}{b} and \\frac{a}{b} to a/b
    4. Remove \\,, \\!, \\$, \\%, ^{\\circ}, \\circ, $, % (formatting)
    5. Remove commas between digits (1,000 → 1000)
    6. Collapse whitespace
    """
    if not text:
        return text

    # 1. Remove \text{...} blocks (units: hours, minutes, pounds, etc.)
    text = _remove_balanced_blocks(text, r'\text{')

    # 2. Remove \overline{...} — keep the inner number
    text = _remove_balanced_blocks(text, r'\overline{', keep_content=True)

    # 3. Convert \dfrac and \frac
    text = _extract_frac(text, r'\dfrac')
    text = _extract_frac(text, r'\frac')

    # 4. Remove \begin{aligned}...\end{aligned} (failed extractions)
    text = _remove_balanced_blocks(text, r'\begin{')
    text = _remove_balanced_blocks(text, r'\end{')

    # 5. Remove LaTeX formatting
    text = text.replace('\\,', '')       # thin space
    text = text.replace('\\!', '')       # negative thin space
    text = text.replace('\\$', '')       # escaped dollar
    text = text.replace('$', '')         # dollar
    text = text.replace('\\%', '')       # escaped percent
    text = text.replace('%', '')         # percent
    text = re.sub(r'\^\{?\\circ\}?', '', text)  # degree symbol ^\circ or ^{\circ}
    text = text.replace('\\circ', '')    # bare \circ
    text = text.replace('\\\\', '')      # line breaks from aligned env
    text = text.replace('£', '')         # pound sterling
    text = text.replace('€', '')         # euro
    text = re.sub(r'\^[234]', '', text)  # superscript ^2, ^3, ^4
    text = text.replace('²', '').replace('³', '')  # unicode superscripts

    # 6. Remove commas between digits (1,000 → 1000)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)

    # 7. Remove remaining backslashes
    text = text.replace('\\', '')

    # 8. Collapse multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()

    # 9. Normalize time format: "07:30" → "730", "9:00" → "900"
    m = re.match(r'^0?(\d{1,2}):(\d{2})$', text)
    if m:
        text = f"{int(m.group(1))}{m.group(2)}"

    return text


def extract_last_number_fallback(text):
    """Last-resort fallback: if text ends with '.', return the last number in it."""
    text = text.strip()
    if not text.endswith('.'):
        return ""
    numbers = find_numbers_in_text(text)
    if not numbers:
        return ""
    return numbers[-1][0]


def filter_data(input_path, output_path):
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            data = json.loads(line.strip())
            raw_boxed = extract_boxed(data['answer'])
            if not raw_boxed:
                raw_boxed = extract_answer_fallback(data['answer'])
            if not raw_boxed:
                raw_boxed = extract_last_number_fallback(data['answer'])
            data['final_answer'] = clean_final_answer(raw_boxed)
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Processed {input_path} -> {output_path}")


def is_valid_numeric(text):
    """Return True if text is a pure number, decimal, or simple fraction."""
    if not text:
        return False
    if re.match(r'^-?\d+(?:\.\d+)?$', text):
        return True
    if re.match(r'^-?\d+(?:\.\d+)?/\d+(?:\.\d+)?$', text):
        return True
    return False


def normalize_number(text):
    """Convert a numeric string to a float value for comparison.

    Handles integers, decimals, fractions, and comma-separated numbers.
    Returns None if conversion fails.
    """
    text = text.strip().replace(',', '')
    if '/' in text:
        parts = text.split('/')
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
    try:
        return float(text)
    except ValueError:
        return None


def compare_with_ground_truth(filtered_path, ground_truth_path, report_path,
                              incorrect_output_path=None,
                              correct_output_path=None):
    """Compare final_answer from filtered file with ground truth answers.

    Reads two JSONL files line-by-line (must be same order), compares each
    final_answer against the ground truth answer, and writes a statistics
    report.  Optionally writes correct / incorrect entries to separate files.
    """
    total = 0
    empty = 0
    non_numeric = 0
    valid = 0
    correct = 0
    incorrect_entries = []
    correct_entries = []

    with open(filtered_path, 'r') as f_filt, \
         open(ground_truth_path, 'r') as f_gt:
        for line_f, line_g in zip(f_filt, f_gt):
            total += 1
            data_f = json.loads(line_f.strip())
            data_g = json.loads(line_g.strip())

            fa = data_f.get('final_answer', '').strip()
            gt = data_g.get('answer', '').strip()

            if not fa:
                empty += 1
                continue

            if not is_valid_numeric(fa):
                non_numeric += 1
                continue

            valid += 1

            fa_val = normalize_number(fa)
            gt_val = normalize_number(gt)

            if fa_val is not None and gt_val is not None:
                if abs(fa_val - gt_val) < 1e-9:
                    correct += 1
                    correct_entries.append({
                        'problem': data_f['problem'],
                        'answer': data_f['answer'],
                        'final_answer': fa,
                        'ground_truth': gt
                    })
                else:
                    incorrect_entries.append({
                        'problem': data_f['problem'],
                        'teacher_answer': data_f['answer'],
                        'final_answer': fa,
                        'ground_truth': gt
                    })

    incorrect_count = valid - correct
    accuracy = correct / total if total > 0 else 0.0
    valid_accuracy = correct / valid if valid > 0 else 0.0

    report = f"""GSM8K Final Answer Comparison Report
================================

Total samples:               {total}
  - Empty final_answer:      {empty}   ({empty/total*100:.1f}%)
  - Non-numeric final_answer:{non_numeric}    ({non_numeric/total*100:.1f}%)
  - Valid numeric:           {valid}   ({valid/total*100:.1f}%)
    - Correct:               {correct}   ({correct/total*100:.1f}%)
    - Incorrect:             {incorrect_count}    ({incorrect_count/total*100:.1f}%)

Accuracy (correct / total):          {accuracy*100:.2f}%
Accuracy (correct / valid numeric):  {valid_accuracy*100:.2f}%
"""

    with open(report_path, 'w') as f:
        f.write(report)

    print(report)
    print(f"Report written to {report_path}")

    if correct_output_path and correct_entries:
        with open(correct_output_path, 'w') as f:
            for entry in correct_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Correct entries written to {correct_output_path} "
              f"({len(correct_entries)} entries)")

    if incorrect_output_path and incorrect_entries:
        with open(incorrect_output_path, 'w') as f:
            for entry in incorrect_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Incorrect entries written to {incorrect_output_path} "
              f"({len(incorrect_entries)} entries)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract \\boxed{} content from answers as final_answer field.'
    )
    parser.add_argument(
        '--input', '-i',
        default='data/gsm8k_train_cot.jsonl',
        help='Input jsonl file path'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/gsm8k_train_cot_filtered.jsonl',
        help='Output jsonl file path'
    )
    parser.add_argument(
        '--ground_truth', '-g',
        default=None,
        help='Ground truth JSONL file (same order as --output) for comparison'
    )
    parser.add_argument(
        '--report', '-r',
        default='data/gsm8k_comparison_report.txt',
        help='Path for comparison report output'
    )
    parser.add_argument(
        '--incorrect_output', '-w',
        default=None,
        help='If set, write valid-numeric but incorrect entries to this JSONL file'
    )
    parser.add_argument(
        '--correct_output', '-c',
        default=None,
        help='If set, write correct entries (student training set) to this JSONL file'
    )
    args = parser.parse_args()
    filter_data(args.input, args.output)
    if args.ground_truth:
        compare_with_ground_truth(args.output, args.ground_truth, args.report,
                                  args.incorrect_output, args.correct_output)
