"""
Filter adaptive-sampling CoT output by extracting boxed answers and comparing
with ground truth.

Key difference from filter.py: the adaptive output contains multiple answers
per problem (tagged with temperature, round, sample_idx), so matching is done
by problem text rather than line-by-line zipping.

Usage:
    python filter_adaptive.py \\
        -i data/gsm8k_train_cot_adaptive.jsonl \\
        -g data/gsm8k_train.jsonl \\
        -o data/gsm8k_train_cot_adaptive_filtered.jsonl \\
        -c data/gsm8k_train_cot_adaptive_correct.jsonl \\
        -w data/gsm8k_train_cot_adaptive_incorrect.jsonl \\
        -b data/gsm8k_train_cot_adaptive_bad.jsonl \\
        -r data/gsm8k_adaptive_comparison_report.txt

    # MATH mode
    python filter_adaptive.py --math \\
        -i data/math_train_cot_adaptive.jsonl \\
        -g data/math_train.jsonl \\
        -o data/math_train_cot_adaptive_filtered.jsonl \\
        -c data/math_train_cot_adaptive_correct.jsonl \\
        -w data/math_train_cot_adaptive_incorrect.jsonl \\
        -b data/math_train_cot_adaptive_bad.jsonl \\
        -r data/math_adaptive_comparison_report.txt
"""

import argparse
import json
import re
import sys
import os

# ======================================================================
#  Boxed extraction & cleaning  (same logic as filter.py)
# ======================================================================

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
    result = []
    i = 0
    marker_len = len(marker)
    while i < len(text):
        if text[i:i + marker_len] == marker:
            rest = text[i + marker_len:]
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
    pattern = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    return [(m.group(), m.start()) for m in re.finditer(pattern, text)]


def _normalize_num(num_str):
    return num_str.replace(',', '')


def extract_answer_fallback(text):
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
    for raw_num, _pos in reversed(before_numbers):
        if _normalize_num(raw_num) in answer_set:
            return raw_num
    return ""


def clean_final_answer(text, math_mode=False):
    if not text:
        return text
    text = _remove_balanced_blocks(text, r'\text{', keep_content=math_mode)
    text = _remove_balanced_blocks(text, r'\overline{', keep_content=True)
    if not math_mode:
        text = _extract_frac(text, r'\dfrac')
        text = _extract_frac(text, r'\frac')
        text = _remove_balanced_blocks(text, r'\begin{')
        text = _remove_balanced_blocks(text, r'\end{')
    text = text.replace('\\,', '')
    text = text.replace('\\!', '')
    text = text.replace('\\$', '')
    text = text.replace('$', '')
    text = text.replace('\\%', '')
    text = text.replace('%', '')
    text = re.sub(r'\^\{?\\circ\}?', '', text)
    text = text.replace('\\circ', '')
    text = text.replace('\\\\', '')
    text = text.replace('£', '')
    text = text.replace('€', '')
    if not math_mode:
        text = re.sub(r'\^[234]', '', text)
        text = text.replace('²', '').replace('³', '')
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    if not math_mode:
        text = text.replace('\\', '')
    text = re.sub(r'\s+', ' ', text).strip()
    if not math_mode:
        m = re.match(r'^0?(\d{1,2}):(\d{2})$', text)
        if m:
            text = f"{int(m.group(1))}{m.group(2)}"
    return text


def extract_last_number_fallback(text):
    text = text.strip()
    if not text.endswith('.'):
        return ""
    numbers = find_numbers_in_text(text)
    if not numbers:
        return ""
    return numbers[-1][0]


_MONTHS = (
    r'January|February|March|April|May|June|'
    r'July|August|September|October|November|December|'
    r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
)


def extract_answer_math_fallback(text):
    marker_match = re.search(r'\*\*Answer:?\*\*\s*:?\s*', text)
    if not marker_match:
        return ""
    answer_text = text[marker_match.end():].strip()
    date_m = re.match(
        rf'({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b',
        answer_text, re.IGNORECASE
    )
    if date_m:
        month = date_m.group(1).capitalize()
        day = date_m.group(2)
        return f'{month} {day}'
    m = re.search(r'\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?', answer_text)
    if not m:
        return ""
    raw = m.group()
    if '/' in raw:
        parts = raw.split('/')
        raw = f'\\frac{{{parts[0]}}}{{{parts[1]}}}'
    return raw


# ======================================================================
#  Answer extraction (pipeline)
# ======================================================================

def extract_final_answer(answer_text, math_mode=False):
    """Run the full extraction pipeline on an answer string."""
    raw_boxed = extract_boxed(answer_text)
    if math_mode:
        if not raw_boxed:
            raw_boxed = extract_answer_math_fallback(answer_text)
    else:
        if not raw_boxed:
            raw_boxed = extract_answer_fallback(answer_text)
        if not raw_boxed:
            raw_boxed = extract_last_number_fallback(answer_text)
    return clean_final_answer(raw_boxed, math_mode=math_mode)


# ======================================================================
#  Comparison helpers
# ======================================================================

def is_valid_numeric(text):
    if not text:
        return False
    if re.match(r'^-?\d+(?:\.\d+)?$', text):
        return True
    if re.match(r'^-?\d+(?:\.\d+)?/\d+(?:\.\d+)?$', text):
        return True
    return False


def normalize_number(text):
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


def _normalize_answer(text):
    return re.sub(r'\s+', ' ', text).strip()


def check_correct(fa, gt, math_mode=False):
    """Return True if final_answer matches ground truth."""
    if math_mode:
        return _normalize_answer(fa) == _normalize_answer(gt)
    else:
        fa_val = normalize_number(fa)
        gt_val = normalize_number(gt)
        if fa_val is not None and gt_val is not None:
            return abs(fa_val - gt_val) < 1e-9
        return False


# ======================================================================
#  Main filter logic
# ======================================================================

def load_ground_truth(gt_path):
    """Load ground truth into a dict: problem_text -> answer."""
    gt_map = {}
    with open(gt_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gt_map[rec['problem']] = rec['answer']
    return gt_map


def process_adaptive(input_path, ground_truth_path, output_path,
                     math_mode=False, bad_output_path=None,
                     correct_output_path=None, incorrect_output_path=None,
                     report_path=None):
    gt_map = load_ground_truth(ground_truth_path)
    print(f"Loaded {len(gt_map)} ground-truth entries from {ground_truth_path}")

    # Counters (sample-level)
    total = 0
    empty = 0
    non_numeric = 0
    valid = 0
    correct = 0

    # Per-round counters: round -> {total, empty, non_numeric, valid, correct}
    round_stats = {}

    # Per-problem tracking
    problem_samples = {}   # problem -> list of dicts (each sample's result)
    problems_with_gt = set()
    problems_missing_gt = set()

    bad_entries = []
    correct_entries = []
    incorrect_entries = []

    with open(input_path, encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1

            problem = rec['problem']
            answer_text = rec['answer']
            round_num = rec.get('round', 0)

            # Look up ground truth by problem text
            gt = gt_map.get(problem)
            if gt is None:
                problems_missing_gt.add(problem)

            # Extract & clean final answer
            fa = extract_final_answer(answer_text, math_mode=math_mode)

            # Build output record: preserve all original fields + add new ones
            out_rec = dict(rec)
            out_rec['final_answer'] = fa
            out_rec['ground_truth'] = gt

            is_empty = not fa
            if math_mode:
                is_non_num = False
            else:
                is_non_num = not is_empty and not is_valid_numeric(fa)
            is_correct = False

            if gt is not None and not is_empty and not is_non_num:
                is_correct = check_correct(fa, gt, math_mode=math_mode)

            out_rec['is_correct'] = is_correct

            # Update sample-level counters
            if is_empty:
                empty += 1
                bad_entries.append(out_rec)
            elif is_non_num:
                non_numeric += 1
                incorrect_entries.append(out_rec)
            else:
                valid += 1
                if is_correct:
                    correct += 1
                    correct_entries.append(out_rec)
                else:
                    incorrect_entries.append(out_rec)

            # Update round stats
            rk = str(round_num)
            if rk not in round_stats:
                round_stats[rk] = {'total': 0, 'empty': 0, 'non_numeric': 0,
                                   'valid': 0, 'correct': 0}
            rs = round_stats[rk]
            rs['total'] += 1
            if is_empty:
                rs['empty'] += 1
            elif is_non_num:
                rs['non_numeric'] += 1
            else:
                rs['valid'] += 1
                if is_correct:
                    rs['correct'] += 1

            # Track per-problem
            if problem not in problem_samples:
                problem_samples[problem] = []
            problem_samples[problem].append(out_rec)

            if gt is not None:
                problems_with_gt.add(problem)

            fout.write(json.dumps(out_rec, ensure_ascii=False) + '\n')

    # ---- Write output files ----
    if bad_output_path and bad_entries:
        with open(bad_output_path, 'w', encoding='utf-8') as f:
            for entry in bad_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Bad entries (empty final_answer): {len(bad_entries)} -> {bad_output_path}")

    if correct_output_path and correct_entries:
        with open(correct_output_path, 'w', encoding='utf-8') as f:
            for entry in correct_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Correct entries: {len(correct_entries)} -> {correct_output_path}")

    if incorrect_output_path and incorrect_entries:
        with open(incorrect_output_path, 'w', encoding='utf-8') as f:
            for entry in incorrect_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Incorrect entries: {len(incorrect_entries)} -> {incorrect_output_path}")

    # ---- Per-problem summary ----
    n_problems = len(problem_samples)
    problems_all_correct = 0
    problems_any_correct = 0
    problems_none_correct = 0

    for problem, samples in problem_samples.items():
        if problem not in gt_map:
            continue
        sample_correct = [s['is_correct'] for s in samples]
        if all(sample_correct):
            problems_all_correct += 1
        if any(sample_correct):
            problems_any_correct += 1
        else:
            problems_none_correct += 1

    problems_with_gt_count = len(problems_with_gt)
    n_missing = len(problems_missing_gt)

    # ---- Report ----
    incorrect_count = valid - correct
    accuracy = correct / total * 100 if total > 0 else 0.0
    valid_accuracy = correct / valid * 100 if valid > 0 else 0.0

    if math_mode:
        header = "MATH Adaptive Final Answer Comparison Report"
    else:
        header = "GSM8K Adaptive Final Answer Comparison Report"

    report = f"""{header}
{'=' * len(header)}

--- Sample-level (total lines) ---
Total samples:                       {total}
  - Empty final_answer:              {empty}   ({empty/total*100:.1f}%)
  - Non-numeric final_answer:        {non_numeric}    ({non_numeric/total*100:.1f}%)
  - Valid:                           {valid}   ({valid/total*100:.1f}%)
    - Correct:                       {correct}   ({correct/total*100:.1f}%)
    - Incorrect:                     {incorrect_count}    ({incorrect_count/total*100:.1f}%)

Sample-level accuracy (correct / total):          {accuracy:.2f}%
Sample-level accuracy (correct / valid):          {valid_accuracy:.2f}%

--- Problem-level ---
Unique problems in adaptive output: {n_problems}
  (ground-truth available for:      {problems_with_gt_count})
Problems with ALL samples correct:  {problems_all_correct}  ({problems_all_correct/problems_with_gt_count*100:.1f}%)*
Problems with ANY sample correct:   {problems_any_correct}  ({problems_any_correct/problems_with_gt_count*100:.1f}%)*
Problems with NO samples correct:   {problems_none_correct}  ({problems_none_correct/problems_with_gt_count*100:.1f}%)*

*percentages based on problems with ground-truth ({problems_with_gt_count})
"""

    if problems_missing_gt:
        report += f"\nWARNING: {n_missing} problems have no matching ground-truth entry.\n"

    # Per-round breakdown
    if len(round_stats) > 1:
        report += "\n--- Per-round breakdown ---\n"
        for rk in sorted(round_stats.keys(), key=int):
            rs = round_stats[rk]
            r_acc = rs['correct'] / rs['total'] * 100 if rs['total'] > 0 else 0.0
            r_vacc = rs['correct'] / rs['valid'] * 100 if rs['valid'] > 0 else 0.0
            report += (
                f"  Round {rk}: total={rs['total']:5d}  empty={rs['empty']:4d}  "
                f"non_num={rs['non_numeric']:4d}  valid={rs['valid']:5d}  "
                f"correct={rs['correct']:5d}  acc={r_acc:5.1f}%  v_acc={r_vacc:5.1f}%\n"
            )

    avg_samples = total / n_problems if n_problems > 0 else 0.0
    report += f"\nAvg samples per problem: {avg_samples:.1f}\n"

    print(report)

    if report_path:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to {report_path}")

    print(f"\nProcessed {input_path} -> {output_path}")


# ======================================================================
#  CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Filter adaptive-sampling CoT output: extract boxed answers, '
                    'match by problem text, compare with ground truth.'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Adaptive CoT JSONL file (input)')
    parser.add_argument('--output', '-o', required=True,
                        help='Filtered JSONL output (all entries with final_answer added)')
    parser.add_argument('--ground_truth', '-g', required=True,
                        help='Ground truth JSONL file')
    parser.add_argument('--report', '-r', default=None,
                        help='Path for comparison report output')
    parser.add_argument('--correct_output', '-c', default=None,
                        help='Write correct entries to this JSONL file')
    parser.add_argument('--incorrect_output', '-w', default=None,
                        help='Write non-empty but incorrect entries to this JSONL')
    parser.add_argument('--bad_output', '-b', default=None,
                        help='Write empty final_answer entries to this JSONL')
    parser.add_argument('--math', action='store_true', default=False,
                        help='MATH dataset mode: string comparison, preserve LaTeX')
    args = parser.parse_args()

    process_adaptive(
        input_path=args.input,
        ground_truth_path=args.ground_truth,
        output_path=args.output,
        math_mode=args.math,
        bad_output_path=args.bad_output,
        correct_output_path=args.correct_output,
        incorrect_output_path=args.incorrect_output,
        report_path=args.report,
    )


if __name__ == '__main__':
    main()
