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


def clean_final_answer(text, math_mode=False):
    """Remove units, formatting, and LaTeX artifacts from extracted boxed content.

    In math_mode, preserves algebraic notation and LaTeX commands (frac, exponents).
    """
    if not text:
        return text

    # 1. Remove \text{...} blocks — in math mode keep inner content
    text = _remove_balanced_blocks(text, r'\text{', keep_content=math_mode)

    # 2. Remove \overline{...} — keep the inner number
    text = _remove_balanced_blocks(text, r'\overline{', keep_content=True)

    if not math_mode:
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
    if not math_mode:
        text = re.sub(r'\^[234]', '', text)  # superscript ^2, ^3, ^4
        text = text.replace('²', '').replace('³', '')  # unicode superscripts

    # 6. Remove commas between digits (1,000 → 1000)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)

    if not math_mode:
        # 7. Remove remaining backslashes
        text = text.replace('\\', '')

    # 8. Collapse multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()

    # 9. Normalize time format: "07:30" → "730", "9:00" → "900"
    if not math_mode:
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


_MONTHS = (
    r'January|February|March|April|May|June|'
    r'July|August|September|October|November|December|'
    r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
)


def extract_answer_math_fallback(text):
    """Math mode fallback: when no \\boxed{}, extract the answer after
    **Answer:**. Handles dates (June 20th → June 20), fractions (4/5 → \\frac{4}{5}),
    and plain numbers."""
    marker_match = re.search(r'\*\*Answer:?\*\*\s*:?\s*', text)
    if not marker_match:
        return ""

    answer_text = text[marker_match.end():].strip()

    # 1. Check for date: month name followed by a number (possibly with ordinal)
    date_m = re.match(
        rf'({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b',
        answer_text, re.IGNORECASE
    )
    if date_m:
        month = date_m.group(1).capitalize()
        day = date_m.group(2)
        return f'{month} {day}'

    # 2. Match first number-like expression: integer, decimal, or fraction a/b
    m = re.search(r'\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?', answer_text)
    if not m:
        return ""

    raw = m.group()
    # Convert a/b to \frac{a}{b}
    if '/' in raw:
        parts = raw.split('/')
        raw = f'\\frac{{{parts[0]}}}{{{parts[1]}}}'

    return raw


def filter_data(input_path, output_path, math_mode=False, bad_output_path=None):
    bad_entries = []
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            data = json.loads(line.strip())
            raw_boxed = extract_boxed(data['answer'])
            if math_mode:
                if not raw_boxed:
                    raw_boxed = extract_answer_math_fallback(data['answer'])
            else:
                if not raw_boxed:
                    raw_boxed = extract_answer_fallback(data['answer'])
                if not raw_boxed:
                    raw_boxed = extract_last_number_fallback(data['answer'])
            data['final_answer'] = clean_final_answer(raw_boxed, math_mode=math_mode)
            if bad_output_path is not None and not data['final_answer']:
                bad_entries.append(data)
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')

    if bad_output_path and bad_entries:
        with open(bad_output_path, 'w') as f:
            for entry in bad_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"Empty final_answer entries: {len(bad_entries)} -> {bad_output_path}")

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


def _category_stats():
    """Return a fresh stats dict for one category (type or level)."""
    return {'total': 0, 'empty': 0, 'non_numeric': 0, 'valid': 0, 'correct': 0}


def _update_stats(stats, is_empty, is_non_numeric, is_correct):
    """Update a stats dict with one sample's outcome."""
    stats['total'] += 1
    if is_empty:
        stats['empty'] += 1
    elif is_non_numeric:
        stats['non_numeric'] += 1
    else:
        stats['valid'] += 1
        if is_correct:
            stats['correct'] += 1


def _format_category_report(cat_stats, cat_name, math_mode=False):
    """Format per-category breakdown for the report."""
    lines = [f"  By {cat_name}:"]
    for value in sorted(cat_stats.keys()):
        s = cat_stats[value]
        acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0.0
        vac = s['correct'] / s['valid'] * 100 if s['valid'] > 0 else 0.0
        if math_mode:
            lines.append(
                f"    {value:12s}  total={s['total']:5d}  empty={s['empty']:4d}  "
                f"valid={s['valid']:5d}  correct={s['correct']:5d}  "
                f"acc={acc:5.1f}%  v_acc={vac:5.1f}%"
            )
        else:
            lines.append(
                f"    {value:12s}  total={s['total']:5d}  empty={s['empty']:4d}  "
                f"non_num={s['non_numeric']:4d}  valid={s['valid']:5d}  "
                f"correct={s['correct']:5d}  acc={acc:5.1f}%  v_acc={vac:5.1f}%"
            )
    return '\n'.join(lines)


def _normalize_answer(text):
    """Normalize answer text for string comparison: collapse whitespace, strip."""
    return re.sub(r'\s+', ' ', text).strip()


def _load_ground_truth(gt_path):
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


def compare_with_ground_truth(filtered_path, ground_truth_path, report_path,
                              incorrect_output_path=None,
                              correct_output_path=None,
                              math_mode=False):
    """Compare final_answer from filtered file with ground truth answers.

    Matches by problem text (not line index), so the filtered file may have
    multiple answers per problem (e.g. after deduplication).

    In math_mode, treats all non-empty answers as valid and uses string
    comparison (with whitespace normalization) instead of numeric comparison.
    """
    gt_map = _load_ground_truth(ground_truth_path)
    print(f"Loaded {len(gt_map)} ground-truth entries from {ground_truth_path}")

    total = 0
    empty = 0
    non_numeric = 0
    valid = 0
    correct = 0
    incorrect_entries = []
    correct_entries = []

    has_type = False
    has_level = False
    type_stats = {}
    level_stats = {}

    # Per-problem tracking
    problem_samples = {}   # problem -> list of is_correct booleans
    missing_gt = set()

    with open(filtered_path, 'r') as f_filt:
        for line_f in f_filt:
            total += 1
            data_f = json.loads(line_f.strip())

            problem = data_f['problem']
            fa = data_f.get('final_answer', '').strip()
            gt = gt_map.get(problem)

            if gt is None:
                missing_gt.add(problem)

            sample_type = str(data_f.get('type', ''))
            sample_level = str(data_f.get('level', ''))

            if sample_type:
                has_type = True
                type_stats.setdefault(sample_type, _category_stats())
            if sample_level:
                has_level = True
                level_stats.setdefault(sample_level, _category_stats())

            is_empty = not fa
            if math_mode:
                is_non_numeric = False
            else:
                is_non_numeric = not is_empty and not is_valid_numeric(fa)
            is_correct = False

            if is_empty:
                empty += 1
            elif is_non_numeric:
                non_numeric += 1
            else:
                valid += 1

                if gt is not None:
                    if math_mode:
                        if _normalize_answer(fa) == _normalize_answer(gt):
                            correct += 1
                            is_correct = True
                            entry = {
                                'problem': problem,
                                'answer': data_f['answer'],
                                'final_answer': fa,
                                'ground_truth': gt
                            }
                            for key in ('type', 'level'):
                                if key in data_f:
                                    entry[key] = data_f[key]
                            correct_entries.append(entry)
                        else:
                            entry = {
                                'problem': problem,
                                'teacher_answer': data_f['answer'],
                                'final_answer': fa,
                                'ground_truth': gt
                            }
                            for key in ('type', 'level'):
                                if key in data_f:
                                    entry[key] = data_f[key]
                            incorrect_entries.append(entry)
                    else:
                        fa_val = normalize_number(fa)
                        gt_val = normalize_number(gt)

                        if fa_val is not None and gt_val is not None:
                            if abs(fa_val - gt_val) < 1e-9:
                                correct += 1
                                is_correct = True
                                entry = {
                                    'problem': problem,
                                    'answer': data_f['answer'],
                                    'final_answer': fa,
                                    'ground_truth': gt
                                }
                                for key in ('type', 'level'):
                                    if key in data_f:
                                        entry[key] = data_f[key]
                                correct_entries.append(entry)
                            else:
                                entry = {
                                    'problem': problem,
                                    'teacher_answer': data_f['answer'],
                                    'final_answer': fa,
                                    'ground_truth': gt
                                }
                                for key in ('type', 'level'):
                                    if key in data_f:
                                        entry[key] = data_f[key]
                                incorrect_entries.append(entry)

            if sample_type:
                _update_stats(type_stats[sample_type], is_empty, is_non_numeric, is_correct)
            if sample_level:
                _update_stats(level_stats[sample_level], is_empty, is_non_numeric, is_correct)

            # Track per-problem
            if problem not in problem_samples:
                problem_samples[problem] = []
            problem_samples[problem].append(is_correct)

    # ---- Problem-level summary ----
    n_problems = len(problem_samples)
    problems_all_correct = 0
    problems_any_correct = 0
    problems_none_correct = 0
    for p, samples in problem_samples.items():
        if p not in gt_map:
            continue
        if all(samples):
            problems_all_correct += 1
        if any(samples):
            problems_any_correct += 1
        else:
            problems_none_correct += 1

    n_with_gt = n_problems - len(missing_gt)

    # ---- Report ----
    incorrect_count = valid - correct
    accuracy = correct / total * 100 if total > 0 else 0.0
    valid_accuracy = correct / valid * 100 if valid > 0 else 0.0

    if math_mode:
        header = "MATH Final Answer Comparison Report"
    else:
        header = "GSM8K Final Answer Comparison Report"

    report = f"""{header}
{'=' * len(header)}

--- Sample-level ---
Total samples:               {total}
  - Empty final_answer:      {empty}   ({empty/total*100:.1f}%)
  - Non-numeric final_answer:{non_numeric}    ({non_numeric/total*100:.1f}%)
  - Valid:                   {valid}   ({valid/total*100:.1f}%)
    - Correct:               {correct}   ({correct/total*100:.1f}%)
    - Incorrect:             {incorrect_count}    ({incorrect_count/total*100:.1f}%)

Accuracy (correct / total):          {accuracy:.2f}%
Accuracy (correct / valid):          {valid_accuracy:.2f}%

--- Problem-level ---
Unique problems in filtered output: {n_problems}
  (ground-truth available for:       {n_with_gt})
Problems with ALL answers correct:   {problems_all_correct}
Problems with ANY answer correct:    {problems_any_correct}
Problems with NO answer correct:     {problems_none_correct}
"""

    if missing_gt:
        report += f"\nWARNING: {len(missing_gt)} problems have no matching ground-truth entry.\n"

    if has_type:
        report += '\n' + _format_category_report(type_stats, 'type', math_mode) + '\n'
    if has_level:
        report += '\n' + _format_category_report(level_stats, 'level', math_mode) + '\n'

    print(report)

    if report_path:
        with open(report_path, 'w') as f:
            f.write(report)
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
        default=None,
        help='Input jsonl file path'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output jsonl file path'
    )
    parser.add_argument(
        '--ground_truth', '-g',
        default=None,
        help='Ground truth JSONL file (same order as --output) for comparison'
    )
    parser.add_argument(
        '--report', '-r',
        default=None,
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
    parser.add_argument(
        '--math', action='store_true',
        default=False,
        help='MATH dataset mode: only extract \\boxed{} content, skip numeric fallbacks'
    )
    parser.add_argument(
        '--bad_output', '-b',
        default=None,
        help='If set, write entries with empty final_answer to this JSONL file'
    )
    args = parser.parse_args()
    filter_data(args.input, args.output, math_mode=args.math,
                bad_output_path=args.bad_output)
    if args.ground_truth:
        compare_with_ground_truth(args.output, args.ground_truth, args.report,
                                  args.incorrect_output, args.correct_output,
                                  math_mode=args.math)
