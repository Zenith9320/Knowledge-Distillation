#!/usr/bin/env python3
"""
Evaluate a model's accuracy on math reasoning test datasets.

Generates CoT answers for each problem, extracts the final answer from
\\boxed{}, and compares with ground truth. Supports both GSM8K and MATH.

Reuses the same prompt format, answer extraction, and comparison logic as
the existing filter.py pipeline for consistent evaluation.

Usage:
    # Evaluate on GSM8K test set
    python evaluate.py --model models/qwen-adaptive/checkpoint-5500 --dataset gsm8k

    # Evaluate on MATH test set
    python evaluate.py --model models/qwen-double_temp --dataset math

    # Evaluate on both datasets
    python evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset both

    # Limit to N samples for quick testing
    python evaluate.py --model models/qwen-single_temp --dataset gsm8k --max_samples 20

    # Increase max tokens for harder problems
    python evaluate.py --model models/qwen-double_temp --dataset math --max_new_tokens 4096
"""

import argparse
import json
import os
import re
import sys
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a math expert. Solve the problem step by step, "
    "showing your reasoning clearly. Put your final numeric answer in \\boxed{}."
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DATASET_CONFIG = {
    "gsm8k": {
        "test_file": "gsm8k_test.jsonl",
    },
    "math": {
        "test_file": "math_test.jsonl",
    },
}


# ---------------------------------------------------------------------------
# Answer extraction (reused from filter.py)
# ---------------------------------------------------------------------------

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
    """Remove `marker{...}` blocks with balanced brace matching."""
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
    """Convert \\frac{a}{b} or \\dfrac{a}{b} to a/b representation."""
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
    """Find all number-like substrings in text, returning list of (raw_string, start_pos)."""
    pattern = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    return [(m.group(), m.start()) for m in re.finditer(pattern, text)]


def _normalize_num(num_str):
    """Remove commas from a number string for comparison."""
    return num_str.replace(',', '')


def extract_answer_fallback(text):
    """Fallback: when no \\boxed{} is found, extract the answer from **Answer:** text."""
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
    """Math mode fallback: when no \\boxed{}, extract the answer after **Answer:**."""
    marker_match = re.search(r'\*\*Answer:?\*\*\s*:?\s*', text)
    if not marker_match:
        return ""

    answer_text = text[marker_match.end():].strip()

    # Check for date
    date_m = re.match(
        rf'({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b',
        answer_text, re.IGNORECASE
    )
    if date_m:
        month = date_m.group(1).capitalize()
        day = date_m.group(2)
        return f'{month} {day}'

    # Match first number-like expression
    m = re.search(r'\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?', answer_text)
    if not m:
        return ""

    raw = m.group()
    if '/' in raw:
        parts = raw.split('/')
        raw = f'\\frac{{{parts[0]}}}{{{parts[1]}}}'

    return raw


def clean_final_answer(text, math_mode=False):
    """Remove units, formatting, and LaTeX artifacts from extracted boxed content."""
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


def extract_final_answer(answer_text, math_mode=False):
    """Extract the final answer from a generated CoT text.

    Follows the same extraction chain as filter.py:
      1. Try \\boxed{} extraction
      2. Fall back to **Answer:** extraction
      3. GSM8K: last-resort last-number fallback
    """
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
    """Convert a numeric string to a float value for comparison."""
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
    """Normalize answer text for string comparison."""
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# DeepSeek tag cleaning
# ---------------------------------------------------------------------------

def clean_deepseek_tags(text: str) -> str:
    """Remove <think>...</think> from DeepSeek-R1 style model outputs."""
    idx = text.find("</think>")
    if idx != -1:
        text = text[idx + len("</think>"):]
    return text.strip()


# ---------------------------------------------------------------------------
# Repetition detection (early stopping)
# ---------------------------------------------------------------------------

class RepetitionStoppingCriteria(StoppingCriteria):
    """Stop generation when the same text pattern repeats excessively.

    Detects when a substring of at least ``min_pattern_len`` characters
    repeats consecutively ``max_repeats`` or more times at the end of the
    generated text.  This prevents the model from looping on tokens like
    ``\\nmoire\\nmoire\\nmoire...`` and wasting compute.

    Uses an efficient sliding-window check that only examines the tail of
    the generated text, avoiding the catastrophic backtracking of a
    backreference regex on long sequences.  Also throttles to check only
    every ``check_every`` tokens to reduce per-step overhead.

    Parameters
    ----------
    tokenizer:
        The tokenizer, used to decode generated token IDs to text.
    min_pattern_len:
        Minimum length (in characters) of a repeating pattern to detect.
    max_repeats:
        Stop when the same pattern appears this many times consecutively.
    input_len:
        Number of prompt tokens to skip when decoding the generated part.
    check_every:
        Only run the repetition check every N new tokens (default: 8).
        Checking every token is wasteful since repetition patterns span
        many tokens.
    """

    def __init__(
        self,
        tokenizer,
        min_pattern_len: int = 4,
        max_repeats: int = 8,
        input_len: int = 0,
        check_every: int = 8,
    ):
        self.tokenizer = tokenizer
        self.min_pattern_len = min_pattern_len
        self.max_repeats = max_repeats
        self.input_len = input_len
        self.check_every = check_every
        self._step_count = 0
        # Upper bound on pattern length to check (repeating very long
        # patterns is rare; capping avoids wasted work).
        self._max_pattern_check = 80

    def __call__(self, input_ids, scores, **kwargs):
        # Throttle: only check every N steps
        self._step_count += 1
        if self._step_count % self.check_every != 0:
            return False

        generated_ids = input_ids[0][self.input_len:]
        min_chars = self.min_pattern_len * self.max_repeats
        if len(generated_ids) < min_chars:
            return False

        # Only decode the *tail* where a repeat could live.
        # We need at most (max_pattern_check * max_repeats) chars from the end.
        tail_len = self._max_pattern_check * self.max_repeats
        tail_ids = generated_ids[-tail_len:]
        tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=False)

        # Also try decoding the raw token text of the last few tokens for
        # tighter detection of token-boundary repeats.
        text_len = len(tail_text)
        if text_len < min_chars:
            return False

        # Check from the end: for each candidate pattern length L, see if
        # the last (L * max_repeats) chars are exactly the same L-char
        # substring repeated max_repeats times.
        max_L = min(self._max_pattern_check, text_len // self.max_repeats)
        for L in range(self.min_pattern_len, max_L + 1):
            block = tail_text[-L * self.max_repeats:]
            pattern = block[:L]
            # Quick check: all L-length slices must equal pattern
            matches = True
            for r in range(1, self.max_repeats):
                if block[r * L:(r + 1) * L] != pattern:
                    matches = False
                    break
            if matches:
                return True

        return False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from a local path or HuggingFace hub name."""
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Model loaded on {device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(
    model,
    tokenizer,
    problem: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    clean_tags: bool = False,
    system_prompt: str | None = None,
    repetition_max_repeats: int | None = None,
    repetition_min_len: int = 4,
) -> str:
    """Generate a CoT answer for a single problem.

    Args:
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        problem: The math problem text.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling top-p.
        clean_tags: If True, strip <think>...</think> tags (for DeepSeek-R1 models).
        system_prompt: Override the default system prompt. If None, uses SYSTEM_PROMPT.
            Pass an empty string "" for no system prompt.
        repetition_max_repeats: If set, stop generation when the same substring
            of length >= ``repetition_min_len`` repeats this many times
            consecutively.  ``None`` disables repetition detection.
        repetition_min_len: Minimum pattern length for repetition detection.

    Returns:
        Generated text (only the new tokens, not the prompt).
    """
    sp = SYSTEM_PROMPT if system_prompt is None else system_prompt
    messages = [
        {"role": "system", "content": sp},
        {"role": "user", "content": f"Problem: {problem}"},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    # Build EOS token IDs: include both the tokenizer's native eos_token
    # (e.g. <|endoftext|> for Qwen2.5-base, <|im_end|> for Instruct) AND
    # <|im_end|> explicitly.  Base models fine-tuned with the chat template
    # generate <|im_end|> to end turns, but their config eos_token_id may
    # still point to <|endoftext|>, causing generation to run until max_new_tokens.
    eos_token_ids = [tokenizer.eos_token_id]
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.eos_token_id:
        eos_token_ids.append(im_end_id)

    # Build stopping criteria (optional repetition detection)
    stopping_criteria = None
    if repetition_max_repeats is not None and repetition_max_repeats > 0:
        stopping_criteria = StoppingCriteriaList([
            RepetitionStoppingCriteria(
                tokenizer,
                min_pattern_len=repetition_min_len,
                max_repeats=repetition_max_repeats,
                input_len=input_len,
            )
        ])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_token_ids,
            stopping_criteria=stopping_criteria,
        )

    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if clean_tags:
        generated_text = clean_deepseek_tags(generated_text)

    return generated_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _suffix_path(path: str, dataset: str) -> str:
    """Insert dataset name before the file extension: report.txt -> report_gsm8k.txt."""
    base, ext = os.path.splitext(path)
    return f"{base}_{dataset}{ext}"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str,
    dataset: str = "gsm8k",
    max_samples: int | None = None,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    clean_tags: bool = False,
    verbose: bool = False,
    output: str | None = None,
    report_path: str | None = None,
    system_prompt: str | None = None,
    repetition_max_repeats: int | None = None,
    repetition_min_len: int = 4,
):
    """Evaluate a model on a test dataset and report accuracy.

    Args:
        model_path: Path to the model (local dir or HF hub name).
        dataset: ``"gsm8k"``, ``"math"``, or ``"both"``.
        max_samples: Limit to first N test samples (None = all).
        max_new_tokens: Max tokens for generation.
        temperature: Sampling temperature.
        top_p: Nucleus sampling top-p.
        clean_tags: Strip <think>...</think> (for DeepSeek-R1 models).
        verbose: Print per-sample results (problem, predicted, ground truth).
        output: If set, write all predictions as JSONL to this path.
        report_path: If set, write the accuracy report to this file.
        system_prompt: Override system prompt. None = default. "" = no system prompt.
        repetition_max_repeats: Stop generation when a substring repeats this
            many times consecutively. None disables repetition detection.
        repetition_min_len: Minimum pattern length for repetition detection.

    Returns:
        dict: Accuracy statistics per dataset.
    """
    if dataset == "both":
        results = {}
        for ds in ["gsm8k", "math"]:
            results[ds] = evaluate(
                model_path=model_path,
                dataset=ds,
                max_samples=max_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                clean_tags=clean_tags,
                verbose=verbose,
                output=_suffix_path(output, ds) if output else None,
                report_path=_suffix_path(report_path, ds) if report_path else None,
                system_prompt=system_prompt,
                repetition_max_repeats=repetition_max_repeats,
                repetition_min_len=repetition_min_len,
            )
        return results

    math_mode = (dataset == "math")
    config = DATASET_CONFIG[dataset]
    test_path = os.path.join(DATA_DIR, config["test_file"])

    if not os.path.exists(test_path):
        print(f"Error: test file not found: {test_path}")
        return None

    # Load model (once per dataset)
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load test data
    test_data = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_data.append(json.loads(line))

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"\nEvaluating on {dataset.upper()} test set ({len(test_data)} samples)")
    print(f"  max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}")

    # Statistics
    total = 0
    empty_answer = 0
    non_numeric = 0
    correct = 0
    valid = 0
    all_results = []  # Collect all results for file output

    # Per-category stats (MATH)
    type_stats = {}
    level_stats = {}

    for record in tqdm(test_data, desc=f"Evaluating {dataset}"):
        problem = record["problem"]
        gt = record["answer"]

        # Generate
        generated = generate_answer(
            model, tokenizer, problem,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            clean_tags=clean_tags,
            system_prompt=system_prompt,
            repetition_max_repeats=repetition_max_repeats,
            repetition_min_len=repetition_min_len,
        )

        # Extract final answer
        fa = extract_final_answer(generated, math_mode=math_mode)

        # Compare
        is_empty = not fa
        is_non_num = False if math_mode else (not is_empty and not is_valid_numeric(fa))
        is_correct = False

        if is_empty:
            empty_answer += 1
        elif is_non_num:
            non_numeric += 1
        else:
            valid += 1
            if math_mode:
                if _normalize_answer(fa) == _normalize_answer(gt):
                    correct += 1
                    is_correct = True
            else:
                fa_val = normalize_number(fa)
                gt_val = normalize_number(gt)
                if fa_val is not None and gt_val is not None:
                    if abs(fa_val - gt_val) < 1e-9:
                        correct += 1
                        is_correct = True

        total += 1

        # Collect result record
        result_record = {
            "problem": problem,
            "ground_truth": gt,
            "generated": generated,
            "final_answer": fa,
            "is_correct": is_correct,
        }
        for key in ("type", "level"):
            if key in record:
                result_record[key] = record[key]
        all_results.append(result_record)

        # Per-category tracking
        rec_type = str(record.get("type", ""))
        rec_level = str(record.get("level", ""))
        if rec_type:
            if rec_type not in type_stats:
                type_stats[rec_type] = {"total": 0, "empty": 0, "correct": 0, "valid": 0}
            type_stats[rec_type]["total"] += 1
            if is_empty:
                type_stats[rec_type]["empty"] += 1
            elif not is_non_num:
                type_stats[rec_type]["valid"] += 1
                if is_correct:
                    type_stats[rec_type]["correct"] += 1
        if rec_level:
            if rec_level not in level_stats:
                level_stats[rec_level] = {"total": 0, "empty": 0, "correct": 0, "valid": 0}
            level_stats[rec_level]["total"] += 1
            if is_empty:
                level_stats[rec_level]["empty"] += 1
            elif not is_non_num:
                level_stats[rec_level]["valid"] += 1
                if is_correct:
                    level_stats[rec_level]["correct"] += 1

        if verbose:
            status = "✓" if is_correct else ("✗" if not is_empty else "∅")
            print(f"\n[{status}] {problem[:80]}...")
            print(f"  Predicted: {fa if fa else '(empty)'}")
            print(f"  Expected:  {gt}")

    # ---- Report ----
    accuracy = correct / total * 100 if total > 0 else 0.0
    valid_accuracy = correct / valid * 100 if valid > 0 else 0.0

    label = "MATH" if math_mode else "GSM8K"
    report = f"""
{'=' * 60}
{label} Evaluation Results
{'=' * 60}
Model: {model_path}
Samples: {total}

--- Sample-level ---
  Correct:                {correct}   ({accuracy:.1f}%)
  Empty final_answer:     {empty_answer}   ({empty_answer / total * 100:.1f}%)\
"""
    if not math_mode:
        report += f"""
  Non-numeric:            {non_numeric}    ({non_numeric / total * 100:.1f}%)\
"""
    report += f"""
  Valid (non-empty):      {valid}   ({valid / total * 100:.1f}%)

Accuracy (correct / total):         {accuracy:.2f}%
Accuracy (correct / valid):         {valid_accuracy:.2f}%
"""

    # Per-type breakdown (MATH)
    if type_stats:
        report += f"\n--- By Type ---\n"
        for t in sorted(type_stats.keys()):
            s = type_stats[t]
            acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0.0
            vacc = s["correct"] / s["valid"] * 100 if s["valid"] > 0 else 0.0
            report += (f"  {t:26s} total={s['total']:5d}  empty={s['empty']:4d}  "
                       f"correct={s['correct']:5d}  acc={acc:5.1f}%  v_acc={vacc:5.1f}%\n")

    # Per-level breakdown (MATH)
    if level_stats:
        report += f"\n--- By Level ---\n"
        for lvl in sorted(level_stats.keys()):
            s = level_stats[lvl]
            acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0.0
            vacc = s["correct"] / s["valid"] * 100 if s["valid"] > 0 else 0.0
            report += (f"  {lvl:12s} total={s['total']:5d}  empty={s['empty']:4d}  "
                       f"correct={s['correct']:5d}  acc={acc:5.1f}%  v_acc={vacc:5.1f}%\n")

    print(report)

    # ---- Write output files ----
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for rec in all_results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Predictions written to: {output}")

    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to: {report_path}")

    return {
        "dataset": dataset,
        "total": total,
        "correct": correct,
        "empty": empty_answer,
        "non_numeric": non_numeric,
        "valid": valid,
        "accuracy": accuracy,
        "valid_accuracy": valid_accuracy,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on math reasoning test datasets"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model path (local directory or HuggingFace hub name)",
    )
    parser.add_argument(
        "--dataset", default="gsm8k", choices=["gsm8k", "math", "both"],
        help="Dataset to evaluate on (default: gsm8k)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit to first N samples (default: all)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4096,
        help="Maximum tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Nucleus sampling top-p (default: 0.95)",
    )
    parser.add_argument(
        "--clean_tags", action="store_true",
        help="Strip <think>...</think> tags (for DeepSeek-R1 models)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-sample predictions",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Write all predictions as JSONL to this path",
    )
    parser.add_argument(
        "--report", "-r", default=None,
        help="Write the accuracy report to this file",
    )
    parser.add_argument(
        "--system_prompt", default=None,
        help="Override system prompt. Pass empty string '' for no system prompt. "
             "Default: math expert prompt.",
    )
    parser.add_argument(
        "--repetition_max_repeats", type=int, default=None,
        help="Stop generation when a substring repeats this many times "
             "consecutively (default: no repetition detection). "
             "Recommended: 8-10 for catching degenerate loops.",
    )
    parser.add_argument(
        "--repetition_min_len", type=int, default=4,
        help="Minimum pattern length for repetition detection (default: 4).",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        dataset=args.dataset,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        clean_tags=args.clean_tags,
        verbose=args.verbose,
        output=args.output,
        report_path=args.report,
        system_prompt=args.system_prompt,
        repetition_max_repeats=args.repetition_max_repeats,
        repetition_min_len=args.repetition_min_len,
    )


if __name__ == "__main__":
    main()