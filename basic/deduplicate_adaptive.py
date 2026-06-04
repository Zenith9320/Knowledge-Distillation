#!/usr/bin/env python3
"""
Apply semantic deduplication to adaptive-sampling training data.

The adaptive strategy generates n=3 samples per problem at T=0.3.  Many of
these samples are near-duplicates (pairwise cosine similarity >= 0.92).
This script:

  1. Groups records by problem within each input file.
  2. For each problem, computes Sentence-BERT embeddings of all answers.
  3. Keeps the first sample, then iterates through remaining samples:
     if a sample's maximum similarity to any already-kept sample is below
     the threshold, it is kept; otherwise discarded.
  4. Merges the deduplicated GSM8K and MATH datasets, then shuffles.

Usage:
    python deduplicate_adaptive.py

    # Custom threshold / output
    python deduplicate_adaptive.py --threshold 0.90 --output data/adaptive_dedup.jsonl
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DEFAULT_FILES = [
    os.path.join(DATA_DIR, "gsm8k_train_cot_adaptive_correct.jsonl"),
    os.path.join(DATA_DIR, "math_train_cot_adaptive_correct.jsonl"),
]

DEFAULT_OUTPUT = os.path.join(DATA_DIR, "train_cot_adaptive_dedup.jsonl")
DEFAULT_THRESHOLD = 0.92
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _ts(msg: str) -> str:
    return f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"


def load_and_group(filepath: str) -> dict[str, list[dict]]:
    """Load a JSONL file and group records by the ``problem`` field."""
    groups = defaultdict(list)
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            groups[rec["problem"]].append(rec)
    return dict(groups)


def deduplicate_groups(
    groups: dict[str, list[dict]],
    model: SentenceTransformer,
    threshold: float,
    batch_size: int,
) -> tuple[list[dict], dict]:
    """Deduplicate within each problem group.

    For a problem with k samples, the first sample is always kept.
    Subsequent samples are kept only if their maximum cosine similarity
    to all already-kept samples is below *threshold*.

    Returns (kept_records, stats).
    """
    kept_all = []
    stats = {
        "problems_total": len(groups),
        "problems_with_1_sample": 0,
        "problems_with_2_samples": 0,
        "problems_with_3_samples": 0,
        "samples_before": 0,
        "samples_after": 0,
        "pairs_above_threshold": 0,
        "pairs_total": 0,
    }

    for problem, records in groups.items():
        k = len(records)
        stats["samples_before"] += k
        if k == 1:
            stats["problems_with_1_sample"] += 1
            kept_all.extend(records)
            stats["samples_after"] += 1
            continue
        elif k == 2:
            stats["problems_with_2_samples"] += 1
        else:
            stats["problems_with_3_samples"] += 1

        # Embed all k answers
        answers = [r["answer"] for r in records]
        embeddings = model.encode(
            answers,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Greedy dedup: keep first, check rest against all kept
        kept_idx = [0]  # always keep the first
        for i in range(1, k):
            max_sim = max(
                float(np.dot(embeddings[i], embeddings[j]))
                for j in kept_idx
            )
            stats["pairs_total"] += 1
            if max_sim <= threshold:
                kept_idx.append(i)       # meaningfully different → keep
            else:
                stats["pairs_above_threshold"] += 1   # near-duplicate → discard

        for idx in kept_idx:
            kept_all.append(records[idx])
        stats["samples_after"] += len(kept_idx)

    return kept_all, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate adaptive-sampling CoT training data"
    )
    parser.add_argument(
        "--files", nargs="+", default=DEFAULT_FILES,
        help="JSONL files to deduplicate (default: GSM8K + MATH adaptive correct)",
    )
    parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT,
        help=f"Output merged+shuffled JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Cosine similarity threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"SentenceTransformer model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for encoding (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Skip the final shuffle step",
    )
    args = parser.parse_args()

    # ---- Load embedding model ----
    print(_ts(f"Loading model: {args.model}"))
    t0 = time.perf_counter()
    model = SentenceTransformer(args.model)
    print(_ts(f"Model loaded in {time.perf_counter() - t0:.1f}s"))

    # ---- Process each file ----
    all_kept = []
    grand_stats = {}

    for filepath in args.files:
        label = os.path.basename(filepath)
        print(f"\n{'=' * 60}")
        print(_ts(f"Processing: {label}"))
        print(f"{'=' * 60}")

        # Load & group
        t0 = time.perf_counter()
        groups = load_and_group(filepath)
        n_problems = len(groups)
        n_samples = sum(len(v) for v in groups.values())
        avg = n_samples / n_problems if n_problems else 0
        print(_ts(f"Loaded {n_samples} samples across {n_problems} problems "
                  f"(avg {avg:.2f} samples/problem)"))
        print(_ts(f"Load time: {time.perf_counter() - t0:.1f}s"))

        # Count distribution
        dist = defaultdict(int)
        for recs in groups.values():
            dist[len(recs)] += 1
        for k in sorted(dist):
            print(f"      {k} sample(s): {dist[k]} problems")

        # Deduplicate
        t0 = time.perf_counter()
        kept, stats = deduplicate_groups(groups, model, args.threshold, args.batch_size)
        elapsed = time.perf_counter() - t0
        print(_ts(f"Dedup complete in {elapsed:.1f}s"))
        print(f"      Samples before: {stats['samples_before']}")
        print(f"      Samples after:  {stats['samples_after']}")
        print(f"      Discarded:      {stats['samples_before'] - stats['samples_after']}")
        retention = stats["samples_after"] / stats["samples_before"] * 100
        print(f"      Retention:      {retention:.1f}%")
        if stats["pairs_total"] > 0:
            above_pct = stats["pairs_above_threshold"] / stats["pairs_total"] * 100
            print(f"      Pairs >= {args.threshold}: "
                  f"{stats['pairs_above_threshold']}/{stats['pairs_total']} "
                  f"({above_pct:.1f}%)")

        all_kept.extend(kept)
        grand_stats[label] = stats

    # ---- Merge & shuffle ----
    print(f"\n{'=' * 60}")
    print(_ts("Merge & shuffle"))
    print(f"{'=' * 60}")
    print(f"  Total before dedup: {sum(s['samples_before'] for s in grand_stats.values())}")
    print(f"  Total after dedup:  {len(all_kept)}")
    overall_retention = len(all_kept) / sum(s["samples_before"] for s in grand_stats.values()) * 100
    print(f"  Overall retention:  {overall_retention:.1f}%")

    if not args.no_shuffle:
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(all_kept)
        print(f"  Shuffled (seed={RANDOM_SEED})")

    # ---- Write output ----
    print(f"\n" + _ts(f"Writing {len(all_kept)} records to {args.output}"))
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in all_kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(_ts(f"Done in {time.perf_counter() - t0:.1f}s"))


if __name__ == "__main__":
    main()
