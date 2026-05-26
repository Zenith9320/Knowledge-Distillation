#!/usr/bin/env python3
"""
Remove near-duplicate answers between two JSONL files by computing
cosine similarity on embeddings. For each problem, if the two
answers are too similar (cosine similarity > threshold), only one
is kept.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

# Enable faster HF downloads if available
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _ts(msg: str) -> str:
    """Prefix message with a timestamp."""
    return f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"


def main():
    parser = argparse.ArgumentParser(description="Deduplicate JSONL answers by embedding similarity")
    parser.add_argument("file1", help="First JSONL file (preferred source)")
    parser.add_argument("file2", help="Second JSONL file")
    parser.add_argument(
        "-o", "--output", default="data/gsm8k_train_cot_dedup.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.92,
        help="Cosine similarity threshold (default: 0.92)"
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for encoding"
    )
    args = parser.parse_args()

    log = logging.getLogger("dedup")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )

    # ---- Step 1: load JSONL ----
    log.info("Loading %s ...", args.file1)
    t0 = time.perf_counter()
    with open(args.file1) as f:
        data1 = [json.loads(line) for line in f]
    log.info("Loaded %d records from %s (%.1fs)", len(data1), args.file1, time.perf_counter() - t0)

    log.info("Loading %s ...", args.file2)
    t0 = time.perf_counter()
    with open(args.file2) as f:
        data2 = [json.loads(line) for line in f]
    log.info("Loaded %d records from %s (%.1fs)", len(data2), args.file2, time.perf_counter() - t0)

    answers1 = [d["answer"] for d in data1]
    answers2 = [d["answer"] for d in data2]

    total_chars_1 = sum(len(a) for a in answers1)
    total_chars_2 = sum(len(a) for a in answers2)
    avg_len_1 = total_chars_1 / len(answers1)
    avg_len_2 = total_chars_2 / len(answers2)
    log.info("Answer stats — file1: avg_len=%.0f chars, file2: avg_len=%.0f chars", avg_len_1, avg_len_2)

    # ---- Step 2: load model ----
    log.info("Loading model: %s （first run will download ~90MB, cached runs are instant）", args.model)
    log.info("  Cache dir: %s", os.path.expanduser("~/.cache/huggingface/hub/"))
    log.info("  Model loading...")
    t0 = time.perf_counter()
    model = SentenceTransformer(args.model)
    log.info("Model loaded in %.1fs", time.perf_counter() - t0)

    # ---- Step 3: encode ----
    log.info(
        "Encoding %d answers from file1 (batch_size=%d, normalize=True) ...",
        len(answers1), args.batch_size,
    )
    t0 = time.perf_counter()
    emb1 = model.encode(
        answers1, batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - t0
    log.info("File1 encoded: %d vectors in %.1fs (%.0f answers/s)", len(emb1), elapsed, len(emb1) / elapsed)

    log.info(
        "Encoding %d answers from file2 (batch_size=%d, normalize=True) ...",
        len(answers2), args.batch_size,
    )
    t0 = time.perf_counter()
    emb2 = model.encode(
        answers2, batch_size=args.batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.perf_counter() - t0
    log.info("File2 encoded: %d vectors in %.1fs (%.0f answers/s)", len(emb2), elapsed, len(emb2) / elapsed)

    # ---- Step 4: cosine similarity & filter ----
    log.info("Computing cosine similarity & filtering (threshold=%.3f) ...", args.threshold)
    kept = 0
    discarded = 0
    results = []
    sims = np.empty(len(data1), dtype=np.float32)

    n_pair = min(len(data1), len(data2))

    t0 = time.perf_counter()
    for i in range(n_pair):
        sim = float(np.dot(emb1[i], emb2[i]))
        sims[i] = sim
        results.append(data1[i])
        kept += 1
        if sim <= args.threshold:
            results.append(data2[i])
            kept += 1
        else:
            discarded += 1

    # Append remaining records from the longer file
    tail_label = ""
    if len(data1) > n_pair:
        tail = data1[n_pair:]
        tail_label = "file1"
    elif len(data2) > n_pair:
        tail = data2[n_pair:]
        tail_label = "file2"
    else:
        tail = []

    if tail:
        results.extend(tail)
        kept += len(tail)
        log.info("Appended %d remaining records from %s", len(tail), tail_label)

    log.info("Similarity computed in %.1fs", time.perf_counter() - t0)

    # Print similarity distribution
    log.info(
        "Similarity stats — min=%.4f  max=%.4f  mean=%.4f  median=%.4f  std=%.4f",
        sims.min(), sims.max(), sims.mean(), float(np.median(sims)), sims.std(),
    )
    bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.85), (0.85, 0.90),
            (0.90, 0.92), (0.92, 0.95), (0.95, 0.97), (0.97, 0.98), (0.98, 0.99), (0.99, 1.0)]
    log.info("Distribution:")
    for lo, hi in bins:
        cnt = int(((sims >= lo) & (sims < hi)).sum())
        log.info("  [%.2f, %.2f): %d", lo, hi, cnt)

    # ---- Step 5: write output ----
    log.info("Writing %d records to %s ...", len(results), args.output)
    t0 = time.perf_counter()
    with open(args.output, "w") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info("Write done in %.1fs", time.perf_counter() - t0)

    log.info("=" * 50)
    log.info(
        "Done.  Kept: %d  |  Discarded: %d  |  Total: %d  |  Retention: %.1f%%",
        kept, discarded, len(data1) + len(data2),
        kept / (len(data1) + len(data2)) * 100,
    )


if __name__ == "__main__":
    main()
