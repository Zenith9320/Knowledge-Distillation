"""
Generate Chain-of-Thought solutions using the teacher model.

The model generates a full solution for each problem. DeepSeek-R1's internal
<think>...</think> reasoning is stripped, keeping only the actual CoT answer.

Output:  {"problem": ..., "answer": "<cleaned CoT solution>"}

Usage:
    python generate_CoT.py                          # both datasets, full
    python generate_CoT.py --dataset gsm8k --max_samples 10
    python generate_CoT.py --dataset math --max_samples 50
"""

import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DATASET_CONFIG = {
    "gsm8k": {
        "input_file": "gsm8k_train.jsonl",
        "output_file": "gsm8k_train_cot.jsonl",
    },
    "math": {
        "input_file": "math_train.jsonl",
        "output_file": "math_train_cot.jsonl",
    },
}


def clean_deepseek_tags(text: str) -> str:
    """Remove everything before and including </think> from generated text.

    DeepSeek-R1 wraps its internal reasoning in <think>...</think> tags.
    The content after </think> is the actual Chain-of-Thought answer that
    should be kept for distillation.
    """
    idx = text.find("</think>")
    if idx != -1:
        text = text[idx + len("</think>"):]
    return text.strip()


def generate(
    model,
    tokenizer,
    content: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
) -> str:
    """Generate text from the model given a user-content string."""
    messages = [{"role": "user", "content": content}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return (generated_text)


def load_model(model_name: str):
    """Load the teacher model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded on {device}")
    return model, tokenizer


def process_dataset(
    model,
    tokenizer,
    dataset_key: str,
    max_samples: int | None = None,
    max_new_tokens: int = 2048,
):
    """CoT generation for a single dataset."""
    config = DATASET_CONFIG[dataset_key]
    input_path = os.path.join(DATA_DIR, config["input_file"])
    output_path = os.path.join(DATA_DIR, config["output_file"])

    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    print(f"\nProcessing {dataset_key}: {len(records)} problems -> {output_path}")

    with open(output_path, "w", encoding="utf-8") as f_out:
        for record in tqdm(records, desc=dataset_key):
            problem = record["problem"]

            rationale_raw = generate(
                model, tokenizer, problem, max_new_tokens=max_new_tokens
            )
            rationale = clean_deepseek_tags(rationale_raw)

            answer = f"{rationale}."

            out = {"problem": problem, "answer": answer}
            if "type" in record:
                out["type"] = record["type"]
            if "level" in record:
                out["level"] = record["level"]
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            f_out.flush()

    print(f"  Done: {len(records)} generated")


def main():
    parser = argparse.ArgumentParser(
        description="Chain-of-Thought generation with the teacher model"
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "math", "both"],
        default="both",
        help="Which dataset to process (default: both)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit to N samples per dataset (for testing)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max tokens for generation (default: 2048)",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Teacher model to use (default: {MODEL_NAME})",
    )
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    model, tokenizer = load_model(args.model)

    datasets = ["gsm8k", "math"] if args.dataset == "both" else [args.dataset]

    for ds_key in datasets:
        process_dataset(
            model,
            tokenizer,
            ds_key,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
