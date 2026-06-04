#!/usr/bin/env python3
"""
Fine-tune the Qwen2.5-0.5B **base** model on Chain-of-Thought training data.

Unlike the Instruct variant, the base model has no pre-existing math reasoning
patterns, so all accuracy gains from distillation are genuine rather than
recovery from distribution-shift interference.

Uses the <|im_start|> chat template (natively supported by Qwen2.5 base)
so that training and inference share the same format.  Label masking is
applied: only assistant-response tokens contribute to the loss.

Usage:
    # Full fine-tune (default)
    python finetune_student_base.py

    # LoRA fine-tune (lower memory)
    python finetune_student_base.py --lora

    # Resume from the latest checkpoint
    python finetune_student_base.py --resume

    # Resume from a specific checkpoint
    python finetune_student_base.py --checkpoint models/student-sft-base/checkpoint-500

    # Custom data / hyperparameters
    python finetune_student_base.py --data data/train_cot_correct_merged.jsonl \\
                                    --epochs 5 --batch_size 8 --lr 1e-5
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"          # base model (no instruction tuning)
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "train_cot_correct_merged.jsonl")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "models", "student-sft-base")

# Qwen2.5 base was pre-trained with <|im_start|> tokens, so the chat template
# works out-of-the-box.  We use a minimal system prompt since the base model
# hasn't been instruction-tuned.
DEFAULT_SYSTEM_PROMPT = (
    "Solve the problem step by step. "
    "Show your reasoning clearly. "
    "Put your final answer in \\boxed{}."
)


# ---------------------------------------------------------------------------
#  Data loading & tokenization
# ---------------------------------------------------------------------------

def load_jsonl_dataset(path: str) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def tokenize(examples: dict, tokenizer, max_length: int, system_prompt: str) -> dict:
    """Tokenize problem+answer pairs with label masking.

    Only the assistant response (CoT answer) contributes to the loss.
    Prompt tokens (system + user) are masked with label=-100.
    """
    input_ids_list = []
    labels_list = []

    for problem, answer in zip(examples["problem"], examples["answer"]):
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem: {problem}"},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": answer},
        ]

        prompt_tokens = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=False,
            return_dict=False,
        )
        full_tokens = tokenizer.apply_chat_template(
            full_messages, tokenize=True, add_generation_prompt=False,
            return_dict=False,
        )

        prompt_len = len(prompt_tokens)

        # Truncate from the left if too long (keep assistant portion intact)
        if len(full_tokens) > max_length:
            full_tokens = full_tokens[-max_length:]
            prompt_len = min(prompt_len, len(full_tokens))

        labels = [-100] * prompt_len + full_tokens[prompt_len:]

        pad_len = max_length - len(full_tokens)
        pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
        full_tokens += [pad_token] * pad_len
        labels += [-100] * pad_len

        input_ids_list.append(full_tokens)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "labels": labels_list}


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, use_lora: bool = False):
    """Load base student model and tokenizer.  Optionally wrap with LoRA."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  LoRA enabled: {trainable:,} trainable / {total:,} total params "
              f"({trainable / total * 100:.1f}%)")

    return model, tokenizer


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-0.5B base model on CoT math reasoning data"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Base model name or path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--data", default=DEFAULT_DATA,
        help="Path to merged training JSONL",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output directory for the fine-tuned model",
    )
    parser.add_argument(
        "--lora", action="store_true",
        help="Use LoRA for memory-efficient fine-tuning",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Per-device training batch size (default: 4)",
    )
    parser.add_argument(
        "--gradient_accumulation", type=int, default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--max_length", type=int, default=1024,
        help="Max token length for training samples (default: 1024)",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.05,
        help="Warmup ratio (default: 0.05)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Save checkpoint every N steps (default: 500)",
    )
    parser.add_argument(
        "--eval_split", type=float, default=0.05,
        help="Fraction of data to use for eval (default: 0.05)",
    )
    parser.add_argument(
        "--system_prompt", default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for training (default: minimal step-by-step prompt)",
    )
    parser.add_argument(
        "--no_system_prompt", action="store_true",
        help="Omit system prompt entirely (some base models perform better without one)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint in the output directory",
    )
    parser.add_argument(
        "--checkpoint",
        help="Resume training from a specific checkpoint directory (overrides --resume)",
    )
    args = parser.parse_args()

    system_prompt = "" if args.no_system_prompt else args.system_prompt

    # 1. Load dataset
    print(f"Loading dataset: {args.data}")
    dataset = load_jsonl_dataset(args.data)
    print(f"  {len(dataset)} samples")

    # 2. Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, use_lora=args.lora)

    # 3. Tokenize
    print("Tokenizing ...")
    dataset = dataset.select_columns(["problem", "answer"])
    dataset = dataset.train_test_split(test_size=args.eval_split, seed=42)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    train_ds = train_ds.map(
        lambda x: tokenize(x, tokenizer, args.max_length, system_prompt),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda x: tokenize(x, tokenizer, args.max_length, system_prompt),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    # 4. Training
    effective_batch_size = args.batch_size * args.gradient_accumulation
    print(f"\nTraining config:")
    print(f"  Model:     {args.model} (base)")
    print(f"  System prompt: {'(none)' if args.no_system_prompt else args.system_prompt[:60] + '...'}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch_size} × {args.gradient_accumulation} = {effective_batch_size}")
    print(f"  LR:        {args.lr}")
    print(f"  Max len:   {args.max_length}")
    print(f"  Train:     {len(train_ds)} samples, Eval: {len(eval_ds)} samples")

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=50,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Resolve checkpoint path
    checkpoint = None
    if args.checkpoint:
        checkpoint = args.checkpoint
    elif args.resume:
        checkpoint = True  # Auto-find latest in output_dir

    print("\nStarting training ...")
    if checkpoint:
        label = checkpoint if isinstance(checkpoint, str) else "auto (latest)"
        print(f"  Resuming from checkpoint: {label}")
    trainer.train(resume_from_checkpoint=checkpoint)

    # 5. Save
    print(f"\nSaving model to {args.output} ...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
