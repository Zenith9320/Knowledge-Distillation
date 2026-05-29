#!/usr/bin/env python3
"""
Fine-tune a base student model (no chat template) on CoT training data.

Uses plain text format: Question: <problem> Answer: <cot solution>
All tokens contribute to the loss.

Usage:
    python finetune_student_base.py
    python finetune_student_base.py --model Qwen/Qwen2.5-0.5B
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


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "train_cot_correct_merged.jsonl")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "models", "student-sft-base")

TEMPLATE = "Question: {problem}\nAnswer: {answer}"


def load_jsonl_dataset(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def tokenize(examples: dict, tokenizer, max_length: int) -> dict:
    """Tokenize question+answer pairs as plain text. All tokens contribute to loss."""
    texts = [
        TEMPLATE.format(problem=p, answer=a)
        for p, a in zip(examples["problem"], examples["answer"])
    ]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
    return tokenized


def load_model_and_tokenizer(model_name: str):
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

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a base student model on CoT math reasoning data"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_split", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Loading dataset: {args.data}")
    dataset = load_jsonl_dataset(args.data)
    print(f"  {len(dataset)} samples")

    model, tokenizer = load_model_and_tokenizer(args.model)

    print("Tokenizing ...")
    dataset = dataset.train_test_split(test_size=args.eval_split, seed=42)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]

    train_ds = train_ds.map(
        lambda x: tokenize(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda x: tokenize(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    effective_batch_size = args.batch_size * args.gradient_accumulation
    print(f"\nTraining config:")
    print(f"  Model: {args.model} (base)")
    print(f"  Format: Question: ... Answer: ...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} × {args.gradient_accumulation} = {effective_batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  Max length: {args.max_length}")
    print(f"  Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

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

    print("\nStarting training ...")
    trainer.train()

    print(f"\nSaving model to {args.output} ...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
