"""
AfriWise LoRA Fine-Tuner for Phi-3 Mini Instruct (CPU-only).

Uses HuggingFace PEFT with LoRA on CPU in FP32.
This is slow but correct. Expected: ~1-3 hours per 100 steps.

Usage:
  python scripts/train_afriwise.py --dataset data/processed/afriwise_v2_chatml.jsonl
  python scripts/train_afriwise.py --dataset data/processed/afriwise_v2_chatml.jsonl --epochs 2 --steps 200
  python scripts/train_afriwise.py --dry-run
"""
import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = "afriwise_adapters_v2"
DEFAULT_DATASET = "data/processed/afriwise_v2_chatml.jsonl"

# CPU-safe LoRA configuration
LORA_CONFIG = {
    "r": 8,               # Low rank — adequate for language grounding, CPU-feasible
    "lora_alpha": 16,     # 2x rank is standard
    "target_modules": [   # Phi-3 attention projection layers
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# CPU Training configuration
TRAINING_CONFIG = {
    "max_seq_length": 512,         # Shorter = faster on CPU
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 20,
    "logging_steps": 10,
    "save_steps": 50,
    "fp16": False,                  # Must be False on CPU
    "bf16": False,                  # Must be False on CPU (no bfloat16 on most CPUs)
    "optim": "adamw_torch",
    "dataloader_num_workers": 0,    # 0 = main process (Windows compatibility)
    "report_to": "none",           # No wandb/tensorboard needed
    "push_to_hub": False,
}


def load_dataset_records(jsonl_path: str, max_records: Optional[int] = None) -> list:
    """Load ChatML-formatted JSONL dataset."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if max_records and i >= max_records:
                break
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {i}: {e}")

    logger.info(f"Loaded {len(records)} training records from {jsonl_path}")
    return records


def chatml_to_prompt(record: dict) -> str:
    """Convert ChatML record to Phi-3 prompt format."""
    messages = record.get("messages", [])
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"<|system|>\n{content}<|end|>\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}<|end|>\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}<|end|>\n"
    return prompt


def main():
    parser = argparse.ArgumentParser(description="AfriWise LoRA Fine-Tuner")
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help=f"JSONL training dataset path (default: {DEFAULT_DATASET})"
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Number of training epochs (default: 1 for CPU; start here)"
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Max training steps (overrides epochs if set)"
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR,
        help=f"Output adapter directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="Limit number of training records (for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load model and dataset but skip training (sanity check)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("AfriWise LoRA Fine-Tuner")
    logger.info(f"  Base model: {BASE_MODEL}")
    logger.info(f"  Dataset:    {args.dataset}")
    logger.info(f"  Output:     {args.output}")
    logger.info(f"  Epochs:     {args.epochs}")
    logger.info(f"  Mode:       {'DRY RUN' if args.dry_run else 'TRAINING'}")
    logger.info("=" * 60)

    # Warnings
    logger.warning("CPU TRAINING NOTE: This will be slow — ~1-3 hours for 100 steps.")
    logger.warning("Close all other applications to maximize available RAM.")

    import torch
    if torch.cuda.is_available():
        logger.info("GPU detected — consider using it for faster training!")
    else:
        logger.info("CPU-only mode confirmed.")

    # ---- Load dataset ----
    records = load_dataset_records(args.dataset, args.max_records)
    prompts = [chatml_to_prompt(r) for r in records]
    logger.info(f"Converted {len(prompts)} records to training prompts.")

    # ---- Load model ----
    logger.info("Loading tokenizer...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=False, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading base model (bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        local_files_only=True,
    )

    # ---- Apply LoRA ----
    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.dry_run:
        logger.info("DRY RUN complete — model and dataset loaded successfully!")
        logger.info("Run without --dry-run to begin training.")
        return

    # ---- Tokenize ----
    logger.info("Tokenizing training data...")
    max_len = TRAINING_CONFIG["max_seq_length"]

    encodings = tokenizer(
        prompts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )

    import torch
    from torch.utils.data import Dataset, DataLoader

    class AfriWiseDataset(Dataset):
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            ids = self.input_ids[idx]
            mask = self.attention_mask[idx]
            # Labels = input_ids; -100 for padding tokens
            labels = ids.clone()
            labels[mask == 0] = -100
            return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    dataset = AfriWiseDataset(encodings["input_ids"], encodings["attention_mask"])

    # ---- Training ----
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

    total_steps = args.steps
    if total_steps is None:
        steps_per_epoch = math.ceil(
            len(dataset) / (
                TRAINING_CONFIG["per_device_train_batch_size"]
                * TRAINING_CONFIG["gradient_accumulation_steps"]
            )
        )
        total_steps = steps_per_epoch * args.epochs

    logger.info(f"Training for {total_steps} steps (~{total_steps // 10} minutes estimated)")

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=total_steps if args.steps else -1,
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        save_total_limit=2,
        fp16=TRAINING_CONFIG["fp16"],
        bf16=TRAINING_CONFIG["bf16"],
        optim=TRAINING_CONFIG["optim"],
        dataloader_num_workers=TRAINING_CONFIG["dataloader_num_workers"],
        report_to=TRAINING_CONFIG["report_to"],
        push_to_hub=TRAINING_CONFIG["push_to_hub"],
        use_cpu=True,     # Force CPU
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    logger.info("Starting training...")
    logger.info("Be patient — CPU training is slow but it works. Press Ctrl+C to stop and save.")

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted — saving current adapter state...")

    # ---- Save ----
    logger.info(f"Saving LoRA adapter to {args.output}...")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    logger.info("=" * 60)
    logger.info(f"Training complete! Adapter saved to: {args.output}")
    logger.info("Next: Run the benchmark to evaluate quality:")
    logger.info(f"  python scripts/benchmark_hallucination.py --adapter {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
