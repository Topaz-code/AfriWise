"""
train_self_supervised_clm.py — Self-Supervised Continual Pretraining (SSL) Engine.

Trains the model via pure Autoregressive Causal Language Modeling (CLM) on raw,
unstructured monolingual literature (Igbo, Bini-Edo, Efik-Ibibio):
- NO synthetic Q&A instruction formatting
- NO prompt-template bias
- Pure next-token prediction loss on native text distribution

Usage:
    python scripts/train_self_supervised_clm.py --corpus data/raw/monolingual_ssl_corpus.txt
"""

import argparse
import io
import logging
import math
import os
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Force UTF-8 on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_CORPUS = Path("data/raw/monolingual_ssl_corpus.txt")
DEFAULT_OUTPUT_DIR = Path("afriwise_ssl_adapters")


class MonolingualBlockDataset(Dataset):
    """Chunks pure text into fixed-length contiguous token blocks for self-supervised CLM."""
    def __init__(self, file_path: Path, tokenizer, block_size: int = 512, max_blocks: int = None):
        self.blocks = []
        logger.info(f"Tokenizing raw monolingual text from: {file_path}")
        text = file_path.read_text(encoding="utf-8")
        tokens = tokenizer(text, truncation=False, return_tensors="pt")["input_ids"].squeeze(0)
        total_tokens = tokens.shape[0]

        # Chunk into non-overlapping blocks
        for i in range(0, total_tokens - block_size, block_size):
            if max_blocks and len(self.blocks) >= max_blocks:
                break
            chunk = tokens[i : i + block_size]
            self.blocks.append({
                "input_ids": chunk,
                "attention_mask": torch.ones_like(chunk),
                "labels": chunk.clone()
            })

        logger.info(f"Created {len(self.blocks)} self-supervised training blocks (block_size={block_size}).")

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("AfriWise: Self-Supervised Continual Pretraining (SSL / CLM)")
    print(f"Device: {device.upper()} | Precision: {'FP16' if device == 'cuda' else 'BFLOAT16'}")
    print(f"Corpus: {args.corpus}")
    print(f"Output Adapter: {args.output_dir}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = MonolingualBlockDataset(
        Path(args.corpus),
        tokenizer=tokenizer,
        block_size=args.block_size,
        max_blocks=args.max_blocks
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()

    trainable_params, all_param = model.get_nb_trainable_parameters()
    print(f"\n[LoRA Architecture]")
    print(f"  Trainable Parameters: {trainable_params:,} ({100 * trainable_params / all_param:.4f}%)")
    print(f"  Total Parameters:     {all_param:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    grad_accum_steps = args.gradient_accumulation_steps
    total_steps = min(len(dataloader) * args.epochs, args.max_steps) if args.max_steps else len(dataloader) * args.epochs

    print(f"\nStarting self-supervised pretraining ({total_steps} steps)...")
    start_time = time.time()
    step = 0
    running_loss = 0.0

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            if args.max_steps and step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            running_loss += loss.item() * grad_accum_steps

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % args.log_interval == 0 or step == total_steps:
                    avg_loss = running_loss / args.log_interval if step % args.log_interval == 0 else running_loss
                    elapsed = time.time() - start_time
                    sps = step / max(1, elapsed)
                    print(f"  [Step {step:04d}/{total_steps:04d}] CLM Loss: {avg_loss:.4f} | Speed: {sps:.2f} step/s")
                    running_loss = 0.0

        if args.max_steps and step >= args.max_steps:
            break

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[OK] Self-Supervised Training Complete in {total_time/60:.2f} minutes.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[OK] Saved adapted SSL weights to: {out_dir.resolve()}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="AfriWise Self-Supervised Continual Pretraining")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="Path to raw text corpus")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for adapters")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps limit")
    parser.add_argument("--max-blocks", type=int, default=1000, help="Max blocks limit")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Grad accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for SSL")
    parser.add_argument("--block-size", type=int, default=384, help="Context block size")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--log-interval", type=int, default=10, help="Log step interval")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
