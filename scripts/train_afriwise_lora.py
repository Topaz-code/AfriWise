"""
train_afriwise_lora.py — Native Fine-Tuning Engine for AfriWise (Phi-3 Mini + LoRA).

Trains LoRA adapter weights directly on the compiled master African language dataset:
- Igbo (Omenala, MAFAND-MT, Proverbs, Grammar)
- Efik / Ibibio (Ibom-MT, Greetings, Cosmology)
- Bini / Edo (Melzian, Creation Myth, Consonant Digraphs, Proverbs)
- Zero-Hallucination Refusal Pairs

Usage:
    python scripts/train_afriwise_lora.py --epochs 1 --max-steps 100 --batch-size 1
"""

import argparse
import io
import json
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

# Force UTF-8 logging on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("data/processed/afriwise_master_training_dataset.jsonl")
DEFAULT_ADAPTER_DIR = Path("afriwise_adapters")
BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


class AfriWiseChatDataset(Dataset):
    """Dataset class parsing ChatML instruction records into tokenized inputs."""
    def __init__(self, jsonl_path: Path, tokenizer, max_seq_len: int = 512, max_samples: int = None):
        self.examples = []
        if not jsonl_path.exists():
            # Fallback to V2 dataset if master not yet compiled
            jsonl_path = Path("data/processed/afriwise_v2_chatml.jsonl")
        
        logger.info(f"Loading training data from: {jsonl_path}")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and len(self.examples) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    msgs = rec.get("messages", [])
                    if len(msgs) >= 2:
                        sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
                        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                        asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                        
                        if user_msg and asst_msg:
                            # Format as standard ChatML
                            text = (
                                f"<|system|>\n{sys_msg}<|end|>\n"
                                f"<|user|>\n{user_msg}<|end|>\n"
                                f"<|assistant|>\n{asst_msg}<|end|>"
                            )
                            tokens = tokenizer(
                                text,
                                max_length=max_seq_len,
                                truncation=True,
                                padding="max_length",
                                return_tensors="pt"
                            )
                            input_ids = tokens["input_ids"].squeeze(0)
                            attention_mask = tokens["attention_mask"].squeeze(0)
                            labels = input_ids.clone()
                            labels[labels == tokenizer.pad_token_id] = -100
                            self.examples.append({
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                                "labels": labels
                            })
                except Exception:
                    continue

        logger.info(f"Loaded {len(self.examples)} tokenized training examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("AfriWise: LoRA Fine-Tuning Engine")
    print(f"Device: {device.upper()} | Precision: {'FP16' if device == 'cuda' else 'BFLOAT16'}")
    print(f"Dataset: {args.dataset}")
    print(f"Adapter Output Directory: {args.output_dir}")
    print("=" * 70)

    logger.info(f"Loading tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load Dataset
    dataset = AfriWiseChatDataset(
        Path(args.dataset),
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Load Base Model
    torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    logger.info(f"Loading base model: {BASE_MODEL_ID} (torch_dtype={torch_dtype})")
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch_dtype,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )

    # 3. Configure LoRA Adapter
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
    print(f"\n[LoRA Model Architecture]")
    print(f"  Trainable Parameters: {trainable_params:,} ({100 * trainable_params / all_param:.4f}%)")
    print(f"  Total Parameters:     {all_param:,}")

    # 4. Training Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    grad_accum_steps = args.gradient_accumulation_steps
    total_steps = min(len(dataloader) * args.epochs, args.max_steps) if args.max_steps else len(dataloader) * args.epochs
    
    print(f"\nStarting training run: {total_steps} total optimization steps...")
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
                    eta_sec = (total_steps - step) / max(0.001, sps)
                    print(f"  [Step {step:04d}/{total_steps:04d}] Loss: {avg_loss:.4f} | Speed: {sps:.2f} step/s | ETA: {eta_sec/60:.1f}m")
                    running_loss = 0.0

        if args.max_steps and step >= args.max_steps:
            break

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[OK] Training complete in {total_time/60:.2f} minutes ({step} steps).")

    # 5. Save LoRA Adapters
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[OK] Saved fine-tuned LoRA weights to: {out_dir.resolve()}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="AfriWise LoRA Fine-Tuning")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to JSONL dataset")
    parser.add_argument("--output-dir", default=str(DEFAULT_ADAPTER_DIR), help="Directory to save adapter")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps limit")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max training samples limit")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per step")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Grad accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=384, help="Max token sequence length")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA Rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA Alpha")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between log output")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
