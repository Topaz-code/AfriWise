"""
Weight Merge & Model Export Script for AfriWise.

Merges the fine-tuned LoRA adapter weights from `afriwise_adapters/` into the
base `microsoft/Phi-3-mini-4k-instruct` model to produce a standalone 16-bit
Hugging Face model directory, ready for GGUF conversion with llama.cpp and Ollama.

Usage:
  python scripts/merge_and_export.py --adapter-dir afriwise_adapters --output-dir afriwise_merged_model
"""

import argparse
import logging
import os
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def merge_lora_weights(
    base_model_id: str,
    adapter_dir: Path,
    output_dir: Path,
    device: str = "cpu"
):
    """Merge LoRA adapter weights with base model and save consolidated model."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        logger.error("Missing dependencies. Run: pip install torch transformers peft")
        sys.exit(1)

    if not adapter_dir.exists():
        logger.error(f"Adapter directory not found at: {adapter_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    abs_adapter_path = str(adapter_dir.resolve())
    logger.info(f"Loading base tokenizer from: {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False, local_files_only=True)

    torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16
    logger.info(f"Loading base model: {base_model_id} ({torch_dtype})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        local_files_only=True,
    )

    logger.info(f"Loading LoRA weights from: {abs_adapter_path}...")
    lora_model = PeftModel.from_pretrained(base_model, abs_adapter_path, local_files_only=True)

    logger.info("Merging LoRA adapter weights into base model...")
    merged_model = lora_model.merge_and_unload()

    logger.info(f"Saving merged model weights to: {output_dir}...")
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    logger.info("✅ Model merge complete!")
    print("\n" + "=" * 70)
    print("🎉 AfriWise Model Successfully Merged & Saved!")
    print(f"📁 Output Directory: {output_dir.resolve()}")
    print("=" * 70)
    print("\n📦 To convert this merged model to GGUF for Ollama:")
    print("1. Clone llama.cpp:")
    print("   git clone https://github.com/ggerganov/llama.cpp.git")
    print("2. Run GGUF conversion script:")
    print(f"   python llama.cpp/convert_hf_to_gguf.py {output_dir.resolve()} --outfile afriwise-phi3-q4_k_m.gguf --outtype q4_k_m")
    print("3. Build Ollama model using the root Modelfile:")
    print("   ollama create afriwise -f Modelfile")
    print("4. Run with Ollama:")
    print("   ollama run afriwise")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Merge AfriWise LoRA weights with base model")
    parser.add_argument("--base-model", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Base model Hugging Face repository")
    parser.add_argument("--adapter-dir", type=Path, default=Path("afriwise_adapters"), help="Path to trained LoRA adapter directory")
    parser.add_argument("--output-dir", type=Path, default=Path("afriwise_merged_model"), help="Output directory for merged model")
    parser.add_argument("--device", type=str, default="cpu", help="Device for merging (cpu or cuda)")
    args = parser.parse_args()

    merge_lora_weights(
        base_model_id=args.base_model,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
