"""
Resumable & Robust Phi-3 Mini Model Downloader.

Downloads microsoft/Phi-3-mini-4k-instruct files individually with retry logic,
resumable chunks, and clear progress reporting without unicode print issues.

Usage:
  python scripts/download_phi3_weights.py
"""
import sys
import time
import io

# Ensure UTF-8 output even on Windows consoles
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from huggingface_hub import hf_hub_download

REPO_ID = "microsoft/Phi-3-mini-4k-instruct"
FILES_TO_DOWNLOAD = [
    "config.json",
    "generation_config.json",
    "modeling_phi3.py",
    "configuration_phi3.py",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "model.safetensors.index.json",
    "model-00002-of-00002.safetensors",
    "model-00001-of-00002.safetensors",
]

def main():
    print("=" * 60)
    print(f"Downloading model: {REPO_ID}")
    print("=" * 60)
    
    for filename in FILES_TO_DOWNLOAD:
        print(f"\n[Fetching] {filename}...")
        success = False
        attempts = 0
        while not success and attempts < 5:
            attempts += 1
            try:
                path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                )
                print(f"  [OK] {filename} ready -> {path}")
                success = True
            except Exception as e:
                print(f"  [Attempt {attempts}/5 failed]: {e}")
                time.sleep(3)
        if not success:
            print(f"CRITICAL: Failed to download {filename} after 5 attempts.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("All Phi-3 Mini 4K Instruct weights downloaded successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
