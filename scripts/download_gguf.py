"""
download_gguf.py - Download Phi-3 Mini Q4_K_M GGUF for CPU inference.

This GGUF file is ~2.3 GB and loads fine even with only 2-3 GB free RAM.
Much lighter than the full bfloat16 model (~7.6 GB).

Usage:
    python scripts/download_gguf.py
"""
import io
import os
import sys
import urllib.request
from pathlib import Path

# Force UTF-8 output on Windows to avoid CP1252 UnicodeEncodeError
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

GGUF_DIR = Path("models/gguf")
GGUF_FILE = GGUF_DIR / "phi-3-mini-4k-instruct-q4_k_m.gguf"

# Primary: official Microsoft GGUF on HuggingFace
HF_URL = (
    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf"
    "/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
)
# Fallback: bartowski's well-tested Q4_K_M conversion
HF_FALLBACK = (
    "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF"
    "/resolve/main/Phi-3-mini-4k-instruct-Q4_K_M.gguf"
)


def download_with_progress(url: str, dest: Path) -> bool:
    """Download with ASCII progress bar (safe for all terminals)."""
    print(f"\nDownloading: {url}")
    print(f"Destination: {dest}")
    try:
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / 1024 / 1024
                total_mb = total_size / 1024 / 1024
                bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
                print(f"\r  [{bar}] {pct}%  {mb:.0f}/{total_mb:.0f} MB", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print(f"\n[OK] Downloaded: {dest.name} ({dest.stat().st_size / 1024**3:.2f} GB)")
        return True
    except Exception as e:
        print(f"\n[FAILED]: {e}")
        if dest.exists():
            dest.unlink()  # Remove partial download
        return False


def main():
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    if GGUF_FILE.exists():
        size_gb = GGUF_FILE.stat().st_size / 1024**3
        print(f"[OK] GGUF already exists: {GGUF_FILE} ({size_gb:.2f} GB)")
        if size_gb < 1.0:
            print("   File looks too small - re-downloading.")
            GGUF_FILE.unlink()
        else:
            print("   Ready to use. Run: python scripts/benchmark_ctransformers.py")
            return

    # Try primary URL
    ok = download_with_progress(HF_URL, GGUF_FILE)
    if not ok:
        print("\nTrying fallback URL...")
        ok = download_with_progress(HF_FALLBACK, GGUF_FILE)

    if not ok:
        print("\n[WARNING] Automatic download failed. Please download manually:")
        print("   1. Open Brave browser")
        print("   2. Go to: https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF")
        print("   3. Download: Phi-3-mini-4k-instruct-Q4_K_M.gguf")
        print(f"   4. Save to: {GGUF_FILE.resolve()}")
        sys.exit(1)

    print("\n[OK] GGUF download complete.")
    print("   Next step: python scripts/benchmark_ctransformers.py")


if __name__ == "__main__":
    main()
