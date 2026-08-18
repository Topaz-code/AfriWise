"""
AfriWise Environment Verification Script.
Verifies all required packages are installed and system has enough RAM.
"""
import sys
import importlib
import psutil

REQUIRED_PACKAGES = [
    "torch", "transformers", "peft", "datasets", "trl",
    "accelerate", "sentencepiece", "huggingface_hub"
]

def check():
    print("=" * 60)
    print("AfriWise Environment Verification")
    print("=" * 60)
    print(f"Python: {sys.version}")
    
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()} (expected: False for CPU build)")
    
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / 1024**3:.1f} GB")
    print(f"Available RAM: {mem.available / 1024**3:.1f} GB")
    
    if mem.available < 6 * 1024**3:
        print("WARNING: Less than 6 GB RAM available -- close other apps before training!")
    
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "OK")
            print(f"  OK {pkg}: {ver}")
        except ImportError:
            print(f"  MISSING: {pkg}")
    
    print("=" * 60)
    print("Environment check complete.")

if __name__ == "__main__":
    check()
