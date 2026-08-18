"""
package_edge_mobile_model.py — Repack AfriWise GGUF Model for Mobile Phones / Google Edge Gallery.
"""
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GGUF_SOURCE = PROJECT_ROOT / "models" / "gguf" / "phi-3-mini-4k-instruct-q4_k_m.gguf"
ROOT_GGUF = PROJECT_ROOT / "afriwise_phi3_q4_k_m.gguf"
EDGE_MANIFEST = PROJECT_ROOT / "afriwise_edge_gallery_spec.json"

def package_mobile_model():
    print("=" * 60)
    print("AfriWise: Mobile & Edge Gallery Model Repackaging")
    print("=" * 60)

    # 1. Hardlink or Symlink or Copy GGUF to root
    if GGUF_SOURCE.exists() and not ROOT_GGUF.exists():
        try:
            print(f"Creating link from {GGUF_SOURCE.name} to {ROOT_GGUF.name}...")
            os.link(str(GGUF_SOURCE), str(ROOT_GGUF))
            print(f"[OK] Hardlink created: {ROOT_GGUF.name}")
        except Exception:
            print("Copying GGUF to root...")
            shutil.copy2(str(GGUF_SOURCE), str(ROOT_GGUF))
            print(f"[OK] GGUF copied: {ROOT_GGUF.name}")
    elif ROOT_GGUF.exists():
        print(f"[OK] Mobile model already in root: {ROOT_GGUF.name} ({ROOT_GGUF.stat().st_size / 1024 / 1024:.2f} MB)")

    # 2. Generate Google Edge Gallery / Mobile Runtime Spec
    spec = {
        "model_name": "AfriWise-Phi3-Mini-African-Languages",
        "version": "2.0.0",
        "description": "Culturally grounded, zero-hallucination assistant for Igbo, Edo/Bini, and Efik/Ibibio languages.",
        "architecture": "Phi-3-Mini-4K-Instruct",
        "quantization": "Q4_K_M (4-bit Medium Quantization)",
        "file_name": "afriwise_phi3_q4_k_m.gguf",
        "file_size_bytes": ROOT_GGUF.stat().st_size if ROOT_GGUF.exists() else GGUF_SOURCE.stat().st_size if GGUF_SOURCE.exists() else 0,
        "supported_languages": [
            {"code": "ig", "name": "Igbo", "orthography": "Onwu 1961 Standard"},
            {"code": "bin", "name": "Edo / Bini", "orthography": "Agheyisi 1986 Standard"},
            {"code": "efi", "name": "Efik", "orthography": "Essien 1983 Standard"},
            {"code": "ibb", "name": "Ibibio", "orthography": "Essien 1990 Standard"},
            {"code": "en", "name": "English", "role": "Bridge & Translation"}
        ],
        "hardware_requirements": {
            "ram_minimum_mb": 3072,
            "ram_recommended_mb": 4096,
            "cpu_threads": 4,
            "gpu_acceleration": "OpenCL / Vulkan / Metal / NPU supported"
        },
        "system_prompt": "You are AfriWise — a culturally accurate assistant for southern Nigerian languages (Igbo, Edo/Bini, Efik/Ibibio). Only state facts you are certain of. If unsure, say 'A maghị m' (Igbo) / 'Mmọdiọkke' (Efik) / 'I ma-ẹre' (Bini).",
        "edge_gallery_metadata": {
            "category": "Education / Cultural Heritage",
            "author": "Topaz & AfriWise Project",
            "license": "CC-BY-NC-4.0",
            "offline_capable": True,
            "fts5_grounding_db": "afriwise_fts5.db"
        }
    }

    with open(EDGE_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print(f"[OK] Google Edge Gallery specification created: {EDGE_MANIFEST.name}")
    print("=" * 60)

if __name__ == "__main__":
    package_mobile_model()
