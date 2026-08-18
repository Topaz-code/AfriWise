"""
download_huggingface_african_gems.py — Ingest all discovered HuggingFace datasets for Igbo and Efik.
"""
import io
import json
import logging
import sys
from pathlib import Path
from datasets import load_dataset

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_DIR = Path("data/raw/massive_african_corpus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target datasets to pull
TARGETS = [
    {"name": "bjorndev/efik-dataset", "lang": "efik", "text_cols": ["efik", "text", "translation"]},
    {"name": "bjorndev/efik-synonyms", "lang": "efik", "text_cols": ["word", "synonym", "text"]},
    {"name": "bjorndev/efik-antonyms", "lang": "efik", "text_cols": ["word", "antonym", "text"]},
    {"name": "michsethowusu/english-efik_sentence-pairs_mt560", "lang": "efik", "text_cols": ["efik", "target", "translation"]},
    {"name": "ignatius/igbo_monolingual", "lang": "igbo", "text_cols": ["text", "sentence", "igbo"]},
    {"name": "ignatius/igbo_english_machine_translation", "lang": "igbo", "text_cols": ["igbo", "text", "translation"]},
    {"name": "iamwille/igbo-translation", "lang": "igbo", "text_cols": ["igbo", "text"]},
]

total_ingested = 0

for target in TARGETS:
    d_name = target["name"]
    lang = target["lang"]
    cols = target["text_cols"]
    print(f"\n--> Fetching HuggingFace Dataset: {d_name}...")
    try:
        ds = load_dataset(d_name, trust_remote_code=True)
        count = 0
        extracted_sentences = []
        for split in ds.keys():
            for row in ds[split]:
                # Extract any text column matching
                for c in cols:
                    if c in row and row[c] and isinstance(row[c], str) and len(row[c].strip()) > 5:
                        extracted_sentences.append(row[c].strip())
                        count += 1
                        break
                    elif c in row and isinstance(row[c], dict):
                        # Nested translation dict (e.g. {'en': ..., 'efi': ...})
                        for subk in ["efi", "ig", "ibo", "target"]:
                            if subk in row[c] and row[c][subk]:
                                extracted_sentences.append(row[c][subk].strip())
                                count += 1
                                break
        
        out_file = OUTPUT_DIR / f"{lang}_hf_{d_name.replace('/', '_')}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for s in extracted_sentences:
                f.write(s + "\n")
        
        print(f"[OK] Ingested {len(extracted_sentences):,} records from {d_name} -> {out_file.name}")
        total_ingested += len(extracted_sentences)
    except Exception as e:
        print(f"[WARN] Skipped {d_name}: {e}")

print("\n" + "=" * 60)
print(f"Total New HuggingFace African Records Ingested: {total_ingested:,}")
print("=" * 60)
