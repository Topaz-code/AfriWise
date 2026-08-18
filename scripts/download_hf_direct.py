"""
download_hf_direct.py — Directly download raw data files from HuggingFace repositories.
"""
import io
import json
import logging
import sys
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_DIR = Path("data/raw/massive_african_corpus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
api = HfApi()

REPOS = [
    {"repo_id": "bjorndev/efik-dataset", "lang": "efik"},
    {"repo_id": "bjorndev/efik-synonyms", "lang": "efik"},
    {"repo_id": "bjorndev/efik-antonyms", "lang": "efik"},
    {"repo_id": "michsethowusu/english-efik_sentence-pairs_mt560", "lang": "efik"},
    {"repo_id": "ignatius/igbo_monolingual", "lang": "igbo"},
    {"repo_id": "ignatius/igbo_english_machine_translation", "lang": "igbo"},
    {"repo_id": "iamwille/igbo-translation", "lang": "igbo"},
]

total_lines = 0

for repo in REPOS:
    repo_id = repo["repo_id"]
    lang = repo["lang"]
    print(f"\n--> Inspecting repository: {repo_id}...")
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        print(f"  Files found: {files}")
        for fname in files:
            if fname.endswith((".csv", ".tsv", ".json", ".jsonl", ".parquet", ".txt")):
                print(f"  Downloading: {fname}...")
                local_path = hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset")
                dest_file = OUTPUT_DIR / f"{lang}_{repo_id.replace('/', '_')}_{Path(fname).name}"
                
                # Extract text lines
                count = 0
                lines = []
                if fname.endswith(".parquet"):
                    df = pd.read_parquet(local_path)
                    for col in df.columns:
                        for val in df[col].dropna():
                            if isinstance(val, str) and len(val.strip()) > 3:
                                lines.append(val.strip())
                                count += 1
                elif fname.endswith((".csv", ".tsv")):
                    sep = "\t" if fname.endswith(".tsv") else ","
                    try:
                        df = pd.read_csv(local_path, sep=sep, on_bad_lines="skip")
                        for col in df.columns:
                            for val in df[col].dropna():
                                if isinstance(val, str) and len(val.strip()) > 3:
                                    lines.append(val.strip())
                                    count += 1
                    except Exception:
                        pass
                elif fname.endswith((".json", ".jsonl", ".txt")):
                    content = Path(local_path).read_text(encoding="utf-8", errors="replace")
                    for l in content.splitlines():
                        if len(l.strip()) > 3:
                            lines.append(l.strip())
                            count += 1

                if lines:
                    with open(dest_file, "w", encoding="utf-8") as out_f:
                        for line in lines:
                            out_f.write(line + "\n")
                    print(f"  [OK] Saved {count:,} records -> {dest_file.name}")
                    total_lines += count
    except Exception as e:
        print(f"  [FAIL] Failed on {repo_id}: {e}")

print("\n" + "=" * 60)
print(f"Total Direct Ingested African Records: {total_lines:,}")
print("=" * 60)
