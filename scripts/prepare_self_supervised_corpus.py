"""
prepare_self_supervised_corpus.py — Unstructured Monolingual Text Builder for SSL.

Extracts pure, raw native literature, news, folktales, and dictionaries:
- NO instruction templates
- NO "system", "user", "assistant" tags
- NO synthetic Q&A formatting

Output:
  data/raw/monolingual_ssl_corpus.txt
"""

import io
import json
import logging
import sys
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"
OUTPUT_FILE = DATA_RAW / "monolingual_ssl_corpus.txt"


def main():
    print("=" * 70)
    print("AfriWise: Prepare Pure Self-Supervised Monolingual Corpus")
    print("=" * 70)

    raw_paragraphs = []

    # 1. Ingest raw Igbo corpus (MasakhaNews / MAFAND / BBC Igbo)
    igbo_file = DATA_RAW / "igbo_corpus.jsonl"
    if igbo_file.exists():
        igbo_count = 0
        with open(igbo_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        txt = rec.get("text", "").strip()
                        if len(txt) > 20:
                            raw_paragraphs.append(txt)
                            igbo_count += 1
                    except Exception:
                        pass
        print(f"[OK] Ingested {igbo_count} raw Igbo articles and paragraphs.")

    # 2. Ingest raw Bini corpus
    bini_file = DATA_RAW / "bini_corpus.jsonl"
    if bini_file.exists():
        bini_count = 0
        with open(bini_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        txt = rec.get("text", "").strip()
                        if len(txt) > 10:
                            raw_paragraphs.append(txt)
                            bini_count += 1
                    except Exception:
                        pass
        print(f"[OK] Ingested {bini_count} raw Bini texts.")

    # 3. Ingest raw Efik/Ibibio corpus
    efik_file = DATA_RAW / "efik_corpus.jsonl"
    if efik_file.exists():
        efik_count = 0
        with open(efik_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        txt = rec.get("text", "").strip()
                        if len(txt) > 10:
                            raw_paragraphs.append(txt)
                            efik_count += 1
                    except Exception:
                        pass
        print(f"[OK] Ingested {efik_count} raw Efik/Ibibio texts.")

    # 4. Ingest raw knowledge base cultural folktales and cosmological texts
    if KNOWLEDGE_BASE.exists():
        kb_count = 0
        for md_file in KNOWLEDGE_BASE.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                # Filter out headers, keep raw narrative prose
                for para in content.split("\n\n"):
                    para_clean = para.strip()
                    if len(para_clean) > 30 and not para_clean.startswith("#"):
                        raw_paragraphs.append(para_clean)
                        kb_count += 1
            except Exception:
                pass
        print(f"[OK] Ingested {kb_count} raw cultural paragraphs from knowledge base.")

    # Write pure unstructured text file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in raw_paragraphs:
            f.write(p + "\n\n")

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Monolingual SSL Corpus Created: {OUTPUT_FILE}")
    print(f"Total Paragraphs/Articles: {len(raw_paragraphs):,}")
    print(f"File Size: {OUTPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
