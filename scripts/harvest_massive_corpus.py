"""
harvest_massive_corpus.py — Massive Multi-Source Corpus Harvester & Knowledge Base Compiler.

Compiles ALL harvested datasets:
- 665,120 Efik-English sentence pairs
- 31,058 Efik general vocabulary & dialogue records
- 12,787 Efik synonyms
- 4,123 Efik antonyms
- 30,966 Igbo verses from eBible
- 22,536 Igbo news/literature records from MasakhaNews & MAFAND
- 21,556 Igbo translation pairs
- Melzian Bini-Edo dictionary & cosmological corpus

Outputs:
- data/raw/massive_african_corpus/master_massive_corpus.txt (Millions of tokens for SSL)
- data/processed/afriwise_fts5.db (SQLite FTS5 Full-Text Database for Instant 0-RAM Grounding)
"""

import io
import json
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "massive_african_corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = PROJECT_ROOT / "data" / "processed" / "afriwise_fts5.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("=" * 70)
    print("AfriWise: Master Corpus Compilation (>750,000 Records)")
    print("=" * 70)
    start_time = time.time()

    all_records = []
    seen = set()

    def add_record(text: str, lang: str, src: str):
        c = clean_text(text)
        if len(c) >= 5 and c not in seen:
            seen.add(c)
            all_records.append((c, lang, src))

    # 1. Harvest all files in massive_african_corpus directory
    print("\n[Phase 1] Scanning massive_african_corpus directory...")
    for fpath in CORPUS_DIR.glob("*.*"):
        if fpath.name in ["master_massive_corpus.txt"]:
            continue
        fname = fpath.name.lower()
        lang = "igbo" if "igbo" in fname or "ibo" in fname else ("efik" if "efik" in fname or "efi" in fname else "bini")
        print(f"  Reading: {fpath.name}...")
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        add_record(line.strip(), lang, fpath.stem)
        except Exception as e:
            print(f"  [WARN] Error reading {fpath.name}: {e}")

    # 2. Harvest raw/ legacy files
    print("\n[Phase 2] Scanning legacy data/raw/ files...")
    for raw_f in (PROJECT_ROOT / "data" / "raw").glob("*.jsonl"):
        lang = "igbo" if "igbo" in raw_f.name else ("efik" if "efik" in raw_f.name else "bini")
        try:
            with open(raw_f, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            txt = rec.get("text", "")
                            if txt:
                                add_record(txt, lang, raw_f.stem)
                        except Exception:
                            pass
        except Exception:
            pass

    # 3. Harvest Knowledge Base Cultural Texts
    print("\n[Phase 3] Scanning Knowledge Base...")
    for kb_f in (PROJECT_ROOT / "knowledge_base").glob("*.md"):
        lang = "bini" if "bini" in kb_f.name or "edo" in kb_f.name else ("efik" if "ibibio" in kb_f.name or "efik" in kb_f.name else "igbo")
        content = kb_f.read_text(encoding="utf-8", errors="replace")
        for line in content.splitlines():
            c = clean_text(line)
            if len(c) > 15 and not c.startswith("#"):
                add_record(c, lang, "knowledge_base")

    # Stats
    lang_counts = {}
    for _, l, _ in all_records:
        lang_counts[l] = lang_counts.get(l, 0) + 1

    print("\n" + "=" * 70)
    print(f"🌟 TOTAL DEDUPLICATED AUTHENTIC RECORDS: {len(all_records):,}")
    for l, cnt in lang_counts.items():
        print(f"   - {l.upper()}: {cnt:,} sentences/records")
    print("=" * 70)

    # 4. Write master monolingual corpus for SSL training
    master_file = CORPUS_DIR / "master_massive_corpus.txt"
    print(f"\nWriting Master SSL Corpus: {master_file}...")
    with open(master_file, "w", encoding="utf-8") as f:
        for s, _, _ in all_records:
            f.write(s + "\n")
    print(f"[OK] Master SSL File Size: {master_file.stat().st_size / 1024 / 1024:.2f} MB")

    # 5. Build SQLite FTS5 Database
    print(f"\nCompiling SQLite FTS5 Database: {DB_PATH}...")
    if DB_PATH.exists():
        try:
            DB_PATH.unlink()
        except Exception:
            pass

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS corpus_fts USING fts5(
            sentence,
            language,
            source,
            tokenize='unicode61'
        );
    """)

    # Batch insert in chunks of 50,000 for speed
    chunk_size = 50000
    for i in range(0, len(all_records), chunk_size):
        chunk = all_records[i : i + chunk_size]
        cursor.executemany("INSERT INTO corpus_fts (sentence, language, source) VALUES (?, ?, ?);", chunk)
        conn.commit()
        print(f"  Inserted {min(i + chunk_size, len(all_records)):,}/{len(all_records):,} records...")

    print("  Optimizing FTS5 index...")
    cursor.execute("INSERT INTO corpus_fts(corpus_fts) VALUES('optimize');")
    conn.commit()
    conn.close()

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Master Database Ready: {DB_PATH.stat().st_size / 1024 / 1024:.2f} MB ({elapsed:.2f}s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
