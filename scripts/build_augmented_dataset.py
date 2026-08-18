"""
AfriWise Augmented Dataset Builder (V2).

Combines:
1. Existing V1 training data (data/processed/training_dataset_chatml.jsonl)
2. Raw corpora downloaded from HuggingFace (data/raw/*.jsonl)
3. Verified cultural Q&A pairs (data/processed/cultural_corpus_chatml.jsonl)

Converts raw text into instruction-tuning ChatML pairs.
Deduplicates, shuffles, and writes final dataset.

Output: data/processed/afriwise_v2_chatml.jsonl

Usage:
  python scripts/build_augmented_dataset.py
  python scripts/build_augmented_dataset.py --no-v1  (skip V1 data)
  python scripts/build_augmented_dataset.py --limit 1000
"""
import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are AfriWise — a culturally accurate assistant for southern Nigerian languages "
    "(Ibibio/Efik, Igbo, and Edo/Bini). Only state facts you are certain of. "
    "If unsure, say: 'A maghị m' (Igbo) / 'Mmọdiọkke' (Efik) / 'I ma-ẹre' (Bini). "
    "Preserve all diacritics: ọ, ụ, ị, ẹ, ñ."
)

V1_DATASET = Path("data/processed/training_dataset_chatml.jsonl")
CULTURAL_CORPUS = Path("data/processed/cultural_corpus_chatml.jsonl")
EXTENDED_CORPUS = Path("data/processed/extended_knowledge_chatml.jsonl")
RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/afriwise_v2_chatml.jsonl")

# Templates for converting raw corpus text into instruction pairs
IGBO_TEMPLATES = [
    ("Translate this Igbo sentence to English.", "{text}"),
    ("What does this Igbo text mean?", "{text}"),
    ("Provide an example of natural Igbo speech.", "{text}"),
    ("Write this sentence in Igbo:", "{text}"),
]

EFIK_TEMPLATES = [
    ("Translate this Efik/Ibibio sentence to English.", "{text}"),
    ("What does this Efik or Ibibio phrase mean?", "{text}"),
    ("Give me an example of Ibibio speech.", "{text}"),
    ("How would you say this in Ibibio/Efik?", "{text}"),
]

BINI_TEMPLATES = [
    ("Translate this Bini-Edo sentence to English.", "{text}"),
    ("What does this Bini (Edo) text mean?", "{text}"),
    ("Provide an example of the Bini language.", "{text}"),
    ("Write something in the Bini-Edo language:", "{text}"),
]

LANG_TEMPLATES = {
    "igbo": IGBO_TEMPLATES,
    "efik": EFIK_TEMPLATES,
    "bini": BINI_TEMPLATES,
}


def text_to_record(text: str, language: str, source: str) -> dict:
    """Convert a raw text entry into a ChatML instruction pair."""
    templates = LANG_TEMPLATES.get(language, IGBO_TEMPLATES)
    user_tmpl, assistant_tmpl = random.choice(templates)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_tmpl},
            {"role": "assistant", "content": assistant_tmpl.format(text=text)},
        ],
        "metadata": {"lang": language, "source": source},
    }


def load_jsonl(path: Path) -> list:
    """Load a JSONL file, return list of parsed records."""
    if not path.exists():
        logger.warning(f"File not found, skipping: {path}")
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"  Skipping bad JSON at {path}:{i}: {e}")
    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def dedup(records: list) -> list:
    """Remove duplicate records based on assistant response content."""
    seen = set()
    unique = []
    for r in records:
        messages = r.get("messages", [])
        asst_content = ""
        for m in messages:
            if m.get("role") == "assistant":
                asst_content = m.get("content", "")
                break
        key = asst_content.strip().lower()[:200]
        if key and key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def main():
    parser = argparse.ArgumentParser(description="AfriWise Augmented Dataset Builder")
    parser.add_argument("--no-v1", action="store_true", help="Skip V1 training data")
    parser.add_argument("--no-raw", action="store_true", help="Skip raw downloaded corpora")
    parser.add_argument("--limit", type=int, default=None, help="Limit total records")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    random.seed(args.seed)
    all_records = []

    # ── Phase 1: Cultural & Extended corpus (ALWAYS include — highest quality) ──
    cultural = load_jsonl(CULTURAL_CORPUS)
    all_records.extend(cultural)
    logger.info(f"Cultural corpus: {len(cultural)} records")

    extended = load_jsonl(EXTENDED_CORPUS)
    all_records.extend(extended)
    logger.info(f"Extended knowledge corpus: {len(extended)} records")

    # ── Phase 2: V1 existing training data ──
    if not args.no_v1:
        v1 = load_jsonl(V1_DATASET)
        all_records.extend(v1)
        logger.info(f"V1 dataset: {len(v1)} records")

    # ── Phase 3: Raw downloaded corpora → instruction pairs ──
    if not args.no_raw:
        for lang in ["igbo", "efik", "bini"]:
            raw_path = RAW_DIR / f"{lang}_corpus.jsonl"
            raw_records = load_jsonl(raw_path)
            converted = 0
            for rec in raw_records:
                text = rec.get("text", "").strip()
                source = rec.get("source", lang)
                if len(text) >= 30:
                    record = text_to_record(text, lang, source)
                    all_records.append(record)
                    converted += 1
            logger.info(f"  Converted {converted} {lang} raw texts → instruction pairs")

    # ── Phase 4: Dedup + shuffle ──
    before = len(all_records)
    all_records = dedup(all_records)
    logger.info(f"After dedup: {before} → {len(all_records)} records")

    random.shuffle(all_records)

    if args.limit:
        all_records = all_records[:args.limit]
        logger.info(f"Limited to {len(all_records)} records")

    # ── Write output ──
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("=" * 60)
    logger.info(f"Final dataset: {len(all_records)} records → {OUTPUT_FILE}")

    # Language breakdown
    lang_count = {}
    for r in all_records:
        meta = r.get("metadata", {})
        lang = meta.get("lang", "unknown")
        lang_count[lang] = lang_count.get(lang, 0) + 1

    for lang, count in sorted(lang_count.items()):
        logger.info(f"  {lang}: {count} records ({100*count/len(all_records):.1f}%)")

    logger.info("=" * 60)
    logger.info("Next step: python scripts/train_afriwise.py")


if __name__ == "__main__":
    main()
