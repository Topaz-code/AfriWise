"""
AfriWise Dataset Downloader.

Downloads open-source corpora for Igbo, Bini-Edo, and Efik-Ibibio
from HuggingFace Hub and saves them as JSONL files.

Format of each JSONL line:
  {"text": "<native language text>", "language": "<lang>", "source": "<dataset>"}

Usage:
  python scripts/download_datasets.py
  python scripts/download_datasets.py --igbo-only
  python scripts/download_datasets.py --skip-download  (use cached files)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

MIN_TEXT_LEN = 20   # Skip very short strings
MAX_PER_SOURCE = 5000  # Cap records per source to keep manageable


def clean_text(text: str) -> str:
    """Basic cleanup of raw text."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove BOM and weird unicode control characters
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "")
    return text


def is_valid(text: str) -> bool:
    """Filter out garbage entries."""
    if not text or len(text) < MIN_TEXT_LEN:
        return False
    if text.isdigit() or text.startswith("http"):
        return False
    return True


def save_corpus(records: list, path: Path, lang: str):
    """Save records to JSONL file."""
    valid = [r for r in records if is_valid(r.get("text", ""))]
    with open(path, "w", encoding="utf-8") as f:
        for rec in valid:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(valid)} {lang} records → {path}")
    return len(valid)


def download_igbo():
    """Download Igbo corpus from multiple HuggingFace sources."""
    from datasets import load_dataset
    records = []

    # Source 1: MAFAND MT (Igbo-English machine translation dataset)
    try:
        logger.info("Downloading masakhane/mafand (Igbo)...")
        ds = load_dataset("masakhane/mafand", "en-ibo", split="train", trust_remote_code=True)
        for row in ds:
            translation = row.get("translation", {})
            igbo_text = translation.get("ibo") or translation.get("ig", "")
            igbo_text = clean_text(igbo_text)
            if igbo_text:
                records.append({"text": igbo_text, "language": "igbo", "source": "mafand-mt"})
            if len(records) >= MAX_PER_SOURCE:
                break
        logger.info(f"  Got {len(records)} from MAFAND")
    except Exception as e:
        logger.warning(f"  MAFAND download failed: {e}")

    # Source 2: AfriXNLI (Igbo)
    before = len(records)
    try:
        logger.info("Downloading masakhane/afrixnli (Igbo)...")
        ds = load_dataset("masakhane/afrixnli", split="validation", trust_remote_code=True)
        count = 0
        for row in ds:
            if row.get("language", "") == "ibo":
                for field in ["premise", "hypothesis"]:
                    text = clean_text(row.get(field, ""))
                    if text:
                        records.append({"text": text, "language": "igbo", "source": "afrixnli"})
                count += 1
                if count >= 2000:
                    break
        logger.info(f"  Got {len(records) - before} from AfriXNLI")
    except Exception as e:
        logger.warning(f"  AfriXNLI download failed: {e}")

    # Source 3: Masakhane NER 2 (Igbo)
    before = len(records)
    try:
        logger.info("Downloading masakhane/masakhaner2 (Igbo)...")
        ds = load_dataset("masakhane/masakhaner2", "ibo", split="train", trust_remote_code=True)
        for row in ds:
            tokens = row.get("tokens", [])
            if tokens:
                text = clean_text(" ".join(tokens))
                if text:
                    records.append({"text": text, "language": "igbo", "source": "masakhaner2"})
            if len(records) > 10000:
                break
        logger.info(f"  Got {len(records) - before} from MasakhaNER2")
    except Exception as e:
        logger.warning(f"  MasakhaNER2 download failed: {e}")

    # Source 4: MasakhaNews (Igbo) - headline + body
    before = len(records)
    try:
        logger.info("Downloading masakhane/masakhanews (Igbo)...")
        ds = load_dataset("masakhane/masakhanews", "ibo", split="train", trust_remote_code=True)
        for row in ds:
            text = clean_text(row.get("text", "") or row.get("headline", ""))
            if text:
                records.append({"text": text, "language": "igbo", "source": "masakhanews"})
            if len(records) > 12000:
                break
        logger.info(f"  Got {len(records) - before} from MasakhaNews")
    except Exception as e:
        logger.warning(f"  MasakhaNews (ibo) failed: {e}")

    # Source 5: Igbo Bible (JW/OPUS based)
    before = len(records)
    try:
        logger.info("Trying bible-nlp/biblenlp-corpus (Igbo ig)...")
        ds = load_dataset(
            "bible-nlp/biblenlp-corpus", "ig", split="train", trust_remote_code=True
        )
        for row in ds:
            text = clean_text(row.get("text", ""))
            if text:
                records.append({"text": text, "language": "igbo", "source": "bible-nlp-ig"})
            if len(records) > 14000:
                break
        logger.info(f"  Got {len(records) - before} from Bible NLP (Igbo)")
    except Exception as e:
        logger.warning(f"  Bible NLP (Igbo) failed: {e}")

    return save_corpus(records, RAW_DIR / "igbo_corpus.jsonl", "igbo")


def download_efik():
    """Download Efik-Ibibio corpus from available sources."""
    from datasets import load_dataset
    records = []

    # Source 1: IBOM MT (Efik)
    try:
        logger.info("Downloading Davlan/ibom-mt-en-efi (Efik)...")
        ds = load_dataset("Davlan/ibom-mt-en-efi", split="train", trust_remote_code=True)
        for row in ds:
            translation = row.get("translation", {})
            efik_text = translation.get("efi") or translation.get("efik", "")
            efik_text = clean_text(efik_text)
            if efik_text:
                records.append({"text": efik_text, "language": "efik", "source": "ibom-mt-efi"})
        logger.info(f"  Got {len(records)} from IBOM-MT")
    except Exception as e:
        logger.warning(f"  IBOM-MT (Efik) failed: {e}")

    # Source 2: Try masakhane/masakhanews for related languages
    before = len(records)
    try:
        logger.info("Trying masakhane/masakhanews (Ibibio - nih)...")
        ds = load_dataset("masakhane/masakhanews", "nih", split="train", trust_remote_code=True)
        for row in ds:
            text = clean_text(row.get("text", "") or row.get("headline", ""))
            if text:
                records.append({"text": text, "language": "efik", "source": "masakhanews-nih"})
            if len(records) > 5000:
                break
        logger.info(f"  Got {len(records) - before} from MasakhaNews (nih)")
    except Exception as e:
        logger.warning(f"  MasakhaNews (nih) failed: {e}")

    # Source 3: Bible NLP (Efik or Ibibio)
    before = len(records)
    for lang_code in ["efi", "ibb"]:
        try:
            logger.info(f"Trying bible-nlp/biblenlp-corpus ({lang_code})...")
            ds = load_dataset(
                "bible-nlp/biblenlp-corpus", lang_code, split="train", trust_remote_code=True
            )
            for row in ds:
                text = clean_text(row.get("text", ""))
                if text:
                    records.append({
                        "text": text, "language": "efik",
                        "source": f"bible-nlp-{lang_code}"
                    })
                if len(records) > 6000:
                    break
            logger.info(f"  Got {len(records) - before} from Bible NLP ({lang_code})")
            before = len(records)
            break
        except Exception as e:
            logger.warning(f"  Bible NLP ({lang_code}) failed: {e}")

    # Fallback: write minimal corpus from manual seed if downloads fail
    if len(records) < 10:
        logger.warning("Efik downloads produced very little data. Writing seed corpus.")
        EFIK_SEEDS = [
            "Emem okut mme owo. Ndito mme owo ye enye ndito ekpe owo.",
            "Nte akamba? Nte idem fo? Uwa mme owo.",
            "Odun ke isong Abasi. Mme owo edi ke inam.",
            "Idim owo ye anwa ima uwa nyin. Ekpe owo bọọng.",
            "Abasi ke ukot ekeñ isiọñ. Ñkpọ eket ke ibuot.",
            "Mme idiọñ edi ke Africa. Ibibio ye Efik edi ke Cross River State.",
            "Emem ami. Mmọdiọkke mme ñkpọ eket.",
            "Idiọñ Ibibio ye idiọñ Efik edi ke Nigeria southeastern.",
            "Akwa ibibio edi iniọñ ye ibuot ke ife.",
            "Mme eka ye ete edi ke ufok.",
            "Nsọñ ke idiọñ eket ke Akwa Ibom State.",
            "Ofiọn ye utin edi ke isiọñ. Emem ke anwa.",
            "Ñkpọ eket edi ke ufok ye inam.",
            "Akamkpa ibibio ye efik edi ke Cross River.",
            "Ekpe masquerade edi ke idiọñ Efik ñkpọ eket.",
            "Mme owo ibibio ye mme owo efik edi ke Cross River State.",
            "Abasi akpa mme owo. Emem ke isiọñ.",
            "Idad mme owo edi ke Africa. Ibibio edi ke southeastern Nigeria.",
            "Mme idiọñ eket edi ke ñkpọ mbọk.",
            "Emem ye ufok edi ke Nigeria.",
        ]
        for seed in EFIK_SEEDS:
            records.append({"text": seed, "language": "efik", "source": "manual_seed"})

    return save_corpus(records, RAW_DIR / "efik_corpus.jsonl", "efik")


def download_bini():
    """Download Bini-Edo corpus from available sources."""
    from datasets import load_dataset
    records = []

    # Source 1: Bible NLP (Bini - bin)
    for lang_code in ["bin", "edo"]:
        try:
            logger.info(f"Trying bible-nlp/biblenlp-corpus ({lang_code} for Bini)...")
            ds = load_dataset(
                "bible-nlp/biblenlp-corpus", lang_code, split="train", trust_remote_code=True
            )
            for row in ds:
                text = clean_text(row.get("text", ""))
                if text:
                    records.append({
                        "text": text, "language": "bini",
                        "source": f"bible-nlp-{lang_code}"
                    })
            logger.info(f"  Got {len(records)} from Bible NLP ({lang_code})")
            break
        except Exception as e:
            logger.warning(f"  Bible NLP ({lang_code}) failed: {e}")

    # Source 2: CC-100 (Bini/Edo - try via huggingface)
    before = len(records)
    try:
        logger.info("Trying cc100 (bin - Bini)...")
        ds = load_dataset("cc100", lang="bin", split="train", streaming=True, trust_remote_code=True)
        count = 0
        for row in ds:
            text = clean_text(row.get("text", ""))
            if text:
                records.append({"text": text[:500], "language": "bini", "source": "cc100-bin"})
                count += 1
            if count >= 2000:
                break
        logger.info(f"  Got {len(records) - before} from CC100 (bin)")
    except Exception as e:
        logger.warning(f"  CC100 (bin) failed: {e}")

    # Fallback: Seed corpus from knowledge base
    if len(records) < 30:
        logger.warning("Bini-Edo downloads produced very little data. Writing knowledge-base seed corpus.")
        BINI_SEEDS = [
            # Creation and cosmology
            "Osanobua vbe erinmwin. Erinmwin ne Osanobua ma gbe.",
            "Agbon ne Osanobua ya gha rua. Oba ya oto s'evbo ebo.",
            "Ehi ne ẹrẹ rua ke Erinmwin. Orhion ne ẹrẹ ma gbe.",
            "Igodomigodo ne ọba ne ẹghe. Idu ne Benin City.",
            "Oba N'Edo ne ẹrẹ rua ke Benin. Omo N'Oba N'Edo Uku Akpolokpolo.",
            # Greetings and basic phrases
            "Ekabo. Ọvbehe? Gha khian? Ọ ya.",
            "Abie. Abie n'egbe. Ọvbehe ne gha khian.",
            "I ya vbo? Ọghe ne iran ọdo na?",
            "Kpere. Ọghe ne ọre gha sẹ.",
            "Oha n'iyoba? Gha khian n'ese?",
            # Language and culture
            "Ẹdo ne ẹghe. Bini ne ọre gha sẹ ke Edo State.",
            "Ẹdo rẹn ẹghe vbe esuku. Bini City ne ọre gha sẹ.",
            "Orhion ne ẹrẹ rua. Ehi ne ọre ne ẹghe.",
            "Olokun ne ẹrẹ rua ke ẹrẹ ẹvbo mẹ.",
            "Obo oguo o vha guese ache. Ẹze ne iran ọdo.",
            # Kingdom and royalty
            "Oba ne ẹrẹ gha sẹ vbe Benin. Ẹvbo mẹ ne Oba ya oto.",
            "Nọnọba ne ẹrẹ rua ke Benin Kingdom.",
            "Igodomigodo ne ẹrẹ gha sẹ vbe ẹrẹ ẹvbo.",
            "Ọ rẹn ẹvbo mẹ vbe iran gha sẹ.",
            "Agbon ne iran gha sẹ. Ẹvbo mẹ ne iran ọdo.",
            # Additional phrases
            "I ma-ẹre. A magh-irẹ. Ọ vbe ẹre irẹ.",
            "Ọ ya vbe iran sẹ. Ne iran gha sẹ vbe ẹrẹ.",
            "Ẹghe ne iran ọdo. Ẹrẹ ne iran sẹ.",
            "Bini ne ẹrẹ rua. Ẹdo ne iran ọdo.",
            "Osanobua gbẹnọ. Ọ ̣rẹn Iran.",
            "Ẹdo ẹvbo ne iran gha sẹ.",
            "Ọba ne ẹrẹ rua ke ẹvbo Bini.",
            "Vb'iran rua ke Erinmwin.",
            "Ẹghe ne ọre gha sẹ ke Agbon.",
            "Orhion ne ẹrẹ gha sẹ vbe iran ọdo.",
        ]
        for seed in BINI_SEEDS:
            records.append({"text": seed, "language": "bini", "source": "knowledge_base_seed"})

    return save_corpus(records, RAW_DIR / "bini_corpus.jsonl", "bini")


def main():
    parser = argparse.ArgumentParser(description="AfriWise Dataset Downloader")
    parser.add_argument("--igbo-only", action="store_true", help="Only download Igbo data")
    parser.add_argument("--efik-only", action="store_true", help="Only download Efik data")
    parser.add_argument("--bini-only", action="store_true", help="Only download Bini data")
    args = parser.parse_args()

    all_langs = not (args.igbo_only or args.efik_only or args.bini_only)
    totals = {}

    if all_langs or args.igbo_only:
        n = download_igbo()
        totals["igbo"] = n

    if all_langs or args.efik_only:
        n = download_efik()
        totals["efik"] = n

    if all_langs or args.bini_only:
        n = download_bini()
        totals["bini"] = n

    logger.info("=" * 50)
    logger.info("Download Summary:")
    for lang, n in totals.items():
        logger.info(f"  {lang}: {n} records")
    logger.info(f"Total: {sum(totals.values())} records")
    logger.info("=" * 50)
    logger.info("Next: python scripts/build_augmented_dataset.py")


if __name__ == "__main__":
    main()
