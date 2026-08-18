"""
AfriWise Data Ingestion: "Millions of Data" Pipeline.
Downloads massive parallel corpora (JW300, OPUS, CC-100) for self-supervised learning on CPU.
"""

import os
import json
import logging
from datasets import load_dataset
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_RAW = Path("data/raw/massive_african_corpus")
DATA_RAW.mkdir(parents=True, exist_ok=True)

def download_opus_data():
    """Downloads millions of sentences from OPUS (if available on HF)."""
    logger.info("Downloading OPUS / JW300 / CC-100 datasets for Igbo, Efik, and Edo/Bini...")
    
    # 1. Igbo (Very large datasets available)
    logger.info("Downloading Igbo massive datasets...")
    try:
        # cc100 is huge
        igbo_cc = load_dataset("cc100", lang="ig", split="train", trust_remote_code=True)
        out_path = DATA_RAW / "igbo_massive.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in igbo_cc:
                f.write(json.dumps({"text": item["text"], "language": "igbo", "source": "cc100"}) + "\n")
        logger.info(f"Saved Igbo massive data to {out_path} ({len(igbo_cc)} rows)")
    except Exception as e:
        logger.error(f"Could not load cc100 for Igbo: {e}")

    # 2. Efik / Ibibio
    logger.info("Downloading Efik massive datasets...")
    try:
        efik_data = load_dataset("Davlan/ibom-mt-en-efi", split="train", trust_remote_code=True)
        out_path = DATA_RAW / "efik_massive.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in efik_data:
                f.write(json.dumps({"text": item["translation"]["efi"], "language": "efik", "source": "ibom-mt"}) + "\n")
        logger.info(f"Saved Efik massive data to {out_path} ({len(efik_data)} rows)")
    except Exception as e:
        logger.error(f"Could not load Efik data: {e}")
        
    # 3. Bini / Edo
    logger.info("Downloading Bini/Edo massive datasets...")
    try:
        bini_data = load_dataset("bible-nlp/biblenlp-corpus", "bin", split="train", trust_remote_code=True)
        out_path = DATA_RAW / "bini_massive.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in bini_data:
                f.write(json.dumps({"text": item["text"], "language": "bini", "source": "biblenlp"}) + "\n")
        logger.info(f"Saved Bini massive data to {out_path} ({len(bini_data)} rows)")
    except Exception as e:
        logger.error(f"Could not load Bini data: {e}")

    logger.info("Data ingestion complete. Ready for self-supervised learning!")

if __name__ == "__main__":
    download_opus_data()
