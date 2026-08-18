"""Tests for raw corpus acquisition."""
import json
from pathlib import Path

DATA_RAW = Path("data/raw")

def test_igbo_corpus_exists():
    corpus = DATA_RAW / "igbo_corpus.jsonl"
    assert corpus.exists(), "Igbo corpus JSONL not found"
    lines = [l for l in corpus.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    assert len(lines) >= 100, f"Expected >=100 Igbo entries, got {len(lines)}"

def test_efik_corpus_exists():
    corpus = DATA_RAW / "efik_corpus.jsonl"
    assert corpus.exists(), "Efik corpus JSONL not found"
    lines = [l for l in corpus.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    assert len(lines) >= 20, f"Expected >=20 Efik entries, got {len(lines)}"

def test_bini_corpus_exists():
    corpus = DATA_RAW / "bini_corpus.jsonl"
    assert corpus.exists(), "Bini corpus JSONL not found"
    lines = [l for l in corpus.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    assert len(lines) >= 10, f"Expected >=10 Bini entries, got {len(lines)}"

def test_corpus_json_valid():
    for lang in ["igbo", "efik", "bini"]:
        corpus = DATA_RAW / f"{lang}_corpus.jsonl"
        if corpus.exists():
            for line in corpus.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    rec = json.loads(line)
                    assert "text" in rec, f"Missing 'text' key in {lang} corpus"
                    assert "language" in rec, f"Missing 'language' key in {lang} corpus"
