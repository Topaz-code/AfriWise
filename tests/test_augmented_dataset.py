"""Tests for augmented instruction-tuning dataset."""
import json
from pathlib import Path

V2_DATASET = Path("data/processed/afriwise_v2_chatml.jsonl")

def _load_all():
    lines = [l for l in V2_DATASET.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    return [json.loads(l) for l in lines]

def test_v2_dataset_exists():
    assert V2_DATASET.exists(), "V2 dataset not found"

def test_v2_dataset_minimum_size():
    records = _load_all()
    assert len(records) >= 100, f"Expected >=100 records, got {len(records)}"

def test_v2_records_have_chatml_format():
    records = _load_all()
    for i, r in enumerate(records[:20]):
        assert "messages" in r, f"Record {i} missing 'messages' key"
        msgs = r["messages"]
        assert len(msgs) >= 2, f"Record {i} has <2 messages"
        roles = [m["role"] for m in msgs]
        assert "user" in roles, f"Record {i} missing user message"
        assert "assistant" in roles, f"Record {i} missing assistant message"

def test_no_empty_assistant_responses():
    records = _load_all()
    for i, r in enumerate(records):
        for msg in r.get("messages", []):
            if msg["role"] == "assistant":
                assert msg["content"].strip(), f"Empty assistant response in record {i}"

def test_language_coverage():
    records = _load_all()
    all_text = " ".join(
        msg["content"]
        for r in records
        for msg in r.get("messages", [])
    )
    bini_ok = "osanobua" in all_text.lower() or "erinmwin" in all_text.lower() or "edo" in all_text.lower()
    assert bini_ok, "No Bini-Edo markers in dataset"
    igbo_ok = "igbo" in all_text.lower() or "mbe" in all_text.lower() or "omenala" in all_text.lower()
    assert igbo_ok, "No Igbo markers in dataset"
    efik_ok = "efik" in all_text.lower() or "ibibio" in all_text.lower() or "emem" in all_text.lower()
    assert efik_ok, "No Efik markers in dataset"
