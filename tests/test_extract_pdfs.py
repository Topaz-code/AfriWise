"""
Unit tests for scripts/extract_pdfs.py.

Tests cover:
- Page text cleaning (headers, footers, page numbers)
- Igbo dictionary parsing (tone marks, multi-word headwords, colons/examples, subentries)
- Ibibio dictionary parsing (OCR characters, POS variations, literals, cross-references)
- Bini dictionary parsing (Melzian format, tone markings, semicolons)
- Dispatcher and error handling
- POS normalization
- End-to-end PDF extraction and JSON output generation
"""

import json
import os
import pytest
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_pdfs import (
    clean_page_text,
    extract_all_dictionaries,
    extract_from_pdf,
    normalize_pos,
    parse_bini_text,
    parse_dictionary_text,
    parse_ibibio_text,
    parse_igbo_text,
)


class TestPageCleaning:
    """Tests for clean_page_text function."""

    def test_clean_empty_text(self):
        assert clean_page_text("", "Igbo") == ""
        assert clean_page_text("   \n\n  ", "Ibibio") == ""

    def test_clean_igbo_headers_and_page_numbers(self):
        raw = """
        Igbo Dictionary: KayWilliamson. Draft of Edition II
        42
        IGBO DICTIONARY
        A.
        egō
        n.
        money
        xxviii
        """
        cleaned = clean_page_text(raw, "Igbo")
        assert "Igbo Dictionary: KayWilliamson" not in cleaned
        assert "42" not in cleaned
        assert "xxviii" not in cleaned
        assert "IGBO DICTIONARY" not in cleaned
        assert "egō\nn.\nmoney" in cleaned

    def test_clean_ibibio_headers(self):
        raw = """
        DOCUMENT RESUME
        FINAL REPORT
        IBIBIO DICTIONARY
        125
        abasi
        n
        God, god
        """
        cleaned = clean_page_text(raw, "Ibibio")
        assert "DOCUMENT RESUME" not in cleaned
        assert "FINAL REPORT" not in cleaned
        assert "125" not in cleaned
        assert "abasi\nn\nGod, god" in cleaned

    def test_clean_bini_headers(self):
        raw = """
        A CONCISE DICTIONARY OF THE BINI LANGUAGE
        15
        ágbà, n., jaw, chin.
        """
        cleaned = clean_page_text(raw, "Bini")
        assert "A CONCISE DICTIONARY" not in cleaned
        assert "15" not in cleaned
        assert "ágbà, n., jaw, chin." in cleaned


class TestPosNormalization:
    """Tests for normalize_pos."""

    def test_pos_normalization_standard(self):
        assert normalize_pos("n") == "n."
        assert normalize_pos("n.") == "n."
        assert normalize_pos("v") == "v."
        assert normalize_pos("v.") == "v."
        assert normalize_pos("aj") == "adj."
        assert normalize_pos("adj.") == "adj."
        assert normalize_pos("adv") == "adv."
        assert normalize_pos("tv") == "v.tr."
        assert normalize_pos("iv") == "v.i."
        assert normalize_pos("v.tr.") == "v.tr."
        assert normalize_pos("v.i.") == "v.i."
        assert normalize_pos("inter]") == "interj."
        assert normalize_pos(None) is None


class TestIgboParsing:
    """Tests for Igbo dictionary parsing."""

    def test_parse_basic_entry(self):
        sample = """
        àbị, àbịị̀
        n.
        yam cv. (sausage-shaped)
        """
        entries = parse_igbo_text(sample)
        assert len(entries) == 1
        assert entries[0]["language"] == "Igbo"
        assert entries[0]["headword"] == "àbị, àbịị̀"
        assert entries[0]["pos"] == "n."
        assert entries[0]["definition"] == "yam cv. (sausage-shaped)"
        assert entries[0]["examples"] == []

    def test_parse_entry_with_tone_marks(self):
        sample = """
        ọ̀kụkụ̀ abùke
        n.
        kind of fowl which never grows to a large size
        """
        entries = parse_igbo_text(sample)
        assert len(entries) == 1
        assert entries[0]["headword"] == "ọ̀kụkụ̀ abùke"
        assert entries[0]["pos"] == "n."
        assert "kind of fowl" in entries[0]["definition"]

    def test_parse_entry_with_examples_colon(self):
        sample = """
        iwe
        n.
        anger
        -dị iwe
        annoy: Ife o mèlụ̀ dị̀ m̀ iwe What he did annoys me
        """
        entries = parse_igbo_text(sample)
        assert len(entries) == 1
        assert entries[0]["headword"] == "iwe"
        assert entries[0]["pos"] == "n."
        assert "anger" in entries[0]["definition"]
        assert len(entries[0]["examples"]) > 0
        assert any("What he did annoys me" in ex for ex in entries[0]["examples"])

    def test_parse_homonym_numbers(self):
        sample = """
        abụ 1.
        n.
        pus
        abụ 2.
        n.
        cat-like animal that sleeps by day
        """
        entries = parse_igbo_text(sample)
        assert len(entries) == 2
        assert entries[0]["headword"] == "abụ 1."
        assert entries[0]["definition"] == "pus"
        assert entries[1]["headword"] == "abụ 2."
        assert "cat-like animal" in entries[1]["definition"]

    def test_parse_cross_reference_entry(self):
        sample = """
        ìteghete
        see teghete
        """
        entries = parse_igbo_text(sample)
        assert len(entries) == 1
        assert entries[0]["headword"] == "ìteghete"
        assert "see teghete" in entries[0]["definition"]


class TestIbibioParsing:
    """Tests for Ibibio dictionary parsing."""

    def test_parse_basic_ibibio_entry(self):
        sample = """
        abasi
        n
        God, god
        abasi Ibbm
        God almighty
        lit
        god of greatness
        """
        entries = parse_ibibio_text(sample)
        assert len(entries) >= 1
        assert entries[0]["language"] == "Ibibio"
        assert entries[0]["headword"] == "abasi"
        assert entries[0]["pos"] == "n."
        assert "God, god" in entries[0]["definition"]

    def test_parse_ibibio_verb_and_examples(self):
        sample = """
        edim
        n
        rain
        edim asilk adep.
        It's raining.
        """
        entries = parse_ibibio_text(sample)
        assert len(entries) == 1
        assert entries[0]["headword"] == "edim"
        assert entries[0]["pos"] == "n."
        assert "rain" in entries[0]["definition"]
        assert len(entries[0]["examples"]) > 0

    def test_parse_ibibio_transitive_verb(self):
        sample = """
        b6
        tv
        speak, say, tell
        b6 afi6!
        Tell him!
        """
        entries = parse_ibibio_text(sample)
        assert len(entries) == 1
        assert entries[0]["headword"] == "b6"
        assert entries[0]["pos"] == "v.tr."
        assert "speak, say, tell" in entries[0]["definition"]


class TestBiniParsing:
    """Tests for Bini dictionary parsing."""

    def test_parse_melzian_format(self):
        sample = """
        ágbà, n., jaw, chin.
        dè, v.tr., buy.
        kpòlò, v.i. & v.tr., sweep; sweep the floor.
        """
        entries = parse_bini_text(sample)
        assert len(entries) == 3
        assert entries[0]["language"] == "Bini"
        assert entries[0]["headword"] == "ágbà"
        assert entries[0]["pos"] == "n."
        assert entries[0]["definition"] == "jaw, chin."

        assert entries[1]["headword"] == "dè"
        assert entries[1]["pos"] == "v.tr."
        assert entries[1]["definition"] == "buy."

        assert entries[2]["headword"] == "kpòlò"
        assert entries[2]["definition"] == "sweep"
        assert "sweep the floor." in entries[2]["examples"]

    def test_parse_bini_with_tones_and_semicolon(self):
        sample = "òhuan, n., sheep; ram; pl. ihuan."
        entries = parse_bini_text(sample)
        assert len(entries) == 1
        assert entries[0]["headword"] == "òhuan"
        assert entries[0]["pos"] == "n."
        assert entries[0]["definition"] == "sheep"
        assert "ram" in entries[0]["examples"]
        assert "pl. ihuan." in entries[0]["examples"]


class TestDispatcherAndEdgeCases:
    """Tests for parse_dictionary_text and error handling."""

    def test_dispatch_valid_languages(self):
        igbo_entries = parse_dictionary_text("Igbo", "egō\nn.\nmoney")
        assert len(igbo_entries) == 1
        assert igbo_entries[0]["language"] == "Igbo"

        ibibio_entries = parse_dictionary_text("Ibibio", "edim\nn\nrain")
        assert len(ibibio_entries) == 1
        assert ibibio_entries[0]["language"] == "Ibibio"

        bini_entries = parse_dictionary_text("Bini", "ágbà, n., jaw, chin.")
        assert len(bini_entries) == 1
        assert bini_entries[0]["language"] == "Bini"

    def test_dispatch_invalid_language(self):
        with pytest.raises(ValueError) as exc:
            parse_dictionary_text("Swahili", "jambo")
        assert "Unsupported language" in str(exc.value)

    def test_nonexistent_pdf_handled_gracefully(self):
        res = extract_from_pdf("nonexistent_path.pdf", "Igbo")
        assert res == []


class TestExtractionIntegration:
    """Integration test with actual PDFs in dictionaries_raw."""

    @pytest.fixture
    def pdf_dir(self):
        return str(PROJECT_ROOT / "dictionaries_raw")

    def test_sample_extraction_runs(self, pdf_dir, tmp_path):
        out_file = tmp_path / "test_dict.json"
        result = extract_all_dictionaries(
            pdf_dir=pdf_dir,
            output_path=str(out_file),
            limit_pages=3
        )
        assert result["total_entries"] > 0
        assert out_file.exists()

        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == result["total_entries"]
        
        # Verify required keys in each extracted entry
        for item in data[:10]:
            assert "language" in item
            assert item["language"] in ["Igbo", "Ibibio", "Bini"]
            assert "headword" in item
            assert isinstance(item["headword"], str)
            assert "pos" in item
            assert "definition" in item
            assert isinstance(item["definition"], str)
            assert "examples" in item
            assert isinstance(item["examples"], list)
