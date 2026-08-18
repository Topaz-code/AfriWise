"""
Unit tests for scripts/build_dataset.py.

Tests cover:
- Text cleaning and POS normalization
- Reverse definition filtering (rejecting 'see e', 'cf.', etc.)
- Direct translation template generation (with and without POS)
- Reverse translation template generation
- Context and example sentence generation
- Bini proverb extraction from cultural corpus markdown
- Proverb QA pair generation (explanation, moral lesson, theme, reverse lookup)
- Cosmology and mythological storytelling QA generation
- Language pedagogy and guardrail QA generation
- Alpaca and ChatML schema converters and validation functions
- File I/O (JSONL saving and loading with UTF-8 verification)
- End-to-end dataset generation with build_all_datasets
"""

import json
from pathlib import Path
import random
import tempfile
import pytest
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_dataset import (
    AFRIWISE_SYSTEM_PROMPT,
    clean_reverse_definition,
    clean_text,
    convert_to_alpaca,
    convert_to_chatml,
    extract_proverbs_from_markdown,
    format_pos_label,
    generate_context_usage_examples,
    generate_cosmology_examples,
    generate_direct_translation_examples,
    generate_pedagogy_examples,
    generate_proverb_examples,
    generate_reverse_translation_examples,
    is_valid_reverse_definition,
    save_jsonl,
    validate_alpaca_record,
    validate_chatml_record,
    build_all_datasets,
)


class TestTextAndPOSUtilities:
    """Tests for text normalization and POS formatting."""

    def test_clean_text_empty_and_spaces(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""
        assert clean_text("  hello   world \n\n new line  ") == "hello world new line"

    def test_format_pos_label(self):
        assert format_pos_label("n.") == "noun"
        assert format_pos_label("v.tr.") == "transitive verb"
        assert format_pos_label("v.i.") == "intransitive verb"
        assert format_pos_label("adj.") == "adjective"
        assert format_pos_label("ideo.") == "ideophone"
        assert format_pos_label(None) is None
        assert format_pos_label("custom_tag") == "custom_tag"

    def test_is_valid_reverse_definition(self):
        # Valid definitions
        assert is_valid_reverse_definition("house, dwelling") is True
        assert is_valid_reverse_definition("to run quickly; hasten") is True

        # Invalid / circular definitions
        assert is_valid_reverse_definition("") is False
        assert is_valid_reverse_definition(None) is False
        assert is_valid_reverse_definition("see e") is False
        assert is_valid_reverse_definition("see also mkpọ") is False
        assert is_valid_reverse_definition("cf. òbòŋ") is False
        assert is_valid_reverse_definition("syn. egō") is False
        assert is_valid_reverse_definition("a" * 300) is False  # Too long

    def test_clean_reverse_definition(self):
        assert clean_reverse_definition("1. money, currency; see ego") == "money, currency"
        assert clean_reverse_definition("king, traditional ruler. cf. obong") == "king, traditional ruler"


class TestDirectTranslationGenerator:
    """Tests for direct translation (Language -> English) generator."""

    def test_generate_direct_translation_basic(self):
        entry = {
            "language": "Ibibio",
            "headword": "ufọk",
            "pos": "n.",
            "definition": "house, building, dwelling",
            "examples": ["ufọk Abasi (church)"]
        }
        rng = random.Random(42)
        results = generate_direct_translation_examples(entry, rng=rng)
        assert len(results) == 1
        rec = results[0]
        assert "ufọk" in rec["instruction"] or "ufọk" in rec["output"]
        assert "house, building, dwelling" in rec["output"]
        assert "Ibibio" in rec["instruction"] or "Ibibio" in rec["output"]
        assert "ufọk Abasi" in rec["output"]
        assert rec["category"] == "direct_translation"

    def test_generate_direct_translation_empty_headword(self):
        entry = {
            "language": "Igbo",
            "headword": "",
            "pos": "n.",
            "definition": "money",
            "examples": []
        }
        results = generate_direct_translation_examples(entry)
        assert results == []


class TestReverseTranslationGenerator:
    """Tests for reverse translation (English -> Target Language) generator."""

    def test_generate_reverse_translation_valid(self):
        entry = {
            "language": "Igbo",
            "headword": "egō",
            "pos": "n.",
            "definition": "money, currency",
            "examples": []
        }
        rng = random.Random(42)
        results = generate_reverse_translation_examples(entry, rng=rng)
        assert len(results) == 1
        rec = results[0]
        assert "money, currency" in rec["instruction"]
        assert "egō" in rec["output"]
        assert "Igbo" in rec["instruction"] or "Igbo" in rec["output"]
        assert rec["category"] == "reverse_translation"

    def test_generate_reverse_translation_circular_skip(self):
        entry = {
            "language": "Igbo",
            "headword": "a",
            "pos": None,
            "definition": "see e",
            "examples": []
        }
        results = generate_reverse_translation_examples(entry)
        assert results == []


class TestContextUsageGenerator:
    """Tests for context and example sentence generation."""

    def test_generate_context_usage_with_examples(self):
        entry = {
            "language": "Ibibio",
            "headword": "dia",
            "pos": "v.tr.",
            "definition": "to eat",
            "examples": ["dia udia (eat food)", "nyin imedia (we have eaten)"]
        }
        rng = random.Random(42)
        results = generate_context_usage_examples(entry, rng=rng)
        assert len(results) == 1
        rec = results[0]
        assert "dia" in rec["instruction"]
        assert "dia udia" in rec["output"]
        assert "nyin imedia" in rec["output"]
        assert rec["category"] == "context_usage"

    def test_generate_context_usage_empty_examples(self):
        entry = {
            "language": "Ibibio",
            "headword": "dia",
            "pos": "v.",
            "definition": "to eat",
            "examples": []
        }
        results = generate_context_usage_examples(entry)
        assert results == []


class TestProverbExtractorAndGenerator:
    """Tests for Bini proverb extraction and QA generation."""

    def test_extract_proverbs_from_markdown(self):
        corpus_path = PROJECT_ROOT / "knowledge_base" / "bini_edo_cultural_corpus.md"
        assert corpus_path.exists(), "Cultural corpus file must exist"
        text = corpus_path.read_text(encoding="utf-8")

        proverbs = extract_proverbs_from_markdown(text)
        assert len(proverbs) == 15
        assert proverbs[0]["number"] == 1
        assert "Omo na gba shi ukoko" in proverbs[0]["proverb"]
        assert "child on the back" in proverbs[0]["translation"]
        assert "comfort zone" in proverbs[0]["cultural_meaning"]

        # Check last proverb
        assert proverbs[-1]["number"] == 15
        assert "orphan" in proverbs[-1]["translation"]

    def test_generate_proverb_examples(self):
        proverbs = [
            {
                "number": 5,
                "proverb": "Obo oguo o vha guese ache",
                "translation": "One hand can’t cover the pot.",
                "cultural_meaning": "Community and collaboration are essential; seek help from others.",
                "theme": "unity and teamwork",
                "language": "Edo (Bini)",
            }
        ]
        rng = random.Random(42)
        pairs = generate_proverb_examples(proverbs, rng=rng)
        # Should generate 4 variations (explanation, moral lesson, theme query, reverse query)
        assert len(pairs) == 4
        instructions = [p["instruction"] for p in pairs]
        assert any("Explain" in ins or "meaning" in ins for ins in instructions)
        assert any("moral lesson" in ins or "wisdom" in ins for ins in instructions)
        assert any("unity and teamwork" in ins for ins in instructions)
        assert all(p["category"] == "proverbs_philosophy" for p in pairs)


class TestCosmologyAndPedagogyGenerators:
    """Tests for cosmology storytelling and pedagogy QA generators."""

    def test_generate_cosmology_examples(self):
        cosmology_pairs = generate_cosmology_examples()
        assert len(cosmology_pairs) >= 15
        topics = [p["instruction"] for p in cosmology_pairs]
        assert any("Igodomigodo" in t for t in topics)
        assert any("Osanobua" in t for t in topics)
        assert any("four pillars" in t.lower() for t in topics)
        assert any("Iso" in t or "sky" in t.lower() for t in topics)
        assert any("Ehi" in t or "soul" in t.lower() for t in topics)
        assert all(p["category"] == "cosmology_storytelling" for p in cosmology_pairs)

    def test_generate_pedagogy_examples(self):
        pedagogy_pairs = generate_pedagogy_examples()
        assert len(pedagogy_pairs) >= 15
        instructions = [p["instruction"] for p in pedagogy_pairs]
        assert any("digraphs" in ins.lower() for ins in instructions)
        assert any("vowels" in ins.lower() for ins in instructions)
        assert any("oba of benin" in ins.lower() or "royalty" in ins.lower() for ins in instructions)
        assert any("downstep" in ins.lower() for ins in instructions)
        assert any("guardrails" in ins.lower() for ins in instructions)
        assert all(p["category"] == "language_pedagogy" for p in pedagogy_pairs)


class TestSchemaValidationAndConversion:
    """Tests for Alpaca and ChatML schema converters and validation."""

    def test_convert_to_alpaca_and_validate(self):
        rec = {
            "instruction": "Translate 'ufọk' into English.",
            "input": "",
            "output": "In Ibibio, 'ufọk' means house.",
        }
        alpaca = convert_to_alpaca(rec)
        assert alpaca == {
            "instruction": "Translate 'ufọk' into English.",
            "input": "",
            "output": "In Ibibio, 'ufọk' means house.",
        }
        assert validate_alpaca_record(alpaca) is True

    def test_invalid_alpaca_record(self):
        assert validate_alpaca_record({"instruction": "", "output": "valid"}) is False
        assert validate_alpaca_record({"instruction": "valid", "output": ""}) is False
        assert validate_alpaca_record("not a dict") is False

    def test_convert_to_chatml_and_validate(self):
        rec = {
            "instruction": "Explain the concept of Ehi in Edo cosmology.",
            "input": "",
            "output": "Ehi is the spiritual guardian and destiny counterpart...",
        }
        chatml = convert_to_chatml(rec)
        assert "messages" in chatml
        messages = chatml["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == AFRIWISE_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Explain the concept of Ehi in Edo cosmology."
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"].startswith("Ehi is the spiritual")
        assert validate_chatml_record(chatml) is True

    def test_invalid_chatml_record(self):
        # Missing system prompt
        bad_chatml = {
            "messages": [
                {"role": "system", "content": "Different system prompt"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        }
        assert validate_chatml_record(bad_chatml) is False

        # Empty assistant content
        empty_assistant = {
            "messages": [
                {"role": "system", "content": AFRIWISE_SYSTEM_PROMPT},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": ""},
            ]
        }
        assert validate_chatml_record(empty_assistant) is False


class TestSaveJsonlAndEndToEndBuild:
    """Tests for file saving and end-to-end dataset generation."""

    def test_save_jsonl(self):
        records = [
            {"instruction": "Test 1", "input": "", "output": "Result 1"},
            {"instruction": "Test 2", "input": "", "output": "Result 2"},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "output.jsonl"
            count = save_jsonl(records, out_file)
            assert count == 2
            assert out_file.exists()

            with open(out_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
            assert len(lines) == 2
            assert lines[0]["instruction"] == "Test 1"

    def test_build_all_datasets_end_to_end(self):
        dict_path = PROJECT_ROOT / "data" / "raw" / "dictionary_extracted.json"
        kb_dir = PROJECT_ROOT / "knowledge_base"

        assert dict_path.exists(), "Raw dictionary extracted JSON must exist"
        assert kb_dir.exists(), "Knowledge base directory must exist"

        alpaca_records, chatml_records, stats = build_all_datasets(
            dict_path=dict_path,
            kb_dir=kb_dir,
            seed=42,
        )

        assert len(alpaca_records) >= 4000
        assert len(chatml_records) == len(alpaca_records)
        assert stats["raw_dictionary_entries"] == 12503
        assert stats["proverbs_qa_count"] == 100  # 25 proverbs (15 Bini + 10 Igbo) * 4 variations
        assert stats["cosmology_qa_count"] >= 18
        assert stats["pedagogy_qa_count"] >= 20
        assert stats["total_alpaca_examples"] == len(alpaca_records)

        # Ensure all records are valid
        for rec in alpaca_records[:50]:
            assert validate_alpaca_record(rec) is True

        for rec in chatml_records[:50]:
            assert validate_chatml_record(rec) is True
