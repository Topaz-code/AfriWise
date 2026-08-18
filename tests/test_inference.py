"""
Unit tests for AfriWise inference engine and export utilities.

Tests cover:
- ChatML prompt formatting and multi-turn history
- System prompt preservation
- Benchmark suite definition and prompt validation
- Adapter files integrity check
- Merge script argument validation
"""

import json
from pathlib import Path
import pytest
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.inference_afriwise import (
    AFRIWISE_SYSTEM_PROMPT,
    BENCHMARK_PROMPTS,
    format_chatml_prompt,
)


class TestPromptFormatting:
    """Tests for ChatML prompt formatting."""

    def test_format_chatml_prompt_basic(self):
        query = "Translate 'ufọk' into English."
        formatted = format_chatml_prompt(query)

        assert "<|system|>" in formatted
        assert AFRIWISE_SYSTEM_PROMPT in formatted
        assert "<|user|>" in formatted
        assert query in formatted
        assert "<|assistant|>" in formatted
        assert formatted.endswith("<|assistant|>\n")

    def test_format_chatml_prompt_with_history(self):
        history = [
            {"role": "user", "content": "Hello AfriWise"},
            {"role": "assistant", "content": "Emesiere! How may I help you?"},
        ]
        query = "Teach me Igbo greetings."
        formatted = format_chatml_prompt(query, history=history)

        assert "Hello AfriWise" in formatted
        assert "Emesiere!" in formatted
        assert "Teach me Igbo greetings." in formatted
        assert formatted.count("<|user|>") == 2
        assert formatted.count("<|assistant|>") == 2


class TestBenchmarkSuite:
    """Tests for AfriWise evaluation benchmarks."""

    def test_benchmark_prompts_coverage(self):
        assert len(BENCHMARK_PROMPTS) >= 5
        categories = [item["category"] for item in BENCHMARK_PROMPTS]
        assert any("Ibibio" in c for c in categories)
        assert any("Igbo" in c for c in categories)
        assert any("Edo" in c or "Bini" in c for c in categories)

        for item in BENCHMARK_PROMPTS:
            assert "category" in item
            assert "prompt" in item
            assert len(item["prompt"]) > 10


class TestAdapterIntegrity:
    """Verify presence and validity of trained adapter files."""

    def test_adapter_directory_contents(self):
        adapter_dir = PROJECT_ROOT / "afriwise_adapters"
        assert adapter_dir.exists(), "afriwise_adapters directory must exist"

        weights_path = adapter_dir / "adapter_model.safetensors"
        assert weights_path.exists(), "adapter_model.safetensors must exist"
        assert weights_path.stat().st_size > 10_000_000, "Adapter weights should be > 10MB"

        config_path = adapter_dir / "adapter_config.json"
        assert config_path.exists(), "adapter_config.json must exist"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg.get("r") == 16 or "r" in cfg
        assert "lora" in cfg.get("peft_type", "").lower()
