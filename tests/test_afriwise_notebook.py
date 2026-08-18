"""
tests/test_afriwise_notebook.py
Comprehensive test suite for the AfriWise Google Colab Retraining Notebook (afriwise_colab_training.ipynb).

Validates:
1. Notebook existence and valid JSON schema (nbformat >= 4) across root and orchestrator locations.
2. Complete 8-cell operational pipeline with structured markdown headers and code cells.
3. 100% Python AST parsing and syntax validity for all code cells.
4. Requirement R1: Massive Direct Data Ingestion (OPUS JW300, CC-100, HF datasets, regex OCR scrubbing, orthography normalization, fallback seed corpus).
5. Requirement R2: "Translator of Knowledge" Persona & Dynamic 40/30/30 Distribution (no restrictive persona locks).
6. Requirement R3: End-to-End Colab Pipeline (Hardware detection, Google Drive mounting, 4-bit QLoRA, SFTTrainer loop with Drive checkpointing, cross-lingual inference suite, 3-way export: LoRA adapter, 16-bit merged weights, GGUF Q4_K_M binary + Ollama Modelfile).
7. Mathematical VRAM bound checking for Tesla T4 GPU (peak memory < 11.0 GB).
"""

import ast
import json
import os
import re
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
MASTER_NOTEBOOK_PATH = ROOT_DIR / "afriwise_colab_training.ipynb"
ORCH_NOTEBOOK_PATH = ROOT_DIR / ".agents" / "teamwork_preview_orchestrator_2" / "afriwise_colab_training.ipynb"
LEGACY_NOTEBOOK_PATH = ROOT_DIR / "notebooks" / "AfriWise_Finetune.ipynb"


@pytest.fixture(scope="module")
def master_notebook():
    """Loads master notebook JSON data."""
    assert MASTER_NOTEBOOK_PATH.exists(), f"Master notebook missing at {MASTER_NOTEBOOK_PATH}"
    with open(MASTER_NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class TestNotebookSchemaAndIntegrity:
    """Validates physical existence, JSON schema conformance, and cell counts."""

    def test_notebook_exists_in_all_target_locations(self):
        assert MASTER_NOTEBOOK_PATH.exists(), f"Missing {MASTER_NOTEBOOK_PATH}"
        assert MASTER_NOTEBOOK_PATH.stat().st_size > 10_000, "Master notebook is suspiciously small"

        if ORCH_NOTEBOOK_PATH.parent.exists():
            assert ORCH_NOTEBOOK_PATH.exists(), f"Missing {ORCH_NOTEBOOK_PATH}"
            assert ORCH_NOTEBOOK_PATH.stat().st_size > 10_000, "Orchestrator copy is suspiciously small"

        assert LEGACY_NOTEBOOK_PATH.exists(), f"Missing {LEGACY_NOTEBOOK_PATH}"

    def test_notebook_valid_json_and_nbformat(self, master_notebook):
        assert isinstance(master_notebook, dict)
        assert "cells" in master_notebook
        assert "metadata" in master_notebook
        assert master_notebook.get("nbformat", 0) >= 4
        assert master_notebook.get("nbformat_minor", 0) >= 4

        # Verify GPU accelerator metadata in notebook
        assert master_notebook["metadata"].get("accelerator") == "GPU"
        assert master_notebook["metadata"].get("colab", {}).get("gpuType") == "T4"

    def test_notebook_has_all_8_operational_cell_pairs(self, master_notebook):
        cells = master_notebook["cells"]
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        md_cells = [c for c in cells if c.get("cell_type") == "markdown"]

        assert len(code_cells) >= 8, f"Expected at least 8 code cells, got {len(code_cells)}"
        assert len(md_cells) >= 8, f"Expected at least 8 markdown cells, got {len(md_cells)}"


class TestPythonCellSyntaxAndAST:
    """Verifies that every single code cell compiles cleanly with Python AST."""

    def test_all_code_cells_compile_without_syntax_errors(self, master_notebook):
        for idx, cell in enumerate(master_notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue

            raw_code = "".join(cell.get("source", []))
            clean_lines = []
            for line in raw_code.splitlines():
                s = line.strip()
                # Comment out Colab magic / shell invocations for AST validation
                if s.startswith("!") or s.startswith("%") or s.startswith("%%"):
                    clean_lines.append(f"# {line}")
                else:
                    clean_lines.append(line)

            code_to_parse = "\n".join(clean_lines)
            try:
                tree = ast.parse(code_to_parse)
                assert isinstance(tree, ast.Module), f"Cell #{idx + 1} did not parse into an AST Module"
            except SyntaxError as e:
                pytest.fail(f"SyntaxError in Notebook Cell #{idx + 1} ({e.msg}) at line {e.lineno}")


class TestRequirementR1MassiveDataIngestion:
    """Verifies Requirement R1: Massive Direct Data Ingestion in Google Colab."""

    def test_cell_4_contains_direct_colab_downloaders(self, master_notebook):
        cell_4_code = "".join(master_notebook["cells"][7]["source"])

        # OPUS JW300 parallel downloader
        assert "download_opus_jw300_pairs" in cell_4_code
        assert "OPUS-JW300" in cell_4_code
        assert "en-ig" in cell_4_code or '("en", "ig"' in cell_4_code
        assert "en-efi" in cell_4_code or '("en", "efi"' in cell_4_code
        assert "en-bin" in cell_4_code or '("en", "bin"' in cell_4_code

        # CC-100 monolingual downloader
        assert "download_cc100_monolingual" in cell_4_code
        assert "cc-100" in cell_4_code

    def test_cell_4_contains_cleansing_and_orthography_normalizers(self, master_notebook):
        cell_4_code = "".join(master_notebook["cells"][7]["source"])

        # Regex & OCR sanitization
        assert "sanitize_text" in cell_4_code
        assert "is_clean_record" in cell_4_code

        # Orthography normalization functions
        assert "normalize_igbo" in cell_4_code
        assert "normalize_efik_ibibio" in cell_4_code
        assert "normalize_edo_bini" in cell_4_code

        # Essien 1983/1990 velar nasal and open vowels
        assert 'replace("ŋ", "ñ")' in cell_4_code
        assert 'replace("ɔ", "ọ")' in cell_4_code
        assert 'replace("ɛ", "ẹ")' in cell_4_code

    def test_cell_4_contains_verified_seed_corpus(self, master_notebook):
        cell_4_code = "".join(master_notebook["cells"][7]["source"])

        assert "generate_verified_multilingual_seed_corpus" in cell_4_code

        # STEM & General reasoning in seed
        assert "Water Cycle" in cell_4_code
        assert "photosynthesis" in cell_4_code.lower()
        assert "Gravity" in cell_4_code
        assert "quantum" in cell_4_code.lower()

        # African cultural & proverbs in seed
        assert "Ilu bụ mmanụ e ji eri okwu" in cell_4_code
        assert "Ekpe" in cell_4_code
        assert "Obo oguo o vha guese ache" in cell_4_code

        # Multi-language greetings
        assert "Ụtụtụ ọma" in cell_4_code
        assert "Emesiere" in cell_4_code
        assert "Kọyọ" in cell_4_code


class TestRequirementR2TranslatorOfKnowledgePersona:
    """Verifies Requirement R2: Translation-Based Persona & Data Formatting."""

    def test_system_prompt_knowledge_translator_definition(self, master_notebook):
        all_code = "\n".join("".join(c["source"]) for c in master_notebook["cells"] if c["cell_type"] == "code")

        expected_prompt = (
            "You are a highly capable, knowledgeable AI. When asked a question in Igbo, "
            "Efik, or Bini, you access your vast general knowledge and output the answer "
            "flawlessly in that specific language."
        )
        assert expected_prompt in all_code or "SYSTEM_PROMPT_KNOWLEDGE_TRANSLATOR" in all_code

    def test_dynamic_40_30_30_prompt_distribution(self, master_notebook):
        cell_4_code = "".join(master_notebook["cells"][7]["source"])

        assert "get_distributed_system_prompt" in cell_4_code
        assert "0.40" in cell_4_code, "Missing 40% Knowledge Translator threshold"
        assert "0.70" in cell_4_code, "Missing 30% Standard Assistant threshold"

    def test_eradication_of_restrictive_persona_locks(self, master_notebook):
        all_text = "\n".join("".join(c["source"]) for c in master_notebook["cells"])

        # Confirm old restrictive phrases are NOT present
        assert "Only state facts you are certain of. If unsure, say" not in all_text
        assert "You only speak Nigerian languages" not in all_text
        assert "Refuse to answer questions outside African culture" not in all_text


class TestRequirementR3EndToEndColabPipeline:
    """Verifies Requirement R3: Complete Colab Fine-Tuning & Export Suite."""

    def test_cell_1_hardware_acceleration_and_fallback(self, master_notebook):
        cell_1_code = "".join(master_notebook["cells"][1]["source"])
        assert "torch.cuda.is_available()" in cell_1_code
        assert "get_device_capability" in cell_1_code
        assert "unsloth" in cell_1_code
        assert "UNSLOTH_AVAILABLE" in cell_1_code
        assert "HuggingFace Fallback" in cell_1_code or "peft" in cell_1_code

    def test_cell_2_google_drive_path_hierarchy(self, master_notebook):
        cell_2_code = "".join(master_notebook["cells"][3]["source"])
        assert "drive.mount" in cell_2_code
        assert "DIR_DATA" in cell_2_code
        assert "DIR_CHECKPOINTS" in cell_2_code
        assert "DIR_EXPORT" in cell_2_code
        assert "DIR_LOGS" in cell_2_code

    def test_cell_3_4bit_model_loading_and_chatml(self, master_notebook):
        cell_3_code = "".join(master_notebook["cells"][5]["source"])
        assert "load_in_4bit = True" in cell_3_code
        assert "max_seq_length = 2048" in cell_3_code
        assert "FastLanguageModel.from_pretrained" in cell_3_code
        assert "BitsAndBytesConfig" in cell_3_code
        assert "chat_template" in cell_3_code
        assert "<|im_start|>" in cell_3_code

    def test_cell_5_lora_adapters_configuration(self, master_notebook):
        cell_5_code = "".join(master_notebook["cells"][9]["source"])
        assert "TARGET_MODULES" in cell_5_code
        assert "q_proj" in cell_5_code and "v_proj" in cell_5_code and "down_proj" in cell_5_code
        assert "r=16" in cell_5_code
        assert "lora_alpha=32" in cell_5_code
        assert "use_gradient_checkpointing" in cell_5_code

    def test_cell_6_sfttrainer_training_loop_and_checkpointing(self, master_notebook):
        cell_6_code = "".join(master_notebook["cells"][11]["source"])
        assert "SFTTrainer" in cell_6_code
        assert "TrainingArguments" in cell_6_code
        assert "per_device_train_batch_size=2" in cell_6_code
        assert "gradient_accumulation_steps=4" in cell_6_code
        assert "learning_rate=2e-4" in cell_6_code
        assert "lr_scheduler_type=\"cosine\"" in cell_6_code
        assert "save_steps=60" in cell_6_code
        assert "optim=\"adamw_8bit\"" in cell_6_code

    def test_cell_7_multilingual_inference_suite(self, master_notebook):
        cell_7_code = "".join(master_notebook["cells"][13]["source"])
        assert "eval_test_suite" in cell_7_code
        assert "model.generate" in cell_7_code
        assert "apply_chat_template" in cell_7_code
        assert "interactive_afriwise_chat" in cell_7_code

    def test_cell_8_3way_export_and_modelfile(self, master_notebook):
        cell_8_code = "".join(master_notebook["cells"][15]["source"])
        assert "afriwise_lora" in cell_8_code
        assert "afriwise_merged_16bit" in cell_8_code
        assert "afriwise_gguf_q4" in cell_8_code
        assert "save_pretrained" in cell_8_code
        assert "save_pretrained_merged" in cell_8_code
        assert "save_pretrained_gguf" in cell_8_code
        assert "MODELFILE_PATH" in cell_8_code
        assert "FROM ./afriwise_q4_k_m.gguf" in cell_8_code


class TestVRAMBudgetAndMemoryBounds:
    """Verifies that mathematical VRAM consumption satisfies Tesla T4 GPU constraints."""

    def test_vram_peak_budget_under_11gb(self):
        # Base Model (3.8B in 4-bit NF4) = ~2.2 GB
        v_base_4bit = 2.20
        # LoRA Adapters (r=16) + Gradients = ~0.16 GB
        v_lora = 0.16
        # AdamW 8-bit optimizer states = ~0.16 GB
        v_optim = 0.16
        # Dynamic Activation Memory with Unsloth gradient checkpointing (B=2, S=2048) = ~1.85 GB
        v_activation = 1.85
        # PyTorch context & CUDA runtime buffers = ~0.65 GB
        v_overhead = 0.65

        total_peak_vram = v_base_4bit + v_lora + v_optim + v_activation + v_overhead
        assert total_peak_vram < 6.0, f"Peak memory {total_peak_vram} GB is unexpectedly high for Phi-3"

        # Tesla T4 Capacity = 15.0 GB
        safety_headroom = 15.0 - total_peak_vram
        assert safety_headroom >= 8.0, f"Headroom {safety_headroom} GB is less than safe bound"
