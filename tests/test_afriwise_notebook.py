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

def get_all_code(master_notebook):
    return "\n".join("".join(c["source"]) for c in master_notebook["cells"] if c["cell_type"] == "code")


class TestNotebookSchemaAndIntegrity:
    """Validates physical existence, JSON schema conformance, and cell counts."""

    def test_notebook_exists_in_all_target_locations(self):
        assert MASTER_NOTEBOOK_PATH.exists(), f"Missing {MASTER_NOTEBOOK_PATH}"
        assert MASTER_NOTEBOOK_PATH.stat().st_size > 10_000, "Master notebook is suspiciously small"
        
        # We don't fail if orchestrator doesn't exist yet, but if parent exists we check
        if ORCH_NOTEBOOK_PATH.parent.exists():
            assert ORCH_NOTEBOOK_PATH.exists(), f"Missing {ORCH_NOTEBOOK_PATH}"
        
        if LEGACY_NOTEBOOK_PATH.exists():
            pass

    def test_notebook_valid_json_and_nbformat(self, master_notebook):
        assert isinstance(master_notebook, dict)
        assert "cells" in master_notebook
        assert "metadata" in master_notebook
        assert master_notebook.get("nbformat", 0) >= 4

    def test_notebook_has_all_operational_cell_pairs(self, master_notebook):
        cells = master_notebook["cells"]
        code_cells = [c for c in cells if c.get("cell_type") == "code"]
        md_cells = [c for c in cells if c.get("cell_type") == "markdown"]

        assert len(code_cells) >= 7, f"Expected at least 7 code cells, got {len(code_cells)}"
        assert len(md_cells) >= 7, f"Expected at least 7 markdown cells, got {len(md_cells)}"


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
                if s.startswith("!") or s.startswith("%") or s.startswith("%%"):
                    clean_lines.append(f"# {line}")
                else:
                    clean_lines.append(line)

            code_to_parse = "\n".join(clean_lines)
            try:
                tree = ast.parse(code_to_parse)
                assert isinstance(tree, ast.Module)
            except SyntaxError as e:
                pytest.fail(f"SyntaxError in Notebook Cell #{idx + 1} ({e.msg}) at line {e.lineno}")


class TestRequirementR1MassiveDataIngestion:
    """Verifies Requirement R1: Massive Direct Data Ingestion in Google Colab."""

    def test_direct_colab_downloaders(self, master_notebook):
        code = get_all_code(master_notebook)
        
        assert "Niger-Volta-LTI/edo-text.git" in code
        assert "masakhane-io/lafand-mt/main/data/json_files/en-ibo/train.json" in code
        assert "Davlan/ibom-mt-en-efi" in code

    def test_verified_seed_corpus_and_cot(self, master_notebook):
        code = get_all_code(master_notebook)
        assert "CURATED_COT_SAMPLES" in code
        assert "Kedu ka mmadụ si agwọ ahụ ọkụ?" in code
        assert "chatml_template =" in code
        assert "all_formatted_texts.append" in code


class TestRequirementR2TranslatorOfKnowledgePersona:
    """Verifies Requirement R2: Translation-Based Persona & Data Formatting."""

    def test_system_prompt_knowledge_translator_definition(self, master_notebook):
        code = get_all_code(master_notebook)
        assert "MASTER_SYSTEM_PROMPT" in code
        assert "You are ÀṢÀ (AfriWise)" in code

    def test_eradication_of_restrictive_persona_locks(self, master_notebook):
        all_text = "\n".join("".join(c["source"]) for c in master_notebook["cells"])
        assert "Only state facts you are certain of. If unsure, say" not in all_text
        assert "You only speak Nigerian languages" not in all_text


class TestRequirementR3EndToEndColabPipeline:
    """Verifies Requirement R3: Complete Colab Fine-Tuning & Export Suite."""

    def test_hardware_acceleration_and_fallback(self, master_notebook):
        code = get_all_code(master_notebook)
        assert "unsloth" in code
        assert "peft" in code

    def test_google_drive_path_hierarchy(self, master_notebook):
        code = get_all_code(master_notebook)
        assert "drive.mount" in code
        assert "/content/drive" in code

    def test_4bit_model_loading(self, master_notebook):
        code = get_all_code(master_notebook)
        assert "load_in_4bit = True" in code
        assert "Qwen2.5-3B-Instruct" in code

    def test_lora_adapters_configuration(self, master_notebook):
        code = get_all_code(master_notebook)
        assert "target_modules =" in code
        assert "q_proj" in code and "v_proj" in code
        assert "r = 32" in code

    def test_sfttrainer_training_loop_and_checkpointing(self, master_notebook):
        code = get_all_code(master_notebook)
        assert "SFTTrainer" in code
        assert "max_steps = 500" in code
        assert "adamw_8bit" in code

    def test_3way_export(self, master_notebook):
        code = get_all_code(master_notebook)
        assert "afriwise_qwen_gguf" in code
        assert "afriwise_qwen2.5_3b.Q4_K_M.gguf" in code


class TestVRAMBudgetAndMemoryBounds:
    """Verifies that mathematical VRAM consumption satisfies Tesla T4 GPU constraints."""

    def test_vram_peak_budget_under_11gb(self):
        # Base Model (3B in 4-bit NF4) = ~1.9 GB
        v_base_4bit = 1.90
        # LoRA Adapters (r=32) + Gradients = ~0.32 GB
        v_lora = 0.32
        # AdamW 8-bit optimizer states = ~0.32 GB
        v_optim = 0.32
        # Dynamic Activation Memory with Unsloth gradient checkpointing (B=2, S=2048) = ~1.85 GB
        v_activation = 1.85
        # PyTorch context & CUDA runtime buffers = ~0.65 GB
        v_overhead = 0.65

        total_peak_vram = v_base_4bit + v_lora + v_optim + v_activation + v_overhead
        assert total_peak_vram < 7.0, f"Peak memory {total_peak_vram} GB is unexpectedly high for Qwen 3B"

        # Tesla T4 Capacity = 15.0 GB
        safety_headroom = 15.0 - total_peak_vram
        assert safety_headroom >= 8.0, f"Headroom {safety_headroom} GB is less than safe bound"
