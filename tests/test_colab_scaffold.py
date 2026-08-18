import ast
import json
import os
import re
import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(ROOT_DIR, "notebooks", "AfriWise_Finetune.ipynb")
MODELFILE_PATH = os.path.join(ROOT_DIR, "Modelfile")
GUIDE_PATH = os.path.join(ROOT_DIR, "notebooks", "COLAB_TRAINING_GUIDE.md")


def test_notebook_file_exists():
    """Verify that the fine-tuning notebook exists in the notebooks/ folder."""
    assert os.path.exists(NOTEBOOK_PATH), f"Notebook missing at {NOTEBOOK_PATH}"
    assert os.path.getsize(NOTEBOOK_PATH) > 0, "Notebook file is empty"


def test_notebook_valid_json_schema():
    """Verify that AfriWise_Finetune.ipynb is a valid Jupyter Notebook JSON schema."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert isinstance(nb_data, dict), "Notebook JSON must be an object"
    assert "cells" in nb_data, "Notebook missing 'cells' key"
    assert "metadata" in nb_data, "Notebook missing 'metadata' key"
    assert "nbformat" in nb_data and nb_data["nbformat"] >= 4, "Notebook format must be >= 4"
    assert isinstance(nb_data["cells"], list), "Notebook cells must be a list"
    assert len(nb_data["cells"]) >= 10, f"Expected at least 10 cells, got {len(nb_data['cells'])}"


def test_notebook_contains_required_pipeline_cells():
    """Verify that all essential fine-tuning pipeline steps are present in notebook cells."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    all_code = ""
    all_markdown = ""

    for cell in nb_data["cells"]:
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if cell_type == "code":
            all_code += "\n" + source
        elif cell_type == "markdown":
            all_markdown += "\n" + source

    # 1. Transformers & dependencies
    assert "transformers" in all_code.lower(), "Missing transformers installation/import"
    assert "peft" in all_code or "peft" in all_markdown, "Missing peft"
    assert "trl" in all_code or "trl" in all_markdown, "Missing trl"
    assert "bitsandbytes" in all_code or "bitsandbytes" in all_markdown, "Missing bitsandbytes"

    # 2. Base model loading with 4-bit
    assert "AutoModelForCausalLM.from_pretrained" in all_code, "Missing AutoModelForCausalLM.from_pretrained call"
    assert "Phi-3" in all_code or "Phi-3.5" in all_code, "Missing Phi-3 base model name"
    assert "BitsAndBytesConfig" in all_code or "load_in_4bit" in all_code, "Missing 4-bit configuration"

    # 3. LoRA configuration
    assert "LoraConfig" in all_code, "Missing LoraConfig call"
    assert "qkv_proj" in all_code or "q_proj" in all_code, "Missing LoRA attention targets"
    assert "r=16" in all_code or "r = 16" in all_code, "LoRA rank r should be 16"

    # 4. ChatML formatting & dataset loading
    assert "apply_chat_template" in all_code, "Missing apply_chat_template setup"
    assert "training_dataset_chatml.jsonl" in all_code, "Missing reference to training_dataset_chatml.jsonl"

    # 5. SFTTrainer setup
    assert "SFTTrainer" in all_code, "Missing SFTTrainer initialization"
    assert "TrainingArguments" in all_code or "SFTConfig" in all_code, "Missing TrainingArguments/SFTConfig initialization"
    assert "2e-4" in all_code, "Expected learning rate of 2e-4"
    assert "cosine" in all_code, "Expected cosine learning rate scheduler"

    # 6. Pre and post inference evaluation
    assert "model.generate" in all_code, "Missing model.generate inference call"

    # 7. Model/Adapter saving
    assert "save_pretrained" in all_code, "Missing save_pretrained call"


def test_notebook_code_cells_valid_python_syntax():
    """Verify that every Python code cell in the notebook compiles without syntax errors."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    for idx, cell in enumerate(nb_data["cells"]):
        if cell.get("cell_type") != "code":
            continue

        raw_source = "".join(cell.get("source", []))
        # Filter out Jupyter/Colab magic commands (lines starting with !, %, %%)
        clean_lines = []
        for line in raw_source.splitlines():
            stripped = line.strip()
            if stripped.startswith("!") or stripped.startswith("%") or stripped.startswith("%%"):
                clean_lines.append(f"# {line}")  # Comment out magics for ast parsing
            else:
                clean_lines.append(line)

        clean_code = "\n".join(clean_lines)
        try:
            ast.parse(clean_code)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in notebook code cell #{idx + 1}: {e}")


def test_modelfile_exists_and_valid():
    """Verify that Modelfile exists and contains valid Ollama directives."""
    assert os.path.exists(MODELFILE_PATH), f"Modelfile missing at {MODELFILE_PATH}"
    with open(MODELFILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify FROM
    assert "FROM" in content, "Modelfile missing FROM directive"
    assert (".gguf" in content.lower() or "phi3" in content.lower()), "Modelfile FROM should point to a .gguf file or base model"

    # Verify System prompt
    assert "SYSTEM" in content, "Modelfile missing SYSTEM directive"
    assert "AfriWise" in content, "Modelfile SYSTEM prompt must mention AfriWise"
    assert "Ibibio" in content, "Modelfile SYSTEM prompt must mention Ibibio"
    assert "Igbo" in content, "Modelfile SYSTEM prompt must mention Igbo"
    assert ("Edo" in content or "Bini" in content), "Modelfile SYSTEM prompt must mention Edo/Bini"

    # Verify Parameters
    assert re.search(r"PARAMETER\s+temperature", content), "Modelfile missing PARAMETER temperature"
    assert re.search(r"PARAMETER\s+top_p", content), "Modelfile missing PARAMETER top_p"
    assert "PARAMETER stop" in content, "Modelfile missing PARAMETER stop directives"

    # Verify Template
    assert "TEMPLATE" in content, "Modelfile missing TEMPLATE directive"
    assert ("<|im_start|>" in content or "<|user|>" in content), "Modelfile TEMPLATE should use valid prompt template tokens"


def test_colab_training_guide_exists_and_complete():
    """Verify that COLAB_TRAINING_GUIDE.md exists and covers all deployment steps."""
    assert os.path.exists(GUIDE_PATH), f"Guide missing at {GUIDE_PATH}"
    with open(GUIDE_PATH, "r", encoding="utf-8") as f:
        guide = f.read()

    assert len(guide) > 500, "Guide is too short"
    assert "Google Colab" in guide, "Guide should mention Google Colab"
    assert "T4" in guide, "Guide should specify T4 GPU"
    assert "training_dataset_chatml.jsonl" in guide, "Guide should reference ChatML dataset"
    assert "Modelfile" in guide, "Guide should mention Modelfile"
    assert "ollama create" in guide, "Guide should document ollama create command"
    assert "ollama run" in guide, "Guide should document ollama run command"
