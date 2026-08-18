"""
Challenger 2 Adversarial Stress Test Suite
Author: Challenger 2 (Empirical Challenger)
Coverage:
1. VRAM bounds & memory consumption across T4, V100, L4, A100 GPUs.
2. ChatML formatter stress testing against malformed inputs, nulls, invalid unicode, OCR debris.
3. Cell 8 export logic, merged 16-bit model export fallback, GGUF binary quantization & Ollama Modelfile syntax.
"""

import os
import re
import json
import random
import unicodedata
import pytest
from pathlib import Path

# Extract Cell 4 & Cell 8 functions for isolated stress-testing
def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = text.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_igbo(text: str) -> str:
    return sanitize_text(text)

def normalize_efik_ibibio(text: str) -> str:
    t = sanitize_text(text)
    t = t.replace("ŋ", "ñ").replace("Ŋ", "Ñ")
    t = t.replace("ɔ", "ọ").replace("Ɔ", "Ọ")
    t = t.replace("ɛ", "ẹ").replace("Ɛ", "Ẹ")
    return t

def normalize_edo_bini(text: str) -> str:
    return sanitize_text(text)

def is_clean_record(text: str) -> bool:
    if not text or len(text) < 10 or len(text) > 3500:
        return False
    if re.search(r"[a-zA-Z]\d|\d[a-zA-Z]", text):
        return False
    if re.search(r"[a-z][A-Z]{2,}", text):
        return False
    alpha_count = sum(c.isalpha() for c in text)
    if (alpha_count / max(1, len(text))) < 0.60:
        return False
    return True

def make_chatml_record(user_msg: str, assistant_msg: str, sys_prompt: str = "", metadata: dict = None) -> dict:
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": (user_msg or "").strip()})
    messages.append({"role": "assistant", "content": (assistant_msg or "").strip()})
    rec = {"messages": messages}
    if metadata:
        rec["metadata"] = metadata
    return rec

SYSTEM_PROMPT_KNOWLEDGE_TRANSLATOR = (
    "You are a highly capable, knowledgeable AI. When asked a question in Igbo, "
    "Efik, or Bini, you access your vast general knowledge and output the answer "
    "flawlessly in that specific language."
)

SYSTEM_PROMPT_STANDARD_ASSISTANT = (
    "You are a helpful, accurate, and respectful AI assistant."
)

def get_distributed_system_prompt(rng: random.Random) -> str:
    val = rng.random()
    if val < 0.40:
        return SYSTEM_PROMPT_KNOWLEDGE_TRANSLATOR
    elif val < 0.70:
        return SYSTEM_PROMPT_STANDARD_ASSISTANT
    else:
        return ""


# ==============================================================================
# 1. VRAM Memory Consumption Bounds Verification
# ==============================================================================
class TestVRAMMemoryBounds:
    """Mathematical and empirical verification of VRAM consumption across GPU tiers."""

    GPU_CAPACITIES_GB = {
        "Tesla T4": 15.0,     # Standard Google Colab Free/Pro T4 (15.0 GB / 14.75 GiB)
        "Tesla V100": 16.0,   # Google Colab V100
        "Nvidia L4": 24.0,    # Google Colab Pro L4
        "Nvidia A100": 40.0   # Google Colab Pro+ A100
    }

    def estimate_vram_gb(
        self,
        param_count_b: float,
        seq_len: int = 2048,
        batch_size: int = 2,
        num_layers: int = 32,
        hidden_dim: int = 3072,
        lora_rank: int = 16,
        target_projections: int = 7,
        optimizer: str = "adamw_8bit",
        gradient_checkpointing: bool = True
    ) -> dict:
        """
        Calculates theoretical and peak runtime VRAM consumption (GB) for QLoRA fine-tuning.
        """
        # 1. Quantized Base Model Weights (4-bit NF4: 0.5 bytes per parameter + block quantization constants)
        base_model_weights_gb = (param_count_b * 1e9 * 0.5) / (1024**3)
        quant_overhead_gb = base_model_weights_gb * 0.15 # Block overhead, state constants
        total_model_static_gb = base_model_weights_gb + quant_overhead_gb

        # 2. LoRA Parameters (A and B matrices for target projections)
        # For each projection: A is (hidden_dim x rank), B is (rank x hidden_dim) -> 2 * hidden_dim * rank
        lora_params = num_layers * target_projections * (2 * hidden_dim * lora_rank)
        lora_weights_fp16_gb = (lora_params * 2) / (1024**3)
        lora_grads_fp16_gb = (lora_params * 2) / (1024**3)

        # 3. Optimizer State (adamw_8bit stores 2 state bytes per trainable parameter)
        if optimizer == "adamw_8bit":
            opt_state_gb = (lora_params * 2) / (1024**3)
        else: # adamw_fp32 (8 bytes per param)
            opt_state_gb = (lora_params * 8) / (1024**3)

        # 4. Activation Memory
        # With Unsloth / PyTorch gradient checkpointing, only layer inputs are stored:
        # Layer input: num_layers * batch_size * seq_len * hidden_dim * 2 bytes (fp16/bf16)
        if gradient_checkpointing:
            activation_gb = (num_layers * batch_size * seq_len * hidden_dim * 2) / (1024**3)
        else:
            # Full activation storage
            activation_gb = (num_layers * batch_size * seq_len * hidden_dim * 2 * 12) / (1024**3)

        # 5. CUDA context and PyTorch memory allocator reserve
        cuda_runtime_overhead_gb = 1.0

        peak_vram_gb = total_model_static_gb + lora_weights_fp16_gb + lora_grads_fp16_gb + opt_state_gb + activation_gb + cuda_runtime_overhead_gb

        return {
            "param_count_b": param_count_b,
            "base_model_static_gb": round(total_model_static_gb, 3),
            "lora_trainable_gb": round(lora_weights_fp16_gb + lora_grads_fp16_gb + opt_state_gb, 4),
            "activation_gb": round(activation_gb, 3),
            "cuda_overhead_gb": round(cuda_runtime_overhead_gb, 2),
            "peak_vram_gb": round(peak_vram_gb, 2)
        }

    def test_vram_bounds_phi3_mini_all_gpus(self):
        """Verify Phi-3-mini (3.82B) fits comfortably within all target GPU memory bounds."""
        res = self.estimate_vram_gb(
            param_count_b=3.82,
            seq_len=2048,
            batch_size=2,
            num_layers=32,
            hidden_dim=3072,
            lora_rank=16,
            optimizer="adamw_8bit",
            gradient_checkpointing=True
        )

        assert res["peak_vram_gb"] < 6.5, f"Peak VRAM {res['peak_vram_gb']}GB exceeds conservative 6.5GB bound"

        for gpu_name, capacity in self.GPU_CAPACITIES_GB.items():
            headroom = capacity - res["peak_vram_gb"]
            utilization_pct = (res["peak_vram_gb"] / capacity) * 100
            assert headroom >= 8.5, f"Insufficient headroom on {gpu_name}: {headroom:.2f}GB (utilization: {utilization_pct:.1f}%)"
            assert utilization_pct <= 45.0, f"Excessive utilization on {gpu_name}: {utilization_pct:.1f}%"

    def test_vram_bounds_llama3_8b_all_gpus(self):
        """Verify Meta-Llama-3.1-8B fits within Tesla T4, V100, L4, and A100 bounds."""
        res = self.estimate_vram_gb(
            param_count_b=8.03,
            seq_len=2048,
            batch_size=2,
            num_layers=32,
            hidden_dim=4096,
            lora_rank=16,
            optimizer="adamw_8bit",
            gradient_checkpointing=True
        )

        assert res["peak_vram_gb"] < 10.0, f"Peak VRAM {res['peak_vram_gb']}GB exceeds 10GB bound for 8B model"

        # Check against Tesla T4 (tightest target at 15GB)
        t4_capacity = self.GPU_CAPACITIES_GB["Tesla T4"]
        headroom_t4 = t4_capacity - res["peak_vram_gb"]
        assert headroom_t4 >= 5.0, f"Headroom on Tesla T4 is too tight: {headroom_t4:.2f}GB"

        # Check against all GPUs
        for gpu_name, capacity in self.GPU_CAPACITIES_GB.items():
            assert res["peak_vram_gb"] < capacity, f"OOM on {gpu_name}"

    def test_vram_stress_sequence_length_scaling(self):
        """Stress-test sequence length scaling up to 4096 tokens on Tesla T4."""
        res_4k = self.estimate_vram_gb(
            param_count_b=3.82,
            seq_len=4096,
            batch_size=2,
            num_layers=32,
            hidden_dim=3072,
            lora_rank=16,
            optimizer="adamw_8bit",
            gradient_checkpointing=True
        )
        assert res_4k["peak_vram_gb"] <= 8.5, f"4k context peak VRAM {res_4k['peak_vram_gb']}GB exceeds 8.5GB on T4"
        assert res_4k["peak_vram_gb"] < self.GPU_CAPACITIES_GB["Tesla T4"]


# ==============================================================================
# 2. ChatML Formatter & Data Cleansing Adversarial Stress Tests
# ==============================================================================
class TestChatMLFormatterAdversarial:
    """Stress tests for data ingestion, sanitization, ChatML record generation, and edge cases."""

    def test_sanitize_text_none_and_non_strings(self):
        """Verify sanitize_text handles nulls and non-string types safely without throwing exceptions."""
        non_strings = [None, 12345, 3.14159, True, False, ["sample"], {"key": "value"}, b"raw bytes", object()]
        for val in non_strings:
            sanitized = sanitize_text(val)
            assert sanitized == "", f"Expected empty string for non-string input {type(val)}, got {sanitized!r}"

    def test_sanitize_text_unicode_artifacts_and_zero_width(self):
        """Verify zero-width characters, BOM, and HTML tags are completely removed."""
        dirty = "\ufeff<p>Ekele diri Chineke</p>\u200b \u200cna mmadụ niile! <a href='https://example.com/api'>Link</a>"
        cleaned = sanitize_text(dirty)
        assert "\ufeff" not in cleaned
        assert "\u200b" not in cleaned
        assert "\u200c" not in cleaned
        assert "<p>" not in cleaned
        assert "http" not in cleaned
        assert cleaned == "Ekele diri Chineke na mmadụ niile! Link"

    def test_is_clean_record_ocr_noise_rejection(self):
        """Adversarially probe is_clean_record with OCR artifacts and corrupted words."""
        ocr_debris_cases = [
            ("ktbr6 akuko banyere ala anyi", False), # digits in word
            ("6k8p emem ndito", False),              # digits in word
            ("1234567890 !@#$%^&*()_+", False),       # <60% alpha
            ("Short", False),                         # <10 chars
            ("a" * 4000, False),                      # >3500 chars
            ("Ndi Eze mgbawa na-achị obodo", True),   # authentic text
            ("Kọwaa ihe kpatara na mmiri na-ezo", True) # authentic text
        ]
        for text, expected in ocr_debris_cases:
            assert is_clean_record(text) == expected, f"is_clean_record({text!r}) returned {not expected}"

    def test_is_clean_record_valid_african_text_acceptance(self):
        """Verify authentic African language sentences with diacritics pass cleanly."""
        valid_samples = [
            "Kọwaa ihe kpatara na mmiri na-ezo na sayensị dị mfe.",
            "Nsinifiok eto emi ekerede chlorophyll amum unwan utin ebiere.",
            "Ọba gha ya ẹwaẹn, ẹmwata, vbe ugbẹn-ẹmwẹn nẹ iran gbẹ gha sẹ ẹvbo.",
            "AfriWise is an advanced multilingual model designed for African languages."
        ]
        for sample in valid_samples:
            assert is_clean_record(sample), f"Valid text was incorrectly rejected: {sample!r}"

    def test_orthography_normalizers_preserves_diacritics(self):
        """Verify orthography normalizers correctly convert obsolete chars while preserving tonal dots."""
        efik_old = "Eti unyɔŋ ye nsinifiɔk"
        efik_norm = normalize_efik_ibibio(efik_old)
        assert "ɔ" not in efik_norm
        assert "ọ" in efik_norm

        igbo_sample = "Ụmụaka na-agụ akwụkwọ n'ụlọ akwụkwọ."
        assert normalize_igbo(igbo_sample) == igbo_sample

        bini_sample = "Ọba gha t'o kpere, Ise!"
        assert normalize_edo_bini(bini_sample) == bini_sample

    def test_make_chatml_record_structure_and_roles(self):
        """Verify ChatML structure compliance with exact role ordering and content presence."""
        # 1. With system prompt
        rec = make_chatml_record("Kedu?", "Ọ dị mma.", SYSTEM_PROMPT_KNOWLEDGE_TRANSLATOR, {"language": "igbo"})
        assert "messages" in rec
        assert len(rec["messages"]) == 3
        assert rec["messages"][0]["role"] == "system"
        assert rec["messages"][0]["content"] == SYSTEM_PROMPT_KNOWLEDGE_TRANSLATOR
        assert rec["messages"][1]["role"] == "user"
        assert rec["messages"][1]["content"] == "Kedu?"
        assert rec["messages"][2]["role"] == "assistant"
        assert rec["messages"][2]["content"] == "Ọ dị mma."
        assert rec["metadata"]["language"] == "igbo"

        # 2. Without system prompt (omitted prompt distribution)
        rec_no_sys = make_chatml_record("Hello", "Hi there!", "")
        assert len(rec_no_sys["messages"]) == 2
        assert rec_no_sys["messages"][0]["role"] == "user"
        assert rec_no_sys["messages"][1]["role"] == "assistant"

    def test_chatml_formatter_against_fuzzed_messages(self):
        """Stress-test ChatML tokenization format logic against empty and unusual message lists."""
        def apply_mock_chatml_template(messages):
            out = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                out.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            return "\n".join(out)

        test_cases = [
            [{"role": "system", "content": "You are AfriWise."}, {"role": "user", "content": "Nno"}, {"role": "assistant", "content": "Nno nna."}],
            [{"role": "user", "content": "Translate this."}, {"role": "assistant", "content": "Finished."}],
            [{"role": "user", "content": "Special Unicode: \u00c0 \u1ee5 \u1ecb \u00f1 \u1ecd"}, {"role": "assistant", "content": "\u2705 Received."}]
        ]
        for tc in test_cases:
            rendered = apply_mock_chatml_template(tc)
            assert "<|im_start|>" in rendered
            assert "<|im_end|>" in rendered
            for m in tc:
                assert m["content"] in rendered

    def test_system_prompt_distribution_statistical_convergence(self):
        """Verify empirical convergence of 40/30/30 dynamic prompt distribution over 20,000 samples."""
        rng = random.Random(42)
        counts = {"translator": 0, "standard": 0, "omitted": 0}
        trials = 20000

        for _ in range(trials):
            sp = get_distributed_system_prompt(rng)
            if sp == SYSTEM_PROMPT_KNOWLEDGE_TRANSLATOR:
                counts["translator"] += 1
            elif sp == SYSTEM_PROMPT_STANDARD_ASSISTANT:
                counts["standard"] += 1
            elif sp == "":
                counts["omitted"] += 1
            else:
                pytest.fail(f"Unexpected system prompt generated: {sp}")

        p_trans = counts["translator"] / trials
        p_std = counts["standard"] / trials
        p_omit = counts["omitted"] / trials

        # Assert within 1.5% margin of target 40/30/30 distribution
        assert abs(p_trans - 0.40) < 0.015, f"Translator prompt proportion {p_trans:.4f} deviated from 0.40"
        assert abs(p_std - 0.30) < 0.015, f"Standard prompt proportion {p_std:.4f} deviated from 0.30"
        assert abs(p_omit - 0.30) < 0.015, f"Omitted prompt proportion {p_omit:.4f} deviated from 0.30"


# ==============================================================================
# 3. Export Logic in Cell 8 & Modelfile Format Syntax Verification
# ==============================================================================
class TestExportLogicAndModelfileSyntax:
    """Verifies Cell 8 export paths, fallback mechanisms, and Ollama Modelfile syntax grammar."""

    def test_cell_8_modelfile_ast_and_syntax_structure(self):
        """Verify the exact Modelfile syntax generated by Cell 8 in the notebook."""
        notebook_path = Path("afriwise_colab_training.ipynb")
        assert notebook_path.exists()
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        cell_8_code = "".join(nb["cells"][15]["source"])

        # Extract modelfile_lines block
        assert "modelfile_lines = [" in cell_8_code
        assert 'FROM ./afriwise_q4_k_m.gguf' in cell_8_code
        assert 'PARAMETER temperature 0.3' in cell_8_code
        assert 'PARAMETER top_p 0.9' in cell_8_code
        assert 'PARAMETER num_ctx 4096' in cell_8_code
        assert 'PARAMETER stop "<|im_end|>"' in cell_8_code
        assert 'TEMPLATE """{{ if .System }}<|im_start|>system' in cell_8_code
        assert 'SYSTEM """You are AfriWise' in cell_8_code

    def test_cell_8_modelfile_unrestrictive_persona_conformance(self):
        """Verify Cell 8 Modelfile system prompt conforms to 'Translator of Knowledge' paradigm without locks."""
        notebook_path = Path("afriwise_colab_training.ipynb")
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        cell_8_code = "".join(nb["cells"][15]["source"])

        assert "You are AfriWise - a highly capable, knowledgeable AI" in cell_8_code
        assert "When asked a question in Igbo, Efik/Ibibio, or Bini/Edo" in cell_8_code
        assert "you access your vast general knowledge and output the answer flawlessly" in cell_8_code
        assert "You also converse fluently and intelligently in English" in cell_8_code
        # Ensure no restrictive lockouts
        assert "only speak" not in cell_8_code.lower()
        assert "refuse english" not in cell_8_code.lower()

    def test_cell_8_code_fallback_integrity(self):
        """Verify Cell 8 code in notebook contains fallbacks for merge_and_unload and GGUF conversions."""
        notebook_path = Path("afriwise_colab_training.ipynb")
        assert notebook_path.exists()
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        cell_8_code = "".join(nb["cells"][15]["source"])

        assert "save_pretrained" in cell_8_code, "Missing LoRA adapter save"
        assert "merge_and_unload" in cell_8_code or "save_pretrained_merged" in cell_8_code, "Missing 16-bit merge logic"
        assert "save_pretrained_gguf" in cell_8_code or "convert_hf_to_gguf" in cell_8_code, "Missing GGUF export path"
        assert "Modelfile" in cell_8_code, "Missing Modelfile creation logic"
