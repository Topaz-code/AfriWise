"""
AfriWise Terminal Chat Interface — V2.

Features:
- Phi-3 Mini Instruct base model
- AfriWise LoRA adapter (V2) for Igbo, Bini-Edo, Efik-Ibibio
- Anti-hallucination guardrails
- Language detection and cultural context

Usage:
  python scripts/chat_afriwise.py
  python scripts/chat_afriwise.py --adapter afriwise_adapters_v2
  python scripts/chat_afriwise.py --adapter afriwise_adapters_v2 --no-stream
"""
import argparse
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)  # Suppress HF warnings during chat
logger = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────

BANNER = r"""
  █████╗ ███████╗██████╗ ██╗██╗    ██╗██╗███████╗███████╗
 ██╔══██╗██╔════╝██╔══██╗██║██║    ██║██║██╔════╝██╔════╝
 ███████║█████╗  ██████╔╝██║██║ █╗ ██║██║███████╗█████╗  
 ██╔══██║██╔══╝  ██╔══██╗██║██║███╗██║██║╚════██║██╔══╝  
 ██║  ██║██║     ██║  ██║██║╚███╔███╔╝██║███████║███████╗
 ╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝ ╚══╝╚══╝ ╚═╝╚══════╝╚══════╝
                                                          
  Southern Nigerian AI: Igbo | Bini-Edo | Efik-Ibibio
"""

SYSTEM_PROMPT = (
    "You are AfriWise — a culturally accurate assistant for southern Nigerian languages "
    "(Ibibio/Efik, Igbo, and Edo/Bini). You specialize in: "
    "1. Translations between English and Igbo, Bini-Edo, and Efik-Ibibio. "
    "2. Cultural knowledge, proverbs, folklore, and history of these three peoples. "
    "3. Language learning and grammar explanations. "
    "CRITICAL RULES: Only state facts you are certain of. "
    "If you are unsure about ANY linguistic or cultural fact, you MUST say: "
    "'A maghị m' (Igbo: I don't know) / 'Mmọdiọkke' (Efik: I don't know) / 'I ma-ẹre' (Bini: I don't know). "
    "NEVER invent words, translations, proverbs, or cultural facts. "
    "Preserve all diacritics correctly: ọ, ụ, ị, ẹ, ñ. "
    "Do NOT claim expertise in Yoruba, Hausa, or other languages — only Igbo, Bini-Edo, and Efik-Ibibio."
)

HELP_TEXT = """
AfriWise Commands:
  /help       Show this help
  /clear      Clear conversation history
  /lang       Toggle language preference (auto-detect / English / Igbo / Bini / Efik)
  /bench      Run a quick 5-question self-test
  /quit       Exit AfriWise

Example questions:
  "How do you say 'good morning' in Igbo?"
  "Tell me the Bini creation story"
  "What is the Ekpe masquerade society?"
  "Translate 'family' to all three languages"
  "What does 'Ilu bu mmanu e ji eri okwu' mean?"
"""

QUICK_BENCHMARK = [
    ("How do you say 'good morning' in Igbo?", ["ututu oma", "ụtụtụ ọma"]),
    ("Who is Osanobua?", ["creator", "supreme", "edo", "osanobua"]),
    ("What does 'emem' mean in Ibibio?", ["peace", "greeting", "hello"]),
    ("What three languages does AfriWise specialize in?", ["igbo", "bini", "efik", "ibibio", "edo"]),
    ("If you don't know something, what do you say in Igbo?", ["a magh", "don't know", "uncertain"]),
]


# ─── Model Loading ──────────────────────────────────────────────────────────────

def load_model(adapter_path: str):
    """Load Phi-3 Mini with AfriWise LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = "microsoft/Phi-3-mini-4k-instruct"

    print("Loading AfriWise... (first load takes 3-5 minutes on CPU)")
    print("(Subsequent loads are faster thanks to caching)")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        base_id, trust_remote_code=False, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel
        print(f"Applying AfriWise cultural adapter ({adapter_path})...")
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
        print("Adapter loaded — cultural knowledge active.")
    else:
        print(f"WARNING: Adapter not found at '{adapter_path}'.")
        print("Running base Phi-3 Mini without AfriWise fine-tuning.")
        print("Language quality will be limited until training is complete.")

    model.eval()
    print("Model ready!\n")
    return model, tokenizer


# ─── Generation ─────────────────────────────────────────────────────────────────

def build_phi3_prompt(history: list) -> str:
    """Convert chat history to Phi-3 format."""
    prompt = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
    for role, content in history:
        if role == "user":
            prompt += f"<|user|>\n{content}<|end|>\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}<|end|>\n"
    prompt += "<|assistant|>\n"
    return prompt


def generate_response(model, tokenizer, history: list, max_new_tokens: int = 400) -> str:
    """Generate AfriWise response."""
    import torch

    prompt = build_phi3_prompt(history)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Strip any leaked prompt artifacts
    for marker in ["<|user|>", "<|system|>", "<|end|>", "<|assistant|>"]:
        response = response.replace(marker, "").strip()

    return response


# ─── Quick Benchmark ────────────────────────────────────────────────────────────

def run_quick_bench(model, tokenizer):
    """Run a 5-question quick benchmark."""
    print("\n" + "─" * 60)
    print("Quick AfriWise Self-Test (5 questions)")
    print("─" * 60)
    passed = 0
    for i, (q, keywords) in enumerate(QUICK_BENCHMARK, 1):
        print(f"\n[{i}] {q}")
        response = generate_response(model, tokenizer, [("user", q)])
        print(f"    A: {response[:200]}")
        ok = any(k.lower() in response.lower() for k in keywords)
        print(f"    {'✓ PASS' if ok else '✗ FAIL'}")
        if ok:
            passed += 1

    print(f"\nResults: {passed}/{len(QUICK_BENCHMARK)} passed")
    print("─" * 60 + "\n")


# ─── Main Chat Loop ──────────────────────────────────────────────────────────────

def chat_loop(model, tokenizer):
    """Interactive terminal chat."""
    history = []  # List of (role, content) tuples

    print(BANNER)
    print("Type your question in English or any of the three languages.")
    print("Type /help for commands, /quit to exit.\n")
    print("─" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! (Ọ dị mma / Emem / Ọ ya!)")
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input.lower() in ("/quit", "/exit", "/bye", "exit", "quit"):
            print("\nGoodbye! (Ọ dị mma / Emem / Ọ ya!)")
            break
        elif user_input.lower() == "/help":
            print(HELP_TEXT)
            continue
        elif user_input.lower() == "/clear":
            history.clear()
            print("Conversation cleared.")
            continue
        elif user_input.lower() == "/bench":
            run_quick_bench(model, tokenizer)
            continue

        # ── Generate response ──
        history.append(("user", user_input))
        print("\nAfriWise: ", end="", flush=True)
        t0 = time.time()

        response = generate_response(model, tokenizer, history)
        elapsed = time.time() - t0

        print(response)
        print(f"\n  [{elapsed:.1f}s]")

        history.append(("assistant", response))

        # Keep context window manageable
        if len(history) > 20:
            history = history[-20:]


# ─── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AfriWise Terminal Chat")
    parser.add_argument(
        "--adapter", default="afriwise_adapters_v2",
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Disable token streaming (simpler output)"
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter)
    chat_loop(model, tokenizer)


if __name__ == "__main__":
    main()
