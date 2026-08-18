"""
Local Inference and Interactive Chat Engine for AfriWise.

Loads the fine-tuned AfriWise LoRA adapters on top of Microsoft Phi-3 Mini (3.8B)
for southern Nigerian languages (Ibibio, Igbo, Edo/Bini).

Modes:
  1. Interactive Chat: python scripts/inference_afriwise.py --chat
  2. Single Prompt:     python scripts/inference_afriwise.py --prompt "Translate 'good morning' into Ibibio and Igbo."
  3. Benchmark Suite:   python scripts/inference_afriwise.py --benchmark
"""

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

AFRIWISE_SYSTEM_PROMPT = (
    "You are AfriWise — a culturally accurate assistant and storyteller for "
    "southern Nigerian languages (Ibibio, Igbo, and Edo/Bini). You prioritize "
    "authenticity, respectful dialogue, and proverbs."
)

BENCHMARK_PROMPTS = [
    {
        "category": "Cross-Language Morning Salutations",
        "prompt": "Translate 'good morning, how is the family?' into Ibibio, Igbo, and Edo.",
    },
    {
        "category": "Ibibio Everyday Greetings",
        "prompt": "How do you greet an elder in the morning and ask how they are in Ibibio?",
    },
    {
        "category": "Igbo Proverbs & Ethics",
        "prompt": "Explain the Igbo proverb 'Onye fee eze, eze eruo ya aka' and its moral lesson.",
    },
    {
        "category": "Edo / Bini Sacred Storytelling",
        "prompt": "Tell the ancient Benin story of how Osanobua's youngest child used the snail shell to create the dry land of Igodomigodo.",
    },
    {
        "category": "Igbo Folklore",
        "prompt": "Tell the Igbo folktale about Mbe (the tortoise) and why his shell is cracked.",
    },
    {
        "category": "Ibibio Folklore",
        "prompt": "Why do the Sun (Utin) and Moon (Ọfiọñ) dwell in the sky according to Ibibio legend?",
    },
    {
        "category": "Oba of Benin Royal Honorifics",
        "prompt": "What are the traditional titles and respectful salutations used when addressing the Oba of Benin?",
    },
]


def format_chatml_prompt(user_query: str, system_prompt: str = AFRIWISE_SYSTEM_PROMPT, history: Optional[List[dict]] = None) -> str:
    """Format prompt using standard Phi-3 / ChatML formatting."""
    formatted = f"<|system|>\n{system_prompt}<|end|>\n"
    if history:
        for turn in history:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            formatted += f"<|{role}|>\n{content}<|end|>\n"
    formatted += f"<|user|>\n{user_query}<|end|>\n<|assistant|>\n"
    return formatted


def load_model_and_tokenizer(
    base_model_id: str = "microsoft/Phi-3-mini-4k-instruct",
    adapter_path: Optional[Path] = None,
    load_in_4bit: bool = False,
    device: Optional[str] = None
):
    """Load base model, apply LoRA adapters if present, and return model + tokenizer."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.error("Required packages (torch, transformers) not installed. Run: pip install torch transformers peft")
        sys.exit(1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading tokenizer from {base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model '{base_model_id}' on {device} (4-bit: {load_in_4bit})...")
    torch_dtype = torch.float16 if device == "cuda" else torch.bfloat16

    model_kwargs = {
        "trust_remote_code": False,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }

    if load_in_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
    if device == "cpu":
        model = model.to("cpu")

    if adapter_path and adapter_path.exists():
        try:
            from peft import PeftModel
            abs_adapter = str(adapter_path.resolve())
            logger.info(f"Applying AfriWise LoRA adapter weights from {abs_adapter}...")
            model = PeftModel.from_pretrained(model, abs_adapter, local_files_only=True)
            logger.info("LoRA adapters successfully attached!")
        except ImportError:
            logger.warning("peft library not found. Running with base model weights.")

    model.eval()
    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: str = "cpu",
    max_new_tokens: int = 400,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
) -> str:
    """Generate fluent response from AfriWise."""
    import torch
    
    formatted_input = format_chatml_prompt(prompt)
    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)

    eos_token_ids = [tokenizer.eos_token_id]
    end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if end_token_id is not None and isinstance(end_token_id, int):
        eos_token_ids.append(end_token_id)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated response
    input_len = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(output_tokens[0][input_len:], skip_special_tokens=True)
    return generated_text.strip()


def run_benchmark(model, tokenizer, device: str):
    """Run AfriWise across key linguistic and cultural benchmarks."""
    print("\n" + "=" * 70)
    print("🌍 AfriWise Multi-Task Linguistic & Cultural Benchmark")
    print("=" * 70)

    for idx, item in enumerate(BENCHMARK_PROMPTS, 1):
        cat = item["category"]
        p = item["prompt"]
        print(f"\n[{idx}/{len(BENCHMARK_PROMPTS)}] Category: {cat}")
        print(f"🗣️ Query: {p}")
        print("-" * 50)
        resp = generate_response(model, tokenizer, p, device=device)
        print(f"🤖 AfriWise:\n{resp}\n")
    print("=" * 70 + "\n")


def run_chat_loop(model, tokenizer, device: str):
    """Interactive multi-turn chat session with AfriWise in the terminal."""
    print("\n" + "=" * 70)
    print("🌍 AfriWise Interactive Chat Terminal")
    print("Southern Nigerian Languages Assistant (Ibibio | Igbo | Edo)")
    print("Type 'exit', 'quit', or 'q' to leave.")
    print("=" * 70 + "\n")

    history = []
    while True:
        try:
            user_input = input("🗣️ You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\n👋 Stay well! (Tie sun / Ka ọ dị / Gha khian n'ese)\n")
                break

            response = generate_response(model, tokenizer, user_input, device=device)
            print(f"\n🤖 AfriWise:\n{response}\n" + "-" * 50)
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})
            # Keep history bounded
            if len(history) > 6:
                history = history[-6:]
        except KeyboardInterrupt:
            print("\n\nSession terminated by user.")
            break


def main():
    parser = argparse.ArgumentParser(description="AfriWise Local Inference & Evaluation Suite")
    parser.add_argument("--base-model", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Base model Hugging Face ID")
    parser.add_argument("--adapter-path", type=Path, default=Path("afriwise_adapters"), help="Path to local LoRA adapter weights")
    parser.add_argument("--prompt", type=str, default=None, help="Run a single prompt and print output")
    parser.add_argument("--chat", action="store_true", help="Launch interactive CLI chat session")
    parser.add_argument("--benchmark", action="store_true", help="Run automated benchmark across all 3 languages")
    parser.add_argument("--4bit", dest="load_in_4bit", action="store_true", help="Load base model in 4-bit NormalFloat")
    parser.add_argument("--device", type=str, default=None, help="Explicit device (cuda or cpu)")
    args = parser.parse_args()

    adapter_p = args.adapter_path if args.adapter_path.exists() else None
    model, tokenizer, device = load_model_and_tokenizer(
        base_model_id=args.base_model,
        adapter_path=adapter_p,
        load_in_4bit=args.load_in_4bit,
        device=args.device,
    )

    if args.benchmark:
        run_benchmark(model, tokenizer, device)
    elif args.prompt:
        resp = generate_response(model, tokenizer, args.prompt, device=device)
        print(f"\n🤖 AfriWise:\n{resp}\n")
    else:
        run_chat_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()
