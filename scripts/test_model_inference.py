"""
Test script for AfriWise model loading and generation with bfloat16 / float32 CPU stability.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import sys
import traceback

def main():
    try:
        print("1. Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", local_files_only=True)
        print("[OK] Tokenizer loaded!")

        # Use bfloat16 on CPU for half-memory and FP32-range stability (no NaNs)
        print("2. Loading Base Model with bfloat16 CPU...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        print("[OK] Base Model loaded in bfloat16! Parameter count:", sum(p.numel() for p in model.parameters()))

        print("3. Attaching AfriWise LoRA Adapter...")
        model = PeftModel.from_pretrained(model, "afriwise_adapters", local_files_only=True)
        print("[OK] LoRA Adapter attached successfully!")

        print("4. Generating test sample...")
        system_prompt = "You are AfriWise — a culturally accurate assistant for southern Nigerian languages (Ibibio/Efik, Igbo, and Edo/Bini)."
        user_query = "Translate 'good morning' into Igbo, Bini-Edo, and Ibibio."
        prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_query}<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=False)

        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("\n" + "=" * 60)
        print("[AFRIWISE RESPONSE]:")
        print(reply.strip())
        print("=" * 60)
    except Exception as e:
        print("ERROR CAUGHT:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
