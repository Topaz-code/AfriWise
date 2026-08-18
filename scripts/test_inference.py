import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

def main():
    base_model_id = "microsoft/Phi-3-mini-4k-instruct"
    adapter_dir = "afriwise_adapters"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    print("Loading base model (this takes a moment)...")
    # Load in bfloat16 to save memory
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    # We do NOT merge_and_unload() to save memory and avoid disk/OOM errors
    model.eval()
    print("\n✅ AfriWise LoRA Model Loaded Successfully!\n")
    
    while True:
        try:
            prompt = input("\nUser: ")
            if prompt.strip().lower() in ["quit", "exit"]:
                break
                
            messages = [
                {"role": "system", "content": "You are AfriWise — a culturally accurate assistant for southern Nigerian languages (Ibibio/Efik, Igbo, and Edo/Bini). Only state facts you are certain of. If unsure, say: 'A maghị m' (Igbo) / 'Mmọdiọkke' (Efik) / 'I ma-ẹre' (Bini). Preserve all diacritics: ọ, ụ, ị, ẹ, ñ."},
                {"role": "user", "content": prompt}
            ]
            
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            
            print("AfriWise: ", end="", flush=True)
            start = time.time()
            
            outputs = model.generate(
                inputs, 
                max_new_tokens=150, 
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )
            
            gen_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            print(gen_text)
            print(f"\n[Generated in {time.time() - start:.1f}s]")
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
