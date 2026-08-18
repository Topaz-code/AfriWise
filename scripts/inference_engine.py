import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import threading

class AfriwiseEngine:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AfriwiseEngine, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        self.base_model_id = "microsoft/Phi-3-mini-4k-instruct"
        self.lora_dir = "afriwise_adapters"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id, 
            trust_remote_code=False, 
            local_files_only=True
        )
        
        print(f"Loading Base Model to {self.device}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            local_files_only=True
        )
        
        print("Applying Native LoRA Adapters...")
        self.model = PeftModel.from_pretrained(base_model, self.lora_dir, local_files_only=True)
        if self.device == "cpu":
            self.model.to(torch.float32) # Better for CPU
            
        # Put into eval mode
        self.model.eval()

    def generate(self, messages, max_new_tokens=150):
        # Format messages into ChatML prompt
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response

def get_engine():
    return AfriwiseEngine()
