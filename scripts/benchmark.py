import json
import time
import random
from pathlib import Path
import ollama
import sys
import io

# Force UTF-8 encoding on standard output streams
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def generate_benchmark_dataset():
    print("Generating benchmark dataset from master corpus...")
    igbo_q, efik_q, bini_q = [], [], []
    
    with open("data/processed/afriwise_master_training_dataset.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            messages = data.get("messages", [])
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt = msg["content"]
                    break
                    
            if not prompt: continue
            
            txt = prompt.lower()
            # More specific detection to ensure we get questions that relate to the languages
            if "igbo" in txt or "ndewo" in txt or "kedụ" in txt or "aha m" in txt:
                igbo_q.append(prompt)
            elif "efik" in txt or "ibibio" in txt or "idem mfo" in txt or "emem" in txt:
                efik_q.append(prompt)
            elif "bini" in txt or "edo" in txt or "oba" in txt or "kọyọ" in txt:
                bini_q.append(prompt)
                
    random.seed(42)
    # Get up to 100 per language (some may be fewer if corpus isn't perfectly balanced)
    igbo_final = random.sample(igbo_q, min(100, len(igbo_q)))
    efik_final = random.sample(efik_q, min(100, len(efik_q)))
    bini_final = random.sample(bini_q, min(100, len(bini_q)))
    
    print(f"Extracted: {len(igbo_final)} Igbo, {len(efik_final)} Efik, {len(bini_final)} Bini.")
    
    # If a category is empty due to heuristic failure, add fallback default questions
    if len(igbo_final) == 0:
        igbo_final = ["What is the meaning of ndewo?", "How do you say good morning in Igbo?"] * 50
    if len(efik_final) == 0:
        efik_final = ["What does idem mfo mean?", "Translate peace to Efik."] * 50
    if len(bini_final) == 0:
        bini_final = ["How do you greet in Bini-Edo?", "What are the six special digraphs in Edo?"] * 50
        
    benchmark_data = {
        "igbo": igbo_final[:100],
        "efik": efik_final[:100],
        "bini": bini_final[:100]
    }
    
    with open("data/processed/benchmark_dataset.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
        
    return benchmark_data

def run_benchmark():
    print("=" * 60)
    print("AfriWise Ollama Benchmark Evaluation (100 Questions per Language)")
    print("=" * 60)
    
    SYSTEM_PROMPT = "You are AfriWise — a culturally accurate assistant for southern Nigerian languages (Ibibio/Efik, Igbo, and Edo/Bini). Only state facts you are certain of. If unsure, say: 'A maghị m' (Igbo) / 'Mmọdiọkke' (Efik) / 'I ma-ẹre' (Bini). Preserve all diacritics: ọ, ụ, ị, ẹ, ñ."
    
    benchmark_data = generate_benchmark_dataset()
        
    results = {}
    
    for lang, questions in benchmark_data.items():
        print(f"\n--- Testing {lang.upper()} ({len(questions)} questions) ---")
        lang_results = []
        for i, q in enumerate(questions):
            try:
                print(f"\rProcessing {lang} question {i+1}/{len(questions)}...", end="", flush=True)
                
                # Dynamic language prompt to prevent cross-language hallucination
                lang_hint = ""
                if lang == "igbo":
                    lang_hint = " The user is speaking Igbo. Reply ONLY in Igbo. Do NOT use Bini or Efik words."
                elif lang == "efik":
                    lang_hint = " The user is speaking Efik/Ibibio. Reply ONLY in Efik/Ibibio. Do NOT use Igbo or Bini words."
                elif lang == "bini":
                    lang_hint = " The user is speaking Bini/Edo. Reply ONLY in Bini/Edo. Do NOT use Igbo or Efik words."

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT + lang_hint},
                    {"role": "user", "content": q}
                ]
                
                start_time = time.time()
                
                response = ollama.chat(
                    model="afriwise",
                    messages=messages,
                    stream=False,
                    options={
                        "temperature": 0.1,
                        "repeat_penalty": 1.0,
                        "top_p": 0.9
                    }
                )
                output = response['message']['content']
                latency = time.time() - start_time
                
                lang_results.append({
                    "question": q,
                    "response": output,
                    "latency": latency
                })
            except Exception as e:
                print(f"\nError on {lang} question {i+1}: {str(e)}")
                break
                
        results[lang] = lang_results
        
    print("\n\nWriting results to benchmark_results_v2.md...")
    
    with open("benchmark_results_v2.md", "w", encoding="utf-8") as f:
        f.write("# AfriWise Benchmark Results V2 (Ollama 16-bit GGUF)\n\n")
        f.write("Native LoRA inference performed using local Ollama engine.\n\n")
        for lang, lang_results in results.items():
            f.write(f"## {lang.upper()}\n")
            avg_latency = sum(r['latency'] for r in lang_results) / len(lang_results) if lang_results else 0
            f.write(f"**Average Latency**: {avg_latency:.2f}s\n\n")
            
            # Print a few random questions for review
            sample = random.sample(lang_results, min(5, len(lang_results)))
            for i, res in enumerate(sample): 
                f.write(f"### Q{i+1}: {res['question']}\n")
                f.write(f"**Response**: {res['response']}\n\n")
                f.write("---\n")
                
    print("Done!")

if __name__ == "__main__":
    run_benchmark()
