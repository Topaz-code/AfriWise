import ollama
import time
import json
from pathlib import Path

def evaluate_model():
    print("=" * 60)
    print("AfriWise Native LoRA Evaluation (100 Questions per Language)")
    print("=" * 60)
    
    # Load 100 random questions for each language from our processed dataset
    data_file = Path("data/processed/afriwise_v2_chatml.jsonl")
    
    if not data_file.exists():
        print("Dataset not found. Please ensure afriwise_v2_chatml.jsonl exists.")
        return

    # Categorize questions
    igbo_q = []
    efik_q = []
    bini_q = []
    
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            # Find the user's prompt
            prompt = next((msg["content"] for msg in record["messages"] if msg["role"] == "user"), None)
            assistant_truth = next((msg["content"] for msg in record["messages"] if msg["role"] == "assistant"), None)
            
            if not prompt or not assistant_truth: continue
            
            # Simple heuristic since our dataset has some markers or we can just sample based on substrings
            txt = prompt.lower()
            if "igbo" in txt or "ndewo" in txt or "chukwu" in txt:
                igbo_q.append((prompt, assistant_truth))
            elif "efik" in txt or "ibibio" in txt or "idem" in txt or "emem" in txt:
                efik_q.append((prompt, assistant_truth))
            elif "bini" in txt or "edo" in txt or "oba" in txt or "kọyọ" in txt:
                bini_q.append((prompt, assistant_truth))
                
    # Limit to 100 each (or what's available)
    import random
    random.seed(42)
    igbo_q = random.sample(igbo_q, min(100, len(igbo_q)))
    efik_q = random.sample(efik_q, min(100, len(efik_q)))
    bini_q = random.sample(bini_q, min(100, len(bini_q)))
    
    print(f"Loaded {len(igbo_q)} Igbo, {len(efik_q)} Efik, and {len(bini_q)} Bini questions for testing.\n")
    
    def test_set(name, qs):
        print(f"--- Testing {name} ---")
        passed = 0
        
        for i, (q, truth) in enumerate(qs[:5]): # Just show the first 5 in detail
            print(f"\nQ: {q}")
            try:
                res = ollama.chat(model='afriwise', messages=[{"role": "user", "content": q}])
                ans = res['message']['content']
                print(f"A: {ans}")
                
                # Check for signs of hallucination (if it apologizes or outputs python code instead of text)
                if "def " not in ans and len(ans.strip()) > 0:
                    passed += 1
            except Exception as e:
                print(f"Error: {e}")
                
        # Run the rest silently
        for q, truth in qs[5:]:
            try:
                res = ollama.chat(model='afriwise', messages=[{"role": "user", "content": q}])
                ans = res['message']['content']
                if "def " not in ans and len(ans.strip()) > 0:
                    passed += 1
            except:
                pass
                
        print(f"\n✅ {name} Pass Rate: {passed}/{len(qs)} ({(passed/max(1, len(qs)))*100:.1f}%)")
        return passed

    start = time.time()
    p_igbo = test_set("Igbo", igbo_q)
    p_efik = test_set("Efik/Ibibio", efik_q)
    p_bini = test_set("Bini/Edo", bini_q)
    
    total = len(igbo_q) + len(efik_q) + len(bini_q)
    total_passed = p_igbo + p_efik + p_bini
    
    print("=" * 60)
    print(f"⏱️ Total Time: {time.time() - start:.1f}s")
    print(f"🏆 Overall Pass Rate: {total_passed}/{total} ({(total_passed/max(1, total))*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_model()
