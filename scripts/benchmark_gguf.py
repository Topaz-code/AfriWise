"""
benchmark_gguf.py — AfriWise Anti-Hallucination Benchmark using GGUF + llama-cpp-python.

Uses Q4_K_M quantized Phi-3 Mini (~2.3 GB RAM) instead of full bfloat16 (~7.6 GB).
This is the correct inference method for a 15.4 GB RAM CPU-only machine.

Usage:
    python scripts/benchmark_gguf.py
    python scripts/benchmark_gguf.py --fast          (structure check only)
    python scripts/benchmark_gguf.py --gguf path/to/model.gguf
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_GGUF = Path("models/gguf/phi-3-mini-4k-instruct-q4_k_m.gguf")

# ============================================================
# AFRIWISE SYSTEM PROMPT — baked into every query
# ============================================================
SYSTEM_PROMPT = (
    "You are AfriWise — a culturally accurate assistant for southern Nigerian languages "
    "(Ibibio/Efik, Igbo, and Edo/Bini). Only state facts you are certain of. "
    "If unsure, say: 'A maghị m' (Igbo) / 'Mmọdiọkke' (Efik) / 'I ma-ẹre' (Bini). "
    "Preserve all diacritics. Never confuse these languages with Yoruba, Hausa, or Swahili."
)

# ============================================================
# GOLD STANDARD BENCHMARK — 26 verified questions
# ============================================================
BENCHMARK = [
    # ===== IGBO (8) =====
    {
        "id": "igbo_001", "lang": "Igbo", "category": "translation",
        "question": "Translate 'good morning' into Igbo.",
        "must_contain_any": ["ututu oma", "ụtụtụ ọ ma"],
        "must_not_contain": ["yoruba", "hausa", "swahili", "emem", "ekabo"],
    },
    {
        "id": "igbo_002", "lang": "Igbo", "category": "proverb",
        "question": "What does the Igbo proverb 'Ilu bu mmanu e ji eri okwu' mean?",
        "must_contain_any": ["palm oil", "words", "proverb", "speech"],
        "must_not_contain": ["swahili", "hausa", "yoruba"],
    },
    {
        "id": "igbo_003", "lang": "Igbo", "category": "culture",
        "question": "What is 'Chi' in Igbo spiritual belief?",
        "must_contain_any": ["personal god", "destiny", "chi", "guardian", "spirit"],
        "must_not_contain": ["quran", "islam is"],
    },
    {
        "id": "igbo_004", "lang": "Igbo", "category": "translation",
        "question": "How do you say 'I don't know' in Igbo?",
        "must_contain_any": ["a maghị m", "a maghi m", "amagh"],
        "must_not_contain": ["emem", "mmodiokke", "i ma-ere"],
    },
    {
        "id": "igbo_005", "lang": "Igbo", "category": "folklore",
        "question": "Why does the tortoise (Mbe) have a cracked shell in Igbo folklore?",
        "must_contain_any": ["sky", "birds", "feathers", "fell", "cracked", "mbe"],
        "must_not_contain": [],
    },
    {
        "id": "igbo_006", "lang": "Igbo", "category": "translation",
        "question": "What is 'family' in Igbo?",
        "must_contain_any": ["ezinụlọ", "ezinulo", "ụmụnna", "umunna"],
        "must_not_contain": [],
    },
    {
        "id": "igbo_007", "lang": "Igbo", "category": "grammar",
        "question": "Is Igbo a tonal language?",
        "must_contain_any": ["yes", "tonal", "tones", "high", "low"],
        "must_not_contain": ["not tonal", "no tones", "non-tonal"],
    },
    {
        "id": "igbo_008", "lang": "Igbo", "category": "translation",
        "question": "Translate 'unity is strength' to Igbo.",
        "must_contain_any": ["igwe bụ ike", "igwe bu ike"],
        "must_not_contain": [],
    },

    # ===== BINI-EDO (8) =====
    {
        "id": "bini_001", "lang": "Bini-Edo", "category": "mythology",
        "question": "Who is Osanobua in Bini-Edo cosmology?",
        "must_contain_any": ["osanobua", "creator", "supreme", "almighty", "god"],
        "must_not_contain": ["allah", "ifa", "yoruba deity"],
    },
    {
        "id": "bini_002", "lang": "Bini-Edo", "category": "culture",
        "question": "What is the full royal title of the Oba of Benin?",
        "must_contain_any": ["omo n'oba", "uku akpolokpolo", "edo"],
        "must_not_contain": [],
    },
    {
        "id": "bini_003", "lang": "Bini-Edo", "category": "creation",
        "question": "In Bini mythology, how was the land (Igodomigodo) created?",
        "must_contain_any": ["snail shell", "sand", "water", "youngest", "igodomigodo", "land"],
        "must_not_contain": [],
    },
    {
        "id": "bini_004", "lang": "Bini-Edo", "category": "orthography",
        "question": "What are the six special consonant digraphs in Bini-Edo?",
        "must_contain_any": ["gh", "kh", "gb", "kp", "vb", "mw"],
        "must_not_contain": ["th", "zh"],
    },
    {
        "id": "bini_005", "lang": "Bini-Edo", "category": "spiritual",
        "question": "What is 'Ehi' in Bini spiritual philosophy?",
        "must_contain_any": ["ehi", "guardian", "spirit", "destiny", "erinmwin", "soul"],
        "must_not_contain": [],
    },
    {
        "id": "bini_006", "lang": "Bini-Edo", "category": "proverb",
        "question": "What does the Bini proverb 'Obo oguo o vha guese ache' mean?",
        "must_contain_any": ["one hand", "pot", "community", "collaboration", "alone"],
        "must_not_contain": [],
    },
    {
        "id": "bini_007", "lang": "Bini-Edo", "category": "geography",
        "question": "Where is the Bini-Edo language primarily spoken?",
        "must_contain_any": ["edo state", "benin city", "nigeria"],
        "must_not_contain": ["accra", "nairobi", "south africa", "kenya"],
    },
    {
        "id": "bini_008", "lang": "Bini-Edo", "category": "translation",
        "question": "What is 'Erinmwin' in Bini-Edo?",
        "must_contain_any": ["erinmwin", "spiritual realm", "heaven", "spirit world"],
        "must_not_contain": [],
    },

    # ===== EFIK-IBIBIO (7) =====
    {
        "id": "efik_001", "lang": "Efik-Ibibio", "category": "greeting",
        "question": "How do you greet someone in Ibibio?",
        "must_contain_any": ["emem", "aya mfo", "nte akamba", "nte idem"],
        "must_not_contain": ["kedu", "ututu", "ẹkabọ"],
    },
    {
        "id": "efik_002", "lang": "Efik-Ibibio", "category": "folklore",
        "question": "Why do the Sun (Utin) and Moon live in the sky in Ibibio legend?",
        "must_contain_any": ["sun", "moon", "utin", "sky", "friends", "stars"],
        "must_not_contain": [],
    },
    {
        "id": "efik_003", "lang": "Efik-Ibibio", "category": "geography",
        "question": "Where do Efik people primarily live?",
        "must_contain_any": ["cross river", "calabar", "nigeria"],
        "must_not_contain": ["lagos", "kano", "abuja", "accra"],
    },
    {
        "id": "efik_004", "lang": "Efik-Ibibio", "category": "culture",
        "question": "What is the Ekpe masquerade society in Efik culture?",
        "must_contain_any": ["ekpe", "masquerade", "society", "secret", "efik"],
        "must_not_contain": [],
    },
    {
        "id": "efik_005", "lang": "Efik-Ibibio", "category": "translation",
        "question": "How do you say 'I don't know' in Efik?",
        "must_contain_any": ["mmodiokke", "mmọdiọkke", "don't know", "not know"],
        "must_not_contain": ["a maghị m", "i ma-ẹre"],
    },
    {
        "id": "efik_006", "lang": "Efik-Ibibio", "category": "translation",
        "question": "What does 'Emem' mean in Ibibio?",
        "must_contain_any": ["peace", "hello", "greeting"],
        "must_not_contain": [],
    },
    {
        "id": "efik_007", "lang": "Efik-Ibibio", "category": "history",
        "question": "What is historically significant about the Efik Bible?",
        "must_contain_any": ["earliest", "first", "19th century", "mission", "bible society", "translation"],
        "must_not_contain": [],
    },

    # ===== CROSS-LANGUAGE (3) =====
    {
        "id": "cross_001", "lang": "ALL", "category": "meta",
        "question": "What three languages does AfriWise specialize in?",
        "must_contain_any": ["igbo", "efik", "bini", "ibibio", "edo"],
        "must_not_contain": ["yoruba is one of the main", "hausa is one", "swahili"],
    },
    {
        "id": "cross_002", "lang": "ALL", "category": "ethics",
        "question": "If I ask you something about a Nigerian language that you don't know, what will you say?",
        "must_contain_any": ["a maghị m", "mmodiokke", "i ma-ẹre", "don't know", "uncertain", "not sure", "cannot confirm"],
        "must_not_contain": ["i know everything", "i am certain of all facts"],
    },
    {
        "id": "cross_003", "lang": "ALL", "category": "translation",
        "question": "Translate 'good morning' into all three AfriWise languages.",
        "must_contain_any": ["ututu", "ekabo", "emem", "abie"],
        "must_not_contain": [],
    },
]


def load_gguf_model(gguf_path: Path):
    """Load Phi-3 Mini GGUF with llama-cpp-python. Uses ~2.3 GB RAM."""
    try:
        from llama_cpp import Llama
    except ImportError:
        logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        sys.exit(1)

    if not gguf_path.exists():
        logger.error(f"GGUF not found: {gguf_path}")
        logger.error("Run first: python scripts/download_gguf.py")
        sys.exit(1)

    size_gb = gguf_path.stat().st_size / 1024**3
    logger.info(f"Loading GGUF: {gguf_path.name} ({size_gb:.2f} GB)")
    logger.info("This takes ~60-90 seconds on CPU. Please wait...")

    model = Llama(
        model_path=str(gguf_path),
        n_ctx=2048,          # context window
        n_threads=4,         # use 4 CPU threads
        n_batch=512,
        verbose=False,
    )
    logger.info("✅ Model loaded.")
    return model


def generate(model, question: str, max_tokens: int = 400) -> str:
    """Generate a response using the GGUF model with AfriWise system prompt."""
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{question}<|end|>\n"
        f"<|assistant|>\n"
    )

    result = model(
        prompt,
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.15,
        stop=["<|end|>", "<|user|>"],
    )
    return result["choices"][0]["text"].strip()


def evaluate(response: str, item: dict) -> dict:
    """Score response against gold standard criteria."""
    resp_lower = response.lower()

    must_any = item.get("must_contain_any", [])
    found_required = any(t.lower() in resp_lower for t in must_any) if must_any else True

    found_forbidden = [t for t in item.get("must_not_contain", []) if t.lower() in resp_lower]

    refusal_patterns = [
        "a maghị m", "mmodiokke", "i ma-ẹre",
        "i don't know", "not certain", "cannot confirm", "i am not sure",
    ]
    has_refusal = any(p in resp_lower for p in refusal_patterns)

    return {
        "id": item["id"],
        "lang": item["lang"],
        "category": item["category"],
        "question": item["question"],
        "response": response,
        "passed": found_required and not found_forbidden,
        "found_required": found_required,
        "hallucinations_detected": found_forbidden,
        "has_honest_refusal": has_refusal,
    }


def main():
    parser = argparse.ArgumentParser(description="AfriWise GGUF Anti-Hallucination Benchmark")
    parser.add_argument("--gguf", default=str(DEFAULT_GGUF), help="Path to GGUF model file")
    parser.add_argument("--output", default=None, help="Output JSON report path")
    parser.add_argument("--fast", action="store_true", help="Print structure only, skip inference")
    args = parser.parse_args()

    if args.fast:
        print(f"\nAfriWise GGUF Benchmark: {len(BENCHMARK)} questions")
        by_lang: dict = {}
        for item in BENCHMARK:
            by_lang.setdefault(item["lang"], []).append(item["id"])
        for lang, ids in by_lang.items():
            print(f"  {lang}: {len(ids)} questions")
        print(f"\nSystem prompt:\n  {SYSTEM_PROMPT[:120]}...")
        return

    gguf_path = Path(args.gguf)
    model = load_gguf_model(gguf_path)

    results = []
    passed_count = 0
    hallucination_count = 0

    print("\n" + "=" * 70)
    print("AfriWise Anti-Hallucination Benchmark (GGUF / CPU-safe)")
    print(f"Model: {gguf_path.name} | Questions: {len(BENCHMARK)}")
    print("=" * 70)

    for i, item in enumerate(BENCHMARK, 1):
        print(f"\n[{i:02d}/{len(BENCHMARK)}] {item['id']} ({item['lang']} / {item['category']})")
        print(f"  Q: {item['question']}")

        response = generate(model, item["question"])
        result = evaluate(response, item)
        results.append(result)

        preview = response[:220] + ("..." if len(response) > 220 else "")
        print(f"  A: {preview}")

        if result["passed"]:
            passed_count += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        if result["hallucinations_detected"]:
            hallucination_count += 1
            status += f" ⚠️  HALLUCINATION: {result['hallucinations_detected']}"

        print(f"  [{status}]")

    total = len(BENCHMARK)
    score = passed_count / total * 100

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS")
    print(f"  Total:          {total}")
    print(f"  Passed:         {passed_count} ({score:.1f}%)")
    print(f"  Failed:         {total - passed_count}")
    print(f"  Hallucinations: {hallucination_count}")

    if score >= 85:
        verdict = "EXCELLENT — ready for deployment"
    elif score >= 70:
        verdict = "GOOD — more training recommended"
    elif score >= 50:
        verdict = "MODERATE — dataset expansion needed"
    else:
        verdict = "NEEDS IMPROVEMENT — check system prompt and retrain"

    print(f"\n  Verdict: {verdict}")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output or f"data/benchmarks/gguf_results_{timestamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": timestamp,
        "model": gguf_path.name,
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "score_pct": round(score, 2),
        "hallucination_count": hallucination_count,
        "verdict": verdict,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nFull report saved: {out_path}")
    return score


if __name__ == "__main__":
    main()
