"""
AfriWise Anti-Hallucination Benchmark Suite.

Tests 25+ factual and linguistic prompts across Bini-Edo, Igbo, and Efik-Ibibio.
Verifies zero hallucination and correct knowledge.

Usage:
  python scripts/benchmark_hallucination.py --adapter afriwise_adapters_v2
  python scripts/benchmark_hallucination.py --adapter afriwise_adapters
  python scripts/benchmark_hallucination.py --fast   (just check structure)
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# GOLD STANDARD BENCHMARK
# All answers verified against authoritative cultural sources
# ============================================================
BENCHMARK = [
    # ===== IGBO (8 questions) =====
    {
        "id": "igbo_001",
        "lang": "Igbo",
        "category": "translation",
        "question": "Translate 'good morning' into Igbo.",
        "must_contain_any": ["ututu oma", "\u1ee5t\u1ee5t\u1ee5 \u1ecd ma"],
        "must_not_contain": ["yoruba", "hausa", "swahili", "emem", "ekabo"],
        "source": "Standard Igbo greetings"
    },
    {
        "id": "igbo_002",
        "lang": "Igbo",
        "category": "proverb",
        "question": "What does the Igbo proverb 'Ilu bu mmanu e ji eri okwu' mean?",
        "must_contain_any": ["palm oil", "words", "proverb", "speech", "mmanu", "okwu"],
        "must_not_contain": ["swahili", "hausa", "yoruba"],
        "source": "Igbo oral tradition"
    },
    {
        "id": "igbo_003",
        "lang": "Igbo",
        "category": "culture",
        "question": "What is 'Chi' in Igbo spiritual belief?",
        "must_contain_any": ["personal god", "destiny", "chi", "guardian", "spirit"],
        "must_not_contain": ["bible is the only", "quran", "islam is"],
        "source": "Igbo traditional religion (Odinala)"
    },
    {
        "id": "igbo_004",
        "lang": "Igbo",
        "category": "translation",
        "question": "How do you say 'I don't know' in Igbo?",
        "must_contain_any": ["a magh\u1ecb m", "a maghi m", "amagh"],
        "must_not_contain": ["emem", "mmodiokke", "i ma-ere"],
        "source": "Standard Igbo"
    },
    {
        "id": "igbo_005",
        "lang": "Igbo",
        "category": "folklore",
        "question": "Why does the tortoise (Mbe) have a cracked shell in Igbo folklore?",
        "must_contain_any": ["sky", "birds", "feathers", "fell", "cracked", "mbe"],
        "must_not_contain": ["efik version", "bini version"],
        "source": "Igbo folklore (widely attested)"
    },
    {
        "id": "igbo_006",
        "lang": "Igbo",
        "category": "translation",
        "question": "What is 'family' in Igbo?",
        "must_contain_any": ["ezin\u1ee5l\u1ecd", "ezinulo", "\u1ee5m\u1ee5nna", "umunna"],
        "must_not_contain": [],
        "source": "Standard Igbo vocabulary"
    },
    {
        "id": "igbo_007",
        "lang": "Igbo",
        "category": "grammar",
        "question": "Is Igbo a tonal language?",
        "must_contain_any": ["yes", "tonal", "tones", "high", "low"],
        "must_not_contain": ["not tonal", "no tones", "non-tonal"],
        "source": "Igbo linguistics"
    },
    {
        "id": "igbo_008",
        "lang": "Igbo",
        "category": "translation",
        "question": "Translate 'unity is strength' to Igbo.",
        "must_contain_any": ["igwe b\u1ee5 ike", "igwe bu ike"],
        "must_not_contain": [],
        "source": "Famous Igbo proverb"
    },

    # ===== BINI-EDO (8 questions) =====
    {
        "id": "bini_001",
        "lang": "Bini-Edo",
        "category": "mythology",
        "question": "Who is Osanobua in Bini-Edo cosmology?",
        "must_contain_any": ["osanobua", "creator", "supreme", "almighty", "god"],
        "must_not_contain": ["allah", "ifa", "yoruba deity"],
        "source": "Edo creation mythology"
    },
    {
        "id": "bini_002",
        "lang": "Bini-Edo",
        "category": "culture",
        "question": "What is the full royal title of the Oba of Benin?",
        "must_contain_any": ["omo n'oba", "uku akpolokpolo", "edo"],
        "must_not_contain": [],
        "source": "Royal court of Benin (Edo State)"
    },
    {
        "id": "bini_003",
        "lang": "Bini-Edo",
        "category": "creation",
        "question": "In Bini mythology, how was the land (Igodomigodo) created?",
        "must_contain_any": ["snail shell", "sand", "water", "youngest", "igodomigodo", "land"],
        "must_not_contain": [],
        "source": "Edo creation myth (widely attested)"
    },
    {
        "id": "bini_004",
        "lang": "Bini-Edo",
        "category": "orthography",
        "question": "What are the six special consonant digraphs in Bini-Edo?",
        "must_contain_any": ["gh", "kh", "gb", "kp", "vb", "mw"],
        "must_not_contain": ["th", "zh"],
        "source": "Bini-Edo orthography standard"
    },
    {
        "id": "bini_005",
        "lang": "Bini-Edo",
        "category": "spiritual",
        "question": "What is 'Ehi' in Bini spiritual philosophy?",
        "must_contain_any": ["ehi", "guardian", "spirit", "destiny", "erinmwin", "soul", "reincarnation"],
        "must_not_contain": [],
        "source": "Edo traditional religion"
    },
    {
        "id": "bini_006",
        "lang": "Bini-Edo",
        "category": "proverb",
        "question": "What does the Bini proverb 'Obo oguo o vha guese ache' mean?",
        "must_contain_any": ["one hand", "pot", "community", "collaboration", "alone"],
        "must_not_contain": [],
        "source": "Bini proverb (attested)"
    },
    {
        "id": "bini_007",
        "lang": "Bini-Edo",
        "category": "geography",
        "question": "Where is the Bini-Edo language primarily spoken?",
        "must_contain_any": ["edo state", "benin city", "nigeria"],
        "must_not_contain": ["accra", "nairobi", "south africa", "kenya"],
        "source": "Geography of Nigerian languages"
    },
    {
        "id": "bini_008",
        "lang": "Bini-Edo",
        "category": "translation",
        "question": "What is 'Erinmwin' in Bini-Edo?",
        "must_contain_any": ["erinmwin", "spiritual realm", "heaven", "spirit world"],
        "must_not_contain": [],
        "source": "Edo cosmology"
    },

    # ===== EFIK-IBIBIO (7 questions) =====
    {
        "id": "efik_001",
        "lang": "Efik-Ibibio",
        "category": "greeting",
        "question": "How do you greet someone in Ibibio?",
        "must_contain_any": ["emem", "aya mfo", "nte akamba", "nte idem"],
        "must_not_contain": ["kedu", "ututu", "ẹkabọ"],
        "source": "Ibibio greetings (attested)"
    },
    {
        "id": "efik_002",
        "lang": "Efik-Ibibio",
        "category": "folklore",
        "question": "Why do the Sun (Utin) and Moon live in the sky in Ibibio legend?",
        "must_contain_any": ["sun", "moon", "utin", "sky", "friends", "stars"],
        "must_not_contain": [],
        "source": "Ibibio oral folklore (attested)"
    },
    {
        "id": "efik_003",
        "lang": "Efik-Ibibio",
        "category": "geography",
        "question": "Where do Efik people primarily live?",
        "must_contain_any": ["cross river", "calabar", "nigeria"],
        "must_not_contain": ["lagos", "kano", "abuja", "accra"],
        "source": "Geography of Nigerian languages"
    },
    {
        "id": "efik_004",
        "lang": "Efik-Ibibio",
        "category": "culture",
        "question": "What is the Ekpe masquerade society in Efik culture?",
        "must_contain_any": ["ekpe", "masquerade", "society", "secret", "efik"],
        "must_not_contain": [],
        "source": "Efik cultural heritage"
    },
    {
        "id": "efik_005",
        "lang": "Efik-Ibibio",
        "category": "translation",
        "question": "How do you say 'I don't know' in Efik?",
        "must_contain_any": ["mmodiokke", "mm\u1ecddi\u1ecdkke", "don't know", "not know"],
        "must_not_contain": ["a magh\u1ecb m", "i ma-\u1eb9re"],
        "source": "Efik language"
    },
    {
        "id": "efik_006",
        "lang": "Efik-Ibibio",
        "category": "translation",
        "question": "What does 'Emem' mean in Ibibio?",
        "must_contain_any": ["peace", "hello", "greeting"],
        "must_not_contain": [],
        "source": "Ibibio vocabulary"
    },
    {
        "id": "efik_007",
        "lang": "Efik-Ibibio",
        "category": "history",
        "question": "What is historically significant about the Efik Bible?",
        "must_contain_any": ["earliest", "first", "19th century", "mission", "bible society", "translation"],
        "must_not_contain": [],
        "source": "History of Bible translation in West Africa"
    },

    # ===== CROSS-LANGUAGE (3 questions) =====
    {
        "id": "cross_001",
        "lang": "ALL",
        "category": "meta",
        "question": "What three languages does AfriWise specialize in?",
        "must_contain_any": ["igbo", "efik", "bini", "ibibio", "edo"],
        "must_not_contain": ["yoruba is one of the main", "hausa is one", "swahili"],
        "source": "AfriWise design spec"
    },
    {
        "id": "cross_002",
        "lang": "ALL",
        "category": "ethics",
        "question": "If I ask you something about a Nigerian language that you don't know, what will you say?",
        "must_contain_any": ["a magh\u1ecb m", "mmodiokke", "i ma-\u1eb9re", "don't know", "uncertain", "not sure", "cannot confirm"],
        "must_not_contain": ["i know everything", "i am certain of all facts"],
        "source": "AfriWise anti-hallucination policy"
    },
    {
        "id": "cross_003",
        "lang": "ALL",
        "category": "translation",
        "question": "Translate 'good morning' into all three AfriWise languages.",
        "must_contain_any": ["ututu", "ekabo", "emem", "abie"],
        "must_not_contain": [],
        "source": "Standard greetings in all 3 languages"
    },
]


def load_model(adapter_path: str):
    """Load Phi-3 Mini with AfriWise LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_id = "microsoft/Phi-3-mini-4k-instruct"
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=False, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model (CPU bfloat16)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False, local_files_only=True
    )

    if adapter_path and Path(adapter_path).exists():
        logger.info(f"Applying LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
        logger.info("Adapter applied successfully.")
    else:
        logger.warning(
            f"No adapter found at '{adapter_path}' — running bare base model. "
            "African language performance will be near zero without the adapter."
        )

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, question: str, max_new_tokens: int = 350) -> str:
    """Generate a response from AfriWise."""
    import torch

    system_prompt = (
        "You are AfriWise — a culturally accurate assistant for southern Nigerian languages "
        "(Ibibio/Efik, Igbo, and Edo/Bini). Only state facts you are certain of. "
        "If unsure, say: 'A magh\u1ecb m' (Igbo) / 'Mm\u1ecddi\u1ecdkke' (Efik) / 'I ma-\u1eb9re' (Bini)."
    )
    prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return gen.strip()


def evaluate_response(response: str, item: dict) -> dict:
    """Check response against gold standard criteria."""
    resp_lower = response.lower()

    must_any = item.get("must_contain_any", [])
    found_required = (
        any(term.lower() in resp_lower for term in must_any) if must_any else True
    )

    must_not = item.get("must_not_contain", [])
    found_forbidden = [term for term in must_not if term.lower() in resp_lower]

    refusal_patterns = [
        "a magh\u1ecb m", "mmodiokke", "i ma-\u1eb9re",
        "i don't know", "not certain", "cannot confirm", "i am not sure"
    ]
    has_refusal = any(p in resp_lower for p in refusal_patterns)

    passed = found_required and not found_forbidden

    return {
        "id": item["id"],
        "lang": item["lang"],
        "category": item["category"],
        "question": item["question"],
        "response": response,
        "passed": passed,
        "found_required": found_required,
        "hallucinations_detected": found_forbidden,
        "has_honest_refusal": has_refusal,
        "source": item["source"],
    }


def main():
    parser = argparse.ArgumentParser(description="AfriWise Anti-Hallucination Benchmark")
    parser.add_argument(
        "--adapter", default="afriwise_adapters_v2",
        help="Path to LoRA adapter directory (default: afriwise_adapters_v2)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON report path (auto-generated if not specified)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip model loading — just print benchmark structure"
    )
    args = parser.parse_args()

    if args.fast:
        print(f"AfriWise Benchmark: {len(BENCHMARK)} questions")
        by_lang = {}
        for item in BENCHMARK:
            by_lang.setdefault(item["lang"], []).append(item["id"])
        for lang, ids in by_lang.items():
            print(f"  {lang}: {len(ids)} questions ({', '.join(ids[:3])}...)")
        return

    model, tokenizer = load_model(args.adapter)

    results = []
    passed = 0
    failed = 0
    hallucinations = 0

    print("\n" + "=" * 70)
    print("AfriWise Anti-Hallucination Benchmark")
    print(f"Adapter: {args.adapter} | Questions: {len(BENCHMARK)}")
    print("=" * 70)

    for i, item in enumerate(BENCHMARK, 1):
        print(f"\n[{i}/{len(BENCHMARK)}] {item['id']} ({item['lang']} / {item['category']})")
        print(f"Q: {item['question']}")

        response = generate(model, tokenizer, item["question"])
        result = evaluate_response(response, item)
        results.append(result)

        preview = response[:200] + ("..." if len(response) > 200 else "")
        print(f"A: {preview}")

        if result["passed"]:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        if result["hallucinations_detected"]:
            hallucinations += 1
            status += f" [HALLUCINATION: {result['hallucinations_detected']}]"

        print(f"   [{status}]")

    # Print summary
    total = len(BENCHMARK)
    score = (passed / total) * 100
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print(f"  Total questions:       {total}")
    print(f"  Passed:                {passed} ({score:.1f}%)")
    print(f"  Failed:                {failed}")
    print(f"  Hallucinations found:  {hallucinations}")

    if score >= 85:
        verdict = "EXCELLENT — ready for deployment"
    elif score >= 70:
        verdict = "GOOD — more training recommended"
    elif score >= 50:
        verdict = "MODERATE — dataset expansion needed"
    else:
        verdict = "NEEDS SIGNIFICANT IMPROVEMENT — check adapter and retrain"

    print(f"\n  Verdict: {verdict}")
    print("=" * 70)

    # Save JSON report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output or f"data/benchmarks/results_{timestamp}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": timestamp,
        "adapter": args.adapter,
        "total": total,
        "passed": passed,
        "failed": failed,
        "score_pct": round(score, 2),
        "hallucination_count": hallucinations,
        "verdict": verdict,
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nFull report saved: {out_path}")
    return score


if __name__ == "__main__":
    main()
