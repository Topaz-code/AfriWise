"""
test_interactive_dialogue.py — Verify conversational natural dialogue and auto-switching.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.rag_engine import get_knowledge_engine

def main():
    engine = get_knowledge_engine()
    test_prompts = [
        "Ndewo m bụ Topaz, onye guzobere Afriwise, gị na gị?",
        "Good night in Igbo",
        "Translate, 'hello my beautiful princess' to Igbo please",
        "Idem mfo?",
        "What is the meaning of idem mfo? what language was that from?",
        "Kọyọ, I re Topaz"
    ]

    for p in test_prompts:
        res = engine.query(p)
        print("=" * 65)
        print(f"USER: {p}")
        print(f"AFRIWISE:\n{res['answer']}")


if __name__ == "__main__":
    main()
