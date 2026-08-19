"""
AfriWise RAG & Universal Cognitive Translation Engine.
Uses a Markdown-based reasoning bridge compatible with standard 3B Instruct models.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LEXICON_PATH = BASE_DIR / "knowledge_base" / "lexicon_grounding.json"

class AfriWiseRAG:
    def __init__(self):
        self.lexicon = {}
        if LEXICON_PATH.exists():
            with open(LEXICON_PATH, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)

    def retrieve_grounding_context(self, user_query: str) -> str:
        q_lower = user_query.lower()
        context_items = []

        # 1. Greetings & Vocabulary
        for item in self.lexicon.get("greetings", []):
            eng_tokens = [w.strip("?,.!") for w in item["english"].lower().split()]
            if any(tok in q_lower for tok in eng_tokens if len(tok) > 3) or \
               any(tok in q_lower for tok in ["morning", "hello", "hi", "name", "who", "thank", "welcome"]):
                context_items.append(
                    f"Greeting Mapping: English '{item['english']}' -> Igbo: '{item['igbo']}' | Efik: '{item['efik']}' | Edo: '{item['edo']}'"
                )

        # 2. Medical Protocols
        if any(w in q_lower for w in ["fever", "ahụ ọkụ", "ufip idem", "ẹrhẹn egbe", "bleed", "wound", "ọbara", "iyịp", "ẹzẹ"]):
            for med in self.lexicon.get("medical_first_aid", []):
                context_items.append(
                    f"Verified First-Aid Protocol: [{med['condition']}]\n"
                    f"- English: {med['english']}\n"
                    f"- Igbo: {med['igbo']}\n"
                    f"- Efik: {med['efik']}\n"
                    f"- Edo: {med['edo']}"
                )

        # 3. Cultural Stories
        if any(w in q_lower for w in ["osanobua", "creation", "benin", "edo", "mbe", "tortoise", "ekpe", "calabar", "omenala"]):
            for cult in self.lexicon.get("cultural_knowledge", []):
                context_items.append(f"Cultural Reference [{cult['topic']}]: {cult['details']}")

        return "\n\n".join(context_items)

    def build_grounded_system_prompt(self, user_query: str) -> str:
        context = self.retrieve_grounding_context(user_query)

        prompt_lines = [
            "You are ÀṢÀ (AfriWise) — an intelligent, culturally grounded AI assistant for Southern Nigerian languages (Igbo, Efik/Ibibio, Edo/Bini) and English.",
            "You MUST think through your response step-by-step before answering. Place all your internal reasoning inside <think> and </think> tags.",
            "",
            "## ABSTENTION RULE & CONTENT SAFETY:",
            "- If asked for an impossible modern technical term with no authentic native equivalent, use exactly: 'A maghị m okwu a na Igbo.' / 'Mmọdiọkke ikọ emi ke Efik.' / 'I ma-ẹre vbe Edo.'",
            "- CRITICAL: Do NOT output religious texts, Bible verses, or references to 'Jehovah' unless explicitly asked about religion. Keep all conversational and medical answers secular and strictly relevant to the prompt."
        ]

        if context:
            prompt_lines.append(f"\n## VERIFIED GROUNDING CONTEXT:\n{context}\n")

        return "\n".join(prompt_lines)

rag_engine = AfriWiseRAG()
