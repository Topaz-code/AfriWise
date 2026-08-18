"""
Dataset Builder Script for Africa AI Project (AfriWise).

Transforms extracted dictionary entries (Ibibio, Igbo, Edo) and cultural knowledge base
files (Bini/Edo, Igbo, Ibibio, Guardrails) into high-quality instruction-tuning datasets.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple

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

POS_FULL_NAMES = {
    "n.": "noun", "v.": "verb", "v.tr.": "transitive verb", "v.i.": "intransitive verb",
    "adj.": "adjective", "adv.": "adverb", "prep.": "preposition", "conj.": "conjunction",
    "pron.": "pronoun", "num.": "numeral", "p.n.": "proper noun", "n.p.": "noun phrase",
    "v.p.": "verb phrase", "aux. v.": "auxiliary verb", "interj.": "interjection",
    "dem.": "demonstrative", "poss.": "possessive", "quant.": "quantifier",
    "ideo.": "ideophone",
}

INVALID_REVERSE_STARTS = ("see ", "see also", "cf.", "syn.", "variant of", "abbr.", "short for", "e.g.")
INVALID_CHARS_RE = re.compile(r"[0-9\?\/\(\)\[\]_=\*\$\^\+~`<>\\\{\}\|]")


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def format_pos_label(pos: Optional[str]) -> Optional[str]:
    if not pos:
        return None
    pos_clean = pos.strip().lower()
    return POS_FULL_NAMES.get(pos_clean, pos_clean)


def is_clean_word(word: str) -> bool:
    if not word or len(word) < 2:
        return False
    if INVALID_CHARS_RE.search(word):
        return False
    letters_only = re.sub(r"[\s\-\'\.,;:!·]", "", word)
    return bool(letters_only)


def clean_definition_text(text: str) -> str:
    """Remove OCR debris and mangled phonetic fragments from definitions."""
    if not text:
        return ""
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", text).strip()
    
    # Filter out tokens that contain digits or irregular OCR noise
    tokens = cleaned.split()
    clean_tokens = []
    for token in tokens:
        # If token has digits or irregular symbols, skip it
        if re.search(r"\d", token):
            continue
        # If token looks like OCR gibberish (e.g. 'nthl', 'kAhl', 'edlOk')
        if len(token) > 2 and re.search(r"[a-z][A-Z]|[A-Z]{2,}[a-z]", token):
            continue
        clean_tokens.append(token)
    
    cleaned_def = " ".join(clean_tokens)
    # Remove hanging punctuation
    cleaned_def = re.sub(r"\s+[\,\;\:\.\-]\s*", ", ", cleaned_def)
    cleaned_def = re.sub(r",\s*,+", ",", cleaned_def)
    return cleaned_def.strip(" ,;:-")


def is_valid_dictionary_entry(entry: Dict[str, Any]) -> bool:
    headword = entry.get("headword", "").strip()
    definition = entry.get("definition", "").strip()
    clean_hw = re.sub(r"\s*\d+[\.\)]?\s*$", "", headword).strip()
    if not clean_hw or len(clean_hw) < 2 or len(clean_hw) > 35:
        return False
    if not is_clean_word(clean_hw):
        return False
    # Reject OCR artifacts with erratic casing inside the word (e.g., 'IkpdAtliki')
    if re.search(r"[a-z][A-Z]", clean_hw):
        return False
    clean_def = clean_definition_text(definition)
    if not clean_def or len(clean_def) < 5 or len(clean_def) > 300:
        return False
    # Reject short stub cross-references (e.g. 'see kpdi')
    if clean_def.lower().startswith(("see ", "see also ", "cf. ")) and len(clean_def.split()) < 4:
        return False
    # Reject definitions with embedded numbers inside words (e.g. '6k8p', '116', '6mim')
    if re.search(r"[a-zA-Z]\d|\d[a-zA-Z]", clean_def):
        return False
    alpha_chars = sum(c.isalpha() for c in clean_def)
    if alpha_chars / len(clean_def) < 0.70:
        return False
    return True


def is_valid_reverse_definition(definition: Optional[str]) -> bool:
    if not definition:
        return False
    defn_clean = clean_text(definition).strip().lower()
    if len(defn_clean) < 2 or len(defn_clean) > 220:
        return False
    for invalid_start in INVALID_REVERSE_STARTS:
        if defn_clean.startswith(invalid_start):
            return False
    return True


def clean_reverse_definition(definition: str) -> str:
    defn = clean_text(definition)
    defn = re.sub(r"^\d+[\.\)]\s*", "", defn)
    defn = re.split(r";\s*see\b|\.\s*see\b|;\s*cf\.|\.\s*cf\.", defn, flags=re.IGNORECASE)[0].strip()
    return defn


DIRECT_TRANSLATION_TEMPLATES = [
    "What does '{headword}' mean in {language}?",
    "Translate the {language} word '{headword}' into English.",
    "What is the meaning of the {language} word '{headword}'?",
    "Define the {language} term '{headword}'.",
]

def generate_direct_translation_examples(entry: Dict[str, Any], rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    if rng is None:
        rng = random.Random(42)
    language = entry.get("language", "African Language").strip()
    headword = clean_text(entry.get("headword"))
    definition = clean_definition_text(entry.get("definition", ""))
    pos = entry.get("pos")
    examples = entry.get("examples") or []
    valid_examples = [clean_text(e) for e in examples if clean_text(e) and not re.search(r"[a-zA-Z]\d|\d[a-zA-Z]", clean_text(e))]
    if not headword or not definition:
        return []
    pos_label = format_pos_label(pos)
    tmpl = rng.choice(DIRECT_TRANSLATION_TEMPLATES)
    instruction = tmpl.format(headword=headword, language=language)
    pos_str = f" ({pos_label})" if pos_label else ""
    ex_str = f"\n\n**Example Usage:**\n- " + "\n- ".join(valid_examples[:2]) if valid_examples else ""
    output = f"In {language}, **'{headword}'**{pos_str} means: {definition}{ex_str}"
    return [{
        "instruction": instruction,
        "input": "",
        "output": output,
        "category": "direct_translation",
        "language": language,
    }]


def generate_reverse_translation_examples(entry: Dict[str, Any], rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    if rng is None:
        rng = random.Random(42)
    language = entry.get("language", "African Language").strip()
    headword = clean_text(entry.get("headword"))
    raw_definition = entry.get("definition")
    if not headword or not is_valid_reverse_definition(raw_definition):
        return []
    clean_def = clean_reverse_definition(raw_definition)
    if not clean_def:
        return []
    return [{
        "instruction": f"How do you say '{clean_def}' in {language}?",
        "input": "",
        "output": f"In {language}, **'{clean_def}'** is expressed as **'{headword}'**.",
        "category": "reverse_translation",
        "language": language,
    }]


def generate_context_usage_examples(entry: Dict[str, Any], rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    if rng is None:
        rng = random.Random(42)
    examples = entry.get("examples") or []
    valid_examples = [clean_text(ex) for ex in examples if clean_text(ex)]
    if not valid_examples:
        return []
    language = entry.get("language", "African Language").strip()
    headword = clean_text(entry.get("headword"))
    definition = clean_text(entry.get("definition", ""))
    if not headword or not definition:
        return []
    ex_lines = "\n".join(f"- {ex}" for ex in valid_examples[:3])
    return [{
        "instruction": f"How is the {language} word '{headword}' used in a sentence?",
        "input": "",
        "output": f"In {language}, **'{headword}'** means: {definition}\n\n**Example Usage:**\n{ex_lines}",
        "category": "context_usage",
        "language": language,
    }]


PROVERB_THEME_MAPPING = {
    1: "hard work, diligence, and purposeful action",
    2: "experience as a teacher, vigilance, and learning from the past",
    3: "living in the present, realistic planning, and contentment",
    4: "respecting boundaries, stewardship, and honor",
    5: "unity and teamwork in community solidarity",
    6: "empathy, kindness, tact, and compassion",
    7: "listening to wise counsel, obedience, and heeding parental guidance",
    8: "excellence, thoroughness, integrity, and avoiding half-measures",
    9: "cause and effect, purposeful action, and understanding motives",
    10: "gratitude, awareness of hardship, and overcoming entitlement",
    11: "dignity, composure, patience, and inner strength",
    12: "valuing the youth, humility, and recognizing every person's worth",
    13: "the fleeting nature of life and valuing time",
    14: "stubbornness, natural consequences, and learning through reality",
    15: "cherishing family roots, ancestral heritage, and communal solidarity",
}

def extract_proverbs_from_markdown(content: str, language: str = "Edo (Bini)") -> List[Dict[str, Any]]:
    proverbs = []
    pattern = re.compile(
        r"(?m)^\s*(\d+)\.\s+\*\*(.+?)\*\*\s*\n\s*-\s*\*(?:Translation|Literal):\*\s*(.+?)\s*\n\s*-\s*\*(?:Cultural Meaning|Meaning):\*\s*(.+?)(?=\n\s*\d+\.|\n\s*##|\Z)",
        re.DOTALL,
    )
    for match in pattern.finditer(content):
        num_str, proverb_text, translation, cultural_meaning = match.groups()
        num = int(num_str)
        proverbs.append({
            "number": num,
            "proverb": clean_text(proverb_text),
            "translation": clean_text(translation),
            "cultural_meaning": clean_text(cultural_meaning),
            "theme": PROVERB_THEME_MAPPING.get(num, "ancestral wisdom and ethical living"),
            "language": language,
        })
    return proverbs


def generate_proverb_examples(proverbs: List[Dict[str, Any]], rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    if rng is None:
        rng = random.Random(42)
    results = []
    for item in proverbs:
        proverb = item["proverb"]
        trans = item["translation"]
        meaning = item["cultural_meaning"]
        theme = item["theme"]
        lang = item["language"]
        results.append({
            "instruction": f"Explain the {lang} proverb: '{proverb}'",
            "input": "",
            "output": f"**{lang} Proverb:** \"{proverb}\"\n- **Literal Translation:** {trans}\n- **Cultural & Philosophical Meaning:** {meaning}\n\nIn traditional {lang} society, elders invoke this proverb to teach {theme}.",
            "category": "proverbs_philosophy",
            "language": lang,
        })
        results.append({
            "instruction": f"What moral lesson does the {lang} saying '{proverb}' teach?",
            "input": "",
            "output": f"The core moral lesson behind **\"{proverb}\"** (*\"{trans}\"*) is: **{meaning}**. It emphasizes {theme}.",
            "category": "proverbs_philosophy",
            "language": lang,
        })
        results.append({
            "instruction": f"What is a traditional {lang} proverb about {theme}?",
            "input": "",
            "output": f"An authentic {lang} proverb concerning **{theme}** is:\n\n👉 **\"{proverb}\"**\n- **Translation:** {trans}\n- **Meaning:** {meaning}",
            "category": "proverbs_philosophy",
            "language": lang,
        })
        results.append({
            "instruction": f"In {lang} culture, how do elders express the idea that \"{meaning}\"?",
            "input": "",
            "output": f"In {lang} tradition, elders express this truth through the proverb:\n\n**\"{proverb}\"**\n*(Translation: {trans})*",
            "category": "proverbs_philosophy",
            "language": lang,
        })
    return results


COSMOLOGY_STORIES = [
    {
        "title": "Creation of the World & Origin of Igodomigodo",
        "prompts": [
            "Tell me the Edo / Bini story of the creation of the world and Osanobua.",
            "How was Igodomigodo and the dry land created according to Benin mythology?",
            "Explain the myth of Osanobua sending his children with the snail shell to create the earth.",
        ],
        "narrative": (
            "In ancient Edo cosmology, the universe was divided between Erinmwin (the spirit world) "
            "and Ágbon (the physical world), which was covered completely in water. Osanobua (God the Creator) "
            "sent his four children to live in the physical realm. The youngest child carried a magical snail shell (ikoko). "
            "Upon Osanobua's command, he poured out the sand from the shell onto the waters. The sand expanded endlessly, "
            "forming the first dry land, which became known as Igodomigodo (the ancient Benin kingdom). Because he created "
            "the dry ground upon which humanity would dwell, the youngest son was crowned the first sovereign ruler (Ogiso)."
        ),
    },
    {
        "title": "Osanobua and His Four Children",
        "prompts": [
            "Who are the children of Osanobua in Bini cosmology?",
            "What divine gifts did the children of Osanobua choose before descending to earth?",
            "Explain the relationship between Osanobua, Olokun, and the ruler of Igodomigodo.",
        ],
        "narrative": (
            "Before departing Erinmwin, Osanobua offered his four divine children their choice of gifts. "
            "The eldest son chose wealth and riches, becoming Olokun, the deity of the oceans, rivers, and boundless prosperity. "
            "The second son chose wisdom and magical power. The third chose sacred medicine and herbs. The youngest, guided by humility, "
            "chose the snail shell. When they reached the boundless waters, the youngest used the shell to create dry land, "
            "making all his siblings come to him for a place to stand. This establishes the Benin principle that humility and stewardship "
            "surpass raw material wealth."
        ),
    },
    {
        "title": "The Four Pillars of Living",
        "prompts": [
            "What are the four pillars of living in Bini philosophy?",
            "Explain the concept of the four pillars (Oto, Ame, Eziza, Erhen) in Edo worldview.",
            "How do the four primordial elements sustain human existence in Edo culture?",
        ],
        "narrative": (
            "In Edo cosmology, life in Ágbon is sustained by the four primordial pillars (elements):\n"
            "1. **Oto (Earth/Land):** The mother of sustenance, stability, agriculture, and physical foundation.\n"
            "2. **Ame (Water):** The fluid realm of Olokun, representing life, purity, cleansing, and prosperity.\n"
            "3. **Eziza (Air/Wind):** The breath of vitality, mystical speed, movement, and medicinal spirit.\n"
            "4. **Erhen (Fire):** The energy of transformation, blacksmithing, spiritual protection, and courage.\n\n"
            "Harmony across these four pillars ensures spiritual balance (omwan) and communal well-being."
        ),
    },
    {
        "title": "The Myth of Iso and the Origin of Farming",
        "prompts": [
            "Tell the story of Iso (the Sky) and why it moved away from the earth in Bini mythology.",
            "Why did the sky retreat from humanity in Edo traditional folklore?",
            "Explain how the myth of Iso gave rise to agriculture and hunting in Benin lore.",
        ],
        "narrative": (
            "In primordial times, Iso (the Sky) hung very low, so close to the earth that humans could reach up and carve "
            "pieces of the sky for food whenever they were hungry. Osanobua placed only one condition: take only what you need "
            "and never waste the food of the sky. One greedy man took far more than he could finish and threw the excess into the mud. "
            "Grievously offended by human wastefulness and ingratitude, Iso retreated high above out of reach. From that day forward, "
            "humans had to toil, farm the soil (Oto), and hunt to feed themselves."
        ),
    },
    {
        "title": "Ehi and the 14 Cycles of Soul Reincarnation",
        "prompts": [
            "Explain the concept of Ehi (the guardian spirit and destiny) in Bini belief.",
            "What is the doctrine of the 14 cycles of soul reincarnation in Edo religion?",
            "How does one's Ehi determine their destiny in Edo spiritual tradition?",
        ],
        "narrative": (
            "In Edo spiritual tradition, every human soul has an **Ehi** (spiritual double and guardian counterpart in Erinmwin). "
            "Before an individual is born into Ágbon, they kneel before Osanobua in the presence of their Ehi and declare their "
            "chosen life destiny (*uhi*). The Ehi remains in Erinmwin to guide and protect the earthly traveler. According to Edo "
            "eschatology, a human soul undergoes 14 cycles of incarnation between Ágbon and Erinmwin. In each cycle, the earthly "
            "self and the Ehi alternate roles, until all 14 cycles are completed and the perfected soul merges back into the divine presence."
        ),
    },
    {
        "title": "Mbe (Tortoise) and the Feast in the Sky (Igbo Folklore)",
        "prompts": [
            "Tell the Igbo folktale about Mbe (the tortoise) and the feast in the sky.",
            "Why does the tortoise have a cracked shell in Igbo folklore?",
            "What moral lesson does the story of Mbe Nwaniga teach about greed and deceit?",
        ],
        "narrative": (
            "In Igbo oral folklore, Mbe Nwaniga (the Tortoise) borrowed feathers from birds to attend a feast in the sky. "
            "He deceitfully took the name 'Unu Niile' (All of You) and ate all the food. In anger, the birds took back their feathers, "
            "leaving Mbe stranded. Mbe fell from the sky onto a pile of hard rocks, shattering his shell. The cracks remain to this day "
            "as an eternal reminder that greed and betrayal lead to destruction."
        ),
    },
    {
        "title": "Utin (Sun) and Ọfiọñ (Moon) in the Sky (Ibibio Legend)",
        "prompts": [
            "Tell the traditional Ibibio story of how the Sun and Moon came to live in the sky.",
            "Share an authentic Ibibio folktale about the Sun, Moon, and Water.",
            "Why do Utin and Ọfiọñ dwell in the sky according to Ibibio myth?",
        ],
        "narrative": (
            "In ancient Ibibio folklore, Utin (the Sun) and Ọfiọñ (the Moon) lived on earth and invited their friend Mmọñ (Water) "
            "to visit. Water brought all the marine creatures and overflowing currents, filling the compound up to the roof. "
            "Utin and Ọfiọñ had to leap into the sky (Enyọñ) to find room, where they decided to dwell permanently, "
            "illuminating the earth with light and beauty."
        ),
    },
]

PEDAGOGY_TOPICS = [
    {
        "topic": "Digraphs and Consonant Orthography",
        "prompts": [
            "What are the special digraphs in the Edo (Bini) language?",
            "Explain the pronunciation of Edo digraphs like GH, KH, GB, KP, VB, and MW.",
            "Why is it an error to ignore digraphs when writing Edo (Bini)?",
            "What common spelling mistakes occur with Bini consonants?",
        ],
        "content": (
            "Edo orthography utilizes 6 vital consonant digraphs that represent distinct phonemes:\n"
            "- **GH /ɣ/:** Voiced velar fricative (e.g., *oghẹghẹ* - joy, *ogha* - if).\n"
            "- **KH /x/:** Voiceless velar fricative (e.g., *ikhuo* - women, *okhian* - walk).\n"
            "- **GB /ɡ͡b/:** Voiced labial-velar plosive (e.g., *agbọn* - world, *egbe* - body).\n"
            "- **KP /k͡p/:** Voiceless labial-velar plosive (e.g., *kpere* - long-lived, *kpọlọ* - sweep).\n"
            "- **VB /ʋ/:** Labiodental approximant (e.g., *ovbiẹ* - sleep, *vb' egbe* - how are you).\n"
            "- **MW /ʋ̃/:** Nasalized labiodental approximant (e.g., *ẹmwan* - people, *omwan* - person).\n\n"
            "**Common Error:** Replacing 'vb' with 'v' or 'mw' with 'm' changes grammatical meaning completely."
        ),
    },
    {
        "topic": "Vowels and Phonetic Subdots",
        "prompts": [
            "Explain the 7-vowel system of the Edo language and how Ẹ and Ọ differ from E and O.",
            "What happens if you omit the subdot under Ẹ and Ọ in Edo writing?",
            "How do open and closed vowels operate in Bini language phonology?",
            "Why is vowel doubling used in Edo orthography?",
        ],
        "content": (
            "Edo uses a strict 7-vowel phonetic system:\n"
            "- **Standard Vowels:** a, e, i, o, u.\n"
            "- **Subdotted / Open Vowels:**\n"
            "  - **Ẹ (ẹ) /ɛ/:** Open 'e' as in English *bed* (e.g., *ẹmwẹn* - word/matter, distinct from *emwen*).\n"
            "  - **Ọ (ọ) /ɔ/:** Open 'o' as in English *law* (e.g., *ọkpa* - rooster/one, distinct from *okpa*).\n\n"
            "- **Vowel Doubling:** Indicates distinct vowel length and morphological emphasis (e.g., *kọyọọ*)."
        ),
    },
    {
        "topic": "Tone Marks and Downstep",
        "prompts": [
            "How does tone function in the Edo (Bini) language?",
            "Explain high tone, low tone, and downstep in Edo tonal grammar.",
            "Give examples where omitting tone marks changes the meaning of a Bini word.",
            "Why is Edo considered a tone language?",
        ],
        "content": (
            "Edo is a register-tone language where pitch contour is phonemic and grammatical:\n"
            "- **High Tone (Á / á):** Raised acoustic pitch (e.g., *íyé* - mother).\n"
            "- **Low Tone (À / à):** Lowered acoustic pitch (e.g., *ìyè* - number/count).\n"
            "- **Downstep (!):** A high tone lowered following an unpronounced low tone.\n\n"
            "**Example of Tone Contrast:**\n"
            "- *Ọ́ bọ́* (High-High): 'He built [it]'\n"
            "- *Ọ̀ bọ̀* (Low-Low): 'He is consulting an oracle'"
        ),
    },
    {
        "topic": "Oba of Benin Reverence & Royal Honorifics",
        "prompts": [
            "How should an assistant address the Oba of Benin with cultural reverence?",
            "What are the sacred titles of the Oba of Benin in traditional Edo language?",
            "Explain the phrase 'Oba gha t'o kpere, Ise!' and its cultural significance.",
            "What royalty guardrails exist when discussing the Benin monarchy?",
        ],
        "content": (
            "In Benin tradition, the Oba is the sacred custodian of the Edo civilization:\n"
            "- **Full Royal Title:** *Omo N'Oba N'Edo Uku Akpolokpolo* (The Sovereign of Benin).\n"
            "- **Salutation / Acclamation:** *Oba Gha T'o Kpere, Ise!* (May the King reign long, Amen!).\n"
            "- **Cultural Etiquette:** The Oba is never addressed casually by his birth name. He is referred to as *Ovbi' Umogun* (Son of the Leopard) or *Omo N'Oba*."
        ),
    },
    {
        "topic": "AfriWise Cultural Guardrails and Sensitivity",
        "prompts": [
            "What cultural guardrails must AfriWise follow when discussing African traditions?",
            "How does AfriWise handle sacred shrines, ancestral deities, and sacred titles?",
            "Why does AfriWise avoid colonial derogatory language when translating indigenous terms?",
            "What are the core ethical guidelines for the AfriWise AI assistant?",
        ],
        "content": (
            "AfriWise enforces strict cultural authenticity and respect guardrails:\n"
            "1. **Respect for Sacred Customs:** Ancestral deities (Olokun, Ogun, Chukwu, Abasi) and traditions are discussed with dignity and anthropological accuracy, avoiding colonialist slurs (such as 'juju' or 'fetish').\n"
            "2. **Linguistic Precision:** Preserves authentic orthography, diacritics, and tone marks.\n"
            "3. **Proverbial Integrity:** Always delivers the moral and philosophical context behind African proverbs."
        ),
    },
]

def generate_cosmology_examples(cosmology_items: Optional[List[Dict[str, Any]]] = None, rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    if rng is None:
        rng = random.Random(42)
    results = []
    stories = cosmology_items or COSMOLOGY_STORIES
    for item in stories:
        narrative = item.get("narrative", "")
        prompts = item.get("prompts", [])
        for p in prompts:
            results.append({
                "instruction": p,
                "input": "",
                "output": narrative,
                "category": "cosmology_storytelling",
                "language": "Edo (Bini)",
            })
    return results


def generate_pedagogy_examples(pedagogy_items: Optional[List[Dict[str, Any]]] = None, rng: Optional[random.Random] = None) -> List[Dict[str, str]]:
    if rng is None:
        rng = random.Random(42)
    results = []
    topics = pedagogy_items or PEDAGOGY_TOPICS
    for item in topics:
        content = item.get("content", "")
        prompts = item.get("prompts", [])
        for p in prompts:
            results.append({
                "instruction": p,
                "input": "",
                "output": content,
                "category": "language_pedagogy",
                "language": "Edo (Bini)",
            })
    return results


def convert_to_alpaca(record: Dict[str, str]) -> Dict[str, str]:
    return {
        "instruction": record.get("instruction", "").strip(),
        "input": record.get("input", "").strip(),
        "output": record.get("output", "").strip(),
    }

def convert_to_chatml(record: Dict[str, str], system_prompt: str = AFRIWISE_SYSTEM_PROMPT) -> Dict[str, Any]:
    instruction = record.get("instruction", "").strip()
    input_text = record.get("input", "").strip()
    user_content = f"{instruction}\n\nContext:\n{input_text}" if input_text else instruction
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": record.get("output", "").strip()},
        ]
    }

def validate_alpaca_record(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict) or "instruction" not in record or "output" not in record:
        return False
    return bool(record["instruction"].strip() and record["output"].strip())

def validate_chatml_record(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict) or "messages" not in record:
        return False
    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) < 3:
        return False
    if messages[0].get("content") != AFRIWISE_SYSTEM_PROMPT:
        return False
    if not messages[1].get("content", "").strip():
        return False
    if not messages[2].get("content", "").strip():
        return False
    return True

CURATED_CONVERSATIONS = [
    {
        "language": "All Languages (Ibibio, Igbo, Edo)",
        "prompts": [
            "Translate 'good morning, how is the family?' into Ibibio and Igbo.",
            "How do you say 'good morning, how is the family?' in Ibibio, Igbo, and Edo?",
            "What are the morning greetings and family inquiries in Ibibio and Igbo?",
        ],
        "response": (
            "Here are the authentic translations across southern Nigerian languages:\n\n"
            "1. **Ibibio:**\n"
            "   - **Greeting:** *\"Emesiere!\"* (Good morning!)\n"
            "   - **Family Inquiry:** *\"Mme ubon mfo etie didie?\"* or *\"Idem ubon mfo?\"* (How is your family?)\n"
            "   - **Full Expression:** *\"Emesiere, mme ubon mfo etie didie?\"*\n\n"
            "2. **Igbo:**\n"
            "   - **Greeting:** *\"Ụtụtụ ọma!\"* or *\"I bọọla chi?\"* (Good morning! / Have you awakened to the dawn?)\n"
            "   - **Family Inquiry:** *\"Kedụ maka ndị ezinụlọ gị?\"* or *\"Ndị be gị kwanụ?\"* (How is your household/family?)\n"
            "   - **Full Expression:** *\"Ụtụtụ ọma, kedụ maka ndị ezinụlọ gị?\"*\n\n"
            "3. **Edo (Bini):**\n"
            "   - **Greeting:** *\"Kọyọ rre!\"* or *\"Ob'owie!\"* (Good morning!)\n"
            "   - **Family Inquiry:** *\"Vb' evbo rre owa?\"* or *\"Egbe emwan rre owa?\"* (How are the people at home?)\n"
            "   - **Full Expression:** *\"Kọyọ, vb' evbo rre owa?\"*"
        )
    },
    {
        "language": "Ibibio",
        "prompts": [
            "How do you greet someone in the morning and ask how they are in Ibibio?",
            "What is the standard morning greeting and response in Ibibio?",
            "Teach me common everyday greetings in Ibibio.",
        ],
        "response": (
            "In Ibibio culture, greetings are polite and foundational to daily interaction:\n\n"
            "- **Good morning:** *\"Emesiere\"* (Response: *\"Emesiere nda\"* or *\"Emesiere o\"*)\n"
            "- **To an elder man:** *\"Emesiere, Ete\"* (Good morning, Father)\n"
            "- **To an elder woman:** *\"Emesiere, Eka\"* (Good morning, Mother)\n"
            "- **How are you?:** *\"Idem mfo?\"* or *\"Abadie?\"*\n"
            "- **I am fine / I am in good health:** *\"Idem mi ọsọn\"* or *\"Aba ke emem\"*\n"
            "- **Thank you:** *\"Sosongo\"* or *\"Sosongo eti-eti\"* (Thank you very much)\n"
            "- **Welcome:** *\"Emedi\"* (to one person) / *\"Emedio\"* (to a group)\n"
            "- **Go well / Safe journey:** *\"Sanga sun\"*\n"
            "- **Stay well:** *\"Tie sun\"*"
        )
    },
    {
        "language": "Igbo",
        "prompts": [
            "How do you greet people in Igbo for different times of the day?",
            "What are common daily greetings and responses in Igbo?",
            "Teach me how to say hello, thank you, and goodbye in Igbo.",
        ],
        "response": (
            "In Igbo tradition (*Omenala Igbo*), respectful greetings establish warm communal rapport:\n\n"
            "- **General Greeting / Hello:** *\"Ndewo\"* or *\"Nnọọ\"* (Welcome)\n"
            "- **Good morning:** *\"Ụtụtụ ọma\"* or *\"I bọọla chi?\"* (Response: *\"Eeh, a bọọla m chi\"*)\n"
            "- **Good afternoon:** *\"Ehihie ọma\"*\n"
            "- **Good evening:** *\"Anyasị ọma\"*\n"
            "- **Good night:** *\"Ka chi bọọ\"* (May the dawn break peacefully)\n"
            "- **How are you?:** *\"Kedụ?\"* or *\"Kedụ ka ị mere?\"*\n"
            "- **I am fine:** *\"Ọ dị mma\"* or *\"A nọ m nke ọma\"*\n"
            "- **Thank you:** *\"Daalụ\"* or *\"Imela\"* / *\"Imeela nke ukwuu\"*\n"
            "- **Safe journey / Goodbye:** *\"Ije ọma\"* or *\"Ka ọ dị\"*"
        )
    },
    {
        "language": "Edo (Bini)",
        "prompts": [
            "How do you greet people in Edo (Bini) throughout the day?",
            "What are the traditional greetings in Benin culture?",
            "How do you say hello, welcome, and thank you in Edo (Bini)?",
        ],
        "response": (
            "In Benin culture, greetings are deeply tied to respect, hierarchy, and family heritage:\n\n"
            "- **General Greeting:** *\"Kọyọ\"* (Hello / Greetings)\n"
            "- **Welcome:** *\"Ob'okhian\"*\n"
            "- **Greeting someone at home / work:** *\"Ob'owa\"* (Welcome home) / *\"Ob'win\"* (Well done at work)\n"
            "- **Morning greeting to elders:** *\"Kọyọ rre\"* or family morning praise salutations (*Laven*, *Lágba*, *Delau*)\n"
            "- **Afternoon greeting:** *\"Ob'avan\"*\n"
            "- **Evening greeting:** *\"Ob'ota\"*\n"
            "- **How are you?:** *\"Vb' egbe?\"* (How is the body?)\n"
            "- **I am fine:** *\"Egbe rre ọma\"* or *\"Ọ gha rrọọ\"*\n"
            "- **Thank you:** *\"Urhese\"* (Thank you very much)\n"
            "- **Safe journey / Goodbye:** *\"Gha khian n'ese\"* (Go well) / *\"O khian kherhe\"*"
        )
    },
]

def save_jsonl(records: List[Dict[str, Any]], file_path: Path) -> int:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def build_all_datasets(
    dict_path: Path,
    kb_dir: Path,
    seed: int = 42,
    max_dict_entries: int = 4000
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    raw_examples: List[Dict[str, str]] = []
    stats: Dict[str, Any] = {
        "raw_dictionary_entries": 0,
        "filtered_dictionary_entries": 0,
        "direct_translation_count": 0,
        "reverse_translation_count": 0,
        "context_usage_count": 0,
        "proverbs_qa_count": 0,
        "cosmology_qa_count": 0,
        "pedagogy_qa_count": 0,
        "total_alpaca_examples": 0,
        "total_chatml_examples": 0,
        "language_distribution": {},
        "category_distribution": {},
    }

    if dict_path.exists():
        with open(dict_path, "r", encoding="utf-8") as f:
            raw_entries = json.load(f)
        stats["raw_dictionary_entries"] = len(raw_entries)
        valid_entries = [e for e in raw_entries if is_valid_dictionary_entry(e)]
        stats["filtered_dictionary_entries"] = len(valid_entries)

        for entry in valid_entries[:max_dict_entries]:
            direct = generate_direct_translation_examples(entry, rng=rng)
            raw_examples.extend(direct)
            stats["direct_translation_count"] += len(direct)

    bini_path = kb_dir / "bini_edo_cultural_corpus.md"
    if bini_path.exists():
        proverbs = extract_proverbs_from_markdown(bini_path.read_text(encoding="utf-8"), language="Edo (Bini)")
        p_pairs = generate_proverb_examples(proverbs, rng=rng)
        raw_examples.extend(p_pairs)
        stats["proverbs_qa_count"] += len(p_pairs)

    igbo_path = kb_dir / "igbo_cultural_corpus.md"
    if igbo_path.exists():
        igbo_p = extract_proverbs_from_markdown(igbo_path.read_text(encoding="utf-8"), language="Igbo")
        igbo_pairs = generate_proverb_examples(igbo_p, rng=rng)
        raw_examples.extend(igbo_pairs)
        stats["proverbs_qa_count"] += len(igbo_pairs)

    # Add Cosmology Stories
    cosmology_pairs = generate_cosmology_examples(COSMOLOGY_STORIES, rng=rng)
    raw_examples.extend(cosmology_pairs)
    stats["cosmology_qa_count"] = len(cosmology_pairs)

    # Add Pedagogy Topics
    pedagogy_pairs = generate_pedagogy_examples(PEDAGOGY_TOPICS, rng=rng)
    raw_examples.extend(pedagogy_pairs)
    stats["pedagogy_qa_count"] = len(pedagogy_pairs)

    # Add Curated Conversations
    for item in CURATED_CONVERSATIONS:
        lang = item.get("language", "General")
        resp = item["response"]
        for prompt in item["prompts"]:
            for _ in range(25):
                raw_examples.append({
                    "instruction": prompt,
                    "input": "",
                    "output": resp,
                    "category": "cultural_conversation",
                    "language": lang,
                })

    rng.shuffle(raw_examples)

    alpaca_records = [convert_to_alpaca(e) for e in raw_examples if validate_alpaca_record(convert_to_alpaca(e))]
    chatml_records = [convert_to_chatml(e) for e in raw_examples if validate_chatml_record(convert_to_chatml(e))]

    for ex in raw_examples:
        l = ex.get("language", "General")
        c = ex.get("category", "general")
        stats["language_distribution"][l] = stats["language_distribution"].get(l, 0) + 1
        stats["category_distribution"][c] = stats["category_distribution"].get(c, 0) + 1

    stats["total_alpaca_examples"] = len(alpaca_records)
    stats["total_chatml_examples"] = len(chatml_records)
    return alpaca_records, chatml_records, stats


def main():
    parser = argparse.ArgumentParser(description="Build clean dataset for AfriWise.")
    parser.add_argument("--dict-path", type=Path, default=Path("data/raw/dictionary_extracted.json"))
    parser.add_argument("--kb-dir", type=Path, default=Path("knowledge_base"))
    parser.add_argument("--output-alpaca", type=Path, default=Path("data/processed/training_dataset.jsonl"))
    parser.add_argument("--output-chatml", type=Path, default=Path("data/processed/training_dataset_chatml.jsonl"))
    args = parser.parse_args()

    alpaca, chatml, stats = build_all_datasets(dict_path=args.dict_path, kb_dir=args.kb_dir)
    save_jsonl(alpaca, args.output_alpaca)
    save_jsonl(chatml, args.output_chatml)
    logger.info("=== Dataset Build Complete ===")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
