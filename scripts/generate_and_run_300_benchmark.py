"""
scripts/generate_and_run_300_benchmark.py — 300-Question Multi-Lingual Grounding & Anti-Hallucination Benchmark.

Validates 100 distinct questions for Igbo, 100 for Efik/Ibibio, and 100 for Bini/Edo (300 total)
spanning greetings, kinship, cosmology, grammar, daily vocabulary, agriculture, proverbs, and complex translation queries.
"""

import io
import json
import logging
import re
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag_engine import get_knowledge_engine

BENCHMARK_OUTPUT = PROJECT_ROOT / "data" / "processed" / "benchmark_300_results.json"


def build_300_questions():
    igbo_questions = [
        ("Translate to Igbo: Good morning", "ụtụtụ ọma", "greeting"),
        ("Translate to Igbo: Good afternoon", "ehihie ọma", "greeting"),
        ("Translate to Igbo: Good evening", "anyasị ọma", "greeting"),
        ("Translate to Igbo: Good night", "ka chi foo", "greeting"),
        ("Translate to Igbo: How are you?", "kedụ", "greeting"),
        ("Translate to Igbo: Thank you", "daalụ", "politeness"),
        ("Translate to Igbo: Welcome", "nnọọ", "greeting"),
        ("Translate to Igbo: Goodbye / Until later", "ka ọ dị", "greeting"),
        ("Translate to Igbo: My name is Topaz", "aha m bụ topaz", "intro"),
        ("Translate to Igbo: What is your name?", "kedụ aha gị", "intro"),
        ("Translate to Igbo: I am happy", "obi dị m ụtọ", "emotion"),
        ("Translate to Igbo: I love you", "ahụrụ m gị n'anya", "emotion"),
        ("Translate to Igbo: God bless you", "chukwu gọzie gị", "blessing"),
        ("Translate to Igbo: Please help me", "biko nyere m aka", "politeness"),
        ("Translate to Igbo: Where are you going?", "kedụ ebe ị na-aga", "question"),
        ("Translate to Igbo: I am going to the market", "ana m aga ahịa", "daily"),
        ("Translate to Igbo: How much is this?", "ego ole", "commerce"),
        ("Translate to Igbo: I want to buy food", "achọrọ m ịzụ nri", "commerce"),
        ("Translate to Igbo: Water", "mmiri", "vocab"),
        ("Translate to Igbo: House / Home", "ụlọ", "vocab"),
        ("Translate to Igbo: Child", "nwa", "kinship"),
        ("Translate to Igbo: Children", "ụmụaka", "kinship"),
        ("Translate to Igbo: Father", "nna", "kinship"),
        ("Translate to Igbo: Mother", "nne", "kinship"),
        ("Translate to Igbo: Brother / Sibling", "nwanne", "kinship"),
        ("Translate to Igbo: Family", "ezinụlọ", "kinship"),
        ("Translate to Igbo: King", "eze", "royalty"),
        ("Translate to Igbo: Queen", "lọọlọ", "royalty"),
        ("Translate to Igbo: Princess", "adaeze", "royalty"),
        ("Translate to Igbo: Doctor", "dọkịta", "profession"),
        ("Translate to Igbo: Teacher", "onye nkuzi", "profession"),
        ("Translate to Igbo: Student", "nwa akwụkwọ", "profession"),
        ("Translate to Igbo: Chicken / Fowl", "ọkụkọ", "animal"),
        ("Translate to Igbo: Goat", "ewu", "animal"),
        ("Translate to Igbo: Dog", "nkịta", "animal"),
        ("Translate to Igbo: Fish", "azụ", "animal"),
        ("Translate to Igbo: Lion", "ọdụm", "animal"),
        ("Translate to Igbo: Leopard", "agụ", "animal"),
        ("Translate to Igbo: Elephant", "enyi", "animal"),
        ("Translate to Igbo: Tortoise", "mbe", "animal"),
        ("Translate to Igbo: Bird", "nnụnụ", "animal"),
        ("Translate to Igbo: Snake", "agwọ", "animal"),
        ("Translate to Igbo: Yam", "ji", "food"),
        ("Translate to Igbo: Cassava", "akpụ", "food"),
        ("Translate to Igbo: Palm oil", "mmanụ", "food"),
        ("Translate to Igbo: Soup", "ofe", "food"),
        ("Translate to Igbo: Rice", "osikapa", "food"),
        ("Translate to Igbo: Bread", "achịcha", "food"),
        ("Translate to Igbo: Breakfast", "nri ụtụtụ", "food"),
        ("Translate to Igbo: Lunch", "nri ehihie", "food"),
        ("Translate to Igbo: Dinner", "nri anyasị", "food"),
        ("Translate to Igbo: Today", "taa", "time"),
        ("Translate to Igbo: Tomorrow", "echi", "time"),
        ("Translate to Igbo: Yesterday", "ụnyaahụ", "time"),
        ("Translate to Igbo: Morning", "ụtụtụ", "time"),
        ("Translate to Igbo: Night", "abalị", "time"),
        ("Translate to Igbo: Sun", "anwụ", "nature"),
        ("Translate to Igbo: Moon", "ọnwa", "nature"),
        ("Translate to Igbo: Rain", "mmiri ozuzo", "nature"),
        ("Translate to Igbo: Road / Way", "ụzọ", "travel"),
        ("Translate to Igbo: Car", "ụgbọ ala", "travel"),
        ("Translate to Igbo: Airplane", "ụgbọ elu", "travel"),
        ("Translate to Igbo: Book", "akwụkwọ", "object"),
        ("Translate to Igbo: Money", "ego", "commerce"),
        ("Translate to Igbo: Peace", "udo", "virtue"),
        ("Translate to Igbo: Truth", "eziokwu", "virtue"),
        ("Translate to Igbo: Life", "ndụ", "virtue"),
        ("Translate to Igbo: Health", "ahụ ike", "health"),
        ("Translate to Igbo: Medicine", "ọgwụ", "health"),
        ("Translate to Igbo: Hospital", "ụlọ ọgwụ", "health"),
        ("Translate to Igbo: School", "ụlọ akwụkwọ", "education"),
        ("Translate to Igbo: Church", "ụlọ ụka", "community"),
        ("Translate to Igbo: Village / Town", "obodo", "community"),
        ("Translate to Igbo: Land / Earth", "ala", "nature"),
        ("Translate to Igbo: Sky / Heaven", "eluigwe", "cosmology"),
        ("Translate to Igbo: God", "chukwu", "cosmology"),
        ("Translate to Igbo: I do not know", "a maghị m", "refusal"),
        ("Translate to Igbo: Speak Igbo", "kwuo igbo", "imperative"),
        ("Translate to Igbo: Listen to me", "gere m ntị", "imperative"),
        ("Translate to Igbo: Sit down", "nọdụ ala", "imperative"),
        ("Translate to Igbo: Stand up", "guzo ọtọ", "imperative"),
        ("Translate to Igbo: Come here", "bịa ebe a", "imperative"),
        ("Translate to Igbo: Go home", "laa ụlọ", "imperative"),
        ("Translate to Igbo: Eat food", "rie nri", "imperative"),
        ("Translate to Igbo: Drink water", "ñụọ mmiri", "imperative"),
        ("Translate to Igbo: Sleep well", "hie ụra nke ọma", "imperative"),
        ("Translate to Igbo: Wake up", "kpọtee", "imperative"),
        ("Explain the Igbo proverb: Ilu bụ mmanụ e ji eri okwu", "mmanụ", "proverb"),
        ("Explain the Igbo proverb: Igwe bụ ike", "strength", "proverb"),
        ("Explain the Igbo proverb: Onye mee ọfọ, ọfọ ana-edu ya", "ọfọ", "proverb"),
        ("Explain the Igbo proverb: Gidi gidi bụ ugwu eze", "king", "proverb"),
        ("Explain the Igbo proverb: Egbe belu ugo belu", "live and let live", "proverb"),
        ("What is Omenala Igbo?", "omenala", "culture"),
        ("Who is Chi in Igbo cosmology?", "chi", "cosmology"),
        ("What is the significance of the kola nut (Ọjị) in Igbo culture?", "ọjị", "custom"),
        ("Tell the story of Mbe the tortoise and the sky feast.", "mbe", "folktale"),
        ("What are the standard sub-dotted vowels in Onwu 1961 Igbo orthography?", "ị", "orthography"),
        ("What are the nine consonant digraphs in Igbo?", "gb", "orthography"),
        ("Translate to Igbo: Hi Topaz, I wanted to ask about how doctors treat small animals like chickens, and what you would love to have for breakfast tomorrow.", "ndewo topaz", "complex_sentence"),
        ("Translate to Igbo: The sun is shining brightly today and we are going to farm yams.", "anwụ", "complex_sentence"),
        ("Translate to Igbo: Respect your elders so that you will live a long and peaceful life.", "ndị okenye", "complex_sentence"),
        ("Translate to Igbo: My mother prepared delicious soup with fresh fish from the river.", "nne m", "complex_sentence"),
    ]

    efik_questions = [
        ("Translate to Efik: Good morning", "amesiere", "greeting"),
        ("Translate to Efik: Good night", "de sung", "greeting"),
        ("Translate to Efik: How are you?", "idem mfo", "greeting"),
        ("Translate to Efik: I am fine", "idem mi ọsọñ", "greeting"),
        ("Translate to Efik: Peace / Hello", "emem", "greeting"),
        ("Translate to Efik: Welcome", "amedi", "greeting"),
        ("Translate to Efik: Thank you", "sọsọñọ", "politeness"),
        ("Translate to Efik: Goodbye / Stay well", "tie sun", "greeting"),
        ("Translate to Efik: My name is Topaz", "enyíñ mmi", "intro"),
        ("Translate to Efik: What is your name?", "enyíñ mfo", "intro"),
        ("Translate to Efik: God bless you", "abasi ọfọfọn", "blessing"),
        ("Translate to Efik: I love you", "mmama fi", "emotion"),
        ("Translate to Efik: Water", "mmọñ", "vocab"),
        ("Translate to Efik: House / Home", "ufọk", "vocab"),
        ("Translate to Efik: Child", "eyen", "kinship"),
        ("Translate to Efik: Children", "nditọ", "kinship"),
        ("Translate to Efik: Father", "ete", "kinship"),
        ("Translate to Efik: Mother", "eka", "kinship"),
        ("Translate to Efik: Brother / Sibling", "eyenenyịn", "kinship"),
        ("Translate to Efik: King / Paramount Ruler", "obong", "royalty"),
        ("Translate to Efik: Queen", "mbre", "royalty"),
        ("Translate to Efik: Market", "urua", "commerce"),
        ("Translate to Efik: Money", "okuk", "commerce"),
        ("Translate to Efik: Food", "udia", "food"),
        ("Translate to Efik: Soup", "efere", "food"),
        ("Translate to Efik: Fish", "iyak", "food"),
        ("Translate to Efik: Meat", "unam", "food"),
        ("Translate to Efik: Chicken / Fowl", "unen", "animal"),
        ("Translate to Efik: Goat", "ebot", "animal"),
        ("Translate to Efik: Dog", "ebua", "animal"),
        ("Translate to Efik: Leopard", "ekpe", "animal"),
        ("Translate to Efik: Elephant", "enini", "animal"),
        ("Translate to Efik: Tortoise", "ikpọk", "animal"),
        ("Translate to Efik: Bird", "inuen", "animal"),
        ("Translate to Efik: Snake", "urukikot", "animal"),
        ("Translate to Efik: Yam", "bia", "food"),
        ("Translate to Efik: Cassava", "iwa", "food"),
        ("Translate to Efik: Palm oil", "adan", "food"),
        ("Translate to Efik: Vegetable soup (Edikang Ikong)", "edikang ikong", "food"),
        ("Translate to Efik: Afang soup", "afang", "food"),
        ("Translate to Efik: Today", "mfin", "time"),
        ("Translate to Efik: Tomorrow", "mkpọñ", "time"),
        ("Translate to Efik: Yesterday", "mkpọñ", "time"),
        ("Translate to Efik: Morning", "usenubọk", "time"),
        ("Translate to Efik: Night", "okoneyo", "time"),
        ("Translate to Efik: Sun", "utin", "nature"),
        ("Translate to Efik: Moon", "ọfiọñ", "nature"),
        ("Translate to Efik: Rain", "edim", "nature"),
        ("Translate to Efik: Road / Path", "usung", "travel"),
        ("Translate to Efik: River", "inwang", "nature"),
        ("Translate to Efik: Forest / Bush", "ikọt", "nature"),
        ("Translate to Efik: Boat / Canoe", "ubom", "travel"),
        ("Translate to Efik: Book", "ñwed", "object"),
        ("Translate to Efik: School", "ufọk ñwed", "education"),
        ("Translate to Efik: Church", "ufọk abasi", "community"),
        ("Translate to Efik: Village / Country", "idung", "community"),
        ("Translate to Efik: Earth / Ground", "isọñ", "nature"),
        ("Translate to Efik: God Almighty", "abasi ibom", "cosmology"),
        ("Translate to Efik: I do not know", "mmọdiọkke", "refusal"),
        ("Translate to Efik: Come here", "di mi", "imperative"),
        ("Translate to Efik: Go there", "ka do", "imperative"),
        ("Translate to Efik: Sit down", "tie ke isọñ", "imperative"),
        ("Translate to Efik: Stand up", "daha da", "imperative"),
        ("Translate to Efik: Eat food", "dia udia", "imperative"),
        ("Translate to Efik: Drink water", "ñwọñ mmọñ", "imperative"),
        ("Translate to Efik: Sleep peacefully", "de sung", "imperative"),
        ("Translate to Efik: Wake up", "demede", "imperative"),
        ("Translate to Efik: Speak Efik", "seme efik", "imperative"),
        ("Translate to Efik: Listen to me", "kpan utọñ", "imperative"),
        ("Explain the Efik proverb: Ete idung iyakke enen", "injustice", "proverb"),
        ("Explain the Efik proverb: Owo itoho ke enyöñ ediduọ", "ancestry", "proverb"),
        ("Explain the Efik proverb: Ekpe esio mkpo, ikọt ekop", "ekpe", "proverb"),
        ("Explain the Efik proverb: Isọñ oro emem buep", "emem", "proverb"),
        ("What is the Ekpe society in Efik history?", "ekpe", "culture"),
        ("Who is the Obong of Calabar?", "obong", "royalty"),
        ("What is the meaning of Emem in Efik culture?", "peace", "culture"),
        ("What does Idem mfo literally translate to in English?", "body", "linguistics"),
        ("Explain the Ibibio folktale of why the Sun and Moon live in the sky.", "utin", "folktale"),
        ("What are the key standard orthography vowels in Essien 1983 Efik?", "ñ", "orthography"),
        ("How does Ibibio pluralize nouns according to Josiah 2020?", "plural", "morphology"),
        ("What is the difference between Efik and Ibibio dialects?", "cross river", "linguistics"),
        ("Translate to Efik: Where are you going today?", "mfin", "question"),
        ("Translate to Efik: How much does this fish cost in the market?", "urua", "market"),
        ("Translate to Efik: We are cooking delicious Afang soup for the festival.", "afang", "complex_sentence"),
        ("Translate to Efik: The elders gathered at the palace to discuss peace.", "emem", "complex_sentence"),
        ("Translate to Efik: The little children are singing happily at school.", "nditọ", "complex_sentence"),
        ("Translate to Efik: My father went to the farm early in the morning.", "ete mmi", "complex_sentence"),
        ("Translate to Efik: May God protect your journey.", "abasi", "blessing"),
        ("Translate to Efik: Respect your parents so that blessings follow you.", "eka", "complex_sentence"),
        ("Translate to Efik: Calabar is a beautiful historic city by the river.", "calabar", "culture"),
        ("Translate to Efik: The fisherman caught large fish in the river yesterday.", "iyak", "complex_sentence"),
        ("Translate to Efik: Peace is better than war and strife.", "emem", "proverb"),
        ("Translate to Efik: Honest words bring honor to the family.", "ufọk", "proverb"),
        ("Translate to Efik: I woke up early to sweep the compound.", "usenubọk", "daily"),
        ("Translate to Efik: The rain cooled the hot afternoon sun.", "edim", "nature"),
        ("Translate to Efik: We welcome all visitors with peace and joy.", "amedi", "greeting"),
        ("Translate to Efik: Let us preserve our ancestral language for our children.", "nditọ", "preservation"),
        ("Translate to Efik: A wise person listens before speaking.", "utọñ", "proverb"),
        ("Translate to Efik: Unity brings prosperity to the whole village.", "idung", "proverb"),
        ("Translate to Efik: What would you like to eat for dinner tonight?", "udia", "complex_sentence"),
    ]

    edo_questions = [
        ("Translate to Bini/Edo: Hello / Greetings", "kọyọ", "greeting"),
        ("Translate to Bini/Edo: Good morning", "ọbowiẹ", "greeting"),
        ("Translate to Bini/Edo: How are you?", "vbèè óye hé", "greeting"),
        ("Translate to Bini/Edo: It is fine / I am fine", "ọ y'ese", "greeting"),
        ("Translate to Bini/Edo: Good night / Until tomorrow", "òkhíen òwie", "greeting"),
        ("Translate to Bini/Edo: Thank you", "uruese", "politeness"),
        ("Translate to Bini/Edo: Welcome", "ekabo", "greeting"),
        ("Translate to Bini/Edo: Journey well / Goodbye", "gha khian n'ese", "greeting"),
        ("Translate to Bini/Edo: My name is Topaz", "enǐ mwẹnrẹn", "intro"),
        ("Translate to Bini/Edo: What is your name?", "enǐ", "intro"),
        ("Translate to Bini/Edo: God bless you", "osanobua", "blessing"),
        ("Translate to Bini/Edo: I love you", "i hoo", "emotion"),
        ("Translate to Bini/Edo: Water", "amẹ", "vocab"),
        ("Translate to Bini/Edo: House / Home", "owa", "vocab"),
        ("Translate to Bini/Edo: Child", "ọmọ", "kinship"),
        ("Translate to Bini/Edo: Children", "ibieka", "kinship"),
        ("Translate to Bini/Edo: Father", "erha", "kinship"),
        ("Translate to Bini/Edo: Mother", "iye", "kinship"),
        ("Translate to Bini/Edo: Brother / Sibling", "ọvbierha", "kinship"),
        ("Translate to Bini/Edo: King of Benin", "oba", "royalty"),
        ("Translate to Bini/Edo: Queen Mother of Benin", "iyoba", "royalty"),
        ("Translate to Bini/Edo: Duke / Community Ruler", "enogie", "royalty"),
        ("Translate to Bini/Edo: Chief / Noble", "ogiegor", "royalty"),
        ("Translate to Bini/Edo: Market", "eki", "commerce"),
        ("Translate to Bini/Edo: Money", "igho", "commerce"),
        ("Translate to Bini/Edo: Food", "evbare", "food"),
        ("Translate to Bini/Edo: Soup", "omwan", "food"),
        ("Translate to Bini/Edo: Yam", "emwin", "food"),
        ("Translate to Bini/Edo: Fish", "ẹhẹn", "food"),
        ("Translate to Bini/Edo: Meat", "ẹnamwẹn", "food"),
        ("Translate to Bini/Edo: Chicken / Fowl", "ọkhọkhọ", "animal"),
        ("Translate to Bini/Edo: Goat", "ewe", "animal"),
        ("Translate to Bini/Edo: Dog", "ẹkwo", "animal"),
        ("Translate to Bini/Edo: Leopard", "ẹkpẹn", "animal"),
        ("Translate to Bini/Edo: Elephant", "eneni", "animal"),
        ("Translate to Bini/Edo: Tortoise", "egi", "animal"),
        ("Translate to Bini/Edo: Bird", "ahiamwen", "animal"),
        ("Translate to Bini/Edo: Snake", "inyẹn", "animal"),
        ("Translate to Bini/Edo: Snail shell", "ikoko", "cosmology"),
        ("Translate to Bini/Edo: Today", "ẹrẹ", "time"),
        ("Translate to Bini/Edo: Tomorrow", "owie", "time"),
        ("Translate to Bini/Edo: Yesterday", "iyẹrẹ", "time"),
        ("Translate to Bini/Edo: Morning", "owie", "time"),
        ("Translate to Bini/Edo: Night", "asọn", "time"),
        ("Translate to Bini/Edo: Sun", "ọwẹn", "nature"),
        ("Translate to Bini/Edo: Moon", "uki", "nature"),
        ("Translate to Bini/Edo: Rain", "amẹ", "nature"),
        ("Translate to Bini/Edo: Road / Path", "ode", "travel"),
        ("Translate to Bini/Edo: River", "ezẹ", "nature"),
        ("Translate to Bini/Edo: Forest / Bush", "oha", "nature"),
        ("Translate to Bini/Edo: Palace", "eguae", "royalty"),
        ("Translate to Bini/Edo: Town / City", "ẹvbo", "community"),
        ("Translate to Bini/Edo: Earth / Land", "oto", "nature"),
        ("Translate to Bini/Edo: Physical World", "agbon", "cosmology"),
        ("Translate to Bini/Edo: Spiritual Realm", "erinmwin", "cosmology"),
        ("Translate to Bini/Edo: God the Creator", "osanobua", "cosmology"),
        ("Translate to Bini/Edo: Guardian Angel / Destiny", "ehi", "cosmology"),
        ("Translate to Bini/Edo: Soul / Spirit", "orhion", "cosmology"),
        ("Translate to Bini/Edo: I do not know", "i ma-ẹre", "refusal"),
        ("Translate to Bini/Edo: Come here", "re vbena", "imperative"),
        ("Translate to Bini/Edo: Go away / Go home", "khian", "imperative"),
        ("Translate to Bini/Edo: Sit down", "totọ", "imperative"),
        ("Translate to Bini/Edo: Stand up", "mudia", "imperative"),
        ("Translate to Bini/Edo: Eat food", "re evbare", "imperative"),
        ("Translate to Bini/Edo: Drink water", "wọn amẹ", "imperative"),
        ("Translate to Bini/Edo: Speak Edo", "guamwẹn edo", "imperative"),
        ("Translate to Bini/Edo: Listen carefully", "danmweh", "imperative"),
        ("What are the six consonant digraphs in Agheyisi 1986 Edo orthography?", "gh", "orthography"),
        ("Explain the Bini proverb: Obo oguo o vha guese ache", "pot", "proverb"),
        ("Explain the Bini proverb: Oba ya oto s'evbo 'ebo", "oba", "proverb"),
        ("Explain the Bini proverb: A ya egbe we ewe, ewe gha gb'orere", "arrogance", "proverb"),
        ("Explain the Bini proverb: Erhimwin ma gbe, agbon i gu'oran", "spiritual", "proverb"),
        ("What is the Bini creation story of Osanobua and the snail shell?", "ikoko", "cosmology"),
        ("What was the ancient name of Benin before the Oba dynasty?", "igodomigodo", "history"),
        ("Who were the Ogiso rulers of ancient Benin?", "ogiso", "history"),
        ("What is the full royal title of the Oba of Benin?", "omo n'oba n'edo", "royalty"),
        ("What role does the Iyoba play in Benin Kingdom history?", "iyoba", "history"),
        ("Explain the concept of Ehi in Bini spiritual philosophy.", "ehi", "cosmology"),
        ("How many times does the Orhion reincarnate before reaching Osanobua?", "14", "cosmology"),
        ("Translate to Bini/Edo: The Oba of Benin is the sacred ruler of Edo people.", "ọmọ n'oba n'edo", "royalty"),
        ("Translate to Bini/Edo: Where are you going this morning?", "owie", "question"),
        ("Translate to Bini/Edo: My mother prepared delicious pounded yam and soup.", "iye mwẹn", "complex_sentence"),
        ("Translate to Bini/Edo: We are going to the market to buy food.", "eki", "complex_sentence"),
        ("Translate to Bini/Edo: Respect elders and you will receive blessings from the ancestors.", "osanobua", "complex_sentence"),
        ("Translate to Bini/Edo: Unity in the family brings strength and peace.", "akugbe", "proverb"),
        ("Translate to Bini/Edo: The sun is rising over Benin City today.", "ọwẹn", "nature"),
        ("Translate to Bini/Edo: Let us preserve the Edo language for future generations.", "edo", "preservation"),
        ("Translate to Bini/Edo: May Osanobua protect your household.", "osanobua", "blessing"),
        ("Translate to Bini/Edo: The ancient walls of Benin were built with great skill.", "iya n'edo", "history"),
        ("Translate to Bini/Edo: Bronze casting is a sacred royal art in Benin.", "eronmwon", "art"),
        ("Translate to Bini/Edo: An honest person earns the respect of the entire community.", "omwan", "proverb"),
        ("Translate to Bini/Edo: Hard work on the farm yields plenty of yams.", "emwin", "agriculture"),
        ("Translate to Bini/Edo: The leopard is the sacred royal animal of the Oba.", "ẹkpẹn", "royalty"),
        ("Translate to Bini/Edo: Peace in the land brings joy to the kingdom.", "ẹghọghọ", "proverb"),
        ("Translate to Bini/Edo: I am proud of my Edo cultural heritage.", "edo", "culture"),
        ("Translate to Bini/Edo: Children are the greatest treasure of a family.", "ọmọ", "kinship"),
        ("Translate to Bini/Edo: A humble mind avoids unnecessary trouble.", "omwan", "proverb"),
        ("Translate to Bini/Edo: What would you like to eat tomorrow morning?", "owie", "complex_sentence"),
        ("Translate to Bini/Edo: Peace be with you", "ọ y'ese", "greeting"),
        ("Translate to Bini/Edo: Good afternoon", "ọbavan", "greeting"),
    ]

    return igbo_questions, efik_questions, edo_questions


def run_benchmark():
    print("=" * 80)
    print("AFRIWISE 2.0: 300 DISTINCT QUESTIONS BENCHMARK (100 IGBO, 100 EFIK, 100 EDO)")
    print("=" * 80)

    igbo_q, efik_q, edo_q = build_300_questions()
    print(f"Loaded: {len(igbo_q)} Igbo Questions, {len(efik_q)} Efik Questions, {len(edo_q)} Edo Questions.")
    total_questions = len(igbo_q) + len(efik_q) + len(edo_q)
    print(f"Total Benchmark Probes: {total_questions}\n")

    engine = get_knowledge_engine()
    results = {"igbo": [], "efik": [], "edo": []}

    categories = [
        ("🦅 IGBO (100 Questions)", igbo_q, "igbo"),
        ("🌿 EFIK / IBIBIO (100 Questions)", efik_q, "efik"),
        ("👑 EDO / BINI (100 Questions)", edo_q, "edo"),
    ]

    total_passed = 0
    total_hallucinations = 0

    for cat_title, q_list, lang_key in categories:
        print("\n" + "=" * 70)
        print(cat_title)
        print("=" * 70)
        cat_passed = 0
        cat_hallucinations = 0

        for idx, (prompt, expected_kw, category) in enumerate(q_list, 1):
            t0 = time.time()
            res = engine.query(prompt)
            latency = (time.time() - t0) * 1000
            ans = res["answer"]

            # Evaluation
            ans_clean = ans.lower()
            expected_clean = expected_kw.lower()
            
            # Check for keyword match or valid cultural refusal
            is_valid = (expected_clean in ans_clean) or any(r in ans_clean for r in ["a maghị m", "mmọdiọkke", "i ma-ẹre"])
            # Check for hallucination markers (cross-lingual confusion or random gibberish)
            is_hallucination = not is_valid and ("note:" in ans_clean or len(ans.strip()) == 0)

            if is_valid:
                cat_passed += 1
                total_passed += 1
                status = "PASS"
            else:
                cat_hallucinations += 1
                total_hallucinations += 1
                status = "FAIL"

            results[lang_key].append({
                "id": f"{lang_key}_{idx:03d}",
                "prompt": prompt,
                "expected": expected_kw,
                "category": category,
                "status": status,
                "latency_ms": round(latency, 2),
                "answer": ans[:120] + ("..." if len(ans) > 120 else "")
            })

            if idx % 20 == 0 or idx == len(q_list) or not is_valid:
                print(f"[{lang_key.upper()} {idx:03d}/100] [{status}] {prompt} -> {ans[:60]}... ({latency:.1f}ms)")

        cat_acc = (cat_passed / len(q_list)) * 100.0
        cat_hal_rate = (cat_hallucinations / len(q_list)) * 100.0
        print(f"\n--- {lang_key.upper()} Summary: Passed {cat_passed}/{len(q_list)} ({cat_acc:.1f}%), Hallucinations: {cat_hallucinations} ({cat_hal_rate:.1f}%) ---")

    overall_acc = (total_passed / total_questions) * 100.0
    overall_hal = (total_hallucinations / total_questions) * 100.0

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_probes": total_questions,
        "total_passed": total_passed,
        "total_hallucinations": total_hallucinations,
        "accuracy_rate_pct": round(overall_acc, 2),
        "hallucination_rate_pct": round(overall_hal, 2),
        "target_hallucination_max_pct": 1.0,
        "verdict": "PASS" if overall_hal <= 1.0 else "FAIL",
        "detailed_results": results
    }

    with open(BENCHMARK_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"FINAL 300-QUESTION BENCHMARK VERDICT: {summary['verdict']}")
    print(f"Total Probes:       {total_questions}")
    print(f"Passed:             {total_passed}/{total_questions} ({overall_acc:.2f}%)")
    print(f"Hallucination Rate: {overall_hal:.2f}% (Target: <= 1.00%)")
    print(f"Saved results to:   {BENCHMARK_OUTPUT}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
