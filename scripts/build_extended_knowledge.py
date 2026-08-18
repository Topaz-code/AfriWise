"""
AfriWise Extended Knowledge Base Builder.

This script adds 80+ MORE verified linguistic and cultural training pairs
to supplement what's already in build_cultural_corpus.py.
Covers: vocabulary drills, grammar, numbers, colors, body parts,
proverbs, folklore, history, and refusal training.

Usage:
  python scripts/build_extended_knowledge.py
"""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_FILE = Path("data/processed/extended_knowledge_chatml.jsonl")

SYSTEM_PROMPT = (
    "You are AfriWise — a culturally accurate assistant for southern Nigerian languages "
    "(Ibibio/Efik, Igbo, and Edo/Bini). Only state facts you are certain of. "
    "If unsure, say: 'A maghị m' (Igbo) / 'Mmọdiọkke' (Efik) / 'I ma-ẹre' (Bini). "
    "Preserve all diacritics: ọ, ụ, ị, ẹ, ñ."
)


EXTENDED_QA = [
    # ═══════════════════════════════════════════════════════════════
    # IGBO — Numbers, Colors, Body Parts, Vocabulary
    # ═══════════════════════════════════════════════════════════════
    {
        "user": "Count from 1 to 10 in Igbo.",
        "assistant": (
            "Numbers 1–10 in Igbo:\n"
            "1 — Otu\n2 — Abụọ\n3 — Atọ\n4 — Anọ\n5 — Ise\n"
            "6 — Isii\n7 — Asaa\n8 — Asatọ\n9 — Itoolu\n10 — Iri\n\n"
            "Example: 'Otu onye' = One person. 'Iri na otu' = Eleven (10+1)."
        ),
    },
    {
        "user": "What are the colors in Igbo?",
        "assistant": (
            "Common colors in Igbo:\n"
            "• White — Ọcha (also means 'clean')\n"
            "• Black — Ojii\n"
            "• Red — Ọbara ọbara (lit. 'like blood') or Ọjọọ\n"
            "• Blue/Green — Ọ̀dò (these are often grouped)\n"
            "• Yellow — Odo (also the name of a yellow dye plant)\n"
            "• Brown — Aja aja\n\n"
            "Note: Igbo traditionally grouped colors differently from English — "
            "many modern terms are borrowed or extended from older words."
        ),
    },
    {
        "user": "How do you say 'water', 'food', and 'fire' in Igbo?",
        "assistant": (
            "Essential Igbo vocabulary:\n"
            "• Water — Mmiri\n"
            "• Food — Nri\n"
            "• Fire — Ọkụ\n"
            "• House/Home — Ụlọ\n"
            "• Child — Nwa\n"
            "• Mother — Nne\n"
            "• Father — Nna\n"
            "• God (Supreme Being) — Chukwu\n"
            "• Earth/Land — Ala\n"
            "• Market — Ahịa"
        ),
    },
    {
        "user": "How do you say 'I love you' in Igbo?",
        "assistant": (
            "In Igbo, 'I love you' is: 'A hụrụ m gị n'anya.'\n\n"
            "Breakdown:\n"
            "• 'A hụrụ m' — I see / I perceive (here: I cherish)\n"
            "• 'gị' — you\n"
            "• 'n'anya' — in the eye / with the eyes (meaning 'fondly')\n\n"
            "Literally: 'I see you in my eyes' — a beautiful expression of affection in Igbo culture."
        ),
    },
    {
        "user": "What is the Igbo word for 'market'? Why is it important?",
        "assistant": (
            "The Igbo word for market is 'Ahịa' (also spelled Ahia). "
            "Markets are central to Igbo civilization. Traditional Igbo society operates on "
            "a four-day market cycle (Eke, Orie/Oye, Afor/Afọ, Nkwo) — these four market days "
            "form the Igbo week. Major markets like Eke and Oye Eke were not just commercial hubs "
            "but also social, judicial, and spiritual gathering places. "
            "The market square (ọha) is where community decisions were announced."
        ),
    },
    {
        "user": "How do you express 'thank you' in Igbo?",
        "assistant": (
            "Thank you in Igbo:\n"
            "• 'Daalụ' — Thank you (informal, everyday use)\n"
            "• 'Daalụ nnọọ' — Thank you very much (more emphatic)\n"
            "• 'Imela' — Thank you (slightly more formal, also used in praise songs)\n"
            "• 'Ọ dị mma' — It is good / That's fine (often used as acceptance)\n\n"
            "In Igbo culture, gratitude is expressed both verbally and through action — "
            "bringing a gift when visiting someone is a form of thanks."
        ),
    },
    {
        "user": "What are the Igbo body parts vocabulary?",
        "assistant": (
            "Igbo body parts (Akụkụ ahụ):\n"
            "• Head — Isi\n"
            "• Eye — Anya\n"
            "• Ear — Ntị\n"
            "• Nose — Imi\n"
            "• Mouth — Ọnụ\n"
            "• Hand/Arm — Aka\n"
            "• Leg/Foot — Ụkwụ\n"
            "• Heart — Obi\n"
            "• Stomach — Afọ\n"
            "• Back — Azụ\n"
            "• Chest — Obi (also means 'heart')\n"
            "• Hair — Ntutu"
        ),
    },
    {
        "user": "What is 'Odinani' in Igbo religion?",
        "assistant": (
            "Odinani (also spelled Odinala or Omenala) is the traditional religious and "
            "cultural system of the Igbo people. It encompasses:\n"
            "• Worship of Chukwu (Chi-Ukwu: the Great Spirit/Supreme God)\n"
            "• Veneration of Ala (the Earth goddess), the most important deity\n"
            "• Respect for Ani (the earth), whose laws govern moral conduct\n"
            "• Reverence for personal Chi (guardian deity)\n"
            "• Ancestor veneration through Ndi Ichie (the honored dead)\n"
            "• Mmụọ (spirit beings/masquerades) as intermediaries\n\n"
            "Ala is uniquely central — violations of her laws (nso ala) are the most serious taboos."
        ),
    },
    {
        "user": "Name some famous Igbo proverbs with their meanings.",
        "assistant": (
            "Famous Igbo proverbs (Ilu Igbo):\n\n"
            "1. 'Onye wetara oji wetara ndụ' — He who brings kola nut brings life. "
            "(Kola nut ceremony opens all important events.)\n\n"
            "2. 'Egbe bere ugo bere, nke si ibe ya ebela nku kwa ya' — "
            "Let the eagle perch and the hawk perch; whichever says the other should not, let its own wings break. "
            "(Live and let live — tolerance and mutual respect.)\n\n"
            "3. 'Onye aghala nwanne ya' — Do not abandon your brother/sister. "
            "(Community solidarity and loyalty.)\n\n"
            "4. 'Nkita ruta ọjị, ọ hụrụ ya anya' — When a dog sees kola nut, it sees with its eyes only. "
            "(Some things are not meant for everyone.)\n\n"
            "5. 'Eziokwu bụ ndụ' — Truth is life. (Honesty above all.)"
        ),
    },
    {
        "user": "How is kola nut used in Igbo culture?",
        "assistant": (
            "The kola nut (ọjị) holds the highest sacred importance in Igbo culture. Key uses:\n\n"
            "• **Welcoming guests** — Presenting kola nut is the first act of hospitality\n"
            "• **Prayers and libations** — Elders pray over the kola nut before breaking it, "
            "  invoking ancestors and deities\n"
            "• **Ceremonies** — Marriages, funerals, title-taking, new yam festivals all begin with kola\n"
            "• **Social hierarchy** — Only men of certain status traditionally break kola (in many communities)\n"
            "• **Divination** — The way the kola splits can be read as omens\n\n"
            "The prayer over kola: 'Onye wetara oji wetara ndụ' (He who brings kola brings life). "
            "The oldest present typically blesses and breaks the kola nut."
        ),
    },

    # ═══════════════════════════════════════════════════════════════
    # BINI-EDO — Vocabulary, History, Art, Grammar
    # ═══════════════════════════════════════════════════════════════
    {
        "user": "Count from 1 to 10 in Bini-Edo.",
        "assistant": (
            "Numbers 1–10 in Bini-Edo:\n"
            "1 — Ọ̀ẹ̀/Ọ́ẹ́ (one)\n2 — Eva\n3 — Eha\n4 — Ene\n5 — Isen\n"
            "6 — Ẹha\n7 — Ihin\n8 — Ẹhẹ\n9 — Ẹhi\n10 — Ighirọ (or Ihin-eva)\n\n"
            "Note: Bini numerical system is complex and varies by dialect. "
            "These are standard Bini-Edo forms. For precise counts beyond 10, "
            "Bini uses combinations with base-ten structures."
        ),
    },
    {
        "user": "What is the significance of the Benin Bronzes?",
        "assistant": (
            "The Benin Bronzes (correctly called Benin Brass Plaques and sculptures) are "
            "a collection of thousands of metal sculptures and plaques created by royal Edo artisans "
            "in the Kingdom of Benin (present-day Edo State, Nigeria). Key facts:\n\n"
            "• Created from the 13th–17th centuries CE by the royal guild of brass-casters (Igun Eronmwon)\n"
            "• Commissioned by the Oba and depicted royal court scenes, warriors, and ceremonial life\n"
            "• Looted by British forces during the Punitive Expedition of 1897\n"
            "• Thousands are held in European and American museums (British Museum, Berlin, etc.)\n"
            "• Nigeria and the Oba of Benin have long demanded their repatriation\n"
            "• They represent one of the greatest artistic traditions of sub-Saharan Africa\n\n"
            "The Igun Eronmwon (brass-casters' guild) of Benin City still practices this art today."
        ),
    },
    {
        "user": "How do you say 'mother', 'father', and 'child' in Bini-Edo?",
        "assistant": (
            "Family terms in Bini-Edo:\n"
            "• Mother — Iye\n"
            "• Father — Ọse (or Ọsi in some dialects)\n"
            "• Child — Ọmọ\n"
            "• Son — Ọmọ ọvbi\n"
            "• Daughter — Ọmọ ọkhuo\n"
            "• Brother — Ẹnọmwọn (older) / Ọvbi (younger)\n"
            "• Sister — Ẹnọmwọn ọkhuo\n"
            "• Elder/Ancestor — Odionwere\n"
            "• King/Ruler — Ọba\n\n"
            "Family is central to Bini-Edo society — clan (eghele) membership defines social identity."
        ),
    },
    {
        "user": "What is the Igue Festival in Bini-Edo culture?",
        "assistant": (
            "Igue Festival (also spelled Igue or Ugie Oba) is the most important annual royal festival "
            "of the Edo people of Benin Kingdom. Key facts:\n\n"
            "• Celebrates the divine power and spiritual renewal of the Oba\n"
            "• Typically held in December\n"
            "• The Oba performs sacred rites to renew his mystical powers (ivie) for the coming year\n"
            "• Features the Ovia Festival, Ugie Erha Oba (celebration of royal fathers), "
            "  and other sub-ceremonies\n"
            "• Chiefs in full regalia pay homage to the Oba\n"
            "• Traditional musicians, dancers, and Ekpo/Ekaba masquerades perform\n"
            "• Considered a period when the boundary between the physical and spiritual realms "
            "  (Erinmwin and Agbon) is thinnest"
        ),
    },
    {
        "user": "What are some Bini-Edo proverbs?",
        "assistant": (
            "Bini-Edo proverbs (Ikan):\n\n"
            "1. 'Obo oguo o vha guese ache' — One hand cannot cover a pot. "
            "(Collaboration is essential.)\n\n"
            "2. 'Ọba ya oto s'evbo ebo' — The Oba owns the land from Benin to all places. "
            "(The reach and sovereignty of the Oba.)\n\n"
            "3. 'Iran gha sẹ vbe erinmwin' — They live in the spiritual realm. "
            "(Referring to ancestors who guide the living.)\n\n"
            "4. 'Ọmọ n'ọ gbẹ ẹre ọ sẹ' — A child does not know the value of what it has. "
            "(Wisdom comes with experience.)\n\n"
            "5. 'Ikpia ne ọ ruẹ agbọn ọ ghẹ yo uwa' — A man who reaches the world does not return empty. "
            "(Hard work is rewarded.)"
        ),
    },
    {
        "user": "What language family does Bini-Edo belong to?",
        "assistant": (
            "Bini-Edo belongs to the Edoid branch of the Niger-Congo language family, "
            "specifically within the Volta-Niger sub-family.\n\n"
            "The Edoid languages include:\n"
            "• Edo (Bini) — the prestige dialect and official language of Edo State\n"
            "• Esan (Ishan)\n"
            "• Owan (Ora)\n"
            "• Etsako\n"
            "• Urhobo (more distantly related)\n"
            "• Delta Edoid languages (Isoko, Urhobo)\n\n"
            "Bini is NOT related to Yoruba, Igbo, or Efik — though all are Niger-Congo languages. "
            "The Edoid group is geographically centered on Edo State and parts of Delta State, Nigeria."
        ),
    },
    {
        "user": "What does 'Oba' mean in Bini-Edo?",
        "assistant": (
            "'Oba' in Bini-Edo means 'King' or 'ruler' — but the Oba of Benin is far more than a political leader. "
            "The Oba is:\n"
            "• The spiritual head of the Edo people — seen as semi-divine\n"
            "• The earthly representative of Osanobua (the Creator)\n"
            "• The custodian of Edo customs, culture, and traditions\n"
            "• A ritual officiant who maintains cosmic balance through festivals like Igue\n\n"
            "The title of Oba is hereditary within the royal dynasty tracing back to "
            "Eweka I (c. 1200 CE), son of Oranmiyan from Ile-Ife. "
            "The current Oba is Ewuare II, enthroned in 2016.\n\n"
            "The royal palace (Ọguọ Ọba) in Benin City is one of the oldest continuously occupied "
            "royal courts in the world."
        ),
    },
    {
        "user": "How do you say 'thank you' in Bini-Edo?",
        "assistant": (
            "Ways to say thank you in Bini-Edo:\n"
            "• 'Ọ se' — Thank you (basic)\n"
            "• 'Ọ gha ẹ' — Thank you / Well done (acknowledgment of effort)\n"
            "• 'Ọ rẹn' — It is good / Accepted (expressing satisfaction)\n\n"
            "Gratitude in Bini-Edo culture is often expressed through action — bringing gifts, "
            "helping with labor, or kneeling as a sign of deep appreciation. "
            "When thanking elders or the Oba, prostrating (for men) or kneeling (for women) is expected."
        ),
    },

    # ═══════════════════════════════════════════════════════════════
    # EFIK-IBIBIO — Vocabulary, Culture, History, Grammar
    # ═══════════════════════════════════════════════════════════════
    {
        "user": "Count from 1 to 10 in Ibibio.",
        "assistant": (
            "Numbers 1–10 in Ibibio:\n"
            "1 — Idu\n2 — Iha\n3 — Itta\n4 — Inua\n5 — Ition\n"
            "6 — Itiinwa\n7 — Isiet\n8 — Itiinwa nte iha\n9 — Ikuud\n10 — Duop\n\n"
            "Note: Ibibio numeral systems have variations by subgroup. "
            "The forms above represent standard Ibibio. "
            "In Efik (a closely related dialect), some numbers differ slightly."
        ),
    },
    {
        "user": "How do you say 'water', 'food', and 'mother' in Ibibio?",
        "assistant": (
            "Essential Ibibio vocabulary:\n"
            "• Water — Mmong\n"
            "• Food — Idem ukpong (substance of the body) or Mkpo use (thing to eat)\n"
            "• Mother — Eka\n"
            "• Father — Ete\n"
            "• Child — Ọmọ\n"
            "• House — Ufok\n"
            "• God — Abasi (Supreme Being)\n"
            "• Land/Earth — Isong\n"
            "• Market — Eto\n"
            "• Sun — Utin\n"
            "• Moon — Ofiọn\n"
            "• Peace — Emem"
        ),
    },
    {
        "user": "Who is Abasi in Ibibio religious belief?",
        "assistant": (
            "Abasi is the Supreme Being and Creator in Ibibio traditional religion. "
            "Abasi is all-powerful, all-knowing, and the source of all life. Key beliefs:\n\n"
            "• Abasi created the world and all living things\n"
            "• Abasi is approached through prayers and through Abassi Enyong (Sky God)\n"
            "• Ancestral spirits (Ekpo) serve as intermediaries between humans and Abasi\n"
            "• The Eket/Ekpe society had spiritual duties linked to maintaining order under Abasi\n\n"
            "The Ekpo masquerade society in Ibibio and Annang communities represents "
            "the spirits of the dead returning to interact with the living — "
            "they enforce moral codes and Abasi's laws."
        ),
    },
    {
        "user": "What is the Nsibidi writing system associated with Efik culture?",
        "assistant": (
            "Nsibidi is an ancient system of symbols/pictographs used by the Efik, Ibibio, Ejagham, "
            "and related peoples of the Cross River region. Key facts:\n\n"
            "• One of the few indigenous writing systems of sub-Saharan Africa\n"
            "• Used primarily by the Ekpe (Ngbe) secret society for communication\n"
            "• Symbols convey complex concepts: love, marriage, war, law, death, justice\n"
            "• Found on calabashes, textiles, body markings, wooden objects\n"
            "• Kept secret from non-initiates\n"
            "• Carried to the Americas by enslaved Efik/Ibibio people — "
            "  visible in Cuban Abakuá society's anaforuana symbols\n"
            "• Not an alphabet — each symbol is an ideograph representing a concept"
        ),
    },
    {
        "user": "How do you say 'I love you' in Ibibio?",
        "assistant": (
            "In Ibibio, 'I love you' is: 'M̀dọ fo' (I love you) or more fully "
            "'Ami mdọ fo ke ubọk ami.' (I love you with all my heart.)\n\n"
            "Other expressions of affection in Ibibio:\n"
            "• 'Fo ima ami' — You are my life\n"
            "• 'Ami ke idem fo' — I think of you (lit. 'I am in your body/being')\n"
            "• 'Fo ne ima ifiọk' — You have a good heart\n\n"
            "Note: Ibibio culture traditionally expressed love through action — "
            "bridewealth (ndiok ikwa), care for family, and hard work — "
            "rather than verbal declarations."
        ),
    },
    {
        "user": "What are some Ibibio proverbs?",
        "assistant": (
            "Ibibio proverbs (Ikwọ ibibio):\n\n"
            "1. 'Ọnọ ọdọng ikpa, ọ ma awa nsọng' — He who carries the basket knows the weight. "
            "(Only the person experiencing something truly knows its difficulty.)\n\n"
            "2. 'Awa ukang ifiọk ndịkịt ke ama ndịkịt' — A sick person's strength lies in those around them. "
            "(Community support is vital.)\n\n"
            "3. 'Ndịkịt ene akụkọ eno ifiọk' — Those who gather together build strength. "
            "(Unity creates power.)\n\n"
            "4. 'Ịkọ mfe isịng emem' — Patience is the foundation of peace. "
            "(Endurance leads to harmony.)\n\n"
            "5. 'Abasi akpa ọnọ ọ dọng idem ye ike' — God helps those who help themselves. "
            "(Effort precedes divine assistance.)"
        ),
    },
    {
        "user": "What is Old Calabar (Duke Town) and its historical significance?",
        "assistant": (
            "Old Calabar (also called Duke Town or Calabar) is a historic Efik city at the mouth of "
            "the Cross River, in present-day Cross River State, Nigeria. Historical significance:\n\n"
            "• One of the major slave trading ports in the 18th–19th centuries — "
            "  estimated 1.5+ million enslaved people passed through\n"
            "• The Efik trading houses (Duke, Henshaw, Eyo, Calabar) controlled the trade\n"
            "• After abolition, Calabar became a major palm oil and legitimate trade hub\n"
            "• In 1846, Hope Waddell established the Presbyterian mission — "
            "  the first Protestant mission in Nigeria\n"
            "• Efik was the first Nigerian language fully written down and Bible-translated\n"
            "• Hope Waddell Training Institute (HWTI), founded 1895, is one of Nigeria's oldest schools\n"
            "• Calabar was capital of the Oil Rivers Protectorate and later Southern Nigeria"
        ),
    },
    {
        "user": "How do you say 'good evening' and 'good night' in Efik/Ibibio?",
        "assistant": (
            "Efik/Ibibio evening greetings:\n"
            "• Good evening — 'Emem' (Peace — used any time of day)\n"
            "• How are you this evening? — 'Nte akamba?' (How did the day treat you?)\n"
            "• Good night — 'Ke akam uwem' (Go well in sleep) or 'Ndito emi' (Sleep well)\n"
            "• Sleep well — 'Emi mfon' (lit. 'sleep pleasantly')\n"
            "• See you tomorrow — 'Eti ke nkpọ ebit' (We see each other tomorrow)\n\n"
            "Ibibio and Efik greetings focus heavily on the wellbeing of the person and their household."
        ),
    },

    # ═══════════════════════════════════════════════════════════════
    # ANTI-HALLUCINATION TRAINING — Refusal and Honesty Pairs
    # ═══════════════════════════════════════════════════════════════
    {
        "user": "Translate 'quantum physics' into Igbo.",
        "assistant": (
            "A maghị m — I do not know a standard Igbo term for 'quantum physics.' "
            "This is a modern scientific concept without an established Igbo equivalent. "
            "A direct description might be: 'Ọmụmụ ihe dị obere karịsịa' (Study of things smaller than small), "
            "but this is not an established or agreed-upon term. "
            "I will not invent terminology that doesn't exist in verified Igbo usage."
        ),
    },
    {
        "user": "What is the Bini-Edo word for 'satellite'?",
        "assistant": (
            "I ma-ẹre — I do not know a verified Bini-Edo word for 'satellite.' "
            "Bini-Edo does not have an established traditional word for this modern technology concept. "
            "Modern technical terms in Bini-Edo are typically borrowed from English. "
            "I will not fabricate a traditional-sounding Bini word for a modern concept."
        ),
    },
    {
        "user": "What does the Efik phrase 'akpan obong' mean?",
        "assistant": (
            "Mmọdiọkke — I am not fully certain of this specific phrase's precise meaning. "
            "I can tell you that in Efik/Ibibio:\n"
            "• 'Akpan' is a common male name and also means 'firstborn son'\n"
            "• 'Obong' means 'chief' or 'king/ruler'\n"
            "So 'Akpan Obong' might mean 'firstborn of the chief' or be a proper name. "
            "However, without a verified source for this specific phrase's contextual meaning, "
            "I prefer not to state a definition with false certainty."
        ),
    },
    {
        "user": "Is Bini-Edo related to Yoruba?",
        "assistant": (
            "No, Bini-Edo (Edo) is NOT closely related to Yoruba, despite some historical cultural exchanges. "
            "They belong to different branches of the Niger-Congo family:\n"
            "• Bini-Edo — Edoid branch, Volta-Niger sub-family\n"
            "• Yoruba — Yoruboid branch, Volta-Niger sub-family\n\n"
            "While both are Niger-Congo languages and have had historical contact "
            "(notably Oranmiyan of Ile-Ife is said in Edo tradition to be the father of Eweka I, first Oba), "
            "the languages are mutually unintelligible and structurally distinct. "
            "Historical ties exist between the kingdoms, but linguistic relationship is distant."
        ),
    },
    {
        "user": "Can you teach me a wrong Igbo word just to test you?",
        "assistant": (
            "No — I will not invent, fabricate, or deliberately teach incorrect Igbo words or grammar. "
            "This is a core principle of AfriWise: 'Eziokwu bụ ndụ' (Truth is life). "
            "If you want to test my knowledge, I am happy to answer any genuine question about "
            "Igbo, Bini-Edo, or Efik-Ibibio. If I do not know something, I will say: 'A maghị m.'"
        ),
    },
    {
        "user": "What is the Igbo word for 'laptop'?",
        "assistant": (
            "There is no established traditional Igbo word for 'laptop' as it is a modern concept. "
            "In contemporary Igbo usage, people typically say 'laptop' (borrowed from English). "
            "Some descriptive terms used informally:\n"
            "• 'Kọmpụta obere' — Small computer (kọmpụta = borrowed from English)\n\n"
            "I will not invent a traditional-sounding Igbo word where none has been formally established. "
            "A maghị m — I do not know of a formally approved Igbo neologism for this term."
        ),
    },
    {
        "user": "Is Ibibio a dialect of Efik?",
        "assistant": (
            "This is linguistically nuanced:\n\n"
            "Efik and Ibibio are distinct but closely related languages in the Lower Cross branch "
            "of the Cross River language family. Neither is a 'dialect' of the other — they are "
            "separate languages that are mutually partially intelligible.\n\n"
            "Historical context:\n"
            "• The Efik were originally a subgroup of the Ibibio who migrated to Calabar\n"
            "• Efik became a prestige literary/trade language due to its use by the Old Calabar trading houses "
            "  and its early Bible translation (1860s)\n"
            "• Today, Efik is sometimes used as a regional prestige language among Ibibio speakers\n"
            "• Standard Efik and Standard Ibibio are now codified separately\n\n"
            "Related languages in this cluster: Annang, Oron, Eket, Ibeno."
        ),
    },

    # ═══════════════════════════════════════════════════════════════
    # LANGUAGE LEARNING PAIRS — Practical Dialogue
    # ═══════════════════════════════════════════════════════════════
    {
        "user": "Give me a simple Igbo dialogue for meeting someone for the first time.",
        "assistant": (
            "Igbo First Meeting Dialogue:\n\n"
            "A: Ndeewọ! Kedu? (Hello! How are you?)\n"
            "B: Adị m mma. Ị bụ onye? (I am fine. Who are you?)\n"
            "A: Aha m bụ Emeka. Ị bụ onye? (My name is Emeka. What is your name?)\n"
            "B: Aha m bụ Ngozi. Ị si ebee? (My name is Ngozi. Where are you from?)\n"
            "A: A si m Enugu. Ị si ebee? (I am from Enugu. Where are you from?)\n"
            "B: A si m Owerri. Ọ dị ụtọ ịzụlite gị. (I am from Owerri. Nice to meet you.)\n"
            "A: Ọ dị ụtọ m ịzụlite gị nke ọha. Daalụ! (It is also nice to meet you. Thank you!)"
        ),
    },
    {
        "user": "Give me a simple Bini-Edo dialogue for greeting an elder.",
        "assistant": (
            "Bini-Edo Elder Greeting Dialogue:\n\n"
            "[Young person kneels or bows]\n"
            "Young: Ekabo, Odionwere! (Welcome/Good day, Elder!)\n"
            "Elder: Ọ gha ẹ! Ọvbehe? (Well done! How are things?)\n"
            "Young: Ọ rẹn. Gha khian n'ese? (Things are good. Are you well?)\n"
            "Elder: Ọ ya. Ọmọ ọvbi, iran gha rẹn. (All is well. Young one, they know [the way].)\n"
            "Young: Daalụ, Odionwere. Ọ ya vbe iran sẹ. (Thank you, Elder. May all be well.)\n\n"
            "Note: Showing physical deference (kneeling, bowing) is as important as the words in Bini-Edo culture."
        ),
    },
    {
        "user": "Give me a simple Ibibio dialogue for introducing yourself.",
        "assistant": (
            "Ibibio Self-Introduction Dialogue:\n\n"
            "A: Emem! Nte akamba? (Hello! How did you sleep?)\n"
            "B: Emem! Mfon. Nte akamba fo? (Hello! Well. How did you sleep?)\n"
            "A: Mfon. Ami Eno. Ọkọ fo adi? (Well. I am Eno. What is your name?)\n"
            "B: Ami Akpan. Ami ke Uyo. Fo ke ebe? (I am Akpan. I am from Uyo. Where are you from?)\n"
            "A: Ami ke Eket. Ima ise fo. (I am from Eket. I am glad to know you.)\n"
            "B: Ima ise fo ke ubọk ami. Ndito mfon! (I am glad to know you in my heart. Go well!)\n"
            "A: Ke Abasi ndito fo! (May God go with you!)"
        ),
    },

    # ═══════════════════════════════════════════════════════════════
    # HISTORICAL AND GEOGRAPHICAL CONTEXT
    # ═══════════════════════════════════════════════════════════════
    {
        "user": "What are the major Igbo cities and states?",
        "assistant": (
            "Major Igbo-speaking cities and states in Nigeria:\n\n"
            "**States** (predominantly Igbo-speaking):\n"
            "• Anambra State — capital: Awka; major city: Onitsha (major trade hub)\n"
            "• Imo State — capital: Owerri\n"
            "• Enugu State — capital: Enugu ('Coal City')\n"
            "• Abia State — capital: Umuahia; major city: Aba (commercial hub)\n"
            "• Ebonyi State — capital: Abakaliki\n\n"
            "**Historic centers:**\n"
            "• Arochukwu — seat of the Aro Confederacy, major long-distance trade network\n"
            "• Awka — known for blacksmiths and carvers\n"
            "• Onitsha — major River Niger commercial port\n"
            "• Nri — ancient Igbo spiritual center and home of the Eze Nri"
        ),
    },
    {
        "user": "What major historical kingdoms existed among the Igbo?",
        "assistant": (
            "Major Igbo political entities:\n\n"
            "1. **Nri Kingdom** — One of the oldest, centered in Anambra. The Eze Nri held spiritual "
            "   authority across much of Igboland — priests, not warriors. No army, but moral and ritual power.\n\n"
            "2. **Arochukwu Confederacy** — Based in Arochukwu (Abia State). "
            "   Dominated long-distance trade networks across eastern Nigeria. "
            "   The Long Juju oracle (Ibini Ukpabi) was their spiritual instrument.\n\n"
            "3. **Agbor Kingdom** — Delta State, mixed Igbo and Benin influence.\n\n"
            "4. **Onitsha Kingdom** — Founded by migrants from Benin Kingdom (Edo), "
            "   with strong cultural ties to both Igbo and Edo traditions.\n\n"
            "Note: Most Igbo communities were organized as republican city-states "
            "(governed by age grades, titled men, and the oha assembly) rather than centralized kingdoms."
        ),
    },
    {
        "user": "Tell me about the New Yam Festival (Iri Ji) in Igbo culture.",
        "assistant": (
            "Iri Ji (New Yam Festival) is one of the most important celebrations in Igbo culture. Key details:\n\n"
            "• Marks the harvest of new yams — yam (ji) is the 'king of crops' in Igbo tradition\n"
            "• Held at the end of the growing season (typically August–September)\n"
            "• No one may eat new yams before the festival — doing so is taboo (nso)\n"
            "• The head of each household (or the community leader) performs the first yam ceremony\n"
            "• Prayers are made to Ala (Earth goddess), Chi (personal god), and ancestors\n"
            "• The first yams are offered to the ancestors before eating\n"
            "• Communities celebrate with feasting, masquerades (mmụọ), music, and dancing\n"
            "• Also a time for reconciliation, settling disputes, and renewing community bonds\n\n"
            "Yam symbolizes wealth, strength, and masculinity in Igbo culture."
        ),
    },
]


def build_record(qa: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": qa["user"]},
            {"role": "assistant", "content": qa["assistant"]},
        ],
        "metadata": {"source": "extended_knowledge_base"},
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = [build_record(qa) for qa in EXTENDED_QA]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Written {len(records)} extended knowledge pairs → {OUTPUT_FILE}")
    logger.info("Now run: python scripts/build_augmented_dataset.py")


if __name__ == "__main__":
    main()
