"""
test_rag_engine.py — Unit Tests for AfriWise Ground-Truth RAG Engine.
"""

import pytest
from scripts.rag_engine import get_knowledge_engine, STANDARD_GREETINGS, REFUSALS


@pytest.fixture(scope="module")
def engine():
    return get_knowledge_engine()


def test_engine_initializes_with_documents(engine):
    assert len(engine.index.documents) > 1000, f"Expected >1000 records, got {len(engine.index.documents)}"


def test_simple_greeting_detection(engine):
    assert engine.is_simple_greeting("hi")
    assert engine.is_simple_greeting("hello")
    assert engine.is_simple_greeting("koyo")
    assert engine.is_simple_greeting("ndewo")
    assert not engine.is_simple_greeting("Explain the creation myth of Osanobua in ancient Benin")


def test_greeting_response_contains_all_three_languages(engine):
    res = engine.query("hi")
    assert res["grounded"] is True
    ans = res["answer"]
    assert "Ndewo" in ans or "Ụtụtụ ọma" in ans
    assert "Kọyọ" in ans or "Ọbowiẹ" in ans
    assert "Emem" in ans or "Amesiere" in ans


def test_igbo_proverb_retrieval(engine):
    res = engine.query("Explain the Igbo proverb Ilu bu mmanu e ji eri okwu")
    assert res["grounded"] is True
    ans = res["answer"].lower()
    assert "palm oil" in ans or "mmanụ" in ans


def test_bini_proverb_retrieval(engine):
    res = engine.query("What is the Bini proverb for one hand cannot cover the pot?")
    assert res["grounded"] is True
    ans = res["answer"]
    assert "Obo oguo o vha guese ache" in ans or "One hand" in ans


def test_efik_emem_meaning(engine):
    res = engine.query("What does Emem mean in Ibibio?")
    assert res["grounded"] is True
    ans = res["answer"].lower()
    assert "peace" in ans


def test_out_of_domain_safe_refusal(engine):
    res = engine.query("Translate quantum string theory into 15th century Edo")
    assert res["grounded"] is True
    ans = res["answer"]
    assert "I ma-ẹre" in ans or "A maghị m" in ans or "Mmọdiọkke" in ans
