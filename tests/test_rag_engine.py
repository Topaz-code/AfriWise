"""
test_rag_engine.py — Unit Tests for AfriWise Ground-Truth RAG Engine.
"""

import pytest
from scripts.rag_engine import AfriWiseRAG


@pytest.fixture(scope="module")
def engine():
    rag = AfriWiseRAG()
    # Mock the lexicon for deterministic testing
    rag.lexicon = {
        "greetings": [
            {
                "english": "good morning",
                "igbo": "ututu oma",
                "efik": "amesiere",
                "edo": "obowie"
            }
        ],
        "medical_first_aid": [
            {
                "condition": "Fever",
                "english": "fever",
                "igbo": "ahu oku",
                "efik": "ufip idem",
                "edo": "erhen egbe"
            }
        ],
        "cultural_knowledge": [
            {
                "topic": "Osanobua",
                "details": "Osanobua is the creator god in Edo culture."
            }
        ]
    }
    return rag


def test_retrieve_grounding_context_greetings(engine):
    context = engine.retrieve_grounding_context("Good morning to you")
    assert "Greeting Mapping: English 'good morning' -> Igbo: 'ututu oma'" in context


def test_retrieve_grounding_context_medical(engine):
    context = engine.retrieve_grounding_context("I have a fever")
    assert "Verified First-Aid Protocol: [Fever]" in context
    assert "- Igbo: ahu oku" in context


def test_retrieve_grounding_context_cultural(engine):
    context = engine.retrieve_grounding_context("tell me about osanobua")
    assert "Cultural Reference [Osanobua]: Osanobua is the creator god in Edo culture." in context


def test_build_grounded_system_prompt_base(engine):
    prompt = engine.build_grounded_system_prompt("unknown random query")
    assert "You are ÀṢÀ (AfriWise)" in prompt
    assert "Place all your internal reasoning inside <think> and </think> tags" in prompt
    assert "## ABSTENTION RULE & CONTENT SAFETY:" in prompt
    assert "VERIFIED GROUNDING CONTEXT" not in prompt


def test_build_grounded_system_prompt_with_context(engine):
    prompt = engine.build_grounded_system_prompt("good morning fever osanobua")
    assert "VERIFIED GROUNDING CONTEXT:" in prompt
    assert "Greeting Mapping: English 'good morning'" in prompt
    assert "Verified First-Aid Protocol: [Fever]" in prompt
    assert "Cultural Reference [Osanobua]" in prompt
