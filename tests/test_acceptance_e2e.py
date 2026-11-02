import pytest

from src.memory_engine import ResearchAssistantEngine


@pytest.mark.acceptance
def test_acceptance_conflict_surface_and_attribution() -> None:
    engine = ResearchAssistantEngine(session_id="acc-1", short_term_token_budget=50)
    engine.ingest_files([("timeline.txt", b"Launch date is Tuesday.")])
    engine.add_user_turn("Launch date is Wednesday.")
    engine.force_compress_buffer()

    result = engine.answer("What is the launch date?")

    assert result.conflicts
    assert result.attributions


@pytest.mark.acceptance
def test_acceptance_document_and_conversation_retrieval() -> None:
    engine = ResearchAssistantEngine(session_id="acc-2", short_term_token_budget=50)
    engine.ingest_files([("science.md", b"Water boils at 100C.")])
    engine.add_user_turn("My preferred unit is Celsius.")
    engine.force_compress_buffer()

    retrieved = engine.retrieve("boiling point preferred unit", top_k=10)
    origins = {item.origin for item in retrieved}

    assert "document" in origins
    assert "conversation" in origins
