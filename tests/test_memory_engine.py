from src.memory_engine import ResearchAssistantEngine


def test_document_ingestion_is_idempotent_by_content_hash() -> None:
    engine = ResearchAssistantEngine(session_id="s1", short_term_token_budget=50)

    files = [
        ("notes.md", b"# Intro\nParis is the capital of France."),
        ("page.html", b"<html><body><h1>Facts</h1><p>Berlin is in Germany.</p></body></html>"),
    ]

    first = engine.ingest_files(files)
    second = engine.ingest_files(files)

    assert first["new_chunks"] > 0
    assert second["new_chunks"] == 0
    assert second["deduplicated_files"] == 2


def test_short_term_overflow_creates_compressed_memory() -> None:
    engine = ResearchAssistantEngine(session_id="s1", short_term_token_budget=10)

    engine.add_user_turn("Alice prefers concise summaries.")
    engine.add_assistant_turn("Understood. I will keep responses concise.")
    engine.add_user_turn("Also remember project codename is Orion.")

    stats = engine.memory_stats()

    assert stats["compression_events"] >= 1
    assert stats["conversation_memories"] >= 1
    assert stats["buffer_tokens"] <= 20


def test_retrieval_searches_document_and_conversation_streams() -> None:
    engine = ResearchAssistantEngine(session_id="s1", short_term_token_budget=60)

    engine.ingest_files([("science.txt", b"Water boils at 100C at sea level.")])
    engine.add_user_turn("My preferred unit is Celsius.")
    engine.add_assistant_turn("Got it, I will answer in Celsius.")
    engine.force_compress_buffer()

    retrieved = engine.retrieve("What is boiling point in my preferred unit?", top_k=8)
    origins = {item.origin for item in retrieved}

    assert "document" in origins
    assert "conversation" in origins


def test_conflict_between_document_and_chat_is_surfaced() -> None:
    engine = ResearchAssistantEngine(session_id="s1", short_term_token_budget=60)

    engine.ingest_files([("policy.txt", b"Launch date is Tuesday.")])
    engine.add_user_turn("Launch date is Wednesday.")
    engine.force_compress_buffer()

    result = engine.answer("What is the launch date?")

    assert result.conflicts
    assert any("launch date" in conflict.lower() for conflict in result.conflicts)
    assert result.attributions


def test_user_confirmed_conclusion_promotes_importance() -> None:
    engine = ResearchAssistantEngine(session_id="s1", short_term_token_budget=50)

    engine.promote_confirmed_fact("The final decision is to use blue branding.")

    convo_memories = [r for r in engine.vector_records() if r.origin == "conversation"]
    assert convo_memories
    assert max(r.importance for r in convo_memories) >= 0.95
