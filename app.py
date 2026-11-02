from __future__ import annotations

import uuid

import streamlit as st

from src.memory_engine import ResearchAssistantEngine

st.set_page_config(page_title="Conversational Agent with Memory", layout="wide")
st.title("Conversational Agent with Memory: Personal research assistant")

if "engine" not in st.session_state:
    st.session_state.engine = ResearchAssistantEngine(
        session_id=str(uuid.uuid4()),
        short_term_token_budget=180,
    )

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

engine: ResearchAssistantEngine = st.session_state.engine

with st.sidebar:
    st.header("Ingestion")
    uploads = st.file_uploader(
        "Upload PDFs, Markdown, text, or HTML",
        type=["pdf", "md", "markdown", "txt", "html", "htm"],
        accept_multiple_files=True,
    )
    if st.button("Ingest uploaded files", use_container_width=True):
        if not uploads:
            st.warning("No files selected.")
        else:
            payload = [(file.name, file.getvalue()) for file in uploads]
            outcome = engine.ingest_files(payload)
            st.success(
                "Ingestion complete: "
                f"{outcome['new_files']} new files, "
                f"{outcome['deduplicated_files']} deduplicated, "
                f"{outcome['new_chunks']} chunks"
            )

    st.divider()
    st.subheader("Memory metrics")
    st.json(engine.memory_stats())

    if st.button("Confirm latest assistant answer", use_container_width=True):
        latest_assistant = next(
            (
                entry["content"]
                for entry in reversed(st.session_state.chat_messages)
                if entry["role"] == "assistant"
            ),
            None,
        )
        if not latest_assistant:
            st.info("No assistant answer available yet.")
        else:
            engine.promote_confirmed_fact(latest_assistant)
            st.success("Promoted latest answer to high-importance memory.")

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("Conversation")
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("attributions"):
                st.caption("Sources: " + "; ".join(message["attributions"]))
            if message.get("conflicts"):
                st.warning("Conflicts: " + " | ".join(message["conflicts"]))

    question = st.chat_input("Ask a question about your documents or prior conversation")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        answer = engine.answer(question)
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer.text,
                "attributions": answer.attributions,
                "conflicts": answer.conflicts,
            }
        )
        st.rerun()

with col2:
    st.subheader("Session summary")
    stats = engine.memory_stats()
    st.metric("Document chunks", stats["document_chunks"])
    st.metric("Conversation memories", stats["conversation_memories"])
    st.metric("Compression events", stats["compression_events"])
    st.metric("Memory hit rate", stats["memory_hit_rate"])
