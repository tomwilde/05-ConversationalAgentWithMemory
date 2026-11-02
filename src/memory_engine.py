from __future__ import annotations

import hashlib
import html
import io
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
CLAIM_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9_\s]{1,40})\s+is\s+([a-zA-Z0-9_\-\s]{1,40})", re.I)


@dataclass
class Turn:
    turn_id: int
    role: str
    text: str
    created_at: datetime


@dataclass
class MemoryRecord:
    record_id: str
    origin: str
    text: str
    token_count: int
    vector: dict[str, float]
    created_at: datetime
    session_id: str | None = None
    turn_id: int | None = None
    importance: float = 0.5
    provenance: dict[str, Any] = field(default_factory=dict)
    mutable: bool = False


@dataclass
class RetrievedItem:
    record_id: str
    origin: str
    text: str
    score: float
    importance: float
    provenance: dict[str, Any]


@dataclass
class AnswerResult:
    text: str
    attributions: list[str]
    conflicts: list[str]
    retrieved: list[RetrievedItem]


class ResearchAssistantEngine:
    def __init__(
        self,
        session_id: str,
        short_term_token_budget: int = 800,
        retrieval_token_budget: int = 500,
    ) -> None:
        self.session_id = session_id
        self.short_term_token_budget = short_term_token_budget
        self.retrieval_token_budget = retrieval_token_budget

        self._turn_counter = 0
        self._record_counter = 0
        self._turns: list[Turn] = []
        self._vector_records: list[MemoryRecord] = []
        self._seen_doc_hashes: set[str] = set()
        self._doc_versions: dict[str, int] = {}

        self._compression_events = 0
        self._retrieval_events = 0
        self._memory_hits = 0
        self._relevance_scores: list[float] = []

    def ingest_files(self, files: list[tuple[str, bytes]]) -> dict[str, int]:
        new_files = 0
        deduplicated_files = 0
        new_chunks = 0

        for name, file_bytes in files:
            digest = hashlib.sha256(file_bytes).hexdigest()
            if digest in self._seen_doc_hashes:
                deduplicated_files += 1
                continue

            text = _extract_text(name, file_bytes)
            if not text.strip():
                continue

            new_files += 1
            self._seen_doc_hashes.add(digest)
            doc_id = _doc_id_from_name(name)
            version = self._doc_versions.get(doc_id, 0) + 1
            self._doc_versions[doc_id] = version

            for index, chunk in enumerate(_chunk_text(text, max_chars=800), start=1):
                record = self._new_record(
                    origin="document",
                    text=chunk,
                    importance=0.5,
                    mutable=False,
                    provenance={
                        "doc_id": doc_id,
                        "version": version,
                        "locator": f"chunk-{index}",
                        "hash": digest,
                    },
                )
                self._vector_records.append(record)
                new_chunks += 1

        return {
            "new_files": new_files,
            "deduplicated_files": deduplicated_files,
            "new_chunks": new_chunks,
        }

    def ingest_web_text(self, title: str, text: str) -> dict[str, int]:
        payload = text.encode("utf-8")
        filename = f"{title or 'web-page'}.html"
        return self.ingest_files([(filename, payload)])

    def add_user_turn(self, text: str) -> None:
        self._add_turn("user", text)

    def add_assistant_turn(self, text: str) -> None:
        self._add_turn("assistant", text)

    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievedItem]:
        self._retrieval_events += 1
        query_vector = _vectorize(query)
        now = datetime.now(UTC)

        scored: list[RetrievedItem] = []
        for record in self._vector_records:
            semantic = _cosine_sim(query_vector, record.vector)
            if semantic <= 0:
                continue

            age_hours = max((now - record.created_at).total_seconds() / 3600, 0)
            recency = 1 / (1 + age_hours)
            origin_weight = 1.0 if record.origin == "document" else 0.95
            score = semantic * (0.65 + 0.2 * recency + 0.15 * record.importance) * origin_weight

            scored.append(
                RetrievedItem(
                    record_id=record.record_id,
                    origin=record.origin,
                    text=record.text,
                    score=score,
                    importance=record.importance,
                    provenance=record.provenance,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        deduped = _dedup_retrieved(scored)
        budgeted = _cap_by_token_budget(deduped, self.retrieval_token_budget)
        selected = budgeted[:top_k]

        if selected and selected[0].score > 0.15:
            self._memory_hits += 1

        self._relevance_scores.extend(item.score for item in selected)
        return selected

    def answer(self, question: str) -> AnswerResult:
        retrieved = self.retrieve(question)
        conflicts = _detect_conflicts(retrieved)
        attributions = _build_attributions(retrieved)

        answer_text = self._generate_answer(question, retrieved, conflicts)
        self.add_user_turn(question)
        self.add_assistant_turn(answer_text)

        return AnswerResult(
            text=answer_text,
            attributions=attributions,
            conflicts=conflicts,
            retrieved=retrieved,
        )

    def promote_confirmed_fact(self, text: str) -> None:
        record = self._new_record(
            origin="conversation",
            text=text,
            importance=0.98,
            mutable=True,
            session_id=self.session_id,
            turn_id=self._turn_counter,
            provenance={
                "session_id": self.session_id,
                "turn_id": self._turn_counter,
                "confirmed": True,
            },
        )
        self._upsert_conversation_memory(record)

    def force_compress_buffer(self) -> None:
        if not self._turns:
            return
        self._compress_turns(self._turns[:])
        self._turns.clear()

    def memory_stats(self) -> dict[str, Any]:
        conversation_records = [r for r in self._vector_records if r.origin == "conversation"]
        hit_rate = self._memory_hits / self._retrieval_events if self._retrieval_events else 0.0
        avg_relevance = (
            sum(self._relevance_scores) / len(self._relevance_scores)
            if self._relevance_scores
            else 0.0
        )
        return {
            "session_id": self.session_id,
            "buffer_tokens": sum(_token_count(t.text) for t in self._turns),
            "conversation_memories": len(conversation_records),
            "document_chunks": len([r for r in self._vector_records if r.origin == "document"]),
            "compression_events": self._compression_events,
            "memory_hit_rate": round(hit_rate, 3),
            "avg_relevance_score": round(avg_relevance, 3),
        }

    def vector_records(self) -> list[MemoryRecord]:
        return list(self._vector_records)

    def chat_history(self) -> list[Turn]:
        return list(self._turns)

    def _add_turn(self, role: str, text: str) -> None:
        self._turn_counter += 1
        self._turns.append(
            Turn(turn_id=self._turn_counter, role=role, text=text, created_at=datetime.now(UTC))
        )
        self._enforce_short_term_budget()

    def _enforce_short_term_budget(self) -> None:
        current_tokens = sum(_token_count(turn.text) for turn in self._turns)
        if current_tokens <= self.short_term_token_budget:
            return

        to_compress: list[Turn] = []
        while self._turns and current_tokens > int(self.short_term_token_budget * 0.7):
            oldest = self._turns.pop(0)
            to_compress.append(oldest)
            current_tokens = sum(_token_count(turn.text) for turn in self._turns)

        if to_compress:
            self._compress_turns(to_compress)

    def _compress_turns(self, turns: list[Turn]) -> None:
        self._compression_events += 1
        source_turn_ids = [turn.turn_id for turn in turns]
        summary = _summarize_turns(turns)
        record = self._new_record(
            origin="conversation",
            text=summary,
            importance=0.7,
            mutable=True,
            session_id=self.session_id,
            turn_id=source_turn_ids[-1],
            provenance={
                "session_id": self.session_id,
                "turn_id": source_turn_ids[-1],
                "source_turn_ids": source_turn_ids,
            },
        )
        self._upsert_conversation_memory(record)

    def _upsert_conversation_memory(self, candidate: MemoryRecord) -> None:
        for record in self._vector_records:
            if record.origin != "conversation":
                continue
            similarity = _cosine_sim(record.vector, candidate.vector)
            if similarity >= 0.92:
                record.text = (
                    record.text if len(record.text) >= len(candidate.text) else candidate.text
                )
                record.vector = _vectorize(record.text)
                record.importance = max(record.importance, candidate.importance)
                record.provenance = {**record.provenance, **candidate.provenance}
                return

        self._vector_records.append(candidate)

    def _new_record(
        self,
        origin: str,
        text: str,
        importance: float,
        mutable: bool,
        provenance: dict[str, Any],
        session_id: str | None = None,
        turn_id: int | None = None,
    ) -> MemoryRecord:
        self._record_counter += 1
        cleaned = text.strip()
        return MemoryRecord(
            record_id=f"rec-{self._record_counter}",
            origin=origin,
            text=cleaned,
            token_count=_token_count(cleaned),
            vector=_vectorize(cleaned),
            created_at=datetime.now(UTC),
            session_id=session_id,
            turn_id=turn_id,
            importance=max(0.0, min(importance, 1.0)),
            provenance={"origin": origin, **provenance},
            mutable=mutable,
        )

    def _generate_answer(
        self,
        question: str,
        retrieved: list[RetrievedItem],
        conflicts: list[str],
    ) -> str:
        generated = _generate_with_openrouter(question, retrieved, conflicts)
        if generated is not None:
            return generated

        if not retrieved:
            return (
                "I do not have enough prior context yet. "
                "Upload documents or continue the conversation."
            )

        doc_snippets = [r.text for r in retrieved if r.origin == "document"]
        convo_snippets = [r.text for r in retrieved if r.origin == "conversation"]

        lines = ["Here is what I found from memory:"]
        if doc_snippets:
            lines.append(f"- According to documents: {doc_snippets[0][:180]}")
        if convo_snippets:
            lines.append(f"- Earlier you mentioned: {convo_snippets[0][:180]}")
        if conflicts:
            lines.append(f"- Conflict detected: {conflicts[0]}")
        lines.append(f"Question: {question}")
        return "\n".join(lines)


def _generate_with_openrouter(
    question: str,
    retrieved: list[RetrievedItem],
    conflicts: list[str],
) -> str | None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    context_lines = []
    for item in retrieved[:8]:
        prefix = "DOC" if item.origin == "document" else "CHAT"
        context_lines.append(f"[{prefix}] {item.text}")

    conflict_line = "\n".join(f"- {item}" for item in conflicts) if conflicts else "None"

    prompt = (
        "Answer using only the supplied memory context when possible. "
        "If evidence conflicts, explicitly state the conflict.\n\n"
        f"Context:\n{os.linesep.join(context_lines) or 'No context'}\n\n"
        f"Conflicts:\n{conflict_line}\n\n"
        f"User question: {question}"
    )

    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        completion = client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise research assistant. "
                        "Cite when a claim comes from document vs prior chat."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
    except Exception:
        return None

    choice = completion.choices[0]
    return choice.message.content if choice.message and choice.message.content else None


def _dedup_retrieved(items: list[RetrievedItem]) -> list[RetrievedItem]:
    seen_hashes: set[str] = set()
    deduped: list[RetrievedItem] = []
    for item in items:
        digest = hashlib.sha1(item.text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        deduped.append(item)
    return deduped


def _cap_by_token_budget(items: list[RetrievedItem], token_budget: int) -> list[RetrievedItem]:
    total = 0
    result: list[RetrievedItem] = []
    for item in items:
        count = _token_count(item.text)
        if total + count > token_budget:
            break
        total += count
        result.append(item)
    return result


def _extract_claims(text: str) -> dict[str, str]:
    claims: dict[str, str] = {}
    for match in CLAIM_RE.finditer(text):
        subject = " ".join(match.group(1).lower().split())
        predicate = " ".join(match.group(2).lower().split())
        claims[subject] = predicate
    return claims


def _detect_conflicts(retrieved: list[RetrievedItem]) -> list[str]:
    by_subject: dict[str, list[tuple[str, str]]] = {}
    for item in retrieved:
        for subject, predicate in _extract_claims(item.text).items():
            by_subject.setdefault(subject, []).append((predicate, item.origin))

    conflicts: list[str] = []
    for subject, values in by_subject.items():
        predicates = {predicate for predicate, _origin in values}
        origins = {origin for _predicate, origin in values}
        if len(predicates) > 1 and {"document", "conversation"}.issubset(origins):
            details = ", ".join(sorted(predicates))
            conflicts.append(f"{subject} has conflicting values: {details}")
    return conflicts


def _build_attributions(retrieved: list[RetrievedItem]) -> list[str]:
    attributions: list[str] = []
    seen: set[str] = set()
    for item in retrieved:
        if item.origin == "document":
            doc_id = item.provenance.get("doc_id", "unknown-doc")
            version = item.provenance.get("version", "?")
            locator = item.provenance.get("locator", "unknown-locator")
            label = f"document:{doc_id}@v{version}:{locator}"
        else:
            session_id = item.provenance.get("session_id", "unknown-session")
            turn_id = item.provenance.get("turn_id", "?")
            label = f"conversation:{session_id}:turn-{turn_id}"

        if label not in seen:
            seen.add(label)
            attributions.append(label)
    return attributions


def _summarize_turns(turns: list[Turn]) -> str:
    if not turns:
        return ""

    snippets: list[str] = []
    facts: list[str] = []
    for turn in turns:
        cleaned = turn.text.strip()
        if cleaned:
            snippets.append(f"{turn.role}: {cleaned}")
        facts.extend(f"{s} is {p}" for s, p in _extract_claims(cleaned).items())

    summary_parts = ["Compressed conversation memory:"]
    summary_parts.extend(snippets[:4])
    if facts:
        unique_facts = list(dict.fromkeys(facts))
        summary_parts.append("facts: " + "; ".join(unique_facts[:5]))
    return "\n".join(summary_parts)


def _doc_id_from_name(name: str) -> str:
    stem = Path(name).stem.lower().strip()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem or "document"


def _extract_text(name: str, file_bytes: bytes) -> str:
    suffix = Path(name).suffix.lower()

    if suffix == ".pdf" and PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except Exception:
            return ""

    decoded = file_bytes.decode("utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        no_tags = re.sub(r"<[^>]+>", " ", decoded)
        return html.unescape(re.sub(r"\s+", " ", no_tags)).strip()

    return decoded


def _chunk_text(text: str, max_chars: int) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []

    words = clean.split(" ")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        added_len = len(word) + (1 if current else 0)
        if current_len + added_len > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += added_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _token_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _vectorize(text: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    for token in WORD_RE.findall(text.lower()):
        normalized = _normalize_token(token)
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1

    total = sum(counts.values()) or 1
    return {token: count / total for token, count in counts.items()}


def _normalize_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]
    return token


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0

    if len(a) > len(b):
        a, b = b, a

    dot = sum(value * b.get(token, 0.0) for token, value in a.items())
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
