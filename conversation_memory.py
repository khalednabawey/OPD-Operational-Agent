from __future__ import annotations

import os
import json
import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import requests
# from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

if TYPE_CHECKING:
    from agent_core import GroqLLM

from contextlib import closing
from policy_rag import _RemoteEmbeddingFunction, _SentenceTransformerEF

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "opd_agent.db")

# Use DATABASE_URL for PostgreSQL/Production, otherwise fallback to local SQLite
DB_URL = DATABASE_URL if DATABASE_URL else f"sqlite:///{SQLITE_DB_PATH}"

# ========== Dataclasses ==========
@dataclass
class ConversationRecord:
    """
    One row = one user query + agent response.
    Stored per-day, not accumulated across days.
    """
    record_id: str
    user_email: str
    bu: str | None
    query: str              # cr6a9_query — the user's exact question
    summary: str            # cr6a9_summary — holistic summary of the daily chat
    full_transcript: str    # cr6a9_fulltranscript — JSON of the entire day's conversation
    last_kpi_mentioned: str | None
    conversation_date: date  # cr6a9_date — the date of this conversation
    message_count: int = 2  # 1 user + 1 assistant
    is_escalated: bool = False


@dataclass
class FAQEntry:
    faq_id: str
    question: str
    answer: str
    category: str
    tags: list[str] = field(default_factory=list)
    related_kpi: str | None = None
    hit_count: int = 0


# ========== Conversation Store (SQLite) — Per-Query, Per-Day ==========
class SQLAlchemyConversationStore:
    """
    Manages database storage for conversation records using SQLAlchemy.
    Supports both SQLite and PostgreSQL via the DB_URL connection string.
    """

    def __init__(self, db_url: str = DB_URL) -> None:
        self.engine = create_engine(db_url)
        # Log database type for verification
        self.db_type = "PostgreSQL" if "postgresql" in str(self.engine.url) else "SQLite"
        print(f"[ConversationStore] Initializing {self.db_type} database connection...")
        self._init_db()

    def _init_db(self) -> None:
        dialect = str(self.engine.url)
        with self.engine.begin() as conn:
            # Performance optimization for SQLite
            if "sqlite" in dialect:
                conn.execute(text("PRAGMA journal_mode=WAL;"))
            
            transcript_type = "JSONB" if "postgresql" in dialect else "TEXT"
            
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS opd_conversations (
                        record_id TEXT PRIMARY KEY,
                        user_email TEXT,
                        bu TEXT,
                        query TEXT,
                        summary TEXT,
                        full_transcript {transcript_type},
                        last_kpi_mentioned TEXT,
                        conversation_date TEXT,
                        message_count INTEGER,
                        is_escalated INTEGER,
                        created_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP)
                )
            """))

    def save_record(self, record: ConversationRecord) -> bool:
        try:
            with self.engine.begin() as conn:
                # Delete existing to simulate 'REPLACE' across different SQL dialects
                conn.execute(text("DELETE FROM opd_conversations WHERE record_id = :id"), {"id": record.record_id})
                conn.execute(text("""
                    INSERT INTO opd_conversations (
                            record_id, user_email, bu, query, summary, full_transcript, 
                            last_kpi_mentioned, conversation_date, message_count, is_escalated
                    ) VALUES (:record_id, :user_email, :bu, :query, :summary, :full_transcript, 
                            :last_kpi_mentioned, :conversation_date, :message_count, :is_escalated)
                """), {
                    "record_id": record.record_id, "user_email": record.user_email, "bu": record.bu,
                    "query": record.query, "summary": record.summary, "full_transcript": record.full_transcript,
                    "last_kpi_mentioned": record.last_kpi_mentioned, "conversation_date": record.conversation_date.isoformat(),
                    "message_count": record.message_count, "is_escalated": 1 if record.is_escalated else 0
                })
            print(f"[ConversationStore] Saved record {record.record_id} to {self.db_type}.")
            return True
        except Exception as e:
            print(f"[ConversationStore] Error saving record: {e}")
            return False

    def _row_to_record(self, row: tuple) -> ConversationRecord:
        # Postgres JSONB columns return dict/list objects; SQLite returns strings.
        # We ensure it's a string for compatibility with the dataclass and json.loads calls.
        transcript = row[5]
        if not isinstance(transcript, str) and transcript is not None:
            transcript = json.dumps(transcript, ensure_ascii=False)

        return ConversationRecord(
            record_id=row[0],
            user_email=row[1],
            bu=row[2],
            query=row[3],
            summary=row[4],
            full_transcript=transcript,
            last_kpi_mentioned=row[6],
            conversation_date=date.fromisoformat(row[7]),
            message_count=row[8],
            is_escalated=bool(row[9])
        )

    def load_by_date(
        self,
        user_email: str,
        target_date: date,
        bu: str | None = None,
    ) -> list[ConversationRecord]:
        sql = "SELECT * FROM opd_conversations WHERE user_email = :user_email AND conversation_date = :date"
        params = {"user_email": user_email, "date": target_date.isoformat()}
        if bu:
            sql += " AND bu = :bu"
            params["bu"] = bu
        sql += " ORDER BY created_at ASC"

        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return [self._row_to_record(row) for row in result]

    def load_date_range(
        self,
        user_email: str,
        start_date: date,
        end_date: date,
        bu: str | None = None,
    ) -> list[ConversationRecord]:
        sql = "SELECT * FROM opd_conversations WHERE user_email = :email AND conversation_date >= :start AND conversation_date <= :end"
        params = {"email": user_email, "start": start_date.isoformat(), "end": end_date.isoformat()}
        if bu:
            sql += " AND bu = :bu"
            params["bu"] = bu
        sql += " ORDER BY conversation_date DESC, created_at ASC"

        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return [self._row_to_record(row) for row in result]

    def load_recent_days(
        self,
        user_email: str,
        days: int = 7,
        bu: str | None = None,
    ) -> list[ConversationRecord]:
        """Get conversations from last N days."""
        end_date = date.today()
        start_date = end_date - timedelta(days=days - 1)
        return self.load_date_range(user_email, start_date, end_date, bu)

    def load_recent_records(
        self,
        user_email: str,
        top_n: int = 5,
        bu: str | None = None,
    ) -> list[ConversationRecord]:
        sql = "SELECT * FROM opd_conversations WHERE user_email = :email"
        params = {"email": user_email}
        if bu:
            sql += " AND bu = :bu"
            params["bu"] = bu
        sql += f" ORDER BY created_at DESC LIMIT {top_n}"

        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params)
            return [self._row_to_record(row) for row in result]


# ========== FAQ Store (SQLite) ==========

class SQLiteFAQStore:
    def __init__(self, db_path: str = SQLITE_DB_PATH) -> None:
        self.db_path = db_path
        self._cache: list[FAQEntry] = []
        self._last_fetch: datetime | None = None
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with closing(sqlite3.connect(self.db_path, timeout=10)) as conn:
            with conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS opd_faqs (
                        faq_id TEXT PRIMARY KEY,
                        question TEXT,
                        answer TEXT,
                        category TEXT,
                        tags TEXT,
                        related_kpi TEXT,
                        hit_count INTEGER DEFAULT 0
                    )
                """)

    def refresh_cache(self) -> None:
        try:
            with closing(sqlite3.connect(self.db_path, timeout=10)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM opd_faqs")
                rows = cursor.fetchall()

            self._cache = []
            for row in rows:
                self._cache.append(
                    FAQEntry(
                        faq_id=row[0],
                        question=row[1],
                        answer=row[2],
                        category=row[3],
                        tags=[t.strip()
                              for t in (row[4] or "").split(",") if t.strip()],
                        related_kpi=row[5],
                        hit_count=row[6],
                    )
                )
            self._last_fetch = datetime.utcnow()
            print(f"[FAQStore] Loaded {len(self._cache)} FAQs from SQLite.")
        except Exception as e:
            print(f"[FAQStore] Error fetching FAQs: {e}")
            self._cache = []

    def get_all(self) -> list[FAQEntry]:
        if not self._cache or (self._last_fetch and (datetime.utcnow() - self._last_fetch).total_seconds() > 300):
            self.refresh_cache()
        return self._cache

    def record_hit(self, faq_id: str) -> None:
        try:
            with closing(sqlite3.connect(self.db_path, timeout=10)) as conn:
                with conn:
                    conn.execute(
                        "UPDATE opd_faqs SET hit_count = hit_count + 1 WHERE faq_id = ?", (faq_id,))
        except Exception as e:
            print(f"[FAQStore] Error recording hit: {e}")


# ========== Semantic FAQ Engine (unchanged) ==========

_FAQ_ENGINE_INSTANCE = None


class FAQEngine:
    def __init__(self, faq_store: DataverseFAQStore, model_name: str = "all-MiniLM-L6-v2", embedding_server_url: str | None = os.getenv("EMBEDDING_SERVER_URL")) -> None:
        self.faq_store = faq_store

        # Use remote embedding server if configured to avoid local model downloads
        if embedding_server_url:
            print(
                f"[FAQEngine] Using remote embedding server at {embedding_server_url}")
            self.embedding_function = _RemoteEmbeddingFunction(
                embedding_server_url)
        else:
            self.embedding_function = _SentenceTransformerEF(model_name)

        self.collection = chromadb.Client(Settings(anonymized_telemetry=False)).create_collection(
            name="opd_faqs", metadata={"hnsw:space": "cosine"}
        )
        self._indexed = False

    def _index(self) -> None:
        if self._indexed:
            return
        faqs = self.faq_store.get_all()
        if not faqs:
            self._indexed = True
            return
        texts = [
            f"Q: {f.question} A: {f.answer} Category: {f.category} KPI: {f.related_kpi or ''}" for f in faqs]
        ids = [f.faq_id for f in faqs]
        embeddings = self.embedding_function(texts)
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts)
        self._indexed = True
        print(f"[FAQEngine] Indexed {len(faqs)} FAQs.")

    def query(self, question: str, n_results: int = 3, min_similarity: float = 0.65) -> list[dict[str, Any]]:
        self._index()
        if not self.faq_store.get_all():
            return []
        q_emb = self.embedding_function([question])
        results = self.collection.query(
            query_embeddings=q_emb, n_results=n_results, include=["distances", "documents"])
        hits = []
        for idx, faq_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][idx]
            similarity = 1 - distance
            if similarity >= min_similarity:
                faq = next((f for f in self.faq_store.get_all()
                           if f.faq_id == faq_id), None)
                if faq:
                    self.faq_store.record_hit(faq_id)
                    hits.append({
                        "faq_id": faq_id,
                        "question": faq.question,
                        "answer": faq.answer,
                        "category": faq.category,
                        "related_kpi": faq.related_kpi,
                        "similarity": round(similarity, 3),
                    })
        return hits

    def format_context(self, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return ""
        lines = ["[FAQ CONTEXT — Previously Answered Questions]"]
        for h in hits:
            lines.append(
                f"Q: {h['question']}\nA: {h['answer']} (Category: {h['category']}, Similarity: {h['similarity']})")
        return "\n\n".join(lines)


def _get_faq_engine() -> FAQEngine | None:
    global _FAQ_ENGINE_INSTANCE
    if _FAQ_ENGINE_INSTANCE is None:
        store = SQLiteFAQStore()
        try:
            _FAQ_ENGINE_INSTANCE = FAQEngine(faq_store=store)
        except Exception as e:
            print(f"[FAQEngine] Init failed: {e}")
            _FAQ_ENGINE_INSTANCE = None
    return _FAQ_ENGINE_INSTANCE


# ========== Answer Summarizer (Lightweight — No Full LLM Call) ==========

class AnswerSummarizer:
    """
    Creates a brief summary of the agent's answer without calling the LLM again.
    Uses heuristics + simple extraction.
    """

    def summarize(self, question: str, answer: str) -> str:
        # Take first 2-3 sentences of the answer, up to 200 chars
        sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        summary = " ".join(sentences[:2]) if len(sentences) >= 2 else answer
        if len(summary) > 250:
            summary = summary[:247] + "..."
        return summary

    def extract_kpi(self, text: str) -> str | None:
        from agent_core import _resolve_kpi_from_text
        return _resolve_kpi_from_text(text)

    def is_escalated(self, question: str, answer: str) -> bool:
        combined = (question + " " + answer).lower()
        escalation_signals = [
            "escalate", "urgent", "critical", "red flag", "🔴",
            "complaint", "unacceptable", "violation", "penalty",
            "salary deduction", "bonus holding", "annual leave deduction",
        ]
        return any(signal in combined for signal in escalation_signals)


# ========== Conversation Manager (Per-Query, Per-Day) ==========

class ConversationManager:
    """
    Each ask() = one record in Dataverse with its own date.
    No accumulation across days. Retrieve by specific date or date range.
    """

    def __init__(self, llm: GroqLLM, user_email: str | None = None, bu: str | None = None) -> None:
        self.llm = llm
        self.user_email = user_email
        self.bu = bu
        self.store = SQLAlchemyConversationStore()
        self.summarizer = AnswerSummarizer()
        self.faq_engine = _get_faq_engine()

    def _generate_daily_summary(self, question: str, answer: str, previous_summary: str | None = None) -> str:
        """Use LLM to generate an incremental concise summary by updating the last summary with the new turn."""
        new_interaction = f"User: {question[:400]}\nAgent: {answer[:400]}"

        if previous_summary:
            prompt = (
                f"EXISTING SUMMARY OF CONVERSATION SO FAR:\n{previous_summary}\n\n"
                f"LATEST INTERACTION TURN:\n{new_interaction}\n\n"
                "TASK: Update the existing summary to incorporate key findings, KPIs, and decisions from the latest interaction. "
                "The result must be a single, holistic summary representing the state of the entire daily session. "
                "CONSTRAINT: Be extremely concise (max 3 sentences). Use numbers and specific names from the data."
            )
        else:
            prompt = (
                f"LATEST INTERACTION:\n{new_interaction}\n\n"
                "TASK: Provide a concise summary of this initial healthcare data analysis interaction. "
                "CONSTRAINT: Be extremely concise (max 3 sentences). Use numbers and specific names from the data."
            )

        system_prompt = (
            "You are a summarization assistant for a Healthcare BI Agent. "
            "Your goal is to maintain a stateful, incremental summary of a conversation that persists key analytical insights."
        )

        try:
            # Pass a strict max_tokens limit to prevent hitting organization-level TPM limits
            print("[ConversationManager] Starting Daily Summary Generation...")
            gen_output = self.llm.generate(
                prompt, system_prompt_override=system_prompt, max_tokens=250
            )
            # Handle potential variance in generate() return signature across different core versions
            if isinstance(gen_output, tuple):
                summary = gen_output[0]
            else:
                summary = gen_output
            return summary
        except Exception as e:
            print(
                f"[ConversationManager] Daily summary generation failed: {e}")
            return self.summarizer.summarize(question, answer)

    def build_memory_context(
        self,
        current_question: str,
        lookback_days: int = 1,  # Default: only today's context
    ) -> str:
        """
        Build context from recent conversations.
        lookback_days=1 means only today (same date).
        lookback_days=7 means last 7 days.
        """
        if not self.user_email:
            return ""

        parts: list[str] = []

        # 1. Same-day or recent conversation history
        today = date.today()
        if lookback_days == 1:
            records = self.store.load_by_date(self.user_email, today, self.bu)
            # Limit to last 3 interactions to prevent system prompt token bloat over long sessions
            records = records[-3:]
            if records:
                parts.append(
                    f"[CHAT HISTORY — Recorded on {today.isoformat()}]")
        else:
            records = self.store.load_recent_days(
                self.user_email, lookback_days, self.bu)
            if records:
                parts.append(
                    f"[RECENT CHAT HISTORY — Recorded over last {lookback_days} days]")

        if records:
            for r in records:
                parts.append(
                    f"• [Turn Date: {r.conversation_date}] Q: {r.query}")
                parts.append(
                    f"  → Daily Summary: {r.summary} (KPI: {r.last_kpi_mentioned or 'N/A'})")

        # 2. Relevant FAQs
        if self.faq_engine:
            faq_hits = self.faq_engine.query(
                current_question, n_results=2, min_similarity=0.70)
            if faq_hits:
                parts.append(self.faq_engine.format_context(faq_hits))

        return "\n\n".join(parts) if parts else ""

    def save(
        self,
        question: str,
        answer: str,
        full_context: str,
    ) -> None:
        """
        Saves or updates the daily conversation record for the user.
        Aggregates the transcript and generates a holistic daily summary.
        """
        if not self.user_email:
            return

        today = date.today()
        # Stable ID for the user/BU per day ensures we update the same row
        id_seed = f"{self.user_email}:{self.bu or 'None'}:{today.isoformat()}"
        record_id = hashlib.md5(id_seed.encode()).hexdigest()

        # 1. Fetch existing transcript for today
        history_messages = []
        accumulated_kpis = set()
        existing_summary = None
        existing = self.store.load_by_date(self.user_email, today, self.bu)
        if existing:
            existing_summary = existing[0].summary
            # Load history
            try:
                history_messages = json.loads(existing[0].full_transcript)
            except (json.JSONDecodeError, IndexError):
                history_messages = []

            # Load previously mentioned KPIs
            if existing[0].last_kpi_mentioned:
                accumulated_kpis.update(
                    [k.strip() for k in existing[0].last_kpi_mentioned.split(",")])

        # 2. Append current turn
        history_messages.append({"role": "user", "content": question})
        history_messages.append({"role": "assistant", "content": answer})

        # 3. Generate incremental summary (rolling update)
        daily_summary = self._generate_daily_summary(question, answer, existing_summary)

        # 4. Extract and accumulate KPIs
        new_kpi = self.summarizer.extract_kpi(question + " " + answer)
        if new_kpi:
            accumulated_kpis.add(new_kpi)

        is_escalated = self.summarizer.is_escalated(question, answer)

        record = ConversationRecord(
            record_id=record_id,
            user_email=self.user_email,
            bu=self.bu,
            query=question,  # Store most recent query
            summary=daily_summary,
            full_transcript=json.dumps(history_messages, ensure_ascii=False),
            last_kpi_mentioned=", ".join(
                sorted(accumulated_kpis)) if accumulated_kpis else None,
            conversation_date=today,
            message_count=len(history_messages),
            is_escalated=is_escalated,
        )

        try:
            self.store.save_record(record)
        except Exception as e:
            print(f"[ConversationManager] Failed to save record: {e}")

    # Convenience methods for external reporting
    def get_today_conversations(self) -> list[ConversationRecord]:
        if not self.user_email:
            return []
        return self.store.load_by_date(self.user_email, date.today(), self.bu)

    def get_conversations_by_date(self, target_date: date) -> list[ConversationRecord]:
        if not self.user_email:
            return []
        return self.store.load_by_date(self.user_email, target_date, self.bu)

    def get_conversations_date_range(self, start: date, end: date) -> list[ConversationRecord]:
        if not self.user_email:
            return []
        return self.store.load_date_range(self.user_email, start, end, self.bu)


if __name__ == "__main__":
    # Test script to verify SQLite initialization and CRUD operations
    print(f"--- Starting Database Test (DB Path: {SQLITE_DB_PATH}) ---")

    # 1. Initialize Stores (This creates the tables)
    conv_store = SQLAlchemyConversationStore()
    faq_store = SQLiteFAQStore()

    test_email = "test.user@andalusiagroup.net"
    test_bu = "SMH"

    # 2. Create a dummy record
    test_record = ConversationRecord(
        record_id="test_hash_12345",
        user_email=test_email,
        bu=test_bu,
        query="What was the revenue for SMH in 2024?",
        summary="The revenue for SMH in 2024 was 5.2M SAR.",
        full_transcript=json.dumps([
            {"role": "user", "content": "What was the revenue for SMH in 2024?"},
            {"role": "assistant", "content": "The revenue for SMH in 2024 was 5.2M SAR."}
        ]),
        last_kpi_mentioned="Total Revenue",
        conversation_date=date.today(),
        is_escalated=False
    )

    # 3. Save the record
    success = conv_store.save_record(test_record)
    if success:
        print("✅ Successfully saved a test conversation record.")
    else:
        print("❌ Failed to save test conversation record.")

    # 4. Verify retrieval
    recent = conv_store.load_recent_records(test_email, top_n=1)
    if recent and recent[0].record_id == "test_hash_12345":
        print(f"✅ Verified data retrieval: Found query '{recent[0].query}'")
    else:
        print("❌ Data retrieval verification failed.")

    # 5. Check FAQ table (should be empty but existing)
    faqs = faq_store.get_all()
    print(f"ℹ️ FAQ table initialized. Current count: {len(faqs)}")

    print("--- Test Complete ---")
