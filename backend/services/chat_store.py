"""
Async chat history store backed by MySQL (via SQLAlchemy 2.0 async).

Architecture:
  - AsyncSession per operation (no session-per-request singleton — avoids
    stale-session issues under concurrent requests).
  - Each public method acquires a fresh session, runs, and closes it.
  - The engine is a module-level singleton; created lazily on first access.

Usage::

    store = get_chat_store()
    session = await store.create_session(agent_id=..., agent_config={...})
    msg = await store.create_message(session_id=..., role="user", content="...")
    history = await store.get_conversation_context(session_id=..., max_turns=6)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from backend.config import settings
from backend.models import Base, ChatFeedback, ChatMessage, ChatSession, FeedbackRating, MessageRole

_log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Engine + session factory (lazy singleton)
# ──────────────────────────────────────────────────────────────────────────────

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_engine() -> AsyncEngine:
    """Create the async engine. Falls back to SQLite when DB_URL is unset."""
    db_url = settings.db_url
    if db_url:
        return create_async_engine(
            db_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            echo=False,
            pool_pre_ping=True,
        )
    # SQLite fallback — useful for dev without Docker
    sqlite_path = str(settings.db_path)
    _log.info("DB_URL not set — using SQLite at %s", sqlite_path)
    return create_async_engine(
        f"sqlite+aiosqlite:///{sqlite_path}",
        echo=False,
        # NullPool avoids SQLite locking issues across concurrent sessions
        poolclass=NullPool,
    )


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


@asynccontextmanager
async def _session_scope() -> AsyncIterator[AsyncSession]:
    """
    Yield a fresh AsyncSession and commit on success / rollback on exception.
    Pattern::

        async with _session_scope() as session:
            result = await session.execute(...)
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


async def init_db() -> None:
    """
    Create all tables if they don't exist.
    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.
    """
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _log.info("Database tables initialised (engine: %s)", engine.url.drivername)


async def close_db() -> None:
    """Dispose the engine pool. Call on application shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


class ChatStore:
    """
    Async CRUD operations for chat sessions, messages, and feedback.
    All methods are standalone — no shared session state.
    """

    # ── Sessions ───────────────────────────────────────────────────────────

    async def create_session(
        self,
        agent_id: str,
        agent_config: dict[str, Any],
        title: str = "New conversation",
    ) -> ChatSession:
        """Create a new conversation session for the given agent."""
        session = ChatSession(
            agent_id=agent_id,
            title=title,
            agent_config_snapshot=agent_config,
            message_count=0,
        )
        async with _session_scope() as db:
            db.add(session)
        _log.debug("Created session %s for agent %s", session.session_id, agent_id)
        return session

    async def get_session(self, session_id: str) -> ChatSession | None:
        """Return a session by ID, or None if not found."""
        async with _session_scope() as db:
            result = await db.execute(
                select(ChatSession).where(ChatSession.session_id == session_id)
            )
            return result.scalar_one_or_none()

    async def list_sessions(
        self,
        agent_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[ChatSession], int]:
        """
        List sessions, optionally filtered by agent_id.
        Returns (sessions, total_count).
        """
        async with _session_scope() as db:
            query = select(ChatSession).order_by(ChatSession.updated_at.desc())
            count_query = select(ChatSession)

            if agent_id:
                query = query.where(ChatSession.agent_id == agent_id)
                count_query = count_query.where(ChatSession.agent_id == agent_id)

            total = (await db.execute(count_query)).scalars().all()
            result = (
                await db.execute(query.limit(limit).offset(offset))
            ).scalars().all()
            return list(result), len(total)

    async def update_session_title(self, session_id: str, title: str) -> ChatSession | None:
        """Update the session title (e.g. after deriving it from the first message)."""
        async with _session_scope() as db:
            result = await db.execute(
                select(ChatSession).where(ChatSession.session_id == session_id)
            )
            session = result.scalar_one_or_none()
            if session is None:
                return None
            session.title = title
        return await self.get_session(session_id)

    async def touch_session(self, session_id: str) -> None:
        """Bump updated_at so list-sorted-by-recent ordering stays correct."""
        async with _session_scope() as db:
            await db.execute(
                select(ChatSession).where(ChatSession.session_id == session_id)
            )
            # Use raw UPDATE to avoid loading the full row
            from sqlalchemy import update
            await db.execute(
                update(ChatSession)
                .where(ChatSession.session_id == session_id)
                .values(updated_at=ChatSession.updated_at)
            )

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages (CASCADE). Returns True if deleted."""
        async with _session_scope() as db:
            result = await db.execute(
                select(ChatSession).where(ChatSession.session_id == session_id)
            )
            session = result.scalar_one_or_none()
            if session is None:
                return False
            await db.delete(session)
        _log.debug("Deleted session %s", session_id)
        return True

    # ── Messages ───────────────────────────────────────────────────────────

    async def create_message(
        self,
        session_id: str,
        role: MessageRole | str,
        content: str,
        *,
        token_count: int | None = None,
        retrieval_context: list[dict[str, Any]] | None = None,
        sources: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        """
        Save a message and increment the parent session's message_count atomically.
        """
        if isinstance(role, str):
            role = MessageRole(role)

        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            token_count=token_count,
            retrieval_context=retrieval_context,
            sources=sources,
        )

        async with _session_scope() as db:
            db.add(message)
            # Increment session message_count
            from sqlalchemy import update
            await db.execute(
                update(ChatSession)
                .where(ChatSession.session_id == session_id)
                .values(
                    message_count=ChatSession.message_count + 1,
                    updated_at=ChatSession.updated_at,
                )
            )
        _log.debug("Created message %s in session %s", message.message_id, session_id)
        return message

    async def get_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ChatMessage]:
        """Return messages for a session, newest last."""
        async with _session_scope() as db:
            result = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def get_message(self, message_id: str) -> ChatMessage | None:
        """Return a single message by ID."""
        async with _session_scope() as db:
            result = await db.execute(
                select(ChatMessage).where(ChatMessage.message_id == message_id)
            )
            return result.scalar_one_or_none()

    async def update_message_content(
        self, message_id: str, content: str
    ) -> ChatMessage | None:
        """Update a message's content (used for retries / edits)."""
        async with _session_scope() as db:
            result = await db.execute(
                select(ChatMessage).where(ChatMessage.message_id == message_id)
            )
            msg = result.scalar_one_or_none()
            if msg is None:
                return None
            msg.content = content
        return await self.get_message(message_id)

    # ── Feedback ───────────────────────────────────────────────────────────

    async def upsert_feedback(
        self,
        message_id: str,
        rating: FeedbackRating | str,
        comment: str | None = None,
    ) -> ChatFeedback:
        """
        Create or update feedback for a message.
        Replaces any existing feedback (unique constraint on message_id).
        """
        if isinstance(rating, str):
            rating = FeedbackRating(rating)

        async with _session_scope() as db:
            result = await db.execute(
                select(ChatFeedback).where(ChatFeedback.message_id == message_id)
            )
            existing = result.scalar_one_or_none()
            if existing:
                existing.rating = rating
                existing.comment = comment
                fb = existing
            else:
                fb = ChatFeedback(
                    message_id=message_id,
                    rating=rating,
                    comment=comment,
                )
                db.add(fb)
        return fb

    async def get_feedback(self, message_id: str) -> ChatFeedback | None:
        """Return feedback for a message, if any."""
        async with _session_scope() as db:
            result = await db.execute(
                select(ChatFeedback).where(ChatFeedback.message_id == message_id)
            )
            return result.scalar_one_or_none()

    async def get_feedback_for_messages(
        self, message_ids: list[str]
    ) -> dict[str, ChatFeedback]:
        """Return a dict mapping message_id → ChatFeedback for the given message IDs."""
        if not message_ids:
            return {}
        async with _session_scope() as db:
            result = await db.execute(
                select(ChatFeedback).where(ChatFeedback.message_id.in_(message_ids))
            )
            rows = result.scalars().all()
        return {r.message_id: r for r in rows}

    # ── Conversation context ───────────────────────────────────────────────

    async def get_conversation_context(
        self,
        session_id: str,
        max_turns: int = 6,
    ) -> list[dict[str, Any]]:
        """
        Return a flat list of {"role": str, "content": str} dicts for the most
        recent `max_turns` (user + assistant = 1 turn) in the session.

        Pass this directly to the RAG pipeline's system prompt augmentation.

        Example::

            context = await store.get_conversation_context(session_id, max_turns=6)
            conversation_block = "\\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in context
            )
            # → "User: What is RAG?\nAssistant: RAG stands for..."
        """
        async with _session_scope() as db:
            result = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            all_messages: list[ChatMessage] = list(result.scalars().all())

        # Trim to last N turns (1 turn = user + assistant pair)
        # If odd number, the last user message is kept as an incomplete turn
        # so the LLM still has context.
        if len(all_messages) > max_turns * 2:
            all_messages = all_messages[-(max_turns * 2):]

        return [
            {"role": m.role.value, "content": m.content}
            for m in all_messages
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton accessor
# ──────────────────────────────────────────────────────────────────────────────

_chat_store: ChatStore | None = None


def get_chat_store() -> ChatStore:
    global _chat_store
    if _chat_store is None:
        _chat_store = ChatStore()
    return _chat_store
