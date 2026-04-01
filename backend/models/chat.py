"""
SQLAlchemy ORM models for chat history.

Tables:
  - chat_sessions  — one per conversation thread
  - chat_messages  — one per user/assistant turn
  - chat_feedback  — optional thumbs up/down per assistant message

Note on relationships: feedback is loaded separately via explicit JOINs to avoid
circular forward-reference issues with SQLAlchemy's mapper configuration.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MessageRole(str, PyEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class FeedbackRating(str, PyEnum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Tables
# ──────────────────────────────────────────────────────────────────────────────


class ChatSession(Base):
    """
    A single conversation thread, scoped to one agent.
    The agent's config is snapshotted at session-creation time so that
    editing an agent's system prompt does not retroactively change past
    conversations.
    """

    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    agent_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="New conversation")
    agent_config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Messages are loaded explicitly; no back_populates to avoid circular mapper config
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage",
        foreign_keys="ChatMessage.session_id",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
        lazy="selectin",
        viewonly=True,
    )

    __table_args__ = (
        Index("ix_chat_sessions_agent_created", "agent_id", "created_at"),
    )


class ChatMessage(Base):
    """
    A single turn in a conversation.
    """

    __tablename__ = "chat_messages"

    message_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole, name="message_role", native_enum=False),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Raw Qdrant chunks at answer time — stored so follow-up questions can reuse this context
    retrieval_context: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    # Formatted source references shown to the user in the UI
    sources: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Explicit session join only
    session: Mapped["ChatSession"] = relationship(
        "ChatSession",
        foreign_keys=[session_id],
        lazy="noload",
        viewonly=True,
    )

    __table_args__ = (
        Index("ix_chat_messages_session_created", "session_id", "created_at"),
    )


class ChatFeedback(Base):
    """
    Optional thumbs-up / thumbs-down on an assistant message.
    One row per message, enforced by UNIQUE constraint.
    """

    __tablename__ = "chat_feedback"

    feedback_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    message_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_messages.message_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    rating: Mapped[FeedbackRating] = mapped_column(
        Enum(FeedbackRating, name="feedback_rating", native_enum=False),
        nullable=False,
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Explicit join to message only — no back_populates to session
    message: Mapped["ChatMessage"] = relationship(
        "ChatMessage",
        foreign_keys=[message_id],
        lazy="noload",
        viewonly=True,
    )
