"""
Pydantic schemas for the chat history API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────────────


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class FeedbackRating(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


# ──────────────────────────────────────────────────────────────────────────────
# Session schemas
# ──────────────────────────────────────────────────────────────────────────────


class AgentConfigSnapshot(BaseModel):
    """Frozen snapshot of the agent's configuration at session-creation time."""

    system_prompt: str
    dataset_id: str
    embedding_model: str
    chat_model: str
    temperature: float = 0.7
    top_k: int = 5


class ChatSessionCreate(BaseModel):
    """Request body for POST /chat/sessions."""

    agent_id: str = Field(min_length=1)
    title: str = Field(default="New conversation", max_length=255)


class ChatSessionResponse(BaseModel):
    """A conversation session."""

    session_id: str
    agent_id: str
    title: str
    message_count: int
    agent_config_snapshot: AgentConfigSnapshot
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionListResponse(BaseModel):
    """Paginated list of sessions."""

    sessions: list[ChatSessionResponse]
    total: int
    limit: int
    offset: int


# ──────────────────────────────────────────────────────────────────────────────
# Message schemas
# ──────────────────────────────────────────────────────────────────────────────


class SourceChunkPayload(BaseModel):
    """A source chunk, serialised from the RAG pipeline's SourceChunk."""

    chunk_id: str
    text: str
    score: float
    source: str | None = None
    cosine_score: float = 0.0
    sparse_score: float = 0.0
    search_type: str = "dense"


class ChatMessageCreate(BaseModel):
    """
    Request body for POST /chat/sessions/{session_id}/messages.
    Mirrors ChatRequest but includes optional session_id for backward compat.
    """

    question: str = Field(min_length=1)
    session_id: str | None = Field(
        default=None,
        description="Optional. If omitted, a new session is created for this agent.",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    use_reranker: bool = Field(default=False)
    reranker_model: str | None = None
    use_hybrid: bool = Field(default=False)


class ChatMessageResponse(BaseModel):
    """A stored chat message."""

    message_id: str
    session_id: str
    role: MessageRole
    content: str
    token_count: int | None = None
    sources: list[SourceChunkPayload] | None = None
    feedback: ChatFeedbackResponse | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionWithMessages(BaseModel):
    """A session with its full message history."""

    session: ChatSessionResponse
    messages: list[ChatMessageResponse]


# ──────────────────────────────────────────────────────────────────────────────
# Streaming response
# ──────────────────────────────────────────────────────────────────────────────


class ChatStreamResponse(BaseModel):
    """
    NDJSON event emitted by POST /chat/sessions/{session_id}/messages/stream.
    Compatible with the existing SSE shape emitted by /chat/{agent_id}/stream.
    """

    event: str = Field(description="'chunk', 'end', or 'error'")
    content: str | None = None
    sources: list[SourceChunkPayload] | None = None
    reranker_model: str | None = None
    use_hybrid: bool = False
    error: str | None = None
    message_id: str | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Feedback schemas
# ──────────────────────────────────────────────────────────────────────────────


class ChatFeedbackUpsert(BaseModel):
    """Request body for POST /chat/messages/{message_id}/feedback."""

    rating: FeedbackRating
    comment: str | None = Field(default=None, max_length=2000)


class ChatFeedbackResponse(BaseModel):
    """A stored feedback entry."""

    feedback_id: str
    message_id: str
    rating: FeedbackRating
    comment: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}
