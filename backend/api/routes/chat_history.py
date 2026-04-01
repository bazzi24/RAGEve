"""
Chat history API routes.

Sessions
  POST   /chat/sessions              — create a new session
  GET    /chat/sessions              — list sessions (filter by agent_id)
  GET    /chat/sessions/{session_id} — get session + message history
  DELETE /chat/sessions/{session_id} — delete a session

Messages
  POST   /chat/sessions/{session_id}/messages      — non-streaming RAG chat (with history)
  POST   /chat/sessions/{session_id}/messages/stream — streaming RAG chat (with history)

Feedback
  POST   /chat/messages/{message_id}/feedback       — thumbs up/down
"""

from __future__ import annotations

import json as _json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from backend.api.routes._limiter import limiter
from backend.schemas.chat_history import (
    AgentConfigSnapshot,
    ChatFeedbackResponse,
    ChatFeedbackUpsert,
    ChatMessageCreate,
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionListResponse,
    ChatSessionResponse,
    ChatSessionWithMessages,
    ChatStreamResponse,
    FeedbackRating,
    MessageRole,
    SourceChunkPayload,
)
from backend.services.chat_store import get_chat_store
from backend.services.ingestion_factory import get_agent_store, get_rag_pipeline

router = APIRouter(prefix="/chat", tags=["chat-history"])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _agent_config_to_snapshot(agent) -> dict[str, Any]:
    return {
        "system_prompt": agent.config.system_prompt,
        "dataset_id": agent.config.dataset_id,
        "embedding_model": agent.config.embedding_model,
        "chat_model": agent.config.chat_model,
        "temperature": agent.config.temperature,
        "top_k": agent.config.top_k,
    }


def _session_to_response(session) -> ChatSessionResponse:
    cfg = session.agent_config_snapshot or {}
    return ChatSessionResponse(
        session_id=session.session_id,
        agent_id=session.agent_id,
        title=session.title,
        message_count=session.message_count,
        agent_config_snapshot=AgentConfigSnapshot(
            system_prompt=cfg.get("system_prompt", ""),
            dataset_id=cfg.get("dataset_id", ""),
            embedding_model=cfg.get("embedding_model", ""),
            chat_model=cfg.get("chat_model", ""),
            temperature=cfg.get("temperature", 0.7),
            top_k=cfg.get("top_k", 5),
        ),
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _message_to_response(message, feedback=None) -> ChatMessageResponse:
    return ChatMessageResponse(
        message_id=message.message_id,
        session_id=message.session_id,
        role=MessageRole(message.role.value),
        content=message.content,
        token_count=message.token_count,
        sources=[
            SourceChunkPayload(**s) for s in (message.sources or [])
        ] if message.sources else None,
        feedback=feedback,
        created_at=message.created_at,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sessions
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("60/minute")
async def create_session(request: Request, payload: ChatSessionCreate):
    """
    Create a new conversation session for an agent.
    The agent's configuration is snapshotted at creation time.
    """
    store = get_chat_store()
    agent_store = get_agent_store()
    agent = agent_store.get(payload.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{payload.agent_id}' not found")

    session = await store.create_session(
        agent_id=payload.agent_id,
        agent_config=_agent_config_to_snapshot(agent),
        title=payload.title,
    )
    return _session_to_response(session)


@router.get("/sessions", response_model=ChatSessionListResponse)
@limiter.limit("120/minute")
async def list_sessions(
    request: Request,
    agent_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List sessions, optionally filtered by agent_id."""
    store = get_chat_store()
    sessions, total = await store.list_sessions(agent_id=agent_id, limit=limit, offset=offset)
    return ChatSessionListResponse(
        sessions=[_session_to_response(s) for s in sessions],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/sessions/{session_id}", response_model=ChatSessionWithMessages)
@limiter.limit("120/minute")
async def get_session_with_messages(request: Request, session_id: str):
    """Get a session and its full message history."""
    store = get_chat_store()
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    messages = await store.get_messages(session_id)

    # Load feedback in one batch query
    feedback_map: dict[str, ChatFeedbackResponse | None] = {}
    if messages:
        fb_rows = await store.get_feedback_for_messages([m.message_id for m in messages])
        for msg in messages:
            fb = fb_rows.get(msg.message_id)
            feedback_map[msg.message_id] = (
                ChatFeedbackResponse(
                    feedback_id=fb.feedback_id,
                    message_id=fb.message_id,
                    rating=FeedbackRating(fb.rating.value),
                    comment=fb.comment,
                    created_at=fb.created_at,
                )
                if fb else None
            )

    return ChatSessionWithMessages(
        session=_session_to_response(session),
        messages=[
            _message_to_response(m, feedback_map.get(m.message_id))
            for m in messages
        ],
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("60/minute")
async def delete_session(request: Request, session_id: str):
    """Delete a session and all its messages."""
    store = get_chat_store()
    deleted = await store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


# ──────────────────────────────────────────────────────────────────────────────
# Core streaming chat — history-aware
# ──────────────────────────────────────────────────────────────────────────────


async def _stream_with_history(
    session_id: str,
    payload: ChatMessageCreate,
) -> AsyncIterator[str]:
    """
    Yield NDJSON tokens for a streaming RAG chat turn with conversation history.

    Flow:
      1. Load conversation context from DB
      2. Save user message to DB
      3. Build system prompt with history
      4. Stream RAG answer, accumulating full text + sources
      5. Save assistant message to DB with retrieval_context + sources
    """
    import time as _time

    store = get_chat_store()
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    cfg = session.agent_config_snapshot or {}

    # 1. Conversation history (last 6 turns = 12 messages)
    history = await store.get_conversation_context(session_id, max_turns=6)

    # 2. Save user message
    user_msg = await store.create_message(
        session_id=session_id,
        role="user",
        content=payload.question,
    )

    # 3. Build conversation block for system prompt
    history_block = ""
    if history:
        history_block = (
            "## Conversation history\n"
            + "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in history
            )
            + "\n\n"
        )

    system_prompt_raw = cfg.get("system_prompt", "")
    if history_block:
        system_prompt = f"{system_prompt_raw}\n\n{history_block}" if system_prompt_raw else history_block
    else:
        system_prompt = system_prompt_raw

    # Update session title from first user message
    if session.message_count == 0:
        title = payload.question[:80].strip()
        await store.update_session_title(session_id, title)

    # 4. Build RAG pipeline
    rag = get_rag_pipeline(
        embedding_model=cfg.get("embedding_model"),
        chat_model=cfg.get("chat_model"),
    )

    top_k = payload.top_k or cfg.get("top_k", 5)
    temperature = payload.temperature or cfg.get("temperature", 0.7)

    # Accumulate full answer + sources across the stream
    full_answer_parts: list[str] = []
    retrieved_sources: list[dict[str, Any]] = []
    reranker_model: str | None = None
    use_hybrid: bool = payload.use_hybrid
    done_emitted = False
    t0 = _time.monotonic()

    import asyncio as _asyncio

    try:
        async with _asyncio.timeout(120):
            async for token in rag.query_stream(
                collection_name=cfg.get("dataset_id", ""),
                question=payload.question,
                system_prompt=system_prompt,
                top_k=top_k,
                score_threshold=payload.score_threshold or 0.0,
                temperature=temperature,
                use_hybrid=payload.use_hybrid,
                use_reranker=payload.use_reranker,
                reranker_model=payload.reranker_model,
            ):
                if isinstance(token, dict):
                    # End marker from the pipeline
                    done_emitted = True
                    retrieved_sources = token.get("sources", [])
                    reranker_model = token.get("reranker_model")
                    use_hybrid = token.get("use_hybrid", False)

                    elapsed = _time.monotonic() - t0
                    yield _json.dumps({
                        "event": "end",
                        "sources": retrieved_sources,
                        "reranker_model": reranker_model,
                        "use_hybrid": use_hybrid,
                        "message_id": user_msg.message_id,
                        "elapsed_s": round(elapsed, 2),
                    }) + "\n"
                else:
                    full_answer_parts.append(token)
                    yield _json.dumps({"event": "chunk", "content": token}) + "\n"

            if not done_emitted:
                yield _json.dumps({
                    "event": "end",
                    "sources": retrieved_sources,
                    "reranker_model": reranker_model,
                    "use_hybrid": use_hybrid,
                    "message_id": user_msg.message_id,
                }) + "\n"

    except (_asyncio.TimeoutError, TimeoutError) as exc:
        raise HTTPException(status_code=504, detail="Request timed out after 120 seconds") from exc
    except Exception as exc:
        yield _json.dumps({
            "event": "error",
            "error": str(exc),
            "message_id": user_msg.message_id,
        }) + "\n"
        return

    # 5. Save assistant message with retrieval context + sources
    full_answer = "".join(full_answer_parts)
    assistant_msg = await store.create_message(
        session_id=session_id,
        role="assistant",
        content=full_answer,
        sources=[s.__dict__ for s in retrieved_sources] if retrieved_sources else None,
    )

    # Touch updated_at without full row load
    try:
        await store.touch_session(session_id)
    except Exception:
        pass  # Non-critical


@router.post(
    "/sessions/{session_id}/messages/stream",
    summary="Streaming RAG chat with conversation history",
)
@limiter.limit("120/minute")
async def chat_stream_with_history(
    request: Request,
    session_id: str,
    payload: ChatMessageCreate,
) -> StreamingResponse:
    """
    Streaming RAG chat that is aware of prior conversation turns.
    Conversation history (up to 6 turns) is injected into the system prompt.

    NDJSON events:
      - {"event": "chunk",    "content": "..."}
      - {"event": "end",      "sources": [...], "message_id": "...", "elapsed_s": ...}
      - {"event": "error",    "error": "...", "message_id": "..."}
    """
    return StreamingResponse(
        _stream_with_history(session_id, payload),
        media_type="application/x-ndjson",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Non-streaming chat — history-aware
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
@limiter.limit("120/minute")
async def chat_non_streaming_with_history(
    request: Request,
    session_id: str,
    payload: ChatMessageCreate,
) -> ChatMessageResponse:
    """
    Non-streaming RAG chat that is aware of prior conversation turns.
    Conversation history (up to 6 turns) is injected into the system prompt.
    """
    store = get_chat_store()
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    cfg = session.agent_config_snapshot or {}

    # Conversation history
    history = await store.get_conversation_context(session_id, max_turns=6)

    history_block = ""
    if history:
        history_block = (
            "## Conversation history\n"
            + "\n".join(
                f"{m['role'].capitalize()}: {m['content']}"
                for m in history
            )
            + "\n\n"
        )

    system_prompt_raw = cfg.get("system_prompt", "")
    system_prompt = (
        f"{system_prompt_raw}\n\n{history_block}"
        if history_block or system_prompt_raw
        else ""
    )

    # Save user message
    user_msg = await store.create_message(
        session_id=session_id,
        role="user",
        content=payload.question,
    )

    # Update session title from first message
    if session.message_count == 0:
        await store.update_session_title(session_id, payload.question[:80].strip())

    # RAG answer
    rag = get_rag_pipeline(
        embedding_model=cfg.get("embedding_model"),
        chat_model=cfg.get("chat_model"),
    )

    answer = await rag.query(
        collection_name=cfg.get("dataset_id", ""),
        question=payload.question,
        system_prompt=system_prompt,
        top_k=payload.top_k or cfg.get("top_k", 5),
        score_threshold=payload.score_threshold or 0.0,
        temperature=payload.temperature or cfg.get("temperature", 0.7),
        use_hybrid=payload.use_hybrid,
        use_reranker=payload.use_reranker,
        reranker_model=payload.reranker_model,
    )

    # Save assistant message
    assistant_msg = await store.create_message(
        session_id=session_id,
        role="assistant",
        content=answer.answer,
        sources=[s.__dict__ for s in answer.sources] if answer.sources else None,
    )

    await store.touch_session(session_id)

    return ChatMessageResponse(
        message_id=assistant_msg.message_id,
        session_id=session_id,
        role=MessageRole.ASSISTANT,
        content=assistant_msg.content,
        sources=[SourceChunkPayload(**s.__dict__) for s in answer.sources] if answer.sources else None,
        created_at=assistant_msg.created_at,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Feedback
# ──────────────────────────────────────────────────────────────────────────────


@router.post(
    "/messages/{message_id}/feedback",
    response_model=ChatFeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit("60/minute")
async def upsert_feedback(request: Request, message_id: str, payload: ChatFeedbackUpsert):
    """
    Submit or update feedback for an assistant message.
    Idempotent — a second POST replaces the previous feedback.
    """
    store = get_chat_store()
    message = await store.get_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail=f"Message '{message_id}' not found")

    fb = await store.upsert_feedback(
        message_id=message_id,
        rating=payload.rating,
        comment=payload.comment,
    )
    return ChatFeedbackResponse(
        feedback_id=fb.feedback_id,
        message_id=fb.message_id,
        rating=FeedbackRating(fb.rating.value),
        comment=fb.comment,
        created_at=fb.created_at,
    )
