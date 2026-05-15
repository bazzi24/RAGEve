"""
Conversation API routes using the new Conversation model.

Endpoints:
  - Conversations CRUD
  - Message appending
  - Streaming chat with conversation history
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from backend.api.dependencies import get_current_user
from backend.api.routes._limiter import limiter
from backend.models_peewee import User
from backend.schemas.conversations import (
    AppendMessageRequest,
    ConversationContextResponse,
    ConversationCreate,
    ConversationListResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageResponse,
)
from backend.services.conversation_store import get_conversation_store
from backend.services.database import run_db_operation
from backend.services.dialog_store import get_dialog_store
from backend.services.ingestion_factory import get_rag_pipeline
from backend.services.tenant_user_store import get_tenant_user_store
from backend.utils.log_sanitizer import sanitize_key

_log = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


async def _ensure_dialog_access(user: User, dialog_id: str):
    dialog_store = get_dialog_store()
    dialog = await run_db_operation(dialog_store.get_dialog, dialog_id)
    if not dialog:
        raise HTTPException(status_code=404, detail=f"Dialog '{dialog_id}' not found")

    if user.is_admin or dialog.tenant_id == user.id:
        return dialog

    tenant_store = get_tenant_user_store()
    role = await run_db_operation(
        tenant_store.get_user_role_in_tenant, user.id, dialog.tenant_id
    )
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this dialog",
        )
    return dialog


async def _ensure_conversation_access(user: User, conv) -> None:
    if user.is_admin:
        return
    if conv.user_id:
        if conv.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this conversation",
            )
        return
    # Legacy conversations may not have user_id: fall back to dialog tenant access.
    await _ensure_dialog_access(user, conv.dialog_id)


@router.post(
    "/", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED
)
@limiter.limit("60/minute")
async def create_conversation(
    payload: ConversationCreate,
    request: Request,  # noqa: F841
    user: User = Depends(get_current_user),
) -> ConversationResponse:
    """Create a new conversation for a dialog (agent)."""
    await _ensure_dialog_access(user, payload.dialog_id)

    store = get_conversation_store()
    conv = await run_db_operation(
        store.create_conversation,
        dialog_id=payload.dialog_id,
        name=payload.name,
        messages=payload.messages,
        reference=payload.reference,
        user_id=user.id,
    )
    conv_dict = conv.to_dict()
    return ConversationResponse(**conv_dict)


@router.get("/", response_model=ConversationListResponse)
@limiter.limit("120/minute")
async def list_conversations(
    request: Request,  # noqa: F841
    dialog_id: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
) -> ConversationListResponse:
    """List conversations, optionally filtered by dialog_id or user_id."""
    effective_user_id = user_id
    if not user.is_admin:
        effective_user_id = user.id
        if dialog_id:
            await _ensure_dialog_access(user, dialog_id)

    store = get_conversation_store()
    conversations, total = await run_db_operation(
        store.list_conversations,
        dialog_id=dialog_id,
        user_id=effective_user_id,
        limit=limit,
        offset=offset,
    )
    return ConversationListResponse(
        conversations=[ConversationResponse(**c) for c in conversations],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{conversation_id}", response_model=ConversationResponse)
@limiter.limit("120/minute")
async def get_conversation(
    conversation_id: str,
    request: Request,
    user: User = Depends(get_current_user),
) -> ConversationResponse:  # noqa: F841
    """Get a conversation by ID, including its message history."""
    store = get_conversation_store()
    conv = await run_db_operation(store.get_conversation, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=404, detail=f"Conversation '{conversation_id}' not found"
        )
    await _ensure_conversation_access(user, conv)
    conv_dict = conv.to_dict()
    return ConversationResponse(**conv_dict)


@router.put("/{conversation_id}", response_model=ConversationResponse)
@limiter.limit("60/minute")
async def update_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
    request: Request,  # noqa: F841
    user: User = Depends(get_current_user),
) -> ConversationResponse:
    """Update conversation metadata (name, reference)."""
    store = get_conversation_store()
    existing = await run_db_operation(store.get_conversation, conversation_id)
    if not existing:
        raise HTTPException(
            status_code=404, detail=f"Conversation '{conversation_id}' not found"
        )
    await _ensure_conversation_access(user, existing)

    updates = payload.dict(exclude_unset=True)
    conv = await run_db_operation(store.update_conversation, conversation_id, **updates)
    if not conv:
        raise HTTPException(
            status_code=404, detail=f"Conversation '{conversation_id}' not found"
        )
    conv_dict = conv.to_dict()
    return ConversationResponse(**conv_dict)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("60/minute")
async def delete_conversation(
    conversation_id: str,
    request: Request,
    user: User = Depends(get_current_user),
) -> None:  # noqa: F841
    """Delete a conversation and all its messages."""
    store = get_conversation_store()
    existing = await run_db_operation(store.get_conversation, conversation_id)
    if not existing:
        raise HTTPException(
            status_code=404, detail=f"Conversation '{conversation_id}' not found"
        )
    await _ensure_conversation_access(user, existing)

    deleted = await run_db_operation(store.delete_conversation, conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Conversation '{conversation_id}' not found"
        )


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
@limiter.limit("120/minute")
async def append_message(
    conversation_id: str,
    payload: AppendMessageRequest,
    request: Request,  # noqa: F841
    user: User = Depends(get_current_user),
) -> MessageResponse:
    """Append a message to the conversation."""
    store = get_conversation_store()
    conv = await run_db_operation(store.get_conversation, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=404, detail=f"Conversation '{conversation_id}' not found"
        )
    await _ensure_conversation_access(user, conv)

    msg = await run_db_operation(
        store.append_message,
        conversation_id,
        payload.role,
        payload.content,
        token_count=payload.token_count,
        sources=payload.sources,
    )
    if msg is None:
        raise HTTPException(
            status_code=404, detail=f"Conversation '{conversation_id}' not found"
        )
    return MessageResponse(**msg)


@router.get("/{conversation_id}/context")
@limiter.limit("120/minute")
async def get_conversation_context(
    request: Request,  # noqa: F841
    conversation_id: str,
    max_turns: int = Query(default=6, ge=1, le=20),
    user: User = Depends(get_current_user),
) -> ConversationContextResponse:
    """Get conversation history formatted for LLM context."""
    store = get_conversation_store()
    conv = await run_db_operation(store.get_conversation, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=404, detail=f"Conversation '{conversation_id}' not found"
        )
    await _ensure_conversation_access(user, conv)

    context = await run_db_operation(
        store.get_conversation_context, conversation_id, max_turns=max_turns
    )
    return ConversationContextResponse(
        messages=context, truncated=False
    )  # TODO: implement truncation flag


@router.post("/{conversation_id}/chat/stream")
@limiter.limit("60/minute")
async def chat_stream_with_conversation(
    conversation_id: str,
    request: Request,  # noqa: F841
    question: str = Query(..., description="User's question"),
    top_k: int = Query(default=5, ge=1, le=50),
    temperature: float = Query(default=0.7, ge=0.0, le=2.0),
    use_hybrid: bool = Query(default=True),
    use_reranker: bool = Query(default=False),
    reranker_model: str | None = Query(default=None),
    score_threshold: float = Query(default=0.0, ge=0.0, le=1.0),
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Streaming RAG chat with conversation history.

    NDJSON events:
      - {"event": "chunk", "content": "..."}
      - {"event": "end", "sources": [...], "message_id": "...", "elapsed_s": ...}
      - {"event": "error", "error": "...", "message_id": "..."}
    """
    import asyncio as _asyncio
    import json as _json

    # 1. Get conversation and dialog
    conv_store = get_conversation_store()
    dialog_store = get_dialog_store()

    conv = await run_db_operation(conv_store.get_conversation, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=404, detail=f"Conversation '{conversation_id}' not found"
        )
    await _ensure_conversation_access(user, conv)

    dialog = await run_db_operation(dialog_store.get_dialog, conv.dialog_id)
    if not dialog:
        raise HTTPException(
            status_code=404, detail=f"Dialog '{conv.dialog_id}' not found"
        )

    # Get tenant for embedding model
    tenant_store = get_tenant_user_store()
    tenant = await run_db_operation(tenant_store.get_tenant, dialog.tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=404, detail=f"Tenant '{dialog.tenant_id}' not found"
        )

    # 2. Append user message
    user_msg = await run_db_operation(
        conv_store.append_message, conversation_id, "user", question
    )

    # 3. Build context from conversation history
    # Use all messages up to this point (including just-added user message)
    history = await run_db_operation(
        conv_store.get_conversation_context, conversation_id, max_turns=6
    )
    # history is list of {"role": ..., "content": ...}

    # Build system prompt with history
    system_prompt_raw = (
        dialog.prompt_config.get("system", "") if dialog.prompt_config else ""
    )
    if history:
        history_block = (
            "## Conversation history\n"
            + "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
            + "\n\n"
        )
        system_prompt = (
            f"{system_prompt_raw}\n\n{history_block}"
            if system_prompt_raw
            else history_block
        )
    else:
        system_prompt = system_prompt_raw

    # 4. Prepare RAG pipeline
    rag = get_rag_pipeline(
        embedding_model=tenant.embd_id,
        chat_model=dialog.llm_id,
    )

    top_k_val = top_k or dialog.top_k
    temp_val = temperature or (
        dialog.llm_setting.get("temperature") if dialog.llm_setting else 0.7
    )

    # Use dialog's knowledge bases
    collection_names = (
        dialog.kb_ids or []
    )  # These are dataset IDs for Qdrant collections

    # For now, use first kb's collection. In future, search across multiple and merge.
    collection_name = collection_names[0] if collection_names else None
    if not collection_name:
        raise HTTPException(
            status_code=400, detail="Dialog has no knowledge base assigned"
        )

    # 5. Stream
    full_answer_parts: list[str] = []
    retrieved_sources: list[dict[str, Any]] = []
    done_emitted = False
    t0 = _time.monotonic()

    try:
        async with _asyncio.timeout(120):
            async for token in rag.query_stream(
                collection_name=collection_name,
                question=question,
                system_prompt=system_prompt,
                top_k=top_k_val,
                score_threshold=score_threshold,
                temperature=temp_val,
                use_hybrid=use_hybrid,
                use_reranker=use_reranker,
                reranker_model=reranker_model,
            ):
                if isinstance(token, dict):
                    done_emitted = True
                    retrieved_sources = token.get("sources", [])
                    reranker_model = token.get("reranker_model")
                    use_hybrid = token.get("use_hybrid", False)

                    elapsed = _time.monotonic() - t0
                    yield _json.dumps(
                        {
                            "event": "end",
                            "sources": retrieved_sources,
                            "reranker_model": reranker_model,
                            "use_hybrid": use_hybrid,
                            "message_id": user_msg.get(
                                "message_id"
                            ),  # Use actual message ID
                            "elapsed_s": round(elapsed, 2),
                        }
                    ) + "\n"
                else:
                    full_answer_parts.append(token)
                    yield _json.dumps({"event": "chunk", "content": token}) + "\n"

            if not done_emitted:
                yield _json.dumps(
                    {
                        "event": "end",
                        "sources": retrieved_sources,
                        "reranker_model": reranker_model,
                        "use_hybrid": use_hybrid,
                        "message_id": user_msg.get(
                            "message_id"
                        ),  # Use actual message ID
                    }
                ) + "\n"

    except (_asyncio.TimeoutError, TimeoutError) as exc:
        raise HTTPException(
            status_code=504, detail="Request timed out after 120 seconds"
        ) from exc
    except Exception as exc:
        _log.exception("Streaming chat failed for conversation %s", sanitize_key(conversation_id))
        yield _json.dumps(
            {
                "event": "error",
                "error": "An internal error occurred",
                "message_id": user_msg.get("message_id") if user_msg else None,
            }
        ) + "\n"
        return

    # 6. Save assistant message
    full_answer = "".join(full_answer_parts)
    await run_db_operation(
        conv_store.append_message,
        conversation_id,
        "assistant",
        full_answer,
        sources=[s for s in retrieved_sources],
    )
