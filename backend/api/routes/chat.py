from __future__ import annotations

import asyncio
import json as _json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.api.dependencies import get_current_user
from backend.api.routes._limiter import limiter
from backend.api.routes.dialogs import _ensure_tenant_access
from backend.models_peewee import User
from backend.schemas.chat import ChatRequest, ChatResponse, SourceChunkSchema
from backend.services.database import run_db_operation
from backend.services.dialog_store import get_dialog_store
from backend.services.ingestion_factory import get_rag_pipeline
from backend.services.tenant_user_store import get_tenant_user_store

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


async def _get_authorized_dialog(dialog_id: str, user: User):
    dialog_store = get_dialog_store()
    dialog = await run_db_operation(dialog_store.get_dialog, dialog_id)
    if not dialog:
        raise HTTPException(status_code=404, detail=f"Dialog '{dialog_id}' not found")

    await _ensure_tenant_access(user, dialog.tenant_id)
    return dialog


@router.post("/{dialog_id}", response_model=ChatResponse)
@limiter.limit("120/minute")
async def chat(
    dialog_id: str,
    payload: ChatRequest,
    request: Request,  # noqa: F841
    user: User = Depends(get_current_user),
) -> ChatResponse:
    """
    Non-streaming RAG chat endpoint using a dialog (agent).
    """
    dialog = await _get_authorized_dialog(dialog_id, user)

    # Fetch tenant for embedding model
    tenant_store = get_tenant_user_store()
    tenant = await run_db_operation(tenant_store.get_tenant, dialog.tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=404, detail=f"Tenant '{dialog.tenant_id}' not found"
        )

    # Determine collection (knowledge base) - use first kb_id
    if not dialog.kb_ids:
        raise HTTPException(
            status_code=400, detail="Dialog has no knowledge base assigned"
        )
    collection_name = dialog.kb_ids[0]

    # Build system prompt
    system_prompt = ""
    if dialog.prompt_config:
        system_prompt = dialog.prompt_config.get("system", "")

    # LLM settings
    temperature = (
        payload.temperature
        if payload.temperature is not None
        else (dialog.llm_setting.get("temperature") if dialog.llm_setting else 0.7)
    )
    top_k = payload.top_k if payload.top_k is not None else dialog.top_k

    # RAG pipeline
    rag = get_rag_pipeline(
        embedding_model=tenant.embd_id,  # Tenant's embedding model
        chat_model=dialog.llm_id,
    )

    answer = await rag.query(
        collection_name=collection_name,
        question=payload.question,
        system_prompt=system_prompt,
        top_k=top_k,
        score_threshold=payload.score_threshold or 0.0,
        temperature=temperature,
        use_reranker=payload.use_reranker,
        reranker_model=payload.reranker_model,
        use_hybrid=payload.use_hybrid,
    )

    return ChatResponse(
        answer=answer.answer,
        sources=[
            SourceChunkSchema(
                chunk_id=s.chunk_id,
                text=s.text,
                score=s.score,
                source=s.source,
                cosine_score=s.cosine_score,
                sparse_score=s.sparse_score,
                search_type=s.search_type,
                pages=s.pages,
                blocks=s.blocks,
                datasetId=s.datasetId,
            )
            for s in answer.sources
        ],
        metadata={
            **answer.metadata,
            "use_hybrid": answer.metadata.get("use_hybrid", False),
        },
    )


async def _stream_rag(
    dialog,
    payload: ChatRequest,
) -> AsyncIterator[str]:
    tenant_store = get_tenant_user_store()
    tenant = await run_db_operation(tenant_store.get_tenant, dialog.tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=404, detail=f"Tenant '{dialog.tenant_id}' not found"
        )

    if not dialog.kb_ids:
        raise HTTPException(
            status_code=400, detail="Dialog has no knowledge base assigned"
        )
    collection_name = dialog.kb_ids[0]

    system_prompt = (
        dialog.prompt_config.get("system", "") if dialog.prompt_config else ""
    )

    temperature = (
        payload.temperature
        if payload.temperature is not None
        else (dialog.llm_setting.get("temperature") if dialog.llm_setting else 0.7)
    )
    top_k = payload.top_k if payload.top_k is not None else dialog.top_k

    rag = get_rag_pipeline(
        embedding_model=tenant.embd_id,
        chat_model=dialog.llm_id,
    )

    sources_list: list[dict] = []
    reranker_model: str | None = None
    use_hybrid: bool = False
    done_emitted = False

    try:
        async with asyncio.timeout(120):
            async for token in rag.query_stream(
                collection_name=collection_name,
                question=payload.question,
                system_prompt=system_prompt,
                top_k=top_k,
                score_threshold=payload.score_threshold or 0.0,
                temperature=temperature,
                use_reranker=payload.use_reranker,
                reranker_model=payload.reranker_model,
                use_hybrid=payload.use_hybrid,
            ):
                if isinstance(token, dict):
                    if "done" in token:
                        done_emitted = True
                        sources_list = token.get("sources", [])
                        reranker_model = token.get("reranker_model")
                        use_hybrid = token.get("use_hybrid", False)
                        payload = {
                            "event": "end",
                            "sources": sources_list,
                            "reranker_model": reranker_model,
                            "use_hybrid": use_hybrid,
                        }
                        yield f"data: {_json.dumps(payload)}\n\n"
                    continue

                yield f"data: {_json.dumps({'event': 'chunk', 'content': token})}\n\n"

            if not done_emitted:
                payload = {
                    "event": "end",
                    "sources": sources_list,
                    "reranker_model": reranker_model,
                    "use_hybrid": use_hybrid,
                }
                yield f"data: {_json.dumps(payload)}\n\n"
    except asyncio.TimeoutError:
        _log.warning("Streaming RAG timeout for dialog %s", dialog.id)
        yield f"data: {_json.dumps({'event': 'error', 'error': 'Request timed out after 120 seconds'})}\n\n"
        return
    except Exception as e:
        _log.exception("Streaming RAG failed for dialog %s", dialog.id)
        yield f"data: {_json.dumps({'event': 'error', 'error': 'An internal error occurred'})}\n\n"
        return


@router.post("/{dialog_id}/stream")
@limiter.limit("120/minute")
async def chat_stream(
    dialog_id: str,
    payload: ChatRequest,
    request: Request,  # noqa: F841
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Streaming RAG chat endpoint using SSE.
    Each chunk is sent as: {"event": "chunk", "content": "..."}
    Final event: {"event": "end", "sources": [...]}
    """
    dialog = await _get_authorized_dialog(dialog_id, user)
    return StreamingResponse(
        _stream_rag(dialog, payload),
        media_type="text/event-stream",
    )
