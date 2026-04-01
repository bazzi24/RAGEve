from __future__ import annotations

from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from backend.api.routes._limiter import limiter
from backend.schemas.chat import ChatRequest, ChatResponse, SourceChunkSchema
from backend.services.ingestion_factory import get_agent_store, get_rag_pipeline


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/{agent_id}", response_model=ChatResponse)
@limiter.limit("120/minute")
async def chat(
    request: Request,
    agent_id: str,
    payload: ChatRequest,
) -> ChatResponse:
    """
    Non-streaming RAG chat endpoint.
    """
    store = get_agent_store()
    agent = store.get(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    rag = get_rag_pipeline(
        embedding_model=agent.config.embedding_model,
        chat_model=agent.config.chat_model,
    )

    answer = await rag.query(
        collection_name=agent.config.dataset_id,
        question=payload.question,
        system_prompt=agent.config.system_prompt,
        top_k=payload.top_k or agent.config.top_k,
        score_threshold=payload.score_threshold or 0.0,
        temperature=payload.temperature or agent.config.temperature,
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
            )
            for s in answer.sources
        ],
        metadata={
            **answer.metadata,
            "use_hybrid": answer.metadata.get("use_hybrid", False),
        },
    )


async def _stream_rag(
    agent_id: str,
    payload: ChatRequest,
) -> AsyncIterator[str]:
    import asyncio
    import json as _json

    store = get_agent_store()
    agent = store.get(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    rag = get_rag_pipeline(
        embedding_model=agent.config.embedding_model,
        chat_model=agent.config.chat_model,
    )

    sources_list: list[dict] = []
    reranker_model: str | None = None
    use_hybrid: bool = False
    done_emitted = False

    try:
        async with asyncio.timeout(120):
            async for token in rag.query_stream(
                collection_name=agent.config.dataset_id,
                question=payload.question,
                system_prompt=agent.config.system_prompt,
                top_k=payload.top_k or agent.config.top_k,
                score_threshold=payload.score_threshold or 0.0,
                temperature=payload.temperature or agent.config.temperature,
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
                        yield _json.dumps({
                            "event": "end",
                            "sources": sources_list,
                            "reranker_model": reranker_model,
                            "use_hybrid": use_hybrid,
                        }) + "\n"
                    continue

                yield _json.dumps({"event": "chunk", "content": token}) + "\n"

            if not done_emitted:
                yield _json.dumps({
                    "event": "end",
                    "sources": sources_list,
                    "reranker_model": reranker_model,
                    "use_hybrid": use_hybrid,
                }) + "\n"
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out after 120 seconds")


@router.post("/{agent_id}/stream")
@limiter.limit("120/minute")
async def chat_stream(request: Request, agent_id: str, payload: ChatRequest) -> StreamingResponse:
    """
    Streaming RAG chat endpoint using SSE.
    Each chunk is sent as: {"event": "chunk", "content": "..."}
    Final event: {"event": "end", "sources": [...]}
    """
    store = get_agent_store()
    agent = store.get(agent_id)

    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    return StreamingResponse(
        _stream_rag(agent_id, payload),
        media_type="application/x-ndjson",
    )
