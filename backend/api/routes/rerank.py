from fastapi import APIRouter

from backend.schemas.rerank import RerankerEntrySchema, RerankersResponse
from rag.retrieval.cross_encoder_reranker import AVAILABLE_RERANKERS

router = APIRouter(prefix="/rerankers", tags=["rerankers"])


@router.get("/", response_model=RerankersResponse)
async def list_rerankers() -> RerankersResponse:
    """
    Return the list of locally available cross-encoder reranker models.

    Models are loaded from HuggingFace on first use (lazy initialisation).
    First call to a model may take 10–60 s depending on network speed.
    Subsequent calls reuse the cached model.

    The model is downloaded automatically by sentence-transformers when
    ``rag.retrieval.CrossEncoderReranker`` is first instantiated.
    """
    entries = [
        RerankerEntrySchema(
            id=m.id,
            display_name=m.display_name,
            description=m.description,
            approx_size_mb=m.approx_size_mb,
        )
        for m in AVAILABLE_RERANKERS
    ]
    return RerankersResponse(rerankers=entries, total=len(entries))
