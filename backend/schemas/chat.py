from pydantic import BaseModel, Field


class SourceChunkSchema(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str | None = None
    # Dense bi-encoder cosine score (preserved through reranking)
    cosine_score: float = 0.0
    # Sparse retrieval score (0.0 when dense-only)
    sparse_score: float = 0.0
    # Search mode used: "dense" | "hybrid"
    search_type: str = "dense"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    stream: bool = Field(default=True)
    # Reranker options — both must be set together to activate reranking.
    use_reranker: bool = Field(default=False, description="Enable cross-encoder reranking")
    reranker_model: str | None = Field(
        default=None,
        description="Cross-encoder model ID (e.g. 'BAAI/bge-reranker-base'). "
        "Required when use_reranker is True.",
    )
    # Hybrid search — uses both dense (Ollama) and sparse (fastembed/Splade++) retrieval
    use_hybrid: bool = Field(
        default=False,
        description="Enable hybrid search: combines dense + sparse retrieval with RRF fusion. "
        "Requires the collection to have been ingested with sparse vectors enabled.",
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunkSchema] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
