from __future__ import annotations

from backend.config import settings
from backend.services.ingestion_service import IngestionService
from rag.embedding.ollama_embedder import OllamaEmbedder
from rag.embedding.sparse_embedder import SparseEmbedder
from rag.generation.ollama_chat import OllamaChat
from rag.retrieval.rag_pipeline import RAGPipeline
from rag.storage.agent_store import AgentStore
from rag.storage.qdrant_store import QdrantStore

# Singleton instances — lazily initialized
_qdrant_store: QdrantStore | None = None
_embedder: OllamaEmbedder | None = None
_sparse_embedder: SparseEmbedder | None = None
_chat_client: OllamaChat | None = None
_agent_store: AgentStore | None = None
_rag_pipeline: RAGPipeline | None = None
_ingestion_service: IngestionService | None = None

# Current active model names (set via /ollama/validate or /agents endpoints)
# Use :latest suffix so bare-name lookups always match Ollama's registry format
_active_embedding_model: str = "nomic-embed-text:latest"
_active_chat_model: str = "llama3.2:latest"


def get_qdrant_store() -> QdrantStore:
    global _qdrant_store
    if _qdrant_store is None:
        _qdrant_store = QdrantStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    return _qdrant_store


def get_embedder(model: str | None = None) -> OllamaEmbedder:
    global _embedder, _active_embedding_model
    target = model or _active_embedding_model
    if _embedder is None or _embedder.model != target:
        _embedder = OllamaEmbedder(base_url=settings.ollama_base_url, model=target)
        _active_embedding_model = target
    return _embedder


def get_sparse_embedder() -> SparseEmbedder:
    """Lazily create a shared SparseEmbedder instance (fastembed)."""
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = SparseEmbedder()
    return _sparse_embedder


def get_chat_client(model: str | None = None) -> OllamaChat:
    global _chat_client, _active_chat_model
    target = model or _active_chat_model
    if _chat_client is None or _chat_client.model != target:
        _chat_client = OllamaChat(base_url=settings.ollama_base_url, model=target)
        _active_chat_model = target
    return _chat_client


def get_agent_store() -> AgentStore:
    global _agent_store
    if _agent_store is None:
        _agent_store = AgentStore(registry_path=settings.data_root / "agents" / "registry.json")
    return _agent_store


def get_rag_pipeline(
    embedding_model: str | None = None,
    chat_model: str | None = None,
    sparse_embedder: SparseEmbedder | None = None,
) -> RAGPipeline:
    return RAGPipeline(
        qdrant_store=get_qdrant_store(),
        embedder=get_embedder(embedding_model),
        chat_model=get_chat_client(chat_model),
        sparse_embedder=sparse_embedder or get_sparse_embedder(),
    )


def set_active_models(embedding_model: str, chat_model: str) -> None:
    global _active_embedding_model, _active_chat_model, _embedder, _chat_client
    _active_embedding_model = embedding_model
    _active_chat_model = chat_model
    # Reset clients so next call re-initializes with new model
    _embedder = None
    _chat_client = None


def get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService(
            qdrant_store=get_qdrant_store(),
            embedder=get_embedder(),
            sparse_embedder=get_sparse_embedder(),
        )
    return _ingestion_service


async def close_qdrant_store() -> None:
    """Close the QdrantStore HTTP client. Call on application shutdown."""
    global _qdrant_store
    if _qdrant_store is not None:
        await _qdrant_store.close()
        _qdrant_store = None
