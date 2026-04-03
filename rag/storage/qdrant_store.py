from __future__ import annotations

import logging
import uuid
import warnings
from dataclasses import dataclass, field
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

# qdrant-client warns when api_key is passed over HTTP (localhost dev = fine, traffic never leaves the machine).
warnings.filterwarnings(
    "ignore",
    message=r"Api key is used with an insecure connection",
    category=UserWarning,
)

_log = logging.getLogger("rag.storage.qdrant_store")


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

DENSE_VECTOR_NAME = "dense"   # Named vector field for Ollama dense embeddings
SPARSE_VECTOR_NAME = "sparse"  # Named vector field for fastembed sparse vectors

# nomic-embed-text outputs 768-dimensional dense vectors
DEFAULT_DENSE_SIZE = 768

# Qdrant HNSW tuning
DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCT = 256

COLLECTION_VECTOR_CONFIG: dict[str, Any] = {
    "vectors": {
        DENSE_VECTOR_NAME: {
            "size": DEFAULT_DENSE_SIZE,
            "distance": "Cosine",
        },
        SPARSE_VECTOR_NAME: {
            "size": 1,  # Sparse vectors use a sparse index, not dense dimension
            "distance": "Dot",
        },
    }
}


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------


@dataclass
class ChunkRecord:
    chunk_id: str
    chunk_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Dense vector from Ollama embedder (list[float], 768-dim)
    dense_vector: list[float] | None = None
    # Sparse vector from fastembed (dict with indices + values)
    sparse_vector: dict[str, Any] | None = None


@dataclass
class SearchResult:
    chunk_id: str
    chunk_text: str
    score: float
    metadata: dict[str, Any]
    # Individual retrieval scores — useful for debugging RRF
    dense_score: float = 0.0
    sparse_score: float = 0.0


# ----------------------------------------------------------------------
# Schema cache — avoids checking collection type on every search
# Key: collection_name, Value: "named" | "unnamed"
# ----------------------------------------------------------------------
_collection_schema_cache: dict[str, str] = {}


class QdrantStore:
    def __init__(
        self,
        url: str,
        timeout: float = 30.0,
        *,
        api_key: str | None = None,
    ) -> None:
        self.base_url = url.rstrip("/")
        self._api_key = api_key
        # Shared httpx AsyncClient with optional API-key header for Qdrant's auth.
        headers = {"api-key": api_key} if api_key else {}
        self._http = httpx.AsyncClient(base_url=url, timeout=timeout, headers=headers)
        # Sync Qdrant SDK client (used for upsert/collection management).
        self.client = QdrantClient(url=url, timeout=timeout, api_key=api_key)

    async def close(self) -> None:
        """Close the async HTTP client. Call this on application shutdown."""
        await self._http.aclose()
        _log.debug("QdrantStore HTTP client closed")

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(
        self,
        collection_name: str,
        dense_size: int = DEFAULT_DENSE_SIZE,
    ) -> bool:
        """
        Create a Qdrant collection with named hybrid vectors.

        Creates two named vector indexes:
          - "dense"  : Ollama 768-dim dense vectors (cosine distance)
          - "sparse" : fastembed sparse vectors (dot product distance)

        Calling this on an existing collection is a no-op (Qdrant returns early).
        """
        try:
            self.client.get_collection(collection_name=collection_name)
            return True  # Already exists
        except (UnexpectedResponse, Exception):
            pass

        dense_config = qdrant_models.VectorParams(
            size=dense_size,
            distance=qdrant_models.Distance.COSINE,
            hnsw_config=qdrant_models.HnswConfigDiff(
                m=DEFAULT_HNSW_M,
                ef_construct=DEFAULT_HNSW_EF_CONSTRUCT,
            ),
        )
        sparse_config = qdrant_models.SparseVectorParams(
            index=qdrant_models.SparseIndexParams(on_disk=False)
        )

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: dense_config,
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: sparse_config,
            },
        )
        _log.info("Collection created: %s", collection_name)
        return True

    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.client.delete_collection(collection_name=collection_name)
            _log.info("Collection deleted: %s", collection_name)
            return True
        except Exception:
            _log.warning("Failed to delete collection: %s", collection_name)
            return False

    def collection_exists(self, collection_name: str) -> bool:
        try:
            self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def collection_has_points(self, collection_name: str) -> bool:
        """
        Return True if a collection exists and contains at least one point.

        This is used by HuggingFace UX status checks to determine whether a
        local dataset has already been ingested into Qdrant.
        """
        info = self.get_collection_info(collection_name)
        if info is None:
            return False
        points_count = info.get("points_count", 0) or 0
        return points_count > 0

    def is_dataset_ingested(self, dataset_id: str) -> bool:
        """Alias for collection_has_points(dataset_id) for readability."""
        return self.collection_has_points(dataset_id)

    def get_collection_info(self, collection_name: str) -> dict[str, Any] | None:
        try:
            info = self.client.get_collection(collection_name=collection_name)
            vectors_count = getattr(info, "vectors_count", None)
            if vectors_count is None:
                vectors_count = info.points_count or 0
            return {
                "name": collection_name,
                "vectors_count": vectors_count,
                "points_count": info.points_count or 0,
                "status": info.status,
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: list[ChunkRecord],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert a list of ChunkRecords into a collection.

        Automatically handles both old (unnamed single-vector) and new
        (Named Vectors with "dense"/"sparse" fields) collection schemas.

        Each point carries both named vectors:
            "dense"  = Ollama 768-dim (cosine-normalized list[float])
            "sparse" = fastembed {indices, values} (dict)

        For old unnamed collections, only the dense vector is stored.
        """
        if not chunks:
            return 0

        self._ensure_collection(collection_name)
        schema_type = self._get_schema_type(collection_name)

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            points = []
            for c in batch:
                payload = {
                    "chunk_id": c.chunk_id,
                    "text": c.chunk_text,
                    **c.metadata,
                }

                if schema_type == "unnamed":
                    # Old single-vector schema — flat list[float]
                    vector = c.dense_vector if c.dense_vector is not None else []
                else:
                    # Named-vector schema — dict with named fields
                    named_vectors: dict[str, Any] = {}
                    if c.dense_vector is not None:
                        named_vectors[DENSE_VECTOR_NAME] = c.dense_vector
                    if c.sparse_vector is not None:
                        named_vectors[SPARSE_VECTOR_NAME] = qdrant_models.SparseVector(
                            indices=c.sparse_vector.get("indices", []),
                            values=c.sparse_vector.get("values", []),
                        )
                    vector = named_vectors  # type: ignore[assignment]

                points.append(
                    qdrant_models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,  # type: ignore[arg-type]
                        payload=payload,
                    )
                )

            self.client.upsert(collection_name=collection_name, points=points)

        _log.debug("upsert_chunks: %s — %d points → batch %d/%d",
                   collection_name, len(chunks),
                   (len(chunks) + batch_size - 1) // batch_size,
                   (len(chunks) + batch_size - 1) // batch_size)

        return len(chunks)

    def _ensure_collection(self, collection_name: str) -> None:
        if not self.collection_exists(collection_name):
            self.create_collection(collection_name)

    # ------------------------------------------------------------------
    # Dense-only search (backward-compatible)
    # ------------------------------------------------------------------

    def _get_schema_type(self, collection_name: str) -> str:
        """Cache-and-detect whether a collection uses Named Vectors or old unnamed schema."""
        if collection_name in _collection_schema_cache:
            return _collection_schema_cache[collection_name]

        try:
            with httpx.Client(base_url=self.base_url, timeout=10.0) as client:
                r = client.get(f"/collections/{collection_name}")
                r.raise_for_status()
                vectors_cfg = r.json().get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
                if isinstance(vectors_cfg, dict) and DENSE_VECTOR_NAME in vectors_cfg:
                    _collection_schema_cache[collection_name] = "named"
                    return "named"
        except Exception:
            pass

        # Assume unnamed (old collections)
        _collection_schema_cache[collection_name] = "unnamed"
        return "unnamed"

    async def dense_search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Search using dense vectors.

        Automatically handles both old (unnamed single-vector) and new
        (Named Vectors with "dense" field) collection schemas.
        """
        schema_type = self._get_schema_type(collection_name)

        if schema_type == "named":
            payload: dict[str, Any] = {
                "vector": {
                    "name": DENSE_VECTOR_NAME,
                    "vector": query_vector,
                },
                "limit": top_k,
                "with_payload": True,
            }
        else:
            # Old collection: unnamed single-vector schema
            payload = {
                "vector": query_vector,
                "limit": top_k,
                "with_payload": True,
            }

        if score_threshold is not None:
            payload["score_threshold"] = score_threshold

        try:
            response = await self._http.post(
                f"/collections/{collection_name}/points/search",
                json=payload,
            )
            response.raise_for_status()
            hits: list[dict[str, Any]] = response.json().get("result", [])
        except Exception as exc:
            _log.error("dense_search: %s — HTTP error: %s", collection_name, exc)
            return []

        _log.debug("dense_search: %s — top_k=%d returned %d results",
                   collection_name, top_k, len(hits))
        return self._hits_to_search_results(hits)

    def _hits_to_search_results(
        self,
        hits: list[dict[str, Any]],
        default_score: float = 0.0,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id=hit.get("payload", {}).get("chunk_id", hit.get("id", "")),
                chunk_text=hit.get("payload", {}).get("text", ""),
                score=hit.get("score", default_score),
                dense_score=hit.get("score", default_score),
                sparse_score=default_score,
                metadata={
                    k: v
                    for k, v in hit.get("payload", {}).items()
                    if k not in ("chunk_id", "text")
                },
            )
            for hit in hits
        ]

    # ------------------------------------------------------------------
    # Hybrid search (dense + sparse with RRF)
    # ------------------------------------------------------------------

    async def hybrid_search(
        self,
        collection_name: str,
        dense_query: list[float],
        sparse_query: dict[str, Any],
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """
        Execute a hybrid search using Qdrant's prefetch + RRF.

        Falls back to dense-only search if the collection does not have
        sparse vectors (old collections with unnamed schema).

        RRF formula: score(r) = Σ  1 / (k + rank(r, i))
        where k=60 (standard constant), rank(r, i) = rank of result r in list i.

        This produces a unified ranking that favours chunks that rank highly
        in BOTH the dense and sparse result sets — exactly what you want for
        queries that contain specific keywords (sparse wins) AND have
        semantic intent (dense wins).

        Parameters
        ----------
        collection_name:
            Qdrant collection to search.
        dense_query:
            Ollama dense embedding of the query.
        sparse_query:
            fastembed sparse encoding of the query:
                {"indices": [int, ...], "values": [float, ...]}
        top_k:
            Final number of results to return after RRF fusion.
        rrf_k:
            RRF constant k. Standard value is 60. Higher = weaker
            bonus for appearing in both lists; lower = stronger synergy bonus.
            Range [1, 1000]. 60 is the Qdrant default.

        Returns
        -------
        list[SearchResult]
            Top-k chunks fused by RRF, sorted descending by RRF score.
        """
        # Fall back to dense-only if this collection predates the hybrid schema
        schema_type = self._get_schema_type(collection_name)
        if schema_type == "unnamed":
            _log.debug(
                "hybrid_search: %s — sparse vectors not found, falling back to dense-only",
                collection_name,
            )
            return self.dense_search(
                collection_name=collection_name,
                query_vector=dense_query,
                top_k=top_k,
            )

        payload: dict[str, Any] = {
            "prefetch": [
                # "vector" (not "query") is required in Qdrant 1.17+ prefetch items
                {"vector": {"name": DENSE_VECTOR_NAME, "vector": dense_query}, "limit": rrf_k},
                {"vector": sparse_query, "limit": rrf_k},
            ],
            "query": {"fusion": "rrf"},
            "limit": top_k,
            "with_payload": True,
        }

        try:
            response = await self._http.post(
                f"/collections/{collection_name}/points/query",
                json=payload,
            )
            response.raise_for_status()
            hits: list[dict[str, Any]] = response.json().get("result", {}).get("points", [])
        except Exception as exc:
            _log.error("hybrid_search: %s — HTTP error: %s", collection_name, exc)
            return []

        return self._hits_to_search_results(hits)

    # ------------------------------------------------------------------
    # Async search (kept for backward compat, dense-only)
    # ------------------------------------------------------------------

    async def async_search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        if not await self.async_collection_exists(collection_name):
            return []
        payload: dict[str, Any] = {
            "vector": {
                "name": DENSE_VECTOR_NAME,
                "vector": query_vector,
            },
            "limit": top_k,
            "with_payload": True,
        }
        if score_threshold is not None:
            payload["score_threshold"] = score_threshold
        response = await self._http.post(
            f"/collections/{collection_name}/points/search",
            json=payload,
        )
        response.raise_for_status()
        hits: list[dict[str, Any]] = response.json().get("result", [])
        return self._hits_to_search_results(hits)

    async def async_collection_exists(self, collection_name: str) -> bool:
        try:
            r = await self._http.get(f"/collections/{collection_name}")
            r.raise_for_status()
            return r.json().get("result", {}).get("status") in ("green", "yellow")
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Aliases for backward compatibility
    # ------------------------------------------------------------------

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Dense-only search — alias for dense_search()."""
        return self.dense_search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    def search_text(
        self,
        collection_name: str,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Alias for dense_search()."""
        return self.dense_search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_by_dataset(self, collection_name: str) -> bool:
        try:
            self.client.delete_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def delete_chunks(
        self,
        collection_name: str,
        chunk_ids: list[str],
    ) -> int:
        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchAny

            self.client.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="chunk_id",
                                match=MatchAny(any=chunk_ids),
                            ),
                        ],
                    ),
                ),
            )
            return len(chunk_ids)
        except Exception:
            return 0
