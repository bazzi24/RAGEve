"""
Thin re-export router for HuggingFace endpoints.

All logic lives in focused sub-modules:
  hf_preview.py       — GET /preview/{dataset_id}, GET /instructions/{dataset_id}
  hf_search.py         — GET /search
  hf_status_texts.py   — GET /status-texts/{dataset_id}
  hf_download.py       — download submit/cancel/status + background worker
  hf_ingest.py         — ingest submit/cancel/status + background worker
  hf_status.py         — shared status registries and persistence
  hf_metadata.py       — card data and README helpers (no routes)

Keeping routes in this file makes the FastAPI include_router call simpler
and keeps all HF prefixes under /datasets/hf.
"""
from fastapi import APIRouter

from backend.api.routes._limiter import limiter  # noqa: F401 — re-exported for other modules
from backend.api.routes.hf_download import router as hf_download_router
from backend.api.routes.hf_ingest import router as hf_ingest_router
from backend.api.routes.hf_preview import router as hf_preview_router
from backend.api.routes.hf_search import router as hf_search_router
from backend.api.routes.hf_status import (  # noqa: F401  — exported for use by other modules
    _hf_download_status,
    _hf_ingest_registry,
)
from backend.api.routes.hf_status_texts import router as hf_status_texts_router

router = APIRouter(prefix="/datasets/hf", tags=["huggingface"])

# Mount sub-routers (each carries its own path prefix)
router.include_router(hf_preview_router)
router.include_router(hf_search_router)
router.include_router(hf_status_texts_router)
router.include_router(hf_download_router)
router.include_router(hf_ingest_router)
