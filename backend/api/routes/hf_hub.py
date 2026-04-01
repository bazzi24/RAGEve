"""
DEPRECATED — all routes moved to focused sub-modules.

Routes:
  hf_preview.py      — GET /preview/{dataset_id}, GET /instructions/{dataset_id}
  hf_search.py      — GET /search
  hf_status_texts.py — GET /status-texts/{dataset_id}

Import the sub-modules directly. This file will be removed in a future version.
"""
from backend.api.routes.hf_preview import router as hf_hub_router  # noqa: F401

__all__ = ["hf_hub_router"]
