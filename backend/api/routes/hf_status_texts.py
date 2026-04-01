"""
HuggingFace Hub status-texts routes — GET /status-texts/{dataset_id}.

Provides:
  get_hf_status_texts() — human-readable status + message from the download registry
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.api.routes import hf_status
from backend.schemas.huggingface import HuggingFaceStatusTextsResponse

router = APIRouter(tags=["huggingface-status"])


@router.get("/status-texts/{dataset_id:path}", response_model=HuggingFaceStatusTextsResponse)
async def get_hf_status_texts(dataset_id: str) -> HuggingFaceStatusTextsResponse:
    """Return human-readable status + message for a dataset."""
    status = hf_status._hf_download_status.get(dataset_id)
    if not status:
        return HuggingFaceStatusTextsResponse(
            dataset_id=dataset_id,
            display_status="not_found",
            display_message="Dataset not found. Enter a valid HuggingFace dataset ID to preview.",
        )

    s = status.get("status", "unknown")
    if s == "queued":
        display_status = "queued"
        display_message = "Waiting to download…"
    elif s == "downloading":
        display_status = "downloading"
        display_message = status.get("message", "Downloading…")
    elif s == "completed":
        if status.get("ingested"):
            display_status = "ready"
            display_message = "Downloaded & indexed — ready for RAG chat"
        else:
            display_status = "downloaded"
            display_message = "Downloaded — ready to ingest into Qdrant"
    elif s == "failed":
        display_status = "error"
        display_message = status.get("error") or status.get("message", "Download failed")
    elif s == "cancelled":
        display_status = "cancelled"
        display_message = "Download cancelled"
    else:
        display_status = "unknown"
        display_message = status.get("message", "")

    return HuggingFaceStatusTextsResponse(
        dataset_id=dataset_id,
        display_status=display_status,
        display_message=display_message,
    )
