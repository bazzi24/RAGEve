"""
HuggingFace Hub search routes — GET /search.

Provides:
  search_hf_datasets() — search HuggingFace Hub for datasets by query
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

_log = logging.getLogger("app")

router = APIRouter(tags=["huggingface-search"])


@router.get("/search")
async def search_hf_datasets(q: str) -> list[dict[str, Any]]:
    """Search HuggingFace Hub for datasets matching the query."""
    import httpx

    if not q or len(q.strip()) < 2:
        return []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://huggingface.co/api/datasets",
                params={"search": q.strip(), "full": "true"},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            results: list[dict[str, Any]] = []
            for item in data[:10]:
                results.append({
                    "id": item.get("id", ""),
                    "private": item.get("private", False),
                    "downloads": item.get("downloads", 0),
                    "likes": item.get("likes", 0),
                    "tags": item.get("tags", [])[:5],
                })
            return results
    except Exception as exc:  # noqa: BLE001
        _log.warning("HF search failed for '%s': %s", q, exc)
        return []
