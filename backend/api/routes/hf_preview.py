"""
HuggingFace Hub preview routes — GET /preview/{dataset_id} and GET /instructions/{dataset_id}.

Provides:
  preview_hf_dataset()  — full dataset preview (configs, splits, columns, card metadata)
  get_hf_download_instructions() — static download instructions + local path info
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.schemas.huggingface import (
    HuggingFaceInstructionsResponse,
    HuggingFacePreviewResponse,
)
from backend.api.routes.hf_metadata import _fetch_hf_card_metadata, _fetch_hf_readme_html

_log = logging.getLogger("app")

router = APIRouter(tags=["huggingface-preview"])


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fmt_size(n: int | None) -> str:
    if n is None:
        return "—"
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


# ── Routes ───────────────────────────────────────────────────────────────────


@router.get("/preview/{dataset_id:path}", response_model=HuggingFacePreviewResponse)
async def preview_hf_dataset(dataset_id: str) -> HuggingFacePreviewResponse:
    """
    Return a full preview of a HuggingFace dataset: configs, splits, columns,
    description, downloads, likes, and rich card metadata (tags, language, license, etc).

    Falls back to Hub API if datasets-server preview is unavailable.
    """
    import httpx

    DS_SERVER = "https://datasets-server.huggingface.co"
    HF_HUB_API = "https://huggingface.co/api"

    dataset_id = dataset_id.strip()

    # 1. Try datasets-server preview endpoint
    preview_data = None
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            preview_resp = await client.get(
                f"{DS_SERVER}/preview",
                params={"dataset": dataset_id},
            )
            if preview_resp.status_code == 200:
                preview_data = preview_resp.json()
    except Exception as exc:  # noqa: BLE001
        _log.warning("datasets-server preview failed for '%s': %s", dataset_id, exc)

    # 2. Fetch Hub API for size / description / downloads / likes / splits
    description: str | None = None
    estimated_bytes: int | None = None
    downloads: int | None = None
    likes: int | None = None

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            hub_resp = await client.get(f"{HF_HUB_API}/datasets/{dataset_id}")
            if hub_resp.status_code == 200:
                hub_data = hub_resp.json()
                description = hub_data.get("description")
                if isinstance(description, dict):
                    description = description.get("en")
                if not description:
                    description = hub_data.get("annotations", {}).get("default", {}).get("en")
                estimated_bytes = hub_data.get("size_bytes")
                downloads = hub_data.get("downloads")
                likes = hub_data.get("likes")
    except Exception as exc:  # noqa: BLE001
        _log.warning("HF Hub API fetch failed for '%s': %s", dataset_id, exc)

    # 3. Discover configs + splits + columns via `datasets` library (no download)
    columns: dict[str, str] = {}
    detected_splits: list[str] = []
    detected_configs: list[str] = []
    default_config: str | None = None

    try:
        from datasets import get_dataset_config_names  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "The 'datasets' package is required for dataset previews. "
            "Install it with: uv add datasets"
        )

    try:
        detected_configs = list(get_dataset_config_names(dataset_id, use_auth_token=settings.hf_token))
    except Exception as exc:  # noqa: BLE001
        _log.warning("get_dataset_config_names failed for '%s': %s", dataset_id, exc)
        detected_configs = []

    if detected_configs:
        default_config = detected_configs[0]

        for cfg in detected_configs:
            try:
                from datasets import get_dataset_split_names, load_dataset  # type: ignore[import-untyped]

                detected_splits = list(
                    get_dataset_split_names(dataset_id, config_name=cfg, use_auth_token=settings.hf_token)
                )
                if detected_splits:
                    sample_ds = load_dataset(
                        dataset_id,
                        name=cfg,
                        split=detected_splits[0],
                        streaming=True,
                        use_auth_token=settings.hf_token,
                    )
                    for idx, row in enumerate(sample_ds):
                        if idx >= 5:
                            break
                        for k, v in row.items():
                            if k not in columns and isinstance(v, dict):
                                continue
                            if k not in columns:
                                if isinstance(v, str):
                                    columns[k] = "string"
                                elif isinstance(v, (int, float)):
                                    columns[k] = "numeric"
                                elif isinstance(v, bool):
                                    columns[k] = "bool"
                                else:
                                    columns[k] = type(v).__name__
                    break
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "Failed to inspect config '%s' of '%s': %s", cfg, dataset_id, exc
                )
                continue
    else:
        try:
            from datasets import get_dataset_split_names  # type: ignore[import-untyped]

            detected_splits = list(
                get_dataset_split_names(dataset_id, use_auth_token=settings.hf_token)
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("get_dataset_split_names failed for '%s': %s", dataset_id, exc)
            detected_splits = []

    # 4. Fetch full card metadata via huggingface_hub
    card_meta: dict[str, Any] = {}
    try:
        card_meta = _fetch_hf_card_metadata(dataset_id, hf_token=settings.hf_token)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to fetch card metadata for '%s': %s", dataset_id, exc)
        card_meta = {}

    # 5. Fetch README HTML
    readme_html: str | None = None
    try:
        readme_html = _fetch_hf_readme_html(dataset_id)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to fetch README for '%s': %s", dataset_id, exc)

    # 6. Detect source
    source = "datasets-server"
    if card_meta.get("card_data") or preview_data is None:
        source = "hf-hub"

    # 7. Resolve default config if not set
    if not default_config and detected_configs:
        default_config = detected_configs[0]

    return HuggingFacePreviewResponse(
        dataset_id=dataset_id,
        full_dataset_id=dataset_id,
        configs=detected_configs,
        default_config=default_config,
        description=description,
        downloads=downloads,
        likes=likes,
        estimated_size_bytes=estimated_bytes,
        estimated_size_human=_fmt_size(estimated_bytes) if estimated_bytes else None,
        splits=detected_splits or (list(preview_data.get("splits", {}).keys()) if preview_data else []),
        columns=columns or (preview_data.get("features", {}) if preview_data else {}),
        source=source,
        tags=card_meta.get("tags", []),
        language=card_meta.get("language", []),
        license=card_meta.get("license"),
        paper_url=card_meta.get("paper_url"),
        card_data=card_meta.get("card_data"),
        readme_html=readme_html,
        leaderboard=card_meta.get("leaderboard"),
        source_detail="HuggingFace Hub" if source == "hf-hub" else "datasets-server",
        message="Dataset preview loaded successfully." if (detected_configs or detected_splits or columns) else "Dataset metadata loaded. Full column list unavailable.",
        valid=True,
    )


@router.get("/instructions/{dataset_id:path}", response_model=HuggingFaceInstructionsResponse)
async def get_hf_download_instructions(dataset_id: str) -> HuggingFaceInstructionsResponse:
    """Return download command + expected local path info."""
    safe_id = dataset_id.replace("/", "__")
    local_path = str(settings.data_root / "hf" / safe_id)

    return HuggingFaceInstructionsResponse(
        dataset_id=dataset_id,
        download_command=f"python -c \"from datasets import load_dataset; ds = load_dataset('{dataset_id}')\"",
        expected_local_path=local_path,
        supported_splits=[],
        supported_file_formats=[".parquet", ".json", ".jsonl", ".csv"],
        column_preview={},
        estimated_size=None,
        message=f"Use the download form to pull this dataset to {local_path}",
    )
