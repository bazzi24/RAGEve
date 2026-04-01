"""
HuggingFace download routes + background worker.

Routes:
  POST /download — submit a dataset download
  GET /download/{dataset_id}/status — get download progress
  POST /download/{dataset_id}/cancel — cancel an in-progress download
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from backend.api.routes._limiter import limiter
from backend.config import settings
from backend.schemas.huggingface import (
    HuggingFaceDownloadRequest,
    HuggingFaceDownloadResponse,
    HuggingFaceDownloadStatusResponse,
)
from backend.api.routes import hf_status

_log = logging.getLogger("app")

router = APIRouter(tags=["huggingface-download"])

# ── Background worker ─────────────────────────────────────────────────────────


async def _download_hf_dataset_to_server(
    dataset_id: str,
    split: str | None,
    config: str | None,
    auto_ingest: bool,
    data_root: Path,
    hf_token: str | None,
    *,
    auto_ingest_row_limit: int | None = None,
    auto_ingest_batch_size: int | None = None,
    auto_ingest_chunk_overlap: int | None = None,
    auto_ingest_max_tokens: int | None = None,
    auto_ingest_text_columns: list[str] | None = None,
    auto_ingest_meta_columns: list[str] | None = None,
    auto_ingest_split: str | None = None,
) -> None:
    """
    Background worker: download HF dataset directly on the backend and save to ./data/hf.

    When auto_ingest=True, automatically triggers HF ingestion after the download
    completes (with smart column detection and defaults).
    """
    lock = hf_status._hf_download_locks.setdefault(dataset_id, asyncio.Lock())

    if lock.locked():
        hf_status._set_download_status(
            dataset_id,
            status="downloading",
            message="A download is already in progress for this dataset",
        )
        return

    async with lock:
        safe_id = hf_status._normalize_dataset_dirname(dataset_id)
        target_dir = data_root / "hf" / safe_id
        _log.info("HF download starting: %s → %s (auto_ingest=%s)", dataset_id, target_dir, auto_ingest)

        hf_status._set_download_status(
            dataset_id,
            status="downloading",
            progress=2,
            message="Connecting to HuggingFace Hub…",
            error=None,
            local_path=str(target_dir),
            started_at=hf_status._utc_now_iso(),
            splits_downloaded=[],
            rows_downloaded=0,
            config=config,
            auto_ingest=auto_ingest,
            ingest_status="idle",
        )

        try:
            try:
                from datasets import (
                    DatasetDict,
                    get_dataset_config_names,
                    get_dataset_split_names,
                    load_dataset,
                )  # type: ignore[import-untyped]
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "The 'datasets' package is required for server-side HF download. "
                    "Install it with: uv add datasets"
                ) from exc

            if hf_status._is_cancelled(dataset_id):
                hf_status._cleanup_partial_dataset(dataset_id, data_root)
                hf_status._set_download_status(
                    dataset_id,
                    status="cancelled",
                    progress=0,
                    message="Download was cancelled.",
                )
                return

            # Resolve split names (config-aware) — timeout to prevent indefinite hangs
            try:
                async with asyncio.timeout(hf_status.MAX_DOWNLOAD_TIMEOUT):
                    split_names = (
                        get_dataset_split_names(dataset_id, config_name=config, use_auth_token=hf_token)
                        if config
                        else get_dataset_split_names(dataset_id, use_auth_token=hf_token)
                    )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Timed out after {hf_status.MAX_DOWNLOAD_TIMEOUT}s while resolving splits for '{dataset_id}'. "
                    "The dataset may be very large or the network may be slow. "
                    "Try again later or use a smaller row_limit."
                ) from None

            if split and split not in split_names:
                raise ValueError(
                    f"Split '{split}' not available. Available: {split_names}"
                )

            to_download = [split] if split else split_names
            if not to_download:
                to_download = ["default"]

            hf_status._set_download_status(
                dataset_id,
                progress=8,
                message=f"Ready to pull {len(to_download)} split(s) from HuggingFace…",
            )

            downloaded_splits: list[str] = []
            rows_total = 0

            with tempfile.TemporaryDirectory(prefix=f"hf-{safe_id}-") as tmp:
                temp_dir = Path(tmp)

                if to_download == ["default"]:
                    if hf_status._is_cancelled(dataset_id):
                        hf_status._cleanup_partial_dataset(dataset_id, data_root)
                        hf_status._set_download_status(
                            dataset_id, status="cancelled", progress=0,
                            message="Download was cancelled.",
                        )
                        return

                    hf_status._set_download_status(
                        dataset_id, progress=10,
                        message="Connecting to HuggingFace…",
                    )
                    tracker = hf_status._install_progress_tracker(dataset_id)

                    _load_kwargs: dict[str, Any] = {"cache_dir": str(temp_dir), "use_auth_token": hf_token}
                    if config:
                        _load_kwargs["name"] = config
                    async with asyncio.timeout(hf_status.MAX_DOWNLOAD_TIMEOUT):
                        ds_default = load_dataset(dataset_id, **_load_kwargs)

                    if tracker:
                        tracker.finalize(80)
                        hf_status._remove_progress_tracker(dataset_id)

                    if isinstance(ds_default, DatasetDict):
                        for split_name, split_ds in ds_default.items():
                            if hf_status._is_cancelled(dataset_id):
                                hf_status._cleanup_partial_dataset(dataset_id, data_root)
                                hf_status._set_download_status(
                                    dataset_id, status="cancelled", progress=0,
                                    message="Download was cancelled.",
                                )
                                return
                            out_path = temp_dir / split_name
                            out_path.mkdir(parents=True, exist_ok=True)
                            split_ds.to_parquet(str(out_path / "data.parquet"))
                            rows_total += len(split_ds)
                            downloaded_splits.append(split_name)
                    else:
                        out_path = temp_dir / "train"
                        out_path.mkdir(parents=True, exist_ok=True)
                        ds_default.to_parquet(str(out_path / "data.parquet"))
                        rows_total += len(ds_default)
                        downloaded_splits.append("train")
                else:
                    total = len(to_download)
                    for idx, split_name in enumerate(to_download, start=1):
                        if hf_status._is_cancelled(dataset_id):
                            hf_status._cleanup_partial_dataset(dataset_id, data_root)
                            hf_status._set_download_status(
                                dataset_id, status="cancelled", progress=0,
                                message="Download was cancelled.",
                            )
                            return

                        start_progress = 10 + int(((idx - 1) / total) * 70)
                        hf_status._set_download_status(
                            dataset_id,
                            progress=start_progress,
                            message=f"Pulling '{split_name}' from HuggingFace… ({idx}/{total})",
                        )

                        tracker = hf_status._install_progress_tracker(dataset_id)

                        _split_load_kwargs: dict[str, Any] = {"split": split_name, "cache_dir": str(temp_dir), "use_auth_token": hf_token}
                        if config:
                            _split_load_kwargs["name"] = config
                        async with asyncio.timeout(hf_status.MAX_DOWNLOAD_TIMEOUT):
                            ds_split = load_dataset(dataset_id, **_split_load_kwargs)

                        if tracker:
                            tracker.finalize(start_progress + int(50 / total))
                            hf_status._remove_progress_tracker(dataset_id)

                        out_path = temp_dir / split_name
                        out_path.mkdir(parents=True, exist_ok=True)
                        ds_split.to_parquet(str(out_path / "data.parquet"))

                        downloaded_splits.append(split_name)
                        rows_total += len(ds_split)

                        done_progress = 10 + int((idx / total) * 70)
                        hf_status._set_download_status(
                            dataset_id,
                            progress=done_progress,
                            message=f"✓ Saved '{split_name}' — {len(ds_split):,} rows",
                            rows_downloaded=rows_total,
                            splits_downloaded=downloaded_splits.copy(),
                        )

                        if hf_status._is_cancelled(dataset_id):
                            hf_status._cleanup_partial_dataset(dataset_id, data_root)
                            hf_status._set_download_status(
                                dataset_id, status="cancelled", progress=0,
                                message="Download was cancelled.",
                            )
                            return

                hf_status._set_download_status(
                    dataset_id,
                    progress=88,
                    message="Copying files to server storage…",
                    rows_downloaded=rows_total,
                )

                _log.info("HF download: %s complete — %d rows, %d splits",
                          dataset_id, rows_total, len(downloaded_splits))

                if hf_status._is_cancelled(dataset_id):
                    hf_status._cleanup_partial_dataset(dataset_id, data_root)
                    hf_status._set_download_status(
                        dataset_id, status="cancelled", progress=0,
                        message="Download was cancelled.",
                    )
                    return

                target_dir.parent.mkdir(parents=True, exist_ok=True)
                if target_dir.exists():
                    import shutil
                    shutil.rmtree(target_dir)
                hf_status._copy_dataset_to_hf_root(temp_dir=temp_dir, target_dir=target_dir)

            # ── Post-download: detect columns ─────────────────────────────────
            columns: dict[str, str] = {}
            suggested_text_col: str | None = None

            try:
                import pandas as pd
                pq_files = list(target_dir.rglob("*.parquet"))
                if pq_files:
                    df = pd.read_parquet(pq_files[0])
                    col_names = list(df.columns)
                    for col in col_names:
                        dtype = str(df[col].dtype)
                        if "object" in dtype or "string" in dtype:
                            columns[col] = "string"
                        elif "int" in dtype or "float" in dtype:
                            columns[col] = "numeric"
                        elif "bool" in dtype:
                            columns[col] = "bool"
                        else:
                            columns[col] = dtype
                    TEXT_CANDIDATES = [
                        "text", "content", "document", "passage", "context",
                        "question", "answer", "sentence", "body", "review",
                        "query", "input", "output", "story", "summary",
                    ]
                    for cand in TEXT_CANDIDATES:
                        if cand in col_names:
                            suggested_text_col = cand
                            break
                    if not suggested_text_col:
                        for col in col_names:
                            if col != "_source_file" and columns.get(col) == "string":
                                suggested_text_col = col
                                break
            except Exception:  # noqa: BLE001
                pass

            # Auto-ingest: trigger HF ingest if requested
            if auto_ingest:
                ingest_split = auto_ingest_split or split or "train"
                text_col = suggested_text_col

                hf_status._set_download_status(
                    dataset_id,
                    ingest_status="ingesting",
                    ingest_message="Starting automatic ingestion…",
                )

                try:
                    from backend.services.ingestion_factory import get_embedder, get_qdrant_store, get_sparse_embedder
                    from rag.ingestion.hf_ingestion import ingest_hf_dataset as _run_ingest

                    qdrant_store = get_qdrant_store()
                    embedder = get_embedder()
                    sparse_embedder = get_sparse_embedder()

                    qdrant_safe_id = dataset_id.replace("/", "_")

                    hf_status._set_download_status(
                        dataset_id,
                        ingest_status="ingesting",
                        ingest_message="Ingesting into Qdrant…",
                    )

                    result = await _run_ingest(
                        dataset_id=dataset_id,
                        qdrant_store=qdrant_store,
                        embedder=embedder,
                        sparse_embedder=sparse_embedder,
                        split=ingest_split,
                        text_columns=auto_ingest_text_columns or ([text_col] if text_col else None),
                        metadata_columns=auto_ingest_meta_columns or None,
                        batch_size=auto_ingest_batch_size or 32,
                        chunk_overlap=auto_ingest_chunk_overlap or 180,
                        max_tokens_per_chunk=auto_ingest_max_tokens or 500,
                        row_limit=auto_ingest_row_limit,
                    )

                    hf_status._set_download_status(
                        dataset_id,
                        status="completed",
                        progress=100,
                        message=f"✓ Downloaded & ingested — {result.get('chunks_embedded', 0):,} chunks",
                        rows_downloaded=rows_total,
                        splits_downloaded=downloaded_splits,
                        ingested=True,
                        ingest_status="completed",
                        ingest_message=f"Ingested {result.get('rows_processed', 0):,} rows, {result.get('chunks_embedded', 0):,} chunks",
                    )
                except Exception as exc:
                    _log.error("Auto-ingest failed for %s: %s", dataset_id, exc)
                    hf_status._set_download_status(
                        dataset_id,
                        status="completed",
                        progress=100,
                        message=f"✓ Downloaded successfully — auto-ingest failed: {exc}",
                        rows_downloaded=rows_total,
                        splits_downloaded=downloaded_splits,
                        ingest_status="failed",
                        ingest_error=str(exc),
                    )
            else:
                hf_status._set_download_status(
                    dataset_id,
                    status="completed",
                    progress=100,
                    message=(
                        f"✓ Downloaded {dataset_id} successfully! "
                        "Refresh the Local Datasets list to see it."
                    ),
                    rows_downloaded=rows_total,
                    splits_downloaded=downloaded_splits,
                    suggested_text_column=suggested_text_col,
                    columns=columns,
                )

        except Exception as exc:  # noqa: BLE001
            _log.error("HF download failed for %s: %s", dataset_id, exc)
            hf_status._cleanup_partial_dataset(dataset_id, data_root)
            hf_status._set_download_status(
                dataset_id,
                status="failed",
                progress=0,
                message="Download failed",
                error=str(exc),
            )
            if auto_ingest:
                hf_status._set_download_status(
                    dataset_id,
                    ingest_status="failed",
                    ingest_error=str(exc),
                )


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post("/download", response_model=HuggingFaceDownloadResponse)
@limiter.limit("30/minute")
async def download_hf_dataset(
    request: Request,
    payload: HuggingFaceDownloadRequest,
    background_tasks: BackgroundTasks,
) -> HuggingFaceDownloadResponse:
    """
    Submit a HuggingFace dataset for download.

    Runs in the background via BackgroundTasks. Poll GET /download/{dataset_id}/status
    for progress. Use POST /download/{dataset_id}/cancel to abort.
    """
    dataset_id = payload.dataset_id.strip()
    if not dataset_id or "/" not in dataset_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "dataset_id must be a valid HF dataset (e.g. 'author/name')")

    # Prevent duplicate downloads
    existing = hf_status._hf_download_status.get(dataset_id, {}).get("status")
    if existing in ("queued", "downloading"):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"A download for '{dataset_id}' is already {existing}. "
            "Cancel it first or wait for it to complete.",
        )

    hf_status._set_download_status(
        dataset_id,
        status="queued",
        progress=0,
        message="Waiting for server…",
        error=None,
        local_path=str(settings.data_root / "hf" / hf_status._normalize_dataset_dirname(dataset_id)),
        started_at=hf_status._utc_now_iso(),
        rows_downloaded=0,
        splits_downloaded=[],
        config=payload.config,
        auto_ingest=payload.auto_ingest,
        ingest_status="idle",
    )

    hf_status._hf_download_cancel_flags[dataset_id] = False

    background_tasks.add_task(
        _download_hf_dataset_to_server,
        dataset_id,
        payload.split,
        payload.config,
        payload.auto_ingest,
        settings.data_root,
        settings.hf_token,
        auto_ingest_row_limit=payload.row_limit,
        auto_ingest_batch_size=payload.batch_size,
        auto_ingest_chunk_overlap=payload.chunk_overlap,
        auto_ingest_max_tokens=payload.max_tokens_per_chunk,
        auto_ingest_text_columns=payload.text_columns if payload.text_columns else None,
        auto_ingest_meta_columns=payload.metadata_columns if payload.metadata_columns else None,
        auto_ingest_split=payload.ingest_split,
    )

    return HuggingFaceDownloadResponse(
        dataset_id=dataset_id,
        status="queued",
        message="Download queued — check status below for progress.",
    )


@router.get("/download/{dataset_id:path}/status", response_model=HuggingFaceDownloadStatusResponse)
async def get_hf_download_status(dataset_id: str) -> HuggingFaceDownloadStatusResponse:
    """Get current in-memory status for a dataset download task."""
    status_obj = hf_status._hf_download_status.get(dataset_id)
    if not status_obj:
        # Check if dataset was already downloaded to disk
        safe_id = dataset_id.replace("/", "__")
        local_path = settings.data_root / "hf" / safe_id
        if local_path.exists():
            return HuggingFaceDownloadStatusResponse(
                dataset_id=dataset_id,
                status="completed",
                progress=100,
                message="Downloaded (server restarted after completion)",
                local_path=str(local_path),
            )
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No download status found for '{dataset_id}'")

    return HuggingFaceDownloadStatusResponse(**status_obj)


@router.post("/download/{dataset_id:path}/cancel", response_model=HuggingFaceDownloadResponse)
async def cancel_hf_download(dataset_id: str) -> HuggingFaceDownloadResponse:
    """Cancel an in-progress or queued dataset download."""
    status_obj = hf_status._hf_download_status.get(dataset_id)
    if status_obj and status_obj.get("status") in ("completed", "failed", "cancelled"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Cannot cancel — download is already {status_obj.get('status')}")

    hf_status._hf_download_cancel_flags[dataset_id] = True

    hf_status._set_download_status(
        dataset_id,
        status="cancelled",
        progress=0,
        message="Cancellation requested…",
    )

    return HuggingFaceDownloadResponse(
        dataset_id=dataset_id,
        status="cancelled",
        message="Cancellation requested. The download will stop shortly.",
    )


# ── Discover endpoint ─────────────────────────────────────────────────────────

@router.get("/discover", response_model=dict)
async def discover_hf_datasets() -> dict:
    """Scan the local HuggingFace data directory and return all discovered datasets."""
    hf_root = settings.data_root / "hf"
    if not hf_root.is_dir():
        return {
            "scan_root": str(hf_root),
            "datasets": [],
            "total_found": 0,
            "message": "HF data directory not found.",
        }

    entries: list[dict] = []
    for item in hf_root.iterdir():
        try:
            if not item.is_dir():
                continue
        except OSError:
            continue
        if item.name.startswith("_") or item.name.startswith("."):
            continue

        # Normalize dataset_id from directory name
        # "user/repo" is stored as "user__repo"
        dataset_id = item.name.replace("__", "/")

        # Collect files
        files: list = []
        total_bytes = 0
        file_formats: set[str] = set()
        for f in item.rglob("*"):
            if f.is_file():
                files.append(f)
                total_bytes += f.stat().st_size
                ext = f.suffix.lower()
                if ext in {".parquet", ".json", ".csv", ".txt", ".jsonl"}:
                    file_formats.add(ext.lstrip("."))
                else:
                    file_formats.add(ext.lstrip(".") or "bin")

        # Extract splits from filename patterns
        # e.g. squad-00000-of-00001.parquet → split is inferred from parent dir or name
        splits: set[str] = set()
        for f in files:
            name = f.stem
            for part in name.split("-"):
                if part in {"train", "test", "validation", "dev"}:
                    splits.add(part)
        if not splits:
            splits = {"train"}

        # Try to read columns from first parquet
        readable_columns: list[str] = []
        for f in files:
            if f.suffix.lower() == ".parquet":
                try:
                    import pandas as pd  # noqa: F401
                    df = pd.read_parquet(f, engine="pyarrow")
                    readable_columns = [c for c in df.columns if df[c].dtype.kind in "ifuO"]
                    break
                except Exception:  # noqa: BLE001
                    pass

        # Check is_ingested
        safe_id = dataset_id.replace("/", "__")
        is_ingested = False
        for ds_id, st in hf_status._hf_download_status.items():
            if st.get("status") == "completed" and ds_id == safe_id:
                is_ingested = True
                break
        if not is_ingested:
            for ing in hf_status._hf_ingest_registry.values():
                if ing.get("status") == "completed" and ing.get("dataset_id") == dataset_id:
                    is_ingested = True
                    break

        entries.append(
            {
                "local_path": str(item),
                "dataset_id": dataset_id,
                "splits": sorted(splits),
                "file_formats": sorted(file_formats),
                "file_count": len(files),
                "total_size_bytes": total_bytes,
                "readable_columns": readable_columns,
                "description": None,
                "is_ingested": is_ingested,
            }
        )

    entries.sort(key=lambda e: e["dataset_id"])
    return {
        "scan_root": str(hf_root),
        "datasets": entries,
        "total_found": len(entries),
        "message": f"Found {len(entries)} local dataset(s).",
    }