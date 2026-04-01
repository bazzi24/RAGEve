from __future__ import annotations

import asyncio
import httpx
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse


# 100 MB per request — enforced via Content-Length header before files are read.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def _check_content_length(request: Request) -> None:
    """Raise 413 if the request's Content-Length exceeds MAX_UPLOAD_BYTES."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Request body exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB size limit.",
        )


from backend.api.routes._limiter import limiter

from backend.config import settings
from backend.schemas.datasets import (
    CollectionDeleteResponse,
    DatasetInfo,
    DatasetListResponse,
    IngestRequest,
    IngestResponse,
    IngestSubmitResponse,
    IngestStatusResponse,
)
from backend.services.file_processor import FileProcessorService, SUPPORTED_EXTENSIONS
from rag.deepdoc.quality_scorer import ChunkProfile

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])
processor = FileProcessorService()

# ------------------------------------------------------------------
# Ingest-status registry (file-backed, survives server restarts)
# ------------------------------------------------------------------
#
# Structure (all in one dict written to _ingest_status.json):
#
#   ingest_id  →  {
#       "ingest_id":  "abc123",
#       "dataset_id":  "my-dataset",
#       "status":      "running",           # queued | running | completed | failed | cancelled
#       "progress":    45,                  # 0-100 overall
#       "current_stage":  "embedding",
#       "message":     "Embedding chunks (1234/5000)",
#       "file_index":   1,
#       "file_total":   2,
#       "current_file": "report.xlsx",
#       "chunks_done":  1234,
#       "chunks_total": 5000,
#       "started_at":   "2026-03-28T10:00:00Z",
#       "completed_at": null,
#       "error":        null,
#       "files":        [],                 # final results when completed/failed
#   }
#
# Non-terminal entries (queued / running) are reloaded from disk on startup so
# polling continues to work after a restart.  Terminal entries (completed /
# failed / cancelled) are pruned on reload to keep the file small.

_ingest_registry: dict[str, dict[str, Any]] = {}
_ingest_registry_lock = asyncio.Lock()


def _ingest_status_file() -> Path:
    path = settings.ingest_status_file
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_ingest_registry() -> dict[str, dict[str, Any]]:
    """Load non-terminal ingest statuses from disk on startup."""
    terminal = {"completed", "failed", "cancelled"}
    try:
        fpath = _ingest_status_file()
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                raw: dict[str, dict[str, Any]] = json.load(f)
            # Prune terminal entries — they have no useful polling state.
            return {k: v for k, v in raw.items() if v.get("status") not in terminal}
    except (json.JSONDecodeError, OSError) as exc:
        # JSONDecodeError: file is corrupted; OSError: permission / missing file edge cases.
        # Neither should crash the app on startup — start with an empty registry.
        _log.warning("Could not load ingest registry from %s: %s", fpath, exc)
    return {}


def _persist_ingest_registry() -> None:
    """Atomically write the full registry to disk."""
    try:
        fpath = _ingest_status_file()
        tmp = fpath.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_ingest_registry, f)
        tmp.replace(fpath)          # atomic on POSIX
    except Exception as exc:
        _log.warning("Failed to persist ingest registry: %s", exc)


# Rehydrate non-terminal statuses from disk on module load.
_ingest_registry = _load_ingest_registry()
for iid, st in _ingest_registry.items():
    _log.info("Resuming ingest %s [%s] — stage=%s", iid, st.get("dataset_id"), st.get("current_stage"))


# ------------------------------------------------------------------
# Helper to update a status entry (writes in-memory + persists to disk)
# ------------------------------------------------------------------

async def _set_ingest_status(ingest_id: str, **kwargs: Any) -> dict[str, Any]:
    """Atomically merge kwargs into the in-memory registry and flush to disk."""
    async with _ingest_registry_lock:
        entry = _ingest_registry.setdefault(ingest_id, {"ingest_id": ingest_id})
        # Always allow updates to mutable state fields.
        for key, val in kwargs.items():
            entry[key] = val
        _persist_ingest_registry()
        return dict(entry)


# ------------------------------------------------------------------
# Background ingest submit + status polling
# ------------------------------------------------------------------

async def _run_background_ingest(
    ingest_id: str,
    dataset_id: str,
    file_paths: list[tuple[str, Path]],   # (original_filename, saved_path)
    req: IngestRequest,
    selected_profile: ChunkProfile | None,
) -> None:
    """
    Background task: ingest every file in ``file_paths`` sequentially,
    updating the shared registry on every stage change.

    Qdrant upserts are idempotent — a cancelled/interrupted run will have left
    partial chunks in the collection; re-submitting the same files with
    ``overwrite=True`` replaces them cleanly.  If the server crashes mid-run,
    the on-disk registry preserves progress and the next server start will show
    ``status=running`` for any interrupted ingest IDs (frontend should offer
    a "resume or cancel" choice).
    """
    from backend.services.ingestion_factory import get_ingestion_service

    ingestion = get_ingestion_service()
    total_files = len(file_paths)
    started_at = datetime.now(timezone.utc).isoformat()

    await _set_ingest_status(
        ingest_id,
        status="running",
        dataset_id=dataset_id,
        started_at=started_at,
        file_total=total_files,
        current_stage="queued",
        progress=0,
        files=[],
        error=None,
        completed_at=None,
    )

    try:
        results: list[dict] = []

        for file_idx, (original_filename, saved_path) in enumerate(file_paths, start=1):
            await _set_ingest_status(
                ingest_id,
                status="running",
                file_index=file_idx,
                current_file=original_filename,
                current_stage="extracting",
                progress=int(((file_idx - 1) / max(total_files, 1)) * 100),
            )

            try:
                result = await ingestion.ingest_file(
                    file_path=saved_path,
                    dataset_id=dataset_id,
                    chunk_size=req.chunk_size,
                    chunk_overlap=req.chunk_overlap,
                    max_tokens_per_chunk=req.max_tokens_per_chunk,
                    force_profile=selected_profile,
                    overwrite=req.overwrite,
                    progress_callback=None,  # structured events already via registry
                )
                results.append(result)

                # Cooperative cancellation: check if the ingest was cancelled
                # while processing this file; bail immediately if so.
                entry = _ingest_registry.get(ingest_id)
                if entry and entry.get("status") == "cancelled":
                    _log.info("Ingest %s cancelled mid-run; stopping after file %d/%d", ingest_id, file_idx, total_files)
                    return

                await _set_ingest_status(
                    ingest_id,
                    chunks_done=result.get("chunks", 0),
                    chunks_total=result.get("chunks", 0),
                    progress=int((file_idx / max(total_files, 1)) * 100),
                    current_stage="completed",
                    message=f"Completed {original_filename}",
                )

            except Exception as exc:
                _log.exception("Ingest failed for file %s in ingest_id=%s", original_filename, ingest_id)
                await _set_ingest_status(
                    ingest_id,
                    status="failed",
                    current_stage="failed",
                    error=str(exc),
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                return  # bail — first file failure aborts the whole ingest job

        completed_at = datetime.now(timezone.utc).isoformat()
        await _set_ingest_status(
            ingest_id,
            status="completed",
            progress=100,
            current_stage="completed",
            message=f"Ingested {sum(r.get('chunks', 0) for r in results)} chunks from {len(results)} file(s).",
            files=results,
            chunks_done=sum(r.get("chunks", 0) for r in results),
            completed_at=completed_at,
        )

    except Exception as exc:
        _log.exception("Background ingest task failed for ingest_id=%s", ingest_id)
        await _set_ingest_status(
            ingest_id,
            status="failed",
            current_stage="failed",
            error=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )


@router.post("/{dataset_id}/ingest/submit", response_model=IngestSubmitResponse)
async def submit_background_ingest(
    request: Request,
    dataset_id: str,
    files: list[UploadFile] = File(...),
    ingest_request: IngestRequest | None = None,
    background_tasks: BackgroundTasks = BackgroundTasks,
) -> IngestSubmitResponse:
    """
    Accept file uploads and start a background ingest task.  Returns
    ``202 Accepted`` immediately with an ``ingest_id`` for polling.

    Poll ``GET /datasets/ingest/{ingest_id}/status`` every 1–2 seconds for
    real-time progress.
    """
    _check_content_length(request)
    req = ingest_request or IngestRequest()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Validate file extensions up front before spawning the background task.
    for upload in files:
        ext = Path(upload.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            )

    # Validate force_profile.
    selected_profile: ChunkProfile | None = None
    if req.force_profile:
        try:
            selected_profile = ChunkProfile(req.force_profile)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid force_profile: {req.force_profile}. "
                f"Valid values: {[p.value for p in ChunkProfile]}",
            )

    # Save all uploaded files to disk before returning (fast, non-blocking).
    # The files must be on disk before _run_background_ingest starts.
    ingest_id = str(uuid.uuid4())
    file_paths: list[tuple[str, Path]] = []

    for upload in files:
        try:
            saved = await processor.save_upload(dataset_id=dataset_id, upload=upload)
            file_paths.append((upload.filename, saved))
        except Exception as exc:
            # Rollback already-saved files.
            for _, p in file_paths:
                p.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Failed to save {upload.filename}: {exc}")

    # Register initial status before scheduling the background task.
    started_at = datetime.now(timezone.utc).isoformat()
    await _set_ingest_status(
        ingest_id,
        status="queued",
        dataset_id=dataset_id,
        started_at=started_at,
        file_total=len(file_paths),
        file_index=0,
        current_file="",
        progress=0,
        current_stage="queued",
        message=f"Queued {len(file_paths)} file(s) for ingestion…",
        files=[],
        error=None,
        completed_at=None,
    )

    background_tasks.add_task(
        _run_background_ingest,
        ingest_id,
        dataset_id,
        file_paths,
        req,
        selected_profile,
    )

    _log.info(
        "Submitted background ingest ingest_id=%s dataset_id=%s files=%d",
        ingest_id, dataset_id, len(file_paths),
    )

    return IngestSubmitResponse(
        ingest_id=ingest_id,
        dataset_id=dataset_id,
        message=f"Ingest submitted — poll GET /datasets/ingest/{ingest_id}/status for progress.",
    )


@router.get("/ingest/{ingest_id}/status", response_model=IngestStatusResponse)
async def get_ingest_status(ingest_id: str) -> IngestStatusResponse:
    """
    Poll ingest progress.  Returns current stage, percentage, chunk counters,
    and (when finished) the full per-file results.

    After a server restart, non-terminal statuses are reloaded from disk so
    polling continues to work.
    """
    async with _ingest_registry_lock:
        entry = _ingest_registry.get(ingest_id)

    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"No ingest found for id '{ingest_id}'.",
        )

    # Compute ETA when running and we have enough data.
    eta_seconds: str | None = None
    if entry.get("status") == "running" and entry.get("chunks_total", 0) > 0:
        done = entry.get("chunks_done", 0) or 0
        total = entry["chunks_total"]
        started_str = entry.get("started_at")
        if done > 0 and started_str:
            try:
                started = datetime.fromisoformat(started_str)
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                rate = done / elapsed if elapsed > 0 else 0
                if rate > 0:
                    remaining = int((total - done) / rate)
                    eta_seconds = f"{remaining}s"
            except Exception:
                pass

    return IngestStatusResponse(
        ingest_id=ingest_id,
        dataset_id=entry.get("dataset_id", ""),
        status=entry.get("status", "unknown"),
        progress=entry.get("progress", 0),
        current_stage=entry.get("current_stage", "idle"),
        message=entry.get("message", ""),
        file_index=entry.get("file_index", 0),
        file_total=entry.get("file_total", 0),
        current_file=entry.get("current_file", ""),
        chunks_done=entry.get("chunks_done", 0),
        chunks_total=entry.get("chunks_total", 0),
        started_at=entry.get("started_at"),
        completed_at=entry.get("completed_at"),
        error=entry.get("error"),
        files=entry.get("files", []),
    )


@router.post("/ingest/{ingest_id}/cancel")
async def cancel_ingest(ingest_id: str) -> dict:
    """
    Cancel a running or queued ingest.  Marks the status as cancelled;
    the background task checks this flag on every stage transition and
    raises ``asyncio.CancelledError`` to abort cleanly.

    Note: cancellation is cooperative — chunks already upserted to Qdrant
    before cancellation are retained.  Re-submit with ``overwrite=True``
    to replace them.
    """
    async with _ingest_registry_lock:
        entry = _ingest_registry.get(ingest_id)

    if entry is None:
        raise HTTPException(status_code=404, detail=f"No ingest found for id '{ingest_id}'.")

    current_status = entry.get("status", "")
    if current_status in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=409,
            detail=f"Ingest is already '{current_status}' — cannot cancel.",
        )

    await _set_ingest_status(
        ingest_id,
        status="cancelled",
        current_stage="cancelled",
        message="Cancelled by user.",
        completed_at=datetime.now(timezone.utc).isoformat(),
    )

    _log.info("Ingest %s marked cancelled.", ingest_id)
    return {"ingest_id": ingest_id, "status": "cancelled", "message": "Cancellation requested."}


# ------------------------------------------------------------------
# List all datasets (collections in Qdrant)
# ------------------------------------------------------------------

@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    skip: int = Query(0, ge=0, description="Number of collections to skip"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of collections to return"),
) -> DatasetListResponse:
    """
    List all datasets by enumerating Qdrant collections.
    Returns metadata for each collection (chunks count, vector size, status).
    """
    from backend.services.ingestion_factory import get_qdrant_store
    import httpx

    store = get_qdrant_store()

    try:
        # Query Qdrant REST API for all collection names
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{store.base_url}/collections")
            r.raise_for_status()
            data = r.json()
            collection_names: list[str] = [
                c["name"] for c in data.get("result", {}).get("collections", [])
            ]
    except httpx.HTTPStatusError as exc:
        _log.error("Qdrant returned HTTP %d while listing collections: %s", exc.response.status_code, exc)
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="Vector store returned an error. Check server logs for details.",
        ) from exc
    except httpx.RequestError as exc:
        _log.error("Qdrant unreachable at %s: %s", store.base_url, exc)
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="Vector store is unreachable. Check that Qdrant is running.",
        ) from exc
    except Exception as exc:
        _log.error("Unexpected error listing Qdrant collections: %s", exc)
        from fastapi import HTTPException

        raise HTTPException(
            status_code=500,
            detail="Failed to list datasets. Check server logs for details.",
        ) from exc

    datasets: list[DatasetInfo] = []
    for name in collection_names[skip : skip + limit]:
        info = store.get_collection_info(name)
        if info is not None:
            datasets.append(
                DatasetInfo(
                    dataset_id=name,
                    collection=name,
                    chunks_count=info.get("points_count", 0),
                    vector_size=info.get("vectors_count", 0),
                    status=info.get("status", "unknown"),
                )
            )

    return DatasetListResponse(datasets=datasets, total=len(collection_names), skip=skip, limit=limit)


# ------------------------------------------------------------------
# Upload + ingest
# ------------------------------------------------------------------


@router.post("/{dataset_id}/upload", response_model=dict)
async def upload_and_ingest(
    request: Request,
    dataset_id: str,
    files: list[UploadFile] = File(...),
    ingest_request: IngestRequest | None = None,
) -> dict:
    """
    Upload files, run deepdoc pipeline (extract + chunk + quality score),
    then embed and upsert to Qdrant.
    """
    _check_content_length(request)
    req = ingest_request or IngestRequest()

    # Validate force_profile
    selected_profile: ChunkProfile | None = None
    if req.force_profile:
        try:
            selected_profile = ChunkProfile(req.force_profile)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid force_profile: {req.force_profile}. "
                f"Valid values: {[p.value for p in ChunkProfile]}",
            )

    from backend.services.ingestion_factory import get_ingestion_service

    ingestion = get_ingestion_service()

    results: list[dict] = []
    for upload in files:
        ext = Path(upload.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
            )

        saved_file = await processor.save_upload(dataset_id=dataset_id, upload=upload)

        result = await ingestion.ingest_file(
            file_path=saved_file,
            dataset_id=dataset_id,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            max_tokens_per_chunk=req.max_tokens_per_chunk,
            force_profile=selected_profile,
            overwrite=req.overwrite,
        )
        results.append(result)

    return {"dataset_id": dataset_id, "files": results}


async def _stream_upload_and_ingest(
    dataset_id: str,
    files: list[UploadFile],
    req: IngestRequest,
    selected_profile: ChunkProfile | None,
) -> AsyncIterator[str]:
    from backend.services.ingestion_factory import get_ingestion_service

    ingestion = get_ingestion_service()
    total_files = len(files)
    done_files = 0

    yield json.dumps(
        {
            "event": "status",
            "stage": "starting",
            "message": f"Starting upload for {total_files} file(s)",
            "progress": 0,
            "dataset_id": dataset_id,
            "file_index": 0,
            "file_total": total_files,
        }
    ) + "\n"

    results: list[dict] = []

    for idx, upload in enumerate(files, start=1):
        ext = Path(upload.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            yield json.dumps(
                {
                    "event": "error",
                    "stage": "failed",
                    "message": f"Unsupported extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
                    "progress": int(((idx - 1) / max(total_files, 1)) * 100),
                    "dataset_id": dataset_id,
                    "file": upload.filename,
                    "file_index": idx,
                    "file_total": total_files,
                }
            ) + "\n"
            return

        yield json.dumps(
            {
                "event": "status",
                "stage": "uploading",
                "message": f"Uploading {upload.filename}",
                "progress": int(((idx - 1) / max(total_files, 1)) * 100),
                "dataset_id": dataset_id,
                "file": upload.filename,
                "file_index": idx,
                "file_total": total_files,
            }
        ) + "\n"

        saved_file = await processor.save_upload(dataset_id=dataset_id, upload=upload)

        progress_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def on_progress(evt: dict) -> None:
            file_share = 100 / max(total_files, 1)
            base_progress = (idx - 1) * file_share
            local_progress = (evt.get("progress") or 0) / 100
            total_progress = int(min(100, base_progress + (local_progress * file_share)))

            payload = {
                "event": "status",
                "stage": evt.get("stage", "processing"),
                "message": evt.get("message", "Processing"),
                "progress": total_progress,
                "dataset_id": dataset_id,
                "file": upload.filename,
                "file_index": idx,
                "file_total": total_files,
            }
            if "chunks_done" in evt:
                payload["chunks_done"] = evt["chunks_done"]
            if "chunks_total" in evt:
                payload["chunks_total"] = evt["chunks_total"]

            await progress_queue.put(json.dumps(payload) + "\n")

        task = asyncio.create_task(
            ingestion.ingest_file(
                file_path=saved_file,
                dataset_id=dataset_id,
                chunk_size=req.chunk_size,
                chunk_overlap=req.chunk_overlap,
                max_tokens_per_chunk=req.max_tokens_per_chunk,
                force_profile=selected_profile,
                overwrite=req.overwrite,
                progress_callback=on_progress,
            )
        )

        while True:
            if task.done() and progress_queue.empty():
                break

            try:
                line = await asyncio.wait_for(progress_queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            if line is not None:
                yield line

        try:
            result = await task
        except Exception as exc:
            yield json.dumps(
                {
                    "event": "error",
                    "stage": "failed",
                    "message": str(exc),
                    "progress": int(((idx - 1) / max(total_files, 1)) * 100),
                    "dataset_id": dataset_id,
                    "file": upload.filename,
                    "file_index": idx,
                    "file_total": total_files,
                }
            ) + "\n"
            return

        results.append(result)
        done_files += 1

        yield json.dumps(
            {
                "event": "file_done",
                "stage": "completed",
                "message": f"Completed {upload.filename}",
                "progress": int((done_files / max(total_files, 1)) * 100),
                "dataset_id": dataset_id,
                "file": upload.filename,
                "file_index": idx,
                "file_total": total_files,
                "result": result,
            }
        ) + "\n"

    yield json.dumps(
        {
            "event": "done",
            "stage": "completed",
            "message": "All files completed",
            "progress": 100,
            "dataset_id": dataset_id,
            "files": results,
        }
    ) + "\n"


@router.post("/{dataset_id}/upload/stream")
@limiter.limit("60/minute")
async def upload_and_ingest_stream(
    request: Request,
    dataset_id: str,
    files: list[UploadFile] = File(...),
    ingest_request: IngestRequest | None = None,
) -> StreamingResponse:
    _check_content_length(request)
    req = ingest_request or IngestRequest()

    selected_profile: ChunkProfile | None = None
    if req.force_profile:
        try:
            selected_profile = ChunkProfile(req.force_profile)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid force_profile: {req.force_profile}. "
                f"Valid values: {[p.value for p in ChunkProfile]}",
            )

    return StreamingResponse(
        _stream_upload_and_ingest(dataset_id, files, req, selected_profile),
        media_type="application/x-ndjson",
    )


@router.post("/{dataset_id}/ingest", response_model=IngestResponse)
async def ingest_existing_files(
    dataset_id: str,
    req: IngestRequest | None = None,
) -> IngestResponse:
    """
    Re-ingest already-uploaded files from data/uploads/{dataset_id}/.
    Useful after changing chunk/embedding settings.
    """
    req = req or IngestRequest()

    # Path-traversal guard: resolve the joined path and verify it stays within upload_root
    resolved = (settings.upload_root / dataset_id).resolve()
    if not str(resolved).startswith(str(settings.upload_root.resolve())):
        raise HTTPException(status_code=400, detail="Invalid dataset_id")

    upload_dir = resolved

    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail=f"No uploaded files found for dataset '{dataset_id}'")

    selected_profile: ChunkProfile | None = None
    if req.force_profile:
        try:
            selected_profile = ChunkProfile(req.force_profile)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid force_profile: {req.force_profile}",
            )

    from backend.services.ingestion_factory import get_ingestion_service

    ingestion = get_ingestion_service()

    total_chunks = 0
    total_chars = 0
    quality_reports: list[dict] = []

    for file_path in upload_dir.iterdir():
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        result = await ingestion.ingest_file(
            file_path=file_path,
            dataset_id=dataset_id,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            max_tokens_per_chunk=req.max_tokens_per_chunk,
            force_profile=selected_profile,
            overwrite=req.overwrite,
        )
        total_chunks += result["chunks"]
        total_chars += result["chars"]
        quality_reports.append(result["quality_report"])

    return IngestResponse(
        dataset_id=dataset_id,
        collection=dataset_id,
        chunks_embedded=total_chunks,
        total_chars=total_chars,
        quality_report={"average_score": sum(r.get("quality_score", 0) for r in quality_reports) / max(len(quality_reports), 1)},
        message=f"Ingested {total_chunks} chunks from {len(quality_reports)} file(s).",
    )


# ------------------------------------------------------------------
# Collection info
# ------------------------------------------------------------------


@router.get("/{dataset_id}", response_model=DatasetInfo)
async def get_dataset_info(dataset_id: str) -> DatasetInfo:
    from backend.services.ingestion_factory import get_qdrant_store

    store = get_qdrant_store()
    info = store.get_collection_info(dataset_id)

    if info is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found in vector store.")

    return DatasetInfo(
        dataset_id=dataset_id,
        collection=dataset_id,
        chunks_count=info.get("points_count", 0),
        vector_size=info.get("vectors_count", 0),
        status=info.get("status", "unknown"),
    )


@router.delete("/{dataset_id}", response_model=CollectionDeleteResponse)
async def delete_dataset(dataset_id: str) -> CollectionDeleteResponse:
    from backend.services.ingestion_factory import get_qdrant_store

    store = get_qdrant_store()
    deleted = store.delete_collection(dataset_id)

    # Also clear upload + chunk dirs
    import shutil

    upload_dir = settings.upload_root / dataset_id
    chunk_dir = settings.chunk_root / dataset_id

    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)

    return CollectionDeleteResponse(
        dataset_id=dataset_id,
        deleted=deleted,
        message=f"Dataset '{dataset_id}' deleted from vector store and local files." if deleted else "Deletion failed or collection did not exist.",
    )
