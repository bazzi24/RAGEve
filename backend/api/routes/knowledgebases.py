"""
Knowledgebase API routes.

Endpoints for managing knowledgebases, documents, files, and ingestion tasks.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import time as _time
from pathlib import Path
from typing import List

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)

# Global registry to keep background tasks alive
_background_tasks: set[asyncio.Task] = set()

from backend.api.dependencies import get_current_user
from backend.api.routes._limiter import limiter
from backend.config_loader import settings
from backend.models_peewee import User
from backend.schemas.knowledgebases import (
    DocumentResponse,
    FileUploadResponse,
    KnowledgebaseCreate,
    KnowledgebaseListResponse,
    KnowledgebaseResponse,
    KnowledgebaseUpdate,
    TaskResponse,
)
from backend.services.cache_service import get_cache_service
from backend.services.database import run_db_operation
from backend.services.ingestion_factory import get_ingestion_service
from backend.services.knowledge_base_store import get_knowledge_base_store
from backend.services.minio_client import get_minio_client
from backend.services.tenant_user_store import get_tenant_user_store
from rag.ingestion.pipeline import SUPPORTED_EXTENSIONS
from rag.storage.qdrant_store import QdrantStore

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledgebases", tags=["knowledgebases"])


async def _get_allowed_tenant_ids(user: User) -> set[str]:
    if user.is_admin:
        return set()
    tenant_store = get_tenant_user_store()
    tenant_objs = await run_db_operation(tenant_store.get_tenants_for_user, user.id)
    return {user.id, *(tenant.id for tenant in tenant_objs)}


async def _ensure_tenant_access(user: User, tenant_id: str) -> None:
    if user.is_admin or tenant_id == user.id:
        return
    tenant_store = get_tenant_user_store()
    role = await run_db_operation(
        tenant_store.get_user_role_in_tenant, user.id, tenant_id
    )
    if role is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this tenant",
        )


async def _ensure_kb_access(user: User, kb_id: str):
    store = get_knowledge_base_store()
    kb = await run_db_operation(store.get_knowledgebase, kb_id)
    if not kb:
        raise HTTPException(
            status_code=404, detail=f"Knowledge base '{kb_id}' not found"
        )
    await _ensure_tenant_access(user, kb.tenant_id)
    return kb


async def _ensure_document_access(user: User, doc_id: str):
    store = get_knowledge_base_store()
    doc = await run_db_operation(store.get_document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    await _ensure_kb_access(user, doc.kb_id)
    return doc


async def _ensure_task_access(user: User, task_id: str):
    store = get_knowledge_base_store()
    task = await run_db_operation(store.get_task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    await _ensure_document_access(user, task.doc_id)
    return task


def _sanitize_filename(filename: str, use_uuid: bool = True) -> str:
    """
    Sanitize a user-provided filename to prevent path traversal attacks.

    Strips directory paths, null bytes, and dangerous characters.
    Returns a safe filename. If use_uuid=True, generates a UUID-based filename
    with the original extension to avoid filename collisions and information leakage.
    """
    import uuid

    if not filename:
        if use_uuid:
            return f"{uuid.uuid4().hex}.txt"
        return "untitled"

    # Remove null bytes
    filename = filename.replace("\x00", "")

    # Get only the basename (strip any path components)
    path = Path(filename)
    safe_name = path.name

    # Remove path traversal sequences
    if ".." in safe_name:
        safe_name = safe_name.replace("..", "_")

    # Replace dangerous path separators
    safe_name = safe_name.replace("/", "_").replace("\\", "_")

    # Strip leading/trailing dots and handle empty names
    safe_name = safe_name.strip(".")
    if not safe_name or safe_name in (".", ".."):
        safe_name = "file" if not use_uuid else uuid.uuid4().hex

    if use_uuid:
        # Preserve extension but use UUID for the stem
        ext = path.suffix.lower()
        # Validate extension is supported
        if ext not in SUPPORTED_EXTENSIONS:
            ext = ".txt"
        return f"{uuid.uuid4().hex}{ext}"
    else:
        return safe_name


@router.post(
    "/", response_model=KnowledgebaseResponse, status_code=status.HTTP_201_CREATED
)
@limiter.limit("60/minute")
async def create_knowledgebase(
    request: Request,
    payload: KnowledgebaseCreate,
    user: User = Depends(get_current_user),
) -> KnowledgebaseResponse:
    """Create a new knowledge base."""
    await _ensure_tenant_access(user, payload.tenant_id)

    store = get_knowledge_base_store()
    kb = await run_db_operation(
        store.create_knowledgebase,
        tenant_id=payload.tenant_id,
        name=payload.name,
        created_by=user.id,
        description=payload.description,
        avatar=payload.avatar,
        parser_ids=payload.parser_ids,
        language=payload.language,
        pagerank=payload.pagerank,
        pipeline_id=payload.pipeline_id,
    )
    kb_dict = kb.to_dict()
    return KnowledgebaseResponse(**kb_dict)


@router.get("/", response_model=KnowledgebaseListResponse)
@limiter.limit("120/minute")
async def list_knowledgebases(
    request: Request,
    tenant_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
) -> KnowledgebaseListResponse:
    """List knowledge bases, optionally filtered by tenant."""
    if tenant_id:
        await _ensure_tenant_access(user, tenant_id)

    store = get_knowledge_base_store()
    kbs, total = await run_db_operation(
        store.list_knowledgebases, tenant_id=tenant_id, limit=limit, offset=offset
    )

    if not user.is_admin:
        allowed_tenants = await _get_allowed_tenant_ids(user)
        kbs = [kb for kb in kbs if kb.get("tenant_id") in allowed_tenants]
        total = len(kbs)

    return KnowledgebaseListResponse(
        knowledgebases=[KnowledgebaseResponse(**kb) for kb in kbs],
        total=total,
    )


@router.get("/{kb_id}", response_model=KnowledgebaseResponse)
@limiter.limit("120/minute")
async def get_knowledgebase(
    request: Request,
    kb_id: str,
    user: User = Depends(get_current_user),
) -> KnowledgebaseResponse:
    """Get a knowledge base by ID."""
    kb = await _ensure_kb_access(user, kb_id)
    kb_dict = kb.to_dict()
    return KnowledgebaseResponse(**kb_dict)


@router.put("/{kb_id}", response_model=KnowledgebaseResponse)
@limiter.limit("60/minute")
async def update_knowledgebase(
    request: Request,
    kb_id: str,
    payload: KnowledgebaseUpdate,
    user: User = Depends(get_current_user),
) -> KnowledgebaseResponse:
    """Update a knowledge base."""
    await _ensure_kb_access(user, kb_id)

    store = get_knowledge_base_store()
    updates = payload.dict(exclude_unset=True)
    kb = await run_db_operation(store.update_knowledgebase, kb_id, **updates)
    if not kb:
        raise HTTPException(
            status_code=404, detail=f"Knowledge base '{kb_id}' not found"
        )
    kb_dict = kb.to_dict()
    return KnowledgebaseResponse(**kb_dict)


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("60/minute")
async def delete_knowledgebase(
    request: Request,
    kb_id: str,
    user: User = Depends(get_current_user),
) -> None:
    """Delete a knowledge base and all its documents, files, and tasks."""
    await _ensure_kb_access(user, kb_id)

    store = get_knowledge_base_store()
    cache_service = get_cache_service()

    # First, verify knowledgebase exists before any destructive operations
    kb_exists = await run_db_operation(store.get_knowledgebase, kb_id)
    if not kb_exists:
        raise HTTPException(
            status_code=404, detail=f"Knowledge base '{kb_id}' not found"
        )

    # Use atomic transaction: delete DB first, then invalidate cache on success
    # If cache invalidation fails, we log but don't fail the request (cache will expire naturally)
    try:
        # Delete knowledgebase and cascade (this is atomic within the DB transaction)
        deleted = await run_db_operation(store.delete_knowledgebase, kb_id)
        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"Knowledge base '{kb_id}' not found"
            )

        # Delete Qdrant collection if it exists (non-blocking)
        try:
            qdrant: QdrantStore = get_ingestion_service().qdrant
            qdrant.delete_collection(kb_id)
        except Exception as e:
            _log.warning("Failed to delete Qdrant collection %s: %s", kb_id, e)

        # Invalidate cache AFTER successful deletion
        try:
            invalidated = await cache_service.invalidate_collection(kb_id)
            _log.info(
                "Invalidated %d cached items for KB %s (delete)", invalidated, kb_id
            )
        except Exception as e:
            _log.warning(
                "Cache invalidation failed for KB %s (deletion succeeded): %s", kb_id, e
            )

    except Exception as e:
        # On any failure after existence check, attempt cache rollback if we invalidated prematurely
        _log.error("Failed to delete knowledge base %s: %s", kb_id, e)
        raise HTTPException(
            status_code=500, detail=f"Failed to delete knowledge base: {str(e)}"
        )


@router.post("/{kb_id}/upload", response_model=list[FileUploadResponse])
@limiter.limit("60/minute")
async def upload_files(
    request: Request,
    kb_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    parser_id: str = Query(default="pdf", description="Parser to use for these files"),
    chunk_size: int | None = Query(default=None, ge=100, le=10000),
    chunk_overlap: int | None = Query(default=None, ge=0, le=1000),
    user: User = Depends(get_current_user),
) -> list[FileUploadResponse]:
    """Upload files to a knowledge base.

    This creates File, Document, and Task records, then kicks off background ingestion.
    Ingestion progress can be tracked via the task_id.
    """
    # Validate knowledgebase exists
    store = get_knowledge_base_store()
    kb = await _ensure_kb_access(user, kb_id)

    # Determine created_by - for now use kb.created_by or a placeholder
    created_by = kb.created_by

    results: list[FileUploadResponse] = []
    minio_client = get_minio_client()

    for upload in files:
        # SECURITY: Check Content-Length header BEFORE reading any data
        # This prevents memory exhaustion from reading huge files into memory
        content_length = upload.size or request.headers.get("content-length")
        if content_length:
            try:
                content_length_int = int(content_length)
                if content_length_int > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File '{upload.filename}' exceeds maximum size of {settings.max_upload_bytes} bytes (Content-Length: {content_length_int})",
                    )
            except (ValueError, TypeError):
                pass  # Invalid content-length, will catch during actual read

        # Validate filename BEFORE processing
        original_filename = upload.filename or "untitled"
        safe_filename = _sanitize_filename(original_filename)
        if safe_filename != original_filename:
            _log.warning(
                "Filename sanitized: '%s' -> '%s'",
                original_filename,
                safe_filename,
            )

        file_ext = Path(safe_filename).suffix.lower()
        file_type = file_ext.lstrip(".") if file_ext else "unknown"

        # SECURITY: Validate file extension against whitelist
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type '{file_ext}' is not supported. Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

        # Stream file to temporary location to avoid memory exhaustion
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=file_ext if file_ext else ".tmp",
                delete=False,
            ) as tmp:
                temp_file_path = tmp.name
                # Stream in chunks to avoid loading entire file into memory
                chunk_size_stream = 1024 * 1024  # 1MB chunks
                total_read = 0
                while True:
                    chunk = await upload.read(chunk_size_stream)
                    if not chunk:
                        break
                    total_read += len(chunk)
                    if total_read > settings.max_upload_bytes:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"File '{safe_filename}' exceeds maximum size of {settings.max_upload_bytes} bytes",
                        )
                    tmp.write(chunk)

            # Get actual file size
            file_size = total_read

            # SECURITY: Validate MIME type using both extension and magic bytes
            from backend.services.file_processor import validate_mime_type

            mime_valid, mime_type, mime_error = validate_mime_type(
                temp_file_path, safe_filename
            )
            if not mime_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid file type: {mime_error}",
                )

            # Read bytes for MinIO upload (file is now on disk, not in memory)
            with open(temp_file_path, "rb") as f:
                file_bytes = f.read()

            # Upload to MinIO
            minio_key = minio_client.get_upload_path(kb_id, safe_filename)
            try:
                await minio_client.upload_file(
                    minio_key, file_bytes, content_type=mime_type
                )
            except Exception as e:
                _log.error("Failed to upload file to MinIO: %s", e)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to store file: {str(e)}",
                )

            # Create File record
            file_rec = await run_db_operation(
                store.create_file,
                name=safe_filename,
                size=file_size,
                file_type=mime_type.split("/")[0] if "/" in mime_type else file_type,
                created_by=created_by,
                source_type="upload",
            )

            # Create Document record
            doc_rec = await run_db_operation(
                store.create_document,
                kb_id=kb_id,
                name=safe_filename,
                parser_id=parser_id,
                created_by=created_by,
                doc_type=file_type,
            )

            # Link file to document
            await run_db_operation(store.link_file_to_document, file_rec.id, doc_rec.id)

            # Create Task for ingestion
            task_rec = await run_db_operation(
                store.create_task,
                doc_id=doc_rec.id,
                task_type="ingestion",
                from_page=0,
                to_page=100000000,
            )

            # Kick off background ingestion
            task = asyncio.create_task(run_ingestion_background(
                task_id=task_rec.id,
                doc_id=doc_rec.id,
                kb_id=kb_id,
                minio_key=minio_key,
                parser_id=parser_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ))
            _background_tasks.add(task)

            def handle_task_result(t: asyncio.Task) -> None:
                try:
                    _background_tasks.discard(t)
                    t.result()
                except Exception as e:
                    _log.exception("Background ingestion task failed: %s", e)

            task.add_done_callback(handle_task_result)

            results.append(
                FileUploadResponse(
                    filename=safe_filename,
                    file_id=file_rec.id,
                    doc_id=doc_rec.id,
                    task_id=task_rec.id,
                    size=file_size,
                    file_type=(
                        mime_type.split("/")[0] if "/" in mime_type else file_type
                    ),
                    status="queued",
                )
            )

        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

    return results


async def run_ingestion_background(
    task_id: str,
    doc_id: str,
    kb_id: str,
    minio_key: str | None = None,
    parser_id: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
):
    """Background task to run ingestion and update Task/Document records."""
    import os

    store = get_knowledge_base_store()
    ingestion = get_ingestion_service()
    minio_client = get_minio_client()
    cache_service = get_cache_service()

    # Mark task as started
    await run_db_operation(store.start_task, task_id)
    t0 = _time.monotonic()

    temp_file_path_local: str | None = None

    try:
        await run_db_operation(
            store.update_document_progress,
            doc_id,
            progress=10.0,
            progress_msg="Starting ingestion",
        )

        # Download file from MinIO
        if not minio_key:
            raise ValueError("No MinIO key available for download")

        await run_db_operation(
            store.update_document_progress,
            doc_id,
            progress=15.0,
            progress_msg="Downloading file from MinIO",
        )
        file_bytes = await minio_client.download_file(minio_key)
        # SECURITY: Do not derive local temp-file path components from user-controlled keys.
        # Use a constant safe suffix for ingestion scratch files.
        safe_suffix = ".bin"
        # Write to temp file (delete=False so it persists for ingestion)
        tmp = tempfile.NamedTemporaryFile(suffix=safe_suffix, delete=False)
        tmp.write(file_bytes)
        tmp.close()
        temp_file_path_local = tmp.name
        file_path = Path(temp_file_path_local)

        # Run ingestion
        result = await ingestion.ingest_file(
            file_path=file_path,
            dataset_id=kb_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        elapsed = _time.monotonic() - t0

        # Invalidate cache for this knowledge base after successful ingestion
        try:
            invalidated = await cache_service.invalidate_collection(kb_id)
            _log.info("Invalidated %d cached items for KB %s", invalidated, kb_id)
        except Exception as e:
            _log.warning("Cache invalidation failed for KB %s: %s", kb_id, e)

        # Update document as complete
        await run_db_operation(
            store.complete_document,
            doc_id,
            duration=elapsed,
            doc_metadata={
                "chunks": result.get("chunks", 0),
                "quality_report": result.get("quality_report", {}),
                "extraction": result.get("extraction", {}),
            },
        )

        # Update task to complete
        await run_db_operation(store.complete_task, task_id, duration=elapsed)
        _log.info("Ingestion completed for doc %s (task %s)", doc_id, task_id)

    except Exception as e:
        _log.exception("Ingestion failed for doc %s (task %s)", doc_id, task_id)
        # Update task and document with failure - wrap each to avoid cascading errors
        try:
            await run_db_operation(
                store.update_task_progress,
                task_id,
                progress=-1.0,  # negative indicates error
                msg=f"Error: {str(e)[:200]}",
            )
        except Exception as db_err:
            _log.error("Failed to update task error status: %s", db_err)

        try:
            await run_db_operation(
                store.update_document_progress,
                doc_id,
                progress=-1.0,
                progress_msg=f"Ingestion failed: {str(e)[:200]}",
            )
        except Exception as db_err:
            _log.error("Failed to update document error status: %s", db_err)
    finally:
        # Clean up temp file created during download
        if temp_file_path_local and os.path.exists(temp_file_path_local):
            try:
                os.unlink(temp_file_path_local)
            except OSError as cleanup_err:
                _log.warning("Failed to clean up temp file %s: %s", temp_file_path_local, cleanup_err)


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
@limiter.limit("120/minute")
async def get_document(
    request: Request,
    doc_id: str,
    user: User = Depends(get_current_user),
) -> DocumentResponse:
    """Get document details by ID."""
    doc = await _ensure_document_access(user, doc_id)
    doc_dict = doc.to_dict()
    return DocumentResponse(**doc_dict)


@router.get("/documents", response_model=List[DocumentResponse])
@limiter.limit("120/minute")
async def list_documents(
    request: Request,
    kb_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
) -> list[DocumentResponse]:
    """List documents, optionally filtered by knowledge base."""
    store = get_knowledge_base_store()
    if kb_id:
        await _ensure_kb_access(user, kb_id)

    docs, _ = await run_db_operation(
        store.list_documents, kb_id=kb_id, limit=limit, offset=offset
    )
    if not user.is_admin and not kb_id:
        allowed_tenants = await _get_allowed_tenant_ids(user)
        filtered_docs: list[dict] = []
        for doc in docs:
            kb = await run_db_operation(store.get_knowledgebase, doc.get("kb_id"))
            if kb and kb.tenant_id in allowed_tenants:
                filtered_docs.append(doc)
        docs = filtered_docs
    return [DocumentResponse(**d) for d in docs]


@router.get("/tasks/{task_id}", response_model=TaskResponse)
@limiter.limit("120/minute")
async def get_task(
    request: Request,
    task_id: str,
    user: User = Depends(get_current_user),
) -> TaskResponse:
    """Get task details by ID."""
    task = await _ensure_task_access(user, task_id)
    task_dict = task.to_dict()
    return TaskResponse(**task_dict)


@router.get("/documents/{doc_id}/tasks", response_model=List[TaskResponse])
@limiter.limit("120/minute")
async def list_document_tasks(
    request: Request,
    doc_id: str,
    user: User = Depends(get_current_user),
) -> list[TaskResponse]:
    """List all tasks for a document."""
    await _ensure_document_access(user, doc_id)

    store = get_knowledge_base_store()
    tasks = await run_db_operation(store.get_document_tasks, doc_id)
    return [TaskResponse(**t.to_dict()) for t in tasks]


@router.post(
    "/{kb_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit("60/minute")
async def create_document_for_kb(
    request: Request,
    kb_id: str,
    payload: dict,
    user: User = Depends(get_current_user),
) -> DocumentResponse:
    """Create a new document within a knowledge base."""
    store = get_knowledge_base_store()
    await _ensure_kb_access(user, kb_id)

    name = payload.get("name")
    parser_id = payload.get("parser_id")
    if not all([name, parser_id]):
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: name, parser_id",
        )
    doc = await run_db_operation(
        store.create_document,
        kb_id=kb_id,
        name=name,
        parser_id=parser_id,
        created_by=user.id,
    )
    doc_dict = doc.to_dict()
    return DocumentResponse(**doc_dict)


@router.post("/documents/{doc_id}/upload", response_model=FileUploadResponse)
@limiter.limit("60/minute")
async def upload_file_to_document(
    request: Request,
    doc_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
) -> FileUploadResponse:
    """Upload a file and attach it to an existing document."""
    store = get_knowledge_base_store()
    doc = await _ensure_document_access(user, doc_id)
    kb_id = doc.kb_id

    # SECURITY: Check Content-Length header BEFORE reading
    content_length = file.size or request.headers.get("content-length")
    if content_length:
        try:
            content_length_int = int(content_length)
            if content_length_int > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{file.filename}' exceeds maximum size of {settings.max_upload_bytes} bytes",
                )
        except (ValueError, TypeError):
            pass

    # Validate filename
    original_filename = file.filename or "untitled"
    safe_filename = _sanitize_filename(original_filename)

    file_ext = Path(safe_filename).suffix.lower()
    file_type = file_ext.lstrip(".") if file_ext else "unknown"

    # Validate extension
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{file_ext}' is not supported. Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Stream to temp file
    temp_file_path = None
    minio_client = get_minio_client()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=file_ext if file_ext else ".tmp",
            delete=False,
        ) as tmp:
            temp_file_path = tmp.name
            chunk_size_stream = 1024 * 1024
            total_read = 0
            while True:
                chunk = await file.read(chunk_size_stream)
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File '{safe_filename}' exceeds maximum size of {settings.max_upload_bytes} bytes",
                    )
                tmp.write(chunk)

        file_size = total_read

        # Validate MIME type
        from backend.services.file_processor import validate_mime_type

        mime_valid, mime_type, mime_error = validate_mime_type(
            temp_file_path, safe_filename
        )
        if not mime_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type: {mime_error}",
            )

        # Read for MinIO upload
        with open(temp_file_path, "rb") as f:
            file_bytes = f.read()

        minio_key = minio_client.get_upload_path(kb_id, safe_filename)
        try:
            await minio_client.upload_file(
                minio_key, file_bytes, content_type=mime_type
            )
        except Exception as e:
            _log.error("Failed to upload file to MinIO: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store file: {str(e)}",
            )

        # Create File record
        file_rec = await run_db_operation(
            store.create_file,
            name=safe_filename,
            size=file_size,
            file_type=mime_type.split("/")[0] if "/" in mime_type else file_type,
            created_by=doc.created_by,
            source_type="upload",
        )
        # Link file to document
        await run_db_operation(store.link_file_to_document, file_rec.id, doc_id)
        # Create Task for ingestion
        task_rec = await run_db_operation(
            store.create_task,
            doc_id=doc_id,
            task_type="ingestion",
            from_page=0,
            to_page=100000000,
        )
        # Kick off background ingestion
        task = asyncio.create_task(run_ingestion_background(
            task_id=task_rec.id,
            doc_id=doc_id,
            kb_id=kb_id,
            minio_key=minio_key,
            parser_id=doc.parser_id,
            chunk_size=None,
            chunk_overlap=None,
        ))
        _background_tasks.add(task)

        def handle_task_result(t: asyncio.Task) -> None:
            try:
                _background_tasks.discard(t)
                t.result()
            except Exception as e:
                _log.exception("Background ingestion task failed: %s", e)

        task.add_done_callback(handle_task_result)
        temp_file_path = None

        return FileUploadResponse(
            filename=safe_filename,
            file_id=file_rec.id,
            doc_id=doc_id,
            task_id=task_rec.id,
            size=file_size,
            file_type=mime_type.split("/")[0] if "/" in mime_type else file_type,
            status="queued",
        )

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass


@router.get("/{kb_id}/documents", response_model=list[DocumentResponse])
@limiter.limit("120/minute")
async def list_documents_for_kb(
    request: Request,
    kb_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
) -> list[DocumentResponse]:
    """List documents for a specific knowledge base."""
    await _ensure_kb_access(user, kb_id)

    store = get_knowledge_base_store()
    docs, _ = await run_db_operation(
        store.list_documents, kb_id=kb_id, limit=limit, offset=offset
    )
    return [DocumentResponse(**d) for d in docs]
