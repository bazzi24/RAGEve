"""
File processor service for RAGEve.

.. deprecated::
    This module is deprecated and will be removed in a future version.
    The active ingestion pipeline now uses :mod:`backend.services.ingestion_service`
    which handles extraction, chunking, embedding, and Qdrant upsert in a streaming fashion.

Handles document ingestion: saving uploads, processing files into chunks,
and persisting results to storage.
"""

from __future__ import annotations

import logging
import os
import random
import uuid
from pathlib import Path
from typing import Tuple

from fastapi import UploadFile

from backend.config_loader import settings
from backend.utils.log_sanitizer import sanitize_key
from rag.ingestion.pipeline import SUPPORTED_EXTENSIONS, run_deepdoc_ingestion

try:
    import filetype

    HAS_FILETYPE = True
except ImportError:
    HAS_FILETYPE = False

_log = logging.getLogger("backend.services.file_processor")

__all__ = ["FileProcessorService", "SUPPORTED_EXTENSIONS"]


# Error classification for retry logic
class ErrorType:
    TRANSIENT = "transient"  # Network issues, temporary service unavailability
    PERMANENT = "permanent"  # Invalid file, unsupported format, corrupt data
    UNKNOWN = "unknown"  # Unclassified errors


# Transient error indicators (substring match)
TRANSIENT_ERROR_PATTERNS = [
    "connection",
    "timeout",
    "network",
    "unreachable",
    "temporary",
    "try again",
    "resource temporarily unavailable",
    "econnreset",
    "broken pipe",
    "reset by peer",
    "service unavailable",
    "gateway timeout",
    "502",
    "503",
    "504",
]

# Permanent error indicators
PERMANENT_ERROR_PATTERNS = [
    "invalid file",
    "unsupported",
    "corrupt",
    "malformed",
    "cannot decode",
    "unsupported format",
    "file type not supported",
    "permission denied",
    "access denied",
    "not found",
]


def classify_error(error: Exception | str) -> str:
    """Classify an error as transient, permanent, or unknown."""
    error_str = str(error).lower()

    for pattern in TRANSIENT_ERROR_PATTERNS:
        if pattern in error_str:
            return ErrorType.TRANSIENT

    for pattern in PERMANENT_ERROR_PATTERNS:
        if pattern in error_str:
            return ErrorType.PERMANENT

    return ErrorType.UNKNOWN


def sanitize_error_message(error: Exception | str, max_length: int = 500) -> str:
    """Sanitize error message for safe storage in database."""
    error_str = str(error)
    # Truncate to prevent DB overflow
    if len(error_str) > max_length:
        error_str = error_str[: max_length - 3] + "..."
    # Remove any control characters
    error_str = "".join(c for c in error_str if ord(c) >= 32 or ord(c) == 9)
    return error_str


def calculate_backoff(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0
) -> float:
    """Calculate exponential backoff with jitter."""
    delay = min(base_delay * (2**attempt), max_delay)
    jitter = random.uniform(0, delay * 0.1)  # 10% jitter
    return delay + jitter


# Whitelist of safe MIME types for upload
ALLOWED_MIME_TYPES = {
    # Documents
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "text/plain",
    "text/markdown",
    "text/html",
    "application/rtf",
    # Images (for OCR)
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/webp",
}

# Blocked executable/dangerous MIME types
BLOCKED_MIME_TYPES = {
    "application/x-msdownload",  # .exe
    "application/x-msdos-program",  # .com
    "application/x-sh",
    "application/x-bat",
    "application/x-csh",
    "text/x-shellscript",
    "application/x-msi",
    "application/vnd.microsoft.portable-executable",
    "application/octet-stream",  # Generic binary - block unless we can positively identify
}


def _sanitize_filename(filename: str, use_uuid: bool = True) -> Tuple[str, str]:
    """
    Sanitize filename and generate storage-safe name.

    Args:
        filename: Original user-provided filename
        use_uuid: If True, generate UUID-based filename; else use sanitized original

    Returns:
        Tuple of (storage_filename, original_extension)
    """
    if not filename:
        # Return default untitled with .txt extension
        if use_uuid:
            return f"{uuid.uuid4().hex}.txt", ".txt"
        return "untitled.txt", ".txt"

    # Remove null bytes
    filename = filename.replace("\x00", "")

    # Get basename and extension
    path = Path(filename)
    stem = path.stem
    ext = path.suffix.lower()

    # Sanitize stem: remove path traversal, dangerous characters
    stem = stem.replace("..", "_").replace("/", "_").replace("\\", "_")
    # Replace spaces with underscores for cleaner storage
    stem = stem.replace(" ", "_")
    # Keep only alphanumeric, dash, underscore
    stem = "".join(c for c in stem if c.isalnum() or c in "-_")[:100]

    # Remove underscores-only stems (e.g., from "..." -> "___")
    stem = stem.strip("_")
    if not stem:
        stem = "file"

    # Ensure extension is in supported list
    if ext not in SUPPORTED_EXTENSIONS:
        # Default to .txt if unknown
        ext = ".txt"

    if use_uuid:
        # Generate UUID-based filename for storage
        storage_name = f"{uuid.uuid4().hex}{ext}"
    else:
        storage_name = f"{stem}{ext}"

    return storage_name, ext


def validate_mime_type(file_path: Path | str, filename: str) -> Tuple[bool, str, str]:
    """
    Validate file MIME type using both extension and magic bytes.

    Args:
        file_path: Path to the uploaded file (on disk)
        filename: Original filename (for extension check)

    Returns:
        Tuple of (is_valid, mime_type, error_message)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False, "", "File does not exist"

    # Check file size is reasonable
    try:
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, "", "File is empty"
        # Convert max_upload_bytes to int (it may be stored as string)
        max_size = (
            int(settings.max_upload_bytes)
            if isinstance(settings.max_upload_bytes, str)
            else settings.max_upload_bytes
        )
        if file_size > max_size:
            return False, "", f"File size {file_size} exceeds limit {max_size}"
    except OSError as e:
        return False, "", f"Cannot access file: {e}"

    # Get extension from filename
    ext = Path(filename).suffix.lower()

    # First, check if extension is in our supported list
    if ext not in SUPPORTED_EXTENSIONS:
        return False, "", f"File extension '{ext}' is not supported"

    # If we have filetype library, check magic bytes
    if HAS_FILETYPE:
        try:
            kind = filetype.guess(str(file_path))
            if kind:
                mime_type = kind.mime

                # Check if MIME type is in our whitelist
                if mime_type in BLOCKED_MIME_TYPES:
                    return False, "", f"Dangerous file type detected: {mime_type}"

                if mime_type not in ALLOWED_MIME_TYPES:
                    # Extension/type mismatch - could be disguised file
                    return (
                        False,
                        "",
                        f"File content ({mime_type}) does not match extension ({ext})",
                    )

                return True, mime_type, ""
            else:
                # Plain text files do not have reliable magic bytes, so fall
                # through to the extension-based MIME mapping below.
                if ext not in {".txt"}:
                    return False, "", "Could not determine file type from content"
        except Exception as e:
            _log.warning("filetype check failed: %s", e)
            # Fall back to extension-only check
    else:
        _log.warning("filetype library not installed, using extension-only validation")

    # Fallback: Use extension-based MIME mapping
    ext_to_mime = {
        ".pdf": "application/pdf",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".ppt": "application/vnd.ms-powerpoint",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".rtf": "application/rtf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
        ".webp": "image/webp",
    }

    mime_type = ext_to_mime.get(ext, "application/octet-stream")

    if mime_type in BLOCKED_MIME_TYPES:
        return False, "", f"Blocked file type: {mime_type}"

    if mime_type not in ALLOWED_MIME_TYPES and mime_type != "application/octet-stream":
        return False, "", f"Unsupported MIME type from extension: {mime_type}"

    return True, mime_type, ""


def _resolve_dataset_dir(root: Path, dataset_id: str) -> Path:
    """Resolve a dataset directory while keeping it inside the configured root."""
    if not dataset_id or dataset_id.strip() in {"", ".", ".."}:
        raise ValueError("Invalid dataset_id: empty or unsafe path")

    root_path = root.resolve()
    dataset_path = (root_path / dataset_id).resolve()
    try:
        dataset_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("Invalid dataset_id: path escapes storage root") from exc

    if dataset_path == root_path:
        raise ValueError("Invalid dataset_id: must identify a child directory")

    return dataset_path


class FileProcessorService:
    def __init__(self) -> None:
        self.chunk_size = settings.default_chunk_size
        self.chunk_overlap = settings.default_chunk_overlap
        self.max_tokens_per_chunk = settings.default_max_tokens_per_chunk

    async def save_upload(
        self,
        dataset_id: str,
        upload: UploadFile,
        file_bytes: bytes | None = None,
    ) -> Path:
        """
        Write an uploaded file to ``data/uploads/{dataset_id}/``.

        Pass ``file_bytes`` when the caller has already called ``await upload.read()``
        to avoid a second read.  When omitted the method reads the bytes itself.
        """
        dataset_dir = _resolve_dataset_dir(settings.upload_root, dataset_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if upload.filename is None:
            raise ValueError("Uploaded file missing filename")

        # SECURITY: Sanitize filename and use UUID for storage
        original_filename = upload.filename
        safe_filename, ext = _sanitize_filename(original_filename, use_uuid=True)

        # Get content (either provided or read)
        if file_bytes is not None:
            content = file_bytes
        else:
            # Stream to avoid memory issues with large files
            temp_file = None
            try:
                # Create temp file for streaming
                import tempfile as tf

                with tf.NamedTemporaryFile(mode="wb", suffix=ext, delete=False) as tmp:
                    temp_file = tmp.name
                    chunk_size = 1024 * 1024
                    total = 0
                    while True:
                        chunk = await upload.read(chunk_size)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > settings.max_upload_bytes:
                            raise ValueError(
                                f"File '{original_filename}' exceeds maximum size of {settings.max_upload_bytes} bytes"
                            )
                        tmp.write(chunk)
                # Read back from temp file
                with open(temp_file, "rb") as f:
                    content = f.read()
            finally:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass

        # Validate file size
        if len(content) > settings.max_upload_bytes:
            raise ValueError(
                f"File '{original_filename}' exceeds maximum size of {settings.max_upload_bytes} bytes"
            )

        # SECURITY: Validate MIME type
        # We need to write to a temp file first to check magic bytes
        temp_check_path = None
        try:
            import tempfile as tf

            with tf.NamedTemporaryFile(mode="wb", suffix=ext, delete=False) as tmp:
                temp_check_path = tmp.name
                tmp.write(content)
        finally:
            pass

        if temp_check_path:
            try:
                mime_valid, mime_type, mime_error = validate_mime_type(
                    temp_check_path, original_filename
                )
                if not mime_valid:
                    # Clean up the file we just wrote
                    try:
                        os.unlink(temp_check_path)
                    except OSError:
                        pass
                    raise ValueError(f"Invalid file type: {mime_error}")
            finally:
                if os.path.exists(temp_check_path):
                    try:
                        os.unlink(temp_check_path)
                    except OSError:
                        pass

        # Write to final destination
        target = dataset_dir / safe_filename
        target.write_bytes(content)

        _log.debug(
            "[%s] File saved: %s (%.1f MB, MIME: %s)",
            sanitize_key(dataset_id),
            sanitize_key(safe_filename),
            len(content) / 1024 / 1024,
            mime_type if "mime_type" in locals() else "unknown",
        )
        return target

    def process_file(self, dataset_id: str, file_path: Path) -> dict:
        result = run_deepdoc_ingestion(
            file_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            max_tokens_per_chunk=self.max_tokens_per_chunk,
        )

        chunks = result["chunks"]
        self._persist_chunks(
            dataset_id=dataset_id, source_file=file_path.name, chunks=chunks
        )

        return {
            "dataset_id": dataset_id,
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "chars": len(result["text"]),
            "chunks": len(chunks),
            "collection": dataset_id,
            "document_analysis": result["document_analysis"],
            "sample_chunk_analysis": result["chunk_analysis"][:5],
            "quality_report": result["quality_report"],
            "layout_summary": result.get("layout_summary"),
            "extraction": result["extraction"],
        }

    def _persist_chunks(
        self, dataset_id: str, source_file: str, chunks: list[tuple[str, list]]
    ) -> None:
        out_dir = _resolve_dataset_dir(settings.chunk_root, dataset_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, (chunk_text, _) in enumerate(chunks):
            chunk_file = out_dir / f"{Path(source_file).stem}.chunk-{idx:04d}.txt"
            chunk_file.write_text(chunk_text, encoding="utf-8")
