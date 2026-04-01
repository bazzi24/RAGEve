"""
Central logging configuration for the Mini RAG Platform.

Sets up RotatingFileHandler for each subsystem logger so that logs persist
across server restarts and are separated by concern for easy grep/tail.

Call setup_logging(log_dir) once at app startup (before any routes are used).
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Log format shared across all handlers.
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _make_handler(log_path: Path, level: int = logging.INFO) -> RotatingFileHandler:
    """Create a rotating file handler that appends to the given path."""
    handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,               # keep 5 backups → max 60 MB per log
    )
    handler.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    handler.setFormatter(formatter)
    return handler


def setup_logging(log_dir: Path) -> None:
    """
    Initialise file-based logging for all application loggers.

    Parameters
    ----------
    log_dir:
        Directory where log files are written. Created if it does not exist.
        Typically: ``settings.data_root / "logs"``.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Define which loggers write to which files ────────────────────────────
    # Format: logger_name (matches `logging.getLogger(name)`) → list of filenames
    logger_files: dict[str, list[str]] = {
        # FastAPI app lifecycle and HTTP access
        "app": ["app.log"],
        # HTTP access logs (captured separately from app.log for easy filtering)
        "uvicorn.access": ["access.log"],
        # File upload + ingest pipeline
        "backend.api.routes.datasets": ["ingest.log"],
        # HuggingFace dataset ingestion
        "rag.ingestion.hf_ingestion": ["hf-ingest.log"],
        # HF dataset download worker
        "backend.api.routes.huggingface": ["hf-download.log"],
        # Ollama embedding (GPU detection, normalization, retries)
        "rag.embedding.ollama_embedder": ["ollama.log"],
        # Qdrant collection + upsert + search operations
        "rag.storage.qdrant_store": ["qdrant.log"],
        # Cross-encoder reranking (model load, scoring)
        "rag.retrieval.cross_encoder_reranker": ["rerank.log"],
        # RAG retrieval pipeline (retrieval mode, chunk counts, reranking)
        "rag.retrieval.rag_pipeline": ["queries.log"],
        # LLM generation (model used, token count, latency)
        "rag.generation.ollama_chat": ["queries.log"],
        # File processor (save/load operations)
        "backend.services.file_processor": ["ingest.log"],
        # Ingestion service (extract/chunk/embed/upsert stages)
        "backend.services.ingestion_service": ["ingest.log"],
    }

    # Capture log levels per logger (defaults to INFO; DEBUG loggers below)
    log_levels: dict[str, int] = {
        "rag.retrieval.rag_pipeline": logging.DEBUG,
        "rag.retrieval.cross_encoder_reranker": logging.DEBUG,
    }

    for logger_name, filenames in logger_files.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_levels.get(logger_name, logging.INFO))
        # Prevent propagation to the root handler (which would double-print to stdout)
        logger.propagate = False

        for fname in filenames:
            handler = _make_handler(log_dir / fname)
            logger.addHandler(handler)

    # Also attach a rotating handler to the uvicorn error logger so that
    # server-side exceptions appear in app.log / access.log as well.
    for name in ("uvicorn.error", "uvicorn"):
        err_logger = logging.getLogger(name)
        err_logger.propagate = False
        err_logger.addHandler(_make_handler(log_dir / "app.log"))

    # ── Root logger: emit everything at WARNING so unconfigured modules are
    #    at least captured in one place (app.log).
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.propagate = False
    if not root.handlers:
        root.addHandler(_make_handler(log_dir / "app.log"))

    logging.info("Logging initialized — logs directory: %s", log_dir)
