from __future__ import annotations

import logging
from pathlib import Path

from fastapi import UploadFile

from backend.config import settings
from rag.ingestion.pipeline import SUPPORTED_EXTENSIONS, run_deepdoc_ingestion

_log = logging.getLogger("backend.services.file_processor")


class FileProcessorService:
    def __init__(self) -> None:
        self.chunk_size = settings.default_chunk_size
        self.chunk_overlap = settings.default_chunk_overlap
        self.max_tokens_per_chunk = settings.default_max_tokens_per_chunk

    async def save_upload(self, dataset_id: str, upload: UploadFile) -> Path:
        dataset_dir = settings.upload_root / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        target = dataset_dir / upload.filename
        content = await upload.read()
        target.write_bytes(content)

        _log.debug("[%s] File saved: %s (%.1f KB)", dataset_id, upload.filename, len(content) / 1024)
        return target

    def process_file(self, dataset_id: str, file_path: Path) -> dict:
        result = run_deepdoc_ingestion(
            file_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            max_tokens_per_chunk=self.max_tokens_per_chunk,
        )

        chunks = result["chunks"]
        self._persist_chunks(dataset_id=dataset_id, source_file=file_path.name, chunks=chunks)

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

    def _persist_chunks(self, dataset_id: str, source_file: str, chunks: list[str]) -> None:
        out_dir = settings.chunk_root / dataset_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, chunk in enumerate(chunks):
            chunk_file = out_dir / f"{Path(source_file).stem}.chunk-{idx:04d}.txt"
            chunk_file.write_text(chunk, encoding="utf-8")
