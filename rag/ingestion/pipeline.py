from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from rag.chunking.adaptive import adaptive_chunk_text
from rag.deepdoc.analyzer import analyze_chunks, analyze_text
from rag.deepdoc.layout_parser import layout_to_readable_text, parse_pdf_layout
from rag.deepdoc.quality_scorer import ChunkProfile, score_and_select_profile
from rag.ingestion.extractors import Extractors
from rag.ingestion.doc_converter import ConversionResult


SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".xlsx", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def extract_text(file_path: Path) -> tuple[str, dict]:
    """
    Extract text from a file, routing to the correct extractor.
    Returns (extracted_text, extraction_metadata).
    """
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        text = Extractors.from_pdf(file_path)
        return text, {"extractor": "pymupdf", "layout_aware": False}

    if ext == ".doc":
        text, conv_result = Extractors.from_doc(file_path)
        meta = {
            "extractor": "doc_converter",
            "converter": conv_result.converter.value,
            "converted": conv_result.success,
            "message": conv_result.message,
            "error": conv_result.error,
        }
        return text, meta

    if ext == ".docx":
        text = Extractors.from_docx(file_path)
        return text, {"extractor": "python-docx", "layout_aware": False}

    if ext == ".xlsx":
        text = Extractors.from_xlsx(file_path)
        return text, {"extractor": "pandas", "layout_aware": False}

    if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
        text = Extractors.from_image(file_path)
        return text, {"extractor": "pytesseract-ocr", "layout_aware": False}

    raise ValueError(f"Unsupported file extension: {ext}")


def run_deepdoc_ingestion(
    file_path: Path,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    max_tokens_per_chunk: int | None = None,
    force_profile: ChunkProfile | None = None,
) -> dict:
    """
    Full deepdoc ingestion pipeline for a single file.

    Steps:
      1. Extract text using format-specific extractor
      2. If PDF: run layout-aware parsing
      3. Compute quality signals and score
      4. Select chunk profile (auto or forced)
      5. Run adaptive chunking
      6. Return structured report with analysis

    Args:
      file_path: path to uploaded file
      chunk_size / chunk_overlap / max_tokens_per_chunk: override auto settings
      force_profile: bypass auto-selection and force a specific chunk profile
    """
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    # 1. Text extraction + metadata
    text, extraction_meta = extract_text(file_path)

    # 2. Layout parsing (PDF only)
    layouts = []
    raw_text = text
    if file_path.suffix.lower() == ".pdf":
        layouts = parse_pdf_layout(file_path)
        raw_text = layout_to_readable_text(layouts)
        extraction_meta["layout_aware"] = True

    # 3. Quality scoring
    quality_report = score_and_select_profile(raw_text)
    signals = quality_report.signals

    # 4. Chunk profile selection
    selected_profile = force_profile or quality_report.profile.profile
    config = quality_report.profile

    effective_chunk_size = chunk_size if chunk_size is not None else config.chunk_size
    effective_overlap = chunk_overlap if chunk_overlap is not None else config.chunk_overlap
    effective_tokens = max_tokens_per_chunk if max_tokens_per_chunk is not None else config.max_tokens_per_chunk

    # 5. Adaptive chunking
    chunks = adaptive_chunk_text(
        raw_text,
        profile=selected_profile,
        override_size=effective_chunk_size,
        override_overlap=effective_overlap,
        override_tokens=effective_tokens,
    )

    # 6. Chunk analysis
    chunk_analysis = analyze_chunks(chunks)

    # 7. Document-level analysis
    doc_analysis = analyze_text(raw_text)

    # 8. Build quality report dict
    quality_dict = {
        "quality_score": quality_report.quality_score,
        "selected_profile": selected_profile.value,
        "profile_reason": config.reason,
        "signals": {
            "alpha_ratio": signals.alpha_ratio,
            "ocr_noise_ratio": signals.ocr_noise_ratio,
            "broken_line_ratio": signals.broken_line_ratio,
            "header_footer_ratio": signals.header_footer_ratio,
            "table_density": signals.table_density,
            "avg_sentence_length": signals.avg_sentence_length,
            "language_script_changes": signals.language_script_changes,
            "repeated_word_ratio": signals.repeated_word_ratio,
            "code_delimiter_ratio": signals.code_delimiter_ratio,
            "issue_tags": signals.issue_tags,
        },
    }

    # 9. Layout summary (PDF only)
    layout_summary = None
    if layouts:
        block_counts: dict[str, int] = {}
        for page in layouts:
            for block in page.blocks:
                block_counts[block.block_type.value] = block_counts.get(block.block_type.value, 0) + 1

        layout_summary = {
            "pages": len(layouts),
            "total_blocks": sum(block_counts.values()),
            "blocks_by_type": block_counts,
        }

    return {
        "text": raw_text,
        "chunks": chunks,
        "extraction": extraction_meta,
        "document_analysis": doc_analysis,
        "chunk_analysis": chunk_analysis,
        "quality_report": quality_dict,
        "layout_summary": layout_summary,
        "chunk_params": {
            "chunk_size": effective_chunk_size,
            "chunk_overlap": effective_overlap,
            "max_tokens_per_chunk": effective_tokens,
        },
    }
