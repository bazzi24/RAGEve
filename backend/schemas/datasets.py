from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    chunk_size: int | None = Field(default=None, description="Override auto-selected chunk size")
    chunk_overlap: int | None = Field(default=None)
    max_tokens_per_chunk: int | None = Field(default=None)
    force_profile: str | None = Field(
        default=None,
        description="Force a specific chunk profile: clean_text, ocr_noisy, table_heavy, code_mixed, general",
    )
    overwrite: bool = Field(default=False, description="Re-ingest and overwrite existing chunks")


class IngestResponse(BaseModel):
    dataset_id: str
    collection: str
    chunks_embedded: int
    total_chars: int
    quality_report: dict = Field(default_factory=dict)
    message: str


class DatasetInfo(BaseModel):
    dataset_id: str
    collection: str
    chunks_count: int
    vector_size: int
    status: str


class DatasetListResponse(BaseModel):
    datasets: list[DatasetInfo] = Field(default_factory=list)
    total: int = 0
    skip: int = 0
    limit: int = 50


class CollectionDeleteResponse(BaseModel):
    dataset_id: str
    deleted: bool
    message: str


# ------------------------------------------------------------------
# Background ingest schemas
# ------------------------------------------------------------------

class IngestSubmitResponse(BaseModel):
    ingest_id: str
    dataset_id: str
    message: str


class IngestStatusResponse(BaseModel):
    ingest_id: str
    dataset_id: str
    status: str  # "queued" | "running" | "completed" | "failed" | "cancelled"
    progress: int = 0        # 0-100 overall
    current_stage: str = "idle"
    message: str = ""
    # Per-file detail
    file_index: int = 0      # 1-based
    file_total: int = 0
    current_file: str = ""
    chunks_done: int = 0
    chunks_total: int = 0
    started_at: str | None = None   # ISO-8601
    completed_at: str | None = None  # ISO-8601
    error: str | None = None
    # Final results (filled when status == "completed")
    files: list[dict] = Field(default_factory=list)
