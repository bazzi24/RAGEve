from pydantic import BaseModel, Field


class HuggingFaceInstructionsRequest(BaseModel):
    dataset_id: str = Field(..., description="HuggingFace dataset id")


class HuggingFaceInstructionsResponse(BaseModel):
    dataset_id: str
    download_command: str
    expected_local_path: str
    supported_splits: list[str] = Field(default_factory=list)
    supported_file_formats: list[str] = Field(default_factory=list)
    column_preview: dict[str, str] = Field(default_factory=dict)
    estimated_size: str | None = None
    message: str


class HuggingFaceDownloadRequest(BaseModel):
    dataset_id: str = Field(..., description="HuggingFace dataset id")
    split: str | None = Field(default=None, description="Optional split to download")
    config: str | None = Field(
        default=None,
        description="Dataset config name (required for multi-config datasets like wikitext). "
        "Use GET /datasets/hf/preview/{id} to discover available configs.",
    )
    auto_ingest: bool = Field(default=False, description="Auto-run ingestion after download.")
    row_limit: int | None = Field(
        default=None, ge=1, description="Row cap for auto-ingest.",
    )
    batch_size: int | None = Field(
        default=None, ge=1, le=2048, description="Embedding batch size for auto-ingest.",
    )
    chunk_overlap: int | None = Field(
        default=None, ge=0, le=500, description="Chunk overlap for auto-ingest.",
    )
    max_tokens_per_chunk: int | None = Field(
        default=None, ge=50, le=2000, description="Max tokens per chunk.",
    )
    text_columns: list[str] = Field(
        default_factory=list,
        description="List of column names to combine for auto-ingest. "
        "Columns are formatted as 'Key: Value' lines.",
    )
    metadata_columns: list[str] = Field(default_factory=list)
    ingest_split: str | None = Field(default=None, description="Split for auto-ingest.")


class HuggingFaceDownloadResponse(BaseModel):
    dataset_id: str
    status: str
    message: str


class HuggingFaceDownloadStatusResponse(BaseModel):
    dataset_id: str
    status: str  # queued | downloading | cancelling | cancelled | completed | failed
    progress: int = Field(default=0, ge=0, le=100)
    message: str = ""
    error: str | None = None
    local_path: str | None = None
    started_at: str | None = None
    updated_at: str | None = None
    rows_downloaded: int | None = None
    splits_downloaded: list[str] = Field(default_factory=list)
    bytes_downloaded: int | None = Field(default=None, description="Bytes downloaded so far")
    total_bytes: int | None = Field(default=None, description="Total bytes to download (estimated)")
    config: str | None = None
    auto_ingest: bool = False
    ingest_status: str | None = None  # idle | ingesting | completed | failed
    ingest_message: str | None = None
    ingest_error: str | None = None
    ingested: bool = False
    suggested_text_column: str | None = None
    columns: dict[str, str] = Field(default_factory=dict)


class HuggingFacePreviewResponse(BaseModel):
    dataset_id: str
    full_dataset_id: str = Field(
        description="Original dataset ID as entered (may contain '/', e.g. 'author/repo'). "
        "Use this for display and API calls — never the filesystem-safe version.",
    )
    configs: list[str] = Field(
        default_factory=list,
        description="Available config names. Empty if dataset has no configs.",
    )
    default_config: str | None = Field(
        default=None,
        description="Recommended config to use (first in list, or None if no configs).",
    )
    description: str | None = None
    downloads: int | None = Field(
        default=None,
        description="Total download count from HuggingFace Hub.",
    )
    likes: int | None = Field(
        default=None,
        description="Total likes/favorites count from HuggingFace Hub.",
    )
    estimated_size_bytes: int | None = None
    estimated_size_human: str | None = None
    splits: list[str] = Field(default_factory=list)
    columns: dict[str, str] = Field(default_factory=dict)
    source: str = Field(
        default="datasets-server",
        description="Provenance of the metadata: 'hf-hub' (full card via huggingface_hub), "
        "'datasets-server' (datasets-server preview API), or 'hub-api' (Hub REST fallback).",
    )
    # ── Rich metadata from huggingface_hub card ─────────────────────────────────
    tags: list[str] = Field(
        default_factory=list,
        description="Curated dataset tags (language, task, size tier, etc.) from card_data.",
    )
    language: list[str] = Field(
        default_factory=list,
        description="Languages detected from card metadata.",
    )
    license: str | None = Field(
        default=None,
        description="License identifier (e.g. 'apache-2.0') from card_data.",
    )
    paper_url: str | None = Field(
        default=None,
        description="URL of the associated research paper.",
    )
    card_data: dict | None = Field(
        default=None,
        description="Full YAML card metadata (tasks, leaderboard, benchmark info, etc.).",
    )
    readme_html: str | None = Field(
        default=None,
        description="README.md content rendered as escaped HTML (capped at 4 000 chars).",
    )
    leaderboard: dict | None = Field(
        default=None,
        description="Leaderboard / benchmark info extracted from card_data.",
    )
    source_detail: str = Field(
        default="HuggingFace Hub",
        description="Human-readable provenance label shown in the UI.",
    )
    message: str
    valid: bool = True


class DiscoveredDataset(BaseModel):
    local_path: str
    dataset_id: str
    splits: list[str] = Field(default_factory=list)
    file_formats: list[str] = Field(default_factory=list)
    file_count: int = 0
    total_size_bytes: int = 0
    readable_columns: list[str] = Field(default_factory=list)
    description: str | None = None
    is_ingested: bool = Field(default=False, description="True if dataset is already indexed in Qdrant.")


class HuggingFaceDiscoveryResponse(BaseModel):
    scan_root: str
    datasets: list[DiscoveredDataset] = Field(default_factory=list)
    total_found: int = 0
    message: str


class HuggingFaceRegisterRequest(BaseModel):
    local_path: str = Field(...)
    dataset_id: str = Field(...)
    split: str = Field(default="train")
    text_columns: list[str] = Field(default_factory=list, description="Column names to combine for embedding.")
    metadata_columns: list[str] = Field(default_factory=list)


class HuggingFaceRegisterResponse(BaseModel):
    dataset_id: str
    registered: bool
    collection: str
    estimated_rows: int | None = None
    splits_available: list[str]
    columns_available: list[str]
    message: str


class HuggingFaceStatusTextsResponse(BaseModel):
    dataset_id: str
    display_status: str
    display_message: str


class DatasetIngestStatus(BaseModel):
    """Per-dataset ingestion status returned by GET /datasets/hf/status."""

    dataset_id: str
    local_path: str
    splits: list[str] = Field(default_factory=list)
    file_count: int = 0
    total_size_bytes: int = 0
    readable_columns: list[str] = Field(default_factory=list)
    description: str | None = None
    is_ingested: bool = Field(
        default=False,
        description=(
            "True when this dataset has been ingested into Qdrant "
            "(its collection exists and contains at least one point)."
        ),
    )
    points_count: int = Field(
        default=0,
        description="Number of vector chunks stored in Qdrant for this dataset.",
    )


class HuggingFaceStatusResponse(BaseModel):
    """Response for GET /datasets/hf/status."""

    datasets: list[DatasetIngestStatus]
    total: int
    message: str
