from pydantic import BaseModel, Field


class ProcessedFileResponse(BaseModel):
    dataset_id: str
    filename: str
    extension: str
    chars: int
    chunks: int
    collection: str
    document_analysis: dict = Field(default_factory=dict)
    sample_chunk_analysis: list[dict] = Field(default_factory=list)
    quality_report: dict = Field(default_factory=dict)
    layout_summary: dict | None = Field(default=None)
    extraction: dict = Field(default_factory=dict)


class UploadSummaryResponse(BaseModel):
    dataset_id: str
    files: list[ProcessedFileResponse] = Field(default_factory=list)
