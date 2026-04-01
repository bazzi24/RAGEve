from pydantic import BaseModel, Field


class OllamaModelDetails(BaseModel):
    """Per-model metadata from Ollama's /api/tags."""

    name: str
    model: str = Field(description="Full model name (same as name in most cases)")
    modified_at: str | None = Field(default=None, description="ISO timestamp of last modification")
    size: int = Field(default=0, description="Size in bytes")
    digest: str | None = Field(default=None, description="SHA256 digest")
    details: dict = Field(
        default_factory=dict,
        description="Ollama-internal details: format, family, parameter_size, quantization_level",
    )

    @property
    def size_mb(self) -> float:
        return self.size / (1024**2)

    @property
    def parameter_size(self) -> str | None:
        return self.details.get("parameter_size")

    @property
    def quantization(self) -> str | None:
        return self.details.get("quantization_level")

    @property
    def family(self) -> str | None:
        return self.details.get("family")


class OllamaModelListResponse(BaseModel):
    """Ollama model list — includes both a flat name list (backward-compatible)
    and per-model detail objects for the Models page."""

    models: list[str] = Field(default_factory=list, description="Flat list of model names (backward-compatible)")
    has_models: bool = Field(default=False)
    message: str | None = Field(default=None)
    model_details: list[OllamaModelDetails] = Field(
        default_factory=list,
        description="Full per-model detail objects from Ollama /api/tags",
    )


class ModelSelectionRequest(BaseModel):
    embedding_model: str
    chat_model: str


class ModelValidationResponse(BaseModel):
    valid: bool
    available_models: list[str] = Field(default_factory=list)
    missing_models: list[str] = Field(default_factory=list)
    message: str
