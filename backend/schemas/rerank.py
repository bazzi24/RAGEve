from pydantic import BaseModel, Field


class RerankerEntrySchema(BaseModel):
    id: str = Field(description="HuggingFace model identifier")
    display_name: str = Field(description="Human-readable name")
    description: str = Field(description="What the model is and when to use it")
    approx_size_mb: int = Field(description="Approximate download size in MB")


class RerankersResponse(BaseModel):
    rerankers: list[RerankerEntrySchema] = Field(default_factory=list)
    total: int = 0
