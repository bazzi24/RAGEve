from pydantic import BaseModel, Field


class AgentConfigSchema(BaseModel):
    system_prompt: str = Field(default="You are a helpful AI assistant.")
    dataset_id: str
    embedding_model: str
    chat_model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=5, ge=1, le=20)


class AgentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="")
    config: AgentConfigSchema


class AgentUpdateConfigSchema(BaseModel):
    """Partial-update schema — all fields optional so omitted ones preserve existing values."""

    system_prompt: str | None = Field(default=None)
    dataset_id: str | None = Field(default=None)
    embedding_model: str | None = Field(default=None)
    chat_model: str | None = Field(default=None)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AgentUpdate(BaseModel):
    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    config: AgentUpdateConfigSchema | None = Field(default=None)


class AgentResponse(BaseModel):
    agent_id: str
    name: str
    description: str
    config: AgentConfigSchema
    created_at: str
    updated_at: str


class AgentListResponse(BaseModel):
    agents: list[AgentResponse] = Field(default_factory=list)
    total: int = 0
