from fastapi import APIRouter, HTTPException, status

from backend.clients.ollama_client import OllamaClient
from backend.config import settings
from backend.schemas.ollama import (
    ModelSelectionRequest,
    ModelValidationResponse,
    OllamaModelListResponse,
    OllamaModelDetails,
)

router = APIRouter(prefix="/ollama", tags=["ollama"])
client = OllamaClient(base_url=settings.ollama_base_url)


@router.get("/models", response_model=OllamaModelListResponse)
async def list_local_models() -> OllamaModelListResponse:
    """
    Scan Ollama local registry and return existing model names plus full detail objects.

    The `model_details` field contains per-model metadata (size, family, parameter_size,
    quantization, modified date) — used by the /models page.
    """
    try:
        names, details = await client.list_models()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot reach Ollama at {settings.ollama_base_url}: {exc}",
        ) from exc

    if not names:
        return OllamaModelListResponse(
            models=[],
            has_models=False,
            message="No local Ollama models found. Please download at least one embedding model and one chat model.",
        )

    return OllamaModelListResponse(
        models=names,
        has_models=True,
        message="Local models discovered.",
        model_details=[
            OllamaModelDetails(
                name=d.name,
                model=d.name,
                size=d.size_bytes,
                modified_at=d.modified_at,
                digest=d.digest,
                details=d.details,
            )
            for d in details
        ],
    )


@router.post("/validate", response_model=ModelValidationResponse)
async def validate_model_selection(payload: ModelSelectionRequest) -> ModelValidationResponse:
    """
    Validate that selected embedding/chat models already exist locally in Ollama.
    """
    try:
        valid, available, missing = await client.validate_models(
            embedding_model=payload.embedding_model,
            chat_model=payload.chat_model,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot validate against Ollama at {settings.ollama_base_url}: {exc}",
        ) from exc

    if not valid:
        return ModelValidationResponse(
            valid=False,
            available_models=available,
            missing_models=missing,
            message=(
                "Some selected models are missing locally. "
                "Please run 'ollama pull <model>' on your machine first."
            ),
        )

    return ModelValidationResponse(
        valid=True,
        available_models=available,
        missing_models=[],
        message="Model selection is valid and available locally.",
    )
