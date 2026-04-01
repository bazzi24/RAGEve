from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx


@dataclass
class OllamaModelInfo:
    """A single model's info, surfaced in the models page."""

    name: str
    size_bytes: int
    modified_at: str | None
    digest: str | None
    details: dict[str, Any]

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024**2)

    @property
    def parameter_size(self) -> str | None:
        return self.details.get("parameter_size")

    @property
    def quantization(self) -> str | None:
        return self.details.get("quantization_level")

    @property
    def family(self) -> str | None:
        return self.details.get("family")


def _normalize_model_name(name: str) -> str:
    """Strip the :latest tag (or any :tag suffix) for comparison purposes."""
    if ":" in name:
        base, _, _ = name.partition(":")
        return base
    return name


def _matches_available(requested: str, available: list[str]) -> bool:
    """
    Check if a requested model matches any available model.
    Handles both exact match and tag-stripped match.
    e.g. requested='llama3.2' matches available='llama3.2:latest'
    """
    if requested in available:
        return True
    # Try with :latest appended (Ollama resolves bare names to :latest automatically)
    with_latest = f"{requested}:latest"
    if with_latest in available:
        return True
    # Try stripping tag from available and comparing base names
    requested_base = _normalize_model_name(requested)
    for avail in available:
        avail_base = _normalize_model_name(avail)
        if requested_base == avail_base:
            return True
    return False


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def list_models(self) -> tuple[list[str], list[OllamaModelInfo]]:
        """
        Fetch local Ollama models from /api/tags.

        Returns:
            A tuple of:
            - unique sorted list of model names (backward-compatible)
            - list of OllamaModelInfo detail objects for the models page
        """
        url = f"{self.base_url}/api/tags"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()

        models_payload = payload.get("models", [])
        names: list[str] = []
        details: list[OllamaModelInfo] = []

        for model in models_payload:
            name = model.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
                details.append(
                    OllamaModelInfo(
                        name=name.strip(),
                        size_bytes=model.get("size", 0),
                        modified_at=model.get("modified_at"),
                        digest=model.get("digest"),
                        details=model.get("details", {}),
                    )
                )

        return sorted(set(names)), details

    async def validate_models(self, embedding_model: str, chat_model: str) -> tuple[bool, list[str], list[str]]:
        """
        Validate that the requested embedding and chat models are available.
        Uses tag-normalized comparison so 'llama3.2' matches 'llama3.2:latest'.
        """
        available, _ = await self.list_models()
        missing: list[str] = []
        for model in [embedding_model, chat_model]:
            if not _matches_available(model, available):
                missing.append(model)

        return (len(missing) == 0, available, missing)

