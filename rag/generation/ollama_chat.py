from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

_log = logging.getLogger("rag.generation.ollama_chat")

# ── Circuit breaker ────────────────────────────────────────────────────────────

@dataclass
class _CircuitBreaker:
    """
    Lightweight fail-fast circuit breaker for the Ollama chat service.

    Tracks consecutive failures and opens the circuit after FAIL_THRESHOLD
    consecutive errors.  While open it raises 503 immediately (no attempt
    to contact Ollama), preventing cascading failures under sustained load.
    After COOLDOWN_SECS it enters half-open state and allows one probe
    request through; that probe either resets the circuit (on success) or
    reopens it (on failure).
    """
    failures: int = 0
    open_until: float = 0.0   # 0 = closed, > now = open

    FAIL_THRESHOLD: int = 5    # consecutive failures before opening
    COOLDOWN_SECS: float = 30.0

    def check(self) -> None:
        """Raise httpx.HTTPStatusError if the circuit is open."""
        if self.open_until > time.monotonic():
            raise httpx.HTTPStatusError(
                "Ollama circuit breaker is open",
                request=None,
                response=httpx.Response(503, json={"error": "Ollama unavailable — service is temporarily overloaded. Please retry."}),
            )

    def record_success(self) -> None:
        """Reset failure counter on a successful call."""
        self.failures = 0

    def record_failure(self) -> None:
        """Increment failures and open the circuit if threshold is reached."""
        self.failures += 1
        if self.failures >= self.FAIL_THRESHOLD:
            self.open_until = time.monotonic() + self.COOLDOWN_SECS
            _log.warning(
                "Ollama circuit breaker opened after %d consecutive failures; "
                "will retry after %.0f s",
                self.failures,
                self.COOLDOWN_SECS,
            )


# Module-level singleton — shared across all OllamaChat instances.
_circuit = _CircuitBreaker()

# ── Chat models ────────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: str   # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatResponse:
    message: ChatMessage
    done: bool
    total_duration_ns: int | None = None
    model: str | None = None


# ── Ollama chat ────────────────────────────────────────────────────────────────

class OllamaChat:
    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ── retry helper ─────────────────────────────────────────────────────────

    @staticmethod
    async def _with_retry(fn, *args, **kwargs) -> Any:
        """
        Retry ``fn`` up to 3 times on httpx.ReadTimeout with exponential back-off
        (30 s → 60 s → 90 s).  All other exceptions (TimeoutException,
        HTTPStatusError, etc.) propagate immediately without retry.
        """
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                return await fn(*args, **kwargs)
            except httpx.ReadTimeout as exc:
                last_exc = exc
                if attempt < 3:
                    delay = 30.0 * attempt
                    _log.warning(
                        "Ollama read timeout (attempt %d/3); retrying in %.0f s…",
                        attempt,
                        delay,
                    )
                    await asyncio.sleep(delay)
            except (httpx.TimeoutException, httpx.HTTPStatusError):
                raise
        # Only reached if ReadTimeout fired 3 times
        raise last_exc

    # ── non-streaming ───────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        num_ctx: int = 4096,
    ) -> ChatResponse:
        _circuit.check()

        ollama_messages: list[dict[str, str]] = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})
        ollama_messages.extend({"role": m.role, "content": m.content} for m in messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }

        url = f"{self.base_url}/api/chat"

        async def _call() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()

        try:
            data = await self._with_retry(_call)
            _circuit.record_success()
        except Exception:
            _circuit.record_failure()
            raise

        raw_msg = data.get("message", {})
        return ChatResponse(
            message=ChatMessage(role=raw_msg.get("role", "assistant"), content=raw_msg.get("content", "")),
            done=data.get("done", True),
            total_duration_ns=data.get("total_duration"),
            model=data.get("model"),
        )

    # ── streaming ─────────────────────────────────────────────────────────

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
        temperature: float = 0.7,
        num_ctx: int = 4096,
    ) -> AsyncIterator[str]:
        _circuit.check()

        ollama_messages: list[dict[str, str]] = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})
        ollama_messages.extend({"role": m.role, "content": m.content} for m in messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }

        url = f"{self.base_url}/api/chat"
        import json as _json

        async def _stream_response():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk: dict[str, Any] = _json.loads(line)
                        except Exception:
                            continue

                        delta = chunk.get("message", {}).get("content", "")
                        if delta:
                            yield delta

                        if chunk.get("done"):
                            break

        # Retry loop: each iteration yields from a fresh async generator.
        # httpx.ReadTimeout → reconnect and retry; all other errors propagate.
        for attempt in range(1, 4):
            try:
                # _stream_response() is an async generator — call it directly (no await)
                # then iterate.  On ReadTimeout the loop retries with a fresh generator.
                async for token in _stream_response():
                    yield token
                # Stream completed without timeout → we're done.
                _circuit.record_success()
                break
            except httpx.ReadTimeout:
                if attempt < 3:
                    delay = 30.0 * attempt
                    _log.warning(
                        "Ollama stream read timeout (attempt %d/3); reconnecting in %.0f s…",
                        attempt,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    _circuit.record_failure()
                    raise
            except httpx.TimeoutException:
                _circuit.record_failure()
                raise
            except httpx.HTTPStatusError:
                _circuit.record_failure()
                raise
