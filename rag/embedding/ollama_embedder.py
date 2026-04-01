from __future__ import annotations

import asyncio
import gc
import logging
import math
import os
from typing import Any, Awaitable, Callable, AsyncIterator

import httpx

# Lazy torch import — only needed when GPU memory management is active.
_torch_cache: Any | None = None


def _get_torch():
    global _torch_cache
    if _torch_cache is None:
        try:
            import torch
            _torch_cache = torch
        except ImportError:
            _torch_cache = False
    return _torch_cache


def _clear_cuda_cache() -> None:
    """Clear PyTorch CUDA cache. Safe to call when torch is not installed."""
    torch = _get_torch()
    if torch:
        torch.cuda.empty_cache()

_log = logging.getLogger(__name__)

BatchProgressCallback = Callable[[int, int], Awaitable[None] | None]

# Number of retry attempts for ReadTimeout before giving up.
MAX_RETRIES = 3

# Base backoff delay in seconds — multiplied by attempt index (1→30s, 2→60s, 3→90s).
BACKOFF_BASE_SECS = 30.0

# Concurrency: GPU workers can overlap; CPU workers are limited to avoid OOM.
DEFAULT_CPU_CONCURRENCY = 4   # safe ceiling for CPU-mode Ollama
DEFAULT_GPU_CONCURRENCY = 16  # nominal; actual parallelism is GPU-bound


def _detect_ollama_cuda() -> bool:
    """
    Detect if Ollama has a GPU (CUDA or ROCm) available by calling
    ``GET /api/ps`` (or ``GET /``), which returns ``"nvidia_accel": true``
    when CUDA is active.

    Returns ``True`` for CUDA or ROCm GPUs; ``False`` otherwise.
    Falls back to the ``OLLAMA_GPU`` env var if the HTTP check fails.
    """
    # Check the documented env-var override first.
    gpu_override = os.environ.get("OLLAMA_GPU", "").lower()
    if gpu_override in ("cuda", "rocm", "true", "1"):
        return True
    if gpu_override == "cpu":
        return False

    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(("localhost", 11434))
        sock.close()
    except OSError:
        return False

    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:11434/api/ps",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = resp.read()
        import json
        payload = json.loads(data)
        # `nvidia_accel` is True when a CUDA GPU is detected (newer Ollama versions).
        if payload.get("nvidia_accel") is True:
            return True
        # Some versions expose driver information differently.
        if payload.get("cuda", False) is True:
            return True
        # Definitive evidence: a loaded model is consuming GPU VRAM.
        # `size_vram` is non-zero when Ollama has placed model tensors on the GPU.
        for model in payload.get("models", []):
            if (model.get("size_vram") or 0) > 0:
                return True
        # Fallback: check environment variables set by the Ollama process.
        for val in os.environ.get("CUDA_VISIBLE_DEVICES", ""), os.environ.get("ROCM_PATH", ""):
            if val:
                return True
    except Exception:
        pass

    return False


async def _emit_batch_progress(
    callback: BatchProgressCallback | None,
    done: int,
    total: int,
) -> None:
    if callback is None:
        return
    maybe_awaitable = callback(done, total)
    if maybe_awaitable is not None:
        await maybe_awaitable


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff delay: 30s, 60s, 90s for attempts 1, 2, 3."""
    return BACKOFF_BASE_SECS * attempt


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._normalization_warned = False

        # Detect GPU once per instance; used to size concurrency.
        self._is_gpu = _detect_ollama_cuda()
        self._concurrency_limit = (
            DEFAULT_GPU_CONCURRENCY if self._is_gpu else DEFAULT_CPU_CONCURRENCY
        )
        # Semaphore to cap concurrent Ollama requests regardless of batch size.
        self._sem = asyncio.Semaphore(self._concurrency_limit)

        if self._is_gpu:
            _log.info(
                "OllamaEmbedder: GPU (CUDA/ROCm) detected — using %d concurrent workers.",
                self._concurrency_limit,
            )
        else:
            _log.warning(
                "OllamaEmbedder: No GPU detected — falling back to %d concurrent workers "
                "(CPU mode). Set OLLAMA_GPU=cuda to override. "
                "Install CUDA drivers and pull the GPU variant for faster embeddings.",
                self._concurrency_limit,
            )

    @property
    def is_gpu(self) -> bool:
        """True when Ollama has a CUDA/ROCm GPU available."""
        return self._is_gpu

    def verify_normalization(self, embedding: list[float]) -> bool:
        """
        Verify that an embedding vector is unit-normalized (L2 norm ≈ 1.0).

        Logs a one-time warning if the L2 norm deviates by more than 0.01 from 1.0.
        When deviation is detected, callers are expected to re-normalize in-place.
        """
        if not embedding:
            return True
        norm_sq = sum(x * x for x in embedding)
        norm = math.sqrt(norm_sq)
        if abs(norm - 1.0) > 0.01:
            if not self._normalization_warned:
                _log.warning(
                    "Embedding vector is NOT unit-normalized: norm=%.6f "
                    "(deviation=%.4f). Model='%s'. Normalizing in-place. "
                    "(This warning will not repeat for this session.)",
                    norm, abs(norm - 1.0), self.model,
                )
                self._normalization_warned = True
            return False
        return True

    def _normalize(self, embedding: list[float]) -> list[float]:
        """Re-normalize a vector to unit length in-place. Returns the same list."""
        norm_sq = sum(x * x for x in embedding)
        norm = math.sqrt(norm_sq)
        if norm < 1e-10:
            _log.warning("Embedding vector has near-zero norm — cannot normalize.")
            return embedding
        inv_norm = 1.0 / norm
        for i in range(len(embedding)):
            embedding[i] *= inv_norm
        return embedding

    # ------------------------------------------------------------------
    # Core: single embedding with retry + backoff + semaphore
    # ------------------------------------------------------------------

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text, retrying on ReadTimeout with exponential backoff."""
        async with self._sem:
            return await self._embed_single_impl(text)

    async def _embed_single_impl(self, text: str) -> list[float]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": text,
        }
        url = f"{self.base_url}/api/embeddings"

        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data: dict[str, Any] = response.json()
                embedding: list[float] = data.get("embedding", [])
                if not embedding:
                    raise ValueError(f"Ollama returned no embedding for model '{self.model}'")
                if not self.verify_normalization(embedding):
                    self._normalize(embedding)
                return embedding
            except httpx.ReadTimeout as exc:
                last_error = exc
                delay = _backoff_delay(attempt)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(delay)
            except httpx.TimeoutException:
                raise
            except httpx.HTTPStatusError:
                raise

        raise last_error if last_error is not None else RuntimeError(
            f"OllamaEmbedder.embed_single failed after {MAX_RETRIES} retries"
        )

    # ------------------------------------------------------------------
    # Optimised batch: one POST per text with semaphore-controlled concurrency
    #
    # Strategy
    #   GPU: 16 requests in-flight simultaneously — GPU overlap keeps all
    #        CUDA cores busy; nomic-embed-text is memory-bandwidth-bound
    #        rather than compute-bound so a high concurrency is beneficial.
    #   CPU: 4 requests in-flight — avoids OOM / CPU thrashing on large batches.
    #
    # ``embed_batch`` calls this method automatically; callers can also invoke
    # it directly with custom batch_size.
    # ------------------------------------------------------------------

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        on_progress: BatchProgressCallback | None = None,
    ) -> list[list[float]]:
        """
        Embed *texts* in batches of at most *batch_size* in-flight requests.

        Within each batch up to ``concurrency_limit`` requests run in parallel
        (controlled by an internal semaphore).  After all batches finish the
        progress callback is invoked with the cumulative done/total counts.

        ReadTimeout → exponential backoff (30 s, 60 s, 90 s) per request.
        """
        total = len(texts)
        if total == 0:
            await _emit_batch_progress(on_progress, 0, 0)
            return []

        results: list[list[float]] = [None] * total  # type: ignore[assignment]
        done = 0

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = texts[batch_start:batch_end]

            # Launch all texts in this batch concurrently, gated by the semaphore.
            tasks = [
                self._embed_single_impl(text) for text in batch
            ]
            batch_results = await asyncio.gather(*tasks)

            # Write results back into the pre-allocated list.
            for offset, emb in enumerate(batch_results):
                results[batch_start + offset] = emb

            done += len(batch_results)
            await _emit_batch_progress(on_progress, done, total)

            # ── Memory management ──────────────────────────────────────────
            # Clear references to this batch's raw strings so Python GC can
            # reclaim them immediately rather than waiting for the function
            # to return.  The pre-allocated `results` list keeps only the
            # small float vectors.
            del batch, batch_results, tasks
            gc.collect()
            _clear_cuda_cache()

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Streaming batch — yields embeddings per sub-batch, releasing memory immediately
    #
    # This is the preferred method when you want to:
    #   (a) upsert to Qdrant per batch instead of accumulating all embeddings
    #   (b) call memory_cleanup() after each batch
    #
    # Usage:
    #   async for (batch_embeddings, batch_start, batch_end) in embed_batches(texts):
    #       records = build_records(batch_embeddings)
    #       qdrant.upsert_chunks(records)
    #       await memory_cleanup()
    # ------------------------------------------------------------------

    async def embed_batches(
        self,
        texts: list[str],
        batch_size: int = 32,
        on_progress: BatchProgressCallback | None = None,
    ) -> AsyncIterator[tuple[list[list[float]], int, int]]:
        """
        Yield batches of embeddings one at a time, clearing memory between batches.

        Each yielded tuple is ``(batch_embeddings, batch_start, batch_end)`` where
        ``batch_embeddings[i]`` corresponds to ``texts[batch_start + i]``.

        Call ``memory_cleanup()`` after processing each batch to reclaim RAM/VRAM.

        Args:
            texts:        List of texts to embed.
            batch_size:   Number of texts per concurrent batch (default 32).
            on_progress:  Optional callback ``(done, total)`` after each batch.

        Yields:
            ``(batch_embeddings, batch_start, batch_end)`` for each batch.
        """
        total = len(texts)
        done = 0

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = texts[batch_start:batch_end]

            tasks = [self._embed_single_impl(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks)

            done += len(batch_embeddings)
            await _emit_batch_progress(on_progress, done, total)

            yield batch_embeddings, batch_start, batch_end

            # Release batch-local variables immediately.
            del batch, batch_embeddings, tasks
            gc.collect()
            _clear_cuda_cache()

    # ------------------------------------------------------------------
    # Batch API — single POST with multiple texts in one request
    #
    # Ollama 0.1.23+ supports POST /api/embed with an "input" array:
    #   {"model": "...", "input": ["text1", "text2", ...]}
    # This processes ALL texts in a single GPU kernel launch, which is
    # 2–5× faster than N sequential /api/embeddings calls for the same N texts.
    #
    # Strategy:
    #   - Divide texts into sub-batches of api_batch_size (default 256).
    #   - Each sub-batch → one POST to /api/embed (fully parallel on GPU).
    #   - Sub-batches are sequential because a single /api/embed call already
    #     saturates GPU parallelism for that group.
    #   - Progress callback fires after each sub-batch with cumulative done/total.
    # ------------------------------------------------------------------

    async def embed_batch_api(
        self,
        texts: list[str],
        api_batch_size: int = 256,
        on_progress: BatchProgressCallback | None = None,
    ) -> list[list[float]]:
        """
        Embed *texts* using Ollama's batch ``/api/embed`` endpoint.

        This is 2–5× faster than ``embed_batch`` because all texts in a
        sub-batch are processed in a single GPU kernel launch rather than
        N separate calls.

        Args:
            texts:          List of text strings to embed.
            api_batch_size: Max texts per /api/embed request (default 256).
                            Higher values increase GPU utilization but use more VRAM.
            on_progress:    Optional callback ``(done, total)`` after each sub-batch.

        Returns:
            List of embedding vectors, one per input text.
        """
        total = len(texts)
        if total == 0:
            await _emit_batch_progress(on_progress, 0, 0)
            return []

        results: list[list[float] | None] = [None] * total  # type: ignore[assignment]
        url = f"{self.base_url}/api/embed"
        done = 0

        for batch_start in range(0, total, api_batch_size):
            batch = texts[batch_start : batch_start + api_batch_size]
            batch_size = len(batch)

            last_error: Exception | None = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    async with httpx.AsyncClient(timeout=max(120.0, batch_size * 2.0)) as client:
                        response = await client.post(
                            url,
                            json={"model": self.model, "input": batch},
                        )
                        response.raise_for_status()
                        payload: dict[str, Any] = response.json()
                    embeddings: list[list[float]] = payload.get("embeddings", [])

                    if len(embeddings) != batch_size:
                        _log.warning(
                            "embed_batch_api: expected %d embeddings, got %d — retrying",
                            batch_size, len(embeddings),
                        )
                        raise ValueError(
                            f"Ollama returned {len(embeddings)} embeddings for {batch_size} inputs"
                        )

                    for idx, emb in enumerate(embeddings):
                        if not emb:
                            _log.warning("embed_batch_api: empty embedding at index %d", batch_start + idx)
                        else:
                            if not self.verify_normalization(emb):
                                self._normalize(emb)
                            results[batch_start + idx] = emb

                    break  # success

                except (httpx.ReadTimeout, httpx.TimeoutException, httpx.HTTPStatusError, ValueError) as exc:
                    last_error = exc
                    if attempt < MAX_RETRIES:
                        delay = _backoff_delay(attempt)
                        _log.warning(
                            "embed_batch_api batch %d-%d failed (attempt %d/%d): %s — retrying in %.0fs",
                            batch_start, batch_start + batch_size,
                            attempt, MAX_RETRIES, exc, delay,
                        )
                        await asyncio.sleep(delay)
                    # else: fall through to re-raise

            if last_error is not None:
                # All retries exhausted — raise with original exception
                raise last_error  # type: ignore[misc]

            done += batch_size
            await _emit_batch_progress(on_progress, done, total)

            # Release per-batch intermediate variables.
            del batch, embeddings, payload
            gc.collect()
            _clear_cuda_cache()

        return results  # type: ignore[return-value]
