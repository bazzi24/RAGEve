"""
Memory management utilities for the ingestion pipeline.

Provides:
- CUDA VRAM clearing (torch.cuda.empty_cache)
- Forced garbage collection after batch loops
- Live VRAM/RAM usage logging
- Memory-safe batch sizing heuristics
"""

from __future__ import annotations

import gc
import logging
from typing import Any

_log = logging.getLogger(__name__)

# Cache the torch import so we don't require torch when not using GPU
_torch_cache: Any = None


def _get_torch():
    """Lazily import torch. Returns None if not available."""
    global _torch_cache
    if _torch_cache is None:
        try:
            import torch as _torch_cache  # type: ignore[assignment]
        except ImportError:
            _torch_cache = False
    return _torch_cache


def cuda_memory_free() -> float:
    """
    Return bytes of free CUDA VRAM, or 0.0 if CUDA is unavailable.
    Call this before deciding batch sizes.
    """
    torch = _get_torch()
    if not torch or not torch.cuda.is_available():
        return 0.0
    return float(
        torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
    )


def cuda_memory_used() -> float:
    """Return bytes of currently allocated CUDA VRAM, or 0.0."""
    torch = _get_torch()
    if not torch or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated())


def clear_cuda_cache() -> None:
    """
    Clear PyTorch CUDA memory cache.

    Call this after every embedding batch to ensure previously held
    activation tensors are released before the next batch starts.
    """
    torch = _get_torch()
    if not torch or not torch.cuda.is_available():
        return

    # Tell PyTorch to release cached allocator memory back to the OS.
    # This is safe to call even when tensors are still alive.
    torch.cuda.empty_cache()

    allocated_mb = cuda_memory_used() / (1024 * 1024)
    total_mb = cuda_memory_free() / (1024 * 1024)
    _log.debug(
        "CUDA cache cleared — used=%.1f MB, free≈%.1f MB",
        allocated_mb, total_mb,
    )


def memory_cleanup() -> None:
    """
    Run forced garbage collection + CUDA cache clear.

    Call this after each embedding batch to break any lingering
    references to intermediate tensors/strings from the previous batch.
    """
    gc.collect()
    clear_cuda_cache()


def log_memory_snapshot(label: str) -> None:
    """Log current RAM and VRAM usage. Useful at stage boundaries."""
    torch = _get_torch()

    # RAM
    try:
        import psutil
        proc = psutil.Process()
        ram_mb = proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        ram_mb = None

    # CUDA
    if torch and torch.cuda.is_available():
        cuda_used_mb = cuda_memory_used() / (1024 * 1024)
        cuda_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        cuda_pct = cuda_used_mb / cuda_total_mb * 100 if cuda_total_mb > 0 else 0
        _log.info(
            "[%s] RAM=%s VRAM=%.0f MB / %.0f MB (%.0f%%)",
            label,
            f"{ram_mb:.0f} MB" if ram_mb is not None else "N/A",
            cuda_used_mb,
            cuda_total_mb,
            cuda_pct,
        )
    elif ram_mb is not None:
        _log.info("[%s] RAM=%s MB", label, f"{ram_mb:.0f}")


def recommend_embed_batch_size(concurrency: int = 32) -> int:
    """
    Return a safe embedding batch size based on available CUDA memory.

    Heuristic:
    - >8 GB free VRAM → 128
    - >4 GB free VRAM → 64
    - >2 GB free VRAM → 32
    - >1 GB free VRAM → 16
    - Otherwise        → 8  (very tight)
    """
    free = cuda_memory_free()
    if free == 0:
        # No CUDA — fall back to conservative default
        return 32

    free_gb = free / (1024**3)

    if free_gb > 8:
        return 128
    elif free_gb > 4:
        return 64
    elif free_gb > 2:
        return 32
    elif free_gb > 1:
        return 16
    else:
        _log.warning(
            "CUDA VRAM critically low (%.1f GB free) — using batch_size=8",
            free_gb,
        )
        return 8
