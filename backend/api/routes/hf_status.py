"""
HuggingFace download + ingest status registries and persistence.

Exports:
  Download status: _hf_download_status, _set_download_status,
                  _hf_download_locks, _hf_download_cancel_flags,
                  _is_cancelled, _get_status_file, _persist_status_to_disk,
                  HFProgressTracker, _install_progress_tracker,
                  _remove_progress_tracker, _copy_dataset_to_hf_root,
                  _cleanup_partial_dataset, MAX_DOWNLOAD_TIMEOUT

  Ingest status:  _hf_ingest_registry, _hf_ingest_status_file,
                  _load_hf_ingest_registry, _persist_hf_ingest_registry,
                  _hf_ingest_lock, _set_hf_ingest_status

All other HF routes import from here so they share the same in-memory state.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import uuid as uuid_lib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import settings

_log = logging.getLogger(__name__)

# ── Download status registry ───────────────────────────────────────────────────

# In-memory download status tracker — persisted to disk so it survives server restarts
_hf_download_status: dict[str, dict[str, Any]] = {}
_hf_download_locks: dict[str, asyncio.Lock] = {}
_hf_download_cancel_flags: dict[str, bool] = {}

# Per-call timeout for datasets library operations (seconds).
# Large or slow datasets (e.g. cold cache, spotty network) will fail fast
# rather than hanging indefinitely.
MAX_DOWNLOAD_TIMEOUT = 3600  # 1 hour — generous for multi-GB datasets on cold cache

_STATUS_FILE: Path | None = None


def _get_status_file() -> Path:
    global _STATUS_FILE
    if _STATUS_FILE is None:
        _STATUS_FILE = settings.hf_status_file
        _STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    return _STATUS_FILE


def _load_status_from_disk() -> dict[str, dict[str, Any]]:
    """Load persisted download statuses from disk. Called once at module import."""
    try:
        fpath = _get_status_file()
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Re-hydrate: only keep non-terminal statuses
            return {
                k: v
                for k, v in raw.items()
                if v.get("status") not in ("completed", "failed", "cancelled")
            }
    except Exception:  # noqa: BLE001
        pass
    return {}


def _persist_status_to_disk() -> None:
    """Write current in-memory status to disk."""
    try:
        with open(_get_status_file(), "w", encoding="utf-8") as f:
            json.dump(_hf_download_status, f)
    except Exception:  # noqa: BLE001
        pass


# Reload any non-terminal statuses from disk on startup
_hf_download_status = _load_status_from_disk()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_dataset_dirname(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def _set_download_status(dataset_id: str, **updates: Any) -> None:
    current = _hf_download_status.get(dataset_id, {
        "dataset_id": dataset_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "error": None,
        "local_path": None,
        "started_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "rows_downloaded": None,
        "splits_downloaded": [],
        "bytes_downloaded": None,
        "total_bytes": None,
        "config": None,
        "auto_ingest": False,
        "ingest_status": None,
        "ingest_message": None,
        "ingest_error": None,
        "ingested": False,
        "suggested_text_column": None,
        "columns": {},
    })
    current.update(updates)
    current["updated_at"] = _utc_now_iso()
    _hf_download_status[dataset_id] = current
    _persist_status_to_disk()


def _copy_dataset_to_hf_root(temp_dir: Path, target_dir: Path) -> None:
    """Copy downloaded dataset artifacts from temporary dir into ./data/hf/{dataset}."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in temp_dir.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def _cleanup_partial_dataset(dataset_id: str, data_root: Path) -> None:
    """Remove partially downloaded files for a dataset (best effort)."""
    safe_id = _normalize_dataset_dirname(dataset_id)
    target_dir = data_root / "hf" / safe_id
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)


def _is_cancelled(dataset_id: str) -> bool:
    return _hf_download_cancel_flags.get(dataset_id, False)


# ── Progress tracking ──────────────────────────────────────────────────────────

# Registry: dataset_id → tracker instance
_progress_callbacks: dict[str, Any] = {}

# Guards so we only patch once per process
_patched_tqdm = False


def _make_progress_tracker(dataset_id: str) -> "HFProgressTracker":
    """Create a progress tracker for a given dataset download."""
    return HFProgressTracker(dataset_id)


class HFProgressTracker:
    """
    Progress tracker that captures tqdm updates from datasets library downloads.

    Works by monkey-patching `datasets.utils.tqdm.tqdm` — the class used internally
    by `datasets.download.download_manager` when downloading files.
    """

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        self.bytes_done: int = 0
        self.bytes_total: int | None = None
        self._last_reported_pct: int = -1

    def update(self, done: int, total: int | None) -> None:
        """Called by tqdm after each progress update."""
        if done < 0:
            return
        self.bytes_done = done
        self.bytes_total = total if total and total > 0 else None

        pct = 0
        if self.bytes_total and self.bytes_total > 0:
            pct = min(100, int((self.bytes_done / self.bytes_total) * 100))
        elif done > 0:
            pct = min(89, int((done % 100_000_000) / 1_000_000))

        if pct != self._last_reported_pct:
            self._last_reported_pct = pct
            _set_download_status(
                self.dataset_id,
                progress=pct,
                bytes_downloaded=self.bytes_done,
                total_bytes=self.bytes_total,
                message=_build_progress_message(self.dataset_id, pct, self.bytes_done, self.bytes_total),
            )

    def finalize(self, final_pct: int) -> None:
        """Called when a split download completes."""
        _set_download_status(
            self.dataset_id,
            progress=final_pct,
            bytes_downloaded=self.bytes_total or self.bytes_done,
            total_bytes=self.bytes_total,
        )


def _build_progress_message(
    dataset_id: str, pct: int, done: int | None, total: int | None
) -> str:
    """Build a human-readable download message."""
    if done is None:
        return "Downloading…"

    def fmt(n: int | None) -> str:
        if n is None:
            return "—"
        if n < 1024:
            return f"{n} B"
        if n < 1024**2:
            return f"{n / 1024:.1f} KB"
        if n < 1024**3:
            return f"{n / 1024**2:.1f} MB"
        return f"{n / 1024**3:.1f} GB"

    if total and total > 0:
        return f"Downloading… {pct}% ({fmt(done)} / {fmt(total)})"
    return f"Downloading… {fmt(done)}"


def _setup_tqdm_patch() -> None:
    """
    Monkey-patch `datasets.utils.tqdm.tqdm` to capture progress callbacks.

    Called once before the first dataset download.
    """
    global _patched_tqdm
    if _patched_tqdm:
        return

    try:
        from tqdm import tqdm as original_tqdm
        from datasets.utils import tqdm as datasets_tqdm_module

        class _TrackingTqdm(original_tqdm):  # type: ignore[misc]
            def update(self, n: int = 1) -> None:  # type: ignore[override]
                super().update(n)
                # Try to find which dataset this tqdm belongs to by matching total
                for tracker in _progress_callbacks.values():
                    tracker_bytes = tracker.bytes_total
                    if tracker_bytes and tracker_bytes == self.total:
                        tracker.update(self.n, tracker_bytes)
                        break

        datasets_tqdm_module.tqdm = _TrackingTqdm  # type: ignore[assignment]
        _patched_tqdm = True
    except Exception:  # noqa: BLE001
        pass


def _install_progress_tracker(dataset_id: str) -> "HFProgressTracker | None":
    """Install a progress tracker for a dataset download. Returns the tracker."""
    try:
        _setup_tqdm_patch()
        tracker = _make_progress_tracker(dataset_id)
        _progress_callbacks[dataset_id] = tracker
        return tracker
    except Exception:  # noqa: BLE001
        return None


def _remove_progress_tracker(dataset_id: str) -> None:
    """Remove a progress tracker after download completes."""
    _progress_callbacks.pop(dataset_id, None)


# ── Ingest status registry ─────────────────────────────────────────────────────

_hf_ingest_registry: dict[str, dict[str, Any]] = {}
_hf_ingest_lock = asyncio.Lock()
_hf_ingest_status_file: Path | None = None


def _hf_ingest_status_file_path() -> Path:
    global _hf_ingest_status_file
    if _hf_ingest_status_file is None:
        _hf_ingest_status_file = settings.hf_ingest_status_file
        _hf_ingest_status_file.parent.mkdir(parents=True, exist_ok=True)
    return _hf_ingest_status_file


def _load_hf_ingest_registry() -> dict[str, dict[str, Any]]:
    """Load persisted ingest statuses from disk."""
    try:
        fpath = _hf_ingest_status_file_path()
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:  # noqa: BLE001
        pass
    return {}


def _persist_hf_ingest_registry() -> None:
    """Write ingest registry to disk."""
    try:
        with open(_hf_ingest_status_file_path(), "w", encoding="utf-8") as f:
            json.dump(_hf_ingest_registry, f)
    except Exception:  # noqa: BLE001
        pass


# Re-hydrate on startup
_hf_ingest_registry = _load_hf_ingest_registry()


async def _set_hf_ingest_status(ingest_id: str, **kwargs: Any) -> dict[str, Any]:
    """Thread-safe update of ingest status + persist to disk."""
    async with _hf_ingest_lock:
        entry = _hf_ingest_registry.setdefault(ingest_id, {"ingest_id": ingest_id})
        entry.update(kwargs)
        _persist_hf_ingest_registry()
        return dict(entry)

