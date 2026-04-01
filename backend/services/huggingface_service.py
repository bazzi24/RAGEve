from __future__ import annotations

import json
from pathlib import Path

import httpx

HF_HUB_API = "https://huggingface.co/api"


def get_download_command(dataset_id: str) -> str:
    safe = dataset_id.replace("/", "__")
    code = (
        f"from huggingface_hub import hf_hub_download, list_repo_files\n"
        f"from pathlib import Path\n"
        f"Path('./data/hf/{safe}').mkdir(parents=True, exist_ok=True)\n"
        f"files = list_repo_files('{dataset_id}', repo_type='dataset')\n"
        f"for f in files:\n"
        f"    p = hf_hub_download(repo_id='{dataset_id}', filename=f, repo_type='dataset', local_dir='./data/hf/{safe}')\n"
        f"    print('downloaded:', p)\n"
        f"print('All files saved to ./data/hf/{safe}/')"
    )
    return code


class DatasetInfo:
    def __init__(
        self,
        id: str,
        sha: str,
        size_bytes: int | None,
        tags: list[str],
        siblings: list[dict],
        private: bool,
        author: str | None,
    ) -> None:
        self.id = id
        self.sha = sha
        self.size_bytes = size_bytes
        self.tags = tags
        self.siblings = siblings
        self.private = private
        self.author = author


async def fetch_dataset_info(dataset_id: str) -> DatasetInfo | None:
    url = f"{HF_HUB_API}/datasets/{dataset_id}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError:
            return None

    siblings = data.get("siblings", [])
    tags = data.get("tags", [])

    return DatasetInfo(
        id=data.get("id", dataset_id),
        sha=data.get("sha", ""),
        size_bytes=data.get("size_bytes"),
        tags=tags,
        siblings=siblings,
        private=data.get("private", False),
        author=data.get("author"),
    )


def _scan_directory_for_datasets(root: Path) -> list[dict]:
    """
    Walk root and detect HuggingFace-compatible dataset folders.
    Files can be in any subdirectory (e.g. squad/plain_text/).
    """
    detected: list[dict] = []

    if not root.exists():
        return detected

    for entry in sorted(root.iterdir()):
        # Skip files (e.g. _ingest_status.json, _download_status.json)
        # Wrap is_dir() in try/except to handle exotic filesystems that raise OSError
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue

        # Recursively find supported files in any subdirectory
        parquet_files: list[Path] = list(entry.rglob("*.parquet"))
        json_files: list[Path] = list(entry.rglob("*.json"))
        csv_files: list[Path] = list(entry.rglob("*.csv"))
        all_files = parquet_files + json_files + csv_files

        if not all_files:
            continue

        # Splits = top-level subdirs named train/test/validation/dev
        def _is_split_dir(d: Path) -> bool:
            try:
                return d.is_dir() and d.name in ("train", "test", "validation", "dev")
            except OSError:
                return False

        splits = sorted(d.name for d in entry.iterdir() if _is_split_dir(d))
        if not splits:
            splits = ["default"]

        file_formats = list({f.suffix.lower() for f in all_files if f.suffix})
        total_size = sum(f.stat().st_size for f in all_files)

        # Read columns from first parquet (most common HF format)
        import pandas as pd

        readable_columns: list[str] = []
        if parquet_files:
            try:
                df = pd.read_parquet(parquet_files[0])
                readable_columns = list(df.columns)
            except Exception:
                pass
        elif csv_files:
            try:
                df = pd.read_csv(csv_files[0], nrows=1, encoding="utf-8")
                readable_columns = list(df.columns)
            except Exception:
                pass
        elif json_files:
            try:
                df = pd.read_json(json_files[0], lines=True, nrows=1, encoding="utf-8")
                readable_columns = list(df.columns)
            except Exception:
                pass

        # Decode filesystem-safe name back to original dataset ID
        # e.g. "th1nhng0__vietnamese-legal-documents" → "th1nhng0/vietnamese-legal-documents"
        fs_safe_name = entry.name
        decoded_id = fs_safe_name.replace("__", "/")

        detected.append({
            "local_path": str(entry.relative_to(root)),
            "dataset_id": decoded_id,
            "splits": splits,
            "file_formats": file_formats,
            "file_count": len(all_files),
            "total_size_bytes": total_size,
            "readable_columns": readable_columns,
            "description": None,
        })

    return detected


def scan_hf_datasets(data_root: Path) -> list[dict]:
    hf_root = data_root / "hf"
    return _scan_directory_for_datasets(hf_root)


def get_registered_datasets(registry_path: Path) -> dict:
    if not registry_path.exists():
        return {}
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def register_dataset(registry_path: Path, dataset_id: str, metadata: dict) -> None:
    registry = get_registered_datasets(registry_path)
    registry[dataset_id] = metadata
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
