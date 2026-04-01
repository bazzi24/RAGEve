"""
HuggingFace Hub metadata helpers — card data and README fetching.

Provides:
  _fetch_hf_card_metadata() — full card metadata (tags, license, language, paper, leaderboard)
  _fetch_hf_readme_html()   — README.md rendered as escaped HTML
"""
from __future__ import annotations

import html as _html_module
import logging
from typing import Any

_log = logging.getLogger("app")


def _fetch_hf_card_metadata(dataset_id: str, hf_token: str | None = None) -> dict[str, Any]:
    """Fetch full dataset card metadata using huggingface_hub HfApi."""
    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]

        api = HfApi()
        info = api.list_dataset_info(dataset_id, files_metadata=False, use_auth_token=hf_token)

        tags: list[str] = list(info.tags) if info.tags else []
        license_str: str | None = info.license

        language: list[str] = []
        paper_url: str | None = None
        leaderboard: dict | None = None
        if info.card_data:
            lang = info.card_data.get("language") or info.card_data.get("languages", [])
            if isinstance(lang, list):
                language = [str(l) for l in lang]
            elif lang:
                language = [str(lang)]
            paper_url = (
                info.card_data.get("paper", {})
            ).get("url") or info.card_data.get("paperswithcode_id")
            leaderboard = info.card_data.get("leaderboard")

        return {
            "card_data": dict(info.card_data) if info.card_data else None,
            "tags": tags,
            "language": language,
            "license": license_str,
            "paper_url": paper_url,
            "leaderboard": leaderboard,
        }
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to fetch HF card metadata for '%s': %s", dataset_id, exc)
        return {
            "card_data": None,
            "tags": [],
            "language": [],
            "license": None,
            "paper_url": None,
            "leaderboard": None,
        }


def _fetch_hf_readme_html(dataset_id: str) -> str | None:
    """Fetch the dataset README.md from the HuggingFace Hub."""
    import httpx

    filenames = ["README.md", "README_fr.md", "README_de.md"]
    for fname in filenames:
        try:
            url = f"https://huggingface.co/datasets/{dataset_id}/raw/main/{fname}"
            resp = httpx.get(url, timeout=10.0)
            if resp.status_code == 200:
                content = resp.text[:4000]
                escaped = _html_module.escape(content)
                return f"<pre style='font-size:12px;line-height:1.6;max-height:400px;overflow:auto'>{escaped}</pre>"
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to fetch README for '%s' (%s): %s", dataset_id, fname, exc)
            continue
    return None
