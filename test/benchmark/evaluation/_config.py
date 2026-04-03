"""
_config.py — Shared configuration for the RAGEve evaluation suite.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure project root is first so rag/backend are always importable.
# __file__ = .../project/test/benchmark/evaluation/FILENAME.py  (4 levels deep)
_project_root = Path(__file__).resolve()
for _ in range(4):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_V1_URL = f"{OLLAMA_BASE_URL}/v1"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
OLLAMA_CHAT_MODEL = "llama3.2:latest"
OLLAMA_EMBED_DIM = 768          # nomic-embed-text dimension

QDRANT_URL = "http://localhost:6333"
API_BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

SQUAD_PARQUET = _project_root / "data" / "hf" / "squad" / "train" / "data.parquet"
DEFAULT_DATASET_ID = "squad"    # Qdrant collection name
DEFAULT_N_SAMPLES = 100          # squad rows to evaluate
DEFAULT_TOP_K = 5               # retrieval top_k

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_DIR = _project_root / "data" / "benchmarks"


@dataclass
class EvalConfig:
    squad_parquet: Path = SQUAD_PARQUET
    dataset_id: str = DEFAULT_DATASET_ID
    n_samples: int = DEFAULT_N_SAMPLES
    top_k: int = DEFAULT_TOP_K
    ollama_base_url: str = OLLAMA_BASE_URL
    ollama_v1_url: str = OLLAMA_V1_URL
    ollama_chat_model: str = OLLAMA_CHAT_MODEL
    ollama_embed_model: str = OLLAMA_EMBED_MODEL
    ollama_embed_dim: int = OLLAMA_EMBED_DIM
    qdrant_url: str = QDRANT_URL
    output_dir: Path = OUTPUT_DIR
