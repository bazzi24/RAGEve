from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAGEve"
    app_env: str = "dev"

    ollama_base_url: str = "http://localhost:11434"

    vector_db_provider: str = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None  # Wire to Qdrant when it is deployed with auth

    # MySQL / chat history database.
    # Use mysql+aiomysql:// for production, mysql+pymysql:// for sync fallback.
    # Leave db_url blank to use SQLite (data/chat.db) for single-node deployments.
    db_url: str | None = None
    db_pool_size: int = 5
    db_max_overflow: int = 10

    # Optional API key for protecting all endpoints.
    # If unset (None or empty), the API is open (local-only deployments).
    api_key: str | None = None

    # Rate limit for all endpoints (requests per minute per IP).
    # Only active when api_key is configured.
    rate_limit_per_minute: int = 120

    # Comma-separated list of allowed CORS origins (no spaces around commas).
    # Example: "https://app.example.com,https://staging.example.com"
    cors_origins: str = "http://localhost:3000,http://localhost:3001,http://localhost:3002"

    # Number of reverse proxies in front of this app (for X-Forwarded-For).
    # Set to 0 to disable proxy-aware IP detection in the rate limiter.
    trusted_proxy_count: int = 1

    # HuggingFace access token — required for private datasets.
    # Leave blank for anonymous access (public datasets only).
    hf_token: str | None = None

    data_root: Path = Path("data")
    upload_dir_name: str = "uploads"
    chunk_dir_name: str = "chunks"
    vector_dir_name: str = "vector"

    default_chunk_size: int = 1200
    default_chunk_overlap: int = 180
    default_max_tokens_per_chunk: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def upload_root(self) -> Path:
        return self.data_root / self.upload_dir_name

    @property
    def chunk_root(self) -> Path:
        return self.data_root / self.chunk_dir_name

    @property
    def vector_root(self) -> Path:
        return self.data_root / self.vector_dir_name

    @property
    def db_path(self) -> Path:
        return self.data_root / "chat.db"

    @property
    def hf_status_file(self) -> Path:
        return self.data_root / "hf" / "_download_status.json"

    @property
    def ingest_status_file(self) -> Path:
        return self.data_root / "_ingest_status.json"

    @property
    def hf_ingest_status_file(self) -> Path:
        return self.data_root / "hf" / "_ingest_status.json"

    @property
    def logs_dir(self) -> Path:
        return self.data_root / "logs"


settings = Settings()
