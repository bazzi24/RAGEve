import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from backend.api.routes._limiter import limiter
from backend.api.routes.agents import router as agents_router
from backend.api.routes.auth import router as auth_router
from backend.api.routes.chat import router as chat_router
from backend.api.routes.conversations import router as conversations_router
from backend.api.routes.datasets import router as datasets_router
from backend.api.routes.dialogs import router as dialogs_router
from backend.api.routes.documents import router as documents_router
from backend.api.routes.files import router as files_router
from backend.api.routes.huggingface import router as hf_router
from backend.api.routes.knowledgebases import router as knowledgebases_router
from backend.api.routes.ollama import router as ollama_router
from backend.api.routes.rerank import router as rerank_router
from backend.config_loader import settings
from backend.logging_config import setup_logging
from backend.models_peewee import close_db as peewee_close_db
from backend.models_peewee import get_database
from backend.models_peewee import init_db as peewee_init_db
from backend.services.cache_service import close_cache, init_cache
from backend.services.database import run_db_operation
from backend.services.ingestion_factory import close_qdrant_store, get_ingestion_service
from backend.services.redis_client import close_redis, init_redis

# Initialise file-based logging before any route handlers run.
setup_logging(settings.logs_dir)

_log = logging.getLogger("app")

app = FastAPI(title=settings.app_name)

# Rate limiter is imported from _limiter and attached to app.state
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    _log.warning("Rate limit exceeded: %s %s", request.method, request.url.path)
    response = JSONResponse(
        status_code=429,
        content={"error": "Too many requests. Please slow down and retry."},
    )
    # Include Retry-After header if available (seconds)
    if hasattr(exc, "retry_after"):
        response.headers["Retry-After"] = str(int(exc.retry_after))
    return response


# ── Global exception handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    _log.exception(
        "Unhandled exception [%s] %s %s", request_id, request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": request_id},
    )


# ── Request ID middleware ─────────────────────────────────────────────────────
@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── API key auth middleware ───────────────────────────────────────────────────
async def _check_api_key(request: Request) -> JSONResponse | None:
    """Return a 401 response if API_KEY is set but the request has no valid key."""
    if not settings.api_key:
        return None  # auth disabled — allow everything
    provided = request.headers.get("X-API-Key", "")
    if not provided or not secrets.compare_digest(provided, settings.api_key):
        _log.warning(
            "Unauthorized request (no/invalid API key): %s %s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=401,
            content={"error": "Missing or invalid X-API-Key header."},
        )
    return None


# ── CORS (driven by CORS_ORIGINS env var) ─────────────────────────────────────
_allowed_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ─────────────────────────────────────────────────
class RequestLogMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status code, latency, and request ID."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        import urllib.parse

        path = request.url.path
        query = request.url.query
        safe_query = urllib.parse.quote_plus(query) if query else ""
        path_log = f"{path}?{safe_query}" if safe_query else path
        request_id = getattr(request.state, "request_id", "-")

        _log.info(
            "%s %s %s %.1fms [reqid=%s]",
            request.method,
            path_log,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response


app.add_middleware(RequestLogMiddleware)


# ── API key auth middleware ───────────────────────────────────────────────────
class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests with 401 when API_KEY is configured and key is missing/invalid."""

    async def dispatch(self, request: Request, call_next):
        if error_response := await _check_api_key(request):
            return error_response
        return await call_next(request)


app.add_middleware(ApiKeyAuthMiddleware)


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    _log.info("─" * 60)
    _log.info("%s started", settings.app_name)
    _log.info("Environment : %s", settings.app_env)
    _log.info("Logs        : %s", settings.logs_dir)
    _log.info("Ollama      : %s", settings.ollama_base_url)
    _log.info("Qdrant      : %s", settings.qdrant_url)
    _log.info("CORS origins: %s", _allowed_origins)
    trusted_proxy_enabled = False
    try:
        trusted_proxy_enabled = int(settings.trusted_proxy_count) > 0
    except (ValueError, TypeError):
        trusted_proxy_enabled = False
    _log.info(
        "Trusted proxies (XFF): %s",
        "enabled" if trusted_proxy_enabled else "disabled",
    )
    try:
        rate_limit = int(settings.rate_limit_per_minute)
    except (ValueError, TypeError):
        rate_limit = 120
    if settings.api_key:
        _log.info("API Auth    : enabled")
    else:
        _log.info("API Auth    : disabled (set API_KEY in .env to enable)")
    _log.info("Rate Limit  : enabled (%d req/min per IP baseline)", rate_limit)
    if settings.hf_token:
        _log.info("HF Token    : configured (private datasets enabled)")
    else:
        _log.info("HF Token    : not set (public datasets only)")
    if settings.db_url:
        _log.info(
            "Chat DB     : MySQL (%s:%s/%s)",
            settings.mysql_host,
            settings.mysql_port,
            settings.mysql_dbname,
        )
    else:
        _log.info("Chat DB     : SQLite (%s)", settings.db_path)
    _log.info(
        "Peewee ORM DB        : MySQL (%s:%s/%s)",
        settings.mysql_host,
        settings.mysql_port,
        settings.mysql_dbname,
    )
    _log.info("─" * 60)
    # Initialise DB tables (creates them on first run)
    await peewee_init_db()
    # Initialize Redis
    await init_redis()
    # Initialize cache service
    await init_cache()
    # Initialize MinIO buckets
    try:
        from botocore.exceptions import ClientError

        from backend.services.minio_client import get_minio_client

        minio_client = get_minio_client()
        buckets = ["uploads", "chunks", "vectors"]
        for bucket in buckets:
            try:
                # Check if bucket exists using head_bucket
                await minio_client.client.head_bucket(Bucket=bucket)
                _log.debug("MinIO bucket exists: %s", bucket)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code")
                if error_code == "404" or error_code == "NoSuchBucket":
                    await minio_client.client.create_bucket(Bucket=bucket)
                    _log.info("Created MinIO bucket: %s", bucket)
                else:
                    _log.warning("Failed to check/create bucket %s: %s", bucket, e)
    except Exception as e:
        _log.warning("MinIO initialization skipped: %s", e)
    # Pre-warm Qdrant store so the first request is fast
    _ = get_ingestion_service()
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────
    _log.info("Shutting down… closing connections")
    await close_qdrant_store()
    await close_redis()
    await close_cache()
    peewee_close_db()
    _log.info("Shutdown complete")


app.router.lifespan_context = lifespan


# ── Health (verifies Ollama + Qdrant + Database + Redis connectivity) ─────────────
@app.get("/health")
async def health():
    ollama_ok = False
    qdrant_ok = False
    db_ok = False
    redis_ok = False
    minio_ok = False

    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = r.status_code == 200
        except Exception:
            pass
        try:
            r = await client.get(f"{settings.qdrant_url}/collections")
            qdrant_ok = r.status_code == 200
        except Exception:
            pass

    # Check database connectivity
    try:
        db = get_database()
        await run_db_operation(db.execute_sql, "SELECT 1")
        db_ok = True
    except Exception as e:
        _log.error("Database health check failed: %s", e)
        db_ok = False

    # Check Redis connectivity
    try:
        from backend.services.redis_client import get_redis_client

        redis_client = get_redis_client()
        redis_ok = await redis_client.ping()
    except Exception as e:
        _log.warning("Redis health check failed: %s", e)
        redis_ok = False

    # Check MinIO connectivity (basic check)
    try:
        from backend.services.minio_client import get_minio_client

        get_minio_client()
        # MinIO client does bucket head check on init, so if we got here it's OK
        minio_ok = True
    except Exception as e:
        _log.warning("MinIO health check failed: %s", e)
        minio_ok = False

    all_ok = ollama_ok and qdrant_ok and db_ok and redis_ok and minio_ok
    status_val = "ok" if all_ok else "degraded"

    return {
        "status": status_val,
        "ollama": "ok" if ollama_ok else "unreachable",
        "qdrant": "ok" if qdrant_ok else "unreachable",
        "database": "ok" if db_ok else "unreachable",
        "redis": "ok" if redis_ok else "unreachable",
        "minio": "ok" if minio_ok else "unreachable",
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(ollama_router)
app.include_router(files_router)
app.include_router(datasets_router)
app.include_router(hf_router)
app.include_router(agents_router)
app.include_router(chat_router)
app.include_router(rerank_router)
app.include_router(dialogs_router)
app.include_router(conversations_router)
app.include_router(knowledgebases_router)
app.include_router(documents_router)
