import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from backend.api.routes.agents import router as agents_router
from backend.api.routes.chat import router as chat_router
from backend.api.routes.chat_history import router as chat_history_router
from backend.api.routes.datasets import router as datasets_router
from backend.api.routes.files import router as files_router
from backend.api.routes.huggingface import router as hf_router
from backend.api.routes.ollama import router as ollama_router
from backend.api.routes.rerank import router as rerank_router
from backend.config import settings
from backend.logging_config import setup_logging
from backend.services.chat_store import close_db, init_db
from backend.services.ingestion_factory import close_qdrant_store, get_qdrant_store

# Initialise file-based logging before any route handlers run.
setup_logging(settings.logs_dir)

_log = logging.getLogger("app")

app = FastAPI(title=settings.app_name)


# ── Client IP helper (used by rate limiter + middleware) ──────────────────────
def _get_client_ip(request: Request) -> str:
    """
    Return the real client IP, accounting for trusted reverse proxies.

    When TRUSTED_PROXY_COUNT > 0 the function reads X-Forwarded-For and
    returns the leftmost (original client) IP from the chain.  For deeper
    chains (e.g. Cloudflare → nginx → backend) set TRUSTED_PROXY_COUNT=2.
    Set to 0 to disable proxy awareness entirely.
    """
    if settings.trusted_proxy_count <= 0:
        return request.client.host if request.client else "127.0.0.1"
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        ips = [ip.strip() for ip in fwd.split(",")]
        idx = len(ips) - settings.trusted_proxy_count
        if 0 <= idx < len(ips):
            return ips[idx]
        return ips[-1]
    return request.client.host if request.client else "127.0.0.1"


# ── Rate limiter (slowapi) ────────────────────────────────────────────────────
# Active only when API_KEY is configured; otherwise no-op.
limiter = Limiter(key_func=_get_client_ip, enabled=bool(settings.api_key))
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    _log.warning("Rate limit exceeded: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=429,
        content={"error": "Too many requests. Please slow down and retry."},
    )


# ── Global exception handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    _log.exception("Unhandled exception [%s] %s %s", request_id, request.method, request.url.path)
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
    _log.info("Trusted proxies (XFF): %d", settings.trusted_proxy_count)
    if settings.api_key:
        _log.info("API Auth    : enabled (%d req/min limit per IP)", settings.rate_limit_per_minute)
    else:
        _log.info("API Auth    : disabled (set API_KEY in .env to enable)")
    if settings.hf_token:
        _log.info("HF Token    : configured (private datasets enabled)")
    else:
        _log.info("HF Token    : not set (public datasets only)")
    if settings.db_url:
        db_desc = settings.db_url.split("@")[1] if "@" in settings.db_url else settings.db_url
        _log.info("Chat DB     : MySQL (%s)", db_desc)
    else:
        _log.info("Chat DB     : SQLite (%s)", settings.db_path)
    _log.info("─" * 60)
    # Initialise DB tables (creates them on first run)
    await init_db()
    # Pre-warm Qdrant store so the first request is fast
    _ = get_qdrant_store()
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────
    _log.info("Shutting down… closing QdrantStore HTTP client")
    await close_qdrant_store()
    await close_db()
    _log.info("Shutdown complete")


app.router.lifespan_context = lifespan

# ── Health (verifies Ollama + Qdrant connectivity) ────────────────────────────
@app.get("/health")
async def health():
    ollama_ok = False
    qdrant_ok = False
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

    status = "ok" if (ollama_ok and qdrant_ok) else "degraded"
    return {
        "status": status,
        "ollama": "ok" if ollama_ok else "unreachable",
        "qdrant": "ok" if qdrant_ok else "unreachable",
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
app.include_router(ollama_router)
app.include_router(files_router)
app.include_router(datasets_router)
app.include_router(hf_router)
app.include_router(agents_router)
app.include_router(chat_history_router)
app.include_router(chat_router)
app.include_router(rerank_router)
