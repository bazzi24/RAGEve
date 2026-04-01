#!/usr/bin/env bash
# ── RAGEve — Backend only ─────────────────────────────────────────────────────
# Starts: Ollama + Qdrant + MySQL (Docker) + FastAPI backend
# Does NOT start the Next.js frontend.
#
# For technical users who run the frontend manually (e.g. npm run dev).
#   ./scripts/backend.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/rageve.sh

# ── Banner ─────────────────────────────────────────────────────────────────────
clear
show_banner

log_info "Starting backend services (no frontend)..."
echo

# ── Ollama ─────────────────────────────────────────────────────────────────────
check_ollama

# ── Docker containers ──────────────────────────────────────────────────────────
if check_docker; then
  log_info "Starting Qdrant + MySQL..."
  docker compose -f docker/docker-compose.yml up -d qdrant mysql 2>&1 | grep -v "^$" || true

  for i in $(seq 1 30); do
    curl -sf http://localhost:6333/collections &>/dev/null && break
    sleep 1
  done
  if curl -sf http://localhost:6333/collections &>/dev/null; then
    log_success "Qdrant ready."
  fi
else
  log_warn "Docker not available."
fi

# ── Backend ───────────────────────────────────────────────────────────────────
log_info "Starting FastAPI on http://localhost:8000 ..."
echo
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
