#!/usr/bin/env bash
# ── RAGEve — Start everything ──────────────────────────────────────────────────
# Auto-installs missing dependencies on first run, then starts:
#   Docker (Qdrant + MySQL) → Ollama → FastAPI backend → Next.js frontend
#
# Non-technical users: just run this.
#   ./scripts/run.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/rageve.sh

# ── Banner ─────────────────────────────────────────────────────────────────────
clear
show_banner

# ── Auto-install on first run ─────────────────────────────────────────────────
if [ ! -f .install_done ]; then
  log_info "First run — running installation (this takes a few minutes)..."
  echo
  if ! ./scripts/install.sh; then
    log_error "Installation failed. See errors above."
    exit 1
  fi
  touch .install_done
  clear
  show_banner
fi

# ── OS ─────────────────────────────────────────────────────────────────────────
detect_os

# ── Docker containers ─────────────────────────────────────────────────────────
if ! check_docker; then
  log_error "Docker must be running. Please start Docker Desktop and re-run."
  exit 1
fi

log_info "Starting Qdrant + MySQL..."
docker compose -f docker/docker-compose.yml up -d qdrant mysql 2>&1 | grep -v "^$" || true

# Wait for Qdrant
for i in $(seq 1 30); do
  curl -sf http://localhost:6333/collections &>/dev/null && break
  sleep 1
done
if curl -sf http://localhost:6333/collections &>/dev/null; then
  log_success "Qdrant ready."
else
  log_error "Qdrant failed to start. Check: docker compose -f docker/docker-compose.yml ps"
  exit 1
fi

# ── Ollama ─────────────────────────────────────────────────────────────────────
check_ollama

# ── Backend ───────────────────────────────────────────────────────────────────
log_info "Starting FastAPI backend..."
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000 &>/dev/null &
BACKEND_PID=$!
echo $BACKEND_PID > .backend.pid

# Wait for backend
for i in $(seq 1 15); do
  curl -sf http://localhost:8000/health &>/dev/null && break
  sleep 1
done
if curl -sf http://localhost:8000/health &>/dev/null; then
  log_success "Backend ready."
else
  log_warn "Backend may still be starting..."
fi

# ── Frontend ───────────────────────────────────────────────────────────────────
log_info "Starting Next.js frontend..."
cd frontend && npm run dev &>/dev/null &
FRONTEND_PID=$!
echo $FRONTEND_PID > .frontend.pid
cd ..

# Wait for frontend
for i in $(seq 1 30); do
  curl -sf http://localhost:3000 &>/dev/null && break
  sleep 1
done
if curl -sf http://localhost:3000 &>/dev/null; then
  log_success "Frontend ready."
else
  log_warn "Frontend may still be compiling..."
fi

# ── Done ───────────────────────────────────────────────────────────────────────
echo
log_success "${BOLD}RAGEve is running!${NC}"
echo
printf "  ${CYAN}%-30s${NC}  http://localhost:3000\n" "Frontend"
printf "  ${CYAN}%-30s${NC}  http://localhost:8000\n" "Backend (FastAPI)"
printf "  ${CYAN}%-30s${NC}  http://localhost:8000/docs\n" "API Docs (Swagger UI)"
printf "  ${CYAN}%-30s${NC}  http://localhost:6333\n" "Qdrant Dashboard"
echo
log_info "Press ${BOLD}Ctrl+C${NC} to stop all services."
echo

# ── Shutdown trap ─────────────────────────────────────────────────────────────
shutdown() {
  log_info "Stopping services..."
  kill $(cat .backend.pid 2>/dev/null) $(cat .frontend.pid 2>/dev/null) 2>/dev/null || true
  rm -f .backend.pid .frontend.pid
  docker compose -f docker/docker-compose.yml down 2>/dev/null || true
  log_success "All services stopped."
  exit 0
}
trap shutdown INT TERM

wait
