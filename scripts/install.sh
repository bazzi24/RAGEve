#!/usr/bin/env bash
# ── RAGEve — One-time setup ───────────────────────────────────────────────────
# Installs all system dependencies, Python packages, Ollama models, and
# Docker containers so that run.sh works immediately after.
#
# Run once after cloning:
#   ./scripts/install.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
SCRIPT_DIR="$(pwd)"
source scripts/rageve.sh

# ── Banner ────────────────────────────────────────────────────────────────────
clear
show_banner

log_info "Starting RAGEve installation..."
log_info "This may take a few minutes on first run (models are ~8 GB)."
echo

# ── OS detection ──────────────────────────────────────────────────────────────
detect_os
log_info "Detected OS: $OS"

if [ "$OS" = "unknown" ]; then
  log_error "Unsupported operating system: $(uname -s)"
  log_info "Supported: macOS, Linux, Windows (WSL2)"
  exit 1
fi

if [ "$OS" = "windows" ]; then
  log_error "WSL2 detected. Run this script from inside WSL2."
  log_info "Make sure you are in the Linux shell, not in Windows CMD or PowerShell."
  exit 1
fi

# ── System dependencies ─────────────────────────────────────────────────────────
log_info "Installing system dependencies..."

install_packages() {
  case "$OS" in
    macos)
      if ! command -v brew &>/dev/null; then
        log_error "Homebrew not found."
        log_info "Install Homebrew: https://brew.sh"
        exit 1
      fi
      brew install curl git 2>/dev/null || true
      ;;
    linux)
      if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq curl git >/dev/null 2>&1
      elif command -v dnf &>/dev/null; then
        sudo dnf install -y curl git >/dev/null 2>&1
      elif command -v pacman &>/dev/null; then
        sudo pacman -Sy --noconfirm curl git >/dev/null 2>&1
      else
        log_warn "Could not detect a supported package manager (apt/dnf/pacman)."
        log_info "Please install curl and git manually, then re-run this script."
        exit 1
      fi
      ;;
  esac
}

install_packages
log_success "System dependencies ready."

# ── uv (Python package manager) ───────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  log_info "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  [ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
  export PATH="$HOME/.local/bin:$PATH"
  log_success "uv installed."
else
  log_success "uv already installed ($(uv --version | cut -d' ' -f1-2))."
fi

# ── Ollama ────────────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  log_info "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  log_success "Ollama installed."
else
  log_success "Ollama already installed ($(ollama --version | head -1))."
fi

# Start Ollama and wait for it to be ready
check_ollama

# ── Ollama models ─────────────────────────────────────────────────────────────
log_info "Checking Ollama models..."
check_ollama_model "nomic-embed-text"   "~274 MB"
check_ollama_model "llama3.2:latest"  "~7.4 GB"

# ── Python dependencies ─────────────────────────────────────────────────────────
log_info "Installing Python packages..."
if [ -f "pyproject.toml" ]; then
  uv sync 2>&1 | grep -v "^$" || true
  log_success "Python packages installed."
else
  log_warn "pyproject.toml not found — skipping Python deps."
fi

# ── Docker containers ──────────────────────────────────────────────────────────
log_info "Starting Docker containers (Qdrant + MySQL)..."
if check_docker; then
  docker compose -f docker/docker-compose.yml up -d qdrant mysql 2>&1 | grep -v "^$" || true

  # Wait for Qdrant to be healthy
  log_info "Waiting for Qdrant to be ready..."
  for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/collections &>/dev/null; then
      log_success "Qdrant is ready."
      break
    fi
    sleep 1
  done
else
  log_warn "Docker not available — skipping containers."
fi

# ── Done ───────────────────────────────────────────────────────────────────────
echo
log_success "RAGEve installation complete!"
echo
log_info "Next steps:"
echo
echo "  ${BOLD}Run everything:${NC}     ./scripts/run.sh"
echo "  ${BOLD}Backend only:${NC}       ./scripts/backend.sh  (for technical users)"
echo
echo "  Frontend:  http://localhost:3000"
echo "  Backend:   http://localhost:8000"
echo "  API docs:  http://localhost:8000/docs"
echo
