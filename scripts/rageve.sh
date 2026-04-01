# ── RAGEve shared library ─────────────────────────────────────────────────────
# Source this file from any script:  source scripts/rageve.sh

# ── Colour codes ───────────────────────────────────────────────────────────────
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m' # No Colour

# ── Log helpers ────────────────────────────────────────────────────────────────
log_info()    { printf "${CYAN}[*]${NC} %s\n" "$*"; }
log_success() { printf "${GREEN}[✓]${NC} %s\n" "$*"; }
log_warn()    { printf "${YELLOW}[!]${NC} %s\n" "$*"; }
log_error()   { printf "${RED}[✗]${NC} %s\n" "$*" >&2; }

# ── ASCII banner ────────────────────────────────────────────────────────────────
show_banner() {
  cat << 'EOF'
  
██████╗  █████╗  ██████╗ ███████╗██╗   ██╗███████╗    
██╔══██╗██╔══██╗██╔════╝ ██╔════╝██║   ██║██╔════╝    
██████╔╝███████║██║  ███╗█████╗  ██║   ██║█████╗      
██╔══██╗██╔══██║██║   ██║██╔══╝  ╚██╗ ██╔╝██╔══╝      
██║  ██║██║  ██║╚██████╔╝███████╗ ╚████╔╝ ███████╗    
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝  ╚═══╝  ╚══════╝    
                                                      

  AI-powered RAG platform — Ollama · Qdrant · FastAPI · Next.js
  https://github.com/bazzi24/RAGEve

EOF
}

# ── OS detection ────────────────────────────────────────────────────────────────
# Sets $OS to: macos | linux | windows | unknown
detect_os() {
  local uname_out
  uname_out=$(uname -s)
  case "$uname_out" in
    Darwin*)  OS=macos ;;
    Linux*)
      if [ -d /proc/sys/fs/binfmt_misc/WSL ] || grep -qi microsoft /proc/version 2>/dev/null; then
        OS=windows
      else
        OS=linux
      fi
      ;;
    *) OS=unknown ;;
  esac
}

# ── Dependency helpers ─────────────────────────────────────────────────────────

# usage: check_command curl "Curl" "brew install curl"
# Exits if command not found.
check_command() {
  local cmd=$1; shift
  local label=$1; shift
  local install_hint=$*
  if ! command -v "$cmd" &>/dev/null; then
    log_error "$label is not installed."
    log_info "Please install it: $install_hint"
    return 1
  fi
}

# Checks if Ollama is running. Starts it in the background if not.
check_ollama() {
  if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    log_info "Starting Ollama..."
    ollama serve &>/dev/null &
    # Wait for it to be ready (up to 20 s)
    for i in $(seq 1 20); do
      sleep 1
      curl -sf http://localhost:11434/api/tags &>/dev/null && break
    done
    if curl -sf http://localhost:11434/api/tags &>/dev/null; then
      log_success "Ollama is running."
    else
      log_error "Ollama failed to start. Is Ollama installed? Run: curl -fsSL https://ollama.com/install.sh | sh"
      return 1
    fi
  fi
}

# Checks if an Ollama model is pulled. Pulls it if not.
# usage: check_ollama_model "nomic-embed-text" "~274 MB"
check_ollama_model() {
  local model=$1; shift
  local size_hint=${1:-""}
  if [ -n "$size_hint" ]; then
    size_hint=" ($size_hint)"
  fi
  if ollama list 2>/dev/null | grep -qE "^${model}[[:space:]]"; then
    log_success "Model '$model' already pulled."
  else
    log_info "Pulling model: ${model}${size_hint}"
    if ! ollama pull "$model"; then
      log_error "Failed to pull model '$model'."
      return 1
    fi
    log_success "Model '$model' ready."
  fi
}

# Checks if Docker is running (docker info succeeds).
check_docker() {
  if ! docker info &>/dev/null; then
    log_error "Docker is not running. Please start Docker Desktop (or dockerd)."
    return 1
  fi
}
