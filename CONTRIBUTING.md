# Contributing to RAGEve

Thank you for your interest in contributing to RAGEve! This guide covers everything you need to know to get started, from setting up your dev environment to opening a pull request.

---

## Fork & Clone

RAGEve uses a **fork-based workflow**. External contributors should fork the repository, then clone their fork locally.

```bash
# Clone your fork
git clone https://github.com/<your-username>/RAGEve.git
cd RAGEve

# Add the upstream repo as a remote (so you can keep your fork in sync)
git remote add upstream https://github.com/bazzi24/RAGEve.git
```

## Keeping Your Fork in Sync

Before starting any new work, sync your fork with the upstream repository to ensure you're working on the latest code.

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
```

---

## Branch Naming

Use the following prefixes for your branches:

| Prefix | When to use |
|---|---|
| `feat/` | New features or enhancements |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes only |
| `refactor/` | Code refactoring without behavior change |
| `test/` | Adding or updating tests |
| `chore/` | Tooling, dependencies, CI/CD changes |
| `hotfix/` | Urgent production fixes |

**Examples:**

```bash
git checkout -b feat/huggingface-parquet-streaming
git checkout -b fix/chat-session-timeout
git checkout -b docs/api-rate-limiting
```

---

## Development Setup

See [README.md](./README.md#-launch-service-from-source-for-development) for full instructions. The short version:

```bash
# One-time setup
cp .env.example .env          # fill in API_KEY, HF_TOKEN, CORS_ORIGINS, etc.
./scripts/install.sh         # installs uv, Ollama, pulls models, starts Docker

# Run the full stack (backend + frontend + Ollama + Qdrant + MySQL)
./scripts/run.sh

# Backend only (if you run the frontend manually)
./scripts/backend.sh
```

### Prerequisites

- **Python 3.12+** — managed by `uv`
- **Ollama** — for local LLM inference
- **Docker** — for Qdrant and MySQL containers
- **Node.js 20+** — for the Next.js frontend

---

## Commit Message Format

RAGEve follows the [Conventional Commits](https://www.conventionalcommits.org/) specification. Every commit **must** use a type prefix.

### Format

```
<type>: <short summary>

[optional body]
```

### Type Prefixes

| Type | Description |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation-only changes |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or correcting tests |
| `chore` | Maintenance tasks (deps, CI, tooling) |
| `hotfix` | Emergency production fix |

### Rules

1. **Use imperative mood** in the summary line: `"add export"` not `"added export"` or `"adds export"`
2. **Keep the summary under 72 characters**
3. **Reference issues** in the body when applicable: `"Closes #42"`
4. **Blank line** between summary and body (if a body is needed)

### Examples

```
feat: add HuggingFace dataset streaming ingest

Implements background ingest for large HF datasets using a staged
NDJSON progress stream. Closes #38.
```

```
fix: prevent circuit breaker from blocking health checks

The /health endpoint now bypasses the Ollama circuit breaker to
report accurate degraded status without triggering a new failure.
```

```
docs: add performance benchmark section to README
```

```
hotfix: restore missing abort controller cleanup in ingest poll
```

---

## Testing

**All tests must pass before submitting a pull request.**

### Running Tests

```bash
# End-to-end tests (from project root)
uv run python test/_test_e2e.py

# Stress tests (optional but recommended for streaming/RAG changes)
uv run python test/_test_stress.py --test all --stream --keep-files
```

### What to Test

- **Bug fixes** — verify the fix resolves the issue; add a regression test if none exists
- **New features** — run the e2e suite to confirm nothing is broken
- **Backend changes** — run the e2e tests, especially if you modified API routes
- **Frontend changes** — verify the UI renders correctly; run `npm run lint` in the `frontend/` directory

### Test File Conventions

- Tests live in `test/` at the project root
- E2E tests: `test/_test_e2e.py`
- Stress tests: `test/_test_stress.py`
- When adding new test coverage, mirror the existing file structure

---

## Pull Request Process

### Before Opening a PR

1. **Sync your branch** with the latest `upstream/main`
2. **Run all tests** — see [Testing](#testing) above
3. **Lint the frontend**:
   ```bash
   cd frontend && npm run lint && cd ..
   ```
4. **Check for console errors** in the browser dev tools
5. **Write a clear PR description** — see below

### PR Description Template

```markdown
## Summary

<!-- 1-3 bullet points: what changed and why -->

## Type

<!-- feat | fix | docs | refactor | test | chore | hotfix -->

## Test Plan

<!-- How did you verify this works? What should reviewers check? -->

## Screenshots (if applicable)

<!-- UI changes: before/after screenshots -->

## Checklist

- [ ] Tests pass locally (`uv run python test/_test_e2e.py`)
- [ ] Conventional commit format followed
- [ ] No console errors in the browser
- [ ] Documentation updated (if applicable)
```

### PR Checklist

- [ ] Branch is up to date with `upstream/main`
- [ ] All tests pass
- [ ] Commit messages use the [Conventional Commits](#commit-message-format) format
- [ ] PR targets `main`
- [ ] No unrelated changes bundled in

---

## Branch Protection

The `main` branch is **protected**. Direct pushes are not allowed. All changes must come through a pull request that satisfies the following requirements:

- **At least 1 reviewer** approval (required for all PRs)
- **All CI checks pass** (tests, linting)
- **Linear history** — squash merges preferred over merge commits

### CI Checks

The following checks run on every pull request:

1. `test/_test_e2e.py` — end-to-end tests
2. `frontend` linting — `npm run lint`
3. Branch name check — must match `feat/`, `fix/`, `docs/`, etc.

### Emergency Hotfixes

For urgent production fixes, use the `hotfix/` branch prefix. Hotfixes still require a PR and CI to pass, but can be fast-tracked for review.

---

## Reporting Bugs

When reporting a bug, please include:

- **Clear description** of the issue
- **Steps to reproduce**
- **Expected vs. actual behavior**
- **Environment details**: OS, Python version, Ollama version, Docker version
- **Relevant logs** — attach the log file from `data/logs/` if applicable
- **Minimum reproduction** — if possible, isolate the issue to a specific action

Use the [GitHub Issues](../issues) page to report bugs. Tag issues with `bug`.

---

## Suggesting Features

Feature requests are welcome! Please open a [GitHub Discussion](../discussions) or issue tagged `enhancement` and include:

- **Problem** — what limitation or missing capability does this address?
- **Proposed solution** — how should it work?
- **Alternatives considered** — what else did you evaluate?

---

## Code of Conduct

Be respectful, inclusive, and constructive. We welcome contributors of all backgrounds and experience levels.

---

## Getting Help

- Open an [issue](../issues) for bugs or feature requests
- Check [README.md](./README.md) for setup and usage documentation
- Review the [CLAUDE.md](./CLAUDE.md) for architecture and codebase context
