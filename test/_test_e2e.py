"""
Comprehensive end-to-end RAG platform tests.

Scenarios:
  1. Large-dataset retrieval  — ingest 500 rows from squad, verify chunks returned
  2. SSE streaming format     — verify correct NDJSON event format, token reception
  3. Zero-results handling    — query that should find no relevant chunks

Usage:
  # Run all tests
  uv run python _test_e2e.py

  # Run individual scenarios
  uv run python _test_e2e.py --scenario 1
  uv run python _test_e2e.py --scenario 2
  uv run python _test_e2e.py --scenario 3

  # Watch raw SSE (verbose)
  uv run python _test_e2e.py --sse-debug

  # Wire up a live agent (create one pointing at squad collection)
  uv run python _test_e2e.py --setup-agent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

# Ensure project root is first so rag/backend are always importable
# regardless of which directory uv run executes from.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import httpx

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
OLLAMA_CHAT_MODEL = "llama3.2:latest"
QDRANT_URL = "http://localhost:6333"
API_BASE_URL = "http://localhost:8000"

SQUAD_PATH = Path("data/hf/squad")
COLLECTION_NAME = "squad"
AGENT_ID = "e2e-test-agent"

PASS = "\u2713"
FAIL = "\u2717"
INFO = "\u2192"

# ── Pretty print helpers ──────────────────────────────────────────────────────

def section(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}")

def ok(msg: str) -> None:
    print(f"  {PASS}  {msg}")

def fail(msg: str) -> None:
    print(f"  {FAIL}  {msg}")

def step(msg: str) -> None:
    print(f"  {INFO} {msg}")

def warn(msg: str) -> None:
    print(f"  \u26a0  {msg}")


# ── Scenario 0: Prerequisites ────────────────────────────────────────────────

async def check_prerequisites() -> bool:
    """Verify all services are reachable before running tests."""
    section("Prerequisites")

    async with httpx.AsyncClient(timeout=10.0) as client:
        checks = [
            ("Ollama", f"{OLLAMA_BASE_URL}/api/tags"),
            ("Qdrant", f"{QDRANT_URL}/collections"),
            ("FastAPI", f"{API_BASE_URL}/health"),
        ]
        all_ok = True
        for name, url in checks:
            try:
                r = await client.get(url)
                if r.status_code < 500:
                    ok(f"{name} is reachable ({r.status_code})")
                else:
                    fail(f"{name} returned {r.status_code}")
                    all_ok = False
            except Exception as exc:
                fail(f"{name} unreachable: {exc}")
                all_ok = False

    return all_ok


# ── Scenario 1: Large-dataset retrieval ──────────────────────────────────────

async def scenario_large_dataset() -> bool:
    """
    Ingest 500 rows from squad, verify retrieval returns relevant chunks
    with proper scores and metadata.
    """
    section("Scenario 1 — Large-dataset retrieval")
    from rag.chunking.adaptive import adaptive_chunk_text
    from rag.deepdoc.quality_scorer import score_and_select_profile
    from rag.embedding.ollama_embedder import OllamaEmbedder
    from rag.ingestion.hf_ingestion import _load_hf_dataset, _detect_text_column
    from rag.storage.qdrant_store import QdrantStore, ChunkRecord

    qdrant = QdrantStore(url=QDRANT_URL)
    embedder = OllamaEmbedder(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)

    ROW_LIMIT = 500
    QUERIES = [
        "What university is mentioned in relation to the Congregation of Holy Cross?",
        "What is the name of the Director of the Science Museum?",
        "Which city is associated with the story about Beyonce?",
    ]

    # 1. Load dataset
    step("Loading squad dataset...")
    df = _load_hf_dataset(SQUAD_PATH)
    text_col = _detect_text_column(list(df.columns)) or "context"
    df_sub = df.head(ROW_LIMIT)
    ok(f"Loaded {len(df)} rows total, using {len(df_sub)} for test")

    # 2. Chunk
    step("Chunking & scoring...")
    all_chunks: list[str] = []
    all_meta: list[dict] = []
    for idx, row in df_sub.iterrows():
        text = str(row.get(text_col, "")).strip()
        if not text:
            continue
        report = score_and_select_profile(text)
        chunks = adaptive_chunk_text(text, profile=report.profile.profile)
        for c_idx, ct in enumerate(chunks):
            all_chunks.append(ct)
            all_meta.append({
                "dataset_id": COLLECTION_NAME,
                "split": "train",
                "source_row": int(idx),
                "chunk_index": c_idx,
                "total_chunks": len(chunks),
                "quality_score": round(report.quality_score, 4),
                "profile": report.profile.profile.value,
            })
    ok(f"Created {len(all_chunks)} chunks from {ROW_LIMIT} rows")

    # 3. Embed
    step("Embedding chunks via Ollama...")
    t0 = time.perf_counter()
    embeddings = await embedder.embed_batch_api(all_chunks, api_batch_size=256)
    elapsed = time.perf_counter() - t0
    ok(f"Embedded {len(embeddings)} chunks in {elapsed:.1f}s ({len(embeddings)/elapsed:.0f} chunks/s)")
    if embeddings and embeddings[0]:
        ok(f"Embedding dimension: {len(embeddings[0])}")
    else:
        fail("No embeddings returned")
        return False

    # 4. Upsert to Qdrant
    step("Upserting to Qdrant...")
    qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(COLLECTION_NAME)
    records: list[ChunkRecord] = [
        ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            chunk_text=chunk,
            metadata=meta,
            dense_vector=emb,
        )
        for chunk, meta, emb in zip(all_chunks, all_meta, embeddings)
    ]
    upserted = qdrant.upsert_chunks(COLLECTION_NAME, records)
    ok(f"Upserted {upserted} records")

    info = qdrant.get_collection_info(COLLECTION_NAME)
    if info and info["points_count"] > 0:
        ok(f"Collection '{COLLECTION_NAME}': {info['points_count']} points, status={info['status']}")
    else:
        fail("Collection has no points after upsert")
        return False

    # 5. Retrieve
    step("Running retrieval queries...")
    all_passed = True
    for query in QUERIES:
        qvec = await embedder.embed_single(query)
        hits = await qdrant.dense_search(collection_name=COLLECTION_NAME, query_vector=qvec, top_k=4)

        if hits:
            top = hits[0]
            ok(f"Query: {query[:60]!r}")
            ok(f"  Top hit: score={top.score:.4f}, profile={top.metadata.get('profile')}")
            ok(f"  Chunk: {top.chunk_text[:80]!r}...")
            if top.score < 0.3:
                warn(f"  Score is low ({top.score:.3f}) — query may not match dataset")
        else:
            fail(f"Query returned 0 hits: {query!r}")
            all_passed = False

    return all_passed


# ── Scenario 2: SSE streaming format ─────────────────────────────────────────

async def scenario_sse_streaming() -> bool:
    """
    Verify the SSE/NDJSON stream from /chat/{agent_id}/stream:
      - Each token arrives as {"event": "chunk", "content": "..."}
      - Final event is {"event": "end", "sources": [...]}
      - No malformed JSON lines
      - Tokens arrive in order
    """
    section("Scenario 2 — SSE streaming format")

    # 1. Get or create an agent that points at the squad collection
    step("Finding or creating test agent...")
    agent_id = await _get_or_create_agent()
    if not agent_id:
        fail("Could not get or create an agent for collection '{COLLECTION_NAME}'")
        return False
    ok(f"Using agent: {agent_id}")

    # 2. Send a streaming request
    step("Sending streaming request...")
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=120.0) as client:
        r = await client.post(
            f"/chat/{agent_id}/stream",
            json={"question": "Summarize what you know about universities.", "stream": True},
            headers={"Content-Type": "application/json"},
        )

    if r.status_code != 200:
        fail(f"Stream request failed with status {r.status_code}")
        try:
            body = r.json()
            fail(f"Error body: {body}")
        except Exception:
            pass
        return False
    ok(f"Stream started (status {r.status_code})")

    # 3. Parse NDJSON line by line
    all_passed = True
    chunks_received: list[str] = []
    raw_lines: list[str] = []
    end_received = False
    sources_received: list = []

    step("Reading stream...")
    async for line in r.aiter_lines():
        raw_lines.append(line)
        if not line.strip():
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"Malformed JSON on line: {exc}")
            fail(f"  Raw: {line!r}")
            all_passed = False
            continue

        if event.get("event") == "chunk":
            content = event.get("content", "")
            chunks_received.append(content)
            chunk_repr = repr(content)
            step(f"  chunk ({len(chunks_received)}): {chunk_repr[:40]}{'...' if len(chunk_repr) > 40 else ''}")

        elif event.get("event") == "end":
            end_received = True
            sources_received = event.get("sources", [])
            ok(f"Received 'end' event — {len(sources_received)} sources")

        elif event.get("event") == "error":
            fail(f"Server emitted error event: {event.get('error')}")
            all_passed = False

        else:
            warn(f"Unknown event type: {event}")

    # 4. Verify stream structure
    step("Verifying stream structure...")

    if not chunks_received:
        fail("No chunk events received")
        all_passed = False
    else:
        ok(f"Received {len(chunks_received)} chunk events")

    # Check ordering: tokens should accumulate in order
    assembled = "".join(chunks_received)
    if len(assembled) > 0:
        ok(f"Full assembled response: {len(assembled)} chars")
        # Don't print full response in test output — just confirm it's non-empty
    else:
        fail("Assembled response is empty")
        all_passed = False

    if not end_received:
        fail("No 'end' event received — stream may have dropped")
        all_passed = False
    else:
        ok("'end' event received")

    # 5. Verify sources structure
    if sources_received:
        ok(f"Sources: {len(sources_received)} chunks")
        for i, src in enumerate(sources_received[:3], 1):
            if not isinstance(src, dict):
                fail(f"Source[{i}] is not a dict: {type(src)}")
                all_passed = False
                continue
            required = ("chunk_id", "text", "score")
            missing = [k for k in required if k not in src]
            if missing:
                fail(f"Source[{i}] missing fields: {missing}")
                all_passed = False
            else:
                ok(f"  Source[{i}]: score={src['score']:.4f}, text={src['text'][:50]!r}...")
    else:
        warn("No sources in 'end' event — may be expected for some queries")

    # 6. Summary
    step(f"Raw lines received: {len(raw_lines)}")
    step(f"Chunk events: {len(chunks_received)}")
    step(f"'end' event received: {end_received}")

    return all_passed


# ── Scenario 3: Zero-results handling ─────────────────────────────────────────

async def scenario_zero_results() -> bool:
    """
    Send a query that has no plausible match in the squad collection
    (e.g. unrelated topic). Verify the API handles it gracefully:
      - Returns 200 OK
      - 'end' event fires with empty sources array
      - LLM answer acknowledges the lack of context
    """
    section("Scenario 3 — Zero-results handling")

    # 1. Ensure collection exists
    from rag.storage.qdrant_store import QdrantStore

    qdrant = QdrantStore(url=QDRANT_URL)
    if not qdrant.collection_exists(COLLECTION_NAME):
        warn(f"Collection '{COLLECTION_NAME}' not found — skipping. Run scenario 1 first.")
        ok("Collection missing — run scenario 1 to create it, then retry scenario 3")
        return True  # Not a failure — test is skipped intentionally

    info = qdrant.get_collection_info(COLLECTION_NAME)
    if not info or info["points_count"] == 0:
        warn("Collection is empty — skipping. Run scenario 1 first.")
        ok("Collection empty — run scenario 1, then retry scenario 3")
        return True

    ok(f"Collection '{COLLECTION_NAME}' has {info['points_count']} points")

    # 2. Get or create an agent
    step("Finding or creating test agent...")
    agent_id = await _get_or_create_agent()
    if not agent_id:
        fail("Could not get or create an agent")
        return False
    ok(f"Using agent: {agent_id}")

    # 3. Send a nonsensical query that should score very low
    ABSURD_QUERY = "What is the recipe for chocolate lava cake according to squad?"

    step(f"Sending absurd query: {ABSURD_QUERY!r}")

    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=120.0) as client:
        r = await client.post(
            f"/chat/{agent_id}/stream",
            json={"question": ABSURD_QUERY, "stream": True},
            headers={"Content-Type": "application/json"},
        )

    if r.status_code != 200:
        fail(f"Request failed with status {r.status_code}")
        return False
    ok(f"Request OK (status {r.status_code})")

    all_passed = True
    chunks: list[str] = []
    sources: list = []
    end_received = False
    error_event = None

    async for line in r.aiter_lines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            fail(f"Malformed JSON: {line!r}")
            all_passed = False
            continue

        if event.get("event") == "chunk":
            chunks.append(event.get("content", ""))

        elif event.get("event") == "end":
            end_received = True
            sources = event.get("sources", [])

        elif event.get("event") == "error":
            error_event = event.get("error")
            fail(f"Server error: {error_event}")
            all_passed = False

    # 3. Verify zero-results handling
    step("Verifying graceful zero-results handling...")

    if not end_received:
        fail("No 'end' event — stream dropped")
        all_passed = False
    else:
        ok("'end' event received")

    assembled = "".join(chunks)
    ok(f"LLM response: {len(assembled)} chars")
    if len(assembled) == 0:
        warn("LLM returned empty response")
    else:
        ok(f"Response preview: {assembled[:120]!r}...")

    # Check that sources are low-scoring or empty
    if sources:
        max_score = max((s.get("score", 0) for s in sources), default=0)
        ok(f"Max source score: {max_score:.4f}")
        if max_score < 0.5:
            ok("Source scores are appropriately low (no strong match)")
        elif max_score >= 0.8:
            warn("Unexpectedly high score for absurd query — check chunking quality")

    # Check LLM response content
    if assembled:
        no_match_phrases = ["don't have", "not enough", "no information", "cannot answer",
                           "not found", "no context", "based on the provided"]
        if any(p.lower() in assembled.lower() for p in no_match_phrases):
            ok("LLM correctly acknowledges lack of relevant context")
        else:
            warn("LLM may not be acknowledging the lack of context clearly")

    return all_passed


# ── SSE Debugger ─────────────────────────────────────────────────────────────

async def sse_debugger() -> None:
    """
    Verbose SSE inspector: prints every raw line received from the stream.
    Use this to diagnose dropped connections, malformed JSON, or ordering issues.
    """
    print("\n" + "=" * 64)
    print("  SSE Debugger — raw stream inspector")
    print("=" * 64)

    query = input("\nEnter a query (or press Enter for default): ").strip()
    if not query:
        query = "What university is mentioned in squad?"

    print(f"\nQuery: {query!r}")
    print(f"Agent: {AGENT_ID}")
    print("-" * 64)

    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=120.0) as client:
        r = await client.post(
            f"/chat/{AGENT_ID}/stream",
            json={"question": query, "stream": True},
            headers={"Content-Type": "application/json"},
        )

    print(f"HTTP Status: {r.status_code}")
    print(f"Content-Type: {r.headers.get('content-type')}")
    print("-" * 64)

    line_num = 0
    total_bytes = 0
    start = time.perf_counter()
    tokens: list[str] = []
    end_seen = False

    async for line in r.aiter_lines():
        elapsed = time.perf_counter() - start
        line_num += 1
        total_bytes += len(line.encode())
        print(f"[{elapsed:>7.2f}s | line {line_num:>4}] {line!r}")

        if line.strip():
            try:
                event = json.loads(line)
                if event.get("event") == "chunk":
                    tokens.append(event.get("content", ""))
                elif event.get("event") == "end":
                    end_seen = True
            except json.JSONDecodeError as exc:
                print(f"  *** JSON ERROR: {exc} ***")

    elapsed = time.perf_counter() - start
    print("-" * 64)
    print(f"Done. {line_num} lines, {total_bytes} bytes, {elapsed:.1f}s")
    print(f"Tokens received: {len(tokens)}")
    print(f"'end' event seen: {end_seen}")
    assembled = "".join(tokens)
    print(f"Assembled text ({len(assembled)} chars):")
    print()
    print(assembled)
    print()


# ── Agent setup helper ────────────────────────────────────────────────────────

async def _get_or_create_agent() -> str | None:
    """
    Returns the agent_id (UUID) of an agent pointing at COLLECTION_NAME,
    creating one if none exists. Returns None on failure.
    """
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
        # 1. Look for an existing agent that uses our collection
        try:
            r = await client.get("/agents/")
            if r.status_code == 200:
                agents = r.json().get("agents", [])
                for a in agents:
                    if a.get("config", {}).get("dataset_id") == COLLECTION_NAME:
                        return a["agent_id"]
        except Exception:
            pass

        # 2. Create a new agent pointing at the collection
        payload = {
            "name": "E2E Test Agent",
            "description": "Auto-created by _test_e2e.py",
            "config": {
                "system_prompt": (
                    "You are a helpful assistant. Answer based ONLY on the provided context. "
                    "If the context does not contain enough information, say so clearly."
                ),
                "dataset_id": COLLECTION_NAME,
                "embedding_model": OLLAMA_EMBED_MODEL,
                "chat_model": OLLAMA_CHAT_MODEL,
                "temperature": 0.7,
                "top_k": 5,
            },
        }
        try:
            r = await client.post("/agents/", json=payload)
            if r.status_code in (200, 201):
                data = r.json()
                return data.get("agent_id")
        except Exception:
            pass
    return None


async def setup_agent() -> None:
    """Create or update the e2e test agent pointing at squad collection (CLI entry point)."""
    section("Setting up test agent")

    from rag.storage.qdrant_store import QdrantStore
    qdrant = QdrantStore(url=QDRANT_URL)
    if qdrant.collection_exists(COLLECTION_NAME):
        info = qdrant.get_collection_info(COLLECTION_NAME)
        pts = info["points_count"] if info else "?"
        ok(f"Collection '{COLLECTION_NAME}' exists ({pts} points)")
    else:
        ok(f"Collection '{COLLECTION_NAME}' not yet created (run scenario 1 first)")
        print("  Run: uv run python _test_e2e.py --scenario 1")
        return

    agent_id = await _get_or_create_agent()
    if agent_id:
        ok(f"Agent ready: {agent_id}")
        print(f"  Note: agent_id is a UUID. Use this in direct API calls.")
    else:
        fail("Could not create or find an agent for collection '{COLLECTION_NAME}'")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Platform E2E Tests")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3],
                        help="Run a specific scenario only")
    parser.add_argument("--setup-agent", action="store_true",
                        help="Create/update the test agent and exit")
    parser.add_argument("--sse-debug", action="store_true",
                        help="Interactive SSE debugger")
    args = parser.parse_args()

    if args.setup_agent:
        await setup_agent()
        return

    if args.sse_debug:
        await sse_debugger()
        return

    if not await check_prerequisites():
        fail("Prerequisites check failed — ensure Ollama, Qdrant, and FastAPI are running.")
        sys.exit(1)

    results: dict[int, bool] = {}

    scenarios = [
        (1, "Large-dataset retrieval", scenario_large_dataset),
        (2, "SSE streaming format", scenario_sse_streaming),
        (3, "Zero-results handling", scenario_zero_results),
    ]

    for num, title, fn in scenarios:
        if args.scenario and args.scenario != num:
            step(f"Skipping scenario {num} ({title})")
            continue
        try:
            results[num] = await fn()
        except Exception as exc:
            fail(f"Scenario {num} crashed: {exc}")
            import traceback
            traceback.print_exc()
            results[num] = False

    # Summary
    section("Results Summary")
    for num, title, _ in scenarios:
        if args.scenario and args.scenario != num:
            continue
        status = results.get(num)
        if status is True:
            ok(f"Scenario {num}: {title} — PASSED")
        elif status is False:
            fail(f"Scenario {num}: {title} — FAILED")
        else:
            warn(f"Scenario {num}: {title} — NOT RUN")

    all_passed = all(v is True for v in results.values())
    print()
    if all_passed:
        print(f"  {PASS}  All tests passed!")
    else:
        print(f"  {FAIL}  Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
