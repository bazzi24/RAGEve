#!/usr/bin/env python3
"""
Stress-test script for the ingestion pipeline.

Tests:
  1. Large PDF  (10 MB+) — generated with pymupdf
  2. Messy CSV  (messy unicode, missing values, mixed types)
  3. Corrupt / unsupported files — graceful 4xx handling
  4. Concurrent uploads — two files at once

Usage:
  uv run python _test_stress.py
  uv run python _test_stress.py --dry-run
  uv run python _test_stress.py --test pdf
  uv run python _test_stress.py --test csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import string
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DATASET_ID = f"stress-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
TEMP_DIR = Path(tempfile.mkdtemp(prefix="rag-stress-"))

LARGE_PDF_TARGET_MB = 10
CSV_ROWS = 50_000
CSV_COLS = 12


@dataclass
class PhaseResult:
    name: str
    elapsed_s: float
    ok: bool
    detail: str = ""


@dataclass
class FileTestResult:
    filename: str
    size_bytes: int
    phases: list[PhaseResult] = field(default_factory=list)
    error: str | None = None

    @property
    def total_s(self) -> float:
        return sum(p.elapsed_s for p in self.phases)

    @property
    def ok(self) -> bool:
        return self.error is None and all(p.ok for p in self.phases)


def generate_large_pdf(output_path: Path, target_mb: int = LARGE_PDF_TARGET_MB) -> int:
    import pymupdf

    print(f"  Generating {target_mb}+ MB PDF …", flush=True)

    base_par = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. "
    )
    block = "".join([base_par] * 20)

    doc = pymupdf.open()
    page_count = 0

    while True:
        page_count += 1
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 72), f"Stress Test Page {page_count}", fontsize=14, fontname="helv")

        y = 100
        while y < 790:
            page.insert_text((72, y), block, fontsize=10, fontname="helv")
            y += 14

        if page_count % 50 == 0:
            doc.save(str(output_path))
            size_mb = output_path.stat().st_size / 1024**2
            print(f"    pages={page_count}, size={size_mb:.2f} MB", flush=True)
            if size_mb >= target_mb:
                break

    doc.save(str(output_path))
    doc.close()

    actual = output_path.stat().st_size
    print(f"  PDF generated: {output_path.name} — {actual / 1024**2:.2f} MB ({page_count} pages)", flush=True)
    return actual


def generate_messy_csv(output_path: Path, rows: int = CSV_ROWS, cols: int = CSV_COLS) -> int:
    print(f"  Generating messy CSV ({rows:,} rows × {cols} cols) …", flush=True)

    unicode_samples = [
        "日本語テスト",
        "Ελληνικά",
        "العربية",
        "🎉 Celebration!",
        "\tTAB\t",
        "  leading spaces  ",
        'embedded "quotes"',
        "line1\nline2",
        "a" * 500,
        "",
    ]

    headers = [
        "ID", "Name", "Age", "Salary", "Is Active", "Join Date",
        "Email", "Department", "Notes", "Score", "Code", "Raw Data",
    ][:cols]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar="\\")
        writer.writerow(headers)

        base_date = date(2010, 1, 1)

        for i in range(rows):
            row = []
            for col_idx in range(cols):
                r = random.random()

                if col_idx == 0:
                    row.append(i if r > 0.05 else "")
                elif col_idx == 1:
                    if r < 0.03:
                        row.append(random.choice(unicode_samples))
                    else:
                        nm = "".join(random.choices(string.ascii_letters, k=random.randint(4, 20)))
                        row.append(f"  {nm}  " if r < 0.1 else nm)
                elif col_idx == 2:
                    row.append(random.choice(["N/A", "unknown", "NA"]) if r < 0.15 else random.randint(18, 80))
                elif col_idx == 3:
                    row.append(random.choice(["", "TBD", "-"]) if r < 0.1 else f"{random.uniform(20000, 200000):.2f}")
                elif col_idx == 4:
                    row.append(random.choice(["yes", "YES", "true", "1", ""]) if r < 0.05 else random.choice(["true", "false"]))
                elif col_idx == 5:
                    row.append(str(base_date + timedelta(days=random.randint(0, 5000))))
                elif col_idx == 6:
                    if r < 0.08:
                        row.append("invalid@@email..com")
                    else:
                        user = "".join(random.choices(string.ascii_lowercase, k=8))
                        row.append(f"{user}@example.com")
                elif col_idx == 7:
                    row.append(random.choice(["Engineering", "Sales", "HR", "Marketing", "", "Operations", "R&D"]))
                elif col_idx == 8:
                    row.append(random.choice(unicode_samples) if r < 0.2 else "Note: " + "".join(random.choices(string.ascii_letters, k=30)))
                elif col_idx == 9:
                    row.append(random.choice(["N/A", "", "err", "99.9*"]) if r < 0.12 else f"{random.uniform(0, 100):.4f}")
                elif col_idx == 10:
                    row.append("".join(random.choices(string.ascii_uppercase + string.digits, k=8)))
                else:
                    row.append(random.choice(unicode_samples))

            writer.writerow(row)
            if (i + 1) % 10000 == 0:
                print(f"    {i + 1:,} rows written …", flush=True)

    actual = output_path.stat().st_size
    print(f"  CSV generated: {output_path.name} — {actual / 1024**2:.2f} MB", flush=True)
    return actual


def curl_upload(dataset_id: str, file_path: Path) -> tuple[int, str, str]:
    cmd = [
        "curl", "-s", "-w", "\n%{http_code}",
        "-X", "POST",
        f"{BACKEND_URL}/datasets/{dataset_id}/upload",
        "-F", f"files=@{file_path}",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=3600)
    body = proc.stdout.decode("utf-8", errors="replace")
    parts = body.rsplit("\n", 1)
    if len(parts) == 2:
        status_body, status_code = parts
    else:
        status_body, status_code = body, "000"
    return int(status_code), status_body, proc.stderr.decode("utf-8", errors="replace")


def _stream_upload(dataset_id: str, file_path: Path):
    """
    Generator that yields NDJSON events from the /upload/stream endpoint.
    Uses httpx so we can stream without buffering the whole response.
    """
    import httpx

    with open(file_path, "rb") as f:
        files = {"files": (file_path.name, f, "application/octet-stream")}
        try:
            with httpx.stream(
                "POST",
                f"{BACKEND_URL}/datasets/{dataset_id}/upload/stream",
                files=files,
                timeout=3600.0,
            ) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    yield json.loads(line)
        except httpx.ReadTimeout:
            yield {"event": "error", "stage": "failed", "message": "Connection refused (Ollama down?)", "progress": 0}


def test_upload_stream(label: str, dataset_id: str, file_path: Path) -> FileTestResult:
    """
    Upload with streaming progress and print every stage transition.
    Returns a FileTestResult with per-phase breakdown.
    """
    size_mb = file_path.stat().st_size / 1024**2
    result = FileTestResult(filename=file_path.name, size_bytes=file_path.stat().st_size)

    print(f"\n{'='*60}")
    print(f"STREAM TEST: {label} ({size_mb:.2f} MB)")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    last_stage, stage_times = None, {}
    all_ok, error_msg, final_result = True, None, None
    embed_events = []

    for evt in _stream_upload(dataset_id, file_path):
        evt_name  = evt.get("event", "?")
        stage     = evt.get("stage", "")
        progress  = evt.get("progress", 0)
        message   = evt.get("message", "")
        elapsed_s = time.perf_counter() - t0

        if evt_name == "status":
            if stage != last_stage:
                if last_stage:
                    stage_times[last_stage] = elapsed_s
                last_stage = stage
                # Print progress bar
                bar_len = 30
                filled  = int(bar_len * progress / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"  [{elapsed_s:6.1f}s] {bar} {progress:3d}%  {stage:<16s}  {message}")

        elif evt_name == "file_done":
            final_result = evt.get("result", {})
            stage_times[stage] = time.perf_counter() - t0
            print(f"  [{elapsed_s:6.1f}s] {'█'*30} 100%  completed")

        elif evt_name == "done":
            all_ok = True
            files = evt.get("files", [])
            stage_times["completed"] = time.perf_counter() - t0
            print(f"\n  ✅ Stream complete — {len(files)} file(s) in {stage_times.get('completed', 0):.1f}s")
            result.phases.append(
                PhaseResult(name="full_pipeline_stream", elapsed_s=stage_times.get("completed", 0),
                            ok=True, detail=f"HTTP 200, {len(files)} files")
            )

        elif evt_name == "error":
            all_ok = False
            error_msg = evt.get("message", "unknown")
            stage_times["failed"] = time.perf_counter() - t0
            print(f"\n  ❌ Error at {progress}%: {error_msg}")
            result.error = error_msg
            result.phases.append(
                PhaseResult(name="full_pipeline_stream", elapsed_s=stage_times.get("failed", 0),
                            ok=False, detail=f"error: {error_msg}")
            )

    # Summary of stage timings
    print(f"\n  Stage breakdown:")
    prev_t = 0.0
    for s, t in stage_times.items():
        print(f"    {s:<16s}  {t - prev_t:6.2f}s  (cumulative {t:.2f}s)")
        prev_t = t

    return result


def test_upload(label: str, result: FileTestResult, dataset_id: str, file_path: Path):
    print(f"\n{'='*60}")
    print(f"TEST: {label} ({file_path.stat().st_size / 1024**2:.2f} MB)")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    status, body, stderr = curl_upload(dataset_id, file_path)
    elapsed = time.perf_counter() - t0

    result.phases.append(
        PhaseResult(name="full_pipeline_http", elapsed_s=elapsed, ok=status in (200, 201), detail=f"HTTP {status}")
    )

    if status == 200:
        try:
            payload = json.loads(body)
            f = (payload.get("files") or [{}])[0]
            print(f"  ✅ HTTP {status}")
            print(f"     chars   : {f.get('chars', '?')}")
            print(f"     chunks  : {f.get('chunks', '?')}")
            print(f"     profile : {f.get('quality_report', {}).get('selected_profile', '?')}")
            print(f"     score   : {f.get('quality_report', {}).get('quality_score', '?')}")
            print(f"     elapsed : {elapsed:.1f}s")
        except Exception:
            print(f"  ⚠️ HTTP 200 but unexpected payload: {body[:250]}")
    else:
        result.error = f"HTTP {status}: {body[:200]}"
        print(f"  ❌ HTTP {status}: {body[:250]}")


def test_unsupported_extension(dataset_id: str):
    bad = TEMP_DIR / "unsupported.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    print(f"\n{'='*60}")
    print("TEST: Unsupported .csv extension (expect 4xx)")
    print(f"{'='*60}")
    status, body, _ = curl_upload(dataset_id, bad)
    print(f"  {'✅' if status >= 400 else '❌'} HTTP {status}")
    if body:
        print(f"     {body[:200]}")


def test_concurrent_uploads(pdf_path: Path, xlsx_path: Path):
    print(f"\n{'='*60}")
    print("TEST: Concurrent uploads (PDF + XLSX)")
    print(f"{'='*60}")

    ds = f"{DATASET_ID}-concurrent"
    outputs: dict[str, tuple[int, float, str]] = {}

    def worker(tag: str, p: Path):
        t0 = time.perf_counter()
        status, body, _ = curl_upload(ds, p)
        outputs[tag] = (status, time.perf_counter() - t0, body)

    t1 = threading.Thread(target=worker, args=("pdf", pdf_path))
    t2 = threading.Thread(target=worker, args=("xlsx", xlsx_path))

    t0 = time.perf_counter()
    t1.start(); t2.start(); t1.join(); t2.join()
    total = time.perf_counter() - t0

    print(f"  Total elapsed (both): {total:.1f}s")
    for tag, (status, elapsed, body) in outputs.items():
        print(f"  {'✅' if status in (200, 201) else '❌'} [{tag}] HTTP {status} in {elapsed:.1f}s")


def print_ui_status_notes():
    print(f"""
{'='*60}
UI STATUS OBSERVATION
{'='*60}
Current Datasets page behavior (from frontend/src/app/datasets/page.tsx):
- On click Upload: setUploading(datasetId) => button shows 'Uploading…' (Processing state)
- Waits for single POST /datasets/{{id}}/upload request to finish
- On success: addToast success + render upload results
- On failure: addToast error
- Finally: setUploading(null)

There are no intermediate status events (extracting/chunking/embedding/upserting).
So the only observable state transition is:
  Processing (spinner) -> Completed (success toast/results)
  Processing (spinner) -> Failed (error toast)

No per-file progress percentage is currently implemented.
{'='*60}
""")


def convert_csv_to_xlsx(csv_path: Path, xlsx_path: Path) -> int:
    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="messy_data")
    return xlsx_path.stat().st_size


def backend_reachable() -> bool:
    try:
        r = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"{BACKEND_URL}/docs"],
            capture_output=True,
            timeout=10,
        )
        return r.stdout.decode().strip() == "200"
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Stress-test ingestion upload path")
    parser.add_argument("--dry-run", action="store_true", help="Generate files only")
    parser.add_argument("--keep-files", action="store_true", help="Keep generated files")
    parser.add_argument("--test", choices=["pdf", "csv", "all"], default="all")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming progress endpoint and print per-stage transitions",
    )
    args = parser.parse_args()

    print(f"Backend : {BACKEND_URL}")
    print(f"Dataset : {DATASET_ID}")
    print(f"Temp dir: {TEMP_DIR}\n")

    pdf_path = TEMP_DIR / "large_stress_test.pdf"
    csv_path = TEMP_DIR / "messy_stress_test.csv"
    xlsx_path = TEMP_DIR / "messy_stress_test.xlsx"

    results: list[FileTestResult] = []

    if args.test in ("pdf", "all"):
        pdf_size = generate_large_pdf(pdf_path, LARGE_PDF_TARGET_MB)
        results.append(FileTestResult(filename=pdf_path.name, size_bytes=pdf_size))

    if args.test in ("csv", "all"):
        csv_size = generate_messy_csv(csv_path, CSV_ROWS, CSV_COLS)
        xlsx_size = convert_csv_to_xlsx(csv_path, xlsx_path)
        print(f"  XLSX converted: {xlsx_path.name} — {xlsx_size / 1024**2:.2f} MB")
        results.append(FileTestResult(filename=xlsx_path.name, size_bytes=xlsx_size))

    if args.dry_run:
        print(f"\nDry-run complete. Files in: {TEMP_DIR}")
        if not args.keep_files:
            shutil.rmtree(TEMP_DIR)
        return

    if not backend_reachable():
        print("❌ Backend not reachable at http://localhost:8000")
        print("   Run: uv run uvicorn backend.main:app --reload --port 8000")
        return

    print_ui_status_notes()

    if args.stream:
        # ── Streaming progress test — prints every stage transition ─────────
        pdf_result: FileTestResult | None = None
        xlsx_result: FileTestResult | None = None

        if args.test in ("pdf", "all"):
            pdf_result = test_upload_stream("Large PDF", DATASET_ID, pdf_path)

        if args.test in ("csv", "all"):
            xlsx_result = test_upload_stream("Messy CSV as XLSX", DATASET_ID, xlsx_path)

        if args.test == "all":
            results = [r for r in [pdf_result, xlsx_result] if r is not None]
        else:
            results = [r for r in [pdf_result, xlsx_result] if r is not None]

    else:
        # ── Non-streaming (original sync upload) ───────────────────────────
        if args.test in ("pdf", "all"):
            pdf_result = next(r for r in results if r.filename == pdf_path.name)
            test_upload("Large PDF", pdf_result, DATASET_ID, pdf_path)

        if args.test in ("csv", "all"):
            xlsx_result = next(r for r in results if r.filename == xlsx_path.name)
            test_upload("Messy CSV as XLSX", xlsx_result, DATASET_ID, xlsx_path)

        test_unsupported_extension(DATASET_ID)

        if args.test == "all" and pdf_path.exists() and xlsx_path.exists():
            test_concurrent_uploads(pdf_path, xlsx_path)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in results:
        status = "✅ OK" if r.ok else f"❌ {r.error or 'FAILED'}"
        print(f"- {r.filename} ({r.size_bytes / 1024**2:.2f} MB): {status}")
        for p in r.phases:
            print(f"    · {p.name}: {p.elapsed_s:.1f}s ({p.detail})")

    print(f"\nDataset used: {DATASET_ID}")

    if not args.keep_files:
        shutil.rmtree(TEMP_DIR)
        print(f"Cleaned temp dir: {TEMP_DIR}")
    else:
        print(f"Kept temp dir: {TEMP_DIR}")


if __name__ == "__main__":
    main()
