"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useDatasetsStore } from "@/stores/useDatasetsStore";
import { useToastStore } from "@/stores/useToastStore";
import { uploadFilesStreaming, deleteDataset, listDatasets } from "@/lib/api/datasets";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { Accordion, AccordionItem } from "@/components/ui/Accordion";
import type { ProcessedFileResponse } from "@/lib/types";
import styles from "./DatasetsPage.module.css";

// ── Limits ───────────────────────────────────────────────────────────────────

const MAX_FILE_BYTES   = 500 * 1024 * 1024;      // 500 MB per file
const MAX_DATASET_BYTES = 100 * 1024 ** 3;        // 100 GB per dataset

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function scoreColor(score: number): string {
  if (score > 0.85) return "var(--success)";
  if (score > 0.7) return "var(--warning)";
  return "var(--error)";
}

// ── Component ────────────────────────────────────────────────────────────────

export default function DatasetsPage() {
  const {
    datasets,
    setDatasets,
    uploadingDatasetId,
    setUploading,
    uploadResults,
    setUploadResults,
    uploadProgress,
    setUploadProgress,
    resetUploadProgress,
    removeDataset,
    setError,
  } = useDatasetsStore();
  const { addToast } = useToastStore();

  // ── Upload form state ──────────────────────────────────────────────────
  const [datasetId, setDatasetId] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Held in a ref so the AbortController is stable across re-renders.
  const cancelRef = useRef<AbortController | null>(null);

  // ── Fetch datasets on mount ──────────────────────────────────────────────
  useEffect(() => {
    listDatasets()
      .then((res) => setDatasets(res.datasets))
      .catch((e) => addToast(`Failed to load datasets: ${e.message}`, "error"));
  }, [setDatasets, addToast]);

  const canUpload = datasetId.trim().length > 0 && selectedFiles.length > 0;
  const isUploading = uploadingDatasetId != null;

  // ── File selection handlers ──────────────────────────────────────────────

  const handleFilesChosen = useCallback((files: File[]) => {
    const oversized: string[] = [];
    for (const f of files) {
      if (f.size > MAX_FILE_BYTES) {
        oversized.push(`${f.name} (${fmtBytes(f.size)} — max ${fmtBytes(MAX_FILE_BYTES)} per file)`);
      }
    }
    if (oversized.length > 0) {
      addToast(`File(s) too large:\n${oversized.join("\n")}`, "error");
      return;
    }

    setSelectedFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name));
      const newFiles = files.filter((f) => !existing.has(f.name));

      // Warn (but don't block) if the new total would exceed the dataset limit.
      const existingTotal = prev.reduce((sum, f) => sum + f.size, 0);
      const newTotal = newFiles.reduce((sum, f) => sum + f.size, 0);
      if (existingTotal + newTotal > MAX_DATASET_BYTES) {
        addToast(
          `Total upload size (${fmtBytes(existingTotal + newTotal)}) exceeds the `
            + `${fmtBytes(MAX_DATASET_BYTES)} dataset limit.`,
          "warning"
        );
      }

      return [...prev, ...newFiles];
    });
  }, [addToast]);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length) handleFilesChosen(files);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length) handleFilesChosen(files);
    // Reset the input so the same file can be re-selected
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleRemoveFile = (name: string) => {
    setSelectedFiles((prev) => prev.filter((f) => f.name !== name));
  };

  const handleClearFiles = () => setSelectedFiles([]);

  // ── Upload ──────────────────────────────────────────────────────────────

  const handleUpload = useCallback(async () => {
    if (!canUpload || isUploading) return;

    // Create a new AbortController for this upload.
    cancelRef.current = new AbortController();

    setUploading(datasetId);
    setUploadResults(datasetId, []);
    setUploadProgress({
      active: true,
      stage: "starting",
      message: "Starting upload…",
      progress: 0,
      file: undefined,
      file_index: undefined,
      file_total: selectedFiles.length,
      chunks_done: undefined,
      chunks_total: undefined,
    });

    let cancelled = false;

    try {
      await uploadFilesStreaming(
        datasetId,
        selectedFiles,
        {
          onStatus: (event) => {
            setUploadProgress({
              active: true,
              stage: event.stage,
              message: event.message,
              progress: event.progress,
              file: event.file,
              file_index: event.file_index,
              file_total: event.file_total,
              chunks_done: event.chunks_done,
              chunks_total: event.chunks_total,
            });
          },
          onFileDone: (event) => {
            const current = useDatasetsStore.getState().uploadResults[datasetId] ?? [];
            setUploadResults(datasetId, [...current, event.result]);
          },
          onDone: async (event) => {
            setUploadResults(datasetId, event.files);
            setUploadProgress({
              active: true,
              stage: "completed",
              message: event.message,
              progress: 100,
            });

            try {
              const res = await listDatasets();
              setDatasets(res.datasets);
            } catch {
              // Non-fatal: upload already succeeded
            }

            const totalChunks = event.files.reduce((sum, f) => sum + f.chunks, 0);
            addToast(
              `${event.files.length} file(s) uploaded — ${totalChunks.toLocaleString()} chunks embedded`,
              "success"
            );
            setSelectedFiles([]);
          },
          onError: (event) => {
            if (cancelled) return;
            const msg = event.message || "Upload failed";
            setUploadProgress({
              active: true,
              stage: "failed",
              message: msg,
              progress: event.progress,
            });
            setError(msg);
            addToast(`Upload failed: ${msg}`, "error");
          },
        },
        undefined,
        cancelRef.current.signal,
      );
    } catch (err) {
      if (cancelled) return;
      const msg = err instanceof Error ? err.message : String(err);
      addToast(`Upload failed: ${msg}`, "error");
      setError(msg);
      setUploadProgress({
        active: true,
        stage: "failed",
        message: msg,
      });
    } finally {
      cancelRef.current = null;
      setUploading(null);
      if (!cancelled) {
        setTimeout(() => {
          resetUploadProgress();
        }, 1500);
      }
    }
  }, [
    canUpload,
    isUploading,
    datasetId,
    selectedFiles,
    setUploading,
    setUploadResults,
    setUploadProgress,
    resetUploadProgress,
    addToast,
    setError,
    setDatasets,
  ]);

  // ── Cancel upload ────────────────────────────────────────────────────────

  const handleCancel = useCallback(() => {
    cancelRef.current?.abort();
    addToast("Upload cancelled", "info");
  }, [addToast]);

  // ── Delete ──────────────────────────────────────────────────────────────

  const handleDelete = async (id: string) => {
    try {
      await deleteDataset(id);
      removeDataset(id);
      addToast(`Dataset "${id}" deleted`, "info");
    } catch (err) {
      addToast(`Delete failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h1 className={styles.title}>Datasets</h1>
      </div>

      {/* ── Upload & Ingest section ─────────────────────────────────── */}
      <div className={styles.section}>
        <div className={styles.sectionTitle}>Upload &amp; Ingest</div>

        {/* Dataset ID field */}
        <div className={styles.datasetIdRow}>
          <div className={styles.datasetIdField}>
            <Input
              id="dataset-id"
              label="Dataset ID"
              placeholder="e.g. my-documents, research-papers"
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              disabled={isUploading}
            />
          </div>
          <div className={styles.uploadActions}>
            <Button
              onClick={handleUpload}
              disabled={!canUpload}
              loading={isUploading}
              title={!canUpload ? "Enter a dataset ID and select files first" : ""}
            >
              {isUploading ? "Uploading…" : selectedFiles.length > 0
                ? `Upload ${selectedFiles.length} file${selectedFiles.length !== 1 ? "s" : ""}`
                : "Upload"}
            </Button>
          </div>
        </div>

        {/* Drop zone — click opens file picker; also accepts drag-and-drop */}
        <div
          className={`${styles.dropzone} ${isDragging ? styles.active : ""} ${selectedFiles.length > 0 ? styles.hasFiles : ""}`}
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click(); }}
          aria-label="Select files to upload"
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.doc,.docx,.xlsx,.png,.jpg,.jpeg,.bmp,.tiff"
            onChange={handleFileInput}
            style={{ display: "none" }}
            aria-hidden="true"
          />

          {selectedFiles.length === 0 ? (
            <>
              <svg className={styles.dropzoneIcon} viewBox="0 0 40 40" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M20 28V14M13 21l7-7 7 7" />
                <path d="M6 28v4a2 2 0 0 0 2 2h24a2 2 0 0 0 2-2v-4" />
              </svg>
              <div className={styles.dropzoneTitle}>
                {isDragging ? "Drop files here" : "Drag & drop files or click to browse"}
              </div>
              <div className={styles.dropzoneSub}>
                PDF, DOCX, XLSX, PNG, JPG, TIFF
              </div>
              <div className={styles.dropzoneLimits}>
                Max {fmtBytes(MAX_FILE_BYTES)} per file · {fmtBytes(MAX_DATASET_BYTES)} per dataset
              </div>
            </>
          ) : (
            <div className={styles.selectedFiles}>
              <div className={styles.selectedFilesHeader}>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M9 2H4a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1V6L9 2z" />
                  <path d="M9 2v4h4" />
                </svg>
                <span>{selectedFiles.length} file{selectedFiles.length !== 1 ? "s" : ""} selected</span>
                <button
                  className={styles.clearBtn}
                  onClick={(e) => { e.stopPropagation(); handleClearFiles(); }}
                  type="button"
                >
                  Clear all
                </button>
              </div>
              <ul className={styles.fileList}>
                {selectedFiles.map((f) => (
                  <li key={f.name} className={styles.fileItem}>
                    <div className={styles.fileName} title={f.name}>{f.name}</div>
                    <div className={styles.fileSize}>{fmtBytes(f.size)}</div>
                    <button
                      className={styles.removeFileBtn}
                      onClick={(e) => { e.stopPropagation(); handleRemoveFile(f.name); }}
                      type="button"
                      title={`Remove ${f.name}`}
                      aria-label={`Remove ${f.name}`}
                    >
                      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M2 2l8 8M10 2l-8 8" />
                      </svg>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {selectedFiles.length > 0 && !canUpload && (
          <p className={styles.uploadHint}>
            Enter a Dataset ID above to enable the Upload button.
          </p>
        )}

        {uploadProgress.active && (
          <div className={styles.progressCard} role="status" aria-live="polite">
            <div className={styles.progressTopRow}>
              <span className={styles.progressStage}>{uploadProgress.stage}</span>
              <div className={styles.progressActions}>
                <span className={styles.progressPct}>{uploadProgress.progress}%</span>
                <button
                  className={styles.cancelBtn}
                  onClick={handleCancel}
                  type="button"
                  title="Cancel upload"
                  aria-label="Cancel upload"
                >
                  Cancel
                </button>
              </div>
            </div>
            <div className={styles.progressMessage}>{uploadProgress.message}</div>
            <div className={styles.progressBar}>
              <div
                className={styles.progressFill}
                style={{ width: `${Math.max(0, Math.min(100, uploadProgress.progress))}%` }}
              />
            </div>
            {(uploadProgress.file || uploadProgress.file_index) && (
              <div className={styles.progressMeta}>
                {uploadProgress.file && <span>{uploadProgress.file}</span>}
                {uploadProgress.file_index != null && uploadProgress.file_total != null && (
                  <span>
                    File {uploadProgress.file_index} / {uploadProgress.file_total}
                  </span>
                )}
                {uploadProgress.chunks_done != null && uploadProgress.chunks_total != null && (
                  <span>
                    Chunks {uploadProgress.chunks_done} / {uploadProgress.chunks_total}
                  </span>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Upload results ───────────────────────────────────────────── */}
      {Object.entries(uploadResults).map(([dsId, files]) => (
        <div key={dsId} className={styles.section}>
          <div className={styles.sectionTitle}>Results — {dsId}</div>
          <div className={styles.resultsList}>
            {files.map((file) => (
              <UploadCard key={file.filename} file={file} scoreColor={scoreColor} />
            ))}
          </div>
        </div>
      ))}

      {/* ── Dataset list ─────────────────────────────────────────────── */}
      {datasets.length > 0 && (
        <div className={styles.section}>
          <div className={styles.sectionTitle}>Uploaded Datasets</div>
          <div className={styles.grid}>
            {datasets.map((ds) => (
              <div key={ds.dataset_id} className={styles.card}>
                <div className={styles.cardHeader}>
                  <span className={styles.cardTitle} title={ds.dataset_id}>{ds.dataset_id}</span>
                  <Badge variant="accent">{ds.status}</Badge>
                </div>
                <div className={styles.cardStats}>
                  <div className={styles.stat}>
                    <span className={styles.statValue}>{ds.chunks_count.toLocaleString()}</span>
                    <span className={styles.statLabel}>chunks</span>
                  </div>
                  <div className={styles.stat}>
                    <span className={styles.statValue}>{ds.vector_size}</span>
                    <span className={styles.statLabel}>vectors</span>
                  </div>
                </div>
                <div className={styles.cardFooter}>
                  <Button
                    variant="danger"
                    size="sm"
                    onClick={() => handleDelete(ds.dataset_id)}
                  >
                    Delete
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Upload card sub-component ─────────────────────────────────────────────────

function UploadCard({
  file,
  scoreColor,
}: {
  file: ProcessedFileResponse;
  scoreColor: (s: number) => string;
}) {
  const qs = file.quality_report?.quality_score ?? 0;

  return (
    <div className={styles.uploadCard}>
      <div className={styles.uploadCardHeader}>
        <div className={styles.uploadCardName} title={file.filename}>{file.filename}</div>
        <Badge variant={qs > 0.85 ? "success" : qs > 0.7 ? "warning" : "error"}>
          {qs.toFixed(2)}
        </Badge>
      </div>

      <div className={styles.uploadCardBody}>
        <div className={styles.statRow}>
          <div className={styles.stat}>
            <span className={styles.statValue}>{file.chars.toLocaleString()}</span>
            <span className={styles.statLabel}>chars</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statValue}>{file.chunks.toLocaleString()}</span>
            <span className={styles.statLabel}>chunks</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statValue}>{file.quality_report?.selected_profile ?? "—"}</span>
            <span className={styles.statLabel}>profile</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statValue}>{file.extraction?.extractor ?? "—"}</span>
            <span className={styles.statLabel}>extractor</span>
          </div>
          {file.layout_summary && (
            <div className={styles.stat}>
              <span className={styles.statValue}>{file.layout_summary.pages}p</span>
              <span className={styles.statLabel}>pages</span>
            </div>
          )}
        </div>

        {/* Quality bar */}
        <div className={styles.scoreBar}>
          <div
            className={styles.scoreFill}
            style={{
              width: `${qs * 100}%`,
              backgroundColor: scoreColor(qs),
            }}
          />
        </div>

        {/* Quality signals accordion */}
        {file.quality_report && (
          <Accordion>
            <AccordionItem title="Quality Report" defaultOpen={false}>
              <div className={styles.signalsGrid}>
                {Object.entries(file.quality_report.signals || {}).map(([key, val]) => (
                  <div key={key} className={styles.signalRow}>
                    <span className={styles.signalKey}>
                      {key.replace(/_/g, " ")}
                    </span>
                    <span className={styles.signalVal}>
                      {typeof val === "number" ? (val as number).toFixed(4) : String(val)}
                    </span>
                  </div>
                ))}
              </div>
            </AccordionItem>
          </Accordion>
        )}
      </div>
    </div>
  );
}
