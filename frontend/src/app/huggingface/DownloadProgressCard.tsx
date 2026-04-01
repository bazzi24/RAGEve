"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/Button";
import type {
  HuggingFaceDownloadStatusResponse,
  HuggingFacePreviewResponse,
} from "@/lib/types";
import styles from "./HuggingFacePage.module.css";

const TERMINAL_STATES = new Set(["completed", "failed", "cancelled"]);

const _fmtBytes = (n: number | null | undefined): string => {
  if (n == null) return "—";
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
};

interface DownloadProgressCardProps {
  downloadStatus: HuggingFaceDownloadStatusResponse;
  preview: HuggingFacePreviewResponse | null;
  autoIngestEnabled: boolean;
  ingestCompleted: boolean;
  ingestFailed: boolean;
  isIngesting: boolean;
  datasets: { dataset_id: string }[];
  setActiveIngestDatasetId: (id: string) => void;
  onPanelUpdate: (id: string, updates: Record<string, unknown>) => void;
  onCancel: () => void;
}

export function DownloadProgressCard({
  downloadStatus,
  preview,
  autoIngestEnabled,
  ingestCompleted,
  ingestFailed,
  isIngesting,
  datasets,
  setActiveIngestDatasetId,
  onPanelUpdate,
  onCancel,
}: DownloadProgressCardProps) {
  const router = useRouter();

  // Prefer preview's full_dataset_id when available; fall back to raw dataset_id
  const datasetName = preview?.full_dataset_id ?? downloadStatus.dataset_id;

  const isCompleted = downloadStatus.status === "completed";
  const isFailed = downloadStatus.status === "failed";
  const isCancelled = downloadStatus.status === "cancelled";
  const isDownloading = !TERMINAL_STATES.has(downloadStatus.status);
  const showSuccessState = isCompleted && !isDownloading;

  // Stage computation
  const stages = [
    { key: "downloading", label: "Downloading" },
    { key: "extracting", label: "Extracting" },
    { key: "embedding", label: "Embedding" },
    { key: "ready", label: "Ready" },
  ];

  const getStageState = (key: string) => {
    if (isFailed) return "error";
    if (isCancelled) return "pending";
    if (key === "ready") {
      if (isCompleted) return "done";
      if (isDownloading) return "pending";
    }
    if (key === "downloading") {
      if (isCompleted) return "done";
      if (isDownloading) return "active";
      return "pending";
    }
    if (key === "extracting") {
      if (isCompleted) return "done";
      if (isDownloading && downloadStatus.progress > 30) return "active";
      return "pending";
    }
    if (key === "embedding") {
      if (isCompleted) return "done";
      if (isIngesting) return "active";
      if (isDownloading && downloadStatus.progress > 60) return "active";
      return "pending";
    }
    return "pending";
  };

  const handleIngestNow = () => {
    const id = downloadStatus.dataset_id;
    setActiveIngestDatasetId(id);
    const ds = datasets.find((d) => d.dataset_id === id);
    if (ds) {
      onPanelUpdate(id, { expanded: true });
    }
  };

  return (
    <div className={styles.progressCard}>
      {/* Header */}
      <div className={styles.dlProgressHeader}>
        <div className={styles.dlProgressHeaderLeft}>
          <span className={styles.dlProgressDatasetName}>
            {isCompleted ? "✓ " : isFailed ? "✗ " : isCancelled ? "— " : ""}
            {datasetName}
            {downloadStatus.config && ` › ${downloadStatus.config}`}
          </span>
          <span className={styles.dlProgressSub}>
            {isIngesting ? "Indexing…" : ingestCompleted ? "Indexed!" : ingestFailed ? "Indexing failed" : ""}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            className={`${styles.dlProgressPct} ${
              isCompleted ? styles.dlProgressPctDone : isFailed ? styles.dlProgressPctError : ""
            }`}
          >
            {isCompleted ? "✓ Done" : isFailed ? "✗ Failed" : isCancelled ? "— Cancelled" : `${downloadStatus.progress}%`}
          </span>
          {isDownloading && (
            <Button variant="danger" size="sm" onClick={onCancel}>
              Cancel
            </Button>
          )}
        </div>
      </div>

      {/* Stage dots strip */}
      {!isCompleted && !isFailed && !isCancelled && (
        <div className={styles.dlProgressStages}>
          {stages.map((s) => {
            const state = getStageState(s.key);
            return (
              <div
                key={s.key}
                className={`${styles.dlStage} ${styles[`dlStage${state.charAt(0).toUpperCase() + state.slice(1)}`]}`}
              >
                <span className={styles.dlStageDot} />
                <span>{s.label}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Progress bar — hidden on terminal states */}
      {!isCompleted && !isFailed && !isCancelled && (
        <>
          <div className={styles.progressBarWrap}>
            <div
              className={`${styles.progressFill} ${
                isIngesting ? "" : styles.progressFillAnimating
              }`}
              style={{ width: `${Math.max(0, Math.min(100, downloadStatus.progress))}%` }}
            />
          </div>
          <div className={styles.dlProgressMeta}>
            <span>
              {downloadStatus.bytes_downloaded != null
                ? _fmtBytes(downloadStatus.bytes_downloaded)
                : "—"}
            </span>
            <span>
              {downloadStatus.total_bytes != null
                ? `/ ${_fmtBytes(downloadStatus.total_bytes)}`
                : "downloading…"}
            </span>
          </div>
        </>
      )}

      {/* Terminal-state full bar */}
      {(isCompleted || isFailed || isCancelled) && (
        <div className={styles.progressBarWrap}>
          <div
            className={`${styles.progressFill} ${
              isCompleted ? styles.progressFillSuccess : isFailed ? styles.progressFillError : ""
            }`}
            style={{ width: "100%" }}
          />
        </div>
      )}

      {/* Ingest sub-bar */}
      {isCompleted && autoIngestEnabled && (
        <div className={styles.ingestSubRow}>
          <div className={styles.ingestSubLabel}>
            {isIngesting
              ? "Indexing…"
              : ingestCompleted
              ? "✓ Indexed & ready to chat!"
              : ingestFailed
              ? "✗ Indexing failed"
              : "Indexing…"}
          </div>
          <div className={styles.ingestSubBar}>
            <div
              className={`${styles.ingestSubFill} ${
                ingestCompleted ? styles.ingestSubFillDone : ingestFailed ? styles.ingestSubFillError : ""
              }`}
              style={{
                width: isIngesting ? "80%" : ingestCompleted ? "100%" : ingestFailed ? "100%" : "0%",
              }}
            />
          </div>
        </div>
      )}

      {/* Message */}
      <div className={styles.progressMsg}>{downloadStatus.message}</div>

      {/* Ingest error */}
      {isIngesting && downloadStatus.ingest_message && (
        <div className={styles.progressSubMsg}>{downloadStatus.ingest_message}</div>
      )}

      {/* Errors */}
      {isFailed && downloadStatus.error && (
        <div className={styles.progressError}>Error: {downloadStatus.error}</div>
      )}
      {ingestFailed && downloadStatus.ingest_error && (
        <div className={styles.progressError}>Indexing error: {downloadStatus.ingest_error}</div>
      )}

      {/* Success CTA */}
      {showSuccessState && (
        <div className={styles.dlProgressCta}>
          {!autoIngestEnabled && !ingestCompleted && (
            <Button onClick={handleIngestNow}>Ingest Now</Button>
          )}
          {ingestCompleted && <Button onClick={() => router.push("/chat")}>Go to Chat</Button>}
          {!ingestCompleted && (
            <Button variant="secondary" onClick={() => router.push("/chat")}>
              Go to Chat
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
