"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { Select } from "@/components/ui/Select";
import { MultiSelect } from "@/components/ui/MultiSelect";
import { Badge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Spinner";
import type {
  DiscoveredDataset,
  HFIngestStatusResponse,
  HFIngestSubmitResult,
} from "@/lib/types";
import { cancelHFIngest } from "@/lib/api/datasets";
import type { ToastVariant } from "@/stores/useToastStore";
import styles from "./HuggingFacePage.module.css";

// ── Types ───────────────────────────────────────────────────────────────────

type IngestPanelState = {
  expanded: boolean;
  loading: boolean;
  ingestId: string | null;
  ingestStatus: HFIngestStatusResponse | null;
  result: HFIngestSubmitResult | null;
  selectedSplit: string;
  selectedTextCols: string[];
  rowLimit: string;
};

// ── Helpers ─────────────────────────────────────────────────────────────────

const fmtSize = (bytes: number) => {
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
};

const scoreColor = (s: number) =>
  s > 0.85 ? "var(--success)" : s > 0.7 ? "var(--warning)" : "var(--error)";

// ── Ingest Progress Card ─────────────────────────────────────────────────────

interface IngestProgressCardProps {
  datasetId: string;
  ingestStatus: HFIngestStatusResponse;
  ingestId: string;
  stopIngestPolling: () => void;
  addToast: (msg: string, variant?: ToastVariant) => void;
  onCancel: () => void;
}

function IngestProgressCard({
  datasetId,
  ingestStatus,
  ingestId,
  stopIngestPolling,
  addToast,
  onCancel,
}: IngestProgressCardProps) {
  const st = ingestStatus;
  return (
    <div className={styles.resultCard} style={{ borderLeft: "3px solid #3b82f6" }}>
      <div className={styles.resultHeader}>
        <div className={styles.resultTitle} style={{ color: "#3b82f6" }}>
          &#x25B2; Ingesting {datasetId}
        </div>
        <Button
          variant="danger"
          size="sm"
          onClick={() => {
            cancelHFIngest(ingestId).then(() => {
              stopIngestPolling();
              onCancel();
            }).catch((e) => addToast(`Cancel failed: ${e}`, "error"));
          }}
        >
          Cancel
        </Button>
      </div>
      <div className={styles.resultStats}>
        <div className={styles.resultStat}>
          <span className={styles.resultStatLabel}>Progress</span>
          <span className={styles.resultStatValue}>{st.progress}%</span>
        </div>
        <div className={styles.resultStat}>
          <span className={styles.resultStatLabel}>Chunks</span>
          <span className={styles.resultStatValue}>
            {st.chunks_done.toLocaleString()} / {st.chunks_total.toLocaleString()}
          </span>
        </div>
        <div className={styles.resultStat}>
          <span className={styles.resultStatLabel}>Stage</span>
          <span className={styles.resultStatValue} style={{ textTransform: "capitalize" }}>
            {st.current_stage}
          </span>
        </div>
      </div>
      <div className={styles.ingestSubBar}>
        <div
          className={styles.ingestSubFill}
          style={{ width: `${st.progress}%` }}
        />
      </div>
      <p className={styles.resultHint}>{st.message}</p>
    </div>
  );
}

// ── Ingest Result Card ──────────────────────────────────────────────────────

interface IngestResultCardProps {
  datasetId: string;
  result: HFIngestSubmitResult;
  isIndexed: boolean;
  onReingest: () => void;
  router: ReturnType<typeof useRouter>;
}

function IngestResultCard({ datasetId, result, isIndexed, onReingest, router }: IngestResultCardProps) {
  const r = result as {
    rows_processed?: number;
    chunks_embedded?: number;
    avg_quality_score?: number;
    text_columns_used?: string[];
    profiles_used?: Record<string, number>;
    message?: string;
  } | null;
  if (!r) return null;

  return (
    <div className={styles.resultCard}>
      <div className={styles.resultHeader}>
        <div className={styles.resultTitle}>
          &#x2713; {isIndexed ? "Indexed" : "Ingestion complete"} — {datasetId}
        </div>
        <div className={styles.resultHeaderActions}>
          <Button variant="ghost" size="sm" onClick={() => router.push("/chat")}>
            Go to Chat →
          </Button>
          {isIndexed && (
            <Button variant="danger" size="sm" onClick={onReingest}>
              Re-ingest
            </Button>
          )}
        </div>
      </div>
      <div className={styles.resultStats}>
        <div className={styles.resultStat}>
          <span className={styles.resultStatLabel}>Rows</span>
          <span className={styles.resultStatValue}>{(r.rows_processed ?? 0).toLocaleString()}</span>
        </div>
        <div className={styles.resultStat}>
          <span className={styles.resultStatLabel}>Chunks</span>
          <span className={styles.resultStatValue}>{(r.chunks_embedded ?? 0).toLocaleString()}</span>
        </div>
        <div className={styles.resultStat}>
          <span className={styles.resultStatLabel}>Quality</span>
          <span
            className={styles.resultStatValue}
            style={{ color: scoreColor(r.avg_quality_score ?? 0) }}
          >
            {(r.avg_quality_score ?? 0).toFixed(3)}
          </span>
        </div>
        <div className={styles.resultStat}>
          <span className={styles.resultStatLabel}>Columns</span>
          <span className={styles.resultStatValue}>{r.text_columns_used?.join(", ")}</span>
        </div>
      </div>
      {r.profiles_used && Object.keys(r.profiles_used).length > 0 && (
        <p className={styles.resultHint}>
          Profiles: {Object.entries(r.profiles_used).map(([k, v]) => `${k} (${v})`).join(", ")}
        </p>
      )}
      {r.message && <p className={styles.resultHint}>{r.message}</p>}
    </div>
  );
}

// ── Library Card ─────────────────────────────────────────────────────────────

interface LibraryCardProps {
  ds: DiscoveredDataset;
  panel: IngestPanelState | undefined;
  onPanelUpdate: (id: string, updates: Partial<IngestPanelState>) => void;
  onIngest: (ds: DiscoveredDataset, force?: boolean) => Promise<void>;
  stopIngestPolling: () => void;
  addToast: (msg: string, variant?: ToastVariant) => void;
  router: ReturnType<typeof useRouter>;
}

function LibraryCard({
  ds,
  panel,
  onPanelUpdate,
  onIngest,
  stopIngestPolling,
  addToast,
  router,
}: LibraryCardProps) {
  const alreadyIngested = ds.is_ingested || panel?.result != null;

  const updatePanel = (updates: Partial<IngestPanelState>) =>
    onPanelUpdate(ds.dataset_id, updates);

  return (
    <div className={styles.libraryCard}>
      <div className={styles.libraryCardTop}>
        <div className={styles.libraryCardIcon}>⬡</div>
        <div className={styles.libraryCardInfo}>
          <div className={styles.libraryCardName}>
            {ds.dataset_id}
            {ds.is_ingested && <span className={styles.ingestedDot} title="Indexed in Qdrant" />}
          </div>
          <div className={styles.libraryCardMeta}>
            <span>{ds.file_count} file(s)</span>
            <span>{fmtSize(ds.total_size_bytes)}</span>
            <span>{ds.splits.join(", ")}</span>
          </div>
          <div className={styles.libraryCardBadges}>
            {ds.is_ingested ? (
              <Badge variant="success">Indexed</Badge>
            ) : (
              <Badge variant="default">Not Indexed</Badge>
            )}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className={styles.libraryCardActions}>
        {alreadyIngested ? (
          <>
            <Button size="sm" onClick={() => router.push("/chat")}>
              Chat
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => void onIngest(ds)}
            >
              Re-ingest
            </Button>
          </>
        ) : (
          <Button
            size="sm"
            onClick={() => updatePanel({ expanded: !panel?.expanded })}
          >
            {panel?.expanded ? "Cancel" : "Ingest →"}
          </Button>
        )}
      </div>

      {/* Ingest panel */}
      {panel?.expanded && !alreadyIngested && (
        <div className={styles.libraryCardIngestPanel}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <Select
              id={`split-${ds.dataset_id}`}
              label="Split"
              value={panel.selectedSplit}
              onChange={(e) => updatePanel({ selectedSplit: e.target.value })}
              options={
                ds.splits.length > 0
                  ? ds.splits.map((s) => ({ value: s, label: s }))
                  : [{ value: "train", label: "train" }]
              }
            />
            <MultiSelect
              id={`col-${ds.dataset_id}`}
              label="Columns to Embed"
              hint="Selected columns combined for semantic search."
              options={ds.readable_columns.map((c) => ({ value: c, label: c }))}
              selected={panel.selectedTextCols}
              onChange={(cols) => updatePanel({ selectedTextCols: cols })}
            />
          </div>
          <div style={{ marginTop: 10 }}>
            <label
              style={{ fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)", display: "block", marginBottom: 4 }}
            >
              Row Limit
            </label>
            <input
              type="text"
              placeholder="Leave empty = full"
              value={panel.rowLimit}
              onChange={(e) => updatePanel({ rowLimit: e.target.value })}
              style={{
                width: "100%",
                background: "var(--bg-primary)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-md)",
                padding: "5px 9px",
                fontSize: 12,
                fontFamily: "var(--font-sans)",
                color: "var(--text-primary)",
                boxSizing: "border-box",
              }}
            />
          </div>
          <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginTop: 12 }}>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => updatePanel({ expanded: false })}
            >
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={() => void onIngest(ds)}
              loading={panel.loading}
            >
              {panel.rowLimit ? `Ingest ${panel.rowLimit} rows` : "Ingest Full Dataset"}
            </Button>
          </div>
        </div>
      )}

      {/* Live ingest progress */}
      {panel?.ingestId != null && panel?.ingestStatus != null && (
        <IngestProgressCard
          datasetId={ds.dataset_id}
          ingestStatus={panel.ingestStatus}
          ingestId={panel.ingestId}
          stopIngestPolling={stopIngestPolling}
          addToast={addToast}
          onCancel={() => {
            onPanelUpdate(ds.dataset_id, {
              ingestId: null,
              ingestStatus: null,
              loading: false,
            });
          }}
        />
      )}

      {/* Completed result */}
      {panel?.result != null && (
        <IngestResultCard
          datasetId={ds.dataset_id}
          result={panel.result}
          isIndexed={!!ds.is_ingested}
          onReingest={() => void onIngest(ds)}
          router={router}
        />
      )}
    </div>
  );
}

// ── Main Library Component ───────────────────────────────────────────────────

type FilterType = "all" | "indexed" | "not_indexed";

interface LocalDatasetsLibraryProps {
  datasets: DiscoveredDataset[];
  ingestPanels: Record<string, IngestPanelState>;
  discovering: boolean;
  downloadStatus: unknown;
  isDownloading: boolean;
  onDiscover: () => Promise<void>;
  onRestartPolling: (datasetId: string) => void;
  onPanelUpdate: (id: string, updates: Partial<IngestPanelState>) => void;
  onIngest: (ds: DiscoveredDataset, force?: boolean) => Promise<void>;
  stopIngestPolling: () => void;
  addToast: (msg: string, variant?: ToastVariant) => void;
  router: ReturnType<typeof useRouter>;
}

export function LocalDatasetsLibrary({
  datasets,
  ingestPanels,
  discovering,
  downloadStatus,
  isDownloading,
  onDiscover,
  onRestartPolling,
  onPanelUpdate,
  onIngest,
  stopIngestPolling,
  addToast,
  router,
}: LocalDatasetsLibraryProps) {
  const [filter, setFilter] = useState<FilterType>("all");

  const indexedCount = datasets.filter((d) => d.is_ingested).length;
  const notIndexedCount = datasets.length - indexedCount;

  const filtered =
    filter === "all"
      ? datasets
      : filter === "indexed"
      ? datasets.filter((d) => d.is_ingested)
      : datasets.filter((d) => !d.is_ingested);

  const handleRefresh = useCallback(async () => {
    await onDiscover();
    const saved = window.localStorage.getItem("hf_active_download_dataset_id");
    if (saved) {
      onRestartPolling(saved);
    }
  }, [onDiscover, onRestartPolling]);

  return (
    <div className={styles.localSection}>
      {/* Header */}
      <div className={styles.libraryHeader}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className={styles.localTitle}>Library</span>
          <span className={styles.libraryCount}>{datasets.length}</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          {/* Filter tabs */}
          <div className={styles.libraryFilterTabs}>
            <button
              className={`${styles.libraryFilterTab} ${filter === "all" ? styles.libraryFilterTabActive : ""}`}
              onClick={() => setFilter("all")}
            >
              All
              <span className={styles.libraryCount}>{datasets.length}</span>
            </button>
            <button
              className={`${styles.libraryFilterTab} ${filter === "indexed" ? styles.libraryFilterTabActive : ""}`}
              onClick={() => setFilter("indexed")}
            >
              Indexed
              <span className={styles.libraryCount}>{indexedCount}</span>
            </button>
            <button
              className={`${styles.libraryFilterTab} ${filter === "not_indexed" ? styles.libraryFilterTabActive : ""}`}
              onClick={() => setFilter("not_indexed")}
            >
              Not Indexed
              <span className={styles.libraryCount}>{notIndexedCount}</span>
            </button>
          </div>

          <Button
            variant="secondary"
            size="sm"
            onClick={() => void handleRefresh()}
            loading={discovering}
          >
            ↻ Refresh
          </Button>
        </div>
      </div>

      {/* Active download banner */}
      {isDownloading && !!downloadStatus && (
        <div className={styles.activeDownloadBanner} style={{ marginBottom: "var(--space-4)" }}>
          <span className={styles.bannerDot} />
          <span>Downloading active — see progress above</span>
        </div>
      )}

      {/* Grid */}
      {datasets.length === 0 ? (
        <div className={styles.libraryCardEmptyState}>
          No local datasets found. Start a download above to get started.
        </div>
      ) : filtered.length === 0 ? (
        <div className={styles.libraryCardEmptyState}>
          {filter === "indexed" ? "No indexed datasets yet." : "All datasets are indexed!"}
        </div>
      ) : (
        <>
          <div className={styles.libraryGrid}>
            {filtered.map((ds) => (
              <LibraryCard
                key={ds.dataset_id}
                ds={ds}
                panel={ingestPanels[ds.dataset_id]}
                onPanelUpdate={(id, updates) => onPanelUpdate(id, updates as Partial<IngestPanelState>)}
                onIngest={onIngest}
                stopIngestPolling={stopIngestPolling}
                addToast={addToast}
                router={router}
              />
            ))}
          </div>
          {filtered.length < datasets.length && (
            <div className={styles.libraryCardCount}>
              Showing {filtered.length} of {datasets.length} datasets
            </div>
          )}
        </>
      )}
    </div>
  );
}
