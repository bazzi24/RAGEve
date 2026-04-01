"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  discoverHFDatasets,
  downloadHFDataset,
  getHFDownloadStatus,
  cancelHFDownload,
  previewHFDataset,
  submitHFIngest,
  getHFIngestStatus,
} from "@/lib/api/datasets";
import { useToastStore } from "@/stores/useToastStore";
import type {
  DiscoveredDataset,
  HuggingFaceDownloadStatusResponse,
  HuggingFacePreviewResponse,
  HFIngestStatusResponse,
  HFIngestSubmitResult,
} from "@/lib/types";
import { HubSearch } from "./HubSearch";
import { DatasetPreview } from "./DatasetPreview";
import { DownloadActionBar } from "./DownloadActionBar";
import { DownloadProgressCard } from "./DownloadProgressCard";
import { LocalDatasetsLibrary } from "./LocalDatasetsLibrary";
import styles from "./HuggingFacePage.module.css";

// ── Constants ────────────────────────────────────────────────────────────────

const ACTIVE_DOWNLOAD_KEY = "hf_active_download_dataset_id";
const TERMINAL_STATES = new Set(["completed", "failed", "cancelled"]);

// ── Ingest Panel State (shared across LocalDatasetsLibrary) ─────────────────

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

// ── Component ────────────────────────────────────────────────────────────────

export default function HuggingFacePage() {
  const router = useRouter();
  const { addToast } = useToastStore();

  // ── Input + preview state ───────────────────────────────────────────────

  const [datasetIdInput, setDatasetIdInput] = useState("");
  const firstMountRef = useRef(true);
  const previewTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const downloadPollRef = useRef<number | null>(null);
  const ingestPollRef = useRef<number | null>(null);

  const [preview, setPreview] = useState<HuggingFacePreviewResponse | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [selectedConfig, setSelectedConfig] = useState("");
  const [autoIngest, setAutoIngest] = useState(false);
  const [autoIngestTextCols, setAutoIngestTextCols] = useState<string[]>([]);
  const [rowLimitInput, setRowLimitInput] = useState("");

  // ── Download state ──────────────────────────────────────────────────────

  const [downloadingDatasetId, setDownloadingDatasetId] = useState<string | null>(null);
  const [downloadStatus, setDownloadStatus] = useState<HuggingFaceDownloadStatusResponse | null>(null);

  // ── Local datasets ───────────────────────────────────────────────────────

  const [discovering, setDiscovering] = useState(false);
  const [datasets, setDatasets] = useState<DiscoveredDataset[]>([]);
  const [ingestPanels, setIngestPanels] = useState<Record<string, IngestPanelState>>({});
  const [activeIngestDatasetId, setActiveIngestDatasetId] = useState<string | null>(null);

  // ── Polling helpers ──────────────────────────────────────────────────────

  const stopDownloadPolling = useCallback(() => {
    if (downloadPollRef.current != null) {
      window.clearInterval(downloadPollRef.current);
      downloadPollRef.current = null;
    }
  }, []);

  const stopIngestPolling = useCallback(() => {
    if (ingestPollRef.current != null) {
      window.clearInterval(ingestPollRef.current);
      ingestPollRef.current = null;
    }
  }, []);

  // ── handleDiscover ──────────────────────────────────────────────────────

  const handleDiscover = useCallback(async () => {
    setDiscovering(true);
    try {
      const result = await discoverHFDatasets();
      setDatasets(result.datasets);

      setIngestPanels((prev) => {
        const next: Record<string, IngestPanelState> = { ...prev };
        for (const ds of result.datasets) {
          if (!next[ds.dataset_id]) {
            next[ds.dataset_id] = {
              expanded: false,
              loading: false,
              ingestId: null,
              ingestStatus: null,
              result: null,
              selectedSplit: ds.splits.includes("train") ? "train" : (ds.splits[0] ?? "train"),
              selectedTextCols: ds.readable_columns.slice(0, 2),
              rowLimit: "",
            };
          }
        }
        return next;
      });
    } catch (err) {
      addToast(`Discovery failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    } finally {
      setDiscovering(false);
    }
  }, [addToast]);

  // ── Download polling ─────────────────────────────────────────────────────

  const startDownloadPolling = useCallback(
    (datasetId: string) => {
      stopDownloadPolling();
      setDownloadingDatasetId(datasetId);

      const poll = async () => {
        try {
          const st = await getHFDownloadStatus(datasetId);
          setDownloadStatus(st);

          if (TERMINAL_STATES.has(st.status)) {
            stopDownloadPolling();
            // Don't persist terminal status — it should not survive page refresh
            window.localStorage.removeItem(ACTIVE_DOWNLOAD_KEY);

            if (st.status === "completed") {
              addToast(
                st.auto_ingest && st.ingested
                  ? `✓ Downloaded & indexed! Ready to chat.`
                  : `✓ ${st.dataset_id} downloaded successfully.`,
                "success"
              );
              void handleDiscover();
              setDownloadingDatasetId(null);
            } else if (st.status === "cancelled") {
              addToast(`Download cancelled.`, "info");
              void handleDiscover();
              setDownloadingDatasetId(null);
            } else if (st.status === "failed") {
              addToast(`Download failed: ${st.error || st.message}`, "error");
              setDownloadingDatasetId(null);
            }
          }
        } catch {
          // Keep polling through transient errors
        }
      };

      poll();
      downloadPollRef.current = window.setInterval(poll, 1500);
    },
    [stopDownloadPolling, addToast, handleDiscover]
  );

  // ── Ingest polling ────────────────────────────────────────────────────────

  const startIngestPolling = useCallback(
    (datasetId: string, ingestId: string) => {
      stopIngestPolling();

      const poll = async () => {
        try {
          const st = await getHFIngestStatus(ingestId);

          setIngestPanels((prev) => ({
            ...prev,
            [datasetId]: {
              ...prev[datasetId],
              ingestStatus: st,
              loading: st.status === "running" || st.status === "queued",
            },
          }));

          if (TERMINAL_STATES.has(st.status)) {
            stopIngestPolling();

            if (st.status === "completed") {
              setIngestPanels((prev) => ({
                ...prev,
                [datasetId]: {
                  ...prev[datasetId],
                  ingestStatus: null,
                  loading: false,
                  result: st.result,
                },
              }));
              const rows = st.result?.rows_processed as number | undefined;
              const chunks = st.result?.chunks_embedded as number | undefined;
              addToast(
                rows != null
                  ? `✓ ${rows.toLocaleString()} rows → ${(chunks ?? 0).toLocaleString()} chunks indexed.`
                  : "✓ Ingestion completed.",
                "success"
              );
              setDatasets((prev) =>
                prev.map((d) => (d.dataset_id === datasetId ? { ...d, is_ingested: true } : d))
              );
              void handleDiscover();
            } else if (st.status === "cancelled") {
              addToast("Ingest cancelled.", "info");
            } else if (st.status === "failed") {
              addToast(`Ingest failed: ${st.error ?? st.message}`, "error");
            }
          }
        } catch {
          // Keep polling through transient errors
        }
      };

      poll();
      ingestPollRef.current = window.setInterval(poll, 2000);
    },
    [stopIngestPolling, addToast, handleDiscover]
  );

  // ── Lifecycle ────────────────────────────────────────────────────────────

  useEffect(() => {
    if (firstMountRef.current) {
      firstMountRef.current = false;
      void handleDiscover();
      const saved = window.localStorage.getItem(ACTIVE_DOWNLOAD_KEY);
      if (saved) {
        void startDownloadPolling(saved);
      }
    }
  }, [handleDiscover, startDownloadPolling]);

  useEffect(() => {
    return () => {
      if (previewTimerRef.current) clearTimeout(previewTimerRef.current);
      stopDownloadPolling();
      stopIngestPolling();
      // Reset search state so the input is always empty on next visit
      setDatasetIdInput("");
      setPreview(null);
      setPreviewError(null);
    };
  }, [stopDownloadPolling, stopIngestPolling]);

  // ── Debounced preview fetch ───────────────────────────────────────────────

  const fetchPreview = useCallback(
    async (id: string) => {
      if (!id.trim() || id.trim().length < 2) {
        setPreview(null);
        setPreviewError(null);
        return;
      }
      setPreviewLoading(true);
      setPreviewError(null);
      try {
        const data = await previewHFDataset(id.trim());
        setPreview(data);
        if (data.default_config && !selectedConfig) {
          setSelectedConfig(data.default_config);
        }
        if (Object.keys(data.columns).length > 0) {
          const firstStringCol =
            Object.entries(data.columns).find(([, t]) => t === "string")?.[0] ??
            Object.keys(data.columns)[0];
          if (firstStringCol) {
            setAutoIngestTextCols([firstStringCol]);
          }
        }
      } catch (err) {
        setPreviewError(err instanceof Error ? err.message : "Could not load preview");
        setPreview(null);
      } finally {
        setPreviewLoading(false);
      }
    },
    [selectedConfig]
  );

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleDatasetIdChange = useCallback(
    (val: string) => {
      setDatasetIdInput(val);
      setSelectedConfig("");
      setAutoIngestTextCols([]);
      setActiveIngestDatasetId(null);
      if (previewTimerRef.current) clearTimeout(previewTimerRef.current);
      if (val.trim().length >= 2) {
        previewTimerRef.current = setTimeout(() => void fetchPreview(val), 600);
      } else {
        setPreview(null);
        setPreviewError(null);
      }
    },
    [fetchPreview]
  );

  const handleChipClick = useCallback(
    (id: string) => {
      setDatasetIdInput(id);
      setSelectedConfig("");
      setAutoIngestTextCols([]);
      setActiveIngestDatasetId(null);
      if (previewTimerRef.current) clearTimeout(previewTimerRef.current);
      previewTimerRef.current = setTimeout(() => void fetchPreview(id), 600);
    },
    [fetchPreview]
  );

  const handleDownload = useCallback(async () => {
    const datasetId = datasetIdInput.trim();
    if (!datasetId) return;
    try {
      window.localStorage.setItem(ACTIVE_DOWNLOAD_KEY, datasetId);
      setDownloadStatus(null);
      await downloadHFDataset({
        dataset_id: datasetId,
        split: undefined,
        config: selectedConfig || undefined,
        auto_ingest: autoIngest,
        row_limit: rowLimitInput ? parseInt(rowLimitInput, 10) : undefined,
        text_columns: autoIngestTextCols.length > 0 ? autoIngestTextCols : undefined,
      });
      void startDownloadPolling(datasetId);
      addToast(`Download started for "${datasetId}"`, "info");
    } catch (err) {
      addToast(`Failed to start download: ${err instanceof Error ? err.message : String(err)}`, "error");
    }
  }, [datasetIdInput, selectedConfig, autoIngest, rowLimitInput, autoIngestTextCols, startDownloadPolling, addToast]);

  const handleCancelDownload = useCallback(async () => {
    const datasetId = downloadingDatasetId ?? datasetIdInput.trim();
    if (!datasetId) return;
    try {
      await cancelHFDownload(datasetId);
      addToast(`Cancel requested for "${datasetId}"`, "info");
    } catch (err) {
      addToast(`Cancel failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    }
  }, [downloadingDatasetId, datasetIdInput, addToast]);

  const updatePanel = useCallback(
    (id: string, updates: Partial<IngestPanelState>) =>
      setIngestPanels((prev) => ({ ...prev, [id]: { ...prev[id], ...updates } })),
    []
  );

  const handleIngest = useCallback(
    async (ds: DiscoveredDataset, force = false) => {
      const panel = ingestPanels[ds.dataset_id];
      if (!panel) return;

      setIngestPanels((prev) => ({
        ...prev,
        [ds.dataset_id]: { ...prev[ds.dataset_id], loading: true },
      }));

      try {
        const submit = await submitHFIngest(ds.dataset_id, {
          split: panel.selectedSplit,
          text_columns: panel.selectedTextCols.length > 0 ? panel.selectedTextCols : undefined,
          row_limit: panel.rowLimit ? parseInt(panel.rowLimit, 10) : undefined,
          force,
        });

        setIngestPanels((prev) => ({
          ...prev,
          [ds.dataset_id]: {
            ...prev[ds.dataset_id],
            ingestId: submit.ingest_id,
            loading: true,
          },
        }));

        void startIngestPolling(ds.dataset_id, submit.ingest_id);
      } catch (err) {
        setIngestPanels((prev) => ({
          ...prev,
          [ds.dataset_id]: { ...prev[ds.dataset_id], loading: false },
        }));
        const msg = err instanceof Error ? err.message : String(err);
        if (msg.includes("409") || msg.toLowerCase().includes("already ingested")) {
          addToast(
            `"${ds.dataset_id}" is already indexed in Qdrant. Use 'Re-ingest' to re-process.`,
            "warning"
          );
          void handleDiscover();
        } else {
          addToast(`Ingest submit failed: ${msg}`, "error");
        }
      }
    },
    [ingestPanels, addToast, handleDiscover, startIngestPolling]
  );

  // ── Derived state ────────────────────────────────────────────────────────

  const isDownloading =
    downloadStatus != null && !TERMINAL_STATES.has(downloadStatus.status);

  const isCompleted = downloadStatus?.status === "completed";
  const isFailed = downloadStatus?.status === "failed";
  const isCancelled = downloadStatus?.status === "cancelled";

  const ingestStatus = downloadStatus?.ingest_status;
  const ingestCompleted = ingestStatus === "completed";
  const ingestFailed = ingestStatus === "failed";
  const isIngesting = ingestStatus === "ingesting";

  const showSuccessState = isCompleted && !isDownloading;
  const autoIngestEnabled = downloadStatus?.auto_ingest === true;

  const textColumnOptions: Array<{ value: string; label: string; typeHint?: string }> =
    preview?.columns
      ? Object.entries(preview.columns).map(([name, type]) => ({
          value: name,
          label: name,
          typeHint: type === "string" ? undefined : type,
        }))
      : [];

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className={styles.page}>

      {/* ── Zone 1: Browse & Select ─────────────────────────────────────── */}
      <HubSearch
        datasetId={datasetIdInput}
        onDatasetIdChange={handleDatasetIdChange}
        onChipClick={handleChipClick}
      />

      {/* ── Zone 1: Preview card ───────────────────────────────────────── */}
      {preview && (
        <DatasetPreview
          preview={preview}
          previewLoading={previewLoading}
          previewError={previewError}
          selectedConfig={selectedConfig}
          onConfigChange={setSelectedConfig}
        />
      )}

      {/* ── Zone 2: Action bar ──────────────────────────────────────────── */}
      {preview && (
        <DownloadActionBar
          preview={preview}
          selectedConfig={selectedConfig}
          autoIngest={autoIngest}
          rowLimitInput={rowLimitInput}
          textColumnOptions={textColumnOptions}
          autoIngestTextCols={autoIngestTextCols}
          isDownloading={isDownloading}
          isCompleted={isCompleted ?? false}
          ingestCompleted={ingestCompleted}
          ingestFailed={ingestFailed}
          isIngesting={isIngesting}
          autoIngestEnabled={autoIngestEnabled}
          onAutoIngestChange={setAutoIngest}
          onAutoIngestTextColsChange={setAutoIngestTextCols}
          onRowLimitChange={setRowLimitInput}
          onDownload={handleDownload}
          onCancel={handleCancelDownload}
        />
      )}

      {/* ── Zone 2: Progress card ────────────────────────────────────────── */}
      {downloadStatus && !TERMINAL_STATES.has(downloadStatus.status) && (
        <div style={{ marginTop: preview ? 12 : 24 }}>
          <DownloadProgressCard
            downloadStatus={downloadStatus}
            preview={preview}
            autoIngestEnabled={autoIngestEnabled}
            ingestCompleted={ingestCompleted}
            ingestFailed={ingestFailed}
            isIngesting={isIngesting}
            datasets={datasets}
            setActiveIngestDatasetId={setActiveIngestDatasetId}
            onPanelUpdate={updatePanel}
            onCancel={handleCancelDownload}
          />
        </div>
      )}

      {/* ── Zone 3: Library ─────────────────────────────────────────────── */}
      <LocalDatasetsLibrary
        datasets={datasets}
        ingestPanels={ingestPanels}
        discovering={discovering}
        downloadStatus={downloadStatus}
        isDownloading={isDownloading}
        onDiscover={handleDiscover}
        onRestartPolling={startDownloadPolling}
        onPanelUpdate={updatePanel}
        onIngest={handleIngest}
        stopIngestPolling={stopIngestPolling}
        addToast={addToast}
        router={router}
      />

    </div>
  );
}
