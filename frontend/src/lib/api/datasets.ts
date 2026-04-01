import { apiFetch } from "./client";
import type {
  DatasetInfo,
  DatasetListResponse,
  HuggingFaceRegisterResponse,
  IngestRequest,
  IngestResponse,
  ProcessedFileResponse,
  CollectionDeleteResponse,
  DiscoveredDataset,
  HFIngestSubmitResponse,
  HFIngestStatusResponse,
  HuggingFaceDiscoveryResponse,
  HuggingFaceInstructionsResponse,
  HuggingFaceRegisterRequest,
  HuggingFaceDownloadRequest,
  HuggingFaceDownloadResponse,
  HuggingFaceDownloadStatusResponse,
  HuggingFacePreviewResponse,
  HuggingFaceStatusResponse,
  HuggingFaceStatusTextsResponse,
  UploadProgressEvent,
  UploadProgressHandlers,
} from "@/lib/types";

// ── Core dataset operations ────────────────────────────────────────────────────────

export async function listDatasets(): Promise<DatasetListResponse> {
  return apiFetch<DatasetListResponse>("/datasets/");
}

export async function uploadFiles(
  datasetId: string,
  files: File[],
  ingestOptions?: IngestRequest
): Promise<{ dataset_id: string; files: ProcessedFileResponse[] }> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  if (ingestOptions) {
    form.append(
      "ingest_request",
      new Blob([JSON.stringify(ingestOptions)], { type: "application/json" })
    );
  }

  const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const response = await fetch(`${base}/datasets/${datasetId}/upload`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `Upload failed: ${response.status}`);
  }

  return response.json();
}

export async function uploadFilesStreaming(
  datasetId: string,
  files: File[],
  handlers: UploadProgressHandlers,
  ingestOptions?: IngestRequest,
  signal?: AbortSignal
): Promise<void> {
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  if (ingestOptions) {
    form.append(
      "ingest_request",
      new Blob([JSON.stringify(ingestOptions)], { type: "application/json" })
    );
  }

  const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const response = await fetch(`${base}/datasets/${datasetId}/upload/stream`, {
    method: "POST",
    body: form,
    signal,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `Upload failed: ${response.status}`);
  }

  if (!response.body) {
    throw new Error("Upload stream body is null");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.trim()) continue;
      let event: UploadProgressEvent;
      try {
        event = JSON.parse(line) as UploadProgressEvent;
      } catch {
        continue;
      }

      if (event.event === "status") {
        handlers.onStatus(event);
      } else if (event.event === "file_done") {
        handlers.onFileDone?.(event);
      } else if (event.event === "done") {
        handlers.onDone(event);
      } else if (event.event === "error") {
        handlers.onError(event);
      }
    }
  }

  if (buffer.trim()) {
    try {
      const event = JSON.parse(buffer) as UploadProgressEvent;
      if (event.event === "status") handlers.onStatus(event);
      else if (event.event === "file_done") handlers.onFileDone?.(event);
      else if (event.event === "done") handlers.onDone(event);
      else if (event.event === "error") handlers.onError(event);
    } catch {
      // ignore trailing malformed JSON
    }
  }
}

export async function ingestDataset(
  datasetId: string,
  options?: IngestRequest
): Promise<IngestResponse> {
  return apiFetch<IngestResponse>(`/datasets/${datasetId}/ingest`, {
    method: "POST",
    body: JSON.stringify(options || {}),
  });
}

export async function getDatasetInfo(datasetId: string): Promise<DatasetInfo> {
  return apiFetch<DatasetInfo>(`/datasets/${datasetId}`);
}

export async function deleteDataset(datasetId: string): Promise<CollectionDeleteResponse> {
  return apiFetch<CollectionDeleteResponse>(`/datasets/${datasetId}`, {
    method: "DELETE",
  });
}

export async function getHFInstructions(
  datasetId: string
): Promise<HuggingFaceInstructionsResponse> {
  return apiFetch<HuggingFaceInstructionsResponse>(
    `/datasets/hf/instructions/${datasetId}`
  );
}

export async function previewHFDataset(
  datasetId: string
): Promise<HuggingFacePreviewResponse> {
  return apiFetch<HuggingFacePreviewResponse>(
    `/datasets/hf/preview/${encodeURIComponent(datasetId)}`
  );
}

export async function getHFStatusTexts(
  datasetId: string
): Promise<HuggingFaceStatusTextsResponse> {
  return apiFetch<HuggingFaceStatusTextsResponse>(
    `/datasets/hf/status-texts/${encodeURIComponent(datasetId)}`
  );
}

export async function discoverHFDatasets(): Promise<HuggingFaceDiscoveryResponse> {
  return apiFetch<HuggingFaceDiscoveryResponse>("/datasets/hf/discover");
}

/** Fetch ingestion status (is_ingested) for all local datasets in one call. */
export async function getHFStatus(): Promise<HuggingFaceStatusResponse> {
  return apiFetch<HuggingFaceStatusResponse>("/datasets/hf/status");
}

/** Search HuggingFace Hub for datasets matching a query string. */
export interface HFDatasetSearchResult {
  id: string;
  downloads: number | null;
  likes: number | null;
  tags: string[];
  description: string;
}

export async function searchHFDatasets(
  query: string
): Promise<HFDatasetSearchResult[]> {
  const params = new URLSearchParams({ q: query });
  return apiFetch<HFDatasetSearchResult[]>(`/datasets/hf/search?${params}`);
}

export async function downloadHFDataset(
  payload: HuggingFaceDownloadRequest
): Promise<HuggingFaceDownloadResponse> {
  return apiFetch<HuggingFaceDownloadResponse>("/datasets/hf/download", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ── HF ingest (background) ──────────────────────────────────────────────────────

/** Submit a HuggingFace dataset ingest as a background task. Returns immediately. */
export async function submitHFIngest(
  datasetId: string,
  payload?: HFIngestRequest
): Promise<HFIngestSubmitResponse> {
  return apiFetch<HFIngestSubmitResponse>(
    `/datasets/hf/${encodeURIComponent(datasetId)}/ingest/submit`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    }
  );
}

/** Poll the status of a background HF ingest task. */
export async function getHFIngestStatus(
  ingestId: string
): Promise<HFIngestStatusResponse> {
  return apiFetch<HFIngestStatusResponse>(
    `/datasets/hf/ingest/${ingestId}/status`
  );
}

/** Cancel a running or queued HF ingest. */
export async function cancelHFIngest(
  ingestId: string
): Promise<{ ingest_id: string; status: string; message: string }> {
  return apiFetch(`/datasets/hf/ingest/${ingestId}/cancel`, {
    method: "POST",
  });
}

/**
 * @deprecated Use `submitHFIngest` + polling `getHFIngestStatus` instead.
 * This fires a synchronous request that blocks the server for the full ingest duration.
 */
export async function ingestHFDataset(
  datasetId: string,
  payload?: HFIngestRequest
): Promise<HFIngestResponse> {
  return apiFetch<HFIngestResponse>(`/datasets/hf/${encodeURIComponent(datasetId)}/ingest`, {
    method: "POST",
    body: JSON.stringify(payload || {}),
  });
}

export async function getHFDownloadStatus(
  datasetId: string
): Promise<HuggingFaceDownloadStatusResponse> {
  return apiFetch<HuggingFaceDownloadStatusResponse>(`/datasets/hf/download/${datasetId}/status`);
}

export async function cancelHFDownload(
  datasetId: string
): Promise<HuggingFaceDownloadResponse> {
  return apiFetch<HuggingFaceDownloadResponse>(`/datasets/hf/download/${datasetId}/cancel`, {
    method: "POST",
  });
}

export async function registerHFDataset(
  payload: HuggingFaceRegisterRequest
): Promise<HuggingFaceRegisterResponse> {
  return apiFetch<HuggingFaceRegisterResponse>("/datasets/hf/register", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export interface HFIngestRequest {
  split?: string;
  text_columns?: string[];
  metadata_columns?: string[];
  row_limit?: number;
  batch_size?: number;
  chunk_overlap?: number;
  max_tokens_per_chunk?: number;
  /** Force re-ingestion even if the dataset is already in Qdrant. */
  force?: boolean;
}

export interface HFIngestResponse {
  dataset_id: string;
  collection: string;
  rows_processed: number;
  chunks_embedded: number;
  avg_quality_score: number;
  profiles_used: Record<string, number>;
  text_columns_used: string[];
  message: string;
}

