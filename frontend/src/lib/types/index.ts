// All TypeScript interfaces mirroring backend schemas

// ── Ollama ──────────────────────────────────────────────────────────────────

export interface OllamaModelDetail {
  name: string;
  model: string;
  size: number; // bytes
  modified_at: string | null;
  digest: string | null;
  details: {
    family?: string;
    parameter_size?: string;
    quantization_level?: string;
    [key: string]: unknown;
  };
}

export interface OllamaModelListResponse {
  models: string[];
  has_models: boolean;
  message?: string;
  model_details: OllamaModelDetail[];
}

export interface ModelSelectionRequest {
  embedding_model: string;
  chat_model: string;
}

export interface ModelValidationResponse {
  valid: boolean;
  available_models: string[];
  missing_models: string[];
  message: string;
}

// ── Files / Datasets ─────────────────────────────────────────────────────────

export interface DocumentAnalysis {
  types: Record<string, number>;
  alpha_ratio: number;
  characters: number;
}

export interface QualitySignals {
  alpha_ratio: number;
  ocr_noise_ratio: number;
  broken_line_ratio: number;
  header_footer_ratio: number;
  table_density: number;
  avg_sentence_length: number;
  language_script_changes: number;
  repeated_word_ratio: number;
  code_delimiter_ratio: number;
  issue_tags: string[];
}

export interface QualityReport {
  quality_score: number;
  selected_profile: ChunkProfile;
  profile_reason: string;
  signals: QualitySignals;
}

export type ChunkProfile =
  | "clean_text"
  | "ocr_noisy"
  | "table_heavy"
  | "code_mixed"
  | "general";

export interface LayoutSummary {
  pages: number;
  total_blocks: number;
  blocks_by_type: Record<string, number>;
}

export interface ProcessedFileResponse {
  dataset_id: string;
  filename: string;
  extension: string;
  chars: number;
  chunks: number;
  collection: string;
  document_analysis: DocumentAnalysis;
  sample_chunk_analysis: DocumentAnalysis[];
  quality_report: QualityReport;
  layout_summary: LayoutSummary | null;
  extraction: {
    extractor: string;
    layout_aware?: boolean;
    converter?: string;
    converted?: boolean;
    message?: string;
    error?: string;
  };
}

export interface UploadSummaryResponse {
  dataset_id: string;
  files: ProcessedFileResponse[];
}

export interface UploadProgressStatusEvent {
  event: "status";
  stage: string;
  message: string;
  progress: number;
  dataset_id: string;
  file?: string;
  file_index?: number;
  file_total?: number;
  chunks_done?: number;
  chunks_total?: number;
}

export interface UploadProgressFileDoneEvent {
  event: "file_done";
  stage: "completed";
  message: string;
  progress: number;
  dataset_id: string;
  file: string;
  file_index: number;
  file_total: number;
  result: ProcessedFileResponse;
}

export interface UploadProgressDoneEvent {
  event: "done";
  stage: "completed";
  message: string;
  progress: 100;
  dataset_id: string;
  files: ProcessedFileResponse[];
}

export interface UploadProgressErrorEvent {
  event: "error";
  stage: "failed";
  message: string;
  progress: number;
  dataset_id: string;
  file?: string;
  file_index?: number;
  file_total?: number;
}

export type UploadProgressEvent =
  | UploadProgressStatusEvent
  | UploadProgressFileDoneEvent
  | UploadProgressDoneEvent
  | UploadProgressErrorEvent;

export type UploadProgressHandlers = {
  onStatus: (event: UploadProgressStatusEvent) => void;
  onFileDone?: (event: UploadProgressFileDoneEvent) => void;
  onDone: (event: UploadProgressDoneEvent) => void;
  onError: (event: UploadProgressErrorEvent) => void;
};

export interface UploadProgressState {
  active: boolean;
  stage: string;
  message: string;
  progress: number;
  file?: string;
  file_index?: number;
  file_total?: number;
  chunks_done?: number;
  chunks_total?: number;
}

export const initialUploadProgressState: UploadProgressState = {
  active: false,
  stage: "idle",
  message: "",
  progress: 0,
};


export interface IngestRequest {
  chunk_size?: number;
  chunk_overlap?: number;
  max_tokens_per_chunk?: number;
  force_profile?: ChunkProfile;
  overwrite?: boolean;
}

export interface IngestResponse {
  dataset_id: string;
  collection: string;
  chunks_embedded: number;
  total_chars: number;
  quality_report: {
    average_score?: number;
    [key: string]: unknown;
  };
  message: string;
}

export interface DatasetInfo {
  dataset_id: string;
  collection: string;
  chunks_count: number;
  vector_size: number;
  status: string;
}

export interface DatasetListResponse {
  datasets: DatasetInfo[];
  total: number;
}

export interface CollectionDeleteResponse {
  dataset_id: string;
  deleted: boolean;
  message: string;
}

// ── HuggingFace ─────────────────────────────────────────────────────────────

export interface HuggingFaceInstructionsResponse {
  dataset_id: string;
  download_command: string;
  expected_local_path: string;
  supported_splits: string[];
  supported_file_formats: string[];
  column_preview: Record<string, string>;
  estimated_size?: string;
  message: string;
}

export interface DiscoveredDataset {
  local_path: string;
  dataset_id: string;
  splits: string[];
  file_formats: string[];
  file_count: number;
  total_size_bytes: number;
  readable_columns: string[];
  description?: string;
  /** True when this dataset has been ingested into Qdrant. */
  is_ingested: boolean;
}

export interface HuggingFaceStatusResponse {
  datasets: DatasetIngestStatus[];
  total: number;
  message: string;
}

export interface DatasetIngestStatus {
  dataset_id: string;
  local_path: string;
  splits: string[];
  file_count: number;
  total_size_bytes: number;
  readable_columns: string[];
  description?: string;
  is_ingested: boolean;
  points_count: number;
}

export interface HuggingFaceDiscoveryResponse {
  scan_root: string;
  datasets: DiscoveredDataset[];
  total_found: number;
  message: string;
}

export interface HuggingFaceRegisterRequest {
  local_path: string;
  dataset_id: string;
  split?: string;
  text_column?: string;
  metadata_columns?: string[];
}

export interface HuggingFaceRegisterResponse {
  dataset_id: string;
  registered: boolean;
  collection: string;
  estimated_rows?: number;
  splits_available: string[];
  columns_available: string[];
  message: string;
}

export interface HuggingFaceDownloadRequest {
  dataset_id: string;
  split?: string;
  config?: string | null;
  auto_ingest?: boolean;
  row_limit?: number | null;
  batch_size?: number | null;
  chunk_overlap?: number | null;
  max_tokens_per_chunk?: number | null;
  text_column?: string | null;
  text_columns?: string[];
  metadata_columns?: string[];
  ingest_split?: string | null;
}

export interface HuggingFaceDownloadResponse {
  dataset_id: string;
  status: "queued" | "downloading" | "cancelling" | "cancelled" | "completed" | "failed";
  message: string;
}

export interface HuggingFaceDownloadStatusResponse {
  dataset_id: string;
  status: "queued" | "downloading" | "cancelling" | "cancelled" | "completed" | "failed";
  progress: number;
  message: string;
  error?: string | null;
  local_path?: string | null;
  started_at?: string | null;
  updated_at?: string | null;
  rows_downloaded?: number | null;
  splits_downloaded?: string[];
  bytes_downloaded?: number | null;
  total_bytes?: number | null;
  config?: string | null;
  auto_ingest?: boolean;
  ingest_status?: "idle" | "ingesting" | "completed" | "failed" | null;
  ingest_message?: string | null;
  ingest_error?: string | null;
  ingested?: boolean;
  suggested_text_column?: string | null;
  columns?: Record<string, string>;
}

export interface HuggingFacePreviewResponse {
  dataset_id: string;
  full_dataset_id: string;
  configs: string[];
  default_config: string | null;
  description: string | null;
  downloads: number | null;
  likes: number | null;
  estimated_size_bytes: number | null;
  estimated_size_human: string | null;
  splits: string[];
  columns: Record<string, string>;
  source: string;
  // Rich metadata from huggingface_hub card
  tags?: string[];
  language?: string[];
  license?: string | null;
  paper_url?: string | null;
  card_data?: Record<string, unknown> | null;
  readme_html?: string | null;
  leaderboard?: Record<string, unknown> | null;
  source_detail?: string;
  message: string;
  valid: boolean;
}

export interface HuggingFaceStatusTextsResponse {
  dataset_id: string;
  display_status: string;
  display_message: string;
}

// ── HF Ingest (background) ──────────────────────────────────────────────────

export interface HFIngestSubmitResult {
  rows_processed?: number;
  chunks_embedded?: number;
  total_chunks?: number;
  avg_quality_score?: number;
  profiles_used?: Record<string, number>;
  text_columns_used?: string[];
  message?: string;
}

export interface HFIngestSubmitResponse {
  ingest_id: string;
  dataset_id: string;
  message: string;
  result: HFIngestSubmitResult | null;
}

export interface HFIngestStatusResponse {
  ingest_id: string;
  dataset_id: string;
  qdrant_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;       // 0–100
  current_stage: string;
  message: string;
  rows_done: number;
  rows_total: number;
  chunks_done: number;
  chunks_total: number;
  batches_done: number;
  batches_total: number;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  result: Record<string, unknown> | null;
}

// ── Rerankers ────────────────────────────────────────────────────────────────

export interface RerankerEntry {
  id: string;
  display_name: string;
  description: string;
  approx_size_mb: number;
}

export interface RerankersResponse {
  rerankers: RerankerEntry[];
  total: number;
}

// ── Agents ───────────────────────────────────────────────────────────────────

export interface AgentConfig {
  system_prompt: string;
  dataset_id: string;
  embedding_model: string;
  chat_model: string;
  temperature: number;
  top_k: number;
}

export interface AgentResponse {
  agent_id: string;
  name: string;
  description: string;
  config: AgentConfig;
  created_at: string;
  updated_at: string;
}

export interface AgentCreate {
  name: string;
  description?: string;
  config: AgentConfig;
}

export interface AgentUpdate {
  name?: string;
  description?: string;
  config?: AgentConfig;
}

export interface AgentListResponse {
  agents: AgentResponse[];
  total: number;
}

// ── Chat ─────────────────────────────────────────────────────────────────────

export interface SourceChunk {
  chunk_id: string;
  text: string;
  score: number;
  source?: string;
  /** Original bi-encoder cosine similarity score (preserved through reranking). */
  cosine_score?: number;
  /** Sparse retrieval score (0.0 when dense-only). */
  sparse_score?: number;
  /** Search mode: "dense" | "hybrid". */
  search_type?: string;
}

export interface ChatMessageItem {
  id: string;
  /** The server-assigned message_id from the DB, if this message was saved. */
  storedMessageId?: string;
  role: "user" | "assistant";
  content: string;
  sources?: SourceChunk[];
  timestamp: number;
}

export interface ChatRequest {
  question: string;
  temperature?: number;
  top_k?: number;
  score_threshold?: number;
  stream?: boolean;
  use_reranker?: boolean;
  reranker_model?: string;
  /** Enable hybrid search (dense + sparse with RRF fusion). */
  use_hybrid?: boolean;
}

export interface ChatResponse {
  answer: string;
  sources: SourceChunk[];
  metadata: Record<string, unknown>;
}

// ── SSE Events ───────────────────────────────────────────────────────────────

export type SSEChunkEvent = { event: "chunk"; content: string };
export type SSEEndEvent = {
  event: "end";
  sources: SourceChunk[];
  reranker_model?: string | null;
  use_hybrid?: boolean;
  message_id?: string;
  elapsed_s?: number;
};
export type SSEErrorEvent = { event: "error"; error: string; message_id?: string };
export type SSEEvent = SSEChunkEvent | SSEEndEvent | SSEErrorEvent;

// ── Chat Sessions / History ──────────────────────────────────────────────────

export interface AgentConfigSnapshot {
  system_prompt: string;
  dataset_id: string;
  embedding_model: string;
  chat_model: string;
  temperature: number;
  top_k: number;
}

export interface ChatSession {
  session_id: string;
  agent_id: string;
  title: string;
  message_count: number;
  agent_config_snapshot: AgentConfigSnapshot;
  created_at: string;
  updated_at: string;
}

export interface ChatFeedback {
  feedback_id: string;
  message_id: string;
  rating: "thumbs_up" | "thumbs_down";
  comment: string | null;
  created_at: string;
}

export interface ChatMessageStored {
  message_id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  token_count: number | null;
  sources: SourceChunk[] | null;
  feedback: ChatFeedback | null;
  created_at: string;
}

export interface ChatSessionWithMessages {
  session: ChatSession;
  messages: ChatMessageStored[];
}

export interface ChatSessionListResponse {
  sessions: ChatSession[];
  total: number;
  limit: number;
  offset: number;
}

export interface CreateSessionRequest {
  agent_id: string;
  title?: string;
}

export interface CreateSessionResponse extends ChatSession {}

export interface FeedbackPayload {
  rating: "thumbs_up" | "thumbs_down";
  comment?: string;
}
