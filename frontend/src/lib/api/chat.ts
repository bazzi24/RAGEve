import type {
  ChatRequest,
  ChatResponse,
  ChatSession,
  ChatSessionWithMessages,
  ChatSessionListResponse,
  CreateSessionRequest,
  FeedbackPayload,
  SSEEvent,
  SourceChunk,
} from "@/lib/types";

const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ──────────────────────────────────────────────────────────────────────────────
// Sessions
// ──────────────────────────────────────────────────────────────────────────────

export async function createSession(payload: CreateSessionRequest): Promise<ChatSession> {
  const res = await fetch(`${BASE}/chat/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Create session failed: ${res.status}`);
  }
  return res.json();
}

export async function listSessions(params?: {
  agent_id?: string;
  limit?: number;
  offset?: number;
}): Promise<ChatSessionListResponse> {
  const qs = new URLSearchParams();
  if (params?.agent_id) qs.set("agent_id", params.agent_id);
  if (params?.limit != null) qs.set("limit", String(params.limit));
  if (params?.offset != null) qs.set("offset", String(params.offset));
  const res = await fetch(`${BASE}/chat/sessions?${qs}`);
  if (!res.ok) throw new Error(`List sessions failed: ${res.status}`);
  return res.json();
}

export async function getSessionWithMessages(sessionId: string): Promise<ChatSessionWithMessages> {
  const res = await fetch(`${BASE}/chat/sessions/${sessionId}`);
  if (!res.ok) throw new Error(`Get session failed: ${res.status}`);
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/chat/sessions/${sessionId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete session failed: ${res.status}`);
}

// ──────────────────────────────────────────────────────────────────────────────
// Streaming with session history
// ──────────────────────────────────────────────────────────────────────────────

export type SessionStreamHandler = {
  onChunk: (content: string) => void;
  onSources: (sources: SourceChunk[], rerankerModel?: string | null, messageId?: string) => void;
  onError: (error: string) => void;
};

export async function chatSessionStreaming(
  sessionId: string,
  payload: ChatRequest,
  handlers: SessionStreamHandler,
  signal: AbortSignal
): Promise<void> {
  const res = await fetch(`${BASE}/chat/sessions/${sessionId}/messages/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Chat stream failed: ${res.status}`);
  }

  if (!res.body) throw new Error("Response body is null");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split("\n");

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event: SSEEvent & { message_id?: string } = JSON.parse(line);
        if (event.event === "chunk") {
          handlers.onChunk(event.content);
        } else if (event.event === "end") {
          handlers.onSources(event.sources || [], event.reranker_model ?? null, event.message_id);
        } else if (event.event === "error") {
          handlers.onError(event.error);
        }
      } catch {
        // Skip malformed lines
      }
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Feedback
// ──────────────────────────────────────────────────────────────────────────────

export async function submitFeedback(
  messageId: string,
  payload: FeedbackPayload
): Promise<void> {
  const res = await fetch(`${BASE}/chat/messages/${messageId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Submit feedback failed: ${res.status}`);
}

export async function chatNonStreaming(
  agentId: string,
  payload: ChatRequest
): Promise<ChatResponse> {
  const response = await fetch(`${BASE}/chat/${agentId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, stream: false }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `Chat failed: ${response.status}`);
  }

  return response.json();
}

export type StreamHandler = {
  onChunk: (content: string) => void;
  onSources: (sources: SourceChunk[], rerankerModel?: string | null) => void;
  onError: (error: string) => void;
};

export async function chatStreaming(
  agentId: string,
  payload: ChatRequest,
  handlers: StreamHandler,
  signal: AbortSignal
): Promise<void> {
  const response = await fetch(`${BASE}/chat/${agentId}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, stream: true }),
    signal,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(err.detail || `Chat failed: ${response.status}`);
  }

  if (!response.body) throw new Error("Response body is null");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split("\n");

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event: SSEEvent = JSON.parse(line);
        if (event.event === "chunk") {
          handlers.onChunk(event.content);
        } else if (event.event === "end") {
          handlers.onSources(event.sources || [], event.reranker_model ?? null);
        } else if (event.event === "error") {
          handlers.onError(event.error);
        }
      } catch {
        // Skip malformed lines
      }
    }
  }
}
