"use client";

import { create } from "zustand";
import type {
  ChatMessageItem,
  ChatSession,
  ChatMessageStored,
  SourceChunk,
  RerankerEntry,
} from "@/lib/types";

interface ChatState {
  // ── Session state ──────────────────────────────────────────────────────────
  sessions: ChatSession[];
  currentSession: ChatSession | null;

  // ── Messages (Zustand-managed, synced from session) ─────────────────────────
  messages: ChatMessageItem[];
  streamingText: string;
  sources: SourceChunk[];
  isStreaming: boolean;

  // ── Agent + RAG settings ───────────────────────────────────────────────────
  selectedAgentId: string | null;
  temperature: number;
  topK: number;
  useReranker: boolean;
  rerankerModel: string | null;
  rerankerModels: RerankerEntry[];
  activeRerankerModel: string | null;
  useHybrid: boolean;

  // ── Actions ────────────────────────────────────────────────────────────────
  setSessions: (sessions: ChatSession[]) => void;
  addSession: (session: ChatSession) => void;
  removeSession: (sessionId: string) => void;
  setCurrentSession: (session: ChatSession | null) => void;
  loadSessionMessages: (messages: ChatMessageStored[]) => void;
  appendMessage: (message: ChatMessageItem) => void;
  appendStreamingText: (text: string) => void;
  finalizeStreaming: (
    sources: SourceChunk[],
    rerankerModel?: string | null,
    storedMessageId?: string
  ) => void;
  clearStreaming: () => void;
  setSources: (sources: SourceChunk[]) => void;
  setIsStreaming: (streaming: boolean) => void;
  setSelectedAgentId: (id: string | null) => void;
  setTemperature: (t: number) => void;
  setTopK: (k: number) => void;
  setUseReranker: (v: boolean) => void;
  setRerankerModel: (modelId: string | null) => void;
  setRerankerModels: (models: RerankerEntry[]) => void;
  setUseHybrid: (v: boolean) => void;
  refreshRerankers: () => Promise<void>;
  clearMessages: () => void;
}

let messageCounter = 0;

function _toItem(m: ChatMessageStored, fallbackId: string): ChatMessageItem {
  return {
    id: m.message_id,
    role: m.role,
    content: m.content,
    sources: m.sources ?? undefined,
    timestamp: m.created_at ? new Date(m.created_at).getTime() : Date.now(),
    storedMessageId: m.message_id,
  };
}

export const useChatStore = create<ChatState>()((set, get) => ({
  sessions: [],
  currentSession: null,
  messages: [],
  streamingText: "",
  sources: [],
  isStreaming: false,
  selectedAgentId: null,
  temperature: 0.7,
  topK: 5,
  useReranker: false,
  rerankerModel: null,
  rerankerModels: [],
  activeRerankerModel: null,
  useHybrid: false,

  setSessions: (sessions) => set({ sessions }),

  addSession: (session) =>
    set((state) => ({
      sessions: [session, ...state.sessions.filter((s) => s.session_id !== session.session_id)],
      currentSession: session,
    })),

  removeSession: (sessionId) =>
    set((state) => ({
      sessions: state.sessions.filter((s) => s.session_id !== sessionId),
      currentSession:
        state.currentSession?.session_id === sessionId ? null : state.currentSession,
    })),

  setCurrentSession: (currentSession) =>
    set({ currentSession, messages: [], streamingText: "", sources: [] }),

  loadSessionMessages: (messages) =>
    set({
      messages: messages.map((m) => _toItem(m, m.message_id)),
    }),

  appendMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),

  appendStreamingText: (text) =>
    set((state) => ({ streamingText: state.streamingText + text })),

  finalizeStreaming: (sources, rerankerModel, storedMessageId) => {
    const { streamingText, messages } = get();
    const finalMessage: ChatMessageItem = {
      id: storedMessageId ?? `msg-${++messageCounter}`,
      role: "assistant",
      content: streamingText,
      sources,
      timestamp: Date.now(),
    };
    set({
      messages: [...messages, finalMessage],
      streamingText: "",
      sources,
      activeRerankerModel: rerankerModel ?? null,
    });
  },

  clearStreaming: () => set({ streamingText: "" }),

  setSources: (sources) => set({ sources }),

  setIsStreaming: (isStreaming) => set({ isStreaming }),

  setSelectedAgentId: (selectedAgentId) =>
    set({
      selectedAgentId,
      currentSession: null,
      messages: [],
      streamingText: "",
      sources: [],
    }),

  setTemperature: (temperature) => set({ temperature }),

  setTopK: (topK) => set({ topK }),

  setUseReranker: (useReranker) => set({ useReranker }),

  setRerankerModel: (rerankerModel) => set({ rerankerModel }),

  setRerankerModels: (rerankerModels) => set({ rerankerModels }),

  setUseHybrid: (useHybrid) => set({ useHybrid }),

  refreshRerankers: async () => {
    try {
      const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${base}/rerankers/`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      set({ rerankerModels: data.rerankers ?? [] });
    } catch {
      // Non-fatal — reranker list just stays as-is
    }
  },

  clearMessages: () =>
    set({
      messages: [],
      streamingText: "",
      sources: [],
      isStreaming: false,
      activeRerankerModel: null,
    }),
}));

