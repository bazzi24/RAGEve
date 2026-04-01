"use client";

import { useRef, useEffect } from "react";
import { chatSessionStreaming } from "@/lib/api/chat";
import { useChatStore } from "@/stores/useChatStore";

export function useChatStream() {
  const abortRef = useRef<AbortController | null>(null);

  // Clean up: abort any in-flight stream when the component using this hook unmounts.
  useEffect(() => {
    return () => {
      if (abortRef.current) {
        abortRef.current.abort();
        abortRef.current = null;
      }
    };
  }, []);

  /**
   * Send a question using the active session.
   * Uses session-aware streaming when a session is selected,
   * falls back to agent-based streaming when no session exists.
   */
  const send = async (
    question: string,
    options?: {
      temperature?: number;
      topK?: number;
      scoreThreshold?: number;
      useReranker?: boolean;
      rerankerModel?: string | null;
      useHybrid?: boolean;
    }
  ) => {
    if (abortRef.current) {
      abortRef.current.abort();
    }

    const controller = new AbortController();
    abortRef.current = controller;

    useChatStore.getState().setIsStreaming(true);
    useChatStore.getState().appendStreamingText("");

    try {
      const { currentSession, selectedAgentId } = useChatStore.getState();

      if (!currentSession) {
        // No session yet — can't stream
        useChatStore.getState().setIsStreaming(false);
        useChatStore.getState().appendStreamingText(
          "\n\n[Error: No session selected. Please select or create a session first.]"
        );
        return;
      }

      await chatSessionStreaming(
        currentSession.session_id,
        {
          question,
          temperature: options?.temperature,
          top_k: options?.topK,
          score_threshold: options?.scoreThreshold,
          use_reranker: options?.useReranker,
          reranker_model: options?.rerankerModel ?? undefined,
          use_hybrid: options?.useHybrid,
        },
        {
          onChunk: (content) => {
            useChatStore.getState().appendStreamingText(content);
          },
          onSources: (sources, rerankerModel, messageId) => {
            useChatStore.getState().finalizeStreaming(sources, rerankerModel, messageId);
            useChatStore.getState().setIsStreaming(false);
          },
          onError: (error) => {
            useChatStore.getState().setIsStreaming(false);
            useChatStore.getState().appendStreamingText(`\n\n[Error: ${error}]`);
          },
        },
        controller.signal
      );
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        useChatStore.getState().setIsStreaming(false);
        return;
      }
      useChatStore.getState().setIsStreaming(false);
      useChatStore.getState().appendStreamingText(
        `\n\n[Error: ${err instanceof Error ? err.message : String(err)}]`
      );
    }
  };

  const stop = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
      useChatStore.getState().setIsStreaming(false);
    }
  };

  return { send, stop };
}

