"use client";

import { useCallback, useEffect, useRef } from "react";
import { useChatStore } from "@/stores/useChatStore";
import { useAgentsStore } from "@/stores/useAgentsStore";
import { useModelStore } from "@/stores/useModelStore";
import { listAgents } from "@/lib/api/agents";
import { getRerankers } from "@/lib/api/rerank";
import { useChatStream } from "@/components/chat/useChatStream";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { ChatInput } from "@/components/chat/ChatInput";
import { SourcesPanel } from "@/components/chat/SourcesPanel";
import { AgentSelector } from "@/components/agents/AgentSelector";
import { Button } from "@/components/ui/Button";
import { useToastStore } from "@/stores/useToastStore";
import { SessionPanel } from "@/components/chat/SessionPanel";
import { getSessionWithMessages } from "@/lib/api/chat";
import styles from "./ChatPage.module.css";

let msgCounter = 0;

export default function ChatPage() {
  const {
    messages,
    streamingText,
    sources,
    isStreaming,
    selectedAgentId,
    currentSession,
    setSelectedAgentId,
    appendMessage,
    loadSessionMessages,
    clearMessages,
    activeRerankerModel,
    setRerankerModels,
  } = useChatStore();
  const { agents, setAgents } = useAgentsStore();
  const { addToast } = useToastStore();
  const { send, stop } = useChatStream();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const prevMsgLen = useRef(0);

  // Load agents on mount
  useEffect(() => {
    listAgents()
      .then((res) => setAgents(res.agents))
      .catch((err) => addToast(`Failed to load agents: ${err.message}`, "error"));
  }, [setAgents, addToast]);

  // Refresh Ollama and reranker model lists on mount.
  useEffect(() => {
    getRerankers()
      .then((res) => setRerankerModels(res.rerankers))
      .catch(() => {
        // Non-fatal — reranking just won't be available
      });
  }, [setRerankerModels]);

  // Pull the Ollama model list fresh on mount.
  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(`${base}/ollama/models`)
      .then((r) => r.json())
      .then((data) => {
        useModelStore.setState({
          availableModels: data.models ?? [],
          modelsLoaded: data.has_models ?? false,
        });
      })
      .catch(() => {
        // Non-fatal
      });
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    const totalLen = messages.length + (streamingText ? 1 : 0);
    if (totalLen !== prevMsgLen.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      prevMsgLen.current = totalLen;
    }
  }, [messages, streamingText]);

  // Load messages when a session is selected
  const handleSessionSelected = useCallback(
    (sessionId: string) => {
      getSessionWithMessages(sessionId)
        .then((res) => {
          loadSessionMessages(res.messages);
        })
        .catch((err) => addToast(`Failed to load messages: ${err.message}`, "error"));
    },
    [loadSessionMessages, addToast]
  );

  const handleSend = useCallback(
    (
      question: string,
      opts: {
        temperature: number;
        topK: number;
        useReranker: boolean;
        rerankerModel: string | null;
        useHybrid: boolean;
      }
    ) => {
      if (!currentSession) {
        addToast("Please create or select a conversation first", "warning");
        return;
      }
      // Add user message immediately to the local store
      const userMsgId = `msg-${++msgCounter}`;
      appendMessage({
        id: userMsgId,
        role: "user",
        content: question,
        timestamp: Date.now(),
      });
      send(question, {
        temperature: opts.temperature,
        topK: opts.topK,
        useReranker: opts.useReranker,
        rerankerModel: opts.useReranker ? opts.rerankerModel : null,
        useHybrid: opts.useHybrid,
      });
    },
    [currentSession, appendMessage, send, addToast]
  );

  // When the selected agent changes, reset messages
  const handleAgentChange = useCallback(
    (agentId: string) => {
      setSelectedAgentId(agentId);
    },
    [setSelectedAgentId]
  );

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h1 className={styles.title}>Chat</h1>
          <AgentSelector
            agents={agents}
            selectedId={selectedAgentId}
            onChange={handleAgentChange}
            disabled={isStreaming}
          />
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            clearMessages();
            addToast("Chat cleared", "info");
          }}
          disabled={isStreaming}
        >
          Clear
        </Button>
      </div>

      <div className={styles.layout}>
        {/* Session sidebar — only shown when an agent is selected */}
        {selectedAgentId && (
          <SessionPanel
            agentId={selectedAgentId}
            onSessionSelected={handleSessionSelected}
          />
        )}

        <div className={styles.chatArea}>
          {!selectedAgentId ? (
            <div className={styles.emptyState}>
              <svg className={styles.emptyIcon} viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M6 8h36v28H30l-6 6V36H6V8z" />
                <path d="M14 18h20M14 24h12" />
              </svg>
              <div className={styles.emptyTitle}>Select an agent</div>
              <div className={styles.emptyDesc}>
                Choose an agent from the dropdown above to start a conversation.
              </div>
            </div>
          ) : !currentSession ? (
            <div className={styles.emptyState}>
              <svg className={styles.emptyIcon} viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M6 8h36v28H30l-6 6V36H6V8z" />
                <path d="M14 18h20M14 24h12" />
              </svg>
              <div className={styles.emptyTitle}>Start a conversation</div>
              <div className={styles.emptyDesc}>
                Click &ldquo;+ New&rdquo; in the sidebar to start a new conversation with this agent.
              </div>
            </div>
          ) : messages.length === 0 && !streamingText ? (
            <div className={styles.emptyState}>
              <svg className={styles.emptyIcon} viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M6 8h36v28H30l-6 6V36H6V8z" />
                <path d="M14 18h20M14 24h12" />
              </svg>
              <div className={styles.emptyTitle}>Ask a question</div>
              <div className={styles.emptyDesc}>
                The model will retrieve relevant context from your datasets and answer.
              </div>
            </div>
          ) : (
            <div className={styles.messages}>
              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}
              {streamingText && (
                <ChatMessage
                  message={{
                    id: "streaming",
                    role: "assistant",
                    content: streamingText,
                    timestamp: Date.now(),
                  }}
                  isStreaming
                />
              )}
              <div ref={messagesEndRef} />
            </div>
          )}

          <ChatInput
            onSend={handleSend}
            onStop={stop}
            disabled={!currentSession}
          />
        </div>

        <SourcesPanel sources={streamingText ? [] : sources} rerankerModel={activeRerankerModel} />
      </div>
    </div>
  );
}

