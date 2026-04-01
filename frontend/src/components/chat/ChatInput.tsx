"use client";

import { useRef, useState } from "react";
import { useChatStore } from "@/stores/useChatStore";
import { useModelStore } from "@/stores/useModelStore";
import { Button } from "@/components/ui/Button";
import { Slider } from "@/components/ui/Slider";
import { Select } from "@/components/ui/Select";
import { useToastStore } from "@/stores/useToastStore";
import styles from "./ChatInput.module.css";

interface ChatInputProps {
  onSend: (
    question: string,
    opts: {
      temperature: number;
      topK: number;
      useReranker: boolean;
      rerankerModel: string | null;
      useHybrid: boolean;
    }
  ) => void;
  onStop: () => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, onStop, disabled }: ChatInputProps) {
  const [text, setText] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const { addToast } = useToastStore();

  const {
    isStreaming,
    temperature,
    topK,
    useReranker,
    rerankerModel,
    rerankerModels,
    useHybrid,
    setTemperature,
    setTopK,
    setUseReranker,
    setRerankerModel,
    setUseHybrid,
    refreshRerankers,
  } = useChatStore();

  const {
    availableModels,
    modelsLoaded,
    refreshOllamaModels,
  } = useModelStore();

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    const trimmed = text.trim();
    if (!trimmed || isStreaming) return;
    onSend(trimmed, { temperature, topK, useReranker, rerankerModel, useHybrid });
    setText("");
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const autoResize = () => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await refreshOllamaModels();
      addToast("Ollama models refreshed", "success");
    } catch (err) {
      addToast(
        `Refresh failed: ${err instanceof Error ? err.message : String(err)}`,
        "error"
      );
    } finally {
      setRefreshing(false);
    }
  };

  const rerankerOptions = rerankerModels.map((m) => ({
    value: m.id,
    label: m.display_name,
  }));

  return (
    <div className={styles.wrapper}>
      {settingsOpen && (
        <div className={styles.settingsPanel}>
          {/* ── Ollama models — quick refresh ───────────────────────────────── */}
          <div className={styles.refreshRow}>
            <span className={styles.refreshLabel}>
              {modelsLoaded
                ? `${availableModels.length} model(s) available`
                : "No Ollama models"}
            </span>
            <button
              className={`${styles.refreshBtn} ${refreshing ? styles.refreshBtnSpinning : ""}`}
              onClick={handleRefresh}
              disabled={refreshing}
              title="Refresh Ollama model list"
              type="button"
              aria-label="Refresh Ollama models"
            >
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M11 6.5A4.5 4.5 0 1 1 6.5 2" />
                <path d="M11 2v3.5H7.5" />
              </svg>
              {refreshing ? "…" : "Refresh"}
            </button>
          </div>

          {/* ── Divider ─────────────────────────────────────────────────────── */}
          <div className={styles.settingsDivider} />

          {/* Temperature */}
          <div className={styles.settingRow}>
            <span className={styles.settingLabel}>Temperature</span>
            <div className={styles.settingControl}>
              <Slider
                min={0}
                max={2}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
              <span className={styles.settingValue}>{temperature.toFixed(1)}</span>
            </div>
          </div>

          {/* Top-K */}
          <div className={styles.settingRow}>
            <span className={styles.settingLabel}>Top-K</span>
            <div className={styles.settingControl}>
              <Slider
                min={1}
                max={20}
                step={1}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
              <span className={styles.settingValue}>{topK}</span>
            </div>
          </div>

          {/* Divider */}
          <div className={styles.settingsDivider} />

          {/* Hybrid search toggle */}
          <div className={styles.settingRow}>
            <div className={styles.settingLabelGroup}>
              <span className={styles.settingLabel}>Hybrid Search</span>
              <span className={styles.settingHint}>
                Combines dense + sparse (Splade++) for keyword-aware retrieval
              </span>
            </div>
            <label
              className={styles.toggle}
              title={useHybrid ? "Disable hybrid search" : "Enable hybrid search"}
            >
              <input
                type="checkbox"
                checked={useHybrid}
                onChange={(e) => setUseHybrid(e.target.checked)}
              />
              <span className={styles.toggleTrack} />
            </label>
          </div>

          {/* Reranking toggle */}
          <div className={styles.settingRow}>
            <div className={styles.settingLabelGroup}>
              <span className={styles.settingLabel}>Reranking</span>
              <span className={styles.settingHint}>
                Score chunks with a cross-encoder for better relevance
              </span>
            </div>
            <label
              className={styles.toggle}
              title={useReranker ? "Disable reranking" : "Enable reranking"}
            >
              <input
                type="checkbox"
                checked={useReranker}
                onChange={(e) => {
                  setUseReranker(e.target.checked);
                  if (e.target.checked && rerankerModels.length > 0 && !rerankerModel) {
                    setRerankerModel(rerankerModels[0].id);
                  }
                }}
              />
              <span className={styles.toggleTrack} />
            </label>
          </div>

          {/* Reranker model selector — only when enabled */}
          {useReranker && (
            <div className={`${styles.settingRow} ${styles.settingRowIndented}`}>
              <span className={styles.settingLabel}>Model</span>
              <div className={styles.settingControl}>
                <Select
                  options={rerankerOptions}
                  value={rerankerModel ?? ""}
                  onChange={(e) => setRerankerModel(e.target.value || null)}
                  placeholder="Select a model…"
                />
              </div>
            </div>
          )}
        </div>
      )}

      <div className={styles.inputRow}>
        <div className={styles.inputWrapper}>
          <textarea
            ref={textareaRef}
            className={styles.textarea}
            value={text}
            onChange={(e) => {
              setText(e.target.value);
              autoResize();
            }}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question… (Enter to send, Shift+Enter for new line)"
            disabled={disabled || isStreaming}
            rows={1}
          />
        </div>
        <div className={styles.actions}>
          {/* Settings gear */}
          <button
            className={`${styles.gearBtn} ${settingsOpen ? styles.gearBtnActive : ""}`}
            onClick={() => setSettingsOpen((v) => !v)}
            title={settingsOpen ? "Close settings" : "Open search settings"}
            type="button"
            aria-label={settingsOpen ? "Close settings" : "Open search settings"}
          >
            <svg width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth="1.4">
              <circle cx="7.5" cy="7.5" r="2.2" />
              <path d="M7.5 1.5v1.6M7.5 11.9v1.6M13.4 7.5h-1.6M3.2 7.5H1.6M11.7 3.3l-1.1 1.1M4.4 10.6L3.3 11.7M11.7 11.7l-1.1-1.1M4.4 4.4L3.3 3.3" />
            </svg>
          </button>

          {isStreaming ? (
            <Button variant="danger" onClick={onStop} title="Stop generation">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                <rect x="2" y="2" width="10" height="10" rx="2" />
              </svg>
            </Button>
          ) : (
            <Button
              onClick={handleSubmit}
              disabled={!text.trim() || disabled}
              title="Send message"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M13 1L1 6l5 2 2 5 5-12z" />
              </svg>
              Send
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
