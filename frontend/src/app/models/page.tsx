"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useModelStore } from "@/stores/useModelStore";
import { useToastStore } from "@/stores/useToastStore";
import { validateModels } from "@/lib/api/ollama";
import { Button } from "@/components/ui/Button";
import { Select } from "@/components/ui/Select";
import type { OllamaModelDetail } from "@/lib/types";
import styles from "./page.module.css";

function fmtBytes(bytes: number): string {
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

interface ModelCardProps {
  model: OllamaModelDetail;
  isEmbeddingDefault: boolean;
  isChatDefault: boolean;
  onSetEmbedding: () => void;
  onSetChat: () => void;
}

function ModelCard({ model, isEmbeddingDefault, isChatDefault, onSetEmbedding, onSetChat }: ModelCardProps) {
  const isEmbedding = !!(
    model.name.includes("embed") ||
    model.name.includes("nomic") ||
    model.name.includes("bge") ||
    model.name.includes("e5") ||
    model.name.includes("gte")
  );

  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <div className={styles.cardName} title={model.name}>
          {model.name}
        </div>
        <div className={styles.cardBadges}>
          {isEmbeddingDefault && (
            <span className={`${styles.badge} ${styles.badgeEmbed}`}>Embedding default</span>
          )}
          {isChatDefault && (
            <span className={`${styles.badge} ${styles.badgeChat}`}>Chat default</span>
          )}
          {isEmbedding && !isEmbeddingDefault && (
            <span className={`${styles.badge} ${styles.badgePossible}`}>Embedding</span>
          )}
        </div>
      </div>

      <div className={styles.cardMeta}>
        {model.details.parameter_size && (
          <div className={styles.metaRow}>
            <span className={styles.metaKey}>Params</span>
            <span className={styles.metaVal}>{model.details.parameter_size}</span>
          </div>
        )}
        {model.details.quantization_level && (
          <div className={styles.metaRow}>
            <span className={styles.metaKey}>Quant</span>
            <span className={styles.metaVal}>{model.details.quantization_level}</span>
          </div>
        )}
        {model.details.family && (
          <div className={styles.metaRow}>
            <span className={styles.metaKey}>Family</span>
            <span className={styles.metaVal}>{model.details.family}</span>
          </div>
        )}
        <div className={styles.metaRow}>
          <span className={styles.metaKey}>Size</span>
          <span className={styles.metaVal}>{fmtBytes(model.size)}</span>
        </div>
        <div className={styles.metaRow}>
          <span className={styles.metaKey}>Modified</span>
          <span className={styles.metaVal}>{fmtDate(model.modified_at)}</span>
        </div>
      </div>

      <div className={styles.cardActions}>
        {!isEmbeddingDefault && (
          <Button
            variant="secondary"
            size="sm"
            onClick={onSetEmbedding}
            title="Set as global embedding model"
          >
            Use as Embedding
          </Button>
        )}
        {!isChatDefault && (
          <Button
            variant="secondary"
            size="sm"
            onClick={onSetChat}
            title="Set as global chat model"
          >
            Use as Chat
          </Button>
        )}
        {(isEmbeddingDefault || isChatDefault) && (
          <span className={styles.inUseLabel}>In use</span>
        )}
      </div>
    </div>
  );
}

export default function ModelsPage() {
  const router = useRouter();
  const { addToast } = useToastStore();
  const {
    embeddingModel,
    chatModel,
    modelDetails,
    modelsLoaded,
    setupComplete,
    ollamaConnected,
    setEmbeddingModel,
    setChatModel,
    refreshOllamaModels,
    setAvailableModels,
  } = useModelStore();

  const [refreshing, setRefreshing] = useState(false);
  const [validating, setValidating] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    setRefreshing(true);
    setFetchError(null);
    try {
      await refreshOllamaModels();
    } catch (err) {
      setFetchError(err instanceof Error ? err.message : "Failed to reach Ollama");
    } finally {
      setRefreshing(false);
    }
  }, [refreshOllamaModels]);

  // Initial load
  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Auto-select embedding candidates when defaults are unset
  useEffect(() => {
    const { embeddingModel: emb, chatModel: chat, modelDetails: details, setEmbeddingModel: setEmb, setChatModel: setChat } = useModelStore.getState();
    if (!emb && details.length > 0) {
      const embed = details.find((m) =>
        m.name.includes("embed") || m.name.includes("nomic") || m.name.includes("bge")
      );
      if (embed) setEmb(embed.name);
    }
    if (!chat && details.length > 0) {
      const chatCandidate = details.find((m) => !m.name.includes("embed"));
      if (chatCandidate) setChat(chatCandidate.name);
    }
  }, [modelDetails]);

  const handleValidate = async () => {
    if (!embeddingModel || !chatModel) return;
    setValidating(true);
    try {
      const result = await validateModels({
        embedding_model: embeddingModel,
        chat_model: chatModel,
      });
      if (result.valid) {
        addToast(`Defaults saved — Embedding: ${embeddingModel}, Chat: ${chatModel}`, "success");
      } else {
        addToast(`Validation failed. Missing: ${result.missing_models.join(", ")}`, "error");
      }
    } catch (err) {
      addToast(`Validation error: ${err instanceof Error ? err.message : String(err)}`, "error");
    } finally {
      setValidating(false);
    }
  };

  const handleSetEmbedding = (name: string) => {
    setEmbeddingModel(name);
    addToast(`Embedding default set to: ${name}`, "success");
  };

  const handleSetChat = (name: string) => {
    setChatModel(name);
    addToast(`Chat default set to: ${name}`, "success");
  };

  // Derive embed vs chat candidates for the dropdowns
  const embedCandidates = modelDetails.filter(
    (m) =>
      m.name.includes("embed") ||
      m.name.includes("nomic") ||
      m.name.includes("bge") ||
      m.name.includes("e5") ||
      m.name.includes("gte")
  );
  const chatCandidates = modelDetails.filter((m) => !embedCandidates.some((e) => e.name === m.name));

  const embedOptions = embedCandidates.length > 0 ? embedCandidates : modelDetails;
  const chatOptions = chatCandidates.length > 0 ? chatCandidates : modelDetails;

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div className={styles.titleRow}>
          <h1 className={styles.title}>Models</h1>
          <div className={styles.headerActions}>
            <Button
              variant="secondary"
              size="sm"
              onClick={fetchModels}
              disabled={refreshing}
              title="Refresh model list from Ollama"
            >
              <svg
                width="13"
                height="13"
                viewBox="0 0 13 13"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                style={{ animation: refreshing ? "spin 0.8s linear infinite" : "none" }}
              >
                <path d="M11 6.5A4.5 4.5 0 1 1 6.5 2" />
                <path d="M11 2v3.5H7.5" />
              </svg>
              {refreshing ? "Refreshing…" : "Refresh"}
            </Button>
          </div>
        </div>

        {/* Connection status */}
        <div className={styles.statusBar}>
          <div className={`${styles.statusDot} ${modelsLoaded ? styles.statusOk : styles.statusError}`} />
          <span className={styles.statusText}>
            {refreshing
              ? "Connecting to Ollama at localhost:11434…"
              : modelsLoaded
              ? `Connected — ${modelDetails.length} model(s) available`
              : "Ollama unreachable — pull models with `ollama pull <name>`"}
          </span>
        </div>
      </div>

      {fetchError && (
        <div className={styles.errorBanner}>{fetchError}</div>
      )}

      {/* Defaults section */}
      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>Global Defaults</h2>
        <p className={styles.sectionDesc}>
          These models are used as defaults for all agents. Override per-agent in the Agents page.
        </p>

        <div className={styles.defaultsGrid}>
          <div className={styles.defaultCard}>
            <label className={styles.defaultLabel} htmlFor="default-embed">
              Embedding Model
            </label>
            <Select
              id="default-embed"
              options={embedOptions.map((m) => ({ value: m.name, label: m.name }))}
              value={embeddingModel}
              onChange={(e) => setEmbeddingModel(e.target.value)}
              placeholder="Select embedding model…"
            />
            {embeddingModel && (
              <p className={styles.defaultHint}>
                {embedCandidates.find((m) => m.name === embeddingModel)
                  ? "Detected as embedding model"
                  : "No embedding model detected — select one above"}
              </p>
            )}
          </div>

          <div className={styles.defaultCard}>
            <label className={styles.defaultLabel} htmlFor="default-chat">
              Chat / LLM Model
            </label>
            <Select
              id="default-chat"
              options={chatOptions.map((m) => ({ value: m.name, label: m.name }))}
              value={chatModel}
              onChange={(e) => setChatModel(e.target.value)}
              placeholder="Select chat model…"
            />
          </div>
        </div>

        <div className={styles.defaultsActions}>
          <Button
            onClick={handleValidate}
            disabled={!embeddingModel || !chatModel || validating}
            loading={validating}
          >
            Validate &amp; Save
          </Button>
          {setupComplete && embeddingModel && chatModel && (
            <span className={styles.savedLabel}>
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M2 7l3.5 3.5L12 3" />
              </svg>
              Defaults saved
            </span>
          )}
        </div>
      </section>

      {/* All models grid */}
      {modelDetails.length > 0 && (
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>
            Pulled Models
            <span className={styles.modelCount}>{modelDetails.length}</span>
          </h2>
          <div className={styles.grid}>
            {modelDetails.map((model) => (
              <ModelCard
                key={model.name}
                model={model}
                isEmbeddingDefault={model.name === embeddingModel}
                isChatDefault={model.name === chatModel}
                onSetEmbedding={() => handleSetEmbedding(model.name)}
                onSetChat={() => handleSetChat(model.name)}
              />
            ))}
          </div>
        </section>
      )}

      {modelDetails.length === 0 && !refreshing && (
        <div className={styles.emptyState}>
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5" className={styles.emptyIcon}>
            <rect x="6" y="8" width="36" height="28" rx="3" />
            <path d="M14 20h20M14 28h12" />
            <path d="M30 8V4M18 8V4" />
          </svg>
          <p className={styles.emptyTitle}>No models pulled yet</p>
          <p className={styles.emptyDesc}>
            Run the following commands to pull models:
          </p>
          <pre className={styles.emptyCode}>{`ollama pull nomic-embed-text
ollama pull llama3.2`}</pre>
          <Button variant="secondary" onClick={fetchModels} disabled={refreshing}>
            Retry Connection
          </Button>
        </div>
      )}
    </div>
  );
}
