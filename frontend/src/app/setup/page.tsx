"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useModelStore } from "@/stores/useModelStore";
import { listModels, validateModels } from "@/lib/api/ollama";
import { Button } from "@/components/ui/Button";
import { Select } from "@/components/ui/Select";
import type { OllamaModelListResponse } from "@/lib/types";
import styles from "./page.module.css";

export default function SetupPage() {
  const router = useRouter();
  const { setEmbeddingModel, setChatModel, setAvailableModels, embeddingModel, chatModel } =
    useModelStore();

  const [loading, setLoading] = useState(false);
  const [modelList, setModelList] = useState<OllamaModelListResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [validating, setValidating] = useState(false);

  const fetchModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await listModels();
      setModelList(result);
      setAvailableModels(result);
    } catch (err) {
      setError(
        `Cannot reach Ollama at http://localhost:11434. Is Ollama running? Error: ${
          err instanceof Error ? err.message : String(err)
        }`
      );
    } finally {
      setLoading(false);
    }
  }, [setAvailableModels]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleValidate = async () => {
    if (!embeddingModel || !chatModel) return;
    setValidating(true);
    setError(null);
    try {
      const result = await validateModels({
        embedding_model: embeddingModel,
        chat_model: chatModel,
      });
      if (result.valid) {
        useModelStore.setState({ setupComplete: true });
        router.push("/datasets");
      } else {
        setError(
          `Missing models: ${result.missing_models.join(", ")}. Run:\nollama pull ${result.missing_models.join("\nollama pull ")}`
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setValidating(false);
    }
  };

  const allModels = modelList?.models || [];
  const embedCandidates = allModels.filter(
    (m) =>
      m.includes("embed") ||
      m.includes("nomic") ||
      m.includes("bge") ||
      m.includes("e5") ||
      m.includes("gte")
  );
  const chatCandidates = allModels.filter(
    (m) => !embedCandidates.includes(m)
  );

  const embedOptions = embedCandidates.length > 0 ? embedCandidates : allModels;
  const chatOptions = chatCandidates.length > 0 ? chatCandidates : allModels;

  const needsModels = !modelList || !modelList.has_models;

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        <div className={styles.logo}>
          <img src="/logo.jpg" alt="RAGEve logo" className={styles.logoIcon} />
          <div className={styles.logoTitle}>RAGEve</div>
          <div className={styles.logoSubtitle}>Setup — Select your Ollama models</div>
        </div>

        {error && <div className={styles.errorBox}>{error}</div>}

        {needsModels && !loading ? (
          <div className={styles.infoBox}>
            <div className={styles.infoTitle}>No local Ollama models found</div>
            <p style={{ color: "var(--text-secondary)", fontSize: "var(--text-sm)", marginBottom: "var(--space-2)" }}>
              Download models from the Ollama library before continuing:
            </p>
            <code className={styles.infoCode}>
              {`# Embedding model\nollama pull nomic-embed-text\n\n# Chat model\nollama pull llama3.2\n\n# Then restart the backend`}
            </code>
            <div style={{ marginTop: "var(--space-4)" }}>
              <Button variant="secondary" onClick={fetchModels}>
                Refresh
              </Button>
            </div>
          </div>
        ) : loading ? (
          <p style={{ color: "var(--text-secondary)", textAlign: "center" }}>
            Scanning Ollama models...
          </p>
        ) : (
          <div className={styles.form}>
            <Select
              id="embed-model"
              label="Embedding Model"
              value={embeddingModel}
              onChange={(e) => setEmbeddingModel(e.target.value)}
              options={embedOptions.map((m) => ({ value: m, label: m }))}
              placeholder="Select embedding model..."
            />
            <Select
              id="chat-model"
              label="Chat Model"
              value={chatModel}
              onChange={(e) => setChatModel(e.target.value)}
              options={chatOptions.map((m) => ({ value: m, label: m }))}
              placeholder="Select chat model..."
            />

            <Button
              onClick={handleValidate}
              disabled={!embeddingModel || !chatModel}
              loading={validating}
              fullWidth
              size="lg"
            >
              Validate &amp; Save
            </Button>

            <p style={{ fontSize: "var(--text-xs)", color: "var(--text-muted)", textAlign: "center" }}>
              {allModels.length} model(s) found: {allModels.join(", ")}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
