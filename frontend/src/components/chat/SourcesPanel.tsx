"use client";

import type { SourceChunk } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import styles from "./SourcesPanel.module.css";

interface SourcesPanelProps {
  sources: SourceChunk[];
  rerankerModel?: string | null;
}

export function SourcesPanel({ sources, rerankerModel }: SourcesPanelProps) {
  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <span className={styles.title}>Sources</span>
        <div className={styles.headerRight}>
          {rerankerModel && (
            <span className={styles.rerankerBadge} title={`Reranked with ${rerankerModel}`}>
              &#x1F504; {rerankerModel.split("/").pop()}
            </span>
          )}
          <span className={styles.count}>{sources.length} chunk{sources.length !== 1 ? "s" : ""}</span>
        </div>
      </div>

      {sources.length === 0 ? (
        <div className={styles.empty}>
          No sources yet.<br />Send a message to see retrieved chunks.
        </div>
      ) : (
        <div className={styles.list}>
          {sources.map((s, i) => (
            <div key={String(s.chunk_id ?? `source-${i}`)} className={styles.sourceCard}>
              <div className={styles.sourceMeta}>
                <div className={styles.sourceHeaderLeft}>
                  <span className={styles.sourceName} title={s.source || "unknown"}>
                    {s.source || "unknown"}
                  </span>
                  {s.search_type === "hybrid" && (
                    <Badge variant="info" title="Retrieved by hybrid search (dense + sparse)">
                      HYBRID
                    </Badge>
                  )}
                </div>
                <div className={styles.scoreRow}>
                  {rerankerModel && s.cosine_score != null ? (
                    <>
                      <Badge variant="muted" title="Bi-encoder cosine similarity">
                        &#x2191; {(s.cosine_score * 100).toFixed(1)}%
                      </Badge>
                      <Badge variant={s.score > 0.85 ? "success" : s.score > 0.7 ? "warning" : "default"} title="Cross-encoder relevance score">
                        &#x1F504; {(s.score * 100).toFixed(1)}%
                      </Badge>
                    </>
                  ) : (
                    <Badge variant={s.score > 0.85 ? "success" : s.score > 0.7 ? "warning" : "default"}>
                      {(s.score * 100).toFixed(1)}%
                    </Badge>
                  )}
                </div>
              </div>
              <div className={styles.sourceText}>{s.text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
