"use client";

import { useState, useEffect } from "react";
import type {
  DatasetInfo,
  KbDocumentResponse,
  KnowledgebaseResponse,
} from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { DocumentList } from "./DocumentList";
import { IngestInterface } from "./IngestInterface";
import { getKnowledgebase } from "@/lib/api/knowledgebases";
import { formatLocalDate } from "@/lib/utils/date";
import styles from "./DatasetCard.module.css";

interface DatasetCardProps {
  dataset: DatasetInfo;
  expanded: boolean;
  documents: KbDocumentResponse[];
  documentsLoading: boolean;
  onClick: () => void;
  onDelete: () => void;
}

export function DatasetCard({
  dataset,
  expanded,
  documents,
  documentsLoading,
  onClick,
  onDelete,
}: DatasetCardProps) {
  const [kbDetail, setKbDetail] = useState<KnowledgebaseResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    if (!expanded) {
      setKbDetail(null);
      return;
    }
    let cancelled = false;
    const fetchDetail = async () => {
      setDetailLoading(true);
      try {
        const kb = await getKnowledgebase(dataset.dataset_id);
        if (cancelled) return;
        setKbDetail(kb);
      } catch {
        // document fetching is handled by toggleDetail in the store
      } finally {
        if (!cancelled) setDetailLoading(false);
      }
    };
    fetchDetail();
    return () => {
      cancelled = true;
    };
  }, [expanded, dataset.dataset_id]);

  return (
    <div className={`${styles.card} ${expanded ? styles.cardExpanded : ""}`}>
      <div className={styles.cardHeader} onClick={onClick}>
        <div className={styles.titleRow}>
          <span className={styles.cardTitle} title={dataset.name}>
            {dataset.name}
          </span>
          <Badge variant={dataset.status === "unknown" ? "muted" : "default"}>
            {dataset.status}
          </Badge>
        </div>
        <svg
          className={`${styles.chevron} ${expanded ? styles.chevronExpanded : ""}`}
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        >
          <path d="M3 6l5 5 5-5" />
        </svg>
      </div>

      <div className={styles.cardStats}>
        <div className={styles.stat}>
          <span className={styles.statValue}>{dataset.chunks_count}</span>
          <span className={styles.statLabel}>Chunks</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statValue}>{dataset.vector_size}</span>
          <span className={styles.statLabel}>Vector Size</span>
        </div>
      </div>

      <div className={styles.cardFooter}>
        <Button
          variant="danger"
          size="sm"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
        >
          Delete
        </Button>
      </div>

      {expanded && (
        <div className={styles.detail}>
          {detailLoading && (
            <div className={styles.detailMessage}>
              Loading dataset details...
            </div>
          )}
          {kbDetail && (
            <div className={styles.detailSection}>
              {kbDetail.description && (
                <p className={styles.description}>{kbDetail.description}</p>
              )}
              <div className={styles.detailMeta}>
                {kbDetail.language && (
                  <span>Language: {kbDetail.language}</span>
                )}
                {kbDetail.create_date && (
                  <span>
                    Created:{" "}
                    {formatLocalDate(kbDetail.create_date)}
                  </span>
                )}
              </div>
            </div>
          )}

          <div className={styles.detailSection}>
            <h4 className={styles.sectionTitle}>Documents</h4>
            <DocumentList
              documents={documents}
              loading={documentsLoading || detailLoading}
              datasetId={dataset.dataset_id}
            />
          </div>

          <IngestInterface datasetId={dataset.dataset_id} />
        </div>
      )}
    </div>
  );
}
