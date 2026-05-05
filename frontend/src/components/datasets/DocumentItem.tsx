"use client";

import type { KbDocumentResponse } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { Accordion, AccordionItem } from "@/components/ui/Accordion";
import { formatLocalDate } from "@/lib/utils/date";
import styles from "./DocumentItem.module.css";

interface DocumentItemProps {
  document: KbDocumentResponse;
  datasetId: string;
}

function getStatus(progress: number): string {
  if (progress === 0) return "queued";
  if (progress < 100) return "processing";
  return "completed";
}

function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return "—";
  return formatLocalDate(dateStr);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function DocumentItem({
  document,
  datasetId: _datasetId,
}: DocumentItemProps) {
  const status = getStatus(document.progress);
  const chunks = String(
    document.meta_fields?.chunks ?? document.doc_metadata?.chunks ?? "—",
  );
  const statusVariant =
    status === "completed"
      ? "success"
      : status === "processing"
        ? "warning"
        : "muted";

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <span className={styles.name} title={document.name}>
          {document.name}
        </span>
        <Badge variant={statusVariant}>{status}</Badge>
      </div>

      <div className={styles.progressBar}>
        <div
          className={styles.progressFill}
          style={{ width: `${document.progress}%` }}
        />
      </div>

      <div className={styles.meta}>
        <span className={styles.metaLabel}>Chunks:</span> {chunks}
        <span className={styles.metaSeparator}>|</span>
        <span className={styles.metaLabel}>Created:</span>{" "}
        {formatDate(document.create_date ?? null)}
        {document.doc_type && (
          <>
            <span className={styles.metaSeparator}>|</span>
            <span className={styles.metaLabel}>Type:</span> {document.doc_type}
          </>
        )}
      </div>

      <Accordion>
        {document.doc_metadata &&
          Object.keys(document.doc_metadata).length > 0 && (
            <AccordionItem title="Document Metadata">
              <div className={styles.kvGrid}>
                {Object.entries(document.doc_metadata).map(([key, value]) => (
                  <div key={key} className={styles.kvRow}>
                    <span className={styles.kvKey}>{key}</span>
                    <span className={styles.kvValue}>{String(value)}</span>
                  </div>
                ))}
              </div>
            </AccordionItem>
          )}
        {document.meta_fields &&
          Object.keys(document.meta_fields).length > 0 && (
            <AccordionItem title="Meta Fields">
              <div className={styles.kvGrid}>
                {Object.entries(document.meta_fields).map(([key, value]) => (
                  <div key={key} className={styles.kvRow}>
                    <span className={styles.kvKey}>{key}</span>
                    <span className={styles.kvValue}>
                      {typeof value === "number"
                        ? value === Math.floor(value)
                          ? value
                          : formatBytes(value)
                        : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </AccordionItem>
          )}
      </Accordion>
    </div>
  );
}
