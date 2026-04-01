"use client";

import { useState } from "react";
import { Button } from "@/components/ui/Button";
import { MultiSelect } from "@/components/ui/MultiSelect";
import { Badge } from "@/components/ui/Badge";
import type { HuggingFacePreviewResponse } from "@/lib/types";
import styles from "./HuggingFacePage.module.css";

// ── Types ───────────────────────────────────────────────────────────────────

interface DownloadActionBarProps {
  preview: HuggingFacePreviewResponse;
  selectedConfig: string;
  autoIngest: boolean;
  rowLimitInput: string;
  textColumnOptions: Array<{ value: string; label: string; typeHint?: string }>;
  autoIngestTextCols: string[];
  isDownloading: boolean;
  isCompleted: boolean;
  ingestCompleted: boolean;
  ingestFailed?: boolean;
  isIngesting: boolean;
  autoIngestEnabled: boolean;
  onAutoIngestChange: (v: boolean) => void;
  onAutoIngestTextColsChange: (cols: string[]) => void;
  onRowLimitChange: (v: string) => void;
  onDownload: () => Promise<void>;
  onCancel: () => Promise<void>;
}

// ── Component ───────────────────────────────────────────────────────────────

export function DownloadActionBar({
  preview,
  selectedConfig,
  autoIngest,
  rowLimitInput,
  textColumnOptions,
  autoIngestTextCols,
  isDownloading,
  isIngesting,
  onAutoIngestChange,
  onAutoIngestTextColsChange,
  onRowLimitChange,
  onDownload,
  onCancel,
}: DownloadActionBarProps) {
  const [ingestSlideOpen, setIngestSlideOpen] = useState(false);

  const handleAutoIngestToggle = (checked: boolean) => {
    onAutoIngestChange(checked);
    if (checked) setIngestSlideOpen(true);
  };

  const isDisabled =
    !preview ||
    (preview.configs.length > 0 && !selectedConfig) ||
    isDownloading;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
      {/* ── Action bar ──────────────────────────────────────────────────── */}
      <div className={styles.actionBar}>
        {/* Dataset info */}
        <div className={styles.actionBarInfo}>
          <span style={{ fontSize: 20 }}>⬡</span>
          <span className={styles.actionBarDatasetName}>{preview.full_dataset_id}</span>
          {preview.estimated_size_human && (
            <Badge variant="default">{preview.estimated_size_human}</Badge>
          )}
          {preview.splits.length > 0 && (
            <Badge variant="default">
              {preview.splits.length} split{preview.splits.length !== 1 ? "s" : ""}
            </Badge>
          )}
        </div>

        {/* Actions */}
        <div className={styles.actionBarActions}>
          {/* Auto-ingest toggle */}
          <label
            className={styles.ingestOptsLabel}
            style={{ cursor: isDownloading ? "not-allowed" : "pointer", opacity: isDownloading ? 0.5 : 1 }}
          >
            <input
              type="checkbox"
              checked={autoIngest}
              onChange={(e) => handleAutoIngestToggle(e.target.checked)}
              disabled={isDownloading}
              style={{ accentColor: "var(--accent)" }}
            />
            Auto-ingest
          </label>

          <Button
            onClick={() => void onDownload()}
            disabled={isDisabled}
            loading={isDownloading && !isIngesting}
          >
            {isDownloading ? "Downloading…" : "↓ Download"}
          </Button>

          {isDownloading && (
            <Button variant="danger" onClick={() => void onCancel()}>
              Cancel
            </Button>
          )}
        </div>
      </div>

      {/* ── Ingest slide-in panel ───────────────────────────────────────── */}
      {(autoIngest || ingestSlideOpen) && !isDownloading && (
        <div className={`${styles.actionBarSlideIn} ${styles.actionBarSlideInOpen}`}>
          <div className={styles.ingestOpts}>
            <div className={styles.ingestOptsRow}>
              {/* Text columns */}
              {textColumnOptions.length > 0 && (
                <div className={styles.ingestOptsField}>
                  <MultiSelect
                    id="text-col-select"
                    label="Columns to Embed"
                    hint="Selected columns combined for semantic search. Other columns stored as metadata."
                    options={textColumnOptions}
                    selected={autoIngestTextCols}
                    onChange={onAutoIngestTextColsChange}
                  />
                </div>
              )}

              {/* Row limit */}
              <div className={styles.ingestOptsField}>
                <label
                  style={{
                    display: "block",
                    fontSize: 10,
                    fontWeight: 600,
                    textTransform: "uppercase",
                    letterSpacing: "0.06em",
                    color: "var(--text-muted)",
                    marginBottom: 4,
                  }}
                >
                  Row Limit
                </label>
                <input
                  className={styles.rowLimitInput}
                  type="text"
                  placeholder="Full dataset (no limit)"
                  value={rowLimitInput}
                  onChange={(e) => onRowLimitChange(e.target.value)}
                />
                <p className={styles.ingestOptsHint}>
                  e.g. 500 for a quick test run
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
