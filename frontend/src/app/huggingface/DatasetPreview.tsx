"use client";

import { useState } from "react";
import { Select } from "@/components/ui/Select";
import { Badge } from "@/components/ui/Badge";
import type { HuggingFacePreviewResponse } from "@/lib/types";
import styles from "./HuggingFacePage.module.css";

// ── Helpers ─────────────────────────────────────────────────────────────────

const fmtCount = (n: number | null | undefined): string | null => {
  if (n == null) return null;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return n.toLocaleString();
};

// ── Props ───────────────────────────────────────────────────────────────────

interface DatasetPreviewProps {
  preview: HuggingFacePreviewResponse;
  previewLoading: boolean;
  previewError: string | null;
  selectedConfig: string;
  onConfigChange: (config: string) => void;
}

// ── Component ────────────────────────────────────────────────────────────────

export function DatasetPreview({
  preview,
  previewLoading,
  previewError,
  selectedConfig,
  onConfigChange,
}: DatasetPreviewProps) {
  type TabKey = "description" | "tags" | "readme" | "columns" | "config";
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<TabKey>("description");

  if (previewLoading) {
    return (
      <div className={styles.previewCard} style={{ minHeight: 80 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
          <span className={styles.lookupSpinner} />
          <span style={{ marginLeft: 8, fontSize: 13, color: "var(--text-muted)" }}>Loading preview…</span>
        </div>
      </div>
    );
  }

  if (previewError && !preview) {
    return (
      <div className={styles.previewCard} style={{ padding: "12px 16px" }}>
        <p style={{ fontSize: 12, color: "var(--error)" }}>{previewError}</p>
      </div>
    );
  }

  if (!preview) return null;

  // Which tabs have content?
  const hasDescription = !!preview.description;
  const hasTags = !!(preview.tags && preview.tags.length > 0);
  const hasReadme = !!preview.readme_html;
  const hasColumns = Object.keys(preview.columns).length > 0;
  const hasConfigs = preview.configs.length > 0;

  const tabs: { key: TabKey; label: string; count?: number }[] = [];
  if (hasDescription) tabs.push({ key: "description", label: "Description" });
  if (hasTags) tabs.push({ key: "tags", label: "Tags", count: preview.tags?.length });
  if (hasReadme) tabs.push({ key: "readme", label: "README" });
  if (hasColumns) tabs.push({ key: "columns", label: "Columns", count: Object.keys(preview.columns).length });
  if (hasConfigs) tabs.push({ key: "config", label: "Config", count: preview.configs.length });

  // Set active tab to first available if current is hidden
  const activeTabValid = tabs.some((t) => t.key === activeTab);
  const effectiveTab: TabKey = activeTabValid ? activeTab : (tabs[0]?.key ?? "description");

  const sourceClass =
    preview.source === "hf-hub"
      ? styles.sourceBadgeBlue
      : preview.source === "datasets-server"
      ? styles.sourceBadgeOrange
      : styles.sourceBadgeGray;

  const sourceLabel =
    preview.source === "hf-hub"
      ? "◆ HF Hub"
      : preview.source === "datasets-server"
      ? "◈ datasets-server"
      : "◎ Hub API";

  return (
    <div className={styles.previewCard}>
      {/* ── Compact Hero ─────────────────────────────────────────────── */}
      <div className={styles.previewHero}>
        <div className={styles.previewIcon}>⬡</div>
        <div className={styles.previewHeroMeta}>
          <div className={styles.previewDatasetName}>{preview.full_dataset_id}</div>
          <div className={styles.previewBadgeRow}>
            {preview.estimated_size_human && (
              <Badge variant="accent">{preview.estimated_size_human}</Badge>
            )}
            <Badge variant="default">
              {preview.splits.length} split{preview.splits.length !== 1 ? "s" : ""}
            </Badge>
            {preview.license && <Badge variant="default">{preview.license}</Badge>}
            <span className={`${styles.previewSourceBadge} ${sourceClass}`}>{sourceLabel}</span>
          </div>
        </div>
      </div>

      {/* ── Stats row ───────────────────────────────────────────────── */}
      <div className={styles.previewStats}>
        {preview.language && preview.language.length > 0 && (
          <div className={styles.previewStat}>
            <span className={styles.statLabel}>Lang</span>
            <span className={styles.statValue}>{preview.language.slice(0, 3).join(", ")}</span>
          </div>
        )}
        {preview.downloads && (
          <div className={styles.previewStat}>
            <span className={styles.statLabel}>↓</span>
            <span className={styles.statValue}>{fmtCount(preview.downloads)}</span>
          </div>
        )}
        {preview.likes && (
          <div className={styles.previewStat}>
            <span className={styles.statLabel}>♥</span>
            <span className={styles.statValue}>{fmtCount(preview.likes)}</span>
          </div>
        )}
        {preview.tags && preview.tags.length > 0 && (
          <div className={styles.previewStat}>
            <span className={styles.statLabel}>Tags</span>
            <span className={styles.statValue}>{preview.tags.slice(0, 3).join(", ")}</span>
          </div>
        )}
      </div>

      {/* ── Description (compact) ─────────────────────────────────────── */}
      {preview.description && (
        <div className={styles.previewSection}>
          <p
            className={`${styles.description} ${
              !detailsOpen && preview.description.length > 180
                ? styles.descriptionClamped
                : ""
            }`}
          >
            {preview.description}
          </p>
          {preview.description.length > 180 && (
            <button
              className={styles.descToggle}
              onClick={() => setDetailsOpen((v) => !v)}
              type="button"
            >
              {detailsOpen ? "▲ Show less" : "▼ Show more"}
            </button>
          )}
        </div>
      )}

      {/* ── Details toggle ────────────────────────────────────────────── */}
      {(hasTags || hasReadme || hasConfigs) && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            padding: "4px 16px 8px",
          }}
        >
          <button
            className={styles.previewDetailsBtn}
            onClick={() => setDetailsOpen((v) => !v)}
            type="button"
          >
            {detailsOpen ? "▲ Hide details" : "▼ Show details"}
          </button>
        </div>
      )}

      {/* ── Expanded tabbed section ──────────────────────────────────── */}
      {detailsOpen && (
        <>
          {/* Tab strip */}
          {tabs.length > 0 && (
            <div className={styles.previewTabStrip}>
              {tabs.map((tab) => (
                <button
                  key={tab.key}
                  className={`${styles.previewTab} ${
                    effectiveTab === tab.key ? styles.previewTabActive : ""
                  }`}
                  onClick={() => setActiveTab(tab.key)}
                  type="button"
                >
                  {tab.label}
                  {tab.count != null && (
                    <span className={styles.previewTabBadge}>{tab.count}</span>
                  )}
                </button>
              ))}
            </div>
          )}

          {/* Tab content */}
          <div className={styles.previewSection} style={{ borderBottom: "none" }}>
            {/* Description tab */}
            {effectiveTab === "description" && preview.description && (
              <p className={styles.description}>{preview.description}</p>
            )}

            {/* Tags tab */}
            {effectiveTab === "tags" && preview.tags && (
              <div className={styles.tags}>
                {preview.tags.map((tag) => (
                  <span key={tag} className={styles.tag}>{tag}</span>
                ))}
              </div>
            )}

            {/* README tab */}
            {effectiveTab === "readme" && preview.readme_html && (
              <div
                className={styles.readmeContent}
                dangerouslySetInnerHTML={{ __html: preview.readme_html }}
              />
            )}

            {/* Columns tab */}
            {effectiveTab === "columns" && (
              <div>
                <div className={styles.columnsRow}>
                  {Object.entries(preview.columns)
                    .slice(0, 12)
                    .map(([name, type]) => (
                      <span
                        key={name}
                        className={`${styles.col} ${type === "string" ? styles.colText : ""}`}
                      >
                        {name}
                        <span className={styles.colType}>{type}</span>
                      </span>
                    ))}
                  {Object.keys(preview.columns).length > 12 && (
                    <span className={styles.colMore}>
                      +{Object.keys(preview.columns).length - 12} more
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* Config tab */}
            {effectiveTab === "config" && (
              <div>
                <p style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8 }}>
                  Select a configuration to continue:
                </p>
                <Select
                  id="config-select"
                  label="Configuration"
                  value={selectedConfig}
                  onChange={(e) => onConfigChange(e.target.value)}
                  options={preview.configs.map((c) => ({ value: c, label: c }))}
                />
              </div>
            )}
          </div>
        </>
      )}

      {/* Config hint when config is required but not selected */}
      {hasConfigs && !selectedConfig && !detailsOpen && (
        <div className={styles.configHintRow}>
          ⚠ Please select a configuration above to continue downloading.
        </div>
      )}
    </div>
  );
}
