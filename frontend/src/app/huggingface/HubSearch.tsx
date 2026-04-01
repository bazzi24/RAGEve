"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { searchHFDatasets } from "@/lib/api/datasets";
import styles from "./HuggingFacePage.module.css";

// ── Types ───────────────────────────────────────────────────────────────────

type SearchMode = "id" | "hub";

interface HFDatasetSearchResult {
  id: string;
  downloads: number | null;
  likes: number | null;
  tags: string[];
  description: string;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

const fmtCount = (n: number | null | undefined): string | null => {
  if (n == null) return null;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return n.toLocaleString();
};

const SUGGESTION_CHIPS = [
  { id: "squad", label: "squad", desc: "QA dataset" },
  { id: "imdb", label: "imdb", desc: "Movie reviews" },
  { id: "wikitext", label: "wikitext", desc: "Wikipedia text" },
  { id: "openai/webgpt_comparisons", label: "webgpt", desc: "GPT answers" },
];

// ── Props ───────────────────────────────────────────────────────────────────

interface HubSearchProps {
  datasetId: string;
  onDatasetIdChange: (id: string) => void;
  onChipClick: (id: string) => void;
}

// ── Component ───────────────────────────────────────────────────────────────

export function HubSearch({ datasetId, onDatasetIdChange, onChipClick }: HubSearchProps) {
  const [searchMode, setSearchMode] = useState<SearchMode>("id");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<HFDatasetSearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounced Hub search
  const fetchHubSearch = useCallback(async (q: string) => {
    if (q.trim().length < 2) {
      setSearchResults([]);
      setDropdownOpen(false);
      return;
    }
    setSearchLoading(true);
    setSearchError(null);
    try {
      const results = await searchHFDatasets(q.trim());
      setSearchResults(results);
      setDropdownOpen(true);
    } catch {
      setSearchError("Search failed. Try again.");
      setSearchResults([]);
    } finally {
      setSearchLoading(false);
    }
  }, []);

  const handleSearchInput = (value: string) => {
    setSearchQuery(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (searchMode === "hub") {
      debounceRef.current = setTimeout(() => void fetchHubSearch(value), 400);
    }
  };

  const handleSelectResult = (result: HFDatasetSearchResult) => {
    onDatasetIdChange(result.id);
    setDropdownOpen(false);
    setSearchQuery("");
  };

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest("[data-search-dropdown]")) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleModeSwitch = (mode: SearchMode) => {
    setSearchMode(mode);
    setSearchQuery("");
    setSearchResults([]);
    setDropdownOpen(false);
  };

  const inputValue = searchMode === "hub" ? searchQuery : datasetId;

  return (
    <div className={styles.pageHeader}>
      {/* Mode toggle */}
      <div className={styles.searchToggleRow}>
        <button
          className={`${styles.searchToggle} ${searchMode === "id" ? styles.searchToggleActive : ""}`}
          onClick={() => handleModeSwitch("id")}
          type="button"
        >
          Browse ID
        </button>
        <button
          className={`${styles.searchToggle} ${searchMode === "hub" ? styles.searchToggleActive : ""}`}
          onClick={() => handleModeSwitch("hub")}
          type="button"
        >
          Search Hub
        </button>
      </div>

      {/* Search input */}
      <div className={styles.searchSection} style={{ position: "relative" }}>
        <div className={styles.searchRow}>
          {searchMode === "hub" && (
            <svg
              className={styles.searchIcon}
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="11" cy="11" r="8" />
              <path d="M21 21l-4.35-4.35" />
            </svg>
          )}
          <input
            className={styles.mainInput}
            type="text"
            value={inputValue}
            onChange={(e) => {
              if (searchMode === "hub") {
                handleSearchInput(e.target.value);
              } else {
                onDatasetIdChange(e.target.value);
              }
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && searchMode === "hub" && searchResults.length > 0) {
                handleSelectResult(searchResults[0]);
              }
            }}
            placeholder={
              searchMode === "hub"
                ? "Search HuggingFace Hub…"
                : "Dataset ID (e.g. squad, imdb, th1nhng0/vietnamese-legal-documents)"
            }
            autoComplete="off"
            spellCheck={false}
            data-search-dropdown="true"
            style={searchMode === "hub" && searchQuery ? { paddingLeft: 32 } : {}}
          />

          {/* Lookup / loading indicator */}
          {searchMode === "hub" ? (
            searchLoading ? (
              <div className={styles.lookupRow}>
                <span className={styles.lookupSpinner} />
                Searching…
              </div>
            ) : searchQuery.length >= 2 && searchResults.length === 0 && !searchError ? (
              <div className={styles.lookupRow} style={{ color: "var(--text-muted)" }}>No results</div>
            ) : null
          ) : datasetId.trim().length >= 2 ? (
            <div className={styles.lookupRow}>
              <span className={styles.lookupSpinner} />
              Looking up…
            </div>
          ) : null}

          {/* Clear button */}
          {inputValue && (
            <button
              className={styles.searchClear}
              onClick={() => {
                if (searchMode === "hub") {
                  setSearchQuery("");
                  setSearchResults([]);
                  setDropdownOpen(false);
                } else {
                  onDatasetIdChange("");
                }
              }}
              type="button"
              title="Clear"
            >
              ✕
            </button>
          )}
        </div>

        {/* Hub search dropdown */}
        {searchMode === "hub" && dropdownOpen && (
          <div className={styles.searchDropdown} data-search-dropdown="true">
            {searchLoading && searchResults.length === 0 ? (
              <div className={styles.searchDropdownLoading}>
                {[1, 2, 3].map((i) => (
                  <div key={i} style={{ height: 44, background: "var(--bg-tertiary)", borderRadius: 6 }} />
                ))}
              </div>
            ) : searchResults.length === 0 && searchQuery.length >= 2 ? (
              <div className={styles.searchNoResults}>
                No datasets found for &ldquo;{searchQuery}&rdquo;
              </div>
            ) : (
              searchResults.map((result) => (
                <div
                  key={result.id}
                  className={styles.searchResultItem}
                  onClick={() => handleSelectResult(result)}
                >
                  <div className={styles.searchResultIcon}>⬡</div>
                  <div className={styles.searchResultInfo}>
                    <div className={styles.searchResultName}>{result.id}</div>
                    <div className={styles.searchResultMeta}>
                      {result.downloads != null && <span>↓ {fmtCount(result.downloads)}</span>}
                      {result.likes != null && <span>♥ {fmtCount(result.likes)}</span>}
                    </div>
                    {result.description && (
                      <div className={styles.searchResultDesc}>{result.description}</div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Suggestion chips — always visible */}
      <div className={styles.chipsRow}>
        <span className={styles.chipsLabel}>Popular:</span>
        {SUGGESTION_CHIPS.map((chip) => (
          <button
            key={chip.id}
            className={`${styles.chip} ${datasetId === chip.id ? styles.chipActive : ""}`}
            onClick={() => onChipClick(chip.id)}
            title={chip.desc}
            type="button"
          >
            {chip.label}
          </button>
        ))}
      </div>
    </div>
  );
}
