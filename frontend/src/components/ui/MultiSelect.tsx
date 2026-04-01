"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./MultiSelect.module.css";

export interface MultiSelectOption {
  value: string;
  label: string;
  /** Optional type hint shown as a small tag (e.g. "string", "numeric") */
  typeHint?: string;
}

export interface MultiSelectProps {
  id: string;
  label?: string;
  options: MultiSelectOption[];
  selected: string[];
  onChange: (selected: string[]) => void;
  hint?: string;
  disabled?: boolean;
}

export function MultiSelect({
  id,
  label,
  options,
  selected,
  onChange,
  hint,
  disabled = false,
}: MultiSelectProps) {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    function handleClick(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [open]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [open]);

  function toggle(value: string) {
    if (selected.includes(value)) {
      onChange(selected.filter((v) => v !== value));
    } else {
      onChange([...selected, value]);
    }
  }

  function selectAll() {
    onChange(options.map((o) => o.value));
  }

  function clearAll() {
    onChange([]);
  }

  const selectedOptions = options.filter((o) => selected.includes(o.value));

  return (
    <div className={styles.wrapper} ref={wrapperRef}>
      {label && (
        <label className={styles.label} htmlFor={`${id}-trigger`}>
          {label}
        </label>
      )}

      {/* Trigger button */}
      <button
        id={`${id}-trigger`}
        type="button"
        className={styles.trigger}
        onClick={() => !disabled && setOpen((v) => !v)}
        disabled={disabled}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <span className={styles.triggerText}>
          {selectedOptions.length === 0 ? (
            <span className={styles.placeholder}>Select columns…</span>
          ) : (
            <span className={styles.selectedCount}>
              {selectedOptions.length} column{selectedOptions.length !== 1 ? "s" : ""} selected
            </span>
          )}
        </span>
        <span className={styles.chevron} aria-hidden="true">
          {open ? "▲" : "▼"}
        </span>
      </button>

      {/* Selected chips */}
      {selectedOptions.length > 0 && (
        <div className={styles.chipList}>
          {selectedOptions.map((opt) => (
            <span key={opt.value} className={styles.chip}>
              {opt.label}
              {opt.typeHint && (
                <span className={styles.chipType}>{opt.typeHint}</span>
              )}
              <button
                type="button"
                className={styles.chipRemove}
                onClick={(e) => {
                  e.stopPropagation();
                  toggle(opt.value);
                }}
                aria-label={`Remove ${opt.label}`}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Hint */}
      {hint && <p className={styles.hint}>{hint}</p>}

      {/* Dropdown */}
      {open && (
        <div className={styles.dropdown} role="listbox" aria-multiselectable="true">
          {/* Select all / Clear all */}
          <div className={styles.dropdownActions}>
            <button
              type="button"
              className={styles.dropdownAction}
              onClick={selectAll}
            >
              Select all
            </button>
            <span className={styles.dropdownSep}>·</span>
            <button
              type="button"
              className={styles.dropdownAction}
              onClick={clearAll}
            >
              Clear
            </button>
          </div>

          {/* Options list */}
          {options.map((opt) => {
            const checked = selected.includes(opt.value);
            return (
              <label key={opt.value} className={styles.dropdownItem}>
                <input
                  type="checkbox"
                  className={styles.itemCheckbox}
                  checked={checked}
                  onChange={() => toggle(opt.value)}
                />
                <span className={styles.itemLabel}>{opt.label}</span>
                {opt.typeHint && (
                  <span className={styles.itemType}>{opt.typeHint}</span>
                )}
              </label>
            );
          })}

          {options.length === 0 && (
            <div className={styles.dropdownEmpty}>No columns available</div>
          )}
        </div>
      )}
    </div>
  );
}
