"use client";

import { useToastStore, type ToastVariant } from "@/stores/useToastStore";
import styles from "./Toast.module.css";

const ICONS: Record<ToastVariant, React.ReactNode> = {
  success: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="8" r="6" />
      <path d="M5 8l2 2 4-4" />
    </svg>
  ),
  error: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="8" r="6" />
      <path d="M5.5 5.5l5 5M10.5 5.5l-5 5" />
    </svg>
  ),
  warning: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M8 2L14.5 13H1.5L8 2z" />
      <path d="M8 6v3M8 11v1" />
    </svg>
  ),
  info: (
    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="8" cy="8" r="6" />
      <path d="M8 7v4M8 5v1" />
    </svg>
  ),
};

export function ToastContainer() {
  const { toasts, removeToast } = useToastStore();

  return (
    <div className={styles.container} role="region" aria-label="Notifications">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`${styles.toast} ${styles[toast.variant]}`}
          role="alert"
        >
          <span className={styles.icon}>{ICONS[toast.variant]}</span>
          <span className={styles.message}>{toast.message}</span>
          <button className={styles.close} onClick={() => removeToast(toast.id)} aria-label="Dismiss">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M1 1l10 10M11 1L1 11" />
            </svg>
          </button>
        </div>
      ))}
    </div>
  );
}
