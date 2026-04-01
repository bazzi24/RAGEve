"use client";

import { useAppStore } from "@/stores/useAppStore";
import { useModelStore } from "@/stores/useModelStore";
import styles from "./TopBar.module.css";

export function TopBar() {
  const { toggleTheme, theme } = useAppStore();
  const { embeddingModel, chatModel } = useModelStore();

  return (
    <header className={styles.topbar}>
      <div className={styles.left}>
        {embeddingModel && (
          <span className={styles.modelBadge} title={`Embed: ${embeddingModel} | Chat: ${chatModel}`}>
            {embeddingModel} / {chatModel}
          </span>
        )}
      </div>

      <div className={styles.right}>
        <button
          className={styles.iconBtn}
          onClick={toggleTheme}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          {theme === "dark" ? (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="8" cy="8" r="3" />
              <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.2 3.2l1.4 1.4M11.4 11.4l1.4 1.4M3.2 12.8l1.4-1.4M11.4 4.6l1.4-1.4" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M13.5 10A6 6 0 0 1 6 2.5a6 6 0 1 0 7.5 7.5z" />
            </svg>
          )}
        </button>
      </div>
    </header>
  );
}
