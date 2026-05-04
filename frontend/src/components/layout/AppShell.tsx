"use client";

import { useAppStore } from "@/stores/useAppStore";
import { Sidebar } from "./Sidebar";
import styles from "./AppShell.module.css";

interface AppShellProps {
  children: React.ReactNode;
  noPadding?: boolean;
}

export function AppShell({ children, noPadding = false }: AppShellProps) {
  useAppStore(); // Subscribe to app state for sidebar collapse etc.

  return (
    <div className={styles.shell}>
      <Sidebar />
      <div className={styles.main}>
        <main className={`${styles.content} ${noPadding ? styles.noPadding : ""}`}>
          {children}
        </main>
      </div>
    </div>
  );
}
