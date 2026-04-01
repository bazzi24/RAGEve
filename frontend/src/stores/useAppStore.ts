"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface AppState {
  theme: "dark" | "light";
  sidebarCollapsed: boolean;
  setTheme: (theme: "dark" | "light") => void;
  toggleTheme: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      theme: "dark",
      sidebarCollapsed: false,
      setTheme: (theme) => {
        set({ theme });
        document.documentElement.setAttribute("data-theme", theme);
      },
      toggleTheme: () => {
        const next = get().theme === "dark" ? "light" : "dark";
        set({ theme: next });
        document.documentElement.setAttribute("data-theme", next);
      },
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
    }),
    { name: "ragve-app" }
  )
);
