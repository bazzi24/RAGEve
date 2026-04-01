"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAppStore } from "@/stores/useAppStore";
import styles from "./Sidebar.module.css";

const NAV_ITEMS = [
  {
    href: "/models",
    label: "Models",
    icon: (
      <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="2" y="2" width="6" height="14" rx="1.5" />
        <rect x="10" y="2" width="6" height="14" rx="1.5" />
      </svg>
    ),
  },
  {
    href: "/datasets",
    label: "Datasets",
    icon: (
      <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="2" y="3" width="14" height="12" rx="2" />
        <path d="M5 7h8M5 10h5" />
      </svg>
    ),
  },
  {
    href: "/huggingface",
    label: "HuggingFace",
    icon: (
      <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="9" cy="9" r="7" />
        <path d="M6 12c1.5-1 2-2 3-3s1.5-2 3-3" />
        <circle cx="9" cy="5" r="1" fill="currentColor" />
      </svg>
    ),
  },
  {
    href: "/agents",
    label: "Agents",
    icon: (
      <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="9" cy="6" r="3" />
        <path d="M4 16c0-2.8 2.2-5 5-5s5 2.2 5 5" />
      </svg>
    ),
  },
  {
    href: "/chat",
    label: "Chat",
    icon: (
      <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M3 3h12v9H10l-3 3V12H3V3z" />
      </svg>
    ),
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const { sidebarCollapsed, setSidebarCollapsed } = useAppStore();

  return (
    <aside className={`${styles.sidebar} ${sidebarCollapsed ? styles.collapsed : ""}`}>
      <div className={styles.logo}>
        <div className={styles.logoIcon}>R</div>
        <span className={styles.logoText}>Mini RAG</span>
      </div>

      <nav className={styles.nav}>
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href || pathname.startsWith(item.href + "/");
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`${styles.navItem} ${active ? styles.active : ""}`}
              title={item.label}
            >
              <span className={styles.navIcon}>{item.icon}</span>
              <span className={styles.navLabel}>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <button
        className={styles.collapseBtn}
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          style={{ transform: sidebarCollapsed ? "rotate(180deg)" : "none", transition: "transform 0.2s" }}
        >
          <path d="M10 3L5 8l5 5" />
        </svg>
      </button>
    </aside>
  );
}
