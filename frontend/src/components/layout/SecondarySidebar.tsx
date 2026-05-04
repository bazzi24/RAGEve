"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import {
  User,
  Bell,
  Palette,
  Cpu,
  BookOpen,
  Code,
  Settings,
} from "lucide-react";
import styles from "./SecondarySidebar.module.css";

export interface NavItem {
  id: string;
  label: string;
  icon: React.ReactNode;
  href: string;
}

export interface NavSection {
  title: string;
  items: NavItem[];
}

interface SecondarySidebarProps {
  sections: NavSection[];
  currentPath?: string;
  className?: string;
}

const DEFAULT_SECTIONS: NavSection[] = [
  {
    title: "General",
    items: [
      {
        id: "account",
        label: "Account",
        icon: <User size={18} />,
        href: "/profile",
      },
      {
        id: "notifications",
        label: "Notifications",
        icon: <Bell size={18} />,
        href: "/profile/notifications",
      },
      {
        id: "appearance",
        label: "Appearance",
        icon: <Palette size={18} />,
        href: "/profile/appearance",
      },
    ],
  },
  {
    title: "System",
    items: [
      {
        id: "models",
        label: "Model Setup",
        icon: <Cpu size={18} />,
        href: "/profile/models",
      },
      {
        id: "knowledge-base",
        label: "Knowledge Base",
        icon: <BookOpen size={18} />,
        href: "/profile/knowledge-base",
      },
      {
        id: "api",
        label: "API Settings",
        icon: <Code size={18} />,
        href: "/profile/api",
      },
      {
        id: "system",
        label: "System",
        icon: <Settings size={18} />,
        href: "/profile/system",
      },
    ],
  },
];

export function SecondarySidebar({
  sections = DEFAULT_SECTIONS,
  currentPath,
  className = "",
}: SecondarySidebarProps) {
  const pathname = usePathname();
  const activePath = currentPath || pathname;

  // Get all items as flat array for comparison
  const allItems = sections.flatMap((section) => section.items);

  // Find the most specific matching item (longest href that matches)
  const matchingItems = allItems.filter((item) => {
    if (activePath === item.href) return true;
    if (activePath.startsWith(item.href + "/")) return true;
    return false;
  });

  const longestMatch = matchingItems.length > 0
    ? matchingItems.sort((a, b) => b.href.length - a.href.length)[0]
    : null;

  return (
    <aside className={`${styles.sidebar} ${className}`.trim()}>
      <nav className={styles.nav} aria-label="Settings navigation">
        {sections.map((section) => (
          <div key={section.title} className={styles.section}>
            <h3 className={styles.sectionTitle}>{section.title}</h3>
            <ul className={styles.navList} role="list">
              {section.items.map((item) => {
                const active = longestMatch?.href === item.href;

                return (
                  <li key={item.id} role="listitem">
                    <Link
                      href={item.href}
                      className={`${styles.navItem} ${active ? styles.active : ""}`}
                      aria-current={active ? "page" : undefined}
                      title={item.label}
                    >
                      <span className={styles.navIcon}>{item.icon}</span>
                      <span className={styles.navLabel}>{item.label}</span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>
    </aside>
  );
}
