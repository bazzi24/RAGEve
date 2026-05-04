"use client";

import { type ReactNode } from "react";
import { usePathname } from "next/navigation";
import { SecondarySidebar } from "@/components/layout/SecondarySidebar";
import {
  User,
  Bell,
  Palette,
  Cpu,
  BookOpen,
  Code,
  Settings,
} from "lucide-react";
import styles from "./ProfileLayout.module.css";

interface ProfileLayoutProps {
  children: ReactNode;
}

// Navigation sections for Profile settings
const PROFILE_NAV_SECTIONS = [
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

export default function ProfileLayout({ children }: ProfileLayoutProps) {
  const pathname = usePathname();

  return (
    <div className={styles.layout}>
      {/* Secondary Sidebar - Sub-navigation */}
      <SecondarySidebar
        sections={PROFILE_NAV_SECTIONS}
        currentPath={pathname}
        className={styles.secondarySidebar}
      />

      {/* Main Content Area */}
      <main className={styles.main}>{children}</main>
    </div>
  );
}
