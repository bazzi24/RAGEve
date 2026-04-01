"use client";

import type { ReactNode } from "react";
import styles from "./Badge.module.css";

export type BadgeVariant = "default" | "accent" | "success" | "warning" | "error" | "info" | "muted";

interface BadgeProps {
  variant?: BadgeVariant;
  children: ReactNode;
  title?: string;
}

export function Badge({ variant = "default", children, title }: BadgeProps) {
  return <span className={`${styles.badge} ${styles[variant]}`} title={title}>{children}</span>;
}
