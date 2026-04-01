"use client";

import { useState, type ReactNode } from "react";
import styles from "./Accordion.module.css";

interface AccordionItemProps {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
}

export function AccordionItem({ title, children, defaultOpen = false }: AccordionItemProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className={styles.item}>
      <button
        className={`${styles.trigger} ${open ? styles.triggerOpen : ""}`}
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <span>{title}</span>
        <svg
          className={`${styles.chevron} ${open ? styles.open : ""}`}
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
        >
          <path d="M3 6l5 5 5-5" />
        </svg>
      </button>
      <div className={`${styles.content} ${open ? styles.open : ""}`}>
        <div className={styles.inner}>
          <div className={styles.panel}>{children}</div>
        </div>
      </div>
    </div>
  );
}

interface AccordionProps {
  children: ReactNode;
}

export function Accordion({ children }: AccordionProps) {
  return <div className={styles.root}>{children}</div>;
}
