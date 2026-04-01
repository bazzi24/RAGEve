"use client";

import type { ReactNode } from "react";
import styles from "./Card.module.css";

interface CardProps {
  children: ReactNode;
}

interface CardHeaderProps {
  title?: string;
  actions?: ReactNode;
}

interface CardBodyProps {
  children: ReactNode;
}

interface CardFooterProps {
  children: ReactNode;
}

export function Card({ children }: CardProps) {
  return <div className={styles.card}>{children}</div>;
}

export function CardHeader({ title, actions }: CardHeaderProps) {
  return (
    <div className={styles.header}>
      {title && <span className={styles.title}>{title}</span>}
      {actions && <div>{actions}</div>}
    </div>
  );
}

export function CardBody({ children }: CardBodyProps) {
  return <div className={styles.body}>{children}</div>;
}

export function CardFooter({ children }: CardFooterProps) {
  return <div className={styles.footer}>{children}</div>;
}
