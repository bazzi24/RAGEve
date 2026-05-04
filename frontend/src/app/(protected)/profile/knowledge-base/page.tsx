"use client";

import { Card, CardBody } from "@/components/ui/Card";
import { BookOpen } from "lucide-react";
import styles from "./page.module.css";

export default function KnowledgeBasePage() {
  return (
    <div className={styles.container}>
      <Card>
        <CardBody>
          <div className={styles.header}>
            <BookOpen size={32} className={styles.headerIcon} />
            <h1 className={styles.title}>Knowledge Base Management</h1>
            <p className={styles.subtitle}>
              View and manage your knowledge bases and document processing settings.
            </p>
          </div>

          <div className={styles.placeholder}>
            <p>Knowledge base settings coming soon.</p>
            <p className={styles.placeholderSub}>
              Default parsers, chunking strategies, and ingestion settings will be configurable here.
            </p>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
