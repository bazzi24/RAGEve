"use client";

import { Card, CardBody } from "@/components/ui/Card";
import { Cpu } from "lucide-react";
import styles from "./page.module.css";

export default function ModelsPage() {
  return (
    <div className={styles.container}>
      <Card>
        <CardBody>
          <div className={styles.header}>
            <Cpu size={32} className={styles.headerIcon} />
            <h1 className={styles.title}>Model Setup</h1>
            <p className={styles.subtitle}>
              Configure LLM and embedding models for your RAG system.
            </p>
          </div>

          <div className={styles.placeholder}>
            <p>Model configuration coming soon.</p>
            <p className={styles.placeholderSub}>
              Select default models, adjust parameters, and manage model providers.
            </p>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
