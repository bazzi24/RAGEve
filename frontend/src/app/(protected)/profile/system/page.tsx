"use client";

import { Card, CardBody } from "@/components/ui/Card";
import { Settings } from "lucide-react";
import styles from "./page.module.css";

export default function SystemPage() {
  return (
    <div className={styles.container}>
      <Card>
        <CardBody>
          <div className={styles.header}>
            <Settings size={32} className={styles.headerIcon} />
            <h1 className={styles.title}>System Settings</h1>
            <p className={styles.subtitle}>
              Configure system-wide preferences and advanced options.
            </p>
          </div>

          <div className={styles.placeholder}>
            <p>System configuration coming soon.</p>
            <p className={styles.placeholderSub}>
              Storage settings, cache configuration, and system diagnostics will be available here.
            </p>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
