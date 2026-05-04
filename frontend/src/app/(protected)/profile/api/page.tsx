"use client";

import { Card, CardBody } from "@/components/ui/Card";
import { Code } from "lucide-react";
import styles from "./page.module.css";

export default function ApiSettingsPage() {
  return (
    <div className={styles.container}>
      <Card>
        <CardBody>
          <div className={styles.header}>
            <Code size={32} className={styles.headerIcon} />
            <h1 className={styles.title}>API Settings</h1>
            <p className={styles.subtitle}>
              Manage API keys, tokens, and programmatic access.
            </p>
          </div>

          <div className={styles.placeholder}>
            <p>API configuration coming soon.</p>
            <p className={styles.placeholderSub}>
              Generate and revoke API tokens, set rate limits, and view usage statistics.
            </p>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
