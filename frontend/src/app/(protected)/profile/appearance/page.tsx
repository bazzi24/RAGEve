"use client";

import { Card, CardBody } from "@/components/ui/Card";
import { Palette } from "lucide-react";
import styles from "./page.module.css";

export default function AppearancePage() {
  return (
    <div className={styles.container}>
      <Card>
        <CardBody>
          <div className={styles.header}>
            <Palette size={32} className={styles.headerIcon} />
            <h1 className={styles.title}>Appearance Settings</h1>
            <p className={styles.subtitle}>
              Customize the look and feel of your interface.
            </p>
          </div>

          <div className={styles.placeholder}>
            <p>Appearance customization coming soon.</p>
            <p className={styles.placeholderSub}>
              Theme selection, color schemes, and layout options will be available here.
            </p>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
