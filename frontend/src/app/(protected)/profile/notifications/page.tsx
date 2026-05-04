"use client";

import { Card, CardBody } from "@/components/ui/Card";
import { Bell } from "lucide-react";
import styles from "./page.module.css";

export default function NotificationsPage() {
  return (
    <div className={styles.container}>
      <Card>
        <CardBody>
          <div className={styles.header}>
            <Bell size={32} className={styles.headerIcon} />
            <h1 className={styles.title}>Notification Settings</h1>
            <p className={styles.subtitle}>
              Configure how you receive notifications and alerts.
            </p>
          </div>

          <div className={styles.placeholder}>
            <p>Notification preferences coming soon.</p>
            <p className={styles.placeholderSub}>
              You'll be able to configure email, push, and in-app notifications.
            </p>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
