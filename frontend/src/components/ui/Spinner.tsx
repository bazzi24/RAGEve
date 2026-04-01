"use client";

import styles from "./Spinner.module.css";

interface SpinnerProps {
  size?: number;
  color?: string;
}

export function Spinner({ size = 20, color = "currentColor" }: SpinnerProps) {
  return (
    <span className={styles.spinner} role="status" aria-label="Loading">
      <svg
        className={styles.circle}
        width={size}
        height={size}
        viewBox="0 0 20 20"
        fill="none"
      >
        <circle
          cx="10"
          cy="10"
          r="8"
          stroke={color}
          strokeWidth="2"
          strokeOpacity="0.2"
        />
        <path
          d="M10 2a8 8 0 0 1 8 8"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
        />
      </svg>
    </span>
  );
}
