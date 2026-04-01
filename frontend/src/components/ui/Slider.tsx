"use client";

import { type InputHTMLAttributes } from "react";
import styles from "./Slider.module.css";

interface SliderProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "type"> {
  label?: string;
  showValue?: boolean;
  valueFormatter?: (value: number) => string;
}

export function Slider({
  label,
  showValue = true,
  valueFormatter = (v) => String(v),
  id,
  value,
  ...props
}: SliderProps) {
  return (
    <div className={styles.wrapper}>
      <div className={styles.header}>
        {label && <label className={styles.label} htmlFor={id}>{label}</label>}
        {showValue && (
          <span className={styles.value}>
            {valueFormatter(Number(value))}
          </span>
        )}
      </div>
      <input
        type="range"
        id={id}
        className={styles.slider}
        value={value}
        {...props}
      />
    </div>
  );
}
