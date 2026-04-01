"use client";

import { forwardRef, type SelectHTMLAttributes } from "react";
import styles from "./Select.module.css";

interface SelectOption {
  value: string;
  label: string;
}

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  options: SelectOption[];
  placeholder?: string;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ label, options, placeholder, id, className = "", ...props }, ref) => {
    return (
      <div className={styles.wrapper}>
        {label && <label className={styles.label} htmlFor={id}>{label}</label>}
        <select
          ref={ref}
          id={id}
          className={`${styles.select} ${className}`}
          {...props}
        >
          {placeholder && (
            <option value="" disabled>
              {placeholder}
            </option>
          )}
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>
    );
  }
);

Select.displayName = "Select";
