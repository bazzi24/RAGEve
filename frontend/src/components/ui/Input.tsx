"use client";

import { forwardRef, type InputHTMLAttributes, type TextareaHTMLAttributes } from "react";
import styles from "./Input.module.css";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  hint?: string;
  error?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, hint, error, className = "", id, ...props }, ref) => {
    return (
      <div className={styles.wrapper}>
        {label && <label className={styles.label} htmlFor={id}>{label}</label>}
        <input
          ref={ref}
          id={id}
          className={`${styles.input} ${error ? styles.error : ""} ${className}`}
          {...props}
        />
        {hint && (
          <span className={`${styles.hint} ${error ? styles.errorText : ""}`}>{hint}</span>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  hint?: string;
  error?: boolean;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ label, hint, error, className = "", id, ...props }, ref) => {
    return (
      <div className={styles.wrapper}>
        {label && <label className={styles.label} htmlFor={id}>{label}</label>}
        <textarea
          ref={ref}
          id={id}
          className={`${styles.input} ${styles.textarea} ${error ? styles.error : ""} ${className}`}
          {...props}
        />
        {hint && (
          <span className={`${styles.hint} ${error ? styles.errorText : ""}`}>{hint}</span>
        )}
      </div>
    );
  }
);

Textarea.displayName = "Textarea";
