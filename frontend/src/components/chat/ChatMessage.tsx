"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessageItem } from "@/lib/types";
import { StreamingCursor } from "./StreamingCursor";
import styles from "./ChatMessage.module.css";

interface ChatMessageProps {
  message: ChatMessageItem;
  isStreaming?: boolean;
}

export function ChatMessage({ message, isStreaming = false }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div className={`${styles.message} ${isUser ? styles.user : styles.assistant}`}>
      <div className={`${styles.avatar} ${isUser ? styles.userAvatar : styles.assistantAvatar}`}>
        {isUser ? (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
            <circle cx="7" cy="5" r="2.5" />
            <path d="M2 12c0-2.8 2.2-5 5-5s5 2.2 5 5" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M2 2h10v7H8l-2 2V9H2V2z" />
          </svg>
        )}
      </div>
      <div className={styles.bubble}>
        <div className={styles.content}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
          {isStreaming && message.role === "assistant" && <StreamingCursor />}
        </div>
        <div className={styles.timestamp}>
          {new Date(message.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </div>
      </div>
    </div>
  );
}
