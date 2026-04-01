"use client";

import { useEffect, useState } from "react";
import { createSession, deleteSession, listSessions } from "@/lib/api/chat";
import { useChatStore } from "@/stores/useChatStore";
import { useToastStore } from "@/stores/useToastStore";
import { Button } from "@/components/ui/Button";
import styles from "./SessionPanel.module.css";

interface SessionPanelProps {
  agentId: string;
  /** Called when a session is selected so the page can load its messages. */
  onSessionSelected: (sessionId: string) => void;
}

export function SessionPanel({ agentId, onSessionSelected }: SessionPanelProps) {
  const { sessions, currentSession, setSessions, addSession, removeSession, setCurrentSession } =
    useChatStore();
  const { addToast } = useToastStore();
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);

  // Reload sessions whenever the agent changes
  useEffect(() => {
    if (!agentId) return;
    setLoading(true);
    listSessions({ agent_id: agentId, limit: 50 })
      .then((res) => {
        setSessions(res.sessions);
        // Keep currentSession if it still belongs to this agent
        if (currentSession && currentSession.agent_id !== agentId) {
          setCurrentSession(null);
        }
      })
      .catch((err) => addToast(`Failed to load sessions: ${err.message}`, "error"))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId]);

  const handleCreateSession = async () => {
    if (!agentId) return;
    setCreating(true);
    try {
      const session = await createSession({ agent_id: agentId });
      addSession(session);
      onSessionSelected(session.session_id);
      addToast("New conversation started", "success");
    } catch (err) {
      addToast(`Failed to create session: ${err instanceof Error ? err.message : String(err)}`, "error");
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await deleteSession(sessionId);
      removeSession(sessionId);
      if (currentSession?.session_id === sessionId) {
        setCurrentSession(null);
      }
      addToast("Conversation deleted", "info");
    } catch (err) {
      addToast(`Failed to delete: ${err instanceof Error ? err.message : String(err)}`, "error");
    }
  };

  const handleSelectSession = (sessionId: string) => {
    const session = sessions.find((s) => s.session_id === sessionId);
    if (session) {
      setCurrentSession(session);
      onSessionSelected(sessionId);
    }
  };

  const agentSessions = sessions.filter((s) => s.agent_id === agentId);

  return (
    <div className={styles.panel}>
      {/* Header */}
      <div className={styles.header}>
        <span className={styles.title}>Conversations</span>
        <Button
          size="sm"
          onClick={handleCreateSession}
          loading={creating}
          disabled={!agentId}
          title="Start a new conversation"
        >
          + New
        </Button>
      </div>

      {/* Session list */}
      <div className={styles.list}>
        {loading && agentSessions.length === 0 ? (
          <div className={styles.empty}>Loading…</div>
        ) : agentSessions.length === 0 ? (
          <div className={styles.empty}>
            No conversations yet.
            <br />
            Click &ldquo;+ New&rdquo; to start.
          </div>
        ) : (
          agentSessions.map((session) => (
            <div
              key={session.session_id}
              className={`${styles.sessionItem} ${
                currentSession?.session_id === session.session_id ? styles.active : ""
              }`}
              onClick={() => handleSelectSession(session.session_id)}
              title={session.title}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") handleSelectSession(session.session_id); }}
            >
              <div className={styles.sessionItemContent}>
                <div className={styles.sessionTitle}>
                  {session.title || "New conversation"}
                </div>
                <div className={styles.sessionMeta}>
                  {session.message_count} message{session.message_count !== 1 ? "s" : ""}
                  {" · "}
                  {formatRelativeTime(session.updated_at)}
                </div>
              </div>
              <button
                className={styles.deleteBtn}
                onClick={(e) => handleDeleteSession(session.session_id, e)}
                title="Delete conversation"
                type="button"
              >
                ×
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function formatRelativeTime(isoString: string): string {
  if (!isoString) return "";
  const diff = Date.now() - new Date(isoString).getTime();
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(isoString).toLocaleDateString();
}
