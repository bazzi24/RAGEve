"use client";

import type { AgentResponse } from "@/lib/types";
import styles from "./AgentSelector.module.css";

interface AgentSelectorProps {
  agents: AgentResponse[];
  selectedId: string | null;
  onChange: (agentId: string) => void;
  disabled?: boolean;
}

export function AgentSelector({ agents, selectedId, onChange, disabled }: AgentSelectorProps) {
  return (
    <select
      className={styles.select}
      value={selectedId ?? ""}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled || agents.length === 0}
    >
      {agents.length === 0 ? (
        <option value="">No agents</option>
      ) : (
        <>
          <option value="">Select an agent…</option>
          {agents.map((agent) => (
            <option key={agent.agent_id} value={agent.agent_id}>
              {agent.name}
            </option>
          ))}
        </>
      )}
    </select>
  );
}
