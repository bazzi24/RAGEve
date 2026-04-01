"use client";

import { create } from "zustand";
import type { AgentResponse } from "@/lib/types";

interface AgentsState {
  agents: AgentResponse[];
  selectedAgentId: string | null;
  loading: boolean;
  error: string | null;

  setAgents: (agents: AgentResponse[]) => void;
  addAgent: (agent: AgentResponse) => void;
  updateAgent: (agent: AgentResponse) => void;
  removeAgent: (agentId: string) => void;
  setSelectedAgentId: (id: string | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useAgentsStore = create<AgentsState>()((set) => ({
  agents: [],
  selectedAgentId: null,
  loading: false,
  error: null,

  setAgents: (agents) => set({ agents }),
  addAgent: (agent) =>
    set((state) => ({
      agents: [...state.agents, agent],
    })),
  updateAgent: (agent) =>
    set((state) => ({
      agents: state.agents.map((a) => (a.agent_id === agent.agent_id ? agent : a)),
    })),
  removeAgent: (agentId) =>
    set((state) => ({
      agents: state.agents.filter((a) => a.agent_id !== agentId),
      selectedAgentId:
        state.selectedAgentId === agentId ? null : state.selectedAgentId,
    })),
  setSelectedAgentId: (selectedAgentId) => set({ selectedAgentId }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
}));
