import { apiFetch } from "./client";
import type {
  AgentCreate,
  AgentListResponse,
  AgentResponse,
  AgentUpdate,
} from "@/lib/types";

export async function listAgents(): Promise<AgentListResponse> {
  return apiFetch<AgentListResponse>("/agents/");
}

export async function createAgent(payload: AgentCreate): Promise<AgentResponse> {
  return apiFetch<AgentResponse>("/agents/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getAgent(agentId: string): Promise<AgentResponse> {
  return apiFetch<AgentResponse>(`/agents/${agentId}`);
}

export async function updateAgent(
  agentId: string,
  payload: AgentUpdate
): Promise<AgentResponse> {
  return apiFetch<AgentResponse>(`/agents/${agentId}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export async function deleteAgent(agentId: string): Promise<void> {
  return apiFetch<void>(`/agents/${agentId}`, { method: "DELETE" });
}
