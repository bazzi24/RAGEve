import { apiFetch } from "./client";
import type {
  OllamaModelListResponse,
  ModelValidationResponse,
  ModelSelectionRequest,
} from "@/lib/types";

export async function listModels(): Promise<OllamaModelListResponse> {
  return apiFetch<OllamaModelListResponse>("/ollama/models");
}

export async function validateModels(
  payload: ModelSelectionRequest
): Promise<ModelValidationResponse> {
  return apiFetch<ModelValidationResponse>("/ollama/validate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
