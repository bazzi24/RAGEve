import { apiFetch } from "./client";
import type { RerankersResponse } from "@/lib/types";

export async function getRerankers(): Promise<RerankersResponse> {
  return apiFetch<RerankersResponse>("/rerankers/");
}
