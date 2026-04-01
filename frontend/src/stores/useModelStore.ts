"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { OllamaModelDetail, OllamaModelListResponse } from "@/lib/types";

interface ModelState {
  // Global default model selections (persisted)
  embeddingModel: string;
  chatModel: string;
  // Live model list
  availableModels: string[];
  modelDetails: OllamaModelDetail[];
  modelsLoaded: boolean;
  modelsRefreshing: boolean;
  setupComplete: boolean;
  // Ollama connection status
  ollamaConnected: boolean;
  ollamaReachable: boolean;

  setEmbeddingModel: (m: string) => void;
  setChatModel: (m: string) => void;
  setAvailableModels: (response: OllamaModelListResponse) => void;
  refreshOllamaModels: () => Promise<void>;
  setOllamaConnected: (v: boolean) => void;
  isSetupComplete: () => boolean;
  clearModels: () => void;
}

export const useModelStore = create<ModelState>()(
  persist(
    (set, get) => ({
      embeddingModel: "",
      chatModel: "",
      availableModels: [],
      modelDetails: [],
      modelsLoaded: false,
      modelsRefreshing: false,
      setupComplete: false,
      ollamaConnected: false,
      ollamaReachable: false,

      setEmbeddingModel: (embeddingModel) => set({ embeddingModel }),
      setChatModel: (chatModel) => set({ chatModel }),

      setAvailableModels: (response) =>
        set({
          availableModels: response.models,
          modelDetails: response.model_details ?? [],
          modelsLoaded: response.has_models,
          setupComplete: response.has_models && !!get().embeddingModel && !!get().chatModel,
        }),

      refreshOllamaModels: async () => {
        set({ modelsRefreshing: true });
        try {
          const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
          const res = await fetch(`${base}/ollama/models`);
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data: OllamaModelListResponse = await res.json();
          set({
            availableModels: data.models,
            modelDetails: data.model_details ?? [],
            modelsLoaded: data.has_models,
            ollamaConnected: true,
            ollamaReachable: true,
            modelsRefreshing: false,
          });
        } catch {
          set({
            modelsRefreshing: false,
            ollamaConnected: false,
            ollamaReachable: false,
          });
          throw new Error("Failed to refresh Ollama models. Is Ollama running?");
        }
      },

      setOllamaConnected: (v) => set({ ollamaConnected: v }),

      isSetupComplete: () => {
        const { embeddingModel, chatModel, setupComplete } = get();
        return setupComplete && !!embeddingModel && !!chatModel;
      },

      clearModels: () =>
        set({ embeddingModel: "", chatModel: "", setupComplete: false }),
    }),
    {
      name: "ragve-models",
      // Only persist the model selections, not the live model list
      partialize: (state) => ({
        embeddingModel: state.embeddingModel,
        chatModel: state.chatModel,
        setupComplete: state.setupComplete,
      }),
    }
  )
);
