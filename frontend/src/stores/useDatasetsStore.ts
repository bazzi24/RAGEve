"use client";

import { create } from "zustand";
import type {
  ProcessedFileResponse,
  DatasetInfo,
  UploadProgressState,
} from "@/lib/types";
import { initialUploadProgressState } from "@/lib/types";

interface DatasetsState {
  datasets: DatasetInfo[];
  uploadingDatasetId: string | null;
  uploadResults: Record<string, ProcessedFileResponse[]>;
  uploadProgress: UploadProgressState;
  loading: boolean;
  error: string | null;

  setDatasets: (datasets: DatasetInfo[]) => void;
  addDataset: (dataset: DatasetInfo) => void;
  removeDataset: (datasetId: string) => void;
  setUploading: (datasetId: string | null) => void;
  setUploadResults: (datasetId: string, results: ProcessedFileResponse[]) => void;
  setUploadProgress: (progress: Partial<UploadProgressState>) => void;
  resetUploadProgress: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useDatasetsStore = create<DatasetsState>()((set) => ({
  datasets: [],
  uploadingDatasetId: null,
  uploadResults: {},
  uploadProgress: initialUploadProgressState,
  loading: false,
  error: null,

  setDatasets: (datasets) => set({ datasets }),
  addDataset: (dataset) =>
    set((state) => ({
      datasets: state.datasets.some((d) => d.dataset_id === dataset.dataset_id)
        ? state.datasets
        : [...state.datasets, dataset],
    })),
  removeDataset: (datasetId) =>
    set((state) => ({
      datasets: state.datasets.filter((d) => d.dataset_id !== datasetId),
      uploadResults: Object.fromEntries(
        Object.entries(state.uploadResults).filter(([k]) => k !== datasetId)
      ),
    })),
  setUploading: (uploadingDatasetId) => set({ uploadingDatasetId }),
  setUploadResults: (datasetId, results) =>
    set((state) => ({
      uploadResults: { ...state.uploadResults, [datasetId]: results },
    })),
  setUploadProgress: (progress) =>
    set((state) => ({
      uploadProgress: { ...state.uploadProgress, ...progress },
    })),
  resetUploadProgress: () => set({ uploadProgress: initialUploadProgressState }),
  setLoading: (loading) => set({ loading }),
  setError: (error) => set({ error }),
}));
