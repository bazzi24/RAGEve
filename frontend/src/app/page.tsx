"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useModelStore } from "@/stores/useModelStore";

export default function HomePage() {
  const { embeddingModel, chatModel, setupComplete } = useModelStore();
  const router = useRouter();

  useEffect(() => {
    if (setupComplete && embeddingModel && chatModel) {
      router.replace("/datasets");
    } else {
      router.replace("/setup");
    }
  }, [setupComplete, embeddingModel, chatModel, router]);

  return null;
}
