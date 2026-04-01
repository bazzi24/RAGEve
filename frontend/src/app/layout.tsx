import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/layout/AppShell";
import { ToastContainer } from "@/components/ui/Toast";

export const metadata: Metadata = {
  title: "RAGEve",
  description: "AI-powered RAG platform with Ollama + Qdrant",
  icons: {
    icon: "/logo.jpg",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <body>
        <AppShell>{children}</AppShell>
        <ToastContainer />
      </body>
    </html>
  );
}
