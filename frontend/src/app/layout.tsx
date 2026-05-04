import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAGEve",
  description: "AI-powered RAG platform with Ollama + Qdrant",
  icons: {
    icon: "/logo.png",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning data-scroll-behavior="smooth">
      <body>{children}</body>
    </html>
  );
}
