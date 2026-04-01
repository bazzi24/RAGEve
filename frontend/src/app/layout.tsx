import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/layout/AppShell";
import { ToastContainer } from "@/components/ui/Toast";

export const metadata: Metadata = {
  title: "Mini RAG Platform",
  description: "AI-powered RAG platform with Ollama + Qdrant",
  icons: {
    icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%234f6ef7'/><text x='16' y='22' font-size='18' text-anchor='middle' fill='white' font-family='monospace'>R</text></svg>",
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
