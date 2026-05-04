"use client";

import { usePathname } from "next/navigation";
import { AppShell } from "@/components/layout/AppShell";
import { ToastContainer } from "@/components/ui/Toast";

export default function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  // Remove padding for profile section to allow full-height sidebar layout
  const noPadding = pathname.startsWith("/profile");

  return (
    <AppShell noPadding={noPadding}>
      {children}
      <ToastContainer />
    </AppShell>
  );
}
