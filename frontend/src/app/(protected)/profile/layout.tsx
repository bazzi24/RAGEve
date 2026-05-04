"use client";

import ProfileLayout from "./ProfileLayout";

export default function ProfileSectionLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <ProfileLayout>{children}</ProfileLayout>;
}
