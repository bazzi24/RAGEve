"use client";

import * as React from "react";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { getMe, updateProfile, changePassword, logout } from "@/lib/api/auth";
import { AuthMeResponse } from "@/lib/api/auth";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Spinner } from "@/components/ui/Spinner";
import { User, Mail, Calendar, Clock, Lock, LogOut } from "lucide-react";
import styles from "./page.module.css";

/* ==========================================================================
   Subcomponents
   ========================================================================== */

interface AlertMessageProps {
  type: "error" | "success";
  message: string;
}

function AlertMessage({ type, message }: AlertMessageProps) {
  return (
    <div className={`${styles.alert} ${type === "error" ? styles.alertError : styles.alertSuccess}`} role="alert">
      {message}
    </div>
  );
}

interface ProfileHeaderProps {
  user: AuthMeResponse;
}

function ProfileHeader({ user }: ProfileHeaderProps) {
  return (
    <div className={styles.header}>
      <div className={styles.avatarContainer}>
        <div className={styles.avatar} aria-hidden="true">
          <User size={40} />
        </div>
        <div className={styles.avatarRing} />
      </div>
      <div className={styles.headerContent}>
        <div className={styles.titleRow}>
          <h1 className={styles.title}>Profile</h1>
          {user.email_verified && (
            <span className={styles.verifiedBadge} style={{
              display: 'inline-flex',
              alignItems: 'center',
              padding: '2px 8px',
              fontSize: '11px',
              fontWeight: 500,
              background: 'rgba(34, 197, 94, 0.15)',
              color: '#22c55e',
              borderRadius: '9999px',
            }}>
              Verified
            </span>
          )}
        </div>
        <p className={styles.subtitle}>@{user.username}</p>
        <p className={styles.timestamp}>
          Member since {user.created_at ? new Date(user.created_at).toLocaleDateString() : "N/A"}
        </p>
      </div>
    </div>
  );
}

interface InfoItemProps {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
}

function InfoItem({ icon, label, value }: InfoItemProps) {
  return (
    <div className={styles.infoItem}>
      <div className={styles.infoIcon}>{icon}</div>
      <div className={styles.infoContent}>
        <span className={styles.infoLabel}>{label}</span>
        <span className={styles.infoValue}>{value}</span>
      </div>
    </div>
  );
}

interface InfoCardProps {
  user: AuthMeResponse;
}

function InfoCard({ user }: InfoCardProps) {
  return (
    <section className={styles.section}>
      <h2 className={styles.sectionTitle}>
        <User size={18} />
        Account Information
      </h2>
      <div className={styles.infoGrid}>
        <InfoItem
          icon={<User size={18} />}
          label="Username"
          value={user.username}
        />
        <InfoItem
          icon={<Mail size={18} />}
          label="Email"
          value={
            <div className={styles.emailRow}>
              <span>{user.email}</span>
              {!user.email_verified && (
                <span style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  padding: '2px 8px',
                  fontSize: '11px',
                  fontWeight: 500,
                  background: 'rgba(234, 179, 8, 0.15)',
                  color: '#eab308',
                  borderRadius: '9999px',
                }}>
                  Unverified
                </span>
              )}
            </div>
          }
        />
        <InfoItem
          icon={<User size={18} />}
          label="Full Name"
          value={user.full_name || "(not set)"}
        />
        <InfoItem
          icon={<Calendar size={18} />}
          label="Member since"
          value={user.created_at ? new Date(user.created_at).toLocaleDateString() : "N/A"}
        />
        <InfoItem
          icon={<Clock size={18} />}
          label="Last login"
          value={user.last_login_at ? new Date(user.last_login_at).toLocaleString() : "N/A"}
        />
      </div>
    </section>
  );
}

interface EditProfileFormProps {
  initialFullName: string;
  initialEmail: string;
  onSubmit: (fullName: string, email: string) => Promise<void>;
  isLoading: boolean;
}

function EditProfileForm({ initialFullName, initialEmail, onSubmit, isLoading }: EditProfileFormProps) {
  const [fullName, setFullName] = useState(initialFullName);
  const [email, setEmail] = useState(initialEmail);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onSubmit(fullName, email);
  };

  return (
    <section className={styles.section}>
      <h2 className={styles.sectionTitle}>
        <User size={18} />
        Edit Profile
      </h2>
      <form onSubmit={handleSubmit} className={styles.form}>
        <div className={styles.formGrid}>
          <Input
            label="Full Name"
            id="fullName"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            disabled={isLoading}
            placeholder="Enter your full name"
            className={styles.input}
          />
          <Input
            label="Email"
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={isLoading}
            hint="Changing email will require re-verification."
            placeholder="you@example.com"
            className={styles.input}
          />
        </div>
        <Button type="submit" disabled={isLoading} loading={isLoading}>
          Save Changes
        </Button>
      </form>
    </section>
  );
}

interface PasswordChangeFormProps {
  onSubmit: (currentPassword: string, newPassword: string) => Promise<void>;
  isLoading: boolean;
}

function PasswordChangeForm({ onSubmit, isLoading }: PasswordChangeFormProps) {
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (newPassword !== confirmPassword) {
      setError("New passwords do not match");
      return;
    }
    if (newPassword.length < 8) {
      setError("New password must be at least 8 characters long");
      return;
    }

    await onSubmit(currentPassword, newPassword);
    setCurrentPassword("");
    setNewPassword("");
    setConfirmPassword("");
  };

  return (
    <section className={styles.section}>
      <h2 className={styles.sectionTitle}>
        <Lock size={18} />
        Change Password
      </h2>
      {error && <AlertMessage type="error" message={error} />}
      <form onSubmit={handleSubmit} className={styles.form}>
        <div className={styles.formGrid}>
          <Input
            label="Current Password"
            id="currentPassword"
            type="password"
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            disabled={isLoading}
            required
            className={styles.input}
          />
          <Input
            label="New Password"
            id="newPassword"
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            disabled={isLoading}
            required
            minLength={8}
            hint="At least 8 characters"
            className={styles.input}
          />
          <Input
            label="Confirm New Password"
            id="confirmPassword"
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            disabled={isLoading}
            required
            className={styles.input}
          />
        </div>
        <Button type="submit" disabled={isLoading} loading={isLoading}>
          Change Password
        </Button>
      </form>
    </section>
  );
}

interface LogoutSectionProps {
  onLogout: () => Promise<void>;
}

function LogoutSection({ onLogout }: LogoutSectionProps) {
  return (
    <div className={styles.logoutSection}>
      <div className={styles.logoutContent}>
        <div className={styles.logoutInfo}>
          <h3 className={styles.logoutTitle}>Sign Out</h3>
          <p className={styles.logoutDesc}>
            End your current session and return to the login page.
          </p>
        </div>
        <Button variant="danger" onClick={onLogout}>
          <LogOut size={16} />
          Log Out
        </Button>
      </div>
    </div>
  );
}

/* ==========================================================================
   Main Profile Page Component
   ========================================================================== */

export default function ProfilePage() {
  const router = useRouter();
  const [user, setUser] = useState<AuthMeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Profile edit state
  const [savingProfile, setSavingProfile] = useState(false);
  const [changingPassword, setChangingPassword] = useState(false);

  useEffect(() => {
    const fetchUser = async () => {
      try {
        const data = await getMe();
        setUser(data);
      } catch {
        router.push("/login");
        return;
      } finally {
        setLoading(false);
      }
    };
    fetchUser();
  }, [router]);

  const handleProfileUpdate = async (fullName: string, email: string) => {
    setError(null);
    setSuccess(null);
    setSavingProfile(true);
    try {
      const updated = await updateProfile({
        full_name: fullName,
        email: email,
      });
      setUser(updated);
      setSuccess("Profile updated successfully");
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to update profile");
      }
    } finally {
      setSavingProfile(false);
    }
  };

  const handlePasswordChange = async (currentPassword: string, newPassword: string) => {
    setError(null);
    setSuccess(null);
    setChangingPassword(true);
    try {
      await changePassword({
        current_password: currentPassword,
        new_password: newPassword,
      });
      setSuccess("Password changed successfully");
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to change password");
      }
    } finally {
      setChangingPassword(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
      router.push("/login");
    } catch (err) {
      console.error("Logout failed", err);
    }
  };

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <Spinner size={32} />
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <div className={styles.container}>
      {error && <AlertMessage type="error" message={error} />}
      {success && <AlertMessage type="success" message={success} />}

      <ProfileHeader user={user} />
      <div className={styles.content}>
        <InfoCard user={user} />
        <EditProfileForm
          initialFullName={user.full_name || ""}
          initialEmail={user.email}
          onSubmit={handleProfileUpdate}
          isLoading={savingProfile}
        />
        <PasswordChangeForm
          onSubmit={handlePasswordChange}
          isLoading={changingPassword}
        />
        <LogoutSection onLogout={handleLogout} />
      </div>
    </div>
  );
}
