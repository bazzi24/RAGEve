/**
 * Date utilities for consistent timezone handling.
 *
 * Backend timestamps are stored in UTC but sent as naive ISO strings
 * (e.g., "2025-05-05T12:34:56" without timezone indicator).
 *
 * JavaScript's `new Date()` interprets these as LOCAL time, causing
 * a timezone offset. These utilities ensure timestamps are treated as UTC.
 */

/**
 * Parse an ISO string from the backend as UTC, or pass through a timestamp number.
 *
 * Backend uses datetime.utcnow().isoformat() which produces naive timestamps.
 * Appending "Z" forces the Date constructor to interpret it as UTC.
 *
 * @param value - ISO format string, timestamp number, or null/undefined
 * @returns Date object representing the correct UTC instant
 */
export function parseUTCDate(value: string | number | null | undefined): Date | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "number") {
    // Already a timestamp (milliseconds since epoch) - just wrap it
    return new Date(value);
  }
  // Append 'Z' if no timezone indicator present to force UTC parsing
  const withTimezone = value.endsWith("Z") || value.includes("+") || value.includes("-")
    ? value
    : value + "Z";
  return new Date(withTimezone);
}

/**
 * Format a backend timestamp for local display.
 *
 * @param value - ISO format string, timestamp number, or null/undefined
 * @param options - Intl.DateTimeFormatOptions
 * @returns Formatted date string in local timezone
 */
export function formatLocalDate(
  value: string | number | null | undefined,
  options?: Intl.DateTimeFormatOptions
): string {
  const date = parseUTCDate(value);
  if (!date) return "";
  return date.toLocaleDateString(undefined, options);
}

/**
 * Format a backend timestamp for local time display.
 *
 * @param value - ISO format string, timestamp number, or null/undefined
 * @param options - Intl.DateTimeFormatOptions
 * @returns Formatted time string in local timezone
 */
export function formatLocalTime(
  value: string | number | null | undefined,
  options?: Intl.DateTimeFormatOptions
): string {
  const date = parseUTCDate(value);
  if (!date) return "";
  return date.toLocaleTimeString(undefined, options);
}

/**
 * Format a backend timestamp for full local date+time display.
 *
 * @param value - ISO format string, timestamp number, or null/undefined
 * @param options - Intl.DateTimeFormatOptions
 * @returns Formatted date+time string in local timezone
 */
export function formatLocalDateTime(
  value: string | number | null | undefined,
  options?: Intl.DateTimeFormatOptions
): string {
  const date = parseUTCDate(value);
  if (!date) return "";
  return date.toLocaleString(undefined, options);
}

/**
 * Get the timestamp in milliseconds since epoch from a backend value.
 *
 * @param value - ISO format string, timestamp number, or null/undefined
 * @returns Unix timestamp in milliseconds, or null if invalid
 */
export function getUTCTimestamp(value: string | number | null | undefined): number | null {
  const date = parseUTCDate(value);
  return date ? date.getTime() : null;
}

/**
 * Format a relative time (e.g., "2 hours ago").
 *
 * @param isoString - ISO format string from backend
 * @returns Human-readable relative time
 */
export function formatRelativeTime(isoString: string | null | undefined): string {
  const date = parseUTCDate(isoString);
  if (!date) return "";

  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return "just now";
}
