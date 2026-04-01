// Base API client with typed error handling

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public body?: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function getBaseUrl(): string {
  if (typeof window !== "undefined") {
    return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  }
  return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
}

/** Truncate long error bodies so they don't overflow UI elements. */
function _truncate(raw: string, max = 300): string {
  return raw.length > max ? raw.slice(0, max) + "…" : raw;
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const base = getBaseUrl();
  const url = `${base}${path}`;

  // 30-second timeout — AbortError is caught and surfaced as a user-friendly error.
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30_000);

  // Optional API key — mirrors the server-side X-API-Key header.
  const apiKey = process.env.NEXT_PUBLIC_API_KEY || "";

  const response = await fetch(url, {
    ...options,
    signal: controller.signal,
    headers: {
      "Content-Type": "application/json",
      ...(apiKey ? { "X-API-Key": apiKey } : {}),
      ...options.headers,
    },
  });

  clearTimeout(timeoutId);

  if (!response.ok) {
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      // Non-JSON responses (Cloudflare 502s, HTML error pages) can be thousands of
      // characters — truncate so they don't overflow UI elements.
      const raw = await response.text();
      body = _truncate(raw);
    }
    throw new ApiError(
      response.status,
      `API error ${response.status}: ${response.statusText}`,
      body
    );
  }

  // Handle empty responses (e.g., 204 No Content)
  const text = await response.text();
  if (!text) return undefined as T;
  return JSON.parse(text) as T;
}

export async function apiFetchFormData<T>(path: string, body: FormData): Promise<T> {
  const base = getBaseUrl();
  const url = `${base}${path}`;

  // 30-second timeout — multipart uploads are typically small; larger ones
  // should go through the streaming endpoint instead.
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30_000);

  const response = await fetch(url, {
    method: "POST",
    body,
    signal: controller.signal,
    // Note: NOT setting Content-Type header — browser sets multipart boundary automatically
  });

  clearTimeout(timeoutId);

  if (!response.ok) {
    let errBody: unknown;
    try {
      errBody = await response.json();
    } catch {
      errBody = _truncate(await response.text());
    }
    throw new ApiError(response.status, `Upload failed: ${response.statusText}`, errBody);
  }

  const text = await response.text();
  if (!text) return undefined as T;
  return JSON.parse(text) as T;
}

export function getSSEUrl(path: string): string {
  const base = getBaseUrl();
  return `${base}${path}`;
}
