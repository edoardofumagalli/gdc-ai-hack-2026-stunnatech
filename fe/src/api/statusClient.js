const DEFAULT_STATUS_URL = "http://localhost:8000/api/status";
const DEFAULT_POLL_MS = 3000;
const DEFAULT_TIMEOUT_MS = 2500;

export function getRuntimeConfig() {
  const params = new URLSearchParams(window.location.search);
  const apiParam = params.get("api");
  const pollParam = Number(params.get("pollMs"));

  return {
    statusUrl: apiParam || import.meta.env.VITE_SEECURE_STATUS_URL || DEFAULT_STATUS_URL,
    pollMs:
      Number.isFinite(pollParam) && pollParam > 0
        ? pollParam
        : Number(import.meta.env.VITE_SEECURE_POLL_MS) || DEFAULT_POLL_MS,
    timeoutMs: Number(import.meta.env.VITE_SEECURE_TIMEOUT_MS) || DEFAULT_TIMEOUT_MS,
  };
}

export async function fetchStatus({ statusUrl, timeoutMs, signal } = {}) {
  const timeoutController = new AbortController();
  const timeoutId = window.setTimeout(() => timeoutController.abort(), timeoutMs || DEFAULT_TIMEOUT_MS);

  const abort = () => timeoutController.abort();
  signal?.addEventListener("abort", abort, { once: true });

  try {
    const response = await fetch(statusUrl || DEFAULT_STATUS_URL, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
      signal: timeoutController.signal,
    });

    if (!response.ok) {
      throw new Error(`Status API returned ${response.status}`);
    }

    return response.json();
  } finally {
    signal?.removeEventListener("abort", abort);
    window.clearTimeout(timeoutId);
  }
}
