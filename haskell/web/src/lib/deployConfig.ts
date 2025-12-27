export type TraderUiDeployConfig = {
  apiBaseUrl: string;
  apiFallbackUrl?: string;
  apiToken: string;
  timeoutsMs?: TraderUiTimeoutsMs;
};

export type TraderUiTimeoutsMs = {
  requestMs?: number;
  signalMs?: number;
  backtestMs?: number;
  tradeMs?: number;
  botStartMs?: number;
  botStatusMs?: number;
};

function readString(raw: unknown): string {
  return typeof raw === "string" ? raw : "";
}

function readNumber(raw: unknown): number | null {
  if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    const n = Number(trimmed);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function normalizeTimeoutMs(raw: unknown): number | undefined {
  const n0 = readNumber(raw);
  if (n0 == null) return undefined;
  const n = Math.round(n0);
  if (n < 1000) return undefined;
  // Avoid giant values overflowing timers / confusing UIs.
  return Math.min(n, 24 * 60 * 60 * 1000);
}

function readTimeouts(raw: unknown): TraderUiTimeoutsMs | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const r = raw as Record<string, unknown>;
  const out: TraderUiTimeoutsMs = {
    requestMs: normalizeTimeoutMs(r.requestMs),
    signalMs: normalizeTimeoutMs(r.signalMs),
    backtestMs: normalizeTimeoutMs(r.backtestMs),
    tradeMs: normalizeTimeoutMs(r.tradeMs),
    botStartMs: normalizeTimeoutMs(r.botStartMs),
    botStatusMs: normalizeTimeoutMs(r.botStatusMs),
  };
  if (!Object.values(out).some((v) => typeof v === "number")) return undefined;
  return out;
}

function readConfigFromGlobal(): TraderUiDeployConfig {
  if (typeof window === "undefined") return { apiBaseUrl: "", apiToken: "" };
  const raw = window.__TRADER_CONFIG__;
  if (!raw || typeof raw !== "object") return { apiBaseUrl: "", apiToken: "" };

  return {
    apiBaseUrl: readString((raw as { apiBaseUrl?: unknown }).apiBaseUrl).trim(),
    apiFallbackUrl: readString((raw as { apiFallbackUrl?: unknown }).apiFallbackUrl).trim(),
    apiToken: readString((raw as { apiToken?: unknown }).apiToken).trim(),
    timeoutsMs: readTimeouts((raw as { timeoutsMs?: unknown }).timeoutsMs),
  };
}

export const TRADER_UI_CONFIG: TraderUiDeployConfig = readConfigFromGlobal();
