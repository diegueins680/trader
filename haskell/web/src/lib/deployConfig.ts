export type TraderUiTimeoutsMs = {
  requestMs?: number;
  signalMs?: number;
  backtestMs?: number;
  tradeMs?: number;
  botStartMs?: number;
  botStatusMs?: number;
};

export type TraderUiDeployConfig = {
  apiBaseUrl: string;
  apiBaseUrlInferred?: boolean;
  apiFallbackUrl?: string;
  apiToken: string;
  timeoutsMs?: TraderUiTimeoutsMs;
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

function readBoolean(raw: unknown): boolean | undefined {
  if (typeof raw === "boolean") return raw;
  if (typeof raw === "number") {
    if (raw === 1) return true;
    if (raw === 0) return false;
  }
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (trimmed === "1") return true;
    if (trimmed === "0") return false;
    const normalized = trimmed.toLowerCase();
    if (normalized === "true") return true;
    if (normalized === "false") return false;
  }
  return undefined;
}

function normalizeTimeoutMs(raw: unknown): number | undefined {
  const n0 = readNumber(raw);
  if (n0 == null) return undefined;
  // Timeout configuration is integer-valued in milliseconds, so only exact safe
  // integers may cross the normalization boundary before range clamping.
  if (!Number.isSafeInteger(n0)) return undefined;
  const n = n0;
  if (n < 1_000) return undefined;
  // Avoid giant values overflowing timers / confusing UIs.
  return Math.min(n, 24 * 60 * 60 * 1_000);
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
    apiBaseUrlInferred: readBoolean((raw as { apiBaseUrlInferred?: unknown }).apiBaseUrlInferred),
    apiFallbackUrl: readString((raw as { apiFallbackUrl?: unknown }).apiFallbackUrl).trim(),
    apiToken: readString((raw as { apiToken?: unknown }).apiToken).trim(),
    timeoutsMs: readTimeouts((raw as { timeoutsMs?: unknown }).timeoutsMs),
  };
}

export const TRADER_UI_CONFIG: TraderUiDeployConfig = readConfigFromGlobal();
