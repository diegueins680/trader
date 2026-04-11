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

function normalizeApiTargetIdentity(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return "";
  const withoutTrailingSlashes = trimmed.replace(/\/+$/, "");
  return withoutTrailingSlashes || (trimmed.startsWith("/") ? "/" : "");
}

function sameApiTarget(left: string, right: string): boolean {
  const leftIdentity = normalizeApiTargetIdentity(left);
  const rightIdentity = normalizeApiTargetIdentity(right);
  return Boolean(leftIdentity) && leftIdentity === rightIdentity;
}

function readApiTargets(raw: Record<string, unknown>): Pick<TraderUiDeployConfig, "apiBaseUrl" | "apiBaseUrlInferred" | "apiFallbackUrl"> {
  const configuredBaseUrl = readString(raw.apiBaseUrl).trim();
  const configuredFallbackUrl = readString(raw.apiFallbackUrl).trim();
  const configuredInferred = readBoolean(raw.apiBaseUrlInferred);

  if (configuredBaseUrl) {
    return {
      apiBaseUrl: configuredBaseUrl,
      apiBaseUrlInferred: configuredInferred,
      apiFallbackUrl:
        configuredFallbackUrl && !sameApiTarget(configuredBaseUrl, configuredFallbackUrl)
          ? configuredFallbackUrl
          : undefined,
    };
  }

  const inferredBaseUrl = "/api";
  return {
    apiBaseUrl: inferredBaseUrl,
    apiBaseUrlInferred: true,
    apiFallbackUrl:
      configuredFallbackUrl && !sameApiTarget(inferredBaseUrl, configuredFallbackUrl)
        ? configuredFallbackUrl
        : undefined,
  };
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

function defaultConfig(): TraderUiDeployConfig {
  return {
    apiBaseUrl: "/api",
    apiBaseUrlInferred: true,
    apiToken: "",
  };
}

function readConfigFromGlobal(): TraderUiDeployConfig {
  if (typeof window === "undefined") return defaultConfig();
  const raw = window.__TRADER_CONFIG__;
  if (!raw || typeof raw !== "object") return defaultConfig();

  const apiTargets = readApiTargets(raw as Record<string, unknown>);
  return {
    ...apiTargets,
    apiToken: readString((raw as { apiToken?: unknown }).apiToken).trim(),
    timeoutsMs: readTimeouts((raw as { timeoutsMs?: unknown }).timeoutsMs),
  };
}

export const TRADER_UI_CONFIG: TraderUiDeployConfig = readConfigFromGlobal();