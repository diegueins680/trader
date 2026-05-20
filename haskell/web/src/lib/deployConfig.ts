export type TraderUiTimeoutsMs = {
  requestMs?: number;
  signalMs?: number;
  backtestMs?: number;
  tradeMs?: number;
  optimizerMs?: number;
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

const flyAppNamePattern = /^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/;
const flyHostLabelPattern = /^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/;

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

function readNonBlankString(raw: unknown): string {
  return readString(raw).trim();
}

function readFirstNonBlankString(...raws: unknown[]): string {
  for (const raw of raws) {
    const value = readNonBlankString(raw);
    if (value) return value;
  }
  return "";
}

function readFirstBoolean(...raws: unknown[]): boolean | undefined {
  for (const raw of raws) {
    const value = readBoolean(raw);
    if (value !== undefined) return value;
  }
  return undefined;
}

function readFirstFlyAppName(...raws: unknown[]): string {
  for (const raw of raws) {
    const value = readNonBlankString(raw).toLowerCase();
    if (!value) continue;
    if (flyAppNamePattern.test(value)) return value;
  }
  return "";
}

function normalizeFlyHost(raw: unknown): string | undefined {
  if (raw === undefined || raw === null) return "fly.dev";
  if (typeof raw !== "string") return undefined;
  const value = raw.trim();
  if (!value) return "fly.dev";
  const normalized = value.replace(/\/+$/, "").toLowerCase();
  if (!normalized || /^https?:\/\//i.test(normalized) || normalized.includes("/")) return undefined;
  const labels = normalized.split(".");
  return labels.every((label) => flyHostLabelPattern.test(label)) ? normalized : undefined;
}

function deriveFlyApiFallbackUrl(raw: Record<string, unknown>): string {
  const backendFlyApp = readFirstFlyAppName(
    raw.backendFlyApp,
    raw.BACKEND_FLY_APP,
    raw.apiFallbackFlyApp,
    raw.API_FALLBACK_FLY_APP,
  );
  if (!backendFlyApp) return "";
  const flyHost = normalizeFlyHost(raw.flyHost ?? raw.FLY_HOST ?? raw.flyDomain ?? raw.FLY_DOMAIN);
  if (!flyHost) return "";
  return `https://${backendFlyApp}.${flyHost}/api`;
}

function normalizeApiTargetIdentity(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return "";
  const withoutTrailingSlashes = trimmed.replace(/\/+$/, "");
  if (!withoutTrailingSlashes) return trimmed.startsWith("/") ? "/" : "";
  if (!/^https?:\/\//i.test(withoutTrailingSlashes)) return withoutTrailingSlashes;
  try {
    const url = new URL(withoutTrailingSlashes);
    const pathname = url.pathname.replace(/\/+$/, "") || "/";
    return `${url.protocol.toLowerCase()}//${url.host.toLowerCase()}${pathname}${url.search}`;
  } catch {
    return withoutTrailingSlashes;
  }
}

function sameApiTarget(left: string, right: string): boolean {
  const leftIdentity = normalizeApiTargetIdentity(left);
  const rightIdentity = normalizeApiTargetIdentity(right);
  return Boolean(leftIdentity) && leftIdentity === rightIdentity;
}

function readApiTargets(
  raw: Record<string, unknown>,
): Pick<TraderUiDeployConfig, "apiBaseUrl" | "apiBaseUrlInferred" | "apiFallbackUrl"> {
  const configuredBaseUrl = readFirstNonBlankString(
    raw.apiBaseUrl,
    raw.API_BASE_URL,
    raw.TRADER_API_BASE_URL,
  );
  const configuredFallbackUrl = readFirstNonBlankString(
    raw.apiFallbackUrl,
    raw.API_FALLBACK_URL,
    raw.TRADER_API_FALLBACK_URL,
  );
  const derivedFallbackUrl = configuredFallbackUrl || deriveFlyApiFallbackUrl(raw);
  const configuredInferred = readFirstBoolean(
    raw.apiBaseUrlInferred,
    raw.API_BASE_URL_INFERRED,
    raw.TRADER_API_BASE_URL_INFERRED,
  );

  if (configuredBaseUrl) {
    return {
      apiBaseUrl: configuredBaseUrl,
      apiBaseUrlInferred: configuredInferred,
      apiFallbackUrl:
        derivedFallbackUrl && !sameApiTarget(configuredBaseUrl, derivedFallbackUrl)
          ? derivedFallbackUrl
          : undefined,
    };
  }

  const inferredBaseUrl = "/api";
  return {
    apiBaseUrl: inferredBaseUrl,
    apiBaseUrlInferred: true,
    apiFallbackUrl:
      derivedFallbackUrl && !sameApiTarget(inferredBaseUrl, derivedFallbackUrl)
        ? derivedFallbackUrl
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
    optimizerMs: normalizeTimeoutMs(r.optimizerMs),
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

  const config = raw as Record<string, unknown>;
  const apiTargets = readApiTargets(config);
  return {
    ...apiTargets,
    apiToken: readFirstNonBlankString(config.apiToken, config.API_TOKEN, config.TRADER_API_TOKEN),
    timeoutsMs: readTimeouts(config.timeoutsMs ?? config.TIMEOUTS_MS ?? config.TRADER_TIMEOUTS_MS),
  };
}

export const TRADER_UI_CONFIG: TraderUiDeployConfig = readConfigFromGlobal();