import type {
  ApiError,
  ApiBinancePositionsRequest,
  ApiBinancePositionsResponse,
  ApiBinanceTradesRequest,
  ApiBinanceTradesResponse,
  ApiParams,
  ApiTradeResponse,
  BacktestResponse,
  BinanceKeysStatus,
  BinanceListenKeyKeepAliveResponse,
  BinanceListenKeyResponse,
  BotStatus,
  CoinbaseKeysStatus,
  LatestSignal,
  OpsPerformanceResponse,
  OpsResponse,
  OptimizerRunRequest,
  OptimizerRunResponse,
  StateSyncImportResponse,
  StateSyncPayload,
} from "./types";
import { TRADER_UI_CONFIG } from "./deployConfig";
import { readJson, writeJson } from "./storage";

export class HttpError extends Error {
  readonly status: number;
  readonly payload?: unknown;
  readonly retryAfterMs: number | null;

  constructor(status: number, message: string, payload?: unknown, retryAfterMs?: number | null) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.payload = payload;
    this.retryAfterMs = typeof retryAfterMs === "number" && Number.isFinite(retryAfterMs) ? retryAfterMs : null;
  }
}

export class UnexpectedResponseError extends Error {
  readonly status: number;
  readonly contentType: string;
  readonly bodySnippet: string;

  constructor(status: number, contentType: string, bodySnippet: string) {
    const label = contentType ? contentType.split(";")[0]?.trim() ?? "unknown" : "unknown";
    super(`Unexpected non-JSON response (${label}). Check your API base or /api proxy.`);
    this.name = "UnexpectedResponseError";
    this.status = status;
    this.contentType = contentType;
    this.bodySnippet = bodySnippet;
  }
}

type FetchJsonOptions = {
  signal?: AbortSignal;
  timeoutMs?: number;
  headers?: Record<string, string>;
  allowFallback?: boolean;
};

type AsyncStartResponse = { jobId: string };
type AsyncPollResponse<T> = { status: "running" | "done" | "error"; result?: T; error?: string };
export type HealthResponse = {
  status: "ok";
  version?: string;
  commit?: string;
  authRequired?: boolean;
  authOk?: boolean;
  computeLimits?: { maxBarsLstm: number; maxEpochs: number; maxHiddenSize: number };
  asyncJobs?: { maxRunning: number; ttlMs: number; persistence: boolean };
  cache?: { enabled: boolean; ttlMs: number; maxEntries: number };
};

export type CacheStatsResponse = {
  enabled: boolean;
  ttlMs: number;
  maxEntries: number;
  signals: { entries: number; hits: number; misses: number };
  backtests: { entries: number; hits: number; misses: number };
  atMs: number;
};

export type CacheClearResponse = { ok: boolean; atMs: number };
type AsyncJobOptions = FetchJsonOptions & {
  onJobId?: (jobId: string) => void;
  retryStart?: boolean;
  maxStartRetries?: number;
};

function resolveUrl(baseUrl: string, path: string): string {
  const base = baseUrl.trim().replace(/\/+$/, "");
  const raw = path.startsWith("/") ? path : `/${path}`;
  const hashIndex = raw.indexOf("#");
  const rawNoHash = hashIndex >= 0 ? raw.slice(0, hashIndex) : raw;
  const hash = hashIndex >= 0 ? raw.slice(hashIndex) : "";
  const queryIndex = rawNoHash.indexOf("?");
  const pathname = queryIndex >= 0 ? rawNoHash.slice(0, queryIndex) : rawNoHash;
  const search = queryIndex >= 0 ? rawNoHash.slice(queryIndex) : "";

  if (/^https?:\/\//.test(base)) {
    const url = new URL(base);
    const basePath = url.pathname.replace(/\/+$/, "");
    url.pathname = `${basePath}${pathname}`.replace(/\/{2,}/g, "/") || "/";
    url.search = search;
    url.hash = hash;
    return url.toString();
  }

  const rel = base.startsWith("/") ? base : `/${base}`;
  return `${rel}${pathname}${search}${hash}`;
}

function normalizeBaseUrl(raw: string): string {
  return raw.trim().replace(/\/+$/, "");
}

const FALLBACK_STORAGE_KEY = "trader_api_fallback_v1";
const FALLBACK_STORAGE_TTL_MS = 12 * 60 * 60 * 1000;

type FallbackStorage = {
  savedAtMs: number;
  blocked: string[];
  preferred: Record<string, string>;
};

function emptyFallbackStorage(): FallbackStorage {
  return { savedAtMs: 0, blocked: [], preferred: {} };
}

function loadFallbackStorage(): FallbackStorage {
  const raw = readJson<FallbackStorage>(FALLBACK_STORAGE_KEY);
  if (!raw || typeof raw !== "object") return emptyFallbackStorage();
  const savedAtMs = typeof raw.savedAtMs === "number" && Number.isFinite(raw.savedAtMs) ? raw.savedAtMs : 0;
  if (!savedAtMs || Date.now() - savedAtMs > FALLBACK_STORAGE_TTL_MS) return emptyFallbackStorage();
  const blocked = Array.isArray(raw.blocked)
    ? raw.blocked
        .filter((entry): entry is string => typeof entry === "string")
        .map((entry) => normalizeBaseUrl(entry))
        .filter(Boolean)
    : [];
  const preferredEntries =
    raw.preferred && typeof raw.preferred === "object" ? Object.entries(raw.preferred as Record<string, unknown>) : [];
  const preferred: Record<string, string> = {};
  for (const [primary, fallback] of preferredEntries) {
    if (typeof fallback !== "string") continue;
    const primaryNorm = normalizeBaseUrl(primary);
    const fallbackNorm = normalizeBaseUrl(fallback);
    if (!primaryNorm || !fallbackNorm || primaryNorm === fallbackNorm) continue;
    preferred[primaryNorm] = fallbackNorm;
  }
  return { savedAtMs, blocked: Array.from(new Set(blocked)), preferred };
}

const fallbackStorage = loadFallbackStorage();
const blockedFallbackBases = new Set<string>(fallbackStorage.blocked);
const preferredFallbackBases = new Map<string, string>(Object.entries(fallbackStorage.preferred));

function persistFallbackStorage() {
  writeJson(FALLBACK_STORAGE_KEY, {
    savedAtMs: Date.now(),
    blocked: Array.from(blockedFallbackBases),
    preferred: Object.fromEntries(preferredFallbackBases),
  } satisfies FallbackStorage);
}

function rememberPreferredFallback(primary: string, fallback: string) {
  if (!primary || !fallback || primary === fallback) return;
  if (blockedFallbackBases.has(fallback)) return;
  if (preferredFallbackBases.get(primary) === fallback) return;
  preferredFallbackBases.set(primary, fallback);
  persistFallbackStorage();
}

function clearPreferredFallback(primary: string) {
  if (!preferredFallbackBases.delete(primary)) return;
  persistFallbackStorage();
}

function blockFallbackBase(fallback: string) {
  if (!fallback || blockedFallbackBases.has(fallback)) return;
  blockedFallbackBases.add(fallback);
  for (const [primary, preferred] of preferredFallbackBases.entries()) {
    if (preferred === fallback) preferredFallbackBases.delete(primary);
  }
  persistFallbackStorage();
}

function isCrossOriginBase(baseUrl: string): boolean {
  if (typeof window === "undefined") return false;
  if (!/^https?:\/\//.test(baseUrl)) return false;
  try {
    return new URL(baseUrl).origin !== window.location.origin;
  } catch {
    return false;
  }
}

function isJsonContentType(raw: string): boolean {
  const ct = raw.toLowerCase();
  return ct.includes("application/json") || ct.includes("+json");
}

function resolveFallbackBase(primaryBase: string): string | null {
  const fallbackRaw = TRADER_UI_CONFIG.apiFallbackUrl?.trim() ?? "";
  if (!fallbackRaw) return null;
  const primary = normalizeBaseUrl(primaryBase);
  const fallback = normalizeBaseUrl(fallbackRaw);
  if (!fallback || fallback === primary) return null;
  if (blockedFallbackBases.has(fallback)) return null;
  return fallback;
}

function resolvePreferredFallback(primaryBase: string, fallbackBase: string | null): string | null {
  if (!fallbackBase) return null;
  const preferred = preferredFallbackBases.get(primaryBase) ?? null;
  if (preferred) {
    if (preferred !== fallbackBase) return null;
    if (blockedFallbackBases.has(preferred)) return null;
    return preferred;
  }
  return null;
}

function mergeHeaders(base: HeadersInit | undefined, extra: Record<string, string> | undefined): HeadersInit | undefined {
  if (!extra || Object.keys(extra).length === 0) return base;
  const merged = new Headers(base);
  for (const [key, value] of Object.entries(extra)) merged.set(key, value);
  return merged;
}

function parseRetryAfterMs(raw: string | null): number | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (/^\d+$/.test(trimmed)) return Math.max(0, Number(trimmed) * 1000);
  const parsed = Date.parse(trimmed);
  if (!Number.isNaN(parsed)) return Math.max(0, parsed - Date.now());
  return null;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      cleanup();
      reject((signal as AbortSignal & { reason?: unknown }).reason ?? new DOMException("Aborted", "AbortError"));
    };

    const timer = window.setTimeout(() => {
      cleanup();
      resolve();
    }, ms);

    const cleanup = () => {
      window.clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    };

    if (signal) {
      if (signal.aborted) return onAbort();
      signal.addEventListener("abort", onAbort, { once: true });
    }
  });
}

function withTimeout(externalSignal: AbortSignal | undefined, timeoutMs: number) {
  const controller = new AbortController();
  let onAbort: (() => void) | null = null;

  if (externalSignal) {
    if (externalSignal.aborted) controller.abort(externalSignal.reason);
    else {
      onAbort = () => controller.abort(externalSignal.reason);
      externalSignal.addEventListener("abort", onAbort, { once: true });
    }
  }

  const timer = window.setTimeout(() => controller.abort(new DOMException("Timeout", "TimeoutError")), timeoutMs);
  return {
    signal: controller.signal,
    cleanup: () => {
      window.clearTimeout(timer);
      if (externalSignal && onAbort) externalSignal.removeEventListener("abort", onAbort);
    },
  };
}

async function readJsonOrText(res: Response, contentType: string): Promise<unknown> {
  const bodyText = await res.text();
  if (isJsonContentType(contentType)) {
    const trimmed = bodyText.trim();
    if (!trimmed) return null;
    try {
      return JSON.parse(trimmed);
    } catch {
      throw new UnexpectedResponseError(res.status, contentType, summarizePayload(bodyText));
    }
  }
  return bodyText;
}

function summarizePayload(payload: unknown): string {
  if (payload == null) return "";
  if (typeof payload === "string") {
    const trimmed = payload.trim();
    return trimmed.length > 320 ? `${trimmed.slice(0, 320)}...` : trimmed;
  }
  try {
    const json = JSON.stringify(payload);
    return json.length > 320 ? `${json.slice(0, 320)}...` : json;
  } catch {
    return "";
  }
}

async function fetchJsonOnce<T>(baseUrl: string, path: string, init: RequestInit, opts?: FetchJsonOptions): Promise<T> {
  const timeoutMs = opts?.timeoutMs ?? TRADER_UI_CONFIG.timeoutsMs?.requestMs ?? 30_000;
  const { signal, cleanup } = withTimeout(opts?.signal, timeoutMs);
  try {
    const url = resolveUrl(baseUrl, path);
    const res = await fetch(url, {
      ...init,
      cache: init.cache ?? "no-store",
      headers: mergeHeaders(init.headers, opts?.headers),
      signal,
    });
    const contentType = res.headers.get("content-type") || "";
    const retryAfterMs = parseRetryAfterMs(res.headers.get("retry-after"));
    const payload = await readJsonOrText(res, contentType);
    if (res.ok && !isJsonContentType(contentType)) {
      throw new UnexpectedResponseError(res.status, contentType, summarizePayload(payload));
    }
    if (!res.ok) {
      const baseMessage =
        typeof payload === "object" && payload && "error" in payload
          ? String((payload as ApiError).error)
          : typeof payload === "string" && payload.trim()
            ? payload.trim()
            : `${res.status} ${res.statusText}`;
      const hint =
        typeof payload === "object" && payload && "hint" in payload && (payload as ApiError).hint
          ? String((payload as ApiError).hint)
          : "";
      const message = hint ? `${baseMessage}\nHint: ${hint}` : baseMessage;
      throw new HttpError(res.status, message, payload, retryAfterMs);
    }
    return payload as T;
  } catch (err) {
    if (signal.aborted) {
      const reason = (signal as AbortSignal & { reason?: unknown }).reason;
      if (reason instanceof DOMException && reason.name === "TimeoutError") throw reason;
    }
    throw err;
  } finally {
    cleanup();
  }
}

async function fetchJson<T>(baseUrl: string, path: string, init: RequestInit, opts?: FetchJsonOptions): Promise<T> {
  const primaryBase = normalizeBaseUrl(baseUrl);
  const fallbackBase = resolveFallbackBase(primaryBase);
  const allowFallback = opts?.allowFallback !== false;
  const preferredBase = allowFallback ? resolvePreferredFallback(primaryBase, fallbackBase) : null;

  if (preferredBase) {
    try {
      return await fetchJsonOnce<T>(preferredBase, path, init, opts);
    } catch (err) {
      clearPreferredFallback(primaryBase);
      if (fallbackBase && preferredBase === fallbackBase && isNetworkError(err) && isCrossOriginBase(fallbackBase)) {
        blockFallbackBase(fallbackBase);
      }
    }
  }
  try {
    return await fetchJsonOnce<T>(primaryBase, path, init, opts);
  } catch (err) {
    if (fallbackBase && allowFallback && shouldFallbackToApiBase(err)) {
      try {
        const out = await fetchJsonOnce<T>(fallbackBase, path, init, opts);
        rememberPreferredFallback(primaryBase, fallbackBase);
        return out;
      } catch (fallbackErr) {
        if (isNetworkError(fallbackErr) && isCrossOriginBase(fallbackBase)) {
          blockFallbackBase(fallbackBase);
          throw err;
        }
        throw fallbackErr;
      }
    }
    throw err;
  }
}

function timeoutError(): DOMException {
  return new DOMException("Timeout", "TimeoutError");
}

function isTimeoutError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "TimeoutError";
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function isNetworkError(err: unknown): boolean {
  return err instanceof TypeError;
}

function shouldFallbackToApiBase(err: unknown): boolean {
  if (err instanceof UnexpectedResponseError) return true;
  if (isAbortError(err) || isTimeoutError(err)) return false;
  if (err instanceof HttpError) return err.status === 502 || err.status === 503 || err.status === 504;
  return isNetworkError(err);
}

function shouldFallbackToGet(err: unknown): boolean {
  if (!(err instanceof HttpError)) return false;
  return err.status === 403 || err.status === 405 || err.status === 501 || err.status === 502 || err.status === 503 || err.status === 504;
}

function asyncJobNotFoundMessage(): string {
  return "Async job not found (server restarted or behind a non-sticky load balancer). Please retry; for multi-instance deployments, enable shared async job storage (TRADER_API_ASYNC_DIR or TRADER_STATE_DIR) or run single-instance.";
}

function describeAsyncTimeout(baseUrl: string, overallTimeoutMs: number, lastError: unknown): string {
  const seconds = Math.max(1, Math.round(overallTimeoutMs / 1000));
  const last =
    lastError instanceof HttpError
      ? `${lastError.status} ${lastError.message}`
      : lastError instanceof Error
        ? lastError.message
        : String(lastError);
  const hint = baseUrl.startsWith("/api")
    ? " Check your CloudFront `/api/*` proxy (or set apiBaseUrl in trader-config.js to your API host)."
    : " Check API connectivity and try again.";
  return `Async request timed out after ${seconds}s while retrying after errors (last error: ${last}).${hint}`;
}

function isBacktestQueueBusy(err: unknown): err is HttpError {
  if (!(err instanceof HttpError)) return false;
  if (err.status !== 429) return false;
  return err.message.toLowerCase().includes("backtest queue is busy");
}

function shouldRetryAsyncStart(err: unknown): boolean {
  if (isTimeoutError(err) || isNetworkError(err)) return true;
  if (err instanceof UnexpectedResponseError) return true;
  return err instanceof HttpError && (err.status === 502 || err.status === 503 || err.status === 504);
}

async function runSyncBacktestWithRetry(
  baseUrl: string,
  params: ApiParams,
  opts?: AsyncJobOptions,
): Promise<BacktestResponse> {
  const startedAt = Date.now();
  const overallTimeoutMs = opts?.timeoutMs ?? 30_000;
  let backoffMs = 750;
  let sawBusy = false;

  for (;;) {
    const elapsed = Date.now() - startedAt;
    const remaining = overallTimeoutMs - elapsed;
    if (remaining <= 0) {
      if (sawBusy) {
        throw new Error("Backtest queue stayed busy. Try again shortly or increase TRADER_API_MAX_BACKTEST_RUNNING.");
      }
      throw timeoutError();
    }

    const requestOpts: FetchJsonOptions = {
      signal: opts?.signal,
      headers: opts?.headers,
      timeoutMs: Math.max(1, remaining),
    };

    try {
      return await fetchJson<BacktestResponse>(
        baseUrl,
        "/backtest",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        },
        requestOpts,
      );
    } catch (err) {
      if (!isBacktestQueueBusy(err)) throw err;
      sawBusy = true;
      const retryAfterMs =
        typeof err.retryAfterMs === "number" && Number.isFinite(err.retryAfterMs) ? Math.max(0, err.retryAfterMs) : backoffMs;
      const delayMs = Math.min(retryAfterMs, remaining);
      await sleep(delayMs, opts?.signal);
      backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
    }
  }
}

async function runAsyncJob<T>(
  baseUrl: string,
  startPath: string,
  pollPath: string,
  params: ApiParams,
  opts?: AsyncJobOptions,
): Promise<T> {
  const startedAt = Date.now();
  const overallTimeoutMs = opts?.timeoutMs ?? 30_000;
  const perRequestTimeoutMs = Math.min(55_000, overallTimeoutMs);
  const notFoundGraceMs = Math.min(2 * 60_000, Math.max(10_000, Math.round(overallTimeoutMs * 0.5)));
  let lastTransientError: unknown = null;
  let notFoundSinceMs: number | null = null;
  const retryStart = opts?.retryStart ?? false;
  const maxStartRetries = opts?.maxStartRetries ?? 2;
  let startRetries = 0;

  let startBackoffMs = 750;
  let start: AsyncStartResponse;
  for (;;) {
    const elapsed = Date.now() - startedAt;
    const remaining = overallTimeoutMs - elapsed;
    if (remaining <= 0) {
      if (lastTransientError) throw new Error(describeAsyncTimeout(baseUrl, overallTimeoutMs, lastTransientError));
      throw timeoutError();
    }

    try {
        start = await fetchJson<AsyncStartResponse>(
          baseUrl,
          startPath,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
          },
        { signal: opts?.signal, headers: opts?.headers, timeoutMs: Math.min(remaining, perRequestTimeoutMs) },
      );
      break;
    } catch (err) {
      // 429 is safe to retry: the server didn't start the async job.
      if (err instanceof HttpError && err.status === 429) {
        lastTransientError = err;
        const retryAfterMs = typeof err.retryAfterMs === "number" && Number.isFinite(err.retryAfterMs) ? Math.max(0, err.retryAfterMs) : null;
        const delayMs = retryAfterMs == null ? startBackoffMs : Math.max(startBackoffMs, retryAfterMs);
        await sleep(Math.min(delayMs, remaining), opts?.signal);
        if (retryAfterMs == null) startBackoffMs = Math.min(5_000, Math.round(startBackoffMs * 1.4));
        continue;
      }
      if (retryStart && shouldRetryAsyncStart(err)) {
        lastTransientError = err;
        if (startRetries >= maxStartRetries) throw err;
        startRetries += 1;
        await sleep(Math.min(startBackoffMs, remaining), opts?.signal);
        startBackoffMs = Math.min(5_000, Math.round(startBackoffMs * 1.4));
        continue;
      }
      throw err;
    }
  }
  if (!start || typeof start !== "object" || !("jobId" in start) || typeof (start as { jobId?: unknown }).jobId !== "string") {
    throw new Error("Invalid async start response");
  }
  opts?.onJobId?.(start.jobId);

  let cancelSent = false;
  const cancel = async () => {
    if (cancelSent) return;
    cancelSent = true;
    const cancelUrl = `${pollPath}/${encodeURIComponent(start.jobId)}/cancel`;
    try {
      await fetchJson<{ status?: string }>(baseUrl, cancelUrl, { method: "POST" }, { headers: opts?.headers, timeoutMs: 10_000 });
    } catch {
      // Best-effort; ignore if the API doesn't support cancel or we can't reach it.
    }
  };

  let pollMethod: "POST" | "GET" = "POST";
  let backoffMs = 750;
  try {
    for (;;) {
      const elapsed = Date.now() - startedAt;
      const remaining = overallTimeoutMs - elapsed;
      if (remaining <= 0) {
        if (lastTransientError) throw new Error(describeAsyncTimeout(baseUrl, overallTimeoutMs, lastTransientError));
        throw timeoutError();
      }

      let status: AsyncPollResponse<T>;
      try {
        const pollUrl = `${pollPath}/${encodeURIComponent(start.jobId)}`;
        try {
          status = await fetchJson<AsyncPollResponse<T>>(
            baseUrl,
            pollUrl,
            { method: pollMethod },
            { signal: opts?.signal, headers: opts?.headers, timeoutMs: Math.min(remaining, perRequestTimeoutMs) },
          );
        } catch (err) {
          if (pollMethod === "POST" && shouldFallbackToGet(err)) {
            status = await fetchJson<AsyncPollResponse<T>>(
              baseUrl,
              pollUrl,
              { method: "GET" },
              { signal: opts?.signal, headers: opts?.headers, timeoutMs: Math.min(remaining, perRequestTimeoutMs) },
            );
            pollMethod = "GET";
          } else {
            throw err;
          }
        }
      } catch (err) {
        if (err instanceof HttpError && (err.status === 401 || err.status === 403)) throw err;
        if (err instanceof HttpError && err.status === 404) throw err;
        if (err instanceof HttpError && err.status === 429) {
          lastTransientError = err;
          const retryAfterMs = typeof err.retryAfterMs === "number" && Number.isFinite(err.retryAfterMs) ? Math.max(0, err.retryAfterMs) : 0;
          const delayMs = Math.min(Math.max(backoffMs, retryAfterMs), remaining);
          await sleep(delayMs, opts?.signal);
          backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
          continue;
        }
        if (isTimeoutError(err)) {
          lastTransientError = err;
          await sleep(Math.min(backoffMs, remaining), opts?.signal);
          backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
          continue;
        }
        if (err instanceof HttpError && err.status >= 500) {
          lastTransientError = err;
          await sleep(Math.min(backoffMs, remaining), opts?.signal);
          backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
          continue;
        }
        if (err instanceof TypeError && err.message.toLowerCase().includes("fetch")) {
          lastTransientError = err;
          await sleep(Math.min(backoffMs, remaining), opts?.signal);
          backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
          continue;
        }
        throw err;
      }

      lastTransientError = null;
      if (!status || typeof status !== "object" || !("status" in status) || typeof (status as { status?: unknown }).status !== "string") {
        throw new Error("Invalid async poll response");
      }
      if (status.status === "done") return status.result as T;
      if (status.status === "error") {
        const msg = status.error || "Async job failed";
        if (msg.trim().toLowerCase() === "not found") {
          lastTransientError = new Error("Async job not found");
          if (notFoundSinceMs == null) notFoundSinceMs = Date.now();
          if (Date.now() - notFoundSinceMs > notFoundGraceMs) {
            throw new Error(asyncJobNotFoundMessage());
          }
          await sleep(Math.min(backoffMs, remaining), opts?.signal);
          backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
          continue;
        }
        throw new Error(msg);
      }
      if (status.status !== "running") throw new Error(`Unexpected async status: ${String(status.status)}`);

      notFoundSinceMs = null;
      await sleep(Math.min(backoffMs, remaining), opts?.signal);
      backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
    }
  } catch (err) {
    if (isAbortError(err) || isTimeoutError(err)) await cancel();
    throw err;
  }
}

export async function health(baseUrl: string, opts?: FetchJsonOptions): Promise<HealthResponse> {
  const out = await fetchJson<{
    status: string;
    authRequired?: boolean;
    authOk?: boolean;
    computeLimits?: { maxBarsLstm: number; maxEpochs: number; maxHiddenSize: number };
    asyncJobs?: { maxRunning: number; ttlMs: number; persistence: boolean };
    cache?: { enabled: boolean; ttlMs: number; maxEntries: number };
  }>(baseUrl, "/health", { method: "GET" }, opts);
  if (out.status !== "ok") throw new Error("Unexpected /health response");
  return {
    status: "ok",
    authRequired: out.authRequired,
    authOk: out.authOk,
    computeLimits: out.computeLimits,
    asyncJobs: out.asyncJobs,
    cache: out.cache,
  };
}

export async function cacheStats(baseUrl: string, opts?: FetchJsonOptions): Promise<CacheStatsResponse> {
  return fetchJson<CacheStatsResponse>(baseUrl, "/cache", { method: "GET" }, opts);
}

export async function cacheClear(baseUrl: string, opts?: FetchJsonOptions): Promise<CacheClearResponse> {
  return fetchJson<CacheClearResponse>(baseUrl, "/cache/clear", { method: "POST" }, opts);
}

export async function signal(baseUrl: string, params: ApiParams, opts?: AsyncJobOptions): Promise<LatestSignal> {
  const asyncOpts = opts ? { ...opts, retryStart: true } : { retryStart: true };
  try {
    return await runAsyncJob<LatestSignal>(baseUrl, "/signal/async", "/signal/async", params, asyncOpts);
  } catch (err) {
    if (err instanceof HttpError && err.status === 404) {
      return fetchJson<LatestSignal>(
        baseUrl,
        "/signal",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        },
        opts,
      );
    }
    throw err;
  }
}

export async function backtest(baseUrl: string, params: ApiParams, opts?: AsyncJobOptions): Promise<BacktestResponse> {
  const asyncOpts = opts ? { ...opts, retryStart: true } : { retryStart: true };
  try {
    return await runAsyncJob<BacktestResponse>(baseUrl, "/backtest/async", "/backtest/async", params, asyncOpts);
  } catch (err) {
    if (err instanceof HttpError && err.status === 404) {
      return runSyncBacktestWithRetry(baseUrl, params, opts);
    }
    throw err;
  }
}

export async function trade(baseUrl: string, params: ApiParams, opts?: AsyncJobOptions): Promise<ApiTradeResponse> {
  try {
    return await runAsyncJob<ApiTradeResponse>(baseUrl, "/trade/async", "/trade/async", params, opts);
  } catch (err) {
    if (err instanceof HttpError && err.status === 404) {
      return fetchJson<ApiTradeResponse>(
        baseUrl,
        "/trade",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        },
        opts,
      );
    }
    throw err;
  }
}

export async function binanceKeysStatus(
  baseUrl: string,
  params: ApiParams,
  opts?: FetchJsonOptions,
): Promise<BinanceKeysStatus> {
  return fetchJson<BinanceKeysStatus>(
    baseUrl,
    "/binance/keys",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function coinbaseKeysStatus(
  baseUrl: string,
  params: ApiParams,
  opts?: FetchJsonOptions,
): Promise<CoinbaseKeysStatus> {
  return fetchJson<CoinbaseKeysStatus>(
    baseUrl,
    "/coinbase/keys",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

type BinanceListenKeyStartParams = Pick<ApiParams, "market" | "binanceTestnet" | "binanceApiKey" | "binanceApiSecret">;
type BinanceListenKeyActionParams = BinanceListenKeyStartParams & { listenKey: string };

export async function binanceListenKey(baseUrl: string, params: BinanceListenKeyStartParams, opts?: FetchJsonOptions): Promise<BinanceListenKeyResponse> {
  return fetchJson<BinanceListenKeyResponse>(
    baseUrl,
    "/binance/listenKey",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function binanceListenKeyKeepAlive(
  baseUrl: string,
  params: BinanceListenKeyActionParams,
  opts?: FetchJsonOptions,
): Promise<BinanceListenKeyKeepAliveResponse> {
  return fetchJson<BinanceListenKeyKeepAliveResponse>(
    baseUrl,
    "/binance/listenKey/keepAlive",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function binanceListenKeyClose(
  baseUrl: string,
  params: BinanceListenKeyActionParams,
  opts?: FetchJsonOptions,
): Promise<BinanceListenKeyKeepAliveResponse> {
  return fetchJson<BinanceListenKeyKeepAliveResponse>(
    baseUrl,
    "/binance/listenKey/close",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function binancePositions(
  baseUrl: string,
  params: ApiBinancePositionsRequest,
  opts?: FetchJsonOptions,
): Promise<ApiBinancePositionsResponse> {
  return fetchJson<ApiBinancePositionsResponse>(
    baseUrl,
    "/binance/positions",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function binanceTrades(
  baseUrl: string,
  params: ApiBinanceTradesRequest,
  opts?: FetchJsonOptions,
): Promise<ApiBinanceTradesResponse> {
  return fetchJson<ApiBinanceTradesResponse>(
    baseUrl,
    "/binance/trades",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function botStart(baseUrl: string, params: ApiParams, opts?: FetchJsonOptions): Promise<BotStatus> {
  return fetchJson<BotStatus>(
    baseUrl,
    "/bot/start",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function botStop(baseUrl: string, opts?: FetchJsonOptions, symbol?: string): Promise<BotStatus> {
  const path = symbol ? `/bot/stop?symbol=${encodeURIComponent(symbol)}` : "/bot/stop";
  return fetchJson<BotStatus>(baseUrl, path, { method: "POST" }, opts);
}

export async function botStatus(baseUrl: string, opts?: FetchJsonOptions, tail?: number, symbol?: string): Promise<BotStatus> {
  const tailSafe = typeof tail === "number" && Number.isFinite(tail) ? Math.trunc(tail) : 0;
  const query = new URLSearchParams();
  if (tailSafe > 0) query.set("tail", String(tailSafe));
  if (symbol) query.set("symbol", symbol);
  const path = query.size > 0 ? `/bot/status?${query.toString()}` : "/bot/status";
  return fetchJson<BotStatus>(baseUrl, path, { method: "GET" }, opts);
}

export async function ops(
  baseUrl: string,
  params?: { kind?: string; limit?: number; since?: number },
  opts?: FetchJsonOptions,
): Promise<OpsResponse> {
  const query = new URLSearchParams();
  if (params?.kind) query.set("kind", params.kind);
  if (typeof params?.limit === "number" && Number.isFinite(params.limit)) query.set("limit", String(Math.trunc(params.limit)));
  if (typeof params?.since === "number" && Number.isFinite(params.since)) query.set("since", String(Math.trunc(params.since)));
  const path = query.size > 0 ? `/ops?${query.toString()}` : "/ops";
  return fetchJson<OpsResponse>(baseUrl, path, { method: "GET" }, opts);
}

export async function opsPerformance(
  baseUrl: string,
  params?: { commitLimit?: number; comboLimit?: number; comboScope?: string; comboOrder?: string },
  opts?: FetchJsonOptions,
): Promise<OpsPerformanceResponse> {
  const query = new URLSearchParams();
  if (typeof params?.commitLimit === "number" && Number.isFinite(params.commitLimit)) {
    query.set("commitLimit", String(Math.trunc(params.commitLimit)));
  }
  if (typeof params?.comboLimit === "number" && Number.isFinite(params.comboLimit)) {
    query.set("comboLimit", String(Math.trunc(params.comboLimit)));
  }
  if (params?.comboScope) query.set("comboScope", params.comboScope);
  if (params?.comboOrder) query.set("comboOrder", params.comboOrder);
  const path = query.size > 0 ? `/ops/performance?${query.toString()}` : "/ops/performance";
  return fetchJson<OpsPerformanceResponse>(baseUrl, path, { method: "GET" }, opts);
}

export async function optimizerCombos(baseUrl: string, opts?: FetchJsonOptions): Promise<unknown> {
  return fetchJson<unknown>(baseUrl, "/optimizer/combos", { method: "GET" }, opts);
}

export async function optimizerRun(
  baseUrl: string,
  params: OptimizerRunRequest,
  opts?: FetchJsonOptions,
): Promise<OptimizerRunResponse> {
  return fetchJson<OptimizerRunResponse>(
    baseUrl,
    "/optimizer/run",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function stateSyncExport(baseUrl: string, opts?: FetchJsonOptions): Promise<StateSyncPayload> {
  const mergedOpts = { ...opts, allowFallback: false };
  return fetchJson<StateSyncPayload>(baseUrl, "/state/sync", { method: "GET" }, mergedOpts);
}

export async function stateSyncImport(
  baseUrl: string,
  payload: StateSyncPayload,
  opts?: FetchJsonOptions,
): Promise<StateSyncImportResponse> {
  const mergedOpts = { ...opts, allowFallback: false };
  return fetchJson<StateSyncImportResponse>(
    baseUrl,
    "/state/sync",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    mergedOpts,
  );
}
