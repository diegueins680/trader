import type {
  ApiError,
  ApiParams,
  ApiTradeResponse,
  BacktestResponse,
  BinanceKeysStatus,
  BinanceListenKeyKeepAliveResponse,
  BinanceListenKeyResponse,
  BotStatus,
  LatestSignal,
} from "./types";

export class HttpError extends Error {
  readonly status: number;
  readonly payload?: unknown;

  constructor(status: number, message: string, payload?: unknown) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.payload = payload;
  }
}

type FetchJsonOptions = {
  signal?: AbortSignal;
  timeoutMs?: number;
  headers?: Record<string, string>;
};

type AsyncStartResponse = { jobId: string };
type AsyncPollResponse<T> = { status: "running" | "done" | "error"; result?: T; error?: string };
export type HealthResponse = {
  status: "ok";
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
type AsyncJobOptions = FetchJsonOptions & { onJobId?: (jobId: string) => void };

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

function mergeHeaders(base: HeadersInit | undefined, extra: Record<string, string> | undefined): HeadersInit | undefined {
  if (!extra || Object.keys(extra).length === 0) return base;
  const merged = new Headers(base);
  for (const [key, value] of Object.entries(extra)) merged.set(key, value);
  return merged;
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

async function readJsonOrText(res: Response): Promise<unknown> {
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

async function fetchJson<T>(baseUrl: string, path: string, init: RequestInit, opts?: FetchJsonOptions): Promise<T> {
  const timeoutMs = opts?.timeoutMs ?? 30_000;
  const { signal, cleanup } = withTimeout(opts?.signal, timeoutMs);
  try {
    const url = resolveUrl(baseUrl, path);
    const res = await fetch(url, {
      ...init,
      cache: init.cache ?? "no-store",
      headers: mergeHeaders(init.headers, opts?.headers),
      signal,
    });
    const payload = await readJsonOrText(res);
    if (!res.ok) {
      const message =
        typeof payload === "object" && payload && "error" in payload
          ? String((payload as ApiError).error)
          : `${res.status} ${res.statusText}`;
      throw new HttpError(res.status, message, payload);
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

function timeoutError(): DOMException {
  return new DOMException("Timeout", "TimeoutError");
}

function isTimeoutError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "TimeoutError";
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
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
    ? " Check your CloudFront `/api/*` proxy (or set the UI “API base URL” to your API host)."
    : " Check API connectivity and try again.";
  return `Async request timed out after ${seconds}s while retrying after errors (last error: ${last}).${hint}`;
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
  let sawJob = false;
  let notFoundSinceMs: number | null = null;

  const start = await fetchJson<AsyncStartResponse>(
    baseUrl,
    startPath,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    { signal: opts?.signal, headers: opts?.headers, timeoutMs: perRequestTimeoutMs },
  );
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
            { method: "POST" },
            { signal: opts?.signal, headers: opts?.headers, timeoutMs: Math.min(remaining, perRequestTimeoutMs) },
          );
        } catch (err) {
          if (err instanceof HttpError && err.status === 405) {
            status = await fetchJson<AsyncPollResponse<T>>(
              baseUrl,
              pollUrl,
              { method: "GET" },
              { signal: opts?.signal, headers: opts?.headers, timeoutMs: Math.min(remaining, perRequestTimeoutMs) },
            );
          } else {
            throw err;
          }
        }
      } catch (err) {
        if (err instanceof HttpError && (err.status === 401 || err.status === 403)) throw err;
        if (err instanceof HttpError && err.status === 404) throw err;
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
          if (!sawJob) {
            if (notFoundSinceMs == null) notFoundSinceMs = Date.now();
            if (Date.now() - notFoundSinceMs > notFoundGraceMs) {
              throw new Error(
                "Async job not found (server restarted or behind a non-sticky load balancer). Please retry; for multi-instance deployments, enable shared async job storage (TRADER_API_ASYNC_DIR) or run single-instance.",
              );
            }
          }
          await sleep(Math.min(backoffMs, remaining), opts?.signal);
          backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
          continue;
        }
        throw new Error(msg);
      }
      if (status.status !== "running") throw new Error(`Unexpected async status: ${String(status.status)}`);

      sawJob = true;
      notFoundSinceMs = null;
      await sleep(Math.min(backoffMs, remaining), opts?.signal);
      backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
    }
  } catch (err) {
    if (isAbortError(err)) await cancel();
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
  try {
    return await runAsyncJob<LatestSignal>(baseUrl, "/signal/async", "/signal/async", params, opts);
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
  try {
    return await runAsyncJob<BacktestResponse>(baseUrl, "/backtest/async", "/backtest/async", params, opts);
  } catch (err) {
    if (err instanceof HttpError && err.status === 404) {
      return fetchJson<BacktestResponse>(
        baseUrl,
        "/backtest",
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

export async function botStop(baseUrl: string, opts?: FetchJsonOptions): Promise<BotStatus> {
  return fetchJson<BotStatus>(baseUrl, "/bot/stop", { method: "POST" }, opts);
}

export async function botStatus(baseUrl: string, opts?: FetchJsonOptions, tail?: number): Promise<BotStatus> {
  const tailSafe = typeof tail === "number" && Number.isFinite(tail) ? Math.trunc(tail) : 0;
  const path = tailSafe > 0 ? `/bot/status?tail=${tailSafe}` : "/bot/status";
  return fetchJson<BotStatus>(baseUrl, path, { method: "GET" }, opts);
}
