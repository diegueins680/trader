import type { ApiError, ApiParams, ApiTradeResponse, BacktestResponse, BinanceKeysStatus, BotStatus, LatestSignal } from "./types";

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
export type HealthResponse = { status: "ok"; authRequired?: boolean; authOk?: boolean };

function resolveUrl(baseUrl: string, path: string): string {
  const base = baseUrl.trim().replace(/\/+$/, "");
  const p = path.startsWith("/") ? path : `/${path}`;

  if (/^https?:\/\//.test(base)) {
    const url = new URL(base);
    const basePath = url.pathname.replace(/\/+$/, "");
    url.pathname = `${basePath}${p}`.replace(/\/{2,}/g, "/") || "/";
    url.search = "";
    url.hash = "";
    return url.toString();
  }

  const rel = base.startsWith("/") ? base : `/${base}`;
  return `${rel}${p}`;
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

async function runAsyncJob<T>(
  baseUrl: string,
  startPath: string,
  pollPath: string,
  params: ApiParams,
  opts?: FetchJsonOptions,
): Promise<T> {
  const startedAt = Date.now();
  const overallTimeoutMs = opts?.timeoutMs ?? 30_000;
  const perRequestTimeoutMs = Math.min(55_000, overallTimeoutMs);

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

  let backoffMs = 750;
  for (;;) {
    const elapsed = Date.now() - startedAt;
    const remaining = overallTimeoutMs - elapsed;
    if (remaining <= 0) throw timeoutError();

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
        await sleep(Math.min(backoffMs, remaining), opts?.signal);
        backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
        continue;
      }
      if (err instanceof HttpError && err.status >= 500) {
        await sleep(Math.min(backoffMs, remaining), opts?.signal);
        backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
        continue;
      }
      if (err instanceof TypeError && err.message.toLowerCase().includes("fetch")) {
        await sleep(Math.min(backoffMs, remaining), opts?.signal);
        backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
        continue;
      }
      throw err;
    }

    if (!status || typeof status !== "object" || !("status" in status) || typeof (status as { status?: unknown }).status !== "string") {
      throw new Error("Invalid async poll response");
    }
    if (status.status === "done") return status.result as T;
    if (status.status === "error") throw new Error(status.error || "Async job failed");
    if (status.status !== "running") throw new Error(`Unexpected async status: ${String(status.status)}`);

    await sleep(Math.min(backoffMs, remaining), opts?.signal);
    backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
  }
}

export async function health(baseUrl: string, opts?: FetchJsonOptions): Promise<HealthResponse> {
  const out = await fetchJson<{ status: string; authRequired?: boolean; authOk?: boolean }>(baseUrl, "/health", { method: "GET" }, opts);
  if (out.status !== "ok") throw new Error("Unexpected /health response");
  return { status: "ok", authRequired: out.authRequired, authOk: out.authOk };
}

export async function signal(baseUrl: string, params: ApiParams, opts?: FetchJsonOptions): Promise<LatestSignal> {
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

export async function backtest(baseUrl: string, params: ApiParams, opts?: FetchJsonOptions): Promise<BacktestResponse> {
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

export async function trade(baseUrl: string, params: ApiParams, opts?: FetchJsonOptions): Promise<ApiTradeResponse> {
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

export async function botStatus(baseUrl: string, opts?: FetchJsonOptions): Promise<BotStatus> {
  return fetchJson<BotStatus>(baseUrl, "/bot/status", { method: "GET" }, opts);
}
