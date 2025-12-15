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

function withTimeout(externalSignal: AbortSignal | undefined, timeoutMs: number) {
  const controller = new AbortController();

  if (externalSignal) {
    if (externalSignal.aborted) controller.abort(externalSignal.reason);
    else {
      externalSignal.addEventListener(
        "abort",
        () => controller.abort(externalSignal.reason),
        { once: true },
      );
    }
  }

  const timer = window.setTimeout(() => controller.abort(new DOMException("Timeout", "TimeoutError")), timeoutMs);
  return { signal: controller.signal, cleanup: () => window.clearTimeout(timer) };
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
    const res = await fetch(url, { ...init, headers: mergeHeaders(init.headers, opts?.headers), signal });
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

export async function health(baseUrl: string, opts?: FetchJsonOptions): Promise<"ok"> {
  const out = await fetchJson<{ status: string }>(baseUrl, "/health", { method: "GET" }, opts);
  if (out.status !== "ok") throw new Error("Unexpected /health response");
  return "ok";
}

export async function signal(baseUrl: string, params: ApiParams, opts?: FetchJsonOptions): Promise<LatestSignal> {
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

export async function backtest(baseUrl: string, params: ApiParams, opts?: FetchJsonOptions): Promise<BacktestResponse> {
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

export async function trade(baseUrl: string, params: ApiParams, opts?: FetchJsonOptions): Promise<ApiTradeResponse> {
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
