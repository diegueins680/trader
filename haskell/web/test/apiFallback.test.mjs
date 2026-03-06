import assert from "node:assert/strict";
import { test } from "node:test";

const apiBundleUrl = new URL("../.tmp/web-tests/api.js", import.meta.url);
const MAX_TIMER_DELAY_MS = 2_147_483_647;

function jsonResponse(status, body) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function loadApiModule(config, fetchImpl) {
  globalThis.window = {
    location: {
      origin: "https://ui.example.com",
      hostname: "ui.example.com",
    },
    setTimeout,
    clearTimeout,
    __TRADER_CONFIG__: config,
  };
  globalThis.fetch = fetchImpl;
  const modUrl = new URL(apiBundleUrl);
  modUrl.searchParams.set("cachebust", `${Date.now()}-${Math.random()}`);
  return import(modUrl.href);
}

function createStorage(seed = {}) {
  const map = new Map(Object.entries(seed));
  return {
    getItem(key) {
      return map.has(key) ? map.get(key) : null;
    },
    setItem(key, value) {
      map.set(key, String(value));
    },
    removeItem(key) {
      map.delete(key);
    },
  };
}

function restoreGlobal(name, priorValue) {
  if (priorValue === undefined) {
    delete globalThis[name];
    return;
  }
  globalThis[name] = priorValue;
}

async function withApiModule(config, fetchImpl, run, options = {}) {
  const priorWindow = globalThis.window;
  const priorFetch = globalThis.fetch;
  const priorLocalStorage = globalThis.localStorage;
  if (Object.prototype.hasOwnProperty.call(options, "localStorage")) {
    globalThis.localStorage = options.localStorage;
  }
  try {
    const api = await loadApiModule(config, fetchImpl);
    return await run(api);
  } finally {
    restoreGlobal("window", priorWindow);
    restoreGlobal("fetch", priorFetch);
    restoreGlobal("localStorage", priorLocalStorage);
  }
}

async function withApiModuleNoWindow(fetchImpl, run) {
  const priorWindow = globalThis.window;
  const priorFetch = globalThis.fetch;
  try {
    delete globalThis.window;
    globalThis.fetch = fetchImpl;
    const modUrl = new URL(apiBundleUrl);
    modUrl.searchParams.set("cachebust", `${Date.now()}-${Math.random()}`);
    const api = await import(modUrl.href);
    return await run(api);
  } finally {
    restoreGlobal("window", priorWindow);
    restoreGlobal("fetch", priorFetch);
  }
}

test("api fallback does not use 401 for explicit direct hosts", async () => {
  const calls = [];
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "/api",
      apiToken: "",
    },
    async (url) => {
      calls.push(String(url));
      return jsonResponse(401, { error: "Unauthorized" });
    },
    async (api) => {
      await assert.rejects(
        () => api.health("https://api.example.com", { timeoutMs: 5_000 }),
        (err) => err?.name === "HttpError" && err.status === 401,
      );
    },
  );
  assert.deepEqual(calls, ["https://api.example.com/health"]);
});

test("api client works without window global", async () => {
  const calls = [];
  await withApiModuleNoWindow(
    async (url) => {
      calls.push(String(url));
      return jsonResponse(200, { status: "ok" });
    },
    async (api) => {
      const out = await api.health("https://api.example.com", { timeoutMs: 5_000 });
      assert.equal(out.status, "ok");
    },
  );
  assert.deepEqual(calls, ["https://api.example.com/health"]);
});

test("api fallback uses 401 for inferred direct hosts with /api fallback", async () => {
  const calls = [];
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: true,
      apiFallbackUrl: "/api",
      apiToken: "",
    },
    async (url) => {
      const href = String(url);
      calls.push(href);
      if (href === "https://api.example.com/health") {
        return jsonResponse(401, { error: "Unauthorized" });
      }
      return jsonResponse(200, { status: "ok" });
    },
    async (api) => {
      const out = await api.health("https://api.example.com", { timeoutMs: 5_000 });
      assert.equal(out.status, "ok");
    },
  );
  assert.deepEqual(calls, ["https://api.example.com/health", "/api/health"]);
});

test("api fallback uses timeout for inferred direct hosts with /api fallback", async () => {
  const calls = [];
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: true,
      apiFallbackUrl: "/api",
      apiToken: "",
    },
    async (url) => {
      const href = String(url);
      calls.push(href);
      if (href === "https://api.example.com/health") {
        throw new DOMException("Timeout", "TimeoutError");
      }
      return jsonResponse(200, { status: "ok" });
    },
    async (api) => {
      const out = await api.health("https://api.example.com", { timeoutMs: 5_000 });
      assert.equal(out.status, "ok");
    },
  );
  assert.deepEqual(calls, ["https://api.example.com/health", "/api/health"]);
});

test("api fallback ignores legacy preferred fallback cache entries", async () => {
  const calls = [];
  const legacyStorage = createStorage({
    trader_api_fallback_v3: JSON.stringify({
      savedAtMs: Date.now(),
      blocked: [],
      preferred: { "https://api.example.com": "/api" },
    }),
  });
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "/api",
      apiToken: "",
    },
    async (url) => {
      calls.push(String(url));
      return jsonResponse(200, { status: "ok" });
    },
    async (api) => {
      const out = await api.health("https://api.example.com", { timeoutMs: 5_000 });
      assert.equal(out.status, "ok");
    },
    { localStorage: legacyStorage },
  );
  assert.deepEqual(calls, ["https://api.example.com/health"]);
});

test("async polling retries transient network type errors without fetch wording", async () => {
  const calls = [];
  let pollCalls = 0;
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      const method = String(init.method || "GET").toUpperCase();
      const href = String(url);
      calls.push(`${method} ${href}`);
      if (method === "POST" && href === "https://api.example.com/signal/async") {
        return jsonResponse(200, { jobId: "job-1" });
      }
      if (method === "POST" && href === "https://api.example.com/signal/async/job-1") {
        pollCalls += 1;
        if (pollCalls === 1) throw new TypeError("Load failed");
        return jsonResponse(200, { status: "done", result: { signal: "UP" } });
      }
      throw new Error(`unexpected request: ${method} ${href}`);
    },
    async (api) => {
      const out = await api.signal("https://api.example.com", { symbol: "BTCUSDT" }, { timeoutMs: 5_000 });
      assert.equal(out.signal, "UP");
    },
  );
  assert.deepEqual(calls, [
    "POST https://api.example.com/signal/async",
    "POST https://api.example.com/signal/async/job-1",
    "POST https://api.example.com/signal/async/job-1",
  ]);
});

test("botStart retries transient 502 responses before failing", async () => {
  const calls = [];
  let starts = 0;
  const retryEvents = [];
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      const method = String(init.method || "GET").toUpperCase();
      const href = String(url);
      calls.push(`${method} ${href}`);
      if (method === "POST" && href === "https://api.example.com/bot/start") {
        starts += 1;
        if (starts < 3) return jsonResponse(502, { error: "Bad Gateway" });
        return jsonResponse(202, { starting: true, symbol: "BTCUSDT" });
      }
      throw new Error(`unexpected request: ${method} ${href}`);
    },
    async (api) => {
      const out = await api.botStart(
        "https://api.example.com",
        { tenantKey: "tenant", binanceSymbol: "BTCUSDT" },
        {
          timeoutMs: 8_000,
          onTransientRetry: (info) => {
            const status =
              info.error && typeof info.error === "object" && "status" in info.error ? Number(info.error.status) : null;
            retryEvents.push({ attempt: info.attempt, maxRetries: info.maxRetries, status, delayMs: info.delayMs });
          },
        },
      );
      assert.equal(out.starting, true);
      assert.equal(out.symbol, "BTCUSDT");
    },
  );
  assert.equal(starts, 3);
  assert.deepEqual(calls, [
    "POST https://api.example.com/bot/start",
    "POST https://api.example.com/bot/start",
    "POST https://api.example.com/bot/start",
  ]);
  assert.equal(retryEvents.length, 2);
  assert.deepEqual(
    retryEvents.map((event) => ({ attempt: event.attempt, maxRetries: event.maxRetries, status: event.status })),
    [
      { attempt: 1, maxRetries: 2, status: 502 },
      { attempt: 2, maxRetries: 2, status: 502 },
    ],
  );
  assert.equal(retryEvents.every((event) => event.delayMs >= 750), true);
});

test("botStart does not retry non-transient validation errors", async () => {
  const calls = [];
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      const method = String(init.method || "GET").toUpperCase();
      const href = String(url);
      calls.push(`${method} ${href}`);
      if (method === "POST" && href === "https://api.example.com/bot/start") {
        return jsonResponse(400, { error: "bot/start requires tenantKey or API keys." });
      }
      throw new Error(`unexpected request: ${method} ${href}`);
    },
    async (api) => {
      await assert.rejects(
        () =>
          api.botStart(
            "https://api.example.com",
            { binanceSymbol: "BTCUSDT" },
            { timeoutMs: 5_000 },
          ),
        (err) => err?.name === "HttpError" && err.status === 400,
      );
    },
  );
  assert.deepEqual(calls, ["POST https://api.example.com/bot/start"]);
});

test("api fallback ignores future-dated fallback cache entries", async () => {
  const calls = [];
  const futureStorage = createStorage({
    trader_api_fallback_v4: JSON.stringify({
      savedAtMs: Date.now() + 24 * 60 * 60 * 1_000,
      blocked: [],
      preferred: { "https://api.example.com": "/api" },
    }),
  });
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "/api",
      apiToken: "",
    },
    async (url) => {
      calls.push(String(url));
      return jsonResponse(200, { status: "ok" });
    },
    async (api) => {
      const out = await api.health("https://api.example.com", { timeoutMs: 5_000 });
      assert.equal(out.status, "ok");
    },
    { localStorage: futureStorage },
  );
  assert.deepEqual(calls, ["https://api.example.com/health"]);
});

test("health clamps huge Retry-After header to safe timer delay", async () => {
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () => {
      const res = jsonResponse(503, { error: "Service unavailable" });
      res.headers.set("retry-after", "999999999999999999999999999999");
      return res;
    },
    async (api) => {
      await assert.rejects(
        () => api.health("https://api.example.com", { timeoutMs: 5_000 }),
        (err) => err?.name === "HttpError" && err.status === 503 && err.retryAfterMs === MAX_TIMER_DELAY_MS,
      );
    },
  );
});

test("api fallback allows inferred /api primary to fail over to cross-origin fallback", async () => {
  const calls = [];
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: true,
      apiFallbackUrl: "https://api.example.com",
      apiToken: "",
    },
    async (url) => {
      const href = String(url);
      calls.push(href);
      if (href === "/api/health") {
        return jsonResponse(502, { error: "Bad Gateway" });
      }
      if (href === "https://api.example.com/health") {
        return jsonResponse(200, { status: "ok" });
      }
      throw new Error(`unexpected request: ${href}`);
    },
    async (api) => {
      const out = await api.health("/api", { timeoutMs: 5_000 });
      assert.equal(out.status, "ok");
    },
  );
  assert.deepEqual(calls, ["/api/health", "https://api.example.com/health"]);
});

test("api fallback allows inferred /api primary timeout failover to cross-origin fallback", async () => {
  const calls = [];
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: true,
      apiFallbackUrl: "https://api.example.com",
      apiToken: "",
    },
    async (url) => {
      const href = String(url);
      calls.push(href);
      if (href === "/api/health") {
        throw new DOMException("Timeout", "TimeoutError");
      }
      if (href === "https://api.example.com/health") {
        return jsonResponse(200, { status: "ok" });
      }
      throw new Error(`unexpected request: ${href}`);
    },
    async (api) => {
      const out = await api.health("/api", { timeoutMs: 5_000 });
      assert.equal(out.status, "ok");
    },
  );
  assert.deepEqual(calls, ["/api/health", "https://api.example.com/health"]);
});

test("api fallback skips inferred /api cross-origin failover for non-GET requests", async () => {
  const calls = [];
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: true,
      apiFallbackUrl: "https://api.example.com",
      apiToken: "",
    },
    async (url) => {
      const href = String(url);
      calls.push(href);
      if (href === "/api/binance/listenKey") {
        return jsonResponse(502, { error: "Bad Gateway" });
      }
      throw new Error(`unexpected request: ${href}`);
    },
    async (api) => {
      await assert.rejects(
        () => api.binanceListenKey("/api", { tenantKey: "tenant" }, { timeoutMs: 5_000 }),
        (err) => err?.name === "HttpError" && err.status === 502,
      );
    },
  );
  assert.deepEqual(calls, ["/api/binance/listenKey"]);
});

test("api client forwards tenant key as X-Tenant-Key from JSON body params", async () => {
  let tenantHeader = null;
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      const href = String(url);
      if (href !== "/api/bot/start") throw new Error(`unexpected request: ${href}`);
      tenantHeader = new Headers(init.headers).get("X-Tenant-Key");
      return jsonResponse(202, { starting: true, symbol: "BTCUSDT" });
    },
    async (api) => {
      const out = await api.botStart(
        "/api",
        { tenantKey: "tenant-body", binanceSymbol: "BTCUSDT" },
        { timeoutMs: 5_000 },
      );
      assert.equal(out.starting, true);
    },
  );
  assert.equal(tenantHeader, "tenant-body");
});

test("api client forwards tenant key as X-Tenant-Key from query params", async () => {
  let tenantHeader = null;
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      const href = String(url);
      if (!href.startsWith("/api/bot/status?")) throw new Error(`unexpected request: ${href}`);
      tenantHeader = new Headers(init.headers).get("X-Tenant-Key");
      return jsonResponse(200, { running: false });
    },
    async (api) => {
      const out = await api.botStatus("/api", { timeoutMs: 5_000 }, 100, undefined, "tenant-query");
      assert.equal(out.running, false);
    },
  );
  assert.equal(tenantHeader, "tenant-query");
});

test("api client skips X-Tenant-Key header for cross-origin direct hosts", async () => {
  let tenantHeader = null;
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: true,
      apiFallbackUrl: "/api",
      apiToken: "",
    },
    async (url, init = {}) => {
      const href = String(url);
      if (!href.startsWith("https://api.example.com/bot/status?")) throw new Error(`unexpected request: ${href}`);
      tenantHeader = new Headers(init.headers).get("X-Tenant-Key");
      return jsonResponse(200, { running: false });
    },
    async (api) => {
      const out = await api.botStatus("https://api.example.com", { timeoutMs: 5_000 }, 100, undefined, "tenant-query");
      assert.equal(out.running, false);
    },
  );
  assert.equal(tenantHeader, null);
});

test("health preserves version and commit metadata", async () => {
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () =>
      jsonResponse(200, {
        status: "ok",
        version: "1.2.3",
        commit: "abcdef123456",
        authRequired: false,
        authOk: true,
        computeLimits: { maxBarsLstm: 10, maxEpochs: 20, maxHiddenSize: 30 },
      }),
    async (api) => {
      const out = await api.health("https://api.example.com", { timeoutMs: 5_000 });
      assert.equal(out.version, "1.2.3");
      assert.equal(out.commit, "abcdef123456");
    },
  );
});
