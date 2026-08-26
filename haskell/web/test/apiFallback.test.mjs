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

function validRunningBotStatus(overrides = {}) {
  return {
    running: true,
    live: false,
    tenantKey: "tenant",
    symbol: "BTCUSDT",
    interval: "3m",
    market: "futures",
    method: "blend",
    threshold: 0.01,
    settings: {
      pollSeconds: 5,
      onlineEpochs: 1,
      trainBars: 100,
      maxPoints: 500,
      tradeEnabled: false,
    },
    halted: false,
    peakEquity: 1,
    dayStartEquity: 1,
    consecutiveOrderErrors: 0,
    startIndex: 0,
    startedAtMs: 1,
    updatedAtMs: 2,
    prices: [100],
    openTimes: [1],
    kalmanPredNext: [null],
    lstmPredNext: [null],
    equityCurve: [1],
    positions: [0],
    operations: [],
    orders: [],
    trades: [],
    ...overrides,
  };
}

function validStartingBotStatus(overrides = {}) {
  return {
    running: false,
    starting: true,
    tenantKey: "tenant",
    symbol: "BTCUSDT",
    interval: "3m",
    market: "futures",
    method: "blend",
    threshold: 0.01,
    openThreshold: 0.01,
    closeThreshold: 0.01,
    startedAtMs: 1,
    ...overrides,
  };
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

test("api client preserves relative base pathname and query for health requests", async () => {
  const calls = [];
  let tenantHeader = null;
  await withApiModule(
    {
      apiBaseUrl: "/api/base?tenantKey=demo&mode=proxy",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      calls.push(String(url));
      tenantHeader = new Headers(init.headers).get("X-Tenant-Key");
      return jsonResponse(200, { status: "ok" });
    },
    async (api) => {
      const out = await api.health("/api/base?tenantKey=demo&mode=proxy", { timeoutMs: 5_000 });
      assert.equal(out.status, "ok");
    },
  );
  assert.equal(tenantHeader, "demo");
  assert.deepEqual(calls, ["/api/base/health?mode=proxy"]);
});

test("api client keeps base and request query params for relative bot URLs", async () => {
  const calls = [];
  let tenantHeader = null;
  await withApiModule(
    {
      apiBaseUrl: "/api/base?tenantKey=demo&mode=proxy",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      calls.push(String(url));
      tenantHeader = new Headers(init.headers).get("X-Tenant-Key");
      return jsonResponse(200, { running: false });
    },
    async (api) => {
      const out = await api.botStatus(
        "/api/base?tenantKey=demo&mode=proxy",
        { timeoutMs: 5_000 },
        100,
        undefined,
        "tenant-request",
      );
      assert.equal(out.running, false);
    },
  );
  assert.equal(tenantHeader, "tenant-request");
  assert.deepEqual(calls, ["/api/base/bot/status?mode=proxy&tail=100"]);
});

test("bot status decoder accepts a valid running safety envelope and is shared by stop", async () => {
  let calls = 0;
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () => {
      calls += 1;
      return calls === 1
        ? jsonResponse(
            200,
            validRunningBotStatus({
              portfolioSelector: {
                mode: "shadow",
                selectionValid: true,
                selection: {
                  generatedAtMs: 1,
                  validUntilMs: 2,
                  evidenceStartMs: 1,
                  evidenceEndMs: 2,
                  members: [{ uuid: "combo", symbol: "BTCUSDT", weight: 0.25 }],
                  metrics: {
                    annualizedReturnP10: 0.12,
                    annualizedReturnP50: 0.2,
                    annualizedReturnP90: 0.3,
                    maxDrawdownP95: 0.08,
                    averageCorrelation: 0,
                    switchingCost: 0,
                    pairedOutperformanceProbability: 0.95,
                  },
                },
              },
            }),
          )
        : jsonResponse(200, { running: 0 });
    },
    async (api) => {
      const status = await api.botStatus("/api", { timeoutMs: 5_000 });
      assert.equal(status.running, true);
      assert.equal(status.live, false);
      assert.deepEqual(status.positions, [0]);
      assert.equal(status.portfolioSelector.mode, "shadow");
      assert.equal(status.portfolioSelector.selection.members[0].weight, 0.25);
      await assert.rejects(
        () => api.botStop("/api", { timeoutMs: 5_000 }),
        (error) => error instanceof api.InvalidApiResponseError && error.endpoint === "/bot/stop",
      );
    },
  );
  assert.equal(calls, 2);
});

test("bot status decoder accepts every backend-only canonical method", async () => {
  const methods = ["meta_hedge_blend", "sma_cross", "sma_cross_regime"];
  for (const method of methods) {
    await withApiModule(
      {
        apiBaseUrl: "/api",
        apiBaseUrlInferred: false,
        apiFallbackUrl: "",
        apiToken: "",
      },
      async () => jsonResponse(200, validRunningBotStatus({ method })),
      async (api) => {
        const status = await api.botStatus("/api", { timeoutMs: 5_000 });
        assert.equal(status.running, true);
        assert.equal(status.method, method);
      },
    );
  }
});

test("bot status decoder rejects non-boolean running state before returning it to the UI", async () => {
  let calls = 0;
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () => {
      calls += 1;
      return jsonResponse(200, { running: "false" });
    },
    async (api) => {
      await assert.rejects(
        () => api.botStatus("/api", { timeoutMs: 5_000 }),
        (error) =>
          error instanceof api.InvalidApiResponseError &&
          error.endpoint === "/bot/status" &&
          error.field === "status.running must be a boolean",
      );
    },
  );
  assert.equal(calls, 1);
});

test("bot status decoder rejects malformed position evidence and incoherent multi-bot summaries", async () => {
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url) => {
      if (String(url).includes("symbol=BAD_POSITION_DOMAIN")) {
        return jsonResponse(200, validRunningBotStatus({ symbol: "BAD_POSITION_DOMAIN", positions: [2] }));
      }
      if (String(url).includes("symbol=BAD_POSITION")) {
        return jsonResponse(200, validRunningBotStatus({ symbol: "BAD_POSITION", positions: [0.5] }));
      }
      return jsonResponse(200, {
        running: true,
        multi: true,
        bots: [{ running: false }],
      });
    },
    async (api) => {
      await assert.rejects(
        () => api.botStatus("/api", { timeoutMs: 5_000 }, undefined, "BAD_POSITION"),
        (error) =>
          error instanceof api.InvalidApiResponseError && error.field === "status.positions[0] must be a safe integer",
      );
      await assert.rejects(
        () => api.botStatus("/api", { timeoutMs: 5_000 }, undefined, "BAD_POSITION_DOMAIN"),
        (error) =>
          error instanceof api.InvalidApiResponseError && error.field === "status.positions[0] must be -1, 0, or 1",
      );
      await assert.rejects(
        () => api.botStatus("/api", { timeoutMs: 5_000 }),
        (error) =>
          error instanceof api.InvalidApiResponseError && error.field === "status.running disagrees with status.bots",
      );
    },
  );
});

test("bot status decoder requires non-empty aligned core running series", async () => {
  const responses = [
    validRunningBotStatus({ prices: [] }),
    validRunningBotStatus({ positions: [] }),
  ];
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () => jsonResponse(200, responses.shift()),
    async (api) => {
      await assert.rejects(
        () => api.botStatus("/api", { timeoutMs: 5_000 }),
        (error) =>
          error instanceof api.InvalidApiResponseError && error.field === "status.prices must not be empty",
      );
      await assert.rejects(
        () => api.botStatus("/api", { timeoutMs: 5_000 }),
        (error) =>
          error instanceof api.InvalidApiResponseError &&
          error.field === "status.positions must align with status.prices",
      );
    },
  );
});

test("bot lifecycle decoder requires and binds tenant identity on running, starting, and snapshot state", async () => {
  const responses = [
    validRunningBotStatus({ tenantKey: undefined }),
    validRunningBotStatus({ tenantKey: "other-tenant" }),
    validStartingBotStatus({ tenantKey: "other-tenant" }),
    {
      running: false,
      snapshotAtMs: 3,
      snapshot: validRunningBotStatus({ tenantKey: "other-tenant" }),
    },
  ];
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () => jsonResponse(200, responses.shift()),
    async (api) => {
      await assert.rejects(
        () => api.botStatus("/api", { timeoutMs: 5_000 }, undefined, "BTCUSDT", "tenant"),
        (error) =>
          error instanceof api.InvalidApiResponseError &&
          error.field === "status.tenantKey must be a non-empty string",
      );
      await assert.rejects(
        () => api.botStatus("/api", { timeoutMs: 5_000 }, undefined, "BTCUSDT", "tenant"),
        (error) =>
          error instanceof api.InvalidApiResponseError &&
          error.field === "status.tenantKey does not match the requested tenant",
      );
      await assert.rejects(
        () => api.botStart("/api", { tenantKey: "tenant", botSymbols: ["BTCUSDT"] }, { timeoutMs: 5_000 }),
        (error) =>
          error instanceof api.InvalidApiResponseError &&
          error.field === "status.tenantKey does not match the requested tenant",
      );
      await assert.rejects(
        () => api.botStop("/api", { timeoutMs: 5_000 }, "BTCUSDT", "tenant"),
        (error) =>
          error instanceof api.InvalidApiResponseError &&
          error.field === "status.snapshot.tenantKey does not match the requested tenant",
      );
    },
  );
});

test("bot lifecycle decoder binds unambiguous status, stop, and start symbols", async () => {
  const responses = [
    validRunningBotStatus({ symbol: "ETHUSDT" }),
    {
      running: false,
      snapshotAtMs: 3,
      snapshot: validRunningBotStatus({ symbol: "ETHUSDT" }),
    },
    validStartingBotStatus({ symbol: "ETHUSDT" }),
  ];
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () => jsonResponse(200, responses.shift()),
    async (api) => {
      await assert.rejects(
        () => api.botStatus("/api", { timeoutMs: 5_000 }, undefined, "BTCUSDT", "tenant"),
        (error) =>
          error instanceof api.InvalidApiResponseError &&
          error.field === "status.symbol does not match the requested symbol",
      );
      await assert.rejects(
        () => api.botStop("/api", { timeoutMs: 5_000 }, "BTCUSDT", "tenant"),
        (error) =>
          error instanceof api.InvalidApiResponseError &&
          error.field === "status.snapshot.symbol does not match the requested symbol",
      );
      await assert.rejects(
        () => api.botStart("/api", { tenantKey: "tenant", botSymbols: ["BTCUSDT"] }, { timeoutMs: 5_000 }),
        (error) =>
          error instanceof api.InvalidApiResponseError &&
          error.field === "status.symbol does not match the requested symbol",
      );
    },
  );
});

test("single-symbol bot start accepts a valid multi response containing an orphan-position bot", async () => {
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () =>
      jsonResponse(202, {
        running: true,
        starting: false,
        multi: true,
        bots: [validRunningBotStatus(), validRunningBotStatus({ symbol: "ETHUSDT" })],
      }),
    async (api) => {
      const status = await api.botStart(
        "/api",
        { tenantKey: "tenant", botSymbols: ["BTCUSDT"] },
        { timeoutMs: 5_000 },
      );
      assert.equal(status.multi, true);
      assert.deepEqual(
        status.bots.map((bot) => bot.symbol),
        ["BTCUSDT", "ETHUSDT"],
      );
    },
  );
});

test("bot start accepts a running multi-bot response with a queued sequential start", async () => {
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () =>
      jsonResponse(202, {
        running: true,
        starting: true,
        multi: true,
        bots: [validRunningBotStatus()],
        queued: [{ symbol: "ETHUSDT", message: "Queued for sequential start after currently active bots are stable." }],
      }),
    async (api) => {
      const status = await api.botStart("/api", { botSymbols: ["BTCUSDT", "ETHUSDT"] }, { timeoutMs: 5_000 });
      assert.equal(status.running, true);
      assert.equal(status.starting, true);
      assert.equal(status.bots.length, 1);
      assert.equal(status.queued.length, 1);
    },
  );
});

test("bot start accepts a queued-only multi-bot response", async () => {
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () =>
      jsonResponse(202, {
        running: false,
        starting: true,
        multi: true,
        bots: [],
        queued: [{ symbol: "ETHUSDT", message: "Queued for sequential start after currently active bots are stable." }],
      }),
    async (api) => {
      const status = await api.botStart("/api", { botSymbols: ["ETHUSDT"] }, { timeoutMs: 5_000 });
      assert.equal(status.running, false);
      assert.equal(status.starting, true);
      assert.deepEqual(status.bots, []);
      assert.equal(status.queued.length, 1);
    },
  );
});

test("bot status accepts backend-bounded long-lookback series that exceed the retention target", async () => {
  const seriesLength = 104;
  const numericSeries = Array.from({ length: seriesLength }, (_, index) => index + 1);
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () =>
      jsonResponse(
        200,
        validRunningBotStatus({
          settings: {
            pollSeconds: 5,
            onlineEpochs: 1,
            trainBars: 100,
            maxPoints: 100,
            tradeEnabled: false,
          },
          prices: numericSeries,
          openTimes: numericSeries,
          kalmanPredNext: numericSeries,
          lstmPredNext: numericSeries,
          equityCurve: numericSeries,
          positions: numericSeries.map(() => 0),
        }),
      ),
    async (api) => {
      const status = await api.botStatus("/api", { timeoutMs: 5_000 });
      assert.equal(status.running, true);
      assert.equal(status.prices.length, seriesLength);
      assert.equal(status.settings.maxPoints, 100);
    },
  );
});

test("bot start decodes after transient retry handling so malformed success cannot duplicate the mutation", async () => {
  let calls = 0;
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () => {
      calls += 1;
      return jsonResponse(200, validRunningBotStatus({ settings: { tradeEnabled: "false" } }));
    },
    async (api) => {
      await assert.rejects(
        () => api.botStart("/api", { binanceSymbol: "BTCUSDT" }, { timeoutMs: 5_000 }),
        (error) =>
          error instanceof api.InvalidApiResponseError &&
          error.endpoint === "/bot/start" &&
          error.field === "status.settings.tradeEnabled must be a boolean",
      );
    },
  );
  assert.equal(calls, 1);
});

test("api client preserves exact safe integer query params for bot status, ops, and ops performance", async () => {
  const calls = [];
  const tenantHeaders = [];
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      calls.push(String(url));
      tenantHeaders.push(new Headers(init.headers).get("X-Tenant-Key"));
      return jsonResponse(200, { ok: true, running: false });
    },
    async (api) => {
      const status = await api.botStatus("/api", { timeoutMs: 5_000 }, 25, undefined, "tenant-status");
      assert.equal(status.running, false);
      await api.ops(
        "/api",
        {
          limit: 10,
          since: 20,
          fromMs: 30,
          toMs: 40,
          bot: true,
          tenantKey: "tenant-ops",
        },
        { timeoutMs: 5_000 },
      );
      await api.opsPerformance(
        "/api",
        { commitLimit: 7, comboLimit: 9, tenantKey: "tenant-perf" },
        { timeoutMs: 5_000 },
      );
    },
  );
  assert.deepEqual(calls, [
    "/api/bot/status?tail=25",
    "/api/ops?limit=10&since=20&fromMs=30&toMs=40&bot=1",
    "/api/ops/performance?commitLimit=7&comboLimit=9",
  ]);
  assert.deepEqual(tenantHeaders, ["tenant-status", "tenant-ops", "tenant-perf"]);
});

test("api client sends Binance trades includeMaxPnl flag in JSON body with tenant header", async () => {
  const calls = [];
  const bodies = [];
  const tenantHeaders = [];
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      calls.push(String(url));
      bodies.push(JSON.parse(String(init.body)));
      tenantHeaders.push(new Headers(init.headers).get("X-Tenant-Key"));
      return jsonResponse(200, {
        market: "futures",
        testnet: false,
        interval: "3m",
        symbols: ["BTCUSDT"],
        allSymbols: false,
        trades: [],
        fetchedAtMs: 1,
      });
    },
    async (api) => {
      const out = await api.binanceTrades(
        "/api",
        {
          market: "futures",
          binanceTestnet: false,
          tenantKey: "tenant-trades",
          symbol: "BTCUSDT",
          interval: "3m",
          limit: 100,
          includeMaxPnl: false,
        },
        { timeoutMs: 5_000 },
      );
      assert.equal(out.market, "futures");
    },
  );
  assert.deepEqual(calls, ["/api/binance/trades"]);
  assert.deepEqual(tenantHeaders, ["tenant-trades"]);
  assert.deepEqual(bodies, [
    {
      market: "futures",
      binanceTestnet: false,
      tenantKey: "tenant-trades",
      symbol: "BTCUSDT",
      interval: "3m",
      limit: 100,
      includeMaxPnl: false,
    },
  ]);
});

test("api client sends Binance revenue accounting inputs in JSON body with tenant header", async () => {
  const calls = [];
  const bodies = [];
  const tenantHeaders = [];
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      calls.push(String(url));
      bodies.push(JSON.parse(String(init.body)));
      tenantHeaders.push(new Headers(init.headers).get("X-Tenant-Key"));
      return jsonResponse(200, {
        market: "futures",
        testnet: false,
        fetchedAtMs: 2,
        ledger: {
          asset: "USDT",
          startAtMs: 1,
          endAtMs: 2,
          incomeRecords: 0,
          tradeRecords: 0,
          incomeMayBeTruncated: false,
          tradesMayBeTruncated: false,
          breakdown: {},
          unrealizedPnl: 0,
          infrastructureCost: 25,
          netRevenue: -25,
          execution: {},
          daily: [],
          symbols: [],
          unclassifiedIncomeTypes: [],
        },
      });
    },
    async (api) => {
      const out = await api.binanceRevenue(
        "/api",
        {
          market: "futures",
          tenantKey: "tenant-revenue",
          asset: "USDT",
          startTimeMs: 1,
          endTimeMs: 2,
          incomeLimit: 1000,
          tradeLimit: 1000,
          infrastructureCost: 25,
          includeUnrealized: false,
        },
        { timeoutMs: 5_000 },
      );
      assert.equal(out.ledger.netRevenue, -25);
    },
  );
  assert.deepEqual(calls, ["/api/binance/revenue"]);
  assert.deepEqual(tenantHeaders, ["tenant-revenue"]);
  assert.deepEqual(bodies, [
    {
      market: "futures",
      tenantKey: "tenant-revenue",
      asset: "USDT",
      startTimeMs: 1,
      endTimeMs: 2,
      incomeLimit: 1000,
      tradeLimit: 1000,
      infrastructureCost: 25,
      includeUnrealized: false,
    },
  ]);
});

test("api client omits fractional and unsafe integer query params instead of truncating them", async () => {
  const calls = [];
  const tenantHeaders = [];
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      calls.push(String(url));
      tenantHeaders.push(new Headers(init.headers).get("X-Tenant-Key"));
      return jsonResponse(200, { ok: true, running: false });
    },
    async (api) => {
      await api.botStatus("/api", { timeoutMs: 5_000 }, 12.5, undefined, "tenant-fractional");
      await api.botStatus("/api", { timeoutMs: 5_000 }, Number.MAX_SAFE_INTEGER + 1, undefined, "tenant-unsafe");
      await api.ops(
        "/api",
        {
          limit: 10.5,
          since: Number.MAX_SAFE_INTEGER + 1,
          fromMs: 30.5,
          toMs: Number.MAX_SAFE_INTEGER + 1,
          bot: true,
          tenantKey: "tenant-ops",
        },
        { timeoutMs: 5_000 },
      );
      await api.opsPerformance(
        "/api",
        {
          commitLimit: 7.5,
          comboLimit: Number.MAX_SAFE_INTEGER + 1,
          tenantKey: "tenant-perf",
        },
        { timeoutMs: 5_000 },
      );
    },
  );
  assert.deepEqual(calls, [
    "/api/bot/status",
    "/api/bot/status",
    "/api/ops?bot=1",
    "/api/ops/performance",
  ]);
  assert.deepEqual(tenantHeaders, ["tenant-fractional", "tenant-unsafe", "tenant-ops", "tenant-perf"]);
});

test("api client preserves absolute base path and merges base query with state sync query", async () => {
  const calls = [];
  let tenantHeader = null;
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com/base?tenantKey=demo&mode=direct",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      calls.push(String(url));
      tenantHeader = new Headers(init.headers).get("X-Tenant-Key");
      return jsonResponse(200, {});
    },
    async (api) => {
      const out = await api.stateSyncExport(
        "https://api.example.com/base?tenantKey=demo&mode=direct",
        { timeoutMs: 5_000, tenantKey: "tenant-request" },
      );
      assert.deepEqual(out, {});
    },
  );
  assert.equal(tenantHeader, "tenant-request");
  assert.deepEqual(calls, ["https://api.example.com/base/state/sync?mode=direct"]);
});

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
        return jsonResponse(202, validStartingBotStatus());
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

test("api fallback keeps root-path bases same-origin in explicit configs", async () => {
  const calls = [];
  await withApiModule(
    {
      apiBaseUrl: "/",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "https://api.example.com",
      apiToken: "",
    },
    async (url) => {
      const href = String(url);
      calls.push(href);
      if (href === "/health") {
        return jsonResponse(502, { error: "Bad Gateway" });
      }
      if (href === "https://api.example.com/health") {
        return jsonResponse(200, { status: "ok" });
      }
      throw new Error(`unexpected request: ${href}`);
    },
    async (api) => {
      await assert.rejects(
        () => api.health("/", { timeoutMs: 5_000 }),
        (err) => err?.name === "HttpError" && err.status === 502,
      );
    },
  );
  assert.deepEqual(calls, ["/health"]);
});

test("api fallback only enables inferred /api cross-origin failover for normalized boolean-like encodings", async () => {
  const cases = [
    { label: "boolean true", value: true, expectFallback: true },
    { label: "string \\\"true\\\"", value: "true", expectFallback: true },
    { label: "string \\\" TrUe \\\"", value: " TrUe ", expectFallback: true },
    { label: "number 1", value: 1, expectFallback: true },
    { label: "string \\\"1\\\"", value: "1", expectFallback: true },
    { label: "string \\\" 1 \\\"", value: " 1 ", expectFallback: true },
    { label: "boolean false", value: false, expectFallback: false },
    { label: "string \\\"false\\\"", value: "false", expectFallback: false },
    { label: "string \\\" FaLsE \\\"", value: " FaLsE ", expectFallback: false },
    { label: "number 0", value: 0, expectFallback: false },
    { label: "string \\\"0\\\"", value: "0", expectFallback: false },
    { label: "string \\\" 0 \\\"", value: " 0 ", expectFallback: false },
    { label: "string \\\"yes\\\"", value: "yes", expectFallback: false },
    { label: "number 2", value: 2, expectFallback: false },
    { label: "string \\\"01\\\"", value: "01", expectFallback: false },
    { label: "null", value: null, expectFallback: false },
  ];

  for (const testCase of cases) {
    const calls = [];
    await withApiModule(
      {
        apiBaseUrl: "/api",
        apiBaseUrlInferred: testCase.value,
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
        if (testCase.expectFallback) {
          const out = await api.health("/api", { timeoutMs: 5_000 });
          assert.equal(out.status, "ok", `${testCase.label} should preserve inferred failover`);
          return;
        }
        await assert.rejects(
          () => api.health("/api", { timeoutMs: 5_000 }),
          (err) => err?.name === "HttpError" && err.status === 502,
          `${testCase.label} should not trigger inferred failover`,
        );
      },
    );
    assert.deepEqual(
      calls,
      testCase.expectFallback ? ["/api/health", "https://api.example.com/health"] : ["/api/health"],
      testCase.label,
    );
  }
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

test("api fallback allows inferred /api cross-origin failover for tenant-scoped non-GET requests", async () => {
  const calls = [];
  let directTenantHeader = null;
  await withApiModule(
    {
      apiBaseUrl: "/api",
      apiBaseUrlInferred: true,
      apiFallbackUrl: "https://api.example.com",
      apiToken: "",
    },
    async (url, init = {}) => {
      const href = String(url);
      calls.push(href);
      if (href === "/api/binance/listenKey") {
        return jsonResponse(502, { error: "Bad Gateway" });
      }
      if (href === "https://api.example.com/binance/listenKey") {
        directTenantHeader = new Headers(init.headers).get("X-Tenant-Key");
        return jsonResponse(200, {
          listenKey: "listen-key-1",
          tenantKey: "tenant",
          market: "spot",
          testnet: false,
          wsUrl: "wss://stream.example.com/ws/listen-key-1",
          keepAliveMs: 60_000,
        });
      }
      throw new Error(`unexpected request: ${href}`);
    },
    async (api) => {
      const out = await api.binanceListenKey("/api", { tenantKey: "tenant" }, { timeoutMs: 5_000 });
      assert.equal(out.listenKey, "listen-key-1");
      assert.equal(out.tenantKey, "tenant");
    },
  );
  assert.deepEqual(calls, ["/api/binance/listenKey", "https://api.example.com/binance/listenKey"]);
  assert.equal(directTenantHeader, "tenant");
});

test("api fallback still skips inferred /api cross-origin failover for unauthenticated non-GET requests", async () => {
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
      if (href === "/api/cache/clear") {
        return jsonResponse(502, { error: "Bad Gateway" });
      }
      throw new Error(`unexpected request: ${href}`);
    },
    async (api) => {
      await assert.rejects(
        () => api.cacheClear("/api", { timeoutMs: 5_000 }),
        (err) => err?.name === "HttpError" && err.status === 502,
      );
    },
  );
  assert.deepEqual(calls, ["/api/cache/clear"]);
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
      return jsonResponse(202, validStartingBotStatus({ tenantKey: "tenant-body" }));
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

test("api client forwards X-Tenant-Key from FormData body with string tenantKey", async () => {
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () => jsonResponse(200, {}),
    async (api) => {
      const fd = new FormData();
      fd.append("tenantKey", "tenant-form");
      fd.append("someFile", new Blob(["data"]), "file.txt");
      const headers = api.withTenantHeader(new Headers(), "/bot/start", fd);
      assert.equal(headers.get("X-Tenant-Key"), "tenant-form");
    },
  );
});

test("api client does not forward X-Tenant-Key for non-string (Blob/File) tenantKey in FormData", async () => {
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async () => jsonResponse(200, {}),
    async (api) => {
      const fd = new FormData();
      fd.append("tenantKey", new Blob(["secret"]), "key.bin");
      const headers = api.withTenantHeader(new Headers(), "/bot/start", fd);
      assert.equal(headers.get("X-Tenant-Key"), null);
    },
  );
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

test("api client forwards X-Tenant-Key for cross-origin direct-host tenant queries", async () => {
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
      if (href !== "https://api.example.com/bot/status?tail=100") throw new Error(`unexpected request: ${href}`);
      tenantHeader = new Headers(init.headers).get("X-Tenant-Key");
      return jsonResponse(200, { running: false });
    },
    async (api) => {
      const out = await api.botStatus("https://api.example.com", { timeoutMs: 5_000 }, 100, undefined, "tenant-query");
      assert.equal(out.running, false);
    },
  );
  assert.equal(tenantHeader, "tenant-query");
});

test("api client forwards X-Tenant-Key for cross-origin direct-host writes", async () => {
  let tenantHeader = null;
  await withApiModule(
    {
      apiBaseUrl: "https://api.example.com",
      apiBaseUrlInferred: false,
      apiFallbackUrl: "",
      apiToken: "",
    },
    async (url, init = {}) => {
      const href = String(url);
      if (href !== "https://api.example.com/bot/start") throw new Error(`unexpected request: ${href}`);
      tenantHeader = new Headers(init.headers).get("X-Tenant-Key");
      return jsonResponse(202, validStartingBotStatus({ tenantKey: "tenant-body" }));
    },
    async (api) => {
      const out = await api.botStart(
        "https://api.example.com",
        { tenantKey: "tenant-body", binanceSymbol: "BTCUSDT" },
        { timeoutMs: 5_000 },
      );
      assert.equal(out.starting, true);
    },
  );
  assert.equal(tenantHeader, "tenant-body");
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

test("tenantKeyFromBody extracts tenant key from FormData string entry", async () => {
  await withApiModule(
    { apiBaseUrl: "https://api.example.com", apiBaseUrlInferred: false, apiFallbackUrl: "", apiToken: "" },
    async () => jsonResponse(200, { status: "ok" }),
    async (api) => {
      const fd = new FormData();
      fd.append("tenantKey", "form-tenant");
      assert.equal(api.tenantKeyFromBody(fd), "form-tenant");
    },
  );
});

test("tenantKeyFromBody returns null for FormData with File entry", async () => {
  await withApiModule(
    { apiBaseUrl: "https://api.example.com", apiBaseUrlInferred: false, apiFallbackUrl: "", apiToken: "" },
    async () => jsonResponse(200, { status: "ok" }),
    async (api) => {
      const fd = new FormData();
      fd.append("tenantKey", new Blob(["data"], { type: "text/plain" }), "key.txt");
      assert.equal(api.tenantKeyFromBody(fd), null);
    },
  );
});

test("tenantKeyFromBody returns null for FormData without tenantKey", async () => {
  await withApiModule(
    { apiBaseUrl: "https://api.example.com", apiBaseUrlInferred: false, apiFallbackUrl: "", apiToken: "" },
    async () => jsonResponse(200, { status: "ok" }),
    async (api) => {
      const fd = new FormData();
      fd.append("otherField", "value");
      assert.equal(api.tenantKeyFromBody(fd), null);
    },
  );
});

test("api client forwards X-Tenant-Key header from FormData body containing tenantKey", async () => {
  await withApiModule(
    { apiBaseUrl: "https://api.example.com", apiBaseUrlInferred: false, apiFallbackUrl: "", apiToken: "" },
    async () => jsonResponse(200, { status: "ok" }),
    async (api) => {
      const fd = new FormData();
      fd.append("tenantKey", "form-tenant");
      const key = api.tenantKeyFromBody(fd);
      const headers = new Headers();
      if (key) headers.set("X-Tenant-Key", key);
      assert.equal(headers.get("X-Tenant-Key"), "form-tenant");
    },
  );
});

test("api client does not set X-Tenant-Key header when FormData tenantKey is a File", async () => {
  await withApiModule(
    { apiBaseUrl: "https://api.example.com", apiBaseUrlInferred: false, apiFallbackUrl: "", apiToken: "" },
    async () => jsonResponse(200, { status: "ok" }),
    async (api) => {
      const fd = new FormData();
      fd.append("tenantKey", new Blob(["data"], { type: "text/plain" }), "key.txt");
      const key = api.tenantKeyFromBody(fd);
      const headers = new Headers();
      if (key) headers.set("X-Tenant-Key", key);
      assert.equal(headers.get("X-Tenant-Key"), null);
    },
  );
});
