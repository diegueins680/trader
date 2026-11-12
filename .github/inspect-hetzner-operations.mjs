// Run inside the API container. Credentials stay in that container; output is a
// bounded operational report. The only POST calls Binance's test-order probe.
// Order journal rows report outcome fields only (no ids, sizes or payloads).
import { createHash } from "node:crypto";
import { open, readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";

const env = process.env;
const secretValues = Object.entries(env)
  .filter(([key, value]) => /secret|token|password|key|credential|db_url|database_url/i.test(key) && value.length >= 4)
  .map(([, value]) => value)
  .sort((a, b) => b.length - a.length);
const safeText = value => {
  let text = String(value ?? "");
  for (const secret of secretValues) text = text.split(secret).join("[redacted]");
  return text.replace(/(signature|apiKey|apiSecret|authorization)\s*[=:]\s*[^\s,&]+/gi, "$1=[redacted]").slice(0, 700);
};
const safeError = error => ({ name: error?.name ?? "Error", message: safeText(error?.message) });
const tenantKey = "binance:" + createHash("sha256").update(env.BINANCE_API_KEY.trim() + ":" + env.BINANCE_API_SECRET.trim()).digest("hex");
const headers = { authorization: "Bearer " + env.TRADER_API_TOKEN, "x-tenant-key": tenantKey };
const request = async (route, payload) => {
  const response = await fetch("http://127.0.0.1:8080" + route, {
    headers: { ...headers, ...(payload ? { "content-type": "application/json" } : {}) },
    method: payload ? "POST" : "GET",
    ...(payload ? { body: JSON.stringify(payload) } : {}),
    signal: AbortSignal.timeout(90000),
  });
  if (!response.ok) throw Error(`HTTP ${response.status} on ${route}`);
  return response.json();
};
const result = { capturedAt: new Date().toISOString(), role: env.TRADER_SERVER_ROLE };
const attempt = async (key, action) => {
  try { result[key] = await action(); } catch (error) { result[key] = { error: safeError(error) }; }
};
await attempt("health", () => request("/health"));
await attempt("ready", () => request("/ready"));
result.configuration = Object.fromEntries([
  "TRADER_SERVER_ROLE", "TRADER_BOT_TRADE", "TRADER_BINANCE_LIVE", "TRADER_BOT_AUTOSTART",
  "TRADER_OPTIMIZER_ENABLED", "TRADER_OPTIMIZER_EVERY_SEC", "TRADER_OPTIMIZER_TIMEOUT_SEC",
  "TRADER_OPTIMIZER_TRIALS", "TRADER_OPTIMIZER_OBJECTIVE", "TRADER_OPTIMIZER_TRIAL_HEAP_CAP",
  "TRADER_TOP_COMBOS_BACKTEST_ENABLED", "TRADER_TOP_COMBOS_SYNC_EVERY_SEC",
  "TRADER_BOT_START_ADOPTION_MAX_POSITION_SIZE_CAP", "TRADER_PORTFOLIO_AUTO_GRADUATE_STARTED_AT_MS",
  "TRADER_PORTFOLIO_SELECTOR_ROLLOUT_MODE", "TRADER_PORTFOLIO_AUTO_GRADUATE_MIN_DAILY_OBSERVATIONS",
].map(key => [key, env[key] ?? null]));
await attempt("fleet", async () => {
  const status = await request("/bot/status?tail=1");
  const bots = status.multi ? status.bots : [status];
  return {
    selector: status.portfolioSelector,
    bots: bots.map(bot => ({
      symbol: bot.symbol, running: bot.running, live: bot.live, interval: bot.interval,
      tradeEnabled: bot.settings?.tradeEnabled, protectionOrders: bot.settings?.protectionOrders,
      halted: bot.halted, error: bot.error ? safeText(bot.error) : null,
      updatedAtMs: bot.updatedAtMs, polledAtMs: bot.polledAtMs, position: bot.positions?.at(-1),
      activeUuid: bot.performance?.closeTiming?.comboUuid,
      consecutiveOrderErrors: bot.consecutiveOrderErrors, dayTrades: bot.dayTrades,
      action: bot.latestSignal?.action, reason: bot.decisionTrace?.reason,
      outcome: bot.decisionTrace?.outcome,
      startedAtMs: bot.startedAtMs, hasOpenTrade: bot.openTrade != null, orderCount: (bot.orders ?? []).length, tradeCount: (bot.trades ?? []).length,
      lastOrder: bot.lastOrder ? { sent: bot.lastOrder.sent, side: bot.lastOrder.side, status: bot.lastOrder.status, filled: (bot.lastOrder.executedQty ?? 0) > 0, message: safeText(bot.lastOrder.message) } : null,
      stages: (bot.decisionTrace?.stages ?? []).slice(0, 24).map(stage => `${safeText(stage.label).slice(0, 60)}=${safeText(stage.status).slice(0, 30)}`),
    })),
  };
});
if (env.TRADER_SERVER_ROLE === "trading") await attempt("binanceTest", async () => {
  const probe = await request("/binance/keys", { platform: "binance", market: "futures", binanceSymbol: "BNBUSDT", orderQuote: 10 });
  return { market: probe.market, testnet: probe.testnet, signed: probe.signed, tradeTest: probe.tradeTest };
});
await attempt("leaderboard", async () => {
  const board = await request("/optimizer/combos");
  const combos = board.combos ?? [];
  const symbols = (env.TRADER_BOT_SYMBOLS ?? "").split(",");
  const pins = (env.TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS ?? "").split(",");
  const missing = ["AAVEUSDT", "ATOMUSDT", "DOGEUSDT", "ETHUSDT", "FILUSDT", "LINKUSDT", "SOLUSDT"];
  const row = combo => ({
    uuid: combo.uuid, symbol: combo.params?.binanceSymbol ?? combo.params?.symbol,
    createdAtMs: combo.createdAtMs, finalEquity: combo.finalEquity, objective: combo.objective,
    processing: combo.processing, method: combo.params?.method, interval: combo.params?.interval,
    bars: combo.params?.bars, lookbackWindow: combo.params?.lookbackWindow,
    hiddenSize: combo.params?.hiddenSize, epochs: combo.params?.epochs,
  });
  const tiers = {};
  for (const combo of combos) { const tier = combo.processing?.tier ?? "unknown"; tiers[tier] = (tiers[tier] ?? 0) + 1; }
  return {
    count: combos.length, tiers, generatedAtMs: board.generatedAtMs,
    newest: combos.slice().sort((a, b) => b.createdAtMs - a.createdAtMs).slice(0, 4).map(row),
    pins: pins.filter(Boolean).map((uuid, i) => { const combo = combos.find(x => x.uuid === uuid); return { symbol: combo?.params?.binanceSymbol ?? combo?.params?.symbol ?? (env.TRADER_SERVER_ROLE === "trading" ? symbols[i] : null), uuid, present: !!combo, processing: combo?.processing, backtest: combo ? { interval: combo.params?.interval, bars: combo.params?.bars, openThreshold: combo.openThreshold ?? combo.params?.openThreshold, tradeCount: combo.metrics?.tradeCount, roundTrips: combo.metrics?.roundTrips, positionChanges: combo.metrics?.positionChanges, exposure: combo.metrics?.exposure, avgHoldingPeriods: combo.metrics?.avgHoldingPeriods, orderQuoteFraction: combo.params?.orderQuoteFraction ?? null, maxPositionSize: combo.params?.maxPositionSize ?? null, minPositionSize: combo.params?.minPositionSize ?? null, confidenceSizing: combo.params?.confidenceSizing ?? null } : null }; }),
    replacements: missing.map(symbol => ({ symbol, strict: combos.filter(combo => (combo.params?.binanceSymbol ?? combo.params?.symbol) === symbol && combo.processing?.tier === "deployable" && !combo.processing?.relaxed).slice(0, 5).map(row) })),
  };
});
await attempt("journal", async () => {
  const root = env.TRADER_JOURNAL_DIR;
  const files = [];
  const walk = async (directory, depth = 0) => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const name = path.join(directory, entry.name);
      if (entry.isDirectory() && depth < 2) await walk(name, depth + 1);
      else if (entry.isFile()) { const info = await stat(name); files.push({ name, mtime: info.mtimeMs, size: info.size }); }
    }
  };
  await walk(root);
  files.sort((a, b) => b.mtime - a.mtime);
  const rows = [];
  const counts = {};
  for (const file of files.slice(0, 3)) {
    const handle = await open(file.name);
    try {
      const length = Math.min(file.size, 8 * 1024 * 1024);
      const buffer = Buffer.alloc(length);
      await handle.read(buffer, 0, length, file.size - length);
      const lines = buffer.toString("utf8").split("\n");
      if (file.size > length) lines.shift();
      for (const line of lines) {
        let event; try { event = JSON.parse(line); } catch { continue; }
        const type = event.type ?? event.kind ?? "unknown";
        counts[type] = (counts[type] ?? 0) + 1;
        if (type === "bot.order") {
          // Outcome fields only: no order ids, quantities or raw exchange payloads.
          const order = event.event?.order ?? {};
          const risk = order.marketRisk;
          const outcome = item => ({ sent: item.sent, mode: item.mode, side: item.side, status: item.status, reduceOnly: item.reduceOnly, filled: (item.executedQty ?? 0) > 0, message: safeText(item.message) });
          rows.push({
            type, atMs: event.atMs, symbol: event.symbol, market: event.market, partial: event.partial === true,
            opSide: event.event?.opSide, ...outcome(order),
            precedingClose: order.precedingClose ? outcome(order.precedingClose) : null,
            marketRisk: risk ? { allowed: risk.allowed, admissionOutcome: risk.admissionOutcome, reasons: (risk.reasons ?? []).slice(0, 8).map(safeText), edgeBps: risk.edgeBps, spreadBps: risk.spreadBps, impactBps: risk.impactBps, markBasisBps: risk.markBasisBps } : null,
          });
          continue;
        }
        if (!/optimizer|graduat|sync.*(fail|error)|autostart.*(fail|error)/i.test(type)) continue;
        const selected = Object.fromEntries(Object.entries(event).filter(([key, value]) => /^(type|atMs|symbol|count|admittedCount|newestCreatedAtMs|boardCount|beforeCount|afterCount|trials|durationMs|elapsedMs|exitCode|reason|summary|error|message|phase|status|objective)$/.test(key) && ["string", "number", "boolean"].includes(typeof value)).map(([key, value]) => [key, typeof value === "string" ? safeText(value) : value]));
        rows.push(selected);
      }
    } finally { await handle.close(); }
  }
  rows.sort((a, b) => (a.atMs ?? 0) - (b.atMs ?? 0));
  return { files: files.slice(0, 3).map(file => ({ name: path.basename(file.name), mtime: new Date(file.mtime).toISOString(), size: file.size })), counts, recent: rows.slice(-35) };
});
await attempt("processes", async () => {
  const matches = [];
  for (const pid of await readdir("/proc")) {
    if (!/^\d+$/.test(pid)) continue;
    try {
      const args = (await readFile(`/proc/${pid}/cmdline`, "utf8")).split("\0");
      if (!args.some(arg => /optimizer|trader-hs|backtest/.test(arg))) continue;
      const status = await readFile(`/proc/${pid}/status`, "utf8");
      matches.push({ pid: Number(pid), executable: path.basename(args[0]), optimizer: args.some(arg => /optimizer/.test(arg)), backtest: args.some(arg => /backtest/.test(arg)), rss: status.match(/^VmRSS:\s*(.*)$/m)?.[1] });
    } catch { /* Process exited. */ }
  }
  return matches;
});
await attempt("memory", async () => {
  const current = Number((await readFile("/sys/fs/cgroup/memory.current", "utf8")).trim());
  const limit = (await readFile("/sys/fs/cgroup/memory.max", "utf8")).trim();
  const events = Object.fromEntries((await readFile("/sys/fs/cgroup/memory.events", "utf8")).trim().split("\n").map(line => {
    const [name, count] = line.split(/\s+/);
    return [name, Number(count)];
  }));
  return { currentBytes: current, limitBytes: limit === "max" ? null : Number(limit), events };
});
console.log(JSON.stringify(result));
