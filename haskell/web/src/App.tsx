import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ApiParams, ApiTradeResponse, BacktestResponse, BotStatus, LatestSignal, Market, Method, Normalization } from "./lib/types";
import { HttpError, backtest, botStart, botStatus, botStop, health, signal, trade } from "./lib/api";
import { copyText } from "./lib/clipboard";
import { readJson, readSessionString, removeSessionKey, writeJson, writeSessionString } from "./lib/storage";
import { fmtMoney, fmtNum, fmtPct, fmtRatio } from "./lib/format";
import { BacktestChart } from "./components/BacktestChart";

type RequestKind = "signal" | "backtest" | "trade";

type RunOptions = {
  silent?: boolean;
};

const API_TARGET = (__TRADER_API_TARGET__ || "http://127.0.0.1:8080").replace(/\/+$/, "");
const API_PORT = (() => {
  try {
    return new URL(API_TARGET).port || "8080";
  } catch {
    return "8080";
  }
})();

type UiState = {
  loading: boolean;
  error: string | null;
  lastKind: RequestKind | null;
  latestSignal: LatestSignal | null;
  backtest: BacktestResponse | null;
  trade: ApiTradeResponse | null;
};

type BotUiState = {
  loading: boolean;
  error: string | null;
  status: BotStatus;
};

type FormState = {
  binanceSymbol: string;
  market: Market;
  interval: string;
  bars: number;
  method: Method;
  threshold: number;
  fee: number;
  backtestRatio: number;
  normalization: Normalization;
  epochs: number;
  hiddenSize: number;
  optimizeOperations: boolean;
  sweepThreshold: boolean;
  binanceTestnet: boolean;
  orderQuote: number;
  orderQuantity: number;
  binanceLive: boolean;
  tradeArmed: boolean;
  autoRefresh: boolean;
  autoRefreshSec: number;
};

const STORAGE_KEY = "trader.ui.form.v1";
const SESSION_TOKEN_KEY = "trader.ui.apiToken.v1";

const SIGNAL_TIMEOUT_MS = 5 * 60_000;
const BACKTEST_TIMEOUT_MS = 10 * 60_000;
const TRADE_TIMEOUT_MS = 5 * 60_000;
const BOT_START_TIMEOUT_MS = 10 * 60_000;

const defaultForm: FormState = {
  binanceSymbol: "BTCUSDT",
  market: "spot",
  interval: "1h",
  bars: 200,
  method: "11",
  threshold: 0.001,
  fee: 0.0005,
  backtestRatio: 0.2,
  normalization: "standard",
  epochs: 30,
  hiddenSize: 16,
  optimizeOperations: false,
  sweepThreshold: false,
  binanceTestnet: false,
  orderQuote: 20,
  orderQuantity: 0,
  binanceLive: false,
  tradeArmed: false,
  autoRefresh: false,
  autoRefreshSec: 20,
};

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

function numFromInput(raw: string, fallback: number): number {
  if (raw.trim() === "") return fallback;
  const n = Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

function escapeSingleQuotes(raw: string): string {
  return raw.replaceAll("'", "'\\''");
}

function errorName(err: unknown): string {
  if (!err || typeof err !== "object" || !("name" in err)) return "";
  return String((err as { name: unknown }).name);
}

function isAbortError(err: unknown): boolean {
  const name = errorName(err);
  if (name === "AbortError") return true;
  if (!(err instanceof Error)) return false;
  return err.message.toLowerCase().includes("aborted");
}

function isTimeoutError(err: unknown): boolean {
  return errorName(err) === "TimeoutError";
}

function actionBadgeClass(action: string): string {
  const a = action.toUpperCase();
  if (a.includes("LONG")) return "badge badgeStrong badgeLong";
  if (a.includes("FLAT")) return "badge badgeStrong badgeFlat";
  return "badge badgeStrong badgeHold";
}

function methodLabel(method: Method): string {
  switch (method) {
    case "11":
      return "Both (agreement gated)";
    case "10":
      return "Kalman only";
    case "01":
      return "LSTM only";
  }
}

function marketLabel(m: Market): string {
  switch (m) {
    case "spot":
      return "Spot";
    case "margin":
      return "Margin";
    case "futures":
      return "Futures";
  }
}

export function App() {
  const [apiOk, setApiOk] = useState<"unknown" | "ok" | "down" | "auth">("unknown");
  const [toast, setToast] = useState<string | null>(null);
  const [apiToken, setApiToken] = useState<string>(() => readSessionString(SESSION_TOKEN_KEY) ?? "");
  const [form, setForm] = useState<FormState>(() => {
    const saved = readJson<Partial<FormState>>(STORAGE_KEY);
    return { ...defaultForm, ...(saved ?? {}) };
  });

  const [state, setState] = useState<UiState>({
    loading: false,
    error: null,
    lastKind: null,
    latestSignal: null,
    backtest: null,
    trade: null,
  });

  const [bot, setBot] = useState<BotUiState>({
    loading: false,
    error: null,
    status: { running: false },
  });

  const abortRef = useRef<AbortController | null>(null);
  const botAbortRef = useRef<AbortController | null>(null);
  const requestSeqRef = useRef(0);
  const botRequestSeqRef = useRef(0);
  const errorRef = useRef<HTMLDivElement | null>(null);
  const signalRef = useRef<HTMLDivElement | null>(null);
  const backtestRef = useRef<HTMLDivElement | null>(null);
  const tradeRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    writeJson(STORAGE_KEY, form);
  }, [form]);

  useEffect(() => {
    const token = apiToken.trim();
    if (!token) removeSessionKey(SESSION_TOKEN_KEY);
    else writeSessionString(SESSION_TOKEN_KEY, token);
  }, [apiToken]);

  const toastTimerRef = useRef<number | null>(null);
  const showToast = useCallback((msg: string) => {
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    setToast(msg);
    toastTimerRef.current = window.setTimeout(() => setToast(null), 1800);
  }, []);

  useEffect(() => {
    return () => {
      if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
      abortRef.current?.abort();
      botAbortRef.current?.abort();
    };
  }, []);

  const authHeaders = useMemo(() => {
    const token = apiToken.trim();
    return token ? { Authorization: `Bearer ${token}` } : undefined;
  }, [apiToken]);

  useEffect(() => {
    let mounted = true;
    health({ timeoutMs: 3000 })
      .then(() => {
        if (!mounted) return;
        setApiOk("ok");
      })
      .catch(() => {
        if (!mounted) return;
        setApiOk("down");
      });
    return () => {
      mounted = false;
    };
  }, []);

  const params: ApiParams = useMemo(() => {
    const base: ApiParams = {
      binanceSymbol: form.binanceSymbol.trim() || undefined,
      market: form.market,
      interval: form.interval.trim() || undefined,
      bars: clamp(Math.trunc(form.bars), 1, 1000),
      method: form.method,
      threshold: Math.max(0, form.threshold),
      fee: Math.max(0, form.fee),
      backtestRatio: clamp(form.backtestRatio, 0.01, 0.99),
      normalization: form.normalization,
      epochs: clamp(Math.trunc(form.epochs), 0, 5000),
      hiddenSize: clamp(Math.trunc(form.hiddenSize), 1, 512),
      binanceTestnet: form.binanceTestnet,
    };

    if (form.optimizeOperations) base.optimizeOperations = true;
    if (form.sweepThreshold) base.sweepThreshold = true;
    if (form.binanceLive) base.binanceLive = true;

    if (form.orderQuantity > 0) base.orderQuantity = form.orderQuantity;
    if (form.orderQuote > 0) base.orderQuote = form.orderQuote;

    return base;
  }, [form]);

  const scrollToResult = useCallback((kind: RequestKind) => {
    const ref = kind === "signal" ? signalRef : kind === "backtest" ? backtestRef : tradeRef;
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  const run = useCallback(
    async (kind: RequestKind, overrideParams?: ApiParams, opts?: RunOptions) => {
      const requestId = ++requestSeqRef.current;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      if (!opts?.silent) scrollToResult(kind);
      setState((s) => ({ ...s, loading: true, error: opts?.silent ? s.error : null, lastKind: kind }));

      try {
        const p = overrideParams ?? params;
        if (!p.binanceSymbol) throw new Error("binanceSymbol is required.");
        if (!p.interval) throw new Error("interval is required.");

        if (kind === "signal") {
          const out = await signal(p, { signal: controller.signal, headers: authHeaders, timeoutMs: SIGNAL_TIMEOUT_MS });
          if (requestId !== requestSeqRef.current) return;
          setState((s) => ({ ...s, latestSignal: out, trade: null, loading: false, error: null }));
          setApiOk("ok");
          if (!opts?.silent) showToast("Signal updated");
        } else if (kind === "backtest") {
          const out = await backtest(p, { signal: controller.signal, headers: authHeaders, timeoutMs: BACKTEST_TIMEOUT_MS });
          if (requestId !== requestSeqRef.current) return;
          setState((s) => ({ ...s, backtest: out, latestSignal: out.latestSignal, trade: null, loading: false, error: null }));
          setApiOk("ok");
          if (!opts?.silent) showToast("Backtest complete");
        } else {
          if (!form.tradeArmed) throw new Error("Trading is locked. Enable “Arm trading” to call /trade.");
          const out = await trade(p, { signal: controller.signal, headers: authHeaders, timeoutMs: TRADE_TIMEOUT_MS });
          if (requestId !== requestSeqRef.current) return;
          setState((s) => ({ ...s, trade: out, latestSignal: out.signal, loading: false, error: null }));
          setApiOk("ok");
          if (!opts?.silent) showToast(out.order.sent ? "Order sent" : "No order");
        }
      } catch (e) {
        if (requestId !== requestSeqRef.current) return;
        if (isAbortError(e)) return;

        let msg = e instanceof Error ? e.message : String(e);
        if (isTimeoutError(e)) msg = "Request timed out. Reduce bars/epochs or increase timeouts.";
        if (e instanceof HttpError && typeof e.payload === "string") {
          const payload = e.payload;
          if (payload.includes("ECONNREFUSED") || payload.includes("connect ECONNREFUSED")) {
            msg = `Backend unreachable. Start it with: cd haskell && cabal run -v0 trader-hs -- --serve --port ${API_PORT}`;
          }
        }

        setApiOk((prev) => {
          if (e instanceof HttpError && (e.status === 401 || e.status === 403)) return "auth";
          const looksDown = msg.toLowerCase().includes("fetch") || (e instanceof HttpError && e.status >= 500) || isTimeoutError(e);
          return looksDown ? "down" : prev;
        });

        if (opts?.silent) {
          setState((s) => ({ ...s, loading: false }));
          return;
        }

        setState((s) => ({ ...s, loading: false, error: msg }));
        showToast("Request failed");
      } finally {
        if (requestId === requestSeqRef.current) abortRef.current = null;
      }
    },
    [authHeaders, form.tradeArmed, params, scrollToResult, showToast],
  );

  const refreshBot = useCallback(
    async (opts?: RunOptions) => {
      const requestId = ++botRequestSeqRef.current;
      botAbortRef.current?.abort();
      const controller = new AbortController();
      botAbortRef.current = controller;

      if (!opts?.silent) setBot((s) => ({ ...s, loading: true, error: null }));

      try {
        const out = await botStatus({ signal: controller.signal, headers: authHeaders, timeoutMs: 10_000 });
        if (requestId !== botRequestSeqRef.current) return;
        setBot((s) => ({ ...s, loading: false, error: null, status: out }));
      } catch (e) {
        if (requestId !== botRequestSeqRef.current) return;
        if (isAbortError(e)) return;
        const msg = e instanceof Error ? e.message : String(e);
        setBot((s) => ({ ...s, loading: false, error: msg }));
      } finally {
        if (requestId === botRequestSeqRef.current) botAbortRef.current = null;
      }
    },
    [authHeaders],
  );

  const startLiveBot = useCallback(async () => {
    setBot((s) => ({ ...s, loading: true, error: null }));
    try {
      const payload: ApiParams = {
        ...params,
        botTrade: form.tradeArmed,
        botOnlineEpochs: 1,
        botMaxPoints: clamp(Math.trunc(form.bars), 100, 100000),
      };
      const out = await botStart(payload, { headers: authHeaders, timeoutMs: BOT_START_TIMEOUT_MS });
      setBot((s) => ({ ...s, loading: false, error: null, status: out }));
      showToast(out.running ? (form.tradeArmed ? "Live bot started (trading armed)" : "Live bot started (paper mode)") : "Bot not running");
    } catch (e) {
      if (isAbortError(e)) return;
      const msg = e instanceof Error ? e.message : String(e);
      setBot((s) => ({ ...s, loading: false, error: msg }));
      showToast("Bot start failed");
    }
  }, [authHeaders, form.bars, form.tradeArmed, params, showToast]);

  const stopLiveBot = useCallback(async () => {
    setBot((s) => ({ ...s, loading: true, error: null }));
    try {
      const out = await botStop({ headers: authHeaders, timeoutMs: 30_000 });
      setBot((s) => ({ ...s, loading: false, error: null, status: out }));
      showToast("Bot stopped");
    } catch (e) {
      if (isAbortError(e)) return;
      const msg = e instanceof Error ? e.message : String(e);
      setBot((s) => ({ ...s, loading: false, error: msg }));
      showToast("Bot stop failed");
    }
  }, [authHeaders, showToast]);

  useEffect(() => {
    void refreshBot({ silent: true });
  }, [refreshBot]);

  useEffect(() => {
    if (!bot.status.running) return;
    const t = window.setInterval(() => {
      if (bot.loading) return;
      void refreshBot({ silent: true });
    }, 2000);
    return () => window.clearInterval(t);
  }, [bot.loading, bot.status.running, refreshBot]);

  useEffect(() => {
    if (!form.autoRefresh || apiOk !== "ok") return;
    const ms = clamp(form.autoRefreshSec, 5, 600) * 1000;
    const t = window.setInterval(() => {
      if (state.loading) return;
      void run("signal", undefined, { silent: true });
    }, ms);
    return () => window.clearInterval(t);
  }, [apiOk, form.autoRefresh, form.autoRefreshSec, run, state.loading]);

  useEffect(() => {
    if (!state.error) return;
    errorRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [state.error]);

  const statusDotClass =
    apiOk === "ok" ? "dot dotOk" : apiOk === "down" ? "dot dotBad" : "dot dotWarn";
  const statusLabel =
    apiOk === "ok"
      ? "API online"
      : apiOk === "auth"
        ? "API auth required"
        : apiOk === "down"
          ? "API unreachable"
          : "API status unknown";

  const curlFor = useMemo(() => {
    const kind = state.lastKind ?? (form.optimizeOperations || form.sweepThreshold ? "backtest" : "signal");
    const endpoint = kind === "signal" ? "/signal" : kind === "backtest" ? "/backtest" : "/trade";
    const json = JSON.stringify(params);
    const safe = escapeSingleQuotes(json);
    const token = apiToken.trim();
    const auth = token ? ` -H 'Authorization: Bearer ${escapeSingleQuotes(token)}'` : "";
    return `curl -s -X POST ${API_TARGET}${endpoint} -H 'Content-Type: application/json'${auth} -d '${safe}'`;
  }, [apiToken, form.optimizeOperations, form.sweepThreshold, params, state.lastKind]);

  return (
    <div className="container">
      <header className="header">
        <div className="brand">
          <div className="logo" aria-hidden="true" />
          <div className="title">
            <h1>Trader UI</h1>
            <p>Configure, backtest, optimize, and trade via the local REST API.</p>
          </div>
        </div>
        <div className="pillRow" aria-live="polite">
          <span className="pill">
            <span className={statusDotClass} aria-hidden="true" />
            {statusLabel}
          </span>
          {state.loading ? (
            <span className="pill">
              <span className="dot dotWarn" aria-hidden="true" />
              Working…
            </span>
          ) : null}
          <span className="pill">
            Proxy: <span style={{ fontFamily: "var(--mono)" }}>/api → {API_TARGET}</span>
          </span>
        </div>
      </header>
      {toast ? (
        <div className="toastFixed" role="status" aria-live="polite" aria-atomic="true">
          {toast}
        </div>
      ) : null}

      <div className="grid">
        <section className="card configCard">
          <div className="cardHeader">
            <h2 className="cardTitle">Configuration</h2>
            <p className="cardSubtitle">Safe defaults, minimal knobs, and clear outputs.</p>
          </div>
          <div className="cardBody">
            <div className="row" style={{ gridTemplateColumns: "1fr" }}>
              <div className="field">
                <label className="label" htmlFor="apiToken">
                  API token (optional)
                </label>
                <div className="row" style={{ gridTemplateColumns: "1fr auto", alignItems: "center" }}>
                  <input
                    id="apiToken"
                    className="input"
                    type="password"
                    value={apiToken}
                    onChange={(e) => setApiToken(e.target.value)}
                    placeholder="TRADER_API_TOKEN"
                    spellCheck={false}
                    autoCapitalize="none"
                    autoCorrect="off"
                    inputMode="text"
                  />
                  <button className="btn" type="button" onClick={() => setApiToken("")} disabled={!apiToken.trim()}>
                    Clear
                  </button>
                </div>
                <div className="hint">Only needed when the backend sets TRADER_API_TOKEN. Stored in session storage (not in the URL).</div>
              </div>
            </div>

            <div className="row">
              <div className="field">
                <label className="label" htmlFor="symbol">
                  Binance symbol
                </label>
                <input
                  id="symbol"
                  className="input"
                  value={form.binanceSymbol}
                  onChange={(e) => setForm((f) => ({ ...f, binanceSymbol: e.target.value.toUpperCase() }))}
                  placeholder="BTCUSDT"
                  spellCheck={false}
                />
                <div className="hint">Use a spot symbol like BTCUSDT (USDT-margined futures also use the same symbol).</div>
              </div>

              <div className="field">
                <label className="label" htmlFor="market">
                  Market
                </label>
                <select
                  id="market"
                  className="select"
                  value={form.market}
                  onChange={(e) =>
                    setForm((f) => {
                      const market = e.target.value as Market;
                      return { ...f, market, binanceTestnet: market === "margin" ? false : f.binanceTestnet };
                    })
                  }
                >
                  <option value="spot">Spot</option>
                  <option value="margin">Margin</option>
                  <option value="futures">Futures (USDT-M)</option>
                </select>
                <div className="hint">Margin orders require live mode. Futures can close positions via reduce-only.</div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 12 }}>
              <div className="field">
                <label className="label" htmlFor="interval">
                  Interval
                </label>
                <input
                  id="interval"
                  className="input"
                  value={form.interval}
                  onChange={(e) => setForm((f) => ({ ...f, interval: e.target.value }))}
                  placeholder="1h"
                  spellCheck={false}
                />
                <div className="hint">Examples: 1m, 5m, 1h, 1d.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="bars">
                  Bars (max 1000)
                </label>
                <input
                  id="bars"
                  className="input"
                  type="number"
                  min={1}
                  max={1000}
                  value={form.bars}
                  onChange={(e) => setForm((f) => ({ ...f, bars: numFromInput(e.target.value, f.bars) }))}
                />
                <div className="hint">Larger values take longer (more training + longer backtest).</div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 12 }}>
              <div className="field">
                <label className="label" htmlFor="method">
                  Method
                </label>
                <select
                  id="method"
                  className="select"
                  value={form.method}
                  onChange={(e) => setForm((f) => ({ ...f, method: e.target.value as Method }))}
                >
                  <option value="11">11 — Both (agreement gated)</option>
                  <option value="10">10 — Kalman only</option>
                  <option value="01">01 — LSTM only</option>
                </select>
                <div className="hint">“11” only trades when both models agree on direction (up/down) outside the threshold.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="threshold">
                  Threshold (fraction)
                </label>
                <input
                  id="threshold"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={form.threshold}
                  onChange={(e) => setForm((f) => ({ ...f, threshold: numFromInput(e.target.value, f.threshold) }))}
                />
                <div className="hint">Deadband for “neutral”. Default 0.001 = 0.1%.</div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 12 }}>
              <div className="field">
                <label className="label" htmlFor="backtestRatio">
                  Backtest ratio
                </label>
                <input
                  id="backtestRatio"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0.01}
                  max={0.99}
                  value={form.backtestRatio}
                  onChange={(e) => setForm((f) => ({ ...f, backtestRatio: numFromInput(e.target.value, f.backtestRatio) }))}
                />
                <div className="hint">Time-split holdout (last portion). Train and backtest are different.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="fee">
                  Fee (fraction)
                </label>
                <input
                  id="fee"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={form.fee}
                  onChange={(e) => setForm((f) => ({ ...f, fee: numFromInput(e.target.value, f.fee) }))}
                />
                <div className="hint">Applied when switching position (long ↔ flat).</div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 12 }}>
              <div className="field">
                <label className="label" htmlFor="norm">
                  Normalization
                </label>
                <select
                  id="norm"
                  className="select"
                  value={form.normalization}
                  onChange={(e) => setForm((f) => ({ ...f, normalization: e.target.value as Normalization }))}
                >
                  <option value="standard">standard</option>
                  <option value="minmax">minmax</option>
                  <option value="log">log</option>
                  <option value="none">none</option>
                </select>
                <div className="hint">Used for the LSTM pipeline.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="epochs">
                  Epochs / Hidden size
                </label>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                  <input
                    id="epochs"
                    className="input"
                    type="number"
                    min={0}
                    value={form.epochs}
                    onChange={(e) => setForm((f) => ({ ...f, epochs: numFromInput(e.target.value, f.epochs) }))}
                  />
                  <input
                    aria-label="Hidden size"
                    className="input"
                    type="number"
                    min={1}
                    value={form.hiddenSize}
                    onChange={(e) => setForm((f) => ({ ...f, hiddenSize: numFromInput(e.target.value, f.hiddenSize) }))}
                  />
                </div>
                <div className="hint">Higher = slower. For quick iteration, reduce epochs.</div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 12 }}>
              <div className="field">
                <label className="label">Optimization</label>
                <div className="pillRow">
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.sweepThreshold}
                      onChange={(e) => setForm((f) => ({ ...f, sweepThreshold: e.target.checked, optimizeOperations: false }))}
                    />
                    Sweep threshold
                  </label>
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.optimizeOperations}
                      onChange={(e) => setForm((f) => ({ ...f, optimizeOperations: e.target.checked, sweepThreshold: false }))}
                    />
                    Optimize operations (method + threshold)
                  </label>
                </div>
                <div className="hint">Runs on the backtest split and selects the best final equity.</div>
              </div>
              <div className="field">
                <label className="label">Options</label>
                <div className="pillRow">
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.binanceTestnet}
                      disabled={form.market === "margin"}
                      onChange={(e) =>
                        setForm((f) => ({
                          ...f,
                          binanceTestnet: f.market === "margin" ? false : e.target.checked,
                        }))
                      }
                    />
                    Testnet (spot/futures)
                  </label>
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.autoRefresh}
                      onChange={(e) => setForm((f) => ({ ...f, autoRefresh: e.target.checked }))}
                    />
                    Auto-refresh
                  </label>
                </div>
                <div className="hint">
                  Auto-refresh every{" "}
                  <input
                    className="input"
                    style={{ height: 32, width: 84, padding: "0 10px", margin: "0 8px" }}
                    type="number"
                    min={5}
                    max={600}
                    value={form.autoRefreshSec}
                    onChange={(e) => setForm((f) => ({ ...f, autoRefreshSec: numFromInput(e.target.value, f.autoRefreshSec) }))}
                  />{" "}
                  seconds.
                </div>
              </div>
            </div>

            <div className="actions">
              <button className="btn btnPrimary" disabled={state.loading} onClick={() => run("signal")}>
                {state.loading && state.lastKind === "signal" ? "Getting signal…" : "Get signal"}
              </button>
              <button className="btn" disabled={state.loading} onClick={() => run("backtest")}>
                {state.loading && state.lastKind === "backtest" ? "Running backtest…" : "Run backtest"}
              </button>
              <button
                className="btn"
                disabled={state.loading}
                onClick={() => {
                  const p = { ...params, sweepThreshold: true, optimizeOperations: false };
                  setForm((f) => ({ ...f, sweepThreshold: true, optimizeOperations: false }));
                  void run("backtest", p);
                }}
              >
                {state.loading && state.lastKind === "backtest" ? "Optimizing…" : "Optimize threshold"}
              </button>
              <button
                className="btn"
                disabled={state.loading}
                onClick={() => {
                  const p = { ...params, optimizeOperations: true, sweepThreshold: false };
                  setForm((f) => ({ ...f, optimizeOperations: true, sweepThreshold: false }));
                  void run("backtest", p);
                }}
              >
                {state.loading && state.lastKind === "backtest" ? "Optimizing…" : "Optimize operations"}
              </button>
            </div>

            <div style={{ marginTop: 14 }}>
              <div className="row" style={{ gridTemplateColumns: "1fr" }}>
                <div className="field">
                  <label className="label">Live bot</label>
                  <div className="actions" style={{ marginTop: 0 }}>
                    <button
                      className="btn btnPrimary"
                      disabled={bot.loading || bot.status.running || state.loading}
                      onClick={startLiveBot}
                      title={form.tradeArmed ? "Trading armed (will send orders)" : "Paper mode (no orders)"}
                    >
                      {bot.loading ? "Starting…" : bot.status.running ? "Running" : "Start live bot"}
                    </button>
                    <button className="btn" disabled={bot.loading || !bot.status.running} onClick={stopLiveBot}>
                      Stop bot
                    </button>
                    <button className="btn" disabled={bot.loading} onClick={() => refreshBot()}>
                      Refresh
                    </button>
                  </div>
                  <div className="hint">
                    Continuously ingests new bars, fine-tunes on each bar, and switches position based on the latest signal. Enable “Arm trading” to actually place
                    Binance orders; otherwise it runs in paper mode. If “Sweep threshold” or “Optimize operations” is enabled, the bot re-optimizes after each
                    buy/sell operation.
                  </div>
                  {bot.error ? <div className="hint" style={{ color: "rgba(239, 68, 68, 0.9)" }}>{bot.error}</div> : null}
                </div>
              </div>
            </div>

            <div style={{ marginTop: 14 }}>
              <div className="row">
                <div className="field">
                  <label className="label">Trade controls</label>
                  <div className="pillRow">
                    <label className="pill">
                      <input
                        type="checkbox"
                        checked={form.binanceLive}
                        onChange={(e) => setForm((f) => ({ ...f, binanceLive: e.target.checked }))}
                      />
                      Live orders
                    </label>
                    <label className="pill">
                      <input
                        type="checkbox"
                        checked={form.tradeArmed}
                        onChange={(e) => setForm((f) => ({ ...f, tradeArmed: e.target.checked }))}
                      />
                      Arm trading
                    </label>
                  </div>
                  <div className="hint">Trading is disabled by default. Only arm it when you’re ready.</div>
                </div>
                <div className="field">
                  <label className="label" htmlFor="orderQuote">
                    Order sizing
                  </label>
                  <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                    <input
                      id="orderQuote"
                      className="input"
                      type="number"
                      min={0}
                      value={form.orderQuote}
                      onChange={(e) => setForm((f) => ({ ...f, orderQuote: numFromInput(e.target.value, f.orderQuote) }))}
                      placeholder="orderQuote"
                    />
                    <input
                      className="input"
                      type="number"
                      min={0}
                      value={form.orderQuantity}
                      onChange={(e) => setForm((f) => ({ ...f, orderQuantity: numFromInput(e.target.value, f.orderQuantity) }))}
                      placeholder="orderQuantity"
                    />
                  </div>
                  <div className="hint">Set one: quote for BUY (spot), or quantity (required for futures in many cases).</div>
                </div>
              </div>

              <div className="actions" style={{ marginTop: 10 }}>
                <button
                  className="btn btnDanger"
                  disabled={state.loading || !form.tradeArmed}
                  onClick={() => run("trade")}
                  title={form.binanceLive ? "LIVE order mode enabled" : "Test order mode (default)"}
                >
                  {state.loading && state.lastKind === "trade" ? "Trading…" : "Trade (uses latest signal)"}
                </button>
                <button
                  className="btn"
                  disabled={state.loading}
                  onClick={() => {
                    abortRef.current?.abort();
                    setState((s) => ({ ...s, loading: false }));
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>

            <p className="footerNote">
              Backend: start with{" "}
              <span style={{ fontFamily: "var(--mono)" }}>
                cd haskell && cabal run -v0 trader-hs -- --serve --port {API_PORT}
              </span>
              . The UI uses a
              same-origin dev proxy (<span style={{ fontFamily: "var(--mono)" }}>/api</span>) to avoid CORS and reduce local attack surface.
            </p>
          </div>
        </section>

        <section className="resultGrid">
          {state.error ? (
            <div className="card" ref={errorRef}>
              <div className="cardHeader">
                <h2 className="cardTitle">Error</h2>
                <p className="cardSubtitle">Fix the request or backend and try again.</p>
              </div>
              <div className="cardBody">
                <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)" }}>
                  {state.error}
                </pre>
              </div>
            </div>
          ) : null}

          <div className="card">
            <div className="cardHeader">
              <h2 className="cardTitle">Live bot</h2>
              <p className="cardSubtitle">Non-stop loop (server-side): fetches new bars, updates the model each bar, and records each buy/sell operation.</p>
            </div>
            <div className="cardBody">
              {bot.status.running ? (
                <>
                  <div className="pillRow" style={{ marginBottom: 10 }}>
                    <span className="badge">{bot.status.symbol}</span>
                    <span className="badge">{bot.status.interval}</span>
                    <span className="badge">{marketLabel(bot.status.market)}</span>
                    <span className="badge">{methodLabel(bot.status.method)}</span>
                    <span className="badge">thr {fmtPct(bot.status.threshold, 3)}</span>
                    <span className="badge">{bot.status.error ? "Error" : "OK"}</span>
                  </div>

                  <BacktestChart
                    prices={bot.status.prices}
                    equityCurve={bot.status.equityCurve}
                    positions={bot.status.positions}
                    trades={bot.status.trades}
                    operations={bot.status.operations}
                    backtestStartIndex={bot.status.startIndex}
                    height={360}
                  />

                  <div className="kv" style={{ marginTop: 12 }}>
                    <div className="k">Equity / Position</div>
                    <div className="v">
                      {fmtRatio(bot.status.equityCurve[bot.status.equityCurve.length - 1] ?? 1, 4)}x /{" "}
                      {(bot.status.positions[bot.status.positions.length - 1] ?? 0) === 1 ? "LONG" : "FLAT"}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Last action</div>
                    <div className="v">{bot.status.latestSignal.action}</div>
                  </div>
                  {bot.status.lastOrder ? (
                    <div className="kv">
                      <div className="k">Last order</div>
                      <div className="v">{bot.status.lastOrder.message}</div>
                    </div>
                  ) : null}
                </>
              ) : (
                <div className="hint">Bot is stopped. Use “Start live bot” on the left.</div>
              )}
            </div>
          </div>

          <div className="card" ref={signalRef}>
            <div className="cardHeader">
              <h2 className="cardTitle">Latest signal</h2>
              <p className="cardSubtitle">{state.latestSignal ? "Computed from the most recent bar." : "Run “Get signal” or “Run backtest” to populate."}</p>
            </div>
            <div className="cardBody">
              {state.latestSignal ? (
                <>
                  <div className="pillRow" style={{ marginBottom: 10 }}>
                    <span className={actionBadgeClass(state.latestSignal.action)}>{state.latestSignal.action}</span>
                    <span className="badge">{methodLabel(state.latestSignal.method)}</span>
                    <span className="badge">{marketLabel(form.market)}</span>
                  </div>
                  <div className="kv">
                    <div className="k">Current price</div>
                    <div className="v">{fmtMoney(state.latestSignal.currentPrice, 4)}</div>
                  </div>
                  <div className="kv">
                    <div className="k">Threshold</div>
                    <div className="v">
                      {fmtNum(state.latestSignal.threshold, 6)} ({fmtPct(state.latestSignal.threshold, 3)})
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Kalman</div>
                    <div className="v">
                      {state.latestSignal.kalmanNext ? fmtMoney(state.latestSignal.kalmanNext, 4) : "disabled"}{" "}
                      {state.latestSignal.kalmanDirection ? `(${state.latestSignal.kalmanDirection})` : ""}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">LSTM</div>
                    <div className="v">
                      {state.latestSignal.lstmNext ? fmtMoney(state.latestSignal.lstmNext, 4) : "disabled"}{" "}
                      {state.latestSignal.lstmDirection ? `(${state.latestSignal.lstmDirection})` : ""}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Chosen</div>
                    <div className="v">{state.latestSignal.chosenDirection ?? "NEUTRAL"}</div>
                  </div>
                </>
              ) : (
                <div className="hint">No signal yet.</div>
              )}
            </div>
          </div>

          <div className="card" ref={backtestRef}>
            <div className="cardHeader">
              <h2 className="cardTitle">Backtest summary</h2>
              <p className="cardSubtitle">Uses a time split (train vs held-out backtest) to avoid lookahead.</p>
            </div>
            <div className="cardBody">
              {state.backtest ? (
                <>
                  <BacktestChart
                    prices={state.backtest.prices}
                    equityCurve={state.backtest.equityCurve}
                    positions={state.backtest.positions}
                    agreementOk={state.backtest.agreementOk}
                    trades={state.backtest.trades}
                    backtestStartIndex={state.backtest.split.backtestStartIndex}
                    height={360}
                  />
                  <div className="pillRow" style={{ marginBottom: 10, marginTop: 12 }}>
                    <span className="badge">Train: {state.backtest.split.train}</span>
                    <span className="badge">Backtest: {state.backtest.split.backtest}</span>
                    <span className="badge">Holdout: {fmtPct(state.backtest.split.backtestRatio, 1)}</span>
                    <span className="badge">{methodLabel(state.backtest.method)}</span>
                  </div>

                  <div className="kv">
                    <div className="k">Best threshold</div>
                    <div className="v">
                      {fmtNum(state.backtest.threshold, 6)} ({fmtPct(state.backtest.threshold, 3)})
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Final equity</div>
                    <div className="v">{fmtRatio(state.backtest.metrics.finalEquity, 4)}</div>
                  </div>
                  <div className="kv">
                    <div className="k">Total return</div>
                    <div className="v">{fmtPct(state.backtest.metrics.totalReturn, 2)}</div>
                  </div>
                  <div className="kv">
                    <div className="k">Annualized return</div>
                    <div className="v">{fmtPct(state.backtest.metrics.annualizedReturn, 2)}</div>
                  </div>
                  <div className="kv">
                    <div className="k">Sharpe / Max DD</div>
                    <div className="v">
                      {fmtNum(state.backtest.metrics.sharpe, 3)} / {fmtPct(state.backtest.metrics.maxDrawdown, 2)}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Exposure / Turnover</div>
                    <div className="v">
                      {fmtPct(state.backtest.metrics.exposure, 1)} / {fmtNum(state.backtest.metrics.turnover, 4)}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Trade count / Win rate</div>
                    <div className="v">
                      {state.backtest.metrics.tradeCount} / {fmtPct(state.backtest.metrics.winRate, 1)}
                    </div>
                  </div>
                </>
              ) : (
                <div className="hint">No backtest yet.</div>
              )}
            </div>
          </div>

          <div className="card" ref={tradeRef}>
            <div className="cardHeader">
              <h2 className="cardTitle">Trade result</h2>
              <p className="cardSubtitle">Only populated after calling /trade. Keys must be provided via env or CLI.</p>
            </div>
            <div className="cardBody">
              {state.trade ? (
                <>
                  <div className="pillRow" style={{ marginBottom: 10 }}>
                    <span className={actionBadgeClass(state.trade.signal.action)}>{state.trade.signal.action}</span>
                    <span className="badge">{state.trade.order.sent ? "Order sent" : "No order"}</span>
                    <span className="badge">{state.trade.order.mode ?? "—"}</span>
                    <span className="badge">{state.trade.order.side ?? "—"}</span>
                  </div>
                  <pre className="code">{JSON.stringify(state.trade, null, 2)}</pre>
                </>
              ) : (
                <div className="hint">No trade attempt yet.</div>
              )}
            </div>
          </div>

          <div className="card">
            <div className="cardHeader">
              <h2 className="cardTitle">Request preview</h2>
              <p className="cardSubtitle">This JSON is what the UI sends to the API.</p>
            </div>
            <div className="cardBody">
              <div className="actions" style={{ marginTop: 0, marginBottom: 10 }}>
                <button
                  className="btn"
                  disabled={state.loading}
                  onClick={async () => {
                    await copyText(JSON.stringify(params, null, 2));
                    showToast("Copied JSON");
                  }}
                >
                  Copy JSON
                </button>
                <button
                  className="btn"
                  disabled={state.loading}
                  onClick={async () => {
                    await copyText(curlFor);
                    showToast("Copied curl");
                  }}
                >
                  Copy curl
                </button>
                <button
                  className="btn"
                  disabled={state.loading}
                  onClick={() => {
                    setForm(defaultForm);
                    showToast("Reset to defaults");
                  }}
                >
                  Reset
                </button>
              </div>
              <pre className="code">{JSON.stringify(params, null, 2)}</pre>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
