import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  ApiParams,
  ApiTradeResponse,
  BacktestResponse,
  BinanceKeysStatus,
  BinanceListenKeyResponse,
  BotOrderEvent,
  BotStatus,
  IntrabarFill,
  LatestSignal,
  Market,
  Method,
  Normalization,
  Positioning,
} from "./lib/types";
import {
  HttpError,
  backtest,
  binanceKeysStatus,
  binanceListenKey,
  binanceListenKeyClose,
  binanceListenKeyKeepAlive,
  botStart,
  botStatus,
  botStop,
  cacheClear,
  cacheStats,
  health,
  signal,
  trade,
} from "./lib/api";
import { copyText } from "./lib/clipboard";
import { TRADER_UI_CONFIG } from "./lib/deployConfig";
import { readJson, readLocalString, readSessionString, removeLocalKey, removeSessionKey, writeJson, writeLocalString, writeSessionString } from "./lib/storage";
import { fmtMoney, fmtNum, fmtPct, fmtRatio } from "./lib/format";
import { API_PORT, API_TARGET } from "./app/apiTarget";
import {
  BACKTEST_TIMEOUT_MS,
  BINANCE_INTERVALS,
  BINANCE_INTERVAL_SET,
  BOT_START_TIMEOUT_MS,
  BOT_STATUS_TAIL_POINTS,
  BOT_STATUS_TIMEOUT_MS,
  BOT_AUTOSTART_RETRY_MS,
  BOT_TELEMETRY_POINTS,
  DATA_LOG_COLLAPSED_MAX_LINES,
  RATE_LIMIT_BASE_MS,
  RATE_LIMIT_MAX_MS,
  RATE_LIMIT_TOAST_MIN_MS,
  SESSION_BINANCE_KEY_KEY,
  SESSION_BINANCE_SECRET_KEY,
  SIGNAL_TIMEOUT_MS,
  STORAGE_KEY,
  STORAGE_ORDER_LOG_PREFS_KEY,
  STORAGE_PERSIST_SECRETS_KEY,
  STORAGE_PROFILES_KEY,
  TRADE_TIMEOUT_MS,
  TUNE_OBJECTIVES,
} from "./app/constants";
import { binanceIntervalSeconds, defaultForm, normalizeFormState, parseDurationSeconds, type FormState, type FormStateJson } from "./app/formState";
import {
  actionBadgeClass,
  clamp,
  escapeSingleQuotes,
  firstReason,
  fmtDurationMs,
  fmtEtaMs,
  fmtProfitFactor,
  fmtTimeMs,
  generateIdempotencyKey,
  indexTopLevelPrimitiveArrays,
  isAbortError,
  isLikelyOrderError,
  isLocalHostname,
  isTimeoutError,
  marketLabel,
  methodLabel,
  normalizeApiBaseUrlInput,
  numFromInput,
} from "./app/utils";
import { BacktestChart } from "./components/BacktestChart";
import { PredictionDiffChart } from "./components/PredictionDiffChart";
import { TelemetryChart } from "./components/TelemetryChart";
import { TopCombosChart, type OptimizationCombo } from "./components/TopCombosChart";

type RequestKind = "signal" | "backtest" | "trade";

type RunOptions = {
  silent?: boolean;
};

type ActiveAsyncJob = {
  kind: RequestKind;
  jobId: string | null;
  startedAtMs: number;
};

type RateLimitState = {
  untilMs: number;
  reason: string;
  lastHitAtMs: number;
};

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

type BotRtEvent = {
  atMs: number;
  message: string;
};

type BotTelemetryPoint = {
  atMs: number;
  pollLatencyMs: number | null;
  driftBps: number | null;
};

type BotRtUiState = {
  lastFetchAtMs: number | null;
  lastFetchDurationMs: number | null;
  lastNewCandles: number;
  lastNewCandlesAtMs: number | null;
  lastKlineUpdates: number;
  lastKlineUpdatesAtMs: number | null;
  telemetry: BotTelemetryPoint[];
  feed: BotRtEvent[];
};

type KeysUiState = {
  loading: boolean;
  error: string | null;
  status: BinanceKeysStatus | null;
  checkedAtMs: number | null;
};

type CacheUiState = {
  loading: boolean;
  error: string | null;
  stats: Awaited<ReturnType<typeof cacheStats>> | null;
};

type ListenKeyUiState = {
  loading: boolean;
  error: string | null;
  info: BinanceListenKeyResponse | null;
  wsStatus: "disconnected" | "connecting" | "connected";
  wsError: string | null;
  lastEventAtMs: number | null;
  lastEvent: string | null;
  keepAliveAtMs: number | null;
  keepAliveError: string | null;
};

type OrderSideFilter = "ALL" | "BUY" | "SELL";

type OrderLogPrefs = {
  filterText: string;
  sentOnly: boolean;
  side: OrderSideFilter;
  limit: number;
  errorsOnly: boolean;
  showOrderId: boolean;
  showStatus: boolean;
  showClientOrderId: boolean;
};

type SavedProfiles = Record<string, FormState>;

type PendingProfileLoad = {
  name: string;
  profile: FormState;
  reasons: string[];
};

export function App() {
  const [apiOk, setApiOk] = useState<"unknown" | "ok" | "down" | "auth">("unknown");
  const [healthInfo, setHealthInfo] = useState<Awaited<ReturnType<typeof health>> | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [revealSecrets, setRevealSecrets] = useState(false);
  const [persistSecrets, setPersistSecrets] = useState<boolean>(() => readJson<boolean>(STORAGE_PERSIST_SECRETS_KEY) ?? false);
  const deployApiBaseUrl = TRADER_UI_CONFIG.apiBaseUrl;
  const apiToken = TRADER_UI_CONFIG.apiToken;
  const [binanceApiKey, setBinanceApiKey] = useState<string>(() => {
    const persisted = readLocalString(SESSION_BINANCE_KEY_KEY) ?? "";
    const session = readSessionString(SESSION_BINANCE_KEY_KEY) ?? "";
    return persistSecrets ? persisted || session : session;
  });
  const [binanceApiSecret, setBinanceApiSecret] = useState<string>(() => {
    const persisted = readLocalString(SESSION_BINANCE_SECRET_KEY) ?? "";
    const session = readSessionString(SESSION_BINANCE_SECRET_KEY) ?? "";
    return persistSecrets ? persisted || session : session;
  });
  const [form, setForm] = useState<FormState>(() => normalizeFormState(readJson<FormStateJson>(STORAGE_KEY)));

  const [profiles, setProfiles] = useState<SavedProfiles>(() => {
    const raw = readJson<Record<string, FormStateJson>>(STORAGE_PROFILES_KEY) ?? {};
    const out: SavedProfiles = {};
    for (const [name, profile] of Object.entries(raw)) out[name] = normalizeFormState(profile);
    return out;
  });
  const [profileName, setProfileName] = useState("");
  const [profileSelected, setProfileSelected] = useState("");
  const [pendingProfileLoad, setPendingProfileLoad] = useState<PendingProfileLoad | null>(null);

  const [confirmLive, setConfirmLive] = useState(false);
  const [confirmArm, setConfirmArm] = useState(false);
  const [pendingMarket, setPendingMarket] = useState<Market | null>(null);

  const orderPrefsInit = readJson<OrderLogPrefs>(STORAGE_ORDER_LOG_PREFS_KEY);
  const [state, setState] = useState<UiState>({
    loading: false,
    error: null,
    lastKind: null,
    latestSignal: null,
    backtest: null,
    trade: null,
  });

  const [activeAsyncJob, setActiveAsyncJob] = useState<ActiveAsyncJob | null>(null);
  const activeAsyncJobRef = useRef<ActiveAsyncJob | null>(null);
  const [activeAsyncTickMs, setActiveAsyncTickMs] = useState(() => Date.now());
  const [rateLimit, setRateLimit] = useState<RateLimitState | null>(null);
  const [rateLimitTickMs, setRateLimitTickMs] = useState(() => Date.now());
  const rateLimitRef = useRef<RateLimitState | null>(null);
  const rateLimitBackoffRef = useRef(RATE_LIMIT_BASE_MS);
  const rateLimitToastAtRef = useRef<number | null>(null);

  const [bot, setBot] = useState<BotUiState>({
    loading: false,
    error: null,
    status: { running: false },
  });

  const [botRt, setBotRt] = useState<BotRtUiState>({
    lastFetchAtMs: null,
    lastFetchDurationMs: null,
    lastNewCandles: 0,
    lastNewCandlesAtMs: null,
    lastKlineUpdates: 0,
    lastKlineUpdatesAtMs: null,
    telemetry: [],
    feed: [],
  });
  const botRtRef = useRef<{
    botKey: string | null;
    lastOpenTimeMs: number | null;
    lastError: string | null;
    lastHalted: boolean | null;
    lastFetchedOpenTimeMs: number | null;
    lastFetchedClose: number | null;
    lastMethod: Method | null;
    lastOpenThreshold: number | null;
    lastCloseThreshold: number | null;
    lastTradeEnabled: boolean | null;
    lastTelemetryPolledAtMs: number | null;
  }>({
    botKey: null,
    lastOpenTimeMs: null,
    lastError: null,
    lastHalted: null,
    lastFetchedOpenTimeMs: null,
    lastFetchedClose: null,
    lastMethod: null,
    lastOpenThreshold: null,
    lastCloseThreshold: null,
    lastTradeEnabled: null,
    lastTelemetryPolledAtMs: null,
  });
  const botStatusFetchedRef = useRef(false);
  const botAutoStartSuppressedRef = useRef(false);
  const botAutoStartRef = useRef<{ lastAttemptAtMs: number }>({ lastAttemptAtMs: 0 });

  const [keys, setKeys] = useState<KeysUiState>({
    loading: false,
    error: null,
    status: null,
    checkedAtMs: null,
  });

  const [cacheUi, setCacheUi] = useState<CacheUiState>({ loading: false, error: null, stats: null });

  const [listenKeyUi, setListenKeyUi] = useState<ListenKeyUiState>({
    loading: false,
    error: null,
    info: null,
    wsStatus: "disconnected",
    wsError: null,
    lastEventAtMs: null,
    lastEvent: null,
    keepAliveAtMs: null,
    keepAliveError: null,
  });
  const listenKeyWsRef = useRef<WebSocket | null>(null);
  const listenKeyKeepAliveTimerRef = useRef<number | null>(null);

  const [orderFilterText, setOrderFilterText] = useState(() => orderPrefsInit?.filterText ?? "");
  const [orderSentOnly, setOrderSentOnly] = useState(() => orderPrefsInit?.sentOnly ?? false);
  const [orderErrorsOnly, setOrderErrorsOnly] = useState(() => orderPrefsInit?.errorsOnly ?? false);
  const [orderSideFilter, setOrderSideFilter] = useState<OrderSideFilter>(() => orderPrefsInit?.side ?? "ALL");
  const [orderLimit, setOrderLimit] = useState(() => orderPrefsInit?.limit ?? 200);
  const [orderShowOrderId, setOrderShowOrderId] = useState(() => orderPrefsInit?.showOrderId ?? false);
  const [orderShowStatus, setOrderShowStatus] = useState(() => orderPrefsInit?.showStatus ?? false);
  const [orderShowClientOrderId, setOrderShowClientOrderId] = useState(() => orderPrefsInit?.showClientOrderId ?? false);

  const [dataLog, setDataLog] = useState<Array<{ timestamp: number; label: string; data: unknown }>>([]);
  const [dataLogExpanded, setDataLogExpanded] = useState(false);
  const [dataLogIndexArrays, setDataLogIndexArrays] = useState(true);
  const [topCombos, setTopCombos] = useState<OptimizationCombo[]>([]);
  const [topCombosLoading, setTopCombosLoading] = useState(true);
  const [topCombosError, setTopCombosError] = useState<string | null>(null);
  const [selectedComboId, setSelectedComboId] = useState<number | null>(null);
  const dataLogRef = useRef<HTMLDivElement | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const botAbortRef = useRef<AbortController | null>(null);
  const keysAbortRef = useRef<AbortController | null>(null);
  const requestSeqRef = useRef(0);
  const botRequestSeqRef = useRef(0);
  const keysRequestSeqRef = useRef(0);
  const errorRef = useRef<HTMLDivElement | null>(null);
  const signalRef = useRef<HTMLDivElement | null>(null);
  const backtestRef = useRef<HTMLDivElement | null>(null);
  const tradeRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    writeJson(STORAGE_KEY, form);
  }, [form]);

  useEffect(() => {
    writeJson(STORAGE_PROFILES_KEY, profiles);
  }, [profiles]);

  useEffect(() => {
    writeJson(STORAGE_ORDER_LOG_PREFS_KEY, {
      filterText: orderFilterText,
      sentOnly: orderSentOnly,
      errorsOnly: orderErrorsOnly,
      side: orderSideFilter,
      limit: orderLimit,
      showOrderId: orderShowOrderId,
      showStatus: orderShowStatus,
      showClientOrderId: orderShowClientOrderId,
    } satisfies OrderLogPrefs);
  }, [
    orderErrorsOnly,
    orderFilterText,
    orderLimit,
    orderSentOnly,
    orderShowClientOrderId,
    orderShowOrderId,
    orderShowStatus,
    orderSideFilter,
  ]);

  useEffect(() => {
    const v = binanceApiKey.trim();
    if (persistSecrets) {
      if (!v) removeLocalKey(SESSION_BINANCE_KEY_KEY);
      else writeLocalString(SESSION_BINANCE_KEY_KEY, v);
      removeSessionKey(SESSION_BINANCE_KEY_KEY);
    } else {
      if (!v) removeSessionKey(SESSION_BINANCE_KEY_KEY);
      else writeSessionString(SESSION_BINANCE_KEY_KEY, v);
      removeLocalKey(SESSION_BINANCE_KEY_KEY);
    }
  }, [binanceApiKey, persistSecrets]);

  useEffect(() => {
    const v = binanceApiSecret.trim();
    if (persistSecrets) {
      if (!v) removeLocalKey(SESSION_BINANCE_SECRET_KEY);
      else writeLocalString(SESSION_BINANCE_SECRET_KEY, v);
      removeSessionKey(SESSION_BINANCE_SECRET_KEY);
    } else {
      if (!v) removeSessionKey(SESSION_BINANCE_SECRET_KEY);
      else writeSessionString(SESSION_BINANCE_SECRET_KEY, v);
      removeLocalKey(SESSION_BINANCE_SECRET_KEY);
    }
  }, [binanceApiSecret, persistSecrets]);

  useEffect(() => {
    writeJson(STORAGE_PERSIST_SECRETS_KEY, persistSecrets);
  }, [persistSecrets]);

  const toastTimerRef = useRef<number | null>(null);
  const showToast = useCallback((msg: string) => {
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    setToast(msg);
    toastTimerRef.current = window.setTimeout(() => setToast(null), 1800);
  }, []);

  const clearRateLimit = useCallback(() => {
    setRateLimit(null);
    rateLimitBackoffRef.current = RATE_LIMIT_BASE_MS;
  }, []);

  const applyRateLimit = useCallback(
    (err: HttpError, opts?: RunOptions) => {
      const now = Date.now();
      const retryAfterMs =
        typeof err.retryAfterMs === "number" && Number.isFinite(err.retryAfterMs) ? Math.max(0, err.retryAfterMs) : null;
      const baseBackoffMs = Math.max(RATE_LIMIT_BASE_MS, rateLimitBackoffRef.current);
      const delayMs = Math.min(RATE_LIMIT_MAX_MS, Math.max(retryAfterMs ?? 0, baseBackoffMs));
      const untilMs = Math.max(now + delayMs, rateLimitRef.current?.untilMs ?? 0);
      const reason =
        retryAfterMs != null && retryAfterMs > 0
          ? `Rate limited by API. Retry-After ${fmtDurationMs(retryAfterMs)}.`
          : "Rate limited by API.";

      setRateLimit({ untilMs, reason, lastHitAtMs: now });
      rateLimitBackoffRef.current = Math.min(RATE_LIMIT_MAX_MS, Math.round(Math.max(baseBackoffMs, retryAfterMs ?? 0) * 1.6));

      const lastToastAt = rateLimitToastAtRef.current ?? 0;
      if (!opts?.silent || now - lastToastAt >= RATE_LIMIT_TOAST_MIN_MS) {
        showToast(`Rate limited. Try again ${fmtEtaMs(Math.max(0, untilMs - now))}.`);
        rateLimitToastAtRef.current = now;
      }

      return untilMs;
    },
    [showToast],
  );

  const handleComboSelect = useCallback(
    (combo: OptimizationCombo) => {
      setForm((prev) => {
        const openThr = combo.openThreshold ?? prev.openThreshold;
        const closeThr = combo.closeThreshold ?? openThr ?? prev.closeThreshold;
        return {
          ...prev,
          interval: combo.params.interval,
          bars: combo.params.bars,
          method: combo.params.method,
          positioning: combo.params.positioning ?? prev.positioning,
          normalization: combo.params.normalization,
          fee: combo.params.fee ?? prev.fee,
          epochs: Math.max(0, Math.trunc(combo.params.epochs)),
          hiddenSize: Math.max(1, Math.trunc(combo.params.hiddenSize)),
          learningRate: combo.params.learningRate,
          valRatio: combo.params.valRatio,
          patience: combo.params.patience,
          gradClip: combo.params.gradClip ?? 0,
          slippage: combo.params.slippage,
          spread: combo.params.spread,
          intrabarFill: combo.params.intrabarFill ?? prev.intrabarFill,
          stopLoss: combo.params.stopLoss ?? 0,
          takeProfit: combo.params.takeProfit ?? 0,
          trailingStop: combo.params.trailingStop ?? 0,
          minHoldBars: combo.params.minHoldBars ?? prev.minHoldBars,
          cooldownBars: combo.params.cooldownBars ?? prev.cooldownBars,
          maxDrawdown: combo.params.maxDrawdown ?? 0,
          maxDailyLoss: combo.params.maxDailyLoss ?? 0,
          maxOrderErrors: combo.params.maxOrderErrors ?? 0,
          kalmanZMin: combo.params.kalmanZMin ?? prev.kalmanZMin,
          kalmanZMax: combo.params.kalmanZMax ?? prev.kalmanZMax,
          maxHighVolProb: combo.params.maxHighVolProb ?? 0,
          maxConformalWidth: combo.params.maxConformalWidth ?? 0,
          maxQuantileWidth: combo.params.maxQuantileWidth ?? 0,
          confirmConformal: combo.params.confirmConformal ?? prev.confirmConformal,
          confirmQuantiles: combo.params.confirmQuantiles ?? prev.confirmQuantiles,
          confidenceSizing: combo.params.confidenceSizing ?? prev.confidenceSizing,
          minPositionSize: combo.params.minPositionSize ?? 0,
          openThreshold: openThr,
          closeThreshold: closeThr,
        };
      });
      setSelectedComboId(combo.id);
      showToast(`Loaded optimizer combo #${combo.id}`);
    },
    [showToast],
  );

  useEffect(() => {
    activeAsyncJobRef.current = activeAsyncJob;
  }, [activeAsyncJob]);

  useEffect(() => {
    if (!state.loading) return;
    setActiveAsyncTickMs(Date.now());
    const t = window.setInterval(() => setActiveAsyncTickMs(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [state.loading]);

  useEffect(() => {
    rateLimitRef.current = rateLimit;
  }, [rateLimit]);

  useEffect(() => {
    if (!rateLimit) return;
    const delayMs = Math.max(0, rateLimit.untilMs - Date.now());
    if (delayMs === 0) {
      setRateLimit(null);
      rateLimitBackoffRef.current = RATE_LIMIT_BASE_MS;
      return;
    }
    const t = window.setTimeout(() => {
      setRateLimit(null);
      rateLimitBackoffRef.current = RATE_LIMIT_BASE_MS;
    }, delayMs);
    return () => window.clearTimeout(t);
  }, [rateLimit]);

  useEffect(() => {
    if (!rateLimit) return;
    setRateLimitTickMs(Date.now());
    const t = window.setInterval(() => setRateLimitTickMs(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, [rateLimit]);

  const profileNames = useMemo(() => Object.keys(profiles).sort((a, b) => a.localeCompare(b)), [profiles]);

  const saveProfile = useCallback(() => {
    const name = profileName.trim();
    if (!name) {
      showToast("Profile name is required");
      return;
    }
    const existed = Object.prototype.hasOwnProperty.call(profiles, name);
    setProfiles((prev) => ({ ...prev, [name]: form }));
    setProfileSelected(name);
    setProfileName("");
    showToast(existed ? `Profile updated: ${name}` : `Profile saved: ${name}`);
  }, [form, profileName, profiles, showToast]);

  const deleteProfile = useCallback(() => {
    const name = profileSelected.trim();
    if (!name) return;
    setProfiles((prev) => {
      const next = { ...prev };
      delete next[name];
      return next;
    });
    setProfileSelected("");
    setPendingProfileLoad(null);
    showToast(`Profile deleted: ${name}`);
  }, [profileSelected, showToast]);

  const requestLoadProfile = useCallback(() => {
    const name = profileSelected.trim();
    if (!name) {
      showToast("Select a profile to load");
      return;
    }
    const raw = profiles[name];
    if (!raw) {
      showToast("Profile not found");
      return;
    }

    const profile: FormState =
      raw.market === "margin"
        ? { ...raw, binanceTestnet: false, binanceLive: true }
        : raw;
    const reasons: string[] = [];
    if (profile.market === "margin" && form.market !== "margin") reasons.push("switch market to Margin");
    if (profile.binanceLive && !form.binanceLive) reasons.push("enable Live orders");
    if (profile.tradeArmed && !form.tradeArmed) reasons.push("arm trading");

    if (reasons.length > 0) {
      setPendingProfileLoad({ name, profile, reasons });
      return;
    }

    setForm(profile);
    setPendingProfileLoad(null);
    showToast(`Profile loaded: ${name}`);
  }, [form.binanceLive, form.market, form.tradeArmed, profileSelected, profiles, showToast]);

  useEffect(() => {
    return () => {
      if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
      abortRef.current?.abort();
      botAbortRef.current?.abort();
      keysAbortRef.current?.abort();
      if (listenKeyKeepAliveTimerRef.current) window.clearInterval(listenKeyKeepAliveTimerRef.current);
      const ws = listenKeyWsRef.current;
      listenKeyWsRef.current = null;
      ws?.close();
    };
  }, []);

  const authHeaders = useMemo(() => {
    const token = apiToken.trim();
    if (!token) return undefined;
    return { Authorization: `Bearer ${token}`, "X-API-Key": token };
  }, [apiToken]);

  const apiBaseCandidate = useMemo(() => normalizeApiBaseUrlInput(deployApiBaseUrl), [deployApiBaseUrl]);

  const apiBaseError = useMemo(() => {
    const raw = deployApiBaseUrl.trim();
    if (!raw) return null;
    const candidate = apiBaseCandidate.trim();
    if (candidate.startsWith("/")) return null;
    if (/^https?:\/\//i.test(candidate)) {
      try {
        // Validate host-like input early to avoid confusing fetch errors later.
        new URL(candidate);
        return null;
      } catch {
        return "Configured apiBaseUrl must be a valid URL (e.g. https://your-api-host) or a path like /api";
      }
    }
    return "Configured apiBaseUrl must start with http(s):// or /api";
  }, [apiBaseCandidate, deployApiBaseUrl]);

  const apiBase = useMemo(() => {
    const raw = apiBaseCandidate.trim();
    if (raw && !apiBaseError) return raw.replace(/\/+$/, "");
    if (!import.meta.env.DEV && /^https?:\/\//.test(API_TARGET)) {
      try {
        const u = new URL(API_TARGET);
        if (u.hostname !== "localhost" && u.hostname !== "127.0.0.1") return API_TARGET.replace(/\/+$/, "");
      } catch {
        // ignore
      }
    }
    return "/api";
  }, [apiBaseCandidate, apiBaseError]);

  useEffect(() => {
    let mounted = true;
    health(apiBase, { timeoutMs: 10_000, headers: authHeaders })
      .then((out) => {
        if (!mounted) return;
        setHealthInfo(out);
        if (out.authRequired && out.authOk !== true) setApiOk("auth");
        else setApiOk("ok");
      })
      .catch(() => {
        if (!mounted) return;
        setHealthInfo(null);
        setApiOk("down");
      });
    return () => {
      mounted = false;
    };
  }, [apiBase, authHeaders]);

  useEffect(() => {
    // Prevent inconsistent state; margin requires live mode.
    if (form.market !== "margin") return;
    if (form.binanceLive) return;
    setForm((f) => ({ ...f, market: "spot" }));
    showToast("Margin requires Live orders (switched back to Spot)");
  }, [form.binanceLive, form.market, showToast]);

  const recheckHealth = useCallback(async () => {
    let h: Awaited<ReturnType<typeof health>>;
    try {
      h = await health(apiBase, { timeoutMs: 10_000, headers: authHeaders });
    } catch {
      setHealthInfo(null);
      setApiOk("down");
      showToast("API unreachable");
      return;
    }

    setHealthInfo(h);
    if (h.authRequired && h.authOk !== true) {
      setApiOk("auth");
      showToast(apiToken.trim() ? "API auth failed" : "API auth required");
      return;
    }

    try {
      await botStatus(apiBase, { timeoutMs: 10_000, headers: authHeaders });
      setApiOk("ok");
      showToast("API online");
    } catch (e) {
      if (isAbortError(e)) return;
      if (e instanceof HttpError && (e.status === 401 || e.status === 403)) {
        setApiOk("auth");
        showToast(apiToken.trim() ? "API auth failed" : "API auth required");
        return;
      }
      setApiOk("down");
      showToast(isTimeoutError(e) ? "API request timed out" : "API unreachable");
    }
  }, [apiBase, apiToken, authHeaders, showToast]);

  const refreshCacheStats = useCallback(async () => {
    setCacheUi((s) => ({ ...s, loading: true, error: null }));
    try {
      const v = await cacheStats(apiBase, { timeoutMs: 10_000, headers: authHeaders });
      setCacheUi({ loading: false, error: null, stats: v });
    } catch (e) {
      if (isAbortError(e)) return;
      if (e instanceof HttpError && (e.status === 401 || e.status === 403)) {
        setCacheUi((s) => ({ ...s, loading: false, error: "Unauthorized (check apiToken / TRADER_API_TOKEN)." }));
        return;
      }
      setCacheUi((s) => ({
        ...s,
        loading: false,
        error: isTimeoutError(e) ? "Request timed out" : "Failed to load cache stats",
      }));
    }
  }, [apiBase, authHeaders]);

  const clearCacheUi = useCallback(async () => {
    setCacheUi((s) => ({ ...s, loading: true, error: null }));
    try {
      await cacheClear(apiBase, { timeoutMs: 10_000, headers: authHeaders });
      showToast("Cache cleared");
      const v = await cacheStats(apiBase, { timeoutMs: 10_000, headers: authHeaders });
      setCacheUi({ loading: false, error: null, stats: v });
    } catch (e) {
      if (isAbortError(e)) return;
      if (e instanceof HttpError && (e.status === 401 || e.status === 403)) {
        setCacheUi((s) => ({ ...s, loading: false, error: "Unauthorized (check apiToken / TRADER_API_TOKEN)." }));
        return;
      }
      setCacheUi((s) => ({
        ...s,
        loading: false,
        error: isTimeoutError(e) ? "Request timed out" : "Failed to clear cache",
      }));
    }
  }, [apiBase, authHeaders, showToast]);

  const commonParams: ApiParams = useMemo(() => {
    const interval = form.interval.trim();
    const intervalOk = BINANCE_INTERVAL_SET.has(interval);
    const barsRaw = Math.trunc(form.bars);
    const bars = barsRaw <= 0 ? 0 : clamp(barsRaw, 2, 1000);
    const base: ApiParams = {
      binanceSymbol: form.binanceSymbol.trim() || undefined,
      market: form.market,
      interval: intervalOk ? interval : undefined,
      bars,
      method: form.method,
      ...(form.positioning !== "long-flat" ? { positioning: form.positioning } : {}),
      openThreshold: Math.max(0, form.openThreshold),
      closeThreshold: Math.max(0, form.closeThreshold),
      fee: Math.max(0, form.fee),
      ...(form.slippage > 0 ? { slippage: clamp(form.slippage, 0, 0.999999) } : {}),
      ...(form.spread > 0 ? { spread: clamp(form.spread, 0, 0.999999) } : {}),
      ...(form.intrabarFill !== "stop-first" ? { intrabarFill: form.intrabarFill } : {}),
      ...(form.stopLoss > 0 ? { stopLoss: clamp(form.stopLoss, 0, 0.999999) } : {}),
      ...(form.takeProfit > 0 ? { takeProfit: clamp(form.takeProfit, 0, 0.999999) } : {}),
      ...(form.trailingStop > 0 ? { trailingStop: clamp(form.trailingStop, 0, 0.999999) } : {}),
      ...(form.minHoldBars > 0 ? { minHoldBars: clamp(Math.trunc(form.minHoldBars), 0, 1_000_000) } : {}),
      ...(form.cooldownBars > 0 ? { cooldownBars: clamp(Math.trunc(form.cooldownBars), 0, 1_000_000) } : {}),
      ...(form.maxDrawdown > 0 ? { maxDrawdown: clamp(form.maxDrawdown, 0, 0.999999) } : {}),
      ...(form.maxDailyLoss > 0 ? { maxDailyLoss: clamp(form.maxDailyLoss, 0, 0.999999) } : {}),
      ...(form.maxOrderErrors >= 1 ? { maxOrderErrors: clamp(Math.trunc(form.maxOrderErrors), 1, 1_000_000) } : {}),
      backtestRatio: clamp(form.backtestRatio, 0.01, 0.99),
      tuneRatio: clamp(form.tuneRatio, 0, 0.99),
      tuneObjective: form.tuneObjective,
      tunePenaltyMaxDrawdown: Math.max(0, form.tunePenaltyMaxDrawdown),
      tunePenaltyTurnover: Math.max(0, form.tunePenaltyTurnover),
      minRoundTrips: clamp(Math.trunc(form.minRoundTrips), 0, 1_000_000),
      walkForwardFolds: clamp(Math.trunc(form.walkForwardFolds), 1, 1000),
      normalization: form.normalization,
      epochs: clamp(Math.trunc(form.epochs), 0, 5000),
      hiddenSize: clamp(Math.trunc(form.hiddenSize), 1, 512),
      lr: Math.max(1e-9, form.learningRate),
      valRatio: clamp(form.valRatio, 0, 1),
      patience: clamp(Math.trunc(form.patience), 0, 1000),
      ...(form.gradClip > 0 ? { gradClip: clamp(form.gradClip, 0, 100) } : {}),
      kalmanZMin: Math.max(0, form.kalmanZMin),
      kalmanZMax: Math.max(Math.max(0, form.kalmanZMin), form.kalmanZMax),
      binanceTestnet: form.binanceTestnet,
    };

    if (form.lookbackBars >= 2) base.lookbackBars = Math.trunc(form.lookbackBars);
    else if (form.lookbackWindow.trim()) base.lookbackWindow = form.lookbackWindow.trim();

    if (form.optimizeOperations) base.optimizeOperations = true;
    if (form.sweepThreshold) base.sweepThreshold = true;

    if (form.maxHighVolProb > 0) base.maxHighVolProb = clamp(form.maxHighVolProb, 0, 1);
    if (form.maxConformalWidth > 0) base.maxConformalWidth = Math.max(0, form.maxConformalWidth);
    if (form.maxQuantileWidth > 0) base.maxQuantileWidth = Math.max(0, form.maxQuantileWidth);
    if (form.confirmConformal) base.confirmConformal = true;
    if (form.confirmQuantiles) base.confirmQuantiles = true;
    if (form.confidenceSizing) base.confidenceSizing = true;
    if (form.minPositionSize > 0) base.minPositionSize = clamp(form.minPositionSize, 0, 1);

    return base;
  }, [form]);

  const estimatedCosts = useMemo(() => {
    const fee = Math.max(0, form.fee);
    const slippage = Math.max(0, form.slippage);
    const spread = Math.max(0, form.spread);
    const perSide = Math.min(0.999999, fee + slippage + spread / 2);
    const roundTrip = Math.min(0.999999, perSide * 2);
    const denom = Math.max(1e-12, (1 - perSide) * (1 - perSide));
    const breakEven = Math.max(0, 1 / denom - 1);
    return { perSide, roundTrip, breakEven };
  }, [form.fee, form.slippage, form.spread]);

  const tradeParams: ApiParams = useMemo(() => {
    const base: ApiParams = { ...commonParams };
    if (form.binanceLive) base.binanceLive = true;
    const k = form.idempotencyKey.trim();
    const idOk = !k || (k.length <= 36 && /^[A-Za-z0-9_-]+$/.test(k));
    if (k && idOk) base.idempotencyKey = k;

    if (form.orderQuantity > 0) base.orderQuantity = form.orderQuantity;
    else if (form.orderQuote > 0) base.orderQuote = form.orderQuote;
    else if (form.orderQuoteFraction > 0 && form.orderQuoteFraction <= 1) {
      base.orderQuoteFraction = form.orderQuoteFraction;
      if (form.maxOrderQuote > 0) base.maxOrderQuote = Math.max(0, form.maxOrderQuote);
    }

    return base;
  }, [
    commonParams,
    form.binanceLive,
    form.idempotencyKey,
    form.maxOrderQuote,
    form.orderQuantity,
    form.orderQuote,
    form.orderQuoteFraction,
  ]);

  const withBinanceKeys = useCallback(
    (p: ApiParams): ApiParams => {
      const key = binanceApiKey.trim();
      const secret = binanceApiSecret.trim();
      if (!key && !secret) return p;
      return {
        ...p,
        ...(key ? { binanceApiKey: key } : {}),
        ...(secret ? { binanceApiSecret: secret } : {}),
      };
    },
    [binanceApiKey, binanceApiSecret],
  );

  const keysParams: ApiParams = useMemo(() => {
    const base: ApiParams = {
      binanceSymbol: form.binanceSymbol.trim() || undefined,
      market: form.market,
      binanceTestnet: form.binanceTestnet,
    };

    const k = form.idempotencyKey.trim();
    const idOk = !k || (k.length <= 36 && /^[A-Za-z0-9_-]+$/.test(k));
    if (k && idOk) base.idempotencyKey = k;

    if (form.orderQuantity > 0) base.orderQuantity = form.orderQuantity;
    else if (form.orderQuote > 0) base.orderQuote = form.orderQuote;
    else if (form.orderQuoteFraction > 0 && form.orderQuoteFraction <= 1) {
      base.orderQuoteFraction = form.orderQuoteFraction;
      if (form.maxOrderQuote > 0) base.maxOrderQuote = Math.max(0, form.maxOrderQuote);
    }

    return withBinanceKeys(base);
  }, [
    form.binanceSymbol,
    form.binanceTestnet,
    form.idempotencyKey,
    form.market,
    form.maxOrderQuote,
    form.orderQuantity,
    form.orderQuote,
    form.orderQuoteFraction,
    withBinanceKeys,
  ]);

  const botOrdersView = useMemo(() => {
    const st = bot.status;
    if (!st.running) return { total: 0, shown: [] as BotOrderEvent[], startIndex: 0 };

    const total = st.orders.length;
    let shown = st.orders;

    const limit = clamp(Math.trunc(orderLimit), 1, 2000);
    shown = shown.slice(-limit);

    if (orderSentOnly) shown = shown.filter((e) => e.order.sent);
    if (orderErrorsOnly) shown = shown.filter((e) => isLikelyOrderError(e.order.message, e.order.sent, e.order.status));
    if (orderSideFilter !== "ALL") shown = shown.filter((e) => e.opSide === orderSideFilter);

    const q = orderFilterText.trim().toLowerCase();
    if (q) {
      shown = shown.filter((e) => {
        const hay = [
          e.opSide,
          e.order.mode ?? "",
          e.order.side ?? "",
          e.order.symbol ?? "",
          e.order.status ?? "",
          e.order.orderId ?? "",
          e.order.clientOrderId ?? "",
          e.order.message ?? "",
        ]
          .join(" ")
          .toLowerCase();
        return hay.includes(q);
      });
    }

    return { total, shown, startIndex: st.startIndex };
  }, [bot.status, orderErrorsOnly, orderFilterText, orderLimit, orderSentOnly, orderSideFilter]);

  const botOrderCopyText = useMemo(() => {
    const st = bot.status;
    if (!st.running) return "";
    const rows = botOrdersView.shown.map((e) => {
      const bar = st.startIndex + e.index;
      const sent = e.order.sent ? "SENT" : "NO";
      const mode = e.order.mode ?? "—";
      return `${fmtTimeMs(e.atMs)} | bar ${bar} | ${e.opSide} @ ${fmtMoney(e.price, 4)} | ${sent} ${mode} | ${e.order.message}`;
    });
    return rows.length ? rows.join("\n") : "No live operations yet.";
  }, [bot.status, botOrdersView.shown]);

  const botRtFeedText = useMemo(() => {
    if (botRt.feed.length === 0) return "No realtime events yet.";
    return botRt.feed.map((e) => `${fmtTimeMs(e.atMs)} | ${e.message}`).join("\n");
  }, [botRt.feed]);

  const botRisk = useMemo(() => {
    const st = bot.status;
    if (!st.running) return null;
    const lastEq = st.equityCurve[st.equityCurve.length - 1] ?? 1;
    const peak = st.peakEquity || 1;
    const dayStart = st.dayStartEquity || 1;
    const dd = peak > 0 ? Math.max(0, 1 - lastEq / peak) : 0;
    const dl = dayStart > 0 ? Math.max(0, 1 - lastEq / dayStart) : 0;
    return { lastEq, dd, dl };
  }, [bot.status]);

  const botRealtime = useMemo(() => {
    const st = bot.status;
    if (!st.running) return null;
    const now = Date.now();
    const processedOpenTime = st.openTimes[st.openTimes.length - 1] ?? null;
    const fetchedLast = st.fetchedLastKline ?? null;
    const fetchedOpenTime = fetchedLast?.openTime ?? null;
    const candleOpenTime = fetchedOpenTime ?? processedOpenTime;
    const intervalSec = binanceIntervalSeconds(st.interval) ?? parseDurationSeconds(st.interval);
    const intervalMs = typeof intervalSec === "number" && Number.isFinite(intervalSec) && intervalSec > 0 ? intervalSec * 1000 : null;
    const expectedCloseMs = candleOpenTime !== null && intervalMs !== null ? candleOpenTime + intervalMs : null;
    const closeEtaMs = expectedCloseMs !== null ? expectedCloseMs - now : null;
    const statusAgeMs = typeof st.updatedAtMs === "number" && Number.isFinite(st.updatedAtMs) ? now - st.updatedAtMs : null;
    const polledAtMs = typeof st.polledAtMs === "number" && Number.isFinite(st.polledAtMs) ? st.polledAtMs : null;
    const pollLatencyMs = typeof st.pollLatencyMs === "number" && Number.isFinite(st.pollLatencyMs) ? st.pollLatencyMs : null;
    const pollAgeMs = polledAtMs !== null ? now - polledAtMs : null;
    const fetchedKlines = typeof st.fetchedKlines === "number" && Number.isFinite(st.fetchedKlines) ? st.fetchedKlines : null;
    const pollSeconds =
      typeof st.settings?.pollSeconds === "number" && Number.isFinite(st.settings.pollSeconds) ? Math.max(0, st.settings.pollSeconds) : null;
    const nextPollEtaMs = polledAtMs !== null && pollSeconds !== null ? polledAtMs + pollSeconds * 1000 - now : null;
    const candleAgeMs = candleOpenTime !== null ? now - candleOpenTime : null;
    const procLagMs = processedOpenTime !== null ? st.updatedAtMs - processedOpenTime : null;
    const closeLagMs = expectedCloseMs !== null ? st.updatedAtMs - expectedCloseMs : null;
    const pollCloseLagMs = polledAtMs !== null && expectedCloseMs !== null ? polledAtMs - expectedCloseMs : null;
    const lastBatchAtMs = typeof st.lastBatchAtMs === "number" && Number.isFinite(st.lastBatchAtMs) ? st.lastBatchAtMs : null;
    const lastBatchSize = typeof st.lastBatchSize === "number" && Number.isFinite(st.lastBatchSize) ? st.lastBatchSize : null;
    const lastBatchMs = typeof st.lastBatchMs === "number" && Number.isFinite(st.lastBatchMs) ? st.lastBatchMs : null;
    const batchAgeMs = lastBatchAtMs !== null ? now - lastBatchAtMs : null;
    const batchPerBarMs = lastBatchMs !== null && lastBatchSize && lastBatchSize > 0 ? lastBatchMs / lastBatchSize : null;
    const lastBarIndex = st.startIndex + Math.max(0, st.prices.length - 1);
    const processedClose = st.prices[st.prices.length - 1] ?? null;
    const fetchedClose = typeof fetchedLast?.close === "number" && Number.isFinite(fetchedLast.close) ? fetchedLast.close : null;
    const closeDelta = typeof processedClose === "number" && Number.isFinite(processedClose) && typeof fetchedClose === "number" ? fetchedClose - processedClose : null;
    const closeDeltaPct =
      closeDelta !== null && typeof processedClose === "number" && Number.isFinite(processedClose) && processedClose !== 0 ? closeDelta / processedClose : null;
    const behindCandles =
      fetchedOpenTime !== null && processedOpenTime !== null && intervalMs !== null
        ? Math.max(0, Math.round((fetchedOpenTime - processedOpenTime) / intervalMs))
        : null;
    return {
      now,
      processedOpenTime,
      fetchedOpenTime,
      candleOpenTime,
      expectedCloseMs,
      closeEtaMs,
      statusAgeMs,
      polledAtMs,
      pollLatencyMs,
      pollAgeMs,
      fetchedKlines,
      pollSeconds,
      nextPollEtaMs,
      candleAgeMs,
      procLagMs,
      closeLagMs,
      pollCloseLagMs,
      lastBatchAtMs,
      lastBatchSize,
      lastBatchMs,
      batchAgeMs,
      batchPerBarMs,
      lastBarIndex,
      processedClose,
      fetchedClose,
      closeDelta,
      closeDeltaPct,
      behindCandles,
      fetchedLast,
    };
  }, [bot.status]);

  const scrollToResult = useCallback((kind: RequestKind) => {
    const ref = kind === "signal" ? signalRef : kind === "backtest" ? backtestRef : tradeRef;
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  const run = useCallback(
    async (kind: RequestKind, overrideParams?: ApiParams, opts?: RunOptions) => {
      const now = Date.now();
      const activeLimit = rateLimitRef.current;
      if (activeLimit && now < activeLimit.untilMs) {
        if (!opts?.silent) {
          showToast(`Rate limited. Try again ${fmtEtaMs(Math.max(0, activeLimit.untilMs - now))}.`);
        }
        return;
      }

      const requestId = ++requestSeqRef.current;
      const startedAtMs = Date.now();
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      if (!opts?.silent) {
        scrollToResult(kind);
        setState((s) => ({ ...s, loading: true, error: null, lastKind: kind }));
        setActiveAsyncJob({ kind, jobId: null, startedAtMs });
      }

      try {
        const p = overrideParams ?? (kind === "trade" ? tradeParams : commonParams);
        if (!p.binanceSymbol) throw new Error("binanceSymbol is required.");
        if (!p.interval) throw new Error("interval is required.");
        const requestHeaders = form.bypassCache ? { ...(authHeaders ?? {}), "Cache-Control": "no-cache" } : authHeaders;

        if (kind === "signal") {
          const out = await signal(apiBase, p, {
            signal: controller.signal,
            headers: requestHeaders,
            timeoutMs: SIGNAL_TIMEOUT_MS,
            onJobId: (jobId) => {
              if (requestId !== requestSeqRef.current) return;
              setActiveAsyncJob({ kind, jobId, startedAtMs });
            },
          });
          if (requestId !== requestSeqRef.current) return;
          if (p.optimizeOperations || p.sweepThreshold) {
            const openThreshold = out.openThreshold ?? out.threshold;
            const closeThreshold = out.closeThreshold ?? out.openThreshold ?? out.threshold;
            setForm((f) => ({
              ...f,
              ...(p.optimizeOperations ? { method: out.method } : {}),
              openThreshold: Math.max(0, openThreshold),
              closeThreshold: Math.max(0, closeThreshold),
            }));
          }
          if (opts?.silent) setState((s) => ({ ...s, latestSignal: out }));
          else {
            setDataLog((logs) => [...logs, { timestamp: Date.now(), label: "Signal Response", data: out }].slice(-100));
            setState((s) => ({ ...s, latestSignal: out, trade: null, loading: false, error: null }));
          }
          setApiOk("ok");
          if (!opts?.silent) showToast("Signal updated");
        } else if (kind === "backtest") {
          const out = await backtest(apiBase, p, {
            signal: controller.signal,
            headers: requestHeaders,
            timeoutMs: BACKTEST_TIMEOUT_MS,
            onJobId: (jobId) => {
              if (requestId !== requestSeqRef.current) return;
              setActiveAsyncJob({ kind, jobId, startedAtMs });
            },
          });
          if (requestId !== requestSeqRef.current) return;
          if (p.optimizeOperations || p.sweepThreshold) {
            const openThreshold = out.openThreshold ?? out.threshold;
            const closeThreshold = out.closeThreshold ?? out.openThreshold ?? out.threshold;
            setForm((f) => ({
              ...f,
              ...(p.optimizeOperations ? { method: out.method } : {}),
              openThreshold: Math.max(0, openThreshold),
              closeThreshold: Math.max(0, closeThreshold),
            }));
          }
          setState((s) => ({ ...s, backtest: out, latestSignal: out.latestSignal, trade: null, loading: false, error: null }));
          setDataLog((logs) => [...logs, { timestamp: Date.now(), label: "Backtest Response", data: out }].slice(-100));
          setApiOk("ok");
          if (!opts?.silent) showToast("Backtest complete");
        } else {
          if (!form.tradeArmed) throw new Error("Trading is locked. Enable “Arm trading” to call /trade.");
          const out = await trade(apiBase, withBinanceKeys(p), {
            signal: controller.signal,
            headers: requestHeaders,
            timeoutMs: TRADE_TIMEOUT_MS,
            onJobId: (jobId) => {
              if (requestId !== requestSeqRef.current) return;
              setActiveAsyncJob({ kind, jobId, startedAtMs });
            },
          });
          if (requestId !== requestSeqRef.current) return;
          if (p.optimizeOperations || p.sweepThreshold) {
            const sig = out.signal;
            const openThreshold = sig.openThreshold ?? sig.threshold;
            const closeThreshold = sig.closeThreshold ?? sig.openThreshold ?? sig.threshold;
            setForm((f) => ({
              ...f,
              ...(p.optimizeOperations ? { method: sig.method } : {}),
              openThreshold: Math.max(0, openThreshold),
              closeThreshold: Math.max(0, closeThreshold),
            }));
          }
          setState((s) => ({ ...s, trade: out, latestSignal: out.signal, loading: false, error: null }));
          setDataLog((logs) => [...logs, { timestamp: Date.now(), label: "Trade Response", data: out }].slice(-100));
          setApiOk("ok");
          if (!opts?.silent) showToast(out.order.sent ? "Order sent" : "No order");
        }
        clearRateLimit();
      } catch (e) {
        if (requestId !== requestSeqRef.current) return;
        if (isAbortError(e)) return;

        let msg = e instanceof Error ? e.message : String(e);
        let showErrorToast = true;
        if (isTimeoutError(e)) msg = "Request timed out. Reduce bars/epochs or increase timeouts.";
        if (e instanceof HttpError && typeof e.payload === "string") {
          const payload = e.payload;
          if (payload.includes("ECONNREFUSED") || payload.includes("connect ECONNREFUSED")) {
            msg = `Backend unreachable. Start it with: cd haskell && cabal run -v0 trader-hs -- --serve --port ${API_PORT}`;
          }
        }
        if (e instanceof HttpError && e.status === 504) {
          msg = apiBase.startsWith("/api")
            ? "CloudFront `/api/*` proxy timed out (504). Point `/api/*` at your API origin (App Runner/ALB/etc) and allow POST/OPTIONS, or set apiBaseUrl in trader-config.js to https://<your-api-host>."
            : "API gateway timed out (504). Try again, or reduce bars/epochs, or scale the API.";
        }
        if (e instanceof HttpError && e.status === 429) {
          const untilMs = applyRateLimit(e, opts);
          msg = `Rate limited. Try again ${fmtEtaMs(Math.max(0, untilMs - Date.now()))}.`;
          showErrorToast = false;
        }

        setApiOk((prev) => {
          if (e instanceof HttpError && (e.status === 401 || e.status === 403)) return "auth";
          const looksDown = msg.toLowerCase().includes("fetch") || (e instanceof HttpError && e.status >= 500);
          return looksDown ? "down" : prev;
        });

        if (opts?.silent) {
          if (e instanceof HttpError && e.status === 400) {
            setForm((f) => (f.autoRefresh ? { ...f, autoRefresh: false } : f));
            const short = msg.replaceAll("\n", " ");
            showToast(`Auto-refresh paused: ${short.length > 140 ? `${short.slice(0, 137)}...` : short}`);
          }
          setState((s) => ({ ...s, loading: false }));
          return;
        }

        setState((s) => ({ ...s, loading: false, error: msg }));
        if (showErrorToast) showToast("Request failed");
      } finally {
        if (requestId === requestSeqRef.current) {
          abortRef.current = null;
          setActiveAsyncJob(null);
        }
      }
    },
    [
      apiBase,
      applyRateLimit,
      authHeaders,
      clearRateLimit,
      commonParams,
      form.bypassCache,
      form.tradeArmed,
      scrollToResult,
      showToast,
      tradeParams,
      withBinanceKeys,
    ],
  );

  const cancelActiveRequest = useCallback(() => {
    if (!state.loading) return;
    const job = activeAsyncJobRef.current;
    abortRef.current?.abort();
    abortRef.current = null;
    setActiveAsyncJob(null);
    setState((s) => ({ ...s, loading: false }));
    showToast(job?.jobId ? `Cancel requested (${job.jobId})` : "Cancel requested");
  }, [showToast, state.loading]);

  const refreshKeys = useCallback(
    async (opts?: RunOptions) => {
      const requestId = ++keysRequestSeqRef.current;
      keysAbortRef.current?.abort();
      const controller = new AbortController();
      keysAbortRef.current = controller;

      setKeys((s) => ({ ...s, loading: true, error: opts?.silent ? s.error : null }));

      try {
        const p = keysParams;
        if (!p.binanceSymbol) throw new Error("binanceSymbol is required.");

        const out = await binanceKeysStatus(apiBase, p, { signal: controller.signal, headers: authHeaders, timeoutMs: 30_000 });
        if (requestId !== keysRequestSeqRef.current) return;
        setKeys({ loading: false, error: null, status: out, checkedAtMs: Date.now() });
        setApiOk("ok");
        if (!opts?.silent) showToast("Key status updated");
      } catch (e) {
        if (requestId !== keysRequestSeqRef.current) return;
        if (isAbortError(e)) return;

        let msg = e instanceof Error ? e.message : String(e);
        if (isTimeoutError(e)) msg = "Key check timed out. Try again, or switch testnet off.";
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
          setKeys((s) => ({ ...s, loading: false }));
          return;
        }

        setKeys((s) => ({ ...s, loading: false, error: msg }));
        showToast("Key check failed");
      } finally {
        if (requestId === keysRequestSeqRef.current) keysAbortRef.current = null;
      }
    },
    [apiBase, authHeaders, keysParams, showToast],
  );

  const stopListenKeyStream = useCallback(
    async (opts?: { close?: boolean; silent?: boolean }) => {
      if (listenKeyKeepAliveTimerRef.current) {
        window.clearInterval(listenKeyKeepAliveTimerRef.current);
        listenKeyKeepAliveTimerRef.current = null;
      }
      const ws = listenKeyWsRef.current;
      listenKeyWsRef.current = null;
      ws?.close();

      const info = listenKeyUi.info;
      if (opts?.close && info) {
        try {
          const base: ApiParams = { market: info.market, binanceTestnet: info.testnet };
          await binanceListenKeyClose(
            apiBase,
            { ...withBinanceKeys(base), listenKey: info.listenKey },
            { headers: authHeaders, timeoutMs: 30_000 },
          );
        } catch (e) {
          if (!opts?.silent) {
            const msg = e instanceof Error ? e.message : String(e);
            showToast(`Listen key close failed: ${msg}`);
          }
        }
      }

      setListenKeyUi((s) => ({
        ...s,
        loading: false,
        error: null,
        info: null,
        wsStatus: "disconnected",
        wsError: null,
        lastEventAtMs: null,
        lastEvent: null,
        keepAliveAtMs: null,
        keepAliveError: null,
      }));
    },
    [apiBase, authHeaders, listenKeyUi.info, showToast, withBinanceKeys],
  );

  const keepAliveListenKeyStream = useCallback(
    async (info: BinanceListenKeyResponse, opts?: { silent?: boolean }) => {
      setListenKeyUi((s) => ({ ...s, keepAliveError: null }));
      try {
        const base: ApiParams = { market: info.market, binanceTestnet: info.testnet };
        const out = await binanceListenKeyKeepAlive(
          apiBase,
          { ...withBinanceKeys(base), listenKey: info.listenKey },
          { headers: authHeaders, timeoutMs: 30_000 },
        );
        setListenKeyUi((s) => ({ ...s, keepAliveAtMs: out.atMs, keepAliveError: null }));
        if (!opts?.silent) showToast("Listen key kept alive");
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setListenKeyUi((s) => ({ ...s, keepAliveError: msg }));
        if (!opts?.silent) showToast("Listen key keep-alive failed");
      }
    },
    [apiBase, authHeaders, showToast, withBinanceKeys],
  );

  const startListenKeyStream = useCallback(async () => {
    if (apiOk !== "ok") return;
    await stopListenKeyStream({ close: false, silent: true });
    setListenKeyUi((s) => ({ ...s, loading: true, error: null, wsError: null, keepAliveError: null, wsStatus: "connecting" }));
    try {
      const base: ApiParams = { market: form.market, binanceTestnet: form.binanceTestnet };
      const out = await binanceListenKey(apiBase, withBinanceKeys(base), { headers: authHeaders, timeoutMs: 30_000 });

      setListenKeyUi((s) => ({ ...s, loading: false, error: null, info: out, wsStatus: "connecting" }));

      const ws = new WebSocket(out.wsUrl);
      listenKeyWsRef.current = ws;
      ws.addEventListener("open", () => {
        if (listenKeyWsRef.current !== ws) return;
        setListenKeyUi((s) => ({ ...s, wsStatus: "connected", wsError: null }));
      });
      ws.addEventListener("close", () => {
        if (listenKeyWsRef.current !== ws) return;
        setListenKeyUi((s) => ({ ...s, wsStatus: "disconnected" }));
      });
      ws.addEventListener("error", () => {
        if (listenKeyWsRef.current !== ws) return;
        setListenKeyUi((s) => ({ ...s, wsError: "WebSocket error" }));
      });
      ws.addEventListener("message", (ev) => {
        if (listenKeyWsRef.current !== ws) return;
        const raw = typeof ev.data === "string" ? ev.data : String(ev.data);
        let pretty = raw;
        try {
          pretty = JSON.stringify(JSON.parse(raw), null, 2);
        } catch {
          // ignore
        }
        if (pretty.length > 8000) pretty = `${pretty.slice(0, 7997)}...`;
        setListenKeyUi((s) => ({ ...s, lastEventAtMs: Date.now(), lastEvent: pretty }));
      });

      void keepAliveListenKeyStream(out, { silent: true });
      const intervalMs = Math.max(60_000, Math.round(out.keepAliveMs * 0.9));
      listenKeyKeepAliveTimerRef.current = window.setInterval(() => void keepAliveListenKeyStream(out, { silent: true }), intervalMs);

      showToast("Listen key started");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setListenKeyUi((s) => ({ ...s, loading: false, error: msg, wsStatus: "disconnected" }));
      showToast("Listen key start failed");
    }
  }, [apiBase, apiOk, authHeaders, form.binanceTestnet, form.market, keepAliveListenKeyStream, showToast, stopListenKeyStream, withBinanceKeys]);

  const refreshBot = useCallback(
    async (opts?: RunOptions) => {
      const requestId = ++botRequestSeqRef.current;
      const startedAtMs = Date.now();
      botAbortRef.current?.abort();
      const controller = new AbortController();
      botAbortRef.current = controller;

      if (!opts?.silent) setBot((s) => ({ ...s, loading: true, error: null }));

      try {
        const out = await botStatus(
          apiBase,
          { signal: controller.signal, headers: authHeaders, timeoutMs: BOT_STATUS_TIMEOUT_MS },
          BOT_STATUS_TAIL_POINTS,
        );
        const finishedAtMs = Date.now();
        if (requestId !== botRequestSeqRef.current) return;
        botStatusFetchedRef.current = true;
        setBotRt((prev) => {
          const base: BotRtUiState = {
            ...prev,
            lastFetchAtMs: finishedAtMs,
            lastFetchDurationMs: Math.max(0, finishedAtMs - startedAtMs),
            lastNewCandles: 0,
            lastKlineUpdates: 0,
          };

          const rt = botRtRef.current;

          if (!out.running) {
            botRtRef.current = {
              botKey: null,
              lastOpenTimeMs: null,
              lastError: null,
              lastHalted: null,
              lastFetchedOpenTimeMs: null,
              lastFetchedClose: null,
              lastMethod: null,
              lastOpenThreshold: null,
              lastCloseThreshold: null,
              lastTradeEnabled: null,
              lastTelemetryPolledAtMs: null,
            };
            return {
              ...base,
              lastNewCandles: 0,
              lastNewCandlesAtMs: null,
              lastKlineUpdates: 0,
              lastKlineUpdatesAtMs: null,
              telemetry: [],
              feed: [],
            };
          }

          const botKey = `${out.market}:${out.symbol}:${out.interval}`;
          let feed = base.feed;
          let telemetry = base.telemetry;
          if (rt.botKey !== botKey) {
            feed = [];
            telemetry = [];
            rt.botKey = botKey;
            rt.lastOpenTimeMs = null;
            rt.lastError = null;
            rt.lastHalted = null;
            rt.lastFetchedOpenTimeMs = null;
            rt.lastFetchedClose = null;
            rt.lastMethod = null;
            rt.lastOpenThreshold = null;
            rt.lastCloseThreshold = null;
            rt.lastTradeEnabled = null;
            rt.lastTelemetryPolledAtMs = null;
          }

          const openTimes = out.openTimes;
          const lastOpen = openTimes[openTimes.length - 1] ?? null;
          const prevLastOpen = rt.lastOpenTimeMs;
          const newTimes = typeof prevLastOpen === "number" ? openTimes.filter((t) => t > prevLastOpen) : [];
          const newCount = newTimes.length;

	          let lastNewCandlesAtMs: number | null = prev.lastNewCandlesAtMs;
	          if (newCount > 0) {
	            lastNewCandlesAtMs = finishedAtMs;
	            const lastNew = newTimes[newTimes.length - 1]!;
	            const idx = openTimes.lastIndexOf(lastNew);
	            const closePx = idx >= 0 ? out.prices[idx] : null;
	            const action = out.latestSignal.action;
	            const pollMs = typeof out.pollLatencyMs === "number" && Number.isFinite(out.pollLatencyMs) ? Math.max(0, Math.round(out.pollLatencyMs)) : null;
	            const batchMs = typeof out.lastBatchMs === "number" && Number.isFinite(out.lastBatchMs) ? Math.max(0, Math.round(out.lastBatchMs)) : null;
	            const batchSize =
	              typeof out.lastBatchSize === "number" && Number.isFinite(out.lastBatchSize) ? Math.max(0, Math.round(out.lastBatchSize)) : null;
	            const perBarMs = batchMs !== null && batchSize && batchSize > 0 ? batchMs / batchSize : null;
	            const msg =
	              `candle +${newCount}: open ${fmtTimeMs(lastNew)}` +
	              (typeof closePx === "number" && Number.isFinite(closePx) ? ` close ${fmtMoney(closePx, 4)}` : "") +
	              (action ? ` • action ${action}` : "") +
	              (pollMs !== null ? ` • poll ${pollMs}ms` : "") +
	              (batchMs !== null ? ` • proc ${batchMs}ms${perBarMs !== null ? ` (${fmtNum(perBarMs, 1)}ms/bar)` : ""}` : "");
	            feed = [{ atMs: finishedAtMs, message: msg }, ...feed].slice(0, 50);
	          }

          let lastKlineUpdatesAtMs: number | null = prev.lastKlineUpdatesAtMs;
          let klineUpdates = 0;
          const fetchedLast = out.fetchedLastKline;
	          if (fetchedLast && typeof fetchedLast.openTime === "number" && Number.isFinite(fetchedLast.openTime)) {
	            const openTime = fetchedLast.openTime;
	            const close = fetchedLast.close;
	            if (typeof close === "number" && Number.isFinite(close)) {
	              const prevFetchedOpen = rt.lastFetchedOpenTimeMs;
	              const prevFetchedClose = rt.lastFetchedClose;
	              if (newCount === 0 && prevFetchedOpen === openTime && typeof prevFetchedClose === "number" && Number.isFinite(prevFetchedClose) && close !== prevFetchedClose) {
	                klineUpdates = 1;
	                lastKlineUpdatesAtMs = finishedAtMs;
	                const d = prevFetchedClose !== 0 ? (close - prevFetchedClose) / prevFetchedClose : null;
	                const msg = `kline update: close ${fmtMoney(close, 4)}${d !== null ? ` (Δ ${fmtPct(d, 2)})` : ""}`;
	                feed = [{ atMs: finishedAtMs, message: msg }, ...feed].slice(0, 50);
	              }
	              rt.lastFetchedOpenTimeMs = openTime;
	              rt.lastFetchedClose = close;
	            }
	          }

	          const polledAtMs = typeof out.polledAtMs === "number" && Number.isFinite(out.polledAtMs) ? out.polledAtMs : null;
	          if (polledAtMs !== null && polledAtMs !== rt.lastTelemetryPolledAtMs) {
	            rt.lastTelemetryPolledAtMs = polledAtMs;
	            const pollLatencyMs = typeof out.pollLatencyMs === "number" && Number.isFinite(out.pollLatencyMs) ? out.pollLatencyMs : null;
	            const processedOpenTime = out.openTimes[out.openTimes.length - 1] ?? null;
	            const processedClose = out.prices[out.prices.length - 1] ?? null;
	            const driftBps =
	              fetchedLast &&
	              typeof fetchedLast.openTime === "number" &&
	              Number.isFinite(fetchedLast.openTime) &&
	              processedOpenTime === fetchedLast.openTime &&
	              typeof fetchedLast.close === "number" &&
	              Number.isFinite(fetchedLast.close) &&
	              typeof processedClose === "number" &&
	              Number.isFinite(processedClose) &&
	              processedClose !== 0
	                ? ((fetchedLast.close - processedClose) / processedClose) * 10000
	                : null;
	            const point: BotTelemetryPoint = { atMs: polledAtMs, pollLatencyMs, driftBps };
	            telemetry = [...telemetry, point].slice(-BOT_TELEMETRY_POINTS);
	          }

	          const openThr = out.openThreshold ?? out.threshold;
	          const closeThr = out.closeThreshold ?? out.openThreshold ?? out.threshold;
	          const tradeEnabled = out.settings?.tradeEnabled ?? null;

	          if (rt.lastMethod !== null && (out.method !== rt.lastMethod || openThr !== rt.lastOpenThreshold || closeThr !== rt.lastCloseThreshold)) {
	            const msg =
	              `params: ${methodLabel(out.method)}` +
	              ` • open ${fmtPct(openThr, 3)}` +
	              ` • close ${fmtPct(closeThr, 3)}` +
	              (typeof tradeEnabled === "boolean" ? ` • trade ${tradeEnabled ? "ON" : "OFF"}` : "");
	            feed = [{ atMs: finishedAtMs, message: msg }, ...feed].slice(0, 50);
	          }

	          if (typeof tradeEnabled === "boolean" && rt.lastTradeEnabled !== null && tradeEnabled !== rt.lastTradeEnabled) {
	            feed = [{ atMs: finishedAtMs, message: `trade ${tradeEnabled ? "enabled" : "disabled"}` }, ...feed].slice(0, 50);
	          }

	          const err = out.error ?? null;
	          if (err && err !== rt.lastError) {
	            feed = [{ atMs: finishedAtMs, message: `error: ${err}` }, ...feed].slice(0, 50);
	          }

          if (rt.lastHalted !== null && rt.lastHalted !== out.halted) {
            feed = [{ atMs: finishedAtMs, message: out.halted ? `halted: ${out.haltReason ?? "true"}` : "resumed" }, ...feed].slice(0, 50);
          }

	          rt.lastOpenTimeMs = lastOpen;
	          rt.lastError = err;
	          rt.lastHalted = out.halted;
	          rt.lastMethod = out.method;
	          rt.lastOpenThreshold = openThr;
	          rt.lastCloseThreshold = closeThr;
	          rt.lastTradeEnabled = typeof tradeEnabled === "boolean" ? tradeEnabled : null;

	          return { ...base, lastNewCandles: newCount, lastNewCandlesAtMs, lastKlineUpdates: klineUpdates, lastKlineUpdatesAtMs, telemetry, feed };
	        });
        setBot((s) => ({ ...s, loading: false, error: null, status: out }));
        setApiOk("ok");
      } catch (e) {
        if (requestId !== botRequestSeqRef.current) return;
        if (isAbortError(e)) return;
        let msg = e instanceof Error ? e.message : String(e);
        if (isTimeoutError(e)) msg = "Bot status timed out. Try again.";
        setBot((s) => ({ ...s, loading: false, error: msg }));

        setApiOk((prev) => {
          if (e instanceof HttpError && (e.status === 401 || e.status === 403)) return "auth";
          const looksDown = msg.toLowerCase().includes("fetch") || (e instanceof HttpError && e.status >= 500) || isTimeoutError(e);
          return looksDown ? "down" : prev;
        });
      } finally {
        if (requestId === botRequestSeqRef.current) botAbortRef.current = null;
      }
    },
    [apiBase, authHeaders],
  );

  type StartBotOptions = { auto?: boolean; forceAdopt?: boolean; silent?: boolean };

  const startLiveBot = useCallback(
    async (opts?: StartBotOptions) => {
      const silent = Boolean(opts?.silent);
      const forceAdopt = Boolean(opts?.forceAdopt);
      if (!opts?.auto) botAutoStartSuppressedRef.current = false;

      const isAdoptError = (msg: string) => {
        const lower = msg.toLowerCase();
        return lower.includes("existing long position") || lower.includes("botadoptexistingposition=true");
      };

      const formatStartError = (err: unknown) => {
        let msg = err instanceof Error ? err.message : String(err);
        let showErrorToast = !silent;
        if (err instanceof HttpError && err.status === 429) {
          const untilMs = applyRateLimit(err, { silent });
          msg = `Rate limited. Try again ${fmtEtaMs(Math.max(0, untilMs - Date.now()))}.`;
          showErrorToast = false;
        } else if (err instanceof HttpError && err.payload && typeof err.payload === "object") {
          try {
            let detail = JSON.stringify(err.payload, null, 2);
            if (detail.length > 2000) detail = `${detail.slice(0, 1997)}...`;
            if (detail !== "{}") msg = `${msg}\n${detail}`;
          } catch {
            // ignore
          }
        }
        return { msg, showErrorToast };
      };

      const runStart = async (adoptOverride: boolean, retrying: boolean) => {
        setBot((s) => ({ ...s, loading: true, error: null }));
        const payload: ApiParams = {
          ...tradeParams,
          botTrade: form.tradeArmed,
          botAdoptExistingPosition: adoptOverride || form.botAdoptExistingPosition,
          ...(form.botPollSeconds > 0 ? { botPollSeconds: clamp(Math.trunc(form.botPollSeconds), 1, 3600) } : {}),
          botOnlineEpochs: clamp(Math.trunc(form.botOnlineEpochs), 0, 50),
          botTrainBars: Math.max(10, Math.trunc(form.botTrainBars)),
          botMaxPoints: clamp(Math.trunc(form.botMaxPoints), 100, 100000),
        };
        const out = await botStart(apiBase, withBinanceKeys(payload), { headers: authHeaders, timeoutMs: BOT_START_TIMEOUT_MS });
        setBot((s) => ({ ...s, loading: false, error: null, status: out }));
        if (adoptOverride && !form.botAdoptExistingPosition) {
          setForm((f) => ({ ...f, botAdoptExistingPosition: true }));
        }
        if (!silent) {
          showToast(
            out.running
              ? form.tradeArmed
                ? "Live bot started (trading armed)"
                : "Live bot started (paper mode)"
              : out.starting
                ? "Live bot starting…"
                : "Bot not running",
          );
        }
        if (!retrying) botAutoStartRef.current.lastAttemptAtMs = Date.now();
      };

      try {
        await runStart(forceAdopt, false);
      } catch (e) {
        if (isAbortError(e)) return;
        const baseMsg = e instanceof Error ? e.message : String(e);
        if (!forceAdopt && isAdoptError(baseMsg)) {
          if (!silent) showToast("Existing position detected. Adopting and retrying…");
          try {
            await runStart(true, true);
            return;
          } catch (e2) {
            if (isAbortError(e2)) return;
            const formatted2 = formatStartError(e2);
            setBot((s) => ({ ...s, loading: false, error: formatted2.msg }));
            if (formatted2.showErrorToast) showToast("Bot start failed");
            return;
          }
        }
        const formatted = formatStartError(e);
        setBot((s) => ({ ...s, loading: false, error: formatted.msg }));
        if (formatted.showErrorToast) showToast("Bot start failed");
      }
    },
    [
      apiBase,
      applyRateLimit,
      authHeaders,
      form.botMaxPoints,
      form.botAdoptExistingPosition,
      form.botOnlineEpochs,
      form.botPollSeconds,
      form.botTrainBars,
      form.tradeArmed,
      showToast,
      tradeParams,
      withBinanceKeys,
    ],
  );

  const stopLiveBot = useCallback(async () => {
    setBot((s) => ({ ...s, loading: true, error: null }));
    try {
      const out = await botStop(apiBase, { headers: authHeaders, timeoutMs: 30_000 });
      setBot((s) => ({ ...s, loading: false, error: null, status: out }));
      botAutoStartSuppressedRef.current = true;
      showToast("Bot stopped");
    } catch (e) {
      if (isAbortError(e)) return;
      const msg = e instanceof Error ? e.message : String(e);
      setBot((s) => ({ ...s, loading: false, error: msg }));
      showToast("Bot stop failed");
    }
  }, [apiBase, authHeaders, showToast]);

  useEffect(() => {
    if (apiOk !== "ok") return;
    void refreshBot({ silent: true });
  }, [apiOk, refreshBot]);

  useEffect(() => {
    if (apiOk !== "ok") return;
    const starting = bot.status.running ? false : bot.status.starting === true;
    if (!bot.status.running && !starting) return;
    const t = window.setInterval(() => {
      if (bot.loading) return;
      void refreshBot({ silent: true });
    }, 2000);
    return () => window.clearInterval(t);
  }, [apiOk, bot.loading, bot.status, refreshBot]);

  useEffect(() => {
    if (!form.autoRefresh || apiOk !== "ok") return;
    const ms = clamp(form.autoRefreshSec, 5, 600) * 1000;
    const t = window.setInterval(() => {
      if (abortRef.current) return;
      const limit = rateLimitRef.current;
      if (limit && Date.now() < limit.untilMs) return;
      const p = commonParams;
      if (!p.binanceSymbol || !p.interval) return;
      void run("signal", undefined, { silent: true });
    }, ms);
    return () => window.clearInterval(t);
  }, [apiOk, form.autoRefresh, form.autoRefreshSec, run, commonParams]);

  useEffect(() => {
    if (!state.error) return;
    errorRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [state.error]);

  useEffect(() => {
    let isCancelled = false;
    setTopCombosLoading(true);
    const fetchPayload = async (): Promise<unknown> => {
      if (apiOk === "ok") {
        try {
          const url = `${apiBase.replace(/\/+$/, "")}/optimizer/combos`;
          const res = await fetch(url, { headers: authHeaders });
          if (res.ok) return res.json();
          throw new Error(`HTTP ${res.status}`);
        } catch {
          // Fall back to the static UI bundle.
        }
      }
      const res = await fetch("/top-combos.json");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    };
    void fetchPayload()
      .then((payload: unknown) => {
        if (isCancelled) return;
        const payloadRec = (payload as Record<string, unknown> | null | undefined) ?? {};
        const rawCombos: unknown[] = Array.isArray(payloadRec.combos) ? (payloadRec.combos as unknown[]) : [];
        const methods: Method[] = ["11", "10", "01"];
        const normalizations: Normalization[] = ["none", "minmax", "standard", "log"];
        const positionings: Positioning[] = ["long-flat", "long-short"];
        const intrabarFills: IntrabarFill[] = ["stop-first", "take-profit-first"];
        const sanitized: OptimizationCombo[] = rawCombos.map((raw, index) => {
          const rawRec = (raw as Record<string, unknown> | null | undefined) ?? {};
          const params = (rawRec.params as Record<string, unknown> | null | undefined) ?? {};
          const method =
            typeof params.method === "string" && methods.includes(params.method as Method)
              ? (params.method as Method)
              : defaultForm.method;
          const normalization =
            typeof params.normalization === "string" && normalizations.includes(params.normalization as Normalization)
              ? (params.normalization as Normalization)
              : defaultForm.normalization;
          const interval = typeof params.interval === "string" && params.interval ? params.interval : defaultForm.interval;
          const bars = typeof params.bars === "number" && Number.isFinite(params.bars) ? Math.trunc(params.bars) : Math.trunc(defaultForm.bars);
          const positioning =
            typeof params.positioning === "string" && positionings.includes(params.positioning as Positioning)
              ? (params.positioning as Positioning)
              : defaultForm.positioning;
          const baseOpenThreshold =
            typeof params.baseOpenThreshold === "number" && Number.isFinite(params.baseOpenThreshold)
              ? Math.max(0, params.baseOpenThreshold)
              : null;
          const baseCloseThreshold =
            typeof params.baseCloseThreshold === "number" && Number.isFinite(params.baseCloseThreshold)
              ? Math.max(0, params.baseCloseThreshold)
              : null;
          const fee = typeof params.fee === "number" && Number.isFinite(params.fee) ? Math.max(0, params.fee) : defaultForm.fee;
          const hiddenSize =
            typeof params.hiddenSize === "number" && Number.isFinite(params.hiddenSize) ? Math.max(1, Math.trunc(params.hiddenSize)) : Math.trunc(defaultForm.hiddenSize);
          const learningRate =
            typeof params.learningRate === "number" && Number.isFinite(params.learningRate) ? params.learningRate : 0.001;
          const valRatio =
            typeof params.valRatio === "number" && Number.isFinite(params.valRatio)
              ? clamp(params.valRatio, 0, 1)
              : defaultForm.valRatio;
          const patience =
            typeof params.patience === "number" && Number.isFinite(params.patience) ? Math.max(0, Math.trunc(params.patience)) : Math.trunc(defaultForm.patience);
          const gradClip =
            typeof params.gradClip === "number" && Number.isFinite(params.gradClip) ? Math.max(0, params.gradClip) : null;
          const epochs = typeof params.epochs === "number" && Number.isFinite(params.epochs) ? Math.max(0, Math.trunc(params.epochs)) : Math.trunc(defaultForm.epochs);
          const slippage = typeof params.slippage === "number" && Number.isFinite(params.slippage) ? params.slippage : defaultForm.slippage;
          const spread = typeof params.spread === "number" && Number.isFinite(params.spread) ? params.spread : defaultForm.spread;
          const intrabarFill =
            typeof params.intrabarFill === "string" && intrabarFills.includes(params.intrabarFill as IntrabarFill)
              ? (params.intrabarFill as IntrabarFill)
              : defaultForm.intrabarFill;
          const kalmanZMin =
            typeof params.kalmanZMin === "number" && Number.isFinite(params.kalmanZMin) ? Math.max(0, params.kalmanZMin) : defaultForm.kalmanZMin;
          const kalmanZMaxRaw =
            typeof params.kalmanZMax === "number" && Number.isFinite(params.kalmanZMax) ? Math.max(0, params.kalmanZMax) : defaultForm.kalmanZMax;
          const kalmanZMax = Math.max(kalmanZMin, kalmanZMaxRaw);
          const maxHighVolProbRaw =
            typeof params.maxHighVolProb === "number" && Number.isFinite(params.maxHighVolProb) ? clamp(params.maxHighVolProb, 0, 1) : null;
          const maxConformalWidthRaw =
            typeof params.maxConformalWidth === "number" && Number.isFinite(params.maxConformalWidth) ? Math.max(0, params.maxConformalWidth) : null;
          const maxQuantileWidthRaw =
            typeof params.maxQuantileWidth === "number" && Number.isFinite(params.maxQuantileWidth) ? Math.max(0, params.maxQuantileWidth) : null;
          const maxHighVolProb = maxHighVolProbRaw != null && maxHighVolProbRaw > 0 ? maxHighVolProbRaw : null;
          const maxConformalWidth = maxConformalWidthRaw != null && maxConformalWidthRaw > 0 ? maxConformalWidthRaw : null;
          const maxQuantileWidth = maxQuantileWidthRaw != null && maxQuantileWidthRaw > 0 ? maxQuantileWidthRaw : null;
          const confirmConformal = typeof params.confirmConformal === "boolean" ? params.confirmConformal : defaultForm.confirmConformal;
          const confirmQuantiles = typeof params.confirmQuantiles === "boolean" ? params.confirmQuantiles : defaultForm.confirmQuantiles;
          const confidenceSizing = typeof params.confidenceSizing === "boolean" ? params.confidenceSizing : defaultForm.confidenceSizing;
          const minPositionSizeRaw =
            typeof params.minPositionSize === "number" && Number.isFinite(params.minPositionSize) ? clamp(params.minPositionSize, 0, 1) : null;
          const minPositionSize = minPositionSizeRaw != null && minPositionSizeRaw > 0 ? minPositionSizeRaw : null;
          const rankRaw = typeof rawRec.rank === "number" && Number.isFinite(rawRec.rank) ? Math.trunc(rawRec.rank) : null;
          const rank = rankRaw != null && rankRaw >= 1 ? rankRaw : null;
          const objective = typeof rawRec.objective === "string" && rawRec.objective ? rawRec.objective : null;
          const score = typeof rawRec.score === "number" && Number.isFinite(rawRec.score) ? rawRec.score : null;
          const metricsRec = (rawRec.metrics as Record<string, unknown> | null | undefined) ?? {};
          const sharpe =
            typeof metricsRec["sharpe"] === "number" && Number.isFinite(metricsRec["sharpe"]) ? (metricsRec["sharpe"] as number) : null;
          const maxDrawdown =
            typeof metricsRec["maxDrawdown"] === "number" && Number.isFinite(metricsRec["maxDrawdown"])
              ? (metricsRec["maxDrawdown"] as number)
              : null;
          const turnover =
            typeof metricsRec["turnover"] === "number" && Number.isFinite(metricsRec["turnover"]) ? (metricsRec["turnover"] as number) : null;
          const roundTrips =
            typeof metricsRec["roundTrips"] === "number" && Number.isFinite(metricsRec["roundTrips"])
              ? Math.trunc(metricsRec["roundTrips"] as number)
              : null;
          const metrics =
            sharpe != null || maxDrawdown != null || turnover != null || roundTrips != null
              ? { sharpe, maxDrawdown, turnover, roundTrips }
              : null;
          const rawSource = typeof rawRec.source === "string" ? rawRec.source : null;
          const source: OptimizationCombo["source"] =
            rawSource === "binance" ? "binance" : rawSource === "csv" ? "csv" : null;
          return {
            id: rank ?? index + 1,
            rank,
            objective,
            score,
            metrics,
            finalEquity: typeof rawRec.finalEquity === "number" && Number.isFinite(rawRec.finalEquity) ? rawRec.finalEquity : 0,
            openThreshold: typeof rawRec.openThreshold === "number" ? rawRec.openThreshold : null,
            closeThreshold: typeof rawRec.closeThreshold === "number" ? rawRec.closeThreshold : null,
            source,
            params: {
              interval,
              bars,
              method,
              positioning,
              normalization,
              baseOpenThreshold,
              baseCloseThreshold,
              fee,
              epochs,
              hiddenSize,
              learningRate,
              valRatio,
              patience,
              gradClip,
              slippage,
              spread,
              intrabarFill,
              stopLoss: typeof params.stopLoss === "number" && Number.isFinite(params.stopLoss) ? params.stopLoss : null,
              takeProfit: typeof params.takeProfit === "number" && Number.isFinite(params.takeProfit) ? params.takeProfit : null,
              trailingStop: typeof params.trailingStop === "number" && Number.isFinite(params.trailingStop) ? params.trailingStop : null,
              maxDrawdown: typeof params.maxDrawdown === "number" && Number.isFinite(params.maxDrawdown) ? params.maxDrawdown : null,
              maxDailyLoss: typeof params.maxDailyLoss === "number" && Number.isFinite(params.maxDailyLoss) ? params.maxDailyLoss : null,
              maxOrderErrors:
                typeof params.maxOrderErrors === "number" && Number.isFinite(params.maxOrderErrors) ? Math.max(1, Math.trunc(params.maxOrderErrors)) : null,
              kalmanZMin,
              kalmanZMax,
              maxHighVolProb,
              maxConformalWidth,
              maxQuantileWidth,
              confirmConformal,
              confirmQuantiles,
              confidenceSizing,
              minPositionSize,
            },
          };
        });
        sanitized.sort((a, b) => {
          const ar = typeof a.rank === "number" && Number.isFinite(a.rank) ? a.rank : null;
          const br = typeof b.rank === "number" && Number.isFinite(b.rank) ? b.rank : null;
          if (ar != null && br != null) return ar - br;
          if (ar != null) return -1;
          if (br != null) return 1;

          const sa = typeof a.score === "number" && Number.isFinite(a.score) ? a.score : null;
          const sb = typeof b.score === "number" && Number.isFinite(b.score) ? b.score : null;
          if (sa != null || sb != null) {
            const diff = (sb ?? Number.NEGATIVE_INFINITY) - (sa ?? Number.NEGATIVE_INFINITY);
            if (diff !== 0) return diff;
          }

          const eq = b.finalEquity - a.finalEquity;
          if (eq !== 0) return eq;
          return a.id - b.id;
        });
        const binanceCombos = sanitized.filter((combo) => combo.source === "binance");
        const preferredCombos = binanceCombos.length > 0 ? binanceCombos : sanitized;
        setTopCombos(preferredCombos.slice(0, 5));
        setTopCombosError(null);
      })
      .catch((err) => {
        if (isCancelled) return;
        setTopCombosError(err instanceof Error ? err.message : "Failed to load optimizer combos.");
      })
      .finally(() => {
        if (isCancelled) return;
        setTopCombosLoading(false);
      });
    return () => {
      isCancelled = true;
    };
  }, [apiBase, apiOk, authHeaders]);

  const statusDotClass =
    apiOk === "ok" ? "dot dotOk" : apiOk === "down" ? "dot dotBad" : "dot dotWarn";
  const statusLabel =
    apiOk === "ok"
      ? "API online"
      : apiOk === "auth"
        ? apiToken.trim()
          ? "API auth failed"
          : "API auth required"
        : apiOk === "down"
          ? "API unreachable"
          : "API status unknown";

  const missingSymbol = !form.binanceSymbol.trim();
  const intervalValue = form.interval.trim();
  const missingInterval = !intervalValue || !BINANCE_INTERVAL_SET.has(intervalValue);
  const lookbackState = useMemo(() => {
    const barsRaw = Math.trunc(form.bars);
    const bars = barsRaw <= 0 ? 0 : clamp(barsRaw, 2, 1000);
    const interval = form.interval.trim();
    const intervalSec = binanceIntervalSeconds(interval);

    const overrideRaw = Math.trunc(form.lookbackBars);
    const overrideOn = overrideRaw >= 2;

    const windowRaw = form.lookbackWindow.trim();
    const windowSec = windowRaw ? parseDurationSeconds(windowRaw) : null;
    const windowBars = windowSec && windowSec > 0 && intervalSec ? Math.ceil(windowSec / intervalSec) : null;

    const effectiveBars = overrideOn ? overrideRaw : windowBars;
    const minBarsRequired = effectiveBars != null ? effectiveBars + 1 : null;

    let error: string | null = null;
    if (overrideOn) {
      if (overrideRaw < 2) error = "Lookback bars must be >= 2 (or 0 to use the window).";
    } else {
      if (!windowRaw) error = "Lookback window is required (e.g. 24h).";
      else if (windowSec == null || windowSec <= 0) error = "Lookback window must look like 24h, 90m, 7d.";
      else if (!intervalSec) error = "Interval is required.";
      else if (windowBars != null && windowBars < 2) error = "Lookback window is too small (needs at least 2 bars).";
    }

    if (!error && effectiveBars != null && effectiveBars >= 2 && barsRaw > 0 && bars <= effectiveBars) {
      error = `Not enough bars for lookback: need bars >= ${effectiveBars + 1} (or reduce lookback).`;
    }

    const summary =
      effectiveBars != null && effectiveBars >= 2
        ? overrideOn
          ? `Effective lookback: ${effectiveBars} bars (override). Need bars ≥ ${effectiveBars + 1}.`
          : windowBars != null
            ? `Effective lookback: ${windowRaw} ≈ ${effectiveBars} bars. Need bars ≥ ${effectiveBars + 1}.`
            : "Effective lookback: —"
        : "Effective lookback: —";

    return { bars, intervalSec, windowBars, overrideOn, effectiveBars, minBarsRequired, error, summary };
  }, [form.bars, form.interval, form.lookbackBars, form.lookbackWindow]);
  const showLocalStartHelp = useMemo(() => {
    if (typeof window === "undefined") return true;
    return isLocalHostname(window.location.hostname);
  }, []);
  const apiBlockedReason = useMemo(() => {
    const authRequired = apiOk === "auth";
    const tokenPresent = Boolean(apiToken.trim());
    const authMsg = authRequired
      ? tokenPresent
        ? "API token rejected. Update apiToken in trader-config.js."
        : "API auth required. Set apiToken in trader-config.js."
      : null;
    const startCmd = `cd haskell && cabal run -v0 trader-hs -- --serve --port ${API_PORT}`;
    const downMsg = showLocalStartHelp
      ? `Backend unreachable. Start it with: ${startCmd}`
      : "Backend unreachable. Configure apiBaseUrl in trader-config.js (or configure CloudFront to forward `/api/*` to your API origin).";
    return firstReason(
      apiBaseError,
      apiOk === "down" ? downMsg : null,
      authMsg,
    );
  }, [apiBaseError, apiOk, apiToken, showLocalStartHelp]);
  const rateLimitEtaMs = rateLimit ? Math.max(0, rateLimit.untilMs - rateLimitTickMs) : null;
  const rateLimitReason =
    rateLimit && rateLimitEtaMs != null ? `${rateLimit.reason} Next retry ${fmtEtaMs(rateLimitEtaMs)}.` : rateLimit?.reason ?? null;

  const apiComputeLimits = healthInfo?.computeLimits ?? null;
  const apiLstmEnabled = form.method !== "10";
  const barsRawForLimits = Math.trunc(form.bars);
  const barsEffectiveForLimits = barsRawForLimits <= 0 ? 500 : barsRawForLimits;
  const barsExceedsApi = Boolean(
    apiComputeLimits && apiLstmEnabled && barsEffectiveForLimits > apiComputeLimits.maxBarsLstm,
  );
  const epochsExceedsApi = Boolean(apiComputeLimits && apiLstmEnabled && form.epochs > apiComputeLimits.maxEpochs);
  const hiddenSizeExceedsApi = Boolean(apiComputeLimits && apiLstmEnabled && form.hiddenSize > apiComputeLimits.maxHiddenSize);
  const apiLimitsReason = barsExceedsApi
    ? `Bars exceed API limit (max ${apiComputeLimits?.maxBarsLstm ?? "?"} for LSTM methods). Reduce bars or use method=10 (Kalman-only).`
    : epochsExceedsApi
      ? `Epochs exceed API limit (max ${apiComputeLimits?.maxEpochs ?? "?"}). Reduce epochs or switch to method=10 (Kalman-only).`
      : hiddenSizeExceedsApi
        ? `Hidden size exceeds API limit (max ${apiComputeLimits?.maxHiddenSize ?? "?"}). Reduce hidden size or switch to method=10 (Kalman-only).`
        : null;
  const requestDisabledReason = firstReason(
    apiBlockedReason,
    rateLimitReason,
    missingSymbol ? "Binance symbol is required." : null,
    missingInterval ? "Interval is required." : null,
    lookbackState.error,
    apiLimitsReason,
  );
  const requestDisabled = state.loading || Boolean(requestDisabledReason);
  const botStarting = bot.status.running ? false : bot.status.starting === true;
  const botStartBlockedReason = firstReason(
    form.positioning === "long-short" ? "Live bot supports Long/Flat only." : null,
    requestDisabledReason,
  );
  const botStartBlocked = bot.loading || botStarting || Boolean(botStartBlockedReason);

  useEffect(() => {
    if (apiOk !== "ok") return;
    if (!botStatusFetchedRef.current) return;
    if (botAutoStartSuppressedRef.current) return;
    if (bot.status.running) return;
    if (botStartBlocked) return;
    const now = Date.now();
    if (now - botAutoStartRef.current.lastAttemptAtMs < BOT_AUTOSTART_RETRY_MS) return;
    botAutoStartRef.current.lastAttemptAtMs = now;
    void startLiveBot({ auto: true, silent: true });
  }, [apiOk, bot.status.running, botStartBlocked, startLiveBot]);
  const orderQuoteFractionError = useMemo(() => {
    const f = form.orderQuoteFraction;
    if (!Number.isFinite(f)) return "Order quote fraction must be a number.";
    if (f <= 0) return null;
    if (f > 1) return "Order quote fraction must be <= 1 (use 0 to disable).";
    return null;
  }, [form.orderQuoteFraction]);
  const tradeOrderSizingError = useMemo(() => {
    if (form.orderQuantity > 0 || form.orderQuote > 0) return null;
    return orderQuoteFractionError;
  }, [form.orderQuantity, form.orderQuote, orderQuoteFractionError]);
  const tradeDisabledReason = firstReason(
    requestDisabledReason,
    tradeOrderSizingError,
    form.positioning === "long-short" && form.market !== "futures" ? "Long/Short trading requires Futures market." : null,
  );

  const orderSizing = useMemo(() => {
    const enabled = {
      orderQuantity: form.orderQuantity > 0,
      orderQuote: form.orderQuote > 0,
      orderQuoteFraction: form.orderQuoteFraction > 0,
    };
    const active = Object.entries(enabled)
      .filter(([, on]) => on)
      .map(([k]) => k) as Array<keyof typeof enabled>;
    const conflicts = active.length > 1;

    let effective: keyof typeof enabled | "none" = "none";
    if (enabled.orderQuantity) effective = "orderQuantity";
    else if (enabled.orderQuote) effective = "orderQuote";
    else if (enabled.orderQuoteFraction) effective = "orderQuoteFraction";

    const label =
      effective === "orderQuantity"
        ? `orderQuantity = ${fmtNum(form.orderQuantity, 8)} (base units)`
        : effective === "orderQuote"
          ? `orderQuote = ${fmtMoney(form.orderQuote, 2)} (quote units)`
          : effective === "orderQuoteFraction"
            ? `orderQuoteFraction = ${fmtPct(form.orderQuoteFraction, 2)}${form.maxOrderQuote > 0 ? ` (cap ${fmtMoney(form.maxOrderQuote, 2)})` : ""}`
            : "none";

    const hint =
      effective === "none"
        ? "Set one sizing input. Precedence: orderQuantity → orderQuote → orderQuoteFraction."
        : `Effective sizing: ${label}. Precedence: orderQuantity → orderQuote → orderQuoteFraction (fraction applies to BUYs).`;

    return { active, conflicts, effective, hint };
  }, [form.maxOrderQuote, form.orderQuantity, form.orderQuote, form.orderQuoteFraction]);

  const idempotencyKeyError = useMemo(() => {
    const k = form.idempotencyKey.trim();
    if (!k) return null;
    if (k.length > 36) return "Idempotency key is too long (max 36 chars).";
    if (!/^[A-Za-z0-9_-]+$/.test(k)) return "Idempotency key may only contain letters, numbers, _ and -.";
    return null;
  }, [form.idempotencyKey]);

  const requestPreviewKind = useMemo<RequestKind>(() => {
    return state.lastKind ?? (form.optimizeOperations || form.sweepThreshold ? "backtest" : "signal");
  }, [form.optimizeOperations, form.sweepThreshold, state.lastKind]);

  const requestPreview = useMemo<ApiParams>(() => {
    return requestPreviewKind === "trade" ? tradeParams : commonParams;
  }, [commonParams, requestPreviewKind, tradeParams]);

  const curlFor = useMemo(() => {
    const endpoint = requestPreviewKind === "signal" ? "/signal" : requestPreviewKind === "backtest" ? "/backtest" : "/trade";
    const json = JSON.stringify(requestPreview);
    const safe = escapeSingleQuotes(json);
    const token = apiToken.trim();
    const auth = token ? ` -H 'Authorization: Bearer ${escapeSingleQuotes(token)}' -H 'X-API-Key: ${escapeSingleQuotes(token)}'` : "";
    const base =
      /^https?:\/\//.test(apiBase)
        ? apiBase
        : typeof window !== "undefined"
          ? `${window.location.origin}${apiBase}`
          : apiBase;
    return `curl -s -X POST ${base}${endpoint} -H 'Content-Type: application/json'${auth} -d '${safe}'`;
  }, [apiBase, apiToken, requestPreview, requestPreviewKind]);

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
            <span className={form.binanceLive ? "dot dotWarn" : "dot"} aria-hidden="true" />
            {form.binanceLive ? "Live orders" : "Test orders"}
          </span>
          <span className="pill">
            <span className={form.tradeArmed ? "dot dotWarn" : "dot"} aria-hidden="true" />
            {form.tradeArmed ? "Trading armed" : "Trading locked"}
          </span>
          <span className="pill">
            {import.meta.env.DEV && apiBase === "/api" ? (
              <>
                Proxy: <span style={{ fontFamily: "var(--mono)" }}>/api → {API_TARGET}</span>
              </>
            ) : (
              <>
                API: <span style={{ fontFamily: "var(--mono)" }}>{apiBase}</span>
              </>
            )}
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
                <label className="label">API</label>
                <div className="kv">
                  <div className="k">Base URL</div>
                  <div className="v">
                    <span className="tdMono">{apiBase}</span>
                  </div>
                </div>
                <div className="kv">
                  <div className="k">Token</div>
                  <div className="v">{apiToken.trim() ? "configured" : "not set"}</div>
                </div>
                <div className="hint" style={{ marginTop: 6 }}>
                  Configured at deploy time via <span style={{ fontFamily: "var(--mono)" }}>trader-config.js</span> (apiBaseUrl, apiToken).
                </div>
                {apiBaseError ? (
                  <div className="hint" style={{ color: "rgba(239, 68, 68, 0.85)", marginTop: 6 }}>
                    {apiBaseError}
                  </div>
                ) : null}
                {healthInfo?.computeLimits ? (
                  <div className="hint" style={{ marginTop: 6 }}>
                    {healthInfo.version ? (
                      <>
                        API build:{" "}
                        <span className="tdMono">
                          {healthInfo.version}
                          {healthInfo.commit ? ` (${healthInfo.commit.slice(0, 12)})` : ""}
                        </span>
                        .{" "}
                      </>
                    ) : null}
                    {typeof healthInfo.authRequired === "boolean" ? (
                      <>
                        Auth:{" "}
                        {healthInfo.authRequired ? (healthInfo.authOk ? "required (ok)" : "required (failed)") : "not required"}.
                      </>
                    ) : null}{" "}
                    API limits: max LSTM bars {healthInfo.computeLimits.maxBarsLstm}, epochs {healthInfo.computeLimits.maxEpochs}, hidden{" "}
                    {healthInfo.computeLimits.maxHiddenSize}.
                    {healthInfo.asyncJobs
                      ? ` Async: max running ${healthInfo.asyncJobs.maxRunning}, TTL ${Math.round(
                          healthInfo.asyncJobs.ttlMs / 60000,
                        )}m, persistence ${healthInfo.asyncJobs.persistence ? "on" : "off"}.`
                      : ""}
                    {healthInfo.cache
                      ? ` Cache: ${healthInfo.cache.enabled ? "on" : "off"} (TTL ${Math.round(healthInfo.cache.ttlMs / 1000)}s, max ${
                          healthInfo.cache.maxEntries
                        }).`
                      : ""}
                  </div>
                ) : null}
                {healthInfo?.cache ? (
                  <div style={{ marginTop: 10 }}>
                    <div className="actions" style={{ marginTop: 0 }}>
                      <button className="btn" type="button" onClick={() => void refreshCacheStats()} disabled={cacheUi.loading || apiOk !== "ok"}>
                        {cacheUi.loading ? "Loading…" : "Refresh cache stats"}
                      </button>
                      <button
                        className="btn"
                        type="button"
                        onClick={() => void clearCacheUi()}
                        disabled={cacheUi.loading || apiOk !== "ok" || healthInfo.cache.enabled === false}
                      >
                        Clear cache
                      </button>
                      <span className="hint">Disable via `TRADER_API_CACHE_TTL_MS=0` if you never want cached results.</span>
                    </div>
                    {cacheUi.error ? (
                      <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)", marginTop: 8 }}>
                        {cacheUi.error}
                      </pre>
                    ) : null}
                    {cacheUi.stats ? (
                      <div className="hint" style={{ marginTop: 8 }}>
                        Signals: {cacheUi.stats.signals.entries} entries ({cacheUi.stats.signals.hits} hit / {cacheUi.stats.signals.misses} miss) • Backtests:{" "}
                        {cacheUi.stats.backtests.entries} entries ({cacheUi.stats.backtests.hits} hit / {cacheUi.stats.backtests.misses} miss) • Updated{" "}
                        {fmtTimeMs(cacheUi.stats.atMs)}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>

            {apiOk === "down" || apiOk === "auth" ? (
              <div className="row" style={{ gridTemplateColumns: "1fr" }}>
                <div className="field">
                  <label className="label">Connection</label>
                  <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)" }}>
                    {apiOk === "down"
                      ? showLocalStartHelp
                        ? `Backend unreachable.\n\nStart it with:\ncd haskell && cabal run -v0 trader-hs -- --serve --port ${API_PORT}`
                        : "Backend unreachable.\n\nConfigure apiBaseUrl in trader-config.js, or configure CloudFront to forward `/api/*` to your API origin."
                      : apiToken.trim()
                        ? "API auth failed.\n\nUpdate apiToken in trader-config.js (it must match the backend’s TRADER_API_TOKEN)."
                        : "API auth required.\n\nSet apiToken in trader-config.js (it must match the backend’s TRADER_API_TOKEN)."}
                  </pre>
                  <div className="actions" style={{ marginTop: 0 }}>
                    {apiOk === "down" && showLocalStartHelp ? (
                      <button
                        className="btn"
                        type="button"
                        onClick={() => {
                          void copyText(`cd haskell && cabal run -v0 trader-hs -- --serve --port ${API_PORT}`);
                          showToast("Copied start command");
                        }}
                      >
                        Copy start command
                      </button>
                    ) : null}
                    <button className="btn" type="button" onClick={() => void recheckHealth()}>
                      Re-check
                    </button>
                  </div>
                </div>
              </div>
            ) : null}

            <div className="row" style={{ gridTemplateColumns: "1fr" }}>
              <div className="field">
                <label className="label">Binance API keys (optional)</label>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr auto auto", alignItems: "center" }}>
                  <input
                    className="input"
                    type={revealSecrets ? "text" : "password"}
                    value={binanceApiKey}
                    onChange={(e) => setBinanceApiKey(e.target.value)}
                    placeholder="BINANCE_API_KEY"
                    spellCheck={false}
                    autoCapitalize="none"
                    autoCorrect="off"
                    inputMode="text"
                  />
                  <input
                    className="input"
                    type={revealSecrets ? "text" : "password"}
                    value={binanceApiSecret}
                    onChange={(e) => setBinanceApiSecret(e.target.value)}
                    placeholder="BINANCE_API_SECRET"
                    spellCheck={false}
                    autoCapitalize="none"
                    autoCorrect="off"
                    inputMode="text"
                  />
                  <button className="btn" type="button" onClick={() => setRevealSecrets((v) => !v)}>
                    {revealSecrets ? "Hide" : "Show"}
                  </button>
                  <button
                    className="btn"
                    type="button"
                    onClick={() => {
                      setBinanceApiKey("");
                      setBinanceApiSecret("");
                    }}
                    disabled={!binanceApiKey.trim() && !binanceApiSecret.trim()}
                  >
                    Clear
                  </button>
                </div>
                <div className="hint">
                  Used for /trade and “Check keys”. Stored in {persistSecrets ? "local storage" : "session storage"}. The request preview/curl omits it.
                </div>
                <div className="pillRow" style={{ marginTop: 10 }}>
                  <label className="pill">
                    <input type="checkbox" checked={persistSecrets} onChange={(e) => setPersistSecrets(e.target.checked)} />
                    Remember Binance keys
                  </label>
                </div>
                <div className="hint">
                  When enabled, the Binance keys are stored in local storage so you can reopen the app later without re-entering them (not recommended on shared
                  machines).
                </div>
              </div>
            </div>

            <div className="row" style={{ gridTemplateColumns: "1fr" }}>
              <div className="field">
                <label className="label">Profiles</label>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr", alignItems: "center" }}>
                  <select className="select" value={profileSelected} onChange={(e) => setProfileSelected(e.target.value)}>
                    <option value="">{profileNames.length ? "Select saved profile…" : "No profiles yet"}</option>
                    {profileNames.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                  <input
                    className="input"
                    value={profileName}
                    onChange={(e) => setProfileName(e.target.value)}
                    placeholder="New profile name"
                    spellCheck={false}
                  />
                </div>
                <div className="actions" style={{ marginTop: 10 }}>
                  <button className="btn" type="button" onClick={saveProfile} disabled={!profileName.trim()}>
                    Save
                  </button>
                  <button className="btn" type="button" onClick={requestLoadProfile} disabled={!profileSelected.trim()}>
                    Load
                  </button>
                  <button className="btn btnDanger" type="button" onClick={deleteProfile} disabled={!profileSelected.trim()}>
                    Delete
                  </button>
                </div>
                <div className="hint">Save/load named config presets. Does not include Binance keys.</div>

                {pendingProfileLoad ? (
                  <>
                    <pre className="code" style={{ borderColor: "rgba(245, 158, 11, 0.35)", marginTop: 10 }}>
                      Loading “{pendingProfileLoad.name}” will: {pendingProfileLoad.reasons.join(", ")}.
                    </pre>
                    <div className="actions" style={{ marginTop: 10 }}>
                      <button
                        className="btn btnPrimary"
                        type="button"
                        onClick={() => {
                          setForm(pendingProfileLoad.profile);
                          setPendingProfileLoad(null);
                          showToast(`Profile loaded: ${pendingProfileLoad.name}`);
                        }}
                      >
                        Load profile
                      </button>
                      <button className="btn" type="button" onClick={() => setPendingProfileLoad(null)}>
                        Cancel
                      </button>
                    </div>
                  </>
              ) : null}
            </div>
          </div>

          <div className="row" style={{ gridTemplateColumns: "1fr" }}>
            <div className="field">
              <label className="label">Optimizer combos</label>
              <TopCombosChart
                combos={topCombos}
                loading={topCombosLoading}
                error={topCombosError}
                selectedId={selectedComboId}
                onSelect={handleComboSelect}
              />
              <div className="hint">Click a combo to preload its parameters into the form (bars=0 runs the full dataset).</div>
            </div>
          </div>

            <div className="row">
              <div className="field">
                <label className="label" htmlFor="symbol">
                  Binance symbol
                </label>
                <input
                  id="symbol"
                  className={missingSymbol ? "input inputError" : "input"}
                  value={form.binanceSymbol}
                  onChange={(e) => setForm((f) => ({ ...f, binanceSymbol: e.target.value.toUpperCase() }))}
                  placeholder="BTCUSDT"
                  spellCheck={false}
                />
                <div className="hint" style={missingSymbol ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
                  {missingSymbol ? "Required." : "Use a spot symbol like BTCUSDT (USDT-margined futures also use the same symbol)."}
                </div>
              </div>

              <div className="field">
                <label className="label" htmlFor="market">
                  Market
                </label>
                <select
                  id="market"
                  className="select"
                  value={form.market}
                  onChange={(e) => {
                    const market = e.target.value as Market;
                    setPendingMarket(null);
                    setPendingProfileLoad(null);
                    if (market === "margin" && !form.binanceLive) {
                      setPendingMarket(market);
                      return;
                    }
                    setForm((f) => ({ ...f, market, binanceTestnet: market === "margin" ? false : f.binanceTestnet }));
                  }}
                >
                  <option value="spot">Spot</option>
                  <option value="margin">Margin</option>
                  <option value="futures">Futures (USDT-M)</option>
                </select>
                <div className="hint">Margin orders require live mode. Futures can close positions via reduce-only.</div>
                {pendingMarket === "margin" ? (
                  <>
                    <pre className="code" style={{ borderColor: "rgba(245, 158, 11, 0.35)", marginTop: 10 }}>
                      Switching to Margin requires enabling Live orders (Binance has no margin test endpoint). This will place real orders once you arm trading and
                      trade.
                    </pre>
                    <div className="actions" style={{ marginTop: 10 }}>
                      <button
                        className="btn btnPrimary"
                        type="button"
                        onClick={() => {
                          setForm((f) => ({ ...f, market: "margin", binanceTestnet: false, binanceLive: true }));
                          setPendingMarket(null);
                          setConfirmLive(false);
                          showToast("Live orders enabled (required for margin)");
                        }}
                      >
                        Enable live + switch
                      </button>
                      <button className="btn" type="button" onClick={() => setPendingMarket(null)}>
                        Cancel
                      </button>
                    </div>
                  </>
                ) : null}
              </div>
            </div>

            <div className="row" style={{ marginTop: 12 }}>
              <div className="field">
                <label className="label" htmlFor="interval">
                  Interval
                </label>
                <select
                  id="interval"
                  className={missingInterval ? "select selectError" : "select"}
                  value={form.interval}
                  onChange={(e) => setForm((f) => ({ ...f, interval: e.target.value }))}
                >
                  {BINANCE_INTERVALS.map((v) => (
                    <option key={v} value={v}>
                      {v}
                    </option>
                  ))}
                </select>
                <div className="hint" style={missingInterval ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
                  {missingInterval ? "Required." : "Binance interval: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M."}
                </div>
              </div>
              <div className="field">
                <label className="label" htmlFor="bars">
                  Bars (0=auto, 2–1000)
                </label>
                <input
                  id="bars"
                  className={barsExceedsApi ? "input inputError" : "input"}
                  type="number"
                  min={0}
                  max={1000}
                  value={form.bars}
                  onChange={(e) => setForm((f) => ({ ...f, bars: numFromInput(e.target.value, f.bars) }))}
                />
                <div className="hint" style={barsExceedsApi ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
                  {barsExceedsApi
                    ? `API limit: max ${apiComputeLimits?.maxBarsLstm ?? "?"} bars for LSTM methods. Reduce bars or use method=10 (Kalman-only).`
                    : "0=auto (Binance uses 500; CSV uses all). For Binance, 2–1000 is allowed. Larger values take longer."
                  }
                </div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 12 }}>
              <div className="field">
                <label className="label" htmlFor="lookbackWindow">
                  Lookback window
                </label>
                <input
                  id="lookbackWindow"
                  className={lookbackState.error && !lookbackState.overrideOn ? "input inputError" : "input"}
                  value={form.lookbackWindow}
                  disabled={form.lookbackBars >= 2}
                  onChange={(e) => setForm((f) => ({ ...f, lookbackWindow: e.target.value }))}
                  placeholder="24h"
                  spellCheck={false}
                />
                <div className="hint" style={lookbackState.error && !lookbackState.overrideOn ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
                  {form.lookbackBars >= 2 ? "Ignored while Lookback bars override is set." : lookbackState.error ?? lookbackState.summary}
                </div>
              </div>
              <div className="field">
                <label className="label" htmlFor="lookbackBars">
                  Lookback bars override (optional)
                </label>
                <input
                  id="lookbackBars"
                  className={lookbackState.error && lookbackState.overrideOn ? "input inputError" : "input"}
                  type="number"
                  min={0}
                  value={form.lookbackBars}
                  onChange={(e) => setForm((f) => ({ ...f, lookbackBars: numFromInput(e.target.value, f.lookbackBars) }))}
                  placeholder="0 (auto)"
                />
                <div className="hint" style={lookbackState.error && lookbackState.overrideOn ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
                  {lookbackState.overrideOn ? lookbackState.error ?? lookbackState.summary : "0 = use lookbackWindow. Set ≥2 to override."}
                </div>
                {form.lookbackBars > 0 || lookbackState.error ? (
                  <div className="actions" style={{ marginTop: 8 }}>
                    <button
                      className="btn"
                      type="button"
                      disabled={lookbackState.bars < 3}
                      onClick={() => setForm((f) => ({ ...f, lookbackBars: Math.max(2, lookbackState.bars - 1) }))}
                    >
                      Fit lookback to bars ({Math.max(0, lookbackState.bars - 1)})
                    </button>
                    <button
                      className="btn"
                      type="button"
                      disabled={form.lookbackBars <= 0}
                      onClick={() => setForm((f) => ({ ...f, lookbackBars: 0 }))}
                    >
                      Clear override
                    </button>
                  </div>
                ) : null}
              </div>
            </div>

            <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
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
                <div className="hint">“11” only trades when both models agree on direction (up/down) outside the open threshold.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="positioning">
                  Positioning
                </label>
                <select
                  id="positioning"
                  className="select"
                  value={form.positioning}
                  onChange={(e) => setForm((f) => ({ ...f, positioning: e.target.value as Positioning }))}
                >
                  <option value="long-flat">Long / Flat</option>
                  <option value="long-short">Long / Short (futures)</option>
                </select>
                <div
                  className="hint"
                  style={
                    form.positioning === "long-short" && form.market !== "futures"
                      ? { color: "rgba(245, 158, 11, 0.9)" }
                      : undefined
                  }
                >
                  {form.positioning === "long-short" && form.market !== "futures"
                    ? "Long/Short trading requires Futures market (switch Market to Futures to trade)."
                    : "Down signals go FLAT (long/flat) or SHORT (long/short)."}
                </div>
              </div>
              <div className="field">
                <label className="label" htmlFor="openThreshold">
                  Open threshold (fraction)
                </label>
                <input
                  id="openThreshold"
                  className={estimatedCosts.breakEven > 0 && form.openThreshold < estimatedCosts.breakEven ? "input inputError" : "input"}
                  type="number"
                  step="0.0001"
                  min={0}
                  value={form.openThreshold}
                  onChange={(e) => setForm((f) => ({ ...f, openThreshold: numFromInput(e.target.value, f.openThreshold) }))}
                />
                <div className="hint">
                  Entry deadband. Default 0.001 = 0.1%. Break-even ≈ {fmtPct(estimatedCosts.breakEven, 3)} (round-trip cost ≈ {fmtPct(estimatedCosts.roundTrip, 3)}).
                  {estimatedCosts.breakEven > 0 && form.openThreshold < estimatedCosts.breakEven
                    ? " Consider increasing open threshold to avoid churn after costs."
                    : null}
                </div>
                <div className="pillRow" style={{ marginTop: 10 }}>
                  <button
                    className="btnSmall"
                    type="button"
                    disabled={!(estimatedCosts.breakEven > 0)}
                    onClick={() => {
                      const be = estimatedCosts.breakEven;
                      const open = Number((be * 2).toFixed(6));
                      const close = Number(be.toFixed(6));
                      setForm((f) => ({ ...f, openThreshold: open, closeThreshold: close }));
                      showToast("Set thresholds to conservative (2× break-even)");
                    }}
                  >
                    Conservative (2× BE)
                  </button>
                  <button
                    className="btnSmall"
                    type="button"
                    disabled={!(estimatedCosts.breakEven > 0)}
                    onClick={() => {
                      const v = Number(estimatedCosts.breakEven.toFixed(6));
                      setForm((f) => ({ ...f, openThreshold: v, closeThreshold: v }));
                      showToast("Set thresholds to break-even");
                    }}
                  >
                    Set open/close to break-even
                  </button>
                  <button
                    className="btnSmall"
                    type="button"
                    onClick={() => {
                      setForm((f) => ({ ...f, openThreshold: defaultForm.openThreshold, closeThreshold: defaultForm.closeThreshold }));
                      showToast("Reset thresholds to defaults");
                    }}
                  >
                    Reset thresholds
                  </button>
                </div>
              </div>
              <div className="field">
                <label className="label" htmlFor="closeThreshold">
                  Close threshold (fraction)
                </label>
                <input
                  id="closeThreshold"
                  className={
                    estimatedCosts.breakEven > 0 && form.closeThreshold < estimatedCosts.breakEven
                      ? "input inputError"
                      : form.closeThreshold > form.openThreshold
                        ? "input inputWarn"
                        : "input"
                  }
                  type="number"
                  step="0.0001"
                  min={0}
                  value={form.closeThreshold}
                  onChange={(e) => setForm((f) => ({ ...f, closeThreshold: numFromInput(e.target.value, f.closeThreshold) }))}
                />
                <div className="hint">
                  Exit deadband. Often smaller than open threshold to reduce churn.
                  {estimatedCosts.breakEven > 0 && form.closeThreshold < estimatedCosts.breakEven ? " Below break-even (may churn after costs)." : null}
                </div>
                {form.closeThreshold > form.openThreshold ? (
                  <div className="hint" style={{ color: "rgba(245, 158, 11, 0.9)" }}>
                    Close threshold is above open threshold (inverted hysteresis). Usually close ≤ open.
                  </div>
                ) : null}
              </div>
            </div>

            <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr 1fr 1fr" }}>
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
                <label className="label" htmlFor="tuneRatio">
                  Tune ratio
                </label>
                <input
                  id="tuneRatio"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  max={0.99}
                  value={form.tuneRatio}
                  onChange={(e) => setForm((f) => ({ ...f, tuneRatio: numFromInput(e.target.value, f.tuneRatio) }))}
                />
                <div className="hint">Used only when optimizing/sweeping: tunes thresholds/method on the last part of the train split.</div>
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
                <div className="hint">
                  Per-side ≈ {fmtPct(Math.max(0, form.fee) + Math.max(0, form.slippage) + Math.max(0, form.spread) / 2, 3)} (fee + slippage + spread/2).
                </div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr 1fr" }}>
              <div className="field">
                <label className="label" htmlFor="slippage">
                  Slippage (fraction per side)
                </label>
                <input
                  id="slippage"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={form.slippage}
                  onChange={(e) => setForm((f) => ({ ...f, slippage: numFromInput(e.target.value, f.slippage) }))}
                />
                <div className="hint">Approx market impact on entry/exit. 0 disables.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="spread">
                  Spread (fraction total)
                </label>
                <input
                  id="spread"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={form.spread}
                  onChange={(e) => setForm((f) => ({ ...f, spread: numFromInput(e.target.value, f.spread) }))}
                />
                <div className="hint">Half is applied per side. 0 disables.</div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr" }}>
              <div className="field">
                <label className="label">Bracket exits (fractions)</label>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
                  <div className="field">
                    <label className="label" htmlFor="stopLoss">
                      Stop-loss
                    </label>
                    <input
                      id="stopLoss"
                      className="input"
                      type="number"
                      step="0.001"
                      min={0}
                      max={0.999}
                      value={form.stopLoss}
                      onChange={(e) => setForm((f) => ({ ...f, stopLoss: numFromInput(e.target.value, f.stopLoss) }))}
                      placeholder="0.02 (2%)"
                    />
                    <div className="hint">{form.stopLoss > 0 ? fmtPct(form.stopLoss, 2) : "0 disables"}</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="takeProfit">
                      Take-profit
                    </label>
                    <input
                      id="takeProfit"
                      className="input"
                      type="number"
                      step="0.001"
                      min={0}
                      max={0.999}
                      value={form.takeProfit}
                      onChange={(e) => setForm((f) => ({ ...f, takeProfit: numFromInput(e.target.value, f.takeProfit) }))}
                      placeholder="0.03 (3%)"
                    />
                    <div className="hint">{form.takeProfit > 0 ? fmtPct(form.takeProfit, 2) : "0 disables"}</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="trailingStop">
                      Trailing stop
                    </label>
                    <input
                      id="trailingStop"
                      className="input"
                      type="number"
                      step="0.001"
                      min={0}
                      max={0.999}
                      value={form.trailingStop}
                      onChange={(e) => setForm((f) => ({ ...f, trailingStop: numFromInput(e.target.value, f.trailingStop) }))}
                      placeholder="0.01 (1%)"
                    />
                    <div className="hint">{form.trailingStop > 0 ? fmtPct(form.trailingStop, 2) : "0 disables"}</div>
                  </div>
	                </div>
	                <div className="row" style={{ gridTemplateColumns: "1fr" }}>
	                  <div className="field">
	                    <label className="label" htmlFor="intrabarFill">
	                      Intrabar fill
	                    </label>
	                    <select
	                      id="intrabarFill"
	                      className="select"
	                      value={form.intrabarFill}
	                      onChange={(e) => setForm((f) => ({ ...f, intrabarFill: e.target.value as IntrabarFill }))}
	                    >
	                      <option value="stop-first">Stop-first (conservative)</option>
	                      <option value="take-profit-first">Take-profit-first (optimistic)</option>
	                    </select>
	                    <div className="hint">If take-profit and stop are both hit within a bar.</div>
	                  </div>
	                </div>
	                <div className="hint">Optional bracket exits (uses OHLC high/low when available; otherwise close-only). Example: 0.02 = 2%.</div>
	              </div>
	            </div>

            <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr" }}>
              <div className="field">
                <label className="label">Trade pacing (bars)</label>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                  <div className="field">
                    <label className="label" htmlFor="minHoldBars">
                      Min hold
                    </label>
                    <input
                      id="minHoldBars"
                      className="input"
                      type="number"
                      step={1}
                      min={0}
                      value={form.minHoldBars}
                      onChange={(e) => setForm((f) => ({ ...f, minHoldBars: numFromInput(e.target.value, f.minHoldBars) }))}
                      placeholder="0"
                    />
                    <div className="hint">
                      {form.minHoldBars > 0 ? `${Math.trunc(Math.max(0, form.minHoldBars))} bars` : "0 disables"} • blocks signal exits, not bracket exits
                    </div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="cooldownBars">
                      Cooldown
                    </label>
                    <input
                      id="cooldownBars"
                      className="input"
                      type="number"
                      step={1}
                      min={0}
                      value={form.cooldownBars}
                      onChange={(e) => setForm((f) => ({ ...f, cooldownBars: numFromInput(e.target.value, f.cooldownBars) }))}
                      placeholder="0"
                    />
                    <div className="hint">{form.cooldownBars > 0 ? `${Math.trunc(Math.max(0, form.cooldownBars))} bars` : "0 disables"} • after exiting</div>
                  </div>
                </div>
                <div className="hint">Helps reduce churn in noisy markets (applies to backtests + live bot; stateless signals/trades ignore state).</div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr" }}>
              <div className="field">
                <label className="label">Risk kill-switches</label>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
                  <div className="field">
                    <label className="label" htmlFor="maxDrawdown">
                      Max drawdown
                    </label>
                    <input
                      id="maxDrawdown"
                      className="input"
                      type="number"
                      step="0.01"
                      min={0}
                      max={0.999}
                      value={form.maxDrawdown}
                      onChange={(e) => setForm((f) => ({ ...f, maxDrawdown: numFromInput(e.target.value, f.maxDrawdown) }))}
                      placeholder="0.20 (20%)"
                    />
                    <div className="hint">{form.maxDrawdown > 0 ? fmtPct(form.maxDrawdown, 2) : "0 disables"}</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="maxDailyLoss">
                      Max daily loss
                    </label>
                    <input
                      id="maxDailyLoss"
                      className="input"
                      type="number"
                      step="0.01"
                      min={0}
                      max={0.999}
                      value={form.maxDailyLoss}
                      onChange={(e) => setForm((f) => ({ ...f, maxDailyLoss: numFromInput(e.target.value, f.maxDailyLoss) }))}
                      placeholder="0.10 (10%)"
                    />
                    <div className="hint">{form.maxDailyLoss > 0 ? fmtPct(form.maxDailyLoss, 2) : "0 disables"}</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="maxOrderErrors">
                      Max order errors
                    </label>
                    <input
                      id="maxOrderErrors"
                      className="input"
                      type="number"
                      step="1"
                      min={0}
                      value={form.maxOrderErrors}
                      onChange={(e) => setForm((f) => ({ ...f, maxOrderErrors: numFromInput(e.target.value, f.maxOrderErrors) }))}
                      placeholder="3"
                    />
                    <div className="hint">{form.maxOrderErrors >= 1 ? `${Math.trunc(form.maxOrderErrors)} errors` : "0 disables"}</div>
                  </div>
                </div>
                <div className="hint">When set, the live bot halts (and forces exit) on max drawdown, max daily loss, or consecutive order failures.</div>
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
                    className={epochsExceedsApi ? "input inputError" : "input"}
                    type="number"
                    min={0}
                    value={form.epochs}
                    onChange={(e) => setForm((f) => ({ ...f, epochs: numFromInput(e.target.value, f.epochs) }))}
                  />
                  <input
                    aria-label="Hidden size"
                    className={hiddenSizeExceedsApi ? "input inputError" : "input"}
                    type="number"
                    min={1}
                    value={form.hiddenSize}
                    onChange={(e) => setForm((f) => ({ ...f, hiddenSize: numFromInput(e.target.value, f.hiddenSize) }))}
                  />
                </div>
                <div className="hint" style={epochsExceedsApi || hiddenSizeExceedsApi ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
                  {!apiLstmEnabled
                    ? "Ignored for Kalman-only."
                    : epochsExceedsApi || hiddenSizeExceedsApi
                      ? `API limits: epochs ≤ ${apiComputeLimits?.maxEpochs ?? "?"}, hidden ≤ ${apiComputeLimits?.maxHiddenSize ?? "?"}.`
                      : "Higher = slower. For quick iteration, reduce epochs."}
                </div>
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
                    Sweep thresholds
                  </label>
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.optimizeOperations}
                      onChange={(e) => setForm((f) => ({ ...f, optimizeOperations: e.target.checked, sweepThreshold: false }))}
                    />
                    Optimize operations (method + thresholds)
                  </label>
                </div>
                <div className="hint">Tunes on the last part of the train split (fit/tune), then evaluates on the held-out backtest.</div>
                <div className="pillRow" style={{ marginTop: 10 }}>
                  <button
                    className="btnSmall"
                    type="button"
                    onClick={() => {
                      setForm((f) => ({
                        ...f,
                        optimizeOperations: true,
                        sweepThreshold: false,
                        minRoundTrips: Math.max(5, Math.trunc(f.minRoundTrips)),
                        walkForwardFolds: Math.max(5, Math.trunc(f.walkForwardFolds)),
                      }));
                      showToast("Preset: safe optimize (min round trips + folds)");
                    }}
                  >
                    Preset: Safe optimize
                  </button>
                  <button
                    className="btnSmall"
                    type="button"
                    onClick={() => {
                      setForm((f) => ({
                        ...f,
                        sweepThreshold: true,
                        optimizeOperations: false,
                        minRoundTrips: Math.max(3, Math.trunc(f.minRoundTrips)),
                        walkForwardFolds: Math.max(3, Math.trunc(f.walkForwardFolds)),
                      }));
                      showToast("Preset: fast sweep (min round trips + folds)");
                    }}
                  >
                    Preset: Fast sweep
                  </button>
                </div>
                <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr 1fr 1fr" }}>
                  <div className="field">
                    <label className="label" htmlFor="tuneObjective">
                      Tune objective
                    </label>
                    <select
                      id="tuneObjective"
                      className="select"
                      value={form.tuneObjective}
                      onChange={(e) => setForm((f) => ({ ...f, tuneObjective: e.target.value }))}
                    >
                      {TUNE_OBJECTIVES.map((o) => (
                        <option key={o} value={o}>
                          {o}
                        </option>
                      ))}
                    </select>
                    <div className="hint">Used by “Optimize thresholds/operations”.</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="tunePenaltyMaxDrawdown">
                      DD penalty
                    </label>
                    <input
                      id="tunePenaltyMaxDrawdown"
                      className="input"
                      type="number"
                      step="0.1"
                      min={0}
                      value={form.tunePenaltyMaxDrawdown}
                      onChange={(e) => setForm((f) => ({ ...f, tunePenaltyMaxDrawdown: numFromInput(e.target.value, f.tunePenaltyMaxDrawdown) }))}
                    />
                    <div className="hint">Applied when objective includes drawdown.</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="tunePenaltyTurnover">
                      Turnover penalty
                    </label>
                    <input
                      id="tunePenaltyTurnover"
                      className="input"
                      type="number"
                      step="0.01"
                      min={0}
                      value={form.tunePenaltyTurnover}
                      onChange={(e) => setForm((f) => ({ ...f, tunePenaltyTurnover: numFromInput(e.target.value, f.tunePenaltyTurnover) }))}
                    />
                    <div className="hint">Applied when objective includes turnover.</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="minRoundTrips">
                      Min round trips
                    </label>
                    <input
                      id="minRoundTrips"
                      className="input"
                      type="number"
                      step="1"
                      min={0}
                      value={form.minRoundTrips}
                      onChange={(e) => setForm((f) => ({ ...f, minRoundTrips: numFromInput(e.target.value, f.minRoundTrips) }))}
                    />
                    <div className="hint">Only used when optimizing/sweeping. 0 disables.</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="walkForwardFolds">
                      Walk-forward folds
                    </label>
                    <input
                      id="walkForwardFolds"
                      className="input"
                      type="number"
                      step="1"
                      min={1}
                      value={form.walkForwardFolds}
                      onChange={(e) => setForm((f) => ({ ...f, walkForwardFolds: numFromInput(e.target.value, f.walkForwardFolds) }))}
                    />
                    <div className="hint">Used for tune scoring + backtest variability.</div>
                  </div>
                </div>
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
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.bypassCache}
                      onChange={(e) => setForm((f) => ({ ...f, bypassCache: e.target.checked }))}
                    />
                    Bypass cache
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
                  seconds. {form.bypassCache ? "Bypass cache adds Cache-Control: no-cache." : ""}
                </div>
              </div>
            </div>

            <div className="actions">
              <button
                className="btn btnPrimary"
                disabled={requestDisabled}
                onClick={() => run("signal")}
                title={requestDisabledReason ?? undefined}
              >
                {state.loading && state.lastKind === "signal" ? "Getting signal…" : "Get signal"}
              </button>
              <button className="btn" disabled={requestDisabled} onClick={() => run("backtest")} title={requestDisabledReason ?? undefined}>
                {state.loading && state.lastKind === "backtest" ? "Running backtest…" : "Run backtest"}
              </button>
              <button
                className="btn"
                disabled={requestDisabled}
                title={requestDisabledReason ?? undefined}
                onClick={() => {
                  const p = { ...commonParams, sweepThreshold: true, optimizeOperations: false };
                  setForm((f) => ({ ...f, sweepThreshold: true, optimizeOperations: false }));
                  void run("backtest", p);
                }}
              >
                {state.loading && state.lastKind === "backtest" ? "Optimizing…" : "Optimize thresholds"}
              </button>
              <button
                className="btn"
                disabled={requestDisabled}
                title={requestDisabledReason ?? undefined}
                onClick={() => {
                  const p = { ...commonParams, optimizeOperations: true, sweepThreshold: false };
                  setForm((f) => ({ ...f, optimizeOperations: true, sweepThreshold: false }));
                  void run("backtest", p);
                }}
              >
                {state.loading && state.lastKind === "backtest" ? "Optimizing…" : "Optimize operations"}
              </button>
              <button className="btn" disabled={!state.loading} onClick={cancelActiveRequest}>
                Cancel
              </button>
            </div>

            {rateLimitReason ? (
              <div className="hint" style={{ marginTop: 10, color: "var(--warn)" }}>
                {rateLimitReason}
              </div>
            ) : null}

            {state.loading ? (
              <div className="hint" style={{ marginTop: 10 }}>
                {activeAsyncJob?.jobId
                  ? `Async job: ${activeAsyncJob.jobId} • ${activeAsyncJob.kind} • ${Math.max(
                      0,
                      Math.floor((activeAsyncTickMs - activeAsyncJob.startedAtMs) / 1000),
                    )}s`
                  : "Starting async job…"}
              </div>
            ) : null}

            <div style={{ marginTop: 14 }}>
              <div className="row" style={{ gridTemplateColumns: "1fr" }}>
                <div className="field">
                  <label className="label">Live bot</label>
                  <div className="actions" style={{ marginTop: 0 }}>
                    <button
                      className="btn btnPrimary"
                      disabled={bot.status.running || botStartBlocked}
                      onClick={() => void startLiveBot()}
                      title={
                        firstReason(
                          botStartBlockedReason,
                          form.tradeArmed ? "Trading armed (will send orders)" : "Paper mode (no orders)",
                        ) ?? undefined
                      }
                    >
                      {bot.loading || botStarting ? "Starting…" : bot.status.running ? "Running" : "Start live bot"}
                    </button>
                    <button className="btn btnDanger" disabled={bot.loading || (!bot.status.running && !botStarting)} onClick={stopLiveBot}>
                      Stop bot
                    </button>
                    <button className="btn" disabled={bot.loading || Boolean(apiBlockedReason)} onClick={() => refreshBot()} title={apiBlockedReason ?? undefined}>
                      Refresh
                    </button>
                  </div>
                  <div className="hint">
                    Continuously ingests new bars, fine-tunes on each bar, and switches position based on the latest signal. Enable “Arm trading” to actually place
                    Binance orders; otherwise it runs in paper mode. If “Sweep thresholds” or “Optimize operations” is enabled, the bot re-optimizes after each
                    buy/sell operation.
                  </div>
                  {bot.error ? <div className="hint" style={{ color: "rgba(239, 68, 68, 0.9)", whiteSpace: "pre-wrap" }}>{bot.error}</div> : null}

                  <details className="details" style={{ marginTop: 10 }}>
                    <summary>Advanced live bot</summary>
                    <div className="row" style={{ marginTop: 10 }}>
                      <div className="field">
                        <label className="label" htmlFor="botPollSeconds">
                          Poll seconds (0 = auto)
                        </label>
                        <input
                          id="botPollSeconds"
                          className="input"
                          type="number"
                          min={0}
                          max={3600}
                          value={form.botPollSeconds}
                          onChange={(e) => setForm((f) => ({ ...f, botPollSeconds: numFromInput(e.target.value, f.botPollSeconds) }))}
                        />
                        <div className="hint">How often the bot checks for a new bar (server-side).</div>
                      </div>
                      <div className="field">
                        <label className="label" htmlFor="botOnlineEpochs">
                          Online epochs
                        </label>
                        <input
                          id="botOnlineEpochs"
                          className="input"
                          type="number"
                          min={0}
                          max={50}
                          value={form.botOnlineEpochs}
                          onChange={(e) => setForm((f) => ({ ...f, botOnlineEpochs: numFromInput(e.target.value, f.botOnlineEpochs) }))}
                        />
                        <div className="hint">0 disables per-bar fine-tuning (faster, less adaptive).</div>
                      </div>
                    </div>
                    <div className="row" style={{ marginTop: 12 }}>
                      <div className="field">
                        <label className="label">Startup position</label>
                        <div className="pillRow">
                          <label className="pill">
                            <input
                              type="checkbox"
                              checked={form.botAdoptExistingPosition}
                              onChange={(e) => setForm((f) => ({ ...f, botAdoptExistingPosition: e.target.checked }))}
                            />
                            Adopt existing long position
                          </label>
                        </div>
                        <div className="hint">If trading is enabled, allow starting while already long (resume management instead of refusing to start).</div>
                      </div>
                    </div>
                    <div className="row" style={{ marginTop: 12 }}>
                      <div className="field">
                        <label className="label" htmlFor="botTrainBars">
                          Train bars (rolling)
                        </label>
                        <input
                          id="botTrainBars"
                          className="input"
                          type="number"
                          min={10}
                          value={form.botTrainBars}
                          onChange={(e) => setForm((f) => ({ ...f, botTrainBars: numFromInput(e.target.value, f.botTrainBars) }))}
                        />
                        <div className="hint">Bars used for online fine-tuning and optimization windows.</div>
                      </div>
                      <div className="field">
                        <label className="label" htmlFor="botMaxPoints">
                          Max points (history)
                        </label>
                        <input
                          id="botMaxPoints"
                          className="input"
                          type="number"
                          min={100}
                          max={100000}
                          value={form.botMaxPoints}
                          onChange={(e) => setForm((f) => ({ ...f, botMaxPoints: numFromInput(e.target.value, f.botMaxPoints) }))}
                        />
                        <div className="hint">Caps in-memory chart/history. Larger uses more memory.</div>
                      </div>
                    </div>
                    <div className="actions" style={{ marginTop: 10 }}>
                      <button
                        className="btn"
                        type="button"
                        onClick={() =>
                          setForm((f) => ({
                            ...f,
                            botPollSeconds: defaultForm.botPollSeconds,
                            botOnlineEpochs: defaultForm.botOnlineEpochs,
                            botTrainBars: defaultForm.botTrainBars,
                            botMaxPoints: defaultForm.botMaxPoints,
                            botAdoptExistingPosition: defaultForm.botAdoptExistingPosition,
                          }))
                        }
                      >
                        Reset defaults
                      </button>
                      <span className="hint">Changes apply the next time you start the bot.</span>
                    </div>
                  </details>
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
                        disabled={form.market === "margin"}
                        onChange={(e) => {
                          setPendingProfileLoad(null);
                          if (!e.target.checked) {
                            setConfirmLive(false);
                            setForm((f) => ({ ...f, binanceLive: false }));
                            return;
                          }
                          setConfirmArm(false);
                          setConfirmLive(true);
                        }}
                      />
                      Live orders
                    </label>
                    <label className="pill">
                      <input
                        type="checkbox"
                        checked={form.tradeArmed}
                        onChange={(e) => {
                          setPendingProfileLoad(null);
                          if (!e.target.checked) {
                            setConfirmArm(false);
                            setForm((f) => ({ ...f, tradeArmed: false }));
                            return;
                          }
                          setConfirmLive(false);
                          setConfirmArm(true);
                        }}
                      />
                      Arm trading
                    </label>
                  </div>
                  <div className="hint">Trading is disabled by default. Only arm it when you’re ready.</div>
                  {form.market === "margin" ? <div className="hint">Live orders are required for margin (forced on).</div> : null}

                  {confirmLive ? (
                    <>
                      <pre className="code" style={{ borderColor: "rgba(245, 158, 11, 0.35)", marginTop: 10 }}>
                        Enable Live orders? This can place real orders on Binance when you trade or start the live bot with trading armed.
                      </pre>
                      <div className="actions" style={{ marginTop: 10 }}>
                        <button
                          className="btn btnPrimary"
                          type="button"
                          onClick={() => {
                            setForm((f) => ({ ...f, binanceLive: true }));
                            setConfirmLive(false);
                            showToast("Live orders enabled");
                          }}
                        >
                          Enable live orders
                        </button>
                        <button className="btn" type="button" onClick={() => setConfirmLive(false)}>
                          Cancel
                        </button>
                      </div>
                    </>
                  ) : null}

                  {confirmArm ? (
                    <>
                      <pre className="code" style={{ borderColor: "rgba(245, 158, 11, 0.35)", marginTop: 10 }}>
                        Arm trading? This unlocks calling /trade and allows the live bot to send orders (paper mode when unarmed).
                      </pre>
                      <div className="actions" style={{ marginTop: 10 }}>
                        <button
                          className="btn btnPrimary"
                          type="button"
                          onClick={() => {
                            setForm((f) => ({ ...f, tradeArmed: true }));
                            setConfirmArm(false);
                            showToast("Trading armed");
                          }}
                        >
                          Arm trading
                        </button>
                        <button className="btn" type="button" onClick={() => setConfirmArm(false)}>
                          Cancel
                        </button>
                      </div>
                    </>
                  ) : null}
                </div>
                <div className="field">
                  <label className="label">Order sizing</label>
                  <div className="hint" style={orderSizing.conflicts ? { color: "rgba(239, 68, 68, 0.9)" } : undefined}>
                    {orderSizing.conflicts ? `Multiple sizing inputs are set (${orderSizing.active.join(", ")}). ` : ""}
                    {orderSizing.hint}
                  </div>

                  {orderSizing.conflicts ? (
                    <div className="actions" style={{ marginTop: 10 }}>
                      <button
                        className="btn"
                        type="button"
                        onClick={() =>
                          setForm((f) => {
                            if (orderSizing.effective === "orderQuantity") return { ...f, orderQuote: 0, orderQuoteFraction: 0 };
                            if (orderSizing.effective === "orderQuote") return { ...f, orderQuantity: 0, orderQuoteFraction: 0 };
                            if (orderSizing.effective === "orderQuoteFraction") return { ...f, orderQuantity: 0, orderQuote: 0 };
                            return f;
                          })
                        }
                      >
                        Keep {orderSizing.effective} and clear others
                      </button>
                    </div>
                  ) : null}

                  <div className="row" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 8 }}>
                    <div className="field">
                      <label className="label" htmlFor="orderQuote">
                        Order quote (e.g., USDT)
                      </label>
                      <input
                        id="orderQuote"
                        className="input"
                        type="number"
                        min={0}
                        value={form.orderQuote}
                        onChange={(e) =>
                          setForm((f) => {
                            const v = numFromInput(e.target.value, f.orderQuote);
                            return v > 0 ? { ...f, orderQuote: v, orderQuantity: 0, orderQuoteFraction: 0 } : { ...f, orderQuote: v };
                          })
                        }
                        placeholder="20"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="orderQuantity">
                        Order quantity (base units)
                      </label>
                      <input
                        id="orderQuantity"
                        className="input"
                        type="number"
                        min={0}
                        value={form.orderQuantity}
                        onChange={(e) =>
                          setForm((f) => {
                            const v = numFromInput(e.target.value, f.orderQuantity);
                            return v > 0 ? { ...f, orderQuantity: v, orderQuote: 0, orderQuoteFraction: 0 } : { ...f, orderQuantity: v };
                          })
                        }
                        placeholder="0.001"
                      />
                    </div>
                  </div>

                  <div className="row" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 10 }}>
                    <div className="field">
                      <label className="label" htmlFor="orderQuoteFraction">
                        Order quote fraction (0 &lt; F ≤ 1; 0 disables)
                      </label>
                      <input
                        id="orderQuoteFraction"
                        className={orderQuoteFractionError ? "input inputError" : "input"}
                        type="number"
                        step="0.01"
                        min={0}
                        max={1}
                        value={form.orderQuoteFraction}
                        onChange={(e) =>
                          setForm((f) => {
                            const v = numFromInput(e.target.value, f.orderQuoteFraction);
                            return v > 0 ? { ...f, orderQuoteFraction: v, orderQuote: 0, orderQuantity: 0 } : { ...f, orderQuoteFraction: v };
                          })
                        }
                        placeholder="0.10 (10%)"
                      />
                      <div className="hint" style={orderQuoteFractionError ? { color: "rgba(239, 68, 68, 0.9)" } : undefined}>
                        {orderQuoteFractionError ?? "Applies to BUYs: uses a fraction of your available quote balance."}
                      </div>
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="maxOrderQuote">
                        Max quote cap (optional)
                      </label>
                      <input
                        id="maxOrderQuote"
                        className="input"
                        type="number"
                        step="1"
                        min={0}
                        disabled={form.orderQuoteFraction <= 0}
                        value={form.maxOrderQuote}
                        onChange={(e) => setForm((f) => ({ ...f, maxOrderQuote: numFromInput(e.target.value, f.maxOrderQuote) }))}
                        placeholder="0 (no cap)"
                      />
                      <div className="hint">
                        {form.orderQuoteFraction > 0 ? "Optional cap when using orderQuoteFraction." : "Enable orderQuoteFraction to use this cap."}
                      </div>
                    </div>
                  </div>

                  <label className="label" htmlFor="idempotencyKey" style={{ marginTop: 10 }}>
                    Idempotency key (optional)
                  </label>
                  <div className="row" style={{ gridTemplateColumns: "1fr auto auto", marginTop: 8, alignItems: "center" }}>
                    <input
                      id="idempotencyKey"
                      className={idempotencyKeyError ? "input inputError" : "input"}
                      value={form.idempotencyKey}
                      onChange={(e) => setForm((f) => ({ ...f, idempotencyKey: e.target.value }))}
                      placeholder="e.g. 1f6a2c7a-…"
                      spellCheck={false}
                      autoCapitalize="none"
                      autoCorrect="off"
                      inputMode="text"
                    />
                    <button
                      className="btn"
                      type="button"
                      onClick={() => setForm((f) => ({ ...f, idempotencyKey: generateIdempotencyKey() }))}
                    >
                      Generate
                    </button>
                    <button
                      className="btn"
                      type="button"
                      disabled={!form.idempotencyKey.trim()}
                      onClick={() => setForm((f) => ({ ...f, idempotencyKey: "" }))}
                    >
                      Clear
                    </button>
                  </div>
                  <div className="hint" style={idempotencyKeyError ? { color: "rgba(239, 68, 68, 0.9)" } : undefined}>
                    {idempotencyKeyError
                      ? `${idempotencyKeyError} (not sent to the API).`
                      : "Use for manual /trade retries. Leave blank for the live bot unless you know what you’re doing."}
                  </div>
                </div>
              </div>

              <div className="actions" style={{ marginTop: 10 }}>
                <button
                  className="btn btnDanger"
                  disabled={state.loading || !form.tradeArmed || Boolean(tradeDisabledReason)}
                  onClick={() => run("trade")}
                  title={tradeDisabledReason ?? (form.binanceLive ? "LIVE order mode enabled" : "Test order mode (default)")}
                >
                  {state.loading && state.lastKind === "trade" ? "Trading…" : "Trade (uses latest signal)"}
                </button>
                <button
                  className="btn"
                  disabled={!state.loading}
                  onClick={cancelActiveRequest}
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
              .{" "}
              {import.meta.env.DEV ? (
                <>
                  The UI uses a same-origin dev proxy (<span style={{ fontFamily: "var(--mono)" }}>/api</span>) to avoid CORS and reduce local attack surface.
                </>
              ) : (
                <>
                  When hosting the UI separately (CloudFront/S3), configure <span style={{ fontFamily: "var(--mono)" }}>trader-config.js</span> (apiBaseUrl,
                  apiToken) and/or route <span style={{ fontFamily: "var(--mono)" }}>/api/*</span> to your backend.
                </>
              )}
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
	                    <span className="badge">open {fmtPct(bot.status.openThreshold ?? bot.status.threshold, 3)}</span>
	                    <span className="badge">
	                      close {fmtPct(bot.status.closeThreshold ?? bot.status.openThreshold ?? bot.status.threshold, 3)}
	                    </span>
	                    <span className="badge">{bot.status.halted ? "HALTED" : "ACTIVE"}</span>
	                    <span className="badge">{bot.status.error ? "Error" : "OK"}</span>
	                  </div>

	                  <BacktestChart
	                    prices={bot.status.prices}
	                    equityCurve={bot.status.equityCurve}
	                    kalmanPredNext={bot.status.kalmanPredNext}
	                    positions={bot.status.positions}
	                    trades={bot.status.trades}
	                    operations={bot.status.operations}
	                    backtestStartIndex={bot.status.startIndex}
	                    height={360}
	                  />

		                  <div style={{ marginTop: 10 }}>
		                    <div className="hint" style={{ marginBottom: 8 }}>
		                      Prediction values vs thresholds (hover for details)
		                    </div>
		                    <PredictionDiffChart
		                      prices={bot.status.prices}
		                      kalmanPredNext={bot.status.kalmanPredNext}
		                      lstmPredNext={bot.status.lstmPredNext}
		                      startIndex={bot.status.startIndex}
		                      height={140}
		                      openThreshold={bot.status.openThreshold ?? bot.status.threshold}
		                      closeThreshold={bot.status.closeThreshold ?? bot.status.openThreshold ?? bot.status.threshold}
		                    />
		                  </div>

		                  <div style={{ marginTop: 10 }}>
		                    <div className="hint" style={{ marginBottom: 8 }}>
		                      Telemetry (Binance poll latency + close drift; hover for details)
		                    </div>
		                    <TelemetryChart points={botRt.telemetry} height={120} label="Live bot telemetry chart" />
		                  </div>

		                  <div className="kv" style={{ marginTop: 12 }}>
		                    <div className="k">Realtime</div>
		                    <div className="v">
		                      <span className="badge" style={{ marginRight: 8 }}>
		                        ui {fmtDurationMs(botRt.lastFetchDurationMs)}
	                      </span>
	                      <span className="badge" style={{ marginRight: 8 }}>
	                        binance {fmtDurationMs(botRealtime?.pollLatencyMs)} • age {fmtDurationMs(botRealtime?.pollAgeMs)}
	                      </span>
	                      <span className="badge" style={{ marginRight: 8 }}>
	                        state {fmtDurationMs(botRealtime?.statusAgeMs)}
	                      </span>
	                      <span
	                        className={botRt.lastNewCandles > 0 ? "badge badgeStrong badgeLong" : "badge"}
	                        style={{ marginRight: 8 }}
	                      >
	                        +{botRt.lastNewCandles} candle{botRt.lastNewCandles === 1 ? "" : "s"}
	                      </span>
	                      <span className={botRt.lastKlineUpdates > 0 ? "badge badgeStrong" : "badge"}>
	                        +{botRt.lastKlineUpdates} update{botRt.lastKlineUpdates === 1 ? "" : "s"}
	                      </span>
	                    </div>
	                  </div>
		                  <div className="kv">
		                    <div className="k">Binance poll</div>
		                    <div className="v">
		                      {botRealtime?.polledAtMs ? (
		                        <>
		                          {fmtTimeMs(botRealtime.polledAtMs)} • {fmtDurationMs(botRealtime.pollLatencyMs)} • klines{" "}
		                          {typeof botRealtime.fetchedKlines === "number" ? botRealtime.fetchedKlines : "—"} • next {fmtEtaMs(botRealtime.nextPollEtaMs)}
		                        </>
		                      ) : (
		                        "—"
		                      )}
		                    </div>
		                  </div>
		                  <div className="kv">
		                    <div className="k">Batch</div>
		                    <div className="v">
		                      {botRealtime?.lastBatchAtMs ? (
		                        <>
		                          {fmtTimeMs(botRealtime.lastBatchAtMs)} •{" "}
		                          {typeof botRealtime.lastBatchSize === "number"
		                            ? `${botRealtime.lastBatchSize} candle${botRealtime.lastBatchSize === 1 ? "" : "s"}`
		                            : "—"}{" "}
		                          • {fmtDurationMs(botRealtime.lastBatchMs)}
		                          {typeof botRealtime.batchPerBarMs === "number" && Number.isFinite(botRealtime.batchPerBarMs)
		                            ? ` (${fmtNum(botRealtime.batchPerBarMs, 1)}ms/bar)`
		                            : ""}
		                          {" • "}age {fmtDurationMs(botRealtime.batchAgeMs)}
		                        </>
		                      ) : (
		                        "—"
		                      )}
		                    </div>
		                  </div>
		                  <div className="kv">
		                    <div className="k">Settings</div>
		                    <div className="v">
		                      {bot.status.settings ? (
		                        <>
		                          poll {bot.status.settings.pollSeconds}s • online epochs {bot.status.settings.onlineEpochs} • train bars{" "}
		                          {bot.status.settings.trainBars} • max points {bot.status.settings.maxPoints} • trade{" "}
		                          {bot.status.settings.tradeEnabled ? "ON" : "OFF"}
		                        </>
		                      ) : (
		                        "—"
		                      )}
		                    </div>
		                  </div>
		                  <div className="kv">
		                    <div className="k">Processed candle</div>
		                    <div className="v">
		                      {botRealtime?.processedOpenTime ? (
		                        <>
		                          {fmtTimeMs(botRealtime.processedOpenTime)} • close{" "}
		                          {typeof botRealtime.processedClose === "number" ? fmtMoney(botRealtime.processedClose, 4) : "—"} • bar{" "}
		                          {botRealtime.lastBarIndex}
		                          {typeof botRealtime.behindCandles === "number" && botRealtime.behindCandles > 0
		                            ? ` • behind ${botRealtime.behindCandles}`
		                            : ""}
		                        </>
		                      ) : (
		                        "—"
		                      )}
		                    </div>
		                  </div>
		                  <div className="kv">
		                    <div className="k">Fetched candle</div>
		                    <div className="v">
		                      {botRealtime?.fetchedLast ? (
		                        <>
		                          {fmtTimeMs(botRealtime.fetchedLast.openTime)}
		                          {typeof botRealtime.behindCandles === "number" && botRealtime.behindCandles > 0
		                            ? ` • ahead +${botRealtime.behindCandles}`
		                            : ""}{" "}
		                          • O {fmtMoney(botRealtime.fetchedLast.open, 4)} H{" "}
		                          {fmtMoney(botRealtime.fetchedLast.high, 4)} L {fmtMoney(botRealtime.fetchedLast.low, 4)} C{" "}
		                          {fmtMoney(botRealtime.fetchedLast.close, 4)}
		                          {typeof botRealtime.closeDelta === "number" && Number.isFinite(botRealtime.closeDelta) ? (
		                            <>
	                              {" "}
	                              • Δ {fmtMoney(botRealtime.closeDelta, 4)}
	                              {typeof botRealtime.closeDeltaPct === "number" && Number.isFinite(botRealtime.closeDeltaPct)
	                                ? ` (${fmtPct(botRealtime.closeDeltaPct, 2)})`
	                                : ""}
	                            </>
	                          ) : null}
	                        </>
	                      ) : (
	                        "—"
	                      )}
	                    </div>
	                  </div>
	                  <div className="kv">
	                    <div className="k">Candle close</div>
	                    <div className="v">
	                      {botRealtime?.expectedCloseMs ? (
	                        <>
	                          {fmtTimeMs(botRealtime.expectedCloseMs)} • {fmtEtaMs(botRealtime.closeEtaMs)} • pollΔ{" "}
	                          {fmtDurationMs(botRealtime.pollCloseLagMs)}
	                        </>
	                      ) : (
	                        "—"
	                      )}
	                    </div>
	                  </div>

                  <div className="kv" style={{ marginTop: 12 }}>
                    <div className="k">Equity / Position</div>
                    <div className="v">
                      {fmtRatio(bot.status.equityCurve[bot.status.equityCurve.length - 1] ?? 1, 4)}x /{" "}
		                      {(() => {
		                        const p = bot.status.positions[bot.status.positions.length - 1] ?? 0;
		                        if (p > 0) return `LONG${Math.abs(p) < 0.9999 ? ` (${fmtPct(Math.abs(p), 1)})` : ""}`;
		                        if (p < 0) return `SHORT${Math.abs(p) < 0.9999 ? ` (${fmtPct(Math.abs(p), 1)})` : ""}`;
		                        return "FLAT";
		                      })()}
		                    </div>
		                  </div>
	                  <div className="kv">
	                    <div className="k">Peak / Drawdown</div>
	                    <div className="v">
	                      {fmtRatio(bot.status.peakEquity, 4)}x / {botRisk ? fmtPct(botRisk.dd, 2) : "—"}
	                    </div>
	                  </div>
	                  <div className="kv">
	                    <div className="k">Day start / Daily loss</div>
	                    <div className="v">
	                      {fmtRatio(bot.status.dayStartEquity, 4)}x / {botRisk ? fmtPct(botRisk.dl, 2) : "—"}
	                    </div>
	                  </div>
	                  <div className="kv">
	                    <div className="k">Halt status</div>
	                    <div className="v">
	                      {bot.status.halted ? `HALTED${bot.status.haltReason ? ` (${bot.status.haltReason})` : ""}` : "Active"}
	                    </div>
	                  </div>
	                  {bot.status.haltedAtMs ? (
	                    <div className="kv">
	                      <div className="k">Halted at</div>
	                      <div className="v">{fmtTimeMs(bot.status.haltedAtMs)}</div>
	                    </div>
	                  ) : null}
                  <div className="kv">
                    <div className="k">Order errors</div>
                    <div className="v">{bot.status.consecutiveOrderErrors}</div>
                  </div>
                  {typeof bot.status.cooldownLeft === "number" && Number.isFinite(bot.status.cooldownLeft) && bot.status.cooldownLeft > 0 ? (
                    <div className="kv">
                      <div className="k">Cooldown</div>
                      <div className="v">{Math.max(0, Math.trunc(bot.status.cooldownLeft))} bar(s) remaining</div>
                    </div>
                  ) : null}
                  <div className="kv">
                    <div className="k">Latest signal</div>
                    <div className="v">{bot.status.latestSignal.action}</div>
                  </div>
                  <div className="kv">
                    <div className="k">Current price</div>
                    <div className="v">{fmtMoney(bot.status.latestSignal.currentPrice, 4)}</div>
                  </div>
                  <div className="kv">
                    <div className="k">Kalman</div>
                    <div className="v">
                      {(() => {
                        const cur = bot.status.latestSignal.currentPrice;
                        const next = bot.status.latestSignal.kalmanNext;
                        const ret = bot.status.latestSignal.kalmanReturn;
                        const z = bot.status.latestSignal.kalmanZ;
                        const ret2 =
                          typeof ret === "number" && Number.isFinite(ret)
                            ? ret
                            : typeof next === "number" && Number.isFinite(next) && cur !== 0
                              ? (next - cur) / cur
                              : null;
                        const nextTxt = typeof next === "number" && Number.isFinite(next) ? fmtMoney(next, 4) : "—";
                        const retTxt = typeof ret2 === "number" && Number.isFinite(ret2) ? fmtPct(ret2, 3) : "—";
                        const zTxt = typeof z === "number" && Number.isFinite(z) ? fmtNum(z, 3) : "—";
                        return `${nextTxt} (${retTxt}) • z ${zTxt} • ${bot.status.latestSignal.kalmanDirection ?? "—"}`;
                      })()}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">LSTM</div>
                    <div className="v">
                      {(() => {
                        const cur = bot.status.latestSignal.currentPrice;
                        const next = bot.status.latestSignal.lstmNext;
                        const ret =
                          typeof next === "number" && Number.isFinite(next) && cur !== 0 ? (next - cur) / cur : null;
                        const nextTxt = typeof next === "number" && Number.isFinite(next) ? fmtMoney(next, 4) : "—";
                        const retTxt = typeof ret === "number" && Number.isFinite(ret) ? fmtPct(ret, 3) : "—";
                        return `${nextTxt} (${retTxt}) • ${bot.status.latestSignal.lstmDirection ?? "—"}`;
                      })()}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Chosen</div>
                    <div className="v">{bot.status.latestSignal.chosenDirection ?? "—"}</div>
                  </div>
	                  {typeof bot.status.latestSignal.confidence === "number" && Number.isFinite(bot.status.latestSignal.confidence) ? (
	                    <div className="kv">
	                      <div className="k">Confidence / Size</div>
	                      <div className="v">
	                        {fmtPct(bot.status.latestSignal.confidence, 1)}
	                        {typeof bot.status.latestSignal.positionSize === "number" && Number.isFinite(bot.status.latestSignal.positionSize)
	                          ? ` • ${fmtPct(bot.status.latestSignal.positionSize, 1)}`
	                          : ""}
	                      </div>
	                    </div>
	                  ) : null}

	                  <details className="details" style={{ marginTop: 12 }}>
	                    <summary>Signal details</summary>
	                    <div style={{ marginTop: 10 }}>
	                      {(() => {
	                        const sig = bot.status.latestSignal;
	                        const r = sig.regimes;
	                        if (!r) return null;
	                        const trend = typeof r.trend === "number" && Number.isFinite(r.trend) ? fmtPct(r.trend, 1) : "—";
	                        const mr = typeof r.mr === "number" && Number.isFinite(r.mr) ? fmtPct(r.mr, 1) : "—";
	                        const hv = typeof r.highVol === "number" && Number.isFinite(r.highVol) ? fmtPct(r.highVol, 1) : "—";
	                        return (
	                          <div className="kv">
	                            <div className="k">Regimes</div>
	                            <div className="v">
	                              trend {trend} • mr {mr} • high vol {hv}
	                            </div>
	                          </div>
	                        );
	                      })()}

	                      {(() => {
	                        const q = bot.status.latestSignal.quantiles;
	                        if (!q) return null;
	                        const q10 = typeof q.q10 === "number" && Number.isFinite(q.q10) ? fmtPct(q.q10, 3) : "—";
	                        const q50 = typeof q.q50 === "number" && Number.isFinite(q.q50) ? fmtPct(q.q50, 3) : "—";
	                        const q90 = typeof q.q90 === "number" && Number.isFinite(q.q90) ? fmtPct(q.q90, 3) : "—";
	                        const w = typeof q.width === "number" && Number.isFinite(q.width) ? fmtPct(q.width, 3) : "—";
	                        return (
	                          <div className="kv">
	                            <div className="k">Quantiles</div>
	                            <div className="v">
	                              q10 {q10} • q50 {q50} • q90 {q90} • width {w}
	                            </div>
	                          </div>
	                        );
	                      })()}

	                      {(() => {
	                        const i = bot.status.latestSignal.conformalInterval;
	                        if (!i) return null;
	                        const lo = typeof i.lo === "number" && Number.isFinite(i.lo) ? fmtPct(i.lo, 3) : "—";
	                        const hi = typeof i.hi === "number" && Number.isFinite(i.hi) ? fmtPct(i.hi, 3) : "—";
	                        const w = typeof i.width === "number" && Number.isFinite(i.width) ? fmtPct(i.width, 3) : "—";
	                        return (
	                          <div className="kv">
	                            <div className="k">Conformal</div>
	                            <div className="v">
	                              lo {lo} • hi {hi} • width {w}
	                            </div>
	                          </div>
	                        );
	                      })()}

	                      {(() => {
	                        const std = bot.status.latestSignal.kalmanStd;
	                        if (typeof std !== "number" || !Number.isFinite(std)) return null;
	                        return (
	                          <div className="kv">
	                            <div className="k">Kalman σ</div>
	                            <div className="v">{fmtPct(std, 3)}</div>
	                          </div>
	                        );
	                      })()}
	                    </div>
	                  </details>

	                  {bot.status.lastOrder ? (
	                    <div className="kv">
	                      <div className="k">Last order</div>
	                      <div className="v">{bot.status.lastOrder.message}</div>
                    </div>
                  ) : null}

                  <details className="details" style={{ marginTop: 12 }}>
                    <summary>Realtime feed</summary>
                    <div className="actions" style={{ marginTop: 10 }}>
                      <button
                        className="btn"
                        type="button"
                        disabled={botRt.feed.length === 0}
                        onClick={() => {
                          void copyText(botRtFeedText);
                          showToast("Copied realtime feed");
                        }}
                      >
                        Copy
                      </button>
                      <button
                        className="btn"
                        type="button"
                        disabled={botRt.feed.length === 0}
                        onClick={() => {
                          setBotRt((s) => ({ ...s, feed: [] }));
                          showToast("Cleared realtime feed");
                        }}
                      >
                        Clear
                      </button>
                    </div>
                    <pre className="code" style={{ marginTop: 10 }}>
                      {botRtFeedText}
                    </pre>
                  </details>

		                  <div style={{ marginTop: 12 }}>
		                    <div className="btChartHeader" style={{ marginBottom: 10 }}>
		                      <div className="btChartTitle">Order log</div>
		                      <div className="btChartMeta">
		                        <span className="badge">
		                          showing {botOrdersView.shown.length}/{botOrdersView.total}
		                        </span>
		                        <span className="badge">latest last</span>
		                      </div>
		                      <div className="btChartActions">
		                        <button
		                          className="btn"
		                          type="button"
		                          disabled={botOrdersView.shown.length === 0}
		                          onClick={() => {
		                            void copyText(botOrderCopyText);
		                            showToast("Copied order log");
		                          }}
		                        >
		                          Copy
		                        </button>
		                        <button
		                          className="btn"
		                          type="button"
		                          disabled={botOrdersView.shown.length === 0}
		                          onClick={() => {
		                            void copyText(JSON.stringify(botOrdersView.shown, null, 2));
		                            showToast("Copied order log JSON");
		                          }}
		                        >
		                          Copy JSON
		                        </button>
		                        <button
		                          className="btn"
		                          type="button"
		                          disabled={!orderFilterText && !orderSentOnly && !orderErrorsOnly && orderSideFilter === "ALL"}
		                          onClick={() => {
		                            setOrderFilterText("");
		                            setOrderSentOnly(false);
		                            setOrderErrorsOnly(false);
		                            setOrderSideFilter("ALL");
		                            showToast("Cleared order log filters");
		                          }}
		                        >
		                          Clear
		                        </button>
		                      </div>
		                    </div>

		                    <div className="pillRow" style={{ marginBottom: 10 }}>
		                      <input
		                        className="input"
		                        style={{ flex: "1 1 240px" }}
		                        value={orderFilterText}
		                        onChange={(e) => setOrderFilterText(e.target.value)}
		                        placeholder="Filter (message / symbol / id)"
		                        spellCheck={false}
		                      />
		                      <select
		                        className="select"
		                        style={{ width: 140 }}
		                        value={orderSideFilter}
		                        onChange={(e) => setOrderSideFilter(e.target.value as OrderSideFilter)}
		                      >
		                        <option value="ALL">All sides</option>
		                        <option value="BUY">BUY</option>
		                        <option value="SELL">SELL</option>
		                      </select>
		                      <select
		                        className="select"
		                        style={{ width: 140 }}
		                        value={String(orderLimit)}
		                        onChange={(e) => setOrderLimit(numFromInput(e.target.value, orderLimit))}
		                        aria-label="Order log limit"
		                      >
		                        <option value="50">Last 50</option>
		                        <option value="200">Last 200</option>
		                        <option value="1000">Last 1000</option>
		                        <option value="2000">Last 2000</option>
		                      </select>
		                      <label className="pill" style={{ userSelect: "none" }}>
		                        <input type="checkbox" checked={orderSentOnly} onChange={(e) => setOrderSentOnly(e.target.checked)} />
		                        Sent only
		                      </label>
		                      <label className="pill" style={{ userSelect: "none" }}>
		                        <input type="checkbox" checked={orderErrorsOnly} onChange={(e) => setOrderErrorsOnly(e.target.checked)} />
		                        Errors only
		                      </label>
		                    </div>

		                    <div className="pillRow" style={{ marginBottom: 10 }}>
		                      <span className="hint" style={{ marginRight: 6 }}>
		                        Columns:
		                      </span>
		                      <label className="pill" style={{ userSelect: "none" }}>
		                        <input type="checkbox" checked={orderShowStatus} onChange={(e) => setOrderShowStatus(e.target.checked)} />
		                        Status
		                      </label>
		                      <label className="pill" style={{ userSelect: "none" }}>
		                        <input type="checkbox" checked={orderShowOrderId} onChange={(e) => setOrderShowOrderId(e.target.checked)} />
		                        Order ID
		                      </label>
		                      <label className="pill" style={{ userSelect: "none" }}>
		                        <input
		                          type="checkbox"
		                          checked={orderShowClientOrderId}
		                          onChange={(e) => setOrderShowClientOrderId(e.target.checked)}
		                        />
		                        Client order ID
		                      </label>
		                    </div>

		                    {botOrdersView.total === 0 ? (
		                      <div className="hint">No live operations yet.</div>
		                    ) : botOrdersView.shown.length === 0 ? (
		                      <div className="hint">No matches. Try clearing filters.</div>
		                    ) : (
		                      <div className="tableWrap" role="region" aria-label="Live bot order log">
		                        <table className="table">
		                          <thead>
		                            <tr>
		                              <th>At</th>
		                              <th>Bar</th>
		                              <th>Side</th>
		                              <th>Price</th>
		                              <th>Sent</th>
		                              <th>Mode</th>
		                              {orderShowStatus ? <th>Status</th> : null}
		                              {orderShowOrderId ? <th>Order ID</th> : null}
		                              {orderShowClientOrderId ? <th>Client order ID</th> : null}
		                              <th>Message</th>
		                            </tr>
		                          </thead>
		                          <tbody>
		                            {botOrdersView.shown.map((e) => {
		                              const bar = botOrdersView.startIndex + e.index;
		                              const mode = e.order.mode ?? "—";
		                              return (
		                                <tr key={`${e.atMs}-${e.index}-${e.opSide}`}>
		                                  <td className="tdMono">{fmtTimeMs(e.atMs)}</td>
		                                  <td className="tdMono">{bar}</td>
		                                  <td>
		                                    <span className={e.opSide === "BUY" ? "badge badgeStrong badgeLong" : "badge badgeStrong badgeFlat"}>
		                                      {e.opSide}
		                                    </span>
		                                  </td>
		                                  <td className="tdMono">{fmtMoney(e.price, 4)}</td>
		                                  <td className="tdMono">{e.order.sent ? "SENT" : "NO"}</td>
		                                  <td className="tdMono">{mode}</td>
		                                  {orderShowStatus ? <td className="tdMono">{e.order.status ?? "—"}</td> : null}
		                                  {orderShowOrderId ? <td className="tdMono">{e.order.orderId ?? "—"}</td> : null}
		                                  {orderShowClientOrderId ? <td className="tdMono">{e.order.clientOrderId ?? "—"}</td> : null}
		                                  <td style={{ whiteSpace: "pre-wrap" }}>{e.order.message}</td>
		                                </tr>
		                              );
		                            })}
		                          </tbody>
		                        </table>
		                      </div>
		                    )}
		                  </div>
		                </>
              ) : (
                <div className="hint">
                  {botStarting
                    ? "Bot is starting… (initializing model). Use “Refresh” to check status."
                    : "Bot is stopped. Use “Start live bot” on the left."}
                </div>
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
                    <div className="k">Open threshold</div>
                    <div className="v">
                      {(() => {
                        const openThr = state.latestSignal.openThreshold ?? state.latestSignal.threshold;
                        return `${fmtNum(openThr, 6)} (${fmtPct(openThr, 3)})`;
                      })()}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Close threshold</div>
                    <div className="v">
                      {(() => {
                        const openThr = state.latestSignal.openThreshold ?? state.latestSignal.threshold;
                        const closeThr = state.latestSignal.closeThreshold ?? openThr;
                        return `${fmtNum(closeThr, 6)} (${fmtPct(closeThr, 3)})`;
                      })()}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Kalman</div>
                    <div className="v">
                      {(() => {
                        const sig = state.latestSignal;
                        const cur = sig.currentPrice;
                        const next = sig.kalmanNext;
                        if (typeof next !== "number" || !Number.isFinite(next)) return "disabled";
                        const ret = sig.kalmanReturn;
                        const z = sig.kalmanZ;
                        const ret2 =
                          typeof ret === "number" && Number.isFinite(ret)
                            ? ret
                            : cur !== 0
                              ? (next - cur) / cur
                              : null;
                        const nextTxt = fmtMoney(next, 4);
                        const retTxt = typeof ret2 === "number" && Number.isFinite(ret2) ? fmtPct(ret2, 3) : "—";
                        const zTxt = typeof z === "number" && Number.isFinite(z) ? fmtNum(z, 3) : "—";
                        return `${nextTxt} (${retTxt}) • z ${zTxt} • ${sig.kalmanDirection ?? "—"}`;
                      })()}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">LSTM</div>
                    <div className="v">
                      {(() => {
                        const sig = state.latestSignal;
                        const cur = sig.currentPrice;
                        const next = sig.lstmNext;
                        if (typeof next !== "number" || !Number.isFinite(next)) return "disabled";
                        const ret = cur !== 0 ? (next - cur) / cur : null;
                        const nextTxt = fmtMoney(next, 4);
                        const retTxt = typeof ret === "number" && Number.isFinite(ret) ? fmtPct(ret, 3) : "—";
                        return `${nextTxt} (${retTxt}) • ${sig.lstmDirection ?? "—"}`;
                      })()}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Chosen</div>
                    <div className="v">{state.latestSignal.chosenDirection ?? "NEUTRAL"}</div>
                  </div>
                  {typeof state.latestSignal.confidence === "number" && Number.isFinite(state.latestSignal.confidence) ? (
                    <div className="kv">
                      <div className="k">Confidence / Size</div>
                      <div className="v">
                        {fmtPct(state.latestSignal.confidence, 1)}
                        {typeof state.latestSignal.positionSize === "number" && Number.isFinite(state.latestSignal.positionSize)
                          ? ` • ${fmtPct(state.latestSignal.positionSize, 1)}`
                          : ""}
                      </div>
                    </div>
                  ) : null}

                  {(() => {
                    const sig = state.latestSignal;
                    const std = sig.kalmanStd;
                    const hasStd = typeof std === "number" && Number.isFinite(std);
                    const hasDetails = Boolean(sig.regimes || sig.quantiles || sig.conformalInterval || hasStd);
                    if (!hasDetails) return null;
                    return (
                      <details className="details" style={{ marginTop: 12 }}>
                        <summary>Signal details</summary>
                        <div style={{ marginTop: 10 }}>
                          {(() => {
                            const r = sig.regimes;
                            if (!r) return null;
                            const trend = typeof r.trend === "number" && Number.isFinite(r.trend) ? fmtPct(r.trend, 1) : "—";
                            const mr = typeof r.mr === "number" && Number.isFinite(r.mr) ? fmtPct(r.mr, 1) : "—";
                            const hv = typeof r.highVol === "number" && Number.isFinite(r.highVol) ? fmtPct(r.highVol, 1) : "—";
                            return (
                              <div className="kv">
                                <div className="k">Regimes</div>
                                <div className="v">
                                  trend {trend} • mr {mr} • high vol {hv}
                                </div>
                              </div>
                            );
                          })()}

                          {(() => {
                            const q = sig.quantiles;
                            if (!q) return null;
                            const q10 = typeof q.q10 === "number" && Number.isFinite(q.q10) ? fmtPct(q.q10, 3) : "—";
                            const q50 = typeof q.q50 === "number" && Number.isFinite(q.q50) ? fmtPct(q.q50, 3) : "—";
                            const q90 = typeof q.q90 === "number" && Number.isFinite(q.q90) ? fmtPct(q.q90, 3) : "—";
                            const w = typeof q.width === "number" && Number.isFinite(q.width) ? fmtPct(q.width, 3) : "—";
                            return (
                              <div className="kv">
                                <div className="k">Quantiles</div>
                                <div className="v">
                                  q10 {q10} • q50 {q50} • q90 {q90} • width {w}
                                </div>
                              </div>
                            );
                          })()}

                          {(() => {
                            const i = sig.conformalInterval;
                            if (!i) return null;
                            const lo = typeof i.lo === "number" && Number.isFinite(i.lo) ? fmtPct(i.lo, 3) : "—";
                            const hi = typeof i.hi === "number" && Number.isFinite(i.hi) ? fmtPct(i.hi, 3) : "—";
                            const w = typeof i.width === "number" && Number.isFinite(i.width) ? fmtPct(i.width, 3) : "—";
                            return (
                              <div className="kv">
                                <div className="k">Conformal</div>
                                <div className="v">
                                  lo {lo} • hi {hi} • width {w}
                                </div>
                              </div>
                            );
                          })()}

                          {hasStd ? (
                            <div className="kv">
                              <div className="k">Kalman σ</div>
                              <div className="v">{fmtPct(std, 3)}</div>
                            </div>
                          ) : null}
                        </div>
                      </details>
                    );
                  })()}
                </>
              ) : (
                <div className="hint">No signal yet.</div>
              )}
            </div>
          </div>

          <div className="card" ref={backtestRef}>
	            <div className="cardHeader">
	              <h2 className="cardTitle">Backtest summary</h2>
	              <p className="cardSubtitle">Uses a time split (train vs held-out backtest). When optimizing, tunes on a fit/tune split inside train.</p>
	            </div>
            <div className="cardBody">
              {state.backtest ? (
                <>
		                  <BacktestChart
			                    prices={state.backtest.prices}
			                    equityCurve={state.backtest.equityCurve}
			                    kalmanPredNext={state.backtest.kalmanPredNext}
			                    positions={state.backtest.positions}
			                    agreementOk={state.backtest.agreementOk}
			                    trades={state.backtest.trades}
			                    backtestStartIndex={state.backtest.split.backtestStartIndex}
			                    height={360}
			                  />
		                  <div className="pillRow" style={{ marginBottom: 10, marginTop: 12 }}>
		                    {state.backtest.split.tune > 0 ? (
		                      <>
		                        <span className="badge">Fit: {state.backtest.split.fit}</span>
		                        <span className="badge">
		                          Tune: {state.backtest.split.tune} ({fmtPct(state.backtest.split.tuneRatio, 1)})
		                        </span>
		                      </>
		                    ) : (
		                      <span className="badge">Train: {state.backtest.split.train}</span>
		                    )}
		                    <span className="badge">Backtest: {state.backtest.split.backtest}</span>
		                    <span className="badge">Holdout: {fmtPct(state.backtest.split.backtestRatio, 1)}</span>
		                    <span className="badge">{methodLabel(state.backtest.method)}</span>
		                  </div>

                  <div className="kv">
                    <div className="k">Best open threshold</div>
                    <div className="v">
                      {(() => {
                        const openThr = state.backtest.openThreshold ?? state.backtest.threshold;
                        return `${fmtNum(openThr, 6)} (${fmtPct(openThr, 3)})`;
                      })()}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Best close threshold</div>
                    <div className="v">
                      {(() => {
                        const openThr = state.backtest.openThreshold ?? state.backtest.threshold;
                        const closeThr = state.backtest.closeThreshold ?? openThr;
                        return `${fmtNum(closeThr, 6)} (${fmtPct(closeThr, 3)})`;
                      })()}
                    </div>
                  </div>
                  {(() => {
                    const minHold = state.backtest.minHoldBars ?? 0;
                    const cooldown = state.backtest.cooldownBars ?? 0;
                    const minHoldN = typeof minHold === "number" && Number.isFinite(minHold) ? Math.max(0, Math.trunc(minHold)) : 0;
                    const cooldownN = typeof cooldown === "number" && Number.isFinite(cooldown) ? Math.max(0, Math.trunc(cooldown)) : 0;
                    if (minHoldN <= 0 && cooldownN <= 0) return null;
                    return (
                      <div className="kv">
                        <div className="k">Min hold / Cooldown</div>
                        <div className="v">
                          {minHoldN} / {cooldownN} bars
                        </div>
                      </div>
                    );
                  })()}
                  {state.backtest.costs ? (
                    <>
                      <div className="kv">
                        <div className="k">Per-side cost (est.)</div>
                        <div className="v">
                          {fmtNum(state.backtest.costs.perSideCost, 6)} ({fmtPct(state.backtest.costs.perSideCost, 3)})
                        </div>
                      </div>
                      <div className="kv">
                        <div className="k">Round-trip cost (approx)</div>
                        <div className="v">
                          {fmtNum(state.backtest.costs.roundTripCost, 6)} ({fmtPct(state.backtest.costs.roundTripCost, 3)})
                        </div>
                      </div>
                      <div className="kv">
                        <div className="k">Break-even (round trip)</div>
                        <div className="v">
                          {(() => {
                            const openThr = state.backtest.openThreshold ?? state.backtest.threshold;
                            const be = state.backtest.costs?.breakEvenThreshold ?? 0;
                            const note = openThr < be && be > 0 ? " (open threshold below break-even)" : "";
                            return `${fmtNum(be, 6)} (${fmtPct(be, 3)})${note}`;
                          })()}
                        </div>
                      </div>
                    </>
                  ) : null}
                  {state.backtest.tuning && state.backtest.split.tune > 0 ? (
                    <div className="kv">
                      <div className="k">Tune objective</div>
                      <div className="v">
                        {state.backtest.tuning.objective}
                        {state.backtest.tuning.minRoundTrips && state.backtest.tuning.minRoundTrips > 0
                          ? ` • min-round-trips=${state.backtest.tuning.minRoundTrips}`
                          : ""}
                        {state.backtest.tuning.tuneStats
                          ? ` • folds=${state.backtest.tuning.tuneStats.folds} score=${fmtNum(state.backtest.tuning.tuneStats.meanScore, 4)}±${fmtNum(state.backtest.tuning.tuneStats.stdScore, 4)}`
                          : ""}
                      </div>
                    </div>
                  ) : null}
                  {state.backtest.tuning?.tuneMetrics ? (
                    <div className="kv">
                      <div className="k">Tune vs backtest</div>
                      <div className="v">
                        {(() => {
                          const tune = state.backtest?.tuning?.tuneMetrics;
                          const bt = state.backtest?.metrics;
                          if (!tune || !bt) return "—";
                          const tuneEq = tune.finalEquity;
                          const btEq = bt.finalEquity;
                          const gap = tuneEq > 0 ? btEq / tuneEq - 1 : null;
                          const gapTxt = gap != null && Number.isFinite(gap) ? ` • gap ${fmtPct(gap, 2)}` : "";
                          return `tune ${fmtRatio(tuneEq, 4)} (${fmtPct(tune.totalReturn, 2)}) • holdout ${fmtRatio(btEq, 4)} (${fmtPct(bt.totalReturn, 2)})${gapTxt}`;
                        })()}
                      </div>
                    </div>
                  ) : null}
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
                    <div className="k">Trades / Round trips / Win rate</div>
                    <div className="v">
                      {state.backtest.metrics.tradeCount} / {state.backtest.metrics.roundTrips} / {fmtPct(state.backtest.metrics.winRate, 1)}
                      {state.backtest.metrics.roundTrips < 3 ? " • low sample" : ""}
                    </div>
                  </div>

                  {state.backtest.baselines && state.backtest.baselines.length > 0 ? (
                    <details className="details" style={{ marginTop: 12 }}>
                      <summary>Baselines</summary>
                      <div style={{ marginTop: 10 }}>
                        {state.backtest.baselines.map((b) => {
                          const baseEq = b.metrics.finalEquity;
                          const modelEq = state.backtest?.metrics.finalEquity ?? 1;
                          const delta = baseEq > 0 ? modelEq / baseEq - 1 : null;
                          return (
                            <div key={b.name} className="kv">
                              <div className="k">{b.name}</div>
                              <div className="v">
                                {fmtRatio(baseEq, 4)} ({fmtPct(b.metrics.totalReturn, 2)})
                                {delta != null && Number.isFinite(delta) ? ` • model: ${fmtPct(delta, 2)}` : ""}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </details>
                  ) : null}

                  <details className="details" style={{ marginTop: 12 }}>
                    <summary>More metrics</summary>
                    <div style={{ marginTop: 10 }}>
                      <div className="kv">
                        <div className="k">Operations (position changes)</div>
                        <div className="v">{state.backtest.metrics.positionChanges}</div>
                      </div>
	                      <div className="kv">
	                        <div className="k">Annualized volatility</div>
	                        <div className="v">{fmtPct(state.backtest.metrics.annualizedVolatility, 2)}</div>
	                      </div>
	                      <div className="kv">
	                        <div className="k">Gross profit / Gross loss</div>
	                        <div className="v">
	                          {fmtPct(state.backtest.metrics.grossProfit, 2)} / {fmtPct(state.backtest.metrics.grossLoss, 2)}
	                        </div>
	                      </div>
	                      <div className="kv">
	                        <div className="k">Profit factor</div>
	                        <div className="v">
	                          {fmtProfitFactor(state.backtest.metrics.profitFactor, state.backtest.metrics.grossProfit, state.backtest.metrics.grossLoss)}
	                        </div>
	                      </div>
	                      <div className="kv">
	                        <div className="k">Avg trade return / Avg hold</div>
	                        <div className="v">
	                          {fmtPct(state.backtest.metrics.avgTradeReturn, 2)} / {fmtNum(state.backtest.metrics.avgHoldingPeriods, 2)} bars
	                        </div>
	                      </div>
	                      <div className="kv">
	                        <div className="k">Agreement rate</div>
	                        <div className="v">
	                          {fmtPct(state.backtest.metrics.agreementRate, 1)}
	                        </div>
	                      </div>
                    </div>
                  </details>

                  {state.backtest.walkForward ? (
                    <details className="details" style={{ marginTop: 12 }}>
                      <summary>Walk-forward</summary>
                      <div style={{ marginTop: 10 }}>
                        <div className="kv">
                          <div className="k">Folds</div>
                          <div className="v">{state.backtest.walkForward.foldCount}</div>
                        </div>
                        <div className="kv">
                          <div className="k">Final equity (mean ± std)</div>
                          <div className="v">
                            {fmtRatio(state.backtest.walkForward.summary.finalEquityMean, 4)} ±{" "}
                            {fmtRatio(state.backtest.walkForward.summary.finalEquityStd, 4)}
                          </div>
                        </div>
                        <div className="kv">
                          <div className="k">Sharpe (mean ± std)</div>
                          <div className="v">
                            {fmtNum(state.backtest.walkForward.summary.sharpeMean, 3)} ±{" "}
                            {fmtNum(state.backtest.walkForward.summary.sharpeStd, 3)}
                          </div>
                        </div>
                        <div className="kv">
                          <div className="k">Max DD (mean ± std)</div>
                          <div className="v">
                            {fmtPct(state.backtest.walkForward.summary.maxDrawdownMean, 2)} ±{" "}
                            {fmtPct(state.backtest.walkForward.summary.maxDrawdownStd, 2)}
                          </div>
                        </div>
                        <div className="kv">
                          <div className="k">Turnover (mean ± std)</div>
                          <div className="v">
                            {fmtNum(state.backtest.walkForward.summary.turnoverMean, 4)} ±{" "}
                            {fmtNum(state.backtest.walkForward.summary.turnoverStd, 4)}
                          </div>
                        </div>
                      </div>
                    </details>
                  ) : null}
                </>
              ) : (
                <div className="hint">No backtest yet.</div>
              )}
            </div>
          </div>

          <div className="card" ref={tradeRef}>
            <div className="cardHeader">
              <h2 className="cardTitle">Trade result</h2>
              <p className="cardSubtitle">Shows current key status, and trade output after calling /trade.</p>
            </div>
            <div className="cardBody">
              <div className="pillRow" style={{ marginBottom: 10 }}>
                <span className="badge">
                  Keys:{" "}
                  {keys.status
                    ? keys.status.hasApiKey && keys.status.hasApiSecret
                      ? "provided"
                      : "missing"
                    : "unknown"}
                </span>
                <span className="badge">
                  {marketLabel(form.market)}
                  {form.binanceTestnet ? " testnet" : ""}
                </span>
                {keys.status?.signed ? <span className="badge">Signed: {keys.status.signed.ok ? "OK" : "FAIL"}</span> : null}
                {keys.status?.tradeTest ? (
                  <span className="badge">
                    Trade: {form.market === "margin" ? "N/A" : keys.status.tradeTest.ok ? "OK" : "FAIL"}
                  </span>
                ) : null}
              </div>

              <div className="actions" style={{ marginTop: 0, marginBottom: 10 }}>
                <button
                  className="btn"
                  type="button"
                  onClick={() => refreshKeys()}
                  disabled={keys.loading || apiOk === "down" || apiOk === "auth"}
                >
                  {keys.loading ? "Checking…" : "Check keys"}
                </button>
                <span className="hint">
                  {keys.checkedAtMs ? `Last checked: ${fmtTimeMs(keys.checkedAtMs)}` : "Uses Binance signed endpoints + /order/test (no real order)."}
                </span>
              </div>

              {keys.error ? (
                <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)", marginBottom: 10 }}>
                  {keys.error}
                </pre>
              ) : null}

              {keys.status ? (
                <>
                  <div className="kv">
                    <div className="k">BINANCE_API_KEY / BINANCE_API_SECRET</div>
                    <div className="v">
                      {keys.status.hasApiKey ? "present" : "missing"} / {keys.status.hasApiSecret ? "present" : "missing"}
                    </div>
                  </div>

                  <div className="kv">
                    <div className="k">Signed check</div>
                    <div className="v">
                      {keys.status.signed ? (
                        <>
                          {keys.status.signed.ok ? "OK" : "FAIL"}{" "}
                          {keys.status.signed.code !== undefined ? `(${keys.status.signed.code}) ` : ""}
                          {keys.status.signed.summary}
                        </>
                      ) : (
                        "—"
                      )}
                    </div>
                  </div>

                  <div className="kv">
                    <div className="k">Trade permission</div>
                    <div className="v">
                      {keys.status.tradeTest ? (
                        <>
                          {keys.status.tradeTest.ok ? "OK" : "FAIL"}{" "}
                          {keys.status.tradeTest.code !== undefined ? `(${keys.status.tradeTest.code}) ` : ""}
                          {keys.status.tradeTest.summary}
                        </>
                      ) : (
                        "—"
                      )}
                    </div>
                  </div>
                </>
              ) : (
                <div className="hint" style={{ marginBottom: 10 }}>
                  Key status not loaded yet.
                </div>
              )}

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
              <h2 className="cardTitle">User data stream (listenKey)</h2>
              <p className="cardSubtitle">Keeps the Binance user-data listen key alive via the API, and connects the browser to Binance WebSocket.</p>
            </div>
            <div className="cardBody">
              <div className="pillRow" style={{ marginBottom: 10 }}>
                <span className="badge">
                  {marketLabel(form.market)}
                  {form.binanceTestnet ? " testnet" : ""}
                </span>
                <span className="badge">WS: {listenKeyUi.wsStatus}</span>
                {listenKeyUi.keepAliveAtMs ? <span className="badge">Keep-alive: {fmtTimeMs(listenKeyUi.keepAliveAtMs)}</span> : null}
              </div>

              <div className="actions" style={{ marginTop: 0, marginBottom: 10 }}>
                <button className="btn" type="button" onClick={() => void startListenKeyStream()} disabled={listenKeyUi.loading || apiOk !== "ok"}>
                  {listenKeyUi.loading ? "Starting…" : listenKeyUi.info ? "Restart stream" : "Start stream"}
                </button>
                <button
                  className="btn"
                  type="button"
                  onClick={() => (listenKeyUi.info ? void keepAliveListenKeyStream(listenKeyUi.info) : undefined)}
                  disabled={!listenKeyUi.info || listenKeyUi.loading || apiOk !== "ok"}
                >
                  Keep alive now
                </button>
                <button
                  className="btn"
                  type="button"
                  onClick={() => void stopListenKeyStream({ close: true })}
                  disabled={!listenKeyUi.info || listenKeyUi.loading || apiOk !== "ok"}
                >
                  Stop
                </button>
                <span className="hint">Binance requires a keep-alive (PUT) at least every ~30 minutes; the UI schedules one every ~{Math.round((listenKeyUi.info?.keepAliveMs ?? 25 * 60_000) * 0.9 / 60_000)} minutes.</span>
              </div>

              {listenKeyUi.error ? (
                <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)", marginBottom: 10 }}>
                  {listenKeyUi.error}
                </pre>
              ) : null}
              {listenKeyUi.keepAliveError ? (
                <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)", marginBottom: 10 }}>
                  {listenKeyUi.keepAliveError}
                </pre>
              ) : null}
              {listenKeyUi.wsError ? (
                <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)", marginBottom: 10 }}>
                  {listenKeyUi.wsError}
                </pre>
              ) : null}

              {listenKeyUi.info ? (
                <>
                  <div className="kv">
                    <div className="k">Listen key</div>
                    <div className="v">
                      <span className="tdMono">
                        {listenKeyUi.info.listenKey.length > 14
                          ? `${listenKeyUi.info.listenKey.slice(0, 6)}…${listenKeyUi.info.listenKey.slice(-6)}`
                          : listenKeyUi.info.listenKey}
                      </span>{" "}
                      <button
                        className="btnSmall"
                        type="button"
                        onClick={async () => {
                          await copyText(listenKeyUi.info?.listenKey ?? "");
                          showToast("Copied listen key");
                        }}
                      >
                        Copy
                      </button>
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">WebSocket URL</div>
                    <div className="v">
                      <span className="tdMono">{listenKeyUi.info.wsUrl}</span>{" "}
                      <button
                        className="btnSmall"
                        type="button"
                        onClick={async () => {
                          await copyText(listenKeyUi.info?.wsUrl ?? "");
                          showToast("Copied WebSocket URL");
                        }}
                      >
                        Copy
                      </button>
                    </div>
                  </div>

                  {listenKeyUi.lastEvent ? (
                    <>
                      <div className="kv">
                        <div className="k">Last event</div>
                        <div className="v">{listenKeyUi.lastEventAtMs ? fmtTimeMs(listenKeyUi.lastEventAtMs) : "—"}</div>
                      </div>
                      <pre className="code">{listenKeyUi.lastEvent}</pre>
                    </>
                  ) : (
                    <div className="hint">No user-data events yet.</div>
                  )}
                </>
              ) : (
                <div className="hint">Not running.</div>
              )}
            </div>
          </div>

          <div className="card">
            <div className="cardHeader">
              <h2 className="cardTitle">Request preview</h2>
              <p className="cardSubtitle">This JSON is what the UI sends to the API (excluding session-stored secrets).</p>
            </div>
            <div className="cardBody">
              <div className="actions" style={{ marginTop: 0, marginBottom: 10 }}>
                <button
                  className="btn"
                  disabled={state.loading}
                  onClick={async () => {
                    await copyText(JSON.stringify(requestPreview, null, 2));
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
              <pre className="code">{JSON.stringify(requestPreview, null, 2)}</pre>
            </div>
          </div>
        </section>
      </div>

      <section className="card" style={{ marginTop: "18px" }}>
        <div className="cardHeader">
          <h2 className="cardTitle">Data Log</h2>
          <p className="cardSubtitle">All incoming API responses (last 100 entries)</p>
        </div>
        <div className="cardBody">
	          <div className="actions" style={{ marginTop: 0, marginBottom: 10 }}>
	            <button
	              className="btn"
	              onClick={() => setDataLog([])}
	            >
	              Clear Log
	            </button>
	            <button
	              className="btn"
	              onClick={() => {
	                const logText = dataLog
	                  .map((entry) => `[${new Date(entry.timestamp).toISOString()}] ${entry.label}:\n${JSON.stringify(entry.data, null, 2)}`)
	                  .join("\n\n");
	                copyText(logText);
	                showToast("Copied log to clipboard");
	              }}
	            >
	              Copy All
	            </button>
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogExpanded} onChange={(e) => setDataLogExpanded(e.target.checked)} />
                Expand
              </label>
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogIndexArrays} onChange={(e) => setDataLogIndexArrays(e.target.checked)} />
                Index arrays
              </label>
	          </div>
          <div
            ref={dataLogRef}
            style={{
              height: "500px",
              overflowY: "auto",
              backgroundColor: "#0a0e27",
              border: "1px solid #374151",
              borderRadius: "6px",
              padding: "12px",
              fontFamily: "var(--mono)",
              fontSize: "12px",
              color: "#e5e7eb",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {dataLog.length === 0 ? (
              <div style={{ color: "#6b7280" }}>No data logged yet. Run a signal, backtest, or trade to see incoming data.</div>
            ) : (
              dataLog.map((entry, idx) => (
                <div key={idx} style={{ marginBottom: "12px", paddingBottom: "12px", borderBottom: "1px solid #1f2937" }}>
                  <div style={{ color: "#60a5fa", marginBottom: "4px" }}>
                    [{new Date(entry.timestamp).toLocaleTimeString()}] <span style={{ color: "#34d399" }}>{entry.label}</span>
	                  </div>
	                  <div style={{ color: "#d1d5db", fontSize: "11px" }}>
                      {(() => {
                        const data = dataLogIndexArrays ? indexTopLevelPrimitiveArrays(entry.data) : entry.data;
                        const json = JSON.stringify(data, null, 2);
                        if (dataLogExpanded) return json;
                        const lines = json.split("\n");
                        const head = lines.slice(0, DATA_LOG_COLLAPSED_MAX_LINES).join("\n");
                        return lines.length > DATA_LOG_COLLAPSED_MAX_LINES ? `${head}\n... (truncated)` : head;
                      })()}
	                  </div>
	                </div>
	              ))
	            )}
          </div>
        </div>
      </section>
    </div>
  );
}
