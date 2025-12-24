import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  ApiBinanceTradesRequest,
  ApiBinanceTradesResponse,
  ApiParams,
  ApiTradeResponse,
  BacktestResponse,
  BinanceKeysStatus,
  BinanceListenKeyResponse,
  BotOrderEvent,
  BotStatus,
  BotStatusMulti,
  BotStatusSingle,
  CoinbaseKeysStatus,
  IntrabarFill,
  LatestSignal,
  Market,
  Method,
  Normalization,
  Platform,
  Positioning,
} from "./lib/types";
import {
  HttpError,
  backtest,
  binanceTrades,
  binanceKeysStatus,
  binanceListenKey,
  binanceListenKeyClose,
  binanceListenKeyKeepAlive,
  botStart,
  botStatus,
  botStop,
  cacheClear,
  cacheStats,
  coinbaseKeysStatus,
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
  BOT_START_TIMEOUT_MS,
  BOT_STATUS_TAIL_POINTS,
  BOT_STATUS_TIMEOUT_MS,
  BOT_AUTOSTART_RETRY_MS,
  BOT_TELEMETRY_POINTS,
  DATA_LOG_COLLAPSED_MAX_LINES,
  PLATFORM_DEFAULT_SYMBOL,
  PLATFORM_INTERVALS,
  PLATFORM_INTERVAL_SET,
  PLATFORM_LABELS,
  PLATFORM_SYMBOLS,
  PLATFORM_SYMBOL_SET,
  PLATFORMS,
  RATE_LIMIT_BASE_MS,
  RATE_LIMIT_MAX_MS,
  RATE_LIMIT_TOAST_MIN_MS,
  SESSION_BINANCE_KEY_KEY,
  SESSION_BINANCE_SECRET_KEY,
  SESSION_COINBASE_KEY_KEY,
  SESSION_COINBASE_SECRET_KEY,
  SESSION_COINBASE_PASSPHRASE_KEY,
  SIGNAL_TIMEOUT_MS,
  STORAGE_KEY,
  STORAGE_ORDER_LOG_PREFS_KEY,
  STORAGE_PERSIST_SECRETS_KEY,
  STORAGE_PROFILES_KEY,
  TRADE_TIMEOUT_MS,
  TUNE_OBJECTIVES,
} from "./app/constants";
import {
  defaultForm,
  normalizeFormState,
  parseDurationSeconds,
  platformIntervalSeconds,
  type FormState,
  type FormStateJson,
} from "./app/formState";
import {
  actionBadgeClass,
  buildRequestIssueDetails,
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
import { TopCombosChart, type OptimizationCombo, type OptimizationComboOperation } from "./components/TopCombosChart";

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

type KeysStatus = BinanceKeysStatus | CoinbaseKeysStatus;

function isCoinbaseKeysStatus(status: KeysStatus): status is CoinbaseKeysStatus {
  return "hasApiPassphrase" in status;
}

function isBinanceKeysStatus(status: KeysStatus): status is BinanceKeysStatus {
  return "market" in status;
}

function isBotStatusMulti(status: BotStatus): status is BotStatusMulti {
  return "multi" in status && status.multi === true;
}

function botStatusSymbol(status: BotStatusSingle): string | null {
  if (status.running) return status.symbol;
  if (status.symbol) return status.symbol;
  if (status.snapshot?.symbol) return status.snapshot.symbol;
  return null;
}

function parseSymbolsInput(raw: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const part of raw.split(/[,\s]+/)) {
    const sym = part.trim().toUpperCase();
    if (!sym) continue;
    if (seen.has(sym)) continue;
    seen.add(sym);
    out.push(sym);
  }
  return out;
}

function parseMaybeInt(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const n = Number(trimmed);
  if (!Number.isFinite(n)) return null;
  return Math.max(0, Math.trunc(n));
}

function parseTimeInputMs(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (/^\d+$/.test(trimmed)) {
    const n = Number(trimmed);
    return Number.isFinite(n) ? n : null;
  }
  const parsed = Date.parse(trimmed);
  return Number.isNaN(parsed) ? null : parsed;
}

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
  status: KeysStatus | null;
  platform: Platform | null;
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

type BinanceTradesUiState = {
  loading: boolean;
  error: string | null;
  response: ApiBinanceTradesResponse | null;
};

type TopCombosSource = "api" | "static";

type TopCombosMeta = {
  source: TopCombosSource;
  generatedAtMs: number | null;
  payloadSource: string | null;
  payloadSources: string[] | null;
  fallbackReason: string | null;
  comboCount: number | null;
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

type ComputeLimits = NonNullable<Awaited<ReturnType<typeof health>>["computeLimits"]>;
type ManualOverrideKey = "method" | "openThreshold" | "closeThreshold";

const CUSTOM_SYMBOL_VALUE = "__custom__";
const TOP_COMBOS_POLL_MS = 30_000;
const TOP_COMBOS_DISPLAY_MAX = 12;
const MIN_LOOKBACK_BARS = 2;

const DURATION_UNITS: Array<{ unit: string; seconds: number }> = [
  { unit: "M", seconds: 30 * 24 * 60 * 60 },
  { unit: "w", seconds: 7 * 24 * 60 * 60 },
  { unit: "d", seconds: 24 * 60 * 60 },
  { unit: "h", seconds: 60 * 60 },
  { unit: "m", seconds: 60 },
  { unit: "s", seconds: 1 },
];

function formatDurationSeconds(totalSeconds: number): string {
  const sec = Math.max(1, Math.round(totalSeconds));
  for (const { unit, seconds } of DURATION_UNITS) {
    if (sec % seconds === 0) return `${sec / seconds}${unit}`;
  }
  return `${sec}s`;
}

function sigNumber(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "";
  const rounded = Math.round(value * 1e8) / 1e8;
  return String(rounded);
}

function coerceNumber(value: number | null | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clampOptionalRatio(value: number | null | undefined): number {
  const v = coerceNumber(value, 0);
  return v > 0 ? clamp(v, 0, 0.999999) : 0;
}

function clampOptionalRange(value: number | null | undefined, min: number, max: number): number {
  const v = coerceNumber(value, 0);
  return v > 0 ? clamp(v, min, max) : 0;
}

function clampOptionalInt(value: number | null | undefined, min: number, max: number): number {
  const v = coerceNumber(value, 0);
  return v > 0 ? clamp(Math.trunc(v), min, max) : 0;
}

function sigText(value: string | null | undefined): string {
  return typeof value === "string" ? value : "";
}

function sigBool(value: boolean | null | undefined): string {
  return value ? "1" : "0";
}

function formatDirectionLabel(value: LatestSignal["closeDirection"]): string {
  if (value === undefined) return "—";
  return value ?? "NEUTRAL";
}

function clampComboForLimits(combo: OptimizationCombo, apiLimits: ComputeLimits | null): {
  bars: number;
  epochs: number;
  hiddenSize: number;
} {
  const lstmEnabled = combo.params.method !== "10";
  let bars = Math.trunc(combo.params.bars);
  if (!Number.isFinite(bars) || bars < 0) bars = 0;
  if (bars > 0) bars = clamp(bars, MIN_LOOKBACK_BARS, 1000);

  let epochs = clamp(Math.trunc(combo.params.epochs), 0, 5000);
  let hiddenSize = clamp(Math.trunc(combo.params.hiddenSize), 1, 512);

  if (lstmEnabled && apiLimits) {
    const maxBarsRaw = Math.trunc(apiLimits.maxBarsLstm);
    if (Number.isFinite(maxBarsRaw) && maxBarsRaw > 0) {
      const maxBars = clamp(maxBarsRaw, MIN_LOOKBACK_BARS, 1000);
      if (bars > 0) bars = Math.min(bars, maxBars);
    }
    epochs = Math.min(epochs, apiLimits.maxEpochs);
    hiddenSize = Math.min(hiddenSize, apiLimits.maxHiddenSize);
  }

  return { bars, epochs, hiddenSize };
}

function applyComboToForm(
  prev: FormState,
  combo: OptimizationCombo,
  apiLimits: ComputeLimits | null,
  manualOverrides?: Set<ManualOverrideKey>,
  allowPositioning = true,
): FormState {
  const nextPlatform = combo.params.platform ?? prev.platform;
  const comboSymbol = combo.params.binanceSymbol?.trim();
  const symbol = comboSymbol && comboSymbol.length > 0 ? comboSymbol : prev.binanceSymbol;
  const interval = combo.params.interval;
  const method = manualOverrides?.has("method") ? prev.method : combo.params.method;
  const comboPositioning = combo.params.positioning ?? prev.positioning;
  const positioning = allowPositioning ? comboPositioning : prev.positioning;
  const normalization = combo.params.normalization;
  const intrabarFill = combo.params.intrabarFill ?? prev.intrabarFill;
  const confirmConformal = combo.params.confirmConformal ?? prev.confirmConformal;
  const confirmQuantiles = combo.params.confirmQuantiles ?? prev.confirmQuantiles;
  const confidenceSizing = combo.params.confidenceSizing ?? prev.confidenceSizing;

  const comboForLimits =
    method === combo.params.method
      ? combo
      : {
          ...combo,
          params: { ...combo.params, method },
        };
  const { bars, epochs, hiddenSize } = clampComboForLimits(comboForLimits, apiLimits);
  const openThrRaw = coerceNumber(combo.openThreshold, prev.openThreshold);
  const closeThrRaw =
    combo.closeThreshold == null ? openThrRaw : coerceNumber(combo.closeThreshold, prev.closeThreshold);
  const openThreshold = manualOverrides?.has("openThreshold") ? prev.openThreshold : Math.max(0, openThrRaw);
  const closeThreshold = manualOverrides?.has("closeThreshold") ? prev.closeThreshold : Math.max(0, closeThrRaw);
  const fee = Math.max(0, coerceNumber(combo.params.fee, prev.fee));
  const learningRate = Math.max(1e-9, coerceNumber(combo.params.learningRate, prev.learningRate));
  const valRatio = clamp(coerceNumber(combo.params.valRatio, prev.valRatio), 0, 1);
  const patience = clamp(Math.trunc(coerceNumber(combo.params.patience, prev.patience)), 0, 1000);
  const gradClipRaw = coerceNumber(combo.params.gradClip, 0);
  const gradClip = gradClipRaw > 0 ? clamp(gradClipRaw, 0, 100) : 0;

  const slippage = clampOptionalRatio(combo.params.slippage);
  const spread = clampOptionalRatio(combo.params.spread);
  const stopLoss = clampOptionalRatio(combo.params.stopLoss);
  const takeProfit = clampOptionalRatio(combo.params.takeProfit);
  const trailingStop = clampOptionalRatio(combo.params.trailingStop);
  const stopLossVolMult = Math.max(0, coerceNumber(combo.params.stopLossVolMult ?? prev.stopLossVolMult, prev.stopLossVolMult));
  const takeProfitVolMult = Math.max(0, coerceNumber(combo.params.takeProfitVolMult ?? prev.takeProfitVolMult, prev.takeProfitVolMult));
  const trailingStopVolMult = Math.max(
    0,
    coerceNumber(combo.params.trailingStopVolMult ?? prev.trailingStopVolMult, prev.trailingStopVolMult),
  );
  const minHoldBars = clampOptionalInt(combo.params.minHoldBars ?? prev.minHoldBars, 0, 1_000_000);
  const maxHoldBars = clampOptionalInt(combo.params.maxHoldBars ?? prev.maxHoldBars, 0, 1_000_000);
  const cooldownBars = clampOptionalInt(combo.params.cooldownBars ?? prev.cooldownBars, 0, 1_000_000);
  const maxDrawdown = clampOptionalRatio(combo.params.maxDrawdown);
  const maxDailyLoss = clampOptionalRatio(combo.params.maxDailyLoss);
  const maxOrderErrors = clampOptionalInt(combo.params.maxOrderErrors, 1, 1_000_000);
  const minEdge = Math.max(0, coerceNumber(combo.params.minEdge ?? prev.minEdge, prev.minEdge));
  const minSignalToNoise = Math.max(0, coerceNumber(combo.params.minSignalToNoise ?? prev.minSignalToNoise, prev.minSignalToNoise));
  const costAwareEdge = combo.params.costAwareEdge ?? prev.costAwareEdge;
  const edgeBuffer = Math.max(0, coerceNumber(combo.params.edgeBuffer ?? prev.edgeBuffer, prev.edgeBuffer));
  const trendLookback = clampOptionalInt(combo.params.trendLookback ?? prev.trendLookback, 0, 1_000_000);
  const maxPositionSize = Math.max(0, coerceNumber(combo.params.maxPositionSize ?? prev.maxPositionSize, prev.maxPositionSize));
  const volTarget = Math.max(0, coerceNumber(combo.params.volTarget ?? prev.volTarget, prev.volTarget));
  const volLookback = Math.max(0, Math.trunc(coerceNumber(combo.params.volLookback ?? prev.volLookback, prev.volLookback)));
  const volEwmaAlphaRaw = coerceNumber(combo.params.volEwmaAlpha ?? prev.volEwmaAlpha, prev.volEwmaAlpha);
  const volEwmaAlpha = volEwmaAlphaRaw > 0 && volEwmaAlphaRaw < 1 ? volEwmaAlphaRaw : 0;
  const volFloor = Math.max(0, coerceNumber(combo.params.volFloor ?? prev.volFloor, prev.volFloor));
  const volScaleMax = Math.max(0, coerceNumber(combo.params.volScaleMax ?? prev.volScaleMax, prev.volScaleMax));
  const maxVolatility = Math.max(0, coerceNumber(combo.params.maxVolatility ?? prev.maxVolatility, prev.maxVolatility));
  const blendWeight = clamp(coerceNumber(combo.params.blendWeight ?? prev.blendWeight, prev.blendWeight), 0, 1);
  const tuneStressVolMult = Math.max(0, coerceNumber(combo.params.tuneStressVolMult ?? prev.tuneStressVolMult, prev.tuneStressVolMult));
  const tuneStressShock = coerceNumber(combo.params.tuneStressShock ?? prev.tuneStressShock, prev.tuneStressShock);
  const tuneStressWeight = Math.max(0, coerceNumber(combo.params.tuneStressWeight ?? prev.tuneStressWeight, prev.tuneStressWeight));
  const walkForwardFolds = clampOptionalInt(combo.params.walkForwardFolds ?? prev.walkForwardFolds, 1, 1000);

  const kalmanZMin = Math.max(0, coerceNumber(combo.params.kalmanZMin, prev.kalmanZMin));
  const kalmanZMax = Math.max(Math.max(0, coerceNumber(combo.params.kalmanZMax, prev.kalmanZMax)), kalmanZMin);
  const maxHighVolProb = clampOptionalRange(combo.params.maxHighVolProb, 0, 1);
  const maxConformalWidthRaw = coerceNumber(combo.params.maxConformalWidth, 0);
  const maxConformalWidth = maxConformalWidthRaw > 0 ? Math.max(0, maxConformalWidthRaw) : 0;
  const maxQuantileWidthRaw = coerceNumber(combo.params.maxQuantileWidth, 0);
  const maxQuantileWidth = maxQuantileWidthRaw > 0 ? Math.max(0, maxQuantileWidthRaw) : 0;
  const minPositionSize = clampOptionalRange(combo.params.minPositionSize, 0, 1);

  let lookbackBars = prev.lookbackBars;
  let lookbackWindow = prev.lookbackWindow;
  const intervalChanged = interval !== prev.interval;
  const prevLookbackBars = lookbackBars;

  if (intervalChanged && prevLookbackBars >= MIN_LOOKBACK_BARS) {
    const prevIntervalSec = platformIntervalSeconds(prev.platform, prev.interval);
    lookbackBars = 0;
    if (prevIntervalSec) {
      lookbackWindow = formatDurationSeconds(prevLookbackBars * prevIntervalSec);
    } else {
      const trimmed = lookbackWindow.trim();
      lookbackWindow = trimmed ? trimmed : defaultForm.lookbackWindow;
    }
  }

  if (lookbackBars >= MIN_LOOKBACK_BARS && bars > 0) {
    const maxLookback = Math.max(0, bars - 1);
    if (maxLookback < MIN_LOOKBACK_BARS) {
      lookbackBars = 0;
    } else if (lookbackBars >= maxLookback) {
      lookbackBars = maxLookback;
    }
  }

  if (lookbackBars < MIN_LOOKBACK_BARS) {
    const intervalSec = platformIntervalSeconds(nextPlatform, interval);
    if (intervalSec) {
      const windowSec = parseDurationSeconds(lookbackWindow);
      const minWindowSec = intervalSec * MIN_LOOKBACK_BARS;
      if (!windowSec || windowSec < minWindowSec) {
        lookbackWindow = formatDurationSeconds(minWindowSec);
      }
    }
  }

  return {
    ...prev,
    binanceSymbol: symbol,
    platform: nextPlatform,
    market: nextPlatform === "binance" ? prev.market : "spot",
    interval,
    bars,
    method,
    positioning,
    normalization,
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
    stopLoss,
    takeProfit,
    trailingStop,
    stopLossVolMult,
    takeProfitVolMult,
    trailingStopVolMult,
    minHoldBars,
    maxHoldBars,
    cooldownBars,
    maxDrawdown,
    maxDailyLoss,
    maxOrderErrors,
    minEdge,
    minSignalToNoise,
    costAwareEdge,
    edgeBuffer,
    trendLookback,
    maxPositionSize,
    volTarget,
    volLookback,
    volEwmaAlpha,
    volFloor,
    volScaleMax,
    maxVolatility,
    blendWeight,
    kalmanZMin,
    kalmanZMax,
    maxHighVolProb,
    maxConformalWidth,
    maxQuantileWidth,
    confirmConformal,
    confirmQuantiles,
    confidenceSizing,
    minPositionSize,
    binanceTestnet: nextPlatform === "binance" ? prev.binanceTestnet : false,
    binanceLive: nextPlatform === "binance" ? prev.binanceLive : false,
    tradeArmed: nextPlatform === "binance" ? prev.tradeArmed : false,
    tuneStressVolMult,
    tuneStressShock,
    tuneStressWeight,
    walkForwardFolds,
    lookbackBars,
    lookbackWindow,
    openThreshold,
    closeThreshold,
  };
}

function comboApplySignature(
  combo: OptimizationCombo,
  apiLimits: ComputeLimits | null,
  baseForm: FormState,
  manualOverrides?: Set<ManualOverrideKey>,
  allowPositioning = true,
): string {
  return formApplySignature(applyComboToForm(baseForm, combo, apiLimits, manualOverrides, allowPositioning));
}

function formApplySignature(form: FormState): string {
  let bars = Math.trunc(form.bars);
  if (!Number.isFinite(bars) || bars < 0) bars = 0;
  if (bars > 0 && bars < MIN_LOOKBACK_BARS) {
    bars = MIN_LOOKBACK_BARS;
  }
  const lookbackBarsRaw = Math.trunc(form.lookbackBars);
  const lookbackOverride = lookbackBarsRaw >= MIN_LOOKBACK_BARS;
  const lookbackBars = lookbackOverride ? lookbackBarsRaw : 0;
  const lookbackWindow = lookbackOverride ? "" : form.lookbackWindow.trim();
  const epochs = Math.max(0, Math.trunc(form.epochs));
  const hiddenSize = Math.max(1, Math.trunc(form.hiddenSize));
  const symbol = form.binanceSymbol.trim().toUpperCase();

  return [
    sigText(symbol),
    sigText(form.platform),
    sigText(form.interval.trim()),
    String(bars),
    lookbackOverride ? String(lookbackBars) : "",
    sigText(lookbackWindow),
    sigText(form.method),
    sigText(form.positioning),
    sigText(form.normalization),
    sigNumber(form.fee),
    String(epochs),
    String(hiddenSize),
    sigNumber(form.learningRate),
    sigNumber(form.valRatio),
    sigNumber(form.patience),
    sigNumber(form.gradClip),
    sigNumber(form.slippage),
    sigNumber(form.spread),
    sigText(form.intrabarFill),
    sigNumber(form.stopLoss),
    sigNumber(form.takeProfit),
    sigNumber(form.trailingStop),
    sigNumber(form.stopLossVolMult),
    sigNumber(form.takeProfitVolMult),
    sigNumber(form.trailingStopVolMult),
    sigNumber(form.minHoldBars),
    sigNumber(form.maxHoldBars),
    sigNumber(form.cooldownBars),
    sigNumber(form.maxDrawdown),
    sigNumber(form.maxDailyLoss),
    sigNumber(form.maxOrderErrors),
    sigNumber(form.minEdge),
    sigNumber(form.minSignalToNoise),
    sigBool(form.costAwareEdge),
    sigNumber(form.edgeBuffer),
    sigNumber(form.trendLookback),
    sigNumber(form.maxPositionSize),
    sigNumber(form.volTarget),
    sigNumber(form.volLookback),
    sigNumber(form.volEwmaAlpha),
    sigNumber(form.volFloor),
    sigNumber(form.volScaleMax),
    sigNumber(form.maxVolatility),
    sigNumber(form.blendWeight),
    sigNumber(form.kalmanZMin),
    sigNumber(form.kalmanZMax),
    sigNumber(form.maxHighVolProb),
    sigNumber(form.maxConformalWidth),
    sigNumber(form.maxQuantileWidth),
    sigBool(form.confirmConformal),
    sigBool(form.confirmQuantiles),
    sigBool(form.confidenceSizing),
    sigNumber(form.minPositionSize),
    sigNumber(form.tuneStressVolMult),
    sigNumber(form.tuneStressShock),
    sigNumber(form.tuneStressWeight),
    sigNumber(form.walkForwardFolds),
    sigNumber(form.openThreshold),
    sigNumber(form.closeThreshold),
  ].join("|");
}

export function App() {
  const [apiOk, setApiOk] = useState<"unknown" | "ok" | "down" | "auth">("unknown");
  const [healthInfo, setHealthInfo] = useState<Awaited<ReturnType<typeof health>> | null>(null);
  const apiComputeLimits = healthInfo?.computeLimits ?? null;
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
  const [coinbaseApiKey, setCoinbaseApiKey] = useState<string>(() => {
    const persisted = readLocalString(SESSION_COINBASE_KEY_KEY) ?? "";
    const session = readSessionString(SESSION_COINBASE_KEY_KEY) ?? "";
    return persistSecrets ? persisted || session : session;
  });
  const [coinbaseApiSecret, setCoinbaseApiSecret] = useState<string>(() => {
    const persisted = readLocalString(SESSION_COINBASE_SECRET_KEY) ?? "";
    const session = readSessionString(SESSION_COINBASE_SECRET_KEY) ?? "";
    return persistSecrets ? persisted || session : session;
  });
  const [coinbaseApiPassphrase, setCoinbaseApiPassphrase] = useState<string>(() => {
    const persisted = readLocalString(SESSION_COINBASE_PASSPHRASE_KEY) ?? "";
    const session = readSessionString(SESSION_COINBASE_PASSPHRASE_KEY) ?? "";
    return persistSecrets ? persisted || session : session;
  });
  const [form, setForm] = useState<FormState>(() => normalizeFormState(readJson<FormStateJson>(STORAGE_KEY)));
  const [customSymbolByPlatform, setCustomSymbolByPlatform] = useState<Record<Platform, string>>(() => ({
    binance: defaultForm.binanceSymbol,
    coinbase: PLATFORM_DEFAULT_SYMBOL.coinbase,
    kraken: PLATFORM_DEFAULT_SYMBOL.kraken,
    poloniex: PLATFORM_DEFAULT_SYMBOL.poloniex,
  }));
  const formRef = useRef(form);
  const apiComputeLimitsRef = useRef<ComputeLimits | null>(apiComputeLimits);
  const [manualOverrides, setManualOverrides] = useState<Set<ManualOverrideKey>>(() => new Set());
  const manualOverridesRef = useRef<Set<ManualOverrideKey>>(manualOverrides);

  apiComputeLimitsRef.current = apiComputeLimits;

  useEffect(() => {
    formRef.current = form;
  }, [form]);

  useEffect(() => {
    manualOverridesRef.current = manualOverrides;
  }, [manualOverrides]);

  const platform = form.platform;
  const isBinancePlatform = platform === "binance";
  const isCoinbasePlatform = platform === "coinbase";
  const keysSupported = isBinancePlatform || isCoinbasePlatform;
  const platformSymbolSet = PLATFORM_SYMBOL_SET[platform];
  const platformSymbols = PLATFORM_SYMBOLS[platform];
  const platformLabel = PLATFORM_LABELS[platform];
  const platformIntervals = PLATFORM_INTERVALS[platform];
  const platformKeyMode = isCoinbasePlatform ? "coinbase" : isBinancePlatform ? "binance" : null;
  const platformKeyLabel =
    platformKeyMode === "coinbase" ? "Coinbase API keys (optional)" : platformKeyMode === "binance" ? "Binance API keys (optional)" : "Platform API keys (optional)";
  const platformKeyHint =
    platformKeyMode === "coinbase"
      ? "Used for /trade (live only) and “Check keys” (Coinbase signed /accounts)."
      : platformKeyMode === "binance"
        ? "Used for /trade and “Check keys”."
        : "Keys can be stored for Binance or Coinbase. Switch Platform to edit them.";
  const platformKeyHasValues =
    platformKeyMode === "coinbase"
      ? Boolean(coinbaseApiKey.trim() || coinbaseApiSecret.trim() || coinbaseApiPassphrase.trim())
      : platformKeyMode === "binance"
        ? Boolean(binanceApiKey.trim() || binanceApiSecret.trim())
        : false;
  const normalizedSymbol = form.binanceSymbol.trim().toUpperCase();
  const symbolIsCustom = !platformSymbolSet.has(normalizedSymbol);
  const symbolSelectValue = symbolIsCustom ? CUSTOM_SYMBOL_VALUE : normalizedSymbol;
  const botSymbolsInput = useMemo(() => parseSymbolsInput(form.botSymbols), [form.botSymbols]);
  const botStartSymbols = useMemo(() => {
    if (botSymbolsInput.length > 0) return botSymbolsInput;
    const fallback = form.binanceSymbol.trim().toUpperCase();
    return fallback ? [fallback] : [];
  }, [botSymbolsInput, form.binanceSymbol]);
  const botMissingSymbol = botStartSymbols.length === 0;

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
  const [botSelectedSymbol, setBotSelectedSymbol] = useState<string | null>(null);

  const [binanceTradesUi, setBinanceTradesUi] = useState<BinanceTradesUiState>({
    loading: false,
    error: null,
    response: null,
  });
  const [binanceTradesSymbolsInput, setBinanceTradesSymbolsInput] = useState(() => form.binanceSymbol.trim());
  const [binanceTradesLimit, setBinanceTradesLimit] = useState(200);
  const [binanceTradesStartInput, setBinanceTradesStartInput] = useState("");
  const [binanceTradesEndInput, setBinanceTradesEndInput] = useState("");
  const [binanceTradesFromIdInput, setBinanceTradesFromIdInput] = useState("");

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
    platform: null,
    checkedAtMs: null,
  });
  const activeKeysStatus = keys.platform === platform ? keys.status : null;
  const keysProvided =
    activeKeysStatus
      ? isCoinbaseKeysStatus(activeKeysStatus)
        ? activeKeysStatus.hasApiKey && activeKeysStatus.hasApiSecret && activeKeysStatus.hasApiPassphrase
        : activeKeysStatus.hasApiKey && activeKeysStatus.hasApiSecret
      : null;
  const keysProvidedLabel = keysProvided === null ? "unknown" : keysProvided ? "provided" : "missing";
  const keysSigned = activeKeysStatus?.signed ?? null;
  const keysTradeTest =
    activeKeysStatus && isBinanceKeysStatus(activeKeysStatus) ? activeKeysStatus.tradeTest ?? null : null;
  const keysCheckedAtMs = keys.platform === platform ? keys.checkedAtMs : null;

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
  const [selectedOrderKey, setSelectedOrderKey] = useState<string | null>(null);

  const [dataLog, setDataLog] = useState<Array<{ timestamp: number; label: string; data: unknown }>>([]);
  const [dataLogExpanded, setDataLogExpanded] = useState(false);
  const [dataLogIndexArrays, setDataLogIndexArrays] = useState(true);
  const [dataLogFilterText, setDataLogFilterText] = useState("");
  const [dataLogAutoScroll, setDataLogAutoScroll] = useState(true);
  const [topCombos, setTopCombos] = useState<OptimizationCombo[]>([]);
  const [topCombosLoading, setTopCombosLoading] = useState(true);
  const [topCombosError, setTopCombosError] = useState<string | null>(null);
  const [topCombosMeta, setTopCombosMeta] = useState<TopCombosMeta>({
    source: "static",
    generatedAtMs: null,
    payloadSource: null,
    payloadSources: null,
    fallbackReason: null,
    comboCount: null,
  });
  const [autoAppliedCombo, setAutoAppliedCombo] = useState<{ id: number; atMs: number } | null>(null);
  const autoAppliedComboRef = useRef<{ id: number | null; atMs: number | null }>({ id: null, atMs: null });
  const [selectedComboId, setSelectedComboId] = useState<number | null>(null);
  const topCombosRef = useRef<OptimizationCombo[]>([]);
  const topCombosSyncRef = useRef<((opts?: { silent?: boolean }) => void) | null>(null);
  const dataLogRef = useRef<HTMLDivElement | null>(null);
  const sectionFlashRef = useRef<HTMLElement | null>(null);
  const sectionFlashTimeoutRef = useRef<number | null>(null);

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
    if (!symbolIsCustom) return;
    setCustomSymbolByPlatform((prev) => {
      const nextValue = normalizedSymbol || prev[platform];
      if (prev[platform] === nextValue) return prev;
      return { ...prev, [platform]: nextValue };
    });
  }, [platform, normalizedSymbol, symbolIsCustom]);

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
    const v = coinbaseApiKey.trim();
    if (persistSecrets) {
      if (!v) removeLocalKey(SESSION_COINBASE_KEY_KEY);
      else writeLocalString(SESSION_COINBASE_KEY_KEY, v);
      removeSessionKey(SESSION_COINBASE_KEY_KEY);
    } else {
      if (!v) removeSessionKey(SESSION_COINBASE_KEY_KEY);
      else writeSessionString(SESSION_COINBASE_KEY_KEY, v);
      removeLocalKey(SESSION_COINBASE_KEY_KEY);
    }
  }, [coinbaseApiKey, persistSecrets]);

  useEffect(() => {
    const v = coinbaseApiSecret.trim();
    if (persistSecrets) {
      if (!v) removeLocalKey(SESSION_COINBASE_SECRET_KEY);
      else writeLocalString(SESSION_COINBASE_SECRET_KEY, v);
      removeSessionKey(SESSION_COINBASE_SECRET_KEY);
    } else {
      if (!v) removeSessionKey(SESSION_COINBASE_SECRET_KEY);
      else writeSessionString(SESSION_COINBASE_SECRET_KEY, v);
      removeLocalKey(SESSION_COINBASE_SECRET_KEY);
    }
  }, [coinbaseApiSecret, persistSecrets]);

  useEffect(() => {
    const v = coinbaseApiPassphrase.trim();
    if (persistSecrets) {
      if (!v) removeLocalKey(SESSION_COINBASE_PASSPHRASE_KEY);
      else writeLocalString(SESSION_COINBASE_PASSPHRASE_KEY, v);
      removeSessionKey(SESSION_COINBASE_PASSPHRASE_KEY);
    } else {
      if (!v) removeSessionKey(SESSION_COINBASE_PASSPHRASE_KEY);
      else writeSessionString(SESSION_COINBASE_PASSPHRASE_KEY, v);
      removeLocalKey(SESSION_COINBASE_PASSPHRASE_KEY);
    }
  }, [coinbaseApiPassphrase, persistSecrets]);

  useEffect(() => {
    writeJson(STORAGE_PERSIST_SECRETS_KEY, persistSecrets);
  }, [persistSecrets]);

  const toastTimerRef = useRef<number | null>(null);
  const showToast = useCallback((msg: string) => {
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    setToast(msg);
    toastTimerRef.current = window.setTimeout(() => setToast(null), 1800);
  }, []);

  const updateManualOverrides = useCallback((updater: (next: Set<ManualOverrideKey>) => void) => {
    setManualOverrides((prev) => {
      const next = new Set(prev);
      updater(next);
      manualOverridesRef.current = next;
      return next;
    });
  }, []);

  const markManualOverrides = useCallback(
    (keys: ManualOverrideKey[]) => {
      updateManualOverrides((next) => {
        for (const key of keys) next.add(key);
      });
    },
    [updateManualOverrides],
  );

  const clearManualOverrides = useCallback(
    (keys?: ManualOverrideKey[]) => {
      updateManualOverrides((next) => {
        if (!keys || keys.length === 0) {
          next.clear();
          return;
        }
        for (const key of keys) next.delete(key);
      });
    },
    [updateManualOverrides],
  );

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

  const applyCombo = useCallback(
    (combo: OptimizationCombo, opts?: { silent?: boolean; respectManual?: boolean; allowPositioning?: boolean }) => {
      const comboSymbol = combo.params.binanceSymbol?.trim();
      const manualOverrides = opts?.respectManual ? manualOverridesRef.current : undefined;
      const allowPositioning = opts?.allowPositioning ?? true;
      setForm((prev) => applyComboToForm(prev, combo, apiComputeLimitsRef.current, manualOverrides, allowPositioning));
      setSelectedComboId(combo.id);
      if (!opts?.silent) {
        showToast(`Loaded optimizer combo #${combo.id}${comboSymbol ? ` (${comboSymbol})` : ""}`);
      }
    },
    [showToast],
  );

  const handleComboPreview = useCallback((combo: OptimizationCombo) => setSelectedComboId(combo.id), []);
  const handleComboApply = useCallback((combo: OptimizationCombo) => applyCombo(combo, { respectManual: true }), [applyCombo]);
  const refreshTopCombos = useCallback(() => {
    topCombosSyncRef.current?.({ silent: false });
  }, []);
  const scrollDataLogToBottom = useCallback(() => {
    const el = dataLogRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, []);

  useEffect(() => {
    topCombosRef.current = topCombos;
  }, [topCombos]);

  useEffect(() => {
    if (!dataLogAutoScroll) return;
    window.requestAnimationFrame(() => {
      if (!dataLogAutoScroll) return;
      scrollDataLogToBottom();
    });
  }, [dataLog, dataLogAutoScroll, scrollDataLogToBottom]);

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

    clearManualOverrides();
    setForm(profile);
    setPendingProfileLoad(null);
    showToast(`Profile loaded: ${name}`);
  }, [clearManualOverrides, form.binanceLive, form.market, form.tradeArmed, profileSelected, profiles, showToast]);

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

  const apiBaseCrossOrigin = useMemo(() => {
    if (!/^https?:\/\//i.test(apiBase)) return false;
    try {
      const apiOrigin = new URL(apiBase).origin;
      if (typeof window === "undefined") return false;
      return apiOrigin !== window.location.origin;
    } catch {
      return false;
    }
  }, [apiBase]);

  const apiBaseCorsHint = useMemo(() => {
    if (!apiBaseCrossOrigin || typeof window === "undefined") return null;
    if (window.location.hostname.endsWith("cloudfront.net")) {
      return "Cross-origin API base. If CloudFront has a /api/* behavior, set apiBaseUrl to /api to avoid CORS/preflight.";
    }
    return "Cross-origin API base. Ensure the API allows this origin, or use a same-origin /api proxy.";
  }, [apiBaseCrossOrigin]);

  const apiBaseAbsolute = useMemo(() => {
    if (/^https?:\/\//.test(apiBase)) return apiBase;
    if (typeof window !== "undefined") return `${window.location.origin}${apiBase}`;
    return apiBase;
  }, [apiBase]);

  const apiHealthUrl = useMemo(() => {
    if (!apiBaseAbsolute) return "";
    return `${apiBaseAbsolute.replace(/\/+$/, "")}/health`;
  }, [apiBaseAbsolute]);

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
    if (!isBinancePlatform) return;
    if (form.market !== "margin") return;
    if (form.binanceLive) return;
    setForm((f) => ({ ...f, market: "spot" }));
    showToast("Margin requires Live orders (switched back to Spot)");
  }, [form.binanceLive, form.market, isBinancePlatform, showToast]);

  useEffect(() => {
    if (!isBinancePlatform) return;
    if (!form.tradeArmed) return;
    if (form.positioning !== "long-short") return;
    if (form.market === "futures") return;
    setPendingMarket(null);
    setForm((f) => ({ ...f, market: "futures" }));
    showToast("Long/Short trading requires Futures (switched Market to Futures)");
  }, [form.market, form.positioning, form.tradeArmed, isBinancePlatform, showToast]);

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
    const intervalOk = PLATFORM_INTERVAL_SET[platform].has(interval);
    const barsRaw = Math.trunc(form.bars);
    const bars = barsRaw <= 0 ? 0 : platform === "binance" ? clamp(barsRaw, 2, 1000) : Math.max(2, barsRaw);
    const volTarget = Math.max(0, form.volTarget);
    const volEwmaAlphaRaw = form.volEwmaAlpha;
    const volEwmaAlpha = volEwmaAlphaRaw > 0 && volEwmaAlphaRaw < 1 ? volEwmaAlphaRaw : 0;
    const volLookbackRaw = Math.max(0, Math.trunc(form.volLookback));
    const volLookback = volTarget > 0 && volEwmaAlpha === 0 ? Math.max(2, volLookbackRaw) : volLookbackRaw;
    const tuneStressVolMult = form.tuneStressVolMult <= 0 ? 1 : form.tuneStressVolMult;
    const base: ApiParams = {
      binanceSymbol: form.binanceSymbol.trim() || undefined,
      platform: form.platform,
      ...(platform === "binance" ? { market: form.market } : {}),
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
      ...(form.stopLossVolMult > 0 ? { stopLossVolMult: Math.max(0, form.stopLossVolMult) } : {}),
      ...(form.takeProfitVolMult > 0 ? { takeProfitVolMult: Math.max(0, form.takeProfitVolMult) } : {}),
      ...(form.trailingStopVolMult > 0 ? { trailingStopVolMult: Math.max(0, form.trailingStopVolMult) } : {}),
      ...(form.minHoldBars > 0 ? { minHoldBars: clamp(Math.trunc(form.minHoldBars), 0, 1_000_000) } : {}),
      ...(form.maxHoldBars > 0 ? { maxHoldBars: clamp(Math.trunc(form.maxHoldBars), 1, 1_000_000) } : {}),
      ...(form.cooldownBars > 0 ? { cooldownBars: clamp(Math.trunc(form.cooldownBars), 0, 1_000_000) } : {}),
      ...(form.maxDrawdown > 0 ? { maxDrawdown: clamp(form.maxDrawdown, 0, 0.999999) } : {}),
      ...(form.maxDailyLoss > 0 ? { maxDailyLoss: clamp(form.maxDailyLoss, 0, 0.999999) } : {}),
      ...(form.maxOrderErrors >= 1 ? { maxOrderErrors: clamp(Math.trunc(form.maxOrderErrors), 1, 1_000_000) } : {}),
      minEdge: Math.max(0, form.minEdge),
      ...(form.minSignalToNoise > 0 ? { minSignalToNoise: Math.max(0, form.minSignalToNoise) } : {}),
      edgeBuffer: Math.max(0, form.edgeBuffer),
      trendLookback: clamp(Math.trunc(form.trendLookback), 0, 1_000_000),
      maxPositionSize: Math.max(0, form.maxPositionSize),
      volLookback,
      volFloor: Math.max(0, form.volFloor),
      volScaleMax: Math.max(0, form.volScaleMax),
      blendWeight: clamp(form.blendWeight, 0, 1),
      backtestRatio: clamp(form.backtestRatio, 0.01, 0.99),
      tuneRatio: clamp(form.tuneRatio, 0, 0.99),
      tuneObjective: form.tuneObjective,
      tunePenaltyMaxDrawdown: Math.max(0, form.tunePenaltyMaxDrawdown),
      tunePenaltyTurnover: Math.max(0, form.tunePenaltyTurnover),
      tuneStressVolMult,
      tuneStressShock: form.tuneStressShock,
      tuneStressWeight: Math.max(0, form.tuneStressWeight),
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
      ...(platform === "binance" ? { binanceTestnet: form.binanceTestnet } : {}),
    };

    if (form.lookbackBars >= 2) base.lookbackBars = Math.trunc(form.lookbackBars);
    else if (form.lookbackWindow.trim()) base.lookbackWindow = form.lookbackWindow.trim();

    if (form.optimizeOperations) base.optimizeOperations = true;
    if (form.sweepThreshold) base.sweepThreshold = true;

    if (form.costAwareEdge) base.costAwareEdge = true;
    if (volTarget > 0) base.volTarget = volTarget;
    if (volEwmaAlpha > 0) base.volEwmaAlpha = volEwmaAlpha;
    if (form.maxVolatility > 0) base.maxVolatility = Math.max(0, form.maxVolatility);

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

  const minEdgeEffective = useMemo(() => {
    const base = Math.max(0, form.minEdge);
    if (!form.costAwareEdge) return base;
    const buffer = Math.max(0, form.edgeBuffer);
    return Math.max(base, estimatedCosts.breakEven + buffer);
  }, [estimatedCosts.breakEven, form.costAwareEdge, form.edgeBuffer, form.minEdge]);

  const tradeParams: ApiParams = useMemo(() => {
    const base: ApiParams = { ...commonParams };
    if (form.platform === "binance" && form.binanceLive) base.binanceLive = true;
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
    form.platform,
  ]);

  const withBinanceKeys = useCallback(
    (p: ApiParams): ApiParams => {
      if (form.platform !== "binance") return p;
      const key = binanceApiKey.trim();
      const secret = binanceApiSecret.trim();
      if (!key && !secret) return p;
      return {
        ...p,
        ...(key ? { binanceApiKey: key } : {}),
        ...(secret ? { binanceApiSecret: secret } : {}),
      };
    },
    [binanceApiKey, binanceApiSecret, form.platform],
  );

  const withPlatformKeys = useCallback(
    (p: ApiParams): ApiParams => {
      if (isBinancePlatform) return withBinanceKeys(p);
      if (!isCoinbasePlatform) return p;
      const key = coinbaseApiKey.trim();
      const secret = coinbaseApiSecret.trim();
      const passphrase = coinbaseApiPassphrase.trim();
      if (!key && !secret && !passphrase) return p;
      return {
        ...p,
        ...(key ? { coinbaseApiKey: key } : {}),
        ...(secret ? { coinbaseApiSecret: secret } : {}),
        ...(passphrase ? { coinbaseApiPassphrase: passphrase } : {}),
      };
    },
    [coinbaseApiKey, coinbaseApiPassphrase, coinbaseApiSecret, isBinancePlatform, isCoinbasePlatform, withBinanceKeys],
  );

  const keysParams: ApiParams = useMemo(() => {
    const base: ApiParams = {
      binanceSymbol: form.binanceSymbol.trim() || undefined,
      platform: form.platform,
      ...(form.platform === "binance" ? { market: form.market, binanceTestnet: form.binanceTestnet } : {}),
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

    return withPlatformKeys(base);
  }, [
    form.binanceSymbol,
    form.binanceTestnet,
    form.idempotencyKey,
    form.market,
    form.maxOrderQuote,
    form.orderQuantity,
    form.orderQuote,
    form.orderQuoteFraction,
    form.platform,
    withPlatformKeys,
  ]);

  const orderRowKey = useCallback((e: BotOrderEvent) => `${e.atMs}-${e.index}-${e.opSide}`, []);
  const botEntries = useMemo<BotStatusSingle[]>(
    () => (isBotStatusMulti(bot.status) ? bot.status.bots : [bot.status as BotStatusSingle]),
    [bot.status],
  );
  const botEntriesWithSymbol = useMemo(
    () =>
      botEntries
        .map((status) => {
          const symbol = botStatusSymbol(status);
          return symbol ? { symbol, status } : null;
        })
        .filter((entry): entry is { symbol: string; status: BotStatusSingle } => Boolean(entry)),
    [botEntries],
  );
  const botStatusBySymbol = useMemo(() => new Map(botEntriesWithSymbol.map((entry) => [entry.symbol, entry.status])), [botEntriesWithSymbol]);
  const botSymbolsActive = useMemo(() => botEntriesWithSymbol.map((entry) => entry.symbol), [botEntriesWithSymbol]);
  const botSymbolOptions = useMemo(
    () =>
      botEntriesWithSymbol.map((entry) => {
        const starting = !entry.status.running && entry.status.starting === true;
        const snapshot = !entry.status.running && Boolean(entry.status.snapshot);
        const label = `${entry.symbol}${entry.status.running ? " (running)" : starting ? " (starting)" : snapshot ? " (snapshot)" : " (stopped)"}`;
        return { symbol: entry.symbol, label, starting, snapshot, running: entry.status.running };
      }),
    [botEntriesWithSymbol],
  );
  const botActiveSymbols = useMemo(
    () =>
      botEntriesWithSymbol
        .filter((entry) => entry.status.running || (!entry.status.running && entry.status.starting === true))
        .map((entry) => entry.symbol),
    [botEntriesWithSymbol],
  );
  const botActiveSymbolSet = useMemo(() => new Set(botActiveSymbols), [botActiveSymbols]);
  const botHasNewSymbols = useMemo(
    () => (botStartSymbols.length > 0 ? botStartSymbols.some((sym) => !botActiveSymbolSet.has(sym)) : false),
    [botActiveSymbolSet, botStartSymbols],
  );
  const botSelectedStatus = useMemo(() => {
    if (botEntriesWithSymbol.length === 0) return null;
    const target = botSelectedSymbol ?? botEntriesWithSymbol[0]!.symbol;
    return botStatusBySymbol.get(target) ?? null;
  }, [botEntriesWithSymbol, botSelectedSymbol, botStatusBySymbol]);
  const botStartErrors = useMemo(() => (isBotStatusMulti(bot.status) ? bot.status.errors ?? [] : []), [bot.status]);
  const botSnapshot = useMemo(
    () => (botSelectedStatus && !botSelectedStatus.running ? botSelectedStatus.snapshot ?? null : null),
    [botSelectedStatus],
  );
  const botDisplay = botSelectedStatus?.running ? botSelectedStatus : botSnapshot;
  const botSnapshotAtMs = botSelectedStatus?.running ? null : botSelectedStatus?.snapshotAtMs ?? null;
  const botHasSnapshot = botSnapshot !== null;

  useEffect(() => {
    if (botEntriesWithSymbol.length === 0) {
      if (botSelectedSymbol !== null) setBotSelectedSymbol(null);
      return;
    }
    const hasSelected = botSelectedSymbol && botEntriesWithSymbol.some((entry) => entry.symbol === botSelectedSymbol);
    if (!hasSelected) setBotSelectedSymbol(botEntriesWithSymbol[0]!.symbol);
  }, [botEntriesWithSymbol, botSelectedSymbol]);

  const botOrdersView = useMemo(() => {
    const st = botDisplay;
    if (!st) return { total: 0, shown: [] as BotOrderEvent[], startIndex: 0 };

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
  }, [botDisplay, orderErrorsOnly, orderFilterText, orderLimit, orderSentOnly, orderSideFilter]);

  const selectedOrderDetails = useMemo(() => {
    const st = botDisplay;
    if (!st || !selectedOrderKey) return null;
    const event = st.orders.find((e) => orderRowKey(e) === selectedOrderKey);
    if (!event) return null;
    const idx = event.index;
    const bar = st.startIndex + idx;
    const close = st.prices[idx] ?? event.price;
    const open = idx > 0 ? st.prices[idx - 1] ?? close : close;
    const eq = st.equityCurve[idx];
    const pos = st.positions[idx] ?? 0;
    const kal = st.kalmanPredNext[idx] ?? null;
    const lstm = st.lstmPredNext[idx] ?? null;
    const closeForRet = typeof close === "number" && Number.isFinite(close) && close !== 0 ? close : null;
    const kalRet = typeof kal === "number" && Number.isFinite(kal) && closeForRet ? (kal - closeForRet) / closeForRet : null;
    const lstmRet = typeof lstm === "number" && Number.isFinite(lstm) && closeForRet ? (lstm - closeForRet) / closeForRet : null;
    return { event, result: event.order, idx, bar, open, close, eq, pos, kal, lstm, kalRet, lstmRet };
  }, [botDisplay, orderRowKey, selectedOrderKey]);

  const selectedOrderJson = useMemo(() => {
    if (!selectedOrderDetails) return "";
    const { event, bar, open, close, eq, pos, kal, lstm, kalRet, lstmRet } = selectedOrderDetails;
    return JSON.stringify(
      {
        ...event,
        bar,
        openPrice: open,
        closePrice: close,
        equity: eq,
        position: pos,
        kalmanPredNext: kal,
        lstmPredNext: lstm,
        kalmanReturn: kalRet,
        lstmReturn: lstmRet,
      },
      null,
      2,
    );
  }, [selectedOrderDetails]);

  const botOrderCopyText = useMemo(() => {
    const st = botDisplay;
    if (!st) return "";
    const rows = botOrdersView.shown.map((e) => {
      const bar = st.startIndex + e.index;
      const sent = e.order.sent ? "SENT" : "NO";
      const mode = e.order.mode ?? "—";
      return `${fmtTimeMs(e.atMs)} | bar ${bar} | ${e.opSide} @ ${fmtMoney(e.price, 4)} | ${sent} ${mode} | ${e.order.message}`;
    });
    return rows.length ? rows.join("\n") : "No live operations yet.";
  }, [botDisplay, botOrdersView.shown]);

  const botRtFeedText = useMemo(() => {
    if (botRt.feed.length === 0) return "No realtime events yet.";
    return botRt.feed.map((e) => `${fmtTimeMs(e.atMs)} | ${e.message}`).join("\n");
  }, [botRt.feed]);

  const binanceTradesCopyText = useMemo(() => {
    const trades = binanceTradesUi.response?.trades ?? [];
    if (trades.length === 0) return "No trades loaded.";
    return trades
      .map((trade) => {
        const side = trade.side ?? (trade.isBuyer === true ? "BUY" : trade.isBuyer === false ? "SELL" : "—");
        const qty = typeof trade.qty === "number" && Number.isFinite(trade.qty) ? fmtNum(trade.qty, 8) : "—";
        const quote = typeof trade.quoteQty === "number" && Number.isFinite(trade.quoteQty) ? fmtMoney(trade.quoteQty, 2) : "—";
        return `${fmtTimeMs(trade.time)} | ${trade.symbol} | ${side} | ${fmtMoney(trade.price, 4)} | ${qty} | ${quote}`;
      })
      .join("\n");
  }, [binanceTradesUi.response]);

  const binanceTradesJson = useMemo(() => {
    const trades = binanceTradesUi.response?.trades ?? [];
    if (trades.length === 0) return "";
    return JSON.stringify(trades, null, 2);
  }, [binanceTradesUi.response]);

  const botRisk = useMemo(() => {
    const st = botDisplay;
    if (!st) return null;
    const lastEq = st.equityCurve[st.equityCurve.length - 1] ?? 1;
    const peak = st.peakEquity || 1;
    const dayStart = st.dayStartEquity || 1;
    const dd = peak > 0 ? Math.max(0, 1 - lastEq / peak) : 0;
    const dl = dayStart > 0 ? Math.max(0, 1 - lastEq / dayStart) : 0;
    return { lastEq, dd, dl };
  }, [botDisplay]);

  const botRealtime = useMemo(() => {
    const st = botDisplay;
    if (!st) return null;
    const now = Date.now();
    const processedOpenTime = st.openTimes[st.openTimes.length - 1] ?? null;
    const fetchedLast = st.fetchedLastKline ?? null;
    const fetchedOpenTime = fetchedLast?.openTime ?? null;
    const candleOpenTime = fetchedOpenTime ?? processedOpenTime;
    const intervalSec = platformIntervalSeconds(platform, st.interval) ?? parseDurationSeconds(st.interval);
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
  }, [botDisplay]);

  const scrollToResult = useCallback((kind: RequestKind) => {
    const ref = kind === "signal" ? signalRef : kind === "backtest" ? backtestRef : tradeRef;
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);
  const scrollToSection = useCallback((id?: string) => {
    if (!id || typeof document === "undefined") return;
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    if (sectionFlashRef.current && sectionFlashRef.current !== el) {
      sectionFlashRef.current.classList.remove("sectionFlash");
    }
    sectionFlashRef.current = el;
    el.classList.remove("sectionFlash");
    void el.offsetWidth;
    el.classList.add("sectionFlash");
    if (sectionFlashTimeoutRef.current) window.clearTimeout(sectionFlashTimeoutRef.current);
    sectionFlashTimeoutRef.current = window.setTimeout(() => {
      el.classList.remove("sectionFlash");
    }, 1200);
  }, []);

  useEffect(() => {
    return () => {
      if (sectionFlashTimeoutRef.current) {
        window.clearTimeout(sectionFlashTimeoutRef.current);
        sectionFlashTimeoutRef.current = null;
      }
    };
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
        if (!p.binanceSymbol) throw new Error("Symbol is required.");
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
            const manualOverrides = manualOverridesRef.current;
            const next: Partial<FormState> = {};
            if (p.optimizeOperations && !manualOverrides.has("method")) next.method = out.method;
            if (!manualOverrides.has("openThreshold")) next.openThreshold = Math.max(0, openThreshold);
            if (!manualOverrides.has("closeThreshold")) next.closeThreshold = Math.max(0, closeThreshold);
            if (Object.keys(next).length > 0) {
              setForm((f) => ({ ...f, ...next }));
            }
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
            const manualOverrides = manualOverridesRef.current;
            const next: Partial<FormState> = {};
            if (p.optimizeOperations && !manualOverrides.has("method")) next.method = out.method;
            if (!manualOverrides.has("openThreshold")) next.openThreshold = Math.max(0, openThreshold);
            if (!manualOverrides.has("closeThreshold")) next.closeThreshold = Math.max(0, closeThreshold);
            if (Object.keys(next).length > 0) {
              setForm((f) => ({ ...f, ...next }));
            }
          }
          setState((s) => ({ ...s, backtest: out, latestSignal: out.latestSignal, trade: null, loading: false, error: null }));
          setDataLog((logs) => [...logs, { timestamp: Date.now(), label: "Backtest Response", data: out }].slice(-100));
          setApiOk("ok");
          if (!opts?.silent) showToast("Backtest complete");
        } else {
          if (!form.tradeArmed) throw new Error("Trading is locked. Enable “Arm trading” to call /trade.");
          const out = await trade(apiBase, withPlatformKeys(p), {
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
            const manualOverrides = manualOverridesRef.current;
            const next: Partial<FormState> = {};
            if (p.optimizeOperations && !manualOverrides.has("method")) next.method = sig.method;
            if (!manualOverrides.has("openThreshold")) next.openThreshold = Math.max(0, openThreshold);
            if (!manualOverrides.has("closeThreshold")) next.closeThreshold = Math.max(0, closeThreshold);
            if (Object.keys(next).length > 0) {
              setForm((f) => ({ ...f, ...next }));
            }
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
        if (e instanceof HttpError && (e.status === 502 || e.status === 503)) {
          msg = apiBase.startsWith("/api")
            ? "CloudFront `/api/*` proxy is unavailable (502/503). Point `/api/*` at your API origin (App Runner/ALB/etc) and allow POST/GET/OPTIONS, or set apiBaseUrl in trader-config.js to https://<your-api-host>."
            : "API gateway unavailable (502/503). Try again, or check the API logs.";
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
      withPlatformKeys,
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
      if (!isBinancePlatform && !isCoinbasePlatform) {
        const msg = "Key checks require Platform=Binance or Coinbase.";
        if (!opts?.silent) showToast(msg);
        setKeys((s) => ({ ...s, loading: false, error: msg, status: null, platform }));
        return;
      }
      const requestId = ++keysRequestSeqRef.current;
      keysAbortRef.current?.abort();
      const controller = new AbortController();
      keysAbortRef.current = controller;

      setKeys((s) => ({ ...s, loading: true, error: opts?.silent ? s.error : null, platform }));

      try {
        const p = keysParams;
        if (!p.binanceSymbol) throw new Error("Symbol is required.");

        const out = isBinancePlatform
          ? await binanceKeysStatus(apiBase, p, { signal: controller.signal, headers: authHeaders, timeoutMs: 30_000 })
          : await coinbaseKeysStatus(apiBase, p, { signal: controller.signal, headers: authHeaders, timeoutMs: 30_000 });
        if (requestId !== keysRequestSeqRef.current) return;
        setKeys({ loading: false, error: null, status: out, platform, checkedAtMs: Date.now() });
        setApiOk("ok");
        if (!opts?.silent) showToast("Key status updated");
      } catch (e) {
        if (requestId !== keysRequestSeqRef.current) return;
        if (isAbortError(e)) return;

        let msg = e instanceof Error ? e.message : String(e);
        if (isTimeoutError(e)) msg = isBinancePlatform ? "Key check timed out. Try again, or switch testnet off." : "Key check timed out. Try again.";
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
          setKeys((s) => ({ ...s, loading: false, platform }));
          return;
        }

        setKeys((s) => ({ ...s, loading: false, error: msg, platform }));
        showToast("Key check failed");
      } finally {
        if (requestId === keysRequestSeqRef.current) keysAbortRef.current = null;
      }
    },
    [apiBase, authHeaders, isBinancePlatform, isCoinbasePlatform, keysParams, platform, showToast],
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
    if (!isBinancePlatform) {
      const msg = "Listen key streams are supported on Binance only.";
      setListenKeyUi((s) => ({ ...s, error: msg, wsStatus: "disconnected" }));
      showToast(msg);
      return;
    }
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
  }, [
    apiBase,
    apiOk,
    authHeaders,
    form.binanceTestnet,
    form.market,
    isBinancePlatform,
    keepAliveListenKeyStream,
    showToast,
    stopListenKeyStream,
    withBinanceKeys,
  ]);

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
        const selectedStatus: BotStatusSingle | null = (() => {
          if (!isBotStatusMulti(out)) return out;
          if (out.bots.length === 0) return null;
          if (botSelectedSymbol) {
            const match = out.bots.find((entry) => botStatusSymbol(entry) === botSelectedSymbol);
            if (match) return match;
          }
          return out.bots[0] ?? null;
        })();
        const selectedRunning = selectedStatus && selectedStatus.running ? selectedStatus : null;
        setBotRt((prev) => {
          const base: BotRtUiState = {
            ...prev,
            lastFetchAtMs: finishedAtMs,
            lastFetchDurationMs: Math.max(0, finishedAtMs - startedAtMs),
            lastNewCandles: 0,
            lastKlineUpdates: 0,
          };

          const rt = botRtRef.current;

          if (!selectedRunning) {
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

          const st = selectedRunning;
          const botKey = `${st.market}:${st.symbol}:${st.interval}`;
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

          const openTimes = st.openTimes;
          const lastOpen = openTimes[openTimes.length - 1] ?? null;
          const prevLastOpen = rt.lastOpenTimeMs;
          const newTimes = typeof prevLastOpen === "number" ? openTimes.filter((t) => t > prevLastOpen) : [];
          const newCount = newTimes.length;

	          let lastNewCandlesAtMs: number | null = prev.lastNewCandlesAtMs;
	          if (newCount > 0) {
	            lastNewCandlesAtMs = finishedAtMs;
	            const lastNew = newTimes[newTimes.length - 1]!;
	            const idx = openTimes.lastIndexOf(lastNew);
	            const closePx = idx >= 0 ? st.prices[idx] : null;
	            const action = st.latestSignal.action;
	            const pollMs = typeof st.pollLatencyMs === "number" && Number.isFinite(st.pollLatencyMs) ? Math.max(0, Math.round(st.pollLatencyMs)) : null;
	            const batchMs = typeof st.lastBatchMs === "number" && Number.isFinite(st.lastBatchMs) ? Math.max(0, Math.round(st.lastBatchMs)) : null;
	            const batchSize =
	              typeof st.lastBatchSize === "number" && Number.isFinite(st.lastBatchSize) ? Math.max(0, Math.round(st.lastBatchSize)) : null;
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
          const fetchedLast = st.fetchedLastKline;
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

	          const polledAtMs = typeof st.polledAtMs === "number" && Number.isFinite(st.polledAtMs) ? st.polledAtMs : null;
	          if (polledAtMs !== null && polledAtMs !== rt.lastTelemetryPolledAtMs) {
	            rt.lastTelemetryPolledAtMs = polledAtMs;
	            const pollLatencyMs = typeof st.pollLatencyMs === "number" && Number.isFinite(st.pollLatencyMs) ? st.pollLatencyMs : null;
	            const processedOpenTime = st.openTimes[st.openTimes.length - 1] ?? null;
	            const processedClose = st.prices[st.prices.length - 1] ?? null;
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

	          const openThr = st.openThreshold ?? st.threshold;
	          const closeThr = st.closeThreshold ?? st.openThreshold ?? st.threshold;
	          const tradeEnabled = st.settings?.tradeEnabled ?? null;

	          if (rt.lastMethod !== null && (st.method !== rt.lastMethod || openThr !== rt.lastOpenThreshold || closeThr !== rt.lastCloseThreshold)) {
	            const msg =
	              `params: ${methodLabel(st.method)}` +
	              ` • open ${fmtPct(openThr, 3)}` +
	              ` • close ${fmtPct(closeThr, 3)}` +
	              (typeof tradeEnabled === "boolean" ? ` • trade ${tradeEnabled ? "ON" : "OFF"}` : "");
	            feed = [{ atMs: finishedAtMs, message: msg }, ...feed].slice(0, 50);
	          }

	          if (typeof tradeEnabled === "boolean" && rt.lastTradeEnabled !== null && tradeEnabled !== rt.lastTradeEnabled) {
	            feed = [{ atMs: finishedAtMs, message: `trade ${tradeEnabled ? "enabled" : "disabled"}` }, ...feed].slice(0, 50);
	          }

	          const err = st.error ?? null;
	          if (err && err !== rt.lastError) {
	            feed = [{ atMs: finishedAtMs, message: `error: ${err}` }, ...feed].slice(0, 50);
	          }

          if (rt.lastHalted !== null && rt.lastHalted !== st.halted) {
            feed = [{ atMs: finishedAtMs, message: st.halted ? `halted: ${st.haltReason ?? "true"}` : "resumed" }, ...feed].slice(0, 50);
          }

	          rt.lastOpenTimeMs = lastOpen;
	          rt.lastError = err;
	          rt.lastHalted = st.halted;
	          rt.lastMethod = st.method;
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
        if (e instanceof HttpError && typeof e.payload === "string") {
          const payload = e.payload;
          if (payload.includes("ECONNREFUSED") || payload.includes("connect ECONNREFUSED")) {
            msg = `Backend unreachable. Start it with: cd haskell && cabal run -v0 trader-hs -- --serve --port ${API_PORT}`;
          }
        }
        if (e instanceof HttpError && (e.status === 502 || e.status === 503)) {
          msg = apiBase.startsWith("/api")
            ? "CloudFront `/api/*` proxy is unavailable (502/503). Point `/api/*` at your API origin (App Runner/ALB/etc) and allow POST/GET/OPTIONS, or set apiBaseUrl in trader-config.js to https://<your-api-host>."
            : "API gateway unavailable (502/503). Try again, or check the API logs.";
        }
        if (e instanceof HttpError && e.status === 504) {
          msg = apiBase.startsWith("/api")
            ? "CloudFront `/api/*` proxy timed out (504). Point `/api/*` at your API origin (App Runner/ALB/etc) and allow POST/OPTIONS, or set apiBaseUrl in trader-config.js to https://<your-api-host>."
            : "API gateway timed out (504). Try again, or reduce bars/epochs, or scale the API.";
        }
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
    [apiBase, authHeaders, botSelectedSymbol],
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
        if (isTimeoutError(err)) msg = "Bot start timed out. Try again.";
        if (err instanceof HttpError && err.status === 429) {
          const untilMs = applyRateLimit(err, { silent });
          msg = `Rate limited. Try again ${fmtEtaMs(Math.max(0, untilMs - Date.now()))}.`;
          showErrorToast = false;
        }
        if (err instanceof HttpError && err.status !== 429 && typeof err.payload === "string") {
          const payload = err.payload;
          if (payload.includes("ECONNREFUSED") || payload.includes("connect ECONNREFUSED")) {
            msg = `Backend unreachable. Start it with: cd haskell && cabal run -v0 trader-hs -- --serve --port ${API_PORT}`;
          }
        }
        if (err instanceof HttpError && err.status !== 429 && err.payload && typeof err.payload === "object") {
          try {
            let detail = JSON.stringify(err.payload, null, 2);
            if (detail.length > 2000) detail = `${detail.slice(0, 1997)}...`;
            if (detail !== "{}") msg = `${msg}\n${detail}`;
          } catch {
            // ignore
          }
        }
        if (err instanceof HttpError && (err.status === 502 || err.status === 503)) {
          msg = apiBase.startsWith("/api")
            ? "CloudFront `/api/*` proxy is unavailable (502/503). Point `/api/*` at your API origin (App Runner/ALB/etc) and allow POST/GET/OPTIONS, or set apiBaseUrl in trader-config.js to https://<your-api-host>."
            : "API gateway unavailable (502/503). Try again, or check the API logs.";
        }
        if (err instanceof HttpError && err.status === 504) {
          msg = apiBase.startsWith("/api")
            ? "CloudFront `/api/*` proxy timed out (504). Point `/api/*` at your API origin (App Runner/ALB/etc) and allow POST/OPTIONS, or set apiBaseUrl in trader-config.js to https://<your-api-host>."
            : "API gateway timed out (504). Try again, or reduce bars/epochs, or scale the API.";
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
        if (botSymbolsInput.length > 0) payload.botSymbols = botSymbolsInput;
        const out = await botStart(apiBase, withPlatformKeys(payload), { headers: authHeaders, timeoutMs: BOT_START_TIMEOUT_MS });
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
      botSymbolsInput,
      form.botMaxPoints,
      form.botAdoptExistingPosition,
      form.botOnlineEpochs,
      form.botPollSeconds,
      form.botTrainBars,
      form.tradeArmed,
      showToast,
      tradeParams,
      withPlatformKeys,
    ],
  );

  const stopLiveBot = useCallback(async (symbol?: string) => {
    setBot((s) => ({ ...s, loading: true, error: null }));
    try {
      const out = await botStop(apiBase, { headers: authHeaders, timeoutMs: 30_000 }, symbol);
      setBot((s) => ({ ...s, loading: false, error: null, status: out }));
      botAutoStartSuppressedRef.current = true;
      showToast(symbol ? `Bot stopped (${symbol})` : "Bot stopped");
    } catch (e) {
      if (isAbortError(e)) return;
      const msg = e instanceof Error ? e.message : String(e);
      setBot((s) => ({ ...s, loading: false, error: msg }));
      showToast("Bot stop failed");
    }
  }, [apiBase, authHeaders, showToast]);

  const binanceTradesSymbols = useMemo(() => parseSymbolsInput(binanceTradesSymbolsInput), [binanceTradesSymbolsInput]);
  const binanceTradesStartMs = useMemo(() => parseTimeInputMs(binanceTradesStartInput), [binanceTradesStartInput]);
  const binanceTradesEndMs = useMemo(() => parseTimeInputMs(binanceTradesEndInput), [binanceTradesEndInput]);
  const binanceTradesFromId = useMemo(() => parseMaybeInt(binanceTradesFromIdInput), [binanceTradesFromIdInput]);
  const binanceTradesLimitSafe = useMemo(
    () => clamp(Math.trunc(binanceTradesLimit), 1, 1000),
    [binanceTradesLimit],
  );
  const binanceTradesInputError = useMemo(
    () =>
      firstReason(
        !isBinancePlatform ? "Binance account trades require platform=binance." : null,
        form.market !== "futures" && binanceTradesSymbols.length === 0 ? "Symbol is required for spot/margin trades." : null,
        binanceTradesStartInput.trim() && binanceTradesStartMs === null
          ? "Start time must be a unix ms timestamp or ISO date."
          : null,
        binanceTradesEndInput.trim() && binanceTradesEndMs === null ? "End time must be a unix ms timestamp or ISO date." : null,
        binanceTradesStartMs !== null && binanceTradesEndMs !== null && binanceTradesEndMs < binanceTradesStartMs
          ? "End time must be after start time."
          : null,
        binanceTradesFromIdInput.trim() && binanceTradesFromId === null ? "From ID must be a number." : null,
      ),
    [
      binanceTradesEndInput,
      binanceTradesEndMs,
      binanceTradesFromId,
      binanceTradesFromIdInput,
      binanceTradesStartInput,
      binanceTradesStartMs,
      binanceTradesSymbols.length,
      form.market,
      isBinancePlatform,
    ],
  );

  const fetchBinanceTrades = useCallback(async () => {
    setBinanceTradesUi((s) => ({ ...s, loading: true, error: null }));
    if (binanceTradesInputError) {
      setBinanceTradesUi((s) => ({ ...s, loading: false, error: binanceTradesInputError }));
      return;
    }
    try {
      const params: ApiBinanceTradesRequest = {
        market: form.market,
        binanceTestnet: form.binanceTestnet,
        ...(binanceTradesSymbols.length === 1 ? { symbol: binanceTradesSymbols[0] } : {}),
        ...(binanceTradesSymbols.length > 1 ? { symbols: binanceTradesSymbols } : {}),
        ...(binanceTradesLimitSafe > 0 ? { limit: binanceTradesLimitSafe } : {}),
        ...(binanceTradesStartMs !== null ? { startTimeMs: binanceTradesStartMs } : {}),
        ...(binanceTradesEndMs !== null ? { endTimeMs: binanceTradesEndMs } : {}),
        ...(binanceTradesFromId !== null ? { fromId: binanceTradesFromId } : {}),
      };
      const out = await binanceTrades(apiBase, withBinanceKeys(params), { headers: authHeaders, timeoutMs: 30_000 });
      setBinanceTradesUi({ loading: false, error: null, response: out });
    } catch (e) {
      if (isAbortError(e)) return;
      const msg = e instanceof Error ? e.message : String(e);
      setBinanceTradesUi((s) => ({ ...s, loading: false, error: msg }));
    }
  }, [
    apiBase,
    authHeaders,
    binanceTradesEndMs,
    binanceTradesFromId,
    binanceTradesInputError,
    binanceTradesLimitSafe,
    binanceTradesStartMs,
    binanceTradesSymbols,
    form.binanceTestnet,
    form.market,
    withBinanceKeys,
  ]);

  useEffect(() => {
    if (apiOk !== "ok") return;
    void refreshBot({ silent: true });
  }, [apiOk, refreshBot]);

  useEffect(() => {
    if (apiOk !== "ok") return;
    const running = bot.status.running;
    const starting = "starting" in bot.status && bot.status.starting === true;
    if (!running && !starting) return;
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
    let inFlight = false;
    const fetchPayload = async (): Promise<{ payload: unknown; source: TopCombosSource; fallbackReason: string | null }> => {
      if (apiOk === "ok") {
        try {
          const url = `${apiBase.replace(/\/+$/, "")}/optimizer/combos`;
          const res = await fetch(url, { headers: authHeaders });
          if (res.ok) return { payload: await res.json(), source: "api", fallbackReason: null };
          throw new Error(`API error (HTTP ${res.status})`);
        } catch (err) {
          const fallbackReason = err instanceof Error ? err.message : "API unreachable";
          const res = await fetch("/top-combos.json");
          if (!res.ok) throw new Error(`Static combos unavailable (HTTP ${res.status})`);
          return { payload: await res.json(), source: "static", fallbackReason };
        }
      }
      const fallbackReason =
        apiOk === "auth"
          ? "API unauthorized"
          : apiOk === "down"
            ? "API unreachable"
            : apiOk === "unknown"
              ? "API unknown"
              : "API unavailable";
      const res = await fetch("/top-combos.json");
      if (!res.ok) throw new Error(`Static combos unavailable (HTTP ${res.status})`);
      return { payload: await res.json(), source: "static", fallbackReason };
    };
    const syncTopCombos = async (opts?: { silent?: boolean }) => {
      if (isCancelled || inFlight) return;
      inFlight = true;
      const silent = opts?.silent ?? false;
      if (!silent) setTopCombosLoading(true);
      try {
        const { payload, source, fallbackReason } = await fetchPayload();
        if (isCancelled) return;
        const payloadRec = (payload as Record<string, unknown> | null | undefined) ?? {};
        const rawCombos: unknown[] = Array.isArray(payloadRec.combos) ? (payloadRec.combos as unknown[]) : [];
        const methods: Method[] = ["11", "10", "01", "blend"];
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
          const rawPlatform = typeof params.platform === "string" ? params.platform : null;
          const platform =
            rawPlatform && PLATFORMS.includes(rawPlatform as Platform)
              ? (rawPlatform as Platform)
              : null;
          const interval = typeof params.interval === "string" && params.interval ? params.interval : defaultForm.interval;
          const bars = typeof params.bars === "number" && Number.isFinite(params.bars) ? Math.trunc(params.bars) : Math.trunc(defaultForm.bars);
          const positioning =
            typeof params.positioning === "string" && positionings.includes(params.positioning as Positioning)
              ? (params.positioning as Positioning)
              : null;
          const rawSymbol =
            typeof params.binanceSymbol === "string"
              ? params.binanceSymbol
              : typeof params.symbol === "string"
                ? params.symbol
                : "";
          const binanceSymbol = rawSymbol.trim().toUpperCase();
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
          const minHoldBars =
            typeof params.minHoldBars === "number" && Number.isFinite(params.minHoldBars)
              ? Math.max(0, Math.trunc(params.minHoldBars))
              : null;
          const maxHoldBars =
            typeof params.maxHoldBars === "number" && Number.isFinite(params.maxHoldBars)
              ? Math.max(0, Math.trunc(params.maxHoldBars))
              : null;
          const cooldownBars =
            typeof params.cooldownBars === "number" && Number.isFinite(params.cooldownBars)
              ? Math.max(0, Math.trunc(params.cooldownBars))
              : null;
          const minEdge =
            typeof params.minEdge === "number" && Number.isFinite(params.minEdge) ? Math.max(0, params.minEdge) : null;
          const costAwareEdge = typeof params.costAwareEdge === "boolean" ? params.costAwareEdge : null;
          const edgeBuffer =
            typeof params.edgeBuffer === "number" && Number.isFinite(params.edgeBuffer) ? Math.max(0, params.edgeBuffer) : null;
          const trendLookback =
            typeof params.trendLookback === "number" && Number.isFinite(params.trendLookback)
              ? Math.max(0, Math.trunc(params.trendLookback))
              : null;
          const maxPositionSize =
            typeof params.maxPositionSize === "number" && Number.isFinite(params.maxPositionSize)
              ? Math.max(0, params.maxPositionSize)
              : null;
          const volTarget =
            typeof params.volTarget === "number" && Number.isFinite(params.volTarget) ? Math.max(0, params.volTarget) : null;
          const volLookback =
            typeof params.volLookback === "number" && Number.isFinite(params.volLookback)
              ? Math.max(0, Math.trunc(params.volLookback))
              : null;
          const volEwmaAlphaRaw =
            typeof params.volEwmaAlpha === "number" && Number.isFinite(params.volEwmaAlpha) ? params.volEwmaAlpha : null;
          const volEwmaAlpha = volEwmaAlphaRaw != null && volEwmaAlphaRaw > 0 && volEwmaAlphaRaw < 1 ? volEwmaAlphaRaw : null;
          const volFloor =
            typeof params.volFloor === "number" && Number.isFinite(params.volFloor) ? Math.max(0, params.volFloor) : null;
          const volScaleMax =
            typeof params.volScaleMax === "number" && Number.isFinite(params.volScaleMax) ? Math.max(0, params.volScaleMax) : null;
          const maxVolatility =
            typeof params.maxVolatility === "number" && Number.isFinite(params.maxVolatility)
              ? Math.max(0, params.maxVolatility)
              : null;
          const periodsPerYear =
            typeof params.periodsPerYear === "number" && Number.isFinite(params.periodsPerYear)
              ? Math.max(0, params.periodsPerYear)
              : null;
          const kalmanMarketTopN =
            typeof params.kalmanMarketTopN === "number" && Number.isFinite(params.kalmanMarketTopN)
              ? Math.max(0, Math.trunc(params.kalmanMarketTopN))
              : null;
          const walkForwardFolds =
            typeof params.walkForwardFolds === "number" && Number.isFinite(params.walkForwardFolds)
              ? Math.max(1, Math.trunc(params.walkForwardFolds))
              : null;
          const blendWeightRaw =
            typeof params.blendWeight === "number" && Number.isFinite(params.blendWeight) ? params.blendWeight : null;
          const blendWeight = blendWeightRaw != null ? clamp(blendWeightRaw, 0, 1) : null;
          const tuneStressVolMult =
            typeof params.tuneStressVolMult === "number" && Number.isFinite(params.tuneStressVolMult)
              ? params.tuneStressVolMult
              : null;
          const tuneStressShock =
            typeof params.tuneStressShock === "number" && Number.isFinite(params.tuneStressShock)
              ? params.tuneStressShock
              : null;
          const tuneStressWeight =
            typeof params.tuneStressWeight === "number" && Number.isFinite(params.tuneStressWeight)
              ? params.tuneStressWeight
              : null;
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
          const operationsRaw = Array.isArray(rawRec.operations) ? rawRec.operations : [];
          const operations = operationsRaw
            .map((rawOp) => {
              const opRec = (rawOp as Record<string, unknown> | null | undefined) ?? {};
              const entryIndex =
                typeof opRec.entryIndex === "number" && Number.isFinite(opRec.entryIndex) ? Math.trunc(opRec.entryIndex) : null;
              const exitIndex =
                typeof opRec.exitIndex === "number" && Number.isFinite(opRec.exitIndex) ? Math.trunc(opRec.exitIndex) : null;
              if (entryIndex == null || exitIndex == null) return null;
              const entryEquity =
                typeof opRec.entryEquity === "number" && Number.isFinite(opRec.entryEquity) ? (opRec.entryEquity as number) : null;
              const exitEquity =
                typeof opRec.exitEquity === "number" && Number.isFinite(opRec.exitEquity) ? (opRec.exitEquity as number) : null;
              const retValue = typeof opRec.return === "number" && Number.isFinite(opRec.return) ? (opRec.return as number) : null;
              const holdingPeriods =
                typeof opRec.holdingPeriods === "number" && Number.isFinite(opRec.holdingPeriods)
                  ? Math.trunc(opRec.holdingPeriods as number)
                  : null;
              const exitReason =
                typeof opRec.exitReason === "string" && opRec.exitReason.trim() ? opRec.exitReason.trim() : null;
              const op: OptimizationComboOperation = {
                entryIndex,
                exitIndex,
                entryEquity,
                exitEquity,
                return: retValue,
                holdingPeriods,
                exitReason,
              };
              return op;
            })
            .filter((op): op is OptimizationComboOperation => op !== null);
          const operationsOut = operations.length > 0 ? operations : null;
          const rawSource = typeof rawRec.source === "string" ? rawRec.source : null;
          const source: OptimizationCombo["source"] =
            rawSource === "binance" || rawSource === "coinbase" || rawSource === "kraken" || rawSource === "poloniex" || rawSource === "csv"
              ? rawSource
              : null;
          const resolvedPlatform =
            platform ?? (source && source !== "csv" ? (source as Platform) : null);
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
            operations: operationsOut,
            params: {
              platform: resolvedPlatform,
              interval,
              bars,
              method,
              positioning,
              normalization,
              binanceSymbol: binanceSymbol ? binanceSymbol : null,
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
              minHoldBars,
              maxHoldBars,
              cooldownBars,
              minEdge,
              minSignalToNoise:
                typeof params.minSignalToNoise === "number" && Number.isFinite(params.minSignalToNoise)
                  ? Math.max(0, params.minSignalToNoise)
                  : null,
              costAwareEdge,
              edgeBuffer,
              trendLookback,
              maxPositionSize,
              volTarget,
              volLookback,
              volEwmaAlpha,
              volFloor,
              volScaleMax,
              maxVolatility,
              periodsPerYear,
              blendWeight,
              walkForwardFolds,
              tuneStressVolMult,
              tuneStressShock,
              tuneStressWeight,
              stopLoss: typeof params.stopLoss === "number" && Number.isFinite(params.stopLoss) ? params.stopLoss : null,
              takeProfit: typeof params.takeProfit === "number" && Number.isFinite(params.takeProfit) ? params.takeProfit : null,
              trailingStop: typeof params.trailingStop === "number" && Number.isFinite(params.trailingStop) ? params.trailingStop : null,
              stopLossVolMult:
                typeof params.stopLossVolMult === "number" && Number.isFinite(params.stopLossVolMult)
                  ? Math.max(0, params.stopLossVolMult)
                  : null,
              takeProfitVolMult:
                typeof params.takeProfitVolMult === "number" && Number.isFinite(params.takeProfitVolMult)
                  ? Math.max(0, params.takeProfitVolMult)
                  : null,
              trailingStopVolMult:
                typeof params.trailingStopVolMult === "number" && Number.isFinite(params.trailingStopVolMult)
                  ? Math.max(0, params.trailingStopVolMult)
                  : null,
              maxDrawdown: typeof params.maxDrawdown === "number" && Number.isFinite(params.maxDrawdown) ? params.maxDrawdown : null,
              maxDailyLoss: typeof params.maxDailyLoss === "number" && Number.isFinite(params.maxDailyLoss) ? params.maxDailyLoss : null,
              maxOrderErrors:
                typeof params.maxOrderErrors === "number" && Number.isFinite(params.maxOrderErrors) ? Math.max(1, Math.trunc(params.maxOrderErrors)) : null,
              kalmanZMin,
              kalmanZMax,
              kalmanMarketTopN,
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
        const limited = sanitized.slice(0, TOP_COMBOS_DISPLAY_MAX);
        setTopCombos(limited);
        const comboCount = rawCombos.length;
        const generatedAtMsRaw = payloadRec.generatedAtMs;
        const generatedAtMs =
          typeof generatedAtMsRaw === "number" && Number.isFinite(generatedAtMsRaw) ? Math.trunc(generatedAtMsRaw) : null;
        const payloadSourceRaw =
          typeof payloadRec.payloadSource === "string" && payloadRec.payloadSource.trim()
            ? payloadRec.payloadSource.trim()
            : typeof payloadRec.source === "string" && payloadRec.source.trim()
              ? payloadRec.source.trim()
              : null;
        const payloadSourcesRaw = Array.isArray(payloadRec.payloadSources) ? payloadRec.payloadSources : [];
        const payloadSources = payloadSourcesRaw
          .map((src) => (typeof src === "string" ? src.trim() : ""))
          .filter((src) => src.length > 0);
        const payloadSourcesFinal = payloadSources.length > 0 ? payloadSources : payloadSourceRaw ? [payloadSourceRaw] : null;
        setTopCombosMeta({
          source,
          generatedAtMs,
          payloadSource: payloadSourceRaw,
          payloadSources: payloadSourcesFinal,
          fallbackReason,
          comboCount,
        });
        setTopCombosError(null);
        const topCombo = limited[0];
        if (topCombo) {
          const currentForm = formRef.current;
          const topSig = comboApplySignature(
            topCombo,
            apiComputeLimitsRef.current,
            currentForm,
            manualOverridesRef.current,
            true,
          );
          const formSig = formApplySignature(currentForm);
          if (topSig !== formSig) {
            applyCombo(topCombo, { silent: true, respectManual: true, allowPositioning: true });
            const now = Date.now();
            setAutoAppliedCombo({ id: topCombo.id, atMs: now });
            const prev = autoAppliedComboRef.current;
            const shouldToast = !prev.atMs || prev.id !== topCombo.id || now - prev.atMs > 120_000;
            autoAppliedComboRef.current = { id: topCombo.id, atMs: now };
            if (shouldToast) {
              showToast(`Auto-applied top combo #${topCombo.id} (manual overrides respected)`);
            }
          }
        }
      } catch (err) {
        if (isCancelled) return;
        const msg = err instanceof Error ? err.message : "Failed to load optimizer combos.";
        if (!silent || topCombosRef.current.length === 0) {
          setTopCombosError(msg);
        }
      } finally {
        inFlight = false;
        if (isCancelled) return;
        if (!silent) setTopCombosLoading(false);
      }
    };

    topCombosSyncRef.current = syncTopCombos;
    void syncTopCombos();
    const t = window.setInterval(() => {
      void syncTopCombos({ silent: true });
    }, TOP_COMBOS_POLL_MS);
    return () => {
      isCancelled = true;
      window.clearInterval(t);
      if (topCombosSyncRef.current === syncTopCombos) {
        topCombosSyncRef.current = null;
      }
    };
  }, [apiBase, apiOk, authHeaders, applyCombo, showToast]);

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
  const methodOverride = manualOverrides.has("method");
  const openThresholdOverride = manualOverrides.has("openThreshold");
  const closeThresholdOverride = manualOverrides.has("closeThreshold");
  const manualOverrideLabels = [
    ...(methodOverride ? ["method"] : []),
    ...(openThresholdOverride ? ["open threshold"] : []),
    ...(closeThresholdOverride ? ["close threshold"] : []),
  ];
  const thresholdOverrideKeys: ManualOverrideKey[] = [];
  if (openThresholdOverride) thresholdOverrideKeys.push("openThreshold");
  if (closeThresholdOverride) thresholdOverrideKeys.push("closeThreshold");
  const autoAppliedAge =
    autoAppliedCombo && autoAppliedCombo.atMs ? fmtDurationMs(Math.max(0, Date.now() - autoAppliedCombo.atMs)) : null;
  const topCombo = topCombos.length > 0 ? topCombos[0] : null;
  const topComboSig = useMemo(() => {
    if (!topCombo) return null;
    return comboApplySignature(topCombo, apiComputeLimits, form, manualOverrides, true);
  }, [apiComputeLimits, form, manualOverrides, topCombo]);
  const botAutoStartReady = useMemo(() => {
    if (!topComboSig) return false;
    return topComboSig === formApplySignature(form);
  }, [form, topComboSig]);

  const missingSymbol = !form.binanceSymbol.trim();
  const intervalValue = form.interval.trim();
  const missingInterval = !intervalValue || !PLATFORM_INTERVAL_SET[platform].has(intervalValue);
  const lookbackState = useMemo(() => {
    const barsRaw = Math.trunc(form.bars);
    const bars = barsRaw <= 0 ? 0 : platform === "binance" ? clamp(barsRaw, 2, 1000) : Math.max(2, barsRaw);
    const interval = form.interval.trim();
    const intervalSec = platformIntervalSeconds(platform, interval);

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
  }, [form.bars, form.interval, form.lookbackBars, form.lookbackWindow, platform]);
  const dataLogFiltered = useMemo(() => {
    const term = dataLogFilterText.trim().toLowerCase();
    if (!term) return dataLog;
    return dataLog.filter((entry) => entry.label.toLowerCase().includes(term));
  }, [dataLog, dataLogFilterText]);
  const dataLogShown = useMemo(
    () => (dataLogFilterText.trim() ? dataLogFiltered : dataLog),
    [dataLog, dataLogFilterText, dataLogFiltered],
  );
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
  const apiStatusIssue = useMemo(() => {
    if (apiBaseError) return apiBaseError;
    if (apiOk === "down") return "API unreachable";
    if (apiOk === "auth") return apiToken.trim() ? "API auth failed" : "API auth required";
    return null;
  }, [apiBaseError, apiOk, apiToken]);
  const rateLimitEtaMs = rateLimit ? Math.max(0, rateLimit.untilMs - rateLimitTickMs) : null;
  const rateLimitReason =
    rateLimit && rateLimitEtaMs != null ? `${rateLimit.reason} Next retry ${fmtEtaMs(rateLimitEtaMs)}.` : rateLimit?.reason ?? null;

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
  const requestIssueInput = useMemo(
    () => ({
      rateLimitReason,
      apiStatusIssue,
      apiBlockedReason,
      apiTargetId: "section-api",
      missingSymbol,
      symbolTargetId: "symbol",
      missingInterval,
      intervalTargetId: "interval",
      lookbackError: lookbackState.error,
      lookbackTargetId: lookbackState.overrideOn ? "lookbackBars" : "lookbackWindow",
      apiLimitsReason,
      apiLimitsTargetId: barsExceedsApi ? "bars" : epochsExceedsApi ? "epochs" : hiddenSizeExceedsApi ? "hiddenSize" : undefined,
    }),
    [
      apiBlockedReason,
      apiLimitsReason,
      apiStatusIssue,
      barsExceedsApi,
      epochsExceedsApi,
      hiddenSizeExceedsApi,
      lookbackState.error,
      lookbackState.overrideOn,
      missingInterval,
      missingSymbol,
      rateLimitReason,
    ],
  );
  const requestIssueDetails = useMemo(() => buildRequestIssueDetails(requestIssueInput), [requestIssueInput]);
  const botIssueDetails = useMemo(
    () =>
      buildRequestIssueDetails({
        ...requestIssueInput,
        missingSymbol: botMissingSymbol,
        symbolTargetId: "botSymbols",
      }),
    [botMissingSymbol, requestIssueInput],
  );
  const requestIssues = useMemo(() => requestIssueDetails.map((issue) => issue.message), [requestIssueDetails]);
  const primaryIssue = requestIssueDetails[0] ?? null;
  const extraIssueCount = Math.max(0, requestIssueDetails.length - 1);
  const requestDisabledReason = primaryIssue?.disabledMessage ?? primaryIssue?.message ?? null;
  const requestDisabled = state.loading || Boolean(requestDisabledReason);
  const botAnyRunning = bot.status.running;
  const botAnyStarting = "starting" in bot.status && bot.status.starting === true;
  const botStartPrimaryIssue = botIssueDetails[0] ?? null;
  const botStartDisabledReason = botStartPrimaryIssue?.disabledMessage ?? botStartPrimaryIssue?.message ?? null;
  const botStarting = apiOk === "ok" && !bot.error && botAnyStarting;
  const botStartBlockedReason = firstReason(
    !isBinancePlatform ? "Live bot is supported on Binance only." : null,
    form.positioning === "long-short" && form.market !== "futures"
      ? "Live bot long/short requires the Futures market."
      : null,
    botStartDisabledReason,
    botStartSymbols.length > 0 && !botHasNewSymbols ? "All requested bot symbols are already running." : null,
  );
  const botStartBlocked = bot.loading || botStarting || Boolean(botStartBlockedReason);
  const longShortBotDisabled = botAnyRunning || botStarting;

  useEffect(() => {
    if (!botAutoStartReady) return;
    if (apiOk !== "ok") return;
    if (!botStatusFetchedRef.current) return;
    if (botAutoStartSuppressedRef.current) return;
    if (botAnyRunning) return;
    if (botStartBlocked) return;
    const now = Date.now();
    if (now - botAutoStartRef.current.lastAttemptAtMs < BOT_AUTOSTART_RETRY_MS) return;
    botAutoStartRef.current.lastAttemptAtMs = now;
    void startLiveBot({ auto: true, silent: true });
  }, [apiOk, botAnyRunning, botAutoStartReady, botStartBlocked, startLiveBot]);
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
  const tradeDisabledDetail = useMemo(() => {
    if (requestDisabledReason) {
      return { message: requestDisabledReason, targetId: requestIssueDetails[0]?.targetId };
    }
    if (!isBinancePlatform && !isCoinbasePlatform) {
      return { message: "Trading is supported on Binance and Coinbase only.", targetId: "platform" };
    }
    if (isCoinbasePlatform && form.positioning === "long-short") {
      return { message: "Coinbase supports spot only (positioning=long-flat).", targetId: "positioning" };
    }
    if (tradeOrderSizingError) {
      return { message: tradeOrderSizingError, targetId: "orderQuoteFraction" };
    }
    if (isBinancePlatform && form.positioning === "long-short" && form.market !== "futures") {
      return { message: "Long/Short trading requires Futures market.", targetId: "market" };
    }
    return null;
  }, [
    form.market,
    form.positioning,
    isBinancePlatform,
    isCoinbasePlatform,
    requestDisabledReason,
    requestIssueDetails,
    tradeOrderSizingError,
  ]);
  const tradeDisabledReason = tradeDisabledDetail?.message ?? null;

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
    const base = apiBaseAbsolute;
    return `curl -s -X POST ${base}${endpoint} -H 'Content-Type: application/json'${auth} -d '${safe}'`;
  }, [apiBaseAbsolute, apiToken, requestPreview, requestPreviewKind]);
  const jumpTargets = [
    { id: "section-api", label: "API" },
    { id: "section-market", label: "Market" },
    { id: "section-lookback", label: "Lookback" },
    { id: "section-thresholds", label: "Thresholds" },
    { id: "section-risk", label: "Risk" },
    { id: "section-optimization", label: "Optimization" },
    { id: "section-livebot", label: "Live bot" },
    { id: "section-trade", label: "Trade" },
  ];

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
            <div className="stickyActions">
              <div className="pillRow">
                <span className={`pill ${requestIssues.length ? "pillWarn" : "pillOk"}`}>
                  {requestIssues.length
                    ? `${requestIssues.length} issue${requestIssues.length === 1 ? "" : "s"} to fix`
                    : "Ready to run"}
                </span>
                {primaryIssue ? (
                  <>
                    <span className="pill pillWarn">{primaryIssue.message}</span>
                    {primaryIssue.targetId ? (
                      <button className="btnSmall" type="button" onClick={() => scrollToSection(primaryIssue.targetId)}>
                        Fix
                      </button>
                    ) : null}
                  </>
                ) : (
                  <span className="pill">All required inputs look good.</span>
                )}
                {extraIssueCount > 0 ? <span className="pill">+{extraIssueCount} more</span> : null}
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
              <div className="pillRow jumpRow">
                <span className="jumpLabel">Jump to</span>
                {jumpTargets.map((target) => (
                  <button
                    key={target.id}
                    className="btnSmall"
                    type="button"
                    onClick={() => scrollToSection(target.id)}
                  >
                    {target.label}
                  </button>
                ))}
              </div>
              {requestIssueDetails.length > 1 ? (
                <details className="details">
                  <summary>Show all issues</summary>
                  <div className="issueList" style={{ marginTop: 10 }}>
                    {requestIssueDetails.map((issue, idx) => (
                      <div key={`${issue.message}-${idx}`} className="issueItem">
                        <span>{issue.message}</span>
                        {issue.targetId ? (
                          <button className="btnSmall" type="button" onClick={() => scrollToSection(issue.targetId)}>
                            Jump
                          </button>
                        ) : null}
                      </div>
                    ))}
                  </div>
                </details>
              ) : null}

              {rateLimitReason ? (
                <div className="hint" style={{ marginTop: 6, color: "var(--warn)" }}>
                  {rateLimitReason}
                </div>
              ) : null}

              {state.loading ? (
                <div className="hint" style={{ marginTop: 6 }}>
                  {activeAsyncJob?.jobId
                    ? `Async job: ${activeAsyncJob.jobId} • ${activeAsyncJob.kind} • ${Math.max(
                        0,
                        Math.floor((activeAsyncTickMs - activeAsyncJob.startedAtMs) / 1000),
                      )}s`
                    : "Starting async job…"}
                </div>
              ) : null}
            </div>
            <div className="row" style={{ gridTemplateColumns: "1fr" }} id="section-api">
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
                <div className="actions" style={{ marginTop: 8 }}>
                  <button className="btnSmall" type="button" onClick={() => void copyText(apiBaseAbsolute)} disabled={Boolean(apiBaseError)}>
                    Copy base URL
                  </button>
                  <button
                    className="btnSmall"
                    type="button"
                    onClick={() => {
                      if (!apiHealthUrl) return;
                      if (typeof window !== "undefined") window.open(apiHealthUrl, "_blank", "noopener,noreferrer");
                    }}
                    disabled={Boolean(apiBaseError) || !apiHealthUrl}
                  >
                    Open /health
                  </button>
                </div>
                <div className="hint" style={{ marginTop: 6 }}>
                  Configured at deploy time via <span style={{ fontFamily: "var(--mono)" }}>trader-config.js</span> (apiBaseUrl, apiToken).
                </div>
                {apiBaseError ? (
                  <div className="hint" style={{ color: "rgba(239, 68, 68, 0.85)", marginTop: 6 }}>
                    {apiBaseError}
                  </div>
                ) : null}
                {apiBaseCorsHint ? (
                  <div className="hint" style={{ marginTop: 6 }}>
                    {apiBaseCorsHint}
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
                <label className="label">{platformKeyLabel}</label>
                <div
                  className="row"
                  style={{
                    gridTemplateColumns: platformKeyMode === "coinbase" ? "1fr 1fr 1fr auto auto" : "1fr 1fr auto auto",
                    alignItems: "center",
                  }}
                >
                  <input
                    className="input"
                    type={revealSecrets ? "text" : "password"}
                    value={platformKeyMode === "coinbase" ? coinbaseApiKey : platformKeyMode === "binance" ? binanceApiKey : ""}
                    onChange={(e) => {
                      const next = e.target.value;
                      if (platformKeyMode === "coinbase") setCoinbaseApiKey(next);
                      else if (platformKeyMode === "binance") setBinanceApiKey(next);
                    }}
                    placeholder={
                      platformKeyMode === "coinbase" ? "COINBASE_API_KEY" : platformKeyMode === "binance" ? "BINANCE_API_KEY" : "Select Binance/Coinbase"
                    }
                    spellCheck={false}
                    autoCapitalize="none"
                    autoCorrect="off"
                    inputMode="text"
                    disabled={!platformKeyMode}
                  />
                  <input
                    className="input"
                    type={revealSecrets ? "text" : "password"}
                    value={platformKeyMode === "coinbase" ? coinbaseApiSecret : platformKeyMode === "binance" ? binanceApiSecret : ""}
                    onChange={(e) => {
                      const next = e.target.value;
                      if (platformKeyMode === "coinbase") setCoinbaseApiSecret(next);
                      else if (platformKeyMode === "binance") setBinanceApiSecret(next);
                    }}
                    placeholder={
                      platformKeyMode === "coinbase" ? "COINBASE_API_SECRET" : platformKeyMode === "binance" ? "BINANCE_API_SECRET" : "Select Binance/Coinbase"
                    }
                    spellCheck={false}
                    autoCapitalize="none"
                    autoCorrect="off"
                    inputMode="text"
                    disabled={!platformKeyMode}
                  />
                  {platformKeyMode === "coinbase" ? (
                    <input
                      className="input"
                      type={revealSecrets ? "text" : "password"}
                      value={coinbaseApiPassphrase}
                      onChange={(e) => setCoinbaseApiPassphrase(e.target.value)}
                      placeholder="COINBASE_API_PASSPHRASE"
                      spellCheck={false}
                      autoCapitalize="none"
                      autoCorrect="off"
                      inputMode="text"
                    />
                  ) : null}
                  <button className="btn" type="button" onClick={() => setRevealSecrets((v) => !v)}>
                    {revealSecrets ? "Hide" : "Show"}
                  </button>
                  <button
                    className="btn"
                    type="button"
                    onClick={() => {
                      if (platformKeyMode === "coinbase") {
                        setCoinbaseApiKey("");
                        setCoinbaseApiSecret("");
                        setCoinbaseApiPassphrase("");
                      } else if (platformKeyMode === "binance") {
                        setBinanceApiKey("");
                        setBinanceApiSecret("");
                      }
                    }}
                    disabled={!platformKeyHasValues}
                  >
                    Clear
                  </button>
                </div>
                <div className="hint">
                  {platformKeyHint} Stored in {persistSecrets ? "local storage" : "session storage"}. The request preview/curl omits it.
                </div>
                {!platformKeyMode ? (
                  <div className="hint" style={{ color: "rgba(245, 158, 11, 0.9)" }}>
                    Keys are only editable when Platform is set to Binance or Coinbase.
                  </div>
                ) : null}
                <div className="pillRow" style={{ marginTop: 10 }}>
                  <label className="pill">
                    <input type="checkbox" checked={persistSecrets} onChange={(e) => setPersistSecrets(e.target.checked)} />
                    Remember platform keys
                  </label>
                </div>
                <div className="hint">
                  When enabled, the platform keys are stored in local storage so you can reopen the app later without re-entering them (not recommended on shared
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
                <div className="hint">Save/load named config presets. Does not include API keys.</div>

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
                          clearManualOverrides();
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
              {(() => {
                const updatedAtMs = topCombosMeta.generatedAtMs;
                const updatedLabel = updatedAtMs ? fmtTimeMs(updatedAtMs) : "—";
                const ageLabel = updatedAtMs ? fmtDurationMs(Math.max(0, Date.now() - updatedAtMs)) : null;
                const sourceLabel = topCombosMeta.source === "api" ? "Source: API" : "Source: static file";
                const reason = topCombosMeta.source === "api" ? null : topCombosMeta.fallbackReason;
                const payloadSources = topCombosMeta.payloadSources;
                const payloadSource = topCombosMeta.payloadSource;
                const payloadLabel =
                  payloadSources && payloadSources.length > 0
                    ? ` • payload ${payloadSources.join(" + ")}`
                    : payloadSource
                      ? ` • payload ${payloadSource}`
                      : "";
                const displayCount = topCombos.length;
                const comboCount = topCombosMeta.comboCount;
                const countLabel =
                  comboCount != null && comboCount > displayCount
                    ? `Showing ${displayCount} of ${comboCount} combos`
                    : `Showing ${displayCount} combo${displayCount === 1 ? "" : "s"}`;
                return (
                  <div style={{ marginBottom: 8 }}>
                    <div className="hint">
                      {sourceLabel}
                      {reason ? ` (${reason})` : ""}
                      {payloadLabel}
                    </div>
                    <div className="hint">
                      Last updated {updatedLabel}
                      {ageLabel ? ` (${ageLabel} ago)` : ""}
                      {" • "}
                      {countLabel}
                    </div>
                  </div>
                );
              })()}
              <div className="pillRow" style={{ marginBottom: 8 }}>
                {autoAppliedCombo ? (
                  <span className="pill">
                    Auto-applied #{autoAppliedCombo.id}
                    {autoAppliedAge ? ` (${autoAppliedAge} ago)` : ""}
                  </span>
                ) : null}
                {manualOverrideLabels.length > 0 ? (
                  <>
                    <span
                      className="pill"
                      style={{ color: "rgba(245, 158, 11, 0.9)", borderColor: "rgba(245, 158, 11, 0.35)" }}
                    >
                      Manual overrides: {manualOverrideLabels.join(", ")}
                    </span>
                    <button className="btnSmall" type="button" onClick={() => clearManualOverrides()}>
                      Unlock overrides
                    </button>
                  </>
                ) : null}
              </div>
              <div className="actions" style={{ marginBottom: 8 }}>
                <button className="btnSmall" type="button" onClick={refreshTopCombos} disabled={topCombosLoading}>
                  {topCombosLoading ? "Refreshing…" : "Refresh combos now"}
                </button>
                <button
                  className="btnSmall"
                  type="button"
                  onClick={() => {
                    if (topCombo) handleComboApply(topCombo);
                  }}
                  disabled={!topCombo}
                >
                  Apply top combo now
                </button>
              </div>
              <TopCombosChart
                combos={topCombos}
                loading={topCombosLoading}
                error={topCombosError}
                selectedId={selectedComboId}
                onSelect={handleComboPreview}
                onApply={handleComboApply}
              />
              <div className="hint">
                Select a combo to preview. Click Apply to load params into the form (and the symbol, when provided). bars=0 uses all CSV data or the exchange default (500).
              </div>
              <div className="hint">
                Top combos auto-apply when available (manual overrides respected). If the bot is idle, it will auto-start once the top combo is applied.
              </div>
            </div>
          </div>

          <div className="sectionHeading" id="section-market">
            <span className="sectionTitle">Market</span>
            <span className="sectionMeta">Pair, market type, interval, bars.</span>
          </div>
          <div className="row rowSingle">
            <div className="field">
              <label className="label" htmlFor="platform">
                Platform
              </label>
              <select
                id="platform"
                className="select"
                value={form.platform}
                onChange={(e) => {
                  const next = e.target.value as Platform;
                  setPendingMarket(null);
                  setPendingProfileLoad(null);
                  setForm((f) => {
                    const symbolSet = PLATFORM_SYMBOL_SET[next];
                    const fallback = customSymbolByPlatform[next] || PLATFORM_DEFAULT_SYMBOL[next];
                    const normalized = f.binanceSymbol.trim().toUpperCase();
                    const nextSymbol = symbolSet.has(normalized) ? normalized : fallback;
                    if (next === "binance") return { ...f, platform: next, binanceSymbol: nextSymbol };
                    return {
                      ...f,
                      platform: next,
                      binanceSymbol: nextSymbol,
                      market: "spot",
                      binanceTestnet: false,
                      binanceLive: false,
                      tradeArmed: false,
                    };
                  });
                }}
              >
                {PLATFORMS.map((entry) => (
                  <option key={entry} value={entry}>
                    {PLATFORM_LABELS[entry]}
                  </option>
                ))}
              </select>
              <div className="hint">
                Exchange platform for price data. Trading supports Binance + Coinbase (spot only); live bot is Binance-only. API keys can be stored per platform.
              </div>
            </div>
          </div>
          <div className="row">
            <div className="field">
              <label className="label" htmlFor="symbol">
                Trading pair
              </label>
              <select
                id="symbol"
                className={missingSymbol ? "select selectError" : "select"}
                value={symbolSelectValue}
                onChange={(e) => {
                  const next = e.target.value;
                  if (next === CUSTOM_SYMBOL_VALUE) {
                    const fallback = customSymbolByPlatform[platform] || "";
                    setForm((f) => ({ ...f, binanceSymbol: fallback }));
                    return;
                  }
                  setForm((f) => ({ ...f, binanceSymbol: next }));
                }}
              >
                {platformSymbols.map((symbol) => (
                  <option key={symbol} value={symbol}>
                    {symbol}
                  </option>
                ))}
                <option value={CUSTOM_SYMBOL_VALUE}>Custom...</option>
              </select>
              {symbolIsCustom ? (
                <input
                  id="symbolCustom"
                  className={missingSymbol ? "input inputError" : "input"}
                  value={form.binanceSymbol}
                  onChange={(e) => {
                    const next = e.target.value.toUpperCase();
                    setCustomSymbolByPlatform((prev) => ({ ...prev, [platform]: next }));
                    setForm((f) => ({ ...f, binanceSymbol: next }));
                  }}
                  placeholder={PLATFORM_DEFAULT_SYMBOL[platform]}
                  spellCheck={false}
                  aria-label="Custom trading pair"
                />
              ) : null}
              <div className="hint" style={missingSymbol ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
                {missingSymbol
                  ? "Required."
                  : symbolIsCustom
                    ? isCoinbasePlatform
                      ? "Type any Coinbase product ID (e.g., BTC-USD)."
                      : `Type any ${platformLabel} symbol.`
                    : `Pick a common ${platformLabel} pair or choose Custom to type another symbol.`}
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
                disabled={!isBinancePlatform}
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
              <div className="hint">
                {isBinancePlatform
                  ? "Margin orders require live mode. Futures can close positions via reduce-only."
                  : "Market selection applies to Binance only."}
              </div>
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
                {platformIntervals.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
              <div className="hint" style={missingInterval ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
                {missingInterval ? "Required." : `${platformLabel} intervals: ${platformIntervals.join(", ")}.`}
              </div>
            </div>
            <div className="field">
              <label className="label" htmlFor="bars">
                Bars (0=auto, {isBinancePlatform ? "2–1000" : ">=2"})
              </label>
              <input
                id="bars"
                className={barsExceedsApi ? "input inputError" : "input"}
                type="number"
                min={0}
                max={isBinancePlatform ? 1000 : undefined}
                value={form.bars}
                onChange={(e) => setForm((f) => ({ ...f, bars: numFromInput(e.target.value, f.bars) }))}
              />
              <div className="hint" style={barsExceedsApi ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
                {barsExceedsApi
                  ? `API limit: max ${apiComputeLimits?.maxBarsLstm ?? "?"} bars for LSTM methods. Reduce bars or use method=10 (Kalman-only).`
                  : isBinancePlatform
                    ? "0=auto (Binance uses 500; CSV uses all). For Binance, 2–1000 is allowed. Larger values take longer."
                    : `0=auto (exchange default 500; CSV uses all). ${platformLabel} requires at least 2 bars. Larger values take longer.`
                }
              </div>
            </div>
          </div>

            <div className="sectionHeading" id="section-lookback">
              <span className="sectionTitle">Lookback</span>
              <span className="sectionMeta">Window length and bar overrides.</span>
            </div>
            <div className="row">
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

            <div className="sectionHeading" id="section-thresholds">
              <span className="sectionTitle">Thresholds</span>
              <span className="sectionMeta">Method, positioning, entry/exit gates.</span>
            </div>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
              <div className="field">
                <label className="label" htmlFor="method">
                  Method
                </label>
                <select
                  id="method"
                  className="select"
                  value={form.method}
                  onChange={(e) => {
                    markManualOverrides(["method"]);
                    setForm((f) => ({ ...f, method: e.target.value as Method }));
                  }}
                >
                  <option value="11">11 — Both (agreement gated)</option>
                  <option value="blend">blend — Weighted average</option>
                  <option value="10">10 — Kalman only</option>
                  <option value="01">01 — LSTM only</option>
                </select>
                <div className="hint">
                  “11” only trades when both models agree on direction (up/down) outside the open threshold. “blend” averages the two predictions.
                </div>
                {methodOverride ? (
                  <div className="pillRow" style={{ marginTop: 6 }}>
                    <span className="pill" style={{ color: "rgba(245, 158, 11, 0.9)", borderColor: "rgba(245, 158, 11, 0.35)" }}>
                      Manual override active
                    </span>
                    <button className="btnSmall" type="button" onClick={() => clearManualOverrides(["method"])}>
                      Unlock method
                    </button>
                  </div>
                ) : null}
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
                  <option value="long-short" disabled={longShortBotDisabled}>
                    Long / Short (futures)
                  </option>
                </select>
                <div
                  className="hint"
                  style={
                    form.positioning === "long-short" && form.market !== "futures"
                      ? { color: "rgba(245, 158, 11, 0.9)" }
                      : undefined
                  }
                >
                  {form.positioning === "long-short"
                    ? `${form.market !== "futures" ? "Long/Short trading requires Futures market when trading is armed. " : ""}Live bot supports Long/Short on futures.`
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
                  onChange={(e) => {
                    markManualOverrides(["openThreshold"]);
                    setForm((f) => ({ ...f, openThreshold: numFromInput(e.target.value, f.openThreshold) }));
                  }}
                />
                <div className="hint">
                  Entry deadband. Default 0.001 = 0.1%. Break-even ≈ {fmtPct(estimatedCosts.breakEven, 3)} (round-trip cost ≈ {fmtPct(estimatedCosts.roundTrip, 3)}).
                  {estimatedCosts.breakEven > 0 && form.openThreshold < estimatedCosts.breakEven
                    ? " Consider increasing open threshold to avoid churn after costs."
                    : null}
                </div>
                {thresholdOverrideKeys.length > 0 ? (
                  <div className="pillRow" style={{ marginTop: 8 }}>
                    <span className="pill" style={{ color: "rgba(245, 158, 11, 0.9)", borderColor: "rgba(245, 158, 11, 0.35)" }}>
                      Manual override: {thresholdOverrideKeys.includes("openThreshold") ? "open" : ""}
                      {thresholdOverrideKeys.includes("openThreshold") && thresholdOverrideKeys.includes("closeThreshold") ? " + " : ""}
                      {thresholdOverrideKeys.includes("closeThreshold") ? "close" : ""} threshold
                      {thresholdOverrideKeys.length > 1 ? "s" : ""}
                    </span>
                    <button className="btnSmall" type="button" onClick={() => clearManualOverrides(thresholdOverrideKeys)}>
                      Unlock thresholds
                    </button>
                  </div>
                ) : null}
                <div className="pillRow" style={{ marginTop: 10 }}>
                  <button
                    className="btnSmall"
                    type="button"
                    disabled={!(estimatedCosts.breakEven > 0)}
                    onClick={() => {
                      const be = estimatedCosts.breakEven;
                      const open = Number((be * 2).toFixed(6));
                      const close = Number(be.toFixed(6));
                      markManualOverrides(["openThreshold", "closeThreshold"]);
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
                      markManualOverrides(["openThreshold", "closeThreshold"]);
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
                      markManualOverrides(["openThreshold", "closeThreshold"]);
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
                  onChange={(e) => {
                    markManualOverrides(["closeThreshold"]);
                    setForm((f) => ({ ...f, closeThreshold: numFromInput(e.target.value, f.closeThreshold) }));
                  }}
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

            <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
              <div className="field">
                <label className="label" htmlFor="minEdge">
                  Min edge (fraction)
                </label>
                <input
                  id="minEdge"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={form.minEdge}
                  onChange={(e) => setForm((f) => ({ ...f, minEdge: numFromInput(e.target.value, f.minEdge) }))}
                />
                <div className="hint">
                  {form.costAwareEdge
                    ? `Cost-aware min edge ≈ ${fmtPct(minEdgeEffective, 3)} (break-even ${fmtPct(estimatedCosts.breakEven, 3)} + buffer).`
                    : "Minimum predicted return to enter. 0 disables."}
                </div>
                <div className="pillRow" style={{ marginTop: 8 }}>
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.costAwareEdge}
                      onChange={(e) => setForm((f) => ({ ...f, costAwareEdge: e.target.checked }))}
                    />
                    Cost-aware edge
                  </label>
                </div>
              </div>
              <div className="field">
                <label className="label" htmlFor="edgeBuffer">
                  Edge buffer (fraction)
                </label>
                <input
                  id="edgeBuffer"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={form.edgeBuffer}
                  onChange={(e) => setForm((f) => ({ ...f, edgeBuffer: numFromInput(e.target.value, f.edgeBuffer) }))}
                />
                <div className="hint">{form.costAwareEdge ? "Extra buffer above break-even." : "Used when cost-aware edge is enabled."}</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="minSignalToNoise">
                  Min signal/vol (x)
                </label>
                <input
                  id="minSignalToNoise"
                  className="input"
                  type="number"
                  step="0.05"
                  min={0}
                  value={form.minSignalToNoise}
                  onChange={(e) => setForm((f) => ({ ...f, minSignalToNoise: numFromInput(e.target.value, f.minSignalToNoise) }))}
                  placeholder="0"
                />
                <div className="hint">
                  {form.minSignalToNoise > 0 ? `${form.minSignalToNoise.toFixed(2)}x sigma` : "0 disables"} - edge / per-bar vol filter.
                </div>
              </div>
              <div className="field">
                <label className="label" htmlFor="blendWeight">
                  Blend weight (Kalman)
                </label>
                <input
                  id="blendWeight"
                  className="input"
                  type="number"
                  step="0.05"
                  min={0}
                  max={1}
                  value={form.blendWeight}
                  onChange={(e) => setForm((f) => ({ ...f, blendWeight: numFromInput(e.target.value, f.blendWeight) }))}
                  disabled={form.method !== "blend"}
                />
                <div className="hint">0 = LSTM only, 1 = Kalman only. Used with method=blend.</div>
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

            <div className="sectionHeading" id="section-risk">
              <span className="sectionTitle">Risk</span>
              <span className="sectionMeta">Stops, pacing, sizing, and kill-switches.</span>
            </div>
            <div className="row" style={{ gridTemplateColumns: "1fr" }}>
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
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr", marginTop: 10 }}>
                  <div className="field">
                    <label className="label" htmlFor="stopLossVolMult">
                      Stop-loss vol mult
                    </label>
                    <input
                      id="stopLossVolMult"
                      className="input"
                      type="number"
                      step="0.05"
                      min={0}
                      value={form.stopLossVolMult}
                      onChange={(e) => setForm((f) => ({ ...f, stopLossVolMult: numFromInput(e.target.value, f.stopLossVolMult) }))}
                      placeholder="0"
                    />
                    <div className="hint">{form.stopLossVolMult > 0 ? `${form.stopLossVolMult.toFixed(2)}x sigma` : "0 disables"}</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="takeProfitVolMult">
                      Take-profit vol mult
                    </label>
                    <input
                      id="takeProfitVolMult"
                      className="input"
                      type="number"
                      step="0.05"
                      min={0}
                      value={form.takeProfitVolMult}
                      onChange={(e) => setForm((f) => ({ ...f, takeProfitVolMult: numFromInput(e.target.value, f.takeProfitVolMult) }))}
                      placeholder="0"
                    />
                    <div className="hint">{form.takeProfitVolMult > 0 ? `${form.takeProfitVolMult.toFixed(2)}x sigma` : "0 disables"}</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="trailingStopVolMult">
                      Trailing vol mult
                    </label>
                    <input
                      id="trailingStopVolMult"
                      className="input"
                      type="number"
                      step="0.05"
                      min={0}
                      value={form.trailingStopVolMult}
                      onChange={(e) => setForm((f) => ({ ...f, trailingStopVolMult: numFromInput(e.target.value, f.trailingStopVolMult) }))}
                      placeholder="0"
                    />
                    <div className="hint">
                      {form.trailingStopVolMult > 0 ? `${form.trailingStopVolMult.toFixed(2)}x sigma` : "0 disables"}
                    </div>
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
                <div className="hint">
                  Optional bracket exits (uses OHLC high/low when available; otherwise close-only). Vol multiples use per-bar sigma and override fixed fractions when available.
                </div>
	              </div>
	            </div>

            <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr" }}>
              <div className="field">
                <label className="label">Trade pacing (bars)</label>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
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
                  <div className="field">
                    <label className="label" htmlFor="maxHoldBars">
                      Max hold
                    </label>
                    <input
                      id="maxHoldBars"
                      className="input"
                      type="number"
                      step={1}
                      min={0}
                      value={form.maxHoldBars}
                      onChange={(e) => setForm((f) => ({ ...f, maxHoldBars: numFromInput(e.target.value, f.maxHoldBars) }))}
                      placeholder="0"
                    />
                    <div className="hint">
                      {form.maxHoldBars > 0 ? `${Math.trunc(Math.max(0, form.maxHoldBars))} bars` : "0 disables"} • forces exit (MAX_HOLD)
                    </div>
                  </div>
                </div>
                <div className="hint">
                  Helps reduce churn in noisy markets (applies to backtests + live bot; stateless signals/trades ignore state).
                </div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr" }}>
              <div className="field">
                <label className="label">Sizing + filters</label>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
                  <div className="field">
                    <label className="label" htmlFor="trendLookback">
                      Trend lookback (SMA bars)
                    </label>
                    <input
                      id="trendLookback"
                      className="input"
                      type="number"
                      step={1}
                      min={0}
                      value={form.trendLookback}
                      onChange={(e) => setForm((f) => ({ ...f, trendLookback: numFromInput(e.target.value, f.trendLookback) }))}
                      placeholder="0"
                    />
                    <div className="hint">
                      {form.trendLookback > 0 ? `${Math.trunc(Math.max(0, form.trendLookback))} bars` : "0 disables"} • filters entries to SMA trend
                    </div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="maxPositionSize">
                      Max position size
                    </label>
                    <input
                      id="maxPositionSize"
                      className="input"
                      type="number"
                      step="0.1"
                      min={0}
                      value={form.maxPositionSize}
                      onChange={(e) => setForm((f) => ({ ...f, maxPositionSize: numFromInput(e.target.value, f.maxPositionSize) }))}
                      placeholder="1"
                    />
                    <div className="hint">Caps size/leverage (1 = full size).</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="maxVolatility">
                      Max volatility (annualized)
                    </label>
                    <input
                      id="maxVolatility"
                      className="input"
                      type="number"
                      step="0.01"
                      min={0}
                      value={form.maxVolatility}
                      onChange={(e) => setForm((f) => ({ ...f, maxVolatility: numFromInput(e.target.value, f.maxVolatility) }))}
                      placeholder="0"
                    />
                    <div className="hint">
                      {form.maxVolatility > 0 ? fmtPct(form.maxVolatility, 2) : "0 disables"} • blocks entries when vol is too high
                    </div>
                  </div>
                </div>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr 1fr", marginTop: 10 }}>
                  <div className="field">
                    <label className="label" htmlFor="volTarget">
                      Vol target
                    </label>
                    <input
                      id="volTarget"
                      className="input"
                      type="number"
                      step="0.01"
                      min={0}
                      value={form.volTarget}
                      onChange={(e) => setForm((f) => ({ ...f, volTarget: numFromInput(e.target.value, f.volTarget) }))}
                      placeholder="0"
                    />
                    <div className="hint">{form.volTarget > 0 ? fmtPct(form.volTarget, 2) : "0 disables"} • annualized</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="volLookback">
                      Vol lookback
                    </label>
                    <input
                      id="volLookback"
                      className="input"
                      type="number"
                      step={1}
                      min={0}
                      value={form.volLookback}
                      onChange={(e) => setForm((f) => ({ ...f, volLookback: numFromInput(e.target.value, f.volLookback) }))}
                      placeholder="20"
                    />
                    <div
                      className="hint"
                      style={
                        form.volTarget > 0 && !(form.volEwmaAlpha > 0 && form.volEwmaAlpha < 1) && form.volLookback < 2
                          ? { color: "rgba(239, 68, 68, 0.85)" }
                          : undefined
                      }
                    >
                      {form.volTarget > 0 && !(form.volEwmaAlpha > 0 && form.volEwmaAlpha < 1) && form.volLookback < 2
                        ? "Must be >=2 when vol target is set (unless EWMA alpha is provided)."
                        : "Realized-vol lookback window (bars)."}
                    </div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="volEwmaAlpha">
                      Vol EWMA alpha
                    </label>
                    <input
                      id="volEwmaAlpha"
                      className="input"
                      type="number"
                      step="0.01"
                      min={0}
                      max={0.999}
                      value={form.volEwmaAlpha}
                      onChange={(e) => setForm((f) => ({ ...f, volEwmaAlpha: numFromInput(e.target.value, f.volEwmaAlpha) }))}
                      placeholder="0"
                    />
                    <div className="hint">Optional; overrides lookback when set (0 disables).</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="volFloor">
                      Vol floor
                    </label>
                    <input
                      id="volFloor"
                      className="input"
                      type="number"
                      step="0.01"
                      min={0}
                      value={form.volFloor}
                      onChange={(e) => setForm((f) => ({ ...f, volFloor: numFromInput(e.target.value, f.volFloor) }))}
                      placeholder="0"
                    />
                    <div className="hint">Annualized floor for sizing (0 disables).</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="volScaleMax">
                      Vol scale max
                    </label>
                    <input
                      id="volScaleMax"
                      className="input"
                      type="number"
                      step="0.1"
                      min={0}
                      value={form.volScaleMax}
                      onChange={(e) => setForm((f) => ({ ...f, volScaleMax: numFromInput(e.target.value, f.volScaleMax) }))}
                      placeholder="1"
                    />
                    <div className="hint">Caps volatility-based scaling.</div>
                  </div>
                </div>
                <div className="hint">Vol sizing scales position by target/realized volatility when vol target is set.</div>
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
                    id="hiddenSize"
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

            <div className="sectionHeading" id="section-optimization">
              <span className="sectionTitle">Optimization</span>
              <span className="sectionMeta">Tuning sweeps, presets, and constraints.</span>
            </div>
            <div className="row">
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
                <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                  <div className="field">
                    <label className="label" htmlFor="tuneStressVolMult">
                      Stress vol mult
                    </label>
                    <input
                      id="tuneStressVolMult"
                      className="input"
                      type="number"
                      step="0.1"
                      min={0}
                      value={form.tuneStressVolMult}
                      onChange={(e) => setForm((f) => ({ ...f, tuneStressVolMult: numFromInput(e.target.value, f.tuneStressVolMult) }))}
                    />
                    <div className="hint">1 disables. &gt;1 increases stress volatility.</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="tuneStressShock">
                      Stress shock
                    </label>
                    <input
                      id="tuneStressShock"
                      className="input"
                      type="number"
                      step="0.001"
                      value={form.tuneStressShock}
                      onChange={(e) => setForm((f) => ({ ...f, tuneStressShock: numFromInput(e.target.value, f.tuneStressShock) }))}
                      placeholder="0"
                    />
                    <div className="hint">Additive return shock (e.g., -0.01).</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="tuneStressWeight">
                      Stress weight
                    </label>
                    <input
                      id="tuneStressWeight"
                      className="input"
                      type="number"
                      step="0.01"
                      min={0}
                      value={form.tuneStressWeight}
                      onChange={(e) => setForm((f) => ({ ...f, tuneStressWeight: numFromInput(e.target.value, f.tuneStressWeight) }))}
                      placeholder="0"
                    />
                    <div className="hint">Penalty weight (0 disables).</div>
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
                    disabled={!form.autoRefresh}
                  />{" "}
                  seconds.{!form.autoRefresh ? " Enable Auto-refresh to use this interval." : ""}{" "}
                  {form.bypassCache ? "Bypass cache adds Cache-Control: no-cache." : ""}
                </div>
              </div>
            </div>

            <div style={{ marginTop: 14 }} id="section-livebot">
              <div className="row" style={{ gridTemplateColumns: "1fr" }}>
                <div className="field">
                  <label className="label">Live bot</label>
                  <div className="actions" style={{ marginTop: 0 }}>
                    <button
                      className="btn btnPrimary"
                      disabled={botStartBlocked}
                      onClick={() => void startLiveBot()}
                      title={
                        firstReason(
                          botStartBlockedReason,
                          form.tradeArmed ? "Trading armed (will send orders)" : "Paper mode (no orders)",
                        ) ?? undefined
                      }
                    >
                      {bot.loading || botStarting ? "Starting…" : botAnyRunning ? "Running" : "Start live bot"}
                    </button>
                    <button
                      className="btn btnDanger"
                      disabled={bot.loading || (!botAnyRunning && !botStarting)}
                      onClick={() => void stopLiveBot()}
                    >
                      {botSymbolsActive.length > 1 ? "Stop all" : "Stop bot"}
                    </button>
                    {botSymbolsActive.length > 1 ? (
                      <button
                        className="btn"
                        disabled={bot.loading || !botSelectedSymbol || botStarting}
                        onClick={() => botSelectedSymbol && stopLiveBot(botSelectedSymbol)}
                      >
                        Stop selected
                      </button>
                    ) : null}
                    <button className="btn" disabled={bot.loading || Boolean(apiBlockedReason)} onClick={() => refreshBot()} title={apiBlockedReason ?? undefined}>
                      Refresh
                    </button>
                  </div>
                  <div className="row" style={{ marginTop: 10 }}>
                    <div className="field" style={{ flex: "1 1 360px" }}>
                      <label className="label" htmlFor="botSymbols">
                        Bot symbols (optional)
                      </label>
                      <input
                        id="botSymbols"
                        className="input"
                        value={form.botSymbols}
                        onChange={(e) => setForm((f) => ({ ...f, botSymbols: e.target.value }))}
                        placeholder="BTCUSDT, ETHUSDT, SOLUSDT"
                      />
                      <div className="hint">
                        Comma-separated list for multi-symbol bots. Leave blank to use the Symbol from Market settings.
                      </div>
                    </div>
                  </div>
                  <div className="hint">
                    Continuously ingests new bars, fine-tunes on each bar, and switches position based on the latest signal. Enable “Arm trading” to actually place
                    Binance orders; otherwise it runs in paper mode. If “Sweep thresholds” or “Optimize operations” is enabled, the bot re-optimizes after each
                    buy/sell operation.
                  </div>
                  {botStartBlockedReason && !botAnyRunning && !botStarting ? (
                    <div className="hint" style={{ color: "rgba(245, 158, 11, 0.9)" }}>
                      Start live bot is disabled: {botStartBlockedReason}
                    </div>
                  ) : null}
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
                              checked={true}
                              disabled
                            />
                            Always adopt existing positions
                          </label>
                        </div>
                        <div className="hint">Existing positions are adopted automatically when the live bot starts.</div>
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

            <div style={{ marginTop: 14 }} id="section-trade">
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
                        Enable Live orders? This can place real orders on Binance or Coinbase when you trade (live bot orders remain Binance-only).
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
              {tradeDisabledDetail ? (
                <div className="issueItem" style={{ marginTop: 10 }}>
                  <span>Trade disabled: {tradeDisabledDetail.message}</span>
                  {tradeDisabledDetail.targetId ? (
                    <button className="btnSmall" type="button" onClick={() => scrollToSection(tradeDisabledDetail.targetId)}>
                      Fix
                    </button>
                  ) : null}
                </div>
              ) : null}
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
              {botDisplay ? (
                <>
                  {botSymbolOptions.length > 1 ? (
                    <div className="pillRow" style={{ marginBottom: 10 }}>
                      <span className="badge">Bots</span>
                      <select
                        className="select"
                        value={botSelectedSymbol ?? ""}
                        onChange={(e) => setBotSelectedSymbol(e.target.value)}
                      >
                        {botSymbolOptions.map((entry) => (
                          <option key={entry.symbol} value={entry.symbol}>
                            {entry.label}
                          </option>
                        ))}
                      </select>
                      <span className="badge">
                        {botActiveSymbols.length}/{botSymbolOptions.length} active
                      </span>
                    </div>
                  ) : null}
                  {botStartErrors.length > 0 ? (
                    <div className="hint" style={{ marginBottom: 10, color: "rgba(239, 68, 68, 0.9)", whiteSpace: "pre-wrap" }}>
                      {botStartErrors.map((err) => `${err.symbol}: ${err.error}`).join("\n")}
                    </div>
                  ) : null}
	                  <div className="pillRow" style={{ marginBottom: 10 }}>
	                    <span className="badge">{botDisplay.symbol}</span>
	                    <span className="badge">{botDisplay.interval}</span>
	                    <span className="badge">{marketLabel(botDisplay.market)}</span>
	                    <span className="badge">{methodLabel(botDisplay.method)}</span>
	                    <span className="badge">open {fmtPct(botDisplay.openThreshold ?? botDisplay.threshold, 3)}</span>
	                    <span className="badge">
	                      close {fmtPct(botDisplay.closeThreshold ?? botDisplay.openThreshold ?? botDisplay.threshold, 3)}
	                    </span>
	                    <span className="badge">{botDisplay.halted ? "HALTED" : "ACTIVE"}</span>
	                    <span className="badge">{botDisplay.error ? "Error" : "OK"}</span>
                      {botHasSnapshot ? <span className="badge">SNAPSHOT</span> : null}
	                  </div>
                    {botHasSnapshot ? (
                      <div className="hint" style={{ marginBottom: 10 }}>
                        Snapshot {botSnapshotAtMs ? `from ${fmtTimeMs(botSnapshotAtMs)}` : "loaded"} (bot not running).
                      </div>
                    ) : null}

	                  <BacktestChart
	                    prices={botDisplay.prices}
	                    equityCurve={botDisplay.equityCurve}
	                    kalmanPredNext={botDisplay.kalmanPredNext}
	                    positions={botDisplay.positions}
	                    trades={botDisplay.trades}
	                    operations={botDisplay.operations}
	                    backtestStartIndex={botDisplay.startIndex}
	                    height={360}
	                  />

		                  <div style={{ marginTop: 10 }}>
		                    <div className="hint" style={{ marginBottom: 8 }}>
		                      Prediction values vs thresholds (hover for details)
		                    </div>
		                    <PredictionDiffChart
		                      prices={botDisplay.prices}
		                      kalmanPredNext={botDisplay.kalmanPredNext}
		                      lstmPredNext={botDisplay.lstmPredNext}
		                      startIndex={botDisplay.startIndex}
		                      height={140}
		                      openThreshold={botDisplay.openThreshold ?? botDisplay.threshold}
		                      closeThreshold={botDisplay.closeThreshold ?? botDisplay.openThreshold ?? botDisplay.threshold}
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
		                      {botDisplay.settings ? (
		                        <>
		                          poll {botDisplay.settings.pollSeconds}s • online epochs {botDisplay.settings.onlineEpochs} • train bars{" "}
		                          {botDisplay.settings.trainBars} • max points {botDisplay.settings.maxPoints} • trade{" "}
		                          {botDisplay.settings.tradeEnabled ? "ON" : "OFF"}
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
                      {fmtRatio(botDisplay.equityCurve[botDisplay.equityCurve.length - 1] ?? 1, 4)}x /{" "}
		                      {(() => {
		                        const p = botDisplay.positions[botDisplay.positions.length - 1] ?? 0;
		                        if (p > 0) return `LONG${Math.abs(p) < 0.9999 ? ` (${fmtPct(Math.abs(p), 1)})` : ""}`;
		                        if (p < 0) return `SHORT${Math.abs(p) < 0.9999 ? ` (${fmtPct(Math.abs(p), 1)})` : ""}`;
		                        return "FLAT";
		                      })()}
		                    </div>
		                  </div>
	                  <div className="kv">
	                    <div className="k">Peak / Drawdown</div>
	                    <div className="v">
	                      {fmtRatio(botDisplay.peakEquity, 4)}x / {botRisk ? fmtPct(botRisk.dd, 2) : "—"}
	                    </div>
	                  </div>
	                  <div className="kv">
	                    <div className="k">Day start / Daily loss</div>
	                    <div className="v">
	                      {fmtRatio(botDisplay.dayStartEquity, 4)}x / {botRisk ? fmtPct(botRisk.dl, 2) : "—"}
	                    </div>
	                  </div>
	                  <div className="kv">
	                    <div className="k">Halt status</div>
	                    <div className="v">
	                      {botDisplay.halted ? `HALTED${botDisplay.haltReason ? ` (${botDisplay.haltReason})` : ""}` : "Active"}
	                    </div>
	                  </div>
	                  {botDisplay.haltedAtMs ? (
	                    <div className="kv">
	                      <div className="k">Halted at</div>
	                      <div className="v">{fmtTimeMs(botDisplay.haltedAtMs)}</div>
	                    </div>
	                  ) : null}
                  <div className="kv">
                    <div className="k">Order errors</div>
                    <div className="v">{botDisplay.consecutiveOrderErrors}</div>
                  </div>
                  {typeof botDisplay.cooldownLeft === "number" && Number.isFinite(botDisplay.cooldownLeft) && botDisplay.cooldownLeft > 0 ? (
                    <div className="kv">
                      <div className="k">Cooldown</div>
                      <div className="v">{Math.max(0, Math.trunc(botDisplay.cooldownLeft))} bar(s) remaining</div>
                    </div>
                  ) : null}
                  <div className="kv">
                    <div className="k">Latest signal</div>
                    <div className="v">{botDisplay.latestSignal.action}</div>
                  </div>
                  <div className="kv">
                    <div className="k">Current price</div>
                    <div className="v">{fmtMoney(botDisplay.latestSignal.currentPrice, 4)}</div>
                  </div>
                  <div className="kv">
                    <div className="k">Kalman</div>
                    <div className="v">
                      {(() => {
                        const cur = botDisplay.latestSignal.currentPrice;
                        const next = botDisplay.latestSignal.kalmanNext;
                        const ret = botDisplay.latestSignal.kalmanReturn;
                        const z = botDisplay.latestSignal.kalmanZ;
                        const ret2 =
                          typeof ret === "number" && Number.isFinite(ret)
                            ? ret
                            : typeof next === "number" && Number.isFinite(next) && cur !== 0
                              ? (next - cur) / cur
                              : null;
                        const nextTxt = typeof next === "number" && Number.isFinite(next) ? fmtMoney(next, 4) : "—";
                        const retTxt = typeof ret2 === "number" && Number.isFinite(ret2) ? fmtPct(ret2, 3) : "—";
                        const zTxt = typeof z === "number" && Number.isFinite(z) ? fmtNum(z, 3) : "—";
                        return `${nextTxt} (${retTxt}) • z ${zTxt} • ${botDisplay.latestSignal.kalmanDirection ?? "—"}`;
                      })()}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">LSTM</div>
                    <div className="v">
                      {(() => {
                        const cur = botDisplay.latestSignal.currentPrice;
                        const next = botDisplay.latestSignal.lstmNext;
                        const ret =
                          typeof next === "number" && Number.isFinite(next) && cur !== 0 ? (next - cur) / cur : null;
                        const nextTxt = typeof next === "number" && Number.isFinite(next) ? fmtMoney(next, 4) : "—";
                        const retTxt = typeof ret === "number" && Number.isFinite(ret) ? fmtPct(ret, 3) : "—";
                        return `${nextTxt} (${retTxt}) • ${botDisplay.latestSignal.lstmDirection ?? "—"}`;
                      })()}
                    </div>
                  </div>
                  <div className="kv">
                    <div className="k">Chosen</div>
                    <div className="v">{botDisplay.latestSignal.chosenDirection ?? "—"}</div>
                  </div>
                  <div className="kv">
                    <div className="k">Close dir</div>
                    <div className="v">{formatDirectionLabel(botDisplay.latestSignal.closeDirection)}</div>
                  </div>
	                  {typeof botDisplay.latestSignal.confidence === "number" && Number.isFinite(botDisplay.latestSignal.confidence) ? (
	                    <div className="kv">
	                      <div className="k">Confidence / Size</div>
	                      <div className="v">
	                        {fmtPct(botDisplay.latestSignal.confidence, 1)}
	                        {typeof botDisplay.latestSignal.positionSize === "number" && Number.isFinite(botDisplay.latestSignal.positionSize)
	                          ? ` • ${fmtPct(botDisplay.latestSignal.positionSize, 1)}`
	                          : ""}
	                      </div>
	                    </div>
	                  ) : null}

	                  <details className="details" style={{ marginTop: 12 }}>
	                    <summary>Signal details</summary>
	                    <div style={{ marginTop: 10 }}>
	                      {(() => {
	                        const sig = botDisplay.latestSignal;
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
	                        const q = botDisplay.latestSignal.quantiles;
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
	                        const i = botDisplay.latestSignal.conformalInterval;
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
	                        const std = botDisplay.latestSignal.kalmanStd;
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

	                  {botDisplay.lastOrder ? (
	                    <div className="kv">
	                      <div className="k">Last order</div>
	                      <div className="v">{botDisplay.lastOrder.message}</div>
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

		                    {selectedOrderDetails ? (
		                      (() => {
		                        const { event, result, bar, open, close, eq, pos, kal, lstm, kalRet, lstmRet } = selectedOrderDetails;
		                        const posLabel = pos > 0 ? "LONG" : pos < 0 ? "SHORT" : "FLAT";
		                        const posSize =
		                          Math.abs(pos) > 0 && Math.abs(pos) < 0.9999 ? ` size ${fmtPct(Math.abs(pos), 1)}` : "";
		                        const eqTxt = typeof eq === "number" && Number.isFinite(eq) ? fmtRatio(eq, 4) : "—";
		                        const kalTxt = typeof kal === "number" && Number.isFinite(kal) ? fmtMoney(kal, 4) : "disabled";
		                        const kalRetTxt = typeof kalRet === "number" && Number.isFinite(kalRet) ? fmtPct(kalRet, 3) : "—";
		                        const lstmTxt = typeof lstm === "number" && Number.isFinite(lstm) ? fmtMoney(lstm, 4) : "disabled";
		                        const lstmRetTxt = typeof lstmRet === "number" && Number.isFinite(lstmRet) ? fmtPct(lstmRet, 3) : "—";
		                        const qtyTxt = typeof result.quantity === "number" && Number.isFinite(result.quantity) ? fmtNum(result.quantity, 8) : "—";
		                        const quoteQtyTxt =
		                          typeof result.quoteQuantity === "number" && Number.isFinite(result.quoteQuantity)
		                            ? fmtMoney(result.quoteQuantity, 2)
		                            : "—";
		                        const execQtyTxt =
		                          typeof result.executedQty === "number" && Number.isFinite(result.executedQty)
		                            ? fmtNum(result.executedQty, 8)
		                            : "—";
		                        const cumQuoteTxt =
		                          typeof result.cummulativeQuoteQty === "number" && Number.isFinite(result.cummulativeQuoteQty)
		                            ? fmtMoney(result.cummulativeQuoteQty, 2)
		                            : "—";
		                        return (
		                          <div className="details" style={{ marginBottom: 10 }}>
		                            <div className="btChartHeader" style={{ marginBottom: 10 }}>
		                              <div className="btChartTitle">Order details</div>
		                              <div className="btChartMeta">
		                                <span className="badge">bar {bar}</span>
		                                <span className="badge">{event.opSide}</span>
		                                <span className="badge">{fmtMoney(close, 4)}</span>
		                                <span className="badge">{result.sent ? "SENT" : "NO"}</span>
		                              </div>
		                              <div className="btChartActions">
		                                <button
		                                  className="btnSmall"
		                                  type="button"
		                                  onClick={() => {
		                                    void copyText(selectedOrderJson);
		                                    showToast("Copied order details JSON");
		                                  }}
		                                >
		                                  Copy JSON
		                                </button>
		                                <button className="btnSmall" type="button" onClick={() => setSelectedOrderKey(null)}>
		                                  Clear
		                                </button>
		                              </div>
		                            </div>
		                            <div className="kv">
		                              <div className="k">Bar / Times</div>
		                              <div className="v">
		                                bar {bar} • open {fmtTimeMs(event.openTime)} • order {fmtTimeMs(event.atMs)}
		                              </div>
		                            </div>
		                            <div className="kv">
		                              <div className="k">Prices</div>
		                              <div className="v">
		                                open (prev close) {fmtMoney(open, 4)} • close {fmtMoney(close, 4)} • order {fmtMoney(event.price, 4)}
		                              </div>
		                            </div>
		                            <div className="kv">
		                              <div className="k">Equity / Position</div>
		                              <div className="v">
		                                {eqTxt} • {posLabel}
		                                {posSize}
		                              </div>
		                            </div>
		                            <div className="kv">
		                              <div className="k">Kalman</div>
		                              <div className="v">
		                                {kalTxt}
		                                {kalTxt !== "disabled" ? ` (${kalRetTxt})` : ""}
		                              </div>
		                            </div>
		                            <div className="kv">
		                              <div className="k">LSTM</div>
		                              <div className="v">
		                                {lstmTxt}
		                                {lstmTxt !== "disabled" ? ` (${lstmRetTxt})` : ""}
		                              </div>
		                            </div>
		                            <div className="kv">
		                              <div className="k">Order</div>
		                              <div className="v">
		                                {result.sent ? "SENT" : "NO"} • {result.mode ?? "—"} • {result.status ?? "—"} •{" "}
		                                {result.side ?? event.opSide ?? "—"} • {result.symbol ?? "—"}
		                              </div>
		                            </div>
		                            <div className="kv">
		                              <div className="k">IDs</div>
		                              <div className="v">
		                                order {result.orderId ?? "—"} • client {result.clientOrderId ?? "—"}
		                              </div>
		                            </div>
		                            <div className="kv">
		                              <div className="k">Amounts</div>
		                              <div className="v">
		                                qty {qtyTxt} • quote {quoteQtyTxt} • executed {execQtyTxt} • cumulative {cumQuoteTxt}
		                              </div>
		                            </div>
		                            <div className="kv">
		                              <div className="k">Message</div>
		                              <div className="v">{result.message}</div>
		                            </div>
		                            {result.response ? (
		                              <div className="kv">
		                                <div className="k">Response</div>
		                                <div className="v">{result.response}</div>
		                              </div>
		                            ) : null}
		                            <details className="details" style={{ marginTop: 10 }}>
		                              <summary>Raw order JSON</summary>
		                              <pre className="code" style={{ marginTop: 10 }}>
		                                {selectedOrderJson}
		                              </pre>
		                            </details>
		                          </div>
		                        );
		                      })()
		                    ) : (
		                      <div className="hint" style={{ marginBottom: 10 }}>
		                        Click an order row to see all details.
		                      </div>
		                    )}

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
		                              const rowKey = orderRowKey(e);
		                              const selected = rowKey === selectedOrderKey;
		                              return (
		                                <tr
		                                  key={rowKey}
		                                  className={`tableRowClickable${selected ? " tableRowSelected" : ""}`}
		                                  onClick={() => setSelectedOrderKey((prev) => (prev === rowKey ? null : rowKey))}
		                                  onKeyDown={(ev) => {
		                                    if (ev.key === "Enter" || ev.key === " ") {
		                                      ev.preventDefault();
		                                      setSelectedOrderKey((prev) => (prev === rowKey ? null : rowKey));
		                                    }
		                                  }}
		                                  tabIndex={0}
		                                  aria-selected={selected}
		                                  title="Click to view order details"
		                                >
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
                    : botStartBlockedReason
                      ? `Bot is stopped. Start live bot is disabled: ${botStartBlockedReason}`
                      : "Bot is stopped. Use “Start live bot” on the left."}
                </div>
              )}
            </div>
          </div>

          <div className="card">
            <div className="cardHeader">
              <h2 className="cardTitle">Binance account trades</h2>
              <p className="cardSubtitle">Full exchange history from your Binance account (API keys required).</p>
            </div>
            <div className="cardBody">
              <div className="row">
                <div className="field" style={{ flex: "2 1 360px" }}>
                  <label className="label" htmlFor="binanceTradesSymbols">
                    Symbols (optional)
                  </label>
                  <input
                    id="binanceTradesSymbols"
                    className="input"
                    value={binanceTradesSymbolsInput}
                    onChange={(e) => setBinanceTradesSymbolsInput(e.target.value)}
                    placeholder="BTCUSDT, ETHUSDT"
                  />
                  <div className="hint">Leave blank for all symbols (futures only). Spot/margin require a symbol.</div>
                </div>
                <div className="field" style={{ flex: "1 1 160px" }}>
                  <label className="label" htmlFor="binanceTradesLimit">
                    Limit
                  </label>
                  <input
                    id="binanceTradesLimit"
                    className="input"
                    type="number"
                    min={1}
                    max={1000}
                    value={binanceTradesLimit}
                    onChange={(e) => setBinanceTradesLimit(numFromInput(e.target.value, binanceTradesLimit))}
                  />
                  <div className="hint">Max 1000 per request.</div>
                </div>
              </div>
              <div className="row" style={{ marginTop: 10 }}>
                <div className="field">
                  <label className="label" htmlFor="binanceTradesStart">
                    Start time (optional)
                  </label>
                  <input
                    id="binanceTradesStart"
                    className="input"
                    value={binanceTradesStartInput}
                    onChange={(e) => setBinanceTradesStartInput(e.target.value)}
                    placeholder="2025-12-23T00:00:00Z or 1700000000000"
                  />
                </div>
                <div className="field">
                  <label className="label" htmlFor="binanceTradesEnd">
                    End time (optional)
                  </label>
                  <input
                    id="binanceTradesEnd"
                    className="input"
                    value={binanceTradesEndInput}
                    onChange={(e) => setBinanceTradesEndInput(e.target.value)}
                    placeholder="2025-12-23T23:59:59Z"
                  />
                </div>
                <div className="field">
                  <label className="label" htmlFor="binanceTradesFromId">
                    From ID (optional)
                  </label>
                  <input
                    id="binanceTradesFromId"
                    className="input"
                    value={binanceTradesFromIdInput}
                    onChange={(e) => setBinanceTradesFromIdInput(e.target.value)}
                    placeholder="Trade ID"
                  />
                </div>
              </div>
              <div className="actions" style={{ marginTop: 10 }}>
                <button
                  className="btn btnPrimary"
                  disabled={binanceTradesUi.loading || Boolean(binanceTradesInputError)}
                  onClick={() => void fetchBinanceTrades()}
                >
                  {binanceTradesUi.loading ? "Loading…" : "Fetch trades"}
                </button>
                <button
                  className="btn"
                  type="button"
                  disabled={!binanceTradesUi.response}
                  onClick={() => {
                    void copyText(binanceTradesCopyText);
                    showToast("Copied trade log");
                  }}
                >
                  Copy
                </button>
                <button
                  className="btn"
                  type="button"
                  disabled={!binanceTradesJson}
                  onClick={() => {
                    void copyText(binanceTradesJson);
                    showToast("Copied trade log JSON");
                  }}
                >
                  Copy JSON
                </button>
                <button
                  className="btn"
                  type="button"
                  disabled={!binanceTradesUi.response && !binanceTradesUi.error}
                  onClick={() => setBinanceTradesUi({ loading: false, error: null, response: null })}
                >
                  Clear
                </button>
              </div>
              {binanceTradesInputError ? (
                <div className="hint" style={{ marginTop: 10, color: "rgba(239, 68, 68, 0.9)" }}>
                  {binanceTradesInputError}
                </div>
              ) : null}
              {binanceTradesUi.error ? (
                <div className="hint" style={{ marginTop: 10, color: "rgba(239, 68, 68, 0.9)", whiteSpace: "pre-wrap" }}>
                  {binanceTradesUi.error}
                </div>
              ) : null}
              {binanceTradesUi.response ? (
                <>
                  <div className="pillRow" style={{ marginTop: 12, marginBottom: 10 }}>
                    <span className="badge">{marketLabel(binanceTradesUi.response.market)}</span>
                    <span className="badge">{binanceTradesUi.response.testnet ? "TESTNET" : "LIVE"}</span>
                    <span className="badge">{binanceTradesUi.response.trades.length} trades</span>
                    <span className="badge">
                      {binanceTradesUi.response.allSymbols
                        ? "all symbols"
                        : binanceTradesUi.response.symbols.length > 0
                          ? binanceTradesUi.response.symbols.join(", ")
                          : "symbol"}
                    </span>
                    <span className="badge">fetched {fmtTimeMs(binanceTradesUi.response.fetchedAtMs)}</span>
                  </div>
                  {binanceTradesUi.response.trades.length === 0 ? (
                    <div className="hint">No trades returned.</div>
                  ) : (
                    <div className="tableWrap" role="region" aria-label="Binance account trades">
                      <table className="table">
                        <thead>
                          <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Price</th>
                            <th>Qty</th>
                            <th>Quote</th>
                            <th>Pos</th>
                            <th>Commission</th>
                            <th>PNL</th>
                            <th>Order</th>
                          </tr>
                        </thead>
                        <tbody>
                          {binanceTradesUi.response.trades.map((trade) => {
                            const side = trade.side ?? (trade.isBuyer === true ? "BUY" : trade.isBuyer === false ? "SELL" : "—");
                            const qtyTxt = Number.isFinite(trade.qty) ? fmtNum(trade.qty, 8) : "—";
                            const quoteTxt = Number.isFinite(trade.quoteQty) ? fmtMoney(trade.quoteQty, 2) : "—";
                            const commissionTxt =
                              trade.commission != null && Number.isFinite(trade.commission)
                                ? `${fmtNum(trade.commission, 8)}${trade.commissionAsset ? ` ${trade.commissionAsset}` : ""}`
                                : "—";
                            const pnlTxt =
                              trade.realizedPnl != null && Number.isFinite(trade.realizedPnl) ? fmtMoney(trade.realizedPnl, 4) : "—";
                            return (
                              <tr key={`${trade.symbol}-${trade.tradeId}`}>
                                <td className="tdMono">{fmtTimeMs(trade.time)}</td>
                                <td className="tdMono">{trade.symbol}</td>
                                <td>
                                  <span className={side === "BUY" ? "badge badgeStrong badgeLong" : side === "SELL" ? "badge badgeStrong badgeFlat" : "badge"}>
                                    {side}
                                  </span>
                                </td>
                                <td className="tdMono">{fmtMoney(trade.price, 4)}</td>
                                <td className="tdMono">{qtyTxt}</td>
                                <td className="tdMono">{quoteTxt}</td>
                                <td className="tdMono">{trade.positionSide ?? "—"}</td>
                                <td className="tdMono">{commissionTxt}</td>
                                <td className="tdMono">{pnlTxt}</td>
                                <td className="tdMono">{trade.orderId ?? "—"}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                  <div className="hint" style={{ marginTop: 8 }}>
                    Binance returns up to 1000 trades per request. Use start/end time or fromId to page deeper history.
                  </div>
                </>
              ) : (
                <div className="hint" style={{ marginTop: 10 }}>
                  No trades loaded yet.
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
                  <div className="kv">
                    <div className="k">Close dir</div>
                    <div className="v">{formatDirectionLabel(state.latestSignal.closeDirection)}</div>
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
				                    agreementOk={state.backtest.method === "01" ? undefined : state.backtest.agreementOk}
				                    trades={state.backtest.trades}
				                    backtestStartIndex={state.backtest.split.backtestStartIndex}
				                    height={360}
				                  />
                        <div style={{ marginTop: 10 }}>
                          <div className="hint" style={{ marginBottom: 8 }}>
                            Prediction values vs thresholds (hover for details)
                          </div>
                          <PredictionDiffChart
                            prices={state.backtest.prices}
                            kalmanPredNext={state.backtest.kalmanPredNext}
                            lstmPredNext={state.backtest.lstmPredNext}
                            startIndex={state.backtest.split.backtestStartIndex}
                            height={140}
                            openThreshold={state.backtest.openThreshold ?? state.backtest.threshold}
                            closeThreshold={
                              state.backtest.closeThreshold ?? state.backtest.openThreshold ?? state.backtest.threshold
                            }
                          />
                        </div>
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
                <span className="badge">Keys: {keysProvidedLabel}</span>
                {isBinancePlatform ? (
                  <span className="badge">
                    {marketLabel(form.market)}
                    {form.binanceTestnet ? " testnet" : ""}
                  </span>
                ) : null}
                {keysSigned ? <span className="badge">Signed: {keysSigned.ok ? "OK" : "FAIL"}</span> : null}
                {keysTradeTest ? (
                  <span className="badge">
                    Trade:{" "}
                    {form.market === "margin"
                      ? "N/A"
                      : keysTradeTest.skipped
                        ? "SKIP"
                        : keysTradeTest.ok
                          ? "OK"
                          : "FAIL"}
                  </span>
                ) : null}
              </div>
              {!isBinancePlatform ? (
                <div className="hint" style={{ color: "rgba(245, 158, 11, 0.9)", marginBottom: 10 }}>
                  {isCoinbasePlatform
                    ? "Coinbase trades are spot-only and live-only (no test endpoint)."
                    : "Trading is supported on Binance and Coinbase only. Key checks are supported on Binance and Coinbase."}
                </div>
              ) : null}

              <div className="actions" style={{ marginTop: 0, marginBottom: 10 }}>
                <button
                  className="btn"
                  type="button"
                  onClick={() => refreshKeys()}
                  disabled={!keysSupported || keys.loading || apiOk === "down" || apiOk === "auth"}
                >
                  {keys.loading ? "Checking…" : "Check keys"}
                </button>
                <span className="hint">
                  {!keysSupported
                    ? "Switch Platform to Binance or Coinbase to check keys."
                    : keysCheckedAtMs
                      ? `Last checked: ${fmtTimeMs(keysCheckedAtMs)}`
                      : isBinancePlatform
                        ? "Uses Binance signed endpoints + /order/test (no real order)."
                        : "Uses Coinbase signed /accounts."}
                </span>
              </div>

              {keys.error ? (
                <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)", marginBottom: 10 }}>
                  {keys.error}
                </pre>
              ) : null}

              {activeKeysStatus ? (
                <>
                  <div className="kv">
                    <div className="k">
                      {isCoinbaseKeysStatus(activeKeysStatus)
                        ? "COINBASE_API_KEY / COINBASE_API_SECRET / COINBASE_API_PASSPHRASE"
                        : "BINANCE_API_KEY / BINANCE_API_SECRET"}
                    </div>
                    <div className="v">
                      {activeKeysStatus.hasApiKey ? "present" : "missing"} / {activeKeysStatus.hasApiSecret ? "present" : "missing"}
                      {isCoinbaseKeysStatus(activeKeysStatus)
                        ? ` / ${activeKeysStatus.hasApiPassphrase ? "present" : "missing"}`
                        : ""}
                    </div>
                  </div>

                  <div className="kv">
                    <div className="k">Signed check</div>
                    <div className="v">
                      {keysSigned ? (
                        <>
                          {keysSigned.ok ? "OK" : "FAIL"} {keysSigned.code !== undefined ? `(${keysSigned.code}) ` : ""}
                          {keysSigned.summary}
                        </>
                      ) : (
                        "—"
                      )}
                    </div>
                  </div>

                  {isBinancePlatform ? (
                    <div className="kv">
                      <div className="k">Trade permission</div>
                      <div className="v">
                        {keysTradeTest ? (
                          <>
                            {keysTradeTest.skipped ? "SKIP" : keysTradeTest.ok ? "OK" : "FAIL"}{" "}
                            {keysTradeTest.code !== undefined ? `(${keysTradeTest.code}) ` : ""}
                            {keysTradeTest.summary}
                          </>
                        ) : (
                          "—"
                        )}
                      </div>
                    </div>
                  ) : null}
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
                <button
                  className="btn"
                  type="button"
                  onClick={() => void startListenKeyStream()}
                  disabled={!isBinancePlatform || listenKeyUi.loading || apiOk !== "ok"}
                >
                  {listenKeyUi.loading ? "Starting…" : listenKeyUi.info ? "Restart stream" : "Start stream"}
                </button>
                <button
                  className="btn"
                  type="button"
                  onClick={() => (listenKeyUi.info ? void keepAliveListenKeyStream(listenKeyUi.info) : undefined)}
                  disabled={!isBinancePlatform || !listenKeyUi.info || listenKeyUi.loading || apiOk !== "ok"}
                >
                  Keep alive now
                </button>
                <button
                  className="btn"
                  type="button"
                  onClick={() => void stopListenKeyStream({ close: true })}
                  disabled={!isBinancePlatform || !listenKeyUi.info || listenKeyUi.loading || apiOk !== "ok"}
                >
                  Stop
                </button>
                <span className="hint">
                  {!isBinancePlatform
                    ? "Listen key streams are available on Binance only."
                    : `Binance requires a keep-alive (PUT) at least every ~30 minutes; the UI schedules one every ~${Math.round((listenKeyUi.info?.keepAliveMs ?? 25 * 60_000) * 0.9 / 60_000)} minutes.`}
                </span>
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
                disabled={dataLogShown.length === 0}
	              onClick={() => {
	                const logText = dataLogShown
	                  .map((entry) => `[${new Date(entry.timestamp).toISOString()}] ${entry.label}:\n${JSON.stringify(entry.data, null, 2)}`)
	                  .join("\n\n");
	                copyText(logText);
	                showToast("Copied log to clipboard");
	              }}
	            >
	              {dataLogFilterText.trim() ? "Copy shown" : "Copy all"}
	            </button>
              <input
                className="input"
                style={{ height: 32, width: 180, padding: "0 10px" }}
                value={dataLogFilterText}
                onChange={(e) => setDataLogFilterText(e.target.value)}
                placeholder="Filter log…"
                spellCheck={false}
                aria-label="Filter data log"
              />
              {dataLogFilterText.trim() ? (
                <button className="btnSmall" type="button" onClick={() => setDataLogFilterText("")}>
                  Clear filter
                </button>
              ) : null}
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogExpanded} onChange={(e) => setDataLogExpanded(e.target.checked)} />
                Expand
              </label>
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogIndexArrays} onChange={(e) => setDataLogIndexArrays(e.target.checked)} />
                Index arrays
              </label>
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogAutoScroll} onChange={(e) => setDataLogAutoScroll(e.target.checked)} />
                Auto-scroll
              </label>
              <button className="btnSmall" type="button" onClick={scrollDataLogToBottom} disabled={dataLog.length === 0}>
                Jump to latest
              </button>
              {dataLogFilterText.trim() ? (
                <span className="hint">
                  Showing {dataLogFiltered.length} of {dataLog.length}
                </span>
              ) : null}
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
            {dataLogShown.length === 0 ? (
              <div style={{ color: "#6b7280" }}>
                {dataLog.length === 0
                  ? "No data logged yet. Run a signal, backtest, or trade to see incoming data."
                  : "No entries match the current filter."}
              </div>
            ) : (
              dataLogShown.map((entry, idx) => (
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
