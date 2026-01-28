import React, { Suspense, lazy, useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import type {
  ApiBinancePositionsRequest,
  ApiBinancePositionsResponse,
  ApiBinanceTradesRequest,
  ApiBinanceTradesResponse,
  ApiParams,
  ApiTradeResponse,
  BacktestResponse,
  BinanceKeysStatus,
  BinanceListenKeyResponse,
  BinancePositionChart,
  BinanceTrade,
  BotOrderEvent,
  BotStatus,
  BotStatusMulti,
  BotStatusRunning,
  BotStatusSingle,
  CoinbaseKeysStatus,
  IntrabarFill,
  LatestSignal,
  Market,
  Method,
  Normalization,
  OpsOperation,
  OpsPerformanceResponse,
  OptimizerRunRequest,
  OptimizerRunResponse,
  OptimizerSource,
  Platform,
  Positioning,
  StateSyncImportResponse,
  StateSyncPayload,
} from "./lib/types";
import {
  HttpError,
  backtest,
  binancePositions,
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
  ops,
  opsPerformance,
  optimizerCombos,
  optimizerRun,
  stateSyncExport,
  stateSyncImport,
  signal,
  trade,
} from "./lib/api";
import { buildTenantKey } from "./lib/tenant";
import { copyText } from "./lib/clipboard";
import { TRADER_UI_CONFIG } from "./lib/deployConfig";
import { readJson, readLocalString, readSessionString, removeLocalKey, removeSessionKey, writeJson, writeLocalString, writeSessionString } from "./lib/storage";
import { fmtMoney, fmtNum, fmtPct, fmtRatio } from "./lib/format";
import { CollapsibleCard } from "./components/CollapsibleCard";
import { ConfigDock, type ConfigDockProps } from "./components/ConfigDock";
import { InfoList, InfoPopover } from "./components/InfoPopover";
import { API_PORT, API_TARGET } from "./app/apiTarget";
import {
  CONFIG_PAGE_IDS,
  CONFIG_PAGE_LABELS,
  CONFIG_PAGE_PANEL_MAP,
  CONFIG_PANEL_HEIGHTS,
  CONFIG_PANEL_IDS,
  normalizeConfigPage,
  normalizeConfigPanelOrder,
  resolveConfigPageForTarget,
} from "./app/configLayout";
import { comboMarketValue, type ComboMarketFilter, type ComboMarketValue } from "./app/comboMarket";
import {
  BACKTEST_TIMEOUT_MS,
  BOT_START_TIMEOUT_MS,
  BOT_STATUS_OPS_FALLBACK_LIMIT,
  BOT_STATUS_OPS_LIMIT,
  BOT_STATUS_POLL_MS,
  BOT_STATUS_TAIL_FALLBACK_POINTS,
  BOT_STATUS_TAIL_POINTS,
  BOT_STATUS_TIMEOUT_MS,
  OPTIMIZER_UI_MAX_BARS,
  OPTIMIZER_UI_MAX_TIMEOUT_SEC,
  OPTIMIZER_UI_MAX_TRIALS,
  BOT_AUTOSTART_RETRY_MS,
  BOT_TELEMETRY_POINTS,
  DATA_LOG_AUTO_SCROLL_SLOP_PX,
  DATA_LOG_COLLAPSED_MAX_LINES,
  DATA_LOG_MAX_ENTRIES,
  PLATFORM_DEFAULT_SYMBOL,
  PLATFORM_DEFAULT_BARS,
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
  STORAGE_CONFIG_PANEL_ORDER_KEY,
  STORAGE_CONFIG_PAGE_KEY,
  STORAGE_CONFIG_TAB_KEY,
  STORAGE_DATA_LOG_KEY,
  STORAGE_DATA_LOG_PREFS_KEY,
  STORAGE_KEY,
  STORAGE_ORDER_LOG_PREFS_KEY,
  STORAGE_PANEL_PREFS_KEY,
  STORAGE_PERSIST_SECRETS_KEY,
  STORAGE_PROFILES_KEY,
  STORAGE_STATE_SYNC_TARGET_KEY,
  STORAGE_STATE_SYNC_TOKEN_KEY,
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
import { DEFAULT_DATA_LOG_PREFS, normalizeDataLog, type DataLogEntry, type DataLogPrefs } from "./app/dataLog";
import {
  actionBadgeClass,
  buildOrphanedPositions,
  buildRequestIssueDetails,
  clamp,
  escapeSingleQuotes,
  firstReason,
  fmtDurationMs,
  fmtEtaMs,
  fmtProfitFactor,
  fmtTimeMs,
  fmtTimeMsWithMs,
  generateIdempotencyKey,
  indexTopLevelPrimitiveArrays,
  isAbortError,
  isLikelyOrderError,
  isLocalHostname,
  isTimeoutError,
  marketLabel,
  methodLabel,
  normalizeApiBaseUrlInput,
  normalizeSymbolKey,
  numFromInput,
} from "./app/utils";
import {
  BINANCE_SYMBOL_PATTERN,
  COMMON_QUOTES,
  COMPLEX_TIPS,
  CUSTOM_SYMBOL_VALUE,
  DIRECTION_HOLD_REASONS,
  DURATION_UNITS,
  EQUITY_TIPS,
  MAX_BACKTEST_RATIO,
  MIN_BACKTEST_BARS,
  MIN_BACKTEST_RATIO,
  MIN_LOOKBACK_BARS,
  RATIO_ROUND_DIGITS,
  RATIO_ROUND_FACTOR,
  SECONDS_PER_YEAR,
  TOP_COMBOS_BOT_TARGET,
  TOP_COMBOS_DISPLAY_DEFAULT,
  TOP_COMBOS_DISPLAY_MIN,
  TOP_COMBOS_POLL_MS,
  TRADE_PNL_EPS,
  TRADE_PNL_TOP_N,
  applyComboToForm,
  backtestTradePhase,
  binanceTradeSideLabel,
  botStatusKey,
  botStatusKeyFromSingle,
  botStatusSymbol,
  buildBacktestOpsCsv,
  buildBacktestTradePnlAnalysis,
  buildBinanceTradePnlAnalysis,
  buildDefaultOptimizerRunForm,
  buildEquityCurve,
  buildOptimizerRunRequest,
  buildPositionSeries,
  clampComboForLimits,
  clampOptionalInt,
  clampOptionalRange,
  clampOptionalRatio,
  coerceNumber,
  comboAnnualizedEquity,
  comboApplySignature,
  createSseParser,
  csvEscape,
  decisionBadgeClass,
  decisionDotClass,
  decisionStatusLabel,
  downloadTextFile,
  emptyBotRtState,
  emptyBotRtTracker,
  formApplySignature,
  formatDatetimeLocal,
  formatDirectionLabel,
  formatDurationSeconds,
  formatSignalDirection,
  inferPeriodsPerYear,
  invalidSymbolsForPlatform,
  isBinanceKeysStatus,
  isBotStatusMulti,
  isCoinbaseKeysStatus,
  isFiniteNumber,
  maxBarsForPlatform,
  maxLookbackForSplit,
  minTrainEndForTune,
  normalizeComboSymbol,
  normalizeHoldReason,
  normalizeIsoInput,
  normalizeListenKeyStreamStatus,
  optimizerSourceForPlatform,
  parseActionReason,
  parseBotStatusOp,
  parseDatetimeLocal,
  parseMaybeInt,
  parseOptimizerExtras,
  parseOptionalInt,
  parseOptionalNumber,
  parseOptionalString,
  parseSymbolsInput,
  parseTimeInputMs,
  pnlBadgeClass,
  positionSideInfo,
  ratioForTrainEnd,
  roundRatioDown,
  roundRatioUp,
  safeJsonParse,
  sanitizeFilenameSegment,
  sigBool,
  sigNumber,
  sigText,
  splitStats,
  symbolFormatExample,
  symbolFormatPattern,
  trimBinanceComboSuffix,
  tuneRatioBounds,
  type ActiveAsyncJob,
  type BinancePnlAnalysis,
  type BinancePnlRow,
  type BinancePositionsUiState,
  type BinanceTradesUiState,
  type BotRtEvent,
  type BotRtTracker,
  type BotRtUiState,
  type BotStatusOp,
  type BotTelemetryPoint,
  type BotUiState,
  type CacheUiState,
  type ComboOrder,
  type CommissionTotal,
  type ComputeLimits,
  type DecisionCheck,
  type DecisionCheckStatus,
  type DecisionSummary,
  type ErrorFix,
  type KeysStatus,
  type KeysUiState,
  type ListenKeyStreamErrorPayload,
  type ListenKeyStreamKeepAlivePayload,
  type ListenKeyStreamStatus,
  type ListenKeyStreamStatusPayload,
  type ListenKeyUiState,
  type ManualOverrideKey,
  type OpsUiState,
  type OptimizerRunForm,
  type OptimizerRunUiState,
  type OrderLogPrefs,
  type OrderSideFilter,
  type PanelPrefs,
  type PendingProfileLoad,
  type RateLimitState,
  type RequestKind,
  type RunOptions,
  type SavedProfiles,
  type SplitStats,
  type TopCombosMeta,
  type TopCombosSource,
  type TradePnlAnalysis,
  type TradePnlRow,
  type TuneRatioBounds,
  type UiState,
} from "./app/appHelpers";
import type { OptimizationCombo, OptimizationComboOperation } from "./components/TopCombosChart";
import type { OptimizerCombosPanelProps } from "./components/OptimizerCombosPanel";
import type { ConfigPageId, ConfigPanelDragState, ConfigPanelId } from "./app/configTypes";

const BacktestChart = lazy(() => import("./components/BacktestChart").then((mod) => ({ default: mod.BacktestChart })));
const BotStateChart = lazy(() => import("./components/BotStateChart").then((mod) => ({ default: mod.BotStateChart })));
const LiveVisuals = lazy(() => import("./components/LiveVisuals").then((mod) => ({ default: mod.LiveVisuals })));
const PredictionDiffChart = lazy(() =>
  import("./components/PredictionDiffChart").then((mod) => ({ default: mod.PredictionDiffChart })),
);
const TelemetryChart = lazy(() => import("./components/TelemetryChart").then((mod) => ({ default: mod.TelemetryChart })));
const OptimizerCombosPanel = lazy(() =>
  import("./components/OptimizerCombosPanel").then((mod) => ({ default: mod.OptimizerCombosPanel })),
);

const BOT_DISPLAY_STALE_MS = 6_000;
const BOT_DISPLAY_STARTING_STALE_MS = Number.POSITIVE_INFINITY;
const BOT_STATUS_RETRYABLE_HTTP = new Set([502, 503, 504]);
const CHART_HEIGHT = "var(--chart-height)";
const CHART_HEIGHT_SIDE = "var(--chart-height-side)";
const CHART_HEIGHT_TIMELINE = "var(--chart-height-timeline)";
const TRADE_PNL_EPS_LABEL = fmtNum(TRADE_PNL_EPS, 9);
const STATE_SYNC_CHUNK_DEFAULT_BYTES = 900_000;
const STATE_SYNC_CHUNK_MAX_BYTES = 50_000_000;
const textEncoder = typeof TextEncoder !== "undefined" ? new TextEncoder() : null;
const textByteLength = (raw: string): number => {
  if (textEncoder) return textEncoder.encode(raw).length;
  if (typeof Blob !== "undefined") return new Blob([raw]).size;
  return raw.length;
};
const jsonByteLength = (value: unknown): number => {
  try {
    return textByteLength(JSON.stringify(value));
  } catch {
    return 0;
  }
};
type ComboImportSummary = {
  comboCount: number;
  generatedAtMs: number | null;
  source: string | null;
};

type ComboImportParseResult = {
  payload: Record<string, unknown> | null;
  summary: ComboImportSummary | null;
  error: string | null;
};

type ComboImportUiState = {
  text: string;
  payload: Record<string, unknown> | null;
  summary: ComboImportSummary | null;
  parseError: string | null;
  importError: string | null;
  importing: boolean;
  importResult: StateSyncImportResponse["topCombos"] | null;
  importedAtMs: number | null;
  stampNow: boolean;
};

const parseTopCombosPayload = (raw: unknown): ComboImportParseResult => {
  if (!raw || typeof raw !== "object") {
    return { payload: null, summary: null, error: "Expected a JSON object." };
  }
  const root = raw as Record<string, unknown>;
  const embedded =
    root.topCombos && typeof root.topCombos === "object" && root.topCombos !== null
      ? (root.topCombos as Record<string, unknown>)
      : null;
  const candidate = embedded ?? root;
  if (!candidate || typeof candidate !== "object") {
    return { payload: null, summary: null, error: "Expected top-combos JSON object." };
  }
  const combos = (candidate as Record<string, unknown>).combos;
  if (!Array.isArray(combos)) {
    return { payload: null, summary: null, error: "Expected an object with a combos array." };
  }
  const generatedAtMsRaw = (candidate as Record<string, unknown>).generatedAtMs;
  const generatedAtMs =
    typeof generatedAtMsRaw === "number" && Number.isFinite(generatedAtMsRaw) ? Math.trunc(generatedAtMsRaw) : null;
  const sourceValue = (candidate as Record<string, unknown>).source;
  const sourceRaw = typeof sourceValue === "string" ? sourceValue.trim() : "";
  const source = sourceRaw ? sourceRaw : null;
  return {
    payload: candidate,
    summary: {
      comboCount: combos.length,
      generatedAtMs,
      source,
    },
    error: null,
  };
};

const parseTopCombosJson = (rawText: string): ComboImportParseResult => {
  const trimmed = rawText.trim();
  if (!trimmed) {
    return { payload: null, summary: null, error: "Paste combos JSON first." };
  }
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    return parseTopCombosPayload(parsed);
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Invalid JSON.";
    return { payload: null, summary: null, error: `Invalid JSON: ${msg}` };
  }
};
const pruneStateSyncPayload = (
  payload: StateSyncPayload,
  opts: { includeBots: boolean; includeCombos: boolean },
): StateSyncPayload => {
  const out: StateSyncPayload = {};
  if (payload.generatedAtMs != null) out.generatedAtMs = payload.generatedAtMs;
  if (opts.includeBots && Array.isArray(payload.botSnapshots) && payload.botSnapshots.length > 0) {
    out.botSnapshots = payload.botSnapshots;
  }
  if (opts.includeCombos && payload.topCombos != null) {
    out.topCombos = payload.topCombos;
  }
  return out;
};
const isStateSyncPayloadEmpty = (payload: StateSyncPayload): boolean => {
  const hasBots = Array.isArray(payload.botSnapshots) && payload.botSnapshots.length > 0;
  const hasCombos = payload.topCombos != null;
  return !hasBots && !hasCombos;
};
const buildStateSyncChunks = (payload: StateSyncPayload, maxBytes: number): StateSyncPayload[] => {
  const chunks: StateSyncPayload[] = [];
  const generatedAtMs = payload.generatedAtMs;
  const botSnapshots = payload.botSnapshots ?? [];
  const topCombos = payload.topCombos;
  const buildBotPayload = (snaps: StateSyncPayload["botSnapshots"]) => {
    const out: StateSyncPayload = {};
    if (generatedAtMs != null) out.generatedAtMs = generatedAtMs;
    if (snaps && snaps.length > 0) out.botSnapshots = snaps;
    return out;
  };

  if (botSnapshots.length > 0) {
    if (maxBytes <= 0) {
      chunks.push(buildBotPayload(botSnapshots));
    } else {
      const prefix = generatedAtMs != null ? `{"generatedAtMs":${generatedAtMs},"botSnapshots":[` : `{"botSnapshots":[`;
      const suffix = "]}";
      const prefixBytes = textByteLength(prefix);
      const suffixBytes = textByteLength(suffix);
      let current: NonNullable<StateSyncPayload["botSnapshots"]> = [];
      let currentBytes = prefixBytes + suffixBytes;
      for (const snap of botSnapshots) {
        const snapJson = JSON.stringify(snap);
        const snapBytes = textByteLength(snapJson);
        const extraBytes = (current.length > 0 ? 1 : 0) + snapBytes;
        if (current.length > 0 && currentBytes + extraBytes > maxBytes) {
          chunks.push(buildBotPayload(current));
          current = [snap];
          currentBytes = prefixBytes + suffixBytes + snapBytes;
          continue;
        }
        current.push(snap);
        currentBytes += extraBytes;
        if (current.length === 1 && currentBytes > maxBytes) {
          chunks.push(buildBotPayload(current));
          current = [];
          currentBytes = prefixBytes + suffixBytes;
        }
      }
      if (current.length > 0) {
        chunks.push(buildBotPayload(current));
      }
    }
  }
  if (topCombos != null) {
    const comboPayload: StateSyncPayload = {};
    if (generatedAtMs != null) comboPayload.generatedAtMs = generatedAtMs;
    comboPayload.topCombos = topCombos;
    chunks.push(comboPayload);
  }
  return chunks;
};
const ACCOUNT_TRADE_PNL_TIPS = {
  outcomes: [
    "Counts trades by exchange realized P&L (realizedPnl).",
    `Win/loss if |P&L| > ${TRADE_PNL_EPS_LABEL}; otherwise flat.`,
    "Win rate = wins / total trades (after filters).",
  ],
  avgWinLoss: [
    "Average realized P&L per winning/losing trade.",
    "Payoff ratio = avg win / |avg loss|.",
    "Fees are shown separately.",
  ],
  bestWorst: [
    "Largest positive/negative realized P&L in this set.",
    "Avg P&L = mean realized P&L across all trades.",
  ],
  totalPnl: [
    "Total P&L = sum of realized P&L across trades.",
    "Total win/loss = sum of positive/negative realized P&L.",
    "Profit factor = gross wins / |gross losses|.",
    "Fees are listed by asset (not subtracted from totals).",
  ],
  totals: [
    "Qty = sum of trade quantities.",
    "Quote = sum of quote quantities.",
    "Scope follows the current filters/symbol selection.",
  ],
};
const BACKTEST_TRADE_PNL_TIPS = {
  outcomes: [
    "Counts trades by per-trade return from the backtest engine.",
    `Win/loss if |return| > ${TRADE_PNL_EPS_LABEL}; otherwise flat.`,
    "Returns are shown as % of equity per trade.",
  ],
  avgWinLoss: [
    "Average return per winning/losing trade.",
    "Payoff ratio = avg win / |avg loss|.",
    "Hold bars show average holding periods for wins/losses.",
  ],
  bestWorst: [
    "Largest positive/negative per-trade return.",
    "Avg return = mean return across all trades.",
  ],
  totals: [
    "Total win/loss = sum of per-trade returns (not compounded).",
    "Profit factor = gross wins / |gross losses|.",
  ],
};
type OpsPerformanceUiState = {
  loading: boolean;
  error: string | null;
  enabled: boolean;
  ready: boolean;
  commitsReady: boolean;
  combosReady: boolean;
  hint: string | null;
  commits: OpsPerformanceResponse["commits"];
  combos: OpsPerformanceResponse["combos"];
  lastFetchedAtMs: number | null;
};
type StateSyncUiState = {
  exporting: boolean;
  exportError: string | null;
  payload: StateSyncPayload | null;
  exportedAtMs: number | null;
  pushing: boolean;
  pushError: string | null;
  pushResult: StateSyncImportResponse | null;
  pushChunkCount: number | null;
  pushedAtMs: number | null;
};
const ChartFallback = ({
  height = CHART_HEIGHT,
  label = "Loading chart…",
}: {
  height?: number | string;
  label?: string;
}) => (
  <div className="chart" style={{ height }}>
    <div className="chartEmpty">{label}</div>
  </div>
);
const ChartSuspense = ({
  height,
  label,
  children,
}: {
  height?: number | string;
  label?: string;
  children: React.ReactNode;
}) => <Suspense fallback={<ChartFallback height={height} label={label} />}>{children}</Suspense>;
const PanelFallback = ({ label }: { label: string }) => <div className="hint">{label}</div>;
function isBinanceTimestampErrorMessage(msg: string): boolean {
  const lowered = msg.toLowerCase();
  return (
    lowered.includes("recvwindow") ||
    lowered.includes("timestamp for this request is outside") ||
    lowered.includes("binance code -1021") ||
    lowered.includes("code -1021")
  );
}
export function App() {
  const [apiOk, setApiOk] = useState<"unknown" | "ok" | "down" | "auth">("unknown");
  const [healthInfo, setHealthInfo] = useState<Awaited<ReturnType<typeof health>> | null>(null);
  const apiComputeLimits = healthInfo?.computeLimits ?? null;
  const [toast, setToast] = useState<string | null>(null);
  const [pageVisible, setPageVisible] = useState(() => (typeof document === "undefined" ? true : !document.hidden));
  const [revealSecrets, setRevealSecrets] = useState(false);
  const [persistSecrets, setPersistSecrets] = useState<boolean>(() => readJson<boolean>(STORAGE_PERSIST_SECRETS_KEY) ?? false);
  const deployApiBaseUrl = TRADER_UI_CONFIG.apiBaseUrl;
  const apiToken = TRADER_UI_CONFIG.apiToken;
  const [stateSyncTargetInput, setStateSyncTargetInput] = useState<string>(() => readLocalString(STORAGE_STATE_SYNC_TARGET_KEY) ?? "");
  const [stateSyncTargetToken, setStateSyncTargetToken] = useState<string>(() => {
    const persisted = readLocalString(STORAGE_STATE_SYNC_TOKEN_KEY) ?? "";
    const session = readSessionString(STORAGE_STATE_SYNC_TOKEN_KEY) ?? "";
    return persistSecrets ? persisted || session : session;
  });
  const [stateSyncIncludeBots, setStateSyncIncludeBots] = useState(true);
  const [stateSyncIncludeCombos, setStateSyncIncludeCombos] = useState(true);
  const [stateSyncChunkBytesInput, setStateSyncChunkBytesInput] = useState(() => String(STATE_SYNC_CHUNK_DEFAULT_BYTES));
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
  const [binanceTenantKey, setBinanceTenantKey] = useState<string | null>(null);
  const [coinbaseTenantKey, setCoinbaseTenantKey] = useState<string | null>(null);
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
    if (typeof document === "undefined") return;
    const onVisibility = () => setPageVisible(!document.hidden);
    onVisibility();
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, []);

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
  const binanceKeyLocalReady = Boolean(binanceApiKey.trim() && binanceApiSecret.trim());
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
  const binancePrivateKeysReady = isBinancePlatform && (binanceKeyLocalReady || keysProvided === true);
  const binancePrivateKeysMissing = isBinancePlatform && !binanceKeyLocalReady && keysProvided === false;
  const binancePrivateKeysUnknown = isBinancePlatform && !binanceKeyLocalReady && keysProvided === null;
  const binanceSignedKeysHint =
    !isBinancePlatform
      ? null
      : binancePrivateKeysMissing
        ? 'Binance API keys missing. Add keys or click "Check keys".'
        : binancePrivateKeysUnknown
          ? 'Add Binance API keys or click "Check keys" to verify server env keys.'
          : null;
  const binancePrivateKeysIssue = useMemo(() => {
    if (!isBinancePlatform) return null;
    if (binancePrivateKeysMissing) {
      return 'Binance API keys missing. Add keys or click "Check keys".';
    }
    if (binancePrivateKeysUnknown) {
      return 'Binance API keys required. Add keys or click "Check keys" to verify server env keys.';
    }
    return null;
  }, [binancePrivateKeysMissing, binancePrivateKeysUnknown, isBinancePlatform]);
  const keysProvidedLabel = keysProvided === null ? "unknown" : keysProvided ? "provided" : "missing";
  const keysSigned = activeKeysStatus?.signed ?? null;
  const keysTradeTest =
    activeKeysStatus && isBinanceKeysStatus(activeKeysStatus) ? activeKeysStatus.tradeTest ?? null : null;
  const keysCheckedAtMs = keys.platform === platform ? keys.checkedAtMs : null;
  const botTradeKeysIssue = useMemo(() => {
    if (!isBinancePlatform || !form.tradeArmed) return null;
    if (binanceKeyLocalReady || keysProvided === true) return null;
    if (keysProvided === false) {
      return "Trading armed requires Binance API keys. Add keys or disable Arm trading for paper mode.";
    }
    return 'Trading armed requires Binance API keys. Add keys or click "Check keys" to verify server env keys, or disable Arm trading for paper mode.';
  }, [binanceKeyLocalReady, form.tradeArmed, isBinancePlatform, keysProvided]);
  const autoKeysCheckRef = useRef(false);
  const autoListenKeyStartRef = useRef(false);
  const binanceTenantKeyFromStatus = useMemo(() => {
    if (keys.platform !== "binance" || !keys.status || !isBinanceKeysStatus(keys.status)) return null;
    const raw = keys.status.tenantKey?.trim();
    return raw ? raw : null;
  }, [keys.platform, keys.status]);
  const coinbaseTenantKeyFromStatus = useMemo(() => {
    if (keys.platform !== "coinbase" || !keys.status || !isCoinbaseKeysStatus(keys.status)) return null;
    const raw = keys.status.tenantKey?.trim();
    return raw ? raw : null;
  }, [keys.platform, keys.status]);
  const binanceTenantKeyResolved = binanceTenantKey ?? binanceTenantKeyFromStatus;
  const coinbaseTenantKeyResolved = coinbaseTenantKey ?? coinbaseTenantKeyFromStatus;
  const activeTenantKey = isBinancePlatform ? binanceTenantKeyResolved : isCoinbasePlatform ? coinbaseTenantKeyResolved : null;
  const normalizedSymbol = form.binanceSymbol.trim().toUpperCase();
  const symbolFormatError = useMemo(() => {
    if (!normalizedSymbol) return null;
    return symbolFormatPattern(platform).test(normalizedSymbol)
      ? null
      : `Symbol must match ${platformLabel} format (e.g., ${symbolFormatExample(platform)}).`;
  }, [normalizedSymbol, platform, platformLabel]);
  const symbolIsCustom = !platformSymbolSet.has(normalizedSymbol);
  const symbolSelectValue = symbolIsCustom ? CUSTOM_SYMBOL_VALUE : normalizedSymbol;
  const botSymbolsInput = useMemo(() => parseSymbolsInput(form.botSymbols), [form.botSymbols]);
  const botSymbolsInvalid = useMemo(() => invalidSymbolsForPlatform("binance", botSymbolsInput), [botSymbolsInput]);
  const botSymbolsFormatError = useMemo(() => {
    if (botSymbolsInvalid.length === 0) return null;
    const invalidList = botSymbolsInvalid.join(", ");
    return `Bot symbols must match Binance format (e.g., ${symbolFormatExample("binance")}). Invalid: ${invalidList}.`;
  }, [botSymbolsInvalid]);
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
  const dataLogPrefsInit = readJson<DataLogPrefs>(STORAGE_DATA_LOG_PREFS_KEY);
  const dataLogPersistInit = dataLogPrefsInit?.persist ?? DEFAULT_DATA_LOG_PREFS.persist;
  const dataLogInit = dataLogPersistInit ? normalizeDataLog(readJson<unknown>(STORAGE_DATA_LOG_KEY)) : [];
  const panelPrefsInit = readJson<PanelPrefs>(STORAGE_PANEL_PREFS_KEY);
  const [panelPrefs, setPanelPrefs] = useState<PanelPrefs>(() => panelPrefsInit ?? {});
  const configPanelOrderInit = readJson<ConfigPanelId[]>(STORAGE_CONFIG_PANEL_ORDER_KEY);
  const [configPanelOrder, setConfigPanelOrder] = useState<ConfigPanelId[]>(() =>
    normalizeConfigPanelOrder(configPanelOrderInit),
  );
  const configPageInit =
    readJson<ConfigPageId | ConfigPanelId>(STORAGE_CONFIG_PAGE_KEY) ?? readJson<ConfigPanelId>(STORAGE_CONFIG_TAB_KEY);
  const [configPage, setConfigPage] = useState<ConfigPageId>(() => normalizeConfigPage(configPageInit));
  const [draggingConfigPanel, setDraggingConfigPanel] = useState<ConfigPanelId | null>(null);
  const [dragOverConfigPanel, setDragOverConfigPanel] = useState<ConfigPanelId | null>(null);
  const [maximizedPanelId, setMaximizedPanelId] = useState<string | null>(null);
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
  const [botStatusOps, setBotStatusOps] = useState<OpsUiState>({
    loading: false,
    error: null,
    enabled: true,
    hint: null,
    ops: [],
    limit: BOT_STATUS_OPS_LIMIT,
    lastFetchedAtMs: null,
  });
  const [opsPerformanceUi, setOpsPerformanceUi] = useState<OpsPerformanceUiState>({
    loading: false,
    error: null,
    enabled: true,
    ready: false,
    commitsReady: false,
    combosReady: false,
    hint: null,
    commits: [],
    combos: [],
    lastFetchedAtMs: null,
  });
  const [stateSyncUi, setStateSyncUi] = useState<StateSyncUiState>({
    exporting: false,
    exportError: null,
    payload: null,
    exportedAtMs: null,
    pushing: false,
    pushError: null,
    pushResult: null,
    pushChunkCount: null,
    pushedAtMs: null,
  });
  const [opsPerformanceCommitLimit, setOpsPerformanceCommitLimit] = useState(40);
  const [opsPerformanceComboLimit, setOpsPerformanceComboLimit] = useState(40);
  const [opsPerformanceComboScope, setOpsPerformanceComboScope] = useState<"latest" | "all">("latest");
  const [opsPerformanceComboOrder, setOpsPerformanceComboOrder] = useState<"delta" | "recent" | "return">("delta");
  const [botStatusStartInput, setBotStatusStartInput] = useState(() => formatDatetimeLocal(Date.now() - 6 * 60 * 60 * 1000));
  const [botStatusEndInput, setBotStatusEndInput] = useState(() => formatDatetimeLocal(Date.now()));

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
  const [binanceTradesFilterSymbolsInput, setBinanceTradesFilterSymbolsInput] = useState("");
  const [binanceTradesFilterSide, setBinanceTradesFilterSide] = useState<OrderSideFilter>("ALL");
  const [binanceTradesFilterStartInput, setBinanceTradesFilterStartInput] = useState("");
  const [binanceTradesFilterEndInput, setBinanceTradesFilterEndInput] = useState("");

  const [binancePositionsUi, setBinancePositionsUi] = useState<BinancePositionsUiState>({
    loading: false,
    error: null,
    response: null,
  });
  const [binancePositionsBars, setBinancePositionsBars] = useState(200);
  const binancePositionsAutoKeyRef = useRef<string | null>(null);

  const [botRtByKey, setBotRtByKey] = useState<Record<string, BotRtUiState>>({});
  const emptyBotRt = useMemo(() => emptyBotRtState(), []);
  const botRtRef = useRef<Record<string, BotRtTracker>>({});
  const botStatusFetchedRef = useRef(false);
  const botStatusTailRef = useRef(BOT_STATUS_TAIL_POINTS);
  const botStatusOpsLimitRef = useRef(BOT_STATUS_OPS_LIMIT);
  const botStatusOpsSinceRef = useRef<number | null>(null);
  const opsPerformanceAbortRef = useRef<AbortController | null>(null);
  const opsPerformanceInFlightRef = useRef(false);
  const botAutoStartSuppressedRef = useRef(false);
  const botAutoStartRef = useRef<{ lastAttemptAtMs: number }>({ lastAttemptAtMs: 0 });

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
  const listenKeyStreamAbortRef = useRef<AbortController | null>(null);
  const listenKeyStreamSeqRef = useRef(0);

  const [orderFilterText, setOrderFilterText] = useState(() => orderPrefsInit?.filterText ?? "");
  const [orderSentOnly, setOrderSentOnly] = useState(() => orderPrefsInit?.sentOnly ?? false);
  const [orderErrorsOnly, setOrderErrorsOnly] = useState(() => orderPrefsInit?.errorsOnly ?? false);
  const [orderSideFilter, setOrderSideFilter] = useState<OrderSideFilter>(() => orderPrefsInit?.side ?? "ALL");
  const [orderLimit, setOrderLimit] = useState(() => orderPrefsInit?.limit ?? 200);
  const [orderShowOrderId, setOrderShowOrderId] = useState(() => orderPrefsInit?.showOrderId ?? false);
  const [orderShowStatus, setOrderShowStatus] = useState(() => orderPrefsInit?.showStatus ?? false);
  const [orderShowClientOrderId, setOrderShowClientOrderId] = useState(() => orderPrefsInit?.showClientOrderId ?? false);
  const [selectedOrderKey, setSelectedOrderKey] = useState<string | null>(null);

  const [dataLog, setDataLog] = useState<DataLogEntry[]>(() => dataLogInit);
  const [dataLogExpanded, setDataLogExpanded] = useState(
    () => dataLogPrefsInit?.expanded ?? DEFAULT_DATA_LOG_PREFS.expanded,
  );
  const [dataLogIndexArrays, setDataLogIndexArrays] = useState(
    () => dataLogPrefsInit?.indexArrays ?? DEFAULT_DATA_LOG_PREFS.indexArrays,
  );
  const [dataLogFilterText, setDataLogFilterText] = useState("");
  const [dataLogAutoScroll, setDataLogAutoScroll] = useState(
    () => dataLogPrefsInit?.autoScroll ?? DEFAULT_DATA_LOG_PREFS.autoScroll,
  );
  const [dataLogLogBackground, setDataLogLogBackground] = useState(
    () => dataLogPrefsInit?.logBackground ?? DEFAULT_DATA_LOG_PREFS.logBackground,
  );
  const [dataLogLogErrors, setDataLogLogErrors] = useState(
    () => dataLogPrefsInit?.logErrors ?? DEFAULT_DATA_LOG_PREFS.logErrors,
  );
  const [dataLogPersist, setDataLogPersist] = useState(() => dataLogPersistInit);
  const [optimizerRunDirty, setOptimizerRunDirty] = useState(false);
  const [optimizerRunForm, setOptimizerRunForm] = useState<OptimizerRunForm>(() => buildDefaultOptimizerRunForm(form.binanceSymbol, platform));
  const [optimizerRunUi, setOptimizerRunUi] = useState<OptimizerRunUiState>({
    loading: false,
    error: null,
    response: null,
    lastRunAtMs: null,
  });
  const [topCombosAll, setTopCombosAll] = useState<OptimizationCombo[]>([]);
  const [comboOrder, setComboOrder] = useState<ComboOrder>("annualized-equity");
  const [comboMinEquityInput, setComboMinEquityInput] = useState("");
  const [comboSymbolFilterInput, setComboSymbolFilterInput] = useState("");
  const [comboMarketFilter, setComboMarketFilter] = useState<ComboMarketFilter>("all");
  const [comboIntervalFilter, setComboIntervalFilter] = useState<string>("all");
  const [comboMethodFilter, setComboMethodFilter] = useState<Method | "all">("all");
  const comboMinEquity = useMemo(() => {
    const trimmed = comboMinEquityInput.trim();
    if (!trimmed) return null;
    const parsed = numFromInput(trimmed, Number.NaN);
    return Number.isFinite(parsed) ? parsed : null;
  }, [comboMinEquityInput]);
  const comboSymbolFilters = useMemo(() => parseSymbolsInput(comboSymbolFilterInput), [comboSymbolFilterInput]);
  const comboSymbolFilterSet = useMemo(() => new Set(comboSymbolFilters), [comboSymbolFilters]);
  const optimizerRunExtras = useMemo(() => parseOptimizerExtras(optimizerRunForm.extraJson), [optimizerRunForm.extraJson]);
  const optimizerRunValidationError = useMemo(() => {
    if (optimizerRunExtras.error) return optimizerRunExtras.error;
    const extras = optimizerRunExtras.value ?? {};
    const extraSourceRaw = typeof extras.source === "string" ? extras.source.trim().toLowerCase() : "";
    const source =
      extraSourceRaw === "binance" ||
      extraSourceRaw === "coinbase" ||
      extraSourceRaw === "kraken" ||
      extraSourceRaw === "poloniex" ||
      extraSourceRaw === "csv"
        ? (extraSourceRaw as OptimizerSource)
        : optimizerRunForm.source;
    const dataPath = optimizerRunForm.dataPath.trim();
    const extraData = typeof extras.data === "string" ? extras.data.trim() : "";
    const symbol = optimizerRunForm.symbol.trim();
    const extraSymbol = typeof extras.binanceSymbol === "string" ? extras.binanceSymbol.trim() : "";
    const high = optimizerRunForm.highColumn.trim() || (typeof extras.highColumn === "string" ? extras.highColumn.trim() : "");
    const low = optimizerRunForm.lowColumn.trim() || (typeof extras.lowColumn === "string" ? extras.lowColumn.trim() : "");
    const intervals =
      typeof extras.intervals === "string" ? extras.intervals.trim() : optimizerRunForm.intervals.trim();
    const lookbackWindow =
      typeof extras.lookbackWindow === "string" ? extras.lookbackWindow.trim() : optimizerRunForm.lookbackWindow.trim();
    const trials = typeof extras.trials === "number" ? extras.trials : parseOptionalInt(optimizerRunForm.trials);
    const timeoutSec =
      typeof extras.timeoutSec === "number" ? extras.timeoutSec : parseOptionalNumber(optimizerRunForm.timeoutSec);
    const barsMin = typeof extras.barsMin === "number" ? extras.barsMin : parseOptionalInt(optimizerRunForm.barsMin);
    const barsMax = typeof extras.barsMax === "number" ? extras.barsMax : parseOptionalInt(optimizerRunForm.barsMax);
    const backtestRatio =
      typeof extras.backtestRatio === "number" ? extras.backtestRatio : parseOptionalNumber(optimizerRunForm.backtestRatio);
    const tuneRatio = typeof extras.tuneRatio === "number" ? extras.tuneRatio : parseOptionalNumber(optimizerRunForm.tuneRatio);

    if (source === "csv") {
      if (!dataPath && !extraData) return "CSV source requires a data path.";
      if ((high && !low) || (!high && low)) return "Provide both High/Low columns or leave both empty.";
    } else if (!symbol && !extraSymbol) {
      return "Symbol is required for exchange sources.";
    }
    if (backtestRatio != null && (backtestRatio < 0 || backtestRatio >= 1)) {
      return "Backtest ratio must be between 0 and 1.";
    }
    if (tuneRatio != null && (tuneRatio < 0 || tuneRatio >= 1)) {
      return "Tune ratio must be between 0 and 1.";
    }
    if (backtestRatio != null && tuneRatio != null && backtestRatio + tuneRatio >= 1) {
      return "Backtest ratio + tune ratio must be < 1.";
    }
    if (trials != null && trials > OPTIMIZER_UI_MAX_TRIALS) {
      return `Trials must be <= ${OPTIMIZER_UI_MAX_TRIALS} (UI guardrail).`;
    }
    if (timeoutSec != null && timeoutSec > OPTIMIZER_UI_MAX_TIMEOUT_SEC) {
      return `TimeoutSec must be <= ${OPTIMIZER_UI_MAX_TIMEOUT_SEC} (UI guardrail).`;
    }
    if (barsMin != null && barsMin > OPTIMIZER_UI_MAX_BARS) {
      return `Bars min exceeds ${OPTIMIZER_UI_MAX_BARS} (UI guardrail).`;
    }
    if (barsMax != null && barsMax > OPTIMIZER_UI_MAX_BARS) {
      return `Bars max exceeds ${OPTIMIZER_UI_MAX_BARS} (UI guardrail).`;
    }
    if (lookbackWindow && intervals) {
      const lookbackSec = parseDurationSeconds(lookbackWindow);
      if (lookbackSec && lookbackSec > 0) {
        const intervalList = intervals.split(/[,\s]+/).map((v) => v.trim()).filter(Boolean);
        for (const interval of intervalList) {
          const intervalSec =
            source === "csv" ? parseDurationSeconds(interval) : platformIntervalSeconds(platform, interval);
          if (!intervalSec || intervalSec <= 0) continue;
          const bars = Math.ceil(lookbackSec / intervalSec);
          if (bars > OPTIMIZER_UI_MAX_BARS) {
            return `LookbackWindow requires ${bars} bars at ${interval}, exceeding ${OPTIMIZER_UI_MAX_BARS} (UI guardrail).`;
          }
        }
      }
    }
    return null;
  }, [
    optimizerRunExtras.error,
    optimizerRunExtras.value,
    optimizerRunForm.backtestRatio,
    optimizerRunForm.barsMax,
    optimizerRunForm.barsMin,
    optimizerRunForm.dataPath,
    optimizerRunForm.highColumn,
    optimizerRunForm.intervals,
    optimizerRunForm.lowColumn,
    optimizerRunForm.lookbackWindow,
    optimizerRunForm.source,
    optimizerRunForm.symbol,
    optimizerRunForm.timeoutSec,
    optimizerRunForm.tuneRatio,
    optimizerRunForm.trials,
    platform,
  ]);
  const optimizerRunRecordJson = useMemo(() => {
    if (!optimizerRunUi.response) return null;
    try {
      return JSON.stringify(optimizerRunUi.response.lastRecord, null, 2);
    } catch {
      return String(optimizerRunUi.response.lastRecord ?? "");
    }
  }, [optimizerRunUi.response]);
  const comboFilterOptions = useMemo(() => {
    const symbols = new Set<string>();
    const markets = new Set<ComboMarketValue>();
    const intervals = new Set<string>();
    const methods = new Set<Method>();
    for (const combo of topCombosAll) {
      const symbol = normalizeSymbolKey(combo.params.binanceSymbol ?? "");
      if (symbol) symbols.add(symbol);
      markets.add(comboMarketValue(combo));
      const interval = combo.params.interval?.trim();
      if (interval) intervals.add(interval);
      methods.add(combo.params.method);
    }
    const intervalList = Array.from(intervals).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    const symbolList = Array.from(symbols).sort((a, b) => a.localeCompare(b));
    const methodOrder: Method[] = ["11", "10", "01", "blend", "router"];
    const methodList = methodOrder.filter((method) => methods.has(method));
    const marketOrder: ComboMarketValue[] = [...PLATFORMS, "csv", "unknown"];
    const marketList = marketOrder.filter((market) => markets.has(market));
    return { symbols: symbolList, markets: marketList, intervals: intervalList, methods: methodList };
  }, [topCombosAll]);
  const topCombosFiltered = useMemo(() => {
    return topCombosAll.filter((combo) => {
      if (comboMinEquity != null && !(combo.finalEquity > comboMinEquity)) return false;
      if (comboSymbolFilterSet.size > 0) {
        const symbol = normalizeSymbolKey(combo.params.binanceSymbol ?? "");
        if (!symbol || !comboSymbolFilterSet.has(symbol)) return false;
      }
      if (comboMarketFilter !== "all") {
        const market = comboMarketValue(combo);
        if (market !== comboMarketFilter) return false;
      }
      if (comboIntervalFilter !== "all" && combo.params.interval !== comboIntervalFilter) return false;
      if (comboMethodFilter !== "all" && combo.params.method !== comboMethodFilter) return false;
      return true;
    });
  }, [
    comboIntervalFilter,
    comboMarketFilter,
    comboMethodFilter,
    comboMinEquity,
    comboSymbolFilterSet,
    topCombosAll,
  ]);
  const topCombosOrdered = useMemo(() => {
    if (comboOrder === "rank") return topCombosFiltered;
    const sorted = [...topCombosFiltered];
    if (comboOrder === "annualized-equity") {
      sorted.sort((a, b) => {
        const aRating = comboAnnualizedEquity(a) ?? a.finalEquity;
        const bRating = comboAnnualizedEquity(b) ?? b.finalEquity;
        const diff = bRating - aRating;
        if (diff !== 0) return diff;
        const eqDiff = b.finalEquity - a.finalEquity;
        if (eqDiff !== 0) return eqDiff;
        return a.id - b.id;
      });
      return sorted;
    }
    sorted.sort((a, b) => {
      const aMs = typeof a.createdAtMs === "number" && Number.isFinite(a.createdAtMs) ? a.createdAtMs : 0;
      const bMs = typeof b.createdAtMs === "number" && Number.isFinite(b.createdAtMs) ? b.createdAtMs : 0;
      if (aMs === bMs) return a.id - b.id;
      return comboOrder === "date-desc" ? bMs - aMs : aMs - bMs;
    });
    return sorted;
  }, [comboOrder, topCombosFiltered]);
  const topComboBotTargets = useMemo(() => {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const combo of topCombosOrdered) {
      const platform = combo.params.platform ?? (combo.source && combo.source !== "csv" ? combo.source : null);
      if (platform && platform !== "binance") continue;
      const rawSymbol = combo.params.binanceSymbol ?? "";
      const normalized = normalizeSymbolKey(rawSymbol);
      if (!normalized) continue;
      if (seen.has(normalized)) continue;
      seen.add(normalized);
      out.push(normalized);
      if (out.length >= TOP_COMBOS_BOT_TARGET) break;
    }
    return out;
  }, [topCombosOrdered]);
  const topComboBotTargetsKey = useMemo(() => topComboBotTargets.join("|"), [topComboBotTargets]);
  const [topCombosDisplayCount, setTopCombosDisplayCount] = useState(() => TOP_COMBOS_DISPLAY_DEFAULT);
  const topCombos = useMemo(
    () => topCombosOrdered.slice(0, topCombosDisplayCount),
    [topCombosOrdered, topCombosDisplayCount],
  );
  const deferredTopCombos = useDeferredValue(topCombos);
  const [topCombosLoading, setTopCombosLoading] = useState(true);
  const [topCombosError, setTopCombosError] = useState<string | null>(null);
  const [topCombosMeta, setTopCombosMeta] = useState<TopCombosMeta>({
    source: "api",
    generatedAtMs: null,
    payloadSource: null,
    payloadSources: null,
    fallbackReason: null,
    comboCount: null,
  });
  const [topCombosPayload, setTopCombosPayload] = useState<Record<string, unknown> | null>(null);
  const [comboImportUi, setComboImportUi] = useState<ComboImportUiState>(() => ({
    text: "",
    payload: null,
    summary: null,
    parseError: null,
    importError: null,
    importing: false,
    importResult: null,
    importedAtMs: null,
    stampNow: false,
  }));
  const topCombosDisplayMax = useMemo(() => {
    const count = topCombosFiltered.length;
    if (count && count > 0) return Math.max(TOP_COMBOS_DISPLAY_MIN, count);
    return Math.max(TOP_COMBOS_DISPLAY_MIN, TOP_COMBOS_DISPLAY_DEFAULT);
  }, [topCombosFiltered.length]);
  const comboHasFilters =
    comboMinEquity != null ||
    comboSymbolFilters.length > 0 ||
    comboMarketFilter !== "all" ||
    comboIntervalFilter !== "all" ||
    comboMethodFilter !== "all";
  const comboExportJson = useMemo(() => {
    if (!topCombosPayload) return "";
    try {
      return JSON.stringify(topCombosPayload, null, 2);
    } catch {
      return "";
    }
  }, [topCombosPayload]);
  const comboExportReady = comboExportJson.length > 0;
  const comboExportFilename = useMemo(() => {
    const stamp = formatDatetimeLocal(topCombosMeta.generatedAtMs ?? Date.now());
    const safeStamp = sanitizeFilenameSegment(stamp, "latest");
    return `top-combos-${safeStamp}.json`;
  }, [topCombosMeta.generatedAtMs]);
  const comboImportReady = useMemo(
    () => Boolean(comboImportUi.payload) && !comboImportUi.parseError,
    [comboImportUi.parseError, comboImportUi.payload],
  );
  const comboImportDisabledReason = useMemo(() => {
    if (!activeTenantKey) return "Tenant key required. Add API keys and check keys.";
    if (comboImportUi.parseError) return comboImportUi.parseError;
    if (!comboImportUi.payload) return "Parse combos JSON first.";
    return null;
  }, [activeTenantKey, comboImportUi.parseError, comboImportUi.payload]);
  const [autoAppliedCombo, setAutoAppliedCombo] = useState<{ id: number; atMs: number } | null>(null);
  const autoAppliedComboRef = useRef<{ id: number | null; atMs: number | null }>({ id: null, atMs: null });
  const [autoApplyTopCombo, setAutoApplyTopCombo] = useState(false);
  const [selectedComboId, setSelectedComboId] = useState<number | null>(null);
  const [pendingComboStart, setPendingComboStart] = useState<{
    signature: string;
    symbols: string[];
  } | null>(null);
  const botAutoStartTopCombosRef = useRef<{ lastAttemptAtMs: number; lastKey: string }>({
    lastAttemptAtMs: 0,
    lastKey: "",
  });
  const pendingComboStartRef = useRef(pendingComboStart);
  const topCombosRef = useRef<OptimizationCombo[]>([]);
  const topCombosSyncRef = useRef<((opts?: { silent?: boolean }) => void) | null>(null);
  const dataLogRef = useRef<HTMLDivElement | null>(null);
  const dataLogPinnedRef = useRef(true);
  const sectionFlashRef = useRef<HTMLElement | null>(null);
  const sectionFlashTimeoutRef = useRef<number | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const botAbortRef = useRef<AbortController | null>(null);
  const keysAbortRef = useRef<AbortController | null>(null);
  const optimizerRunAbortRef = useRef<AbortController | null>(null);
  const botStatusOpsAbortRef = useRef<AbortController | null>(null);
  const topCombosAbortRef = useRef<AbortController | null>(null);
  const binanceTradesAbortRef = useRef<AbortController | null>(null);
  const binancePositionsAbortRef = useRef<AbortController | null>(null);
  const botStatusOpsInFlightRef = useRef(false);
  const requestSeqRef = useRef(0);
  const botRequestSeqRef = useRef(0);
  const keysRequestSeqRef = useRef(0);
  const optimizerRunRequestSeqRef = useRef(0);
  const errorRef = useRef<HTMLDetailsElement>(null!);
  const signalRef = useRef<HTMLDetailsElement>(null!);
  const backtestRef = useRef<HTMLDetailsElement>(null!);
  const tradeRef = useRef<HTMLDetailsElement>(null!);

  useEffect(() => {
    writeJson(STORAGE_KEY, form);
  }, [form]);

  const syncOptimizerRunForm = useCallback(() => {
    setOptimizerRunForm((prev) => ({
      ...prev,
      source: prev.source === "csv" ? "csv" : optimizerSourceForPlatform(form.platform),
      symbol: form.binanceSymbol.trim().toUpperCase(),
      intervals: form.interval.trim() || prev.intervals,
      lookbackWindow: form.lookbackWindow.trim() || prev.lookbackWindow,
    }));
    setOptimizerRunDirty(false);
  }, [form.binanceSymbol, form.interval, form.lookbackWindow, form.platform]);

  useEffect(() => {
    if (optimizerRunDirty) return;
    syncOptimizerRunForm();
  }, [form.binanceSymbol, form.interval, form.lookbackWindow, form.platform, optimizerRunDirty, syncOptimizerRunForm]);

  useEffect(() => {
    pendingComboStartRef.current = pendingComboStart;
  }, [pendingComboStart]);

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
    writeJson(STORAGE_DATA_LOG_PREFS_KEY, {
      expanded: dataLogExpanded,
      indexArrays: dataLogIndexArrays,
      autoScroll: dataLogAutoScroll,
      logBackground: dataLogLogBackground,
      logErrors: dataLogLogErrors,
      persist: dataLogPersist,
    } satisfies DataLogPrefs);
  }, [dataLogAutoScroll, dataLogExpanded, dataLogIndexArrays, dataLogLogBackground, dataLogLogErrors, dataLogPersist]);

  useEffect(() => {
    if (!dataLogPersist) {
      removeLocalKey(STORAGE_DATA_LOG_KEY);
      return;
    }
    writeJson(STORAGE_DATA_LOG_KEY, dataLog);
  }, [dataLog, dataLogPersist]);

  useEffect(() => {
    writeJson(STORAGE_PANEL_PREFS_KEY, panelPrefs);
  }, [panelPrefs]);

  useEffect(() => {
    if (typeof document === "undefined") return;
    document.body.classList.toggle("panelMaximized", Boolean(maximizedPanelId));
    return () => {
      document.body.classList.remove("panelMaximized");
    };
  }, [maximizedPanelId]);

  useEffect(() => {
    if (!maximizedPanelId || typeof window === "undefined") return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMaximizedPanelId(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [maximizedPanelId]);

  useEffect(() => {
    if (!maximizedPanelId || typeof document === "undefined") return;
    const panel = document.querySelector(`[data-panel="${maximizedPanelId}"]`);
    if (!panel) setMaximizedPanelId(null);
  });

  useEffect(() => {
    writeJson(STORAGE_CONFIG_PANEL_ORDER_KEY, configPanelOrder);
  }, [configPanelOrder]);
  useEffect(() => {
    writeJson(STORAGE_CONFIG_PAGE_KEY, configPage);
  }, [configPage]);

  const setPanelOpen = useCallback((panelId: string, open: boolean) => {
    setPanelPrefs((prev) => (prev[panelId] === open ? prev : { ...prev, [panelId]: open }));
  }, []);

  const isPanelOpen = useCallback(
    (panelId: string, defaultOpen: boolean) => {
      const stored = panelPrefs[panelId];
      return typeof stored === "boolean" ? stored : defaultOpen;
    },
    [panelPrefs],
  );
  const selectConfigPage = useCallback(
    (pageId: ConfigPageId) => {
      setConfigPage(pageId);
      setPanelOpen("panel-config", true);
      const panelId = CONFIG_PAGE_PANEL_MAP[pageId];
      setPanelOpen(panelId, true);
      setPanelOpen(pageId, true);
    },
    [setPanelOpen],
  );

  const handlePanelToggle = useCallback(
    (panelId: string) => (event: React.SyntheticEvent<HTMLDetailsElement>) => {
      const nextOpen = event.currentTarget.open;
      setPanelOpen(panelId, nextOpen);
      if (!nextOpen) {
        setMaximizedPanelId((prev) => (prev === panelId ? null : prev));
      }
    },
    [setPanelOpen],
  );

  const togglePanelMaximize = useCallback(
    (panelId: string) => {
      setPanelOpen(panelId, true);
      setMaximizedPanelId((prev) => (prev === panelId ? null : panelId));
    },
    [setPanelOpen],
  );

  const reorderConfigPanels = useCallback((sourceId: ConfigPanelId, targetId: ConfigPanelId) => {
    setConfigPanelOrder((prev) => {
      if (sourceId === targetId) return prev;
      const fromIndex = prev.indexOf(sourceId);
      const toIndex = prev.indexOf(targetId);
      if (fromIndex < 0 || toIndex < 0) return prev;
      const next = [...prev];
      next.splice(fromIndex, 1);
      next.splice(toIndex, 0, sourceId);
      return next;
    });
  }, []);

  const handleConfigPanelDragStart = useCallback(
    (panelId: ConfigPanelId) => (event: React.DragEvent<HTMLButtonElement>) => {
      setDraggingConfigPanel(panelId);
      setDragOverConfigPanel(null);
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", panelId);
    },
    [],
  );

  const handleConfigPanelDragOver = useCallback(
    (panelId: ConfigPanelId) => (event: React.DragEvent<HTMLElement>) => {
      if (!draggingConfigPanel || draggingConfigPanel === panelId) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      if (dragOverConfigPanel !== panelId) {
        setDragOverConfigPanel(panelId);
      }
    },
    [draggingConfigPanel, dragOverConfigPanel],
  );

  const handleConfigPanelDrop = useCallback(
    (panelId: ConfigPanelId) => (event: React.DragEvent<HTMLElement>) => {
      event.preventDefault();
      const sourceIdRaw = draggingConfigPanel ?? event.dataTransfer.getData("text/plain");
      const sourceId = CONFIG_PANEL_IDS.includes(sourceIdRaw as ConfigPanelId)
        ? (sourceIdRaw as ConfigPanelId)
        : null;
      if (!sourceId || sourceId === panelId) {
        setDragOverConfigPanel(null);
        setDraggingConfigPanel(null);
        return;
      }
      reorderConfigPanels(sourceId, panelId);
      setDragOverConfigPanel(null);
      setDraggingConfigPanel(null);
    },
    [draggingConfigPanel, reorderConfigPanels],
  );

  const handleConfigPanelDragEnd = useCallback(() => {
    setDragOverConfigPanel(null);
    setDraggingConfigPanel(null);
  }, []);

  const openPanelAncestors = useCallback(
    (el: HTMLElement) => {
      let current: HTMLElement | null = el;
      let opened = false;
      while (current) {
        const details = current.closest("details[data-panel]") as HTMLDetailsElement | null;
        if (!details) return opened;
        const panelId = details.getAttribute("data-panel");
        if (panelId && !details.open) {
          setPanelOpen(panelId, true);
          opened = true;
        }
        current = details.parentElement;
      }
      return opened;
    },
    [setPanelOpen],
  );

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
    let cancelled = false;
    const key = binanceApiKey.trim();
    const secret = binanceApiSecret.trim();
    if (!key || !secret) {
      setBinanceTenantKey(null);
      return () => {
        cancelled = true;
      };
    }
    void (async () => {
      try {
        const next = await buildTenantKey("binance", key, secret);
        if (!cancelled) setBinanceTenantKey(next);
      } catch {
        if (!cancelled) setBinanceTenantKey(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [binanceApiKey, binanceApiSecret]);

  useEffect(() => {
    let cancelled = false;
    const key = coinbaseApiKey.trim();
    const secret = coinbaseApiSecret.trim();
    const passphrase = coinbaseApiPassphrase.trim();
    if (!key || !secret || !passphrase) {
      setCoinbaseTenantKey(null);
      return () => {
        cancelled = true;
      };
    }
    void (async () => {
      try {
        const next = await buildTenantKey("coinbase", key, secret, passphrase);
        if (!cancelled) setCoinbaseTenantKey(next);
      } catch {
        if (!cancelled) setCoinbaseTenantKey(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [coinbaseApiKey, coinbaseApiPassphrase, coinbaseApiSecret]);

  useEffect(() => {
    const v = stateSyncTargetToken.trim();
    if (persistSecrets) {
      if (!v) removeLocalKey(STORAGE_STATE_SYNC_TOKEN_KEY);
      else writeLocalString(STORAGE_STATE_SYNC_TOKEN_KEY, v);
      removeSessionKey(STORAGE_STATE_SYNC_TOKEN_KEY);
    } else {
      if (!v) removeSessionKey(STORAGE_STATE_SYNC_TOKEN_KEY);
      else writeSessionString(STORAGE_STATE_SYNC_TOKEN_KEY, v);
      removeLocalKey(STORAGE_STATE_SYNC_TOKEN_KEY);
    }
  }, [persistSecrets, stateSyncTargetToken]);

  useEffect(() => {
    const v = stateSyncTargetInput.trim();
    if (!v) removeLocalKey(STORAGE_STATE_SYNC_TARGET_KEY);
    else writeLocalString(STORAGE_STATE_SYNC_TARGET_KEY, v);
  }, [stateSyncTargetInput]);

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
  const handleComboApply = useCallback(
    (combo: OptimizationCombo) => {
      const baseForm = formRef.current;
      const manualOverrides = manualOverridesRef.current;
      const nextForm = applyComboToForm(baseForm, combo, apiComputeLimitsRef.current, manualOverrides, true);
      const nextSignature = formApplySignature(nextForm);
      const symbols = parseSymbolsInput(nextForm.binanceSymbol);
      pendingComboStartRef.current = { signature: nextSignature, symbols };
      setPendingComboStart({ signature: nextSignature, symbols });
      applyCombo(combo, { respectManual: true, allowPositioning: true });
    },
    [applyCombo],
  );
  const handleComboStart = useCallback(
    (combo: OptimizationCombo) => {
      const baseForm = formRef.current;
      const nextForm = applyComboToForm(baseForm, combo, apiComputeLimitsRef.current, undefined, true);
      const nextSignature = formApplySignature(nextForm);
      const symbols = parseSymbolsInput(nextForm.binanceSymbol);
      pendingComboStartRef.current = { signature: nextSignature, symbols };
      setPendingComboStart({ signature: nextSignature, symbols });
      applyCombo(combo, { respectManual: false, allowPositioning: true });
    },
    [applyCombo],
  );
  const refreshTopCombos = useCallback(() => {
    topCombosSyncRef.current?.({ silent: false });
  }, []);
  const updateOptimizerRunForm = useCallback((updates: Partial<OptimizerRunForm>) => {
    setOptimizerRunDirty(true);
    setOptimizerRunForm((prev) => ({ ...prev, ...updates }));
  }, []);
  const resetOptimizerRunForm = useCallback(() => {
    setOptimizerRunForm(buildDefaultOptimizerRunForm(form.binanceSymbol, platform));
    setOptimizerRunUi((prev) => ({ ...prev, error: null }));
    setOptimizerRunDirty(false);
  }, [form.binanceSymbol, platform]);
  const syncOptimizerRunSymbolInterval = useCallback(() => {
    setOptimizerRunDirty(true);
    setOptimizerRunForm((prev) => ({
      ...prev,
      symbol: form.binanceSymbol.trim().toUpperCase(),
      intervals: form.interval.trim() || prev.intervals,
    }));
  }, [form.binanceSymbol, form.interval]);
  const applyEquityPreset = useCallback(() => {
    setOptimizerRunDirty(true);
    setOptimizerRunForm((prev) => ({
      ...prev,
      objective: "annualized-equity",
      tuneObjective: "annualized-equity",
      trials: "100",
      timeoutSec: "120",
      minRoundTrips: "5",
      walkForwardEmbargoBarsMin: "1",
      walkForwardEmbargoBarsMax: "1",
      rebalanceCostMultMin: "1",
      rebalanceCostMultMax: "1",
    }));
    showToast("Preset: annualized equity focus");
  }, [showToast]);
  const cancelOptimizerRun = useCallback(() => {
    if (!optimizerRunUi.loading) return;
    optimizerRunRequestSeqRef.current += 1;
    optimizerRunAbortRef.current?.abort();
    optimizerRunAbortRef.current = null;
    setOptimizerRunUi((prev) => ({ ...prev, loading: false }));
    showToast("Optimizer run cancelled");
  }, [optimizerRunUi.loading, showToast]);
  const authHeaders = useMemo(() => {
    const token = apiToken.trim();
    if (!token) return undefined;
    return { "X-API-Key": token };
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

  const stateSyncTargetBase = useMemo(() => normalizeApiBaseUrlInput(stateSyncTargetInput), [stateSyncTargetInput]);
  const stateSyncTargetError = useMemo(() => {
    const candidate = stateSyncTargetBase.trim();
    if (!candidate) return null;
    if (candidate.startsWith("/")) return null;
    if (!/^https?:\/\//i.test(candidate)) return "Target API base must start with http(s):// or a /path.";
    try {
      new URL(candidate);
    } catch {
      return "Target API base must be a valid URL (e.g., https://your-api-host).";
    }
    return null;
  }, [stateSyncTargetBase]);
  const stateSyncTargetAbsolute = useMemo(() => {
    const candidate = stateSyncTargetBase.trim();
    if (!candidate) return "";
    if (/^https?:\/\//i.test(candidate)) return candidate;
    if (typeof window !== "undefined") return `${window.location.origin}${candidate}`;
    return candidate;
  }, [stateSyncTargetBase]);
  const stateSyncTargetCrossOrigin = useMemo(() => {
    if (!stateSyncTargetAbsolute || typeof window === "undefined") return false;
    if (!/^https?:\/\//i.test(stateSyncTargetAbsolute)) return false;
    try {
      return new URL(stateSyncTargetAbsolute).origin !== window.location.origin;
    } catch {
      return false;
    }
  }, [stateSyncTargetAbsolute]);
  const stateSyncTargetCorsHint = useMemo(() => {
    if (!stateSyncTargetCrossOrigin) return null;
    return "Target API is cross-origin; it must allow this UI origin (CORS) for /state/sync.";
  }, [stateSyncTargetCrossOrigin]);
  const stateSyncTargetHeaders = useMemo(() => {
    const token = stateSyncTargetToken.trim();
    return token ? { "X-API-Key": token } : undefined;
  }, [stateSyncTargetToken]);
  const stateSyncTargetMissing = !stateSyncTargetBase.trim();
  const stateSyncChunkBytes = useMemo(() => {
    const parsed = numFromInput(stateSyncChunkBytesInput, STATE_SYNC_CHUNK_DEFAULT_BYTES);
    if (!Number.isFinite(parsed)) return STATE_SYNC_CHUNK_DEFAULT_BYTES;
    if (parsed <= 0) return 0;
    return Math.trunc(Math.min(parsed, STATE_SYNC_CHUNK_MAX_BYTES));
  }, [stateSyncChunkBytesInput]);
  const stateSyncSelectionEmpty = !stateSyncIncludeBots && !stateSyncIncludeCombos;
  const stateSyncPayloadJson = useMemo(
    () => (stateSyncUi.payload ? JSON.stringify(stateSyncUi.payload, null, 2) : ""),
    [stateSyncUi.payload],
  );
  const stateSyncPayloadSummary = useMemo(() => {
    const payload = stateSyncUi.payload;
    if (!payload) return null;
    const botCount = payload.botSnapshots?.length ?? 0;
    let comboCount: number | null = null;
    let comboGeneratedAt: number | null = null;
    const topCombos = payload.topCombos;
    if (topCombos && typeof topCombos === "object") {
      const topCombosRecord = topCombos as { combos?: unknown; generatedAtMs?: unknown };
      comboCount = Array.isArray(topCombosRecord.combos) ? topCombosRecord.combos.length : null;
      comboGeneratedAt =
        typeof topCombosRecord.generatedAtMs === "number" && Number.isFinite(topCombosRecord.generatedAtMs)
          ? topCombosRecord.generatedAtMs
          : null;
    }
    return {
      botCount,
      comboCount,
      generatedAtMs: payload.generatedAtMs ?? comboGeneratedAt ?? null,
    };
  }, [stateSyncUi.payload]);
  const stateSyncChunkPlan = useMemo(() => {
    const payload = stateSyncUi.payload;
    if (!payload) return null;
    const chunks = buildStateSyncChunks(payload, stateSyncChunkBytes);
    const payloadBytes = jsonByteLength(payload);
    const chunkSizes = chunks.map((chunk) => jsonByteLength(chunk));
    const largestChunkBytes = chunkSizes.length > 0 ? Math.max(...chunkSizes) : 0;
    const combosBytes =
      payload.topCombos != null
        ? jsonByteLength({
            ...(payload.generatedAtMs != null ? { generatedAtMs: payload.generatedAtMs } : {}),
            topCombos: payload.topCombos,
          })
        : null;
    return {
      payloadBytes,
      chunkCount: chunks.length,
      largestChunkBytes,
      combosBytes,
    };
  }, [stateSyncUi.payload, stateSyncChunkBytes]);

  const appendDataLog = useCallback(
    (label: string, data: unknown, opts?: { background?: boolean; error?: boolean }) => {
      if (opts?.background && !dataLogLogBackground) return;
      if (opts?.error && !dataLogLogErrors) return;
      setDataLog((logs) => [...logs, { timestamp: Date.now(), label, data }].slice(-DATA_LOG_MAX_ENTRIES));
    },
    [dataLogLogBackground, dataLogLogErrors],
  );

  const buildDataLogError = useCallback((err: unknown, message: string) => {
    const data: Record<string, unknown> = { message };
    if (err instanceof HttpError) {
      data.status = err.status;
      if (err.payload !== undefined) data.payload = err.payload;
      return data;
    }
    if (err instanceof Error) {
      data.name = err.name;
      return data;
    }
    if (err !== undefined) data.payload = err;
    return data;
  }, []);

  const exportStateSync = useCallback(async (): Promise<StateSyncPayload | null> => {
    setStateSyncUi((prev) => ({ ...prev, exporting: true, exportError: null }));
    try {
      if (stateSyncSelectionEmpty) {
        const msg = "Select bot snapshots and/or combos to export.";
        setStateSyncUi((prev) => ({ ...prev, exporting: false, exportError: msg }));
        return null;
      }
      if (!activeTenantKey) {
        const msg = "Tenant key required. Add API keys (or check keys) to export state.";
        setStateSyncUi((prev) => ({ ...prev, exporting: false, exportError: msg }));
        return null;
      }
      const out = await stateSyncExport(apiBase, { headers: authHeaders, timeoutMs: 30_000, tenantKey: activeTenantKey });
      const filtered = pruneStateSyncPayload(out, { includeBots: stateSyncIncludeBots, includeCombos: stateSyncIncludeCombos });
      if (isStateSyncPayloadEmpty(filtered)) {
        const msg = "Nothing to export for the selected payload options.";
        setStateSyncUi((prev) => ({ ...prev, exporting: false, exportError: msg, payload: null, exportedAtMs: null }));
        return null;
      }
      setStateSyncUi((prev) => ({
        ...prev,
        exporting: false,
        exportError: null,
        payload: filtered,
        exportedAtMs: Date.now(),
        pushChunkCount: null,
      }));
      appendDataLog("State Sync Export", filtered);
      setApiOk("ok");
      showToast("State export ready");
      return filtered;
    } catch (e) {
      let msg = e instanceof Error ? e.message : "Failed to export state.";
      if (e instanceof HttpError) {
        if (e.status === 401) msg = "Unauthorized (check apiToken / TRADER_API_TOKEN).";
        if (e.status === 404) msg = "State sync endpoint not available on this API.";
      }
      if (isTimeoutError(e)) msg = "State export timed out.";
      setStateSyncUi((prev) => ({ ...prev, exporting: false, exportError: msg }));
      appendDataLog("State Sync Export Error", buildDataLogError(e, msg), { error: true });
      return null;
    }
  }, [
    activeTenantKey,
    apiBase,
    appendDataLog,
    authHeaders,
    buildDataLogError,
    showToast,
    stateSyncIncludeBots,
    stateSyncIncludeCombos,
    stateSyncSelectionEmpty,
  ]);

  const sendStateSync = useCallback(
    async (payloadOverride: StateSyncPayload | null = null) => {
      const targetBase = stateSyncTargetBase.trim();
      if (stateSyncTargetMissing) {
        setStateSyncUi((prev) => ({ ...prev, pushError: "Enter a target API base URL." }));
        return;
      }
      if (stateSyncTargetError) {
        setStateSyncUi((prev) => ({ ...prev, pushError: stateSyncTargetError }));
        return;
      }
      const payload = payloadOverride ?? stateSyncUi.payload;
      if (!payload) {
        setStateSyncUi((prev) => ({ ...prev, pushError: "Export state first." }));
        return;
      }
      const payloadChunks = buildStateSyncChunks(payload, stateSyncChunkBytes);
      if (payloadChunks.length === 0) {
        setStateSyncUi((prev) => ({ ...prev, pushError: "State sync payload is empty." }));
        return;
      }
      setStateSyncUi((prev) => ({
        ...prev,
        pushing: true,
        pushError: null,
        pushResult: null,
        pushChunkCount: payloadChunks.length,
      }));
      try {
        let lastOut: StateSyncImportResponse | null = null;
        for (const [idx, chunk] of payloadChunks.entries()) {
          try {
            lastOut = await stateSyncImport(targetBase, chunk, {
              headers: stateSyncTargetHeaders,
              timeoutMs: 30_000,
              tenantKey: activeTenantKey ?? undefined,
            });
          } catch (e) {
            let msg = e instanceof Error ? e.message : "Failed to send state sync payload.";
            if (e instanceof HttpError) {
              if (e.status === 401) msg = "Unauthorized (check target API token).";
              if (e.status === 404) msg = "Target API does not expose /state/sync.";
            }
            if (isTimeoutError(e)) msg = "State sync push timed out.";
            if (payloadChunks.length > 1) {
              msg = `Failed after ${idx} of ${payloadChunks.length} payloads. ${msg}`;
            }
            setStateSyncUi((prev) => ({ ...prev, pushing: false, pushError: msg }));
            appendDataLog(
              "State Sync Push Error",
              { ...buildDataLogError(e, msg), chunk: idx + 1, chunks: payloadChunks.length },
              { error: true },
            );
            return;
          }
        }
        setStateSyncUi((prev) => ({
          ...prev,
          pushing: false,
          pushError: null,
          pushResult: lastOut,
          pushedAtMs: Date.now(),
          pushChunkCount: payloadChunks.length,
        }));
        appendDataLog("State Sync Push", {
          target: stateSyncTargetAbsolute || targetBase,
          response: lastOut,
          chunks: payloadChunks.length,
        });
        showToast(payloadChunks.length > 1 ? `State sync sent (${payloadChunks.length} chunks)` : "State sync sent");
      } catch (e) {
        let msg = e instanceof Error ? e.message : "Failed to send state sync payload.";
        if (e instanceof HttpError) {
          if (e.status === 401) msg = "Unauthorized (check target API token).";
          if (e.status === 404) msg = "Target API does not expose /state/sync.";
        }
        if (isTimeoutError(e)) msg = "State sync push timed out.";
        setStateSyncUi((prev) => ({ ...prev, pushing: false, pushError: msg }));
        appendDataLog("State Sync Push Error", buildDataLogError(e, msg), { error: true });
      }
    },
    [
      activeTenantKey,
      appendDataLog,
      buildDataLogError,
      showToast,
      stateSyncChunkBytes,
      stateSyncTargetAbsolute,
      stateSyncTargetBase,
      stateSyncTargetError,
      stateSyncTargetHeaders,
      stateSyncTargetMissing,
      stateSyncUi.payload,
    ],
  );

  const exportAndSendStateSync = useCallback(async () => {
    if (stateSyncTargetMissing) {
      setStateSyncUi((prev) => ({ ...prev, pushError: "Enter a target API base URL." }));
      return;
    }
    if (stateSyncTargetError) {
      setStateSyncUi((prev) => ({ ...prev, pushError: stateSyncTargetError }));
      return;
    }
    const payload = await exportStateSync();
    if (!payload) return;
    await sendStateSync(payload);
  }, [exportStateSync, sendStateSync, stateSyncTargetError, stateSyncTargetMissing]);

  const updateComboImportText = useCallback((text: string) => {
    setComboImportUi((prev) => ({
      ...prev,
      text,
      payload: null,
      summary: null,
      parseError: null,
      importError: null,
      importResult: null,
      importedAtMs: null,
    }));
  }, []);

  const setComboImportStampNow = useCallback((value: boolean) => {
    setComboImportUi((prev) => ({ ...prev, stampNow: value }));
  }, []);

  const parseComboImportText = useCallback(
    (textOverride?: string) => {
      const text = textOverride ?? comboImportUi.text;
      const result = parseTopCombosJson(text);
      setComboImportUi((prev) => ({
        ...prev,
        text,
        payload: result.payload,
        summary: result.summary,
        parseError: result.error,
        importError: null,
        importResult: null,
        importedAtMs: null,
      }));
    },
    [comboImportUi.text],
  );

  const handleComboImportFile = useCallback(
    (file: File | null) => {
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const text = typeof reader.result === "string" ? reader.result : "";
        const result = parseTopCombosJson(text);
        setComboImportUi((prev) => ({
          ...prev,
          text,
          payload: result.payload,
          summary: result.summary,
          parseError: result.error,
          importError: null,
          importResult: null,
          importedAtMs: null,
        }));
      };
      reader.onerror = () => {
        setComboImportUi((prev) => ({
          ...prev,
          text: "",
          payload: null,
          summary: null,
          parseError: "Failed to read file.",
          importError: null,
          importResult: null,
          importedAtMs: null,
        }));
      };
      reader.readAsText(file);
    },
    [],
  );

  const copyComboExport = useCallback(async () => {
    if (!comboExportReady) {
      showToast("No combos to export yet");
      return;
    }
    await copyText(comboExportJson);
    showToast("Copied combos JSON");
  }, [comboExportJson, comboExportReady, showToast]);

  const downloadComboExport = useCallback(() => {
    if (!comboExportReady) {
      showToast("No combos to export yet");
      return;
    }
    downloadTextFile(comboExportFilename, comboExportJson, "application/json");
  }, [comboExportFilename, comboExportJson, comboExportReady, showToast]);

  const importCombos = useCallback(async () => {
    if (!activeTenantKey) {
      setComboImportUi((prev) => ({ ...prev, importError: "Tenant key required. Add API keys and check keys." }));
      return;
    }
    if (!comboImportUi.payload) {
      setComboImportUi((prev) => ({ ...prev, importError: "Parse combos JSON first." }));
      return;
    }
    setComboImportUi((prev) => ({ ...prev, importing: true, importError: null }));
    try {
      const payload =
        comboImportUi.stampNow
          ? { ...comboImportUi.payload, generatedAtMs: Date.now() }
          : comboImportUi.payload;
      const out = await stateSyncImport(
        apiBase,
        { generatedAtMs: Date.now(), topCombos: payload } satisfies StateSyncPayload,
        { headers: authHeaders, timeoutMs: 30_000, tenantKey: activeTenantKey },
      );
      setComboImportUi((prev) => ({
        ...prev,
        importing: false,
        importError: null,
        importResult: out.topCombos ?? null,
        importedAtMs: Date.now(),
      }));
      appendDataLog("Combos Import", { response: out, combos: payload });
      showToast(`Combos import ${out.topCombos?.action ?? "complete"}`);
      refreshTopCombos();
    } catch (e) {
      let msg = e instanceof Error ? e.message : "Failed to import combos.";
      if (e instanceof HttpError) {
        if (e.status === 401) msg = "Unauthorized (check apiToken / TRADER_API_TOKEN).";
        if (e.status === 400) msg = "Invalid combos JSON (expected object with combos array).";
      }
      if (isTimeoutError(e)) msg = "Combos import timed out.";
      setComboImportUi((prev) => ({ ...prev, importing: false, importError: msg }));
      appendDataLog("Combos Import Error", buildDataLogError(e, msg), { error: true });
    }
  }, [
    activeTenantKey,
    apiBase,
    appendDataLog,
    authHeaders,
    buildDataLogError,
    comboImportUi.payload,
    comboImportUi.stampNow,
    refreshTopCombos,
    showToast,
  ]);

  const clearComboImport = useCallback(() => {
    setComboImportUi({
      text: "",
      payload: null,
      summary: null,
      parseError: null,
      importError: null,
      importing: false,
      importResult: null,
      importedAtMs: null,
      stampNow: false,
    });
  }, []);

  const runOptimizer = useCallback(async () => {
    if (optimizerRunValidationError) {
      setOptimizerRunUi((prev) => ({ ...prev, error: optimizerRunValidationError }));
      showToast(optimizerRunValidationError);
      return;
    }

    const requestId = ++optimizerRunRequestSeqRef.current;
    optimizerRunAbortRef.current?.abort();
    const controller = new AbortController();
    optimizerRunAbortRef.current = controller;
    setOptimizerRunUi((prev) => ({ ...prev, loading: true, error: null }));

    try {
      const extraTimeoutSec = (() => {
        const raw = optimizerRunExtras.value?.timeoutSec;
        if (typeof raw === "number" && Number.isFinite(raw)) return raw;
        if (typeof raw === "string") {
          const parsed = Number(raw);
          return Number.isFinite(parsed) ? parsed : null;
        }
        return null;
      })();
      const timeoutSec = extraTimeoutSec ?? parseOptionalNumber(optimizerRunForm.timeoutSec);
      const timeoutMs =
        typeof timeoutSec === "number" && Number.isFinite(timeoutSec) && timeoutSec > 0
          ? Math.max(1000, Math.round(timeoutSec * 1000))
          : BACKTEST_TIMEOUT_MS;
      const payload = buildOptimizerRunRequest(optimizerRunForm, optimizerRunExtras.value);
      const out = await optimizerRun(apiBase, payload, {
        signal: controller.signal,
        headers: authHeaders,
        timeoutMs,
      });
      if (requestId !== optimizerRunRequestSeqRef.current) return;
      setOptimizerRunUi({ loading: false, error: null, response: out, lastRunAtMs: Date.now() });
      appendDataLog("Optimizer Run", out);
      setApiOk("ok");
      showToast("Optimizer run complete");
      refreshTopCombos();
    } catch (e) {
      if (requestId !== optimizerRunRequestSeqRef.current) return;
      if (isAbortError(e)) return;

      let msg = e instanceof Error ? e.message : String(e);
      let showErrorToast = true;
      if (isTimeoutError(e)) msg = "Optimizer run timed out. Increase timeout or reduce trials.";
      if (e instanceof HttpError && typeof e.payload === "string") {
        const payload = e.payload;
        if (payload.includes("ECONNREFUSED") || payload.includes("connect ECONNREFUSED")) {
          msg = `Backend unreachable. Start it with: cd haskell && cabal run -v0 trader-hs -- --serve --port ${API_PORT}`;
        }
      }
      if (e instanceof HttpError && (e.status === 502 || e.status === 503)) {
        msg = apiBase.startsWith("/api")
          ? "CloudFront `/api/*` proxy is unavailable (502/503). Point `/api/*` at your API origin and allow POST/GET/OPTIONS."
          : "API gateway unavailable (502/503). Try again, or check the API logs.";
      }
      if (e instanceof HttpError && e.status === 504) {
        msg = apiBase.startsWith("/api")
          ? "CloudFront `/api/*` proxy timed out (504). Point `/api/*` at your API origin and allow POST/OPTIONS."
          : "API gateway timed out (504). Try again, or reduce trials/timeout, or scale the API.";
      }
      if (e instanceof HttpError && e.status === 429) {
        const untilMs = applyRateLimit(e);
        msg = `Rate limited. Try again ${fmtEtaMs(Math.max(0, untilMs - Date.now()))}.`;
        showErrorToast = false;
      }

      setApiOk((prev) => {
        if (e instanceof HttpError && (e.status === 401 || e.status === 403)) return "auth";
        const looksDown = msg.toLowerCase().includes("fetch") || (e instanceof HttpError && e.status >= 500);
        return looksDown ? "down" : prev;
      });

      setOptimizerRunUi((prev) => ({ ...prev, loading: false, error: msg }));
      appendDataLog("Optimizer Run Error", buildDataLogError(e, msg), { error: true });
      if (showErrorToast) showToast("Optimizer run failed");
    } finally {
      if (requestId === optimizerRunRequestSeqRef.current) {
        optimizerRunAbortRef.current = null;
      }
    }
  }, [
    apiBase,
    applyRateLimit,
    appendDataLog,
    authHeaders,
    buildDataLogError,
    optimizerRunExtras.value,
    optimizerRunForm,
    optimizerRunValidationError,
    refreshTopCombos,
    showToast,
  ]);
  const scrollDataLogToBottom = useCallback(() => {
    const el = dataLogRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    dataLogPinnedRef.current = true;
  }, []);
  const handleDataLogScroll = useCallback(() => {
    const el = dataLogRef.current;
    if (!el) return;
    const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
    dataLogPinnedRef.current = gap <= DATA_LOG_AUTO_SCROLL_SLOP_PX;
  }, []);
  const downloadBacktestOps = useCallback(() => {
    if (!state.backtest) return;
    const csv = buildBacktestOpsCsv(state.backtest);
    const symbolPart = sanitizeFilenameSegment(form.binanceSymbol, "backtest");
    const intervalPart = sanitizeFilenameSegment(form.interval, "interval");
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    downloadTextFile(`backtest-ops-${symbolPart}-${intervalPart}-${stamp}.csv`, csv, "text/csv");
  }, [form.binanceSymbol, form.interval, state.backtest]);

  useEffect(() => {
    topCombosRef.current = topCombosAll;
  }, [topCombosAll]);

  useEffect(() => {
    if (topCombosDisplayCount <= topCombosDisplayMax) return;
    setTopCombosDisplayCount(topCombosDisplayMax);
  }, [topCombosDisplayCount, topCombosDisplayMax]);

  useEffect(() => {
    if (!dataLogAutoScroll) return;
    if (!dataLogPinnedRef.current) return;
    window.requestAnimationFrame(() => {
      if (!dataLogAutoScroll) return;
      if (!dataLogPinnedRef.current) return;
      scrollDataLogToBottom();
    });
  }, [dataLog, dataLogAutoScroll, scrollDataLogToBottom]);
  useEffect(() => {
    if (!dataLogAutoScroll) return;
    scrollDataLogToBottom();
  }, [dataLogAutoScroll, scrollDataLogToBottom]);

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
      listenKeyStreamAbortRef.current?.abort();
      listenKeyStreamAbortRef.current = null;
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    health(apiBase, { timeoutMs: 10_000, headers: authHeaders })
      .then((out) => {
        if (!mounted) return;
        setHealthInfo(out);
        if (out.authRequired && out.authOk !== true) setApiOk("auth");
        else setApiOk("ok");
        appendDataLog("Health Response (auto)", out, { background: true });
      })
      .catch((e) => {
        if (!mounted) return;
        setHealthInfo(null);
        setApiOk("down");
        const msg = e instanceof Error ? e.message : "API unreachable";
        appendDataLog("Health Error (auto)", buildDataLogError(e, msg), { background: true, error: true });
      });
    return () => {
      mounted = false;
    };
  }, [apiBase, appendDataLog, authHeaders, buildDataLogError]);

  useEffect(() => {
    botStatusTailRef.current = BOT_STATUS_TAIL_POINTS;
    botStatusOpsLimitRef.current = BOT_STATUS_OPS_LIMIT;
  }, [apiBase]);

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
    } catch (e) {
      setHealthInfo(null);
      setApiOk("down");
      showToast("API unreachable");
      const msg = e instanceof Error ? e.message : "API unreachable";
      appendDataLog("Health Error", buildDataLogError(e, msg), { error: true });
      return;
    }

    setHealthInfo(h);
    appendDataLog("Health Response", h);
    if (h.authRequired && h.authOk !== true) {
      setApiOk("auth");
      showToast(apiToken.trim() ? "API auth failed" : "API auth required");
      return;
    }

    try {
      if (activeTenantKey) {
        await botStatus(apiBase, { timeoutMs: 10_000, headers: authHeaders }, undefined, undefined, activeTenantKey);
      }
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
  }, [activeTenantKey, apiBase, apiToken, appendDataLog, authHeaders, buildDataLogError, showToast]);

  const refreshCacheStats = useCallback(async () => {
    setCacheUi((s) => ({ ...s, loading: true, error: null }));
    try {
      const v = await cacheStats(apiBase, { timeoutMs: 10_000, headers: authHeaders });
      setCacheUi({ loading: false, error: null, stats: v });
      appendDataLog("Cache Stats Response", v);
    } catch (e) {
      if (isAbortError(e)) return;
      let msg = "Failed to load cache stats";
      if (e instanceof HttpError && (e.status === 401 || e.status === 403)) {
        msg = "Unauthorized (check apiToken / TRADER_API_TOKEN).";
        setCacheUi((s) => ({ ...s, loading: false, error: msg }));
        appendDataLog("Cache Stats Error", buildDataLogError(e, msg), { error: true });
        return;
      }
      if (isTimeoutError(e)) msg = "Request timed out";
      setCacheUi((s) => ({
        ...s,
        loading: false,
        error: msg,
      }));
      appendDataLog("Cache Stats Error", buildDataLogError(e, msg), { error: true });
    }
  }, [apiBase, appendDataLog, authHeaders, buildDataLogError]);

  const clearCacheUi = useCallback(async () => {
    setCacheUi((s) => ({ ...s, loading: true, error: null }));
    try {
      const cleared = await cacheClear(apiBase, { timeoutMs: 10_000, headers: authHeaders });
      appendDataLog("Cache Clear Response", cleared);
      showToast("Cache cleared");
      try {
        const v = await cacheStats(apiBase, { timeoutMs: 10_000, headers: authHeaders });
        setCacheUi({ loading: false, error: null, stats: v });
        appendDataLog("Cache Stats Response", v);
      } catch (e) {
        if (isAbortError(e)) return;
        let msg = "Failed to load cache stats";
        if (e instanceof HttpError && (e.status === 401 || e.status === 403)) {
          msg = "Unauthorized (check apiToken / TRADER_API_TOKEN).";
        } else if (isTimeoutError(e)) {
          msg = "Request timed out";
        }
        setCacheUi((s) => ({
          ...s,
          loading: false,
          error: msg,
        }));
        appendDataLog("Cache Stats Error", buildDataLogError(e, msg), { error: true });
      }
    } catch (e) {
      if (isAbortError(e)) return;
      let msg = "Failed to clear cache";
      if (e instanceof HttpError && (e.status === 401 || e.status === 403)) {
        msg = "Unauthorized (check apiToken / TRADER_API_TOKEN).";
        setCacheUi((s) => ({ ...s, loading: false, error: msg }));
        appendDataLog("Cache Clear Error", buildDataLogError(e, msg), { error: true });
        return;
      }
      if (isTimeoutError(e)) msg = "Request timed out";
      setCacheUi((s) => ({
        ...s,
        loading: false,
        error: msg,
      }));
      appendDataLog("Cache Clear Error", buildDataLogError(e, msg), { error: true });
    }
  }, [apiBase, appendDataLog, authHeaders, buildDataLogError, showToast]);

  const commonParams: ApiParams = useMemo(() => {
    const interval = form.interval.trim();
    const intervalOk = PLATFORM_INTERVAL_SET[platform].has(interval);
    const barsRaw = Math.trunc(form.bars);
    const barsCap = maxBarsForPlatform(platform, form.method, apiComputeLimits);
    const bars =
      barsRaw <= 0
        ? 0
        : (() => {
            const normalized = Math.max(MIN_LOOKBACK_BARS, barsRaw);
            if (Number.isFinite(barsCap)) return Math.min(normalized, barsCap);
            return normalized;
          })();
    const slippage = clamp(form.slippage, 0, 0.999999);
    const spread = clamp(form.spread, 0, 0.999999);
    const minHoldBars = clamp(Math.trunc(form.minHoldBars), 0, 1_000_000);
    const maxHoldBars = clamp(Math.trunc(form.maxHoldBars), 0, 1_000_000);
    const cooldownBars = clamp(Math.trunc(form.cooldownBars), 0, 1_000_000);
    const minSignalToNoise = Math.max(0, form.minSignalToNoise);
    const volTarget = Math.max(0, form.volTarget);
    const volEwmaAlphaRaw = form.volEwmaAlpha;
    const volEwmaAlpha = volEwmaAlphaRaw > 0 && volEwmaAlphaRaw < 1 ? volEwmaAlphaRaw : 0;
    const volLookbackRaw = Math.max(0, Math.trunc(form.volLookback));
    const volLookback = volTarget > 0 && volEwmaAlpha === 0 ? Math.max(2, volLookbackRaw) : volLookbackRaw;
    const maxVolatility = Math.max(0, form.maxVolatility);
    const rebalanceBars = Math.max(0, Math.trunc(form.rebalanceBars));
    const rebalanceThreshold = Math.max(0, form.rebalanceThreshold);
    const rebalanceCostMult = Math.max(0, form.rebalanceCostMult);
    const fundingRate = Number.isFinite(form.fundingRate) ? form.fundingRate : 0;
    const rebalanceGlobal = form.rebalanceGlobal;
    const rebalanceResetOnSignal = form.rebalanceResetOnSignal;
    const fundingBySide = form.fundingBySide;
    const fundingOnOpen = form.fundingOnOpen;
    const tuneStressVolMult = form.tuneStressVolMult <= 0 ? 1 : form.tuneStressVolMult;
    const minPositionSize = clamp(form.minPositionSize, 0, 1);
    const walkForwardEmbargoBars = Math.max(0, Math.trunc(form.walkForwardEmbargoBars));
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
      slippage,
      spread,
      ...(form.intrabarFill !== "stop-first" ? { intrabarFill: form.intrabarFill } : {}),
      ...(form.stopLoss > 0 ? { stopLoss: clamp(form.stopLoss, 0, 0.999999) } : {}),
      ...(form.takeProfit > 0 ? { takeProfit: clamp(form.takeProfit, 0, 0.999999) } : {}),
      ...(form.trailingStop > 0 ? { trailingStop: clamp(form.trailingStop, 0, 0.999999) } : {}),
      ...(form.stopLossVolMult > 0 ? { stopLossVolMult: Math.max(0, form.stopLossVolMult) } : {}),
      ...(form.takeProfitVolMult > 0 ? { takeProfitVolMult: Math.max(0, form.takeProfitVolMult) } : {}),
      ...(form.trailingStopVolMult > 0 ? { trailingStopVolMult: Math.max(0, form.trailingStopVolMult) } : {}),
      minHoldBars,
      maxHoldBars,
      cooldownBars,
      ...(form.maxDrawdown > 0 ? { maxDrawdown: clamp(form.maxDrawdown, 0, 0.999999) } : {}),
      ...(form.maxDailyLoss > 0 ? { maxDailyLoss: clamp(form.maxDailyLoss, 0, 0.999999) } : {}),
      ...(form.maxOrderErrors >= 1 ? { maxOrderErrors: clamp(Math.trunc(form.maxOrderErrors), 1, 1_000_000) } : {}),
      minEdge: Math.max(0, form.minEdge),
      minSignalToNoise,
      costAwareEdge: form.costAwareEdge,
      edgeBuffer: Math.max(0, form.edgeBuffer),
      trendLookback: clamp(Math.trunc(form.trendLookback), 0, 1_000_000),
      maxPositionSize: Math.max(0, form.maxPositionSize),
      volTarget,
      volLookback,
      volFloor: Math.max(0, form.volFloor),
      volScaleMax: Math.max(0, form.volScaleMax),
      maxVolatility,
      rebalanceBars,
      rebalanceThreshold,
      rebalanceCostMult,
      rebalanceGlobal,
      rebalanceResetOnSignal,
      fundingRate,
      fundingBySide,
      fundingOnOpen,
      blendWeight: clamp(form.blendWeight, 0, 1),
      routerLookback: clamp(Math.trunc(form.routerLookback), 2, 1_000_000),
      routerMinScore: clamp(form.routerMinScore, 0, 1),
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
      walkForwardEmbargoBars,
      normalization: form.normalization,
      epochs: clamp(Math.trunc(form.epochs), 0, 5000),
      hiddenSize: clamp(Math.trunc(form.hiddenSize), 1, 512),
      lr: Math.max(1e-9, form.learningRate),
      valRatio: clamp(form.valRatio, 0, 1),
      patience: clamp(Math.trunc(form.patience), 0, 1000),
      ...(form.gradClip > 0 ? { gradClip: clamp(form.gradClip, 0, 100) } : {}),
      kalmanZMin: Math.max(0, form.kalmanZMin),
      kalmanZMax: Math.max(Math.max(0, form.kalmanZMin), form.kalmanZMax),
      confirmConformal: form.confirmConformal,
      confirmQuantiles: form.confirmQuantiles,
      confidenceSizing: form.confidenceSizing,
      minPositionSize,
      ...(platform === "binance" ? { binanceTestnet: form.binanceTestnet } : {}),
    };

    if (form.lookbackBars >= 2) base.lookbackBars = Math.trunc(form.lookbackBars);
    else if (form.lookbackWindow.trim()) base.lookbackWindow = form.lookbackWindow.trim();

    if (form.method !== "router") {
      if (form.optimizeOperations) base.optimizeOperations = true;
      if (form.sweepThreshold) base.sweepThreshold = true;
    }

    if (volEwmaAlpha > 0) base.volEwmaAlpha = volEwmaAlpha;

    if (form.maxHighVolProb > 0) base.maxHighVolProb = clamp(form.maxHighVolProb, 0, 1);
    if (form.maxConformalWidth > 0) base.maxConformalWidth = Math.max(0, form.maxConformalWidth);
    if (form.maxQuantileWidth > 0) base.maxQuantileWidth = Math.max(0, form.maxQuantileWidth);

    return base;
  }, [apiComputeLimits, form]);

  useEffect(() => {
    if (form.method !== "router") return;
    if (!form.optimizeOperations && !form.sweepThreshold) return;
    setForm((f) => ({ ...f, optimizeOperations: false, sweepThreshold: false }));
  }, [form.method, form.optimizeOperations, form.sweepThreshold]);

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
      const tenantKey = binanceTenantKeyResolved?.trim();
      if (!key && !secret && !tenantKey) return p;
      return {
        ...p,
        ...(tenantKey ? { tenantKey } : {}),
        ...(key ? { binanceApiKey: key } : {}),
        ...(secret ? { binanceApiSecret: secret } : {}),
      };
    },
    [binanceApiKey, binanceApiSecret, binanceTenantKeyResolved, form.platform],
  );

  const withPlatformKeys = useCallback(
    (p: ApiParams): ApiParams => {
      if (isBinancePlatform) return withBinanceKeys(p);
      if (!isCoinbasePlatform) return p;
      const key = coinbaseApiKey.trim();
      const secret = coinbaseApiSecret.trim();
      const passphrase = coinbaseApiPassphrase.trim();
      const tenantKey = coinbaseTenantKeyResolved?.trim();
      if (!key && !secret && !passphrase && !tenantKey) return p;
      return {
        ...p,
        ...(tenantKey ? { tenantKey } : {}),
        ...(key ? { coinbaseApiKey: key } : {}),
        ...(secret ? { coinbaseApiSecret: secret } : {}),
        ...(passphrase ? { coinbaseApiPassphrase: passphrase } : {}),
      };
    },
    [
      coinbaseApiKey,
      coinbaseApiPassphrase,
      coinbaseApiSecret,
      coinbaseTenantKeyResolved,
      isBinancePlatform,
      isCoinbasePlatform,
      withBinanceKeys,
    ],
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
  const botEntriesWithSymbolLive = useMemo(
    () =>
      botEntries
        .map((status) => {
          const symbol = botStatusSymbol(status);
          return symbol ? { symbol, status } : null;
        })
        .filter((entry): entry is { symbol: string; status: BotStatusSingle } => Boolean(entry)),
    [botEntries],
  );
  const botEntriesStarting = useMemo(() => {
    if ("starting" in bot.status && bot.status.starting === true) return true;
    return botEntriesWithSymbolLive.some((entry) => !entry.status.running && entry.status.starting === true);
  }, [bot.status, botEntriesWithSymbolLive]);
  const botEntriesStaleLimitMs = botEntriesStarting ? BOT_DISPLAY_STARTING_STALE_MS : BOT_DISPLAY_STALE_MS;
  const botEntriesCacheRef = useRef<{ entries: { symbol: string; status: BotStatusSingle }[]; atMs: number } | null>(null);
  useEffect(() => {
    if (botEntriesWithSymbolLive.length === 0) return;
    const prev = botEntriesCacheRef.current;
    const now = Date.now();
    const prevAgeMs = prev ? now - prev.atMs : null;
    const shouldReplace =
      !prev || botEntriesWithSymbolLive.length >= prev.entries.length || (prevAgeMs != null && prevAgeMs > botEntriesStaleLimitMs);
    if (shouldReplace) {
      botEntriesCacheRef.current = { entries: botEntriesWithSymbolLive, atMs: now };
    }
  }, [botEntriesStaleLimitMs, botEntriesWithSymbolLive]);
  const botEntriesCacheAgeMs = botEntriesCacheRef.current ? Date.now() - botEntriesCacheRef.current.atMs : null;
  const botEntriesCacheFresh = botEntriesCacheAgeMs != null && botEntriesCacheAgeMs <= botEntriesStaleLimitMs;
  const botEntriesShrank =
    botEntriesCacheRef.current && botEntriesWithSymbolLive.length < botEntriesCacheRef.current.entries.length;
  const botEntriesUseCache = botEntriesCacheFresh && (botEntriesWithSymbolLive.length === 0 || botEntriesShrank);
  const botEntriesWithSymbol =
    botEntriesUseCache && botEntriesCacheRef.current ? botEntriesCacheRef.current.entries : botEntriesWithSymbolLive;
  const botRunningEntries = useMemo(
    () =>
      botEntriesWithSymbol.filter(
        (entry): entry is { symbol: string; status: BotStatusRunning } => entry.status.running,
      ),
    [botEntriesWithSymbol],
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
  const botActiveSymbolSet = useMemo(
    () => new Set(botActiveSymbols.map((sym) => normalizeSymbolKey(sym))),
    [botActiveSymbols],
  );
  const botStartSymbolsNormalized = useMemo(() => botStartSymbols.map((sym) => normalizeSymbolKey(sym)), [botStartSymbols]);
  const botHasNewSymbols = useMemo(
    () => (botStartSymbolsNormalized.length > 0 ? botStartSymbolsNormalized.some((sym) => !botActiveSymbolSet.has(sym)) : false),
    [botActiveSymbolSet, botStartSymbolsNormalized],
  );
  const botSelectedStatus = useMemo(() => {
    if (botEntriesWithSymbol.length === 0) return null;
    const target = botSelectedSymbol ?? botEntriesWithSymbol[0]!.symbol;
    return botStatusBySymbol.get(target) ?? null;
  }, [botEntriesWithSymbol, botSelectedSymbol, botStatusBySymbol]);
  const botStartErrors = useMemo(() => (isBotStatusMulti(bot.status) ? bot.status.errors ?? [] : []), [bot.status]);
  const botRunningCharts = useMemo(() => {
    if (botSelectedStatus?.running) {
      const selectedKey = normalizeSymbolKey(botSelectedStatus.symbol);
      return botRunningEntries.filter((entry) => normalizeSymbolKey(entry.symbol) !== selectedKey);
    }
    return botRunningEntries;
  }, [botRunningEntries, botSelectedStatus]);
  const botStartingReason = useMemo(() => {
    if (!botSelectedStatus || botSelectedStatus.running) return null;
    if (botSelectedStatus.starting !== true) return null;
    return botSelectedStatus.startingReason ?? null;
  }, [botSelectedStatus]);
  const botSnapshot = useMemo(
    () => (botSelectedStatus && !botSelectedStatus.running ? botSelectedStatus.snapshot ?? null : null),
    [botSelectedStatus],
  );
  const botSelectedStarting = useMemo(() => {
    if (!botSelectedStatus || botSelectedStatus.running) return false;
    return botSelectedStatus.starting === true;
  }, [botSelectedStatus]);
  const botDisplayCandidate = botSelectedStatus?.running ? botSelectedStatus : botSnapshot;
  const botDisplayKeyCandidate = botDisplayCandidate ? botStatusKey(botDisplayCandidate) : null;
  const botSelectedKey = useMemo(
    () => (botSelectedStatus ? botStatusKeyFromSingle(botSelectedStatus) : null),
    [botSelectedStatus],
  );
  const botDisplayCacheRef = useRef<Record<string, { data: BotStatusRunning; atMs: number }>>({});
  const botDisplayLastKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (!botDisplayCandidate || !botDisplayKeyCandidate) return;
    botDisplayCacheRef.current[botDisplayKeyCandidate] = { data: botDisplayCandidate, atMs: Date.now() };
    botDisplayLastKeyRef.current = botDisplayKeyCandidate;
  }, [botDisplayCandidate, botDisplayKeyCandidate]);
  useEffect(() => {
    if (!botSelectedKey) return;
    if (!botDisplayCacheRef.current[botSelectedKey]) return;
    botDisplayLastKeyRef.current = botSelectedKey;
  }, [botSelectedKey]);
  const botSelectedCacheEntry = botSelectedKey ? botDisplayCacheRef.current[botSelectedKey] ?? null : null;
  const botDisplayCacheKey = botSelectedCacheEntry ? botSelectedKey : botDisplayLastKeyRef.current;
  const botDisplayCacheEntry = botDisplayCacheKey ? botDisplayCacheRef.current[botDisplayCacheKey] ?? null : null;
  const botDisplayCacheAgeMs = botDisplayCandidate
    ? 0
    : botDisplayCacheEntry
      ? Date.now() - botDisplayCacheEntry.atMs
      : null;
  const botDisplayStaleLimitMs = botSelectedStarting ? BOT_DISPLAY_STARTING_STALE_MS : BOT_DISPLAY_STALE_MS;
  const botDisplayStale =
    botDisplayCandidate == null && botDisplayCacheAgeMs != null && botDisplayCacheAgeMs <= botDisplayStaleLimitMs;
  const botDisplay = botDisplayCandidate ?? (botDisplayStale && botDisplayCacheEntry ? botDisplayCacheEntry.data : null);
  const botSnapshotAtMs = botSelectedStatus?.running ? null : botSelectedStatus?.snapshotAtMs ?? null;
  const botHasSnapshot = botSnapshot !== null;
  const botDisplayStaleLabel =
    botDisplayStale && botDisplayCacheAgeMs != null ? fmtDurationMs(Math.max(0, botDisplayCacheAgeMs)) : null;
  const botDisplayKey = botDisplay ? botStatusKey(botDisplay) : null;
  const botRt = botDisplayKey ? botRtByKey[botDisplayKey] ?? emptyBotRt : emptyBotRt;
  const botStatusRange = useMemo(() => {
    const startMs = parseDatetimeLocal(botStatusStartInput);
    const endMs = parseDatetimeLocal(botStatusEndInput);
    if (startMs == null || endMs == null) return { startMs: null, endMs: null, error: "Start/end must be valid dates." };
    if (startMs > endMs) return { startMs, endMs, error: "Start must be before end." };
    return { startMs, endMs, error: null };
  }, [botStatusEndInput, botStatusStartInput]);
  const botStatusTargetSymbol = botSelectedSymbol ?? botDisplay?.symbol ?? null;
  const botStatusOpsAll = useMemo(() => {
    const parsed = botStatusOps.ops.map((op) => parseBotStatusOp(op)).filter((op): op is BotStatusOp => op !== null);
    return parsed.sort((a, b) => a.atMs - b.atMs);
  }, [botStatusOps.ops]);
  const botStatusOpsParsed = useMemo(
    () => botStatusOpsAll.filter((op) => (botStatusTargetSymbol ? op.symbol === botStatusTargetSymbol : true)),
    [botStatusOpsAll, botStatusTargetSymbol],
  );
  const botStatusPoints = useMemo(
    () => botStatusOpsParsed.map((op) => ({ atMs: op.atMs, running: op.running })),
    [botStatusOpsParsed],
  );
  const botStatusPointsBySymbol = useMemo(() => {
    const map = new Map<string, Array<{ atMs: number; running: boolean }>>();
    for (const op of botStatusOpsAll) {
      if (!op.symbol) continue;
      const key = normalizeSymbolKey(op.symbol);
      const list = map.get(key) ?? [];
      list.push({ atMs: op.atMs, running: op.running });
      if (!map.has(key)) map.set(key, list);
    }
    for (const list of map.values()) {
      list.sort((a, b) => a.atMs - b.atMs);
    }
    return map;
  }, [botStatusOpsAll]);
  const botStatusOpsWindow = useMemo(() => {
    if (botStatusOpsParsed.length === 0) return null;
    return {
      startMs: botStatusOpsParsed[0]!.atMs,
      endMs: botStatusOpsParsed[botStatusOpsParsed.length - 1]!.atMs,
    };
  }, [botStatusOpsParsed]);
  const botStatusRangeWarning = useMemo(() => {
    if (!botStatusOps.enabled) return null;
    if (!botStatusOpsWindow) return null;
    if (botStatusRange.error || botStatusRange.startMs == null || botStatusRange.endMs == null) return null;
    const startsBefore = botStatusRange.startMs < botStatusOpsWindow.startMs;
    const endsAfter = botStatusRange.endMs > botStatusOpsWindow.endMs;
    if (!startsBefore && !endsAfter) return null;
    const rangeNote =
      startsBefore && endsAfter
        ? "Selected range extends beyond available data."
        : startsBefore
          ? "Selected range starts before available data."
          : "Selected range ends after available data.";
    const windowNote = `Data window: ${fmtTimeMs(botStatusOpsWindow.startMs)} to ${fmtTimeMs(botStatusOpsWindow.endMs)}.`;
    const limitNote =
      botStatusOps.ops.length >= botStatusOps.limit
        ? `Showing latest ${botStatusOps.limit} ops.`
        : `Showing ${botStatusOps.ops.length} ops.`;
    return `${rangeNote} ${windowNote} ${limitNote}`;
  }, [botStatusOps.enabled, botStatusOps.limit, botStatusOps.ops.length, botStatusOpsWindow, botStatusRange]);

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
  const deferredBotOrders = useDeferredValue(botOrdersView.shown);

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

  const binanceTradesFilterSymbols = useMemo(
    () => parseSymbolsInput(binanceTradesFilterSymbolsInput),
    [binanceTradesFilterSymbolsInput],
  );
  const binanceTradesFilterStartMs = useMemo(() => parseTimeInputMs(binanceTradesFilterStartInput), [binanceTradesFilterStartInput]);
  const binanceTradesFilterEndMs = useMemo(() => {
    const trimmed = binanceTradesFilterEndInput.trim();
    if (!trimmed) return null;
    if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) {
      const parsed = parseTimeInputMs(trimmed);
      return parsed === null ? null : parsed + 24 * 60 * 60 * 1000 - 1;
    }
    return parseTimeInputMs(trimmed);
  }, [binanceTradesFilterEndInput]);
  const binanceTradesFilterError = useMemo(() => {
    if (binanceTradesFilterStartInput.trim() && binanceTradesFilterStartMs === null) {
      return "Filter start date must be a unix ms timestamp or ISO date.";
    }
    if (binanceTradesFilterEndInput.trim() && binanceTradesFilterEndMs === null) {
      return "Filter end date must be a unix ms timestamp or ISO date.";
    }
    if (
      binanceTradesFilterStartMs !== null &&
      binanceTradesFilterEndMs !== null &&
      binanceTradesFilterEndMs < binanceTradesFilterStartMs
    ) {
      return "Filter end date must be after the start date.";
    }
    return null;
  }, [
    binanceTradesFilterEndInput,
    binanceTradesFilterEndMs,
    binanceTradesFilterStartInput,
    binanceTradesFilterStartMs,
  ]);
  const binanceTradesFiltered = useMemo(() => {
    const trades = binanceTradesUi.response?.trades ?? [];
    if (trades.length === 0) return trades;
    let filtered = trades;
    if (binanceTradesFilterSymbols.length > 0) {
      const symbols = new Set(binanceTradesFilterSymbols);
      filtered = filtered.filter((trade) => symbols.has(trade.symbol.toUpperCase()));
    }
    if (binanceTradesFilterSide !== "ALL") {
      filtered = filtered.filter((trade) => binanceTradeSideLabel(trade) === binanceTradesFilterSide);
    }
    if (binanceTradesFilterStartMs !== null) {
      filtered = filtered.filter((trade) => trade.time >= binanceTradesFilterStartMs);
    }
    if (binanceTradesFilterEndMs !== null) {
      filtered = filtered.filter((trade) => trade.time <= binanceTradesFilterEndMs);
    }
    return filtered;
  }, [
    binanceTradesFilterEndMs,
    binanceTradesFilterSide,
    binanceTradesFilterStartMs,
    binanceTradesFilterSymbols,
    binanceTradesUi.response,
  ]);
  const binanceTradesFilteredTotals = useMemo(() => {
    let totalPnl = 0;
    let totalPnlCount = 0;
    const commissionByAsset = new Map<string, CommissionTotal>();
    for (const trade of binanceTradesFiltered) {
      const pnl = trade.realizedPnl;
      if (typeof pnl === "number" && Number.isFinite(pnl)) {
        totalPnl += pnl;
        totalPnlCount += 1;
      }
      const commission = trade.commission;
      if (typeof commission === "number" && Number.isFinite(commission)) {
        const assetKey = trade.commissionAsset?.trim() ? trade.commissionAsset.trim() : "unknown";
        const existing = commissionByAsset.get(assetKey);
        if (existing) {
          existing.total += commission;
          existing.count += 1;
        } else {
          commissionByAsset.set(assetKey, { asset: assetKey, total: commission, count: 1 });
        }
      }
    }
    const commissionTotals = Array.from(commissionByAsset.values()).sort((a, b) => a.asset.localeCompare(b.asset));
    return { totalPnl, totalPnlCount, commissionTotals };
  }, [binanceTradesFiltered]);
  const binanceTradesFilterActive =
    binanceTradesFilterSymbols.length > 0 ||
    binanceTradesFilterSide !== "ALL" ||
    Boolean(binanceTradesFilterStartInput.trim()) ||
    Boolean(binanceTradesFilterEndInput.trim());
  const binanceTradesTotalCount = binanceTradesUi.response?.trades.length ?? 0;
  const binanceTradesFilteredCount = binanceTradesFiltered.length;
  const binanceTradesFilteredPnlLabel =
    binanceTradesFilteredTotals.totalPnlCount > 0 ? fmtMoney(binanceTradesFilteredTotals.totalPnl, 4) : "—";
  const binanceTradesFilteredCommissionLabel =
    binanceTradesFilteredTotals.commissionTotals.length > 0
      ? binanceTradesFilteredTotals.commissionTotals.map((row) => `${fmtNum(row.total, 8)} ${row.asset}`).join(" • ")
      : "—";

  const binanceTradesCopyText = useMemo(() => {
    const trades = binanceTradesFiltered;
    if (trades.length === 0) {
      return binanceTradesUi.response?.trades.length ? "No trades match filters." : "No trades loaded.";
    }
    return trades
      .map((trade) => {
        const side = binanceTradeSideLabel(trade);
        const qty = typeof trade.qty === "number" && Number.isFinite(trade.qty) ? fmtNum(trade.qty, 8) : "—";
        const quote = typeof trade.quoteQty === "number" && Number.isFinite(trade.quoteQty) ? fmtMoney(trade.quoteQty, 2) : "—";
        return `${fmtTimeMsWithMs(trade.time)} | ${trade.symbol} | ${side} | ${fmtMoney(trade.price, 4)} | ${qty} | ${quote}`;
      })
      .join("\n");
  }, [binanceTradesFiltered, binanceTradesUi.response]);

  const binanceTradesJson = useMemo(() => {
    const trades = binanceTradesFiltered;
    if (trades.length === 0) return "";
    return JSON.stringify(trades, null, 2);
  }, [binanceTradesFiltered]);

  const binanceTradesAnalysis = useMemo(() => {
    return buildBinanceTradePnlAnalysis(binanceTradesFiltered);
  }, [binanceTradesFiltered]);

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

  const botLastPosition = useMemo(() => {
    const st = botDisplay;
    if (!st) return null;
    const last = st.positions[st.positions.length - 1];
    return typeof last === "number" && Number.isFinite(last) ? last : null;
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
    const el = ref.current;
    if (!el) return;
    const openedPanels = openPanelAncestors(el);
    const runScroll = () => {
      const prefersReducedMotion =
        typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const behavior: ScrollBehavior = prefersReducedMotion ? "auto" : "smooth";
      el.scrollIntoView({ behavior, block: "start" });
    };
    if (openedPanels && typeof window !== "undefined") {
      window.setTimeout(runScroll, 0);
    } else {
      runScroll();
    }
  }, [openPanelAncestors]);
  const scrollToSection = useCallback((id?: string) => {
    if (!id || typeof document === "undefined") return;
    const targetPage = resolveConfigPageForTarget(id);
    const shouldSwitchPage = Boolean(targetPage && targetPage !== configPage);
    if (targetPage && targetPage !== configPage) {
      selectConfigPage(targetPage);
    }

    const runScroll = () => {
      const el = document.getElementById(id);
      if (!el) return;
      const openedPanels = openPanelAncestors(el);

      const doScroll = () => {
        const prefersReducedMotion =
          typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        const behavior: ScrollBehavior = prefersReducedMotion ? "auto" : "smooth";
        el.scrollIntoView({ behavior, block: "start" });

        const focusSelector =
          "input:not([type='hidden']):not([disabled]), select:not([disabled]), textarea:not([disabled]), button:not([disabled]), [tabindex]:not([tabindex='-1'])";
        const focusTarget = el.matches(focusSelector) ? el : el.querySelector<HTMLElement>(focusSelector);
        if (focusTarget) {
          focusTarget.focus({ preventScroll: true });
        } else {
          if (!el.hasAttribute("tabindex")) el.setAttribute("tabindex", "-1");
          el.focus({ preventScroll: true });
        }

        if (sectionFlashRef.current && sectionFlashRef.current !== el) {
          sectionFlashRef.current.classList.remove("sectionFlash");
        }
        sectionFlashRef.current = el;
        if (sectionFlashTimeoutRef.current) {
          window.clearTimeout(sectionFlashTimeoutRef.current);
          sectionFlashTimeoutRef.current = null;
        }
        if (!prefersReducedMotion) {
          el.classList.remove("sectionFlash");
          void el.offsetWidth;
          el.classList.add("sectionFlash");
          sectionFlashTimeoutRef.current = window.setTimeout(() => {
            el.classList.remove("sectionFlash");
          }, 1200);
        }
      };

      if (openedPanels && typeof window !== "undefined") {
        window.setTimeout(doScroll, 0);
      } else {
        doScroll();
      }
    };

    if (shouldSwitchPage && typeof window !== "undefined") {
      window.setTimeout(runScroll, 0);
    } else {
      runScroll();
    }
  }, [configPage, openPanelAncestors, selectConfigPage]);

  useEffect(() => {
    return () => {
      if (sectionFlashTimeoutRef.current) {
        window.clearTimeout(sectionFlashTimeoutRef.current);
        sectionFlashTimeoutRef.current = null;
      }
    };
  }, []);

  const adjustBacktestParams = (params: ApiParams): { params: ApiParams; changes: { bars?: number; backtestRatio?: number; message: string } | null } => {
    const interval = params.interval ?? "";
    const platformValue = params.platform ?? platform;
    const intervalSec = platformIntervalSeconds(platformValue, interval);
    const overrideBars = Math.trunc(params.lookbackBars ?? 0);
    const windowRaw = (params.lookbackWindow ?? "").trim();
    const windowSec = windowRaw ? parseDurationSeconds(windowRaw) : null;
    const windowBars =
      windowSec && windowSec > 0 && intervalSec ? Math.ceil(windowSec / intervalSec) : null;
    const lookbackBars = overrideBars >= MIN_LOOKBACK_BARS ? overrideBars : windowBars;
    if (lookbackBars == null || lookbackBars < MIN_LOOKBACK_BARS) return { params, changes: null };

    const barsRaw = Math.trunc(params.bars ?? 0);
    if (!Number.isFinite(barsRaw) || barsRaw <= 0) return { params, changes: null };

    const bars = Math.max(MIN_LOOKBACK_BARS, barsRaw);
    const backtestRatioRaw = typeof params.backtestRatio === "number" && Number.isFinite(params.backtestRatio) ? params.backtestRatio : 0.2;
    const backtestRatio = clamp(backtestRatioRaw, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
    const tuneRatio = typeof params.tuneRatio === "number" && Number.isFinite(params.tuneRatio) ? clamp(params.tuneRatio, 0, 0.99) : 0;
    const tuningEnabled = Boolean(params.optimizeOperations || params.sweepThreshold);

    const method = params.method ?? form.method;
    const maxBars = maxBarsForPlatform(platformValue, method, apiComputeLimits);
    const barsCap = Number.isFinite(maxBars) ? Math.trunc(maxBars) : bars;

    const current = splitStats(bars, backtestRatio, lookbackBars, tuneRatio, tuningEnabled);
    if (current.trainOk && current.backtestOk && current.tuneOk && current.fitOk) {
      if (bars !== barsRaw || backtestRatio !== backtestRatioRaw) {
        return {
          params: { ...params, bars, backtestRatio },
          changes: {
            bars: bars !== barsRaw ? bars : undefined,
            backtestRatio: backtestRatio !== backtestRatioRaw ? backtestRatio : undefined,
            message: `Adjusted split inputs: bars=${bars}, backtestRatio=${fmtPct(backtestRatio, 1)}.`,
          },
        };
      }
      return { params, changes: null };
    }

    const minTrainBars = lookbackBars + 1;
    const maxTrainEndForBars = Math.max(0, bars - MIN_BACKTEST_BARS);
    const minTrainEnd = minTrainEndForTune(minTrainBars, tuneRatio, tuningEnabled, maxTrainEndForBars);
    const minBarsForTrain = Math.ceil(minTrainEnd / Math.max(1e-6, 1 - backtestRatio));
    const minBarsForBacktest = Math.ceil(MIN_BACKTEST_BARS / Math.max(1e-6, backtestRatio));
    let candidateBars = Math.max(bars, minTrainBars + MIN_BACKTEST_BARS, minBarsForTrain, minBarsForBacktest);
    if (candidateBars > barsCap) candidateBars = barsCap;

    if (candidateBars > bars) {
      let adjustedBars = candidateBars;
      while (adjustedBars <= barsCap) {
        const stats = splitStats(adjustedBars, backtestRatio, lookbackBars, tuneRatio, tuningEnabled);
        if (stats.trainOk && stats.backtestOk && stats.tuneOk && stats.fitOk) {
          return {
            params: { ...params, bars: adjustedBars, backtestRatio },
            changes: {
              bars: adjustedBars,
              message: `Adjusted bars to ${adjustedBars} to satisfy the split.`,
            },
          };
        }
        adjustedBars += 1;
      }
    }

    const maxTrainEnd = Math.max(0, bars - MIN_BACKTEST_BARS);
    const minTrainEndAdj = minTrainEndForTune(minTrainBars, tuneRatio, tuningEnabled, maxTrainEnd);
    if (minTrainEndAdj <= maxTrainEnd) {
      let targetTrainEnd = current.trainEndRaw;
      if (!current.trainOk || !current.tuneOk || !current.fitOk) targetTrainEnd = minTrainEndAdj;
      if (!current.backtestOk) targetTrainEnd = maxTrainEnd;
      const adjustedRatio = ratioForTrainEnd(bars, targetTrainEnd);
      const stats = splitStats(bars, adjustedRatio, lookbackBars, tuneRatio, tuningEnabled);
      if (stats.trainOk && stats.backtestOk && stats.tuneOk && stats.fitOk) {
        return {
          params: { ...params, bars, backtestRatio: adjustedRatio },
          changes: {
            backtestRatio: adjustedRatio,
            message: `Adjusted backtest ratio to ${fmtPct(adjustedRatio, 1)} to satisfy the split.`,
          },
        };
      }
    }

    return { params, changes: null };
  };

  const run = useCallback(
    async (kind: RequestKind, overrideParams?: ApiParams, opts?: RunOptions) => {
      const now = Date.now();
      const logBackground = Boolean(opts?.silent);
      const logSuffix = logBackground ? " (auto)" : "";
      const logLabelBase = kind === "signal" ? "Signal" : kind === "backtest" ? "Backtest" : "Trade";
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
        const baseParams = overrideParams ?? (kind === "trade" ? tradeParams : commonParams);
        const adjusted =
          kind === "backtest" ? adjustBacktestParams(baseParams) : { params: baseParams, changes: null };
        const p = adjusted.params;
        if (adjusted.changes && !opts?.silent) {
          setForm((prev) => {
            let next = prev;
            if (adjusted.changes?.bars !== undefined && adjusted.changes.bars !== prev.bars) {
              next = { ...next, bars: adjusted.changes.bars };
            }
            if (adjusted.changes?.backtestRatio !== undefined && adjusted.changes.backtestRatio !== prev.backtestRatio) {
              next = { ...next, backtestRatio: adjusted.changes.backtestRatio };
            }
            return next === prev ? prev : next;
          });
          showToast(adjusted.changes.message);
        }
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
          appendDataLog(`${logLabelBase} Response${logSuffix}`, out, { background: logBackground });
          if (opts?.silent) {
            setState((s) => ({ ...s, latestSignal: out }));
          } else {
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
          appendDataLog(`${logLabelBase} Response${logSuffix}`, out, { background: logBackground });
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
          appendDataLog(`${logLabelBase} Response${logSuffix}`, out, { background: logBackground });
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
            ? "CloudFront `/api/*` proxy is unavailable (502/503). Point `/api/*` at your API origin (App Runner/ALB/etc) and allow POST/GET/OPTIONS. If you are not using a proxy, set apiBaseUrl in trader-config.js to https://<your-api-host> (CORS required)."
            : "API gateway unavailable (502/503). Try again, or check the API logs.";
        }
        if (e instanceof HttpError && e.status === 504) {
          msg = apiBase.startsWith("/api")
            ? "CloudFront `/api/*` proxy timed out (504). Point `/api/*` at your API origin (App Runner/ALB/etc) and allow POST/OPTIONS. If you are not using a proxy, set apiBaseUrl in trader-config.js to https://<your-api-host> (CORS required)."
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

        appendDataLog(`${logLabelBase} Error${logSuffix}`, buildDataLogError(e, msg), {
          background: logBackground,
          error: true,
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
      appendDataLog,
      authHeaders,
      buildDataLogError,
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
        appendDataLog(`Key Status${opts?.silent ? " (auto)" : ""}`, out, { background: Boolean(opts?.silent) });
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

        appendDataLog(`Key Status Error${opts?.silent ? " (auto)" : ""}`, buildDataLogError(e, msg), {
          background: Boolean(opts?.silent),
          error: true,
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
    [apiBase, appendDataLog, authHeaders, buildDataLogError, isBinancePlatform, isCoinbasePlatform, keysParams, platform, showToast],
  );

  const stopListenKeyStream = useCallback(
    async (opts?: { close?: boolean; silent?: boolean }) => {
      listenKeyStreamAbortRef.current?.abort();
      listenKeyStreamAbortRef.current = null;
      listenKeyStreamSeqRef.current += 1;

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

  const openListenKeyStream = useCallback(
    async (opts?: { silent?: boolean }) => {
      const controller = new AbortController();
      const streamId = ++listenKeyStreamSeqRef.current;
      listenKeyStreamAbortRef.current?.abort();
      listenKeyStreamAbortRef.current = controller;

      try {
        const tenantKey = binanceTenantKeyResolved?.trim();
        if (!tenantKey) {
          throw new Error("Tenant key required. Add Binance API keys (or check keys) to open the listen key stream.");
        }
        const requestHeaders = { ...(authHeaders ?? {}), Accept: "text/event-stream" };
        const url = `${apiBase}/binance/listenKey/stream?tenantKey=${encodeURIComponent(tenantKey)}`;
        const res = await fetch(url, {
          method: "GET",
          headers: requestHeaders,
          signal: controller.signal,
        });
        if (!res.ok) {
          const msg = await res.text();
          throw new Error(msg || `Listen key stream failed (${res.status}).`);
        }
        if (!res.body) {
          throw new Error("Listen key stream unavailable.");
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        const parse = createSseParser((event, data) => {
          if (listenKeyStreamSeqRef.current !== streamId) return;
          if (event === "status") {
            const payload = safeJsonParse<ListenKeyStreamStatusPayload>(data);
            const statusRaw = typeof payload?.status === "string" ? payload.status : "disconnected";
            const message = typeof payload?.message === "string" ? payload.message : null;
            const nextStatus = normalizeListenKeyStreamStatus(statusRaw);
            setListenKeyUi((s) => ({
              ...s,
              wsStatus: nextStatus,
              wsError: nextStatus === "connected" ? null : message ?? s.wsError,
            }));
            return;
          }
          if (event === "keepalive") {
            const payload = safeJsonParse<ListenKeyStreamKeepAlivePayload>(data);
            if (payload && typeof payload.atMs === "number") {
              setListenKeyUi((s) => ({ ...s, keepAliveAtMs: payload.atMs ?? null, keepAliveError: null }));
            }
            return;
          }
          if (event === "binance") {
            let pretty = data;
            try {
              pretty = JSON.stringify(JSON.parse(data), null, 2);
            } catch {
              // ignore
            }
            if (pretty.length > 8000) pretty = `${pretty.slice(0, 7997)}...`;
            setListenKeyUi((s) => ({ ...s, lastEventAtMs: Date.now(), lastEvent: pretty }));
            return;
          }
          if (event === "error") {
            const payload = safeJsonParse<ListenKeyStreamErrorPayload>(data);
            const message = typeof payload?.message === "string" ? payload.message : data;
            setListenKeyUi((s) => ({ ...s, wsError: message || s.wsError }));
          }
        });

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          parse(decoder.decode(value, { stream: true }));
        }

        if (listenKeyStreamSeqRef.current === streamId) {
          setListenKeyUi((s) => ({ ...s, wsStatus: "disconnected" }));
        }
      } catch (e) {
        if (isAbortError(e)) return;
        if (listenKeyStreamSeqRef.current !== streamId) return;
        const msg = e instanceof Error ? e.message : String(e);
        setListenKeyUi((s) => ({ ...s, wsError: msg, wsStatus: "disconnected" }));
        if (!opts?.silent) showToast("Listen key stream failed");
      } finally {
        if (listenKeyStreamAbortRef.current === controller) {
          listenKeyStreamAbortRef.current = null;
        }
      }
    },
    [apiBase, authHeaders, binanceTenantKeyResolved, showToast],
  );

  const startListenKeyStream = useCallback(
    async (opts?: { silent?: boolean }) => {
      const silent = Boolean(opts?.silent);
      if (!isBinancePlatform) {
        const msg = "Listen key streams are supported on Binance only.";
        setListenKeyUi((s) => ({ ...s, error: msg, wsStatus: "disconnected" }));
        if (!silent) showToast(msg);
        return;
      }
      if (apiOk !== "ok") return;
      await stopListenKeyStream({ close: false, silent: true });
      setListenKeyUi((s) => ({ ...s, loading: true, error: null, wsError: null, keepAliveError: null, wsStatus: "connecting" }));
      try {
        const base: ApiParams = { market: form.market, binanceTestnet: form.binanceTestnet };
        const out = await binanceListenKey(apiBase, withBinanceKeys(base), { headers: authHeaders, timeoutMs: 30_000 });

        setListenKeyUi((s) => ({ ...s, loading: false, error: null, info: out, wsStatus: "connecting" }));
        void openListenKeyStream({ silent });
        if (!silent) showToast("Listen key started");
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setListenKeyUi((s) => ({ ...s, loading: false, error: msg, wsStatus: "disconnected" }));
        if (!silent) showToast("Listen key start failed");
      }
    },
    [
      apiBase,
      apiOk,
      authHeaders,
      form.binanceTestnet,
      form.market,
      isBinancePlatform,
      openListenKeyStream,
      showToast,
      stopListenKeyStream,
      withBinanceKeys,
    ],
  );

  useEffect(() => {
    if (autoKeysCheckRef.current) return;
    if (apiOk !== "ok") return;
    if (!keysSupported) return;
    if (!form.binanceSymbol.trim()) return;
    autoKeysCheckRef.current = true;
    void refreshKeys({ silent: true });
  }, [apiOk, form.binanceSymbol, keysSupported, refreshKeys]);

  useEffect(() => {
    if (autoListenKeyStartRef.current) return;
    if (listenKeyUi.info || listenKeyUi.wsStatus !== "disconnected") {
      autoListenKeyStartRef.current = true;
      return;
    }
    if (apiOk !== "ok") return;
    if (!isBinancePlatform) return;
    if (listenKeyUi.loading) return;
    if (!binanceKeyLocalReady && keysProvided !== true) return;
    autoListenKeyStartRef.current = true;
    void startListenKeyStream({ silent: true });
  }, [
    apiOk,
    binanceKeyLocalReady,
    isBinancePlatform,
    keysProvided,
    listenKeyUi.info,
    listenKeyUi.loading,
    listenKeyUi.wsStatus,
    startListenKeyStream,
  ]);

  const refreshBot = useCallback(
    async (opts?: RunOptions) => {
      const silent = Boolean(opts?.silent);
      if (silent && botAbortRef.current) return;
      const requestId = ++botRequestSeqRef.current;
      const startedAtMs = Date.now();
      if (!silent) botAbortRef.current?.abort();
      const controller = new AbortController();
      botAbortRef.current = controller;

      if (!silent) setBot((s) => ({ ...s, loading: true, error: null }));

      try {
        if (!activeTenantKey) {
          const msg = "Tenant key required. Add API keys (or check keys) to load bot status.";
          if (!silent) setBot((s) => ({ ...s, loading: false, error: msg }));
          return;
        }
        const tailPoints = botStatusTailRef.current;
        let out: BotStatus | BotStatusMulti;
        try {
          out = await botStatus(
            apiBase,
            { signal: controller.signal, headers: authHeaders, timeoutMs: BOT_STATUS_TIMEOUT_MS },
            tailPoints,
            undefined,
            activeTenantKey,
          );
        } catch (e) {
          const shouldFallback =
            tailPoints > BOT_STATUS_TAIL_FALLBACK_POINTS &&
            (isTimeoutError(e) || (e instanceof HttpError && BOT_STATUS_RETRYABLE_HTTP.has(e.status)));
          if (shouldFallback) {
            const fallbackTail = Math.min(BOT_STATUS_TAIL_FALLBACK_POINTS, tailPoints);
            out = await botStatus(
              apiBase,
              { signal: controller.signal, headers: authHeaders, timeoutMs: BOT_STATUS_TIMEOUT_MS },
              fallbackTail,
              undefined,
              activeTenantKey,
            );
            botStatusTailRef.current = fallbackTail;
          } else {
            throw e;
          }
        }
        const finishedAtMs = Date.now();
        if (requestId !== botRequestSeqRef.current) return;
        botStatusFetchedRef.current = true;
        const botStatuses = isBotStatusMulti(out) ? out.bots : [out];
        const runningStatuses = botStatuses.filter((status): status is BotStatusRunning => status.running);
        setBotRtByKey((prev) => {
          if (runningStatuses.length === 0) {
            const hasPrev = Object.keys(prev).length > 0;
            const hasRef = Object.keys(botRtRef.current).length > 0;
            if (!hasPrev && !hasRef) return prev;
            botRtRef.current = {};
            return {};
          }

          const next: Record<string, BotRtUiState> = {};
          const nextRef: Record<string, BotRtTracker> = {};

          for (const st of runningStatuses) {
            const botKey = botStatusKey(st);
            const prevState = prev[botKey] ?? emptyBotRtState();
            const tracker = botRtRef.current[botKey] ?? emptyBotRtTracker();
            const base: BotRtUiState = {
              ...prevState,
              lastFetchAtMs: finishedAtMs,
              lastFetchDurationMs: Math.max(0, finishedAtMs - startedAtMs),
              lastNewCandles: 0,
              lastKlineUpdates: 0,
            };

            let feed = base.feed;
            let telemetry = base.telemetry;

            const openTimes = st.openTimes;
            const lastOpen = openTimes[openTimes.length - 1] ?? null;
            const prevLastOpen = tracker.lastOpenTimeMs;
            const newTimes = typeof prevLastOpen === "number" ? openTimes.filter((t) => t > prevLastOpen) : [];
            const newCount = newTimes.length;

            let lastNewCandlesAtMs: number | null = prevState.lastNewCandlesAtMs;
            if (newCount > 0) {
              lastNewCandlesAtMs = finishedAtMs;
              const lastNew = newTimes[newTimes.length - 1]!;
              const idx = openTimes.lastIndexOf(lastNew);
              const closePx = idx >= 0 ? st.prices[idx] : null;
              const action = st.latestSignal.action;
              const pollMs =
                typeof st.pollLatencyMs === "number" && Number.isFinite(st.pollLatencyMs) ? Math.max(0, Math.round(st.pollLatencyMs)) : null;
              const batchMs =
                typeof st.lastBatchMs === "number" && Number.isFinite(st.lastBatchMs) ? Math.max(0, Math.round(st.lastBatchMs)) : null;
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

            let lastKlineUpdatesAtMs: number | null = prevState.lastKlineUpdatesAtMs;
            let klineUpdates = 0;
            const fetchedLast = st.fetchedLastKline;
            if (fetchedLast && typeof fetchedLast.openTime === "number" && Number.isFinite(fetchedLast.openTime)) {
              const openTime = fetchedLast.openTime;
              const close = fetchedLast.close;
              if (typeof close === "number" && Number.isFinite(close)) {
                const prevFetchedOpen = tracker.lastFetchedOpenTimeMs;
                const prevFetchedClose = tracker.lastFetchedClose;
                if (
                  newCount === 0 &&
                  prevFetchedOpen === openTime &&
                  typeof prevFetchedClose === "number" &&
                  Number.isFinite(prevFetchedClose) &&
                  close !== prevFetchedClose
                ) {
                  klineUpdates = 1;
                  lastKlineUpdatesAtMs = finishedAtMs;
                  const d = prevFetchedClose !== 0 ? (close - prevFetchedClose) / prevFetchedClose : null;
                  const msg = `kline update: close ${fmtMoney(close, 4)}${d !== null ? ` (Δ ${fmtPct(d, 2)})` : ""}`;
                  feed = [{ atMs: finishedAtMs, message: msg }, ...feed].slice(0, 50);
                }
                tracker.lastFetchedOpenTimeMs = openTime;
                tracker.lastFetchedClose = close;
              }
            }

            const polledAtMs = typeof st.polledAtMs === "number" && Number.isFinite(st.polledAtMs) ? st.polledAtMs : null;
            if (polledAtMs !== null && polledAtMs !== tracker.lastTelemetryPolledAtMs) {
              tracker.lastTelemetryPolledAtMs = polledAtMs;
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

            if (
              tracker.lastMethod !== null &&
              (st.method !== tracker.lastMethod || openThr !== tracker.lastOpenThreshold || closeThr !== tracker.lastCloseThreshold)
            ) {
              const msg =
                `params: ${methodLabel(st.method)}` +
                ` • open ${fmtPct(openThr, 3)}` +
                ` • close ${fmtPct(closeThr, 3)}` +
                (typeof tradeEnabled === "boolean" ? ` • trade ${tradeEnabled ? "ON" : "OFF"}` : "");
              feed = [{ atMs: finishedAtMs, message: msg }, ...feed].slice(0, 50);
            }

            if (typeof tradeEnabled === "boolean" && tracker.lastTradeEnabled !== null && tradeEnabled !== tracker.lastTradeEnabled) {
              feed = [{ atMs: finishedAtMs, message: `trade ${tradeEnabled ? "enabled" : "disabled"}` }, ...feed].slice(0, 50);
            }

            const err = st.error ?? null;
            if (err && err !== tracker.lastError) {
              feed = [{ atMs: finishedAtMs, message: `error: ${err}` }, ...feed].slice(0, 50);
            }

            if (tracker.lastHalted !== null && tracker.lastHalted !== st.halted) {
              feed = [{ atMs: finishedAtMs, message: st.halted ? `halted: ${st.haltReason ?? "true"}` : "resumed" }, ...feed].slice(0, 50);
            }

            tracker.lastOpenTimeMs = lastOpen;
            tracker.lastError = err;
            tracker.lastHalted = st.halted;
            tracker.lastMethod = st.method;
            tracker.lastOpenThreshold = openThr;
            tracker.lastCloseThreshold = closeThr;
            tracker.lastTradeEnabled = typeof tradeEnabled === "boolean" ? tradeEnabled : null;

            next[botKey] = {
              ...base,
              lastNewCandles: newCount,
              lastNewCandlesAtMs,
              lastKlineUpdates: klineUpdates,
              lastKlineUpdatesAtMs,
              telemetry,
              feed,
            };
            nextRef[botKey] = tracker;
          }

          botRtRef.current = nextRef;
          return next;
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
            ? "CloudFront `/api/*` proxy is unavailable (502/503). Point `/api/*` at your API origin (App Runner/ALB/etc) and allow POST/GET/OPTIONS. If you are not using a proxy, set apiBaseUrl in trader-config.js to https://<your-api-host> (CORS required)."
            : "API gateway unavailable (502/503). Try again, or check the API logs.";
        }
        if (e instanceof HttpError && e.status === 504) {
          msg = apiBase.startsWith("/api")
            ? "CloudFront `/api/*` proxy timed out (504). Point `/api/*` at your API origin (App Runner/ALB/etc) and allow POST/OPTIONS. If you are not using a proxy, set apiBaseUrl in trader-config.js to https://<your-api-host> (CORS required)."
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
    [activeTenantKey, apiBase, authHeaders],
  );

  type StartBotOptions = { auto?: boolean; forceAdopt?: boolean; silent?: boolean; symbolsOverride?: string[] };

  const startLiveBot = useCallback(
    async (opts?: StartBotOptions) => {
      const silent = Boolean(opts?.silent);
      const forceAdopt = Boolean(opts?.forceAdopt);
      const symbolsOverride = opts?.symbolsOverride ? parseSymbolsInput(opts.symbolsOverride.join(",")) : [];
      if (!opts?.auto) botAutoStartSuppressedRef.current = false;
      const startSymbols = symbolsOverride.length > 0 ? symbolsOverride : botSymbolsInput;
      const primarySymbolRaw = startSymbols[0] ?? form.binanceSymbol.trim();
      const primarySymbol = primarySymbolRaw ? normalizeSymbolKey(primarySymbolRaw) : "";
      const startSymbolsNormalized = startSymbols.map((sym) => normalizeSymbolKey(sym)).filter(Boolean);
      const requestedSymbols =
        startSymbolsNormalized.length > 0 ? startSymbolsNormalized : primarySymbol ? [primarySymbol] : [];
      if (requestedSymbols.length === 0) {
        if (!silent) {
          setBot((s) => ({ ...s, error: "Symbol is required to start the live bot." }));
          showToast("Bot start failed");
        }
        return;
      }
      const missingSymbols = requestedSymbols.filter((sym) => !botActiveSymbolSet.has(sym));
      const shouldSelectPrimary = primarySymbol && (!opts?.auto || botSelectedSymbol == null);

      if (primarySymbol && missingSymbols.length === 0) {
        setBot((s) => ({ ...s, error: null }));
        if (shouldSelectPrimary) setBotSelectedSymbol(primarySymbol);
        if (!silent) {
          const msg =
            requestedSymbols.length > 1
              ? "All requested bot symbols are already running."
              : `Live bot already running for ${primarySymbol}.`;
          showToast(msg);
        }
        return;
      }

      const isAdoptError = (msg: string) => {
        const lower = msg.toLowerCase();
        return lower.includes("existing long position") || lower.includes("botadoptexistingposition=true");
      };

      const isAlreadyRunningError = (err: unknown) => {
        if (!(err instanceof HttpError)) return false;
        const msg = err.message.toLowerCase();
        if (msg.includes("already running") || msg.includes("bot is starting")) return true;
        const payload = err.payload;
        if (!payload || typeof payload !== "object") return false;
        if ("error" in payload && typeof payload.error === "string") {
          const errMsg = payload.error.toLowerCase();
          if (errMsg.includes("already running") || errMsg.includes("bot is starting")) return true;
        }
        if ("errors" in payload && Array.isArray(payload.errors)) {
          return payload.errors.some((entry) => {
            if (!entry || typeof entry !== "object") return false;
            const errText = "error" in entry && typeof entry.error === "string" ? entry.error.toLowerCase() : "";
            return errText.includes("already running") || errText.includes("bot is starting");
          });
        }
        return false;
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
          ...(form.botProtectionOrders ? { botProtectionOrders: true } : {}),
          botAdoptExistingPosition: true,
          ...(form.botPollSeconds > 0 ? { botPollSeconds: clamp(Math.trunc(form.botPollSeconds), 1, 3600) } : {}),
          botOnlineEpochs: clamp(Math.trunc(form.botOnlineEpochs), 0, 50),
          botTrainBars: Math.max(10, Math.trunc(form.botTrainBars)),
          botMaxPoints: clamp(Math.trunc(form.botMaxPoints), 100, 100000),
        };
        if (primarySymbol) payload.binanceSymbol = primarySymbol;
        if (startSymbols.length > 0) payload.botSymbols = startSymbols;
        const out = await botStart(apiBase, withPlatformKeys(payload), { headers: authHeaders, timeoutMs: BOT_START_TIMEOUT_MS });
        setBot((s) => ({ ...s, loading: false, error: null, status: out }));
        appendDataLog(`Bot Start Response${silent ? " (auto)" : ""}`, out, { background: silent });
        if (symbolsOverride.length > 0) {
          if (shouldSelectPrimary) setBotSelectedSymbol(primarySymbol || null);
        }
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
        if (isAlreadyRunningError(e)) {
          setBot((s) => ({ ...s, loading: false, error: null }));
          if (shouldSelectPrimary) setBotSelectedSymbol(primarySymbol);
          if (!silent) showToast(`Live bot already running${primarySymbol ? ` for ${primarySymbol}` : ""}.`);
          return;
        }
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
            appendDataLog(`Bot Start Error${silent ? " (auto)" : ""}`, buildDataLogError(e2, formatted2.msg), {
              background: silent,
              error: true,
            });
            if (formatted2.showErrorToast) showToast("Bot start failed");
            return;
          }
        }
        const formatted = formatStartError(e);
        setBot((s) => ({ ...s, loading: false, error: formatted.msg }));
        appendDataLog(`Bot Start Error${silent ? " (auto)" : ""}`, buildDataLogError(e, formatted.msg), {
          background: silent,
          error: true,
        });
        if (formatted.showErrorToast) showToast("Bot start failed");
      }
    },
    [
      apiBase,
      applyRateLimit,
      appendDataLog,
      authHeaders,
      botActiveSymbolSet,
      botSelectedSymbol,
      botSymbolsInput,
      buildDataLogError,
      form.binanceSymbol,
      form.botMaxPoints,
      form.botAdoptExistingPosition,
      form.botOnlineEpochs,
      form.botPollSeconds,
      form.botProtectionOrders,
      form.botTrainBars,
      form.tradeArmed,
      showToast,
      tradeParams,
      withPlatformKeys,
    ],
  );

  const fetchBotStatusOps = useCallback(
    async (opts?: RunOptions) => {
      if (apiOk !== "ok") return;
      if (botStatusOpsInFlightRef.current) return;
      botStatusOpsInFlightRef.current = true;
      botStatusOpsAbortRef.current?.abort();
      const controller = new AbortController();
      botStatusOpsAbortRef.current = controller;

      if (!opts?.silent) setBotStatusOps((s) => ({ ...s, loading: true, error: null }));

      const mergeOps = (prevOps: OpsOperation[], nextOps: OpsOperation[], limit: number) => {
        if (prevOps.length === 0) return nextOps.slice(Math.max(0, nextOps.length - limit));
        const map = new Map<number, OpsOperation>();
        for (const op of prevOps) map.set(op.id, op);
        for (const op of nextOps) map.set(op.id, op);
        const merged = Array.from(map.values()).sort((a, b) => (a.atMs - b.atMs) || (a.id - b.id));
        if (merged.length > limit) merged.splice(0, merged.length - limit);
        return merged;
      };

      try {
        const limit = botStatusOpsLimitRef.current;
        const sinceRaw = botStatusOpsSinceRef.current;
        const since =
          typeof sinceRaw === "number" && Number.isFinite(sinceRaw) && sinceRaw > 0
            ? Math.max(0, Math.trunc(sinceRaw) - 1)
            : undefined;
        const out = await ops(
          apiBase,
          { kind: "bot.status", limit, ...(since ? { since } : {}) },
          { headers: authHeaders, timeoutMs: 30_000, signal: controller.signal },
        );
        const incoming = Array.isArray(out.ops) ? out.ops : [];
        setBotStatusOps((prev) => {
          const merged = mergeOps(prev.ops, incoming, limit);
          const lastAt = merged.length > 0 ? merged[merged.length - 1]!.atMs : prev.lastFetchedAtMs;
          if (typeof lastAt === "number" && Number.isFinite(lastAt)) botStatusOpsSinceRef.current = lastAt;
          return {
            loading: false,
            error: null,
            enabled: out.enabled,
            hint: out.hint ?? null,
            ops: merged,
            limit,
            lastFetchedAtMs: Date.now(),
          };
        });
        appendDataLog(`Ops Response (bot.status)${opts?.silent ? " (auto)" : ""}`, out, {
          background: Boolean(opts?.silent),
        });
        botStatusOpsLimitRef.current = limit;
      } catch (e) {
        if (isAbortError(e)) return;
        const msg = e instanceof Error ? e.message : String(e);
        if (e instanceof HttpError && BOT_STATUS_RETRYABLE_HTTP.has(e.status) && botStatusOpsLimitRef.current > BOT_STATUS_OPS_FALLBACK_LIMIT) {
          try {
            const fallbackLimit = BOT_STATUS_OPS_FALLBACK_LIMIT;
            const out = await ops(
              apiBase,
              { kind: "bot.status", limit: fallbackLimit },
              { headers: authHeaders, timeoutMs: 30_000, signal: controller.signal },
            );
            const incoming = Array.isArray(out.ops) ? out.ops : [];
            setBotStatusOps((prev) => {
              const merged = mergeOps(prev.ops, incoming, fallbackLimit);
              const lastAt = merged.length > 0 ? merged[merged.length - 1]!.atMs : prev.lastFetchedAtMs;
              if (typeof lastAt === "number" && Number.isFinite(lastAt)) botStatusOpsSinceRef.current = lastAt;
              return {
                loading: false,
                error: null,
                enabled: out.enabled,
                hint: out.hint ?? null,
                ops: merged,
                limit: fallbackLimit,
                lastFetchedAtMs: Date.now(),
              };
            });
            appendDataLog(`Ops Response (bot.status fallback)${opts?.silent ? " (auto)" : ""}`, out, {
              background: Boolean(opts?.silent),
            });
            botStatusOpsLimitRef.current = fallbackLimit;
            return;
          } catch (fallbackErr) {
            if (isAbortError(fallbackErr)) return;
            const fallbackMsg = fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr);
            setBotStatusOps((s) => ({ ...s, loading: false, error: fallbackMsg }));
            appendDataLog(`Ops Error (bot.status fallback)${opts?.silent ? " (auto)" : ""}`, buildDataLogError(fallbackErr, fallbackMsg), {
              background: Boolean(opts?.silent),
              error: true,
            });
            return;
          }
        }
        setBotStatusOps((s) => ({ ...s, loading: false, error: msg }));
        appendDataLog(`Ops Error (bot.status)${opts?.silent ? " (auto)" : ""}`, buildDataLogError(e, msg), {
          background: Boolean(opts?.silent),
          error: true,
        });
      } finally {
        if (botStatusOpsAbortRef.current === controller) botStatusOpsAbortRef.current = null;
        botStatusOpsInFlightRef.current = false;
      }
    },
    [apiBase, apiOk, appendDataLog, authHeaders, buildDataLogError],
  );

  const fetchOpsPerformance = useCallback(
    async (opts?: RunOptions) => {
      if (apiOk !== "ok") return;
      if (opsPerformanceInFlightRef.current) return;
      opsPerformanceInFlightRef.current = true;
      opsPerformanceAbortRef.current?.abort();
      const controller = new AbortController();
      opsPerformanceAbortRef.current = controller;

      if (!opts?.silent) setOpsPerformanceUi((s) => ({ ...s, loading: true, error: null }));

      try {
        const out = await opsPerformance(
          apiBase,
          {
            commitLimit: opsPerformanceCommitLimit,
            comboLimit: opsPerformanceComboLimit,
            comboScope: opsPerformanceComboScope,
            comboOrder: opsPerformanceComboOrder,
          },
          { headers: authHeaders, timeoutMs: 30_000, signal: controller.signal },
        );
        setOpsPerformanceUi({
          loading: false,
          error: null,
          enabled: out.enabled,
          ready: out.ready,
          commitsReady: out.commitsReady,
          combosReady: out.combosReady,
          hint: out.hint ?? null,
          commits: Array.isArray(out.commits) ? out.commits : [],
          combos: Array.isArray(out.combos) ? out.combos : [],
          lastFetchedAtMs: Date.now(),
        });
        appendDataLog(`Ops Performance${opts?.silent ? " (auto)" : ""}`, out, { background: Boolean(opts?.silent) });
      } catch (e) {
        if (isAbortError(e)) return;
        const msg = e instanceof Error ? e.message : String(e);
        setOpsPerformanceUi((s) => ({ ...s, loading: false, error: msg }));
        appendDataLog(`Ops Performance Error${opts?.silent ? " (auto)" : ""}`, buildDataLogError(e, msg), {
          background: Boolean(opts?.silent),
          error: true,
        });
      } finally {
        if (opsPerformanceAbortRef.current === controller) opsPerformanceAbortRef.current = null;
        opsPerformanceInFlightRef.current = false;
      }
    },
    [
      apiBase,
      apiOk,
      appendDataLog,
      authHeaders,
      buildDataLogError,
      opsPerformanceComboLimit,
      opsPerformanceComboOrder,
      opsPerformanceComboScope,
      opsPerformanceCommitLimit,
    ],
  );

  const opsPerformancePanelOpen = isPanelOpen("panel-performance", false);

  const shouldPollBotStatusOps = useMemo(
    () => isPanelOpen("panel-live-bot", true) || botRunningCharts.length > 0,
    [botRunningCharts.length, isPanelOpen],
  );

  useEffect(() => {
    if (apiOk !== "ok") return;
    if (!pageVisible && !shouldPollBotStatusOps) return;
    void fetchBotStatusOps({ silent: true });
    const pollMs = pageVisible ? 60_000 : 180_000;
    const t = window.setInterval(() => {
      if (!pageVisible && !shouldPollBotStatusOps) return;
      void fetchBotStatusOps({ silent: true });
    }, pollMs);
    return () => window.clearInterval(t);
  }, [apiOk, fetchBotStatusOps, pageVisible, shouldPollBotStatusOps]);

  useEffect(() => {
    const tailTarget =
      pageVisible && shouldPollBotStatusOps ? BOT_STATUS_TAIL_POINTS : BOT_STATUS_TAIL_FALLBACK_POINTS;
    botStatusTailRef.current = tailTarget;
  }, [pageVisible, shouldPollBotStatusOps]);

  useEffect(() => {
    const limitTarget =
      pageVisible && shouldPollBotStatusOps ? BOT_STATUS_OPS_LIMIT : BOT_STATUS_OPS_FALLBACK_LIMIT;
    botStatusOpsLimitRef.current = limitTarget;
  }, [pageVisible, shouldPollBotStatusOps]);

  useEffect(() => {
    if (!opsPerformancePanelOpen || apiOk !== "ok") return;
    void fetchOpsPerformance({ silent: true });
  }, [
    apiOk,
    fetchOpsPerformance,
    opsPerformanceComboLimit,
    opsPerformanceComboOrder,
    opsPerformanceComboScope,
    opsPerformanceCommitLimit,
    opsPerformancePanelOpen,
  ]);

  const stopLiveBot = useCallback(async (symbol?: string) => {
    setBot((s) => ({ ...s, loading: true, error: null }));
    try {
      if (!activeTenantKey) {
        const msg = "Tenant key required. Add API keys (or check keys) to stop the bot.";
        setBot((s) => ({ ...s, loading: false, error: msg }));
        showToast("Bot stop failed");
        return;
      }
      const out = await botStop(apiBase, { headers: authHeaders, timeoutMs: 30_000 }, symbol, activeTenantKey);
      setBot((s) => ({ ...s, loading: false, error: null, status: out }));
      appendDataLog("Bot Stop Response", out);
      botAutoStartSuppressedRef.current = true;
      showToast(symbol ? `Bot stopped (${symbol})` : "Bot stopped");
    } catch (e) {
      if (isAbortError(e)) return;
      const msg = e instanceof Error ? e.message : String(e);
      setBot((s) => ({ ...s, loading: false, error: msg }));
      appendDataLog("Bot Stop Error", buildDataLogError(e, msg), { error: true });
      showToast("Bot stop failed");
    }
  }, [activeTenantKey, apiBase, authHeaders, showToast]);

  const binanceTradesSymbols = useMemo(() => parseSymbolsInput(binanceTradesSymbolsInput), [binanceTradesSymbolsInput]);
  const binanceTradesSymbolsInvalid = useMemo(
    () => invalidSymbolsForPlatform("binance", binanceTradesSymbols),
    [binanceTradesSymbols],
  );
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
        binancePrivateKeysIssue,
        form.market !== "futures" && binanceTradesSymbols.length === 0 ? "Symbol is required for spot/margin trades." : null,
        binanceTradesSymbolsInvalid.length > 0
          ? `Symbols must match Binance format (e.g., ${symbolFormatExample("binance")}). Invalid: ${binanceTradesSymbolsInvalid.join(", ")}.`
          : null,
        binanceTradesStartInput.trim() && binanceTradesStartMs === null
          ? "Start time must be a unix ms timestamp or ISO date."
          : null,
        binanceTradesEndInput.trim() && binanceTradesEndMs === null ? "End time must be a unix ms timestamp or ISO date." : null,
        binanceTradesStartMs !== null && binanceTradesEndMs !== null && binanceTradesEndMs < binanceTradesStartMs
          ? "End time must be after start time."
          : null,
        binanceTradesFromIdInput.trim() && binanceTradesFromId === null ? "From ID must be a non-negative integer." : null,
      ),
    [
      binanceTradesEndInput,
      binanceTradesEndMs,
      binanceTradesFromId,
      binanceTradesFromIdInput,
      binanceTradesStartInput,
      binanceTradesStartMs,
      binanceTradesSymbols.length,
      binanceTradesSymbolsInvalid,
      binancePrivateKeysIssue,
      form.market,
      isBinancePlatform,
    ],
  );

  const binancePositionsBarsError = useMemo(() => {
    if (!Number.isFinite(binancePositionsBars)) return "Chart bars must be a number.";
    if (binancePositionsBars <= 0) return "Chart bars must be greater than zero.";
    return null;
  }, [binancePositionsBars]);
  const binancePositionsLimitSafe = useMemo(
    () => clamp(Math.trunc(binancePositionsBars), 10, 1000),
    [binancePositionsBars],
  );
  const binancePositionsInputError = useMemo(
    () =>
      firstReason(
        !isBinancePlatform ? "Binance positions require platform=binance." : null,
        form.market !== "futures" ? "Binance positions are supported for futures only." : null,
        binancePrivateKeysIssue,
        binancePositionsBarsError,
      ),
    [binancePositionsBarsError, binancePrivateKeysIssue, form.market, isBinancePlatform],
  );

  const fetchBinanceTrades = useCallback(async () => {
    binanceTradesAbortRef.current?.abort();
    const controller = new AbortController();
    binanceTradesAbortRef.current = controller;
    setBinanceTradesUi((s) => ({ ...s, loading: true, error: null }));
    if (binanceTradesInputError) {
      setBinanceTradesUi((s) => ({ ...s, loading: false, error: binanceTradesInputError }));
      if (binanceTradesAbortRef.current === controller) binanceTradesAbortRef.current = null;
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
      const out = await binanceTrades(apiBase, withBinanceKeys(params), {
        headers: authHeaders,
        timeoutMs: 30_000,
        signal: controller.signal,
      });
      setBinanceTradesUi({ loading: false, error: null, response: out });
    } catch (e) {
      if (isAbortError(e)) return;
      const msg = e instanceof Error ? e.message : String(e);
      setBinanceTradesUi((s) => ({ ...s, loading: false, error: msg }));
    } finally {
      if (binanceTradesAbortRef.current === controller) binanceTradesAbortRef.current = null;
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

  const fetchBinancePositions = useCallback(async (opts?: { retrying?: boolean }) => {
    binancePositionsAbortRef.current?.abort();
    const controller = new AbortController();
    binancePositionsAbortRef.current = controller;
    setBinancePositionsUi((s) => ({ ...s, loading: true, error: null }));
    if (binancePositionsInputError) {
      setBinancePositionsUi((s) => ({ ...s, loading: false, error: binancePositionsInputError }));
      if (binancePositionsAbortRef.current === controller) binancePositionsAbortRef.current = null;
      return;
    }
    try {
      const params: ApiBinancePositionsRequest = {
        market: form.market,
        binanceTestnet: form.binanceTestnet,
        interval: form.interval.trim(),
        limit: binancePositionsLimitSafe,
      };
      const out = await binancePositions(apiBase, withBinanceKeys(params), {
        headers: authHeaders,
        timeoutMs: 30_000,
        signal: controller.signal,
      });
      setBinancePositionsUi({ loading: false, error: null, response: out });
    } catch (e) {
      if (isAbortError(e)) return;
      const msg = e instanceof Error ? e.message : String(e);
      const isTimestampError = isBinanceTimestampErrorMessage(msg);
      if (isTimestampError && !opts?.retrying) {
        setBinancePositionsUi((s) => ({
          ...s,
          loading: true,
          error: "Binance timestamp out of sync (code -1021). Retrying after time sync…",
        }));
        window.setTimeout(() => {
          void fetchBinancePositions({ retrying: true });
        }, 750);
        return;
      }
      const finalMsg = isTimestampError
        ? "Binance timestamp out of sync (code -1021). Ensure system time is synced and the Binance time endpoint is reachable, then retry."
        : msg;
      setBinancePositionsUi((s) => ({ ...s, loading: false, error: finalMsg }));
    } finally {
      if (binancePositionsAbortRef.current === controller) binancePositionsAbortRef.current = null;
    }
  }, [
    apiBase,
    authHeaders,
    binancePositionsInputError,
    binancePositionsLimitSafe,
    form.binanceTestnet,
    form.interval,
    form.market,
    withBinanceKeys,
  ]);

  const binancePositionsList = useMemo(() => {
    const raw = binancePositionsUi.response?.positions ?? [];
    return [...raw].sort((a, b) => a.symbol.localeCompare(b.symbol));
  }, [binancePositionsUi.response?.positions]);
  const binancePositionsCharts = useMemo(() => {
    const charts = binancePositionsUi.response?.charts ?? [];
    const map = new Map<string, BinancePositionChart>();
    for (const chart of charts) map.set(chart.symbol, chart);
    return map;
  }, [binancePositionsUi.response?.charts]);
  const orphanPositions = useMemo(
    () =>
      buildOrphanedPositions(binancePositionsList, botEntriesWithSymbol, {
        market: binancePositionsUi.response?.market ?? form.market,
      }),
    [binancePositionsList, binancePositionsUi.response?.market, botEntriesWithSymbol, form.market],
  );

  useEffect(() => {
    if (apiOk !== "ok") return;
    if (binancePositionsInputError) return;
    if (!binancePrivateKeysReady) return;
    if (!pageVisible) return;
    const interval = form.interval.trim();
    if (!interval) return;
    const authKey = `${apiBase}:${apiToken.trim()}:${binanceApiKey.trim()}:${binanceApiSecret.trim()}`;
    const key = `${form.market}:${form.binanceTestnet ? "t" : "f"}:${interval}:${binancePositionsLimitSafe}:${authKey}`;
    if (binancePositionsAutoKeyRef.current === key) return;
    binancePositionsAutoKeyRef.current = key;
    void fetchBinancePositions();
  }, [
    apiOk,
    apiBase,
    authHeaders,
    apiToken,
    binanceApiKey,
    binanceApiSecret,
    binancePrivateKeysReady,
    binancePositionsInputError,
    binancePositionsLimitSafe,
    fetchBinancePositions,
    form.binanceTestnet,
    form.interval,
    form.market,
    pageVisible,
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
    const pollMs = pageVisible ? BOT_STATUS_POLL_MS : Math.max(BOT_STATUS_POLL_MS * 3, 30_000);
    const t = window.setInterval(() => {
      if (bot.loading) return;
      void refreshBot({ silent: true });
    }, pollMs);
    return () => window.clearInterval(t);
  }, [apiOk, bot.loading, bot.status, pageVisible, refreshBot]);

  useEffect(() => {
    if (!form.autoRefresh || apiOk !== "ok" || !pageVisible) return;
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
  }, [apiOk, form.autoRefresh, form.autoRefreshSec, pageVisible, run, commonParams]);

  useEffect(() => {
    if (!state.error) return;
    errorRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [state.error]);

  useEffect(() => {
    let isCancelled = false;
    let inFlight = false;
    const syncTopCombos = async (opts?: { silent?: boolean }) => {
      if (isCancelled || inFlight) return;
      inFlight = true;
      topCombosAbortRef.current?.abort();
      const controller = new AbortController();
      topCombosAbortRef.current = controller;
      const silent = opts?.silent ?? false;
      if (!silent) setTopCombosLoading(true);
      try {
        if (apiOk !== "ok") {
          const fallbackReason =
            apiOk === "auth"
              ? "API unauthorized"
              : apiOk === "down"
                ? "API unreachable"
                : apiOk === "unknown"
                  ? "API unknown"
                  : "API unavailable";
          throw new Error(fallbackReason);
        }
        const payload = await optimizerCombos(apiBase, { headers: authHeaders, signal: controller.signal });
        const source: TopCombosSource = "api";
        const fallbackReason: string | null = null;
        if (isCancelled) return;
        const payloadObj = payload && typeof payload === "object" ? (payload as Record<string, unknown>) : null;
        const payloadRec = payloadObj ?? {};
        const rawCombos: unknown[] = Array.isArray(payloadRec.combos) ? (payloadRec.combos as unknown[]) : [];
        const generatedAtMsRaw = payloadRec.generatedAtMs;
        const generatedAtMs =
          typeof generatedAtMsRaw === "number" && Number.isFinite(generatedAtMsRaw) ? Math.trunc(generatedAtMsRaw) : null;
        const methods: Method[] = ["11", "10", "01", "blend", "router"];
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
          const rawSource = typeof rawRec.source === "string" ? rawRec.source : null;
          const source: OptimizationCombo["source"] =
            rawSource === "binance" || rawSource === "coinbase" || rawSource === "kraken" || rawSource === "poloniex" || rawSource === "csv"
              ? rawSource
              : null;
          const resolvedPlatform =
            platform ?? (source && source !== "csv" ? (source as Platform) : null);
          const binanceSymbol = normalizeComboSymbol(rawSymbol, resolvedPlatform);
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
          const rebalanceBars =
            typeof params.rebalanceBars === "number" && Number.isFinite(params.rebalanceBars)
              ? Math.max(0, Math.trunc(params.rebalanceBars))
              : null;
          const rebalanceThreshold =
            typeof params.rebalanceThreshold === "number" && Number.isFinite(params.rebalanceThreshold)
              ? Math.max(0, params.rebalanceThreshold)
              : null;
          const rebalanceCostMult =
            typeof params.rebalanceCostMult === "number" && Number.isFinite(params.rebalanceCostMult)
              ? Math.max(0, params.rebalanceCostMult)
              : null;
          const rebalanceGlobal = typeof params.rebalanceGlobal === "boolean" ? params.rebalanceGlobal : null;
          const rebalanceResetOnSignal = typeof params.rebalanceResetOnSignal === "boolean" ? params.rebalanceResetOnSignal : null;
          const fundingRate =
            typeof params.fundingRate === "number" && Number.isFinite(params.fundingRate) ? params.fundingRate : null;
          const fundingBySide = typeof params.fundingBySide === "boolean" ? params.fundingBySide : null;
          const fundingOnOpen = typeof params.fundingOnOpen === "boolean" ? params.fundingOnOpen : null;
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
          const walkForwardEmbargoBars =
            typeof params.walkForwardEmbargoBars === "number" && Number.isFinite(params.walkForwardEmbargoBars)
              ? Math.max(0, Math.trunc(params.walkForwardEmbargoBars))
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
          const orderQuoteRaw =
            typeof params.orderQuote === "number" && Number.isFinite(params.orderQuote) ? params.orderQuote : null;
          const orderQuote = orderQuoteRaw != null && orderQuoteRaw > 0 ? Math.max(0, orderQuoteRaw) : null;
          const orderQuantityRaw =
            typeof params.orderQuantity === "number" && Number.isFinite(params.orderQuantity) ? params.orderQuantity : null;
          const orderQuantity = orderQuantityRaw != null && orderQuantityRaw > 0 ? Math.max(0, orderQuantityRaw) : null;
          const orderQuoteFractionRaw =
            typeof params.orderQuoteFraction === "number" && Number.isFinite(params.orderQuoteFraction) ? params.orderQuoteFraction : null;
          const orderQuoteFraction =
            orderQuoteFractionRaw != null && orderQuoteFractionRaw > 0 ? clamp(orderQuoteFractionRaw, 0, 1) : null;
          const maxOrderQuoteRaw =
            typeof params.maxOrderQuote === "number" && Number.isFinite(params.maxOrderQuote) ? params.maxOrderQuote : null;
          const maxOrderQuote = maxOrderQuoteRaw != null && maxOrderQuoteRaw > 0 ? Math.max(0, maxOrderQuoteRaw) : null;
          const createdAtMsRaw = rawRec.createdAtMs;
          const createdAtMs =
            typeof createdAtMsRaw === "number" && Number.isFinite(createdAtMsRaw) ? Math.trunc(createdAtMsRaw) : null;
          const createdAtMsFinal = createdAtMs ?? generatedAtMs;
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
          const annualizedReturn =
            typeof metricsRec["annualizedReturn"] === "number" && Number.isFinite(metricsRec["annualizedReturn"])
              ? (metricsRec["annualizedReturn"] as number)
              : null;
          const metrics =
            sharpe != null || maxDrawdown != null || turnover != null || roundTrips != null || annualizedReturn != null
              ? { sharpe, maxDrawdown, turnover, roundTrips, annualizedReturn }
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
          return {
            id: rank ?? index + 1,
            rank,
            createdAtMs: createdAtMsFinal,
            objective,
            score,
            metrics,
            finalEquity: typeof rawRec.finalEquity === "number" && Number.isFinite(rawRec.finalEquity) ? rawRec.finalEquity : 0,
            openThreshold: typeof rawRec.openThreshold === "number" ? rawRec.openThreshold : null,
            closeThreshold: typeof rawRec.closeThreshold === "number" ? rawRec.closeThreshold : null,
            source,
            operations: operationsOut,
            params: {
              ...params,
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
              rebalanceBars,
              rebalanceThreshold,
              rebalanceCostMult,
              rebalanceGlobal,
              fundingRate,
              fundingBySide,
              rebalanceResetOnSignal,
              fundingOnOpen,
              periodsPerYear,
              blendWeight,
              walkForwardFolds,
              walkForwardEmbargoBars,
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
              orderQuote,
              orderQuantity,
              orderQuoteFraction,
              maxOrderQuote,
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
          const aAnnEq = comboAnnualizedEquity(a);
          const bAnnEq = comboAnnualizedEquity(b);
          if (aAnnEq != null || bAnnEq != null) {
            const diff = (bAnnEq ?? Number.NEGATIVE_INFINITY) - (aAnnEq ?? Number.NEGATIVE_INFINITY);
            if (diff !== 0) return diff;
          }

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
        if (payloadObj && Array.isArray(payloadObj.combos)) {
          setTopCombosPayload(payloadObj);
        }
        setTopCombosAll(sanitized);
        const comboCount = rawCombos.length;
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
        const topCombo = sanitized[0];
        if (topCombo && autoApplyTopCombo) {
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
            if (!pendingComboStartRef.current) {
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
        }
      } catch (err) {
        if (isCancelled || isAbortError(err)) return;
        const msg = err instanceof Error ? err.message : "Failed to load optimizer combos.";
        if (!silent || topCombosRef.current.length === 0) {
          setTopCombosError(msg);
        }
      } finally {
        inFlight = false;
        if (isCancelled) return;
        if (topCombosAbortRef.current === controller) topCombosAbortRef.current = null;
        if (!silent) setTopCombosLoading(false);
      }
    };

    topCombosSyncRef.current = syncTopCombos;
    void syncTopCombos();
    const t = window.setInterval(() => {
      void syncTopCombos({ silent: true });
    }, pageVisible ? TOP_COMBOS_POLL_MS : TOP_COMBOS_POLL_MS * 3);
    return () => {
      isCancelled = true;
      window.clearInterval(t);
      topCombosAbortRef.current?.abort();
      topCombosAbortRef.current = null;
      if (topCombosSyncRef.current === syncTopCombos) {
        topCombosSyncRef.current = null;
      }
    };
  }, [apiBase, apiOk, authHeaders, applyCombo, autoApplyTopCombo, pageVisible, showToast]);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      botAbortRef.current?.abort();
      keysAbortRef.current?.abort();
      optimizerRunAbortRef.current?.abort();
      listenKeyStreamAbortRef.current?.abort();
      botStatusOpsAbortRef.current?.abort();
      topCombosAbortRef.current?.abort();
      binanceTradesAbortRef.current?.abort();
      binancePositionsAbortRef.current?.abort();
    };
  }, []);

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
  const topCombo = topCombosAll.length > 0 ? topCombosAll[0] : null;
  const topComboDisplay: OptimizationCombo | null = topCombosOrdered.length > 0 ? topCombosOrdered[0] ?? null : null;
  const selectedCombo = useMemo(() => {
    if (selectedComboId == null) return null;
    return topCombosAll.find((combo) => combo.id === selectedComboId) ?? null;
  }, [selectedComboId, topCombosAll]);
  const selectedComboForm = useMemo(() => {
    if (!selectedCombo) return null;
    return applyComboToForm(form, selectedCombo, apiComputeLimits, undefined, true);
  }, [apiComputeLimits, form, selectedCombo]);
  const selectedComboLabel = selectedCombo ? `#${selectedCombo.rank ?? selectedCombo.id}` : null;
  const selectedComboSymbol = selectedComboForm ? selectedComboForm.binanceSymbol.trim().toUpperCase() : "";
  const selectedComboStartLabel =
    selectedComboLabel && selectedComboSymbol
      ? `Start bot with ${selectedComboLabel} (${selectedComboSymbol})`
      : selectedComboLabel
        ? `Start bot with ${selectedComboLabel}`
        : "Start bot with selected combo";
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
    const barsCap = maxBarsForPlatform(platform, form.method, apiComputeLimits);
    const bars =
      barsRaw <= 0
        ? 0
        : (() => {
            const normalized = Math.max(MIN_LOOKBACK_BARS, barsRaw);
            if (Number.isFinite(barsCap)) return Math.min(normalized, barsCap);
            return normalized;
          })();
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
  }, [apiComputeLimits, form.bars, form.interval, form.lookbackBars, form.lookbackWindow, form.method, platform]);
  const splitPreview = useMemo(() => {
    const bars = lookbackState.bars;
    const lookbackBars = lookbackState.effectiveBars;
    const backtestRatio =
      typeof form.backtestRatio === "number" && Number.isFinite(form.backtestRatio) ? clamp(form.backtestRatio, 0.01, 0.99) : 0.2;
    const tuneRatio =
      typeof form.tuneRatio === "number" && Number.isFinite(form.tuneRatio) ? clamp(form.tuneRatio, 0, 0.99) : 0;
    const tuningEnabled = form.optimizeOperations || form.sweepThreshold;

    if (!Number.isFinite(bars) || bars <= 0) {
      return { summary: "Split preview: set Bars to compute train/backtest sizes.", warning: false };
    }
    if (lookbackBars == null || lookbackBars < 2) {
      return { summary: "Split preview: set a lookback window or override to compute minimum bars.", warning: false };
    }

    const trainEndRaw = Math.floor(bars * (1 - backtestRatio) + 1e-9);
    const backtestBars = Math.max(0, bars - trainEndRaw);
    const minTrainBars = lookbackBars + 1;
    const trainOk = trainEndRaw >= minTrainBars;
    const backtestOk = backtestBars >= 2;

    let tuneBars = 0;
    let fitBars = trainEndRaw;
    let tuneOk = true;
    let fitOk = true;
    if (tuningEnabled) {
      tuneBars = Math.max(0, Math.min(trainEndRaw, Math.floor(trainEndRaw * tuneRatio)));
      fitBars = Math.max(0, trainEndRaw - tuneBars);
      tuneOk = tuneBars >= 2;
      fitOk = fitBars >= minTrainBars;
    }

    const minBarsForTrain = Math.ceil(minTrainBars / Math.max(1e-6, 1 - backtestRatio));
    const baseSummary = tuningEnabled
      ? `Split preview: bars=${bars} → fit=${fitBars}, tune=${tuneBars}, backtest=${backtestBars}.`
      : `Split preview: bars=${bars} → train=${trainEndRaw}, backtest=${backtestBars}.`;
    const minSummary = `Min bars for lookback=${lookbackBars}: ${minBarsForTrain}.`;
    const warnings: string[] = [];
    if (!trainOk) warnings.push(`Train must be ≥ ${minTrainBars} bars.`);
    if (!backtestOk) warnings.push("Backtest needs ≥ 2 bars.");
    if (tuningEnabled && !tuneOk) warnings.push("Tune needs ≥ 2 bars.");
    if (tuningEnabled && !fitOk) warnings.push(`Fit must be ≥ ${minTrainBars} bars.`);
    const summary = warnings.length > 0 ? `${baseSummary} ${minSummary} ${warnings.join(" ")}` : `${baseSummary} ${minSummary}`;

    return { summary, warning: warnings.length > 0 };
  }, [
    form.backtestRatio,
    form.tuneRatio,
    form.optimizeOperations,
    form.sweepThreshold,
    lookbackState.bars,
    lookbackState.effectiveBars,
  ]);
  const errorFix = useMemo<ErrorFix | null>(() => {
    if (!state.error) return null;
    const error = state.error;
    const errorLower = error.toLowerCase();
    const fitError = errorLower.includes("fit window too small");
    const tuneError = errorLower.includes("tune window too small");
    const backtestRatioError = errorLower.includes("backtest-ratio") || errorLower.includes("backtest ratio");
    const lookbackError =
      errorLower.includes("lookback") &&
      (errorLower.includes("not enough") || errorLower.includes("need") || errorLower.includes("train/backtest"));
    if (!fitError && !tuneError && !backtestRatioError && !lookbackError) return null;

    const tuningEnabled = form.optimizeOperations || form.sweepThreshold;
    const isBacktest = state.lastKind === "backtest";

    const lookbackMatch = error.match(/lookback=(\d+)/i);
    const lookbackFromError = lookbackMatch ? Number(lookbackMatch[1]) : null;
    const lookbackBars =
      Number.isFinite(lookbackFromError) && lookbackFromError != null
        ? lookbackFromError
        : lookbackState.effectiveBars;

    const bars = lookbackState.bars;
    const backtestRatio =
      typeof form.backtestRatio === "number" && Number.isFinite(form.backtestRatio)
        ? clamp(form.backtestRatio, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO)
        : 0.2;
    const currentTuneRatio =
      typeof form.tuneRatio === "number" && Number.isFinite(form.tuneRatio) ? clamp(form.tuneRatio, 0, 0.99) : 0;
    const barsCap = maxBarsForPlatform(platform, form.method, apiComputeLimits);
    const gotPricesMatch = error.match(/got\s+(\d+)/i);
    const gotPrices = gotPricesMatch ? Number(gotPricesMatch[1]) : null;
    const needBarsMatch = error.match(/need bars >= (\d+)/i);
    const needPricesMatch = error.match(/need >= (\d+) prices/i);
    const neededBars = needBarsMatch ? Number(needBarsMatch[1]) : needPricesMatch ? Number(needPricesMatch[1]) : null;

    const makeBarsFix = (nextBars: number, reason: string): ErrorFix | null => {
      if (!Number.isFinite(nextBars) || nextBars <= 0) return null;
      const normalized = Math.max(MIN_LOOKBACK_BARS, Math.trunc(nextBars));
      if (bars > 0 && normalized <= bars) return null;
      if (Number.isFinite(barsCap) && normalized > barsCap) return null;
      return {
        label: `Suggested fix: set bars to ${normalized}.`,
        action: "bars",
        value: normalized,
        targetId: "bars",
        toast: `Bars set to ${normalized} ${reason}.`,
      };
    };

    const makeBacktestRatioFix = (nextRatio: number, reason: string): ErrorFix | null => {
      if (!Number.isFinite(nextRatio)) return null;
      const normalized = clamp(nextRatio, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
      if (Math.abs(normalized - backtestRatio) < 1e-6) return null;
      const label = fmtNum(normalized, RATIO_ROUND_DIGITS);
      return {
        label: `Suggested fix: set backtest ratio to ${label}.`,
        action: "backtestRatio",
        value: normalized,
        targetId: "backtestRatio",
        toast: `Backtest ratio set to ${label} ${reason}.`,
      };
    };

    const makeTuneRatioFix = (nextRatio: number, reason: string): ErrorFix | null => {
      if (!Number.isFinite(nextRatio)) return null;
      const normalized = clamp(nextRatio, 0, 0.99);
      if (Math.abs(normalized - currentTuneRatio) < 1e-6) return null;
      const label = fmtNum(normalized, RATIO_ROUND_DIGITS);
      return {
        label: `Suggested fix: set tune ratio to ${label}.`,
        action: "tuneRatio",
        value: normalized,
        targetId: "tuneRatio",
        toast: `Tune ratio set to ${label} ${reason}.`,
      };
    };

    const tuneFix = (() => {
      if (!isBacktest || !tuningEnabled) return null;
      if (!Number.isFinite(lookbackBars) || lookbackBars == null || lookbackBars < MIN_LOOKBACK_BARS) return null;
      if (!Number.isFinite(bars) || bars <= 0) return null;
      const bounds = tuneRatioBounds(bars, backtestRatio, lookbackBars);
      if (!bounds) return null;
      if (bounds.maxRatio < bounds.minRatio || bounds.maxTuneBars < bounds.minTuneBars) return null;

      if (fitError) {
        if (currentTuneRatio <= bounds.maxRatio + 1e-6) return null;
        let nextTuneRatio = roundRatioDown(Math.min(currentTuneRatio, bounds.maxRatio));
        if (nextTuneRatio < bounds.minRatio) nextTuneRatio = bounds.maxRatio;
        if (nextTuneRatio >= currentTuneRatio || nextTuneRatio < bounds.minRatio - 1e-6) return null;
        return makeTuneRatioFix(nextTuneRatio, "to keep fit >= lookback");
      }

      if (tuneError) {
        if (currentTuneRatio >= bounds.minRatio - 1e-6) return null;
        let nextTuneRatio = roundRatioUp(Math.max(currentTuneRatio, bounds.minRatio));
        if (nextTuneRatio > bounds.maxRatio) nextTuneRatio = bounds.minRatio;
        if (nextTuneRatio <= currentTuneRatio || nextTuneRatio > bounds.maxRatio + 1e-6) return null;
        return makeTuneRatioFix(nextTuneRatio, "to reach at least 2 tune bars");
      }

      return null;
    })();

    const backtestAdjust = isBacktest ? adjustBacktestParams(commonParams) : null;
    const backtestAdjustFix = (() => {
      const changes = backtestAdjust?.changes ?? null;
      if (!changes) return null;
      if (changes.bars !== undefined) {
        return makeBarsFix(changes.bars, "to satisfy the split");
      }
      if (changes.backtestRatio !== undefined) {
        return makeBacktestRatioFix(changes.backtestRatio, "to satisfy the split");
      }
      return null;
    })();

    const backtestBarsFix = backtestAdjustFix?.action === "bars" ? backtestAdjustFix : null;
    const backtestRatioFix = backtestAdjustFix?.action === "backtestRatio" ? backtestAdjustFix : null;

    const simpleBarsFix = (() => {
      if (isBacktest) return null;
      const minBars = Number.isFinite(neededBars) && neededBars != null ? neededBars : lookbackBars != null ? lookbackBars + 1 : null;
      if (!Number.isFinite(minBars) || minBars == null) return null;
      return makeBarsFix(minBars, "to fit the lookback");
    })();

    const lookbackFix: ErrorFix | null = (() => {
      if (!Number.isFinite(lookbackBars) || lookbackBars == null || lookbackBars < MIN_LOOKBACK_BARS) return null;
      let maxLookback: number | null = null;
      if (Number.isFinite(gotPrices) && gotPrices != null && gotPrices > 1) {
        maxLookback = gotPrices - 1;
      } else if (isBacktest) {
        maxLookback = maxLookbackForSplit(bars, backtestRatio, currentTuneRatio, tuningEnabled);
      } else if (Number.isFinite(bars) && bars > 0) {
        maxLookback = bars - 1;
      }
      if (!Number.isFinite(maxLookback) || maxLookback == null) return null;
      const normalizedMax = Math.trunc(maxLookback);
      if (normalizedMax < MIN_LOOKBACK_BARS || normalizedMax >= lookbackBars) return null;

      const intervalSec = lookbackState.intervalSec;
      if (lookbackState.overrideOn || !intervalSec) {
        const fix: ErrorFix = {
          label: `Suggested fix: set lookback bars to ${normalizedMax}.`,
          action: "lookbackBars",
          value: normalizedMax,
          targetId: "lookbackBars",
          toast: `Lookback bars set to ${normalizedMax} to fit available bars.`,
        };
        return fix;
      }

      const window = formatDurationSeconds(normalizedMax * intervalSec);
      const fix: ErrorFix = {
        label: `Suggested fix: set lookback window to ${window}.`,
        action: "lookbackWindow",
        value: window,
        targetId: "lookbackWindow",
        toast: `Lookback window set to ${window} to fit available bars.`,
      };
      return fix;
    })();

    if (fitError) return tuneFix ?? lookbackFix ?? backtestBarsFix ?? null;
    if (tuneError) return tuneFix ?? backtestRatioFix ?? backtestBarsFix ?? null;
    if (backtestRatioError) return backtestRatioFix ?? backtestBarsFix ?? null;
    if (lookbackError) return (isBacktest ? backtestBarsFix : simpleBarsFix) ?? lookbackFix ?? null;
    return (isBacktest ? backtestBarsFix ?? backtestRatioFix : simpleBarsFix ?? lookbackFix) ?? null;
  }, [
    apiComputeLimits,
    commonParams,
    form.backtestRatio,
    form.method,
    form.optimizeOperations,
    form.sweepThreshold,
    form.tuneRatio,
    lookbackState.bars,
    lookbackState.effectiveBars,
    lookbackState.intervalSec,
    lookbackState.overrideOn,
    platform,
    state.error,
    state.lastKind,
  ]);
  const dataLogFiltered = useMemo(() => {
    const term = dataLogFilterText.trim().toLowerCase();
    if (!term) return dataLog;
    return dataLog.filter((entry) => entry.label.toLowerCase().includes(term));
  }, [dataLog, dataLogFilterText]);
  const dataLogShown = useMemo(
    () => (dataLogFilterText.trim() ? dataLogFiltered : dataLog),
    [dataLog, dataLogFilterText, dataLogFiltered],
  );
  const dataLogShownDeferred = useDeferredValue(dataLogShown);
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
      : "Backend unreachable. Configure apiBaseUrl in trader-config.js (CORS required for cross-origin) or configure CloudFront to forward `/api/*` to your API origin.";
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
  const barsDefaultForLimits = PLATFORM_DEFAULT_BARS[platform] ?? 500;
  const barsEffectiveForLimits = barsRawForLimits <= 0 ? barsDefaultForLimits : barsRawForLimits;
  const barsMaxForInput = maxBarsForPlatform(platform, form.method, apiComputeLimits);
  const barsMaxLabel = Number.isFinite(barsMaxForInput) ? Math.trunc(barsMaxForInput) : null;
  const barsRangeLabel = barsMaxLabel ? `${MIN_LOOKBACK_BARS}–${barsMaxLabel}` : `>=${MIN_LOOKBACK_BARS}`;
  const barsAutoHint = isBinancePlatform
    ? `0=auto (Binance uses ${barsDefaultForLimits}; CSV uses all).`
    : `0=auto (exchange default ${barsDefaultForLimits}; CSV uses all).`;
  const barsRangeHint = barsMaxLabel
    ? `${platformLabel} allows ${barsRangeLabel}${apiLstmEnabled ? " for LSTM methods" : ""}. Larger values take longer.`
    : `${platformLabel} requires at least ${MIN_LOOKBACK_BARS} bars. Larger values take longer.`;
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
      symbolError: symbolFormatError,
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
      symbolFormatError,
    ],
  );
  const requestIssueDetails = useMemo(() => buildRequestIssueDetails(requestIssueInput), [requestIssueInput]);
  const botIssueDetails = useMemo(() => {
    const base = buildRequestIssueDetails({
      ...requestIssueInput,
      missingSymbol: botMissingSymbol,
      symbolTargetId: "botSymbols",
      symbolError: botSymbolsFormatError,
    });
    if (!botTradeKeysIssue) return base;
    return [...base, { message: botTradeKeysIssue, targetId: "platformKeys" }];
  }, [botMissingSymbol, botSymbolsFormatError, botTradeKeysIssue, requestIssueInput]);
  const requestIssues = useMemo(() => requestIssueDetails.map((issue) => issue.message), [requestIssueDetails]);
  const primaryIssue = requestIssueDetails[0] ?? null;
  const extraIssueCount = Math.max(0, requestIssueDetails.length - 1);
  const requestDisabledReason = primaryIssue?.disabledMessage ?? primaryIssue?.message ?? null;
  const requestDisabled = state.loading || Boolean(requestDisabledReason);
  const applyErrorFix = useCallback(() => {
    if (!errorFix) return;
    if (errorFix.action === "tuneRatio") {
      setForm((prev) => {
        const nextTuneRatio = clamp(errorFix.value, 0, 0.99);
        return nextTuneRatio === prev.tuneRatio ? prev : { ...prev, tuneRatio: nextTuneRatio };
      });
    } else if (errorFix.action === "backtestRatio") {
      setForm((prev) => {
        const nextRatio = clamp(errorFix.value, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
        return nextRatio === prev.backtestRatio ? prev : { ...prev, backtestRatio: nextRatio };
      });
    } else if (errorFix.action === "bars") {
      setForm((prev) => {
        const nextBars = Math.max(MIN_LOOKBACK_BARS, Math.trunc(errorFix.value));
        return nextBars === prev.bars ? prev : { ...prev, bars: nextBars };
      });
    } else if (errorFix.action === "lookbackBars") {
      setForm((prev) => {
        const nextBars = Math.max(MIN_LOOKBACK_BARS, Math.trunc(errorFix.value));
        return nextBars === prev.lookbackBars ? prev : { ...prev, lookbackBars: nextBars };
      });
    } else if (errorFix.action === "lookbackWindow") {
      setForm((prev) => {
        const nextWindow = errorFix.value.trim();
        if (nextWindow === prev.lookbackWindow && prev.lookbackBars === 0) return prev;
        return { ...prev, lookbackWindow: nextWindow, lookbackBars: 0 };
      });
    }
    showToast(errorFix.toast);
    if (errorFix.targetId) scrollToSection(errorFix.targetId);
  }, [errorFix, scrollToSection, showToast]);
  const handlePrimaryIssueFix = useCallback(() => {
    const targetId = primaryIssue?.targetId;
    if (!targetId) return;
    if (apiComputeLimits && apiLstmEnabled) {
      if (targetId === "bars" && barsExceedsApi) {
        const maxBarsRaw = maxBarsForPlatform(platform, form.method, apiComputeLimits);
        if (Number.isFinite(maxBarsRaw) && maxBarsRaw > 0) {
          const maxBars = Math.max(MIN_LOOKBACK_BARS, Math.trunc(maxBarsRaw));
          setForm((prev) => {
            const currentBars = Math.trunc(prev.bars);
            const nextBars = currentBars > 0 ? Math.min(currentBars, maxBars) : maxBars;
            return nextBars === prev.bars ? prev : { ...prev, bars: nextBars };
          });
          showToast(`Bars set to ${maxBars} (API limit).`);
        }
      } else if (targetId === "epochs" && epochsExceedsApi) {
        const maxEpochs = Math.trunc(apiComputeLimits.maxEpochs);
        if (Number.isFinite(maxEpochs) && maxEpochs >= 0) {
          setForm((prev) => {
            const nextEpochs = clamp(Math.trunc(prev.epochs), 0, maxEpochs);
            return nextEpochs === prev.epochs ? prev : { ...prev, epochs: nextEpochs };
          });
          showToast(`Epochs set to ${maxEpochs} (API limit).`);
        }
      } else if (targetId === "hiddenSize" && hiddenSizeExceedsApi) {
        const maxHiddenSize = Math.trunc(apiComputeLimits.maxHiddenSize);
        if (Number.isFinite(maxHiddenSize) && maxHiddenSize > 0) {
          setForm((prev) => {
            const nextHidden = clamp(Math.trunc(prev.hiddenSize), 1, maxHiddenSize);
            return nextHidden === prev.hiddenSize ? prev : { ...prev, hiddenSize: nextHidden };
          });
          showToast(`Hidden size set to ${maxHiddenSize} (API limit).`);
        }
      }
    }
    scrollToSection(targetId);
  }, [
    apiComputeLimits,
    apiLstmEnabled,
    barsExceedsApi,
    epochsExceedsApi,
    form.method,
    hiddenSizeExceedsApi,
    platform,
    primaryIssue?.targetId,
    scrollToSection,
    showToast,
  ]);
  const botAnyRunning = bot.status.running;
  const botAnyStarting = "starting" in bot.status && bot.status.starting === true;
  const botStartPrimaryIssue = botIssueDetails[0] ?? null;
  const botStartDisabledReason = botStartPrimaryIssue?.disabledMessage ?? botStartPrimaryIssue?.message ?? null;
  const botStarting = apiOk === "ok" && !bot.error && botAnyStarting;
  const botStartingHint = useMemo(() => {
    if (!botStarting) return null;
    if (botStartingReason) {
      return `Bot is starting… ${botStartingReason} Use “Refresh” to check status.`;
    }
    return "Bot is starting… (initializing model). Use “Refresh” to check status.";
  }, [botStarting, botStartingReason]);
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
  const comboStartPending = pendingComboStart !== null;
  const comboStartBlockedReason = useMemo(() => {
    if (!selectedComboForm) return null;
    const comboSymbol = selectedComboForm.binanceSymbol.trim().toUpperCase();
    const interval = selectedComboForm.interval.trim();
    const intervalOk =
      interval.length > 0 && PLATFORM_INTERVAL_SET[selectedComboForm.platform].has(interval);
    return firstReason(
      rateLimitReason,
      apiBlockedReason ?? apiStatusIssue,
      botTradeKeysIssue,
      selectedComboForm.platform !== "binance" ? "Live bot is supported on Binance only." : null,
      !comboSymbol ? "Combo is missing a symbol." : null,
      !intervalOk ? "Combo is missing a valid interval." : null,
      selectedComboForm.positioning === "long-short" && selectedComboForm.market !== "futures"
        ? "Live bot long/short requires the Futures market."
        : null,
    );
  }, [apiBlockedReason, apiStatusIssue, botTradeKeysIssue, rateLimitReason, selectedComboForm]);
  const comboStartBlocked = bot.loading || botStarting || comboStartPending || Boolean(comboStartBlockedReason);
  const topComboAutoStartIssues = useMemo(
    () =>
      buildRequestIssueDetails({
        rateLimitReason,
        apiStatusIssue,
        apiBlockedReason,
        apiTargetId: "section-api",
        missingSymbol: false,
        symbolError: null,
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
      rateLimitReason,
    ],
  );
  const topComboAutoStartBlockedReason = useMemo(
    () =>
      firstReason(
        topComboAutoStartIssues[0]?.disabledMessage ?? topComboAutoStartIssues[0]?.message,
        !isBinancePlatform ? "Live bot is supported on Binance only." : null,
        form.positioning === "long-short" && form.market !== "futures"
          ? "Live bot long/short requires the Futures market."
          : null,
        botTradeKeysIssue,
      ),
    [botTradeKeysIssue, form.market, form.positioning, isBinancePlatform, topComboAutoStartIssues],
  );

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
  useEffect(() => {
    if (apiOk !== "ok") return;
    if (!botStatusFetchedRef.current) return;
    if (botAutoStartSuppressedRef.current) return;
    if (comboStartPending) return;
    if (bot.loading || botStarting) return;
    if (topComboAutoStartBlockedReason) return;
    if (topComboBotTargets.length === 0) return;

    const missing = topComboBotTargets.filter((sym) => !botActiveSymbolSet.has(normalizeSymbolKey(sym)));
    if (missing.length === 0) return;

    const now = Date.now();
    const last = botAutoStartTopCombosRef.current;
    if (last.lastKey === topComboBotTargetsKey && now - last.lastAttemptAtMs < BOT_AUTOSTART_RETRY_MS) return;
    botAutoStartTopCombosRef.current = { lastAttemptAtMs: now, lastKey: topComboBotTargetsKey };
    void startLiveBot({ auto: true, silent: true, symbolsOverride: topComboBotTargets });
  }, [
    apiOk,
    bot.loading,
    botActiveSymbolSet,
    botStarting,
    comboStartPending,
    startLiveBot,
    topComboAutoStartBlockedReason,
    topComboBotTargets,
    topComboBotTargetsKey,
  ]);
  useEffect(() => {
    if (!pendingComboStart) return;
    if (formApplySignature(form) !== pendingComboStart.signature) return;
    setPendingComboStart(null);
    if (comboStartBlockedReason) {
      showToast(`Start bot disabled: ${comboStartBlockedReason}`);
      return;
    }
    void startLiveBot({ symbolsOverride: pendingComboStart.symbols });
  }, [comboStartBlockedReason, form, pendingComboStart, showToast, startLiveBot]);
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
  const latestSignalDecision = useMemo<DecisionSummary | null>(() => {
    const sig = state.latestSignal;
    if (!sig) return null;

    const action = sig.action ?? "";
    const isHold = action.toUpperCase().startsWith("HOLD");
    const rawReason = parseActionReason(action);
    const reason = rawReason ? rawReason.replace(/_/g, " ") : null;
    const reasonKey = normalizeHoldReason(rawReason);
    const directionHold = reasonKey ? DIRECTION_HOLD_REASONS.has(reasonKey) : false;

    const openThrRaw = sig.openThreshold ?? sig.threshold ?? 0;
    const openThr = isFiniteNumber(openThrRaw) ? openThrRaw : 0;

    const currentPrice = sig.currentPrice;
    const edgeFromPred = (pred: number | null | undefined): number | null => {
      if (!isFiniteNumber(pred) || !isFiniteNumber(currentPrice) || currentPrice === 0) return null;
      const edge = Math.abs(pred / currentPrice - 1);
      return Number.isFinite(edge) ? edge : null;
    };

    const kalEdge = edgeFromPred(sig.kalmanNext);
    const lstmEdge = edgeFromPred(sig.lstmNext);
    const blendWeight = clamp(form.blendWeight, 0, 1);
    const blendNext =
      isFiniteNumber(sig.kalmanNext) && isFiniteNumber(sig.lstmNext)
        ? blendWeight * sig.kalmanNext + (1 - blendWeight) * sig.lstmNext
        : null;
    const blendEdge = edgeFromPred(blendNext);

    let edgeForMethod: number | null = null;
    let edgeSource = "";
    switch (sig.method) {
      case "11":
        edgeForMethod = kalEdge != null && lstmEdge != null ? Math.min(kalEdge, lstmEdge) : null;
        edgeSource = "min(kalman,lstm)";
        break;
      case "10":
        edgeForMethod = kalEdge;
        edgeSource = "kalman";
        break;
      case "01":
        edgeForMethod = lstmEdge;
        edgeSource = "lstm";
        break;
      case "blend":
        edgeForMethod = blendEdge;
        edgeSource = "blend";
        break;
      case "router":
        edgeForMethod = null;
        edgeSource = `router (kal ${kalEdge != null ? fmtPct(kalEdge, 3) : "n/a"}, lstm ${
          lstmEdge != null ? fmtPct(lstmEdge, 3) : "n/a"
        }, blend ${blendEdge != null ? fmtPct(blendEdge, 3) : "n/a"})`;
        break;
      default:
        edgeForMethod = null;
        edgeSource = "unknown";
        break;
    }

    const volEstimate = isFiniteNumber(sig.volatility) ? sig.volatility : null;
    const periodsPerYear = inferPeriodsPerYear(form.platform, form.interval);
    const volPerBar = volEstimate != null && periodsPerYear ? volEstimate / Math.sqrt(periodsPerYear) : null;

    const checks: DecisionCheck[] = [];

    const kalDir = sig.kalmanDirection ?? null;
    const lstmDir = sig.lstmDirection ?? null;
    const chosenDir = sig.chosenDirection ?? null;
    const directionDetail = `Kalman ${formatSignalDirection(kalDir)} / LSTM ${formatSignalDirection(lstmDir)} / chosen ${formatSignalDirection(chosenDir)}`;
    const directionStatus: DecisionCheckStatus = chosenDir ? "ok" : kalDir || lstmDir ? "warn" : "bad";
    checks.push({ id: "direction", label: "Direction vote", status: directionStatus, detail: directionDetail });

    if (sig.method === "11") {
      let status: DecisionCheckStatus = "warn";
      let detail = "One or both models are neutral.";
      if (kalDir && lstmDir) {
        status = kalDir === lstmDir ? "ok" : "bad";
        detail = kalDir === lstmDir ? `Agree on ${kalDir}.` : `Disagree (${kalDir} vs ${lstmDir}).`;
      }
      checks.push({ id: "agreement", label: "Agreement gate", status, detail });
    } else {
      checks.push({
        id: "agreement",
        label: "Agreement gate",
        status: "skip",
        detail: `Not required for ${methodLabel(sig.method)}.`,
      });
    }

    const kalmanEnabled = sig.method !== "01";
    if (!kalmanEnabled) {
      checks.push({
        id: "kalman-z",
        label: "Kalman z-score",
        status: "skip",
        detail: "Kalman disabled for this method.",
      });
    } else if (form.kalmanZMin <= 0) {
      checks.push({
        id: "kalman-z",
        label: "Kalman z-score",
        status: "skip",
        detail: "Min z gate disabled.",
      });
    } else if (!isFiniteNumber(sig.kalmanZ)) {
      checks.push({
        id: "kalman-z",
        label: "Kalman z-score",
        status: "warn",
        detail: "Kalman z not available.",
      });
    } else {
      const status: DecisionCheckStatus = sig.kalmanZ >= form.kalmanZMin ? "ok" : "bad";
      const detail = `z ${fmtNum(sig.kalmanZ, 2)} >= min ${fmtNum(form.kalmanZMin, 2)}`;
      checks.push({ id: "kalman-z", label: "Kalman z-score", status, detail });
    }

    if (!kalmanEnabled) {
      checks.push({
        id: "hmm-high-vol",
        label: "High-vol regime",
        status: "skip",
        detail: "Kalman disabled for this method.",
      });
    } else if (form.maxHighVolProb <= 0) {
      checks.push({
        id: "hmm-high-vol",
        label: "High-vol regime",
        status: "skip",
        detail: "High-vol gate disabled.",
      });
    } else if (!sig.regimes || !isFiniteNumber(sig.regimes.highVol)) {
      checks.push({
        id: "hmm-high-vol",
        label: "High-vol regime",
        status: "warn",
        detail: "Regime estimate missing.",
      });
    } else {
      const status: DecisionCheckStatus = sig.regimes.highVol <= form.maxHighVolProb ? "ok" : "bad";
      const detail = `high vol ${fmtPct(sig.regimes.highVol, 1)} <= max ${fmtPct(form.maxHighVolProb, 1)}`;
      checks.push({ id: "hmm-high-vol", label: "High-vol regime", status, detail });
    }

    if (!kalmanEnabled) {
      checks.push({
        id: "conformal-width",
        label: "Conformal width",
        status: "skip",
        detail: "Kalman disabled for this method.",
      });
    } else if (form.maxConformalWidth <= 0) {
      checks.push({
        id: "conformal-width",
        label: "Conformal width",
        status: "skip",
        detail: "Width gate disabled.",
      });
    } else if (!sig.conformalInterval || !isFiniteNumber(sig.conformalInterval.width)) {
      checks.push({
        id: "conformal-width",
        label: "Conformal width",
        status: "warn",
        detail: "Conformal interval missing.",
      });
    } else {
      const status: DecisionCheckStatus = sig.conformalInterval.width <= form.maxConformalWidth ? "ok" : "bad";
      const detail = `width ${fmtPct(sig.conformalInterval.width, 2)} <= max ${fmtPct(form.maxConformalWidth, 2)}`;
      checks.push({ id: "conformal-width", label: "Conformal width", status, detail });
    }

    if (!kalmanEnabled) {
      checks.push({
        id: "quantile-width",
        label: "Quantile width",
        status: "skip",
        detail: "Kalman disabled for this method.",
      });
    } else if (form.maxQuantileWidth <= 0) {
      checks.push({
        id: "quantile-width",
        label: "Quantile width",
        status: "skip",
        detail: "Width gate disabled.",
      });
    } else if (!sig.quantiles || !isFiniteNumber(sig.quantiles.width)) {
      checks.push({
        id: "quantile-width",
        label: "Quantile width",
        status: "warn",
        detail: "Quantile estimate missing.",
      });
    } else {
      const status: DecisionCheckStatus = sig.quantiles.width <= form.maxQuantileWidth ? "ok" : "bad";
      const detail = `width ${fmtPct(sig.quantiles.width, 2)} <= max ${fmtPct(form.maxQuantileWidth, 2)}`;
      checks.push({ id: "quantile-width", label: "Quantile width", status, detail });
    }

    if (!kalmanEnabled) {
      checks.push({
        id: "conformal-confirm",
        label: "Conformal confirm",
        status: "skip",
        detail: "Kalman disabled for this method.",
      });
    } else if (!form.confirmConformal) {
      checks.push({
        id: "conformal-confirm",
        label: "Conformal confirm",
        status: "skip",
        detail: "Confirmation disabled.",
      });
    } else if (!sig.conformalInterval) {
      checks.push({
        id: "conformal-confirm",
        label: "Conformal confirm",
        status: "warn",
        detail: "Conformal interval missing.",
      });
    } else if (!kalDir) {
      checks.push({
        id: "conformal-confirm",
        label: "Conformal confirm",
        status: "warn",
        detail: "No Kalman direction.",
      });
    } else {
      const status: DecisionCheckStatus =
        (kalDir === "UP" && sig.conformalInterval.lo > openThr) ||
        (kalDir === "DOWN" && sig.conformalInterval.hi < -openThr)
          ? "ok"
          : "bad";
      const detail =
        kalDir === "UP"
          ? `lo ${fmtPct(sig.conformalInterval.lo, 3)} > thr ${fmtPct(openThr, 3)}`
          : `hi ${fmtPct(sig.conformalInterval.hi, 3)} < ${fmtPct(-openThr, 3)}`;
      checks.push({ id: "conformal-confirm", label: "Conformal confirm", status, detail });
    }

    if (!kalmanEnabled) {
      checks.push({
        id: "quantile-confirm",
        label: "Quantile confirm",
        status: "skip",
        detail: "Kalman disabled for this method.",
      });
    } else if (!form.confirmQuantiles) {
      checks.push({
        id: "quantile-confirm",
        label: "Quantile confirm",
        status: "skip",
        detail: "Confirmation disabled.",
      });
    } else if (!sig.quantiles) {
      checks.push({
        id: "quantile-confirm",
        label: "Quantile confirm",
        status: "warn",
        detail: "Quantile estimate missing.",
      });
    } else if (!kalDir) {
      checks.push({
        id: "quantile-confirm",
        label: "Quantile confirm",
        status: "warn",
        detail: "No Kalman direction.",
      });
    } else {
      const status: DecisionCheckStatus =
        (kalDir === "UP" && sig.quantiles.q10 > openThr) ||
        (kalDir === "DOWN" && sig.quantiles.q90 < -openThr)
          ? "ok"
          : "bad";
      const detail =
        kalDir === "UP"
          ? `q10 ${fmtPct(sig.quantiles.q10, 3)} > thr ${fmtPct(openThr, 3)}`
          : `q90 ${fmtPct(sig.quantiles.q90, 3)} < ${fmtPct(-openThr, 3)}`;
      checks.push({ id: "quantile-confirm", label: "Quantile confirm", status, detail });
    }

    if (form.minSignalToNoise <= 0) {
      checks.push({
        id: "signal-to-noise",
        label: "Signal-to-noise",
        status: "skip",
        detail: "SNR filter disabled.",
      });
    } else {
      const edgeLabel = edgeForMethod != null ? fmtPct(edgeForMethod, 3) : "—";
      const volLabel = volPerBar != null ? fmtPct(volPerBar, 3) : "—";
      const snr = edgeForMethod != null && volPerBar != null ? edgeForMethod / volPerBar : null;
      const snrLabel = snr != null && Number.isFinite(snr) ? fmtNum(snr, 2) : "—";
      const detail = `edge ${edgeLabel} (${edgeSource}) / vol ${volLabel} = ${snrLabel} (min ${fmtNum(form.minSignalToNoise, 2)})`;
      let status: DecisionCheckStatus = "bad";
      if (sig.method === "router" && edgeForMethod == null) {
        status = "warn";
      } else if (edgeForMethod != null && volPerBar != null) {
        status = snr != null && Number.isFinite(snr) && snr >= form.minSignalToNoise ? "ok" : "bad";
      }
      checks.push({ id: "signal-to-noise", label: "Signal-to-noise", status, detail });
    }

    if (form.maxVolatility <= 0) {
      checks.push({
        id: "max-volatility",
        label: "Max volatility",
        status: "skip",
        detail: "Max volatility gate disabled.",
      });
    } else if (volEstimate == null) {
      checks.push({
        id: "max-volatility",
        label: "Max volatility",
        status: "bad",
        detail: `No volatility estimate (max ${fmtPct(form.maxVolatility, 2)}).`,
      });
    } else {
      const status: DecisionCheckStatus = volEstimate <= form.maxVolatility ? "ok" : "bad";
      const detail = `vol ${fmtPct(volEstimate, 2)} <= max ${fmtPct(form.maxVolatility, 2)}`;
      checks.push({ id: "max-volatility", label: "Max volatility", status, detail });
    }

    if (form.volTarget <= 0) {
      checks.push({
        id: "vol-target",
        label: "Vol target warmup",
        status: "skip",
        detail: "Vol target disabled.",
      });
    } else if (volEstimate == null) {
      checks.push({
        id: "vol-target",
        label: "Vol target warmup",
        status: "bad",
        detail: `Waiting for volatility estimate (target ${fmtPct(form.volTarget, 2)}).`,
      });
    } else {
      const detail = `vol ${fmtPct(volEstimate, 2)} target ${fmtPct(form.volTarget, 2)}`;
      checks.push({ id: "vol-target", label: "Vol target warmup", status: "ok", detail });
    }

    if (!isFiniteNumber(sig.positionSize)) {
      checks.push({
        id: "position-size",
        label: "Position size",
        status: "warn",
        detail: "Position size missing.",
      });
    } else {
      const size = sig.positionSize;
      const sizeParts = [`size ${fmtPct(size, 1)}`];
      if (isFiniteNumber(sig.confidence)) sizeParts.push(`confidence ${fmtPct(sig.confidence, 1)}`);
      if (form.confidenceSizing && form.minPositionSize > 0) {
        sizeParts.push(`min ${fmtPct(form.minPositionSize, 1)}`);
      } else if (!form.confidenceSizing) {
        sizeParts.push("confidence sizing off");
      }
      if (form.maxPositionSize > 0) sizeParts.push(`max ${fmtPct(form.maxPositionSize, 1)}`);
      const detail = sizeParts.join(" / ");
      let status: DecisionCheckStatus = size > 0 ? "ok" : "bad";
      if (form.confidenceSizing && form.minPositionSize > 0 && size < form.minPositionSize) {
        status = "bad";
      }
      if (directionHold && status === "ok") {
        status = "warn";
      }
      checks.push({ id: "position-size", label: "Position size", status, detail });
    }

    return { isHold, reason, checks };
  }, [
    form.blendWeight,
    form.confirmConformal,
    form.confirmQuantiles,
    form.confidenceSizing,
    form.interval,
    form.kalmanZMin,
    form.maxConformalWidth,
    form.maxHighVolProb,
    form.maxPositionSize,
    form.maxQuantileWidth,
    form.maxVolatility,
    form.minPositionSize,
    form.minSignalToNoise,
    form.platform,
    form.volTarget,
    state.latestSignal,
  ]);
  const configPageList = useMemo(
    () => CONFIG_PAGE_IDS.map((pageId) => ({ id: pageId, label: CONFIG_PAGE_LABELS[pageId] })),
    [],
  );
  const configPanelOrderIndex = useMemo(() => {
    const index = {} as Record<ConfigPanelId, number>;
    configPanelOrder.forEach((panelId, idx) => {
      index[panelId] = idx;
    });
    return index;
  }, [configPanelOrder]);
  const activeConfigPanel = CONFIG_PAGE_PANEL_MAP[configPage];
  const configPanelStyle = useCallback(
    (panelId: ConfigPanelId): React.CSSProperties =>
      ({
        "--panel-height": CONFIG_PANEL_HEIGHTS[panelId],
      }) as React.CSSProperties,
    [],
  );
  const configPanelDragState = useMemo(
    () => ({ draggingId: draggingConfigPanel, overId: dragOverConfigPanel }),
    [draggingConfigPanel, dragOverConfigPanel],
  );
  const isPanelMaximized = useCallback((panelId: string) => maximizedPanelId === panelId, [maximizedPanelId]);
  const configPanelHandlers = {
    dragState: configPanelDragState,
    onDragStart: handleConfigPanelDragStart,
    onDragOver: handleConfigPanelDragOver,
    onDrop: handleConfigPanelDrop,
    onDragEnd: handleConfigPanelDragEnd,
  };
  const apiStatusBadgeClass =
    apiOk === "ok" ? "badge badgeOk" : apiOk === "auth" ? "badge badgeWarn" : apiOk === "down" ? "badge badgeBad" : "badge";
  const liveModeBadgeClass = form.binanceLive ? "badge badgeWarn" : "badge";
  const tradeArmBadgeClass = form.tradeArmed ? "badge badgeWarn" : "badge";
  const botStatusBadge = bot.error
    ? { label: "Bot error", className: "badge badgeBad" }
    : botRunningEntries.length > 0
      ? { label: "Bot running", className: "badge badgeOk" }
      : botActiveSymbols.length > 0
        ? { label: "Bot starting", className: "badge badgeWarn" }
        : { label: "Bot stopped", className: "badge" };
  const latestSignalSummary = state.latestSignal
    ? {
        action: state.latestSignal.action,
        direction: state.latestSignal.chosenDirection ?? "NEUTRAL",
        method: methodLabel(state.latestSignal.method),
      }
    : null;
  const backtestTradeAnalysis = useMemo(
    () => (state.backtest ? buildBacktestTradePnlAnalysis(state.backtest) : null),
    [state.backtest],
  );
  const backtestSummary = state.backtest
    ? {
        equity: fmtRatio(state.backtest.metrics.finalEquity, 4),
        sharpe: fmtNum(state.backtest.metrics.sharpe, 2),
        trades: state.backtest.metrics.tradeCount,
      }
    : null;
  const tradeOrder = state.trade?.order ?? null;
  const combosOpen = isPanelOpen("panel-combos", true);
  const configOpen = isPanelOpen("panel-config", true);
  const dockLayoutClass = `dockLayout dockLayoutConfigPage${combosOpen ? "" : " dockLayoutCompactBottom"}${configOpen ? "" : " dockLayoutCompactTop"}`;
  const configDockProps: ConfigDockProps = {
    configOpen,
    handlePanelToggle,
    isPanelOpen,
    isPanelMaximized,
    togglePanelMaximize,
    requestIssues,
    primaryIssue,
    handlePrimaryIssueFix,
    extraIssueCount,
    requestDisabled,
    requestDisabledReason,
    run,
    state,
    commonParams,
    setForm,
    cancelActiveRequest,
    requestIssueDetails,
    scrollToSection,
    rateLimitReason,
    activeAsyncJob,
    activeAsyncTickMs,
    activeConfigPanel,
    configPanelOrderIndex,
    configPanelStyle,
    configPanelHandlers,
    configPage,
    apiBase,
    apiToken,
    apiBaseAbsolute,
    apiBaseError,
    apiHealthUrl,
    apiBaseCorsHint,
    healthInfo,
    cacheUi,
    apiOk,
    showLocalStartHelp,
    refreshCacheStats,
    clearCacheUi,
    recheckHealth,
    showToast,
    platformKeyLabel,
    platformKeyMode,
    revealSecrets,
    setRevealSecrets,
    coinbaseApiKey,
    coinbaseApiSecret,
    coinbaseApiPassphrase,
    setCoinbaseApiKey,
    setCoinbaseApiSecret,
    setCoinbaseApiPassphrase,
    binanceApiKey,
    binanceApiSecret,
    setBinanceApiKey,
    setBinanceApiSecret,
    platformKeyHasValues,
    platformKeyHint,
    keysLoading: keys.loading,
    keysCheckedAtMs,
    refreshKeys,
    persistSecrets,
    setPersistSecrets,
    profileSelected,
    setProfileSelected,
    profileNames,
    profileName,
    setProfileName,
    saveProfile,
    requestLoadProfile,
    deleteProfile,
    pendingProfileLoad,
    clearManualOverrides,
    setPendingProfileLoad,
    form,
    setPendingMarket,
    customSymbolByPlatform,
    setCustomSymbolByPlatform,
    missingSymbol,
    symbolFormatError,
    symbolSelectValue,
    platformSymbols,
    symbolIsCustom,
    platformLabel,
    platform,
    isCoinbasePlatform,
    isBinancePlatform,
    pendingMarket,
    setConfirmLive,
    missingInterval,
    platformIntervals,
    barsRangeLabel,
    barsExceedsApi,
    barsMaxLabel,
    apiComputeLimits,
    barsAutoHint,
    barsRangeHint,
    lookbackState,
    splitPreview,
    markManualOverrides,
    methodOverride,
    thresholdOverrideKeys,
    estimatedCosts,
    minEdgeEffective,
    longShortBotDisabled,
    apiLstmEnabled,
    epochsExceedsApi,
    hiddenSizeExceedsApi,
    optimizerRunForm,
    updateOptimizerRunForm,
    optimizerRunValidationError,
    optimizerRunExtras,
    optimizerRunUi,
    runOptimizer,
    cancelOptimizerRun,
    syncOptimizerRunForm,
    applyEquityPreset,
    resetOptimizerRunForm,
    optimizerRunRecordJson,
    botStartBlocked,
    botStartBlockedReason,
    bot,
    botStarting,
    botAnyRunning,
    startLiveBot,
    stopLiveBot,
    botSymbolsActive,
    botSelectedSymbol,
    apiBlockedReason,
    refreshBot,
    botSymbolsFormatError,
    confirmLive,
    confirmArm,
    setConfirmArm,
    orderSizing,
    orderQuoteFractionError,
    idempotencyKeyError,
    tradeDisabledReason,
    tradeDisabledDetail,
  };

  const optimizerCombosProps = {
    apiOk,
    autoAppliedCombo,
    autoAppliedAge,
    autoApplyTopCombo,
    setAutoApplyTopCombo,
    manualOverrideLabels,
    clearManualOverrides,
    topCombosMeta,
    topCombos,
    topCombosFiltered,
    topCombosAll,
    deferredTopCombos,
    topCombosLoading,
    topCombosError,
    selectedComboId,
    selectedCombo,
    topComboDisplay,
    selectedComboStartLabel,
    comboStartBlocked,
    comboStartBlockedReason,
    comboStartPending,
    comboMinEquity,
    comboSymbolFilters,
    comboMarketFilter,
    comboIntervalFilter,
    comboMethodFilter,
    comboOrder,
    comboHasFilters,
    comboFilterOptions,
    comboMinEquityInput,
    comboSymbolFilterInput,
    topCombosDisplayCount,
    topCombosDisplayMax,
    setTopCombosDisplayCount,
    setComboOrder,
    setComboMinEquityInput,
    setComboSymbolFilterInput,
    setComboMarketFilter,
    setComboIntervalFilter,
    setComboMethodFilter,
    refreshTopCombos,
    handleComboApply,
    handleComboPreview,
    handleComboStart,
    optimizerRunForm,
    setOptimizerRunForm,
    setOptimizerRunDirty,
    optimizerRunExtras,
    optimizerRunUi,
    optimizerRunValidationError,
    optimizerRunRecordJson,
    runOptimizer,
    cancelOptimizerRun,
    syncOptimizerRunSymbolInterval,
    applyEquityPreset,
    resetOptimizerRunForm,
    comboExportReady,
    comboExportFilename,
    copyComboExport,
    downloadComboExport,
    comboImportText: comboImportUi.text,
    setComboImportText: updateComboImportText,
    parseComboImportText: () => parseComboImportText(),
    comboImportParseError: comboImportUi.parseError,
    comboImportSummary: comboImportUi.summary,
    comboImporting: comboImportUi.importing,
    comboImportError: comboImportUi.importError,
    comboImportResult: comboImportUi.importResult,
    comboImportAtMs: comboImportUi.importedAtMs,
    comboImportReady,
    comboImportDisabledReason,
    comboImportStampNow: comboImportUi.stampNow,
    setComboImportStampNow,
    handleComboImportFile,
    importCombos,
    clearComboImport,
  } satisfies OptimizerCombosPanelProps;

  const shortCommitHash = (hash?: string | null, size = 8): string => {
    if (!hash) return "—";
    const trimmed = hash.trim();
    if (!trimmed) return "—";
    return trimmed.length > size ? `${trimmed.slice(0, size)}…` : trimmed;
  };

  const shortComboUuid = (uuid?: string | null, size = 6): string => {
    if (!uuid) return "—";
    const trimmed = uuid.trim();
    if (!trimmed) return "—";
    return trimmed.length > size ? `${trimmed.slice(0, size)}…` : trimmed;
  };

  return (
    <div className="container">
      {toast ? (
        <div className="toastFixed" role="status" aria-live="polite" aria-atomic="true">
          {toast}
        </div>
      ) : null}
      <div className={dockLayoutClass}>
        <div className="dockTop">
          <details
            className={`card cardCollapsible headerCard${isPanelMaximized("panel-header") ? " cardMaximized" : ""}`}
            open={isPanelOpen("panel-header", true)}
            onToggle={handlePanelToggle("panel-header")}
            data-panel="panel-header"
          >
            <summary className="cardSummary headerSummary">
              <div className="brand">
                <div className="logo" aria-hidden="true" />
                <div className="title">
                  <h1>Trader UI</h1>
                  <p>Configure, backtest, optimize, and trade via the local REST API.</p>
                </div>
              </div>
              <div className="headerActions">
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
                <div className="cardControls">
                  <button
                    className="cardControl"
                    type="button"
                    aria-pressed={isPanelMaximized("panel-header")}
                    aria-label={isPanelMaximized("panel-header") ? "Restore panel size" : "Maximize panel"}
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      togglePanelMaximize("panel-header");
                    }}
                  >
                    {isPanelMaximized("panel-header") ? "Restore" : "Maximize"}
                  </button>
                  <span className="cardToggle" aria-hidden="true">
                    <span className="cardToggleLabel" data-open="Collapse" data-closed="Expand" />
                    <span className="cardToggleIcon" />
                  </span>
                </div>
              </div>
              <nav className="menuBar menuBarHeader" aria-label="Configuration pages">
                <span className="jumpLabel">Menu</span>
                {configPageList.map((page) => (
                  <button
                    key={page.id}
                    className={`menuItem${configPage === page.id ? " menuItemActive" : ""}`}
                    type="button"
                    aria-current={configPage === page.id ? "page" : undefined}
                    onClick={() => selectConfigPage(page.id)}
                  >
                    {page.label}
                  </button>
                ))}
              </nav>
            </summary>
          </details>
          <ConfigDock {...configDockProps} />
      </div>
      <main className="dockMain">
        <section className="resultGrid">
          {state.error ? (
            <CollapsibleCard
              panelId="panel-error"
              open={isPanelOpen("panel-error", true)}
              onToggle={handlePanelToggle("panel-error")}
              maximized={isPanelMaximized("panel-error")}
              onToggleMaximize={() => togglePanelMaximize("panel-error")}
              title="Error"
              subtitle="Fix the request or backend and try again."
              containerRef={errorRef}
            >
              <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)" }}>
                {state.error}
              </pre>
              {errorFix ? (
                <div className="issueItem" style={{ marginTop: 10 }}>
                  <span>{errorFix.label}</span>
                  <button className="btnSmall" type="button" onClick={applyErrorFix}>
                    Apply fix
                  </button>
                </div>
              ) : null}
            </CollapsibleCard>
          ) : null}

          <CollapsibleCard
            panelId="panel-overview"
            open={isPanelOpen("panel-overview", true)}
            onToggle={handlePanelToggle("panel-overview")}
            maximized={isPanelMaximized("panel-overview")}
            onToggleMaximize={() => togglePanelMaximize("panel-overview")}
            title="Overview"
            subtitle="At-a-glance status for connection, execution mode, and latest outputs."
          >
            <div className="summaryGrid">
                <div className="summaryItem">
                  <div className="summaryLabel">Connection</div>
                  <div className="summaryValue">
                    <span className={statusDotClass} aria-hidden="true" />
                    <span className={apiStatusBadgeClass}>{statusLabel}</span>
                    <span className="summaryMeta" title={apiBaseAbsolute || apiBase}>
                      {apiBaseAbsolute || apiBase}
                    </span>
                  </div>
                </div>
                <div className="summaryItem">
                  <div className="summaryLabel">Market</div>
                  <div className="summaryValue">
                    <span className="badge">{platformLabel}</span>
                    <span className="badge">{normalizedSymbol || "—"}</span>
                    <span className="badge">{form.interval || "—"}</span>
                    <span className="badge">{marketLabel(form.market)}</span>
                  </div>
                </div>
                <div className="summaryItem">
                  <div className="summaryLabel">Execution</div>
                  <div className="summaryValue">
                    <span className={liveModeBadgeClass}>{form.binanceLive ? "Live orders" : "Test orders"}</span>
                    <span className={tradeArmBadgeClass}>{form.tradeArmed ? "Trading armed" : "Trading locked"}</span>
                    <span className={botStatusBadge.className}>{botStatusBadge.label}</span>
                    <span className="badge">{botActiveSymbols.length} active</span>
                    {botDisplay?.symbol ? <span className="badge">{botDisplay.symbol}</span> : null}
                    {botDisplay?.halted ? <span className="badge badgeWarn">Halted</span> : null}
                    {bot.error ? (
                      <span className="summaryMeta" title={bot.error}>
                        {bot.error}
                      </span>
                    ) : null}
                  </div>
                </div>
                <div className="summaryItem">
                  <div className="summaryLabel">Latest signal</div>
                  <div className="summaryValue">
                    {latestSignalSummary ? (
                      <>
                        <span className={actionBadgeClass(latestSignalSummary.action)}>{latestSignalSummary.action}</span>
                        <span className="badge">{latestSignalSummary.direction}</span>
                        <span className="badge">{latestSignalSummary.method}</span>
                      </>
                    ) : (
                      <span className="summaryEmpty">No signal yet</span>
                    )}
                  </div>
                </div>
                <div className="summaryItem">
                  <div className="summaryLabel">Backtest</div>
                  <div className="summaryValue">
                    {backtestSummary ? (
                      <>
                        <span className="badge">Equity {backtestSummary.equity}</span>
                        <span className="badge">Sharpe {backtestSummary.sharpe}</span>
                        <span className="badge">{backtestSummary.trades} trades</span>
                      </>
                    ) : (
                      <span className="summaryEmpty">No backtest yet</span>
                    )}
                  </div>
                </div>
                <div className="summaryItem">
                  <div className="summaryLabel">Trade</div>
                  <div className="summaryValue">
                    {tradeOrder ? (
                      <>
                        <span className={tradeOrder.sent ? "badge badgeOk" : "badge badgeBad"}>
                          {tradeOrder.sent ? "Sent" : "Rejected"}
                        </span>
                        {tradeOrder.side ? <span className="badge">{tradeOrder.side}</span> : null}
                        {tradeOrder.symbol ? <span className="badge">{tradeOrder.symbol}</span> : null}
                        {tradeOrder.status ? (
                          <span className="summaryMeta" title={tradeOrder.status}>
                            {tradeOrder.status}
                          </span>
                        ) : tradeOrder.message ? (
                          <span className="summaryMeta" title={tradeOrder.message}>
                            {tradeOrder.message}
                          </span>
                        ) : null}
                      </>
                    ) : (
                      <span className="summaryEmpty">No trade yet</span>
                    )}
                  </div>
                </div>
            </div>
          </CollapsibleCard>

          <CollapsibleCard
            panelId="panel-live-bot"
            open={isPanelOpen("panel-live-bot", true)}
            onToggle={handlePanelToggle("panel-live-bot")}
            maximized={isPanelMaximized("panel-live-bot")}
            onToggleMaximize={() => togglePanelMaximize("panel-live-bot")}
            title="Live bot"
            subtitle="Non-stop loop (server-side): fetches new bars, updates the model each bar, and records each buy/sell operation."
            className="chartCard"
          >
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
                      {botDisplayStale ? <span className="badge badgeWarn">STALE</span> : null}
	                  </div>
                    {botHasSnapshot ? (
                      <div className="hint" style={{ marginBottom: 10 }}>
                        Snapshot {botSnapshotAtMs ? `from ${fmtTimeMs(botSnapshotAtMs)}` : "loaded"} (bot not running).
                      </div>
                    ) : null}
                    {botDisplayStale ? (
                      <div className="hint" style={{ marginBottom: 10, color: "rgba(245, 158, 11, 0.9)" }}>
                        Showing last bot status{botDisplayStaleLabel ? ` (${botDisplayStaleLabel} old)` : ""}. Live status unavailable.
                      </div>
                    ) : null}

                    <Suspense fallback={<PanelFallback label="Loading live visuals…" />}>
                      <LiveVisuals
                        prices={botDisplay.prices}
                        signal={botDisplay.latestSignal}
                        position={botLastPosition}
                        risk={botRisk}
                        halted={botDisplay.halted}
                        cooldownLeft={botDisplay.cooldownLeft ?? null}
                        orderErrors={botDisplay.consecutiveOrderErrors ?? null}
                        candleAgeMs={botRealtime?.candleAgeMs ?? null}
                        closeEtaMs={botRealtime?.closeEtaMs ?? null}
                        statusAgeMs={botRealtime?.statusAgeMs ?? null}
                      />
                    </Suspense>

                    <div className="analysisDeck analysisDeckSplit">
                      <div className="analysisDeckMain">
                        <ChartSuspense height={CHART_HEIGHT}>
                          <BacktestChart
                            prices={botDisplay.prices}
                            equityCurve={botDisplay.equityCurve}
                            openTimes={botDisplay.openTimes}
                            kalmanPredNext={botDisplay.kalmanPredNext}
                            positions={botDisplay.positions}
                            trades={botDisplay.trades}
                            operations={botDisplay.operations}
                            backtestStartIndex={botDisplay.startIndex}
                            height={CHART_HEIGHT}
                          />
                        </ChartSuspense>
                      </div>
                      <div className="analysisDeckSide">
                        <div className="chartBlock">
                          <div className="hint">Prediction values vs thresholds (hover for details)</div>
                          <ChartSuspense height={CHART_HEIGHT_SIDE}>
                            <PredictionDiffChart
                              prices={botDisplay.prices}
                              openTimes={botDisplay.openTimes}
                              kalmanPredNext={botDisplay.kalmanPredNext}
                              lstmPredNext={botDisplay.lstmPredNext}
                              startIndex={botDisplay.startIndex}
                              height={CHART_HEIGHT_SIDE}
                              openThreshold={botDisplay.openThreshold ?? botDisplay.threshold}
                              closeThreshold={botDisplay.closeThreshold ?? botDisplay.openThreshold ?? botDisplay.threshold}
                            />
                          </ChartSuspense>
                        </div>
                        <div className="chartBlock">
                          <div className="hint">Telemetry (Binance poll latency + close drift; hover for details)</div>
                          <ChartSuspense height={CHART_HEIGHT_SIDE}>
                            <TelemetryChart points={botRt.telemetry} height={CHART_HEIGHT_SIDE} label="Live bot telemetry chart" />
                          </ChartSuspense>
                        </div>
                      </div>
                    </div>

                  <div style={{ marginTop: 10 }}>
                    <div className="hint" style={{ marginBottom: 8 }}>
                      Bot state timeline (live/offline from ops log).
                    </div>
                    <div className="row" style={{ marginBottom: 8, gridTemplateColumns: "1fr 1fr auto", alignItems: "end" }}>
                      <div className="field">
                        <label className="label" htmlFor="botStatusStart">
                          Chart start
                        </label>
                        <input
                          id="botStatusStart"
                          className="input"
                          type="datetime-local"
                          value={botStatusStartInput}
                          onChange={(e) => setBotStatusStartInput(e.target.value)}
                        />
                      </div>
                      <div className="field">
                        <label className="label" htmlFor="botStatusEnd">
                          Chart end
                        </label>
                        <input
                          id="botStatusEnd"
                          className="input"
                          type="datetime-local"
                          value={botStatusEndInput}
                          onChange={(e) => setBotStatusEndInput(e.target.value)}
                        />
                      </div>
                      <button
                        className="btn"
                        type="button"
                        disabled={botStatusOps.loading || apiOk !== "ok"}
                        onClick={() => void fetchBotStatusOps()}
                      >
                        {botStatusOps.loading ? "Loading..." : "Refresh"}
                      </button>
                    </div>
                    {botStatusRange.error ? (
                      <div className="hint" style={{ marginBottom: 8, color: "rgba(239, 68, 68, 0.9)" }}>
                        {botStatusRange.error}
                      </div>
                    ) : null}
                    {!botStatusOps.enabled ? (
                      <div className="hint">{botStatusOps.hint ?? "Enable TRADER_DB_URL to track bot status history."}</div>
                    ) : botStatusRange.startMs !== null && botStatusRange.endMs !== null && !botStatusRange.error ? (
                      <ChartSuspense height={CHART_HEIGHT_TIMELINE} label="Loading timeline…">
                        <BotStateChart
                          points={botStatusPoints}
                          startMs={botStatusRange.startMs}
                          endMs={botStatusRange.endMs}
                          height={CHART_HEIGHT_TIMELINE}
                        />
                      </ChartSuspense>
                    ) : (
                      <div className="chart" style={{ height: CHART_HEIGHT_TIMELINE }}>
                        <div className="chartEmpty">Select a valid time range</div>
                      </div>
                    )}
                    {botStatusRangeWarning ? (
                      <div className="hint" style={{ marginTop: 6 }}>
                        {botStatusRangeWarning}
                      </div>
                    ) : null}
                    {botStatusOps.error ? (
                      <div className="hint" style={{ marginTop: 6, color: "rgba(239, 68, 68, 0.9)" }}>
                        {botStatusOps.error}
                      </div>
                    ) : null}
                    <div className="pillRow" style={{ marginTop: 8 }}>
                      <span className="badge badgeStrong badgeLong">LIVE</span>
                      <span className="badge badgeHold">OFFLINE</span>
                      {botStatusOps.lastFetchedAtMs ? <span className="badge">Synced {fmtTimeMs(botStatusOps.lastFetchedAtMs)}</span> : null}
                    </div>
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
		                          {typeof botDisplay.settings.protectionOrders === "boolean"
		                            ? ` • protection ${botDisplay.settings.protectionOrders ? "ON" : "OFF"}`
		                            : ""}
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
                          if (!botDisplayKey) return;
                          setBotRtByKey((prev) => {
                            const cur = prev[botDisplayKey];
                            if (!cur || cur.feed.length === 0) return prev;
                            return { ...prev, [botDisplayKey]: { ...cur, feed: [] } };
                          });
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
                            {deferredBotOrders.map((e) => {
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
                    ? botStartingHint ?? "Bot is starting… (initializing model). Use “Refresh” to check status."
                    : botStartBlockedReason
                      ? `Bot is stopped. Start live bot is disabled: ${botStartBlockedReason}`
                      : "Bot is stopped. Use “Start live bot” on the left."}
                </div>
              )}
          </CollapsibleCard>

          {botRunningCharts.map((entry) => {
            const st = entry.status;
            const botStatePoints = botStatusPointsBySymbol.get(normalizeSymbolKey(st.symbol)) ?? [];
            const botStateRangeOk = botStatusRange.startMs !== null && botStatusRange.endMs !== null && !botStatusRange.error;
            const panelId = `panel-bot-${botStatusKey(st).replace(/[^a-z0-9-]/gi, "-").toLowerCase()}`;
            const subtitle = `${marketLabel(st.market)} / ${st.interval} / ${methodLabel(st.method)}`;
            return (
              <CollapsibleCard
                key={panelId}
                panelId={panelId}
                open={isPanelOpen(panelId, true)}
                onToggle={handlePanelToggle(panelId)}
                maximized={isPanelMaximized(panelId)}
                onToggleMaximize={() => togglePanelMaximize(panelId)}
                title={`Bot ${st.symbol}`}
                subtitle={subtitle}
                className="chartCard"
              >
                <div className="pillRow" style={{ marginBottom: 10 }}>
                  <span className="badge">{st.symbol}</span>
                  <span className="badge">{st.interval}</span>
                  <span className="badge">{marketLabel(st.market)}</span>
                  <span className="badge">{methodLabel(st.method)}</span>
                  <span className="badge">open {fmtPct(st.openThreshold ?? st.threshold, 3)}</span>
                  <span className="badge">close {fmtPct(st.closeThreshold ?? st.openThreshold ?? st.threshold, 3)}</span>
                  <span className="badge">{st.halted ? "HALTED" : "ACTIVE"}</span>
                  <span className="badge">{st.error ? "Error" : "OK"}</span>
                </div>
                <ChartSuspense height={CHART_HEIGHT}>
                  <BacktestChart
                    prices={st.prices}
                    equityCurve={st.equityCurve}
                    openTimes={st.openTimes}
                    kalmanPredNext={st.kalmanPredNext}
                    positions={st.positions}
                    trades={st.trades}
                    operations={st.operations}
                    backtestStartIndex={st.startIndex}
                    height={CHART_HEIGHT}
                  />
                </ChartSuspense>
                <div style={{ marginTop: 8 }}>
                  <div className="hint" style={{ marginBottom: 6 }}>
                    Bot state timeline
                  </div>
                  {botStateRangeOk ? (
                    <ChartSuspense height={160} label="Loading timeline...">
                      <BotStateChart
                        points={botStatePoints}
                        startMs={botStatusRange.startMs}
                        endMs={botStatusRange.endMs}
                        height={160}
                        label={`Bot state timeline (${st.symbol})`}
                      />
                    </ChartSuspense>
                  ) : (
                    <div className="chart" style={{ height: 160 }}>
                      <div className="chartEmpty">Select a valid time range</div>
                    </div>
                  )}
                </div>
              </CollapsibleCard>
            );
          })}

          <CollapsibleCard
            panelId="panel-binance-trades"
            open={isPanelOpen("panel-binance-trades", true)}
            onToggle={handlePanelToggle("panel-binance-trades")}
            maximized={isPanelMaximized("panel-binance-trades")}
            onToggleMaximize={() => togglePanelMaximize("panel-binance-trades")}
            title="Binance account trades"
            subtitle="Full exchange history from your Binance account (API keys required)."
          >
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
                    <span className="badge">{binanceTradesTotalCount} trades</span>
                    {binanceTradesFilterActive ? <span className="badge">filtered {binanceTradesFilteredCount}</span> : null}
                    <span className="badge">
                      {binanceTradesUi.response.allSymbols
                        ? "all symbols"
                        : binanceTradesUi.response.symbols.length > 0
                          ? binanceTradesUi.response.symbols.join(", ")
                          : "symbol"}
                    </span>
                    <span className="badge">fetched {fmtTimeMs(binanceTradesUi.response.fetchedAtMs)}</span>
                  </div>
                  <div className="pillRow" style={{ marginBottom: 10 }}>
                    <input
                      className="input"
                      style={{ flex: "1 1 220px" }}
                      value={binanceTradesFilterSymbolsInput}
                      onChange={(e) => setBinanceTradesFilterSymbolsInput(e.target.value)}
                      placeholder="Filter symbols (BTCUSDT, ETHUSDT)"
                      spellCheck={false}
                      aria-label="Filter trade symbols"
                    />
                    <select
                      className="select"
                      style={{ width: 140 }}
                      value={binanceTradesFilterSide}
                      onChange={(e) => setBinanceTradesFilterSide(e.target.value as OrderSideFilter)}
                      aria-label="Filter trade side"
                    >
                      <option value="ALL">All sides</option>
                      <option value="BUY">BUY</option>
                      <option value="SELL">SELL</option>
                    </select>
                    <input
                      className="input"
                      style={{ flex: "1 1 200px" }}
                      value={binanceTradesFilterStartInput}
                      onChange={(e) => setBinanceTradesFilterStartInput(e.target.value)}
                      placeholder="Filter start date"
                      aria-label="Filter start date"
                    />
                    <input
                      className="input"
                      style={{ flex: "1 1 200px" }}
                      value={binanceTradesFilterEndInput}
                      onChange={(e) => setBinanceTradesFilterEndInput(e.target.value)}
                      placeholder="Filter end date"
                      aria-label="Filter end date"
                    />
                    <button
                      className="btn"
                      type="button"
                      disabled={!binanceTradesFilterActive && !binanceTradesFilterError}
                      onClick={() => {
                        setBinanceTradesFilterSymbolsInput("");
                        setBinanceTradesFilterSide("ALL");
                        setBinanceTradesFilterStartInput("");
                        setBinanceTradesFilterEndInput("");
                        showToast("Cleared trade filters");
                      }}
                    >
                      Clear filters
                    </button>
                  </div>
                  {binanceTradesFilterError ? (
                    <div className="hint" style={{ marginTop: 6, color: "rgba(239, 68, 68, 0.9)" }}>
                      {binanceTradesFilterError}
                    </div>
                  ) : (
                    <div className="hint" style={{ marginTop: 6 }}>
                      Filter dates accept unix ms timestamps or ISO dates (YYYY-MM-DD or YYYY-MM-DDTHH:MM).
                    </div>
                  )}
                  <div className="pillRow" style={{ marginTop: 8 }}>
                    {binanceTradesFilterActive ? (
                      <span className="badge">
                        showing {binanceTradesFilteredCount} of {binanceTradesTotalCount}
                      </span>
                    ) : null}
                    <span
                      className={pnlBadgeClass(
                        binanceTradesFilteredTotals.totalPnlCount > 0 ? binanceTradesFilteredTotals.totalPnl : null,
                      )}
                    >
                      Total P&amp;L {binanceTradesFilteredPnlLabel}
                    </span>
                    <span className="badge">Total commission {binanceTradesFilteredCommissionLabel}</span>
                  </div>
                  {binanceTradesFilteredCount > 0 ? (
                    binanceTradesAnalysis ? (
                      <details className="details" style={{ marginTop: 12 }}>
                        <summary>Trade P&amp;L analysis</summary>
                        <div style={{ marginTop: 10 }}>
                          {(() => {
                            const stats = binanceTradesAnalysis;
                            const response = binanceTradesUi.response;
                            if (!response) return null;
                            const winRateLabel = stats.winRate != null && Number.isFinite(stats.winRate) ? fmtPct(stats.winRate, 1) : "—";
                            const avgWinLabel = stats.avgWin != null && Number.isFinite(stats.avgWin) ? fmtMoney(stats.avgWin, 4) : "—";
                            const avgLossLabel = stats.avgLoss != null && Number.isFinite(stats.avgLoss) ? fmtMoney(stats.avgLoss, 4) : "—";
                            const avgPnlLabel = stats.avgPnl != null && Number.isFinite(stats.avgPnl) ? fmtMoney(stats.avgPnl, 4) : "—";
                            const maxWinLabel = stats.maxWin != null && Number.isFinite(stats.maxWin) ? fmtMoney(stats.maxWin, 4) : "—";
                            const maxLossLabel = stats.maxLoss != null && Number.isFinite(stats.maxLoss) ? fmtMoney(stats.maxLoss, 4) : "—";
                            const totalWinLabel = Number.isFinite(stats.totalWin) ? fmtMoney(stats.totalWin, 4) : "—";
                            const totalLossLabel = Number.isFinite(stats.totalLoss) ? fmtMoney(stats.totalLoss, 4) : "—";
                            const totalPnlLabel = Number.isFinite(stats.totalPnl) ? fmtMoney(stats.totalPnl, 4) : "—";
                            const totalQtyLabel = Number.isFinite(stats.totalQty) ? fmtNum(stats.totalQty, 8) : "—";
                            const totalQuoteLabel = Number.isFinite(stats.totalQuoteQty) ? fmtMoney(stats.totalQuoteQty, 2) : "—";
                            const profitFactorLabel =
                              stats.profitFactor == null
                                ? "—"
                                : Number.isFinite(stats.profitFactor)
                                  ? fmtNum(stats.profitFactor, 3)
                                  : "∞";
                            const payoffRatioLabel =
                              stats.payoffRatio == null
                                ? "—"
                                : Number.isFinite(stats.payoffRatio)
                                  ? fmtNum(stats.payoffRatio, 3)
                                  : "∞";
                            const commissionLabel =
                              stats.commissionTotals.length > 0
                                ? stats.commissionTotals.map((c) => `${fmtNum(c.total, 6)} ${c.asset}`).join(" • ")
                                : "—";
                            const totalsScope = binanceTradesFilterActive
                              ? "Totals across filtered trades."
                              : response.allSymbols || response.symbols.length > 1
                                ? response.allSymbols
                                  ? "Totals across all symbols."
                                  : `Totals across ${response.symbols.length} symbols.`
                                : null;
                            return (
                              <>
                                <div className="summaryGrid">
                                  <div className="summaryItem">
                                    <div className="labelRow">
                                      <div className="summaryLabel">Outcomes</div>
                                      <InfoPopover label="Outcomes">
                                        <InfoList items={ACCOUNT_TRADE_PNL_TIPS.outcomes} />
                                      </InfoPopover>
                                    </div>
                                    <div className="summaryValue">
                                      <span className="badge badgeStrong badgeLong">{stats.wins} wins</span>
                                      <span className="badge badgeStrong badgeFlat">{stats.losses} losses</span>
                                      <span className="badge">{stats.breakeven} flat</span>
                                      <span className="summaryMeta">Win rate {winRateLabel} • trades {stats.count}</span>
                                    </div>
                                  </div>
                                  <div className="summaryItem">
                                    <div className="labelRow">
                                      <div className="summaryLabel">Avg win / loss</div>
                                      <InfoPopover label="Average win/loss">
                                        <InfoList items={ACCOUNT_TRADE_PNL_TIPS.avgWinLoss} />
                                      </InfoPopover>
                                    </div>
                                    <div className="summaryValue">
                                      <span className={pnlBadgeClass(stats.avgWin)}>{avgWinLabel}</span>
                                      <span className={pnlBadgeClass(stats.avgLoss)}>{avgLossLabel}</span>
                                      <span className="summaryMeta">Payoff ratio {payoffRatioLabel}</span>
                                    </div>
                                  </div>
                                  <div className="summaryItem">
                                    <div className="labelRow">
                                      <div className="summaryLabel">Best / worst trade</div>
                                      <InfoPopover label="Best/worst trade">
                                        <InfoList items={ACCOUNT_TRADE_PNL_TIPS.bestWorst} />
                                      </InfoPopover>
                                    </div>
                                    <div className="summaryValue">
                                      <span className={pnlBadgeClass(stats.maxWin)}>{maxWinLabel}</span>
                                      <span className={pnlBadgeClass(stats.maxLoss)}>{maxLossLabel}</span>
                                      <span className="summaryMeta">Avg P&amp;L {avgPnlLabel}</span>
                                    </div>
                                  </div>
                                  <div className="summaryItem">
                                    <div className="labelRow">
                                      <div className="summaryLabel">Total P&amp;L</div>
                                      <InfoPopover label="Total P&L" align="left">
                                        <InfoList items={ACCOUNT_TRADE_PNL_TIPS.totalPnl} />
                                      </InfoPopover>
                                    </div>
                                    <div className="summaryValue">
                                      <span className={pnlBadgeClass(stats.totalPnl)}>{totalPnlLabel}</span>
                                      <span className={pnlBadgeClass(stats.totalWin)}>{totalWinLabel}</span>
                                      <span className={pnlBadgeClass(stats.totalLoss)}>{totalLossLabel}</span>
                                      <span className="summaryMeta">Profit factor {profitFactorLabel} • Fees {commissionLabel}</span>
                                    </div>
                                  </div>
                                  <div className="summaryItem">
                                    <div className="labelRow">
                                      <div className="summaryLabel">Totals</div>
                                      <InfoPopover label="Totals" align="left">
                                        <InfoList items={ACCOUNT_TRADE_PNL_TIPS.totals} />
                                      </InfoPopover>
                                    </div>
                                    <div className="summaryValue">
                                      <span className="badge">Qty {totalQtyLabel}</span>
                                      <span className="badge">Quote {totalQuoteLabel}</span>
                                      {totalsScope ? <span className="summaryMeta">{totalsScope}</span> : null}
                                    </div>
                                  </div>
                                </div>
                                <div className="chartBlock" style={{ marginTop: 12 }}>
                                  <div className="hint">Top winners</div>
                                  {stats.topWins.length > 0 ? (
                                    <div className="tableWrap" role="region" aria-label="Top winning account trades">
                                      <table className="table">
                                        <thead>
                                          <tr>
                                            <th>Time</th>
                                            <th>Symbol</th>
                                            <th>Side</th>
                                            <th>Price</th>
                                            <th>Qty</th>
                                            <th>Pos</th>
                                            <th>PNL</th>
                                            <th>Commission</th>
                                            <th>Order</th>
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {stats.topWins.map((row) => {
                                            const sideClass =
                                              row.side === "BUY"
                                                ? "badge badgeStrong badgeLong"
                                                : row.side === "SELL"
                                                  ? "badge badgeStrong badgeFlat"
                                                  : "badge";
                                            const qtyTxt = Number.isFinite(row.qty) ? fmtNum(row.qty, 8) : "—";
                                            const pnlTxt = Number.isFinite(row.realizedPnl) ? fmtMoney(row.realizedPnl, 4) : "—";
                                            const commissionTxt =
                                              row.commission != null && Number.isFinite(row.commission)
                                                ? `${fmtNum(row.commission, 8)}${row.commissionAsset ? ` ${row.commissionAsset}` : ""}`
                                                : "—";
                                            return (
                                              <tr key={`binance-win-${row.tradeId}`}>
                                                <td className="tdMono">{fmtTimeMsWithMs(row.time)}</td>
                                                <td className="tdMono">{row.symbol}</td>
                                                <td>
                                                  <span className={sideClass}>{row.side}</span>
                                                </td>
                                                <td className="tdMono">{fmtMoney(row.price, 4)}</td>
                                                <td className="tdMono">{qtyTxt}</td>
                                                <td className="tdMono">{row.positionSide ?? "—"}</td>
                                                <td>
                                                  <span className={pnlBadgeClass(row.realizedPnl)}>{pnlTxt}</span>
                                                </td>
                                                <td className="tdMono">{commissionTxt}</td>
                                                <td className="tdMono">{row.orderId ?? "—"}</td>
                                              </tr>
                                            );
                                          })}
                                        </tbody>
                                      </table>
                                    </div>
                                  ) : (
                                    <div className="hint">No winning trades in this sample.</div>
                                  )}
                                </div>
                                <div className="chartBlock" style={{ marginTop: 12 }}>
                                  <div className="hint">Top losers</div>
                                  {stats.topLosses.length > 0 ? (
                                    <div className="tableWrap" role="region" aria-label="Top losing account trades">
                                      <table className="table">
                                        <thead>
                                          <tr>
                                            <th>Time</th>
                                            <th>Symbol</th>
                                            <th>Side</th>
                                            <th>Price</th>
                                            <th>Qty</th>
                                            <th>Pos</th>
                                            <th>PNL</th>
                                            <th>Commission</th>
                                            <th>Order</th>
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {stats.topLosses.map((row) => {
                                            const sideClass =
                                              row.side === "BUY"
                                                ? "badge badgeStrong badgeLong"
                                                : row.side === "SELL"
                                                  ? "badge badgeStrong badgeFlat"
                                                  : "badge";
                                            const qtyTxt = Number.isFinite(row.qty) ? fmtNum(row.qty, 8) : "—";
                                            const pnlTxt = Number.isFinite(row.realizedPnl) ? fmtMoney(row.realizedPnl, 4) : "—";
                                            const commissionTxt =
                                              row.commission != null && Number.isFinite(row.commission)
                                                ? `${fmtNum(row.commission, 8)}${row.commissionAsset ? ` ${row.commissionAsset}` : ""}`
                                                : "—";
                                            return (
                                              <tr key={`binance-loss-${row.tradeId}`}>
                                                <td className="tdMono">{fmtTimeMsWithMs(row.time)}</td>
                                                <td className="tdMono">{row.symbol}</td>
                                                <td>
                                                  <span className={sideClass}>{row.side}</span>
                                                </td>
                                                <td className="tdMono">{fmtMoney(row.price, 4)}</td>
                                                <td className="tdMono">{qtyTxt}</td>
                                                <td className="tdMono">{row.positionSide ?? "—"}</td>
                                                <td>
                                                  <span className={pnlBadgeClass(row.realizedPnl)}>{pnlTxt}</span>
                                                </td>
                                                <td className="tdMono">{commissionTxt}</td>
                                                <td className="tdMono">{row.orderId ?? "—"}</td>
                                              </tr>
                                            );
                                          })}
                                        </tbody>
                                      </table>
                                    </div>
                                  ) : (
                                    <div className="hint">No losing trades in this sample.</div>
                                  )}
                                </div>
                                <div className="hint" style={{ marginTop: 8 }}>
                                  P&amp;L uses Binance realizedPnl per fill. Spot/margin trades typically omit realizedPnl.
                                </div>
                              </>
                            );
                          })()}
                        </div>
                      </details>
                    ) : (
                      <div className="hint" style={{ marginTop: 10 }}>
                        Realized P&amp;L is unavailable for these trades. Binance only returns realizedPnl for futures fills.
                      </div>
                    )
                  ) : null}
                  {binanceTradesFilteredCount === 0 ? (
                    <div className="hint">
                      {binanceTradesTotalCount > 0 ? "No trades match the filters." : "No trades returned."}
                    </div>
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
                          {binanceTradesFiltered.map((trade) => {
                            const side = binanceTradeSideLabel(trade);
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
                                <td className="tdMono">{fmtTimeMsWithMs(trade.time)}</td>
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
          </CollapsibleCard>

          <CollapsibleCard
            panelId="panel-latest-signal"
            open={isPanelOpen("panel-latest-signal", true)}
            onToggle={handlePanelToggle("panel-latest-signal")}
            maximized={isPanelMaximized("panel-latest-signal")}
            onToggleMaximize={() => togglePanelMaximize("panel-latest-signal")}
            title="Latest signal"
            subtitle={state.latestSignal ? "Computed from the most recent bar." : "Run “Get signal” or “Run backtest” to populate."}
            containerRef={signalRef}
          >
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

                  {latestSignalDecision ? (
                    <div className="decisionBlock">
                      <div className="decisionHeader">
                        <div>
                          <div className="decisionTitle">Decision logic</div>
                          <div className="decisionSubtitle">
                            {latestSignalDecision.isHold
                              ? `Holding${latestSignalDecision.reason ? `: ${latestSignalDecision.reason}` : "."}`
                              : "Signal cleared gates and is eligible to operate."}
                          </div>
                        </div>
                        <span className={decisionBadgeClass(latestSignalDecision.isHold ? "bad" : "ok")}>
                          {latestSignalDecision.isHold ? "hold" : "operate"}
                        </span>
                      </div>
                      <div className="decisionGrid">
                        {latestSignalDecision.checks.map((check) => (
                          <div key={check.id} className="decisionRow">
                            <span className={decisionDotClass(check.status)} aria-hidden="true" />
                            <div>
                              <div className="decisionLabel">{check.label}</div>
                              <div className="decisionDetail">{check.detail}</div>
                            </div>
                            <span className={decisionBadgeClass(check.status)}>{decisionStatusLabel(check.status)}</span>
                          </div>
                        ))}
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
          </CollapsibleCard>

          <CollapsibleCard
            panelId="panel-positions"
            open={isPanelOpen("panel-positions", true)}
            onToggle={handlePanelToggle("panel-positions")}
            maximized={isPanelMaximized("panel-positions")}
            onToggleMaximize={() => togglePanelMaximize("panel-positions")}
            title="Open positions"
            subtitle="Charts for every open Binance futures position (positionRisk + klines)."
            summaryId="section-positions"
            className="chartCard"
          >
              <div className="row" style={{ marginBottom: 10 }}>
                <div className="field">
                  <label className="label" htmlFor="positionsBars">
                    Chart bars
                  </label>
                  <input
                    id="positionsBars"
                    className={binancePositionsBarsError ? "input inputError" : "input"}
                    type="number"
                    min={10}
                    max={1000}
                    value={binancePositionsBars}
                    onChange={(e) => setBinancePositionsBars(numFromInput(e.target.value, binancePositionsBars))}
                  />
                  <div className="hint">Uses the current interval ({form.interval}).</div>
                </div>
                <div className="field">
                  <div className="label">Actions</div>
                  <div className="actions" style={{ marginTop: 0 }}>
                    <button
                      className="btn"
                      type="button"
                      onClick={() => void fetchBinancePositions()}
                      disabled={binancePositionsUi.loading || apiOk !== "ok" || Boolean(binancePositionsInputError)}
                    >
                      {binancePositionsUi.loading ? "Refreshing…" : "Refresh positions"}
                    </button>
                    {binancePositionsUi.response ? (
                      <span className="badge">Updated {fmtTimeMs(binancePositionsUi.response.fetchedAtMs)}</span>
                    ) : null}
                    <span className="badge">{binancePositionsList.length} positions</span>
                  </div>
                  {binancePositionsInputError ? (
                    <div className="hint" style={{ color: "rgba(239, 68, 68, 0.9)" }}>
                      {binancePositionsInputError}
                    </div>
                  ) : (
                    <div className="hint">{binanceSignedKeysHint ?? "Requires Binance API keys with futures access."}</div>
                  )}
                </div>
              </div>

              {binancePositionsUi.error ? (
                <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)", marginBottom: 10 }}>
                  {binancePositionsUi.error}
                </pre>
              ) : null}

              {binancePositionsUi.response ? (
                binancePositionsList.length === 0 ? (
                  <div className="hint">No open futures positions detected.</div>
                ) : (
                  <div style={{ display: "grid", gap: 18 }}>
                    {binancePositionsList.map((pos) => {
                      const chart = binancePositionsCharts.get(pos.symbol);
                      const prices = chart?.prices ?? [];
                      const positionAmt = pos.positionAmt;
                      const sideInfo = positionSideInfo(positionAmt, pos.positionSide);
                      const posDir = sideInfo.dir;
                      const sideLabel = sideInfo.label;
                      const sideKey = sideInfo.key;
                      const pnlClass = pos.unrealizedPnl >= 0 ? "badge badgeLong" : "badge badgeFlat";
                      const positionsSeries = buildPositionSeries(prices, posDir);
                      const equityCurve = buildEquityCurve(prices, posDir);
                      return (
                        <div key={`${pos.symbol}:${sideKey}`}>
                          <div className="pillRow" style={{ marginBottom: 10 }}>
                            <span className="badge badgeStrong">{pos.symbol}</span>
                            <span className={`badge ${posDir > 0 ? "badgeLong" : "badgeFlat"}`}>{sideLabel}</span>
                            <span className="badge">size {fmtNum(Math.abs(positionAmt), 6)}</span>
                            <span className="badge">entry {fmtNum(pos.entryPrice, 6)}</span>
                            <span className="badge">mark {fmtNum(pos.markPrice, 6)}</span>
                            <span className={pnlClass}>PNL {fmtMoney(pos.unrealizedPnl, 4)}</span>
                            {typeof pos.breakEvenPrice === "number" && Number.isFinite(pos.breakEvenPrice) && pos.breakEvenPrice > 0 ? (
                              <span className="badge">break-even {fmtNum(pos.breakEvenPrice, 6)}</span>
                            ) : null}
                            {typeof pos.liquidationPrice === "number" && Number.isFinite(pos.liquidationPrice) && pos.liquidationPrice > 0 ? (
                              <span className="badge">liq {fmtNum(pos.liquidationPrice, 6)}</span>
                            ) : null}
                            {typeof pos.leverage === "number" && Number.isFinite(pos.leverage) ? (
                              <span className="badge">lev {fmtNum(pos.leverage, 2)}x</span>
                            ) : null}
                            {pos.marginType ? <span className="badge">{pos.marginType}</span> : null}
                          </div>
                          {prices.length > 1 ? (
                            <ChartSuspense height={CHART_HEIGHT}>
                              <BacktestChart
                                prices={prices}
                                equityCurve={equityCurve}
                                openTimes={chart?.openTimes}
                                positions={positionsSeries}
                                trades={[]}
                                height={CHART_HEIGHT}
                              />
                            </ChartSuspense>
                          ) : (
                            <div className="chart" style={{ height: CHART_HEIGHT }}>
                              <div className="chartEmpty">No chart data available.</div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )
              ) : (
                <div className="hint">No positions loaded yet.</div>
              )}
          </CollapsibleCard>

          <CollapsibleCard
            panelId="panel-orphaned-operations"
            open={isPanelOpen("panel-orphaned-operations", true)}
            onToggle={handlePanelToggle("panel-orphaned-operations")}
            maximized={isPanelMaximized("panel-orphaned-operations")}
            onToggleMaximize={() => togglePanelMaximize("panel-orphaned-operations")}
            title="Orphaned operations"
            subtitle="Open futures positions that are not currently adopted by a running/starting bot."
            className="chartCard"
          >
              <div className="pillRow" style={{ marginBottom: 10 }}>
                <span className="badge">
                  {binancePositionsUi.response ? orphanPositions.length : "—"} orphaned
                </span>
                {binancePositionsUi.response ? (
                  <span className="badge">Updated {fmtTimeMs(binancePositionsUi.response.fetchedAtMs)}</span>
                ) : null}
                <span className="badge">{binancePositionsList.length} total positions</span>
              </div>

              {binancePositionsInputError ? (
                <div className="hint" style={{ color: "rgba(239, 68, 68, 0.9)", marginBottom: 10 }}>
                  {binancePositionsInputError}
                </div>
              ) : null}

              {binancePositionsUi.error ? (
                <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)", marginBottom: 10 }}>
                  {binancePositionsUi.error}
                </pre>
              ) : null}

              {binancePositionsUi.response ? (
                orphanPositions.length === 0 ? (
                  <div className="hint">No orphaned operations detected.</div>
                ) : (
                  <div style={{ display: "grid", gap: 18 }}>
                    {orphanPositions.map(({ pos, status, reason }) => {
                      const chart = binancePositionsCharts.get(pos.symbol);
                      const prices = chart?.prices ?? [];
                      const positionAmt = pos.positionAmt;
                      const sideInfo = positionSideInfo(positionAmt, pos.positionSide);
                      const posDir = sideInfo.dir;
                      const sideLabel = sideInfo.label;
                      const sideKey = sideInfo.key;
                      const pnlClass = pos.unrealizedPnl >= 0 ? "badge badgeLong" : "badge badgeFlat";
                      const positionsSeries = buildPositionSeries(prices, posDir);
                      const equityCurve = buildEquityCurve(prices, posDir);
                      const tradeEnabled = status?.running
                        ? status.settings?.tradeEnabled
                        : status?.snapshot?.settings?.tradeEnabled;
                      const statusLabelBase = status
                        ? status.running
                          ? "running"
                          : status.starting
                            ? "starting"
                            : status.snapshot
                              ? "snapshot"
                              : "stopped"
                        : null;
                      const statusLabel =
                        statusLabelBase && tradeEnabled === false ? `${statusLabelBase} (trade off)` : statusLabelBase;
                      return (
                        <div key={`orphan-${pos.symbol}-${sideKey}`}>
                          <div className="pillRow" style={{ marginBottom: 10 }}>
                            <span className="badge badgeStrong">{pos.symbol}</span>
                            <span className={`badge ${posDir > 0 ? "badgeLong" : "badgeFlat"}`}>{sideLabel}</span>
                            {statusLabel ? <span className="badge">{statusLabel}</span> : null}
                            <span className="badge">{reason}</span>
                            <span className="badge">size {fmtNum(Math.abs(positionAmt), 6)}</span>
                            <span className="badge">entry {fmtNum(pos.entryPrice, 6)}</span>
                            <span className="badge">mark {fmtNum(pos.markPrice, 6)}</span>
                            <span className={pnlClass}>PNL {fmtMoney(pos.unrealizedPnl, 4)}</span>
                            {typeof pos.breakEvenPrice === "number" && Number.isFinite(pos.breakEvenPrice) && pos.breakEvenPrice > 0 ? (
                              <span className="badge">break-even {fmtNum(pos.breakEvenPrice, 6)}</span>
                            ) : null}
                            {typeof pos.liquidationPrice === "number" && Number.isFinite(pos.liquidationPrice) && pos.liquidationPrice > 0 ? (
                              <span className="badge">liq {fmtNum(pos.liquidationPrice, 6)}</span>
                            ) : null}
                            {typeof pos.leverage === "number" && Number.isFinite(pos.leverage) ? (
                              <span className="badge">lev {fmtNum(pos.leverage, 2)}x</span>
                            ) : null}
                            {pos.marginType ? <span className="badge">{pos.marginType}</span> : null}
                          </div>
                          {prices.length > 1 ? (
                            <ChartSuspense height={CHART_HEIGHT}>
                              <BacktestChart
                                prices={prices}
                                equityCurve={equityCurve}
                                openTimes={chart?.openTimes}
                                positions={positionsSeries}
                                trades={[]}
                                height={CHART_HEIGHT}
                              />
                            </ChartSuspense>
                          ) : (
                            <div className="chart" style={{ height: CHART_HEIGHT }}>
                              <div className="chartEmpty">No chart data available.</div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )
              ) : (
                <div className="hint">Load open positions above to see orphaned operations.</div>
              )}
          </CollapsibleCard>

          <CollapsibleCard
            panelId="panel-backtest-summary"
            open={isPanelOpen("panel-backtest-summary", true)}
            onToggle={handlePanelToggle("panel-backtest-summary")}
            maximized={isPanelMaximized("panel-backtest-summary")}
            onToggleMaximize={() => togglePanelMaximize("panel-backtest-summary")}
            title="Backtest summary"
            subtitle="Uses a time split (train vs held-out backtest). When optimizing, tunes on a fit/tune split inside train."
            containerRef={backtestRef}
            className="chartCard"
          >
              {state.backtest ? (
                <>
                  <div className="analysisDeck">
                    <div className="analysisDeckMain">
                      <ChartSuspense height={CHART_HEIGHT}>
                        <BacktestChart
                          prices={state.backtest.prices}
                          equityCurve={state.backtest.equityCurve}
                          openTimes={state.backtest.openTimes}
                          kalmanPredNext={state.backtest.kalmanPredNext}
                          positions={state.backtest.positions}
                          agreementOk={state.backtest.method === "01" ? undefined : state.backtest.agreementOk}
                          trades={state.backtest.trades}
                          backtestStartIndex={state.backtest.split.backtestStartIndex}
                          height={CHART_HEIGHT}
                          actions={
                            <button className="btn" type="button" onClick={downloadBacktestOps}>
                              Download log
                            </button>
                          }
                        />
                      </ChartSuspense>
                    </div>
                    <div className="analysisDeckSide">
                      <div className="chartBlock">
                        <div className="hint">Prediction values vs thresholds (hover for details)</div>
                        <ChartSuspense height={CHART_HEIGHT_SIDE}>
                          <PredictionDiffChart
                            prices={state.backtest.prices}
                            openTimes={state.backtest.openTimes}
                            kalmanPredNext={state.backtest.kalmanPredNext}
                            lstmPredNext={state.backtest.lstmPredNext}
                            startIndex={state.backtest.split.backtestStartIndex}
                            height={CHART_HEIGHT_SIDE}
                            openThreshold={state.backtest.openThreshold ?? state.backtest.threshold}
                            closeThreshold={
                              state.backtest.closeThreshold ?? state.backtest.openThreshold ?? state.backtest.threshold
                            }
                          />
                        </ChartSuspense>
                      </div>
                    </div>
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

                  <details className="details" style={{ marginTop: 12 }}>
                    <summary>Trade P&amp;L analysis</summary>
                    <div style={{ marginTop: 10 }}>
                      {backtestTradeAnalysis ? (
                        (() => {
                          const stats = backtestTradeAnalysis;
                          const winRateLabel = stats.winRate != null && Number.isFinite(stats.winRate) ? fmtPct(stats.winRate, 1) : "—";
                          const avgWinLabel = stats.avgWin != null && Number.isFinite(stats.avgWin) ? fmtPct(stats.avgWin, 2) : "—";
                          const avgLossLabel = stats.avgLoss != null && Number.isFinite(stats.avgLoss) ? fmtPct(stats.avgLoss, 2) : "—";
                          const avgReturnLabel = stats.avgReturn != null && Number.isFinite(stats.avgReturn) ? fmtPct(stats.avgReturn, 3) : "—";
                          const maxWinLabel = stats.maxWin != null && Number.isFinite(stats.maxWin) ? fmtPct(stats.maxWin, 2) : "—";
                          const maxLossLabel = stats.maxLoss != null && Number.isFinite(stats.maxLoss) ? fmtPct(stats.maxLoss, 2) : "—";
                          const totalWinLabel = Number.isFinite(stats.totalWin) ? fmtPct(stats.totalWin, 2) : "—";
                          const totalLossLabel = Number.isFinite(stats.totalLoss) ? fmtPct(stats.totalLoss, 2) : "—";
                          const profitFactorLabel =
                            stats.profitFactor == null
                              ? "—"
                              : Number.isFinite(stats.profitFactor)
                                ? fmtNum(stats.profitFactor, 3)
                                : "∞";
                          const payoffRatioLabel =
                            stats.payoffRatio == null
                              ? "—"
                              : Number.isFinite(stats.payoffRatio)
                                ? fmtNum(stats.payoffRatio, 3)
                                : "∞";
                          const avgHoldWinLabel =
                            stats.avgHoldWin != null && Number.isFinite(stats.avgHoldWin) ? fmtNum(stats.avgHoldWin, 2) : "—";
                          const avgHoldLossLabel =
                            stats.avgHoldLoss != null && Number.isFinite(stats.avgHoldLoss) ? fmtNum(stats.avgHoldLoss, 2) : "—";
                          const holdMeta =
                            stats.avgHoldWin != null || stats.avgHoldLoss != null
                              ? ` • hold ${avgHoldWinLabel} / ${avgHoldLossLabel} bars`
                              : "";
                          return (
                            <>
                              <div className="summaryGrid">
                                <div className="summaryItem">
                                  <div className="labelRow">
                                    <div className="summaryLabel">Outcomes</div>
                                    <InfoPopover label="Outcomes">
                                      <InfoList items={BACKTEST_TRADE_PNL_TIPS.outcomes} />
                                    </InfoPopover>
                                  </div>
                                  <div className="summaryValue">
                                    <span className="badge badgeStrong badgeLong">{stats.wins} wins</span>
                                    <span className="badge badgeStrong badgeFlat">{stats.losses} losses</span>
                                    <span className="badge">{stats.breakeven} flat</span>
                                    <span className="summaryMeta">Win rate {winRateLabel} • total {stats.count}</span>
                                  </div>
                                </div>
                                <div className="summaryItem">
                                  <div className="labelRow">
                                    <div className="summaryLabel">Avg win / loss</div>
                                    <InfoPopover label="Average win/loss">
                                      <InfoList items={BACKTEST_TRADE_PNL_TIPS.avgWinLoss} />
                                    </InfoPopover>
                                  </div>
                                  <div className="summaryValue">
                                    <span className={pnlBadgeClass(stats.avgWin)}>{avgWinLabel}</span>
                                    <span className={pnlBadgeClass(stats.avgLoss)}>{avgLossLabel}</span>
                                    <span className="summaryMeta">Payoff ratio {payoffRatioLabel}{holdMeta}</span>
                                  </div>
                                </div>
                                <div className="summaryItem">
                                  <div className="labelRow">
                                    <div className="summaryLabel">Best / worst trade</div>
                                    <InfoPopover label="Best/worst trade">
                                      <InfoList items={BACKTEST_TRADE_PNL_TIPS.bestWorst} />
                                    </InfoPopover>
                                  </div>
                                  <div className="summaryValue">
                                    <span className={pnlBadgeClass(stats.maxWin)}>{maxWinLabel}</span>
                                    <span className={pnlBadgeClass(stats.maxLoss)}>{maxLossLabel}</span>
                                    <span className="summaryMeta">Avg return {avgReturnLabel}</span>
                                  </div>
                                </div>
                                <div className="summaryItem">
                                  <div className="labelRow">
                                    <div className="summaryLabel">Trade return totals</div>
                                    <InfoPopover label="Trade return totals" align="left">
                                      <InfoList items={BACKTEST_TRADE_PNL_TIPS.totals} />
                                    </InfoPopover>
                                  </div>
                                  <div className="summaryValue">
                                    <span className={pnlBadgeClass(stats.totalWin)}>{totalWinLabel}</span>
                                    <span className={pnlBadgeClass(stats.totalLoss)}>{totalLossLabel}</span>
                                    <span className="summaryMeta">Profit factor {profitFactorLabel}</span>
                                  </div>
                                </div>
                              </div>
                              <div className="chartBlock" style={{ marginTop: 12 }}>
                                <div className="hint">Top winners</div>
                                {stats.topWins.length > 0 ? (
                                  <div className="tableWrap" role="region" aria-label="Top winning trades">
                                    <table className="table">
                                      <thead>
                                        <tr>
                                          <th>#</th>
                                          <th>Phase</th>
                                          <th>Entry</th>
                                          <th>Exit</th>
                                          <th>Hold</th>
                                          <th>Return</th>
                                          <th>P&amp;L</th>
                                          <th>Exit reason</th>
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {stats.topWins.map((row) => {
                                          const entryTitle = row.entryTime != null ? fmtTimeMs(row.entryTime) : undefined;
                                          const exitTitle = row.exitTime != null ? fmtTimeMs(row.exitTime) : undefined;
                                          const pnlTxt = Number.isFinite(row.pnl) ? fmtNum(row.pnl, 4) : "—";
                                          return (
                                            <tr key={`bt-win-${row.idx}`}>
                                              <td className="tdMono">{row.idx}</td>
                                              <td>
                                                <span className="badge">{row.phase}</span>
                                              </td>
                                              <td className="tdMono" title={entryTitle}>
                                                {row.entryIndex}
                                              </td>
                                              <td className="tdMono" title={exitTitle}>
                                                {row.exitIndex}
                                              </td>
                                              <td className="tdMono">{row.holdingPeriods}</td>
                                              <td>
                                                <span className={pnlBadgeClass(row.return)}>{fmtPct(row.return, 2)}</span>
                                              </td>
                                              <td>
                                                <span className={pnlBadgeClass(row.pnl)}>{pnlTxt}</span>
                                              </td>
                                              <td>{row.exitReason ?? "—"}</td>
                                            </tr>
                                          );
                                        })}
                                      </tbody>
                                    </table>
                                  </div>
                                ) : (
                                  <div className="hint">No winning trades.</div>
                                )}
                              </div>
                              <div className="chartBlock" style={{ marginTop: 12 }}>
                                <div className="hint">Top losers</div>
                                {stats.topLosses.length > 0 ? (
                                  <div className="tableWrap" role="region" aria-label="Top losing trades">
                                    <table className="table">
                                      <thead>
                                        <tr>
                                          <th>#</th>
                                          <th>Phase</th>
                                          <th>Entry</th>
                                          <th>Exit</th>
                                          <th>Hold</th>
                                          <th>Return</th>
                                          <th>P&amp;L</th>
                                          <th>Exit reason</th>
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {stats.topLosses.map((row) => {
                                          const entryTitle = row.entryTime != null ? fmtTimeMs(row.entryTime) : undefined;
                                          const exitTitle = row.exitTime != null ? fmtTimeMs(row.exitTime) : undefined;
                                          const pnlTxt = Number.isFinite(row.pnl) ? fmtNum(row.pnl, 4) : "—";
                                          return (
                                            <tr key={`bt-loss-${row.idx}`}>
                                              <td className="tdMono">{row.idx}</td>
                                              <td>
                                                <span className="badge">{row.phase}</span>
                                              </td>
                                              <td className="tdMono" title={entryTitle}>
                                                {row.entryIndex}
                                              </td>
                                              <td className="tdMono" title={exitTitle}>
                                                {row.exitIndex}
                                              </td>
                                              <td className="tdMono">{row.holdingPeriods}</td>
                                              <td>
                                                <span className={pnlBadgeClass(row.return)}>{fmtPct(row.return, 2)}</span>
                                              </td>
                                              <td>
                                                <span className={pnlBadgeClass(row.pnl)}>{pnlTxt}</span>
                                              </td>
                                              <td>{row.exitReason ?? "—"}</td>
                                            </tr>
                                          );
                                        })}
                                      </tbody>
                                    </table>
                                  </div>
                                ) : (
                                  <div className="hint">No losing trades.</div>
                                )}
                              </div>
                            </>
                          );
                        })()
                      ) : (
                        <div className="hint">No trades available for P&amp;L analysis yet.</div>
                      )}
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
          </CollapsibleCard>

          <CollapsibleCard
            panelId="panel-trade-result"
            open={isPanelOpen("panel-trade-result", true)}
            onToggle={handlePanelToggle("panel-trade-result")}
            maximized={isPanelMaximized("panel-trade-result")}
            onToggleMaximize={() => togglePanelMaximize("panel-trade-result")}
            title="Trade result"
            subtitle="Shows current key status, and trade output after calling /trade."
            containerRef={tradeRef}
          >
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
                  <span className={keysTradeTest.skipped ? "badge badgeWarn" : "badge"}>
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
              {keysTradeTest?.skipped ? (
                <div className="issueItem" style={{ marginBottom: 10 }}>
                  <span>{keysTradeTest.summary}</span>
                </div>
              ) : null}
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
          </CollapsibleCard>

          <CollapsibleCard
            panelId="panel-user-data-stream"
            open={isPanelOpen("panel-user-data-stream", true)}
            onToggle={handlePanelToggle("panel-user-data-stream")}
            maximized={isPanelMaximized("panel-user-data-stream")}
            onToggleMaximize={() => togglePanelMaximize("panel-user-data-stream")}
            title="User data stream (listenKey)"
            subtitle="Backend keeps the Binance user-data listen key alive and relays Binance WebSocket events to the browser."
          >
              <div className="pillRow" style={{ marginBottom: 10 }}>
                <span className="badge">
                  {marketLabel(form.market)}
                  {form.binanceTestnet ? " testnet" : ""}
                </span>
                <span className="badge">Stream: {listenKeyUi.wsStatus}</span>
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
                    : `Binance requires a keep-alive (PUT) at least every ~30 minutes; the backend schedules one every ~${Math.round((listenKeyUi.info?.keepAliveMs ?? 25 * 60_000) * 0.9 / 60_000)} minutes.`}
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
          </CollapsibleCard>

          <CollapsibleCard
            panelId="panel-state-sync"
            open={isPanelOpen("panel-state-sync", false)}
            onToggle={handlePanelToggle("panel-state-sync")}
            maximized={isPanelMaximized("panel-state-sync")}
            onToggleMaximize={() => togglePanelMaximize("panel-state-sync")}
            title="State sync"
            subtitle="Export bot snapshots + optimizer combos and push them to another API."
          >
              <div className="row" style={{ alignItems: "end" }}>
                <div className="field" style={{ flex: "2 1 320px" }}>
                  <label className="label" htmlFor="stateSyncTargetBase">
                    Target API base
                  </label>
                  <input
                    id="stateSyncTargetBase"
                    className="input"
                    value={stateSyncTargetInput}
                    onChange={(e) => setStateSyncTargetInput(e.target.value)}
                    placeholder="https://api.example.com or /api"
                    spellCheck={false}
                  />
                  <div className="hint">POSTs to `/state/sync` on the target API.</div>
                </div>
                <div className="field" style={{ flex: "1 1 220px" }}>
                  <label className="label" htmlFor="stateSyncTargetToken">
                    Target API token (optional)
                  </label>
                  <input
                    id="stateSyncTargetToken"
                    className="input"
                    type={revealSecrets ? "text" : "password"}
                    value={stateSyncTargetToken}
                    onChange={(e) => setStateSyncTargetToken(e.target.value)}
                    placeholder="X-API-Key"
                    spellCheck={false}
                  />
                  <div className="hint">Sent as `X-API-Key` when provided.</div>
                </div>
              </div>

              <div className="hint" style={{ marginTop: 6 }}>
                Source: <span className="tdMono">{apiBaseAbsolute || apiBase}</span>
                {stateSyncTargetAbsolute ? (
                  <>
                    {" "}
                    → Target: <span className="tdMono">{stateSyncTargetAbsolute}</span>
                  </>
                ) : null}
              </div>
              {stateSyncTargetError ? (
                <div className="hint" style={{ marginTop: 6, color: "rgba(239, 68, 68, 0.9)" }}>
                  {stateSyncTargetError}
                </div>
              ) : null}
              {stateSyncTargetCorsHint ? (
                <div className="hint" style={{ marginTop: 6 }}>
                  {stateSyncTargetCorsHint}
                </div>
              ) : null}

              <div className="actions" style={{ marginTop: 10 }}>
                <button
                  className="btn"
                  type="button"
                  disabled={stateSyncUi.exporting || apiOk !== "ok" || stateSyncSelectionEmpty}
                  onClick={() => void exportStateSync()}
                >
                  {stateSyncUi.exporting ? "Exporting…" : "Export"}
                </button>
                <button
                  className="btn btnPrimary"
                  type="button"
                  disabled={
                    stateSyncUi.pushing ||
                    stateSyncUi.exporting ||
                    stateSyncTargetMissing ||
                    Boolean(stateSyncTargetError) ||
                    !stateSyncUi.payload
                  }
                  onClick={() => void sendStateSync()}
                >
                  {stateSyncUi.pushing ? "Sending…" : "Send to target"}
                </button>
                <button
                  className="btn"
                  type="button"
                  disabled={
                    stateSyncUi.pushing ||
                    stateSyncUi.exporting ||
                    stateSyncTargetMissing ||
                    Boolean(stateSyncTargetError) ||
                    stateSyncSelectionEmpty
                  }
                  onClick={() => void exportAndSendStateSync()}
                >
                  {stateSyncUi.exporting || stateSyncUi.pushing ? "Working…" : "Export & send"}
                </button>
                <button
                  className="btn"
                  type="button"
                  disabled={!stateSyncUi.payload}
                  onClick={async () => {
                    await copyText(stateSyncPayloadJson);
                    showToast("Copied state sync payload");
                  }}
                >
                  Copy JSON
                </button>
                <button
                  className="btn"
                  type="button"
                  disabled={!stateSyncUi.payload && !stateSyncUi.pushResult && !stateSyncUi.exportError && !stateSyncUi.pushError}
                  onClick={() =>
                    setStateSyncUi((prev) => ({
                      ...prev,
                      payload: null,
                      exportError: null,
                      exportedAtMs: null,
                      pushError: null,
                      pushResult: null,
                      pushChunkCount: null,
                      pushedAtMs: null,
                    }))
                  }
                >
                  Clear
                </button>
              </div>

              <div className="pillRow" style={{ marginTop: 8 }}>
                <span className="hint" style={{ marginRight: 6 }}>
                  Payload:
                </span>
                <label className="pill" style={{ userSelect: "none" }}>
                  <input
                    type="checkbox"
                    checked={stateSyncIncludeBots}
                    onChange={(e) => setStateSyncIncludeBots(e.target.checked)}
                  />
                  Bot snapshots
                </label>
                <label className="pill" style={{ userSelect: "none" }}>
                  <input
                    type="checkbox"
                    checked={stateSyncIncludeCombos}
                    onChange={(e) => setStateSyncIncludeCombos(e.target.checked)}
                  />
                  Optimizer combos
                </label>
              </div>
              <div className="row" style={{ marginTop: 8, alignItems: "end" }}>
                <div className="field" style={{ flex: "1 1 240px" }}>
                  <label className="label" htmlFor="stateSyncChunkBytes">
                    Max bytes per request
                  </label>
                  <input
                    id="stateSyncChunkBytes"
                    className="input"
                    value={stateSyncChunkBytesInput}
                    onChange={(e) => setStateSyncChunkBytesInput(e.target.value)}
                    placeholder={String(STATE_SYNC_CHUNK_DEFAULT_BYTES)}
                    spellCheck={false}
                  />
                  <div className="hint">0 disables bot chunking; combos are sent separately.</div>
                </div>
              </div>
              {stateSyncSelectionEmpty ? (
                <div className="hint" style={{ marginTop: 6, color: "rgba(239, 68, 68, 0.9)" }}>
                  Select bot snapshots and/or optimizer combos to export.
                </div>
              ) : null}

              {stateSyncUi.exportError ? (
                <div className="hint" style={{ marginTop: 8, color: "rgba(239, 68, 68, 0.9)" }}>
                  {stateSyncUi.exportError}
                </div>
              ) : null}
              {stateSyncUi.pushError ? (
                <div className="hint" style={{ marginTop: 8, color: "rgba(239, 68, 68, 0.9)" }}>
                  {stateSyncUi.pushError}
                </div>
              ) : null}

              {stateSyncPayloadSummary ? (
                <div className="pillRow" style={{ marginTop: 10 }}>
                  <span className="badge">Bots {stateSyncPayloadSummary.botCount}</span>
                  <span className="badge">
                    Combos {stateSyncPayloadSummary.comboCount == null ? "—" : stateSyncPayloadSummary.comboCount}
                  </span>
                  {stateSyncPayloadSummary.generatedAtMs ? (
                    <span className="badge">Generated {fmtTimeMs(stateSyncPayloadSummary.generatedAtMs)}</span>
                  ) : null}
                  {stateSyncUi.exportedAtMs ? <span className="badge">Exported {fmtTimeMs(stateSyncUi.exportedAtMs)}</span> : null}
                </div>
              ) : (
                <div className="hint" style={{ marginTop: 10 }}>
                  No export yet.
                </div>
              )}
              {stateSyncChunkPlan ? (
                <div className="hint" style={{ marginTop: 6 }}>
                  Payload {fmtNum(stateSyncChunkPlan.payloadBytes, 0)} bytes; sending {stateSyncChunkPlan.chunkCount} request
                  {stateSyncChunkPlan.chunkCount === 1 ? "" : "s"}
                  {stateSyncChunkBytes > 0 ? ` (max ${fmtNum(stateSyncChunkBytes, 0)} bytes)` : "; bot chunking disabled"}
                  {stateSyncChunkPlan.largestChunkBytes > 0 && stateSyncChunkBytes > 0 && stateSyncChunkPlan.largestChunkBytes > stateSyncChunkBytes
                    ? " — some chunks exceed the limit; increase max bytes or reduce payload."
                    : ""}
                </div>
              ) : null}

              {stateSyncUi.pushResult ? (
                <div className="pillRow" style={{ marginTop: 8 }}>
                  <span className="badge">Target {stateSyncUi.pushResult.ok ? "OK" : "Error"}</span>
                  {stateSyncUi.pushResult.topCombos?.action ? (
                    <span className="badge">Combos {stateSyncUi.pushResult.topCombos.action}</span>
                  ) : null}
                  {stateSyncUi.pushChunkCount ? <span className="badge">Chunks {stateSyncUi.pushChunkCount}</span> : null}
                  {stateSyncUi.pushedAtMs ? <span className="badge">Sent {fmtTimeMs(stateSyncUi.pushedAtMs)}</span> : null}
                </div>
              ) : null}

              {stateSyncUi.payload ? (
                <details className="details" style={{ marginTop: 10 }}>
                  <summary>Payload JSON</summary>
                  <pre className="code" style={{ marginTop: 10 }}>
                    {stateSyncPayloadJson}
                  </pre>
                </details>
              ) : null}
          </CollapsibleCard>

          <CollapsibleCard
            panelId="panel-request-preview"
            open={isPanelOpen("panel-request-preview", false)}
            onToggle={handlePanelToggle("panel-request-preview")}
            maximized={isPanelMaximized("panel-request-preview")}
            onToggleMaximize={() => togglePanelMaximize("panel-request-preview")}
            title="Request preview"
            subtitle="This JSON is what the UI sends to the API (excluding session-stored secrets)."
          >
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
          </CollapsibleCard>

          <CollapsibleCard
            panelId="panel-performance"
            open={isPanelOpen("panel-performance", false)}
            onToggle={handlePanelToggle("panel-performance")}
            maximized={isPanelMaximized("panel-performance")}
            onToggleMaximize={() => togglePanelMaximize("panel-performance")}
            title="Performance vs code"
            subtitle="Per-commit rollups and deltas from the ops log (bot.status/bot.order)."
          >
            <div className="row" style={{ marginBottom: 10, gridTemplateColumns: "repeat(4, minmax(0, 1fr)) auto", alignItems: "end" }}>
              <div className="field">
                <label className="label" htmlFor="opsPerfCommitLimit">
                  Commit rows
                </label>
                <input
                  id="opsPerfCommitLimit"
                  className="input"
                  type="number"
                  min={1}
                  max={200}
                  value={opsPerformanceCommitLimit}
                  onChange={(e) =>
                    setOpsPerformanceCommitLimit(
                      clamp(Math.trunc(numFromInput(e.target.value, opsPerformanceCommitLimit)), 1, 200),
                    )
                  }
                />
              </div>
              <div className="field">
                <label className="label" htmlFor="opsPerfComboLimit">
                  Combo rows
                </label>
                <input
                  id="opsPerfComboLimit"
                  className="input"
                  type="number"
                  min={1}
                  max={200}
                  value={opsPerformanceComboLimit}
                  onChange={(e) =>
                    setOpsPerformanceComboLimit(
                      clamp(Math.trunc(numFromInput(e.target.value, opsPerformanceComboLimit)), 1, 200),
                    )
                  }
                />
              </div>
              <div className="field">
                <label className="label" htmlFor="opsPerfComboScope">
                  Combo scope
                </label>
                <select
                  id="opsPerfComboScope"
                  className="select"
                  value={opsPerformanceComboScope}
                  onChange={(e) => setOpsPerformanceComboScope(e.target.value === "all" ? "all" : "latest")}
                >
                  <option value="latest">Latest commit</option>
                  <option value="all">All history</option>
                </select>
              </div>
              <div className="field">
                <label className="label" htmlFor="opsPerfComboOrder">
                  Combo order
                </label>
                <select
                  id="opsPerfComboOrder"
                  className="select"
                  value={opsPerformanceComboOrder}
                  onChange={(e) => {
                    const value = e.target.value;
                    if (value === "recent" || value === "return" || value === "delta") setOpsPerformanceComboOrder(value);
                  }}
                >
                  <option value="delta">Worst delta return</option>
                  <option value="recent">Most recent</option>
                  <option value="return">Highest return</option>
                </select>
              </div>
              <button
                className="btn"
                type="button"
                disabled={opsPerformanceUi.loading || apiOk !== "ok"}
                onClick={() => void fetchOpsPerformance()}
              >
                {opsPerformanceUi.loading ? "Loading..." : "Refresh"}
              </button>
            </div>

            {!opsPerformanceUi.enabled ? (
              <div className="hint">{opsPerformanceUi.hint ?? "Enable TRADER_DB_URL to load performance rollups."}</div>
            ) : opsPerformanceUi.hint ? (
              <div className="hint">{opsPerformanceUi.hint}</div>
            ) : null}

            {opsPerformanceUi.error ? (
              <div className="hint" style={{ color: "rgba(239, 68, 68, 0.9)", marginBottom: 6 }}>
                {opsPerformanceUi.error}
              </div>
            ) : null}

            <div className="pillRow" style={{ marginBottom: 10 }}>
              <span className="badge">Commits {opsPerformanceUi.commits.length}</span>
              <span className="badge">Combos {opsPerformanceUi.combos.length}</span>
              {opsPerformanceUi.lastFetchedAtMs ? <span className="badge">Synced {fmtTimeMs(opsPerformanceUi.lastFetchedAtMs)}</span> : null}
            </div>

            <div style={{ marginBottom: 12 }}>
              <div className="hint" style={{ marginBottom: 6 }}>
                Commit deltas
              </div>
              {opsPerformanceUi.commits.length === 0 ? (
                <div className="hint">No commit rollups yet.</div>
              ) : (
                <div className="tableWrap" role="region" aria-label="Commit performance rollups">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Commit</th>
                        <th>Committed</th>
                        <th>Median return</th>
                        <th>Δ return</th>
                        <th>Median DD</th>
                        <th>Δ DD</th>
                        <th>Worst DD</th>
                        <th>Δ worst</th>
                        <th>Samples</th>
                      </tr>
                    </thead>
                    <tbody>
                      {opsPerformanceUi.commits.map((row) => {
                        const commitLabel = shortCommitHash(row.commitHash);
                        const commitTitle = row.commitHash ?? "—";
                        const committedLabel =
                          typeof row.committedAtMs === "number" && Number.isFinite(row.committedAtMs)
                            ? fmtTimeMs(row.committedAtMs)
                            : "—";
                        const medianReturn = typeof row.medianReturn === "number" && Number.isFinite(row.medianReturn)
                          ? fmtPct(row.medianReturn, 2)
                          : "—";
                        const deltaReturn = typeof row.deltaMedianReturn === "number" && Number.isFinite(row.deltaMedianReturn)
                          ? fmtPct(row.deltaMedianReturn, 2)
                          : "—";
                        const medianDd = typeof row.medianDrawdown === "number" && Number.isFinite(row.medianDrawdown)
                          ? fmtPct(row.medianDrawdown, 2)
                          : "—";
                        const deltaDd = typeof row.deltaMedianDrawdown === "number" && Number.isFinite(row.deltaMedianDrawdown)
                          ? fmtPct(row.deltaMedianDrawdown, 2)
                          : "—";
                        const worstDd = typeof row.worstDrawdown === "number" && Number.isFinite(row.worstDrawdown)
                          ? fmtPct(row.worstDrawdown, 2)
                          : "—";
                        const deltaWorst = typeof row.deltaWorstDrawdown === "number" && Number.isFinite(row.deltaWorstDrawdown)
                          ? fmtPct(row.deltaWorstDrawdown, 2)
                          : "—";
                        const samples = `${row.rollups ?? 0} rollups • ${row.combos ?? 0} combos • ${row.symbols ?? 0} symbols`;
                        return (
                          <tr key={`commit-${row.gitCommitId}`}>
                            <td className="tdMono" title={commitTitle}>
                              {commitLabel}
                            </td>
                            <td>{committedLabel}</td>
                            <td>
                              <span className={pnlBadgeClass(row.medianReturn)}>{medianReturn}</span>
                            </td>
                            <td>
                              <span className={pnlBadgeClass(row.deltaMedianReturn)}>{deltaReturn}</span>
                            </td>
                            <td>
                              <span className={pnlBadgeClass(row.medianDrawdown)}>{medianDd}</span>
                            </td>
                            <td>
                              <span className={pnlBadgeClass(row.deltaMedianDrawdown)}>{deltaDd}</span>
                            </td>
                            <td>
                              <span className={pnlBadgeClass(row.worstDrawdown)}>{worstDd}</span>
                            </td>
                            <td>
                              <span className={pnlBadgeClass(row.deltaWorstDrawdown)}>{deltaWorst}</span>
                            </td>
                            <td>{samples}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div>
              <div className="hint" style={{ marginBottom: 6 }}>
                Combo deltas
              </div>
              {!opsPerformanceUi.combosReady ? (
                <div className="hint">Combo deltas are unavailable until rollups are built.</div>
              ) : opsPerformanceUi.combos.length === 0 ? (
                <div className="hint">No combo deltas available for the selected scope.</div>
              ) : (
                <div className="tableWrap" role="region" aria-label="Combo performance deltas">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Market</th>
                        <th>Interval</th>
                        <th>Combo</th>
                        <th>Return</th>
                        <th>Δ return</th>
                        <th>Max DD</th>
                        <th>Δ DD</th>
                        <th>Commit</th>
                      </tr>
                    </thead>
                    <tbody>
                      {opsPerformanceUi.combos.map((row, idx) => {
                        const symbol = row.symbol ?? "—";
                        const market = row.market ? marketLabel(row.market as Market) : "—";
                        const interval = row.interval ?? "—";
                        const combo = shortComboUuid(row.comboUuid);
                        const commitLabel = shortCommitHash(row.commitHash, 7);
                        const commitTitle = row.commitHash ?? "—";
                        const ret = typeof row.return === "number" && Number.isFinite(row.return) ? fmtPct(row.return, 2) : "—";
                        const deltaRet = typeof row.deltaReturn === "number" && Number.isFinite(row.deltaReturn)
                          ? fmtPct(row.deltaReturn, 2)
                          : "—";
                        const dd = typeof row.maxDrawdown === "number" && Number.isFinite(row.maxDrawdown)
                          ? fmtPct(row.maxDrawdown, 2)
                          : "—";
                        const deltaDd = typeof row.deltaDrawdown === "number" && Number.isFinite(row.deltaDrawdown)
                          ? fmtPct(row.deltaDrawdown, 2)
                          : "—";
                        return (
                          <tr key={`combo-${row.comboUuid ?? idx}-${row.gitCommitId}`}>
                            <td className="tdMono">{symbol}</td>
                            <td>{market}</td>
                            <td>{interval}</td>
                            <td className="tdMono" title={row.comboUuid ?? "—"}>
                              {combo}
                            </td>
                            <td>
                              <span className={pnlBadgeClass(row.return)}>{ret}</span>
                            </td>
                            <td>
                              <span className={pnlBadgeClass(row.deltaReturn)}>{deltaRet}</span>
                            </td>
                            <td>
                              <span className={pnlBadgeClass(row.maxDrawdown)}>{dd}</span>
                            </td>
                            <td>
                              <span className={pnlBadgeClass(row.deltaDrawdown)}>{deltaDd}</span>
                            </td>
                            <td className="tdMono" title={commitTitle}>
                              {commitLabel}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </CollapsibleCard>
        </section>

        <CollapsibleCard
          panelId="panel-data-log"
          open={isPanelOpen("panel-data-log", false)}
          onToggle={handlePanelToggle("panel-data-log")}
          maximized={isPanelMaximized("panel-data-log")}
          onToggleMaximize={() => togglePanelMaximize("panel-data-log")}
          title="Data Log"
          subtitle={`Recent API responses (last ${DATA_LOG_MAX_ENTRIES} entries)`}
          style={{ marginTop: "18px" }}
        >
	          <div className="actions dataLogActions">
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
                className="input dataLogFilter"
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
                <input type="checkbox" checked={dataLogLogErrors} onChange={(e) => setDataLogLogErrors(e.target.checked)} />
                Log errors
              </label>
              <label className="pill" style={{ userSelect: "none" }}>
                <input
                  type="checkbox"
                  checked={dataLogLogBackground}
                  onChange={(e) => setDataLogLogBackground(e.target.checked)}
                />
                Log auto-refresh
              </label>
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogIndexArrays} onChange={(e) => setDataLogIndexArrays(e.target.checked)} />
                Index arrays
              </label>
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogAutoScroll} onChange={(e) => setDataLogAutoScroll(e.target.checked)} />
                Auto-scroll
              </label>
              <label className="pill" style={{ userSelect: "none" }}>
                <input type="checkbox" checked={dataLogPersist} onChange={(e) => setDataLogPersist(e.target.checked)} />
                Remember
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
          <div ref={dataLogRef} className="dataLogBox" onScroll={handleDataLogScroll}>
            {dataLogShownDeferred.length === 0 ? (
              <div className="dataLogEmpty">
                {dataLog.length === 0
                  ? "No data logged yet. Run a signal, backtest, trade, or bot action to see incoming data."
                  : "No entries match the current filter."}
              </div>
            ) : (
              dataLogShownDeferred.map((entry, idx) => (
                <div key={idx} className="dataLogEntry">
                  <div className="dataLogEntryHeader">
                    [{new Date(entry.timestamp).toLocaleTimeString()}] <span className="dataLogEntryLabel">{entry.label}</span>
                  </div>
	                  <div className="dataLogEntryBody">
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
        </CollapsibleCard>
      </main>
      <div className="dockBottom">
        <CollapsibleCard
          panelId="panel-combos"
          open={isPanelOpen("panel-combos", true)}
          onToggle={handlePanelToggle("panel-combos")}
          maximized={isPanelMaximized("panel-combos")}
          onToggleMaximize={() => togglePanelMaximize("panel-combos")}
          title="Optimizer Combos"
          subtitle="Browse, apply, and run optimizer payloads."
          className="combosCard"
        >
          {combosOpen ? (
            <Suspense fallback={<PanelFallback label="Loading optimizer combos…" />}>
              <OptimizerCombosPanel {...optimizerCombosProps} />
            </Suspense>
          ) : (
            <div className="hint">Expand to load optimizer combos.</div>
          )}
        </CollapsibleCard>
      </div>
    </div>
  </div>
  );
}
