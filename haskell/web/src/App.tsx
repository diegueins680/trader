import React, { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  OptimizerRunRequest,
  OptimizerRunResponse,
  OptimizerSource,
  Platform,
  Positioning,
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
  optimizerCombos,
  optimizerRun,
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
  DATA_LOG_AUTO_SCROLL_SLOP_PX,
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
  STORAGE_CONFIG_PANEL_ORDER_KEY,
  STORAGE_KEY,
  STORAGE_ORDER_LOG_PREFS_KEY,
  STORAGE_PANEL_PREFS_KEY,
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
  buildOrphanedPositions,
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
  normalizeSymbolKey,
  numFromInput,
} from "./app/utils";
import type { OptimizationCombo, OptimizationComboOperation } from "./components/TopCombosChart";

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

type PanelPrefs = Record<string, boolean>;

type KeysStatus = BinanceKeysStatus | CoinbaseKeysStatus;

type OpsUiState = {
  loading: boolean;
  error: string | null;
  enabled: boolean;
  hint: string | null;
  ops: OpsOperation[];
  lastFetchedAtMs: number | null;
};

type BotStatusOp = {
  atMs: number;
  running: boolean;
  live: boolean;
  symbol: string | null;
};

type PanelToggleHandler = (event: React.SyntheticEvent<HTMLDetailsElement>) => void;

type ConfigPanelId =
  | "config-access"
  | "config-market"
  | "config-strategy"
  | "config-optimization"
  | "config-execution";

type ConfigPanelDragState = {
  draggingId: ConfigPanelId | null;
  overId: ConfigPanelId | null;
};

type CollapsibleCardProps = {
  panelId: string;
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  open: boolean;
  onToggle: PanelToggleHandler;
  className?: string;
  summaryId?: string;
  style?: React.CSSProperties;
  containerRef?: React.RefObject<HTMLDetailsElement>;
  maximized?: boolean;
  onToggleMaximize?: () => void;
};

type CollapsibleSectionProps = {
  panelId: string;
  title: string;
  meta?: string;
  children: React.ReactNode;
  open: boolean;
  onToggle: PanelToggleHandler;
};

type ConfigPanelProps = {
  panelId: ConfigPanelId;
  title: string;
  subtitle?: string;
  draggable?: boolean;
  order: number;
  open: boolean;
  onToggle: PanelToggleHandler;
  maximized: boolean;
  onToggleMaximize: () => void;
  dragState: ConfigPanelDragState;
  onDragStart: (panelId: ConfigPanelId) => (event: React.DragEvent<HTMLButtonElement>) => void;
  onDragOver: (panelId: ConfigPanelId) => (event: React.DragEvent<HTMLElement>) => void;
  onDrop: (panelId: ConfigPanelId) => (event: React.DragEvent<HTMLElement>) => void;
  onDragEnd: () => void;
  style?: React.CSSProperties;
  children: React.ReactNode;
};

type InfoPopoverProps = {
  label: string;
  children: React.ReactNode;
  align?: "left" | "right";
};

type InfoListProps = {
  items: string[];
};

const InfoPopover = ({ label, children, align = "right" }: InfoPopoverProps) => (
  <details className={`infoPopover${align === "left" ? " infoPopoverLeft" : ""}`}>
    <summary className="infoButton" aria-label={label} title={label}>
      i
    </summary>
    <div className="infoContent" role="note">
      {children}
    </div>
  </details>
);

const InfoList = ({ items }: InfoListProps) => (
  <ul className="infoList">
    {items.map((item) => (
      <li key={item}>{item}</li>
    ))}
  </ul>
);

const BacktestChart = lazy(() => import("./components/BacktestChart").then((mod) => ({ default: mod.BacktestChart })));
const BotStateChart = lazy(() => import("./components/BotStateChart").then((mod) => ({ default: mod.BotStateChart })));
const LiveVisuals = lazy(() => import("./components/LiveVisuals").then((mod) => ({ default: mod.LiveVisuals })));
const PredictionDiffChart = lazy(() =>
  import("./components/PredictionDiffChart").then((mod) => ({ default: mod.PredictionDiffChart })),
);
const TelemetryChart = lazy(() => import("./components/TelemetryChart").then((mod) => ({ default: mod.TelemetryChart })));
const TopCombosChart = lazy(() => import("./components/TopCombosChart").then((mod) => ({ default: mod.TopCombosChart })));

const CollapsibleCard = ({
  panelId,
  title,
  subtitle,
  children,
  open,
  onToggle,
  className,
  summaryId,
  style,
  containerRef,
  maximized = false,
  onToggleMaximize,
}: CollapsibleCardProps) => (
  <details
    className={`card cardCollapsible${maximized ? " cardMaximized" : ""}${className ? ` ${className}` : ""}`}
    open={open}
    onToggle={onToggle}
    data-panel={panelId}
    ref={containerRef}
    style={style}
  >
    <summary className="cardHeader cardSummary" id={summaryId}>
      <div className="cardHeaderText">
        <h2 className="cardTitle">{title}</h2>
        {subtitle ? <p className="cardSubtitle">{subtitle}</p> : null}
      </div>
      <div className="cardControls">
        {onToggleMaximize ? (
          <button
            className="cardControl"
            type="button"
            aria-pressed={maximized}
            aria-label={maximized ? "Restore panel size" : "Maximize panel"}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onToggleMaximize();
            }}
          >
            {maximized ? "Restore" : "Maximize"}
          </button>
        ) : null}
        <span className="cardToggle" aria-hidden="true">
          <span className="cardToggleLabel" data-open="Collapse" data-closed="Expand" />
          <span className="cardToggleIcon" />
        </span>
      </div>
    </summary>
    <div className="cardBody">{children}</div>
  </details>
);

const CollapsibleSection = ({ panelId, title, meta, children, open, onToggle }: CollapsibleSectionProps) => (
  <details className="sectionPanel" open={open} onToggle={onToggle} data-panel={panelId}>
    <summary className="sectionHeading" id={panelId}>
      <span className="sectionTitle">{title}</span>
      {meta ? <span className="sectionMeta">{meta}</span> : null}
      <span className="sectionToggle" aria-hidden="true">
        <span className="sectionToggleLabel" data-open="Hide" data-closed="Show" />
        <span className="sectionToggleIcon" />
      </span>
    </summary>
    <div className="sectionBody">{children}</div>
  </details>
);

const ConfigPanel = ({
  panelId,
  title,
  subtitle,
  draggable = true,
  order,
  open,
  onToggle,
  maximized,
  onToggleMaximize,
  dragState,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
  style,
  children,
}: ConfigPanelProps) => {
  const isDragOver = dragState.overId === panelId && dragState.draggingId !== panelId;
  const isDragging = dragState.draggingId === panelId;
  return (
    <details
      className={`configPanel${maximized ? " configPanelMaximized" : ""}${isDragOver ? " configPanelDrop" : ""}${
        isDragging ? " configPanelDragging" : ""
      }`}
      open={open}
      onToggle={onToggle}
      style={{ order, ...style }}
      onDragOver={onDragOver(panelId)}
      onDrop={onDrop(panelId)}
      data-panel={panelId}
    >
      <summary className="configPanelHeader configPanelSummary">
        <div className="configPanelHeaderText">
          <span className="configPanelTitle">{title}</span>
          {subtitle ? <span className="configPanelSubtitle">{subtitle}</span> : null}
        </div>
        <div className="configPanelControls">
          {draggable ? (
            <button
              className="configPanelHandle"
              type="button"
              draggable={!maximized}
              onDragStart={onDragStart(panelId)}
              onDragEnd={onDragEnd}
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
              }}
              aria-label={`Drag ${title} panel`}
              title="Drag to reorder"
            >
              Drag
            </button>
          ) : null}
          <button
            className="configPanelControl"
            type="button"
            aria-pressed={maximized}
            aria-label={maximized ? "Restore panel size" : "Maximize panel"}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onToggleMaximize();
            }}
          >
            {maximized ? "Restore" : "Maximize"}
          </button>
          <span className="configPanelToggle" aria-hidden="true">
            <span className="configPanelToggleLabel" data-open="Minimize" data-closed="Expand" />
            <span className="configPanelToggleIcon" />
          </span>
        </div>
      </summary>
      <div className="configPanelBody">{children}</div>
    </details>
  );
};

const BOT_STATUS_OPS_LIMIT = 5000;
const BOT_DISPLAY_STALE_MS = 6_000;
const BOT_DISPLAY_STARTING_STALE_MS = Number.POSITIVE_INFINITY;
const CHART_HEIGHT = "var(--chart-height)";
const CHART_HEIGHT_SIDE = "var(--chart-height-side)";
const CHART_HEIGHT_TIMELINE = "var(--chart-height-timeline)";
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
const CONFIG_SECTION_IDS = [
  "section-market",
  "section-lookback",
  "section-thresholds",
  "section-risk",
  "section-optimizer-run",
  "section-optimization",
  "section-livebot",
  "section-trade",
] as const;
const CONFIG_PANEL_IDS: ConfigPanelId[] = [
  "config-access",
  "config-market",
  "config-strategy",
  "config-optimization",
  "config-execution",
];
const CONFIG_PANEL_HEIGHTS: Record<ConfigPanelId, string> = {
  "config-access": "clamp(260px, 32vh, 360px)",
  "config-market": "clamp(280px, 38vh, 420px)",
  "config-strategy": "clamp(320px, 50vh, 600px)",
  "config-optimization": "clamp(320px, 50vh, 600px)",
  "config-execution": "clamp(320px, 50vh, 600px)",
};
const normalizeConfigPanelOrder = (order: unknown): ConfigPanelId[] => {
  const seen = new Set<ConfigPanelId>();
  const out: ConfigPanelId[] = [];
  if (Array.isArray(order)) {
    for (const value of order) {
      if (CONFIG_PANEL_IDS.includes(value as ConfigPanelId)) {
        const id = value as ConfigPanelId;
        if (!seen.has(id)) {
          seen.add(id);
          out.push(id);
        }
      }
    }
  }
  for (const id of CONFIG_PANEL_IDS) {
    if (!seen.has(id)) out.push(id);
  }
  return out;
};
const EQUITY_TIPS = {
  preset: [
    'Use "Preset: Equity focus", then bump Trials/Timeout to widen the search.',
    'To bias toward raw equity, reduce or clear drawdown/turnover penalties while keeping Objective/Tune objective on "annualized-equity".',
    "For short windows (e.g., 48h), shorter intervals (15m/30m/1h) increase sample size; keep Backtest + Tune ratios < 1.",
  ],
  trials: ["Higher Trials/Timeout expands the search (slower) and can lift annualized equity."],
  objective: ['Keep Objective and Tune objective on "annualized-equity" for equity-focused ranking.'],
  penalties: ["Reduce or clear drawdown/turnover penalties to favor raw equity."],
  intervals: ["For short windows (e.g., 48h), shorter intervals (15m/30m/1h) increase sample size."],
  ratios: ["Keep Backtest ratio + Tune ratio < 1 to leave enough training data."],
};
const COMPLEX_TIPS = {
  method: [
    "11 requires Kalman + LSTM agreement beyond the open threshold; fewer trades, higher confidence.",
    "blend averages predictions; blend weight sets the Kalman vs LSTM mix.",
    "router picks the best recent model using router lookback; min score gates to HOLD.",
  ],
  thresholds: [
    "Open threshold is the entry deadband; below break-even can churn after costs.",
    "Close threshold is often <= open threshold to reduce whipsaw.",
  ],
  edge: [
    "Min edge is the minimum predicted return to trade; cost-aware edge adds break-even + buffer.",
    "Edge buffer adds extra margin above break-even when cost-aware edge is on.",
  ],
  snr: ["Signal/vol (SNR) filters trades when predicted edge is small versus recent volatility."],
  blend: ["0 = LSTM only, 1 = Kalman only. Only used with method=blend."],
  router: ["Lookback controls how much recent history the router uses; longer is smoother but slower to adapt.", "Min score gates low-confidence periods to HOLD."],
  split: ["Backtest ratio is the held-out tail; tune ratio is only used for optimization/sweeps.", "Backtest + tune must be < 1 to leave training data."],
  lstm: ["Normalization affects scaling for LSTM only; keep consistent with training.", "Epochs/hidden size trade off fit vs runtime and overfitting."],
  optimization: ["Sweep thresholds searches open/close gates only.", "Optimize operations also tries methods and thresholds; router disables both."],
  tuneObjective: ["Tune objective defines the score used during fit/tune; it can differ from backtest objective."],
  walkForward: ["Walk-forward folds split data into sequential folds to estimate stability."],
};

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

function botStatusKey(status: { market: Market; symbol: string; interval: string }): string {
  return `${status.market}:${normalizeSymbolKey(status.symbol)}:${status.interval}`;
}

function botStatusKeyFromSingle(status: BotStatusSingle): string | null {
  const symbol = botStatusSymbol(status);
  if (!symbol) return null;
  const market = status.running ? status.market : status.market ?? status.snapshot?.market;
  const interval = status.running ? status.interval : status.interval ?? status.snapshot?.interval;
  if (!market || !interval) return null;
  return botStatusKey({ market, symbol, interval });
}

function formatDatetimeLocal(ms: number): string {
  if (!Number.isFinite(ms)) return "";
  const d = new Date(ms);
  const pad = (v: number) => String(v).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function parseDatetimeLocal(raw: string): number | null {
  if (!raw.trim()) return null;
  const parsed = Date.parse(raw);
  return Number.isNaN(parsed) ? null : parsed;
}

function parseBotStatusOp(op: OpsOperation): BotStatusOp | null {
  if (!op || op.kind !== "bot.status") return null;
  if (typeof op.atMs !== "number" || !Number.isFinite(op.atMs)) return null;
  const rec = (op.result as Record<string, unknown> | null | undefined) ?? {};
  const running = typeof rec.running === "boolean" ? rec.running : null;
  if (running == null) return null;
  const live = typeof rec.live === "boolean" ? rec.live : false;
  const symbol = typeof rec.symbol === "string" ? rec.symbol : null;
  return { atMs: op.atMs, running, live, symbol };
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

function symbolFormatPattern(platform: Platform): RegExp {
  switch (platform) {
    case "coinbase":
      return /^[A-Z0-9]+-[A-Z0-9]+$/;
    case "poloniex":
      return /^[A-Z0-9]+_[A-Z0-9]+$/;
    case "binance":
    case "kraken":
    default:
      return /^[A-Z0-9]{3,30}$/;
  }
}

const COMMON_QUOTES = ["USDT", "USDC", "FDUSD", "TUSD", "BUSD", "BTC", "ETH", "BNB"];
const BINANCE_SYMBOL_PATTERN = /^[A-Z0-9]{3,30}$/;

function trimBinanceComboSuffix(value: string): string | null {
  const compact = value.replace(/[^A-Z0-9]/g, "");
  if (!compact) return null;
  let best: string | null = null;
  for (const quote of COMMON_QUOTES) {
    let idx = compact.indexOf(quote);
    while (idx >= 0) {
      const end = idx + quote.length;
      if (end < compact.length) {
        const suffix = compact.slice(end);
        if (/\d/.test(suffix)) {
          const candidate = compact.slice(0, end);
          if (BINANCE_SYMBOL_PATTERN.test(candidate) && !COMMON_QUOTES.includes(candidate)) {
            if (!best || candidate.length > best.length) best = candidate;
          }
        }
      }
      idx = compact.indexOf(quote, idx + 1);
    }
  }
  return best;
}

function normalizeComboSymbol(raw: string, platform: Platform | null): string {
  const value = raw.trim().toUpperCase();
  if (!value) return value;
  const resolvedPlatform: Platform = platform ?? "binance";
  const pattern = symbolFormatPattern(resolvedPlatform);
  const isBinanceLike = resolvedPlatform === "binance" || resolvedPlatform === "kraken";

  if (isBinanceLike) {
    const trimmed = trimBinanceComboSuffix(value);
    if (trimmed) return trimmed;
  }

  if (pattern.test(value)) return value;

  if (resolvedPlatform === "coinbase") {
    const parts = value.split("-");
    if (parts.length >= 2) {
      const candidate = `${parts[0]}-${parts[1]}`;
      if (pattern.test(candidate)) return candidate;
    }
    return value;
  }

  if (resolvedPlatform === "poloniex") {
    const parts = value.split("_");
    if (parts.length >= 2) {
      const candidate = `${parts[0]}_${parts[1]}`;
      if (pattern.test(candidate)) return candidate;
    }
    return value;
  }

  if (isBinanceLike) {
    const tokens = value.split(/[^A-Z0-9]+/).filter(Boolean);
    if (tokens.length >= 2) {
      const joined = `${tokens[0]}${tokens[1]}`;
      if (tokens.length === 2 && pattern.test(joined)) return joined;
      if (tokens.length >= 3 && /^[0-9]+[A-Z]$/.test(tokens[2] ?? "") && pattern.test(joined)) return joined;
    }
    if (tokens.length >= 1 && pattern.test(tokens[0] ?? "")) return tokens[0] ?? value;
  }
  return value;
}

function symbolFormatExample(platform: Platform): string {
  switch (platform) {
    case "coinbase":
      return "BTC-USD";
    case "poloniex":
      return "BTC_USDT";
    case "kraken":
      return "XBTUSD";
    case "binance":
    default:
      return "BTCUSDT";
  }
}

function invalidSymbolsForPlatform(platform: Platform, symbols: string[]): string[] {
  const pattern = symbolFormatPattern(platform);
  return symbols.filter((sym) => !pattern.test(sym));
}

function parseMaybeInt(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const n = Number(trimmed);
  if (!Number.isFinite(n)) return null;
  const rounded = Math.trunc(n);
  return rounded < 0 ? null : rounded;
}

function normalizeIsoInput(raw: string): string | null {
  const trimmed = raw.trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) return `${trimmed}T00:00:00Z`;
  if (
    !/^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,3})?)?(?:Z|[+-]\d{2}:\d{2})?$/.test(
      trimmed,
    )
  ) {
    return null;
  }
  return trimmed.replace(" ", "T");
}

function parseTimeInputMs(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (/^\d+$/.test(trimmed)) {
    const n = Number(trimmed);
    return Number.isFinite(n) ? n : null;
  }
  const iso = normalizeIsoInput(trimmed);
  if (!iso) return null;
  const parsed = Date.parse(iso);
  return Number.isNaN(parsed) ? null : parsed;
}

function sanitizeFilenameSegment(raw: string, fallback: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return fallback;
  const cleaned = trimmed.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-+/, "").replace(/-+$/, "");
  return cleaned || fallback;
}

function csvEscape(value: unknown): string {
  if (value == null) return "";
  const text = String(value);
  if (text === "") return "";
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, "\"\"")}"` : text;
}

function backtestTradePhase(split: BacktestResponse["split"], entryIndex: number): string {
  if (entryIndex >= split.backtestStartIndex) return "backtest";
  if (split.tune > 0 && entryIndex >= split.tuneStartIndex) return "tune";
  return split.tune > 0 ? "fit" : "train";
}

function buildBacktestOpsCsv(backtest: BacktestResponse): string {
  const header = [
    "tradeIndex",
    "phase",
    "entryIndex",
    "exitIndex",
    "entryPrice",
    "exitPrice",
    "entryEquity",
    "exitEquity",
    "return",
    "holdingPeriods",
    "exitReason",
  ].join(",");
  const prices = backtest.prices ?? [];
  const rows = backtest.trades.map((trade, idx) => {
    const entryPrice = prices[trade.entryIndex];
    const exitPrice = prices[trade.exitIndex];
    const phase = backtestTradePhase(backtest.split, trade.entryIndex);
    return [
      idx + 1,
      phase,
      trade.entryIndex,
      trade.exitIndex,
      Number.isFinite(entryPrice) ? entryPrice : "",
      Number.isFinite(exitPrice) ? exitPrice : "",
      trade.entryEquity,
      trade.exitEquity,
      trade.return,
      trade.holdingPeriods,
      trade.exitReason ?? "",
    ]
      .map(csvEscape)
      .join(",");
  });
  return [header, ...rows].join("\n");
}

function downloadTextFile(filename: string, contents: string, contentType = "text/plain"): void {
  if (typeof window === "undefined") return;
  const blob = new Blob([contents], { type: contentType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function buildPositionSeries(prices: number[], side: number): number[] {
  if (prices.length === 0) return [];
  if (!Number.isFinite(side) || side === 0) return Array.from({ length: prices.length }, () => 0);
  const dir = side > 0 ? 1 : -1;
  return Array.from({ length: prices.length }, () => dir);
}

function buildEquityCurve(prices: number[], side: number): number[] {
  if (prices.length === 0) return [];
  const dir = side > 0 ? 1 : side < 0 ? -1 : 0;
  const out = [1];
  for (let i = 1; i < prices.length; i += 1) {
    const prev = prices[i - 1] ?? 0;
    const cur = prices[i] ?? prev;
    const last = out[out.length - 1] ?? 1;
    if (!Number.isFinite(prev) || !Number.isFinite(cur) || prev === 0 || cur === 0 || dir === 0) {
      out.push(last);
      continue;
    }
    const ratio = dir > 0 ? cur / prev : prev / cur;
    out.push(last * ratio);
  }
  return out;
}

function positionSideInfo(positionAmt: number, positionSide?: string | null): { dir: number; label: string; key: string } {
  const raw = positionSide?.trim().toUpperCase();
  const side = raw && raw !== "BOTH" ? raw : null;
  const dir = side === "SHORT" ? -1 : side === "LONG" ? 1 : positionAmt > 0 ? 1 : positionAmt < 0 ? -1 : 0;
  const label = side ?? (dir > 0 ? "LONG" : dir < 0 ? "SHORT" : "FLAT");
  const key = side ?? (dir > 0 ? "LONG" : dir < 0 ? "SHORT" : "FLAT");
  return { dir, label, key };
}

type ListenKeyStreamStatus = "disconnected" | "connecting" | "connected" | "stopped";

type ListenKeyStreamStatusPayload = { status?: string; message?: string; atMs?: number };
type ListenKeyStreamKeepAlivePayload = { atMs?: number };
type ListenKeyStreamErrorPayload = { message?: string; atMs?: number };

function normalizeListenKeyStreamStatus(raw: string): ListenKeyStreamStatus {
  switch (raw) {
    case "connecting":
    case "connected":
    case "disconnected":
    case "stopped":
      return raw;
    default:
      return "disconnected";
  }
}

function safeJsonParse<T = unknown>(raw: string): T | null {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

// Minimal SSE parser for fetch streams that handles chunk boundaries.
function createSseParser(onEvent: (event: string, data: string) => void): (chunk: string) => void {
  let buffer = "";
  return (chunk: string) => {
    buffer += chunk.replace(/\r/g, "");
    while (true) {
      const boundary = buffer.indexOf("\n\n");
      if (boundary === -1) return;
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (!block.trim()) continue;
      let eventName = "message";
      const dataLines: string[] = [];
      for (const line of block.split("\n")) {
        if (!line || line.startsWith(":")) continue;
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
          continue;
        }
        if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trimStart());
        }
      }
      onEvent(eventName, dataLines.join("\n"));
    }
  };
}

function parseOptionalNumber(raw: string): number | undefined {
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  const parsed = numFromInput(trimmed, Number.NaN);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseOptionalInt(raw: string): number | undefined {
  const parsed = parseOptionalNumber(raw);
  if (parsed == null) return undefined;
  return Math.trunc(parsed);
}

function parseOptionalString(raw: string): string | undefined {
  const trimmed = raw.trim();
  return trimmed ? trimmed : undefined;
}

type UiState = {
  loading: boolean;
  error: string | null;
  lastKind: RequestKind | null;
  latestSignal: LatestSignal | null;
  backtest: BacktestResponse | null;
  trade: ApiTradeResponse | null;
};

type ErrorFix =
  | {
      label: string;
      action: "tuneRatio";
      value: number;
      targetId?: string;
      toast: string;
    }
  | {
      label: string;
      action: "backtestRatio";
      value: number;
      targetId?: string;
      toast: string;
    }
  | {
      label: string;
      action: "bars";
      value: number;
      targetId?: string;
      toast: string;
    }
  | {
      label: string;
      action: "lookbackBars";
      value: number;
      targetId?: string;
      toast: string;
    }
  | {
      label: string;
      action: "lookbackWindow";
      value: string;
      targetId?: string;
      toast: string;
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

type BotRtTracker = {
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
};

const emptyBotRtState = (): BotRtUiState => ({
  lastFetchAtMs: null,
  lastFetchDurationMs: null,
  lastNewCandles: 0,
  lastNewCandlesAtMs: null,
  lastKlineUpdates: 0,
  lastKlineUpdatesAtMs: null,
  telemetry: [],
  feed: [],
});

const emptyBotRtTracker = (): BotRtTracker => ({
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
  wsStatus: ListenKeyStreamStatus;
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

type BinancePositionsUiState = {
  loading: boolean;
  error: string | null;
  response: ApiBinancePositionsResponse | null;
};

type OptimizerRunUiState = {
  loading: boolean;
  error: string | null;
  response: OptimizerRunResponse | null;
  lastRunAtMs: number | null;
};

type OptimizerRunForm = {
  source: OptimizerSource;
  symbol: string;
  dataPath: string;
  priceColumn: string;
  highColumn: string;
  lowColumn: string;
  platforms: string;
  intervals: string;
  lookbackWindow: string;
  barsMin: string;
  barsMax: string;
  barsAutoProb: string;
  barsDistribution: "" | "uniform" | "log";
  trials: string;
  timeoutSec: string;
  seed: string;
  seedTrials: string;
  seedRatio: string;
  survivorFraction: string;
  perturbScaleDouble: string;
  perturbScaleInt: string;
  earlyStopNoImprove: string;
  objective: string;
  tuneObjective: string;
  backtestRatio: string;
  tuneRatio: string;
  penaltyMaxDrawdown: string;
  penaltyTurnover: string;
  normalizations: string;
  epochsMin: string;
  epochsMax: string;
  hiddenSizeMin: string;
  hiddenSizeMax: string;
  lrMin: string;
  lrMax: string;
  patienceMax: string;
  gradClipMin: string;
  gradClipMax: string;
  pDisableGradClip: string;
  slippageMax: string;
  spreadMax: string;
  minRoundTrips: string;
  minWinRate: string;
  minSharpe: string;
  minAnnualizedReturn: string;
  minCalmar: string;
  minProfitFactor: string;
  maxTurnover: string;
  minExposure: string;
  minWalkForwardSharpeMean: string;
  maxWalkForwardSharpeStd: string;
  walkForwardFoldsMin: string;
  walkForwardFoldsMax: string;
  minHoldBarsMin: string;
  minHoldBarsMax: string;
  cooldownBarsMin: string;
  cooldownBarsMax: string;
  maxHoldBarsMin: string;
  maxHoldBarsMax: string;
  minEdgeMin: string;
  minEdgeMax: string;
  minSignalToNoiseMin: string;
  minSignalToNoiseMax: string;
  edgeBufferMin: string;
  edgeBufferMax: string;
  trendLookbackMin: string;
  trendLookbackMax: string;
  pCostAwareEdge: string;
  stopMin: string;
  stopMax: string;
  tpMin: string;
  tpMax: string;
  trailMin: string;
  trailMax: string;
  methodWeightBlend: string;
  blendWeightMin: string;
  blendWeightMax: string;
  disableLstmPersistence: boolean;
  noSweepThreshold: boolean;
  extraJson: string;
};

type TopCombosSource = "api";

type TopCombosMeta = {
  source: TopCombosSource;
  generatedAtMs: number | null;
  payloadSource: string | null;
  payloadSources: string[] | null;
  fallbackReason: string | null;
  comboCount: number | null;
};

type ComboOrder = "annualized-equity" | "rank" | "date-desc" | "date-asc";

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

function optimizerSourceForPlatform(platform: Platform): OptimizerSource {
  switch (platform) {
    case "coinbase":
      return "coinbase";
    case "kraken":
      return "kraken";
    case "poloniex":
      return "poloniex";
    default:
      return "binance";
  }
}

function buildDefaultOptimizerRunForm(symbol: string, platform: Platform): OptimizerRunForm {
  return {
    source: optimizerSourceForPlatform(platform),
    symbol: symbol.trim().toUpperCase(),
    dataPath: "",
    priceColumn: "close",
    highColumn: "",
    lowColumn: "",
    platforms: "",
    intervals: "1h,2h,4h,6h,12h,1d",
    lookbackWindow: "7d",
    barsMin: "",
    barsMax: "",
    barsAutoProb: "",
    barsDistribution: "",
    trials: "50",
    timeoutSec: "60",
    seed: "42",
    seedTrials: "",
    seedRatio: "",
    survivorFraction: "",
    perturbScaleDouble: "",
    perturbScaleInt: "",
    earlyStopNoImprove: "",
    objective: "annualized-equity",
    tuneObjective: "annualized-equity",
    backtestRatio: "0.2",
    tuneRatio: "0.25",
    penaltyMaxDrawdown: "",
    penaltyTurnover: "",
    normalizations: "none,minmax,standard,log",
    epochsMin: "",
    epochsMax: "",
    hiddenSizeMin: "",
    hiddenSizeMax: "",
    lrMin: "",
    lrMax: "",
    patienceMax: "",
    gradClipMin: "",
    gradClipMax: "",
    pDisableGradClip: "",
    slippageMax: "",
    spreadMax: "",
    minRoundTrips: "",
    minWinRate: "",
    minSharpe: "",
    minAnnualizedReturn: "",
    minCalmar: "",
    minProfitFactor: "",
    maxTurnover: "",
    minExposure: "",
    minWalkForwardSharpeMean: "",
    maxWalkForwardSharpeStd: "",
    walkForwardFoldsMin: "",
    walkForwardFoldsMax: "",
    minHoldBarsMin: "",
    minHoldBarsMax: "",
    cooldownBarsMin: "",
    cooldownBarsMax: "",
    maxHoldBarsMin: "",
    maxHoldBarsMax: "",
    minEdgeMin: "",
    minEdgeMax: "",
    minSignalToNoiseMin: "",
    minSignalToNoiseMax: "",
    edgeBufferMin: "",
    edgeBufferMax: "",
    trendLookbackMin: "",
    trendLookbackMax: "",
    pCostAwareEdge: "",
    stopMin: "",
    stopMax: "",
    tpMin: "",
    tpMax: "",
    trailMin: "",
    trailMax: "",
    methodWeightBlend: "",
    blendWeightMin: "",
    blendWeightMax: "",
    disableLstmPersistence: false,
    noSweepThreshold: false,
    extraJson: "",
  };
}

function parseOptimizerExtras(raw: string): { value: Record<string, unknown> | null; error: string | null } {
  const trimmed = raw.trim();
  if (!trimmed) return { value: null, error: null };
  try {
    const parsed = JSON.parse(trimmed);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { value: null, error: "Extra options must be a JSON object." };
    }
    return { value: parsed as Record<string, unknown>, error: null };
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Invalid JSON.";
    return { value: null, error: `Invalid JSON: ${msg}` };
  }
}

function buildOptimizerRunRequest(form: OptimizerRunForm, extras: Record<string, unknown> | null): OptimizerRunRequest {
  const req: OptimizerRunRequest = {
    source: form.source,
  };

  const symbol = parseOptionalString(form.symbol);
  const dataPath = parseOptionalString(form.dataPath);
  const priceColumn = parseOptionalString(form.priceColumn);
  const highColumn = parseOptionalString(form.highColumn);
  const lowColumn = parseOptionalString(form.lowColumn);

  if (form.source === "csv") {
    if (dataPath) req.data = dataPath;
    if (priceColumn) req.priceColumn = priceColumn;
    if (highColumn && lowColumn) {
      req.highColumn = highColumn;
      req.lowColumn = lowColumn;
    }
  } else {
    if (symbol) req.binanceSymbol = symbol.toUpperCase();
    const platforms = parseOptionalString(form.platforms);
    if (platforms) req.platforms = platforms;
  }

  const intervals = parseOptionalString(form.intervals);
  if (intervals) req.intervals = intervals;
  const lookbackWindow = parseOptionalString(form.lookbackWindow);
  if (lookbackWindow) req.lookbackWindow = lookbackWindow;

  const barsMin = parseOptionalInt(form.barsMin);
  if (barsMin != null) req.barsMin = barsMin;
  const barsMax = parseOptionalInt(form.barsMax);
  if (barsMax != null) req.barsMax = barsMax;
  const barsAutoProb = parseOptionalNumber(form.barsAutoProb);
  if (barsAutoProb != null) req.barsAutoProb = barsAutoProb;
  if (form.barsDistribution) req.barsDistribution = form.barsDistribution;

  const trials = parseOptionalInt(form.trials);
  if (trials != null) req.trials = trials;
  const timeoutSec = parseOptionalNumber(form.timeoutSec);
  if (timeoutSec != null) req.timeoutSec = timeoutSec;
  const seed = parseOptionalInt(form.seed);
  if (seed != null) req.seed = seed;
  const seedTrials = parseOptionalInt(form.seedTrials);
  if (seedTrials != null) req.seedTrials = seedTrials;
  const seedRatio = parseOptionalNumber(form.seedRatio);
  if (seedRatio != null) req.seedRatio = seedRatio;
  const survivorFraction = parseOptionalNumber(form.survivorFraction);
  if (survivorFraction != null) req.survivorFraction = survivorFraction;
  const perturbScaleDouble = parseOptionalNumber(form.perturbScaleDouble);
  if (perturbScaleDouble != null) req.perturbScaleDouble = perturbScaleDouble;
  const perturbScaleInt = parseOptionalInt(form.perturbScaleInt);
  if (perturbScaleInt != null) req.perturbScaleInt = perturbScaleInt;
  const earlyStopNoImprove = parseOptionalInt(form.earlyStopNoImprove);
  if (earlyStopNoImprove != null) req.earlyStopNoImprove = earlyStopNoImprove;

  const objective = parseOptionalString(form.objective);
  if (objective) req.objective = objective;
  const tuneObjective = parseOptionalString(form.tuneObjective);
  if (tuneObjective) req.tuneObjective = tuneObjective;
  const backtestRatio = parseOptionalNumber(form.backtestRatio);
  if (backtestRatio != null) req.backtestRatio = backtestRatio;
  const tuneRatio = parseOptionalNumber(form.tuneRatio);
  if (tuneRatio != null) req.tuneRatio = tuneRatio;
  const penaltyMaxDrawdown = parseOptionalNumber(form.penaltyMaxDrawdown);
  if (penaltyMaxDrawdown != null) req.penaltyMaxDrawdown = penaltyMaxDrawdown;
  const penaltyTurnover = parseOptionalNumber(form.penaltyTurnover);
  if (penaltyTurnover != null) req.penaltyTurnover = penaltyTurnover;

  const normalizations = parseOptionalString(form.normalizations);
  if (normalizations) req.normalizations = normalizations;
  const epochsMin = parseOptionalInt(form.epochsMin);
  if (epochsMin != null) req.epochsMin = epochsMin;
  const epochsMax = parseOptionalInt(form.epochsMax);
  if (epochsMax != null) req.epochsMax = epochsMax;
  const hiddenSizeMin = parseOptionalInt(form.hiddenSizeMin);
  if (hiddenSizeMin != null) req.hiddenSizeMin = hiddenSizeMin;
  const hiddenSizeMax = parseOptionalInt(form.hiddenSizeMax);
  if (hiddenSizeMax != null) req.hiddenSizeMax = hiddenSizeMax;
  const lrMin = parseOptionalNumber(form.lrMin);
  if (lrMin != null) req.lrMin = lrMin;
  const lrMax = parseOptionalNumber(form.lrMax);
  if (lrMax != null) req.lrMax = lrMax;
  const patienceMax = parseOptionalInt(form.patienceMax);
  if (patienceMax != null) req.patienceMax = patienceMax;
  const gradClipMin = parseOptionalNumber(form.gradClipMin);
  if (gradClipMin != null) req.gradClipMin = gradClipMin;
  const gradClipMax = parseOptionalNumber(form.gradClipMax);
  if (gradClipMax != null) req.gradClipMax = gradClipMax;
  const pDisableGradClip = parseOptionalNumber(form.pDisableGradClip);
  if (pDisableGradClip != null) req.pDisableGradClip = pDisableGradClip;

  const slippageMax = parseOptionalNumber(form.slippageMax);
  if (slippageMax != null) req.slippageMax = slippageMax;
  const spreadMax = parseOptionalNumber(form.spreadMax);
  if (spreadMax != null) req.spreadMax = spreadMax;

  const minRoundTrips = parseOptionalInt(form.minRoundTrips);
  if (minRoundTrips != null) req.minRoundTrips = minRoundTrips;
  const minWinRate = parseOptionalNumber(form.minWinRate);
  if (minWinRate != null) req.minWinRate = minWinRate;
  const minSharpe = parseOptionalNumber(form.minSharpe);
  if (minSharpe != null) req.minSharpe = minSharpe;
  const minAnnualizedReturn = parseOptionalNumber(form.minAnnualizedReturn);
  if (minAnnualizedReturn != null) req.minAnnualizedReturn = minAnnualizedReturn;
  const minCalmar = parseOptionalNumber(form.minCalmar);
  if (minCalmar != null) req.minCalmar = minCalmar;
  const minProfitFactor = parseOptionalNumber(form.minProfitFactor);
  if (minProfitFactor != null) req.minProfitFactor = minProfitFactor;
  const maxTurnover = parseOptionalNumber(form.maxTurnover);
  if (maxTurnover != null) req.maxTurnover = maxTurnover;
  const minExposure = parseOptionalNumber(form.minExposure);
  if (minExposure != null) req.minExposure = minExposure;
  const minWalkForwardSharpeMean = parseOptionalNumber(form.minWalkForwardSharpeMean);
  if (minWalkForwardSharpeMean != null) req.minWalkForwardSharpeMean = minWalkForwardSharpeMean;
  const maxWalkForwardSharpeStd = parseOptionalNumber(form.maxWalkForwardSharpeStd);
  if (maxWalkForwardSharpeStd != null) req.maxWalkForwardSharpeStd = maxWalkForwardSharpeStd;

  const walkForwardFoldsMin = parseOptionalInt(form.walkForwardFoldsMin);
  if (walkForwardFoldsMin != null) req.walkForwardFoldsMin = walkForwardFoldsMin;
  const walkForwardFoldsMax = parseOptionalInt(form.walkForwardFoldsMax);
  if (walkForwardFoldsMax != null) req.walkForwardFoldsMax = walkForwardFoldsMax;

  const minHoldBarsMin = parseOptionalInt(form.minHoldBarsMin);
  if (minHoldBarsMin != null) req.minHoldBarsMin = minHoldBarsMin;
  const minHoldBarsMax = parseOptionalInt(form.minHoldBarsMax);
  if (minHoldBarsMax != null) req.minHoldBarsMax = minHoldBarsMax;
  const cooldownBarsMin = parseOptionalInt(form.cooldownBarsMin);
  if (cooldownBarsMin != null) req.cooldownBarsMin = cooldownBarsMin;
  const cooldownBarsMax = parseOptionalInt(form.cooldownBarsMax);
  if (cooldownBarsMax != null) req.cooldownBarsMax = cooldownBarsMax;
  const maxHoldBarsMin = parseOptionalInt(form.maxHoldBarsMin);
  if (maxHoldBarsMin != null) req.maxHoldBarsMin = maxHoldBarsMin;
  const maxHoldBarsMax = parseOptionalInt(form.maxHoldBarsMax);
  if (maxHoldBarsMax != null) req.maxHoldBarsMax = maxHoldBarsMax;

  const minEdgeMin = parseOptionalNumber(form.minEdgeMin);
  if (minEdgeMin != null) req.minEdgeMin = minEdgeMin;
  const minEdgeMax = parseOptionalNumber(form.minEdgeMax);
  if (minEdgeMax != null) req.minEdgeMax = minEdgeMax;
  const minSignalToNoiseMin = parseOptionalNumber(form.minSignalToNoiseMin);
  if (minSignalToNoiseMin != null) req.minSignalToNoiseMin = minSignalToNoiseMin;
  const minSignalToNoiseMax = parseOptionalNumber(form.minSignalToNoiseMax);
  if (minSignalToNoiseMax != null) req.minSignalToNoiseMax = minSignalToNoiseMax;
  const edgeBufferMin = parseOptionalNumber(form.edgeBufferMin);
  if (edgeBufferMin != null) req.edgeBufferMin = edgeBufferMin;
  const edgeBufferMax = parseOptionalNumber(form.edgeBufferMax);
  if (edgeBufferMax != null) req.edgeBufferMax = edgeBufferMax;
  const trendLookbackMin = parseOptionalInt(form.trendLookbackMin);
  if (trendLookbackMin != null) req.trendLookbackMin = trendLookbackMin;
  const trendLookbackMax = parseOptionalInt(form.trendLookbackMax);
  if (trendLookbackMax != null) req.trendLookbackMax = trendLookbackMax;
  const pCostAwareEdge = parseOptionalNumber(form.pCostAwareEdge);
  if (pCostAwareEdge != null) req.pCostAwareEdge = pCostAwareEdge;

  const stopMin = parseOptionalNumber(form.stopMin);
  if (stopMin != null) req.stopMin = stopMin;
  const stopMax = parseOptionalNumber(form.stopMax);
  if (stopMax != null) req.stopMax = stopMax;
  const tpMin = parseOptionalNumber(form.tpMin);
  if (tpMin != null) req.tpMin = tpMin;
  const tpMax = parseOptionalNumber(form.tpMax);
  if (tpMax != null) req.tpMax = tpMax;
  const trailMin = parseOptionalNumber(form.trailMin);
  if (trailMin != null) req.trailMin = trailMin;
  const trailMax = parseOptionalNumber(form.trailMax);
  if (trailMax != null) req.trailMax = trailMax;

  const methodWeightBlend = parseOptionalNumber(form.methodWeightBlend);
  if (methodWeightBlend != null) req.methodWeightBlend = methodWeightBlend;
  const blendWeightMin = parseOptionalNumber(form.blendWeightMin);
  if (blendWeightMin != null) req.blendWeightMin = blendWeightMin;
  const blendWeightMax = parseOptionalNumber(form.blendWeightMax);
  if (blendWeightMax != null) req.blendWeightMax = blendWeightMax;

  if (form.disableLstmPersistence) req.disableLstmPersistence = true;
  if (form.noSweepThreshold) req.noSweepThreshold = true;

  if (extras) {
    Object.assign(req, extras);
  }

  return req;
}

const CUSTOM_SYMBOL_VALUE = "__custom__";
const TOP_COMBOS_POLL_MS = 30_000;
const TOP_COMBOS_DISPLAY_DEFAULT = 5;
const TOP_COMBOS_DISPLAY_MIN = 1;
const TOP_COMBOS_BOT_TARGET = 5;
const MIN_LOOKBACK_BARS = 2;
const MIN_BACKTEST_BARS = 2;
const MIN_BACKTEST_RATIO = 0.01;
const MAX_BACKTEST_RATIO = 0.99;

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

type DecisionCheckStatus = "ok" | "warn" | "bad" | "skip";

type DecisionCheck = {
  id: string;
  label: string;
  status: DecisionCheckStatus;
  detail: string;
};

type DecisionSummary = {
  isHold: boolean;
  reason: string | null;
  checks: DecisionCheck[];
};

const DIRECTION_HOLD_REASONS = new Set([
  "DIRECTIONS_DISAGREE",
  "BOTH_NEUTRAL",
  "KALMAN_NEUTRAL",
  "LSTM_NEUTRAL",
  "BLEND_NEUTRAL",
  "ROUTER_NEUTRAL",
]);

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatSignalDirection(value: LatestSignal["chosenDirection"]): string {
  return value ?? "NEUTRAL";
}

function parseActionReason(action: string): string | null {
  const match = /\(([^)]+)\)/.exec(action);
  if (!match) return null;
  const reason = match[1]?.trim() ?? "";
  return reason ? reason : null;
}

function normalizeHoldReason(reason: string | null): string | null {
  if (!reason) return null;
  const clean = reason.trim();
  if (!clean) return null;
  return clean.toUpperCase().replace(/\s+/g, "_");
}

function decisionDotClass(status: DecisionCheckStatus): string {
  switch (status) {
    case "ok":
      return "dot dotOk";
    case "bad":
      return "dot dotBad";
    case "warn":
      return "dot dotWarn";
    default:
      return "dot";
  }
}

function decisionBadgeClass(status: DecisionCheckStatus): string {
  switch (status) {
    case "ok":
      return "badge badgeOk";
    case "bad":
      return "badge badgeBad";
    case "warn":
      return "badge badgeWarn";
    default:
      return "badge";
  }
}

function decisionStatusLabel(status: DecisionCheckStatus): string {
  switch (status) {
    case "ok":
      return "pass";
    case "bad":
      return "block";
    case "warn":
      return "needs data";
    default:
      return "off";
  }
}

const SECONDS_PER_YEAR = 365 * 24 * 60 * 60;

function inferPeriodsPerYear(platform: Platform, interval: string): number | null {
  const seconds = platformIntervalSeconds(platform, interval);
  if (!seconds || seconds <= 0) return null;
  const out = SECONDS_PER_YEAR / seconds;
  return Number.isFinite(out) && out > 0 ? out : null;
}

type SplitStats = {
  trainEndRaw: number;
  backtestBars: number;
  tuneBars: number;
  fitBars: number;
  trainOk: boolean;
  backtestOk: boolean;
  tuneOk: boolean;
  fitOk: boolean;
};

type TuneRatioBounds = {
  trainEndRaw: number;
  minTrainBars: number;
  minTuneBars: number;
  maxTuneBars: number;
  minRatio: number;
  maxRatio: number;
};

const RATIO_ROUND_DIGITS = 3;
const RATIO_ROUND_FACTOR = 10 ** RATIO_ROUND_DIGITS;

function splitStats(
  bars: number,
  backtestRatio: number,
  lookbackBars: number,
  tuneRatio: number,
  tuningEnabled: boolean,
): SplitStats {
  const ratio = clamp(backtestRatio, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
  const trainEndRaw = Math.floor(bars * (1 - ratio) + 1e-9);
  const backtestBars = Math.max(0, bars - trainEndRaw);
  const minTrainBars = lookbackBars + 1;
  const trainOk = trainEndRaw >= minTrainBars;
  const backtestOk = backtestBars >= MIN_BACKTEST_BARS;
  let tuneBars = 0;
  let fitBars = trainEndRaw;
  let tuneOk = true;
  let fitOk = true;
  if (tuningEnabled) {
    const tuneRatioSafe = clamp(tuneRatio, 0, 0.99);
    tuneBars = Math.max(0, Math.min(trainEndRaw, Math.floor(trainEndRaw * tuneRatioSafe)));
    fitBars = Math.max(0, trainEndRaw - tuneBars);
    tuneOk = tuneBars >= 2;
    fitOk = fitBars >= minTrainBars;
  }
  return { trainEndRaw, backtestBars, tuneBars, fitBars, trainOk, backtestOk, tuneOk, fitOk };
}

function roundRatioDown(value: number): number {
  return Math.floor(value * RATIO_ROUND_FACTOR) / RATIO_ROUND_FACTOR;
}

function roundRatioUp(value: number): number {
  return Math.ceil(value * RATIO_ROUND_FACTOR) / RATIO_ROUND_FACTOR;
}

function tuneRatioBounds(bars: number, backtestRatio: number, lookbackBars: number): TuneRatioBounds | null {
  if (!Number.isFinite(bars) || bars <= 0) return null;
  if (!Number.isFinite(lookbackBars) || lookbackBars < MIN_LOOKBACK_BARS) return null;
  const ratio = clamp(backtestRatio, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
  const trainEndRaw = Math.floor(bars * (1 - ratio) + 1e-9);
  const minTrainBars = lookbackBars + 1;
  if (trainEndRaw < minTrainBars) return null;
  const minTuneBars = 2;
  const maxTuneBars = trainEndRaw - minTrainBars;
  const minRatio = minTuneBars / trainEndRaw;
  const maxRatio = maxTuneBars / trainEndRaw;
  return {
    trainEndRaw,
    minTrainBars,
    minTuneBars,
    maxTuneBars,
    minRatio: clamp(minRatio, 0, 0.99),
    maxRatio: clamp(maxRatio, 0, 0.99),
  };
}

function maxBarsForPlatform(platform: Platform, method: Method, apiLimits: ComputeLimits | null): number {
  let maxBars = platform === "binance" ? 1000 : Number.POSITIVE_INFINITY;
  if (method !== "10" && apiLimits) {
    const maxBarsRaw = Math.trunc(apiLimits.maxBarsLstm);
    if (Number.isFinite(maxBarsRaw) && maxBarsRaw > 0) {
      maxBars = Math.min(maxBars, maxBarsRaw);
    }
  }
  return maxBars;
}

function maxLookbackForSplit(bars: number, backtestRatio: number, tuneRatio: number, tuningEnabled: boolean): number | null {
  if (!Number.isFinite(bars) || bars <= 0) return null;
  const ratio = clamp(backtestRatio, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
  const trainEndRaw = Math.floor(bars * (1 - ratio) + 1e-9);
  if (trainEndRaw <= 0) return null;
  if (!tuningEnabled) return trainEndRaw - 1;
  const tuneRatioSafe = clamp(tuneRatio, 0, 0.99);
  const tuneBars = Math.max(0, Math.min(trainEndRaw, Math.floor(trainEndRaw * tuneRatioSafe)));
  const fitBars = Math.max(0, trainEndRaw - tuneBars);
  return fitBars - 1;
}

function minTrainEndForTune(minTrainBars: number, tuneRatio: number, tuningEnabled: boolean, maxTrainEnd: number): number {
  if (!tuningEnabled) return minTrainBars;
  const ratio = clamp(tuneRatio, 0, 0.99);
  if (ratio <= 0) return minTrainBars;
  for (let trainEnd = minTrainBars; trainEnd <= maxTrainEnd; trainEnd += 1) {
    const tuneBars = Math.floor(trainEnd * ratio);
    const fitBars = trainEnd - tuneBars;
    if (tuneBars >= 2 && fitBars >= minTrainBars) return trainEnd;
  }
  return minTrainBars;
}

function ratioForTrainEnd(bars: number, trainEnd: number): number {
  const raw = 1 - (trainEnd + 0.5) / Math.max(1, bars);
  return clamp(raw, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
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
  const comboSymbolRaw = combo.params.binanceSymbol?.trim() ?? "";
  const normalizedComboSymbol = comboSymbolRaw ? normalizeComboSymbol(comboSymbolRaw, nextPlatform) : "";
  const comboSymbol =
    normalizedComboSymbol && symbolFormatPattern(nextPlatform).test(normalizedComboSymbol) ? normalizedComboSymbol : "";
  const prevSymbol = prev.binanceSymbol.trim().toUpperCase();
  const prevSymbolValid = symbolFormatPattern(nextPlatform).test(prevSymbol);
  const fallbackSymbol = PLATFORM_DEFAULT_SYMBOL[nextPlatform] ?? prev.binanceSymbol;
  const symbol = comboSymbol || (prevSymbolValid ? prevSymbol : fallbackSymbol);
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
  const rebalanceBars = clampOptionalInt(combo.params.rebalanceBars ?? prev.rebalanceBars, 0, 1_000_000);
  const rebalanceThreshold = Math.max(
    0,
    coerceNumber(combo.params.rebalanceThreshold ?? prev.rebalanceThreshold, prev.rebalanceThreshold),
  );
  const rebalanceGlobal = combo.params.rebalanceGlobal ?? prev.rebalanceGlobal;
  const rebalanceResetOnSignal = combo.params.rebalanceResetOnSignal ?? prev.rebalanceResetOnSignal;
  const fundingRate = coerceNumber(combo.params.fundingRate ?? prev.fundingRate, prev.fundingRate);
  const fundingBySide = combo.params.fundingBySide ?? prev.fundingBySide;
  const fundingOnOpen = combo.params.fundingOnOpen ?? prev.fundingOnOpen;
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
  const comboOrderQuantity = coerceNumber(combo.params.orderQuantity, 0);
  const comboOrderQuote = coerceNumber(combo.params.orderQuote, 0);
  const comboOrderQuoteFraction = clampOptionalRange(combo.params.orderQuoteFraction, 0, 1);
  const comboMaxOrderQuote = Math.max(0, coerceNumber(combo.params.maxOrderQuote, 0));
  const hasComboSizing =
    combo.params.orderQuantity != null ||
    combo.params.orderQuote != null ||
    combo.params.orderQuoteFraction != null ||
    combo.params.maxOrderQuote != null;

  let orderQuantity = prev.orderQuantity;
  let orderQuote = prev.orderQuote;
  let orderQuoteFraction = prev.orderQuoteFraction;
  let maxOrderQuote = prev.maxOrderQuote;

  if (hasComboSizing) {
    if (comboOrderQuantity > 0) {
      orderQuantity = comboOrderQuantity;
      orderQuote = 0;
      orderQuoteFraction = 0;
      maxOrderQuote = 0;
    } else if (comboOrderQuote > 0) {
      orderQuote = comboOrderQuote;
      orderQuantity = 0;
      orderQuoteFraction = 0;
      maxOrderQuote = 0;
    } else if (comboOrderQuoteFraction > 0) {
      orderQuoteFraction = comboOrderQuoteFraction;
      orderQuantity = 0;
      orderQuote = 0;
      maxOrderQuote = comboMaxOrderQuote > 0 ? comboMaxOrderQuote : 0;
    } else {
      orderQuantity = 0;
      orderQuote = 0;
      orderQuoteFraction = 0;
      maxOrderQuote = 0;
    }
  }

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
    rebalanceBars,
    rebalanceThreshold,
    rebalanceGlobal,
    rebalanceResetOnSignal,
    fundingRate,
    fundingBySide,
    fundingOnOpen,
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
    orderQuantity,
    orderQuote,
    orderQuoteFraction,
    maxOrderQuote,
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

function comboAnnualizedEquity(combo: OptimizationCombo): number | null {
  const annReturn = combo.metrics?.annualizedReturn;
  if (typeof annReturn !== "number" || !Number.isFinite(annReturn)) return null;
  const annEq = annReturn + 1;
  return Number.isFinite(annEq) ? annEq : null;
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
    sigNumber(form.orderQuantity),
    sigNumber(form.orderQuote),
    sigNumber(form.orderQuoteFraction),
    sigNumber(form.maxOrderQuote),
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
    sigNumber(form.rebalanceBars),
    sigNumber(form.rebalanceThreshold),
    sigBool(form.rebalanceGlobal),
    sigBool(form.rebalanceResetOnSignal),
    sigNumber(form.fundingRate),
    sigBool(form.fundingBySide),
    sigBool(form.fundingOnOpen),
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
  const binanceKeyLocalReady = Boolean(binanceApiKey.trim() && binanceApiSecret.trim());
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
  const panelPrefsInit = readJson<PanelPrefs>(STORAGE_PANEL_PREFS_KEY);
  const [panelPrefs, setPanelPrefs] = useState<PanelPrefs>(() => panelPrefsInit ?? {});
  const configPanelOrderInit = readJson<ConfigPanelId[]>(STORAGE_CONFIG_PANEL_ORDER_KEY);
  const [configPanelOrder, setConfigPanelOrder] = useState<ConfigPanelId[]>(() =>
    normalizeConfigPanelOrder(configPanelOrderInit),
  );
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
    lastFetchedAtMs: null,
  });
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

  const [dataLog, setDataLog] = useState<Array<{ timestamp: number; label: string; data: unknown }>>([]);
  const [dataLogExpanded, setDataLogExpanded] = useState(false);
  const [dataLogIndexArrays, setDataLogIndexArrays] = useState(true);
  const [dataLogFilterText, setDataLogFilterText] = useState("");
  const [dataLogAutoScroll, setDataLogAutoScroll] = useState(true);
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
  const comboMinEquity = useMemo(() => {
    const trimmed = comboMinEquityInput.trim();
    if (!trimmed) return null;
    const parsed = numFromInput(trimmed, Number.NaN);
    return Number.isFinite(parsed) ? parsed : null;
  }, [comboMinEquityInput]);
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
    return null;
  }, [
    optimizerRunExtras.error,
    optimizerRunExtras.value,
    optimizerRunForm.backtestRatio,
    optimizerRunForm.dataPath,
    optimizerRunForm.highColumn,
    optimizerRunForm.lowColumn,
    optimizerRunForm.source,
    optimizerRunForm.symbol,
    optimizerRunForm.tuneRatio,
  ]);
  const optimizerRunRecordJson = useMemo(() => {
    if (!optimizerRunUi.response) return null;
    try {
      return JSON.stringify(optimizerRunUi.response.lastRecord, null, 2);
    } catch {
      return String(optimizerRunUi.response.lastRecord ?? "");
    }
  }, [optimizerRunUi.response]);
  const topCombosFiltered = useMemo(() => {
    if (comboMinEquity == null) return topCombosAll;
    return topCombosAll.filter((combo) => combo.finalEquity > comboMinEquity);
  }, [comboMinEquity, topCombosAll]);
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
  const topCombosDisplayMax = useMemo(() => {
    const count = topCombosFiltered.length;
    if (count && count > 0) return Math.max(TOP_COMBOS_DISPLAY_MIN, count);
    return Math.max(TOP_COMBOS_DISPLAY_MIN, TOP_COMBOS_DISPLAY_DEFAULT);
  }, [topCombosFiltered.length]);
  const [autoAppliedCombo, setAutoAppliedCombo] = useState<{ id: number; atMs: number } | null>(null);
  const autoAppliedComboRef = useRef<{ id: number | null; atMs: number | null }>({ id: null, atMs: null });
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

  const setPanelOpen = useCallback((panelId: string, open: boolean) => {
    setPanelPrefs((prev) => (prev[panelId] === open ? prev : { ...prev, [panelId]: open }));
  }, []);

  const setPanelsOpen = useCallback((panelIds: readonly string[], open: boolean) => {
    setPanelPrefs((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const panelId of panelIds) {
        if (next[panelId] !== open) {
          next[panelId] = open;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, []);

  const isPanelOpen = useCallback(
    (panelId: string, defaultOpen: boolean) => {
      const stored = panelPrefs[panelId];
      return typeof stored === "boolean" ? stored : defaultOpen;
    },
    [panelPrefs],
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
      setDataLog((logs) => [...logs, { timestamp: Date.now(), label: "Optimizer Run", data: out }].slice(-100));
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
      if (showErrorToast) showToast("Optimizer run failed");
    } finally {
      if (requestId === optimizerRunRequestSeqRef.current) {
        optimizerRunAbortRef.current = null;
      }
    }
  }, [
    apiBase,
    applyRateLimit,
    authHeaders,
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
    const fundingRate = Number.isFinite(form.fundingRate) ? form.fundingRate : 0;
    const rebalanceGlobal = form.rebalanceGlobal;
    const rebalanceResetOnSignal = form.rebalanceResetOnSignal;
    const fundingBySide = form.fundingBySide;
    const fundingOnOpen = form.fundingOnOpen;
    const tuneStressVolMult = form.tuneStressVolMult <= 0 ? 1 : form.tuneStressVolMult;
    const minPositionSize = clamp(form.minPositionSize, 0, 1);
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
  }, [form]);

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
      botStatusOps.ops.length >= BOT_STATUS_OPS_LIMIT
        ? `Showing latest ${BOT_STATUS_OPS_LIMIT} ops.`
        : `Showing ${botStatusOps.ops.length} ops.`;
    return `${rangeNote} ${windowNote} ${limitNote}`;
  }, [botStatusOps.enabled, botStatusOps.ops.length, botStatusOpsWindow, botStatusRange]);

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
    const el = document.getElementById(id);
    if (!el) return;
    const openedPanels = openPanelAncestors(el);

    const runScroll = () => {
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
      window.setTimeout(runScroll, 0);
    } else {
      runScroll();
    }
  }, [openPanelAncestors]);

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

    const lstmEnabled = params.method !== "10";
    let maxBars = platformValue === "binance" ? 1000 : Number.POSITIVE_INFINITY;
    const apiBarsLimit = apiComputeLimits?.maxBarsLstm;
    if (lstmEnabled && typeof apiBarsLimit === "number" && Number.isFinite(apiBarsLimit) && apiBarsLimit > 0) {
      maxBars = Math.min(maxBars, Math.trunc(apiBarsLimit));
    }
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
        const requestHeaders = { ...(authHeaders ?? {}), Accept: "text/event-stream" };
        const res = await fetch(`${apiBase}/binance/listenKey/stream`, {
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
            if (typeof payload?.atMs === "number") {
              setListenKeyUi((s) => ({ ...s, keepAliveAtMs: payload.atMs, keepAliveError: null }));
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
    [apiBase, authHeaders, showToast],
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
      void openListenKeyStream();
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
    openListenKeyStream,
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
    [apiBase, authHeaders],
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
      botActiveSymbolSet,
      botSelectedSymbol,
      botSymbolsInput,
      form.binanceSymbol,
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

  const fetchBotStatusOps = useCallback(
    async (opts?: RunOptions) => {
      if (apiOk !== "ok") return;
      if (!opts?.silent) setBotStatusOps((s) => ({ ...s, loading: true, error: null }));
      try {
        const out = await ops(apiBase, { kind: "bot.status", limit: BOT_STATUS_OPS_LIMIT }, { headers: authHeaders, timeoutMs: 30_000 });
        setBotStatusOps({
          loading: false,
          error: null,
          enabled: out.enabled,
          hint: out.hint ?? null,
          ops: Array.isArray(out.ops) ? out.ops : [],
          lastFetchedAtMs: Date.now(),
        });
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setBotStatusOps((s) => ({ ...s, loading: false, error: msg }));
      }
    },
    [apiBase, apiOk, authHeaders],
  );

  useEffect(() => {
    if (apiOk !== "ok") return;
    void fetchBotStatusOps({ silent: true });
    const t = window.setInterval(() => void fetchBotStatusOps({ silent: true }), 60_000);
    return () => window.clearInterval(t);
  }, [apiOk, fetchBotStatusOps]);

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
        binancePrivateKeysMissing ? 'Binance API keys missing. Add keys or click "Check keys".' : null,
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
      form.market,
      binancePrivateKeysMissing,
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
        binancePrivateKeysMissing ? 'Binance API keys missing. Add keys or click "Check keys".' : null,
        binancePositionsBarsError,
      ),
    [binancePositionsBarsError, binancePrivateKeysMissing, form.market, isBinancePlatform],
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

  const fetchBinancePositions = useCallback(async () => {
    setBinancePositionsUi((s) => ({ ...s, loading: true, error: null }));
    if (binancePositionsInputError) {
      setBinancePositionsUi((s) => ({ ...s, loading: false, error: binancePositionsInputError }));
      return;
    }
    try {
      const params: ApiBinancePositionsRequest = {
        market: form.market,
        binanceTestnet: form.binanceTestnet,
        interval: form.interval.trim(),
        limit: binancePositionsLimitSafe,
      };
      const out = await binancePositions(apiBase, withBinanceKeys(params), { headers: authHeaders, timeoutMs: 30_000 });
      setBinancePositionsUi({ loading: false, error: null, response: out });
    } catch (e) {
      if (isAbortError(e)) return;
      const msg = e instanceof Error ? e.message : String(e);
      setBinancePositionsUi((s) => ({ ...s, loading: false, error: msg }));
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
      const payload = await optimizerCombos(apiBase, { headers: authHeaders });
      return { payload, source: "api", fallbackReason: null };
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
              rebalanceGlobal,
              fundingRate,
              fundingBySide,
              rebalanceResetOnSignal,
              fundingOnOpen,
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
  const topCombo = topCombosAll.length > 0 ? topCombosAll[0] : null;
  const topComboDisplay = topCombosOrdered.length > 0 ? topCombosOrdered[0] : null;
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
        const maxBarsRaw = Math.trunc(apiComputeLimits.maxBarsLstm);
        if (Number.isFinite(maxBarsRaw) && maxBarsRaw > 0) {
          const maxBars = clamp(maxBarsRaw, MIN_LOOKBACK_BARS, 1000);
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
    hiddenSizeExceedsApi,
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
  const topComboAutoStartBlockedReason = useMemo(
    () =>
      firstReason(
        rateLimitReason,
        apiBlockedReason ?? apiStatusIssue,
        !isBinancePlatform ? "Live bot is supported on Binance only." : null,
        form.positioning === "long-short" && form.market !== "futures"
          ? "Live bot long/short requires the Futures market."
          : null,
        botTradeKeysIssue,
      ),
    [apiBlockedReason, apiStatusIssue, botTradeKeysIssue, form.market, form.positioning, isBinancePlatform, rateLimitReason],
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
  const jumpTargets = [
    { id: "section-api", label: "API" },
    { id: "section-market", label: "Market" },
    { id: "section-lookback", label: "Lookback" },
    { id: "section-thresholds", label: "Thresholds" },
    { id: "section-risk", label: "Risk" },
    { id: "section-optimizer-run", label: "Optimizer run" },
    { id: "section-optimization", label: "Optimization" },
    { id: "section-livebot", label: "Live bot" },
    { id: "section-trade", label: "Trade" },
    { id: "section-positions", label: "Positions" },
  ];
  const configPanelOrderIndex = useMemo(() => {
    const index = {} as Record<ConfigPanelId, number>;
    configPanelOrder.forEach((panelId, idx) => {
      index[panelId] = idx;
    });
    return index;
  }, [configPanelOrder]);
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
  const dockLayoutClass = `dockLayout${combosOpen ? "" : " dockLayoutCompactBottom"}${configOpen ? "" : " dockLayoutCompactTop"}`;

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
            </summary>
          </details>
          <CollapsibleCard
            panelId="panel-config"
            open={configOpen}
            onToggle={handlePanelToggle("panel-config")}
            maximized={isPanelMaximized("panel-config")}
            onToggleMaximize={() => togglePanelMaximize("panel-config")}
            title="Configuration"
            subtitle="Safe defaults, minimal knobs, and clear outputs."
            className="configCard"
          >
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
                      <button className="btnSmall" type="button" onClick={handlePrimaryIssueFix}>
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
              <div className="pillRow jumpRow">
                <span className="jumpLabel">Panels</span>
                <button className="btnSmall" type="button" onClick={() => setPanelsOpen(CONFIG_SECTION_IDS, true)}>
                  Expand all
                </button>
                <button className="btnSmall" type="button" onClick={() => setPanelsOpen(CONFIG_SECTION_IDS, false)}>
                  Collapse all
                </button>
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
            <div className="configPanels">
              <ConfigPanel
                panelId="config-access"
                title="Access & Profiles"
                subtitle="API health, keys, and saved setups."
                order={configPanelOrderIndex["config-access"]}
                open={isPanelOpen("config-access", true)}
                onToggle={handlePanelToggle("config-access")}
                maximized={isPanelMaximized("config-access")}
                onToggleMaximize={() => togglePanelMaximize("config-access")}
                style={configPanelStyle("config-access")}
                {...configPanelHandlers}
              >
            <div className="row" style={{ gridTemplateColumns: "1fr" }} id="section-api">
              <div className="field" id="platformKeys">
                <div className="label">API</div>
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
                  <div className="label">Connection</div>
                  <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)" }}>
                    {apiOk === "down"
                      ? showLocalStartHelp
                        ? `Backend unreachable.\n\nStart it with:\ncd haskell && cabal run -v0 trader-hs -- --serve --port ${API_PORT}`
                        : "Backend unreachable.\n\nConfigure apiBaseUrl in trader-config.js (CORS required for cross-origin) or configure CloudFront to forward `/api/*` to your API origin."
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
                <div className="label">{platformKeyLabel}</div>
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
                    aria-label={platformKeyMode === "coinbase" ? "Coinbase API key" : platformKeyMode === "binance" ? "Binance API key" : "API key"}
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
                    aria-label={
                      platformKeyMode === "coinbase" ? "Coinbase API secret" : platformKeyMode === "binance" ? "Binance API secret" : "API secret"
                    }
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
                      aria-label="Coinbase API passphrase"
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
                <div className="label">Profiles</div>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr", alignItems: "center" }}>
                  <select className="select" value={profileSelected} onChange={(e) => setProfileSelected(e.target.value)} aria-label="Saved profiles">
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
                    aria-label="New profile name"
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
              </ConfigPanel>
              
              <ConfigPanel
                panelId="config-market"
                title="Market & Lookback"
                subtitle="Platform, symbol, interval, and window sizing."
                order={configPanelOrderIndex["config-market"]}
                open={isPanelOpen("config-market", true)}
                onToggle={handlePanelToggle("config-market")}
                maximized={isPanelMaximized("config-market")}
                onToggleMaximize={() => togglePanelMaximize("config-market")}
                style={configPanelStyle("config-market")}
                {...configPanelHandlers}
              >
          <CollapsibleSection
            panelId="section-market"
            open={isPanelOpen("section-market", true)}
            onToggle={handlePanelToggle("section-market")}
            title="Market"
            meta="Pair, market type, interval, bars."
          >
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
                className={missingSymbol || Boolean(symbolFormatError) ? "select selectError" : "select"}
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
                  className={missingSymbol || Boolean(symbolFormatError) ? "input inputError" : "input"}
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
              <div className="hint" style={missingSymbol || symbolFormatError ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
                {missingSymbol
                  ? "Required."
                  : symbolFormatError
                    ? symbolFormatError
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

          </CollapsibleSection>

          <CollapsibleSection
            panelId="section-lookback"
            open={isPanelOpen("section-lookback", true)}
            onToggle={handlePanelToggle("section-lookback")}
            title="Lookback"
            meta="Window length and bar overrides."
          >
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

          </CollapsibleSection>
              </ConfigPanel>
              <ConfigPanel
                panelId="config-strategy"
                title="Strategy & Risk"
                subtitle="Thresholds, exits, sizing, and safeguards."
                order={configPanelOrderIndex["config-strategy"]}
                open={isPanelOpen("config-strategy", true)}
                onToggle={handlePanelToggle("config-strategy")}
                maximized={isPanelMaximized("config-strategy")}
                onToggleMaximize={() => togglePanelMaximize("config-strategy")}
                style={configPanelStyle("config-strategy")}
                {...configPanelHandlers}
              >
          <CollapsibleSection
            panelId="section-thresholds"
            open={isPanelOpen("section-thresholds", true)}
            onToggle={handlePanelToggle("section-thresholds")}
            title="Thresholds"
            meta="Method, positioning, entry/exit gates."
          >
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
              <div className="field">
                <div className="labelRow">
                  <label className="label" htmlFor="method">
                    Method
                  </label>
                  <InfoPopover label="Method details">
                    <InfoList items={COMPLEX_TIPS.method} />
                  </InfoPopover>
                </div>
                <select
                  id="method"
                  className="select"
                  value={form.method}
                  onChange={(e) => {
                    const nextMethod = e.target.value as Method;
                    markManualOverrides(["method"]);
                    setForm((f) => ({
                      ...f,
                      method: nextMethod,
                      ...(nextMethod === "router" ? { optimizeOperations: false, sweepThreshold: false } : {}),
                    }));
                  }}
                >
                  <option value="11">11 — Both (agreement gated)</option>
                  <option value="blend">blend — Weighted average</option>
                  <option value="router">router — Adaptive router</option>
                  <option value="10">10 — Kalman only</option>
                  <option value="01">01 — LSTM only</option>
                </select>
                <div className="hint">
                  “11” only trades when both models agree on direction (up/down) outside the open threshold. “blend” averages the two predictions. “router” picks the best recent model.
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
                <div className="labelRow">
                  <label className="label" htmlFor="openThreshold">
                    Open threshold (fraction)
                  </label>
                  <InfoPopover label="Threshold details">
                    <InfoList items={COMPLEX_TIPS.thresholds} />
                  </InfoPopover>
                </div>
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
                <div className="labelRow">
                  <label className="label" htmlFor="closeThreshold">
                    Close threshold (fraction)
                  </label>
                  <InfoPopover label="Threshold details">
                    <InfoList items={COMPLEX_TIPS.thresholds} />
                  </InfoPopover>
                </div>
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
                <div className="labelRow">
                  <label className="label" htmlFor="minEdge">
                    Min edge (fraction)
                  </label>
                  <InfoPopover label="Edge filters">
                    <InfoList items={COMPLEX_TIPS.edge} />
                  </InfoPopover>
                </div>
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
                <div className="labelRow">
                  <label className="label" htmlFor="edgeBuffer">
                    Edge buffer (fraction)
                  </label>
                  <InfoPopover label="Edge filters">
                    <InfoList items={COMPLEX_TIPS.edge} />
                  </InfoPopover>
                </div>
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
                <div className="labelRow">
                  <label className="label" htmlFor="minSignalToNoise">
                    Min signal/vol (x)
                  </label>
                  <InfoPopover label="Signal-to-noise filter">
                    <InfoList items={COMPLEX_TIPS.snr} />
                  </InfoPopover>
                </div>
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
                <div className="labelRow">
                  <label className="label" htmlFor="blendWeight">
                    Blend weight (Kalman)
                  </label>
                  <InfoPopover label="Blend weight">
                    <InfoList items={COMPLEX_TIPS.blend} />
                  </InfoPopover>
                </div>
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

            <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr 1fr" }}>
              <div className="field">
                <div className="labelRow">
                  <label className="label" htmlFor="routerLookback">
                    Router lookback (bars)
                  </label>
                  <InfoPopover label="Router settings">
                    <InfoList items={COMPLEX_TIPS.router} />
                  </InfoPopover>
                </div>
                <input
                  id="routerLookback"
                  className="input"
                  type="number"
                  step="1"
                  min={2}
                  value={form.routerLookback}
                  onChange={(e) => setForm((f) => ({ ...f, routerLookback: numFromInput(e.target.value, f.routerLookback) }))}
                  disabled={form.method !== "router"}
                />
                <div className="hint">Used with method=router; evaluates recent signal accuracy.</div>
              </div>
              <div className="field">
                <div className="labelRow">
                  <label className="label" htmlFor="routerMinScore">
                    Router min score
                  </label>
                  <InfoPopover label="Router settings">
                    <InfoList items={COMPLEX_TIPS.router} />
                  </InfoPopover>
                </div>
                <input
                  id="routerMinScore"
                  className="input"
                  type="number"
                  step="0.05"
                  min={0}
                  max={1}
                  value={form.routerMinScore}
                  onChange={(e) => setForm((f) => ({ ...f, routerMinScore: numFromInput(e.target.value, f.routerMinScore) }))}
                  disabled={form.method !== "router"}
                />
                <div className="hint">Accuracy × coverage threshold; below = hold.</div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr 1fr 1fr" }}>
              <div className="field">
                <div className="labelRow">
                  <label className="label" htmlFor="backtestRatio">
                    Backtest ratio
                  </label>
                  <InfoPopover label="Backtest and tune split">
                    <InfoList items={COMPLEX_TIPS.split} />
                  </InfoPopover>
                </div>
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
                <div className="labelRow">
                  <label className="label" htmlFor="tuneRatio">
                    Tune ratio
                  </label>
                  <InfoPopover label="Backtest and tune split">
                    <InfoList items={COMPLEX_TIPS.split} />
                  </InfoPopover>
                </div>
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
            <div
              className="hint"
              style={{
                marginTop: 6,
                color: splitPreview.warning ? "rgba(239, 68, 68, 0.85)" : undefined,
              }}
            >
              {splitPreview.summary}
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

          </CollapsibleSection>

          <CollapsibleSection
            panelId="section-risk"
            open={isPanelOpen("section-risk", true)}
            onToggle={handlePanelToggle("section-risk")}
            title="Risk"
            meta="Stops, pacing, sizing, and kill-switches."
          >
            <div className="row" style={{ gridTemplateColumns: "1fr" }}>
              <div className="field">
              <div className="label">Bracket exits (fractions)</div>
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
              <div className="label">Trade pacing (bars)</div>
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
              <div className="label">Sizing + filters</div>
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
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr", marginTop: 10 }}>
                  <div className="field">
                    <label className="label" htmlFor="rebalanceBars">
                      Rebalance bars
                    </label>
                    <input
                      id="rebalanceBars"
                      className="input"
                      type="number"
                      step={1}
                      min={0}
                      value={form.rebalanceBars}
                      onChange={(e) => setForm((f) => ({ ...f, rebalanceBars: numFromInput(e.target.value, f.rebalanceBars) }))}
                      placeholder="0"
                    />
                    <div className="hint">
                      {form.rebalanceBars > 0 ? `${Math.trunc(Math.max(0, form.rebalanceBars))} bars` : "0 disables"} • resize toward target
                    </div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="rebalanceThreshold">
                      Rebalance threshold
                    </label>
                    <input
                      id="rebalanceThreshold"
                      className="input"
                      type="number"
                      step="0.01"
                      min={0}
                      value={form.rebalanceThreshold}
                      onChange={(e) =>
                        setForm((f) => ({ ...f, rebalanceThreshold: numFromInput(e.target.value, f.rebalanceThreshold) }))
                      }
                      placeholder="0"
                    />
                    <div className="hint">
                      {form.rebalanceThreshold > 0 ? form.rebalanceThreshold.toFixed(2) : "0 disables"} • min abs size delta
                    </div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="fundingRate">
                      Funding rate (annualized)
                    </label>
                    <input
                      id="fundingRate"
                      className="input"
                      type="number"
                      step="0.01"
                      value={form.fundingRate}
                      onChange={(e) => setForm((f) => ({ ...f, fundingRate: numFromInput(e.target.value, f.fundingRate) }))}
                      placeholder="0"
                    />
                    <div className="hint">{form.fundingRate !== 0 ? fmtPct(form.fundingRate, 2) : "0 disables"} • backtests only</div>
                  </div>
                </div>
                <div className="pillRow" style={{ marginTop: 8 }}>
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.rebalanceGlobal}
                      onChange={(e) => setForm((f) => ({ ...f, rebalanceGlobal: e.target.checked }))}
                    />
                    Rebalance global cadence
                  </label>
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.fundingBySide}
                      onChange={(e) => setForm((f) => ({ ...f, fundingBySide: e.target.checked }))}
                    />
                    Funding by side
                  </label>
                </div>
                <div className="pillRow" style={{ marginTop: 6 }}>
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.rebalanceResetOnSignal}
                      onChange={(e) => setForm((f) => ({ ...f, rebalanceResetOnSignal: e.target.checked }))}
                    />
                    Reset rebalance on signal
                  </label>
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.fundingOnOpen}
                      onChange={(e) => setForm((f) => ({ ...f, fundingOnOpen: e.target.checked }))}
                    />
                    Funding on open bar
                  </label>
                </div>
                <div className="hint">Defaults: rebalancing anchors to entry age; funding is side-agnostic and only charged if the position survives the bar.</div>
                <div className="hint">Vol sizing scales position by target/realized volatility when vol target is set.</div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr" }}>
              <div className="field">
              <div className="label">Risk kill-switches</div>
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
                <div className="labelRow">
                  <label className="label" htmlFor="norm">
                    Normalization
                  </label>
                  <InfoPopover label="LSTM normalization">
                    <InfoList items={COMPLEX_TIPS.lstm} />
                  </InfoPopover>
                </div>
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
                <div className="labelRow">
                  <label className="label" htmlFor="epochs">
                    Epochs / Hidden size
                  </label>
                  <InfoPopover label="LSTM epochs and hidden size">
                    <InfoList items={COMPLEX_TIPS.lstm} />
                  </InfoPopover>
                </div>
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

          </CollapsibleSection>
              </ConfigPanel>
              <ConfigPanel
                panelId="config-optimization"
                title="Optimization & Runs"
                subtitle="Tune sweeps and launch optimizer runs."
                order={configPanelOrderIndex["config-optimization"]}
                open={isPanelOpen("config-optimization", true)}
                onToggle={handlePanelToggle("config-optimization")}
                maximized={isPanelMaximized("config-optimization")}
                onToggleMaximize={() => togglePanelMaximize("config-optimization")}
                style={configPanelStyle("config-optimization")}
                {...configPanelHandlers}
              >
          <CollapsibleSection
            panelId="section-optimizer-run"
            open={isPanelOpen("section-optimizer-run", true)}
            onToggle={handlePanelToggle("section-optimizer-run")}
            title="Optimizer run"
            meta="Kick off /optimizer/run with the current config or a CSV source."
          >
            <div className="row">
              <div className="field">
                <div className="label">Request</div>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
                  <div className="field">
                    <label className="label" htmlFor="optimizerRunSource">
                      Source
                    </label>
                    <select
                      id="optimizerRunSource"
                      className="select"
                      value={optimizerRunForm.source}
                      onChange={(e) => updateOptimizerRunForm({ source: e.target.value as OptimizerSource })}
                    >
                      <option value="binance">Binance</option>
                      <option value="coinbase">Coinbase</option>
                      <option value="kraken">Kraken</option>
                      <option value="poloniex">Poloniex</option>
                      <option value="csv">CSV (file path)</option>
                    </select>
                    <div className="hint">Exchange determines symbol format; CSV bypasses the API.</div>
                  </div>
                  {optimizerRunForm.source === "csv" ? (
                    <>
                      <div className="field">
                        <label className="label" htmlFor="optimizerRunDataPath">
                          CSV path
                        </label>
                        <input
                          id="optimizerRunDataPath"
                          className={optimizerRunValidationError ? "input inputError" : "input"}
                          value={optimizerRunForm.dataPath}
                          onChange={(e) => updateOptimizerRunForm({ dataPath: e.target.value })}
                          placeholder="data/my-prices.csv"
                        />
                        <div className="hint">Required for CSV source.</div>
                      </div>
                      <div className="field">
                        <label className="label" htmlFor="optimizerRunPriceColumn">
                          Price / High / Low
                        </label>
                        <input
                          id="optimizerRunPriceColumn"
                          className="input"
                          value={optimizerRunForm.priceColumn}
                          onChange={(e) => updateOptimizerRunForm({ priceColumn: e.target.value })}
                          placeholder="close"
                        />
                        <div className="row" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 6 }}>
                          <input
                            className={optimizerRunValidationError ? "input inputError" : "input"}
                            value={optimizerRunForm.highColumn}
                            onChange={(e) => updateOptimizerRunForm({ highColumn: e.target.value })}
                            placeholder="high (optional)"
                          />
                          <input
                            className={optimizerRunValidationError ? "input inputError" : "input"}
                            value={optimizerRunForm.lowColumn}
                            onChange={(e) => updateOptimizerRunForm({ lowColumn: e.target.value })}
                            placeholder="low (optional)"
                          />
                        </div>
                        <div className="hint">Provide both High/Low or leave both blank.</div>
                      </div>
                    </>
                  ) : (
                    <div className="field" style={{ gridColumn: "span 2" }}>
                      <label className="label" htmlFor="optimizerRunSymbol">
                        Symbol
                      </label>
                      <input
                        id="optimizerRunSymbol"
                        className={optimizerRunValidationError ? "input inputError" : "input"}
                        value={optimizerRunForm.symbol}
                        onChange={(e) => updateOptimizerRunForm({ symbol: e.target.value.toUpperCase() })}
                        placeholder="BTCUSDT"
                      />
                      <div className="hint">Defaults to the current symbol/platform.</div>
                    </div>
                  )}
                </div>

                <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerRunIntervals">
                        Intervals
                      </label>
                      <InfoPopover label="Equity tip: intervals">
                        <InfoList items={EQUITY_TIPS.intervals} />
                      </InfoPopover>
                    </div>
                    <input
                      id="optimizerRunIntervals"
                      className="input"
                      value={optimizerRunForm.intervals}
                      onChange={(e) => updateOptimizerRunForm({ intervals: e.target.value })}
                      placeholder="1h,4h,1d"
                    />
                    <div className="hint">Comma-separated list.</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="optimizerRunLookback">
                      Lookback window
                    </label>
                    <input
                      id="optimizerRunLookback"
                      className="input"
                      value={optimizerRunForm.lookbackWindow}
                      onChange={(e) => updateOptimizerRunForm({ lookbackWindow: e.target.value })}
                      placeholder="7d"
                    />
                    <div className="hint">Same format as main form (e.g., 7d, 30d).</div>
                  </div>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerRunTrials">
                        Trials / Timeout / Seed
                      </label>
                      <InfoPopover label="Equity tip: trials and timeout">
                        <InfoList items={EQUITY_TIPS.trials} />
                      </InfoPopover>
                    </div>
                    <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
                      <input
                        id="optimizerRunTrials"
                        className="input"
                        type="number"
                        min={1}
                        value={optimizerRunForm.trials}
                        onChange={(e) => updateOptimizerRunForm({ trials: e.target.value })}
                        placeholder="50"
                      />
                      <input
                        className="input"
                        type="number"
                        min={5}
                        value={optimizerRunForm.timeoutSec}
                        onChange={(e) => updateOptimizerRunForm({ timeoutSec: e.target.value })}
                        placeholder="60"
                      />
                      <input
                        className="input"
                        type="number"
                        value={optimizerRunForm.seed}
                        onChange={(e) => updateOptimizerRunForm({ seed: e.target.value })}
                        placeholder="42"
                      />
                    </div>
                    <div className="hint">Numbers are optional; blanks are omitted.</div>
                  </div>
                </div>

                <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerRunObjective">
                        Objective
                      </label>
                      <InfoPopover label="Equity tip: objective">
                        <InfoList items={EQUITY_TIPS.objective} />
                      </InfoPopover>
                    </div>
                    <input
                      id="optimizerRunObjective"
                      className="input"
                      value={optimizerRunForm.objective}
                      onChange={(e) => updateOptimizerRunForm({ objective: e.target.value })}
                      placeholder="annualized-equity"
                    />
                    <div className="hint">Matches backend objective names.</div>
                  </div>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerRunBacktestRatio">
                        Backtest / Tune ratio
                      </label>
                      <InfoPopover label="Equity tip: backtest and tune ratios">
                        <InfoList items={EQUITY_TIPS.ratios} />
                      </InfoPopover>
                    </div>
                    <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                      <input
                        id="optimizerRunBacktestRatio"
                        className="input"
                        type="number"
                        min={0}
                        max={0.9}
                        step="0.01"
                        value={optimizerRunForm.backtestRatio}
                        onChange={(e) => updateOptimizerRunForm({ backtestRatio: e.target.value })}
                        placeholder="0.2"
                      />
                      <input
                        id="optimizerRunTuneRatio"
                        className="input"
                        type="number"
                        min={0}
                        max={0.9}
                        step="0.01"
                        value={optimizerRunForm.tuneRatio}
                        onChange={(e) => updateOptimizerRunForm({ tuneRatio: e.target.value })}
                        placeholder="0.25"
                      />
                    </div>
                    <div className="hint">Leave blank to use defaults.</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="optimizerRunExtra">
                      Extra JSON (advanced)
                    </label>
                    <textarea
                      id="optimizerRunExtra"
                      className={optimizerRunExtras.error ? "input inputError" : "input"}
                      value={optimizerRunForm.extraJson}
                      onChange={(e) => updateOptimizerRunForm({ extraJson: e.target.value })}
                      placeholder='{"minSharpe":1.0,"minRoundTrips":5}'
                      rows={4}
                      spellCheck={false}
                    />
                    <div className="hint">{optimizerRunExtras.error ?? "Merged into the payload verbatim."}</div>
                  </div>
                </div>

                <div className="actions" style={{ marginTop: 10 }}>
                  <button
                    className="btn btnPrimary"
                    type="button"
                    disabled={optimizerRunUi.loading || Boolean(optimizerRunValidationError)}
                    onClick={runOptimizer}
                  >
                    {optimizerRunUi.loading ? "Running optimizer…" : "Run optimizer"}
                  </button>
                  <button className="btn" type="button" disabled={!optimizerRunUi.loading} onClick={cancelOptimizerRun}>
                    Cancel run
                  </button>
                  <button
                    className="btn"
                    type="button"
                    onClick={() => {
                      syncOptimizerRunForm();
                    }}
                  >
                    Sync from config
                  </button>
                  <button className="btn" type="button" onClick={applyEquityPreset}>
                    Preset: Equity focus
                  </button>
                  <InfoPopover label="Equity options" align="left">
                    <InfoList items={EQUITY_TIPS.preset} />
                  </InfoPopover>
                  <button className="btn" type="button" onClick={resetOptimizerRunForm}>
                    Reset
                  </button>
                  <span className="hint">
                    {optimizerRunValidationError
                      ? optimizerRunValidationError
                      : optimizerRunUi.loading
                        ? "Submitting to /optimizer/run…"
                        : "Uses the same auth/proxy settings as other requests."}
                  </span>
                </div>
              </div>

              <div className="field">
                <div className="label">Result</div>
                {optimizerRunUi.error ? (
                  <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)" }}>
                    {optimizerRunUi.error}
                  </pre>
                ) : null}
                {optimizerRunUi.response ? (
                  <div className="hint" style={{ marginBottom: 8 }}>
                    Last run: {optimizerRunUi.lastRunAtMs ? fmtTimeMs(optimizerRunUi.lastRunAtMs) : "just now"}
                  </div>
                ) : null}
                {optimizerRunUi.response?.stdout ? (
                  <details className="details" open>
                    <summary>Stdout</summary>
                    <pre className="code">{optimizerRunUi.response.stdout}</pre>
                  </details>
                ) : null}
                {optimizerRunUi.response?.stderr ? (
                  <details className="details">
                    <summary>Stderr</summary>
                    <pre className="code">{optimizerRunUi.response.stderr}</pre>
                  </details>
                ) : null}
                {optimizerRunRecordJson ? (
                  <details className="details" open>
                    <summary>Last record</summary>
                    <pre className="code">{optimizerRunRecordJson}</pre>
                  </details>
                ) : (
                  <div className="hint">No optimizer run yet.</div>
                )}
              </div>
            </div>
          </CollapsibleSection>

          <CollapsibleSection
            panelId="section-optimization"
            open={isPanelOpen("section-optimization", true)}
            onToggle={handlePanelToggle("section-optimization")}
            title="Optimization"
            meta="Tuning sweeps, presets, and constraints."
          >
            <div className="row">
              <div className="field">
              <div className="labelRow">
                <div className="label">Optimization</div>
                <InfoPopover label="Optimization modes">
                  <InfoList items={COMPLEX_TIPS.optimization} />
                </InfoPopover>
              </div>
                <div className="pillRow">
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.sweepThreshold}
                      disabled={form.method === "router"}
                      onChange={(e) => setForm((f) => ({ ...f, sweepThreshold: e.target.checked, optimizeOperations: false }))}
                    />
                    Sweep thresholds
                  </label>
                  <label className="pill">
                    <input
                      type="checkbox"
                      checked={form.optimizeOperations}
                      disabled={form.method === "router"}
                      onChange={(e) => setForm((f) => ({ ...f, optimizeOperations: e.target.checked, sweepThreshold: false }))}
                    />
                    Optimize operations (method + thresholds)
                  </label>
                </div>
                <div className="hint">
                  Tunes on the last part of the train split (fit/tune), then evaluates on the held-out backtest.
                  {form.method === "router" ? " Router mode disables optimize/sweep." : ""}
                </div>
                <div className="pillRow" style={{ marginTop: 10 }}>
                  <button
                    className="btnSmall"
                    type="button"
                    disabled={form.method === "router"}
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
                    disabled={form.method === "router"}
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
                    <div className="labelRow">
                      <label className="label" htmlFor="tuneObjective">
                        Tune objective
                      </label>
                      <InfoPopover label="Tune objective">
                        <InfoList items={COMPLEX_TIPS.tuneObjective} />
                      </InfoPopover>
                    </div>
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
                    <div className="labelRow">
                      <label className="label" htmlFor="walkForwardFolds">
                        Walk-forward folds
                      </label>
                      <InfoPopover label="Walk-forward folds">
                        <InfoList items={COMPLEX_TIPS.walkForward} />
                      </InfoPopover>
                    </div>
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
              <div className="label">Options</div>
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
                    aria-label="Auto-refresh interval in seconds"
                  />{" "}
                  seconds.{!form.autoRefresh ? " Enable Auto-refresh to use this interval." : ""}{" "}
                  {form.bypassCache ? "Bypass cache adds Cache-Control: no-cache." : ""}
                </div>
              </div>
            </div>

          </CollapsibleSection>
              </ConfigPanel>
              <ConfigPanel
                panelId="config-execution"
                title="Live Bot & Trade"
                subtitle="Arm trading, run bots, and size orders."
                order={configPanelOrderIndex["config-execution"]}
                open={isPanelOpen("config-execution", true)}
                onToggle={handlePanelToggle("config-execution")}
                maximized={isPanelMaximized("config-execution")}
                onToggleMaximize={() => togglePanelMaximize("config-execution")}
                style={configPanelStyle("config-execution")}
                {...configPanelHandlers}
              >
          <CollapsibleSection
            panelId="section-livebot"
            open={isPanelOpen("section-livebot", true)}
            onToggle={handlePanelToggle("section-livebot")}
            title="Live bot"
            meta="Start, stop, and tune the continuous loop."
          >
              <div className="row" style={{ gridTemplateColumns: "1fr" }}>
                <div className="field">
                  <div className="label">Live bot</div>
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
                      {botSymbolsFormatError ? (
                        <div className="hint" style={{ color: "rgba(239, 68, 68, 0.9)" }}>
                          {botSymbolsFormatError}
                        </div>
                      ) : null}
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
                        <div className="label">Startup position</div>
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
          </CollapsibleSection>

          <CollapsibleSection
            panelId="section-trade"
            open={isPanelOpen("section-trade", true)}
            onToggle={handlePanelToggle("section-trade")}
            title="Trade"
            meta="Arm trading, size orders, and run /trade."
          >
              <div className="row">
                <div className="field">
                  <div className="label">Trade controls</div>
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
                  <div className="label">Order sizing</div>
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

          </CollapsibleSection>
              </ConfigPanel>
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
                  apiToken, and optional apiFallbackUrl for CORS-enabled failover) and/or route <span style={{ fontFamily: "var(--mono)" }}>/api/*</span> to your backend.
                </>
              )}
            </p>
        </CollapsibleCard>
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
        </section>

        <CollapsibleCard
          panelId="panel-data-log"
          open={isPanelOpen("panel-data-log", false)}
          onToggle={handlePanelToggle("panel-data-log")}
          maximized={isPanelMaximized("panel-data-log")}
          onToggleMaximize={() => togglePanelMaximize("panel-data-log")}
          title="Data Log"
          subtitle="All incoming API responses (last 100 entries)"
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
          <div ref={dataLogRef} className="dataLogBox" onScroll={handleDataLogScroll}>
            {dataLogShown.length === 0 ? (
              <div className="dataLogEmpty">
                {dataLog.length === 0
                  ? "No data logged yet. Run a signal, backtest, or trade to see incoming data."
                  : "No entries match the current filter."}
              </div>
            ) : (
              dataLogShown.map((entry, idx) => (
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
          <div className="row" style={{ gridTemplateColumns: "1fr" }}>
            <div className="field">
              <div className="label">Optimizer combos</div>
              {(() => {
                const updatedAtMs = topCombosMeta.generatedAtMs;
                const updatedLabel = updatedAtMs ? fmtTimeMs(updatedAtMs) : "—";
                const ageLabel = updatedAtMs ? fmtDurationMs(Math.max(0, Date.now() - updatedAtMs)) : null;
                const sourceLabel = "Source: API";
                const payloadSources = topCombosMeta.payloadSources;
                const payloadSource = topCombosMeta.payloadSource;
                const payloadLabel =
                  payloadSources && payloadSources.length > 0
                    ? ` • payload ${payloadSources.join(" + ")}`
                    : payloadSource
                      ? ` • payload ${payloadSource}`
                      : "";
                const displayCount = topCombos.length;
                const filteredCount = topCombosFiltered.length;
                const totalCount = topCombosMeta.comboCount ?? topCombosAll.length;
                const filterLabel =
                  comboMinEquity != null ? ` (min final equity > ${fmtRatio(comboMinEquity, 4)})` : "";
                const countLabel =
                  comboMinEquity != null
                    ? `Showing ${displayCount} of ${filteredCount} combos${filterLabel}`
                    : totalCount > displayCount
                      ? `Showing ${displayCount} of ${totalCount} combos`
                      : `Showing ${displayCount} combo${displayCount === 1 ? "" : "s"}`;
                const totalLabel =
                  comboMinEquity != null && totalCount > filteredCount ? ` • ${totalCount} total` : "";
                return (
                  <div style={{ marginBottom: 8 }}>
                    <div className="hint">
                      {sourceLabel}
                      {payloadLabel}
                    </div>
                    <div className="hint">
                      Last updated {updatedLabel}
                      {ageLabel ? ` (${ageLabel} ago)` : ""}
                      {" • "}
                      {countLabel}
                      {totalLabel}
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
              <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center", marginBottom: 8 }}>
                <label className="label" htmlFor="comboDisplayCount">
                  Combos to show
                </label>
                <input
                  id="comboDisplayCount"
                  className="input"
                  type="number"
                  min={TOP_COMBOS_DISPLAY_MIN}
                  max={topCombosDisplayMax}
                  step={1}
                  value={topCombosDisplayCount}
                  onChange={(e) => {
                    const rawValue = numFromInput(e.target.value, topCombosDisplayCount);
                    const next = clamp(Math.trunc(rawValue), TOP_COMBOS_DISPLAY_MIN, topCombosDisplayMax);
                    setTopCombosDisplayCount(next);
                  }}
                  style={{ width: 120 }}
                />
                <label className="label" htmlFor="comboOrder">
                  Order by
                </label>
                <select
                  id="comboOrder"
                  className="select"
                  value={comboOrder}
                  onChange={(e) => setComboOrder(e.target.value as ComboOrder)}
                  style={{ minWidth: 180 }}
                >
                  <option value="annualized-equity">Annualized equity</option>
                  <option value="rank">Rank (score/final equity)</option>
                  <option value="date-desc">Date (newest)</option>
                  <option value="date-asc">Date (oldest)</option>
                </select>
                <label className="label" htmlFor="comboMinEquity">
                  Min final equity
                </label>
                <input
                  id="comboMinEquity"
                  className="input"
                  type="number"
                  step="0.0001"
                  value={comboMinEquityInput}
                  onChange={(e) => setComboMinEquityInput(e.target.value)}
                  placeholder="e.g. 1.5"
                  style={{ width: 140 }}
                />
              </div>
              {comboMinEquity != null && topCombosFiltered.length === 0 ? (
                <div className="hint" style={{ marginBottom: 8 }}>
                  No combos match the final equity filter.
                </div>
              ) : null}
              <div className="actions" style={{ marginBottom: 8 }}>
                <button className="btnSmall" type="button" onClick={refreshTopCombos} disabled={topCombosLoading}>
                  {topCombosLoading ? "Refreshing…" : "Refresh combos now"}
                </button>
                <button
                  className="btnSmall"
                  type="button"
                  onClick={() => {
                    if (topComboDisplay) handleComboApply(topComboDisplay);
                  }}
                  disabled={!topComboDisplay}
                >
                  Apply top combo now
                </button>
                {selectedCombo ? (
                  <button
                    className="btnSmall btnPrimary"
                    type="button"
                    onClick={() => handleComboStart(selectedCombo)}
                    disabled={comboStartBlocked}
                    title={comboStartBlockedReason ?? undefined}
                  >
                    {comboStartPending ? "Starting…" : selectedComboStartLabel}
                  </button>
                ) : null}
              </div>
              {selectedCombo && comboStartBlockedReason ? (
                <div className="hint" style={{ marginBottom: 8, color: "rgba(239, 68, 68, 0.85)" }}>
                  Start bot with selected combo is disabled: {comboStartBlockedReason}
                </div>
              ) : null}
              <div className="combosList">
              <details className="details" style={{ marginBottom: 12 }}>
                <summary>Run optimizer (create combos)</summary>
                <div onChange={() => setOptimizerRunDirty((prev) => (prev ? prev : true))}>
                <div className="row" style={{ marginTop: 10 }}>
                  <div className="field">
                    <label className="label" htmlFor="optimizerSource">
                      Source
                    </label>
                    <select
                      id="optimizerSource"
                      className="select"
                      value={optimizerRunForm.source}
                      onChange={(e) =>
                        setOptimizerRunForm((prev) => ({
                          ...prev,
                          source: e.target.value as OptimizerSource,
                        }))
                      }
                    >
                      <option value="binance">Binance</option>
                      <option value="coinbase">Coinbase</option>
                      <option value="kraken">Kraken</option>
                      <option value="poloniex">Poloniex</option>
                      <option value="csv">CSV</option>
                    </select>
                    <div className="hint">Choose the source used for optimizer data (CSV requires a path below).</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="optimizerSymbol">
                      Symbol
                    </label>
                    <input
                      id="optimizerSymbol"
                      className="input"
                      value={optimizerRunForm.symbol}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, symbol: e.target.value }))}
                      placeholder="BTCUSDT"
                      spellCheck={false}
                      disabled={optimizerRunForm.source === "csv"}
                    />
                    <div className="hint">Required for exchange sources; ignored for CSV.</div>
                  </div>
                </div>
                {optimizerRunForm.source === "csv" ? (
                  <>
                    <div className="row" style={{ marginTop: 10 }}>
                      <div className="field">
                        <label className="label" htmlFor="optimizerDataPath">
                          CSV path
                        </label>
                        <input
                          id="optimizerDataPath"
                          className="input"
                          value={optimizerRunForm.dataPath}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, dataPath: e.target.value }))}
                          placeholder="../data/sample_prices.csv"
                          spellCheck={false}
                        />
                        <div className="hint">Path is resolved on the API host.</div>
                      </div>
                      <div className="field">
                        <label className="label" htmlFor="optimizerPriceColumn">
                          Price column
                        </label>
                        <input
                          id="optimizerPriceColumn"
                          className="input"
                          value={optimizerRunForm.priceColumn}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, priceColumn: e.target.value }))}
                          placeholder="close"
                          spellCheck={false}
                        />
                        <div className="hint">Defaults to close when omitted.</div>
                      </div>
                    </div>
                    <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
                      <div className="field">
                        <label className="label" htmlFor="optimizerHighColumn">
                          High column (optional)
                        </label>
                        <input
                          id="optimizerHighColumn"
                          className="input"
                          value={optimizerRunForm.highColumn}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, highColumn: e.target.value }))}
                          placeholder="high"
                          spellCheck={false}
                        />
                      </div>
                      <div className="field">
                        <label className="label" htmlFor="optimizerLowColumn">
                          Low column (optional)
                        </label>
                        <input
                          id="optimizerLowColumn"
                          className="input"
                          value={optimizerRunForm.lowColumn}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, lowColumn: e.target.value }))}
                          placeholder="low"
                          spellCheck={false}
                        />
                      </div>
                    </div>
                  </>
                ) : null}
                <div className="row" style={{ marginTop: 10 }}>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerIntervals">
                        Intervals
                      </label>
                      <InfoPopover label="Equity tip: intervals">
                        <InfoList items={EQUITY_TIPS.intervals} />
                      </InfoPopover>
                    </div>
                    <input
                      id="optimizerIntervals"
                      className="input"
                      value={optimizerRunForm.intervals}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, intervals: e.target.value }))}
                      placeholder="1h,2h,4h,6h,12h,1d"
                      spellCheck={false}
                    />
                    <div className="hint">Comma-separated list; leave blank for API defaults.</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="optimizerLookbackWindow">
                      Lookback window
                    </label>
                    <input
                      id="optimizerLookbackWindow"
                      className="input"
                      value={optimizerRunForm.lookbackWindow}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, lookbackWindow: e.target.value }))}
                      placeholder="7d"
                      spellCheck={false}
                    />
                    <div className="hint">Duration string like 48h, 7d, 30d.</div>
                  </div>
                </div>
                <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1.2fr 0.8fr 0.8fr" }}>
                  <div className="field">
                    <label className="label" htmlFor="optimizerBarsMin">
                      Bars range
                    </label>
                    <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                      <input
                        id="optimizerBarsMin"
                        className="input"
                        type="number"
                        value={optimizerRunForm.barsMin}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, barsMin: e.target.value }))}
                        placeholder="min"
                      />
                      <input
                        aria-label="Bars max"
                        className="input"
                        type="number"
                        value={optimizerRunForm.barsMax}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, barsMax: e.target.value }))}
                        placeholder="max"
                      />
                    </div>
                    <div className="hint">0 or blank lets the optimizer choose.</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="optimizerBarsAutoProb">
                      Bars auto prob
                    </label>
                    <input
                      id="optimizerBarsAutoProb"
                      className="input"
                      type="number"
                      step="0.01"
                      min={0}
                      max={1}
                      value={optimizerRunForm.barsAutoProb}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, barsAutoProb: e.target.value }))}
                      placeholder="0.25"
                    />
                    <div className="hint">Chance to use bars=0 (exchange default).</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="optimizerBarsDistribution">
                      Bars distribution
                    </label>
                    <select
                      id="optimizerBarsDistribution"
                      className="select"
                      value={optimizerRunForm.barsDistribution}
                      onChange={(e) =>
                        setOptimizerRunForm((prev) => ({
                          ...prev,
                          barsDistribution: e.target.value as OptimizerRunForm["barsDistribution"],
                        }))
                      }
                    >
                      <option value="">Default (uniform)</option>
                      <option value="uniform">uniform</option>
                      <option value="log">log</option>
                    </select>
                  </div>
                </div>
                <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerTrials">
                        Trials
                      </label>
                      <InfoPopover label="Equity tip: trials and timeout">
                        <InfoList items={EQUITY_TIPS.trials} />
                      </InfoPopover>
                    </div>
                    <input
                      id="optimizerTrials"
                      className="input"
                      type="number"
                      min={1}
                      value={optimizerRunForm.trials}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, trials: e.target.value }))}
                      placeholder="50"
                    />
                  </div>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerTimeoutSec">
                        Timeout (sec)
                      </label>
                      <InfoPopover label="Equity tip: trials and timeout">
                        <InfoList items={EQUITY_TIPS.trials} />
                      </InfoPopover>
                    </div>
                    <input
                      id="optimizerTimeoutSec"
                      className="input"
                      type="number"
                      min={1}
                      step="1"
                      value={optimizerRunForm.timeoutSec}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, timeoutSec: e.target.value }))}
                      placeholder="60"
                    />
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="optimizerSeed">
                      Seed
                    </label>
                    <input
                      id="optimizerSeed"
                      className="input"
                      type="number"
                      min={0}
                      step="1"
                      value={optimizerRunForm.seed}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, seed: e.target.value }))}
                      placeholder="42"
                    />
                  </div>
                </div>
                <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerObjective">
                        Objective
                      </label>
                      <InfoPopover label="Equity tip: objective">
                        <InfoList items={EQUITY_TIPS.objective} />
                      </InfoPopover>
                    </div>
                    <select
                      id="optimizerObjective"
                      className="select"
                      value={optimizerRunForm.objective}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, objective: e.target.value }))}
                    >
                      <option value="">Default</option>
                      {TUNE_OBJECTIVES.map((o) => (
                        <option key={o} value={o}>
                          {o}
                        </option>
                      ))}
                    </select>
                    <div className="hint">Controls which combos survive.</div>
                  </div>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerTuneObjective">
                        Tune objective
                      </label>
                      <InfoPopover label="Equity tip: objective">
                        <InfoList items={EQUITY_TIPS.objective} />
                      </InfoPopover>
                    </div>
                    <select
                      id="optimizerTuneObjective"
                      className="select"
                      value={optimizerRunForm.tuneObjective}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, tuneObjective: e.target.value }))}
                    >
                      <option value="">Default</option>
                      {TUNE_OBJECTIVES.map((o) => (
                        <option key={o} value={o}>
                          {o}
                        </option>
                      ))}
                    </select>
                    <div className="hint">Used during fit/tune scoring.</div>
                  </div>
                </div>
                <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerBacktestRatio">
                        Backtest ratio
                      </label>
                      <InfoPopover label="Equity tip: backtest and tune ratios">
                        <InfoList items={EQUITY_TIPS.ratios} />
                      </InfoPopover>
                    </div>
                    <input
                      id="optimizerBacktestRatio"
                      className="input"
                      type="number"
                      step="0.01"
                      min={0}
                      max={0.99}
                      value={optimizerRunForm.backtestRatio}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, backtestRatio: e.target.value }))}
                      placeholder="0.2"
                    />
                  </div>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerTuneRatio">
                        Tune ratio
                      </label>
                      <InfoPopover label="Equity tip: backtest and tune ratios">
                        <InfoList items={EQUITY_TIPS.ratios} />
                      </InfoPopover>
                    </div>
                    <input
                      id="optimizerTuneRatio"
                      className="input"
                      type="number"
                      step="0.01"
                      min={0}
                      max={0.99}
                      value={optimizerRunForm.tuneRatio}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, tuneRatio: e.target.value }))}
                      placeholder="0.25"
                    />
                  </div>
                </div>
                <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerPenaltyMaxDrawdown">
                        DD penalty
                      </label>
                      <InfoPopover label="Equity tip: penalties">
                        <InfoList items={EQUITY_TIPS.penalties} />
                      </InfoPopover>
                    </div>
                    <input
                      id="optimizerPenaltyMaxDrawdown"
                      className="input"
                      type="number"
                      step="0.1"
                      min={0}
                      value={optimizerRunForm.penaltyMaxDrawdown}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, penaltyMaxDrawdown: e.target.value }))}
                      placeholder="1.5"
                    />
                  </div>
                  <div className="field">
                    <div className="labelRow">
                      <label className="label" htmlFor="optimizerPenaltyTurnover">
                        Turnover penalty
                      </label>
                      <InfoPopover label="Equity tip: penalties">
                        <InfoList items={EQUITY_TIPS.penalties} />
                      </InfoPopover>
                    </div>
                    <input
                      id="optimizerPenaltyTurnover"
                      className="input"
                      type="number"
                      step="0.1"
                      min={0}
                      value={optimizerRunForm.penaltyTurnover}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, penaltyTurnover: e.target.value }))}
                      placeholder="0.2"
                    />
                  </div>
                </div>
                <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
                  <div className="field">
                    <label className="label" htmlFor="optimizerSlippageMax">
                      Slippage max
                    </label>
                    <input
                      id="optimizerSlippageMax"
                      className="input"
                      type="number"
                      step="0.0001"
                      min={0}
                      value={optimizerRunForm.slippageMax}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, slippageMax: e.target.value }))}
                      placeholder="0.0005"
                    />
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="optimizerSpreadMax">
                      Spread max
                    </label>
                    <input
                      id="optimizerSpreadMax"
                      className="input"
                      type="number"
                      step="0.0001"
                      min={0}
                      value={optimizerRunForm.spreadMax}
                      onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, spreadMax: e.target.value }))}
                      placeholder="0.0005"
                    />
                  </div>
                </div>
                <div className="actions" style={{ marginTop: 12 }}>
                  <button
                    className="btn btnPrimary"
                    type="button"
                    onClick={() => void runOptimizer()}
                    disabled={optimizerRunUi.loading || Boolean(optimizerRunValidationError) || apiOk === "down" || apiOk === "auth"}
                    title={optimizerRunValidationError ?? undefined}
                  >
                    {optimizerRunUi.loading ? "Running…" : "Run optimizer"}
                  </button>
                  <button className="btn" type="button" onClick={cancelOptimizerRun} disabled={!optimizerRunUi.loading}>
                    Cancel
                  </button>
                  <button className="btn" type="button" onClick={syncOptimizerRunSymbolInterval}>
                    Use current symbol/interval
                  </button>
                  <button className="btn" type="button" onClick={applyEquityPreset}>
                    Preset: Equity focus
                  </button>
                  <InfoPopover label="Equity options" align="left">
                    <InfoList items={EQUITY_TIPS.preset} />
                  </InfoPopover>
                  <button className="btn" type="button" onClick={resetOptimizerRunForm}>
                    Reset defaults
                  </button>
                </div>
                <div className="hint" style={{ marginTop: 8 }}>
                  Runs <code>/optimizer/run</code> to generate new combos and refreshes the list above. For annualized equity, keep objective/tune objective on
                  <code>annualized-equity</code> and increase trials/timeout.
                </div>
                {optimizerRunValidationError ? (
                  <div className="hint" style={{ marginTop: 8, color: "rgba(239, 68, 68, 0.85)" }}>
                    {optimizerRunValidationError}
                  </div>
                ) : null}
                <details className="details" style={{ marginTop: 12 }}>
                  <summary>Sampling + model ranges</summary>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerPlatforms">
                        Platforms (optional)
                      </label>
                      <input
                        id="optimizerPlatforms"
                        className="input"
                        value={optimizerRunForm.platforms}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, platforms: e.target.value }))}
                        placeholder="binance,coinbase"
                        spellCheck={false}
                        disabled={optimizerRunForm.source === "csv"}
                      />
                      <div className="hint">Overrides the source platform for multi-exchange runs.</div>
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerNormalizations">
                        Normalizations
                      </label>
                      <input
                        id="optimizerNormalizations"
                        className="input"
                        value={optimizerRunForm.normalizations}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, normalizations: e.target.value }))}
                        placeholder="none,minmax,standard,log"
                        spellCheck={false}
                      />
                      <div className="hint">Comma-separated list for LSTM runs.</div>
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerEpochsMin">
                        Epochs range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerEpochsMin"
                          className="input"
                          type="number"
                          min={0}
                          value={optimizerRunForm.epochsMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, epochsMin: e.target.value }))}
                          placeholder="0"
                        />
                        <input
                          aria-label="Epochs max"
                          className="input"
                          type="number"
                          min={0}
                          value={optimizerRunForm.epochsMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, epochsMax: e.target.value }))}
                          placeholder="10"
                        />
                      </div>
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerHiddenMin">
                        Hidden size range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerHiddenMin"
                          className="input"
                          type="number"
                          min={1}
                          value={optimizerRunForm.hiddenSizeMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, hiddenSizeMin: e.target.value }))}
                          placeholder="8"
                        />
                        <input
                          aria-label="Hidden size max"
                          className="input"
                          type="number"
                          min={1}
                          value={optimizerRunForm.hiddenSizeMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, hiddenSizeMax: e.target.value }))}
                          placeholder="64"
                        />
                      </div>
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerLrMin">
                        Learning rate range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerLrMin"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.lrMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, lrMin: e.target.value }))}
                          placeholder="0.0001"
                        />
                        <input
                          aria-label="Learning rate max"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.lrMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, lrMax: e.target.value }))}
                          placeholder="0.01"
                        />
                      </div>
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerPatienceMax">
                        Patience max
                      </label>
                      <input
                        id="optimizerPatienceMax"
                        className="input"
                        type="number"
                        min={0}
                        value={optimizerRunForm.patienceMax}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, patienceMax: e.target.value }))}
                        placeholder="20"
                      />
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerGradClipMin">
                        Grad clip range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerGradClipMin"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.gradClipMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, gradClipMin: e.target.value }))}
                          placeholder="0.001"
                        />
                        <input
                          aria-label="Grad clip max"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.gradClipMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, gradClipMax: e.target.value }))}
                          placeholder="1.0"
                        />
                      </div>
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerDisableGradClipProb">
                        Disable grad clip prob
                      </label>
                      <input
                        id="optimizerDisableGradClipProb"
                        className="input"
                        type="number"
                        step="0.01"
                        min={0}
                        max={1}
                        value={optimizerRunForm.pDisableGradClip}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, pDisableGradClip: e.target.value }))}
                        placeholder="0.7"
                      />
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerSeedTrials">
                        Seed trials
                      </label>
                      <input
                        id="optimizerSeedTrials"
                        className="input"
                        type="number"
                        min={0}
                        value={optimizerRunForm.seedTrials}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, seedTrials: e.target.value }))}
                        placeholder="0"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerSeedRatio">
                        Seed ratio
                      </label>
                      <input
                        id="optimizerSeedRatio"
                        className="input"
                        type="number"
                        min={0}
                        max={1}
                        step="0.01"
                        value={optimizerRunForm.seedRatio}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, seedRatio: e.target.value }))}
                        placeholder="0.0"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerSurvivorFraction">
                        Survivor fraction
                      </label>
                      <input
                        id="optimizerSurvivorFraction"
                        className="input"
                        type="number"
                        min={0}
                        max={1}
                        step="0.01"
                        value={optimizerRunForm.survivorFraction}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, survivorFraction: e.target.value }))}
                        placeholder="0.5"
                      />
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerPerturbScaleDouble">
                        Perturb scale (float)
                      </label>
                      <input
                        id="optimizerPerturbScaleDouble"
                        className="input"
                        type="number"
                        step="0.01"
                        min={0}
                        value={optimizerRunForm.perturbScaleDouble}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, perturbScaleDouble: e.target.value }))}
                        placeholder="0.1"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerPerturbScaleInt">
                        Perturb scale (int)
                      </label>
                      <input
                        id="optimizerPerturbScaleInt"
                        className="input"
                        type="number"
                        step="1"
                        min={0}
                        value={optimizerRunForm.perturbScaleInt}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, perturbScaleInt: e.target.value }))}
                        placeholder="2"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerEarlyStop">
                        Early stop (no improve)
                      </label>
                      <input
                        id="optimizerEarlyStop"
                        className="input"
                        type="number"
                        step="1"
                        min={0}
                        value={optimizerRunForm.earlyStopNoImprove}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, earlyStopNoImprove: e.target.value }))}
                        placeholder="0"
                      />
                    </div>
                  </div>
                  <div className="pillRow" style={{ marginTop: 10 }}>
                    <label className="pill">
                      <input
                        type="checkbox"
                        checked={optimizerRunForm.disableLstmPersistence}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, disableLstmPersistence: e.target.checked }))}
                      />
                      Disable LSTM persistence
                    </label>
                    <label className="pill">
                      <input
                        type="checkbox"
                        checked={optimizerRunForm.noSweepThreshold}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, noSweepThreshold: e.target.checked }))}
                      />
                      No threshold sweep
                    </label>
                  </div>
                </details>
                <details className="details" style={{ marginTop: 12 }}>
                  <summary>Quality filters + constraints</summary>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMinRoundTrips">
                        Min round trips
                      </label>
                      <input
                        id="optimizerMinRoundTrips"
                        className="input"
                        type="number"
                        min={0}
                        value={optimizerRunForm.minRoundTrips}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minRoundTrips: e.target.value }))}
                        placeholder="0"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMinWinRate">
                        Min win rate
                      </label>
                      <input
                        id="optimizerMinWinRate"
                        className="input"
                        type="number"
                        step="0.01"
                        min={0}
                        max={1}
                        value={optimizerRunForm.minWinRate}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minWinRate: e.target.value }))}
                        placeholder="0.0"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMinSharpe">
                        Min Sharpe
                      </label>
                      <input
                        id="optimizerMinSharpe"
                        className="input"
                        type="number"
                        step="0.1"
                        min={0}
                        value={optimizerRunForm.minSharpe}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minSharpe: e.target.value }))}
                        placeholder="0.0"
                      />
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMinAnnualizedReturn">
                        Min annualized return
                      </label>
                      <input
                        id="optimizerMinAnnualizedReturn"
                        className="input"
                        type="number"
                        step="0.01"
                        min={0}
                        value={optimizerRunForm.minAnnualizedReturn}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minAnnualizedReturn: e.target.value }))}
                        placeholder="0.0"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMinCalmar">
                        Min Calmar
                      </label>
                      <input
                        id="optimizerMinCalmar"
                        className="input"
                        type="number"
                        step="0.1"
                        min={0}
                        value={optimizerRunForm.minCalmar}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minCalmar: e.target.value }))}
                        placeholder="0.0"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMinProfitFactor">
                        Min profit factor
                      </label>
                      <input
                        id="optimizerMinProfitFactor"
                        className="input"
                        type="number"
                        step="0.1"
                        min={0}
                        value={optimizerRunForm.minProfitFactor}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minProfitFactor: e.target.value }))}
                        placeholder="0.0"
                      />
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMaxTurnover">
                        Max turnover
                      </label>
                      <input
                        id="optimizerMaxTurnover"
                        className="input"
                        type="number"
                        step="0.01"
                        min={0}
                        value={optimizerRunForm.maxTurnover}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, maxTurnover: e.target.value }))}
                        placeholder="0.0"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMinExposure">
                        Min exposure
                      </label>
                      <input
                        id="optimizerMinExposure"
                        className="input"
                        type="number"
                        step="0.01"
                        min={0}
                        max={1}
                        value={optimizerRunForm.minExposure}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minExposure: e.target.value }))}
                        placeholder="0.0"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMinWfSharpeMean">
                        Min WF Sharpe mean
                      </label>
                      <input
                        id="optimizerMinWfSharpeMean"
                        className="input"
                        type="number"
                        step="0.1"
                        min={0}
                        value={optimizerRunForm.minWalkForwardSharpeMean}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minWalkForwardSharpeMean: e.target.value }))}
                        placeholder="0.0"
                      />
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMaxWfSharpeStd">
                        Max WF Sharpe std
                      </label>
                      <input
                        id="optimizerMaxWfSharpeStd"
                        className="input"
                        type="number"
                        step="0.1"
                        min={0}
                        value={optimizerRunForm.maxWalkForwardSharpeStd}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, maxWalkForwardSharpeStd: e.target.value }))}
                        placeholder="0.0"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerWalkForwardMin">
                        Walk-forward folds min
                      </label>
                      <input
                        id="optimizerWalkForwardMin"
                        className="input"
                        type="number"
                        min={1}
                        value={optimizerRunForm.walkForwardFoldsMin}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, walkForwardFoldsMin: e.target.value }))}
                        placeholder="7"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerWalkForwardMax">
                        Walk-forward folds max
                      </label>
                      <input
                        id="optimizerWalkForwardMax"
                        className="input"
                        type="number"
                        min={1}
                        value={optimizerRunForm.walkForwardFoldsMax}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, walkForwardFoldsMax: e.target.value }))}
                        placeholder="7"
                      />
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMinEdgeMin">
                        Min edge range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerMinEdgeMin"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.minEdgeMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minEdgeMin: e.target.value }))}
                          placeholder="min"
                        />
                        <input
                          aria-label="Min edge max"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.minEdgeMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minEdgeMax: e.target.value }))}
                          placeholder="max"
                        />
                      </div>
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMinSnrMin">
                        Min SNR range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerMinSnrMin"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.minSignalToNoiseMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minSignalToNoiseMin: e.target.value }))}
                          placeholder="min"
                        />
                        <input
                          aria-label="Min SNR max"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.minSignalToNoiseMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minSignalToNoiseMax: e.target.value }))}
                          placeholder="max"
                        />
                      </div>
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerEdgeBufferMin">
                        Edge buffer range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerEdgeBufferMin"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.edgeBufferMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, edgeBufferMin: e.target.value }))}
                          placeholder="min"
                        />
                        <input
                          aria-label="Edge buffer max"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.edgeBufferMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, edgeBufferMax: e.target.value }))}
                          placeholder="max"
                        />
                      </div>
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerTrendLookbackMin">
                        Trend lookback range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerTrendLookbackMin"
                          className="input"
                          type="number"
                          min={0}
                          value={optimizerRunForm.trendLookbackMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, trendLookbackMin: e.target.value }))}
                          placeholder="min"
                        />
                        <input
                          aria-label="Trend lookback max"
                          className="input"
                          type="number"
                          min={0}
                          value={optimizerRunForm.trendLookbackMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, trendLookbackMax: e.target.value }))}
                          placeholder="max"
                        />
                      </div>
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10 }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerCostAwareEdge">
                        Cost-aware edge prob
                      </label>
                      <input
                        id="optimizerCostAwareEdge"
                        className="input"
                        type="number"
                        step="0.01"
                        min={0}
                        max={1}
                        value={optimizerRunForm.pCostAwareEdge}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, pCostAwareEdge: e.target.value }))}
                        placeholder="0.0"
                      />
                      <div className="hint">Probability of enabling cost-aware edge (0 disables).</div>
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMinHoldBarsMin">
                        Min hold bars range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerMinHoldBarsMin"
                          className="input"
                          type="number"
                          min={0}
                          value={optimizerRunForm.minHoldBarsMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minHoldBarsMin: e.target.value }))}
                          placeholder="min"
                        />
                        <input
                          aria-label="Min hold bars max"
                          className="input"
                          type="number"
                          min={0}
                          value={optimizerRunForm.minHoldBarsMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minHoldBarsMax: e.target.value }))}
                          placeholder="max"
                        />
                      </div>
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerCooldownBarsMin">
                        Cooldown bars range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerCooldownBarsMin"
                          className="input"
                          type="number"
                          min={0}
                          value={optimizerRunForm.cooldownBarsMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, cooldownBarsMin: e.target.value }))}
                          placeholder="min"
                        />
                        <input
                          aria-label="Cooldown bars max"
                          className="input"
                          type="number"
                          min={0}
                          value={optimizerRunForm.cooldownBarsMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, cooldownBarsMax: e.target.value }))}
                          placeholder="max"
                        />
                      </div>
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMaxHoldBarsMin">
                        Max hold bars range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerMaxHoldBarsMin"
                          className="input"
                          type="number"
                          min={0}
                          value={optimizerRunForm.maxHoldBarsMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, maxHoldBarsMin: e.target.value }))}
                          placeholder="min"
                        />
                        <input
                          aria-label="Max hold bars max"
                          className="input"
                          type="number"
                          min={0}
                          value={optimizerRunForm.maxHoldBarsMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, maxHoldBarsMax: e.target.value }))}
                          placeholder="max"
                        />
                      </div>
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerStopMin">
                        Stop loss range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerStopMin"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.stopMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, stopMin: e.target.value }))}
                          placeholder="min"
                        />
                        <input
                          aria-label="Stop loss max"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.stopMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, stopMax: e.target.value }))}
                          placeholder="max"
                        />
                      </div>
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerTpMin">
                        Take profit range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerTpMin"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.tpMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, tpMin: e.target.value }))}
                          placeholder="min"
                        />
                        <input
                          aria-label="Take profit max"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.tpMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, tpMax: e.target.value }))}
                          placeholder="max"
                        />
                      </div>
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerTrailMin">
                        Trailing stop range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerTrailMin"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.trailMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, trailMin: e.target.value }))}
                          placeholder="min"
                        />
                        <input
                          aria-label="Trailing stop max"
                          className="input"
                          type="number"
                          step="0.0001"
                          min={0}
                          value={optimizerRunForm.trailMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, trailMax: e.target.value }))}
                          placeholder="max"
                        />
                      </div>
                    </div>
                  </div>
                  <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerMethodWeightBlend">
                        Blend method weight
                      </label>
                      <input
                        id="optimizerMethodWeightBlend"
                        className="input"
                        type="number"
                        step="0.1"
                        min={0}
                        value={optimizerRunForm.methodWeightBlend}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, methodWeightBlend: e.target.value }))}
                        placeholder="0.0"
                      />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="optimizerBlendWeightMin">
                        Blend weight range
                      </label>
                      <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                        <input
                          id="optimizerBlendWeightMin"
                          className="input"
                          type="number"
                          step="0.01"
                          min={0}
                          max={1}
                          value={optimizerRunForm.blendWeightMin}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, blendWeightMin: e.target.value }))}
                          placeholder="min"
                        />
                        <input
                          aria-label="Blend weight max"
                          className="input"
                          type="number"
                          step="0.01"
                          min={0}
                          max={1}
                          value={optimizerRunForm.blendWeightMax}
                          onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, blendWeightMax: e.target.value }))}
                          placeholder="max"
                        />
                      </div>
                    </div>
                  </div>
                </details>
                <details className="details" style={{ marginTop: 12 }}>
                  <summary>Advanced JSON overrides</summary>
                  <div className="row" style={{ marginTop: 10 }}>
                    <div className="field">
                      <label className="label" htmlFor="optimizerExtraJson">
                        Extra JSON fields
                      </label>
                      <textarea
                        id="optimizerExtraJson"
                        className="input"
                        value={optimizerRunForm.extraJson}
                        onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, extraJson: e.target.value }))}
                        placeholder='{"minPositionSizeMin": 0.1, "pConfirmQuantiles": 0.2}'
                        rows={4}
                        spellCheck={false}
                      />
                      <div className="hint">Merged into the request (overrides form inputs when keys overlap).</div>
                      {optimizerRunExtras.error ? (
                        <div className="hint" style={{ color: "rgba(239, 68, 68, 0.85)" }}>
                          {optimizerRunExtras.error}
                        </div>
                      ) : null}
                    </div>
                  </div>
                </details>
                {optimizerRunUi.lastRunAtMs ? (
                  <div className="hint" style={{ marginTop: 12 }}>
                    Last optimizer run {fmtTimeMs(optimizerRunUi.lastRunAtMs)}.
                  </div>
                ) : null}
                {optimizerRunUi.response ? (
                  <div style={{ marginTop: 10 }}>
                    <div className="label">Last optimizer output</div>
                    {optimizerRunRecordJson ? <pre className="code">{optimizerRunRecordJson}</pre> : null}
                    {optimizerRunUi.response.stdout ? <pre className="code">{optimizerRunUi.response.stdout}</pre> : null}
                    {optimizerRunUi.response.stderr ? (
                      <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)" }}>
                        {optimizerRunUi.response.stderr}
                      </pre>
                    ) : null}
                  </div>
                ) : null}
                {optimizerRunUi.error && !optimizerRunValidationError ? (
                  <div className="hint" style={{ marginTop: 8, color: "rgba(239, 68, 68, 0.85)" }}>
                    {optimizerRunUi.error}
                  </div>
                ) : null}
                </div>
              </details>
              <Suspense fallback={<PanelFallback label="Loading optimizer combos…" />}>
                <TopCombosChart
                  combos={topCombos}
                  loading={topCombosLoading}
                  error={topCombosError}
                  selectedId={selectedComboId}
                  onSelect={handleComboPreview}
                  onApply={handleComboApply}
                />
              </Suspense>
              <div className="hint">
                Select a combo to preview. Click Apply to load params into the form and auto-start a live bot for that symbol (Binance only). bars=0 uses all CSV data or the exchange default (500).
              </div>
              <div className="hint">
                Top combos auto-apply when available (manual overrides respected). If the bot is idle, it will auto-start once the top combo is applied.
              </div>
              </div>
            </div>
          </div>
        </CollapsibleCard>
      </div>
    </div>
  </div>
  );
}
