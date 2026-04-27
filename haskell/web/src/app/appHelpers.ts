import type {
  ApiParams,
  ApiBinancePositionsResponse,
  ApiBinanceTradesResponse,
  ApiTradeResponse,
  BacktestResponse,
  BinanceKeysStatus,
  BinanceListenKeyResponse,
  BinanceTrade,
  BotStatus,
  BotStatusMulti,
  BotStatusSingle,
  CoinbaseKeysStatus,
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
} from "../lib/types";
import type { OptimizationCombo, OptimizationComboOperation } from "../components/TopCombosChart";
import { defaultForm, parseDurationSeconds, platformIntervalSeconds } from "./formState";
import type { FormState } from "./formState";
import type { cacheStats, health } from "../lib/api";
import { fmtPct } from "../lib/format";
import { PLATFORM_DEFAULT_BARS, PLATFORM_DEFAULT_SYMBOL } from "./constants";
import { preferredExchangePlatform } from "./contracts";
import { METHOD_TIPS } from "./methodMeta";
import {
  BINANCE_SYMBOL_PATTERN,
  COMMON_QUOTES,
  invalidSymbolsForPlatform,
  normalizeComboSymbol,
  sanitizeSymbolForPlatform,
  symbolFormatExample,
  symbolFormatPattern,
  trimBinanceComboSuffix,
} from "./symbols";
import {
  clamp,
  isEffectivelyFlatPositionAmount,
  normalizePositionSide,
  normalizeSymbolKey,
  numFromInput,
  positionSideFromAmount,
} from "./utils";

export type RequestKind = "signal" | "backtest" | "trade";

export type RunOptions = {
  silent?: boolean;
};

export type ActiveAsyncJob = {
  kind: RequestKind;
  jobId: string | null;
  startedAtMs: number;
};

export type RateLimitState = {
  untilMs: number;
  reason: string;
  lastHitAtMs: number;
};

export type PanelPrefs = Record<string, boolean>;

export type KeysStatus = BinanceKeysStatus | CoinbaseKeysStatus;

export type OpsUiState = {
  loading: boolean;
  error: string | null;
  enabled: boolean;
  hint: string | null;
  ops: OpsOperation[];
  limit: number;
  lastFetchedAtMs: number | null;
};

export type OpsPerformanceUiState = {
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

export type BotStatusOp = {
  atMs: number;
  running: boolean;
  live: boolean;
  symbol: string | null;
};

export type BotOrderOp = {
  atMs: number;
  symbol: string;
  market: Market;
  interval: string;
  openTime: number;
  side: "BUY" | "SELL";
  price: number | null;
  position: number | null;
  sent: boolean;
};

function parseMarket(raw: unknown): Market | null {
  if (raw === "spot" || raw === "margin" || raw === "futures") return raw;
  return null;
}
export function isCoinbaseKeysStatus(status: KeysStatus): status is CoinbaseKeysStatus {
  return "hasApiPassphrase" in status;
}

export function isBinanceKeysStatus(status: KeysStatus): status is BinanceKeysStatus {
  return "market" in status;
}

export function isBotStatusMulti(status: BotStatus): status is BotStatusMulti {
  return ("multi" in status && status.multi === true) || ("bots" in status && Array.isArray(status.bots));
}

export function botStatusSymbol(status: BotStatusSingle): string | null {
  if (status.running) return status.symbol;
  if (status.symbol) return status.symbol;
  if (status.snapshot?.symbol) return status.snapshot.symbol;
  return null;
}

export function botStatusKey(status: { market: Market; symbol: string; interval: string }): string {
  return `${status.market}:${normalizeSymbolKey(status.symbol)}:${status.interval}`;
}

export function botStatusKeyFromSingle(status: BotStatusSingle): string | null {
  const symbol = botStatusSymbol(status);
  if (!symbol) return null;
  const market = status.running ? status.market : status.market ?? status.snapshot?.market;
  const interval = status.running ? status.interval : status.interval ?? status.snapshot?.interval;
  if (!market || !interval) return null;
  return botStatusKey({ market, symbol, interval });
}

const LOCAL_DATETIME_RE = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,3}))?)?$/;

function localDateTimePartsMatch(parsedMs: number, match: RegExpExecArray): boolean {
  const [
    ,
    yearRaw,
    monthRaw,
    dayRaw,
    hourRaw,
    minuteRaw,
    secondRaw = "0",
    millisecondRaw = "0",
  ] = match;
  const observed = new Date(parsedMs);
  if (!Number.isFinite(observed.getTime())) return false;
  return (
    observed.getFullYear() === Number(yearRaw) &&
    observed.getMonth() + 1 === Number(monthRaw) &&
    observed.getDate() === Number(dayRaw) &&
    observed.getHours() === Number(hourRaw) &&
    observed.getMinutes() === Number(minuteRaw) &&
    observed.getSeconds() === Number(secondRaw) &&
    observed.getMilliseconds() === Number(millisecondRaw.padEnd(3, "0"))
  );
}

export function formatDatetimeLocal(ms: number): string {
  if (!Number.isFinite(ms)) return "";
  const d = new Date(ms);
  if (!Number.isFinite(d.getTime())) return "";
  const pad = (v: number) => String(v).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function parseDatetimeLocal(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const match = LOCAL_DATETIME_RE.exec(trimmed);
  if (!match) return null;
  const parsed = Date.parse(trimmed.replace(" ", "T"));
  if (Number.isNaN(parsed)) return null;
  return localDateTimePartsMatch(parsed, match) ? parsed : null;
}

export function parseBotStatusOp(op: OpsOperation): BotStatusOp | null {
  if (!op || op.kind !== "bot.status") return null;
  if (typeof op.atMs !== "number" || !Number.isFinite(op.atMs)) return null;
  const rec = (op.result as Record<string, unknown> | null | undefined) ?? {};
  const running = typeof rec.running === "boolean" ? rec.running : null;
  if (running == null) return null;
  const live = typeof rec.live === "boolean" ? rec.live : false;
  const symbol = typeof rec.symbol === "string" ? rec.symbol : null;
  return { atMs: op.atMs, running, live, symbol };
}

export function parseBotOrderOp(op: OpsOperation): BotOrderOp | null {
  if (!op || op.kind !== "bot.order") return null;
  if (typeof op.atMs !== "number" || !Number.isFinite(op.atMs)) return null;
  const rec = (op.result as Record<string, unknown> | null | undefined) ?? {};
  const symbol = typeof rec.symbol === "string" ? rec.symbol : null;
  const market = parseMarket(rec.market);
  const interval = typeof rec.interval === "string" ? rec.interval : null;
  const event = (rec.event as Record<string, unknown> | null | undefined) ?? null;
  if (!symbol || !market || !interval || !event) return null;
  const opSide = event.opSide;
  if (opSide !== "BUY" && opSide !== "SELL") return null;
  const index = typeof event.index === "number" && Number.isFinite(event.index) ? event.index : null;
  const openTime = typeof event.openTime === "number" && Number.isFinite(event.openTime) ? event.openTime : null;
  const atMs = typeof event.atMs === "number" && Number.isFinite(event.atMs) ? event.atMs : null;
  const price = typeof event.price === "number" && Number.isFinite(event.price) ? event.price : null;
  if (index == null || openTime == null || atMs == null) return null;
  const order = (event.order as Record<string, unknown> | null | undefined) ?? null;
  const sent = typeof order?.sent === "boolean" ? order.sent : false;
  const position =
    typeof rec.position === "number" && Number.isFinite(rec.position) ? rec.position : null;
  return {
    atMs,
    symbol,
    market,
    interval,
    openTime,
    side: opSide,
    price,
    position,
    sent,
  };
}

export function parseSymbolsInput(raw: string): string[] {
  // Whitespace around symbol delimiters is formatting noise, not a list split.
  const normalized = raw.replace(/\s*([/_-])\s*/g, "$1");
  const seen = new Set<string>();
  const out: string[] = [];
  for (const part of normalized.split(/[,\s]+/)) {
    const sym = part.trim().toUpperCase();
    if (!sym) continue;
    if (seen.has(sym)) continue;
    seen.add(sym);
    out.push(sym);
  }
  return out;
}
export {
  BINANCE_SYMBOL_PATTERN,
  COMMON_QUOTES,
  invalidSymbolsForPlatform,
  normalizeComboSymbol,
  sanitizeSymbolForPlatform,
  symbolFormatExample,
  symbolFormatPattern,
  trimBinanceComboSuffix,
};
export const EQUITY_TIPS = {
  preset: [
    'Use "Preset: Equity focus", then bump Trials/Timeout to widen the search.',
    'To maximize ROI, keep Objective/Tune objective on "roi"; reduce penalties only when you intentionally want higher risk.',
    "For short windows (e.g., 48h), shorter intervals (15m/30m/1h) increase sample size; keep Backtest + Tune ratios < 1.",
  ],
  trials: ["Higher Trials/Timeout expands the search (slower) and can improve ROI."],
  objective: ['Keep Objective and Tune objective on "roi" for risk-adjusted ROI ranking.'],
  penalties: ["Reduce or clear drawdown/turnover penalties to favor raw equity."],
  intervals: ["For short windows (e.g., 48h), shorter intervals (15m/30m/1h) increase sample size."],
  ratios: ["Keep Backtest ratio + Tune ratio < 1 to leave enough training data."],
};
export const COMPLEX_TIPS = {
  method: METHOD_TIPS,
  thresholds: [
    "Open threshold is the entry deadband; below break-even can churn after costs.",
    "Close threshold is often <= open threshold to reduce whipsaw.",
  ],
  edge: [
    "Min edge is the minimum predicted return to trade; cost-aware edge adds break-even + buffer.",
    "Edge buffer adds extra margin above break-even when cost-aware edge is on.",
  ],
  snr: ["Signal/vol (SNR) filters trades when predicted edge is small versus recent volatility."],
  blend: [
    "0 = LSTM only, 1 = Kalman only. Used with method=blend/conf_blend/conf_pick/conformal_clip/cost_pick/harmonic_blend/disagreement_guard/median_blend/neutral_guard/risk_parity_blend/consensus_boost/anchor_blend/tension_gate/entropy_blend/coherence_gate/divergence_gate/fractal_blend/phase_cancel/softmax_blend/smooth_softmax_blend/hedge_blend/net_softmax_blend/edge_blend/edge_pick/geo_blend/regime_switch.",
  ],
  router: ["Lookback controls how much recent history the router uses; longer is smoother but slower to adapt.", "Min score gates low-confidence periods to HOLD."],
  split: ["Backtest ratio is the held-out tail; tune ratio is only used for optimization/sweeps.", "Backtest + tune must be < 1 to leave training data."],
  lstm: ["Normalization affects scaling for LSTM only; keep consistent with training.", "Epochs/hidden size trade off fit vs runtime and overfitting."],
  optimization: ["Sweep thresholds searches open/close gates only.", "Optimize operations also tries methods and thresholds; router/bandit_router disable both."],
  tuneObjective: ["Tune objective defines the score used during fit/tune; it can differ from backtest objective."],
  walkForward: [
    "Walk-forward folds split data into sequential folds to estimate stability.",
    "Embargo bars drop samples near fold edges to reduce leakage.",
  ],
};

export function parseMaybeInt(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const n = Number(trimmed);
  // Integer-only request fields must round-trip exactly through JS numbers.
  if (!Number.isSafeInteger(n) || n < 0) return null;
  return Object.is(n, -0) ? 0 : n;
}

export function normalizeIsoInput(raw: string): string | null {
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

export function parseTimeInputMs(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (/^\d+$/.test(trimmed)) {
    const n = Number(trimmed);
    return Number.isSafeInteger(n) ? n : null;
  }
  const iso = normalizeIsoInput(trimmed);
  if (!iso) return null;
  const parsed = Date.parse(iso);
  if (Number.isNaN(parsed)) return null;
  return normalizedIsoMatchesParsedTime(iso, parsed) ? parsed : null;
}

function normalizedIsoMatchesParsedTime(iso: string, parsedMs: number): boolean {
  const match =
    /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,3}))?)?(?:(Z)|([+-])(\d{2}):(\d{2}))?$/.exec(iso);
  if (!match) return false;
  const [
    ,
    yearRaw,
    monthRaw,
    dayRaw,
    hourRaw,
    minuteRaw,
    secondRaw = "0",
    millisecondRaw = "0",
    zuluRaw,
    offsetSignRaw,
    offsetHourRaw = "0",
    offsetMinuteRaw = "0",
  ] = match;
  const expectedYear = Number(yearRaw);
  const expectedMonth = Number(monthRaw);
  const expectedDay = Number(dayRaw);
  const expectedHour = Number(hourRaw);
  const expectedMinute = Number(minuteRaw);
  const expectedSecond = Number(secondRaw);
  const expectedMillisecond = Number(millisecondRaw.padEnd(3, "0"));
  const offsetMinutes =
    zuluRaw === "Z"
      ? 0
      : offsetSignRaw
        ? (offsetSignRaw === "-" ? -1 : 1) * (Number(offsetHourRaw) * 60 + Number(offsetMinuteRaw))
        : null;
  const observed = new Date(offsetMinutes == null ? parsedMs : parsedMs + offsetMinutes * 60_000);
  const readPart = (useUtc: boolean, local: () => number, utc: () => number) => (useUtc ? utc() : local());
  const useUtc = offsetMinutes != null;
  return (
    readPart(useUtc, () => observed.getFullYear(), () => observed.getUTCFullYear()) === expectedYear &&
    readPart(useUtc, () => observed.getMonth() + 1, () => observed.getUTCMonth() + 1) === expectedMonth &&
    readPart(useUtc, () => observed.getDate(), () => observed.getUTCDate()) === expectedDay &&
    readPart(useUtc, () => observed.getHours(), () => observed.getUTCHours()) === expectedHour &&
    readPart(useUtc, () => observed.getMinutes(), () => observed.getUTCMinutes()) === expectedMinute &&
    readPart(useUtc, () => observed.getSeconds(), () => observed.getUTCSeconds()) === expectedSecond &&
    readPart(useUtc, () => observed.getMilliseconds(), () => observed.getUTCMilliseconds()) === expectedMillisecond
  );
}

export function sanitizeFilenameSegment(raw: string, fallback: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return fallback;
  const cleaned = trimmed.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-+/, "").replace(/-+$/, "");
  return cleaned || fallback;
}

export function csvEscape(value: unknown): string {
  if (value == null) return "";
  const text = String(value);
  if (text === "") return "";
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, "\"\"")}"` : text;
}

export const TRADE_PNL_EPS = 1e-9;
export const TRADE_PNL_TOP_N = 5;

export type TradePnlRow = {
  idx: number;
  entryIndex: number;
  exitIndex: number;
  entryEquity: number;
  exitEquity: number;
  return: number;
  holdingPeriods: number;
  exitReason: string | null;
  phase: string;
  pnl: number;
  entryTime: number | null;
  exitTime: number | null;
  entryIp: string | null;
  exitIp: string | null;
};

export type TradePnlAnalysis = {
  count: number;
  wins: number;
  losses: number;
  breakeven: number;
  winRate: number | null;
  avgWin: number | null;
  avgLoss: number | null;
  avgReturn: number | null;
  maxWin: number | null;
  maxLoss: number | null;
  totalWin: number;
  totalLoss: number;
  profitFactor: number | null;
  payoffRatio: number | null;
  avgHoldWin: number | null;
  avgHoldLoss: number | null;
  topWins: TradePnlRow[];
  topLosses: TradePnlRow[];
};

export type CommissionTotal = {
  asset: string;
  total: number;
  count: number;
};

export type BinancePnlRow = {
  idx: number;
  tradeId: number;
  orderId: number | null;
  time: number;
  entryTime: number | null;
  exitTime: number | null;
  symbol: string;
  side: string;
  price: number;
  qty: number;
  quoteQty: number;
  positionSide: string | null;
  realizedPnl: number;
  commission: number | null;
  commissionAsset: string | null;
  entryIp: string | null;
  exitIp: string | null;
};

export type BinancePnlAnalysis = {
  count: number;
  wins: number;
  losses: number;
  breakeven: number;
  winRate: number | null;
  avgWin: number | null;
  avgLoss: number | null;
  avgPnl: number | null;
  maxWin: number | null;
  maxLoss: number | null;
  totalWin: number;
  totalLoss: number;
  totalPnl: number;
  totalQty: number;
  totalQuoteQty: number;
  profitFactor: number | null;
  payoffRatio: number | null;
  commissionTotals: CommissionTotal[];
  topWins: BinancePnlRow[];
  topLosses: BinancePnlRow[];
};

export function pnlBadgeClass(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "badge";
  if (value > TRADE_PNL_EPS) return "badge badgeStrong badgeLong";
  if (value < -TRADE_PNL_EPS) return "badge badgeStrong badgeFlat";
  return "badge badgeHold";
}

export function binanceTradeSideLabel(trade: BinanceTrade): "BUY" | "SELL" | "—" {
  const raw = trade.side?.toUpperCase();
  if (raw === "BUY" || raw === "SELL") return raw;
  if (trade.isBuyer === true) return "BUY";
  if (trade.isBuyer === false) return "SELL";
  return "—";
}

export type BinanceTradeIpMeta = {
  entryIp: string | null;
  exitIp: string | null;
  entryTime: number | null;
  exitTime: number | null;
};

type TradeLot = {
  qty: number;
  ip: string | null;
  openedAtMs: number | null;
  tradeKey: string | null;
};

type ConsumedTradeLot = {
  ip: string | null;
  openedAtMs: number | null;
  tradeKey: string | null;
  fullyClosed: boolean;
};

const BINANCE_TRADE_IP_EPS = 1e-12;

export function binanceTradeKey(trade: BinanceTrade): string {
  return `${normalizeSymbolKey(trade.symbol)}-${trade.tradeId}`;
}

export function isLikelyBinanceCloseFill(trade: BinanceTrade): boolean {
  const pnl = trade.realizedPnl;
  return typeof pnl === "number" && Number.isFinite(pnl) && Math.abs(pnl) > BINANCE_TRADE_IP_EPS;
}

function normalizeTradeIp(raw?: string | null): string | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  return trimmed ? trimmed : null;
}

function normalizeTradeTime(raw: unknown): number | null {
  return typeof raw === "number" && Number.isFinite(raw) ? raw : null;
}

function pushLot(
  store: Map<string, TradeLot[]>,
  key: string,
  qty: number,
  ip: string | null,
  openedAtMs: number | null,
  tradeKey: string | null,
): void {
  if (!Number.isFinite(qty) || qty <= BINANCE_TRADE_IP_EPS) return;
  const lots = store.get(key) ?? [];
  lots.push({ qty, ip, openedAtMs, tradeKey });
  store.set(key, lots);
}

function consumeLots(store: Map<string, TradeLot[]>, key: string, qty: number): ConsumedTradeLot[] {
  if (!Number.isFinite(qty) || qty <= BINANCE_TRADE_IP_EPS) return [];
  const lots = store.get(key);
  if (!lots || lots.length === 0) return [];
  let remaining = qty;
  const nextLots: TradeLot[] = [];
  const consumed: ConsumedTradeLot[] = [];
  for (const lot of lots) {
    if (remaining <= BINANCE_TRADE_IP_EPS) {
      nextLots.push(lot);
      continue;
    }
    const take = Math.min(lot.qty, remaining);
    remaining -= take;
    const leftover = lot.qty - take;
    const fullyClosed = leftover <= BINANCE_TRADE_IP_EPS;
    consumed.push({ ip: lot.ip, openedAtMs: lot.openedAtMs, tradeKey: lot.tradeKey, fullyClosed });
    if (leftover > BINANCE_TRADE_IP_EPS) {
      nextLots.push({ ...lot, qty: leftover });
    }
  }
  if (nextLots.length > 0) {
    store.set(key, nextLots);
  } else {
    store.delete(key);
  }
  return consumed;
}

function joinIps(ips: string[]): string | null {
  if (ips.length === 0) return null;
  return ips.join(" • ");
}

function consumedIps(consumed: ConsumedTradeLot[]): string[] {
  return Array.from(new Set(consumed.map((item) => item.ip).filter((ip): ip is string => Boolean(ip))));
}

function consumedOpenTime(consumed: ConsumedTradeLot[]): number | null {
  let out: number | null = null;
  for (const item of consumed) {
    const t = item.openedAtMs;
    if (typeof t !== "number" || !Number.isFinite(t)) continue;
    out = out === null ? t : Math.min(out, t);
  }
  return out;
}

function applyExitToConsumedLots(
  meta: Map<string, BinanceTradeIpMeta>,
  consumed: ConsumedTradeLot[],
  exitIp: string | null,
  exitTime: number | null,
): void {
  if (!exitIp && exitTime == null) return;
  for (const item of consumed) {
    if (!item.fullyClosed) continue;
    const tradeKey = item.tradeKey;
    if (!tradeKey) continue;
    const existing = meta.get(tradeKey);
    if (!existing) continue;
    if (existing.exitIp || existing.exitTime != null) continue;
    meta.set(tradeKey, { ...existing, exitIp: exitIp ?? existing.exitIp, exitTime: exitTime ?? existing.exitTime });
  }
}

export function buildBinanceTradeIpMap(trades: BinanceTrade[]): Map<string, BinanceTradeIpMeta> {
  const meta = new Map<string, BinanceTradeIpMeta>();
  if (trades.length === 0) return meta;
  const sorted = [...trades].sort((a, b) => {
    const aTime = Number.isFinite(a.time) ? a.time : Number.POSITIVE_INFINITY;
    const bTime = Number.isFinite(b.time) ? b.time : Number.POSITIVE_INFINITY;
    if (aTime !== bTime) return aTime - bTime;
    const aTradeId = Number.isFinite(a.tradeId) ? a.tradeId : 0;
    const bTradeId = Number.isFinite(b.tradeId) ? b.tradeId : 0;
    return aTradeId - bTradeId;
  });
  const longLots = new Map<string, TradeLot[]>();
  const shortLots = new Map<string, TradeLot[]>();
  const netPos = new Map<string, number>();

  for (const trade of sorted) {
    const key = binanceTradeKey(trade);
    const side = binanceTradeSideLabel(trade);
    const qty = trade.qty;
    const tradeTime = normalizeTradeTime(trade.time);
    if (side === "—" || !Number.isFinite(qty) || qty <= BINANCE_TRADE_IP_EPS) {
      meta.set(key, { entryIp: null, exitIp: null, entryTime: null, exitTime: null });
      continue;
    }

    const symbolKey = normalizeSymbolKey(trade.symbol);
    const posSide = normalizePositionSide(trade.positionSide) ?? "BOTH";
    const orderIp = normalizeTradeIp(trade.originIp);
    let entryIp: string | null = null;
    let exitIp: string | null = null;
    let entryTime: number | null = null;
    let exitTime: number | null = null;

    if (posSide === "LONG") {
      const lotKey = `${symbolKey}::LONG`;
      if (side === "BUY") {
        pushLot(longLots, lotKey, qty, orderIp, tradeTime, key);
        entryIp = orderIp;
        entryTime = tradeTime;
      } else if (side === "SELL") {
        const consumed = consumeLots(longLots, lotKey, qty);
        entryIp = joinIps(consumedIps(consumed));
        exitIp = orderIp;
        entryTime = consumedOpenTime(consumed);
        exitTime = tradeTime;
        applyExitToConsumedLots(meta, consumed, orderIp, tradeTime);
      }
    } else if (posSide === "SHORT") {
      const lotKey = `${symbolKey}::SHORT`;
      if (side === "SELL") {
        pushLot(shortLots, lotKey, qty, orderIp, tradeTime, key);
        entryIp = orderIp;
        entryTime = tradeTime;
      } else if (side === "BUY") {
        const consumed = consumeLots(shortLots, lotKey, qty);
        entryIp = joinIps(consumedIps(consumed));
        exitIp = orderIp;
        entryTime = consumedOpenTime(consumed);
        exitTime = tradeTime;
        applyExitToConsumedLots(meta, consumed, orderIp, tradeTime);
      }
    } else {
      const netKey = `${symbolKey}::BOTH`;
      const net = netPos.get(netKey) ?? 0;
      const longKey = `${symbolKey}::BOTH:LONG`;
      const shortKey = `${symbolKey}::BOTH:SHORT`;

      if (side === "BUY") {
        if (net >= 0) {
          pushLot(longLots, longKey, qty, orderIp, tradeTime, key);
          entryIp = orderIp;
          entryTime = tradeTime;
        } else {
          const closeQty = Math.min(qty, Math.abs(net));
          if (closeQty > BINANCE_TRADE_IP_EPS) {
            const consumed = consumeLots(shortLots, shortKey, closeQty);
            entryIp = joinIps(consumedIps(consumed));
            exitIp = orderIp;
            entryTime = consumedOpenTime(consumed);
            exitTime = tradeTime;
            applyExitToConsumedLots(meta, consumed, orderIp, tradeTime);
          }
          const openQty = qty - closeQty;
          if (openQty > BINANCE_TRADE_IP_EPS) {
            pushLot(longLots, longKey, openQty, orderIp, tradeTime, key);
            if (!entryIp && entryTime == null) {
              entryIp = orderIp;
              entryTime = tradeTime;
            }
          }
        }
        netPos.set(netKey, net + qty);
      } else {
        if (net <= 0) {
          pushLot(shortLots, shortKey, qty, orderIp, tradeTime, key);
          entryIp = orderIp;
          entryTime = tradeTime;
        } else {
          const closeQty = Math.min(qty, net);
          if (closeQty > BINANCE_TRADE_IP_EPS) {
            const consumed = consumeLots(longLots, longKey, closeQty);
            entryIp = joinIps(consumedIps(consumed));
            exitIp = orderIp;
            entryTime = consumedOpenTime(consumed);
            exitTime = tradeTime;
            applyExitToConsumedLots(meta, consumed, orderIp, tradeTime);
          }
          const openQty = qty - closeQty;
          if (openQty > BINANCE_TRADE_IP_EPS) {
            pushLot(shortLots, shortKey, openQty, orderIp, tradeTime, key);
            if (!entryIp && entryTime == null) {
              entryIp = orderIp;
              entryTime = tradeTime;
            }
          }
        }
        netPos.set(netKey, net - qty);
      }
    }

    // When we cannot pair inventory from the visible window, non-zero realized PnL
    // still indicates a close fill, so keep its close timestamp/IP.
    if (exitTime == null && isLikelyBinanceCloseFill(trade)) {
      exitIp = exitIp ?? orderIp;
      exitTime = tradeTime;
    }

    meta.set(key, { entryIp, exitIp, entryTime, exitTime });
  }

  return meta;
}

export type PositionOpenTimeEstimate = {
  openedAtMs: number | null;
  isLowerBound: boolean;
};

function tradeDeltaForPosition(trade: BinanceTrade, posSide: "LONG" | "SHORT"): number {
  const side = binanceTradeSideLabel(trade);
  if (side === "—") return 0;
  const qty = trade.qty;
  if (!Number.isFinite(qty) || qty <= 0) return 0;
  const tradeSide = normalizePositionSide(trade.positionSide);
  if (tradeSide && tradeSide !== "BOTH" && tradeSide !== posSide) return 0;
  if (posSide === "LONG") return side === "BUY" ? qty : -qty;
  return side === "SELL" ? -qty : qty;
}

export function inferBinancePositionOpenTime(
  position: ApiBinancePositionsResponse["positions"][number],
  trades: BinanceTrade[],
): PositionOpenTimeEstimate | null {
  const posAmt = position.positionAmt;
  if (!Number.isFinite(posAmt) || isEffectivelyFlatPositionAmount(posAmt)) return null;
  const sideRaw = normalizePositionSide(position.positionSide);
  const posSide = sideRaw && sideRaw !== "BOTH" ? sideRaw : positionSideFromAmount(posAmt);
  if (!posSide) return null;
  const symKey = normalizeSymbolKey(position.symbol);
  const symTrades = trades.filter((trade) => normalizeSymbolKey(trade.symbol) === symKey);
  if (symTrades.length === 0) return null;
  const sorted = [...symTrades].sort((a, b) => b.time - a.time);
  let remaining = (posSide === "LONG" ? 1 : -1) * Math.abs(posAmt);
  let oldestRelevant: number | null = null;
  for (const trade of sorted) {
    if (!Number.isFinite(trade.time)) continue;
    const delta = tradeDeltaForPosition(trade, posSide);
    if (!Number.isFinite(delta) || delta === 0) continue;
    oldestRelevant = oldestRelevant === null ? trade.time : Math.min(oldestRelevant, trade.time);
    const next = remaining - delta;
    if (Math.sign(remaining) !== 0 && (next === 0 || Math.sign(next) !== Math.sign(remaining))) {
      return { openedAtMs: trade.time, isLowerBound: false };
    }
    remaining = next;
  }
  if (oldestRelevant !== null) return { openedAtMs: oldestRelevant, isLowerBound: true };
  return null;
}

export function backtestTradePhase(split: BacktestResponse["split"], entryIndex: number): string {
  if (entryIndex >= split.backtestStartIndex) return "backtest";
  if (split.tune > 0 && entryIndex >= split.tuneStartIndex) return "tune";
  return split.tune > 0 ? "fit" : "train";
}

export function buildBacktestOpsCsv(backtest: BacktestResponse): string {
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
    "entryIp",
    "exitIp",
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
      trade.entryIp ?? "",
      trade.exitIp ?? "",
    ]
      .map(csvEscape)
      .join(",");
  });
  return [header, ...rows].join("\n");
}

export function buildBacktestTradePnlAnalysis(backtest: BacktestResponse): TradePnlAnalysis | null {
  const trades = backtest.trades ?? [];
  if (trades.length === 0) return null;
  const openTimes = Array.isArray(backtest.openTimes) ? backtest.openTimes : null;
  const rows: TradePnlRow[] = trades
    .map((trade, idx) => {
      if (!Number.isFinite(trade.return)) return null;
      const entryTimeRaw = openTimes?.[trade.entryIndex];
      const exitTimeRaw = openTimes?.[trade.exitIndex];
      const entryTime = typeof entryTimeRaw === "number" && Number.isFinite(entryTimeRaw) ? entryTimeRaw : null;
      const exitTime = typeof exitTimeRaw === "number" && Number.isFinite(exitTimeRaw) ? exitTimeRaw : null;
      return {
        idx: idx + 1,
        entryIndex: trade.entryIndex,
        exitIndex: trade.exitIndex,
        entryEquity: trade.entryEquity,
        exitEquity: trade.exitEquity,
        return: trade.return,
        holdingPeriods: trade.holdingPeriods,
        exitReason: trade.exitReason ?? null,
        phase: backtestTradePhase(backtest.split, trade.entryIndex),
        pnl: trade.exitEquity - trade.entryEquity,
        entryTime,
        exitTime,
        entryIp: trade.entryIp ?? null,
        exitIp: trade.exitIp ?? null,
      };
    })
    .filter((row): row is TradePnlRow => Boolean(row));
  if (rows.length === 0) return null;

  let wins = 0;
  let losses = 0;
  let breakeven = 0;
  let sumWin = 0;
  let sumLoss = 0;
  let sumReturn = 0;
  let maxWin: number | null = null;
  let maxLoss: number | null = null;
  let holdWin = 0;
  let holdLoss = 0;

  for (const row of rows) {
    const r = row.return;
    sumReturn += r;
    if (r > TRADE_PNL_EPS) {
      wins += 1;
      sumWin += r;
      holdWin += row.holdingPeriods;
      maxWin = maxWin === null ? r : Math.max(maxWin, r);
    } else if (r < -TRADE_PNL_EPS) {
      losses += 1;
      sumLoss += r;
      holdLoss += row.holdingPeriods;
      maxLoss = maxLoss === null ? r : Math.min(maxLoss, r);
    } else {
      breakeven += 1;
    }
  }

  const count = rows.length;
  const avgWin = wins > 0 ? sumWin / wins : null;
  const avgLoss = losses > 0 ? sumLoss / losses : null;
  const avgReturn = count > 0 ? sumReturn / count : null;
  const winRate = count > 0 ? wins / count : null;
  const profitFactor = sumLoss < 0 ? sumWin / Math.abs(sumLoss) : sumWin > 0 ? Infinity : null;
  const payoffRatio = avgWin !== null && avgLoss !== null && avgLoss !== 0 ? avgWin / Math.abs(avgLoss) : null;
  const avgHoldWin = wins > 0 ? holdWin / wins : null;
  const avgHoldLoss = losses > 0 ? holdLoss / losses : null;
  const topWins = rows
    .filter((row) => row.return > TRADE_PNL_EPS)
    .sort((a, b) => b.return - a.return)
    .slice(0, TRADE_PNL_TOP_N);
  const topLosses = rows
    .filter((row) => row.return < -TRADE_PNL_EPS)
    .sort((a, b) => a.return - b.return)
    .slice(0, TRADE_PNL_TOP_N);

  return {
    count,
    wins,
    losses,
    breakeven,
    winRate,
    avgWin,
    avgLoss,
    avgReturn,
    maxWin,
    maxLoss,
    totalWin: sumWin,
    totalLoss: sumLoss,
    profitFactor,
    payoffRatio,
    avgHoldWin,
    avgHoldLoss,
    topWins,
    topLosses,
  };
}

export function buildBinanceTradePnlAnalysis(trades: BinanceTrade[]): BinancePnlAnalysis | null {
  const rows: BinancePnlRow[] = [];
  const commissionByAsset = new Map<string, CommissionTotal>();
  for (let i = 0; i < trades.length; i += 1) {
    const trade = trades[i];
    if (!trade) continue;
    const pnl = trade.realizedPnl;
    if (typeof pnl !== "number" || !Number.isFinite(pnl)) continue;
    const side = binanceTradeSideLabel(trade);
    const commission = trade.commission;
    const commissionAsset = trade.commissionAsset ?? null;
    const entryTime = typeof trade.entryTime === "number" && Number.isFinite(trade.entryTime) ? trade.entryTime : null;
    const exitTime = typeof trade.exitTime === "number" && Number.isFinite(trade.exitTime) ? trade.exitTime : null;
    if (typeof commission === "number" && Number.isFinite(commission)) {
      const assetKey = commissionAsset ?? "unknown";
      const existing = commissionByAsset.get(assetKey);
      if (existing) {
        existing.total += commission;
        existing.count += 1;
      } else {
        commissionByAsset.set(assetKey, { asset: assetKey, total: commission, count: 1 });
      }
    }
    rows.push({
      idx: i + 1,
      tradeId: trade.tradeId,
      orderId: trade.orderId ?? null,
      time: trade.time,
      entryTime,
      exitTime,
      symbol: trade.symbol,
      side,
      price: trade.price,
      qty: trade.qty,
      quoteQty: trade.quoteQty,
      positionSide: trade.positionSide ?? null,
      realizedPnl: pnl,
      commission: commission ?? null,
      commissionAsset,
      entryIp: trade.entryIp ?? null,
      exitIp: trade.exitIp ?? null,
    });
  }

  if (rows.length === 0) return null;

  let wins = 0;
  let losses = 0;
  let breakeven = 0;
  let sumWin = 0;
  let sumLoss = 0;
  let sumPnl = 0;
  let sumQty = 0;
  let sumQuoteQty = 0;
  let maxWin: number | null = null;
  let maxLoss: number | null = null;

  for (const row of rows) {
    const qty = row.qty;
    if (Number.isFinite(qty)) sumQty += qty;
    const quoteQty = row.quoteQty;
    if (Number.isFinite(quoteQty)) sumQuoteQty += quoteQty;
    const pnl = row.realizedPnl;
    sumPnl += pnl;
    if (pnl > TRADE_PNL_EPS) {
      wins += 1;
      sumWin += pnl;
      maxWin = maxWin === null ? pnl : Math.max(maxWin, pnl);
    } else if (pnl < -TRADE_PNL_EPS) {
      losses += 1;
      sumLoss += pnl;
      maxLoss = maxLoss === null ? pnl : Math.min(maxLoss, pnl);
    } else {
      breakeven += 1;
    }
  }

  const count = rows.length;
  const avgWin = wins > 0 ? sumWin / wins : null;
  const avgLoss = losses > 0 ? sumLoss / losses : null;
  const avgPnl = count > 0 ? sumPnl / count : null;
  const winRate = count > 0 ? wins / count : null;
  const profitFactor = sumLoss < 0 ? sumWin / Math.abs(sumLoss) : sumWin > 0 ? Infinity : null;
  const payoffRatio = avgWin !== null && avgLoss !== null && avgLoss !== 0 ? avgWin / Math.abs(avgLoss) : null;
  const topWins = rows
    .filter((row) => row.realizedPnl > TRADE_PNL_EPS)
    .sort((a, b) => b.realizedPnl - a.realizedPnl)
    .slice(0, TRADE_PNL_TOP_N);
  const topLosses = rows
    .filter((row) => row.realizedPnl < -TRADE_PNL_EPS)
    .sort((a, b) => a.realizedPnl - b.realizedPnl)
    .slice(0, TRADE_PNL_TOP_N);
  const commissionTotals = Array.from(commissionByAsset.values()).sort((a, b) => a.asset.localeCompare(b.asset));

  return {
    count,
    wins,
    losses,
    breakeven,
    winRate,
    avgWin,
    avgLoss,
    avgPnl,
    maxWin,
    maxLoss,
    totalWin: sumWin,
    totalLoss: sumLoss,
    totalPnl: sumPnl,
    totalQty: sumQty,
    totalQuoteQty: sumQuoteQty,
    profitFactor,
    payoffRatio,
    commissionTotals,
    topWins,
    topLosses,
  };
}

export function downloadTextFile(filename: string, contents: string, contentType = "text/plain"): void {
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

export function buildPositionSeries(prices: number[], side: number, entryPrice?: number | null): number[] {
  if (prices.length === 0) return [];
  if (!Number.isFinite(side) || side === 0) return Array.from({ length: prices.length }, () => 0);
  const dir = side > 0 ? 1 : -1;
  const entry =
    typeof entryPrice === "number" && Number.isFinite(entryPrice) && entryPrice > 0 ? entryPrice : null;
  if (entry == null) return Array.from({ length: prices.length }, () => dir);

  let bestIdx = 0;
  let bestDiff = Number.POSITIVE_INFINITY;
  for (let i = 0; i < prices.length; i += 1) {
    const p = prices[i] ?? Number.NaN;
    if (!Number.isFinite(p)) continue;
    const diff = Math.abs(p - entry);
    if (diff < bestDiff || (diff === bestDiff && i > bestIdx)) {
      bestDiff = diff;
      bestIdx = i;
    }
  }

  return prices.map((_, i) => (i < bestIdx ? 0 : dir));
}

export function buildEquityCurve(prices: number[], side: number): number[] {
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

export function positionSideInfo(positionAmt: number, positionSide?: string | null): { dir: number; label: string; key: string } {
  if (!Number.isFinite(positionAmt) || isEffectivelyFlatPositionAmount(positionAmt)) {
    return { dir: 0, label: "FLAT", key: "FLAT" };
  }
  const raw = positionSide?.trim().toUpperCase();
  const side = raw && raw !== "BOTH" ? raw : null;
  const dir = side === "SHORT" ? -1 : side === "LONG" ? 1 : positionAmt > 0 ? 1 : positionAmt < 0 ? -1 : 0;
  const label = side ?? (dir > 0 ? "LONG" : dir < 0 ? "SHORT" : "FLAT");
  const key = side ?? (dir > 0 ? "LONG" : dir < 0 ? "SHORT" : "FLAT");
  return { dir, label, key };
}

export function isOpenBinancePosition(position: ApiBinancePositionsResponse["positions"][number]): boolean {
  return positionSideInfo(position.positionAmt, position.positionSide).dir !== 0;
}

export function buildOpenBinancePositionSymbolSet(positions: ApiBinancePositionsResponse["positions"]): Set<string> {
  const out = new Set<string>();
  for (const position of positions) {
    if (isOpenBinancePosition(position)) out.add(normalizeSymbolKey(position.symbol));
  }
  return out;
}

export type ListenKeyStreamStatus = "disconnected" | "connecting" | "connected" | "stopped";

export type ListenKeyStreamStatusPayload = { status?: string; message?: string; atMs?: number };
export type ListenKeyStreamKeepAlivePayload = { atMs?: number };
export type ListenKeyStreamErrorPayload = { message?: string; atMs?: number };

export function normalizeListenKeyStreamStatus(raw: string): ListenKeyStreamStatus {
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

export function safeJsonParse<T = unknown>(raw: string): T | null {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

// Minimal SSE parser for fetch streams that handles chunk boundaries.
export function createSseParser(onEvent: (event: string, data: string) => void): (chunk: string) => void {
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

export function parseOptionalNumber(raw: string): number | undefined {
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  const parsed = numFromInput(trimmed, Number.NaN);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function parseOptionalInt(raw: string): number | undefined {
  const parsed = parseOptionalNumber(raw);
  if (parsed == null || !Number.isSafeInteger(parsed)) return undefined;
  return parsed;
}

export type OptionalWholeNumberField = {
  label: string;
  raw: string;
  override?: unknown;
};

const OPTIMIZER_EXTRA_WHOLE_NUMBER_KEYS = [
  "barsMin",
  "barsMax",
  "trials",
  "seed",
  "seedTrials",
  "perturbScaleInt",
  "earlyStopNoImprove",
  "epochsMin",
  "epochsMax",
  "hiddenSizeMin",
  "hiddenSizeMax",
  "patienceMax",
  "minRoundTrips",
  "walkForwardFoldsMin",
  "walkForwardFoldsMax",
  "walkForwardEmbargoBarsMin",
  "walkForwardEmbargoBarsMax",
  "minHoldBarsMin",
  "minHoldBarsMax",
  "cooldownBarsMin",
  "cooldownBarsMax",
  "maxHoldBarsMin",
  "maxHoldBarsMax",
  "trendLookbackMin",
  "trendLookbackMax",
] as const;

const OPTIMIZER_EXTRA_FINITE_NUMBER_KEYS = ["timeoutSec", "backtestRatio", "tuneRatio"] as const;

const OPTIMIZER_EXTRA_TRIMMED_STRING_KEYS = [
  "data",
  "priceColumn",
  "highColumn",
  "lowColumn",
  "intervals",
  "platforms",
  "lookbackWindow",
  "objective",
  "tuneObjective",
  "normalizations",
] as const;

function readOptionalTrimmedStringOverride(raw: unknown): { provided: boolean; value: string | null } {
  if (raw == null) return { provided: false, value: null };
  if (typeof raw !== "string") return { provided: true, value: null };
  const trimmed = raw.trim();
  if (!trimmed) return { provided: false, value: null };
  return { provided: true, value: trimmed };
}

function readOptionalFiniteNumberOverride(raw: unknown): { provided: boolean; value: number | null } {
  if (raw == null) return { provided: false, value: null };
  if (typeof raw === "number") {
    return { provided: true, value: Number.isFinite(raw) ? raw : null };
  }
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!trimmed) return { provided: false, value: null };
    return { provided: true, value: parseOptionalNumber(trimmed) ?? null };
  }
  return { provided: true, value: null };
}

function readOptionalWholeNumberOverride(raw: unknown): { provided: boolean; value: number | null } {
  if (raw == null) return { provided: false, value: null };
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!trimmed) return { provided: false, value: null };
    return { provided: true, value: parseOptionalInt(trimmed) ?? null };
  }
  if (typeof raw === "number") {
    return { provided: true, value: Number.isSafeInteger(raw) ? raw : null };
  }
  return { provided: true, value: null };
}

function normalizeOptimizerSourceOverride(raw: unknown): OptimizerSource | null {
  const override = readOptionalTrimmedStringOverride(raw);
  if (!override.provided || override.value == null) return null;
  switch (override.value.toLowerCase()) {
    case "binance":
    case "coinbase":
    case "kraken":
    case "poloniex":
    case "csv":
      return override.value.toLowerCase() as OptimizerSource;
    default:
      return null;
  }
}

function normalizeKnownOptimizerRunExtras(extras: Record<string, unknown>): Record<string, unknown> {
  const normalized: Record<string, unknown> = { ...extras };
  for (const key of OPTIMIZER_EXTRA_WHOLE_NUMBER_KEYS) {
    const override = readOptionalWholeNumberOverride(normalized[key]);
    if (!override.provided || override.value == null) {
      delete normalized[key];
      continue;
    }
    normalized[key] = override.value;
  }
  for (const key of OPTIMIZER_EXTRA_FINITE_NUMBER_KEYS) {
    const override = readOptionalFiniteNumberOverride(normalized[key]);
    if (!override.provided || override.value == null) {
      delete normalized[key];
      continue;
    }
    normalized[key] = override.value;
  }
  for (const key of OPTIMIZER_EXTRA_TRIMMED_STRING_KEYS) {
    const override = readOptionalTrimmedStringOverride(normalized[key]);
    if (!override.provided || override.value == null) {
      delete normalized[key];
      continue;
    }
    normalized[key] = override.value;
  }
  const source = normalizeOptimizerSourceOverride(normalized.source);
  if (source == null) delete normalized.source;
  else normalized.source = source;
  const symbolOverride = readOptionalTrimmedStringOverride(normalized.binanceSymbol);
  if (!symbolOverride.provided || symbolOverride.value == null) delete normalized.binanceSymbol;
  else normalized.binanceSymbol = symbolOverride.value.toUpperCase();
  return normalized;
}

const CSV_ONLY_OPTIMIZER_REQUEST_KEYS = ["data", "priceColumn", "highColumn", "lowColumn"] as const;
const EXCHANGE_ONLY_OPTIMIZER_REQUEST_KEYS = ["binanceSymbol", "platforms"] as const;

function enforceOptimizerRequestSourceCompatibility(req: OptimizerRunRequest): void {
  if (req.source === "csv") {
    for (const key of EXCHANGE_ONLY_OPTIMIZER_REQUEST_KEYS) delete req[key];
    return;
  }
  for (const key of CSV_ONLY_OPTIMIZER_REQUEST_KEYS) delete req[key];
}

export function findOptionalWholeNumberFieldError(fields: OptionalWholeNumberField[]): string | null {
  for (const field of fields) {
    const override = readOptionalWholeNumberOverride(field.override);
    if (override.provided) {
      if (override.value == null) {
        return `${field.label} must be a whole number.`;
      }
      continue;
    }
    const trimmed = field.raw.trim();
    if (!trimmed) continue;
    if (parseOptionalInt(trimmed) == null) {
      return `${field.label} must be a whole number.`;
    }
  }
  return null;
}

export function parseOptionalString(raw: string): string | undefined {
  const trimmed = raw.trim();
  return trimmed ? trimmed : undefined;
}

export type UiState = {
  loading: boolean;
  error: string | null;
  lastKind: RequestKind | null;
  latestSignal: LatestSignal | null;
  backtest: BacktestResponse | null;
  trade: ApiTradeResponse | null;
};

export type ErrorFix =
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

export type BotUiState = {
  loading: boolean;
  error: string | null;
  status: BotStatus;
};

export type BotRtEvent = {
  atMs: number;
  message: string;
};

export type BotTelemetryPoint = {
  atMs: number;
  pollLatencyMs: number | null;
  driftBps: number | null;
};

export type BotRtUiState = {
  lastFetchAtMs: number | null;
  lastFetchDurationMs: number | null;
  lastNewCandles: number;
  lastNewCandlesAtMs: number | null;
  lastKlineUpdates: number;
  lastKlineUpdatesAtMs: number | null;
  telemetry: BotTelemetryPoint[];
  feed: BotRtEvent[];
};

export type BotRtTracker = {
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

export const emptyBotRtState = (): BotRtUiState => ({
  lastFetchAtMs: null,
  lastFetchDurationMs: null,
  lastNewCandles: 0,
  lastNewCandlesAtMs: null,
  lastKlineUpdates: 0,
  lastKlineUpdatesAtMs: null,
  telemetry: [],
  feed: [],
});

export const emptyBotRtTracker = (): BotRtTracker => ({
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

export type KeysUiState = {
  loading: boolean;
  error: string | null;
  status: KeysStatus | null;
  platform: Platform | null;
  checkedAtMs: number | null;
};

export type CacheUiState = {
  loading: boolean;
  error: string | null;
  stats: Awaited<ReturnType<typeof cacheStats>> | null;
};

export type ListenKeyUiState = {
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

export type BinanceTradesUiState = {
  loading: boolean;
  error: string | null;
  response: ApiBinanceTradesResponse | null;
};

export type BinancePositionsUiState = {
  loading: boolean;
  error: string | null;
  response: ApiBinancePositionsResponse | null;
};

export type OptimizerRunUiState = {
  loading: boolean;
  error: string | null;
  response: OptimizerRunResponse | null;
  lastRunAtMs: number | null;
};

export type OptimizerRunForm = {
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
  walkForwardEmbargoBarsMin: string;
  walkForwardEmbargoBarsMax: string;
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
  rebalanceCostMultMin: string;
  rebalanceCostMultMax: string;
  pCostAwareEdge: string;
  stopMin: string;
  stopMax: string;
  tpMin: string;
  tpMax: string;
  trailMin: string;
  trailMax: string;
  methodWeightBlend: string;
  methodWeightConfBlend: string;
  methodWeightConfPick: string;
  methodWeightConformalClip: string;
  methodWeightCostPick: string;
  methodWeightHarmonicBlend: string;
  methodWeightDisagreementGuard: string;
  methodWeightMedianBlend: string;
  methodWeightNeutralGuard: string;
  methodWeightRiskParityBlend: string;
  methodWeightConsensusBoost: string;
  methodWeightAnchorBlend: string;
  methodWeightTensionGate: string;
  methodWeightEntropyBlend: string;
  methodWeightCoherenceGate: string;
  methodWeightDivergenceGate: string;
  methodWeightFractalBlend: string;
  methodWeightPhaseCancel: string;
  methodWeightSoftmaxBlend: string;
  methodWeightSmoothSoftmaxBlend: string;
  methodWeightHedgeBlend: string;
  methodWeightNetSoftmaxBlend: string;
  methodWeightEdgeBlend: string;
  methodWeightEdgePick: string;
  methodWeightGeoBlend: string;
  methodWeightRegimeSwitch: string;
  methodWeightBanditRouter: string;
  blendWeightMin: string;
  blendWeightMax: string;
  disableLstmPersistence: boolean;
  noSweepThreshold: boolean;
  extraJson: string;
};

export type TopCombosSource = "api" | "repo" | "cache";

export type TopCombosMeta = {
  source: TopCombosSource;
  generatedAtMs: number | null;
  payloadSource: string | null;
  payloadSources: string[] | null;
  fallbackReason: string | null;
  comboCount: number | null;
  rawCount: number | null;
  droppedCount: number | null;
  dedupedCount: number | null;
};

export type ComboOrder = "annualized-equity" | "rank" | "date-desc" | "date-asc";

type OptimizerRunFormTextKey = {
  [K in keyof OptimizerRunForm]: OptimizerRunForm[K] extends string ? K : never;
}[keyof OptimizerRunForm];

type OptimizerCorrelationPoint = {
  x: number;
  roi: number;
};

type OptimizerCorrelationRangeConfig = {
  min?: OptimizerRunFormTextKey;
  max?: OptimizerRunFormTextKey;
  extraMin?: string;
  extraMax?: string;
  integer?: boolean;
  lowerBound?: number;
  upperBound?: number;
  widthRatio?: number;
};

type OptimizerCorrelationStat = {
  key: string;
  label: string;
  correlation: number;
  sampleCount: number;
  target: number;
  xMin: number;
  xMax: number;
};

export type OptimizerCorrelationGuess = {
  patch: Partial<OptimizerRunForm>;
  extras: Record<string, number>;
  basis: string[];
  sampleCount: number;
  correlationCount: number;
};

const OPTIMIZER_GUESS_MIN_SAMPLES = 4;
const OPTIMIZER_GUESS_MIN_ABS_CORRELATION = 0.18;
const OPTIMIZER_GUESS_MAX_FIELDS = 6;
const OPTIMIZER_GUESS_TOP_ROI_FRACTION = 0.25;
const OPTIMIZER_GUESS_DEFAULT_WIDTH_RATIO = 0.22;

const OPTIMIZER_CORRELATION_RANGE_CONFIGS: Record<string, OptimizerCorrelationRangeConfig> = {
  bars: { min: "barsMin", max: "barsMax", integer: true, lowerBound: 1 },
  epochs: { min: "epochsMin", max: "epochsMax", integer: true, lowerBound: 1 },
  hiddenSize: { min: "hiddenSizeMin", max: "hiddenSizeMax", integer: true, lowerBound: 1 },
  learningRate: { min: "lrMin", max: "lrMax", lowerBound: 1e-12, widthRatio: 0.3 },
  patience: { max: "patienceMax", integer: true, lowerBound: 0 },
  gradClip: { min: "gradClipMin", max: "gradClipMax", lowerBound: 0 },
  slippage: { max: "slippageMax", lowerBound: 0 },
  spread: { max: "spreadMax", lowerBound: 0 },
  minHoldBars: { min: "minHoldBarsMin", max: "minHoldBarsMax", integer: true, lowerBound: 0 },
  cooldownBars: { min: "cooldownBarsMin", max: "cooldownBarsMax", integer: true, lowerBound: 0 },
  maxHoldBars: { min: "maxHoldBarsMin", max: "maxHoldBarsMax", integer: true, lowerBound: 0 },
  minEdge: { min: "minEdgeMin", max: "minEdgeMax", lowerBound: 0 },
  minSignalToNoise: { min: "minSignalToNoiseMin", max: "minSignalToNoiseMax", lowerBound: 0 },
  edgeBuffer: { min: "edgeBufferMin", max: "edgeBufferMax", lowerBound: 0 },
  trendLookback: { min: "trendLookbackMin", max: "trendLookbackMax", integer: true, lowerBound: 0 },
  stopLoss: { min: "stopMin", max: "stopMax", lowerBound: 0 },
  takeProfit: { min: "tpMin", max: "tpMax", lowerBound: 0 },
  trailingStop: { min: "trailMin", max: "trailMax", lowerBound: 0 },
  rebalanceCostMult: { min: "rebalanceCostMultMin", max: "rebalanceCostMultMax", lowerBound: 0 },
  blendWeight: { min: "blendWeightMin", max: "blendWeightMax", lowerBound: 0, upperBound: 1 },
  walkForwardFolds: { min: "walkForwardFoldsMin", max: "walkForwardFoldsMax", integer: true, lowerBound: 1 },
  walkForwardEmbargoBars: { min: "walkForwardEmbargoBarsMin", max: "walkForwardEmbargoBarsMax", integer: true, lowerBound: 0 },
  maxHighVolProb: { extraMin: "maxHighVolProbMin", extraMax: "maxHighVolProbMax", lowerBound: 0, upperBound: 1 },
  maxConformalWidth: { extraMin: "maxConformalWidthMin", extraMax: "maxConformalWidthMax", lowerBound: 0 },
  maxQuantileWidth: { extraMin: "maxQuantileWidthMin", extraMax: "maxQuantileWidthMax", lowerBound: 0 },
  periodsPerYear: { extraMin: "periodsPerYearMin", extraMax: "periodsPerYearMax", lowerBound: 0 },
  kalmanZMin: { extraMin: "kalmanZMinMin", extraMax: "kalmanZMinMax", lowerBound: 0 },
  kalmanZMax: { extraMin: "kalmanZMaxMin", extraMax: "kalmanZMaxMax", lowerBound: 0 },
  kalmanMarketTopN: { extraMin: "kalmanMarketTopNMin", extraMax: "kalmanMarketTopNMax", integer: true, lowerBound: 1 },
  maxOrderErrors: { extraMin: "maxOeMin", extraMax: "maxOeMax", integer: true, lowerBound: 0 },
  maxDrawdown: { extraMin: "maxDdMin", extraMax: "maxDdMax", lowerBound: 0 },
  maxDailyLoss: { extraMin: "maxDlMin", extraMax: "maxDlMax", lowerBound: 0 },
  thresholdFactorAlpha: { extraMin: "thresholdFactorAlphaMin", extraMax: "thresholdFactorAlphaMax", lowerBound: 0, upperBound: 1 },
  thresholdFactorMin: { extraMin: "thresholdFactorMinMin", extraMax: "thresholdFactorMinMax", lowerBound: 0 },
  thresholdFactorMax: { extraMin: "thresholdFactorMaxMin", extraMax: "thresholdFactorMaxMax", lowerBound: 0 },
  thresholdFactorFloor: { extraMin: "thresholdFactorFloorMin", extraMax: "thresholdFactorFloorMax", lowerBound: 0 },
  thresholdFactorWeight: { extraMin: "thresholdFactorWeightMin", extraMax: "thresholdFactorWeightMax" },
  volTarget: { extraMin: "volTargetMin", extraMax: "volTargetMax", lowerBound: 0 },
  volLookback: { extraMin: "volLookbackMin", extraMax: "volLookbackMax", integer: true, lowerBound: 1 },
  volEwmaAlpha: { extraMin: "volEwmaAlphaMin", extraMax: "volEwmaAlphaMax", lowerBound: 0, upperBound: 1 },
  volFloor: { extraMin: "volFloorMin", extraMax: "volFloorMax", lowerBound: 0 },
  volScaleMax: { extraMin: "volScaleMaxMin", extraMax: "volScaleMaxMax", lowerBound: 0 },
  maxVolatility: { extraMin: "maxVolatilityMin", extraMax: "maxVolatilityMax", lowerBound: 0 },
  minPositionSize: { extraMin: "minPositionSizeMin", extraMax: "minPositionSizeMax", lowerBound: 0 },
  maxPositionSize: { extraMin: "maxPositionSizeMin", extraMax: "maxPositionSizeMax", lowerBound: 0 },
};

function optimizerGuessParamNumber(raw: unknown): number | null {
  if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  if (typeof raw === "boolean") return raw ? 1 : 0;
  return null;
}

function optimizerGuessRoi(combo: OptimizationCombo): number | null {
  const roi = combo.finalEquity - 1;
  return Number.isFinite(roi) ? roi : null;
}

function optimizerGuessParamLabel(key: string): string {
  return key
    .replace(/^base/, "")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function optimizerGuessPearson(points: OptimizerCorrelationPoint[]): number | null {
  const n = points.length;
  if (n < OPTIMIZER_GUESS_MIN_SAMPLES) return null;
  const meanX = points.reduce((sum, p) => sum + p.x, 0) / n;
  const meanY = points.reduce((sum, p) => sum + p.roi, 0) / n;
  let numerator = 0;
  let denomX = 0;
  let denomY = 0;
  for (const point of points) {
    const dx = point.x - meanX;
    const dy = point.roi - meanY;
    numerator += dx * dy;
    denomX += dx * dx;
    denomY += dy * dy;
  }
  const denom = Math.sqrt(denomX * denomY);
  if (denom <= 0) return null;
  return numerator / denom;
}

function optimizerGuessMedian(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    const left = sorted[mid - 1];
    const right = sorted[mid];
    return left == null || right == null ? null : (left + right) / 2;
  }
  return sorted[mid] ?? null;
}

function clampOptimizerGuessValue(value: number, config: OptimizerCorrelationRangeConfig): number {
  const lower = config.lowerBound;
  const upper = config.upperBound;
  let out = value;
  if (lower != null) out = Math.max(lower, out);
  if (upper != null) out = Math.min(upper, out);
  return out;
}

function optimizerGuessRange(stat: OptimizerCorrelationStat, config: OptimizerCorrelationRangeConfig): { lo: number; hi: number } {
  const span = Math.max(0, stat.xMax - stat.xMin);
  const widthRatio = config.widthRatio ?? OPTIMIZER_GUESS_DEFAULT_WIDTH_RATIO;
  const radius = config.integer ? Math.max(1, Math.round((span * widthRatio) / 2)) : Math.max(span * widthRatio * 0.5, Math.abs(stat.target) * 0.03, 1e-12);
  let lo = clampOptimizerGuessValue(stat.target - radius, config);
  let hi = clampOptimizerGuessValue(stat.target + radius, config);
  lo = Math.max(stat.xMin, lo);
  hi = Math.min(stat.xMax, hi);
  if (config.integer) {
    lo = Math.floor(lo);
    hi = Math.ceil(hi);
    if (config.lowerBound != null) {
      lo = Math.max(config.lowerBound, lo);
      hi = Math.max(config.lowerBound, hi);
    }
  }
  if (lo > hi) [lo, hi] = [hi, lo];
  return { lo, hi };
}

function formatOptimizerGuessNumber(value: number, integer: boolean | undefined): string {
  if (integer) return String(Math.round(value));
  const rounded = Math.round(value * 1e8) / 1e8;
  return String(rounded);
}

function setOptimizerGuessPatch(
  patch: Partial<OptimizerRunForm>,
  key: OptimizerRunFormTextKey | undefined,
  value: string,
): void {
  if (!key) return;
  (patch as Partial<Record<OptimizerRunFormTextKey, string>>)[key] = value;
}

function optimizerSourceFromCombo(combo: OptimizationCombo): Exclude<OptimizerSource, "csv"> | null {
  if (combo.source === "binance" || combo.source === "coinbase" || combo.source === "kraken" || combo.source === "poloniex") {
    return combo.source;
  }
  const platform = combo.params.platform;
  if (platform === "binance" || platform === "coinbase" || platform === "kraken" || platform === "poloniex") {
    return platform;
  }
  return null;
}

export function buildOptimizerCorrelationGuess(combos: OptimizationCombo[]): OptimizerCorrelationGuess | null {
  const finiteCombos = combos.filter((combo) => optimizerGuessRoi(combo) != null);
  if (finiteCombos.length < OPTIMIZER_GUESS_MIN_SAMPLES) return null;

  const pointsByKey = new Map<string, OptimizerCorrelationPoint[]>();
  for (const combo of finiteCombos) {
    const roi = optimizerGuessRoi(combo);
    if (roi == null) continue;
    for (const key of Object.keys(OPTIMIZER_CORRELATION_RANGE_CONFIGS)) {
      const value = optimizerGuessParamNumber(combo.params[key]);
      if (value == null) continue;
      const points = pointsByKey.get(key) ?? [];
      points.push({ x: value, roi });
      pointsByKey.set(key, points);
    }
  }

  const stats: OptimizerCorrelationStat[] = [];
  for (const [key, points] of pointsByKey) {
    const xs = points.map((point) => point.x);
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    if (!(xMax > xMin)) continue;
    const correlation = optimizerGuessPearson(points);
    if (correlation == null || Math.abs(correlation) < OPTIMIZER_GUESS_MIN_ABS_CORRELATION) continue;
    const topCount = Math.max(2, Math.ceil(points.length * OPTIMIZER_GUESS_TOP_ROI_FRACTION));
    const topPoints = [...points].sort((a, b) => b.roi - a.roi).slice(0, topCount);
    const target = optimizerGuessMedian(topPoints.map((point) => point.x));
    if (target == null || !Number.isFinite(target)) continue;
    stats.push({
      key,
      label: optimizerGuessParamLabel(key),
      correlation,
      sampleCount: points.length,
      target,
      xMin,
      xMax,
    });
  }

  stats.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));

  const patch: Partial<OptimizerRunForm> = {
    minRoundTrips: "3",
    minExposure: "0.01",
    timeoutSec: "120",
    seedRatio: "0.55",
    survivorFraction: "0.5",
    perturbScaleDouble: "0.2",
    perturbScaleInt: "4",
  };
  const extras: Record<string, number> = {};
  const basis: string[] = [];
  let correlationCount = 0;

  const topCombo = [...finiteCombos].sort((a, b) => b.finalEquity - a.finalEquity)[0] ?? null;
  if (topCombo) {
    const source = optimizerSourceFromCombo(topCombo);
    const symbol = typeof topCombo.params.binanceSymbol === "string" ? topCombo.params.binanceSymbol.trim().toUpperCase() : "";
    if (source) {
      patch.source = source;
      patch.platforms = source;
      if (symbol) patch.symbol = symbol;
    }
    if (typeof topCombo.params.interval === "string" && topCombo.params.interval.trim()) {
      patch.intervals = topCombo.params.interval.trim();
    }
    if (typeof topCombo.params.normalization === "string" && topCombo.params.normalization.trim()) {
      patch.normalizations = topCombo.params.normalization.trim();
    }
  }

  for (const stat of stats) {
    if (correlationCount >= OPTIMIZER_GUESS_MAX_FIELDS) break;
    const config = OPTIMIZER_CORRELATION_RANGE_CONFIGS[stat.key];
    if (!config) continue;
    const { lo, hi } = optimizerGuessRange(stat, config);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) continue;
    const loText = formatOptimizerGuessNumber(lo, config.integer);
    const hiText = formatOptimizerGuessNumber(hi, config.integer);
    setOptimizerGuessPatch(patch, config.min, loText);
    setOptimizerGuessPatch(patch, config.max, hiText);
    if (config.extraMin) extras[config.extraMin] = config.integer ? Math.round(lo) : lo;
    if (config.extraMax) extras[config.extraMax] = config.integer ? Math.round(hi) : hi;
    basis.push(`${stat.label} r ${stat.correlation.toFixed(2)} -> ${loText}..${hiText}`);
    correlationCount += 1;
  }

  if (correlationCount === 0) return null;
  return {
    patch,
    extras,
    basis,
    sampleCount: finiteCombos.length,
    correlationCount,
  };
}

export type OrderSideFilter = "ALL" | "BUY" | "SELL";

export type OrderLogPrefs = {
  filterText: string;
  sentOnly: boolean;
  side: OrderSideFilter;
  limit: number;
  errorsOnly: boolean;
  showOrderId: boolean;
  showStatus: boolean;
  showClientOrderId: boolean;
};

export type SavedProfiles = Record<string, FormState>;

export type PendingProfileLoad = {
  name: string;
  profile: FormState;
  reasons: string[];
};

export type ComputeLimits = NonNullable<Awaited<ReturnType<typeof health>>["computeLimits"]>;
export type ManualOverrideKey = "method" | "openThreshold" | "closeThreshold";

export function optimizerSourceForPlatform(platform: Platform): OptimizerSource {
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

export function buildDefaultOptimizerRunForm(symbol: string, platform: Platform): OptimizerRunForm {
  return {
    source: optimizerSourceForPlatform(platform),
    symbol: sanitizeSymbolForPlatform(platform, symbol) ?? symbol.trim().toUpperCase(),
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
    objective: "roi",
    tuneObjective: "roi",
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
    minRoundTrips: "5",
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
    walkForwardEmbargoBarsMin: "1",
    walkForwardEmbargoBarsMax: "1",
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
    rebalanceCostMultMin: "",
    rebalanceCostMultMax: "",
    pCostAwareEdge: "",
    stopMin: "",
    stopMax: "",
    tpMin: "",
    tpMax: "",
    trailMin: "",
    trailMax: "",
    methodWeightBlend: "",
    methodWeightConfBlend: "",
    methodWeightConfPick: "",
    methodWeightConformalClip: "",
    methodWeightCostPick: "",
    methodWeightHarmonicBlend: "",
    methodWeightDisagreementGuard: "",
    methodWeightMedianBlend: "",
    methodWeightNeutralGuard: "",
    methodWeightRiskParityBlend: "",
    methodWeightConsensusBoost: "",
    methodWeightAnchorBlend: "",
    methodWeightTensionGate: "",
    methodWeightEntropyBlend: "",
    methodWeightCoherenceGate: "",
    methodWeightDivergenceGate: "",
    methodWeightFractalBlend: "",
    methodWeightPhaseCancel: "",
    methodWeightSoftmaxBlend: "",
    methodWeightSmoothSoftmaxBlend: "",
    methodWeightHedgeBlend: "",
    methodWeightNetSoftmaxBlend: "",
    methodWeightEdgeBlend: "",
    methodWeightEdgePick: "",
    methodWeightGeoBlend: "",
    methodWeightRegimeSwitch: "",
    methodWeightBanditRouter: "",
    blendWeightMin: "",
    blendWeightMax: "",
    disableLstmPersistence: false,
    noSweepThreshold: false,
    extraJson: "",
  };
}

export function parseOptimizerExtras(raw: string): { value: Record<string, unknown> | null; error: string | null } {
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

export function buildOptimizerRunRequest(form: OptimizerRunForm, extras: Record<string, unknown> | null): OptimizerRunRequest {
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
  const walkForwardEmbargoBarsMin = parseOptionalInt(form.walkForwardEmbargoBarsMin);
  if (walkForwardEmbargoBarsMin != null) req.walkForwardEmbargoBarsMin = walkForwardEmbargoBarsMin;
  const walkForwardEmbargoBarsMax = parseOptionalInt(form.walkForwardEmbargoBarsMax);
  if (walkForwardEmbargoBarsMax != null) req.walkForwardEmbargoBarsMax = walkForwardEmbargoBarsMax;

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
  const rebalanceCostMultMin = parseOptionalNumber(form.rebalanceCostMultMin);
  if (rebalanceCostMultMin != null) req.rebalanceCostMultMin = rebalanceCostMultMin;
  const rebalanceCostMultMax = parseOptionalNumber(form.rebalanceCostMultMax);
  if (rebalanceCostMultMax != null) req.rebalanceCostMultMax = rebalanceCostMultMax;
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
  const methodWeightConfBlend = parseOptionalNumber(form.methodWeightConfBlend);
  if (methodWeightConfBlend != null) req.methodWeightConfBlend = methodWeightConfBlend;
  const methodWeightConfPick = parseOptionalNumber(form.methodWeightConfPick);
  if (methodWeightConfPick != null) req.methodWeightConfPick = methodWeightConfPick;
  const methodWeightConformalClip = parseOptionalNumber(form.methodWeightConformalClip);
  if (methodWeightConformalClip != null) req.methodWeightConformalClip = methodWeightConformalClip;
  const methodWeightCostPick = parseOptionalNumber(form.methodWeightCostPick);
  if (methodWeightCostPick != null) req.methodWeightCostPick = methodWeightCostPick;
  const methodWeightHarmonicBlend = parseOptionalNumber(form.methodWeightHarmonicBlend);
  if (methodWeightHarmonicBlend != null) req.methodWeightHarmonicBlend = methodWeightHarmonicBlend;
  const methodWeightDisagreementGuard = parseOptionalNumber(form.methodWeightDisagreementGuard);
  if (methodWeightDisagreementGuard != null) req.methodWeightDisagreementGuard = methodWeightDisagreementGuard;
  const methodWeightMedianBlend = parseOptionalNumber(form.methodWeightMedianBlend);
  if (methodWeightMedianBlend != null) req.methodWeightMedianBlend = methodWeightMedianBlend;
  const methodWeightNeutralGuard = parseOptionalNumber(form.methodWeightNeutralGuard);
  if (methodWeightNeutralGuard != null) req.methodWeightNeutralGuard = methodWeightNeutralGuard;
  const methodWeightRiskParityBlend = parseOptionalNumber(form.methodWeightRiskParityBlend);
  if (methodWeightRiskParityBlend != null) req.methodWeightRiskParityBlend = methodWeightRiskParityBlend;
  const methodWeightConsensusBoost = parseOptionalNumber(form.methodWeightConsensusBoost);
  if (methodWeightConsensusBoost != null) req.methodWeightConsensusBoost = methodWeightConsensusBoost;
  const methodWeightAnchorBlend = parseOptionalNumber(form.methodWeightAnchorBlend);
  if (methodWeightAnchorBlend != null) req.methodWeightAnchorBlend = methodWeightAnchorBlend;
  const methodWeightTensionGate = parseOptionalNumber(form.methodWeightTensionGate);
  if (methodWeightTensionGate != null) req.methodWeightTensionGate = methodWeightTensionGate;
  const methodWeightEntropyBlend = parseOptionalNumber(form.methodWeightEntropyBlend);
  if (methodWeightEntropyBlend != null) req.methodWeightEntropyBlend = methodWeightEntropyBlend;
  const methodWeightCoherenceGate = parseOptionalNumber(form.methodWeightCoherenceGate);
  if (methodWeightCoherenceGate != null) req.methodWeightCoherenceGate = methodWeightCoherenceGate;
  const methodWeightDivergenceGate = parseOptionalNumber(form.methodWeightDivergenceGate);
  if (methodWeightDivergenceGate != null) req.methodWeightDivergenceGate = methodWeightDivergenceGate;
  const methodWeightFractalBlend = parseOptionalNumber(form.methodWeightFractalBlend);
  if (methodWeightFractalBlend != null) req.methodWeightFractalBlend = methodWeightFractalBlend;
  const methodWeightPhaseCancel = parseOptionalNumber(form.methodWeightPhaseCancel);
  if (methodWeightPhaseCancel != null) req.methodWeightPhaseCancel = methodWeightPhaseCancel;
  const methodWeightSoftmaxBlend = parseOptionalNumber(form.methodWeightSoftmaxBlend);
  if (methodWeightSoftmaxBlend != null) req.methodWeightSoftmaxBlend = methodWeightSoftmaxBlend;
  const methodWeightSmoothSoftmaxBlend = parseOptionalNumber(form.methodWeightSmoothSoftmaxBlend);
  if (methodWeightSmoothSoftmaxBlend != null) req.methodWeightSmoothSoftmaxBlend = methodWeightSmoothSoftmaxBlend;
  const methodWeightHedgeBlend = parseOptionalNumber(form.methodWeightHedgeBlend);
  if (methodWeightHedgeBlend != null) req.methodWeightHedgeBlend = methodWeightHedgeBlend;
  const methodWeightNetSoftmaxBlend = parseOptionalNumber(form.methodWeightNetSoftmaxBlend);
  if (methodWeightNetSoftmaxBlend != null) req.methodWeightNetSoftmaxBlend = methodWeightNetSoftmaxBlend;
  const methodWeightEdgeBlend = parseOptionalNumber(form.methodWeightEdgeBlend);
  if (methodWeightEdgeBlend != null) req.methodWeightEdgeBlend = methodWeightEdgeBlend;
  const methodWeightEdgePick = parseOptionalNumber(form.methodWeightEdgePick);
  if (methodWeightEdgePick != null) req.methodWeightEdgePick = methodWeightEdgePick;
  const methodWeightGeoBlend = parseOptionalNumber(form.methodWeightGeoBlend);
  if (methodWeightGeoBlend != null) req.methodWeightGeoBlend = methodWeightGeoBlend;
  const methodWeightRegimeSwitch = parseOptionalNumber(form.methodWeightRegimeSwitch);
  if (methodWeightRegimeSwitch != null) req.methodWeightRegimeSwitch = methodWeightRegimeSwitch;
  const methodWeightBanditRouter = parseOptionalNumber(form.methodWeightBanditRouter);
  if (methodWeightBanditRouter != null) req.methodWeightBanditRouter = methodWeightBanditRouter;
  const blendWeightMin = parseOptionalNumber(form.blendWeightMin);
  if (blendWeightMin != null) req.blendWeightMin = blendWeightMin;
  const blendWeightMax = parseOptionalNumber(form.blendWeightMax);
  if (blendWeightMax != null) req.blendWeightMax = blendWeightMax;

  if (form.disableLstmPersistence) req.disableLstmPersistence = true;
  if (form.noSweepThreshold) req.noSweepThreshold = true;

  if (extras) {
    const normalizedExtras = normalizeKnownOptimizerRunExtras(extras);
    // Keep extra JSON forward-compatible, but normalize the known typed
    // override keys before merging so validation and request emission agree.
    Object.assign(req, normalizedExtras);
  }
  enforceOptimizerRequestSourceCompatibility(req);

  return req;
}

export const CUSTOM_SYMBOL_VALUE = "__custom__";
export const TOP_COMBOS_POLL_MS = 30_000;
export const TOP_COMBOS_DISPLAY_DEFAULT = 5;
export const TOP_COMBOS_DISPLAY_MIN = 1;
export const TOP_COMBOS_BOT_TARGET = 50;
export const MIN_LOOKBACK_BARS = 2;
export const MIN_BACKTEST_BARS = 2;
export const MIN_BACKTEST_RATIO = 0.01;
export const MAX_BACKTEST_RATIO = 0.99;

export function minBarsRequiredForLookback(
  platform: Platform,
  interval: string,
  lookbackBarsRaw: number,
  lookbackWindowRaw: string,
): number | null {
  const overrideRaw = Math.trunc(lookbackBarsRaw);
  if (Number.isFinite(overrideRaw) && overrideRaw >= MIN_LOOKBACK_BARS) {
    return overrideRaw + 1;
  }

  const intervalSec = platformIntervalSeconds(platform, interval.trim());
  const windowRaw = lookbackWindowRaw.trim();
  if (!intervalSec || !windowRaw) return null;

  const windowSec = parseDurationSeconds(windowRaw);
  if (!windowSec || windowSec <= 0) return null;

  const windowBars = Math.ceil(windowSec / intervalSec);
  if (!Number.isFinite(windowBars) || windowBars < MIN_LOOKBACK_BARS) return null;
  return windowBars + 1;
}

export function autoAdjustBarsForLookback(
  barsRaw: number,
  minBarsRequired: number | null,
  platform: Platform,
  method: Method,
  apiLimits: ComputeLimits | null,
): number | null {
  if (!Number.isFinite(minBarsRequired ?? NaN) || minBarsRequired == null) return null;

  const requiredBars = Math.max(MIN_LOOKBACK_BARS, Math.trunc(minBarsRequired));
  const barsCap = maxBarsForPlatform(platform, method, apiLimits);
  if (Number.isFinite(barsCap) && requiredBars > barsCap) return null;

  const requestedBars = Math.trunc(barsRaw);
  if (Number.isFinite(requestedBars) && requestedBars > 0) {
    const normalizedBars = Math.max(MIN_LOOKBACK_BARS, requestedBars);
    return normalizedBars < requiredBars ? requiredBars : null;
  }

  const autoBars = PLATFORM_DEFAULT_BARS[platform] ?? 500;
  return autoBars < requiredBars ? requiredBars : null;
}

export const DURATION_UNITS: Array<{ unit: string; seconds: number }> = [
  { unit: "M", seconds: 30 * 24 * 60 * 60 },
  { unit: "w", seconds: 7 * 24 * 60 * 60 },
  { unit: "d", seconds: 24 * 60 * 60 },
  { unit: "h", seconds: 60 * 60 },
  { unit: "m", seconds: 60 },
  { unit: "s", seconds: 1 },
];

export function formatDurationSeconds(totalSeconds: number): string {
  const sec = Math.max(1, Math.round(totalSeconds));
  for (const { unit, seconds } of DURATION_UNITS) {
    if (sec % seconds === 0) return `${sec / seconds}${unit}`;
  }
  return `${sec}s`;
}

export function sigNumber(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "";
  const rounded = Math.round(value * 1e8) / 1e8;
  return String(rounded);
}

export function readExactSafeInteger(raw: unknown): number | null {
  return typeof raw === "number" && Number.isSafeInteger(raw) ? raw : null;
}

export function readNonNegativeExactSafeInteger(raw: unknown): number | null {
  const n = readExactSafeInteger(raw);
  return n != null && n >= 0 ? n : null;
}

export function coerceNumber(value: number | null | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function coerceExactSafeInteger(value: number | null | undefined, fallback: number): number {
  return typeof value === "number" && Number.isSafeInteger(value) ? value : fallback;
}

export function clampOptionalRatio(value: number | null | undefined): number {
  const v = coerceNumber(value, 0);
  return v > 0 ? clamp(v, 0, 0.999999) : 0;
}

export function clampOptionalRange(value: number | null | undefined, min: number, max: number): number {
  const v = coerceNumber(value, 0);
  return v > 0 ? clamp(v, min, max) : 0;
}

export function clampOptionalInt(value: number | null | undefined, fallback: number, min: number, max: number): number {
  const v = coerceExactSafeInteger(value, fallback);
  return v > 0 ? clamp(v, min, max) : 0;
}

export function sanitizeOptimizationComboOperation(raw: unknown): OptimizationComboOperation | null {
  const opRec = (raw as Record<string, unknown> | null | undefined) ?? {};
  const entryIndex = readNonNegativeExactSafeInteger(opRec.entryIndex);
  const exitIndex = readNonNegativeExactSafeInteger(opRec.exitIndex);
  if (entryIndex == null || exitIndex == null || exitIndex < entryIndex) return null;

  const entryEquity =
    typeof opRec.entryEquity === "number" && Number.isFinite(opRec.entryEquity) ? (opRec.entryEquity as number) : null;
  const exitEquity =
    typeof opRec.exitEquity === "number" && Number.isFinite(opRec.exitEquity) ? (opRec.exitEquity as number) : null;
  const retValue = typeof opRec.return === "number" && Number.isFinite(opRec.return) ? (opRec.return as number) : null;
  const holdingPeriods = readNonNegativeExactSafeInteger(opRec.holdingPeriods);
  const exitReason = typeof opRec.exitReason === "string" && opRec.exitReason.trim() ? opRec.exitReason.trim() : null;

  return {
    entryIndex,
    exitIndex,
    entryEquity,
    exitEquity,
    return: retValue,
    holdingPeriods,
    exitReason,
  };
}

export function sigText(value: string | null | undefined): string {
  return typeof value === "string" ? value : "";
}

export function sigBool(value: boolean | null | undefined): string {
  return value ? "1" : "0";
}

export function formatDirectionLabel(value: LatestSignal["closeDirection"]): string {
  if (value === undefined) return "—";
  return value ?? "NEUTRAL";
}

export type DecisionCheckStatus = "ok" | "warn" | "bad" | "skip";

export type DecisionCheck = {
  id: string;
  label: string;
  status: DecisionCheckStatus;
  detail: string;
};

export type DecisionSummary = {
  isHold: boolean;
  reason: string | null;
  checks: DecisionCheck[];
};

export const DIRECTION_HOLD_REASONS = new Set([
  "DIRECTIONS_DISAGREE",
  "BOTH_NEUTRAL",
  "KALMAN_NEUTRAL",
  "LSTM_NEUTRAL",
  "BLEND_NEUTRAL",
  "ROUTER_NEUTRAL",
]);

export function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatSignalDirection(value: LatestSignal["chosenDirection"]): string {
  return value ?? "NEUTRAL";
}

export function parseActionReason(action: string): string | null {
  const match = /\(([^)]+)\)/.exec(action);
  if (!match) return null;
  const reason = match[1]?.trim() ?? "";
  return reason ? reason : null;
}

export function normalizeHoldReason(reason: string | null): string | null {
  if (!reason) return null;
  const clean = reason.trim();
  if (!clean) return null;
  return clean.toUpperCase().replace(/\s+/g, "_");
}

export function decisionDotClass(status: DecisionCheckStatus): string {
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

export function decisionBadgeClass(status: DecisionCheckStatus): string {
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

export function decisionStatusLabel(status: DecisionCheckStatus): string {
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

export const SECONDS_PER_YEAR = 365 * 24 * 60 * 60;

export function inferPeriodsPerYear(platform: Platform, interval: string): number | null {
  const seconds = platformIntervalSeconds(platform, interval);
  if (!seconds || seconds <= 0) return null;
  const out = SECONDS_PER_YEAR / seconds;
  return Number.isFinite(out) && out > 0 ? out : null;
}

export type SplitStats = {
  trainEndRaw: number;
  backtestBars: number;
  tuneBars: number;
  fitBars: number;
  trainOk: boolean;
  backtestOk: boolean;
  tuneOk: boolean;
  fitOk: boolean;
};

export type TuneRatioBounds = {
  trainEndRaw: number;
  minTrainBars: number;
  minTuneBars: number;
  maxTuneBars: number;
  minRatio: number;
  maxRatio: number;
};

export type BacktestSplitAdjustmentChanges = {
  bars?: number;
  backtestRatio?: number;
  message: string;
};

export type BacktestSplitAdjustment = {
  params: ApiParams;
  changes: BacktestSplitAdjustmentChanges | null;
};

export type BacktestSplitAdjustmentOptions = {
  platform: Platform;
  method: Method;
  apiLimits: ComputeLimits | null;
};

export const RATIO_ROUND_DIGITS = 3;
export const RATIO_ROUND_FACTOR = 10 ** RATIO_ROUND_DIGITS;

export function splitStats(
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

export function roundRatioDown(value: number): number {
  return Math.floor(value * RATIO_ROUND_FACTOR) / RATIO_ROUND_FACTOR;
}

export function roundRatioUp(value: number): number {
  return Math.ceil(value * RATIO_ROUND_FACTOR) / RATIO_ROUND_FACTOR;
}

export function tuneRatioBounds(bars: number, backtestRatio: number, lookbackBars: number): TuneRatioBounds | null {
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

export function maxBarsForPlatform(platform: Platform, method: Method, apiLimits: ComputeLimits | null): number {
  let maxBars = method === "10" ? (platform === "binance" ? 1000 : Number.POSITIVE_INFINITY) : 1000;
  if (method !== "10") {
    const maxBarsRaw = apiLimits ? Math.trunc(apiLimits.maxBarsLstm) : NaN;
    if (Number.isFinite(maxBarsRaw) && maxBarsRaw > 0) {
      maxBars = maxBarsRaw;
    }
  }
  return maxBars;
}

export function maxLookbackForSplit(bars: number, backtestRatio: number, tuneRatio: number, tuningEnabled: boolean): number | null {
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

export function minTrainEndForTune(minTrainBars: number, tuneRatio: number, tuningEnabled: boolean, maxTrainEnd: number): number {
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

export function ratioForTrainEnd(bars: number, trainEnd: number): number {
  const raw = 1 - (trainEnd + 0.5) / Math.max(1, bars);
  return clamp(raw, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
}

function splitStatsValid(stats: SplitStats): boolean {
  return stats.trainOk && stats.backtestOk && stats.tuneOk && stats.fitOk;
}

function adjustedBacktestRatioForBars(
  bars: number,
  preferredRatio: number,
  lookbackBars: number,
  tuneRatio: number,
  tuningEnabled: boolean,
): number | null {
  const current = splitStats(bars, preferredRatio, lookbackBars, tuneRatio, tuningEnabled);
  if (splitStatsValid(current)) return preferredRatio;

  const minTrainBars = lookbackBars + 1;
  const maxTrainEnd = Math.max(0, bars - MIN_BACKTEST_BARS);
  const minTrainEnd = minTrainEndForTune(minTrainBars, tuneRatio, tuningEnabled, maxTrainEnd);
  if (minTrainEnd > maxTrainEnd) return null;

  let targetTrainEnd = current.trainEndRaw;
  if (!current.trainOk || !current.tuneOk || !current.fitOk) targetTrainEnd = minTrainEnd;
  if (!current.backtestOk) targetTrainEnd = maxTrainEnd;
  targetTrainEnd = clamp(Math.trunc(targetTrainEnd), minTrainEnd, maxTrainEnd);

  const candidateTrainEnds = Array.from(new Set([targetTrainEnd, minTrainEnd, maxTrainEnd]));
  for (const trainEnd of candidateTrainEnds) {
    const ratio = ratioForTrainEnd(bars, trainEnd);
    const stats = splitStats(bars, ratio, lookbackBars, tuneRatio, tuningEnabled);
    if (splitStatsValid(stats)) return ratio;
  }

  return null;
}

function buildBacktestSplitAdjustment(
  params: ApiParams,
  requestedBars: number,
  requestedRatio: number,
  adjustedBars: number,
  adjustedRatio: number,
): BacktestSplitAdjustment {
  const barsChanged = adjustedBars !== requestedBars;
  const ratioChanged = Math.abs(adjustedRatio - requestedRatio) >= 1e-9;
  if (!barsChanged && !ratioChanged) return { params, changes: null };

  const changes: BacktestSplitAdjustmentChanges = {
    message:
      barsChanged && ratioChanged
        ? `Adjusted bars to ${adjustedBars} and backtest ratio to ${fmtPct(adjustedRatio, 1)} to satisfy the split.`
        : barsChanged
          ? `Adjusted bars to ${adjustedBars} to satisfy the split.`
          : `Adjusted backtest ratio to ${fmtPct(adjustedRatio, 1)} to satisfy the split.`,
  };
  if (barsChanged) changes.bars = adjustedBars;
  if (ratioChanged) changes.backtestRatio = adjustedRatio;

  return {
    params: { ...params, bars: adjustedBars, backtestRatio: adjustedRatio },
    changes,
  };
}

export function adjustBacktestParamsForSplit(
  params: ApiParams,
  options: BacktestSplitAdjustmentOptions,
): BacktestSplitAdjustment {
  const platformValue = params.platform ?? options.platform;
  const method = params.method ?? options.method;
  const interval = params.interval ?? "";
  const intervalSec = platformIntervalSeconds(platformValue, interval);
  const overrideBars = Math.trunc(params.lookbackBars ?? 0);
  const windowRaw = (params.lookbackWindow ?? "").trim();
  const windowSec = windowRaw ? parseDurationSeconds(windowRaw) : null;
  const windowBars = windowSec && windowSec > 0 && intervalSec ? Math.ceil(windowSec / intervalSec) : null;
  const lookbackBars = overrideBars >= MIN_LOOKBACK_BARS ? overrideBars : windowBars;
  if (lookbackBars == null || lookbackBars < MIN_LOOKBACK_BARS) return { params, changes: null };

  const barsRaw = Math.trunc(params.bars ?? 0);
  if (!Number.isFinite(barsRaw) || barsRaw <= 0) return { params, changes: null };

  const barsCapRaw = maxBarsForPlatform(platformValue, method, options.apiLimits);
  const barsCap = Number.isFinite(barsCapRaw) ? Math.max(MIN_LOOKBACK_BARS, Math.trunc(barsCapRaw)) : Number.POSITIVE_INFINITY;
  const bars = Math.min(Math.max(MIN_LOOKBACK_BARS, barsRaw), barsCap);
  const backtestRatioRaw =
    typeof params.backtestRatio === "number" && Number.isFinite(params.backtestRatio) ? params.backtestRatio : 0.2;
  const backtestRatio = clamp(backtestRatioRaw, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
  const tuneRatio = typeof params.tuneRatio === "number" && Number.isFinite(params.tuneRatio) ? clamp(params.tuneRatio, 0, 0.99) : 0;
  const tuningEnabled = Boolean(params.optimizeOperations || params.sweepThreshold);

  const current = splitStats(bars, backtestRatio, lookbackBars, tuneRatio, tuningEnabled);
  if (splitStatsValid(current)) {
    return buildBacktestSplitAdjustment(params, barsRaw, backtestRatioRaw, bars, backtestRatio);
  }

  const minTrainBars = lookbackBars + 1;
  const maxTrainEndForBars = Math.max(0, bars - MIN_BACKTEST_BARS);
  const tuneRatioSafe = clamp(tuneRatio, 0, 0.99);
  const minTrainEndSearchCap = tuningEnabled
    ? Math.max(
        maxTrainEndForBars,
        minTrainBars,
        Math.ceil(minTrainBars / Math.max(1e-6, 1 - tuneRatioSafe)) + 2,
        tuneRatioSafe > 0 ? Math.ceil(MIN_BACKTEST_BARS / tuneRatioSafe) + 2 : minTrainBars,
      )
    : maxTrainEndForBars;
  const minTrainEnd = minTrainEndForTune(minTrainBars, tuneRatio, tuningEnabled, minTrainEndSearchCap);
  const minBarsForTrain = Math.ceil(minTrainEnd / Math.max(1e-6, 1 - backtestRatio));
  const minBarsForBacktest = Math.ceil(MIN_BACKTEST_BARS / Math.max(1e-6, backtestRatio));
  let candidateBars = Math.max(bars, minTrainBars + MIN_BACKTEST_BARS, minBarsForTrain, minBarsForBacktest);
  if (candidateBars > barsCap) candidateBars = barsCap;

  if (candidateBars > bars) {
    let adjustedBars = candidateBars;
    const preserveRatioSearchLimit = Number.isFinite(barsCap) ? barsCap : candidateBars + 1000;
    while (adjustedBars <= preserveRatioSearchLimit) {
      const stats = splitStats(adjustedBars, backtestRatio, lookbackBars, tuneRatio, tuningEnabled);
      if (splitStatsValid(stats)) {
        return buildBacktestSplitAdjustment(params, barsRaw, backtestRatioRaw, adjustedBars, backtestRatio);
      }
      adjustedBars += 1;
    }

    const adjustedRatio = adjustedBacktestRatioForBars(candidateBars, backtestRatio, lookbackBars, tuneRatio, tuningEnabled);
    if (adjustedRatio != null) {
      return buildBacktestSplitAdjustment(params, barsRaw, backtestRatioRaw, candidateBars, adjustedRatio);
    }
  }

  const adjustedRatio = adjustedBacktestRatioForBars(bars, backtestRatio, lookbackBars, tuneRatio, tuningEnabled);
  if (adjustedRatio != null) {
    return buildBacktestSplitAdjustment(params, barsRaw, backtestRatioRaw, bars, adjustedRatio);
  }

  return { params, changes: null };
}

export function clampComboForLimits(
  combo: OptimizationCombo,
  apiLimits: ComputeLimits | null,
  platform: Platform,
  fallback: { bars: number; epochs: number; hiddenSize: number },
): {
  bars: number;
  epochs: number;
  hiddenSize: number;
} {
  const lstmEnabled = combo.params.method !== "10";
  let bars = coerceExactSafeInteger(combo.params.bars, fallback.bars);
  if (!Number.isFinite(bars) || bars < 0) bars = 0;
  if (bars > 0) {
    bars = Math.max(MIN_LOOKBACK_BARS, bars);
    const barsCap = maxBarsForPlatform(platform, combo.params.method, apiLimits);
    if (Number.isFinite(barsCap)) {
      bars = Math.min(bars, barsCap);
    }
  }

  let epochs = clamp(coerceExactSafeInteger(combo.params.epochs, fallback.epochs), 0, 5000);
  let hiddenSize = clamp(coerceExactSafeInteger(combo.params.hiddenSize, fallback.hiddenSize), 1, 512);

  if (lstmEnabled && apiLimits) {
    epochs = Math.min(epochs, apiLimits.maxEpochs);
    hiddenSize = Math.min(hiddenSize, apiLimits.maxHiddenSize);
  }

  return { bars, epochs, hiddenSize };
}

export function applyComboToForm(
  prev: FormState,
  combo: OptimizationCombo,
  apiLimits: ComputeLimits | null,
  manualOverrides?: Set<ManualOverrideKey>,
  allowPositioning = true,
): FormState {
  const nextPlatform = preferredExchangePlatform(combo.params.platform, combo.source) ?? prev.platform;
  const comboSymbolRaw = combo.params.binanceSymbol?.trim() ?? "";
  const comboSymbol = comboSymbolRaw ? sanitizeSymbolForPlatform(nextPlatform, comboSymbolRaw) : null;
  const prevSymbol = sanitizeSymbolForPlatform(nextPlatform, prev.binanceSymbol);
  const fallbackSymbol =
    sanitizeSymbolForPlatform(nextPlatform, PLATFORM_DEFAULT_SYMBOL[nextPlatform] ?? prev.binanceSymbol)
    ?? (PLATFORM_DEFAULT_SYMBOL[nextPlatform] ?? prev.binanceSymbol);
  const symbol = comboSymbol ?? prevSymbol ?? fallbackSymbol;
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
  let { bars, epochs, hiddenSize } = clampComboForLimits(comboForLimits, apiLimits, nextPlatform, {
    bars: prev.bars,
    epochs: prev.epochs,
    hiddenSize: prev.hiddenSize,
  });
  const openThrRaw = coerceNumber(combo.openThreshold, prev.openThreshold);
  const closeThrRaw =
    combo.closeThreshold == null ? openThrRaw : coerceNumber(combo.closeThreshold, prev.closeThreshold);
  const openThreshold = manualOverrides?.has("openThreshold") ? prev.openThreshold : Math.max(0, openThrRaw);
  const closeThreshold = manualOverrides?.has("closeThreshold") ? prev.closeThreshold : Math.max(0, closeThrRaw);
  const fee = Math.max(0, coerceNumber(combo.params.fee, prev.fee));
  const learningRate = Math.max(1e-9, coerceNumber(combo.params.learningRate, prev.learningRate));
  const valRatio = clamp(coerceNumber(combo.params.valRatio, prev.valRatio), 0, 0.999999);
  const patience = clamp(coerceExactSafeInteger(combo.params.patience, prev.patience), 0, 1000);
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
  const minHoldBars = clampOptionalInt(combo.params.minHoldBars, prev.minHoldBars, 0, 1_000_000);
  const maxHoldBars = clampOptionalInt(combo.params.maxHoldBars, prev.maxHoldBars, 0, 1_000_000);
  const cooldownBars = clampOptionalInt(combo.params.cooldownBars, prev.cooldownBars, 0, 1_000_000);
  const maxDrawdown = clampOptionalRatio(combo.params.maxDrawdown);
  const maxDailyLoss = clampOptionalRatio(combo.params.maxDailyLoss);
  const maxOrderErrors =
    combo.params.maxOrderErrors == null
      ? 0
      : clampOptionalInt(combo.params.maxOrderErrors, prev.maxOrderErrors, 1, 1_000_000);
  const minEdge = Math.max(0, coerceNumber(combo.params.minEdge ?? prev.minEdge, prev.minEdge));
  const minSignalToNoise = Math.max(0, coerceNumber(combo.params.minSignalToNoise ?? prev.minSignalToNoise, prev.minSignalToNoise));
  const costAwareEdge = combo.params.costAwareEdge ?? prev.costAwareEdge;
  const edgeBuffer = Math.max(0, coerceNumber(combo.params.edgeBuffer ?? prev.edgeBuffer, prev.edgeBuffer));
  const trendLookback = clampOptionalInt(combo.params.trendLookback, prev.trendLookback, 0, 1_000_000);
  const maxPositionSize = Math.max(0, coerceNumber(combo.params.maxPositionSize ?? prev.maxPositionSize, prev.maxPositionSize));
  const volTarget = Math.max(0, coerceNumber(combo.params.volTarget ?? prev.volTarget, prev.volTarget));
  const volLookback = Math.max(0, coerceExactSafeInteger(combo.params.volLookback, prev.volLookback));
  const volEwmaAlphaRaw = coerceNumber(combo.params.volEwmaAlpha ?? prev.volEwmaAlpha, prev.volEwmaAlpha);
  const volEwmaAlpha = volEwmaAlphaRaw > 0 && volEwmaAlphaRaw < 1 ? volEwmaAlphaRaw : 0;
  const volFloor = Math.max(0, coerceNumber(combo.params.volFloor ?? prev.volFloor, prev.volFloor));
  const volScaleMax = Math.max(0, coerceNumber(combo.params.volScaleMax ?? prev.volScaleMax, prev.volScaleMax));
  const maxVolatility = Math.max(0, coerceNumber(combo.params.maxVolatility ?? prev.maxVolatility, prev.maxVolatility));
  const rebalanceBars = clampOptionalInt(combo.params.rebalanceBars, prev.rebalanceBars, 0, 1_000_000);
  const rebalanceThreshold = Math.max(
    0,
    coerceNumber(combo.params.rebalanceThreshold ?? prev.rebalanceThreshold, prev.rebalanceThreshold),
  );
  const rebalanceCostMult = Math.max(
    0,
    coerceNumber(combo.params.rebalanceCostMult ?? prev.rebalanceCostMult, prev.rebalanceCostMult),
  );
  const rebalanceGlobal = combo.params.rebalanceGlobal ?? prev.rebalanceGlobal;
  const rebalanceResetOnSignal = combo.params.rebalanceResetOnSignal ?? prev.rebalanceResetOnSignal;
  const fundingRate = coerceNumber(combo.params.fundingRate ?? prev.fundingRate, prev.fundingRate);
  const fundingBySide = combo.params.fundingBySide ?? prev.fundingBySide;
  const fundingOnOpen = combo.params.fundingOnOpen ?? prev.fundingOnOpen;
  const blendWeight = clamp(coerceNumber(combo.params.blendWeight ?? prev.blendWeight, prev.blendWeight), 0, 1);
  const routerLookback = clamp(coerceExactSafeInteger(combo.params.routerLookback, prev.routerLookback), 2, 1_000_000);
  const routerMinScore = clamp(coerceNumber(combo.params.routerMinScore, prev.routerMinScore), 0, 1);
  const tuneStressVolMult = Math.max(0, coerceNumber(combo.params.tuneStressVolMult ?? prev.tuneStressVolMult, prev.tuneStressVolMult));
  const tuneStressShock = coerceNumber(combo.params.tuneStressShock ?? prev.tuneStressShock, prev.tuneStressShock);
  const tuneStressWeight = Math.max(0, coerceNumber(combo.params.tuneStressWeight ?? prev.tuneStressWeight, prev.tuneStressWeight));
  const walkForwardFolds = clampOptionalInt(combo.params.walkForwardFolds, prev.walkForwardFolds, 1, 1000);
  const walkForwardEmbargoBars = clampOptionalInt(
    combo.params.walkForwardEmbargoBars,
    prev.walkForwardEmbargoBars,
    0,
    1_000_000,
  );

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

  const minBarsRequired = minBarsRequiredForLookback(nextPlatform, interval, lookbackBars, lookbackWindow);
  const adjustedBars = autoAdjustBarsForLookback(bars, minBarsRequired, nextPlatform, method, apiLimits);
  if (adjustedBars != null) bars = adjustedBars;

  const liveOrdersSupported = nextPlatform === "binance" || nextPlatform === "coinbase";

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
    rebalanceCostMult,
    rebalanceGlobal,
    rebalanceResetOnSignal,
    fundingRate,
    fundingBySide,
    fundingOnOpen,
    blendWeight,
    routerLookback,
    routerMinScore,
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
    // Applying a combo should preserve manual trade readiness on supported live-order
    // platforms while still clearing those toggles for read-only exchanges.
    binanceLive: liveOrdersSupported ? prev.binanceLive : false,
    tradeArmed: liveOrdersSupported ? prev.tradeArmed : false,
    orderQuantity,
    orderQuote,
    orderQuoteFraction,
    maxOrderQuote,
    tuneStressVolMult,
    tuneStressShock,
    tuneStressWeight,
    walkForwardFolds,
    walkForwardEmbargoBars,
    lookbackBars,
    lookbackWindow,
    openThreshold,
    closeThreshold,
  };
}

export function comboApplySignature(
  combo: OptimizationCombo,
  apiLimits: ComputeLimits | null,
  baseForm: FormState,
  manualOverrides?: Set<ManualOverrideKey>,
  allowPositioning = true,
): string {
  return formApplySignature(applyComboToForm(baseForm, combo, apiLimits, manualOverrides, allowPositioning));
}

export function comboAnnualizedEquity(combo: OptimizationCombo): number | null {
  const annReturn = combo.metrics?.annualizedReturn;
  if (typeof annReturn !== "number" || !Number.isFinite(annReturn)) return null;
  const annEq = annReturn + 1;
  return Number.isFinite(annEq) ? annEq : null;
}

export function formApplySignature(form: FormState): string {
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
    sigNumber(form.rebalanceCostMult),
    sigBool(form.rebalanceGlobal),
    sigBool(form.rebalanceResetOnSignal),
    sigNumber(form.fundingRate),
    sigBool(form.fundingBySide),
    sigBool(form.fundingOnOpen),
    sigNumber(form.blendWeight),
    sigNumber(form.routerLookback),
    sigNumber(form.routerMinScore),
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
    sigNumber(form.walkForwardEmbargoBars),
    sigNumber(form.openThreshold),
    sigNumber(form.closeThreshold),
  ].join("|");
}
