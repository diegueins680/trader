import type { IntrabarFill, Market, Method, Normalization, Platform, Positioning } from "../lib/types";
import { BINANCE_INTERVAL_SECONDS, PLATFORM_DEFAULT_SYMBOL, PLATFORM_INTERVAL_SET, TUNE_OBJECTIVE_SET } from "./constants";
import { METHOD_IDS } from "./contracts";
import { sanitizeSymbolForPlatform } from "./symbols";
import { clamp } from "./utils";

export type FormState = {
  binanceSymbol: string;
  botSymbols: string;
  platform: Platform;
  market: Market;
  interval: string;
  bars: number;
  lookbackWindow: string;
  lookbackBars: number;
  method: Method;
  positioning: Positioning;
  openThreshold: number;
  closeThreshold: number;
  fee: number;
  slippage: number;
  spread: number;
  intrabarFill: IntrabarFill;
  stopLoss: number;
  takeProfit: number;
  trailingStop: number;
  stopLossVolMult: number;
  takeProfitVolMult: number;
  trailingStopVolMult: number;
  minHoldBars: number;
  maxHoldBars: number;
  cooldownBars: number;
  maxDrawdown: number;
  maxDailyLoss: number;
  maxOrderErrors: number;
  minEdge: number;
  minSignalToNoise: number;
  costAwareEdge: boolean;
  edgeBuffer: number;
  trendLookback: number;
  maxPositionSize: number;
  volTarget: number;
  volLookback: number;
  volEwmaAlpha: number;
  volFloor: number;
  volScaleMax: number;
  maxVolatility: number;
  rebalanceBars: number;
  rebalanceThreshold: number;
  rebalanceCostMult: number;
  rebalanceGlobal: boolean;
  rebalanceResetOnSignal: boolean;
  fundingRate: number;
  fundingBySide: boolean;
  fundingOnOpen: boolean;
  blendWeight: number;
  routerLookback: number;
  routerMinScore: number;
  backtestRatio: number;
  tuneRatio: number;
  tuneObjective: string;
  tunePenaltyMaxDrawdown: number;
  tunePenaltyTurnover: number;
  tuneStressVolMult: number;
  tuneStressShock: number;
  tuneStressWeight: number;
  minRoundTrips: number;
  walkForwardFolds: number;
  walkForwardEmbargoBars: number;
  normalization: Normalization;
  epochs: number;
  learningRate: number;
  valRatio: number;
  patience: number;
  gradClip: number;
  hiddenSize: number;
  kalmanZMin: number;
  kalmanZMax: number;
  maxHighVolProb: number;
  maxConformalWidth: number;
  maxQuantileWidth: number;
  confirmConformal: boolean;
  confirmQuantiles: boolean;
  confidenceSizing: boolean;
  minPositionSize: number;
  optimizeOperations: boolean;
  sweepThreshold: boolean;
  binanceTestnet: boolean;
  orderQuote: number;
  orderQuantity: number;
  orderQuoteFraction: number;
  maxOrderQuote: number;
  idempotencyKey: string;
  binanceLive: boolean;
  tradeArmed: boolean;
  bypassCache: boolean;
  autoRefresh: boolean;
  autoRefreshSec: number;
  positionsOpenTimeCacheSec: number;

  // Live bot (advanced)
  botPollSeconds: number;
  botOnlineEpochs: number;
  botTrainBars: number;
  botMaxPoints: number;
  botProtectionOrders: boolean;
  botAdoptExistingPosition: boolean;
};

export const defaultForm: FormState = {
  binanceSymbol: "BTCUSDT",
  botSymbols: "",
  platform: "binance",
  market: "spot",
  interval: "1h",
  bars: 500,
  lookbackWindow: "7d",
  lookbackBars: 0,
  method: "11",
  positioning: "long-flat",
  openThreshold: 0.002,
  closeThreshold: 0.002,
  fee: 0.0008,
  slippage: 0.0002,
  spread: 0.0002,
  intrabarFill: "stop-first",
  stopLoss: 0,
  takeProfit: 0,
  trailingStop: 0,
  stopLossVolMult: 0,
  takeProfitVolMult: 0,
  trailingStopVolMult: 0,
  minHoldBars: 4,
  maxHoldBars: 36,
  cooldownBars: 2,
  maxDrawdown: 0,
  maxDailyLoss: 0,
  maxOrderErrors: 0,
  minEdge: 0.0004,
  minSignalToNoise: 0.8,
  costAwareEdge: true,
  edgeBuffer: 0.0002,
  trendLookback: 30,
  maxPositionSize: 0.8,
  volTarget: 0.7,
  volLookback: 30,
  volEwmaAlpha: 0,
  volFloor: 0.15,
  volScaleMax: 1,
  maxVolatility: 1.5,
  rebalanceBars: 24,
  rebalanceThreshold: 0.05,
  rebalanceCostMult: 0,
  rebalanceGlobal: false,
  rebalanceResetOnSignal: false,
  fundingRate: 0.1,
  fundingBySide: false,
  fundingOnOpen: false,
  blendWeight: 0.5,
  routerLookback: 30,
  routerMinScore: 0.25,
  backtestRatio: 0.2,
  tuneRatio: 0.25,
  tuneObjective: "roi",
  tunePenaltyMaxDrawdown: 1.5,
  tunePenaltyTurnover: 0.2,
  tuneStressVolMult: 1.0,
  tuneStressShock: 0,
  tuneStressWeight: 0,
  minRoundTrips: 5,
  walkForwardFolds: 7,
  walkForwardEmbargoBars: 1,
  normalization: "standard",
  epochs: 30,
  learningRate: 0.001,
  valRatio: 0.3,
  patience: 10,
  gradClip: 0,
  hiddenSize: 16,
  kalmanZMin: 0.5,
  kalmanZMax: 3,
  maxHighVolProb: 0,
  maxConformalWidth: 0,
  maxQuantileWidth: 0,
  confirmConformal: true,
  confirmQuantiles: true,
  confidenceSizing: true,
  minPositionSize: 0.15,
  optimizeOperations: false,
  sweepThreshold: false,
  binanceTestnet: false,
  orderQuote: 100,
  orderQuantity: 0,
  orderQuoteFraction: 0,
  maxOrderQuote: 0,
  idempotencyKey: "",
  binanceLive: false,
  tradeArmed: false,
  bypassCache: false,
  autoRefresh: false,
  autoRefreshSec: 20,
  positionsOpenTimeCacheSec: 60,

  botPollSeconds: 0,
  botOnlineEpochs: 1,
  botTrainBars: 800,
  botMaxPoints: 2000,
  botProtectionOrders: false,
  botAdoptExistingPosition: true,
};

const METHOD_SET = new Set<Method>(METHOD_IDS);
const NORMALIZATION_SET = new Set<Normalization>(["none", "minmax", "standard", "log"]);

export type FormStateJson = Partial<FormState> & {
  threshold?: unknown; // legacy field (maps to openThreshold/closeThreshold)
  platform?: unknown;
  interval?: unknown;
  positioning?: unknown;
  intrabarFill?: unknown;
  lookbackWindow?: unknown;
  lookbackBars?: unknown;
};

export function binanceIntervalSeconds(interval: string): number | null {
  const sec = BINANCE_INTERVAL_SECONDS[interval];
  return typeof sec === "number" && Number.isFinite(sec) ? sec : null;
}

export function platformIntervalSeconds(platform: Platform, interval: string): number | null {
  if (platform === "binance") return binanceIntervalSeconds(interval);
  return parseDurationSeconds(interval);
}

function platformDefaultSymbol(platform: Platform): string {
  return PLATFORM_DEFAULT_SYMBOL[platform];
}

function normalizeSymbol(raw: unknown, platform: Platform): string {
  const s = typeof raw === "string" ? raw : "";
  return sanitizeSymbolForPlatform(platform, s) ?? platformDefaultSymbol(platform);
}

export function parseDurationSeconds(raw: string): number | null {
  const s = raw.trim();
  const m = /^(\d+)([A-Za-z])$/.exec(s);
  if (!m) return null;

  const n = Number(m[1]);
  if (!Number.isSafeInteger(n)) return null;
  const unitRaw = m[2] ?? "";
  const unit = unitRaw === "M" ? "M" : unitRaw.toLowerCase();

  const mult =
    unit === "s"
      ? 1
      : unit === "m"
        ? 60
        : unit === "h"
          ? 60 * 60
          : unit === "d"
            ? 24 * 60 * 60
            : unit === "w"
              ? 7 * 24 * 60 * 60
              : unit === "M"
                ? 30 * 24 * 60 * 60
                : null;
  if (!mult) return null;
  const seconds = n * mult;
  return Number.isSafeInteger(seconds) ? seconds : null;
}

function normalizePlatformInterval(platform: Platform, raw: unknown, fallback: string): string {
  const value = normalizeIntervalCode(raw);
  return PLATFORM_INTERVAL_SET[platform].has(value) ? value : fallback;
}

function normalizeLookbackWindow(raw: unknown, fallback: string): string {
  if (typeof raw !== "string") return fallback;
  const value = raw.trim();
  const sec = parseDurationSeconds(value);
  return sec && sec > 0 ? value : fallback;
}

function normalizeIntervalCode(raw: unknown): string {
  if (typeof raw !== "string") return "";
  const value = raw.trim();
  const match = /^(\d+)([A-Za-z])$/.exec(value);
  if (!match) return value;
  const digits = match[1] ?? "";
  const unit = match[2] ?? "";
  return `${digits}${unit === "M" ? "M" : unit.toLowerCase()}`;
}

function parseFiniteInteger(raw: unknown): number | null {
  // Restored whole-number fields must remain exactly representable after JSON/string hydration.
  if (typeof raw === "number" && Number.isSafeInteger(raw)) return raw;
  if (typeof raw === "string") {
    const n = Number(raw);
    if (Number.isSafeInteger(n)) return n;
  }
  return null;
}

function normalizeWholeNumber(raw: unknown, fallback: number, lo: number, hi: number): number {
  const n = parseFiniteInteger(raw);
  return n == null ? fallback : clamp(n, lo, hi);
}

function normalizeLookbackBars(raw: unknown, fallback: number): number {
  const n = parseFiniteInteger(raw);
  if (n == null) return fallback;
  return n >= 2 ? n : 0;
}

function normalizeBool(raw: unknown, fallback: boolean): boolean {
  if (typeof raw === "boolean") return raw;
  if (raw === 1 || raw === 0) return raw === 1;
  if (typeof raw === "string") {
    const value = raw.trim().toLowerCase();
    if (value === "1" || value === "true") return true;
    if (value === "0" || value === "false") return false;
  }
  return fallback;
}

function normalizeFiniteNumber(raw: unknown, fallback: number, lo: number, hi: number): number {
  if (typeof raw === "number" && Number.isFinite(raw)) return clamp(raw, lo, hi);
  if (typeof raw === "string") {
    const n = Number(raw);
    if (Number.isFinite(n)) return clamp(n, lo, hi);
  }
  return fallback;
}

type NumericFormKey = {
  [K in keyof FormState]-?: FormState[K] extends number ? K : never;
}[keyof FormState];

function coerceFiniteNumber(raw: unknown, fallback: number): number {
  if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  if (typeof raw === "string") {
    const n = Number(raw);
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

function normalizeRestoredNumericFields(raw: Record<string, unknown>): Pick<FormState, NumericFormKey> {
  const out = {} as Pick<FormState, NumericFormKey>;
  for (const [key, fallback] of Object.entries(defaultForm) as Array<[keyof FormState, FormState[keyof FormState]]>) {
    if (typeof fallback !== "number") continue;
    const numericKey = key as NumericFormKey;
    out[numericKey] = coerceFiniteNumber(raw[numericKey], fallback) as FormState[NumericFormKey];
  }
  return out;
}

function normalizePositioning(raw: unknown, fallback: Positioning): Positioning {
  const value =
    typeof raw === "string"
      ? raw
          .trim()
          .toLowerCase()
          .replace(/[^a-z0-9]/g, "")
      : "";
  if (value === "longflat" || value === "lf" || value === "flat" || value === "longonly" || value === "long") return "long-flat";
  if (value === "longshort" || value === "ls") return "long-short";
  return fallback;
}

function normalizeIntrabarFill(raw: unknown, fallback: IntrabarFill): IntrabarFill {
  const value =
    typeof raw === "string"
      ? raw
          .trim()
          .toLowerCase()
          .replace(/[^a-z0-9]/g, "")
      : "";
  if (value === "stopfirst" || value === "slfirst" || value === "stop" || value === "worst") return "stop-first";
  if (value === "takeprofitfirst" || value === "tpfirst" || value === "takeprofit" || value === "tp" || value === "best") {
    return "take-profit-first";
  }
  return fallback;
}

function normalizePlatform(raw: unknown, fallback: Platform): Platform {
  const value = typeof raw === "string" ? raw.trim().toLowerCase() : "";
  if (value === "binance" || value === "coinbase" || value === "kraken" || value === "poloniex") return value;
  return fallback;
}

function normalizeMarket(raw: unknown, fallback: Market): Market {
  const value = typeof raw === "string" ? raw.trim().toLowerCase() : "";
  if (value === "spot" || value === "margin" || value === "futures") return value;
  return fallback;
}

function normalizeTuneObjective(raw: unknown, fallback: string): string {
  const s = typeof raw === "string" ? raw.trim().toLowerCase().replace(/\s+/g, "").replace(/_/g, "-") : "";
  const canonical =
    s === "finalequity" || s === "final-equity"
      ? "final-equity"
      : s === "annualizedequity" || s === "annualized-equity" || s === "annualizedreturn" || s === "annualized-return"
        ? "annualized-equity"
        : s === "roi" || s === "riskadjustedroi" || s === "risk-adjusted-roi"
          ? "roi"
          : s === "sharpe"
            ? "sharpe"
            : s === "calmar"
              ? "calmar"
              : s === "equitydd" || s === "equity-dd"
                ? "equity-dd"
                : s === "equityddturnover" || s === "equity-dd-turnover"
                  ? "equity-dd-turnover"
                  : "";
  if (canonical && TUNE_OBJECTIVE_SET.has(canonical)) return canonical;
  if (TUNE_OBJECTIVE_SET.has(fallback)) return fallback;
  return "roi";
}

function normalizeMethod(raw: unknown, fallback: Method): Method {
  const s = typeof raw === "string" ? raw.trim() : "";
  if (METHOD_SET.has(s as Method)) return s as Method;
  if (METHOD_SET.has(fallback)) return fallback;
  return defaultForm.method;
}

function normalizeNormalization(raw: unknown, fallback: Normalization): Normalization {
  const s = typeof raw === "string" ? raw.trim().toLowerCase() : "";
  if (NORMALIZATION_SET.has(s as Normalization)) return s as Normalization;
  if (NORMALIZATION_SET.has(fallback)) return fallback;
  return defaultForm.normalization;
}

function normalizeNonNegativeFiniteNumber(raw: unknown, fallback: number): number {
  return normalizeFiniteNumber(raw, fallback, 0, Number.MAX_VALUE);
}

export function normalizeFormState(raw: FormStateJson | null | undefined): FormState {
  const merged = { ...defaultForm, ...(raw ?? {}) };
  const rawRec = (raw as Record<string, unknown> | null | undefined) ?? {};
  // Invariant: normalized form state exposes finite numbers for every numeric field,
  // even when older/local storage serialized them as strings.
  const numericFields = normalizeRestoredNumericFields(rawRec);
  const botSymbols = typeof rawRec.botSymbols === "string" ? rawRec.botSymbols : merged.botSymbols;
  const legacyThreshold = rawRec.threshold;
  const openThreshold = normalizeFiniteNumber(rawRec.openThreshold ?? legacyThreshold ?? merged.openThreshold, defaultForm.openThreshold, 0, 1e9);
  const closeThreshold = normalizeFiniteNumber(
    rawRec.closeThreshold ?? legacyThreshold ?? (rawRec.openThreshold != null ? openThreshold : merged.closeThreshold),
    defaultForm.closeThreshold,
    0,
    1e9,
  );
  const kalmanZMin = normalizeFiniteNumber(rawRec.kalmanZMin ?? merged.kalmanZMin, defaultForm.kalmanZMin, 0, 1e9);
  const kalmanZMaxRaw = normalizeFiniteNumber(
    rawRec.kalmanZMax ?? merged.kalmanZMax,
    defaultForm.kalmanZMax,
    0,
    1e9,
  );
  const kalmanZMax = Math.max(kalmanZMin, kalmanZMaxRaw);
  const platform = normalizePlatform(rawRec.platform ?? merged.platform, defaultForm.platform);
  const restoredBinanceMarket =
    platform === "binance" ? normalizeMarket(rawRec.market ?? merged.market, defaultForm.market) : "spot";
  let market = restoredBinanceMarket;
  const restoredBinanceMargin = platform === "binance" && restoredBinanceMarket === "margin";
  const liveOrdersSupported = platform === "binance" || platform === "coinbase";
  const binanceLiveCandidate =
    liveOrdersSupported ? normalizeBool(rawRec.binanceLive ?? merged.binanceLive, defaultForm.binanceLive) : false;
  // Margin requires live orders; prefer a safe fallback over implicitly enabling live mode.
  const binanceLive = restoredBinanceMargin && !binanceLiveCandidate ? false : binanceLiveCandidate;
  if (restoredBinanceMargin && !binanceLiveCandidate) market = "spot";
  // Trade arming is meaningful for Binance + Coinbase; disable it for non-trading platforms.
  const tradeArmed =
    platform === "binance" || platform === "coinbase"
      ? normalizeBool(rawRec.tradeArmed ?? merged.tradeArmed, defaultForm.tradeArmed)
      : false;
  // Persisted margin restores must clear the stale testnet toggle even when market falls back to spot.
  const binanceTestnet =
    platform === "binance" && !restoredBinanceMargin
      ? normalizeBool(rawRec.binanceTestnet ?? merged.binanceTestnet, defaultForm.binanceTestnet)
      : false;
  // Restore-time symbol hydration must keep the field usable for the active platform:
  // canonicalize compatible aliases and otherwise fall back to that platform's default symbol.
  const binanceSymbol = normalizeSymbol(rawRec.binanceSymbol ?? merged.binanceSymbol, platform);
  // Keep restored manual sizing fields numeric before the sizing UI computes badges, cap state, and trade readiness.
  const orderQuote = normalizeFiniteNumber(rawRec.orderQuote ?? merged.orderQuote, defaultForm.orderQuote, 0, 1e9);
  const orderQuantity = normalizeFiniteNumber(rawRec.orderQuantity ?? merged.orderQuantity, defaultForm.orderQuantity, 0, 1e9);
  const orderQuoteFraction = normalizeFiniteNumber(
    rawRec.orderQuoteFraction ?? merged.orderQuoteFraction,
    defaultForm.orderQuoteFraction,
    -1e9,
    1e9,
  );
  const maxOrderQuote = normalizeFiniteNumber(rawRec.maxOrderQuote ?? merged.maxOrderQuote, defaultForm.maxOrderQuote, 0, 1e9);
  // Integer-backed request fields should restore as exact safe integers so the UI state
  // cannot diverge from the values later emitted by the request builder.
  const bars = normalizeWholeNumber(rawRec.bars ?? merged.bars, defaultForm.bars, 0, 1e9);
  const epochs = normalizeWholeNumber(rawRec.epochs ?? merged.epochs, defaultForm.epochs, 0, 5000);
  const hiddenSize = normalizeWholeNumber(rawRec.hiddenSize ?? merged.hiddenSize, defaultForm.hiddenSize, 1, 512);
  const patience = normalizeWholeNumber(rawRec.patience ?? merged.patience, defaultForm.patience, 0, 1000);
  // Integer-only bar/count controls and live-bot poll cadence should not reopen with
  // fractional or over-cap values that the UI later clamps differently when building requests.
  const minHoldBars = normalizeWholeNumber(rawRec.minHoldBars ?? merged.minHoldBars, defaultForm.minHoldBars, 0, 1_000_000);
  const maxHoldBars = normalizeWholeNumber(rawRec.maxHoldBars ?? merged.maxHoldBars, defaultForm.maxHoldBars, 0, 1_000_000);
  const cooldownBars = normalizeWholeNumber(rawRec.cooldownBars ?? merged.cooldownBars, defaultForm.cooldownBars, 0, 1_000_000);
  const maxOrderErrors = normalizeWholeNumber(rawRec.maxOrderErrors ?? merged.maxOrderErrors, defaultForm.maxOrderErrors, 0, 1_000_000);
  const trendLookback = normalizeWholeNumber(rawRec.trendLookback ?? merged.trendLookback, defaultForm.trendLookback, 0, 1_000_000);
  const volLookback = normalizeWholeNumber(rawRec.volLookback ?? merged.volLookback, defaultForm.volLookback, 0, 1e9);
  const rebalanceBars = normalizeWholeNumber(rawRec.rebalanceBars ?? merged.rebalanceBars, defaultForm.rebalanceBars, 0, 1e9);
  const routerLookback = normalizeWholeNumber(rawRec.routerLookback ?? merged.routerLookback, defaultForm.routerLookback, 2, 1_000_000);
  const botPollSeconds = normalizeWholeNumber(rawRec.botPollSeconds ?? merged.botPollSeconds, defaultForm.botPollSeconds, 0, 3600);
  const botOnlineEpochs = normalizeWholeNumber(
    rawRec.botOnlineEpochs ?? merged.botOnlineEpochs,
    defaultForm.botOnlineEpochs,
    0,
    50,
  );
  const botTrainBars = normalizeWholeNumber(rawRec.botTrainBars ?? merged.botTrainBars, defaultForm.botTrainBars, 10, 1e9);
  const botMaxPoints = normalizeWholeNumber(rawRec.botMaxPoints ?? merged.botMaxPoints, defaultForm.botMaxPoints, 100, 100_000);
  const { threshold: _ignoredThreshold, ...mergedNoLegacy } = merged as FormState & { threshold?: unknown };
  return {
    ...mergedNoLegacy,
    ...numericFields,
    botSymbols,
    platform,
    market,
    binanceTestnet,
    binanceLive,
    tradeArmed,
    binanceSymbol,
    bars,
    method: normalizeMethod(rawRec.method ?? merged.method, defaultForm.method),
    interval: normalizePlatformInterval(platform, raw?.interval ?? merged.interval, defaultForm.interval),
    positioning: normalizePositioning(raw?.positioning ?? merged.positioning, defaultForm.positioning),
    lookbackWindow: normalizeLookbackWindow(raw?.lookbackWindow ?? merged.lookbackWindow, defaultForm.lookbackWindow),
    lookbackBars: normalizeLookbackBars(raw?.lookbackBars ?? merged.lookbackBars, defaultForm.lookbackBars),
    openThreshold,
    closeThreshold,
    fee: normalizeNonNegativeFiniteNumber(rawRec.fee ?? merged.fee, defaultForm.fee),
    slippage: normalizeFiniteNumber(rawRec.slippage ?? merged.slippage, defaultForm.slippage, 0, 0.999999),
    spread: normalizeFiniteNumber(rawRec.spread ?? merged.spread, defaultForm.spread, 0, 0.999999),
    intrabarFill: normalizeIntrabarFill(rawRec.intrabarFill ?? merged.intrabarFill, defaultForm.intrabarFill),
    stopLoss: normalizeFiniteNumber(rawRec.stopLoss ?? merged.stopLoss, defaultForm.stopLoss, 0, 0.999999),
    takeProfit: normalizeFiniteNumber(rawRec.takeProfit ?? merged.takeProfit, defaultForm.takeProfit, 0, 0.999999),
    trailingStop: normalizeFiniteNumber(rawRec.trailingStop ?? merged.trailingStop, defaultForm.trailingStop, 0, 0.999999),
    stopLossVolMult: normalizeFiniteNumber(
      rawRec.stopLossVolMult ?? merged.stopLossVolMult,
      defaultForm.stopLossVolMult,
      0,
      1e9,
    ),
    takeProfitVolMult: normalizeFiniteNumber(
      rawRec.takeProfitVolMult ?? merged.takeProfitVolMult,
      defaultForm.takeProfitVolMult,
      0,
      1e9,
    ),
    trailingStopVolMult: normalizeFiniteNumber(
      rawRec.trailingStopVolMult ?? merged.trailingStopVolMult,
      defaultForm.trailingStopVolMult,
      0,
      1e9,
    ),
    minHoldBars,
    maxHoldBars,
    cooldownBars,
    maxDrawdown: normalizeFiniteNumber(rawRec.maxDrawdown ?? merged.maxDrawdown, defaultForm.maxDrawdown, 0, 0.999999),
    maxDailyLoss: normalizeFiniteNumber(rawRec.maxDailyLoss ?? merged.maxDailyLoss, defaultForm.maxDailyLoss, 0, 0.999999),
    maxOrderErrors,
    backtestRatio: normalizeFiniteNumber(rawRec.backtestRatio ?? merged.backtestRatio, defaultForm.backtestRatio, 0.01, 0.99),
    tuneRatio: normalizeFiniteNumber(rawRec.tuneRatio ?? merged.tuneRatio, defaultForm.tuneRatio, 0, 0.99),
    tuneObjective: normalizeTuneObjective(rawRec.tuneObjective ?? merged.tuneObjective, defaultForm.tuneObjective),
    tunePenaltyMaxDrawdown: normalizeFiniteNumber(rawRec.tunePenaltyMaxDrawdown ?? merged.tunePenaltyMaxDrawdown, defaultForm.tunePenaltyMaxDrawdown, 0, 1e9),
    tunePenaltyTurnover: normalizeFiniteNumber(rawRec.tunePenaltyTurnover ?? merged.tunePenaltyTurnover, defaultForm.tunePenaltyTurnover, 0, 1e9),
    tuneStressVolMult: normalizeFiniteNumber(rawRec.tuneStressVolMult ?? merged.tuneStressVolMult, defaultForm.tuneStressVolMult, 0, 1e9),
    tuneStressShock: normalizeFiniteNumber(rawRec.tuneStressShock ?? merged.tuneStressShock, defaultForm.tuneStressShock, -1e9, 1e9),
    tuneStressWeight: normalizeFiniteNumber(rawRec.tuneStressWeight ?? merged.tuneStressWeight, defaultForm.tuneStressWeight, 0, 1e9),
    minRoundTrips: normalizeWholeNumber(rawRec.minRoundTrips ?? merged.minRoundTrips, defaultForm.minRoundTrips, 0, 1_000_000),
    walkForwardFolds: normalizeWholeNumber(rawRec.walkForwardFolds ?? merged.walkForwardFolds, defaultForm.walkForwardFolds, 1, 1000),
    walkForwardEmbargoBars: normalizeWholeNumber(
      rawRec.walkForwardEmbargoBars ?? merged.walkForwardEmbargoBars,
      defaultForm.walkForwardEmbargoBars,
      0,
      1e9,
    ),
    normalization: normalizeNormalization(rawRec.normalization ?? merged.normalization, defaultForm.normalization),
    kalmanZMin,
    kalmanZMax,
    minEdge: normalizeFiniteNumber(rawRec.minEdge ?? merged.minEdge, defaultForm.minEdge, 0, 1e9),
    minSignalToNoise: normalizeFiniteNumber(
      rawRec.minSignalToNoise ?? merged.minSignalToNoise,
      defaultForm.minSignalToNoise,
      0,
      1e9,
    ),
    costAwareEdge: normalizeBool(rawRec.costAwareEdge ?? merged.costAwareEdge, defaultForm.costAwareEdge),
    edgeBuffer: normalizeFiniteNumber(rawRec.edgeBuffer ?? merged.edgeBuffer, defaultForm.edgeBuffer, 0, 1e9),
    trendLookback,
    maxPositionSize: normalizeFiniteNumber(rawRec.maxPositionSize ?? merged.maxPositionSize, defaultForm.maxPositionSize, 0, 1e9),
    volTarget: normalizeFiniteNumber(rawRec.volTarget ?? merged.volTarget, defaultForm.volTarget, 0, 1e9),
    volLookback,
    volEwmaAlpha: normalizeFiniteNumber(rawRec.volEwmaAlpha ?? merged.volEwmaAlpha, defaultForm.volEwmaAlpha, 0, 0.999999),
    volFloor: normalizeFiniteNumber(rawRec.volFloor ?? merged.volFloor, defaultForm.volFloor, 0, 1e9),
    volScaleMax: normalizeFiniteNumber(rawRec.volScaleMax ?? merged.volScaleMax, defaultForm.volScaleMax, 0, 1e9),
    maxVolatility: normalizeFiniteNumber(rawRec.maxVolatility ?? merged.maxVolatility, defaultForm.maxVolatility, 0, 1e9),
    rebalanceBars,
    rebalanceThreshold: normalizeFiniteNumber(
      rawRec.rebalanceThreshold ?? merged.rebalanceThreshold,
      defaultForm.rebalanceThreshold,
      0,
      1e9,
    ),
    rebalanceCostMult: normalizeFiniteNumber(
      rawRec.rebalanceCostMult ?? merged.rebalanceCostMult,
      defaultForm.rebalanceCostMult,
      0,
      1e9,
    ),
    rebalanceGlobal: normalizeBool(rawRec.rebalanceGlobal ?? merged.rebalanceGlobal, defaultForm.rebalanceGlobal),
    rebalanceResetOnSignal: normalizeBool(
      rawRec.rebalanceResetOnSignal ?? merged.rebalanceResetOnSignal,
      defaultForm.rebalanceResetOnSignal,
    ),
    fundingRate: normalizeFiniteNumber(rawRec.fundingRate ?? merged.fundingRate, defaultForm.fundingRate, -1e9, 1e9),
    fundingBySide: normalizeBool(rawRec.fundingBySide ?? merged.fundingBySide, defaultForm.fundingBySide),
    fundingOnOpen: normalizeBool(rawRec.fundingOnOpen ?? merged.fundingOnOpen, defaultForm.fundingOnOpen),
    blendWeight: normalizeFiniteNumber(rawRec.blendWeight ?? merged.blendWeight, defaultForm.blendWeight, 0, 1),
    routerLookback,
    routerMinScore: normalizeFiniteNumber(rawRec.routerMinScore ?? merged.routerMinScore, defaultForm.routerMinScore, 0, 1),
    maxHighVolProb: normalizeFiniteNumber(rawRec.maxHighVolProb ?? merged.maxHighVolProb, 0, 0, 1),
    maxConformalWidth: normalizeFiniteNumber(rawRec.maxConformalWidth ?? merged.maxConformalWidth, 0, 0, 1e9),
    maxQuantileWidth: normalizeFiniteNumber(rawRec.maxQuantileWidth ?? merged.maxQuantileWidth, 0, 0, 1e9),
    confirmConformal: normalizeBool(rawRec.confirmConformal ?? merged.confirmConformal, defaultForm.confirmConformal),
    confirmQuantiles: normalizeBool(rawRec.confirmQuantiles ?? merged.confirmQuantiles, defaultForm.confirmQuantiles),
    confidenceSizing: normalizeBool(rawRec.confidenceSizing ?? merged.confidenceSizing, defaultForm.confidenceSizing),
    optimizeOperations: normalizeBool(rawRec.optimizeOperations ?? merged.optimizeOperations, defaultForm.optimizeOperations),
    sweepThreshold: normalizeBool(rawRec.sweepThreshold ?? merged.sweepThreshold, defaultForm.sweepThreshold),
    bypassCache: normalizeBool(rawRec.bypassCache ?? merged.bypassCache, defaultForm.bypassCache),
    autoRefresh: normalizeBool(rawRec.autoRefresh ?? merged.autoRefresh, defaultForm.autoRefresh),
    autoRefreshSec: normalizeFiniteNumber(rawRec.autoRefreshSec ?? merged.autoRefreshSec, defaultForm.autoRefreshSec, 5, 600),
    positionsOpenTimeCacheSec: normalizeFiniteNumber(
      rawRec.positionsOpenTimeCacheSec ?? merged.positionsOpenTimeCacheSec,
      defaultForm.positionsOpenTimeCacheSec,
      0,
      86_400,
    ),
    epochs,
    learningRate: normalizeFiniteNumber(rawRec.learningRate ?? merged.learningRate, defaultForm.learningRate, 0, 1),
    valRatio: normalizeFiniteNumber(rawRec.valRatio ?? merged.valRatio, defaultForm.valRatio, 0, 1),
    patience,
    gradClip: normalizeFiniteNumber(rawRec.gradClip ?? merged.gradClip, defaultForm.gradClip, 0, 10),
    hiddenSize,
    minPositionSize: normalizeFiniteNumber(rawRec.minPositionSize ?? merged.minPositionSize, defaultForm.minPositionSize, 0, 1),
    orderQuote,
    orderQuantity,
    orderQuoteFraction,
    maxOrderQuote,
    botPollSeconds,
    botOnlineEpochs,
    botTrainBars,
    botMaxPoints,
    botProtectionOrders: normalizeBool(rawRec.botProtectionOrders ?? merged.botProtectionOrders, defaultForm.botProtectionOrders),
    botAdoptExistingPosition: true,
  };
}
