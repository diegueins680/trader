import type { IntrabarFill, Market, Method, Normalization, Platform, Positioning } from "../lib/types";
import { BINANCE_INTERVAL_SECONDS, PLATFORM_INTERVAL_SET, TUNE_OBJECTIVE_SET } from "./constants";
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
  rebalanceGlobal: boolean;
  rebalanceResetOnSignal: boolean;
  fundingRate: number;
  fundingBySide: boolean;
  fundingOnOpen: boolean;
  blendWeight: number;
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

  // Live bot (advanced)
  botPollSeconds: number;
  botOnlineEpochs: number;
  botTrainBars: number;
  botMaxPoints: number;
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
  rebalanceGlobal: false,
  rebalanceResetOnSignal: false,
  fundingRate: 0.1,
  fundingBySide: false,
  fundingOnOpen: false,
  blendWeight: 0.5,
  backtestRatio: 0.2,
  tuneRatio: 0.25,
  tuneObjective: "equity-dd-turnover",
  tunePenaltyMaxDrawdown: 1.5,
  tunePenaltyTurnover: 0.2,
  tuneStressVolMult: 1.0,
  tuneStressShock: 0,
  tuneStressWeight: 0,
  minRoundTrips: 0,
  walkForwardFolds: 7,
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
  orderQuote: 20,
  orderQuantity: 0,
  orderQuoteFraction: 0,
  maxOrderQuote: 0,
  idempotencyKey: "",
  binanceLive: false,
  tradeArmed: false,
  bypassCache: false,
  autoRefresh: false,
  autoRefreshSec: 20,

  botPollSeconds: 0,
  botOnlineEpochs: 1,
  botTrainBars: 800,
  botMaxPoints: 2000,
  botAdoptExistingPosition: true,
};

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

export function parseDurationSeconds(raw: string): number | null {
  const s = raw.trim();
  const m = /^(\d+)([A-Za-z])$/.exec(s);
  if (!m) return null;

  const n = Number(m[1]);
  if (!Number.isFinite(n)) return null;
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
  return n * mult;
}

function normalizePlatformInterval(platform: Platform, raw: unknown, fallback: string): string {
  if (typeof raw !== "string") return fallback;
  const value = raw.trim();
  return PLATFORM_INTERVAL_SET[platform].has(value) ? value : fallback;
}

function normalizeLookbackWindow(raw: unknown, fallback: string): string {
  if (typeof raw !== "string") return fallback;
  const value = raw.trim();
  const sec = parseDurationSeconds(value);
  return sec && sec > 0 ? value : fallback;
}

function normalizeLookbackBars(raw: unknown, fallback: number): number {
  if (typeof raw === "number" && Number.isFinite(raw)) {
    const n = Math.trunc(raw);
    return n >= 2 ? n : 0;
  }
  if (typeof raw === "string") {
    const n = Number(raw);
    if (!Number.isFinite(n)) return fallback;
    const i = Math.trunc(n);
    return i >= 2 ? i : 0;
  }
  return fallback;
}

function normalizeBool(raw: unknown, fallback: boolean): boolean {
  if (typeof raw === "boolean") return raw;
  if (raw === 1 || raw === "1" || raw === "true") return true;
  if (raw === 0 || raw === "0" || raw === "false") return false;
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

function normalizePositioning(raw: unknown, fallback: Positioning): Positioning {
  if (raw === "long-flat" || raw === "long-short") return raw;
  return fallback;
}

function normalizeIntrabarFill(raw: unknown, fallback: IntrabarFill): IntrabarFill {
  if (raw === "stop-first" || raw === "take-profit-first") return raw;
  return fallback;
}

function normalizePlatform(raw: unknown, fallback: Platform): Platform {
  if (raw === "binance" || raw === "coinbase" || raw === "kraken" || raw === "poloniex") return raw;
  return fallback;
}

function normalizeTuneObjective(raw: unknown, fallback: string): string {
  const s = typeof raw === "string" ? raw.trim() : "";
  if (s && TUNE_OBJECTIVE_SET.has(s)) return s;
  if (TUNE_OBJECTIVE_SET.has(fallback)) return fallback;
  return "equity-dd-turnover";
}

export function normalizeFormState(raw: FormStateJson | null | undefined): FormState {
  const merged = { ...defaultForm, ...(raw ?? {}) };
  const rawRec = (raw as Record<string, unknown> | null | undefined) ?? {};
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
  const { threshold: _ignoredThreshold, ...mergedNoLegacy } = merged as FormState & { threshold?: unknown };
  return {
    ...mergedNoLegacy,
    botSymbols,
    platform,
    interval: normalizePlatformInterval(platform, raw?.interval ?? merged.interval, defaultForm.interval),
    positioning: normalizePositioning(raw?.positioning ?? merged.positioning, defaultForm.positioning),
    lookbackWindow: normalizeLookbackWindow(raw?.lookbackWindow ?? merged.lookbackWindow, defaultForm.lookbackWindow),
    lookbackBars: normalizeLookbackBars(raw?.lookbackBars ?? merged.lookbackBars, defaultForm.lookbackBars),
    openThreshold,
    closeThreshold,
    slippage: normalizeFiniteNumber(rawRec.slippage ?? merged.slippage, defaultForm.slippage, 0, 0.999999),
    spread: normalizeFiniteNumber(rawRec.spread ?? merged.spread, defaultForm.spread, 0, 0.999999),
    intrabarFill: normalizeIntrabarFill(rawRec.intrabarFill ?? merged.intrabarFill, defaultForm.intrabarFill),
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
    minHoldBars: normalizeFiniteNumber(rawRec.minHoldBars ?? merged.minHoldBars, defaultForm.minHoldBars, 0, 1e9),
    maxHoldBars: normalizeFiniteNumber(rawRec.maxHoldBars ?? merged.maxHoldBars, defaultForm.maxHoldBars, 0, 1e9),
    cooldownBars: normalizeFiniteNumber(rawRec.cooldownBars ?? merged.cooldownBars, defaultForm.cooldownBars, 0, 1e9),
    tuneRatio: normalizeFiniteNumber(rawRec.tuneRatio ?? merged.tuneRatio, defaultForm.tuneRatio, 0, 0.99),
    tuneObjective: normalizeTuneObjective(rawRec.tuneObjective ?? merged.tuneObjective, defaultForm.tuneObjective),
    tunePenaltyMaxDrawdown: normalizeFiniteNumber(rawRec.tunePenaltyMaxDrawdown ?? merged.tunePenaltyMaxDrawdown, defaultForm.tunePenaltyMaxDrawdown, 0, 1e9),
    tunePenaltyTurnover: normalizeFiniteNumber(rawRec.tunePenaltyTurnover ?? merged.tunePenaltyTurnover, defaultForm.tunePenaltyTurnover, 0, 1e9),
    tuneStressVolMult: normalizeFiniteNumber(rawRec.tuneStressVolMult ?? merged.tuneStressVolMult, defaultForm.tuneStressVolMult, 0, 1e9),
    tuneStressShock: normalizeFiniteNumber(rawRec.tuneStressShock ?? merged.tuneStressShock, defaultForm.tuneStressShock, -1e9, 1e9),
    tuneStressWeight: normalizeFiniteNumber(rawRec.tuneStressWeight ?? merged.tuneStressWeight, defaultForm.tuneStressWeight, 0, 1e9),
    minRoundTrips: normalizeFiniteNumber(rawRec.minRoundTrips ?? merged.minRoundTrips, defaultForm.minRoundTrips, 0, 1e9),
    walkForwardFolds: normalizeFiniteNumber(rawRec.walkForwardFolds ?? merged.walkForwardFolds, defaultForm.walkForwardFolds, 1, 1000),
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
    trendLookback: normalizeFiniteNumber(rawRec.trendLookback ?? merged.trendLookback, defaultForm.trendLookback, 0, 1e9),
    maxPositionSize: normalizeFiniteNumber(rawRec.maxPositionSize ?? merged.maxPositionSize, defaultForm.maxPositionSize, 0, 1e9),
    volTarget: normalizeFiniteNumber(rawRec.volTarget ?? merged.volTarget, defaultForm.volTarget, 0, 1e9),
    volLookback: normalizeFiniteNumber(rawRec.volLookback ?? merged.volLookback, defaultForm.volLookback, 0, 1e9),
    volEwmaAlpha: normalizeFiniteNumber(rawRec.volEwmaAlpha ?? merged.volEwmaAlpha, defaultForm.volEwmaAlpha, 0, 0.999999),
    volFloor: normalizeFiniteNumber(rawRec.volFloor ?? merged.volFloor, defaultForm.volFloor, 0, 1e9),
    volScaleMax: normalizeFiniteNumber(rawRec.volScaleMax ?? merged.volScaleMax, defaultForm.volScaleMax, 0, 1e9),
    maxVolatility: normalizeFiniteNumber(rawRec.maxVolatility ?? merged.maxVolatility, defaultForm.maxVolatility, 0, 1e9),
    rebalanceBars: normalizeFiniteNumber(rawRec.rebalanceBars ?? merged.rebalanceBars, defaultForm.rebalanceBars, 0, 1e9),
    rebalanceThreshold: normalizeFiniteNumber(
      rawRec.rebalanceThreshold ?? merged.rebalanceThreshold,
      defaultForm.rebalanceThreshold,
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
    maxHighVolProb: normalizeFiniteNumber(rawRec.maxHighVolProb ?? merged.maxHighVolProb, 0, 0, 1),
    maxConformalWidth: normalizeFiniteNumber(rawRec.maxConformalWidth ?? merged.maxConformalWidth, 0, 0, 1e9),
    maxQuantileWidth: normalizeFiniteNumber(rawRec.maxQuantileWidth ?? merged.maxQuantileWidth, 0, 0, 1e9),
    confirmConformal: normalizeBool(rawRec.confirmConformal ?? merged.confirmConformal, defaultForm.confirmConformal),
    confirmQuantiles: normalizeBool(rawRec.confirmQuantiles ?? merged.confirmQuantiles, defaultForm.confirmQuantiles),
    confidenceSizing: normalizeBool(rawRec.confidenceSizing ?? merged.confidenceSizing, defaultForm.confidenceSizing),
    bypassCache: normalizeBool(rawRec.bypassCache ?? merged.bypassCache, defaultForm.bypassCache),
    learningRate: normalizeFiniteNumber(rawRec.learningRate ?? merged.learningRate, defaultForm.learningRate, 0, 1),
    valRatio: normalizeFiniteNumber(rawRec.valRatio ?? merged.valRatio, defaultForm.valRatio, 0, 1),
    patience: normalizeFiniteNumber(rawRec.patience ?? merged.patience, defaultForm.patience, 0, 100),
    gradClip: normalizeFiniteNumber(rawRec.gradClip ?? merged.gradClip, defaultForm.gradClip, 0, 10),
    minPositionSize: normalizeFiniteNumber(rawRec.minPositionSize ?? merged.minPositionSize, 0, 0, 1),
    botAdoptExistingPosition: true,
  };
}
