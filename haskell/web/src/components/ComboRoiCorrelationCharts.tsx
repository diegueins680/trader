import React, { useEffect, useMemo, useState } from "react";
import { comboMarketLabel, comboMarketValue } from "../app/comboMarket";
import { methodLabelFromMeta } from "../app/methodMeta";
import { fmtPct, fmtRatio } from "../lib/format";
import type { OptimizationCombo } from "./TopCombosChart";

type ComboRoiPoint = {
  comboId: number;
  x: number;
  roi: number;
};

type ComboRoiCorrelation = {
  key: string;
  label: string;
  correlation: number;
  sampleCount: number;
  points: ComboRoiPoint[];
  xMin: number;
  xMax: number;
  xActualMin: number;
  xActualMax: number;
  roiMin: number;
  roiMax: number;
  roiActualMin: number;
  roiActualMax: number;
  slope: number;
  intercept: number;
};

type ComboProfitabilitySummary = {
  comboCount: number;
  profitableCount: number;
  profitRate: number;
  avgRoi: number | null;
  medianRoi: number | null;
  bestRoi: number | null;
  worstRoi: number | null;
  avgAnnualizedReturn: number | null;
  avgSharpe: number | null;
  avgMaxDrawdown: number | null;
  totalRoundTrips: number | null;
};

type ComboGroupDimension = "method" | "symbol" | "interval" | "strategy";

type ComboGroupStat = {
  key: string;
  label: string;
  count: number;
  avgRoi: number;
  medianRoi: number;
  bestRoi: number;
  worstRoi: number;
  profitRate: number;
  avgAnnualizedReturn: number | null;
  avgSharpe: number | null;
  avgMaxDrawdown: number | null;
  avgWinRate: number | null;
  totalRoundTrips: number | null;
};

type RiskReturnPoint = {
  comboId: number;
  label: string;
  method: string;
  symbol: string;
  xDrawdown: number;
  yReturn: number;
  roi: number;
  sharpe: number | null;
  roundTrips: number | null;
};

type HeatmapAxis = {
  key: string;
  label: string;
};

type MethodSymbolHeatmapCell = {
  methodKey: string;
  symbolKey: string;
  methodLabel: string;
  symbolLabel: string;
  count: number;
  avgRoi: number;
  profitRate: number;
  avgSharpe: number | null;
};

type MethodSymbolHeatmap = {
  methods: HeatmapAxis[];
  symbols: HeatmapAxis[];
  cells: MethodSymbolHeatmapCell[];
  roiMaxAbs: number;
};

type ConditionalCorrelationSignal = {
  key: string;
  label: string;
  correlation: number;
  sampleCount: number;
  stableContexts: number;
  checkedContexts: number;
  strongestContext: string | null;
  strongestCorrelation: number | null;
};

type MetricCorrelationSignal = {
  key: string;
  label: string;
  metric: "annualizedReturn" | "sharpe" | "maxDrawdown";
  correlation: number;
  sampleCount: number;
};

type InteractionSignal = {
  key: string;
  leftLabel: string;
  rightLabel: string;
  lift: number;
  avgRoi: number;
  sampleCount: number;
  sideA: "high" | "low";
  sideB: "high" | "low";
};

type Props = {
  combos: OptimizationCombo[];
  loading: boolean;
};

const MIN_CORRELATION_SAMPLES = 4;
const MAX_CORRELATION_CHARTS = 8;
const MAX_RENDERED_POINTS = 180;
const MAX_GROUP_BARS = 8;
const MAX_HEATMAP_METHODS = 7;
const MAX_HEATMAP_SYMBOLS = 9;
const MAX_RISK_RETURN_POINTS = 220;
const CHART_W = 420;
const CHART_H = 180;
const CHART_PAD = { l: 34, r: 14, t: 14, b: 30 };
const DETAIL_CHART_W = 980;
const DETAIL_CHART_H = 560;
const DETAIL_CHART_PAD = { l: 76, r: 30, t: 34, b: 62 };
const RISK_CHART_W = 620;
const RISK_CHART_H = 260;
const RISK_CHART_PAD = { l: 52, r: 18, t: 18, b: 38 };

const PARAM_LABELS: Record<string, string> = {
  baseCloseThreshold: "Base close threshold",
  baseOpenThreshold: "Base open threshold",
  blendWeight: "Blend weight",
  bars: "Bars",
  closeThreshold: "Close threshold",
  cooldownBars: "Cooldown bars",
  edgeBuffer: "Edge buffer",
  epochs: "Epochs",
  fee: "Fee",
  gradClip: "Grad clip",
  hiddenSize: "Hidden size",
  kalmanDt: "Kalman dt",
  kalmanMarketTopN: "Kalman market top N",
  kalmanMeasurementVar: "Kalman measurement var",
  kalmanProcessVar: "Kalman process var",
  kalmanZMax: "Kalman Z max",
  kalmanZMin: "Kalman Z min",
  learningRate: "Learning rate",
  maxConformalWidth: "Max conformal width",
  maxDrawdown: "Max drawdown",
  maxHighVolProb: "Max high-vol prob",
  maxHoldBars: "Max hold bars",
  maxPositionSize: "Max position size",
  maxQuantileWidth: "Max quantile width",
  maxVolatility: "Max volatility",
  minEdge: "Min edge",
  minHoldBars: "Min hold bars",
  minPositionSize: "Min position size",
  minSignalToNoise: "Min signal/noise",
  openThreshold: "Open threshold",
  orderQuote: "Order quote",
  orderQuoteFraction: "Order quote fraction",
  patience: "Patience",
  periodsPerYear: "Periods/year",
  rebalanceBars: "Rebalance bars",
  rebalanceCostMult: "Rebalance cost mult",
  rebalanceThreshold: "Rebalance threshold",
  routerLookback: "Router lookback",
  routerRegimeMinBars: "Router regime bars",
  routerRegimeMinFraction: "Router regime fraction",
  routerMinScore: "Router min score",
  slippage: "Slippage",
  spread: "Spread",
  stopLoss: "Stop loss",
  stopLossVolMult: "Stop-loss vol mult",
  takeProfit: "Take profit",
  takeProfitVolMult: "Take-profit vol mult",
  trailingStop: "Trailing stop",
  trailingStopVolMult: "Trailing-stop vol mult",
  trendLookback: "Trend lookback",
  tuneStressShock: "Stress shock",
  tuneStressVolMult: "Stress vol mult",
  tuneStressWeight: "Stress weight",
  valRatio: "Validation ratio",
  volEwmaAlpha: "Vol EWMA alpha",
  volFloor: "Vol floor",
  volLookback: "Vol lookback",
  volScaleMax: "Vol scale max",
  volTarget: "Vol target",
  walkForwardEmbargoBars: "WF embargo bars",
  walkForwardFolds: "WF folds",
};

function paramNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  return null;
}

function comboRoi(combo: OptimizationCombo): number | null {
  const roi = combo.finalEquity - 1;
  return Number.isFinite(roi) ? roi : null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function metricNumber(combo: OptimizationCombo, key: string): number | null {
  const metrics = combo.metrics as Record<string, unknown> | null | undefined;
  return finiteNumber(metrics?.[key]);
}

function comboAnnualizedReturn(combo: OptimizationCombo): number | null {
  return metricNumber(combo, "annualizedReturn");
}

function comboSharpe(combo: OptimizationCombo): number | null {
  return metricNumber(combo, "sharpe");
}

function comboMaxDrawdown(combo: OptimizationCombo): number | null {
  const drawdown = metricNumber(combo, "maxDrawdown");
  return drawdown == null ? null : Math.abs(drawdown);
}

function comboWinRate(combo: OptimizationCombo): number | null {
  return metricNumber(combo, "winRate");
}

function comboRoundTrips(combo: OptimizationCombo): number | null {
  return (
    metricNumber(combo, "roundTrips") ??
    metricNumber(combo, "tradeCount") ??
    (combo.operations && combo.operations.length > 0 ? combo.operations.length : null)
  );
}

function fmtParamValue(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    if (Number.isInteger(value)) return value.toString();
    const fixed = value.toFixed(6);
    return fixed.replace(/\.?0+$/, "");
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return value.trim() ? value : "—";
  try {
    const json = JSON.stringify(value);
    return json.length > 140 ? `${json.slice(0, 137)}...` : json;
  } catch {
    return String(value);
  }
}

function comboMethodLabel(combo: OptimizationCombo): string {
  return methodLabelFromMeta(combo.params.method);
}

function comboSymbolLabel(combo: OptimizationCombo): string {
  const raw = combo.params.binanceSymbol;
  if (typeof raw === "string" && raw.trim()) return raw.trim().toUpperCase();
  const market = comboMarketLabel(comboMarketValue(combo));
  return market === "CSV" ? "CSV input" : market;
}

function comboIntervalLabel(combo: OptimizationCombo): string {
  const raw = combo.params.interval;
  return typeof raw === "string" && raw.trim() ? raw.trim() : "unknown";
}

function compactDimensionValue(value: unknown): string | null {
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed || trimmed.toLowerCase() === "none") return null;
    return trimmed;
  }
  if (typeof value === "boolean") return value ? "enabled" : null;
  return null;
}

function comboStrategyLabel(combo: OptimizationCombo): string {
  const parts = [
    comboMethodLabel(combo),
    compactDimensionValue(combo.params.positioning),
    compactDimensionValue(combo.params.normalization),
  ].filter((part): part is string => Boolean(part));
  return parts.length > 0 ? parts.join(" · ") : comboMethodLabel(combo);
}

function average(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const current = sorted[mid];
  if (current == null) return null;
  if (sorted.length % 2 === 1) return current;
  const prev = sorted[mid - 1];
  return prev == null ? current : (prev + current) / 2;
}

function sum(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((acc, value) => acc + value, 0);
}

function buildProfitabilitySummary(combos: OptimizationCombo[]): ComboProfitabilitySummary {
  const roiValues: number[] = [];
  const annualizedReturns: number[] = [];
  const sharpeValues: number[] = [];
  const drawdownValues: number[] = [];
  const roundTripValues: number[] = [];

  for (const combo of combos) {
    const roi = comboRoi(combo);
    if (roi != null) roiValues.push(roi);

    const annualizedReturn = comboAnnualizedReturn(combo);
    if (annualizedReturn != null) annualizedReturns.push(annualizedReturn);

    const sharpe = comboSharpe(combo);
    if (sharpe != null) sharpeValues.push(sharpe);

    const drawdown = comboMaxDrawdown(combo);
    if (drawdown != null) drawdownValues.push(drawdown);

    const roundTrips = comboRoundTrips(combo);
    if (roundTrips != null) roundTripValues.push(roundTrips);
  }

  const profitableCount = roiValues.filter((roi) => roi > 0).length;
  return {
    comboCount: combos.length,
    profitableCount,
    profitRate: roiValues.length > 0 ? profitableCount / roiValues.length : 0,
    avgRoi: average(roiValues),
    medianRoi: median(roiValues),
    bestRoi: roiValues.length > 0 ? Math.max(...roiValues) : null,
    worstRoi: roiValues.length > 0 ? Math.min(...roiValues) : null,
    avgAnnualizedReturn: average(annualizedReturns),
    avgSharpe: average(sharpeValues),
    avgMaxDrawdown: average(drawdownValues),
    totalRoundTrips: sum(roundTripValues),
  };
}

type MutableComboGroup = {
  key: string;
  label: string;
  roiValues: number[];
  annualizedReturns: number[];
  sharpeValues: number[];
  drawdownValues: number[];
  winRates: number[];
  roundTrips: number[];
};

function comboGroupIdentity(combo: OptimizationCombo, dimension: ComboGroupDimension): { key: string; label: string } {
  switch (dimension) {
    case "method":
      return { key: `method:${combo.params.method}`, label: comboMethodLabel(combo) };
    case "symbol": {
      const label = comboSymbolLabel(combo);
      return { key: `symbol:${label}`, label };
    }
    case "interval": {
      const label = comboIntervalLabel(combo);
      return { key: `interval:${label}`, label };
    }
    case "strategy": {
      const label = comboStrategyLabel(combo);
      return { key: `strategy:${combo.params.method}:${combo.params.positioning ?? ""}:${combo.params.normalization}`, label };
    }
    default:
      return { key: "unknown", label: "Unknown" };
  }
}

function buildGroupStats(
  combos: OptimizationCombo[],
  dimension: ComboGroupDimension,
  maxGroups = MAX_GROUP_BARS,
): ComboGroupStat[] {
  const byGroup = new Map<string, MutableComboGroup>();

  for (const combo of combos) {
    const roi = comboRoi(combo);
    if (roi == null) continue;
    const identity = comboGroupIdentity(combo, dimension);
    const group =
      byGroup.get(identity.key) ??
      ({
        ...identity,
        roiValues: [],
        annualizedReturns: [],
        sharpeValues: [],
        drawdownValues: [],
        winRates: [],
        roundTrips: [],
      } satisfies MutableComboGroup);
    group.roiValues.push(roi);

    const annualizedReturn = comboAnnualizedReturn(combo);
    if (annualizedReturn != null) group.annualizedReturns.push(annualizedReturn);

    const sharpe = comboSharpe(combo);
    if (sharpe != null) group.sharpeValues.push(sharpe);

    const drawdown = comboMaxDrawdown(combo);
    if (drawdown != null) group.drawdownValues.push(drawdown);

    const winRate = comboWinRate(combo);
    if (winRate != null) group.winRates.push(winRate);

    const roundTrips = comboRoundTrips(combo);
    if (roundTrips != null) group.roundTrips.push(roundTrips);

    byGroup.set(identity.key, group);
  }

  const stats: ComboGroupStat[] = [];
  for (const group of byGroup.values()) {
    const avgRoi = average(group.roiValues);
    const medianRoi = median(group.roiValues);
    if (avgRoi == null || medianRoi == null) continue;
    const profitableCount = group.roiValues.filter((roi) => roi > 0).length;
    stats.push({
      key: group.key,
      label: group.label,
      count: group.roiValues.length,
      avgRoi,
      medianRoi,
      bestRoi: Math.max(...group.roiValues),
      worstRoi: Math.min(...group.roiValues),
      profitRate: profitableCount / group.roiValues.length,
      avgAnnualizedReturn: average(group.annualizedReturns),
      avgSharpe: average(group.sharpeValues),
      avgMaxDrawdown: average(group.drawdownValues),
      avgWinRate: average(group.winRates),
      totalRoundTrips: sum(group.roundTrips),
    });
  }

  stats.sort((a, b) => {
    const avgRoi = b.avgRoi - a.avgRoi;
    if (avgRoi !== 0) return avgRoi;
    const count = b.count - a.count;
    if (count !== 0) return count;
    return a.label.localeCompare(b.label);
  });

  return stats.slice(0, maxGroups);
}

function buildRiskReturnPoints(combos: OptimizationCombo[]): RiskReturnPoint[] {
  const points: RiskReturnPoint[] = [];

  for (const combo of combos) {
    const roi = comboRoi(combo);
    if (roi == null) continue;
    const yReturn = comboAnnualizedReturn(combo) ?? roi;
    if (!Number.isFinite(yReturn)) continue;
    const xDrawdown = comboMaxDrawdown(combo) ?? 0;
    points.push({
      comboId: combo.id,
      label: `#${combo.rank ?? combo.id}`,
      method: comboMethodLabel(combo),
      symbol: comboSymbolLabel(combo),
      xDrawdown,
      yReturn,
      roi,
      sharpe: comboSharpe(combo),
      roundTrips: comboRoundTrips(combo),
    });
  }

  if (points.length <= MAX_RISK_RETURN_POINTS) return points;
  points.sort((a, b) => Math.abs(b.roi) - Math.abs(a.roi));
  return points.slice(0, MAX_RISK_RETURN_POINTS);
}

type AxisCounter = HeatmapAxis & { count: number };

function addAxisCount(map: Map<string, AxisCounter>, key: string, label: string): void {
  const current = map.get(key);
  if (current) {
    current.count += 1;
  } else {
    map.set(key, { key, label, count: 1 });
  }
}

function topAxes(map: Map<string, AxisCounter>, max: number): HeatmapAxis[] {
  return [...map.values()]
    .sort((a, b) => {
      const count = b.count - a.count;
      if (count !== 0) return count;
      return a.label.localeCompare(b.label);
    })
    .slice(0, max)
    .map(({ key, label }) => ({ key, label }));
}

function buildMethodSymbolHeatmap(combos: OptimizationCombo[]): MethodSymbolHeatmap | null {
  const methodCounts = new Map<string, AxisCounter>();
  const symbolCounts = new Map<string, AxisCounter>();
  const cellGroups = new Map<
    string,
    {
      methodKey: string;
      symbolKey: string;
      methodLabel: string;
      symbolLabel: string;
      roiValues: number[];
      sharpeValues: number[];
    }
  >();

  for (const combo of combos) {
    const roi = comboRoi(combo);
    if (roi == null) continue;
    const methodKey = `method:${combo.params.method}`;
    const methodLabel = comboMethodLabel(combo);
    const symbolLabel = comboSymbolLabel(combo);
    const symbolKey = `symbol:${symbolLabel}`;
    addAxisCount(methodCounts, methodKey, methodLabel);
    addAxisCount(symbolCounts, symbolKey, symbolLabel);

    const cellKey = `${methodKey}\u0000${symbolKey}`;
    const cell =
      cellGroups.get(cellKey) ??
      ({
        methodKey,
        symbolKey,
        methodLabel,
        symbolLabel,
        roiValues: [],
        sharpeValues: [],
      } satisfies {
        methodKey: string;
        symbolKey: string;
        methodLabel: string;
        symbolLabel: string;
        roiValues: number[];
        sharpeValues: number[];
      });
    cell.roiValues.push(roi);
    const sharpe = comboSharpe(combo);
    if (sharpe != null) cell.sharpeValues.push(sharpe);
    cellGroups.set(cellKey, cell);
  }

  const methods = topAxes(methodCounts, MAX_HEATMAP_METHODS);
  const symbols = topAxes(symbolCounts, MAX_HEATMAP_SYMBOLS);
  if (methods.length === 0 || symbols.length === 0) return null;
  const methodKeys = new Set(methods.map((axis) => axis.key));
  const symbolKeys = new Set(symbols.map((axis) => axis.key));
  const cells: MethodSymbolHeatmapCell[] = [];

  for (const cell of cellGroups.values()) {
    if (!methodKeys.has(cell.methodKey) || !symbolKeys.has(cell.symbolKey)) continue;
    const avgRoi = average(cell.roiValues);
    if (avgRoi == null) continue;
    cells.push({
      methodKey: cell.methodKey,
      symbolKey: cell.symbolKey,
      methodLabel: cell.methodLabel,
      symbolLabel: cell.symbolLabel,
      count: cell.roiValues.length,
      avgRoi,
      profitRate: cell.roiValues.filter((roi) => roi > 0).length / cell.roiValues.length,
      avgSharpe: average(cell.sharpeValues),
    });
  }

  if (cells.length === 0) return null;
  const roiMaxAbs = Math.max(0.01, ...cells.map((cell) => Math.abs(cell.avgRoi)));
  return { methods, symbols, cells, roiMaxAbs };
}

function niceParamLabel(key: string): string {
  const known = PARAM_LABELS[key];
  if (known) return known;
  const words = key
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .trim();
  if (!words) return key;
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function extendRange(min: number, max: number): [number, number] {
  if (min === max) {
    const delta = Math.abs(min) || 1;
    return [min - delta, max + delta];
  }
  const pad = (max - min) * 0.08;
  return [min - pad, max + pad];
}

function pearson(points: ComboRoiPoint[]): { correlation: number; slope: number; intercept: number } | null {
  if (points.length < MIN_CORRELATION_SAMPLES) return null;
  const n = points.length;
  const meanX = points.reduce((acc, point) => acc + point.x, 0) / n;
  const meanY = points.reduce((acc, point) => acc + point.roi, 0) / n;
  let cov = 0;
  let varX = 0;
  let varY = 0;
  for (const point of points) {
    const dx = point.x - meanX;
    const dy = point.roi - meanY;
    cov += dx * dy;
    varX += dx * dx;
    varY += dy * dy;
  }
  if (varX <= 0 || varY <= 0) return null;
  const correlation = cov / Math.sqrt(varX * varY);
  const slope = cov / varX;
  const intercept = meanY - slope * meanX;
  if (!Number.isFinite(correlation) || !Number.isFinite(slope) || !Number.isFinite(intercept)) return null;
  return { correlation, slope, intercept };
}

function downsamplePoints(points: ComboRoiPoint[]): ComboRoiPoint[] {
  if (points.length <= MAX_RENDERED_POINTS) return points;
  const sampled: ComboRoiPoint[] = [];
  const last = points.length - 1;
  for (let i = 0; i < MAX_RENDERED_POINTS; i += 1) {
    const idx = Math.round((i * last) / (MAX_RENDERED_POINTS - 1));
    const point = points[idx];
    if (point) sampled.push(point);
  }
  return sampled;
}

function comboRoiPointKey(point: ComboRoiPoint): string {
  return `${point.comboId}:${point.x}:${point.roi}`;
}

function strongestRoiPoint(points: ComboRoiPoint[]): ComboRoiPoint | null {
  if (points.length === 0) return null;
  return points.reduce((best, point) => (point.roi > best.roi ? point : best), points[0]!);
}

export function buildComboRoiCorrelations(combos: OptimizationCombo[]): ComboRoiCorrelation[] {
  const byParam = new Map<string, ComboRoiPoint[]>();

  for (const combo of combos) {
    const roi = comboRoi(combo);
    if (roi == null) continue;

    const pushParam = (key: string, rawValue: unknown) => {
      const x = paramNumber(rawValue);
      if (x == null) return;
      const points = byParam.get(key);
      const point = { comboId: combo.id, x, roi };
      if (points) {
        points.push(point);
      } else {
        byParam.set(key, [point]);
      }
    };

    pushParam("openThreshold", combo.openThreshold);
    pushParam("closeThreshold", combo.closeThreshold);

    const rawParams = (combo.params ?? {}) as Record<string, unknown>;
    for (const [key, value] of Object.entries(rawParams)) {
      if (key === "openThreshold" || key === "closeThreshold") continue;
      pushParam(key, value);
    }
  }

  const correlations: ComboRoiCorrelation[] = [];
  for (const [key, points] of byParam) {
    const stats = pearson(points);
    if (!stats) continue;
    const distinctValues = new Set(points.map((point) => point.x));
    if (distinctValues.size < 2) continue;
    const xValues = points.map((point) => point.x);
    const roiValues = points.map((point) => point.roi);
    const xActualMin = Math.min(...xValues);
    const xActualMax = Math.max(...xValues);
    const roiActualMin = Math.min(...roiValues);
    const roiActualMax = Math.max(...roiValues);
    const [xMin, xMax] = extendRange(xActualMin, xActualMax);
    const [roiMin, roiMax] = extendRange(roiActualMin, roiActualMax);
    correlations.push({
      key,
      label: niceParamLabel(key),
      correlation: stats.correlation,
      sampleCount: points.length,
      points,
      xMin,
      xMax,
      xActualMin,
      xActualMax,
      roiMin,
      roiMax,
      roiActualMin,
      roiActualMax,
      slope: stats.slope,
      intercept: stats.intercept,
    });
  }

  correlations.sort((a, b) => {
    const strength = Math.abs(b.correlation) - Math.abs(a.correlation);
    if (strength !== 0) return strength;
    const samples = b.sampleCount - a.sampleCount;
    if (samples !== 0) return samples;
    return a.label.localeCompare(b.label);
  });

  return correlations.slice(0, MAX_CORRELATION_CHARTS);
}

function comboParamValue(combo: OptimizationCombo, key: string): number | null {
  if (key === "openThreshold") return paramNumber(combo.openThreshold);
  if (key === "closeThreshold") return paramNumber(combo.closeThreshold);
  const rawParams = (combo.params ?? {}) as Record<string, unknown>;
  return paramNumber(rawParams[key]);
}

function comboContextLabel(combo: OptimizationCombo, dimension: ComboGroupDimension): string {
  switch (dimension) {
    case "method":
      return comboMethodLabel(combo);
    case "symbol":
      return comboSymbolLabel(combo);
    case "interval":
      return comboIntervalLabel(combo);
    case "strategy":
      return comboStrategyLabel(combo);
    default:
      return "Unknown";
  }
}

function buildConditionalCorrelationSignals(
  correlations: ComboRoiCorrelation[],
  comboById: Map<number, OptimizationCombo>,
): ConditionalCorrelationSignal[] {
  const signals: ConditionalCorrelationSignal[] = [];
  const dimensions: ComboGroupDimension[] = ["symbol", "interval", "method", "strategy"];

  for (const chart of correlations) {
    const conditional: Array<{ label: string; correlation: number; sampleCount: number }> = [];
    for (const dimension of dimensions) {
      const grouped = new Map<string, ComboRoiPoint[]>();
      for (const point of chart.points) {
        const combo = comboById.get(point.comboId);
        if (!combo) continue;
        const key = `${dimension}:${comboContextLabel(combo, dimension)}`;
        const group = grouped.get(key) ?? [];
        group.push(point);
        grouped.set(key, group);
      }
      for (const [key, points] of grouped) {
        if (points.length < MIN_CORRELATION_SAMPLES || new Set(points.map((point) => point.x)).size < 2) continue;
        const stats = pearson(points);
        if (!stats) continue;
        conditional.push({
          label: key.replace(/^[^:]+:/, ""),
          correlation: stats.correlation,
          sampleCount: points.length,
        });
      }
    }
    if (conditional.length === 0) continue;
    const sign = Math.sign(chart.correlation) || 1;
    const stable = conditional.filter((item) => Math.sign(item.correlation) === sign && Math.abs(item.correlation) >= 0.18);
    const strongest = conditional.reduce((best, item) => (Math.abs(item.correlation) > Math.abs(best.correlation) ? item : best), conditional[0]!);
    signals.push({
      key: chart.key,
      label: chart.label,
      correlation: chart.correlation,
      sampleCount: chart.sampleCount,
      stableContexts: stable.length,
      checkedContexts: conditional.length,
      strongestContext: strongest.label,
      strongestCorrelation: strongest.correlation,
    });
  }

  signals.sort((a, b) => {
    const stable = b.stableContexts / Math.max(1, b.checkedContexts) - a.stableContexts / Math.max(1, a.checkedContexts);
    if (stable !== 0) return stable;
    return Math.abs(b.correlation) - Math.abs(a.correlation);
  });
  return signals.slice(0, 6);
}

function metricCorrelationPoints(
  chart: ComboRoiCorrelation,
  comboById: Map<number, OptimizationCombo>,
  metric: MetricCorrelationSignal["metric"],
): Array<{ x: number; y: number }> {
  const points: Array<{ x: number; y: number }> = [];
  for (const point of chart.points) {
    const combo = comboById.get(point.comboId);
    if (!combo) continue;
    const value =
      metric === "annualizedReturn"
        ? comboAnnualizedReturn(combo)
        : metric === "sharpe"
          ? comboSharpe(combo)
          : comboMaxDrawdown(combo);
    if (value == null) continue;
    points.push({ x: point.x, y: value });
  }
  return points;
}

function pearsonXY(points: Array<{ x: number; y: number }>): number | null {
  if (points.length < MIN_CORRELATION_SAMPLES) return null;
  const meanX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
  const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
  let cov = 0;
  let varX = 0;
  let varY = 0;
  for (const point of points) {
    const dx = point.x - meanX;
    const dy = point.y - meanY;
    cov += dx * dy;
    varX += dx * dx;
    varY += dy * dy;
  }
  if (varX <= 0 || varY <= 0) return null;
  const corr = cov / Math.sqrt(varX * varY);
  return Number.isFinite(corr) ? corr : null;
}

function buildMetricCorrelationSignals(
  correlations: ComboRoiCorrelation[],
  comboById: Map<number, OptimizationCombo>,
): MetricCorrelationSignal[] {
  const metrics: MetricCorrelationSignal["metric"][] = ["annualizedReturn", "sharpe", "maxDrawdown"];
  const signals: MetricCorrelationSignal[] = [];
  for (const chart of correlations) {
    for (const metric of metrics) {
      const points = metricCorrelationPoints(chart, comboById, metric);
      const correlation = pearsonXY(points);
      if (correlation == null || Math.abs(correlation) < 0.18) continue;
      signals.push({
        key: `${chart.key}:${metric}`,
        label: chart.label,
        metric,
        correlation,
        sampleCount: points.length,
      });
    }
  }
  signals.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
  return signals.slice(0, 6);
}

function buildInteractionSignals(correlations: ComboRoiCorrelation[], combos: OptimizationCombo[]): InteractionSignal[] {
  const candidates = correlations.slice(0, 8);
  const roiValues = combos.map((combo) => comboRoi(combo)).filter((roi): roi is number => roi != null);
  const baseAvg = average(roiValues);
  if (candidates.length < 2 || baseAvg == null) return [];
  const targetFor = (chart: ComboRoiCorrelation) => {
    const topCount = Math.max(2, Math.ceil(chart.points.length * 0.25));
    return median([...chart.points].sort((a, b) => b.roi - a.roi).slice(0, topCount).map((point) => point.x));
  };
  const targets = new Map(candidates.map((chart) => [chart.key, targetFor(chart)]));
  const matches = (combo: OptimizationCombo, chart: ComboRoiCorrelation): boolean => {
    const target = targets.get(chart.key);
    const value = comboParamValue(combo, chart.key);
    if (target == null || value == null) return false;
    return chart.correlation >= 0 ? value >= target : value <= target;
  };
  const interactions: InteractionSignal[] = [];

  for (let i = 0; i < candidates.length; i += 1) {
    const left = candidates[i];
    if (!left) continue;
    for (let j = i + 1; j < candidates.length; j += 1) {
      const right = candidates[j];
      if (!right) continue;
      const matchedRois = combos
        .filter((combo) => matches(combo, left) && matches(combo, right))
        .map((combo) => comboRoi(combo))
        .filter((roi): roi is number => roi != null);
      if (matchedRois.length < 3) continue;
      const avgRoi = average(matchedRois);
      if (avgRoi == null) continue;
      const lift = avgRoi - baseAvg;
      if (!Number.isFinite(lift) || lift <= 0) continue;
      interactions.push({
        key: `${left.key}:${right.key}`,
        leftLabel: left.label,
        rightLabel: right.label,
        lift,
        avgRoi,
        sampleCount: matchedRois.length,
        sideA: left.correlation >= 0 ? "high" : "low",
        sideB: right.correlation >= 0 ? "high" : "low",
      });
    }
  }

  interactions.sort((a, b) => {
    const lift = b.lift - a.lift;
    if (lift !== 0) return lift;
    return b.sampleCount - a.sampleCount;
  });
  return interactions.slice(0, 5);
}

function fmtCompact(value: number): string {
  const abs = Math.abs(value);
  if (!Number.isFinite(value)) return "-";
  if (abs === 0) return "0";
  if (abs >= 1000) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  if (abs >= 0.01) return value.toFixed(4);
  return value.toExponential(1);
}

function fmtCorrelation(value: number): string {
  const rounded = value.toFixed(2);
  return rounded === "-0.00" ? "0.00" : rounded;
}

function scale(value: number, min: number, max: number, outMin: number, outMax: number): number {
  const span = max - min || 1;
  return outMin + ((value - min) / span) * (outMax - outMin);
}

function fmtMaybePct(value: number | null, digits = 1): string {
  return value == null ? "—" : fmtPct(value, digits);
}

function fmtMaybeNumber(value: number | null, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function fmtMaybeCount(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return Math.round(value).toString();
}

function MetricTile({
  label,
  value,
  detail,
  tone = "neutral",
}: {
  label: string;
  value: string;
  detail?: string;
  tone?: "gain" | "loss" | "neutral";
}) {
  return (
    <div className={`comboMetricTile comboMetricTile${tone === "gain" ? "Gain" : tone === "loss" ? "Loss" : "Neutral"}`}>
      <div className="comboMetricLabel">{label}</div>
      <div className="comboMetricValue">{value}</div>
      {detail ? <div className="comboMetricDetail">{detail}</div> : null}
    </div>
  );
}

function ComboProfitabilityStrip({ summary }: { summary: ComboProfitabilitySummary }) {
  const avgTone = summary.avgRoi == null ? "neutral" : summary.avgRoi >= 0 ? "gain" : "loss";
  const medianTone = summary.medianRoi == null ? "neutral" : summary.medianRoi >= 0 ? "gain" : "loss";
  return (
    <div className="comboMetricGrid" aria-label="Filtered combo profitability summary">
      <MetricTile
        label="Combos"
        value={summary.comboCount.toString()}
        detail={`${summary.profitableCount} profitable`}
      />
      <MetricTile label="Profit rate" value={fmtPct(summary.profitRate, 0)} />
      <MetricTile label="Avg ROI" value={fmtMaybePct(summary.avgRoi, 1)} tone={avgTone} />
      <MetricTile label="Median ROI" value={fmtMaybePct(summary.medianRoi, 1)} tone={medianTone} />
      <MetricTile label="Best ROI" value={fmtMaybePct(summary.bestRoi, 1)} tone="gain" />
      <MetricTile label="Worst ROI" value={fmtMaybePct(summary.worstRoi, 1)} tone="loss" />
      <MetricTile label="Avg annualized" value={fmtMaybePct(summary.avgAnnualizedReturn, 1)} />
      <MetricTile label="Avg Sharpe" value={fmtMaybeNumber(summary.avgSharpe, 2)} />
      <MetricTile label="Avg MaxDD" value={fmtMaybePct(summary.avgMaxDrawdown, 1)} tone="loss" />
      <MetricTile label="Round trips" value={fmtMaybeCount(summary.totalRoundTrips)} />
    </div>
  );
}

function ComboGroupBars({ title, groups }: { title: string; groups: ComboGroupStat[] }) {
  const maxAbs = Math.max(0.01, ...groups.map((group) => Math.abs(group.avgRoi)));

  return (
    <div className="comboAnalyticsPanel">
      <div className="comboAnalyticsPanelHeader">
        <div className="comboRoiParam">{title}</div>
        {groups.length > 0 ? <span className="badge">{groups.length}</span> : null}
      </div>
      {groups.length === 0 ? (
        <div className="hint">No grouped ROI data.</div>
      ) : (
        <div className="comboGroupBars">
          {groups.map((group) => {
            const width = Math.max(3, (Math.abs(group.avgRoi) / maxAbs) * 100);
            const gain = group.avgRoi >= 0;
            const titleText = `${group.label}: avg ROI ${fmtPct(group.avgRoi, 2)}, median ${fmtPct(
              group.medianRoi,
              2,
            )}, profit rate ${fmtPct(group.profitRate, 0)}, n=${group.count}`;
            return (
              <div key={group.key} className="comboGroupBarRow" title={titleText}>
                <div className="comboGroupBarLabel">
                  <span>{group.label}</span>
                  <span>{group.count}</span>
                </div>
                <div className="comboGroupBarTrack">
                  <div
                    className={`comboGroupBarFill${gain ? " comboGroupBarGain" : " comboGroupBarLoss"}`}
                    style={{ width: `${width}%` }}
                  />
                </div>
                <div className="comboGroupBarMeta">
                  <span>{fmtPct(group.avgRoi, 1)}</span>
                  <span>Win {fmtPct(group.profitRate, 0)}</span>
                  <span>Sharpe {fmtMaybeNumber(group.avgSharpe, 2)}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function metricSignalLabel(metric: MetricCorrelationSignal["metric"]): string {
  switch (metric) {
    case "annualizedReturn":
      return "Annualized";
    case "sharpe":
      return "Sharpe";
    case "maxDrawdown":
      return "MaxDD";
    default:
      return metric;
  }
}

function ComboSignalPanels({
  conditionalSignals,
  metricSignals,
  interactionSignals,
}: {
  conditionalSignals: ConditionalCorrelationSignal[];
  metricSignals: MetricCorrelationSignal[];
  interactionSignals: InteractionSignal[];
}) {
  return (
    <div className="comboAnalyticsSignalGrid">
      <div className="comboAnalyticsPanel">
        <div className="comboAnalyticsPanelHeader">
          <div className="comboRoiParam">Stable signals</div>
          {conditionalSignals.length > 0 ? <span className="badge">{conditionalSignals.length}</span> : null}
        </div>
        {conditionalSignals.length === 0 ? (
          <div className="hint">No stable context signals.</div>
        ) : (
          <div className="comboSignalList">
            {conditionalSignals.map((signal) => (
              <div key={signal.key} className="comboSignalRow">
                <div>
                  <span>{signal.label}</span>
                  <small>
                    {signal.stableContexts}/{signal.checkedContexts} contexts
                  </small>
                </div>
                <strong>r {fmtCorrelation(signal.correlation)}</strong>
                <small>
                  {signal.strongestContext ?? "mixed"}{" "}
                  {signal.strongestCorrelation == null ? "" : `r ${fmtCorrelation(signal.strongestCorrelation)}`}
                </small>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="comboAnalyticsPanel">
        <div className="comboAnalyticsPanelHeader">
          <div className="comboRoiParam">Risk signals</div>
          {metricSignals.length > 0 ? <span className="badge">{metricSignals.length}</span> : null}
        </div>
        {metricSignals.length === 0 ? (
          <div className="hint">No secondary metric signals.</div>
        ) : (
          <div className="comboSignalList">
            {metricSignals.map((signal) => (
              <div key={signal.key} className="comboSignalRow">
                <div>
                  <span>{signal.label}</span>
                  <small>{metricSignalLabel(signal.metric)}</small>
                </div>
                <strong>r {fmtCorrelation(signal.correlation)}</strong>
                <small>{signal.sampleCount} points</small>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="comboAnalyticsPanel">
        <div className="comboAnalyticsPanelHeader">
          <div className="comboRoiParam">Interactions</div>
          {interactionSignals.length > 0 ? <span className="badge">{interactionSignals.length}</span> : null}
        </div>
        {interactionSignals.length === 0 ? (
          <div className="hint">No lifted parameter pairs.</div>
        ) : (
          <div className="comboSignalList">
            {interactionSignals.map((signal) => (
              <div key={signal.key} className="comboSignalRow">
                <div>
                  <span>
                    {signal.leftLabel} {signal.sideA} + {signal.rightLabel} {signal.sideB}
                  </span>
                  <small>{signal.sampleCount} combos</small>
                </div>
                <strong>{fmtPct(signal.lift, 1)}</strong>
                <small>Avg {fmtPct(signal.avgRoi, 1)}</small>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function RiskReturnScatter({ points }: { points: RiskReturnPoint[] }) {
  if (points.length === 0) {
    return (
      <div className="comboAnalyticsPanel">
        <div className="comboAnalyticsPanelHeader">
          <div className="comboRoiParam">Risk / return map</div>
        </div>
        <div className="hint">No risk-return metrics available.</div>
      </div>
    );
  }

  const x1 = RISK_CHART_PAD.l;
  const x2 = RISK_CHART_W - RISK_CHART_PAD.r;
  const y1 = RISK_CHART_PAD.t;
  const y2 = RISK_CHART_H - RISK_CHART_PAD.b;
  const xActualMax = Math.max(0.01, ...points.map((point) => point.xDrawdown));
  const xMax = xActualMax * 1.08;
  const yValues = points.map((point) => point.yReturn);
  const yActualMin = Math.min(...yValues);
  const yActualMax = Math.max(...yValues);
  const [yMin, yMax] = extendRange(yActualMin, yActualMax);
  const xFor = (value: number) => scale(value, 0, xMax, x1, x2);
  const yFor = (value: number) => scale(value, yMin, yMax, y2, y1);
  const zeroY = yMin <= 0 && yMax >= 0 ? yFor(0) : null;

  return (
    <div className="comboAnalyticsPanel">
      <div className="comboAnalyticsPanelHeader">
        <div className="comboRoiParam">Risk / return map</div>
        <span className="badge">{points.length} points</span>
      </div>
      <svg
        viewBox={`0 0 ${RISK_CHART_W} ${RISK_CHART_H}`}
        className="comboRiskSvg"
        role="img"
        aria-label="Combo return versus max drawdown"
      >
        <line x1={x1} x2={x2} y1={y2} y2={y2} className="comboRoiAxis" />
        <line x1={x1} x2={x1} y1={y1} y2={y2} className="comboRoiAxis" />
        {zeroY != null ? <line x1={x1} x2={x2} y1={zeroY} y2={zeroY} className="comboRoiZeroLine" /> : null}
        {points.map((point) => {
          const roundTrips = point.roundTrips ?? 0;
          const radius = Math.max(3, Math.min(9, 3 + Math.log10(Math.max(1, roundTrips)) * 2.2));
          return (
            <g key={`${point.comboId}-${point.symbol}-${point.method}`}>
              <circle
                cx={xFor(point.xDrawdown)}
                cy={yFor(point.yReturn)}
                r={radius}
                className={point.roi >= 0 ? "comboRoiPointGain" : "comboRoiPointLoss"}
              />
              <title>
                {point.label} {point.symbol} · {point.method} · return {fmtPct(point.yReturn, 2)} · MaxDD{" "}
                {fmtPct(point.xDrawdown, 2)} · ROI {fmtPct(point.roi, 2)} · Sharpe {fmtMaybeNumber(point.sharpe, 2)}
              </title>
            </g>
          );
        })}
        <text x={x1} y={RISK_CHART_H - 9} className="comboRoiAxisLabel" textAnchor="start">
          MaxDD 0%
        </text>
        <text x={x2} y={RISK_CHART_H - 9} className="comboRoiAxisLabel" textAnchor="end">
          {fmtPct(xActualMax, 0)}
        </text>
        <text x={x1 - 9} y={y1 + 6} className="comboRoiAxisLabel" textAnchor="end">
          {fmtPct(yActualMax, 0)}
        </text>
        <text x={x1 - 9} y={y2 + 5} className="comboRoiAxisLabel" textAnchor="end">
          {fmtPct(yActualMin, 0)}
        </text>
      </svg>
    </div>
  );
}

function heatmapCellStyle(cell: MethodSymbolHeatmapCell, heatmap: MethodSymbolHeatmap): React.CSSProperties {
  const intensity = Math.min(1, Math.abs(cell.avgRoi) / heatmap.roiMaxAbs);
  const alpha = 0.12 + intensity * 0.58;
  const rgb = cell.avgRoi >= 0 ? "20, 184, 166" : "239, 68, 68";
  return {
    background: `rgba(${rgb}, ${alpha})`,
    borderColor: `rgba(${rgb}, ${Math.min(0.82, alpha + 0.12)})`,
  };
}

function MethodSymbolHeatmap({ heatmap }: { heatmap: MethodSymbolHeatmap | null }) {
  if (!heatmap) {
    return (
      <div className="comboAnalyticsPanel comboAnalyticsPanelWide">
        <div className="comboAnalyticsPanelHeader">
          <div className="comboRoiParam">Method / symbol heatmap</div>
        </div>
        <div className="hint">No method-symbol ROI matrix available.</div>
      </div>
    );
  }

  const cellMap = new Map(heatmap.cells.map((cell) => [`${cell.methodKey}\u0000${cell.symbolKey}`, cell]));
  const gridTemplateColumns = `minmax(104px, 1.25fr) repeat(${heatmap.symbols.length}, minmax(72px, 1fr))`;

  return (
    <div className="comboAnalyticsPanel comboAnalyticsPanelWide">
      <div className="comboAnalyticsPanelHeader">
        <div className="comboRoiParam">Method / symbol heatmap</div>
        <span className="badge">{heatmap.cells.length} cells</span>
      </div>
      <div className="comboHeatmap" style={{ gridTemplateColumns }}>
        <div className="comboHeatmapCorner">Avg ROI</div>
        {heatmap.symbols.map((symbol) => (
          <div key={symbol.key} className="comboHeatmapAxis comboHeatmapSymbol">
            {symbol.label}
          </div>
        ))}
        {heatmap.methods.map((method) => (
          <React.Fragment key={method.key}>
            <div className="comboHeatmapAxis comboHeatmapMethod">{method.label}</div>
            {heatmap.symbols.map((symbol) => {
              const cell = cellMap.get(`${method.key}\u0000${symbol.key}`);
              if (!cell) {
                return (
                  <div key={`${method.key}-${symbol.key}`} className="comboHeatmapCell comboHeatmapCellEmpty">
                    —
                  </div>
                );
              }
              return (
                <div
                  key={`${method.key}-${symbol.key}`}
                  className="comboHeatmapCell"
                  style={heatmapCellStyle(cell, heatmap)}
                  title={`${cell.methodLabel} / ${cell.symbolLabel}: avg ROI ${fmtPct(cell.avgRoi, 2)}, win ${fmtPct(
                    cell.profitRate,
                    0,
                  )}, n=${cell.count}, Sharpe ${fmtMaybeNumber(cell.avgSharpe, 2)}`}
                >
                  <span>{fmtPct(cell.avgRoi, 0)}</span>
                  <small>{cell.count}</small>
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

type ChartPad = {
  l: number;
  r: number;
  t: number;
  b: number;
};

type ComboRoiScatterProps = {
  chart: ComboRoiCorrelation;
  points?: ComboRoiPoint[];
  width?: number;
  height?: number;
  pad?: ChartPad;
  className?: string;
  interactive?: boolean;
  selectedPointKey?: string | null;
  onPointHover?: (point: ComboRoiPoint | null) => void;
  onPointSelect?: (point: ComboRoiPoint) => void;
  comboById?: Map<number, OptimizationCombo>;
};

function comboPointTitle(
  chart: ComboRoiCorrelation,
  point: ComboRoiPoint,
  combo: OptimizationCombo | null | undefined,
): string {
  const comboLabel = combo ? `#${combo.rank ?? combo.id} ${comboSymbolLabel(combo)} · ${comboMethodLabel(combo)}` : `#${point.comboId}`;
  const sharpe = combo ? comboSharpe(combo) : null;
  return `${comboLabel} · ${chart.label} ${fmtCompact(point.x)} · ROI ${fmtPct(point.roi, 2)} · Sharpe ${fmtMaybeNumber(sharpe, 2)}`;
}

function ComboRoiScatter({
  chart,
  points,
  width = CHART_W,
  height = CHART_H,
  pad = CHART_PAD,
  className = "comboRoiSvg",
  interactive = false,
  selectedPointKey = null,
  onPointHover,
  onPointSelect,
  comboById,
}: ComboRoiScatterProps) {
  const renderedPoints = points ?? downsamplePoints(chart.points);
  const x1 = pad.l;
  const x2 = width - pad.r;
  const y1 = pad.t;
  const y2 = height - pad.b;
  const xFor = (value: number) => scale(value, chart.xMin, chart.xMax, x1, x2);
  const yFor = (value: number) => scale(value, chart.roiMin, chart.roiMax, y2, y1);
  const trendStartY = chart.slope * chart.xMin + chart.intercept;
  const trendEndY = chart.slope * chart.xMax + chart.intercept;
  const zeroY = chart.roiMin <= 0 && chart.roiMax >= 0 ? yFor(0) : null;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className={className} role="img" aria-label={`${chart.label} ROI correlation`}>
      <line x1={x1} x2={x2} y1={y2} y2={y2} className="comboRoiAxis" />
      <line x1={x1} x2={x1} y1={y1} y2={y2} className="comboRoiAxis" />
      {zeroY != null ? <line x1={x1} x2={x2} y1={zeroY} y2={zeroY} className="comboRoiZeroLine" /> : null}
      <line
        x1={xFor(chart.xMin)}
        x2={xFor(chart.xMax)}
        y1={yFor(trendStartY)}
        y2={yFor(trendEndY)}
        className="comboRoiTrend"
      />
      {renderedPoints.map((point, idx) => {
        const pointKey = comboRoiPointKey(point);
        const selected = selectedPointKey === pointKey;
        const combo = comboById?.get(point.comboId);
        const pointClass = [
          point.roi >= 0 ? "comboRoiPointGain" : "comboRoiPointLoss",
          interactive ? "comboRoiPointInteractive" : "",
          selected ? "comboRoiPointSelected" : "",
        ]
          .filter(Boolean)
          .join(" ");
        return (
          <circle
            key={`${chart.key}-${idx}-${pointKey}`}
            cx={xFor(point.x)}
            cy={yFor(point.roi)}
            r={interactive && selected ? "6.5" : interactive ? "5.2" : "3.2"}
            className={pointClass}
            role={interactive ? "button" : undefined}
            tabIndex={interactive ? 0 : undefined}
            aria-label={interactive ? comboPointTitle(chart, point, combo) : undefined}
            onMouseEnter={interactive ? () => onPointHover?.(point) : undefined}
            onMouseLeave={interactive ? () => onPointHover?.(null) : undefined}
            onFocus={interactive ? () => onPointHover?.(point) : undefined}
            onBlur={interactive ? () => onPointHover?.(null) : undefined}
            onClick={interactive ? () => onPointSelect?.(point) : undefined}
            onKeyDown={
              interactive
                ? (event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onPointSelect?.(point);
                    }
                  }
                : undefined
            }
          >
            <title>{comboPointTitle(chart, point, combo)}</title>
          </circle>
        );
      })}
      <text x={x1} y={height - 7} className="comboRoiAxisLabel" textAnchor="start">
        {fmtCompact(chart.xActualMin)}
      </text>
      <text x={x2} y={height - 7} className="comboRoiAxisLabel" textAnchor="end">
        {fmtCompact(chart.xActualMax)}
      </text>
      <text x={x1 - 8} y={y1 + 5} className="comboRoiAxisLabel" textAnchor="end">
        {fmtPct(chart.roiActualMax, 0)}
      </text>
      <text x={x1 - 8} y={y2 + 5} className="comboRoiAxisLabel" textAnchor="end">
        {fmtPct(chart.roiActualMin, 0)}
      </text>
    </svg>
  );
}

function ComboRoiPointMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="comboRoiInspectorMetric">
      <div className="comboRoiInspectorMetricLabel">{label}</div>
      <div className="comboRoiInspectorMetricValue">{value}</div>
    </div>
  );
}

function comboPointParamEntries(combo: OptimizationCombo | null, chart: ComboRoiCorrelation, point: ComboRoiPoint): Array<[string, unknown]> {
  const entries = new Map<string, unknown>();
  if (combo) {
    entries.set("openThreshold", combo.openThreshold);
    entries.set("closeThreshold", combo.closeThreshold);
    const rawParams = (combo.params ?? {}) as Record<string, unknown>;
    for (const [key, value] of Object.entries(rawParams)) {
      if (value !== undefined && !entries.has(key)) entries.set(key, value);
    }
  }
  if (!entries.has(chart.key)) entries.set(chart.key, point.x);
  return [...entries.entries()].sort(([a], [b]) => {
    if (a === chart.key) return -1;
    if (b === chart.key) return 1;
    return a.localeCompare(b);
  });
}

function ComboRoiPointInspector({
  chart,
  point,
  combo,
}: {
  chart: ComboRoiCorrelation;
  point: ComboRoiPoint | null;
  combo: OptimizationCombo | null;
}) {
  if (!point) {
    return (
      <aside className="comboRoiInspector">
        <div className="hint">No point selected.</div>
      </aside>
    );
  }

  const paramEntries = comboPointParamEntries(combo, chart, point);
  const title = combo ? `#${combo.rank ?? combo.id} · ${comboSymbolLabel(combo)}` : `#${point.comboId}`;
  const subtitle = combo
    ? `${comboMarketLabel(comboMarketValue(combo))} · ${comboMethodLabel(combo)} · ${comboIntervalLabel(combo)} · bars=${combo.params.bars}`
    : chart.label;
  const objectiveLabel = combo?.objective && typeof combo.objective === "string" ? combo.objective : "—";
  const scoreLabel = typeof combo?.score === "number" && Number.isFinite(combo.score) ? combo.score.toFixed(4) : "—";

  return (
    <aside className="comboRoiInspector" aria-label="Selected correlation point">
      <div>
        <div className="comboRoiInspectorTitle">{title}</div>
        <div className="comboRoiInspectorSubtitle">{subtitle}</div>
      </div>
      <div className="comboRoiInspectorGrid">
        <ComboRoiPointMetric label={chart.label} value={fmtParamValue(point.x)} />
        <ComboRoiPointMetric label="ROI" value={fmtPct(point.roi, 2)} />
        <ComboRoiPointMetric label="Final equity" value={combo ? fmtRatio(combo.finalEquity, 4) : "—"} />
        <ComboRoiPointMetric label="Annualized" value={combo ? fmtMaybePct(comboAnnualizedReturn(combo), 2) : "—"} />
        <ComboRoiPointMetric label="Sharpe" value={combo ? fmtMaybeNumber(comboSharpe(combo), 2) : "—"} />
        <ComboRoiPointMetric label="MaxDD" value={combo ? fmtMaybePct(comboMaxDrawdown(combo), 2) : "—"} />
        <ComboRoiPointMetric label="Win rate" value={combo ? fmtMaybePct(comboWinRate(combo), 0) : "—"} />
        <ComboRoiPointMetric label="Round trips" value={combo ? fmtMaybeCount(comboRoundTrips(combo)) : "—"} />
        <ComboRoiPointMetric label="Objective" value={objectiveLabel} />
        <ComboRoiPointMetric label="Score" value={scoreLabel} />
      </div>
      <div className="comboRoiInspectorParams">
        <div className="comboRoiInspectorSectionTitle">Parameters</div>
        <div className="comboRoiInspectorParamList">
          {paramEntries.map(([key, value]) => (
            <div key={`${point.comboId}-${key}`} className={key === chart.key ? "comboRoiInspectorParam comboRoiInspectorParamActive" : "comboRoiInspectorParam"}>
              <span>{niceParamLabel(key)}</span>
              <strong>{fmtParamValue(value)}</strong>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

function ComboRoiExpandedDialog({
  chart,
  comboById,
  onClose,
}: {
  chart: ComboRoiCorrelation;
  comboById: Map<number, OptimizationCombo>;
  onClose: () => void;
}) {
  const [selectedPointKey, setSelectedPointKey] = useState<string | null>(() => {
    const point = strongestRoiPoint(chart.points);
    return point ? comboRoiPointKey(point) : null;
  });
  const [hoveredPointKey, setHoveredPointKey] = useState<string | null>(null);

  useEffect(() => {
    const point = strongestRoiPoint(chart.points);
    setSelectedPointKey(point ? comboRoiPointKey(point) : null);
    setHoveredPointKey(null);
  }, [chart]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  const activePointKey = hoveredPointKey ?? selectedPointKey;
  const activePoint =
    (activePointKey ? chart.points.find((point) => comboRoiPointKey(point) === activePointKey) : null) ??
    strongestRoiPoint(chart.points);
  const activeCombo = activePoint ? comboById.get(activePoint.comboId) ?? null : null;
  const chartSelectedPointKey = activePoint ? comboRoiPointKey(activePoint) : selectedPointKey;

  return (
    <div
      className="comboRoiModalBackdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="comboRoiModal" role="dialog" aria-modal="true" aria-label={`${chart.label} ROI correlation details`}>
        <div className="comboRoiModalHeader">
          <div>
            <div className="comboRoiModalTitle">{chart.label}</div>
            <div className="comboRoiModalMeta">
              <span>{chart.sampleCount} points</span>
              <span>
                {fmtCompact(chart.xActualMin)} to {fmtCompact(chart.xActualMax)}
              </span>
              <span>
                ROI {fmtPct(chart.roiActualMin, 0)} to {fmtPct(chart.roiActualMax, 0)}
              </span>
            </div>
          </div>
          <div className="comboRoiModalActions">
            <span className="badge">r {fmtCorrelation(chart.correlation)}</span>
            <button className="btnSmall comboRoiModalClose" type="button" onClick={onClose} aria-label="Close expanded chart">
              ×
            </button>
          </div>
        </div>
        <div className="comboRoiModalContent">
          <div className="comboRoiExpandedChartWrap">
            <ComboRoiScatter
              chart={chart}
              points={chart.points}
              width={DETAIL_CHART_W}
              height={DETAIL_CHART_H}
              pad={DETAIL_CHART_PAD}
              className="comboRoiSvg comboRoiSvgExpanded"
              interactive
              selectedPointKey={chartSelectedPointKey}
              onPointHover={(point) => setHoveredPointKey(point ? comboRoiPointKey(point) : null)}
              onPointSelect={(point) => setSelectedPointKey(comboRoiPointKey(point))}
              comboById={comboById}
            />
          </div>
          <ComboRoiPointInspector chart={chart} point={activePoint} combo={activeCombo} />
        </div>
      </div>
    </div>
  );
}

export const ComboRoiCorrelationCharts = React.memo(function ComboRoiCorrelationCharts({ combos, loading }: Props) {
  const [expandedChartKey, setExpandedChartKey] = useState<string | null>(null);
  const comboById = useMemo(() => new Map(combos.map((combo) => [combo.id, combo])), [combos]);
  const analytics = useMemo(
    () => {
      const correlations = buildComboRoiCorrelations(combos);
      return {
        summary: buildProfitabilitySummary(combos),
        methodGroups: buildGroupStats(combos, "method"),
        symbolGroups: buildGroupStats(combos, "symbol"),
        intervalGroups: buildGroupStats(combos, "interval"),
        strategyGroups: buildGroupStats(combos, "strategy"),
        riskReturnPoints: buildRiskReturnPoints(combos),
        heatmap: buildMethodSymbolHeatmap(combos),
        correlations,
        conditionalSignals: buildConditionalCorrelationSignals(correlations, comboById),
        metricSignals: buildMetricCorrelationSignals(correlations, comboById),
        interactionSignals: buildInteractionSignals(correlations, combos),
      };
    },
    [comboById, combos],
  );
  const charts = analytics.correlations;
  const expandedChart = expandedChartKey ? charts.find((chart) => chart.key === expandedChartKey) ?? null : null;

  useEffect(() => {
    if (expandedChartKey && !expandedChart) setExpandedChartKey(null);
  }, [expandedChartKey, expandedChart]);

  return (
    <section className="comboRoiSection" aria-label="Optimizer combo analytics">
      <div className="comboRoiHeader">
        <div>
          <div className="label">Combo analytics</div>
          <div className="hint">
            ROI, risk, method, strategy, symbol, and parameter views{" "}
            {combos.length > 0 ? `from ${combos.length} filtered combos` : "from filtered combos"}
          </div>
        </div>
        {combos.length > 0 ? (
          <span className="badge">
            AvgEq {fmtRatio(analytics.summary.avgRoi != null ? analytics.summary.avgRoi + 1 : 1, 4)}
          </span>
        ) : null}
      </div>
      {loading && combos.length === 0 ? (
        <div className="hint">Loading combo analytics...</div>
      ) : combos.length === 0 ? (
        <div className="hint">No filtered combos to analyze.</div>
      ) : (
        <>
          <ComboProfitabilityStrip summary={analytics.summary} />
          <div className="comboAnalyticsDashboard">
            <RiskReturnScatter points={analytics.riskReturnPoints} />
            <MethodSymbolHeatmap heatmap={analytics.heatmap} />
          </div>
          <div className="comboAnalyticsGroupGrid">
            <ComboGroupBars title="Methods" groups={analytics.methodGroups} />
            <ComboGroupBars title="Symbols" groups={analytics.symbolGroups} />
            <ComboGroupBars title="Intervals" groups={analytics.intervalGroups} />
            <ComboGroupBars title="Strategies" groups={analytics.strategyGroups} />
          </div>
          <ComboSignalPanels
            conditionalSignals={analytics.conditionalSignals}
            metricSignals={analytics.metricSignals}
            interactionSignals={analytics.interactionSignals}
          />
          <div className="comboRoiSubsection">
            <div className="comboRoiHeader">
              <div>
                <div className="label">Parameter / ROI correlation</div>
                <div className="hint">Numeric and boolean optimizer knobs ranked by absolute Pearson r.</div>
              </div>
              {charts.length > 0 ? <span className="badge">Top {charts.length}</span> : null}
            </div>
            {charts.length === 0 ? (
              <div className="hint">Not enough varied numeric combo parameters yet.</div>
            ) : (
              <div className="comboRoiGrid">
                {charts.map((chart) => (
                  <div
                    key={chart.key}
                    className="comboRoiChart"
                    role="button"
                    tabIndex={0}
                    aria-label={`Open ${chart.label} ROI correlation details`}
                    onClick={() => setExpandedChartKey(chart.key)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setExpandedChartKey(chart.key);
                      }
                    }}
                  >
                    <div className="comboRoiChartHeader">
                      <div className="comboRoiParam">{chart.label}</div>
                      <span className="badge">r {fmtCorrelation(chart.correlation)}</span>
                    </div>
                    <div className="comboRoiMeta">
                      <span>{chart.sampleCount} points</span>
                      <span>
                        ROI {fmtPct(chart.roiActualMin, 0)} to {fmtPct(chart.roiActualMax, 0)}
                      </span>
                    </div>
                    <ComboRoiScatter chart={chart} />
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
      {expandedChart ? (
        <ComboRoiExpandedDialog
          chart={expandedChart}
          comboById={comboById}
          onClose={() => setExpandedChartKey(null)}
        />
      ) : null}
    </section>
  );
});
