import React, { useMemo } from "react";
import { fmtPct } from "../lib/format";
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

type Props = {
  combos: OptimizationCombo[];
  loading: boolean;
};

const MIN_CORRELATION_SAMPLES = 4;
const MAX_CORRELATION_CHARTS = 8;
const MAX_RENDERED_POINTS = 180;
const CHART_W = 420;
const CHART_H = 180;
const CHART_PAD = { l: 34, r: 14, t: 14, b: 30 };

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
      points: downsamplePoints(points),
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

function ComboRoiScatter({ chart }: { chart: ComboRoiCorrelation }) {
  const x1 = CHART_PAD.l;
  const x2 = CHART_W - CHART_PAD.r;
  const y1 = CHART_PAD.t;
  const y2 = CHART_H - CHART_PAD.b;
  const xFor = (value: number) => scale(value, chart.xMin, chart.xMax, x1, x2);
  const yFor = (value: number) => scale(value, chart.roiMin, chart.roiMax, y2, y1);
  const trendStartY = chart.slope * chart.xMin + chart.intercept;
  const trendEndY = chart.slope * chart.xMax + chart.intercept;
  const zeroY = chart.roiMin <= 0 && chart.roiMax >= 0 ? yFor(0) : null;

  return (
    <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="comboRoiSvg" role="img" aria-label={`${chart.label} ROI correlation`}>
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
      {chart.points.map((point, idx) => (
        <circle
          key={`${chart.key}-${idx}-${point.comboId}-${point.x}-${point.roi}`}
          cx={xFor(point.x)}
          cy={yFor(point.roi)}
          r="3.2"
          className={point.roi >= 0 ? "comboRoiPointGain" : "comboRoiPointLoss"}
        />
      ))}
      <text x={x1} y={CHART_H - 7} className="comboRoiAxisLabel" textAnchor="start">
        {fmtCompact(chart.xActualMin)}
      </text>
      <text x={x2} y={CHART_H - 7} className="comboRoiAxisLabel" textAnchor="end">
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

export const ComboRoiCorrelationCharts = React.memo(function ComboRoiCorrelationCharts({ combos, loading }: Props) {
  const charts = useMemo(() => buildComboRoiCorrelations(combos), [combos]);

  return (
    <section className="comboRoiSection" aria-label="Combo parameter ROI correlations">
      <div className="comboRoiHeader">
        <div>
          <div className="label">Parameter / ROI correlation</div>
          <div className="hint">ROI {combos.length > 0 ? `from ${combos.length} filtered combos` : "from filtered combos"}</div>
        </div>
        {charts.length > 0 ? <span className="badge">Top {charts.length}</span> : null}
      </div>
      {loading && combos.length === 0 ? (
        <div className="hint">Loading ROI correlations...</div>
      ) : charts.length === 0 ? (
        <div className="hint">Not enough varied numeric combo parameters yet.</div>
      ) : (
        <div className="comboRoiGrid">
          {charts.map((chart) => (
            <div key={chart.key} className="comboRoiChart">
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
    </section>
  );
});
