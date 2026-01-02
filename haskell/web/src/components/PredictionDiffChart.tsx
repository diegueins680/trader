import React, { useId, useMemo, useRef, useState } from "react";
import { fmtTimeMs } from "../app/utils";

type Series = Array<number | null | undefined>;

type Props = {
  prices: number[];
  openTimes?: number[] | null;
  kalmanPredNext?: Series;
  lstmPredNext?: Series;
  startIndex?: number;
  height?: number | string;
  label?: string;
  openThreshold?: number;
  closeThreshold?: number;
};

const DEFAULT_CHART_HEIGHT = "clamp(360px, 75vh, 960px)";

type ErrorMode = "abs" | "pct";

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

function isFiniteNumber(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

function fmt(n: number, digits = 6): string {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function pct(n: number, digits = 2): string {
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
}

function decimalsForStep(step: number, maxDecimals = 6): number {
  if (!Number.isFinite(step) || step === 0) return 0;
  let d = 0;
  let s = Math.abs(step);
  while (d < maxDecimals && Math.abs(Math.round(s) - s) > 1e-9) {
    s *= 10;
    d += 1;
  }
  return d;
}

function niceStep(span: number, ticks: number): number {
  const s = Math.abs(span);
  if (!Number.isFinite(s) || s === 0) return 1;
  const rough = s / Math.max(1, ticks - 1);
  const pow10 = 10 ** Math.floor(Math.log10(rough));
  const err = rough / pow10;
  const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 2.25 ? 2.5 : err >= 1.5 ? 2 : 1;
  return mult * pow10;
}

function niceTicks(min: number, max: number, ticks = 5): { ticks: number[]; min: number; max: number; step: number } {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { ticks: [0, 1], min: 0, max: 1, step: 1 };
  let a = min;
  let b = max;
  if (a === b) {
    const d = Math.abs(a) || 1;
    a -= d;
    b += d;
  }
  const step = niceStep(b - a, ticks);
  const lo = Math.floor(a / step) * step;
  const hi = Math.ceil(b / step) * step;
  const out: number[] = [];
  for (let v = lo; v <= hi + step / 2; v += step) out.push(v);
  return { ticks: out, min: lo, max: hi, step };
}

function buildNextPriceError(prices: number[], predNext: Series | undefined, mode: ErrorMode): Array<number | null> {
  const n = prices.length;
  const out: Array<number | null> = Array.from({ length: n }, () => null);
  if (!predNext || n < 2) return out;
  for (let i = 0; i < n - 1; i++) {
    const pred = predNext[i];
    const actual = prices[i + 1];
    if (!isFiniteNumber(pred) || !isFiniteNumber(actual)) continue;
    if (mode === "pct") {
      if (actual === 0) continue;
      out[i] = (pred - actual) / actual;
    } else {
      out[i] = pred - actual;
    }
  }
  return out;
}

type Pads = { l: number; r: number; t: number; b: number };

function pathFor(series: Array<number | null>, w: number, h: number, pad: Pads, min: number, max: number): string {
  const n = series.length;
  if (n < 2) return "";
  const span = max - min || 1;
  const xFor = (i: number) => pad.l + (i * (w - pad.l - pad.r)) / Math.max(1, n - 1);
  const yFor = (v: number) => {
    const t = (v - min) / span;
    return pad.t + (1 - clamp(t, 0, 1)) * (h - pad.t - pad.b);
  };

  let d = "";
  let penDown = false;
  let pts = 0;
  for (let i = 0; i < n; i++) {
    const v = series[i];
    if (!isFiniteNumber(v)) {
      penDown = false;
      continue;
    }
    const x = xFor(i);
    const y = yFor(v);
    if (!penDown) {
      d += `M ${x} ${y} `;
      penDown = true;
    } else {
      d += `L ${x} ${y} `;
    }
    pts += 1;
  }
  if (pts < 2) return "";
  return d.trim();
}

export function PredictionDiffChart({
  prices,
  openTimes,
  kalmanPredNext,
  lstmPredNext,
  startIndex = 0,
  height = DEFAULT_CHART_HEIGHT,
  label = "Prediction error chart",
  openThreshold,
  closeThreshold,
}: Props) {
  const w = 1000;
  const h = 240;
  const pad: Pads = { l: 66, r: 18, t: 18, b: 34 };
  const nPred = Math.max(0, prices.length - 1);
  const resolvedHeight = typeof height === "string" ? height : DEFAULT_CHART_HEIGHT;
  const minHeight = typeof height === "number" ? height : undefined;

  const wrapRef = useRef<HTMLDivElement | null>(null);
  const group = useId();
  const [mode, setMode] = useState<ErrorMode>("abs");
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [pointer, setPointer] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  const { kalErr, lstmErr, min, max, kalPreds, lstmPreds } = useMemo(() => {
    const kalErr = buildNextPriceError(prices, kalmanPredNext, mode);
    const lstmErr = buildNextPriceError(prices, lstmPredNext, mode);
    
    // Collect actual predictions (next price values)
    const kalPreds = kalmanPredNext ? [...kalmanPredNext] : [];
    const lstmPreds = lstmPredNext ? [...lstmPredNext] : [];
    
    const finiteVals = [...kalErr, ...lstmErr].filter((v): v is number => isFiniteNumber(v));
    if (finiteVals.length === 0) return { kalErr, lstmErr, min: -1, max: 1, kalPreds, lstmPreds };
    const min = Math.min(0, ...finiteVals);
    const max = Math.max(0, ...finiteVals);
    if (min === max) return { kalErr, lstmErr, min: min - 1, max: max + 1, kalPreds, lstmPreds };
    return { kalErr, lstmErr, min, max, kalPreds, lstmPreds };
  }, [kalmanPredNext, lstmPredNext, mode, prices]);

  const yAxis = useMemo(() => niceTicks(min, max, 5), [max, min]);
  const yMin = yAxis.min;
  const yMax = yAxis.max;
  const kalPath = useMemo(() => pathFor(kalErr.slice(0, nPred), w, h, pad, yMin, yMax), [kalErr, nPred, yMax, yMin]);
  const lstmPath = useMemo(() => pathFor(lstmErr.slice(0, nPred), w, h, pad, yMin, yMax), [lstmErr, nPred, yMax, yMin]);

  const empty = nPred < 2 || (!kalPath && !lstmPath);
  const span = yMax - yMin || 1;
  const y0 = pad.t + (1 - clamp((0 - yMin) / span, 0, 1)) * (h - pad.t - pad.b);

  const xFor = (i: number) => pad.l + (i * (w - pad.l - pad.r)) / Math.max(1, nPred - 1);
  const yFor = (v: number) => {
    const t = (v - yMin) / span;
    return pad.t + (1 - clamp(t, 0, 1)) * (h - pad.t - pad.b);
  };

  const hover = useMemo(() => {
    if (hoverIdx === null) return null;
    if (nPred < 1) return null;
    const idx = clamp(hoverIdx, 0, nPred - 1);
    const openTime = openTimes?.[idx];
    const atMs = typeof openTime === "number" && Number.isFinite(openTime) ? openTime : null;
    const actualNext = prices[idx + 1]!;
    const kalPred = kalmanPredNext?.[idx];
    const lstmPred = lstmPredNext?.[idx];
    const kalE = kalErr[idx];
    const lstmE = lstmErr[idx];
    const currentPrice = prices[idx]!;
    const kalDiff = isFiniteNumber(kalPred) ? kalPred - currentPrice : null;
    const lstmDiff = isFiniteNumber(lstmPred) ? lstmPred - currentPrice : null;
    return {
      idx,
      atMs,
      currentPrice,
      actualNext,
      kalPred: isFiniteNumber(kalPred) ? kalPred : null,
      lstmPred: isFiniteNumber(lstmPred) ? lstmPred : null,
      kalDiff,
      lstmDiff,
      kalErr: isFiniteNumber(kalE) ? kalE : null,
      lstmErr: isFiniteNumber(lstmE) ? lstmE : null,
    };
  }, [hoverIdx, kalErr, kalmanPredNext, lstmErr, lstmPredNext, nPred, openTimes, prices]);

  const tooltipStyle = useMemo(() => {
    if (!pointer || !hover) return { display: "none" } as React.CSSProperties;
    const pad = 12;
    const tw = 300;
    const left = clamp(pointer.x + 12, pad, pointer.w - tw - pad);
    const top = clamp(pointer.y + 12, pad, pointer.h - 170 - pad);
    return { left, top, width: tw } as React.CSSProperties;
  }, [hover, pointer]);

  const onPointerMove = (e: React.PointerEvent) => {
    const el = wrapRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setPointer({ x, y, w: rect.width, h: rect.height });

    if (nPred < 2) return;
    const xSvg = (x / Math.max(1, rect.width)) * w;
    const t = clamp((xSvg - pad.l) / Math.max(1, w - pad.l - pad.r), 0, 1);
    const idx = clamp(Math.round(t * (nPred - 1)), 0, nPred - 1);
    setHoverIdx(idx);
  };

  const onPointerLeave = () => {
    setPointer(null);
    setHoverIdx(null);
  };

  return (
    <div>
      <div className="pillRow" style={{ marginBottom: 8, justifyContent: "space-between" }}>
        <div className="hint">{mode === "pct" ? "Percent error" : "Absolute error"}</div>
        <div className="pillRow">
          <label className="pill" style={{ userSelect: "none" }}>
            <input type="radio" name={`${group}-errMode`} checked={mode === "abs"} onChange={() => setMode("abs")} />
            Abs
          </label>
          <label className="pill" style={{ userSelect: "none" }}>
            <input type="radio" name={`${group}-errMode`} checked={mode === "pct"} onChange={() => setMode("pct")} />
            %
          </label>
        </div>
      </div>

      <div
      ref={wrapRef}
      className="chart"
      style={{ height: resolvedHeight, minHeight, position: "relative" }}
      role="img"
      aria-label={label}
        onPointerMove={onPointerMove}
        onPointerLeave={onPointerLeave}
      >
        {hover ? (
          <div className="btTooltip" style={tooltipStyle} aria-hidden={false}>
          <div className="btTooltipTitle">
            <span className="badge">bar {startIndex + hover.idx}</span>
            <span className="badge">{mode === "pct" ? "percent" : "absolute"}</span>
          </div>
          {hover.atMs !== null ? (
            <div className="btTooltipRow">
              <div className="k">time</div>
              <div className="v">{fmtTimeMs(hover.atMs)}</div>
            </div>
          ) : null}
          <div className="btTooltipRow">
            <div className="k">current close</div>
            <div className="v">{fmt(hover.currentPrice, 6)}</div>
            </div>
            <div className="btTooltipRow">
              <div className="k">next close (actual)</div>
              <div className="v">{fmt(hover.actualNext, 6)}</div>
            </div>
            <div className="btTooltipRow">
              <div className="k">kalman pred (next)</div>
              <div className="v">{hover.kalPred === null ? "—" : fmt(hover.kalPred, 6)}</div>
            </div>
            {hover.kalDiff !== null ? (
              <div className="btTooltipRow">
                <div className="k">kalman diff from current</div>
                <div className="v">{fmt(hover.kalDiff, 6)} ({pct(hover.kalDiff / hover.currentPrice, 2)})</div>
              </div>
            ) : null}
            <div className="btTooltipRow">
              <div className="k">kalman err</div>
              <div className="v">{hover.kalErr === null ? "—" : mode === "pct" ? pct(hover.kalErr, 3) : fmt(hover.kalErr, 6)}</div>
            </div>
            <div className="btTooltipRow">
              <div className="k">lstm pred (next)</div>
              <div className="v">{hover.lstmPred === null ? "—" : fmt(hover.lstmPred, 6)}</div>
            </div>
            {hover.lstmDiff !== null ? (
              <div className="btTooltipRow">
                <div className="k">lstm diff from current</div>
                <div className="v">{fmt(hover.lstmDiff, 6)} ({pct(hover.lstmDiff / hover.currentPrice, 2)})</div>
              </div>
            ) : null}
            <div className="btTooltipRow">
              <div className="k">lstm err</div>
              <div className="v">{hover.lstmErr === null ? "—" : mode === "pct" ? pct(hover.lstmErr, 3) : fmt(hover.lstmErr, 6)}</div>
            </div>
            {typeof openThreshold === "number" || typeof closeThreshold === "number" ? (
              <>
                <div style={{ borderTop: "1px solid rgba(255,255,255,0.1)", marginTop: 6, paddingTop: 6 }}>
                  {typeof openThreshold === "number" ? (
                    <div className="btTooltipRow">
                      <div className="k">open threshold</div>
                      <div className="v">{fmt(openThreshold, 6)} ({pct(openThreshold, 2)})</div>
                    </div>
                  ) : null}
                  {typeof closeThreshold === "number" ? (
                    <div className="btTooltipRow">
                      <div className="k">close threshold</div>
                      <div className="v">{fmt(closeThreshold, 6)} ({pct(closeThreshold, 2)})</div>
                    </div>
                  ) : null}
                </div>
              </>
            ) : null}
          </div>
        ) : null}

        {empty ? (
          <div className="chartEmpty">Not enough data</div>
        ) : (
          <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="chartSvg" aria-hidden="true">
            <g>
              {yAxis.ticks.map((tv) => {
                const y = yFor(tv);
                const isZero = Math.abs(tv) < (yAxis.step || 1) * 1e-6;
                const label =
                  mode === "pct"
                    ? `${(tv * 100).toFixed(decimalsForStep(yAxis.step * 100))}%`
                    : tv.toFixed(decimalsForStep(yAxis.step));
                return (
                  <g key={`y-${tv}`}>
                    <line
                      x1={pad.l}
                      x2={w - pad.r}
                      y1={y}
                      y2={y}
                      stroke={isZero ? "rgba(255,255,255,0.16)" : "rgba(255,255,255,0.07)"}
                      strokeWidth={isZero ? "2" : "1"}
                    />
                    <text x={pad.l - 10} y={y + 4} fill="rgba(255,255,255,0.62)" fontSize="14" fontFamily="monospace" textAnchor="end">
                      {label === "-0" || label === "-0.0" || label === "-0.00" || label === "-0.000" ? label.slice(1) : label}
                    </text>
                  </g>
                );
              })}
            </g>

            <g>
              {(() => {
                const target = clamp(Math.round((w - pad.l - pad.r) / 180) + 1, 2, 7);
                const ticks: number[] = [];
                for (let k = 0; k < target; k += 1) {
                  const idx = Math.round((k * Math.max(1, nPred - 1)) / Math.max(1, target - 1));
                  if (ticks[ticks.length - 1] !== idx) ticks.push(idx);
                }
                return ticks.map((idx) => {
                  const x = xFor(idx);
                  return (
                    <g key={`x-${idx}`}>
                      <line x1={x} x2={x} y1={h - pad.b} y2={h - pad.b + 6} stroke="rgba(255,255,255,0.14)" strokeWidth="1" />
                      <text x={x} y={h - pad.b + 22} fill="rgba(255,255,255,0.62)" fontSize="14" fontFamily="monospace" textAnchor="middle">
                        {startIndex + idx}
                      </text>
                    </g>
                  );
                });
              })()}
            </g>

            {kalPath ? <path d={kalPath} fill="none" stroke="rgba(14, 165, 233, 0.85)" strokeWidth="3.5" /> : null}
            {lstmPath ? <path d={lstmPath} fill="none" stroke="rgba(124, 58, 237, 0.9)" strokeWidth="3.5" /> : null}

            {hover ? (
              <>
                <line
                  x1={xFor(hover.idx)}
                  x2={xFor(hover.idx)}
                  y1={pad.t}
                  y2={h - pad.b}
                  stroke="rgba(255,255,255,0.16)"
                  strokeWidth="1.5"
                />
                {hover.kalErr !== null ? <circle cx={xFor(hover.idx)} cy={yFor(hover.kalErr)} r={5} fill="rgba(14, 165, 233, 0.95)" /> : null}
                {hover.lstmErr !== null ? <circle cx={xFor(hover.idx)} cy={yFor(hover.lstmErr)} r={5} fill="rgba(124, 58, 237, 0.95)" /> : null}
              </>
            ) : null}

            <g>
              <rect x={pad.l} y={pad.t - 12} width={500} height={28} rx={8} fill="rgba(0,0,0,0.35)" />
              <circle cx={pad.l + 12} cy={pad.t + 2} r={4} fill="rgba(14, 165, 233, 0.85)" />
              <text x={pad.l + 22} y={pad.t + 6} fill="rgba(255,255,255,0.78)" fontSize="12" fontFamily="monospace">
                Kalman {mode === "pct" ? "% error" : "error"}
              </text>
              <circle cx={pad.l + 160} cy={pad.t + 2} r={4} fill="rgba(124, 58, 237, 0.9)" />
              <text x={pad.l + 170} y={pad.t + 6} fill="rgba(255,255,255,0.78)" fontSize="12" fontFamily="monospace">
                LSTM {mode === "pct" ? "% error" : "error"}
              </text>
            </g>
          </svg>
        )}
      </div>
    </div>
  );
}
