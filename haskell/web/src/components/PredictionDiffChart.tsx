import React, { useId, useMemo, useRef, useState } from "react";

type Series = Array<number | null | undefined>;

type Props = {
  prices: number[];
  kalmanPredNext?: Series;
  lstmPredNext?: Series;
  height?: number;
  label?: string;
};

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

function pathFor(series: Array<number | null>, w: number, h: number, padX: number, padY: number, min: number, max: number): string {
  const n = series.length;
  if (n < 2) return "";
  const span = max - min || 1;
  const xFor = (i: number) => padX + (i * (w - padX * 2)) / Math.max(1, n - 1);
  const yFor = (v: number) => {
    const t = (v - min) / span;
    return padY + (1 - clamp(t, 0, 1)) * (h - padY * 2);
  };

  let d = "";
  let penDown = false;
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
  }
  return d.trim();
}

export function PredictionDiffChart({ prices, kalmanPredNext, lstmPredNext, height = 140, label = "Prediction error chart" }: Props) {
  const w = 1000;
  const h = 240;
  const padX = 10;
  const padY = 18;

  const wrapRef = useRef<HTMLDivElement | null>(null);
  const group = useId();
  const [mode, setMode] = useState<ErrorMode>("abs");
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [pointer, setPointer] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  const { kalErr, lstmErr, min, max } = useMemo(() => {
    const kalErr = buildNextPriceError(prices, kalmanPredNext, mode);
    const lstmErr = buildNextPriceError(prices, lstmPredNext, mode);
    const finiteVals = [...kalErr, ...lstmErr].filter((v): v is number => isFiniteNumber(v));
    if (finiteVals.length === 0) return { kalErr, lstmErr, min: -1, max: 1 };
    const min = Math.min(0, ...finiteVals);
    const max = Math.max(0, ...finiteVals);
    if (min === max) return { kalErr, lstmErr, min: min - 1, max: max + 1 };
    return { kalErr, lstmErr, min, max };
  }, [kalmanPredNext, lstmPredNext, mode, prices]);

  const kalPath = useMemo(() => pathFor(kalErr, w, h, padX, padY, min, max), [kalErr, max, min]);
  const lstmPath = useMemo(() => pathFor(lstmErr, w, h, padX, padY, min, max), [lstmErr, max, min]);

  const empty = prices.length < 2 || (!kalPath && !lstmPath);
  const span = max - min || 1;
  const y0 = padY + (1 - clamp((0 - min) / span, 0, 1)) * (h - padY * 2);

  const xFor = (i: number) => padX + (i * (w - padX * 2)) / Math.max(1, prices.length - 1);
  const yFor = (v: number) => {
    const t = (v - min) / span;
    return padY + (1 - clamp(t, 0, 1)) * (h - padY * 2);
  };

  const hover = useMemo(() => {
    if (hoverIdx === null) return null;
    const n = prices.length;
    if (n < 2) return null;
    const idx = clamp(hoverIdx, 0, n - 2);
    const actualNext = prices[idx + 1]!;
    const kalPred = kalmanPredNext?.[idx];
    const lstmPred = lstmPredNext?.[idx];
    const kalE = kalErr[idx];
    const lstmE = lstmErr[idx];
    return {
      idx,
      actualNext,
      kalPred: isFiniteNumber(kalPred) ? kalPred : null,
      lstmPred: isFiniteNumber(lstmPred) ? lstmPred : null,
      kalErr: isFiniteNumber(kalE) ? kalE : null,
      lstmErr: isFiniteNumber(lstmE) ? lstmE : null,
    };
  }, [hoverIdx, kalErr, kalmanPredNext, lstmErr, lstmPredNext, prices]);

  const tooltipStyle = useMemo(() => {
    if (!pointer || !hover) return { display: "none" } as React.CSSProperties;
    const pad = 12;
    const tw = 300;
    const left = clamp(pointer.x + 12, pad, pointer.w - tw - pad);
    const top = clamp(pointer.y + 12, pad, pointer.h - 140 - pad);
    return { left, top, width: tw } as React.CSSProperties;
  }, [hover, pointer]);

  const onPointerMove = (e: React.PointerEvent) => {
    const el = wrapRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setPointer({ x, y, w: rect.width, h: rect.height });

    const n = prices.length;
    if (n < 2) return;
    const xSvg = (x / Math.max(1, rect.width)) * w;
    const t = clamp((xSvg - padX) / Math.max(1, w - padX * 2), 0, 1);
    const idx = clamp(Math.round(t * (n - 1)), 0, n - 2);
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
        style={{ height, position: "relative" }}
        role="img"
        aria-label={label}
        onPointerMove={onPointerMove}
        onPointerLeave={onPointerLeave}
      >
        {hover ? (
          <div className="btTooltip" style={tooltipStyle} aria-hidden={false}>
            <div className="btTooltipTitle">
              <span className="badge">bar {hover.idx}</span>
              <span className="badge">{mode === "pct" ? "percent" : "absolute"}</span>
            </div>
            <div className="btTooltipRow">
              <div className="k">next close</div>
              <div className="v">{fmt(hover.actualNext, 6)}</div>
            </div>
            <div className="btTooltipRow">
              <div className="k">kalman pred</div>
              <div className="v">{hover.kalPred === null ? "—" : fmt(hover.kalPred, 6)}</div>
            </div>
            <div className="btTooltipRow">
              <div className="k">kalman err</div>
              <div className="v">{hover.kalErr === null ? "—" : mode === "pct" ? pct(hover.kalErr, 3) : fmt(hover.kalErr, 6)}</div>
            </div>
            <div className="btTooltipRow">
              <div className="k">lstm pred</div>
              <div className="v">{hover.lstmPred === null ? "—" : fmt(hover.lstmPred, 6)}</div>
            </div>
            <div className="btTooltipRow">
              <div className="k">lstm err</div>
              <div className="v">{hover.lstmErr === null ? "—" : mode === "pct" ? pct(hover.lstmErr, 3) : fmt(hover.lstmErr, 6)}</div>
            </div>
          </div>
        ) : null}

        {empty ? (
          <div className="chartEmpty">Not enough data</div>
        ) : (
          <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="chartSvg" aria-hidden="true">
            <line x1={padX} x2={w - padX} y1={y0} y2={y0} stroke="rgba(255,255,255,0.16)" strokeWidth="2" />
            {kalPath ? <path d={kalPath} fill="none" stroke="rgba(14, 165, 233, 0.85)" strokeWidth="3.5" /> : null}
            {lstmPath ? <path d={lstmPath} fill="none" stroke="rgba(124, 58, 237, 0.9)" strokeWidth="3.5" /> : null}

            {hover ? (
              <>
                <line
                  x1={xFor(hover.idx)}
                  x2={xFor(hover.idx)}
                  y1={padY}
                  y2={h - padY}
                  stroke="rgba(255,255,255,0.16)"
                  strokeWidth="1.5"
                />
                {hover.kalErr !== null ? <circle cx={xFor(hover.idx)} cy={yFor(hover.kalErr)} r={5} fill="rgba(14, 165, 233, 0.95)" /> : null}
                {hover.lstmErr !== null ? <circle cx={xFor(hover.idx)} cy={yFor(hover.lstmErr)} r={5} fill="rgba(124, 58, 237, 0.95)" /> : null}
              </>
            ) : null}

            <g>
              <rect x={padX} y={padY - 12} width={440} height={28} rx={8} fill="rgba(0,0,0,0.35)" />
              <circle cx={padX + 12} cy={padY + 2} r={4} fill="rgba(14, 165, 233, 0.85)" />
              <text x={padX + 22} y={padY + 6} fill="rgba(255,255,255,0.78)" fontSize="12" fontFamily="monospace">
                Kalman error
              </text>
              <circle cx={padX + 128} cy={padY + 2} r={4} fill="rgba(124, 58, 237, 0.9)" />
              <text x={padX + 138} y={padY + 6} fill="rgba(255,255,255,0.78)" fontSize="12" fontFamily="monospace">
                LSTM error {mode === "pct" ? "(pred_next - next_close) / next_close" : "(pred_next - next_close)"}
              </text>
            </g>
          </svg>
        )}
      </div>
    </div>
  );
}
