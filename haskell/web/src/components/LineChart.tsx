import React from "react";

type Props = {
  series: number[];
  height?: number | string;
  label?: string;
};

const DEFAULT_CHART_HEIGHT = "var(--chart-height)";

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
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

export function LineChart({ series, height = DEFAULT_CHART_HEIGHT, label = "Series chart" }: Props) {
  const w = 1000;
  const h = 240;
  const pad = { l: 66, r: 14, t: 14, b: 34 };
  const resolvedHeight = typeof height === "string" ? height : DEFAULT_CHART_HEIGHT;
  const minHeight = typeof height === "number" ? height : undefined;

  if (series.length < 2) {
    return (
      <div className="chart" style={{ height: resolvedHeight, minHeight }} aria-label={label}>
        <div className="chartEmpty">Not enough points</div>
      </div>
    );
  }

  const safe = series.filter((v) => Number.isFinite(v));
  const min = Math.min(...safe);
  const max = Math.max(...safe);
  const yAxis = niceTicks(min, max, 5);
  const yMin = yAxis.min;
  const yMax = yAxis.max;
  const span = yMax - yMin || 1;

  const pts = series.map((v, i) => {
    const x = pad.l + (i * (w - pad.l - pad.r)) / (series.length - 1);
    const t = (v - yMin) / span;
    const y = pad.t + (1 - clamp(t, 0, 1)) * (h - pad.t - pad.b);
    return { x, y };
  });

  const line = pts.map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`)).join(" ");
  const first = pts[0]!;
  const last = pts[pts.length - 1]!;
  const yBase = h - pad.b;
  const area = `${line} L ${last.x} ${yBase} L ${first.x} ${yBase} Z`;

  return (
    <div className="chart" style={{ height: resolvedHeight, minHeight }} role="img" aria-label={label}>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="chartSvg">
        <g>
          {yAxis.ticks.map((tv) => {
            const t = (tv - yMin) / span;
            const y = pad.t + (1 - clamp(t, 0, 1)) * (h - pad.t - pad.b);
            const v = Math.abs(tv) < (yAxis.step || 1) * 1e-9 ? 0 : tv;
            const label = v.toFixed(decimalsForStep(yAxis.step));
            return (
              <g key={`y-${tv}`}>
                <line x1={pad.l} x2={w - pad.r} y1={y} y2={y} stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
                <text x={pad.l - 10} y={y + 6} fill="rgba(255,255,255,0.62)" fontSize="18" fontFamily="monospace" textAnchor="end">
                  {label === "-0" || label === "-0.0" || label === "-0.00" || label === "-0.000" ? label.slice(1) : label}
                </text>
              </g>
            );
          })}
        </g>

        <defs>
          <linearGradient id="eqFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(14, 165, 233, 0.28)" />
            <stop offset="85%" stopColor="rgba(14, 165, 233, 0.02)" />
          </linearGradient>
          <linearGradient id="eqStroke" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="rgba(14, 165, 233, 0.85)" />
            <stop offset="55%" stopColor="rgba(20, 184, 166, 0.95)" />
            <stop offset="100%" stopColor="rgba(34, 197, 94, 0.85)" />
          </linearGradient>
        </defs>

        <path d={area} fill="url(#eqFill)" />
        <path d={line} fill="none" stroke="url(#eqStroke)" strokeWidth="6" strokeLinecap="round" />

        <g>
          {(() => {
            const target = clamp(Math.round((w - pad.l - pad.r) / 240) + 1, 2, 6);
            const ticks: number[] = [];
            for (let k = 0; k < target; k += 1) {
              const idx = Math.round((k * (series.length - 1)) / Math.max(1, target - 1));
              if (ticks[ticks.length - 1] !== idx) ticks.push(idx);
            }
            return ticks.map((idx) => {
              const x = pad.l + (idx * (w - pad.l - pad.r)) / Math.max(1, series.length - 1);
              return (
                <g key={`x-${idx}`}>
                  <line x1={x} x2={x} y1={h - pad.b} y2={h - pad.b + 7} stroke="rgba(255,255,255,0.14)" strokeWidth="1" />
                  <text x={x} y={h - pad.b + 24} fill="rgba(255,255,255,0.62)" fontSize="18" fontFamily="monospace" textAnchor="middle">
                    {idx}
                  </text>
                </g>
              );
            });
          })()}
        </g>
      </svg>
    </div>
  );
}
