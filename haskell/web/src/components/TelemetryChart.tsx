import React, { useMemo, useRef, useState } from "react";

export type TelemetryPoint = {
  atMs: number;
  pollLatencyMs: number | null;
  driftBps: number | null;
};

type Props = {
  points: TelemetryPoint[];
  height?: number;
  label?: string;
};

type Pads = { l: number; r: number; t: number; b: number };

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

function isFiniteNumber(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
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

function fmtTimeMs(ms: number): string {
  if (!Number.isFinite(ms)) return "—";
  try {
    return new Date(ms).toLocaleString();
  } catch {
    return String(ms);
  }
}

function fmtMs(ms: number | null): string {
  if (!isFiniteNumber(ms)) return "—";
  return `${Math.round(ms)}ms`;
}

function fmtBps(bps: number | null): string {
  if (!isFiniteNumber(bps)) return "—";
  const sign = bps > 0 ? "+" : "";
  return `${sign}${bps.toFixed(Math.abs(bps) < 10 ? 2 : Math.abs(bps) < 100 ? 1 : 0)} bp`;
}

function pathFor(series: Array<number | null>, xFor: (i: number) => number, yFor: (v: number) => number): string {
  const n = series.length;
  if (n < 2) return "";
  let d = "";
  let penDown = false;
  let pts = 0;
  for (let i = 0; i < n; i += 1) {
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
  return pts < 2 ? "" : d.trim();
}

export function TelemetryChart({ points, height = 120, label = "Telemetry chart" }: Props) {
  const w = 1000;
  const h = 240;
  const pad: Pads = { l: 66, r: 66, t: 18, b: 34 };

  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [pointer, setPointer] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  const { pollSeries, driftSeries, leftAxis, rightAxis, pollPath, driftPath, xFor, yForLeft, yForRight } = useMemo(() => {
    const pollSeries = points.map((p) => (isFiniteNumber(p.pollLatencyMs) ? p.pollLatencyMs : null));
    const driftSeries = points.map((p) => (isFiniteNumber(p.driftBps) ? p.driftBps : null));

    const pollFinite = pollSeries.filter((v): v is number => isFiniteNumber(v));
    const pollMax = pollFinite.length ? Math.max(0, ...pollFinite) : 1;
    const leftAxis = niceTicks(0, pollMax === 0 ? 1 : pollMax, 5);

    const driftFinite = driftSeries.filter((v): v is number => isFiniteNumber(v));
    const driftMin = driftFinite.length ? Math.min(0, ...driftFinite) : -1;
    const driftMax = driftFinite.length ? Math.max(0, ...driftFinite) : 1;
    const rightAxis = niceTicks(driftMin, driftMax, 5);

    const spanLeft = leftAxis.max - leftAxis.min || 1;
    const spanRight = rightAxis.max - rightAxis.min || 1;

    const n = points.length;
    const xFor = (i: number) => pad.l + (i * (w - pad.l - pad.r)) / Math.max(1, n - 1);
    const yForLeft = (v: number) => {
      const t = (v - leftAxis.min) / spanLeft;
      return pad.t + (1 - clamp(t, 0, 1)) * (h - pad.t - pad.b);
    };
    const yForRight = (v: number) => {
      const t = (v - rightAxis.min) / spanRight;
      return pad.t + (1 - clamp(t, 0, 1)) * (h - pad.t - pad.b);
    };

    const pollPath = pathFor(pollSeries, xFor, yForLeft);
    const driftPath = pathFor(driftSeries, xFor, yForRight);

    return { pollSeries, driftSeries, leftAxis, rightAxis, pollPath, driftPath, xFor, yForLeft, yForRight };
  }, [points]);

  const n = points.length;
  const empty = n < 2 || (!pollPath && !driftPath);

  const hover = useMemo(() => {
    if (hoverIdx === null) return null;
    if (n < 1) return null;
    const idx = clamp(hoverIdx, 0, n - 1);
    const p = points[idx]!;
    const pollLatencyMs = pollSeries[idx] ?? null;
    const driftBps = driftSeries[idx] ?? null;
    const driftPct = isFiniteNumber(driftBps) ? driftBps / 100 : null;
    return { idx, atMs: p.atMs, pollLatencyMs, driftBps, driftPct };
  }, [driftSeries, hoverIdx, n, points, pollSeries]);

  const tooltipStyle = useMemo(() => {
    if (!pointer || !hover) return { display: "none" } as React.CSSProperties;
    const pad = 12;
    const tw = 290;
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

    if (n < 2) return;
    const xSvg = (x / Math.max(1, rect.width)) * w;
    const t = clamp((xSvg - pad.l) / Math.max(1, w - pad.l - pad.r), 0, 1);
    const idx = clamp(Math.round(t * (n - 1)), 0, n - 1);
    setHoverIdx(idx);
  };

  const onPointerLeave = () => {
    setPointer(null);
    setHoverIdx(null);
  };

  const hoverX = hover ? xFor(hover.idx) : null;
  const hoverYLeft = hover && isFiniteNumber(hover.pollLatencyMs) ? yForLeft(hover.pollLatencyMs) : null;
  const hoverYRight = hover && isFiniteNumber(hover.driftBps) ? yForRight(hover.driftBps) : null;
  const driftZeroY = yForRight(0);

  return (
    <div
      ref={wrapRef}
      className="chart btChart"
      style={{ height, position: "relative" }}
      role="img"
      aria-label={label}
      onPointerMove={onPointerMove}
      onPointerLeave={onPointerLeave}
    >
      {hover ? (
        <div className="btTooltip" style={tooltipStyle} aria-hidden={false}>
          <div className="btTooltipTitle">
            <span className="badge">poll</span>
            <span className="badge">{fmtTimeMs(hover.atMs)}</span>
          </div>
          <div className="btTooltipRow">
            <div className="k">binance poll latency</div>
            <div className="v">{fmtMs(hover.pollLatencyMs)}</div>
          </div>
          <div className="btTooltipRow">
            <div className="k">close drift</div>
            <div className="v">
              {fmtBps(hover.driftBps)}
              {isFiniteNumber(hover.driftPct) ? ` (${hover.driftPct.toFixed(3)}%)` : ""}
            </div>
          </div>
        </div>
      ) : null}

      {empty ? (
        <div className="chartEmpty">Not enough data</div>
      ) : (
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="chartSvg" aria-hidden="true">
          <g>
            {leftAxis.ticks.map((tv) => {
              const y = yForLeft(tv);
              const isZero = Math.abs(tv) < (leftAxis.step || 1) * 1e-6;
              const label = `${tv.toFixed(decimalsForStep(leftAxis.step))}ms`;
              return (
                <g key={`yl-${tv}`}>
                  <line
                    x1={pad.l}
                    x2={w - pad.r}
                    y1={y}
                    y2={y}
                    stroke={isZero ? "rgba(255,255,255,0.16)" : "rgba(255,255,255,0.07)"}
                    strokeWidth={isZero ? "2" : "1"}
                  />
                  <text x={pad.l - 10} y={y + 4} fill="rgba(255,255,255,0.62)" fontSize="14" fontFamily="monospace" textAnchor="end">
                    {label}
                  </text>
                </g>
              );
            })}
          </g>

          <g>
            {rightAxis.ticks.map((tv) => {
              const y = yForRight(tv);
              const label = `${tv.toFixed(decimalsForStep(rightAxis.step))}bp`;
              return (
                <g key={`yr-${tv}`}>
                  <text x={w - pad.r + 10} y={y + 4} fill="rgba(255,255,255,0.55)" fontSize="14" fontFamily="monospace" textAnchor="start">
                    {label === "-0bp" || label === "-0.0bp" || label === "-0.00bp" || label === "-0.000bp" ? label.slice(1) : label}
                  </text>
                </g>
              );
            })}
          </g>

          <line
            x1={pad.l}
            x2={w - pad.r}
            y1={driftZeroY}
            y2={driftZeroY}
            stroke="rgba(124, 58, 237, 0.22)"
            strokeWidth="2"
            strokeDasharray="6 6"
          />

          {pollPath ? <path d={pollPath} fill="none" stroke="rgba(14, 165, 233, 0.95)" strokeWidth="3" strokeLinecap="round" /> : null}
          {driftPath ? <path d={driftPath} fill="none" stroke="rgba(124, 58, 237, 0.95)" strokeWidth="3" strokeLinecap="round" /> : null}

          {hoverX !== null ? (
            <line x1={hoverX} x2={hoverX} y1={pad.t} y2={h - pad.b} stroke="rgba(255,255,255,0.16)" strokeWidth="1" />
          ) : null}
          {hoverX !== null && hoverYLeft !== null ? <circle cx={hoverX} cy={hoverYLeft} r="6" fill="rgba(14, 165, 233, 0.95)" /> : null}
          {hoverX !== null && hoverYRight !== null ? <circle cx={hoverX} cy={hoverYRight} r="6" fill="rgba(124, 58, 237, 0.95)" /> : null}
        </svg>
      )}
    </div>
  );
}

