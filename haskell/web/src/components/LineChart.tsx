import React from "react";

type Props = {
  series: number[];
  height?: number;
  label?: string;
};

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

export function LineChart({ series, height = 96, label = "Series chart" }: Props) {
  const w = 1000;
  const h = 240;
  const padX = 6;
  const padY = 14;

  if (series.length < 2) {
    return (
      <div className="chart" style={{ height }} aria-label={label}>
        <div className="chartEmpty">Not enough points</div>
      </div>
    );
  }

  const safe = series.filter((v) => Number.isFinite(v));
  const min = Math.min(...safe);
  const max = Math.max(...safe);
  const span = max - min || 1;

  const pts = series.map((v, i) => {
    const x = padX + (i * (w - padX * 2)) / (series.length - 1);
    const t = (v - min) / span;
    const y = padY + (1 - clamp(t, 0, 1)) * (h - padY * 2);
    return { x, y };
  });

  const line = pts.map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`)).join(" ");
  const first = pts[0];
  const last = pts[pts.length - 1];
  const area = `${line} L ${last.x} ${h - padY} L ${first.x} ${h - padY} Z`;

  return (
    <div className="chart" style={{ height }} role="img" aria-label={label}>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="chartSvg">
        <defs>
          <linearGradient id="eqFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(124, 58, 237, 0.35)" />
            <stop offset="85%" stopColor="rgba(124, 58, 237, 0.02)" />
          </linearGradient>
          <linearGradient id="eqStroke" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="rgba(14, 165, 233, 0.85)" />
            <stop offset="55%" stopColor="rgba(124, 58, 237, 0.95)" />
            <stop offset="100%" stopColor="rgba(34, 197, 94, 0.85)" />
          </linearGradient>
        </defs>

        <path d={area} fill="url(#eqFill)" />
        <path d={line} fill="none" stroke="url(#eqStroke)" strokeWidth="6" strokeLinecap="round" />
      </svg>
    </div>
  );
}

