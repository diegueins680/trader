import React, { useMemo } from "react";

type Series = Array<number | null | undefined>;

type Props = {
  prices: number[];
  kalmanPredNext?: Series;
  lstmPredNext?: Series;
  height?: number;
  label?: string;
};

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

function isFiniteNumber(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

function buildNextPriceError(prices: number[], predNext?: Series): Array<number | null> {
  const n = prices.length;
  const out: Array<number | null> = Array.from({ length: n }, () => null);
  if (!predNext || n < 2) return out;
  for (let i = 0; i < n - 1; i++) {
    const pred = predNext[i];
    const actual = prices[i + 1];
    if (isFiniteNumber(pred) && isFiniteNumber(actual)) out[i] = pred - actual;
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

  const { kalErr, lstmErr, min, max } = useMemo(() => {
    const kalErr = buildNextPriceError(prices, kalmanPredNext);
    const lstmErr = buildNextPriceError(prices, lstmPredNext);
    const finiteVals = [...kalErr, ...lstmErr].filter((v): v is number => isFiniteNumber(v));
    if (finiteVals.length === 0) return { kalErr, lstmErr, min: -1, max: 1 };
    const min = Math.min(0, ...finiteVals);
    const max = Math.max(0, ...finiteVals);
    if (min === max) return { kalErr, lstmErr, min: min - 1, max: max + 1 };
    return { kalErr, lstmErr, min, max };
  }, [kalmanPredNext, lstmPredNext, prices]);

  const kalPath = useMemo(() => pathFor(kalErr, w, h, padX, padY, min, max), [kalErr, max, min]);
  const lstmPath = useMemo(() => pathFor(lstmErr, w, h, padX, padY, min, max), [lstmErr, max, min]);

  const empty = prices.length < 2 || (!kalPath && !lstmPath);
  const span = max - min || 1;
  const y0 = padY + (1 - clamp((0 - min) / span, 0, 1)) * (h - padY * 2);

  return (
    <div className="chart" style={{ height }} role="img" aria-label={label}>
      {empty ? (
        <div className="chartEmpty">Not enough data</div>
      ) : (
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="chartSvg">
          <line x1={padX} x2={w - padX} y1={y0} y2={y0} stroke="rgba(255,255,255,0.16)" strokeWidth="2" />
          {kalPath ? <path d={kalPath} fill="none" stroke="rgba(14, 165, 233, 0.85)" strokeWidth="3.5" /> : null}
          {lstmPath ? <path d={lstmPath} fill="none" stroke="rgba(124, 58, 237, 0.9)" strokeWidth="3.5" /> : null}

          <g>
            <rect x={padX} y={padY - 12} width={260} height={28} rx={8} fill="rgba(0,0,0,0.35)" />
            <circle cx={padX + 12} cy={padY + 2} r={4} fill="rgba(14, 165, 233, 0.85)" />
            <text x={padX + 22} y={padY + 6} fill="rgba(255,255,255,0.78)" fontSize="12" fontFamily="monospace">
              Kalman err
            </text>
            <circle cx={padX + 112} cy={padY + 2} r={4} fill="rgba(124, 58, 237, 0.9)" />
            <text x={padX + 122} y={padY + 6} fill="rgba(255,255,255,0.78)" fontSize="12" fontFamily="monospace">
              LSTM err (pred_next - next_price)
            </text>
          </g>
        </svg>
      )}
    </div>
  );
}

