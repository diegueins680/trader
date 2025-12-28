import React, { useMemo } from "react";

export type BotStatePoint = {
  atMs: number;
  running: boolean;
};

type Segment = {
  startMs: number;
  endMs: number;
  running: boolean;
};

type Props = {
  points: BotStatePoint[];
  startMs: number;
  endMs: number;
  height?: number;
  label?: string;
};

function fmtTimeShort(ms: number): string {
  if (!Number.isFinite(ms)) return "--";
  try {
    return new Date(ms).toLocaleString(undefined, { month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" });
  } catch {
    return String(ms);
  }
}

function buildSegments(points: BotStatePoint[], startMs: number, endMs: number): { segments: Segment[]; hasData: boolean } {
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || endMs <= startMs) {
    return { segments: [], hasData: false };
  }

  const sorted = [...points]
    .filter((p) => Number.isFinite(p.atMs))
    .sort((a, b) => a.atMs - b.atMs);

  if (sorted.length === 0) return { segments: [], hasData: false };
  if (!sorted.some((p) => p.atMs <= endMs)) return { segments: [], hasData: false };

  let state = false;
  for (const p of sorted) {
    if (p.atMs <= startMs) state = p.running;
    else break;
  }

  const segments: Segment[] = [];
  let cursor = startMs;

  for (const p of sorted) {
    if (p.atMs < startMs) continue;
    if (p.atMs > endMs) break;
    const t = p.atMs;
    if (t > cursor) segments.push({ startMs: cursor, endMs: t, running: state });
    state = p.running;
    cursor = t;
  }

  if (cursor < endMs) segments.push({ startMs: cursor, endMs, running: state });

  return { segments, hasData: true };
}

export function BotStateChart({ points, startMs, endMs, height = 90, label = "Bot state timeline" }: Props) {
  const w = 1000;
  const h = 180;
  const pad = { l: 16, r: 16, t: 12, b: 26 };

  const { segments, hasData } = useMemo(() => buildSegments(points, startMs, endMs), [points, startMs, endMs]);
  const empty = !hasData || segments.length === 0;

  const span = Math.max(1, endMs - startMs);
  const chartWidth = w - pad.l - pad.r;
  const chartHeight = h - pad.t - pad.b;

  return (
    <div className="chart" style={{ height }} role="img" aria-label={label}>
      {empty ? (
        <div className="chartEmpty">No status data</div>
      ) : (
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="chartSvg" aria-hidden="true">
          <g>
            {segments.map((seg, idx) => {
              const x = pad.l + ((seg.startMs - startMs) / span) * chartWidth;
              const wSeg = Math.max(0, ((seg.endMs - seg.startMs) / span) * chartWidth);
              return (
                <rect
                  key={`seg-${idx}`}
                  x={x}
                  y={pad.t}
                  width={wSeg}
                  height={chartHeight}
                  className={seg.running ? "stateChartLive" : "stateChartOffline"}
                  stroke="rgba(255, 255, 255, 0.06)"
                  strokeWidth={1}
                />
              );
            })}
          </g>
          <g>
            <text x={pad.l} y={h - 8} textAnchor="start" className="stateChartAxis">
              {fmtTimeShort(startMs)}
            </text>
            <text x={w - pad.r} y={h - 8} textAnchor="end" className="stateChartAxis">
              {fmtTimeShort(endMs)}
            </text>
          </g>
        </svg>
      )}
    </div>
  );
}
