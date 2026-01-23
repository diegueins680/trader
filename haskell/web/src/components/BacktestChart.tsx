import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { downsampleArray, downsampleIndices, downsampleOptionalArray, fmtTimeMs, remapIndexToSample } from "../app/utils";

type Trade = {
  entryIndex: number;
  exitIndex: number;
  entryEquity: number;
  exitEquity: number;
  return: number;
  holdingPeriods: number;
  exitReason?: string | null;
  open?: boolean;
};

type Operation = {
  index: number;
  side: "BUY" | "SELL";
  price?: number;
};

type Props = {
  prices: number[];
  equityCurve: number[];
  openTimes?: number[] | null;
  kalmanPredNext?: Array<number | null>;
  lstmPredNext?: Array<number | null>;
  positions: number[];
  agreementOk?: boolean[];
  trades: Trade[];
  operations?: Operation[];
  backtestStartIndex?: number;
  height?: number | string;
  actions?: React.ReactNode;
};

type View = { start: number; end: number };

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

const DEFAULT_CHART_HEIGHT = "var(--chart-height)";

const CHART_MARGIN = { l: 14, r: 78, t: 10, b: 26 };
const MIN_ZOOM_WINDOW = 6;
const MAX_BACKTEST_POINTS = 2000;

function fmt(n: number, digits = 4): string {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function pct(n: number, digits = 2): string {
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
}

function fmtRatio(n: number, digits = 4): string {
  if (!Number.isFinite(n)) return "—";
  return `${n.toFixed(digits)}x`;
}

function fmtSigned(n: number, digits = 4): string {
  if (!Number.isFinite(n)) return "—";
  const sign = n > 0 ? "+" : n < 0 ? "-" : "";
  return `${sign}${Math.abs(n).toFixed(digits)}`;
}

function fmtSignedPct(n: number, digits = 2): string {
  if (!Number.isFinite(n)) return "—";
  const sign = n > 0 ? "+" : n < 0 ? "-" : "";
  return `${sign}${pct(Math.abs(n), digits)}`;
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

function findTrade(trades: Trade[], idx: number): Trade | null {
  for (const t of trades) {
    if (idx >= t.entryIndex && idx <= t.exitIndex) return t;
  }
  return null;
}

function findOp(ops: Operation[] | undefined, idx: number): Operation | null {
  if (!ops) return null;
  for (let i = ops.length - 1; i >= 0; i -= 1) {
    const op = ops[i]!;
    if (op.index === idx) return op;
  }
  return null;
}

function posSign(p: number, eps = 1e-9): -1 | 0 | 1 {
  if (!Number.isFinite(p) || Math.abs(p) < eps) return 0;
  return p > 0 ? 1 : -1;
}

function equityAt(equityCurve: number[], idx: number): number {
  const v = equityCurve[idx];
  if (typeof v === "number" && Number.isFinite(v)) return v;
  const last = equityCurve.length > 0 ? equityCurve[equityCurve.length - 1] : undefined;
  if (typeof last === "number" && Number.isFinite(last)) return last;
  return 1;
}

function deriveOpenTrade(trades: Trade[], pos: number[], equityCurve: number[], n: number): Trade | null {
  if (n < 2) return null;

  const lastPos = pos[n - 1] ?? 0;
  const lastSign = posSign(lastPos);
  if (lastSign === 0) return null;

  // Backend may start including open trades in the future; avoid duplication.
  if (findTrade(trades, n - 1)) return null;

  let entryIndex = n - 1;
  for (let i = n - 1; i > 0; i -= 1) {
    const prev = pos[i - 1] ?? 0;
    if (posSign(prev) === lastSign) entryIndex = i - 1;
    else break;
  }

  const entryEquity = equityAt(equityCurve, entryIndex);
  const exitEquity = equityAt(equityCurve, n - 1);
  const ret = entryEquity > 0 ? exitEquity / entryEquity - 1 : 0;

  return {
    entryIndex,
    exitIndex: n - 1,
    entryEquity,
    exitEquity,
    return: ret,
    holdingPeriods: n - 1 - entryIndex,
    open: true,
    exitReason: null,
  };
}

export const BacktestChart = React.memo(function BacktestChart({
  prices,
  equityCurve,
  openTimes,
  kalmanPredNext,
  lstmPredNext,
  positions,
  agreementOk,
  trades,
  operations,
  backtestStartIndex = 0,
  height = DEFAULT_CHART_HEIGHT,
  actions,
}: Props) {
  const nFull = prices.length;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const resolvedHeight = typeof height === "string" ? height : DEFAULT_CHART_HEIGHT;
  const minHeight = typeof height === "number" ? height : undefined;
  const initialHeight = typeof height === "number" ? height : 0;
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: initialHeight });
  const [view, setView] = useState<View>(() => ({ start: 0, end: Math.max(1, nFull - 1) }));
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(null);
  const [locked, setLocked] = useState(false);

  const dragRef = useRef<{
    startX: number;
    startView: View;
    moved: boolean;
  } | null>(null);

  const posFull = useMemo<number[]>(() => {
    if (nFull === 0) return [];
    if (positions.length >= nFull) return positions.slice(0, nFull);
    const last = positions.length > 0 ? positions[positions.length - 1]! : 0;
    return [...positions, ...Array.from({ length: nFull - positions.length }, () => last)];
  }, [nFull, positions]);

  const equityFull = useMemo<number[]>(() => {
    if (nFull === 0) return [];
    if (equityCurve.length >= nFull) return equityCurve.slice(0, nFull);
    const last = equityCurve.length > 0 ? equityCurve[equityCurve.length - 1]! : 1;
    return [...equityCurve, ...Array.from({ length: nFull - equityCurve.length }, () => last)];
  }, [equityCurve, nFull]);

  const openTimesFull = useMemo<number[] | null>(() => {
    if (!openTimes) return null;
    if (nFull === 0) return [];
    if (openTimes.length >= nFull) return openTimes.slice(0, nFull);
    const last = openTimes.length > 0 ? openTimes[openTimes.length - 1]! : 0;
    return [...openTimes, ...Array.from({ length: nFull - openTimes.length }, () => last)];
  }, [nFull, openTimes]);

  const kalmanFull = useMemo<Array<number | null> | null>(() => {
    if (!kalmanPredNext) return null;
    if (nFull === 0) return [];
    const out = kalmanPredNext.slice(0, nFull).map((v) => (typeof v === "number" && Number.isFinite(v) ? v : null));
    if (out.length >= nFull) return out;
    return [...out, ...Array.from({ length: nFull - out.length }, () => null)];
  }, [kalmanPredNext, nFull]);

  const lstmFull = useMemo<Array<number | null> | null>(() => {
    if (!lstmPredNext) return null;
    if (nFull === 0) return [];
    const out = lstmPredNext.slice(0, nFull).map((v) => (typeof v === "number" && Number.isFinite(v) ? v : null));
    if (out.length >= nFull) return out;
    return [...out, ...Array.from({ length: nFull - out.length }, () => null)];
  }, [lstmPredNext, nFull]);

  const agreeFull = useMemo<boolean[] | null>(() => {
    if (!agreementOk) return null;
    if (nFull === 0) return [];
    if (agreementOk.length >= nFull) return agreementOk.slice(0, nFull);
    const last = agreementOk.length > 0 ? agreementOk[agreementOk.length - 1]! : false;
    return [...agreementOk, ...Array.from({ length: nFull - agreementOk.length }, () => last)];
  }, [agreementOk, nFull]);

  const sampled = useMemo(() => {
    const indices = downsampleIndices(nFull, MAX_BACKTEST_POINTS);
    if (indices.length === nFull) {
      return {
        indices,
        prices,
        equityCurve: equityFull,
        positions: posFull,
        openTimes: openTimesFull,
        kalman: kalmanFull,
        lstm: lstmFull,
        agree: agreeFull,
      };
    }
    return {
      indices,
      prices: downsampleArray(prices, indices),
      equityCurve: downsampleArray(equityFull, indices),
      positions: downsampleArray(posFull, indices),
      openTimes: downsampleOptionalArray(openTimesFull, indices) ?? null,
      kalman: downsampleOptionalArray(kalmanFull, indices) ?? null,
      lstm: downsampleOptionalArray(lstmFull, indices) ?? null,
      agree: downsampleOptionalArray(agreeFull, indices) ?? null,
    };
  }, [agreeFull, equityFull, kalmanFull, lstmFull, nFull, openTimesFull, posFull, prices]);

  const n = sampled.prices.length;
  const indexFor = useCallback(
    (idx: number) => {
      if (idx < 0 || idx >= sampled.indices.length) return idx;
      return sampled.indices[idx] ?? idx;
    },
    [sampled.indices],
  );
  const needsRemap = sampled.indices.length !== nFull;
  const remapIndex = useCallback(
    (idx: number) => (needsRemap ? remapIndexToSample(sampled.indices, idx) : idx),
    [needsRemap, sampled.indices],
  );

  const pos = sampled.positions;
  const kalman = sampled.kalman;
  const agree = sampled.agree;
  const pricesSample = sampled.prices;
  const equitySample = sampled.equityCurve;
  const openTimesSample = sampled.openTimes;

  const operationsSample = useMemo(() => {
    if (!operations) return undefined;
    if (!needsRemap) return operations;
    return operations
      .map((op) => ({ ...op, index: remapIndex(op.index) }))
      .filter((op) => op.index >= 0 && op.index < n);
  }, [n, needsRemap, operations, remapIndex]);

  const tradesSample = useMemo(() => {
    if (!needsRemap) return trades;
    return trades
      .map((t) => ({
        ...t,
        entryIndex: remapIndex(t.entryIndex),
        exitIndex: remapIndex(t.exitIndex),
      }))
      .filter((t) => t.entryIndex <= t.exitIndex && t.exitIndex >= 0);
  }, [needsRemap, remapIndex, trades]);

  const openTrade = useMemo(
    () => deriveOpenTrade(tradesSample, pos, equitySample, n),
    [equitySample, n, pos, tradesSample],
  );
  const tradesAll = useMemo(() => (openTrade ? [...tradesSample, openTrade] : tradesSample), [openTrade, tradesSample]);

  const legend = useMemo(() => {
    const hasShort = pos.some((p) => p < 0);
    const showKalman = Boolean(kalman && kalman.some((v) => typeof v === "number" && Number.isFinite(v)));
    return {
      showAgreement: agree !== null,
      showTrades: tradesSample.length > 0 || openTrade !== null,
      showOps: Boolean(operationsSample && operationsSample.length > 0),
      showKalman,
      hasShort,
    };
  }, [agree, kalman, openTrade, operationsSample, pos, tradesSample.length]);

  const viewSummary = useMemo(() => {
    if (n < 2) return null;
    const start = clamp(view.start, 0, n - 1);
    const end = clamp(view.end, 0, n - 1);
    if (end < start) return null;

    let priceStart: number | null = null;
    let priceEnd: number | null = null;
    let priceMin = Number.POSITIVE_INFINITY;
    let priceMax = Number.NEGATIVE_INFINITY;
    let eqStart: number | null = null;
    let eqEnd: number | null = null;
    let eqMin = Number.POSITIVE_INFINITY;
    let eqMax = Number.NEGATIVE_INFINITY;
    const eqFallback = equitySample.length > 0 ? equitySample[equitySample.length - 1]! : null;

    for (let i = start; i <= end; i += 1) {
      const p = pricesSample[i];
      if (typeof p === "number" && Number.isFinite(p)) {
        if (priceStart === null) priceStart = p;
        priceEnd = p;
        priceMin = Math.min(priceMin, p);
        priceMax = Math.max(priceMax, p);
      }

      const e = equitySample[i] ?? eqFallback;
      if (typeof e === "number" && Number.isFinite(e)) {
        if (eqStart === null) eqStart = e;
        eqEnd = e;
        eqMin = Math.min(eqMin, e);
        eqMax = Math.max(eqMax, e);
      }
    }

    if (priceStart === null || priceEnd === null) return null;
    const priceChange = priceEnd - priceStart;
    const priceChangePct = priceStart !== 0 ? priceChange / priceStart : null;
    const eqChange = eqStart !== null && eqEnd !== null ? eqEnd - eqStart : null;
    const eqChangePct = eqStart && eqChange !== null ? eqChange / eqStart : null;

    return {
      bars: end - start + 1,
      priceEnd,
      priceMin: Number.isFinite(priceMin) ? priceMin : null,
      priceMax: Number.isFinite(priceMax) ? priceMax : null,
      priceChange,
      priceChangePct,
      eqEnd,
      eqMin: Number.isFinite(eqMin) ? eqMin : null,
      eqMax: Number.isFinite(eqMax) ? eqMax : null,
      eqChange,
      eqChangePct,
    };
  }, [equitySample, n, pricesSample, view]);

  useEffect(() => {
    setView({ start: 0, end: Math.max(1, n - 1) });
    setHoverIdx(null);
    setPointer(null);
    setLocked(false);
  }, [n]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const cr = entry.contentRect;
      setSize({ w: Math.max(1, Math.floor(cr.width)), h: Math.max(1, Math.floor(cr.height)) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;

    const onWheel = (e: WheelEvent) => {
      if (n < 2) return;
      e.preventDefault();

      const rect = el.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const w = Math.max(1, size.w - CHART_MARGIN.l - CHART_MARGIN.r);

      setView((prev) => {
        const start = prev.start;
        const end = prev.end;
        const len = end - start + 1;
        const t = clamp((x - CHART_MARGIN.l) / w, 0, 1);
        const pivot = clamp(Math.round(start + t * (len - 1)), 0, n - 1);

        const zoomIn = e.deltaY < 0;
        const factor = zoomIn ? 0.86 : 1.18;
        const minWin = Math.min(MIN_ZOOM_WINDOW, n);
        const nextLen = clamp(Math.round(len * factor), minWin, n);
        const ratio = len <= 1 ? 0 : (pivot - start) / (len - 1);
        const nextStart = clamp(Math.round(pivot - ratio * (nextLen - 1)), 0, n - nextLen);
        const nextEnd = nextStart + nextLen - 1;
        return { start: nextStart, end: nextEnd };
      });
    };

    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [n, size.w]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    c.width = Math.floor(size.w * dpr);
    c.height = Math.floor(size.h * dpr);
    c.style.width = `${size.w}px`;
    c.style.height = `${size.h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, size.w, size.h);
    if (n < 2 || size.w < 2 || size.h < 2) return;

    const start = clamp(view.start, 0, n - 2);
    const end = clamp(view.end, start + 1, n - 1);
    const len = end - start + 1;

    const marginL = CHART_MARGIN.l;
    const marginR = CHART_MARGIN.r;
    const marginT = CHART_MARGIN.t;
    const marginB = CHART_MARGIN.b;

    const w = size.w - marginL - marginR;
    const h = size.h - marginT - marginB;

    const hPrice = Math.floor(h * 0.56);
    const hEquity = Math.floor(h * 0.28);
    const hPos = h - hPrice - hEquity;

    const yPrice0 = marginT;
    const yEquity0 = marginT + hPrice;
    const yPos0 = marginT + hPrice + hEquity;

    const xFor = (i: number) => marginL + ((i - start) / (len - 1)) * w;

    let pMin = Number.POSITIVE_INFINITY;
    let pMax = Number.NEGATIVE_INFINITY;
    let eMin = Number.POSITIVE_INFINITY;
    let eMax = Number.NEGATIVE_INFINITY;

    for (let i = start; i <= end; i += 1) {
      const p = pricesSample[i]!;
      const k = kalman?.[i];
      const e = equitySample[i] ?? equitySample[equitySample.length - 1] ?? 1;
      if (Number.isFinite(p)) {
        pMin = Math.min(pMin, p);
        pMax = Math.max(pMax, p);
      }
      if (typeof k === "number" && Number.isFinite(k)) {
        pMin = Math.min(pMin, k);
        pMax = Math.max(pMax, k);
      }
      if (Number.isFinite(e)) {
        eMin = Math.min(eMin, e);
        eMax = Math.max(eMax, e);
      }
    }

    const padRange = (min: number, max: number) => {
      if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 };
      const span = max - min || 1;
      const pad = span * 0.08;
      return { min: min - pad, max: max + pad };
    };

    const pr = padRange(pMin, pMax);
    const er = padRange(eMin, eMax);

    const priceTicksN = clamp(Math.round(hPrice / 54) + 1, 3, 6);
    const equityTicksN = clamp(Math.round(hEquity / 54) + 1, 3, 6);
    const prTicks = niceTicks(pr.min, pr.max, priceTicksN);
    const erTicks = niceTicks(er.min, er.max, equityTicksN);
    const prNice = { min: prTicks.min, max: prTicks.max };
    const erNice = { min: erTicks.min, max: erTicks.max };

    const yScale = (v: number, r: { min: number; max: number }, y0: number, hh: number) => {
      const t = (v - r.min) / (r.max - r.min || 1);
      return y0 + (1 - clamp(t, 0, 1)) * hh;
    };

    // Panels separators
    ctx.strokeStyle = "rgba(255,255,255,0.09)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(marginL, yEquity0);
    ctx.lineTo(marginL + w, yEquity0);
    ctx.moveTo(marginL, yPos0);
    ctx.lineTo(marginL + w, yPos0);
    ctx.stroke();

    // Subtle grid lines (horizontal) + y-axis labels
    const yGrid = (ticks: { ticks: number[]; step: number }, r: { min: number; max: number }, y0: number, hh: number, suffix = "") => {
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      ctx.fillStyle = "rgba(255,255,255,0.58)";
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";

      const d = decimalsForStep(ticks.step);
      for (const tv of ticks.ticks) {
        const y = yScale(tv, r, y0, hh);
        ctx.beginPath();
        ctx.moveTo(marginL, y);
        ctx.lineTo(marginL + w, y);
        ctx.stroke();

        const v = Math.abs(tv) < (ticks.step || 1) * 1e-9 ? 0 : tv;
        const label = `${v.toFixed(d)}${suffix}`;
        ctx.fillText(label, marginL + w + 6, y);
      }
    };
    yGrid(prTicks, prNice, yPrice0, hPrice);
    yGrid(erTicks, erNice, yEquity0, hEquity, "x");

    // Trade highlight regions
    for (const t of tradesAll) {
      if (t.exitIndex < start || t.entryIndex > end) continue;
      const a = clamp(t.entryIndex, start, end);
      const b = clamp(t.exitIndex, start, end);
      const x0 = xFor(a);
      const x1 = xFor(b);
      ctx.fillStyle = t.open ? "rgba(250, 204, 21, 0.10)" : "rgba(34, 197, 94, 0.08)";
      ctx.fillRect(x0, yPrice0, Math.max(1, x1 - x0), hPrice);
    }

    // Agreement stripe (thin)
    if (agree) {
      const stripeH = Math.max(4, Math.floor(hPos * 0.18));
      for (let i = start; i <= end; i += 1) {
        const ok = agree[i] ?? false;
        if (!ok) continue;
        const x0 = xFor(i);
        const x1 = xFor(Math.min(end, i + 1));
        ctx.fillStyle = "rgba(20, 184, 166, 0.22)";
        ctx.fillRect(x0, yPos0, Math.max(1, x1 - x0), stripeH);
      }
    }

    // Positions (bottom pane)
    for (let i = start; i < end; i += 1) {
      const p = pos[i] ?? 0;
      if (p === 0) continue;
      const x0 = xFor(i);
      const x1 = xFor(i + 1);
      const dir = p > 0 ? 1 : -1;
      const size = clamp(Math.abs(p), 0, 1);
      const paneY = yPos0 + Math.floor(hPos * 0.22);
      const paneH = Math.floor(hPos * 0.62);
      const barH = Math.max(1, Math.floor(paneH * size));
      const y = paneY + Math.floor((paneH - barH) / 2);
      const alpha = 0.08 + 0.22 * size;
      ctx.fillStyle = dir > 0 ? `rgba(34, 197, 94, ${alpha})` : `rgba(239, 68, 68, ${alpha})`;
      ctx.fillRect(x0, y, Math.max(1, x1 - x0), barH);
    }

    // Price line
    const priceGrad = ctx.createLinearGradient(marginL, 0, marginL + w, 0);
    priceGrad.addColorStop(0, "rgba(14, 165, 233, 0.95)");
    priceGrad.addColorStop(0.55, "rgba(20, 184, 166, 0.95)");
    priceGrad.addColorStop(1, "rgba(34, 197, 94, 0.9)");
    ctx.strokeStyle = priceGrad;
    ctx.lineWidth = 2.25;
    ctx.beginPath();
    for (let i = start; i <= end; i += 1) {
      const x = xFor(i);
      const y = yScale(pricesSample[i]!, prNice, yPrice0, hPrice);
      if (i === start) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Kalman prediction (overlay)
    if (legend.showKalman && kalman) {
      ctx.save();
      ctx.strokeStyle = "rgba(245, 158, 11, 0.92)";
      ctx.lineWidth = 1.8;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();

      let started = false;
      for (let i = start; i <= end; i += 1) {
        const v = kalman[i];
        if (typeof v !== "number" || !Number.isFinite(v)) {
          started = false;
          continue;
        }
        const x = xFor(i);
        const y = yScale(v, prNice, yPrice0, hPrice);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.restore();
    }

    // Equity line
    const eqGrad = ctx.createLinearGradient(marginL, 0, marginL + w, 0);
    eqGrad.addColorStop(0, "rgba(20, 184, 166, 0.85)");
    eqGrad.addColorStop(1, "rgba(255, 255, 255, 0.68)");
    ctx.strokeStyle = eqGrad;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = start; i <= end; i += 1) {
      const x = xFor(i);
      const e = equitySample[i] ?? equitySample[equitySample.length - 1] ?? 1;
      const y = yScale(e, erNice, yEquity0, hEquity);
      if (i === start) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Trade markers + connectors
    const tri = (x: number, y: number, up: boolean, color: string) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      if (up) {
        ctx.moveTo(x, y - 7);
        ctx.lineTo(x - 6, y + 6);
        ctx.lineTo(x + 6, y + 6);
      } else {
        ctx.moveTo(x, y + 7);
        ctx.lineTo(x - 6, y - 6);
        ctx.lineTo(x + 6, y - 6);
      }
      ctx.closePath();
      ctx.fill();
    };

    for (const t of tradesAll) {
      if (t.exitIndex < start || t.entryIndex > end) continue;
      const ei = clamp(t.entryIndex, start, end);
      const xi = xFor(ei);
      const entryPrice = pricesSample[ei]!;
      const yi = yScale(entryPrice, prNice, yPrice0, hPrice);

      const xo = xFor(clamp(t.exitIndex, start, end));
      const exitPrice = pricesSample[clamp(t.exitIndex, start, end)]!;
      const yo = yScale(exitPrice, prNice, yPrice0, hPrice);

      const isOpen = t.open === true;
      const dir = posSign(pos[clamp(t.entryIndex, 0, n - 1)] ?? 0);

      ctx.save();
      ctx.strokeStyle = isOpen
        ? "rgba(250, 204, 21, 0.85)"
        : t.return >= 0
          ? "rgba(34, 197, 94, 0.55)"
          : "rgba(239, 68, 68, 0.55)";
      ctx.lineWidth = isOpen ? 1.35 : 1.2;
      if (isOpen) ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(xi, yi);
      ctx.lineTo(xo, yo);
      ctx.stroke();
      ctx.restore();

      const entryUp = dir >= 0;
      const entryColor = entryUp ? "rgba(34, 197, 94, 0.95)" : "rgba(239, 68, 68, 0.92)";
      tri(xi, yi, entryUp, entryColor);

      if (isOpen) {
        ctx.fillStyle = "rgba(250, 204, 21, 0.95)";
        ctx.beginPath();
        ctx.arc(xo, yo, 3.8, 0, Math.PI * 2);
        ctx.fill();
      } else {
        const exitUp = dir < 0;
        const exitColor = exitUp ? "rgba(34, 197, 94, 0.95)" : "rgba(239, 68, 68, 0.92)";
        tri(xo, yo, exitUp, exitColor);
      }
    }

    // Operations (buy/sell markers)
    if (operationsSample) {
      for (const op of operationsSample) {
        if (op.index < start || op.index > end) continue;
        const idx = clamp(op.index, start, end);
        const x = xFor(idx);
        const p = op.price ?? pricesSample[idx]!;
        const y = yScale(p, prNice, yPrice0, hPrice);
        if (op.side === "BUY") tri(x, y, true, "rgba(34, 197, 94, 0.95)");
        if (op.side === "SELL") tri(x, y, false, "rgba(239, 68, 68, 0.92)");
      }
    }

    // Crosshair
    if (hoverIdx !== null && hoverIdx >= start && hoverIdx <= end) {
      const x = xFor(hoverIdx);
      ctx.strokeStyle = "rgba(255,255,255,0.16)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, marginT);
      ctx.lineTo(x, marginT + h);
      ctx.stroke();

      const hp = yScale(pricesSample[hoverIdx]!, prNice, yPrice0, hPrice);
      const he = yScale(equitySample[hoverIdx] ?? 1, erNice, yEquity0, hEquity);

      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.beginPath();
      ctx.arc(x, hp, 3.3, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, he, 3.1, 0, Math.PI * 2);
      ctx.fill();
    }

    // X-axis bar scale (bottom)
    ctx.strokeStyle = "rgba(255,255,255,0.14)";
    ctx.lineWidth = 1;
    ctx.fillStyle = "rgba(255,255,255,0.58)";
    ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";

    const xTicksN = clamp(Math.round(w / 170) + 1, 2, 7);
    const xTicks: number[] = [];
    for (let k = 0; k < xTicksN; k += 1) {
      const idx = Math.round(start + (k * Math.max(1, len - 1)) / Math.max(1, xTicksN - 1));
      if (xTicks[xTicks.length - 1] !== idx) xTicks.push(idx);
    }
    const yBase = marginT + h;
    for (const idx of xTicks) {
      const x = xFor(idx);
      ctx.beginPath();
      ctx.moveTo(x, yBase);
      ctx.lineTo(x, yBase + 5);
      ctx.stroke();
      ctx.fillText(String(backtestStartIndex + indexFor(idx)), x, yBase + 7);
    }
  }, [
    agree,
    backtestStartIndex,
    equitySample,
    height,
    hoverIdx,
    kalman,
    legend.showKalman,
    n,
    operationsSample,
    pos,
    pricesSample,
    size.h,
    size.w,
    tradesAll,
    view.end,
    view.start,
    indexFor,
  ]);

  const hover = useMemo(() => {
    if (hoverIdx === null || hoverIdx < 0 || hoverIdx >= n) return null;
    const price = pricesSample[hoverIdx]!;
    const eq = equitySample[hoverIdx] ?? equitySample[equitySample.length - 1] ?? 1;
    const openTime = openTimesSample?.[hoverIdx];
    const atMs = typeof openTime === "number" && Number.isFinite(openTime) ? openTime : null;
    const kalPred = legend.showKalman ? kalman?.[hoverIdx] ?? null : null;
    const position = pos[hoverIdx] ?? 0;
    const ok = agree ? (agree[hoverIdx] ?? false) : null;
    const trade = findTrade(tradesAll, hoverIdx);
    const op = findOp(operationsSample, hoverIdx);
    const bar = backtestStartIndex + indexFor(hoverIdx);
    return { idx: hoverIdx, bar, price, atMs, kalPred, eq, position, ok, trade, op };
  }, [
    agree,
    backtestStartIndex,
    equitySample,
    hoverIdx,
    indexFor,
    kalman,
    legend.showKalman,
    n,
    openTimesSample,
    operationsSample,
    pos,
    pricesSample,
    tradesAll,
  ]);

  const tooltipStyle = useMemo(() => {
    if (!pointer) return { display: "none" } as React.CSSProperties;
    const pad = 14;
    const w = 270;
    const h = 190;
    const left = clamp(pointer.x + 12, pad, size.w - w - pad);
    const top = clamp(pointer.y + 12, pad, size.h - h - pad);
    return { left, top, width: w } as React.CSSProperties;
  }, [pointer, size.h, size.w]);

  const resetView = () => {
    setView({ start: 0, end: Math.max(1, n - 1) });
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const el = wrapRef.current;
    if (!el || n < 2) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (dragRef.current) {
      const { startX, startView } = dragRef.current;
      const w = size.w - CHART_MARGIN.l - CHART_MARGIN.r;
      const len = clamp(startView.end, startView.start + 1, n - 1) - startView.start + 1;
      const dx = x - startX;
      const deltaIdx = Math.round((dx / Math.max(1, w)) * (len - 1));
      const newStart = clamp(startView.start - deltaIdx, 0, n - len);
      const newEnd = newStart + len - 1;
      setView({ start: newStart, end: newEnd });
      dragRef.current.moved = dragRef.current.moved || Math.abs(dx) > 3;
      if (!locked) {
        setPointer({ x, y });
      }
      return;
    }

    if (locked) return;
    setPointer({ x, y });

    const w = size.w - CHART_MARGIN.l - CHART_MARGIN.r;
    const start = view.start;
    const end = view.end;
    const len = end - start + 1;
    const t = clamp((x - CHART_MARGIN.l) / Math.max(1, w), 0, 1);
    const idx = clamp(Math.round(start + t * (len - 1)), 0, n - 1);
    setHoverIdx(idx);
  };

  const onPointerLeave = () => {
    if (locked) return;
    setPointer(null);
    setHoverIdx(null);
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (n < 2) return;
    (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
    const el = wrapRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const x = e.clientX - rect.left;
    dragRef.current = { startX: x, startView: view, moved: false };
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const d = dragRef.current;
    dragRef.current = null;
    if (!d) return;

    if (!d.moved) {
      // click toggles lock
      setLocked((v) => !v);
      if (!locked) {
        // keep pointer/hover in place
        const el = wrapRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        setPointer({ x, y });
      }
    }
  };

  const onDoubleClick = () => {
    setLocked(false);
    resetView();
  };

  const empty = n < 2 || equityCurve.length < 2;
  const showBuySell = legend.showTrades || legend.showOps;
  const priceChangeLabel =
    viewSummary && viewSummary.priceChange !== null
      ? viewSummary.priceChangePct !== null
        ? fmtSignedPct(viewSummary.priceChangePct, 2)
        : fmtSigned(viewSummary.priceChange, 4)
      : "—";
  const priceLabel =
    viewSummary && viewSummary.priceEnd !== null
      ? priceChangeLabel === "—"
        ? fmt(viewSummary.priceEnd, 4)
        : `${fmt(viewSummary.priceEnd, 4)} (${priceChangeLabel})`
      : "—";
  const priceRangeLabel =
    viewSummary && viewSummary.priceMin !== null && viewSummary.priceMax !== null
      ? `${fmt(viewSummary.priceMin, 4)} - ${fmt(viewSummary.priceMax, 4)}`
      : "—";
  const eqLabel =
    viewSummary && viewSummary.eqEnd !== null
      ? viewSummary.eqChangePct !== null
        ? `${fmtRatio(viewSummary.eqEnd, 4)} (${fmtSignedPct(viewSummary.eqChangePct, 2)})`
        : fmtRatio(viewSummary.eqEnd, 4)
      : "—";

  return (
    <div className="btChartWrap">
      <div className="btChartHeader">
        <div className="btChartTitle">Chart</div>
        <div className="btChartMeta">
          <span className="badge">
            Window {backtestStartIndex + indexFor(view.start)}-{backtestStartIndex + indexFor(view.end)}
          </span>
          <span className="badge">Zoom: {Math.round(((view.end - view.start + 1) / Math.max(1, n)) * 100)}%</span>
          {viewSummary ? <span className="badge">Bars {viewSummary.bars}</span> : null}
          {viewSummary ? <span className="badge">Price {priceLabel}</span> : null}
          {viewSummary ? <span className="badge">Price range {priceRangeLabel}</span> : null}
          {viewSummary && viewSummary.eqEnd !== null ? <span className="badge">Equity {eqLabel}</span> : null}
          <span className="badge">{locked ? "Locked" : "Hover"}</span>
        </div>
        <div className="btChartActions">
          <button className="btn" type="button" onClick={resetView} disabled={empty}>
            Reset zoom
          </button>
          {actions}
        </div>
      </div>

      <div className="btLegend" role="list" aria-label="Chart legend">
        <div className="btLegendItem" role="listitem">
          <span className="btLegendSwatch btLegendPrice" aria-hidden="true" />
          Price
        </div>
        {legend.showKalman ? (
          <div className="btLegendItem" role="listitem">
            <span className="btLegendSwatch btLegendKalman" aria-hidden="true" />
            Kalman
          </div>
        ) : null}
        <div className="btLegendItem" role="listitem">
          <span className="btLegendSwatch btLegendEquity" aria-hidden="true" />
          Equity
        </div>
        <div className="btLegendItem" role="listitem">
          <span className="btLegendSwatch btLegendLong" aria-hidden="true" />
          Long
        </div>
        {legend.hasShort ? (
          <div className="btLegendItem" role="listitem">
            <span className="btLegendSwatch btLegendShort" aria-hidden="true" />
            Short
          </div>
        ) : null}
        {legend.showAgreement ? (
          <div className="btLegendItem" role="listitem">
            <span className="btLegendSwatch btLegendAgree" aria-hidden="true" />
            Agree
          </div>
        ) : null}
        {legend.showTrades ? (
          <div className="btLegendItem" role="listitem">
            <span className="btLegendSwatch btLegendTrade" aria-hidden="true" />
            Trade window
          </div>
        ) : null}
        {showBuySell ? (
          <>
            <div className="btLegendItem" role="listitem">
              <span className="btLegendMarker btLegendBuy" aria-hidden="true">
                ▲
              </span>
              BUY
            </div>
            <div className="btLegendItem" role="listitem">
              <span className="btLegendMarker btLegendSell" aria-hidden="true">
                ▼
              </span>
              SELL
            </div>
          </>
        ) : null}
      </div>

      <div
        className="chart btChart"
        ref={wrapRef}
        style={{ height: resolvedHeight, minHeight }}
        onPointerMove={onPointerMove}
        onPointerLeave={onPointerLeave}
        onPointerDown={onPointerDown}
        onPointerUp={onPointerUp}
        onDoubleClick={onDoubleClick}
      >
        <canvas className="btCanvas" ref={canvasRef} />

        {empty ? <div className="chartEmpty">Not enough data</div> : null}

	        <div className="btTooltip" style={tooltipStyle} aria-hidden={!hover}>
	          {hover ? (
	            <>
              <div className="btTooltipTitle">
                Bar <span style={{ fontFamily: "var(--mono)" }}>#{hover.bar}</span>{" "}
                <span className="badge">{hover.position > 0 ? "LONG" : hover.position < 0 ? "SHORT" : "FLAT"}</span>{" "}
                {hover.position !== 0 && Math.abs(hover.position) < 0.9999 ? (
                  <span className="badge">size {pct(Math.abs(hover.position), 1)}</span>
                ) : null}{" "}
                {hover.ok !== null ? <span className="badge">{hover.ok ? "AGREE" : "NO-AGREE"}</span> : null}
              </div>
              {hover.atMs !== null ? (
                <div className="btTooltipRow">
                  <div className="k">Time</div>
                  <div className="v">{fmtTimeMs(hover.atMs)}</div>
                </div>
              ) : null}
              <div className="btTooltipRow">
                <div className="k">Close</div>
                <div className="v">{fmt(hover.price, 4)}</div>
              </div>
              {legend.showKalman ? (
                <div className="btTooltipRow">
                  <div className="k">Kalman</div>
                  <div className="v">{hover.kalPred === null ? "—" : fmt(hover.kalPred, 4)}</div>
                </div>
              ) : null}
              <div className="btTooltipRow">
                <div className="k">Equity</div>
                <div className="v">{fmt(hover.eq, 4)}x</div>
              </div>
              {hover.trade ? (
                <>
                  <div className="btTooltipRow">
                    <div className="k">Trade</div>
                    <div className="v">
                      {hover.trade.entryIndex} → {hover.trade.exitIndex} ({hover.trade.holdingPeriods}p)
                    </div>
                  </div>
                  {hover.trade.open ? (
                    <div className="btTooltipRow">
                      <div className="k">Status</div>
                      <div className="v">OPEN</div>
                    </div>
                  ) : hover.trade.exitReason ? (
                    <div className="btTooltipRow">
                      <div className="k">Exit</div>
                      <div className="v">{hover.trade.exitReason}</div>
                    </div>
                  ) : null}
                  <div className="btTooltipRow">
                    <div className="k">Trade PnL</div>
                    <div className="v">{pct(hover.trade.return, 2)}</div>
                  </div>
                </>
              ) : (
                <div className="btTooltipRow">
                  <div className="k">Trade</div>
                  <div className="v">—</div>
                </div>
              )}
              {hover.op ? (
                <div className="btTooltipRow">
                  <div className="k">Op</div>
                  <div className="v">{hover.op.side}</div>
                </div>
              ) : null}
              <div className="btTooltipHint">Wheel to zoom · Drag to pan · Click to lock · Double‑click to reset</div>
            </>
          ) : (
            <div className="btTooltipHint">Hover the chart</div>
          )}
        </div>
      </div>
    </div>
  );
});
