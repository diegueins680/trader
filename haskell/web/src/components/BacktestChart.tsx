import React, { useEffect, useMemo, useRef, useState } from "react";

type Trade = {
  entryIndex: number;
  exitIndex: number;
  entryEquity: number;
  exitEquity: number;
  return: number;
  holdingPeriods: number;
  exitReason?: string;
};

type Operation = {
  index: number;
  side: "BUY" | "SELL";
  price?: number;
};

type Props = {
  prices: number[];
  equityCurve: number[];
  positions: number[];
  agreementOk?: boolean[];
  trades: Trade[];
  operations?: Operation[];
  backtestStartIndex?: number;
  height?: number;
};

type View = { start: number; end: number };

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

function fmt(n: number, digits = 4): string {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function pct(n: number, digits = 2): string {
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(digits)}%`;
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

export function BacktestChart({
  prices,
  equityCurve,
  positions,
  agreementOk,
  trades,
  operations,
  backtestStartIndex = 0,
  height = 340,
}: Props) {
  const n = prices.length;
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: height });
  const [view, setView] = useState<View>(() => ({ start: 0, end: Math.max(1, n - 1) }));
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(null);
  const [locked, setLocked] = useState(false);

  const dragRef = useRef<{
    startX: number;
    startView: View;
    moved: boolean;
  } | null>(null);

  const pos = useMemo(() => {
    if (n === 0) return [];
    if (positions.length >= n) return positions.slice(0, n);
    const last = positions.length > 0 ? positions[positions.length - 1] : 0;
    return [...positions, ...Array.from({ length: n - positions.length }, () => last)];
  }, [n, positions]);

  const agree = useMemo(() => {
    if (!agreementOk) return null;
    if (n === 0) return [];
    if (agreementOk.length >= n) return agreementOk.slice(0, n);
    const last = agreementOk.length > 0 ? agreementOk[agreementOk.length - 1] : false;
    return [...agreementOk, ...Array.from({ length: n - agreementOk.length }, () => last)];
  }, [agreementOk, n]);

  const legend = useMemo(() => {
    return {
      showAgreement: agree !== null,
      showTrades: trades.length > 0,
      showOps: Boolean(operations && operations.length > 0),
    };
  }, [agree, operations, trades.length]);

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
      const marginL = 14;
      const marginR = 10;
      const w = Math.max(1, size.w - marginL - marginR);

      setView((prev) => {
        const start = prev.start;
        const end = prev.end;
        const len = end - start + 1;
        const t = clamp((x - marginL) / w, 0, 1);
        const pivot = clamp(Math.round(start + t * (len - 1)), 0, n - 1);

        const zoomIn = e.deltaY < 0;
        const factor = zoomIn ? 0.86 : 1.18;
        const minWin = Math.min(60, n);
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
    if (n < 2) return;

    const start = clamp(view.start, 0, n - 2);
    const end = clamp(view.end, start + 1, n - 1);
    const len = end - start + 1;

    const marginL = 14;
    const marginR = 10;
    const marginT = 10;
    const marginB = 12;

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
      const p = prices[i]!;
      const e = equityCurve[i] ?? equityCurve[equityCurve.length - 1] ?? 1;
      if (Number.isFinite(p)) {
        pMin = Math.min(pMin, p);
        pMax = Math.max(pMax, p);
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

    // Subtle grid lines (horizontal)
    const grid = (y0: number, hh: number) => {
      ctx.strokeStyle = "rgba(255,255,255,0.05)";
      ctx.lineWidth = 1;
      for (let k = 1; k <= 3; k += 1) {
        const yy = y0 + (k / 4) * hh;
        ctx.beginPath();
        ctx.moveTo(marginL, yy);
        ctx.lineTo(marginL + w, yy);
        ctx.stroke();
      }
    };
    grid(yPrice0, hPrice);
    grid(yEquity0, hEquity);

    // Trade highlight regions
    for (const t of trades) {
      if (t.exitIndex < start || t.entryIndex > end) continue;
      const a = clamp(t.entryIndex, start, end);
      const b = clamp(t.exitIndex, start, end);
      const x0 = xFor(a);
      const x1 = xFor(b);
      ctx.fillStyle = "rgba(34, 197, 94, 0.08)";
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
        ctx.fillStyle = "rgba(124, 58, 237, 0.22)";
        ctx.fillRect(x0, yPos0, Math.max(1, x1 - x0), stripeH);
      }
    }

    // Positions (bottom pane)
    for (let i = start; i < end; i += 1) {
      const p = pos[i] ?? 0;
      if (p === 0) continue;
      const x0 = xFor(i);
      const x1 = xFor(i + 1);
      ctx.fillStyle = p === 1 ? "rgba(34, 197, 94, 0.18)" : "rgba(239, 68, 68, 0.18)";
      ctx.fillRect(x0, yPos0 + Math.floor(hPos * 0.22), Math.max(1, x1 - x0), Math.floor(hPos * 0.62));
    }

    // Price line
    const priceGrad = ctx.createLinearGradient(marginL, 0, marginL + w, 0);
    priceGrad.addColorStop(0, "rgba(14, 165, 233, 0.95)");
    priceGrad.addColorStop(0.55, "rgba(124, 58, 237, 0.95)");
    priceGrad.addColorStop(1, "rgba(34, 197, 94, 0.9)");
    ctx.strokeStyle = priceGrad;
    ctx.lineWidth = 2.25;
    ctx.beginPath();
    for (let i = start; i <= end; i += 1) {
      const x = xFor(i);
      const y = yScale(prices[i]!, pr, yPrice0, hPrice);
      if (i === start) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Equity line
    const eqGrad = ctx.createLinearGradient(marginL, 0, marginL + w, 0);
    eqGrad.addColorStop(0, "rgba(124, 58, 237, 0.85)");
    eqGrad.addColorStop(1, "rgba(255, 255, 255, 0.68)");
    ctx.strokeStyle = eqGrad;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = start; i <= end; i += 1) {
      const x = xFor(i);
      const e = equityCurve[i] ?? equityCurve[equityCurve.length - 1] ?? 1;
      const y = yScale(e, er, yEquity0, hEquity);
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

    for (const t of trades) {
      if (t.exitIndex < start || t.entryIndex > end) continue;
      const ei = clamp(t.entryIndex, start, end);
      const xi = xFor(ei);
      const entryPrice = prices[ei]!;
      const yi = yScale(entryPrice, pr, yPrice0, hPrice);

      const xo = xFor(clamp(t.exitIndex, start, end));
      const exitPrice = prices[clamp(t.exitIndex, start, end)]!;
      const yo = yScale(exitPrice, pr, yPrice0, hPrice);

      ctx.strokeStyle = t.return >= 0 ? "rgba(34, 197, 94, 0.55)" : "rgba(239, 68, 68, 0.55)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(xi, yi);
      ctx.lineTo(xo, yo);
      ctx.stroke();

      tri(xi, yi, true, "rgba(34, 197, 94, 0.95)");
      tri(xo, yo, false, "rgba(239, 68, 68, 0.92)");
    }

    // Operations (buy/sell markers)
    if (operations) {
      for (const op of operations) {
        if (op.index < start || op.index > end) continue;
        const idx = clamp(op.index, start, end);
        const x = xFor(idx);
        const p = op.price ?? prices[idx]!;
        const y = yScale(p, pr, yPrice0, hPrice);
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

      const hp = yScale(prices[hoverIdx]!, pr, yPrice0, hPrice);
      const he = yScale(equityCurve[hoverIdx] ?? 1, er, yEquity0, hEquity);

      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.beginPath();
      ctx.arc(x, hp, 3.3, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, he, 3.1, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [agree, equityCurve, height, hoverIdx, n, operations, pos, prices, size.h, size.w, trades, view.end, view.start]);

  const hover = useMemo(() => {
    if (hoverIdx === null || hoverIdx < 0 || hoverIdx >= n) return null;
    const price = prices[hoverIdx]!;
    const eq = equityCurve[hoverIdx] ?? equityCurve[equityCurve.length - 1] ?? 1;
    const position = pos[hoverIdx] ?? 0;
    const ok = agree ? (agree[hoverIdx] ?? false) : null;
    const trade = findTrade(trades, hoverIdx);
    const op = findOp(operations, hoverIdx);
    const bar = backtestStartIndex + hoverIdx;
    return { idx: hoverIdx, bar, price, eq, position, ok, trade, op };
  }, [agree, backtestStartIndex, equityCurve, hoverIdx, n, operations, pos, prices, trades]);

  const tooltipStyle = useMemo(() => {
    if (!pointer) return { display: "none" } as React.CSSProperties;
    const pad = 14;
    const w = 270;
    const h = 160;
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
      const marginL = 14;
      const marginR = 10;
      const w = size.w - marginL - marginR;
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

    const marginL = 14;
    const marginR = 10;
    const w = size.w - marginL - marginR;
    const start = view.start;
    const end = view.end;
    const len = end - start + 1;
    const t = clamp((x - marginL) / Math.max(1, w), 0, 1);
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

  return (
    <div className="btChartWrap">
      <div className="btChartHeader">
        <div className="btChartTitle">Chart</div>
        <div className="btChartMeta">
          <span className="badge">Range: {view.start}–{view.end}</span>
          <span className="badge">Zoom: {Math.round(((view.end - view.start + 1) / Math.max(1, n)) * 100)}%</span>
          <span className="badge">{locked ? "Locked" : "Hover"}</span>
        </div>
        <div className="btChartActions">
          <button className="btn" type="button" onClick={resetView} disabled={empty}>
            Reset zoom
          </button>
        </div>
      </div>

      <div className="btLegend" role="list" aria-label="Chart legend">
        <div className="btLegendItem" role="listitem">
          <span className="btLegendSwatch btLegendPrice" aria-hidden="true" />
          Price
        </div>
        <div className="btLegendItem" role="listitem">
          <span className="btLegendSwatch btLegendEquity" aria-hidden="true" />
          Equity
        </div>
        <div className="btLegendItem" role="listitem">
          <span className="btLegendSwatch btLegendLong" aria-hidden="true" />
          Long
        </div>
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
              BUY / entry
            </div>
            <div className="btLegendItem" role="listitem">
              <span className="btLegendMarker btLegendSell" aria-hidden="true">
                ▼
              </span>
              SELL / exit
            </div>
          </>
        ) : null}
      </div>

      <div
        className="chart btChart"
        ref={wrapRef}
        style={{ height }}
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
                <span className="badge">{hover.position === 1 ? "LONG" : hover.position === -1 ? "SHORT" : "FLAT"}</span>{" "}
                {hover.ok !== null ? <span className="badge">{hover.ok ? "AGREE" : "NO-AGREE"}</span> : null}
              </div>
              <div className="btTooltipRow">
                <div className="k">Close</div>
                <div className="v">{fmt(hover.price, 4)}</div>
              </div>
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
                  {hover.trade.exitReason ? (
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
}
