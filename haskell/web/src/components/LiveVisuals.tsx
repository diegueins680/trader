import React, { useMemo } from "react";
import type { LatestSignal } from "../lib/types";
import { fmtMoney, fmtPct } from "../lib/format";
import { fmtDurationMs, fmtEtaMs } from "../app/utils";

type Props = {
  prices: number[];
  signal: LatestSignal;
  position: number | null;
  risk: { dd: number; dl: number; lastEq: number } | null;
  halted: boolean;
  cooldownLeft?: number | null;
  orderErrors?: number | null;
  candleAgeMs?: number | null;
  closeEtaMs?: number | null;
  statusAgeMs?: number | null;
};

type SparklineData = {
  path: string;
  area: string;
  trend: "up" | "down" | "flat";
  last: number;
  delta: number;
  deltaPct: number | null;
};

const SPARK_WIDTH = 240;
const SPARK_HEIGHT = 72;
const SPARK_PAD = 6;

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function buildSparkline(prices: number[]): SparklineData | null {
  const slice = prices.slice(Math.max(0, prices.length - 60)).filter((v) => Number.isFinite(v));
  if (slice.length < 2) return null;

  const last = slice[slice.length - 1]!;
  const prev = slice[slice.length - 2]!;
  const delta = last - prev;
  const deltaPct = prev !== 0 ? delta / prev : null;
  const trend: SparklineData["trend"] = delta > 0 ? "up" : delta < 0 ? "down" : "flat";

  const min = Math.min(...slice);
  const max = Math.max(...slice);
  const span = max - min || 1;
  const xFor = (i: number) => SPARK_PAD + (i * (SPARK_WIDTH - SPARK_PAD * 2)) / Math.max(1, slice.length - 1);
  const yFor = (v: number) => SPARK_PAD + (1 - clamp((v - min) / span, 0, 1)) * (SPARK_HEIGHT - SPARK_PAD * 2);

  let path = "";
  for (let i = 0; i < slice.length; i += 1) {
    const x = xFor(i);
    const y = yFor(slice[i]!);
    path += `${i === 0 ? "M" : "L"} ${x} ${y} `;
  }

  const yBase = SPARK_HEIGHT - SPARK_PAD;
  const area = `${path}L ${xFor(slice.length - 1)} ${yBase} L ${xFor(0)} ${yBase} Z`;

  return { path: path.trim(), area: area.trim(), trend, last, delta, deltaPct };
}

function signalDirectionLabel(dir: LatestSignal["chosenDirection"]): string {
  if (dir === "UP") return "LONG";
  if (dir === "DOWN") return "SHORT";
  return "FLAT";
}

function signalDirectionValue(dir: LatestSignal["chosenDirection"]): number {
  if (dir === "UP") return 1;
  if (dir === "DOWN") return -1;
  return 0;
}

function actionTone(action: string): "long" | "short" | "flat" {
  const head = action.trim().split(/\s+/)[0]?.toUpperCase() ?? "";
  if (head.includes("BUY") || head.includes("LONG")) return "long";
  if (head.includes("SELL") || head.includes("SHORT")) return "short";
  return "flat";
}

function fmtSignedMoney(value: number, digits = 4): string {
  if (!Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${fmtMoney(Math.abs(value), digits)}`;
}

function fmtSignedPct(value: number | null, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${fmtPct(Math.abs(value), digits)}`;
}

export function LiveVisuals({
  prices,
  signal,
  position,
  risk,
  halted,
  cooldownLeft,
  orderErrors,
  candleAgeMs,
  closeEtaMs,
  statusAgeMs,
}: Props) {
  const spark = useMemo(() => buildSparkline(prices), [prices]);

  const currentPrice = signal.currentPrice;
  const priceDelta = spark?.delta ?? null;
  const priceDeltaPct = spark?.deltaPct ?? null;
  const sparkTrend = spark?.trend ?? "flat";

  const confidence = isFiniteNumber(signal.confidence) ? clamp(signal.confidence, 0, 1) : null;
  const positionSize = isFiniteNumber(signal.positionSize) ? clamp(signal.positionSize, 0, 1) : null;

  const signalDirection = signal.chosenDirection;
  const signalValue = signalDirectionValue(signalDirection);
  const signalPct = ((signalValue + 1) / 2) * 100;

  const posValue = isFiniteNumber(position) ? clamp(position, -1, 1) : 0;
  const posPct = ((posValue + 1) / 2) * 100;
  const posLabel = posValue > 0.001 ? "LONG" : posValue < -0.001 ? "SHORT" : "FLAT";
  const posMagnitude = Math.abs(posValue) > 0.001 ? fmtPct(Math.abs(posValue), 1) : "0%";

  const candleProgress = useMemo(() => {
    if (!isFiniteNumber(candleAgeMs) || !isFiniteNumber(closeEtaMs)) return null;
    if (closeEtaMs <= 0) return 1;
    const total = candleAgeMs + closeEtaMs;
    if (!Number.isFinite(total) || total <= 0) return null;
    return clamp(candleAgeMs / total, 0, 1);
  }, [candleAgeMs, closeEtaMs]);

  const actionText = signal.action ?? "HOLD";
  const actionBadgeClass = actionTone(actionText);
  const directionBadgeClass = signalDirection === "UP" ? "long" : signalDirection === "DOWN" ? "short" : "flat";

  const riskBadge = halted ? "halted" : typeof cooldownLeft === "number" && cooldownLeft > 0 ? "cooldown" : "active";

  return (
    <div className="liveVizGrid" aria-label="Live trading visual aids" role="group">
      <div className="liveVizCard">
        <div className="liveVizHeader">
          <div>
            <div className="liveVizLabel">Market pulse</div>
            <div className="liveVizValue">
              {fmtMoney(currentPrice, 4)}
              <span className={`liveVizDelta liveVizDelta${sparkTrend === "up" ? "Up" : sparkTrend === "down" ? "Down" : "Flat"}`}>
                {priceDelta != null ? `${fmtSignedMoney(priceDelta, 4)} (${fmtSignedPct(priceDeltaPct, 2)})` : "—"}
              </span>
            </div>
          </div>
          <span className={`badge liveVizBadge liveVizBadge${sparkTrend === "down" ? "Short" : sparkTrend === "up" ? "Long" : "Flat"}`}>
            {sparkTrend === "up" ? "rising" : sparkTrend === "down" ? "falling" : "steady"}
          </span>
        </div>
        <div className="liveVizSparklineWrap">
          {spark ? (
            <svg
              className={`liveSparkline liveSparkline${sparkTrend === "up" ? "Up" : sparkTrend === "down" ? "Down" : "Flat"}`}
              viewBox={`0 0 ${SPARK_WIDTH} ${SPARK_HEIGHT}`}
              preserveAspectRatio="none"
              role="img"
              aria-label="Recent price sparkline"
            >
              <path className="liveSparklineArea" d={spark.area} />
              <path className="liveSparklineLine" d={spark.path} />
            </svg>
          ) : (
            <div className="liveVizEmpty">Waiting for price data</div>
          )}
        </div>
        <div className="liveVizMeta">
          <span className="badge">Status {fmtDurationMs(statusAgeMs)}</span>
          <span className="badge">Candle {fmtEtaMs(closeEtaMs)}</span>
        </div>
        <div className={`liveMeter liveMeterThin ${candleProgress == null ? "liveMeterMuted" : ""}`} aria-hidden="true">
          <div className="liveMeterFill" style={{ width: `${Math.round((candleProgress ?? 0) * 100)}%` }} />
        </div>
      </div>

      <div className="liveVizCard">
        <div className="liveVizHeader">
          <div className="liveVizLabel">Signal compass</div>
          <span className={`badge liveVizBadge liveVizBadge${actionBadgeClass === "long" ? "Long" : actionBadgeClass === "short" ? "Short" : "Flat"}`}>
            {actionText}
          </span>
        </div>
        <div className="liveVizBlock">
          <div className="liveVizRowHeader">
            <span>Direction</span>
            <span className={`liveVizValueTag liveVizValueTag${directionBadgeClass === "long" ? "Long" : directionBadgeClass === "short" ? "Short" : "Flat"}`}>
              {signalDirectionLabel(signalDirection)}
            </span>
          </div>
          <div className="liveLaneLabels">
            <span>Short</span>
            <span>Flat</span>
            <span>Long</span>
          </div>
          <div className="liveLaneTrack">
            <span className="liveLaneCenter" />
            <span
              className={`liveLaneMarker liveLaneMarker${directionBadgeClass === "long" ? "Long" : directionBadgeClass === "short" ? "Short" : "Flat"}`}
              style={{ left: `${signalPct}%` }}
            />
          </div>
        </div>
        <div className="liveVizBlock">
          <div className="liveVizRowHeader">
            <span>Confidence</span>
            <span className="liveVizRowValue">
              {confidence != null ? fmtPct(confidence, 1) : "—"}
              {positionSize != null ? ` • size ${fmtPct(positionSize, 1)}` : ""}
            </span>
          </div>
          <div className={`liveMeter ${confidence == null ? "liveMeterMuted" : ""}`}>
            <div className="liveMeterFill" style={{ width: `${Math.round((confidence ?? 0) * 100)}%` }} />
          </div>
          {positionSize != null ? (
            <div className="liveMeter liveMeterThin liveMeterAlt" aria-hidden="true">
              <div className="liveMeterFill" style={{ width: `${Math.round(positionSize * 100)}%` }} />
            </div>
          ) : null}
        </div>
        <div className="liveVizBlock">
          <div className="liveVizRowHeader">
            <span>Position</span>
            <span className="liveVizRowValue">{`${posLabel} ${posMagnitude}`}</span>
          </div>
          <div className="liveLaneTrack">
            <span className="liveLaneCenter" />
            <span
              className={`liveLaneMarker liveLaneMarker${posLabel === "LONG" ? "Long" : posLabel === "SHORT" ? "Short" : "Flat"}`}
              style={{ left: `${posPct}%` }}
            />
          </div>
        </div>
      </div>

      <div className="liveVizCard">
        <div className="liveVizHeader">
          <div className="liveVizLabel">Risk buffer</div>
          <span className={`badge liveVizBadge liveVizBadge${riskBadge === "halted" ? "Short" : riskBadge === "cooldown" ? "Warn" : "Ok"}`}>
            {riskBadge === "halted" ? "HALTED" : riskBadge === "cooldown" ? "COOLDOWN" : "ACTIVE"}
          </span>
        </div>
        {risk ? (
          <>
            <div className="liveVizBlock">
              <div className="liveVizRowHeader">
                <span>Drawdown</span>
                <span className="liveVizRowValue">{fmtPct(risk.dd, 2)}</span>
              </div>
              <div className="liveMeter liveMeterWarn">
                <div className="liveMeterFill" style={{ width: `${Math.round(clamp(risk.dd, 0, 1) * 100)}%` }} />
              </div>
            </div>
            <div className="liveVizBlock">
              <div className="liveVizRowHeader">
                <span>Daily loss</span>
                <span className="liveVizRowValue">{fmtPct(risk.dl, 2)}</span>
              </div>
              <div className="liveMeter liveMeterWarn">
                <div className="liveMeterFill" style={{ width: `${Math.round(clamp(risk.dl, 0, 1) * 100)}%` }} />
              </div>
            </div>
          </>
        ) : (
          <div className="liveVizEmpty">No risk snapshot yet</div>
        )}
        <div className="liveVizMeta">
          <span className="badge">Order errors {orderErrors ?? "—"}</span>
          <span className="badge">
            Cooldown {typeof cooldownLeft === "number" && cooldownLeft > 0 ? `${Math.max(0, Math.trunc(cooldownLeft))} bar(s)` : "—"}
          </span>
        </div>
      </div>
    </div>
  );
}
