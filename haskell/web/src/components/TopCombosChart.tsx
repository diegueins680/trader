import React from "react";
import type { IntrabarFill, Method, Normalization, Platform, Positioning } from "../lib/types";
import { fmtPct, fmtRatio } from "../lib/format";
import { PLATFORM_LABELS } from "../app/constants";

export type OptimizationComboParams = {
  binanceSymbol?: string | null;
  platform?: Platform | null;
  interval: string;
  bars: number;
  method: Method;
  blendWeight?: number | null;
  positioning?: Positioning | null;
  normalization: Normalization;
  baseOpenThreshold?: number | null;
  baseCloseThreshold?: number | null;
  fee?: number;
  epochs: number;
  hiddenSize: number;
  learningRate: number;
  valRatio: number;
  patience: number;
  gradClip?: number | null;
  slippage: number;
  spread: number;
  intrabarFill?: IntrabarFill;
  stopLoss?: number | null;
  takeProfit?: number | null;
  trailingStop?: number | null;
  stopLossVolMult?: number | null;
  takeProfitVolMult?: number | null;
  trailingStopVolMult?: number | null;
  minHoldBars?: number | null;
  cooldownBars?: number | null;
  maxHoldBars?: number | null;
  minEdge?: number | null;
  minSignalToNoise?: number | null;
  edgeBuffer?: number | null;
  costAwareEdge?: boolean | null;
  trendLookback?: number | null;
  maxDrawdown?: number | null;
  maxDailyLoss?: number | null;
  maxOrderErrors?: number | null;
  kalmanZMin?: number | null;
  kalmanZMax?: number | null;
  kalmanMarketTopN?: number | null;
  maxHighVolProb?: number | null;
  maxConformalWidth?: number | null;
  maxQuantileWidth?: number | null;
  confirmConformal?: boolean;
  confirmQuantiles?: boolean;
  confidenceSizing?: boolean;
  minPositionSize?: number | null;
  maxPositionSize?: number | null;
  volTarget?: number | null;
  volLookback?: number | null;
  volEwmaAlpha?: number | null;
  volFloor?: number | null;
  volScaleMax?: number | null;
  maxVolatility?: number | null;
  periodsPerYear?: number | null;
  walkForwardFolds?: number | null;
  tuneStressVolMult?: number | null;
  tuneStressShock?: number | null;
  tuneStressWeight?: number | null;
};

export type OptimizationComboOperation = {
  entryIndex: number;
  exitIndex: number;
  entryEquity?: number | null;
  exitEquity?: number | null;
  return?: number | null;
  holdingPeriods?: number | null;
  exitReason?: string | null;
};

export type OptimizationCombo = {
  id: number;
  rank?: number | null;
  finalEquity: number;
  objective?: string | null;
  score?: number | null;
  metrics?: {
    sharpe?: number | null;
    maxDrawdown?: number | null;
    turnover?: number | null;
    roundTrips?: number | null;
  } | null;
  openThreshold: number | null;
  closeThreshold: number | null;
  params: OptimizationComboParams;
  source: "binance" | "kraken" | "poloniex" | "csv" | null;
  operations?: OptimizationComboOperation[] | null;
};

type Props = {
  combos: OptimizationCombo[];
  loading: boolean;
  error: string | null;
  selectedId: number | null;
  onSelect: (combo: OptimizationCombo) => void;
  onApply: (combo: OptimizationCombo) => void;
};

export function TopCombosChart({ combos, loading, error, selectedId, onSelect, onApply }: Props) {
  if (loading) {
    return <div className="hint">Looking for optimizer combos…</div>;
  }
  if (combos.length === 0) {
    if (error) {
      return (
        <div className="hint" style={{ color: "rgba(239, 68, 68, 0.85)" }}>
          {error}
        </div>
      );
    }
    return (
      <div className="hint">
        No combos available yet. Run the optimizer script with <code>--top-json haskell/web/public/top-combos.json</code> (or your own path) and refresh the UI.
      </div>
    );
  }

  const maxEq = combos.reduce((acc, combo) => Math.max(acc, combo.finalEquity), 0.0) || 1.0;
  const fmtOptRatio = (v: number | null | undefined, digits = 4): string =>
    typeof v === "number" && Number.isFinite(v) ? fmtRatio(v, digits) : "—";
  const fmtOptPct = (v: number | null | undefined, digits = 2): string =>
    typeof v === "number" && Number.isFinite(v) ? fmtPct(v, digits) : "—";
  const fmtOptInt = (v: number | null | undefined): string =>
    typeof v === "number" && Number.isFinite(v) ? Math.trunc(v).toString() : "—";

  return (
    <div className="topCombosChart">
      {error ? (
        <div className="hint" style={{ color: "rgba(239, 68, 68, 0.85)" }}>
          {error} Showing last known combos.
        </div>
      ) : null}
      {combos.map((combo) => {
        const barsLabel = combo.params.bars <= 0 ? "auto" : combo.params.bars.toString();
        const platform = combo.params.platform ?? (combo.source && combo.source !== "csv" ? combo.source : null);
        const sourceLabel = platform ? PLATFORM_LABELS[platform] : combo.source === "csv" ? "CSV" : "Unknown";
        const symbolLabel = combo.params.binanceSymbol ? combo.params.binanceSymbol : null;
        const barWidth = Math.max(1, (combo.finalEquity / maxEq) * 100);
        const objectiveLabel = typeof combo.objective === "string" && combo.objective ? combo.objective : null;
        const scoreLabel =
          typeof combo.score === "number" && Number.isFinite(combo.score) ? combo.score.toFixed(4) : null;
        const sharpeLabel =
          typeof combo.metrics?.sharpe === "number" && Number.isFinite(combo.metrics.sharpe)
            ? combo.metrics.sharpe.toFixed(2)
            : null;
        const maxDdLabel =
          typeof combo.metrics?.maxDrawdown === "number" && Number.isFinite(combo.metrics.maxDrawdown)
            ? fmtPct(combo.metrics.maxDrawdown, 1)
            : null;
        const turnoverLabel =
          typeof combo.metrics?.turnover === "number" && Number.isFinite(combo.metrics.turnover)
            ? combo.metrics.turnover.toFixed(3)
            : null;
        const roundTripsLabel =
          typeof combo.metrics?.roundTrips === "number" && Number.isFinite(combo.metrics.roundTrips)
            ? Math.trunc(combo.metrics.roundTrips).toString()
            : null;
        const openLabel = combo.openThreshold != null ? fmtRatio(combo.openThreshold, 4) : "—";
        const closeLabel =
          combo.closeThreshold != null
            ? fmtRatio(combo.closeThreshold, 4)
            : openLabel !== "—"
              ? openLabel
              : "—";
        const operations = combo.operations ?? [];
        const hasOps = operations.length > 0;
        return (
          <div
            key={combo.id}
            className={`comboRow${selectedId === combo.id ? " comboRowSelected" : ""}`}
            role="button"
            tabIndex={0}
            onClick={() => onSelect(combo)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect(combo);
              }
            }}
          >
            <div className="comboRowHeader">
              <div>
                <div className="comboTitle">
                  #{combo.rank ?? combo.id} · {sourceLabel}
                  {symbolLabel ? ` · ${symbolLabel}` : ""}
                  {" · "}
                  {combo.params.interval} · bars={barsLabel}
                </div>
                <div className="comboDetail">
                  {(() => {
                    const lrValue = combo.params.learningRate;
                    const lrLabel =
                      typeof lrValue === "number" && Number.isFinite(lrValue)
                        ? lrValue.toExponential(2)
                        : "—";
                    const valValue = combo.params.valRatio;
                    const valLabel =
                      typeof valValue === "number" && Number.isFinite(valValue)
                        ? valValue.toFixed(2)
                        : "—";
                    const gradClipLabel =
                      typeof combo.params.gradClip === "number" && Number.isFinite(combo.params.gradClip)
                        ? ` · GradClip ${combo.params.gradClip.toFixed(4)}`
                        : "";
                    return (
                      <>
                        Method {combo.params.method} · Norm {combo.params.normalization} · Epochs {combo.params.epochs} · Hidden{" "}
                        {combo.params.hiddenSize} · LR {lrLabel} · Val {valLabel} · Patience {combo.params.patience}
                        {gradClipLabel}
                      </>
                    );
                  })()}
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
                <div className="comboEquity">{fmtRatio(combo.finalEquity, 4)}</div>
                <button
                  className="btnSmall"
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onApply(combo);
                  }}
                >
                  Apply
                </button>
              </div>
            </div>
            <div className="comboBarWrapper">
              <div className="comboBar" style={{ width: `${barWidth}%` }} />
            </div>
            <div className="comboDetailRow">
              {objectiveLabel ? <span className="badge">Obj {objectiveLabel}</span> : null}
              {scoreLabel ? <span className="badge">Score {scoreLabel}</span> : null}
              {sharpeLabel ? <span className="badge">Sharpe {sharpeLabel}</span> : null}
              {maxDdLabel ? <span className="badge">MaxDD {maxDdLabel}</span> : null}
              {turnoverLabel ? <span className="badge">Turn {turnoverLabel}</span> : null}
              {roundTripsLabel ? <span className="badge">RT {roundTripsLabel}</span> : null}
              {hasOps ? <span className="badge">Ops {operations.length}</span> : null}
              <span className="badge">Open {openLabel}</span>
              <span className="badge">Close {closeLabel}</span>
            </div>
            {hasOps ? (
              <div className="comboOpsTooltip" aria-hidden="true">
                <div className="comboOpsTitle">
                  Operations <span className="badge">{operations.length}</span>
                </div>
                <div className="comboOpsHeader">
                  <span>Entry</span>
                  <span>Exit</span>
                  <span>Equity</span>
                  <span>Return</span>
                  <span>Hold</span>
                  <span>Reason</span>
                </div>
                <div className="comboOpsList">
                  {operations.map((op, idx) => {
                    const entryEq = fmtOptRatio(op.entryEquity, 4);
                    const exitEq = fmtOptRatio(op.exitEquity, 4);
                    const eqLabel = entryEq !== "—" || exitEq !== "—" ? `${entryEq} → ${exitEq}` : "—";
                    const retLabel = fmtOptPct(op.return, 2);
                    const holdLabel = fmtOptInt(op.holdingPeriods);
                    const reasonLabel =
                      typeof op.exitReason === "string" && op.exitReason.trim() ? op.exitReason.trim() : "—";
                    return (
                      <div key={`${combo.id}-op-${idx}`} className="comboOpsRow">
                        <span>#{op.entryIndex}</span>
                        <span>#{op.exitIndex}</span>
                        <span>{eqLabel}</span>
                        <span>{retLabel}</span>
                        <span>{holdLabel}</span>
                        <span>{reasonLabel}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
