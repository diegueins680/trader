import React from "react";
import type { IntrabarFill, Method, Normalization, Positioning } from "../lib/types";
import { fmtPct, fmtRatio } from "../lib/format";

export type OptimizationComboParams = {
  binanceSymbol?: string | null;
  interval: string;
  bars: number;
  method: Method;
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
  minHoldBars?: number | null;
  cooldownBars?: number | null;
  maxDrawdown?: number | null;
  maxDailyLoss?: number | null;
  maxOrderErrors?: number | null;
  kalmanZMin?: number | null;
  kalmanZMax?: number | null;
  maxHighVolProb?: number | null;
  maxConformalWidth?: number | null;
  maxQuantileWidth?: number | null;
  confirmConformal?: boolean;
  confirmQuantiles?: boolean;
  confidenceSizing?: boolean;
  minPositionSize?: number | null;
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
  source: "binance" | "csv" | null;
};

type Props = {
  combos: OptimizationCombo[];
  loading: boolean;
  error: string | null;
  selectedId: number | null;
  onSelect: (combo: OptimizationCombo) => void;
};

export function TopCombosChart({ combos, loading, error, selectedId, onSelect }: Props) {
  if (loading) {
    return <div className="hint">Looking for optimizer combos…</div>;
  }
  if (error) {
    return (
      <div className="hint" style={{ color: "rgba(239, 68, 68, 0.85)" }}>
        {error}
      </div>
    );
  }
  if (combos.length === 0) {
    return (
      <div className="hint">
        No combos available yet. Run the optimizer script with <code>--top-json haskell/web/public/top-combos.json</code> (or your own path) and refresh the UI.
      </div>
    );
  }

  const maxEq = combos.reduce((acc, combo) => Math.max(acc, combo.finalEquity), 0.0) || 1.0;

  return (
    <div className="topCombosChart">
      {combos.map((combo) => {
        const barsLabel = combo.params.bars <= 0 ? "auto" : combo.params.bars.toString();
        const sourceLabel = combo.source === "binance" ? "Binance" : combo.source === "csv" ? "CSV" : "Unknown";
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
        return (
          <button
            type="button"
            key={combo.id}
            className={`comboRow${selectedId === combo.id ? " comboRowSelected" : ""}`}
            onClick={() => onSelect(combo)}
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
              <div className="comboEquity">{fmtRatio(combo.finalEquity, 4)}</div>
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
              <span className="badge">Open {openLabel}</span>
              <span className="badge">Close {closeLabel}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
