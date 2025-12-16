import React from "react";
import type { Method, Normalization } from "../lib/types";
import { fmtRatio } from "../lib/format";

export type OptimizationComboParams = {
  interval: string;
  bars: number;
  method: Method;
  normalization: Normalization;
  epochs: number;
  hiddenSize: number;
  learningRate: number;
  valRatio: number;
  patience: number;
  gradClip?: number | null;
  slippage: number;
  spread: number;
  stopLoss?: number | null;
  takeProfit?: number | null;
  trailingStop?: number | null;
  maxDrawdown?: number | null;
  maxDailyLoss?: number | null;
  maxOrderErrors?: number | null;
};

export type OptimizationCombo = {
  id: number;
  finalEquity: number;
  openThreshold: number | null;
  closeThreshold: number | null;
  params: OptimizationComboParams;
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
        const barWidth = Math.max(1, (combo.finalEquity / maxEq) * 100);
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
                  #{combo.id} · {combo.params.interval} · bars={barsLabel}
                </div>
                <div className="comboDetail">
                  Method {combo.params.method} · Norm {combo.params.normalization} · Epochs {combo.params.epochs} · Hidden {combo.params.hiddenSize} · LR {combo.params.learningRate.toExponential(2)} · Val {combo.params.valRatio.toFixed(2)} · Patience {combo.params.patience}
                  {combo.params.gradClip ? ` · GradClip ${combo.params.gradClip.toFixed(4)}` : ""}
                </div>
              </div>
              <div className="comboEquity">{fmtRatio(combo.finalEquity, 4)}</div>
            </div>
            <div className="comboBarWrapper">
              <div className="comboBar" style={{ width: `${barWidth}%` }} />
            </div>
            <div className="comboDetailRow">
              <span className="badge">Open {openLabel}</span>
              <span className="badge">Close {closeLabel}</span>
            </div>
          </button>
        );
      })}
    </div>
  );
}
