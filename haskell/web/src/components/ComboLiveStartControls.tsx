import React from "react";
import type { OptimizationCombo } from "./TopCombosChart";

export type ComboLiveStartApiState = "unknown" | "ok" | "down" | "auth";

export type ComboLiveStartControlsProps = {
  apiOk: ComboLiveStartApiState;
  topCombosLoading: boolean;
  topComboDisplay: OptimizationCombo | null;
  selectedCombo: OptimizationCombo | null;
  selectedComboStartLabel: string;
  comboStartBlocked: boolean;
  comboStartBlockedReason: string | null;
  comboStartPending: boolean;
  refreshTopCombos: () => void;
  handleComboApply: (combo: OptimizationCombo) => void;
  handleComboStart: (combo: OptimizationCombo) => void;
};

export function comboLiveStartDisabledReason(
  apiOk: ComboLiveStartApiState,
  comboStartBlocked: boolean,
  comboStartBlockedReason: string | null,
): string | null {
  if (apiOk === "auth") return comboStartBlockedReason ?? "API authentication is required.";
  if (apiOk === "down") return comboStartBlockedReason ?? "API is unavailable.";
  if (apiOk !== "ok") return comboStartBlockedReason ?? "Waiting for a fresh API status.";
  if (comboStartBlocked) return comboStartBlockedReason ?? "Live bot start is not ready.";
  return null;
}

export function ComboLiveStartControls({
  apiOk,
  topCombosLoading,
  topComboDisplay,
  selectedCombo,
  selectedComboStartLabel,
  comboStartBlocked,
  comboStartBlockedReason,
  comboStartPending,
  refreshTopCombos,
  handleComboApply,
  handleComboStart,
}: ComboLiveStartControlsProps) {
  const startDisabledReason = comboLiveStartDisabledReason(apiOk, comboStartBlocked, comboStartBlockedReason);
  const startDisabled = Boolean(startDisabledReason);

  return (
    <>
      <div className="actions" style={{ marginBottom: 8 }}>
        <button className="btnSmall" type="button" onClick={refreshTopCombos} disabled={topCombosLoading}>
          {topCombosLoading ? "Refreshing…" : "Refresh combos now"}
        </button>
        <button
          className="btnSmall"
          type="button"
          onClick={() => {
            if (topComboDisplay) handleComboApply(topComboDisplay);
          }}
          disabled={!topComboDisplay}
        >
          Apply top combo now
        </button>
        {selectedCombo ? (
          <button
            className="btnSmall btnPrimary"
            type="button"
            onClick={() => {
              // Keep the gate in the event handler as well as the disabled UI so
              // stale/programmatic events cannot cross the live-start boundary.
              if (startDisabled) return;
              handleComboStart(selectedCombo);
            }}
            disabled={startDisabled}
            title={startDisabledReason ?? undefined}
          >
            {comboStartPending ? "Starting…" : selectedComboStartLabel}
          </button>
        ) : null}
      </div>
      {selectedCombo && startDisabledReason ? (
        <div className="hint" style={{ marginBottom: 8, color: "rgba(239, 68, 68, 0.85)" }}>
          Start bot with selected combo is disabled: {startDisabledReason}
        </div>
      ) : null}
    </>
  );
}
