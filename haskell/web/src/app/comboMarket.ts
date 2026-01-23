import type { Platform } from "../lib/types";
import type { OptimizationCombo } from "../components/TopCombosChart";
import { PLATFORM_LABELS } from "./constants";

export type ComboMarketValue = Platform | "csv" | "unknown";
export type ComboMarketFilter = ComboMarketValue | "all";

export const comboMarketValue = (combo: OptimizationCombo): ComboMarketValue => {
  const platform = combo.params.platform ?? (combo.source && combo.source !== "csv" ? combo.source : null);
  if (platform) return platform;
  if (combo.source === "csv") return "csv";
  return "unknown";
};

export const comboMarketLabel = (value: ComboMarketValue): string => {
  if (value === "csv") return "CSV";
  if (value === "unknown") return "Unknown";
  return PLATFORM_LABELS[value] ?? value;
};
