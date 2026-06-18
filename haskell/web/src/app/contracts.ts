export const PLATFORM_IDS = ["binance", "coinbase", "kraken", "poloniex"] as const;
export type Platform = (typeof PLATFORM_IDS)[number];

function normalizePlatformKey(raw: string): string {
  return raw.trim().toLowerCase().replace(/[^a-z0-9]/g, "");
}

export function canonicalExchangePlatform(raw: unknown): Platform | null {
  if (typeof raw !== "string") return null;
  const key = normalizePlatformKey(raw);
  if (!key) return null;
  if (key === "kraken") return "kraken";
  if (key.startsWith("binance")) return "binance";
  if (key.startsWith("coinbase")) return "coinbase";
  if (key.startsWith("poloniex")) return "poloniex";
  return null;
}

export function canonicalComboSource(raw: unknown): Platform | "csv" | null {
  const platform = canonicalExchangePlatform(raw);
  if (platform) return platform;
  if (typeof raw !== "string") return null;
  return raw.trim().toLowerCase() === "csv" ? "csv" : null;
}

export function preferredExchangePlatform(platformRaw: unknown, sourceRaw: unknown): Platform | null {
  const platform = canonicalExchangePlatform(platformRaw);
  if (platform) return platform;
  const source = canonicalComboSource(sourceRaw);
  return source && source !== "csv" ? source : null;
}

export const METHOD_IDS = [
  "11",
  "10",
  "01",
  "blend",
  "conf_blend",
  "conf_pick",
  "conformal_clip",
  "cost_pick",
  "harmonic_blend",
  "disagreement_guard",
  "median_blend",
  "neutral_guard",
  "risk_parity_blend",
  "consensus_boost",
  "anchor_blend",
  "tension_gate",
  "entropy_blend",
  "coherence_gate",
  "divergence_gate",
  "fractal_blend",
  "phase_cancel",
  "softmax_blend",
  "smooth_softmax_blend",
  "hedge_blend",
  "net_softmax_blend",
  "edge_blend",
  "edge_pick",
  "geo_blend",
  "regime_switch",
  "router",
  "bandit_router",
  "cross_sectional_momentum",
  "online_nn",
  "ta_trend",
  "ta_reversion",
  "ta_breakout",
  "ta_best",
  "ta_regime_switch",
  "kalman_physics_error",
] as const;

export type Method = (typeof METHOD_IDS)[number];
