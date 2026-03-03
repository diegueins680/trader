export const PLATFORM_IDS = ["binance", "coinbase", "kraken", "poloniex"] as const;
export type Platform = (typeof PLATFORM_IDS)[number];

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
] as const;

export type Method = (typeof METHOD_IDS)[number];
