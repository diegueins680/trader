import type { Method } from "../lib/types";

export type MethodUiMeta = {
  id: Method;
  optionTitle: string;
  label: string;
  configHint: string;
  tip: string;
};

export const METHOD_UI_META = [
  {
    id: "11",
    optionTitle: "Both (agreement gated)",
    label: "Both (agreement gated)",
    configHint: "only trades when both models agree on direction (up/down) outside the open threshold.",
    tip: "11 requires Kalman + LSTM agreement beyond the open threshold; fewer trades, higher confidence.",
  },
  {
    id: "blend",
    optionTitle: "Weighted average",
    label: "Blend (weighted average)",
    configHint: "uses a fixed average.",
    tip: "blend averages predictions; blend weight sets the Kalman vs LSTM mix.",
  },
  {
    id: "conf_blend",
    optionTitle: "Confidence blend",
    label: "Conf blend (confidence-weighted)",
    configHint: "uses confidence-weighted mixing.",
    tip: "conf_blend adjusts Kalman vs LSTM mix per bar using confidence (Kalman z vs LSTM edge confidence).",
  },
  {
    id: "conf_pick",
    optionTitle: "Confidence pick",
    label: "Conf pick (confidence winner)",
    configHint: "picks the higher-confidence model per bar.",
    tip: "conf_pick selects Kalman or LSTM per bar using the stronger confidence score.",
  },
  {
    id: "conformal_clip",
    optionTitle: "Conformal clip",
    label: "Conformal clip (clips blended return to conformal/quantile band)",
    configHint: "clips blended returns to the conformal interval when available.",
    tip: "conformal_clip clips the blended return into the conformal interval (or quantile band) when available.",
  },
  {
    id: "cost_pick",
    optionTitle: "Cost-aware pick",
    label: "Cost pick (net-edge winner)",
    configHint: "picks the higher post-cost edge.",
    tip: "cost_pick selects Kalman or LSTM per bar using post-cost edge (net of round-trip costs).",
  },
  {
    id: "harmonic_blend",
    optionTitle: "Harmonic-return blend",
    label: "Harmonic blend (return harmonic mean)",
    configHint: "uses a harmonic mean in return space.",
    tip: "harmonic_blend mixes Kalman/LSTM predictions using a harmonic mean in return space (more conservative on outliers).",
  },
  {
    id: "disagreement_guard",
    optionTitle: "Disagreement-aware pick",
    label: "Disagreement guard (lower-edge on conflict)",
    configHint: "picks lower-edge predictions when models conflict.",
    tip: "disagreement_guard picks higher-edge model when Kalman/LSTM agree, but lower-edge model when they disagree.",
  },
  {
    id: "median_blend",
    optionTitle: "Median-robust blend",
    label: "Median blend (robust middle return)",
    configHint: "uses a median-robust return blend.",
    tip: "median_blend uses the median of Kalman/LSTM/blend return ratios to damp outlier predictions.",
  },
  {
    id: "neutral_guard",
    optionTitle: "Neutral-on-disagreement guard",
    label: "Neutral guard (flat on conflict)",
    configHint: "goes flat on model conflict.",
    tip: "neutral_guard goes neutral when Kalman/LSTM directions conflict, and otherwise follows the more conservative edge.",
  },
  {
    id: "risk_parity_blend",
    optionTitle: "Inverse-edge risk parity blend",
    label: "Risk parity blend (inverse-edge weighted)",
    configHint: "inversely weights each model by edge magnitude.",
    tip: "risk_parity_blend down-weights the more extreme edge, blending by inverse edge magnitude to reduce forecast concentration.",
  },
  {
    id: "consensus_boost",
    optionTitle: "Consensus-strength guard",
    label: "Consensus boost (flat on conflict, strong on agree)",
    configHint: "boosts the stronger edge when models agree and goes flat when they conflict.",
    tip: "consensus_boost goes flat on model conflict, and on agreement chooses the higher-edge forecast.",
  },
  {
    id: "anchor_blend",
    optionTitle: "Conflict-anchor blend",
    label: "Anchor blend (pulls toward current price on conflict)",
    configHint: "tethers conflict bars back toward spot.",
    tip: "anchor_blend continuously pulls the blend back to current price when Kalman/LSTM conflict or disagree in edge strength.",
  },
  {
    id: "tension_gate",
    optionTitle: "Conflict-tension gate",
    label: "Tension gate (partial neutralization on conflict)",
    configHint: "keeps agreement conviction but partially neutralizes conflicts.",
    tip: "tension_gate uses stronger conviction on agreement, but partially neutralizes toward spot on directional conflict.",
  },
  {
    id: "entropy_blend",
    optionTitle: "Entropy-aware blend",
    label: "Entropy blend (uncertainty-aware spot anchoring)",
    configHint: "shrinks toward spot when model-edge uncertainty is high.",
    tip: "entropy_blend uses edge-entropy to adaptively shrink blend predictions toward spot when model uncertainty is high.",
  },
  {
    id: "coherence_gate",
    optionTitle: "Coherence gate",
    label: "Coherence gate (agreement coherence conflict guard)",
    configHint: "amplifies coherent agreement and dampens incoherent conflicts.",
    tip: "coherence_gate measures return coherence; it amplifies coherent agreement and soft-gates incoherent conflicts toward spot.",
  },
  {
    id: "divergence_gate",
    optionTitle: "Divergence gate",
    label: "Divergence gate (shrinks blended return on disagreement)",
    configHint: "shrinks blended returns toward spot as model returns diverge.",
    tip: "divergence_gate shrinks blended returns toward spot as Kalman/LSTM return divergence grows relative to the open threshold.",
  },
  {
    id: "fractal_blend",
    optionTitle: "Fractal blend",
    label: "Fractal blend (signed-root nonlinear fusion)",
    configHint: "fuses signed-root returns to suppress outlier dominance.",
    tip: "fractal_blend blends signed square-root returns, then maps back to return space to reduce outlier dominance.",
  },
  {
    id: "phase_cancel",
    optionTitle: "Phase cancel",
    label: "Phase cancel (anti-phase conflict neutralization)",
    configHint: "neutralizes anti-phase conflicts between models.",
    tip: "phase_cancel detects anti-phase Kalman/LSTM returns and compresses conflicting edges toward neutral.",
  },
  {
    id: "softmax_blend",
    optionTitle: "Softmax blend",
    label: "Softmax blend (softmax edge weights)",
    configHint: "blends with a softmax-style edge weighting.",
    tip: "softmax_blend uses a softmax-style edge weighting so the higher-edge model gets more weight (blend weight is a bias/fallback).",
  },
  {
    id: "smooth_softmax_blend",
    optionTitle: "Smooth softmax blend",
    label: "Smooth softmax blend (EMA-smooth softmax weights)",
    configHint: "smooths that softmax weighting over time.",
    tip: "smooth_softmax_blend smooths the softmax edge weights over time (EMA) to reduce twitchy per-bar switching.",
  },
  {
    id: "hedge_blend",
    optionTitle: "Hedge blend",
    label: "Hedge blend (online exp-weights mix)",
    configHint: "adapts the blend online based on realized error.",
    tip: "hedge_blend adapts the Kalman/LSTM mix online (exp-weights) based on realized prediction error.",
  },
  {
    id: "net_softmax_blend",
    optionTitle: "Net softmax blend",
    label: "Net softmax blend (post-cost softmax edge weights)",
    configHint: "uses post-cost edge in that softmax weighting.",
    tip: "net_softmax_blend is like softmax_blend but uses post-cost edge (net of round-trip costs) in the softmax weighting.",
  },
  {
    id: "edge_blend",
    optionTitle: "Edge-weighted blend",
    label: "Edge blend (edge-weighted)",
    configHint: "weights by instantaneous edge.",
    tip: "edge_blend adapts Kalman vs LSTM weights from each model's instantaneous edge magnitude.",
  },
  {
    id: "edge_pick",
    optionTitle: "Edge pick",
    label: "Edge pick (edge winner)",
    configHint: "picks the higher-edge model per bar.",
    tip: "edge_pick selects Kalman or LSTM per bar using the larger absolute edge.",
  },
  {
    id: "geo_blend",
    optionTitle: "Geometric blend",
    label: "Geo blend (geometric)",
    configHint: "blends in log-return space.",
    tip: "geo_blend mixes Kalman/LSTM returns in log-space for a multiplicative geometric blend.",
  },
  {
    id: "regime_switch",
    optionTitle: "Regime switch",
    label: "Regime switch (vol/z adaptive)",
    configHint: "toggles by volatility/z-score context.",
    tip: "regime_switch prefers LSTM in calm periods, Kalman on strong z-scores, and blend in high-volatility regimes.",
  },
  {
    id: "router",
    optionTitle: "Adaptive router",
    label: "Router (adaptive)",
    configHint: "picks the best recent model.",
    tip: "router picks the best recent model using router lookback; min score gates to HOLD.",
  },
  {
    id: "bandit_router",
    optionTitle: "Bandit router",
    label: "Bandit router (UCB adaptive)",
    configHint: "adds exploration while picking the best recent model.",
    tip: "bandit_router adds an exploration bonus so under-sampled models can still be selected when promising.",
  },
  {
    id: "10",
    optionTitle: "Kalman only",
    label: "Kalman only",
    configHint: "uses only Kalman predictions.",
    tip: "10 uses Kalman-only predictions.",
  },
  {
    id: "kalman_physics_error",
    optionTitle: "Kalman physics error",
    label: "Kalman physics error",
    configHint: "uses the Kalman state plus physics-error model.",
    tip: "kalman_physics_error uses the Kalman state plus a physics-error model over the latest 1000 bars.",
  },
  {
    id: "01",
    optionTitle: "LSTM only",
    label: "LSTM only",
    configHint: "uses only LSTM predictions.",
    tip: "01 uses LSTM-only predictions.",
  },
] as const satisfies readonly MethodUiMeta[];

type ListedMethodIds = (typeof METHOD_UI_META)[number]["id"];
type MissingMethodIds = Exclude<Method, ListedMethodIds>;
const _assertAllMethodsListed: MissingMethodIds extends never ? true : never = true;

const METHOD_UI_META_BY_ID = new Map<Method, MethodUiMeta>();
for (const meta of METHOD_UI_META) {
  if (METHOD_UI_META_BY_ID.has(meta.id)) {
    throw new Error(`duplicate method metadata entry: ${meta.id}`);
  }
  METHOD_UI_META_BY_ID.set(meta.id, meta);
}

export const METHOD_CONFIG_HINT = METHOD_UI_META.map((meta) => `"${meta.id}" ${meta.configHint}`).join(" ");

export const METHOD_TIPS = METHOD_UI_META.map((meta) => meta.tip);

export function methodLabelFromMeta(method: Method): string {
  return METHOD_UI_META_BY_ID.get(method)?.label ?? "Unknown";
}
