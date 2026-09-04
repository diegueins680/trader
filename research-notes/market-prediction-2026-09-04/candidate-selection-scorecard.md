# Candidate selection scorecard

Scores use the literature-review weighting and are deliberately conservative. “Evidence” means evidence for the exact repository use, not the broad academic method.

| Candidate | Academic mechanism | Credibility (20) | Independent evidence (15) | Leakage/cost discipline (20) | Crypto + repository fit (20) | Reproducibility/licensing (10) | Production feasibility (15) | Total / 100 | Disposition at 2026-09-04 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| HAR-style volatility forecast used only for risk gating and interval scaling | Andersen et al.; Corsi; GARCH literature | 18 | 13 | 14 | 16 | 9 | 15 | 85 | **Prototype prospectively; continue research.** No directional authority |
| Depth-normalized best-quote order-flow imbalance with explicit cost/latency abstention | Cont et al.; Kolm et al.; Kyle | 17 | 11 | 8 | 15 | 3 | 8 | 62 | **Monitor / data-blocked.** No L2 event archive or fill replay |
| Missingness-aware ridge/GBDT distributional forecast with split/adaptive calibration and a no-trade region | Gu et al.; quantile regression; Gibbs–Candès | 14 | 9 | 14 | 17 | 9 | 14 | 77 | **Prototype prospectively after feature schema v2.** No current evidence |
| Faithful PatchTST/Transformer/TCN | Nie et al.; Vaswani et al.; Bai et al. | 11 | 7 | 4 | 5 | 8 | 4 | 39 | **Reject now.** Architecture benchmarks do not establish net crypto value |
| Another residual momentum/reversal/basis parameter sweep | trend/pairs/carry literature | 12 | 10 | 2 | 12 | 8 | 14 | 58 | **Reject.** Existing development region is contaminated and variants failed |
| Existing prospective cross-sectional funding carry | Koijen et al.; crypto funding literature | 13 | 7 | 18 | 19 | 9 | 13 | 79 | **Continue existing registration only.** Evaluation prohibited before 2027-01-20T13:00:00Z |

## Shortlist

Only three new ideas are shortlisted:

1. `har_rv_risk_gate_v1` — a volatility/uncertainty challenger, never a directional signal by itself.
2. `depth_normalized_ofi_v1` — preregistered but blocked until causally complete, lawfully licensed L2 data and replay infrastructure exist.
3. `missingness_aware_calibrated_shallow_v1` — a small probabilistic challenger contingent on an explicit missingness schema and genuinely new data.

The previously registered `cross_sectional_funding_carry_v1` is not counted as a new candidate or modified here. Its one-shot path and trial count remain exactly as registered.

## Rejection logic

- **Sequence foundation/deep models:** rejected before experimentation. They add dependencies, memory, artifact risk, and semantic migration work without repository-specific net-performance evidence.
- **Residual momentum/reversal successors:** rejected before experimentation. Forty-five attempts are already counted in the related lifetime family; the same development data are contaminated, two risk-controlled versions breached the unchanged drawdown gate, and the historical final holdout remains sealed.
- **Alternative data sweep:** rejected. The current external-data breadth creates a large multiplicity surface, and missingness is not yet explicit enough for a clean learned-model interpretation.
- **Prediction-bias auto-correction:** not shortlisted. Issue #119 is a useful operational observation, but a rolling correction requires a frozen loss, update delay, bounds, and economic OOS protocol; its current proposal is not confirmation evidence.

## Gate common to all shortlisted candidates

A candidate can be integrated only as disabled challenger after all of the following are true: exact causal availability; nested purged walk-forward evidence; current champion and all applicable simple/model baselines on identical rows and costs; net improvement; DSR probability at least 0.95; PBO no more than 0.20 where selection occurs; positive lower confidence bound on the principal risk-adjusted metric; no unacceptable drawdown/tail regression; symbol/fold/regime robustness; baseline, 1.5x, 2x, extreme-cost, and added-delay survival; complete ablations; CPU/latency/memory compliance; deterministic provenance; and fail-closed behavior. No threshold may be relaxed after results are seen.
