# Ablation and robustness evidence

## Audit result

No new return series was calculated. That is a methodological result, not missing work: every available period is either contaminated by the 45-trial adaptive residual/funding sequence, part of the still-sealed 1,227-return historical holdout, or inside the active prospective carry window where performance calculations are prohibited until 2027-01-20T13:00:00Z. The new candidates begin on genuinely new data at 2027-01-21T00:00:00Z.

Accordingly, this document reports only completed registered evidence and readiness gaps. Empty cells are never converted into favorable metrics.

## Completed prototype evidence

| Campaign / mechanism | Ablation or robustness evidence | Result | Decision |
|---|---|---|---|
| Residual momentum + derivatives, 15 trials | Base vs funding/basis vs OI vs taker flow vs all features at 24h/72h/168h; symbol/regime and cost/delay gates | Joint derivative coverage, OOS sample, confidence, 2x cost, added delay, DSR, and PBO gates did not all pass | Reject |
| Historical residual momentum + funding-only, 6 trials | Base vs funding-only at 24h/72h/168h; seven outer folds; regimes; symbols; 2x cost; one-bar delay; DSR/PBO; lifetime correction | Whole-development winner looked favorable, but nested OOS Sharpe was -1.296, CI [-2.674, 0.035], total return -76.69%, drawdown 81.53%; 2x cost and delay worsened it | Reject; holdout unopened |
| Residual reversal turnover, 12 counted paths | 1-bar vs 3-bar cadence plus registered phases/horizons | A required 24h/3-bar path exhausted modeled equity before statistical analysis | Mechanical reject; holdout unopened |
| Risk-controlled rank hysteresis v1, 6 trials | Exit-rank 1 controls vs exit-rank 3 treatment; 0.50 gross; risk shock and liquidation accounting | First required control breached 20% drawdown at 21.1756% | Risk reject; remaining trials unevaluated; holdout unopened |
| Risk-controlled rank hysteresis v2, 6 trials | Sole pre-registered exposure intervention to 0.25 gross; all other gates unchanged | First required control still breached at 20.4847% | Risk reject; no further sizing sweep; holdout unopened |
| Prospective funding carry, one trial | One fixed daily ranking path; no model selection | Metadata acquisition only; performance unavailable by protocol | Continue collection; no conclusion |

## Robustness interpretation

- **Cost:** the one completed statistical campaign fails more strongly at 2x cost. The reversal variants fail before cost robustness can rescue them.
- **Delay:** the historical funding candidate remains negative with one additional bar. OFI has no event replay, so latency robustness is unavailable and blocks the candidate.
- **Symbols/regimes:** the historical funding path passed minimum symbol and regime-coverage checks but failed regime-loss and worst-fold gates. Coverage is not performance.
- **Drawdown/tails:** residual reversal fails the unchanged 20% drawdown boundary even after the single registered exposure reduction. No lower exposure may be selected on those rows.
- **Missing/stale/non-finite data:** existing unit/property tests cover many fail-closed predictor paths. The model audit found that unavailable optional features can still be numerically indistinguishable from observed zero; the new shallow candidate remains blocked until schema v2.
- **Artifacts:** the existing LSTM cache lacks mission-grade provenance; no new persisted model is created. Therefore artifact corruption/compatibility tests are not falsely reported as passed for a nonexistent challenger.
- **Cross-exchange:** context adapters exist, but no equivalent-data cross-exchange economic replication is complete.
- **Adjacent timeframes/horizons:** existing residual work covers 24h/72h/168h on eight-hour bars. The newly required 1/3/6 bar tests at 1h/4h/8h await new data and remain unexecuted.

## Registered future ablations

- HAR risk gate: remove each 1/6/24-bar volatility component; remove spot context; scaling-only; abstention-only.
- OFI: remove depth normalization, quote cancellations, spot OFI, pooling, and cost-based abstention; compare depth levels 1/5/10.
- Missingness-aware shallow model: price-only, each derivatives family removed, missingness-mask removal, spot-context removal, calibration removal, abstention removal, and each model family removed.

Every future result must include bull/bear/sideways, volatility, liquidity, symbol, exchange-where-equivalent, cost, delay, missing/stale/gap, non-finite, corruption, training failure, timeout, and distribution-shift slices. A missing slice is a failed completeness gate, not an invitation to average it away.
