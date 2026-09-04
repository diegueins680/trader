# Final decision memo — market-prediction review

Decision date: 2026-09-04.

## Decision

**No candidate passed. Preserve the current champion. Do not integrate a new trading predictor.**

The only production code change in this work is semantic/provenance plumbing: stable legacy sequence-predictor identifiers gain truthful versioned implementation identities and compatible aliases. This does not change a forecast, combo, saved configuration, artifact, risk threshold, deployment mode, or order path.

## Why no model was integrated

1. The strongest completed repository reproduction—historical cross-sectional residual momentum with funding—failed nested outer-OOS economics, confidence, regime, worst-fold, 2x-cost, delay, DSR, and lifetime-correction gates despite a favorable whole-development backtest.
2. The turnover-controlled reversal follow-up exhausted modeled equity. Its two risk-controlled successors breached the unchanged 20% maximum-drawdown rule at 0.50 and 0.25 gross.
3. Those development rows are permanently contaminated by adaptive choices and 45 counted attempts. Trying a smaller exposure, signal inversion, another threshold, or a deeper model would be post-outcome rescue.
4. The shared 1,227-return historical holdout remains reserved and unopened. Development did not earn the right to inspect it.
5. `cross_sectional_funding_carry_v1` is prospective attempt 46. Its protocol forbids returns, ranks, weights, PnL, risk, and performance statistics before 2027-01-20T13:00:00Z; this work did not calculate them.
6. The April–September 2026 tracked derivative cache overlaps controlled research time and is too short/sparse for an independent confirmation. It was hashed and inventoried, not backtested here.
7. OFI lacks event-level L2 history and fill replay; the calibrated shallow candidate lacks explicit missingness in the production feature schema; HAR-RV needs genuinely new prospective evidence. These are research blockers, not gates to waive.

## Candidate dispositions

| Candidate | Decision | Next lawful step |
|---|---|---|
| HAR-style volatility risk gate | Continue research | Begin new-data collection 2027-01-21; evaluate only as a direction-preserving risk/abstention challenger |
| Depth-normalized OFI | Monitor / blocked | Obtain approval and lawful point-in-time L2 data or accumulate public events; build sequence-valid fill replay before performance work |
| Missingness-aware calibrated shallow model | Continue research / blocked | Introduce feature-schema v2 with explicit availability and safe abstention, then wait for the registered future window |
| Faithful TCN/PatchTST/Transformer or time-series foundation model | Reject | Reconsider only after an independently supported net-economic mechanism and CPU/artifact case exists |
| More residual momentum/reversal tuning | Reject | Do not use the contaminated development region or sealed holdout for adaptive rescue |
| Existing prospective funding carry | Continue its existing registration | Metadata-only acquisition until the frozen evaluation time; one one-shot read afterward if acquisition gates pass |

## Compatibility and model fidelity

The current `tcn`, `patch_tst`, and `transformer` implementations are lightweight proxies, not namesake neural networks. Their historical semantics remain unchanged:

- `tcn` ↔ `dilated_lag_ridge_v1`;
- `patch_tst` ↔ `patch_summary_ridge_v1`;
- `transformer` ↔ `similarity_attention_v1`.

Legacy serialization still emits the old codes. A faithful successor must use a new explicit model identifier and compatibility version. No saved configuration acquires different behavior.

## Risk and formal-contract findings

- `FEATURE-MISSINGNESS-001` is open: an unavailable optional feature may be encoded as the same zero as an observed value. Affected learned candidates are not promotion-eligible until availability/coverage is explicit.
- `PREDICTOR-IDENTITY-001` is mitigated, not erased: aliases and diagnostics state actual behavior while legacy codes remain for compatibility.
- Formal contracts now bind future-only registrations, artifact provenance requirements, safe unavailable output, legacy semantic stability, disabled challenger isolation, no automatic promotion, and no live authorization.
- Production optimizer `deflatedSharpeProxy`/`pboProxy` remain heuristics. Formal DSR and CSCV-PBO in the research harness are the promotion standard.

## Operational lifecycle

No candidate advances beyond offline research. None enters historical replay, shadow, paper, or live execution. Before any future challenger may enter shadow, it must additionally expose independent metrics, feature and prediction drift, residual/bias, calibration decay, regime degradation, inference latency/failure rate, cost divergence, and an explicit disable/rollback control. Automated retraining may create only offline/shadow artifacts and cannot promote them.

The repository's tracked Hetzner trading profile already authorizes the pre-existing reviewed live fleet. This branch does not change that pre-existing state, does not authorize any research candidate, and placed no order. It would be inaccurate to claim that the repository as a whole was already live-disabled; the accurate safety statement is that **this research work neither enabled live trading nor sent a live order**.

## Related GitHub state

The audit considered open reliability issues [#102](https://github.com/diegueins680/trader/issues/102), [#103](https://github.com/diegueins680/trader/issues/103), and [#104](https://github.com/diegueins680/trader/issues/104), which reinforce signal/backtest/live parity, data QA, and halt-scenario requirements. [#119](https://github.com/diegueins680/trader/issues/119) proposes bias neutralization, but the named script is absent at the audited revision and its proposal is not a preregistered net-OOS result. Open pull requests #205 and #206 are dependency updates unrelated to this work.

## Acceptance outcome

- Academic basis: documented.
- Repository gap: documented.
- Fresh causal evidence: **not yet available**.
- Incremental net OOS performance: **not demonstrated**.
- Statistical, cost, delay, drawdown, and robustness gates: **not passed by any new candidate**.
- Production implementation: **none**.
- Recommendation: **No candidate passed. Continue only the three preregistered research paths and the existing prospective carry protocol.**
