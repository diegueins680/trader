# Residual Reversal Turnover Campaign Development Result - 2026-07-15

## Decision

**Reject this trial family and do not open the final holdout.** A registered primary development path exhausted modeled portfolio equity, making the campaign `mechanically_invalid`. The registered `1,227` final returns remain reserved and unconsumed.

## Protocol

- Campaign: `residual_reversal_turnover_v1`
- Registration commit: `ea97838c93a6bdc01f055053bb5c275dab8b2f08`
- Merge commit: `f0e132fb4fb33afe5f03bd263666358627fa9db2`
- Registration SHA-256: `3e40d0ac76bdf98868a5767b3953ab590ee5c029eb4f86bd597f918c2961ecff`
- Source campaign: `residual_momentum_funding_only_v1`
- Equivalent repository-root command: `python3 scripts/research/run_historical_reversal_campaign.py --source-campaign-dir .tmp/research/historical-funding-campaign-v1 --snapshot-dir .tmp/research/historical-funding-snapshot-v1 --output-dir .tmp/research/historical-reversal-campaign-v1`
- Final-holdout flag: absent

The official run used the merged implementation and only the predecessor's hash-pinned `4,910`-bar development panel and settlement CSV. It did not open the raw snapshot or evaluate any final-holdout return.

This campaign is not a clean preregistration relative to development. The registration discloses the adaptive hypothesis and prohibited pre-merge development smoke. The official result is therefore an integrity-confirming execution of the locked failure policy, not new independent development evidence.

## Integrity

- Predecessor campaign manifest SHA-256: `686ddb5ae44c4b2e461ffedd5c2b8199da4399b75d14c9075636ac062a008776`
- Predecessor registration SHA-256: `cedaea5af05c880af732ceb1a78c39d4056efc2b1065157bc7a1ce31ff684d9f`
- Source snapshot manifest SHA-256: `0e970ef24bbda0a2ceff24af5b83bc1a70de60a96ab4940a729c505e9a9c801e`
- Registered development panel SHA-256: `09a09f0e1065733be623625fa0d67e6a67dad92c53cd8fa92b6d9caa1040674a`
- Registered development settlements SHA-256: `2ccba74489f96b9ce0e58842594a524b7c306012086443cbea7305d4011ee899`
- Registered full panel SHA-256: `11d0af89e1603fad91bceffae3847dfdaed819bf56a45fe6f9d6bec45c953c0a`
- Registered full settlements SHA-256: `cc0daa4d86d8c2d64285cf12baae86dcb1d9cb5606525074cf023f77bd59c795`
- Campaign manifest file SHA-256: `5d3779b5a47f9d3a981262449fb80b21706d164804d04eb3b2e9b324bc4a802d`
- Mechanical failure file SHA-256: `e096375f16921c77611a14de1de93ea4ba6ade57b887cd3a21aa1bfe50dd00ed`
- Summary file SHA-256: `e2e430b5c352ba578b4a90500e90e11c8753fd084cd4e6715b8069ed9de4a2ed`
- Implementation SHA-256: `4066cf10bbfbb8b8818035c088c9b63558d35eb1946c8eed844b10c1dcd68333`
- Holdout identity SHA-256: `5b09051047e062332269b7ef9c47f8fc595ef555378b943c32e08d4106c6ac5c`

The evidence directory contains only `.campaign.lock`, `campaign-manifest.json`, `mechanical-failure.json`, and `summary.json`. No `final-holdout-returns.csv`, `final-holdout-result.json`, or `final-holdout-opened.json` exists. The shared holdout registry has no entry for the registered identity.

## Development Result

The locked trigger is `1 + netReturn <= 0` on any of the six primary phase-zero development paths. It fired with the following evidence:

| Measure | Result |
| --- | --- |
| Campaign status | `mechanically_invalid` |
| Trial | `resrev_24h_rebalance_3bar` |
| Failure reason | `portfolio_equity_exhausted` |
| Interval left close | `1611907199999` |
| Outcome close | `1611935999999` |
| Bankruptcy-free gate | failed |
| Final holdout | reserved and unopened |

The runner stopped before nested rolling selection, DSR, CSCV/PBO, lifetime correction, paired rebalance inference, regime analysis, or stress promotion gates. It did not clip the failed return, fabricate an absorbing post-bankruptcy series, restart the portfolio, substitute another trial, or tune the strategy.

## Interpretation

This is a mechanical rejection, not merely a weak Sharpe estimate. Under the registered drift-aware futures accounting, the primary path became insolvent before the campaign could establish a statistically testable edge. It does not resolve the registered execution-model limitation for paths that remain positive; the earlier disclosed near-zero development paths illustrate how drifted exposure and turnover can become extreme as equity approaches zero.

The phase-zero failure cannot be rescued by selecting the completed phase-one or phase-two observations disclosed by the earlier smoke. Doing so would be post-outcome trial substitution and is expressly forbidden.

## Next Direction

Any follow-up must be a new adaptive campaign rather than a modification of this one:

1. Register an execution-risk layer before evaluation, including leverage, maintenance-margin, equity-buffer, and absolute-turnover limits.
2. Treat forced risk rebalancing and liquidation costs as part of the strategy, not as post-hoc diagnostics.
3. Test turnover reduction through one fixed mechanism, such as rank hysteresis or a fixed cadence, without selecting a favorable phase after observing outcomes.
4. Pre-register a dependence-conservative confidence block and justify it against the longest overlapping signal plus sensitivity analysis for persistent portfolio state.
5. Count every new configuration in the lifetime trial family and keep the same final chronological holdout sealed until a complete development campaign passes every locked gate.

The immediate conclusion is to stop this residual-reversal family in its registered form.
