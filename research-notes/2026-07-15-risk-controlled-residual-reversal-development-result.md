# Risk-Controlled Residual Reversal Campaign Development Result - 2026-07-15

## Decision

**Reject this trial family and do not open the final holdout.** The first registered primary development path breached the fixed `20%` maximum-drawdown rule, making the campaign `risk_invalid`. The registered `1,227` final returns remain reserved and unconsumed.

## Protocol

- Campaign: `residual_reversal_rank_hysteresis_risk_v1`
- Final registration commit: `d97305dcbaa6da3c0047c8d5903c58ec1a6c93b7`
- Merge commit: `9772bb6a5e68259d58b39584e6178976ff2a50ed`
- Pull request: `#183`
- Registration SHA-256: `9491cb3ddb94ce346900872707cf393c62339cec410e21893f20cb2318fe701d`
- Equivalent repository-root command: `python3 scripts/research/run_historical_risk_controlled_reversal_campaign.py --predecessor-campaign-dir .tmp/research/historical-reversal-campaign-v1 --source-campaign-dir .tmp/research/historical-funding-campaign-v1 --snapshot-dir .tmp/research/historical-funding-snapshot-v1 --output-dir .tmp/research/historical-risk-controlled-reversal-campaign-v1`
- Final-holdout flag: absent

The post-merge development run used the merged implementation and only the predecessor funding campaign's hash-pinned `4,910`-bar development panel and settlement CSV. It did not open the raw snapshot or evaluate any final-holdout return.

This adaptive campaign is not a clean preregistration relative to development. The locked registration incorporates the two earlier development results and discloses that contamination. This execution is therefore an integrity-preserving test of the fixed risk and failure rules, not independent evidence for a new hypothesis.

## Integrity

- Rejected predecessor registration SHA-256: `3e40d0ac76bdf98868a5767b3953ab590ee5c029eb4f86bd597f918c2961ecff`
- Rejected predecessor manifest SHA-256: `5d3779b5a47f9d3a981262449fb80b21706d164804d04eb3b2e9b324bc4a802d`
- Rejected predecessor failure SHA-256: `e096375f16921c77611a14de1de93ea4ba6ade57b887cd3a21aa1bfe50dd00ed`
- Rejected predecessor summary SHA-256: `e2e430b5c352ba578b4a90500e90e11c8753fd084cd4e6715b8069ed9de4a2ed`
- Funding source registration SHA-256: `cedaea5af05c880af732ceb1a78c39d4056efc2b1065157bc7a1ce31ff684d9f`
- Funding source campaign manifest SHA-256: `686ddb5ae44c4b2e461ffedd5c2b8199da4399b75d14c9075636ac062a008776`
- Source snapshot manifest SHA-256: `0e970ef24bbda0a2ceff24af5b83bc1a70de60a96ab4940a729c505e9a9c801e`
- Registered development panel SHA-256: `09a09f0e1065733be623625fa0d67e6a67dad92c53cd8fa92b6d9caa1040674a`
- Registered development settlements SHA-256: `2ccba74489f96b9ce0e58842594a524b7c306012086443cbea7305d4011ee899`
- Registered full panel SHA-256: `11d0af89e1603fad91bceffae3847dfdaed819bf56a45fe6f9d6bec45c953c0a`
- Registered full settlements SHA-256: `cc0daa4d86d8c2d64285cf12baae86dcb1d9cb5606525074cf023f77bd59c795`
- Campaign manifest file SHA-256: `fc03ebec3a0ef236fdc4e871af6bc23b24fcdf9aca60859a6f40e49b1d00f14d`
- Risk failure file SHA-256: `36caf58583dd5d715ab8da6282ca3527e1688d01bc1c1bc7426716d65c75a7bb`
- Risk ledger file SHA-256: `960f4005a1ad3151a635e9c89515c6b364abaf938b8f917062c2dcc6a428cd0d`
- Summary file SHA-256: `69cca89d2b5a601627179eca0f998b5a96392f6ea04e5eb48145118b55bb5d5f`
- Evidence index file SHA-256: `8311bbde8894752f74a0edb5f83a709e755ebebfaf561aec6981a69e2dba4db5`
- Implementation SHA-256: `9ec15c25b7c883cf94a34ddca2088cfd8e9475517ae011b05dc3cf7d38a17443`
- Holdout identity SHA-256: `957dc15233747f575d8bee043754fc707e073bfca3a96addfe39b41c4e82fecd`

The evidence directory contains only `.campaign.lock`, `campaign-manifest.json`, `evidence-index.json`, `risk-failure.json`, `risk-ledger.json`, and `summary.json`. No `final-holdout-opened.json`, `final-holdout-result.json`, or `final-holdout-returns.csv` exists. A separate post-run check of the canonical checkout's holdout registry found no entry for the registered identity.

## Development Result

The registered primary-path safety rule requires all six control and treatment paths to complete without a risk breach. The first path failed with the following evidence:

| Measure | Result |
| --- | ---: |
| Campaign status | `risk_invalid` |
| Path | `primary_development` |
| Trial | `resrev_24h_exit1_control` |
| Failure reason | `maximum_drawdown` |
| Risk field | `drawdown` |
| Registered limit | `0.20` |
| Observed drawdown | `0.21175623967159896` |
| Interval left close | `1606809599999` |
| Outcome close | `1606838399999` |
| Equity before modeled liquidation | `0.814215047523834` |
| Modeled immediate liquidation turnover | `0.5039719277139856` |
| Modeled immediate liquidation cost | `0.000410341527074321` equity |
| Every primary path risk-safe and complete | failed |
| Final holdout | reserved and unopened |

The runner stopped before evaluating the remaining five primary paths, nested rolling selection, DSR, CSCV/PBO, lifetime correction, paired hysteresis inference, turnover ratios, regimes, or cost and delay stresses. The modeled immediate liquidation is failure evidence only; it did not cure the breach, append a synthetic cash path, or permit trial substitution.

## Post-Run Conformance Audit

Static review after the immutable result was written found that the merged runner passed `H._close_frame(panel)` directly into residual construction. That helper alphabetizes symbol columns, while the registration fixes the nonalphabetical `stableTieOrder` shown below and the core uses DataFrame column order for stable score ties:

- Executed order: `ADAUSDT`, `AVAXUSDT`, `BNBUSDT`, `BTCUSDT`, `DOGEUSDT`, `ETHUSDT`, `LINKUSDT`, `LTCUSDT`, `SOLUSDT`, `XRPUSDT`
- Registered order: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `AVAXUSDT`, `LINKUSDT`, `LTCUSDT`

A development-only forensic replay reordered the verified close frame before residual construction. Through the stopping point it found zero exact `24h` score-tie rows, zero target-weight differences, and the identical trial, timestamps, drawdown, equity, turnover, and liquidation-cost evidence. The maximum residual numerical difference from the changed floating-point summation order was `1.426636586643326e-14`. The immutable artifact records `risk_invalid`, and the registered-order replay independently reaches the identical risk rejection. The replay does not make the original execution bit-for-bit protocol-conforming; it establishes only that the ordering defect did not affect the stopping decision or evidence. This post-outcome replay is protocol-conformance forensics, not additional performance evidence; it did not access the snapshot or holdout, and the immutable recorded output was not overwritten or rerun.

The audit also found that the default registry directory is derived from the executing worktree's repository root. This no-flag run used a merged temporary worktree, so its default registry path was not the canonical checkout's registry. The distinction did not affect this result: the primary path failed before any holdout reservation or snapshot read, and no local or canonical marker or holdout artifact was created. A future holdout opening must use the canonical checkout or a redesigned truly cross-worktree registry before claiming shared one-shot protection.

## Interpretation

This result rejects the registered campaign on safety and completeness, not on a confidence interval for hysteresis edge. Because the matched `24h` exit-rank `1` control breached before the treatment matrix completed, the run provides no registered statistical comparison between exit-rank `3` hysteresis and its controls.

The risk layer prevented the earlier near-zero-equity failure mode from being carried into selection, but the executed `0.50`-gross control path exceeded the registered drawdown budget. Relaxing the drawdown rule, dropping the failed control, selecting another horizon, or opening the holdout would be post-outcome rescue and is prohibited. The close-only limitation also remains: an endpoint breach does not quantify potentially worse intrabar liquidation, impact, or fill risk.

## Next Direction

Any follow-up must be a new adaptive campaign rather than a modification of this one:

1. Correct and integration-test propagation of the registered symbol order in a new campaign version; do not patch and rerun this immutable campaign output.
2. Make the one-shot registry genuinely shared across worktrees or require and verify execution from one canonical checkout before any future holdout opening.
3. Count every new configuration beyond the existing `39` lifetime attempts.
4. Choose one ex-ante portfolio-risk mechanism, such as a fixed lower gross target or a fully specified volatility budget, without selecting its setting from this failure path.
5. Require the same cash entry, drifted-position accounting, terminal liquidation, endpoint risk, and adverse-shock rules; do not weaken the failed `20%` drawdown boundary after observing it.
6. Use a new development period or explicitly label reuse of this development history as contaminated; do not describe a rerun on the same rows as independent confirmation.
7. Keep the same final chronological holdout sealed until a complete newly registered development campaign passes every safety and statistical gate.

The immediate conclusion is to stop this risk-controlled hysteresis family in its registered form.
