# Risk-Controlled Residual Reversal V2 Development Result - 2026-07-15

## Decision

**Reject this trial family and do not open the final holdout.** The first registered primary development path breached the unchanged `20%` maximum-drawdown rule at the fixed `0.25` gross exposure, making the campaign `risk_invalid`. The registered `1,227` final returns remain reserved and unconsumed.

## Protocol

- Campaign: `residual_reversal_rank_hysteresis_risk_v2`
- Protocol commit: `0e247dab0bf53b2fcde28e4c75f3bd689e79e7eb`
- Final implementation commit: `bc80af2f34ae27b9f89695616038da2a23fc54eb`
- Merge commit: `b7852bccdf44077fe789d169d78e2e2e2a465f51`
- Pull request: `#185`
- Registration SHA-256: `b097722d1527e3f6203ca28dc50edaeec8a15d0f27a0600405a4fede5e6130cb`
- Executed command: `python3 scripts/research/run_historical_risk_controlled_reversal_campaign_v2.py --predecessor-campaign-dir /Users/diegosaa/GitHub/trader/.tmp/research/historical-reversal-campaign-v1 --source-campaign-dir /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-campaign-v1 --snapshot-dir /Users/diegosaa/GitHub/trader/.tmp/research/historical-funding-snapshot-v1 --output-dir /Users/diegosaa/GitHub/trader/.tmp/research/historical-risk-controlled-reversal-campaign-v2`
- Final-holdout flag: absent
- Development-audit digest: absent

The post-merge run used the merged implementation and only the hash-pinned `4,910`-bar development panel and settlement CSV. The raw snapshot loader is unreachable on this no-flag path, and the run did not evaluate a final-holdout return.

This was the separately registered adaptive follow-up to v1's risk rejection. Reusing the same development rows is explicitly contaminated and advances lifetime accounting from `39` to `45` trials. The only economic intervention was the ex-ante reduction from `0.50` to `0.25` gross, split `+0.125/-0.125`; no exposure sweep or post-result setting selection occurred.

## Integrity

- Rejected reversal predecessor registration SHA-256: `3e40d0ac76bdf98868a5767b3953ab590ee5c029eb4f86bd597f918c2961ecff`
- Rejected reversal predecessor manifest SHA-256: `5d3779b5a47f9d3a981262449fb80b21706d164804d04eb3b2e9b324bc4a802d`
- Rejected reversal predecessor failure SHA-256: `e096375f16921c77611a14de1de93ea4ba6ade57b887cd3a21aa1bfe50dd00ed`
- Rejected reversal predecessor summary SHA-256: `e2e430b5c352ba578b4a90500e90e11c8753fd084cd4e6715b8069ed9de4a2ed`
- Rejected adaptive v1 registration SHA-256: `9491cb3ddb94ce346900872707cf393c62339cec410e21893f20cb2318fe701d`
- Rejected adaptive v1 manifest SHA-256: `fc03ebec3a0ef236fdc4e871af6bc23b24fcdf9aca60859a6f40e49b1d00f14d`
- Rejected adaptive v1 failure SHA-256: `36caf58583dd5d715ab8da6282ca3527e1688d01bc1c1bc7426716d65c75a7bb`
- Rejected adaptive v1 ledger SHA-256: `960f4005a1ad3151a635e9c89515c6b364abaf938b8f917062c2dcc6a428cd0d`
- Rejected adaptive v1 summary SHA-256: `69cca89d2b5a601627179eca0f998b5a96392f6ea04e5eb48145118b55bb5d5f`
- Rejected adaptive v1 evidence-index SHA-256: `8311bbde8894752f74a0edb5f83a709e755ebebfaf561aec6981a69e2dba4db5`
- Funding source registration SHA-256: `cedaea5af05c880af732ceb1a78c39d4056efc2b1065157bc7a1ce31ff684d9f`
- Funding source campaign manifest SHA-256: `686ddb5ae44c4b2e461ffedd5c2b8199da4399b75d14c9075636ac062a008776`
- Source snapshot manifest SHA-256: `0e970ef24bbda0a2ceff24af5b83bc1a70de60a96ab4940a729c505e9a9c801e`
- Registered development panel SHA-256: `09a09f0e1065733be623625fa0d67e6a67dad92c53cd8fa92b6d9caa1040674a`
- Registered development settlements SHA-256: `2ccba74489f96b9ce0e58842594a524b7c306012086443cbea7305d4011ee899`
- Registered full panel SHA-256: `11d0af89e1603fad91bceffae3847dfdaed819bf56a45fe6f9d6bec45c953c0a`
- Registered full settlements SHA-256: `cc0daa4d86d8c2d64285cf12baae86dcb1d9cb5606525074cf023f77bd59c795`
- Campaign manifest file SHA-256: `36bc28a5a2e1b485e3cf9f4903f54dc8fee1d5f9a04d6cc62e868481ba354e60`
- Risk failure file SHA-256: `91ca052dbc12d072e1497b3e6c2684f92085ade09f52b8778cba09edce7a1452`
- Risk ledger file SHA-256: `899dc4303510d773f8a69554de02b17ac0b465f9de738509de3491769d2e2c86`
- Summary file SHA-256: `ba463bd7d0d68fd75bcc9770209b56aa6f3ed734565a0e92b629a76544bde48e`
- Evidence index file SHA-256: `bb127418a7c519f8686a52ee3f4003ddc0d83e97ecf7a650a65828ba3c3819e0`
- Implementation SHA-256: `364e29c4eac70d56debf97c544a83c78f68e01fee16894db01a49af09df45bab`
- Holdout identity SHA-256: `da00e904b950ed16395f33148f6572b47526eef54557722352e1242a376b6e37`

The evidence directory contains only `.campaign.lock`, `campaign-manifest.json`, `evidence-index.json`, `risk-failure.json`, `risk-ledger.json`, and `summary.json`. No development-ready receipt or `final-holdout-opened.json`, `final-holdout-result.json`, or `final-holdout-returns.csv` exists. The canonical shared registry has no marker for the registered holdout identity. A second no-flag invocation from the same merged worktree validated and returned the immutable terminal chain without rerunning analysis.

## Development Result

The registered primary-path rule requires all six control and treatment paths to complete without a risk breach. The first path failed with the following evidence:

| Measure | Result |
| --- | ---: |
| Campaign status | `risk_invalid` |
| Registered gross exposure | `0.25` |
| Registered long / short targets | `+0.125 / -0.125` |
| Path | `primary_development` |
| Trial | `resrev_24h_exit1_control` |
| Failure reason | `maximum_drawdown` |
| Evaluation stage | `outcome_endpoint` |
| Risk field | `drawdown` |
| Registered limit | `0.20` |
| Observed drawdown | `0.20484692888931988` |
| Interval left close | `1608739199999` |
| Outcome close | `1608767999999` |
| Equity before modeled liquidation | `0.8082353603571534` |
| Modeled immediate liquidation turnover | `0.22876013090015015` |
| Modeled immediate liquidation cost fraction | `0.00022876013090015015` |
| Modeled immediate liquidation cost | `0.00018489202683343243` equity |
| Every primary path risk-safe and complete | failed |
| Final holdout | `1,227` rows, reserved and unopened |
| Holdout open requested | `false` |
| Holdout blocked by | `everyPrimaryPathRiskSafeAndComplete` |

The runner stopped before evaluating the remaining five primary paths, nested rolling selection, DSR, CSCV/PBO, lifetime correction, paired hysteresis inference, turnover ratios, regimes, or cost and delay stresses. The modeled immediate liquidation is failure evidence only; it did not cure the breach, append a synthetic cash path, permit trial substitution, or authorize a ready receipt.

## Post-Run Evidence Audit

The merged validator accepts the complete terminal chain from the official worktree. A forensic reload with the identical merged implementation from a different linked worktree validated the index, artifact hashes, and current input bytes, then rejected the campaign manifest's exact semantics. The only differences were the absolute checkout prefixes in `predecessor.registration`, `adaptivePredecessor.registration`, and `registeredDevelopment.sourceRegistration`: the official manifest records `/private/tmp/trader-residual-v2-official`.

This path binding did not change any source byte, digest, trial calculation, failure evidence, or holdout state. It does mean idempotent reload is tied to the official path as well as the merged implementation. Preserve that worktree, or recreate merge commit `b7852bccdf44077fe789d169d78e2e2e2a465f51` at `/private/tmp/trader-residual-v2-official`, before relying on the runner to revalidate this output. A future campaign should use repository-relative or otherwise identity-stable registration locators without weakening exact byte and hash checks. The audit did not read the raw snapshot or final holdout.

## Interpretation

Halving gross exposure prevented the earlier v1 breach at outcome close `1606838399999`, but the same first control path still crossed the fixed drawdown boundary later, at `1608767999999`. The intervention reduced the breach observation from `21.1756%` to `20.4847%`; it did not make the campaign risk-valid.

This is a safety rejection, not a confidence result for rank hysteresis. Because the matched `24h` exit-rank `1` control breached before the treatment matrix completed, the run provides no registered comparison between exit-rank `3` hysteresis and its controls and no evidence that any trial has an edge.

Trying `0.125` gross, relaxing the drawdown limit, dropping the failed control, or selecting another horizon on these same rows would be an exposure sweep or post-outcome rescue. The family has now consumed its single planned risk intervention and all `45` counted attempts. The close-only execution limitation also remains: an endpoint breach does not quantify potentially worse intrabar liquidation, impact, or fills.

## Next Direction

Stop this residual-reversal family on the registered development history. Any new campaign should be economically distinct and should not use another exposure reduction to relabel the same signal as risk-valid.

1. Accumulate a fresh development period before making another performance claim; do not use the still-sealed final window as replacement development data.
2. Prefer a lower-turnover mechanism with independent economic motivation, such as cross-sectional carry or trend-confirmed basis normalization, rather than another residual-reversal parameter.
3. Register one small treatment set, its controls, risk budget, costs, delays, and lifetime trial count before evaluation; do not run a broad feature or threshold sweep.
4. Require chronological nested selection and all current endpoint, shock, terminal-liquidation, and dependence-conservative inference gates.
5. Keep the existing final holdout sealed. A future economically distinct campaign should use a newly designated holdout from data that was not observed during this sequence.

The immediate conclusion is to stop iterating this family and gather genuinely new evidence.
