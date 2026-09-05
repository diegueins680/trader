# Trader Firm — Risk Register

`formal/risk-register.json` is the canonical machine-readable source for risk
IDs, severities, and lifecycle statuses. This table and the typed Haskell
projection in `app/Trader/Formal/RiskRegister.hs` must contain exactly the same
entries in canonical ID order; automation rejects drift and duplicate IDs.

Severity describes impact (`LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`). Status
describes lifecycle (`OPEN`, `MITIGATED`, or `CLOSED`). A mitigation does not
change severity and a fixed risk is `CLOSED`, never encoded as a severity.

| ID | Risk | Severity | Owner | Status | Next Action |
|---|---|---|---|---|---|
| AUTOLOOP-DOWN-003 | Autoloop process was not alive at the last recorded operational review | CRITICAL | trader-firm-cto | OPEN | Restart or diagnose the supervisor and replace stale evidence with a fresh health witness |
| AUTOLOOP-RESET-2026-05-30 | Autoloop cycle counter reset and broke continuity assumptions | CRITICAL | trader-firm-cto | OPEN | Establish and verify durable monotone cycle identity across supervisor restarts |
| AUTOLOOP-SINGLETON-001 | Multiple autoloop instances may race on the same repository | HIGH | trader-firm-cto | OPEN | Enforce one process with a verified lock and stale-owner recovery |
| AUTOLOOP-STALL-001 | Autoloop stall detection depended on manual observation | CRITICAL | trader-firm-cto | CLOSED | Heartbeat telemetry and a bounded stale-heartbeat alert are implemented |
| BINARY-HANG-001 | The trader binary could hang while draining after a termination signal | MEDIUM | trader-firm-cto | MITIGATED | Close after a serve-mode PostgreSQL subprocess termination witness |
| CIO-DEAFNESS-001 | The CIO reporting lane missed recorded deadlines | CRITICAL | trader-firm-ceo | OPEN | Obtain a current owner report and explicitly close or reassign the obligation |
| EXECUTION-DATASET-001 | Backtest dataset generation is not fully reproducible | MEDIUM | trader-firm-data | OPEN | Seed randomness and record the source dataset hash in test output |
| EXECUTION-MISSING-001 | The execution reporting lane missed recorded trade-log deadlines | CRITICAL | trader-firm-execution | OPEN | Obtain a current execution report and explicitly close or reassign the obligation |
| EXPECTANCY-INVALID-001 | Missing or non-finite expectancy could bypass a configured minimum | CRITICAL | trader-firm-risk | CLOSED | `specRiskHalt` rejects malformed expectancy and bounded verification covers it |
| FEATURE-MISSINGNESS-001 | Optional predictor features can encode unavailable evidence as the same numeric zero as an observed value | HIGH | trader-firm-research | OPEN | Preserve the new derivatives first-seen ledgers, define equivalent policies for remaining sources, and migrate production feature builders and artifacts under explicit compatibility versioning before promotion eligibility |
| GITHUB-502-001 | Transient GitHub API failures can interrupt automation | MEDIUM | trader-firm-cto | OPEN | Replace stale outage evidence with current bounded-retry operational validation |
| KALMAN-NUMSTAB-001 | Kalman numerical instability could produce zero trades or hangs | MEDIUM | trader-firm-cto | OPEN | Obtain current operational validation of the implemented numerical guards |
| LEVERAGE-INVALID-001 | Malformed leverage configuration could bypass position-size protection | CRITICAL | trader-firm-risk | CLOSED | `specRiskHalt` rejects malformed leverage and live futures leverage is capped |
| LEVERAGE-SANITY-001 | Corrupted or absurd venue leverage evidence could bypass size limits | CRITICAL | trader-firm-risk | CLOSED | Venue evidence is capped and configuration validation rejects malformed values |
| LOSS-STREAK-LIMIT-INVALID-001 | A negative loss-streak limit could silently disable protection | CRITICAL | trader-firm-risk | CLOSED | Negative limits fail closed while zero remains the documented disabled boundary |
| MARKET-DATA-TIMESTAMP-OVERFLOW-001 | Timestamp overflow could make stale or discontinuous evidence appear valid | CRITICAL | trader-firm-data | CLOSED | Checked arithmetic fails closed across market-data time validation |
| MAX-POSITION-GUARDRAIL-001 | Malformed maximum-position configuration could silently disable every trade | HIGH | trader-firm-risk | CLOSED | Checked simulation rejects non-positive or non-finite configuration |
| PREDICTOR-IDENTITY-001 | Legacy TCN, PatchTST, and Transformer identifiers can overstate the fidelity of lightweight proxy implementations | HIGH | trader-firm-research | MITIGATED | Preserve legacy semantics, expose accurate versioned implementation identities, and require a new model ID for any faithful architecture |
| RISK-LIMIT-001 | Daily, weekly, and drawdown limits were not enforced in the live loop | HIGH | trader-firm-risk | CLOSED | Runtime invariant checks and guardrail regressions are implemented |
| RISK-LIMIT-NON-FINITE-001 | Non-finite risk limits could silently disable halt checks | CRITICAL | trader-firm-risk | CLOSED | `specRiskHalt` rejects non-finite limits and bounded verification covers it |
| RISK-METRIC-INVALID-001 | Malformed loss or drawdown evidence could bypass live halt checks | CRITICAL | trader-firm-risk | CLOSED | `specRiskHalt` validates risk evidence before threshold comparisons |
| SCHEMA-001 | Live trade-log schema could drift from its declared contract | CRITICAL | trader-firm-cio | CLOSED | The schema contract and executable validation are tracked |
| THRESHOLD-FACTOR-001 | `thresholdFactor` may not be wired into simulation configuration | MEDIUM | trader-firm-research | OPEN | Confirm the research contract and add a simulation integration witness |
| TRADE-LOG-GAP-001 | Trade-log records lacked required exit and halt evidence | HIGH | trader-firm-cio | CLOSED | The schema includes `exit_reason` and the trade-log implementation is tracked |
| TRADE-LOG-GAP-002 | Trade logs lack a native snapshot of derived risk-state metrics | MEDIUM | trader-firm-cio | OPEN | Define whether schema or deterministic derivation owns the snapshot |
| TRAILING-STOP-001 | A trailing-stop exit may re-enter on the same bar | MEDIUM | trader-firm-execution | OPEN | Add and verify a bar-level re-entry lock after trailing-stop exits |
| VOL-TARGET-001 | A stateful volatility-target regression was reported in the Haskell tests | CRITICAL | trader-firm-cto | OPEN | Reproduce against the current canonical Haskell wrapper and close or update the stale report |
| VOL-TARGET-INVALID-001 | Malformed volatility-target configuration could bypass scaling limits | CRITICAL | trader-firm-risk | CLOSED | `specRiskHalt` rejects malformed targets and bounded verification covers it |
| ZERO-VIABLE-SIGNAL-001 | No strategy signal had met the recorded long-sample viability threshold | CRITICAL | trader-firm-research | OPEN | Run and record a current long-dataset viability evaluation |

## Update rule

Change `formal/risk-register.json` first, then update both projections in the
same commit. IDs are permanent. Reopening a risk changes its status rather than
creating a duplicate row; a materially different risk receives a new ID.

Last reconciled: 2026-09-04.
