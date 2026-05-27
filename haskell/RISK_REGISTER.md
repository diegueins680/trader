# Trader Firm — Risk Register

| ID | Risk | Severity | Owner | Status | Next Action |
|---|---|---|---|---|---|
| SCHEMA-001 | Live trade log schema drift (16-field / ISO-8601 contract) | **CRITICAL** | trader-firm-cio | **RESOLVED** 2026-05-20 | CIO delivered schema; Data validated |
| VOL-TARGET-001 | `volConfStatefulCloseDirection` regression breaks `cabal test` | **CRITICAL** | trader-firm-cto | OPEN | Deadline 2026-05-25 06:00 UTC missed; repro: `cd haskell && cabal test trader-tests`; escalate to CEO if not fixed by 2026-05-27 06:00 UTC |
| THRESHOLD-FACTOR-001 | `thresholdFactor` not wired into simulation config | Medium | trader-firm-research | OPEN | Confirm research spec; handoff to Execution for wiring |
| TRAILING-STOP-001 | Trailing-stop exit may re-enter on same bar (race) | Medium | trader-firm-execution | OPEN | Add bar-level re-entry lock after trailing-stop exit |
| RISK-LIMIT-001 | Daily / weekly / drawdown limits not enforced in live trading loop | High | trader-firm-risk | **RESOLVED** 2026-05-24 | Runtime spec-coupled invariant checks landed; guardrail tests pass |
| BINARY-HANG-001 | `trader-hs` binary occasionally hangs on shutdown (SIGTERM) | Medium | trader-firm-cto | OPEN | Add timeout + explicit `exitWith` in signal handler |
| GITHUB-502-001 | Autoloop GitHub API 502 retries are unbounded | **CRITICAL** | trader-firm-cto | OPEN | Deadline 2026-05-26 22:00 UTC missed; fix scoped (exponential backoff + cap); escalate to CEO if not deployed by 2026-05-27 06:00 UTC |
| EXECUTION-DATASET-001 | Dataset generation for backtests is non-deterministic | Medium | trader-firm-data | OPEN | Seed RNG and snapshot dataset hash in test output |
| AUTOLOOP-SINGLETON-001 | Multiple autoloop instances may race on same repo | High | trader-firm-cto | OPEN | Add PID file / lockfile in `scripts/autoloop-forever.mjs` |
| AUTOLOOP-STALL-001 | Autoloop stall detection is manual (no heartbeat) | **CRITICAL** | trader-firm-cto | **RESOLVED** 2026-05-25 | Telemetry recovered (cycle 163 at 01:10 UTC); heartbeat NDJSON emitted every 60s; alert if >5 min gap |
| **CIO-DEAFNESS-001** | CIO missed deadlines; no report since 18:31 UTC | **CRITICAL** | trader-firm-ceo | OPEN | CEO to ping CIO; GO/NO-GO deadlines 2026-05-25 22:00 UTC and 2026-05-26 22:00 UTC both missed; escalate to owner |
| **ZERO-VIABLE-SIGNAL-001** | No method achieves Sharpe >= 0.20 on >= 5000-bar data | **CRITICAL** | trader-firm-cio / trader-firm-research | OPEN | Research deadline 2026-05-25 23:30 UTC missed; no new data since 2026-05-25 03:52 UTC; escalate to CEO |
| **KALMAN-NUMSTAB-001** | Kalman filter numerical instability / zero trades / hangs | **CRITICAL** | trader-firm-cto | OPEN | CTO deadline 2026-05-25 23:30 UTC missed; diagnosis artifact not delivered; escalate to CEO if not delivered by 2026-05-27 06:00 UTC |
| **EXECUTION-MISSING-001** | Execution missed 5+ consecutive deadlines; no trade-log spec | **CRITICAL** | trader-firm-execution | OPEN | Execution deadlines 2026-05-25 23:00 UTC and 2026-05-26 22:00 UTC both missed; escalate to CEO |
| **AUTOLOOP-SINGLETON-001** | Multiple autoloop instances may race on same repo | High | trader-firm-cto | OPEN | PID 70643 healthy (cycle 48); metrics path divergence noted; add PID file / lockfile in `scripts/autoloop-forever.mjs` |
| **TRADE-LOG-GAP-001** | Missing `exit_reason` / `halt_reason` in trade-log schema | **HIGH** | trader-firm-cio | OPEN | Add `exit_reason` field to schema v1.1; update Execution emitters by 2026-05-27 18:00 UTC |
| **TRADE-LOG-GAP-002** | Missing native drawdown, daily/weekly loss, expectancy fields | **MEDIUM** | trader-firm-cio | OPEN | Add computed risk-state snapshot or derived fields to schema v1.1 by 2026-05-27 18:00 UTC |

---

*Last updated: 2026-05-27 12:10 UTC by trader-firm-risk*
*Next review: 2026-05-27 18:00 UTC or upon any status change*
