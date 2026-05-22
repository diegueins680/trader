# Self-Healing & Continuous Improvement — Trader-Firm

## 1. Philosophy

> *"The system should fix itself before the pager goes off, and learn from every incident so the same root cause never pages twice."*

This document defines the self-healing and continuous-improvement (SHCI) contract for the `trader-firm` repo. It is a living specification: every incident, every near-miss, and every deliberate experiment updates one or more sections here.

---

## 2. Self-Healing Architecture

### 2.1 Layers

| Layer | Responsibility | Current Implementation | Gap |
|-------|---------------|------------------------|-----|
| **L1: Process** | Keep critical processes alive | `launchd` LaunchAgents (`ai.openclaw.trader.api`, `ai.openclaw.trader.monitor`) | None |
| **L2: Health Check** | Detect degradation before failure | `trader-monitor.sh` polls port every 120s; OpenClaw cron every 5 min | Stale snapshot rate not yet auto-remediated |
| **L3: Auto-Remediation** | Restart / failover / degrade gracefully | `KeepAlive` restarts API; monitor restarts on port-down | No automatic bot-worker pool scaling |
| **L4: State Recovery** | Resume from last known good state | Bot snapshots resume closed-trade memory; autoloop recovery branches | Orphaned-position adoption needs operator confirmation |
| **L5: Learning** | Update code / config / thresholds from incident | Autoloop bounded cycle + formal methods review | Not yet wired to auto-PR for routine fixes |

### 2.2 Health Signals & Auto-Actions

| Signal | Threshold | Auto-Action | Escalation |
|--------|-----------|-------------|------------|
| API port down | 2 consecutive failures | `launchctl kickstart` | Telegram alert after 3rd failure |
| Stale snapshot rate | > 50% | Log warning; queue bot restart | Telegram alert after 10 min sustained |
| Bot worker crash | Any non-zero exit | `launchd` auto-restart | Telegram alert if > 3 in 10 min |
| Dirty worktree (autoloop) | On cycle start | Auto-commit to `autoloop/recovery/*` | Block cycle if commit fails |
| CI failure (autoloop) | Any non-zero exit | HLint auto-fix → retry; else open repair branch | Telegram alert if repair branch created |
| Negative Sharpe (backtest) | < -2.0 on default method | Flag for regime-switch review; no auto-trade | Engineering review within 24h |
| Daily loss halt triggered | Daily loss ≥ `--daily-loss-halt` | Flatten positions; block new entries until reset | Telegram alert immediately |

---

## 3. Continuous Improvement Loop

### 3.1 The Loop

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   OBSERVE   │───>│   ANALYZE   │───>│   DECIDE    │───>│    ACT      │
│  (metrics,  │    │  (formal    │    │  (autoloop  │    │  (code,     │
│   logs,     │    │   methods,  │    │   planner,  │    │   config,   │
│   incidents)│    │   backtests)│    │   operator) │    │   deploy)   │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
     ^                                                           │
     └───────────────────────────────────────────────────────────┘
                         (measure impact, feed back to OBSERVE)
```

### 3.2 Observability Requirements

Every improvement must be **measurable**. Before merging any change:

1. **Baseline metric** is recorded.
2. **Hypothesis** is stated in `ENGINEERING_REVIEW_YYYY-MM-DD.md`.
3. **Experiment** is run (backtest, walk-forward, or shadow live).
4. **Result** is compared to baseline with statistical significance where possible.
5. **Rollback criteria** are defined upfront.

### 3.3 Formal Methods as Improvement Guardrail

The formal verification modules (`Trader.Formal.Execution`, `Trader.Formal.Risk`) are not one-time proofs. They are **living contracts**:

- Any change to execution quantity logic must update the naive spec and re-run the exhaustive grid.
- Any change to risk halt logic must update the risk spec and re-run the verification.
- A CI job runs `stack test` on every PR; formal verification failures block merge.

---

## 4. Incident Response & Learning

### 4.1 Incident Levels

| Level | Definition | Response Time | Post-Incident Review |
|-------|-----------|---------------|----------------------|
| **Sev 1** | Loss of capital, system unavailability, data corruption | Immediate (< 15 min) | Within 24h; update FORMAL_METHODS.md if contract violated |
| **Sev 2** | Degraded performance, stale data, missed signals | < 1 hour | Within 48h; update HEARTBEAT.md monitoring if needed |
| **Sev 3** | Cosmetic issues, non-critical alerts, doc drift | < 24 hours | Next engineering review; update relevant docs |

### 4.2 Post-Incident Review Template

Every Sev 1/2 incident produces a file: `artifacts/incidents/INCIDENT_YYYY-MM-DD_{short-name}.md`

```markdown
# INCIDENT_YYYY-MM-DD_{short-name}

## Summary
One-line description.

## Timeline
- HH:MM UTC — Detection (how?)
- HH:MM UTC — Impact start
- HH:MM UTC — Mitigation applied
- HH:MM UTC — Recovery complete

## Root Cause
Technical root cause, not proximate cause.

## Formal Contract Impact
Did this violate any clause in FORMAL_METHODS.md? If yes, which and why?

## Fix
Code/config change that prevents recurrence.

## Verification
How do we know the fix works? (Test, backtest, formal proof)

## Prevention
What self-healing or monitoring change prevents this class of incident?
```

---

## 5. Autoloop as Continuous Improvement Engine

### 5.1 Autoloop Contract (from `mission.md`)

The autoloop is the **primary CI engine** for the repo:

- **Forever runner**: `scripts/autoloop-forever.sh` — keeps the bounded cycle alive.
- **Bounded cycle**: `scripts/autoloop-forever.mjs` — does one unit of work (review, fix, test, push).
- **Safe verification**: `bash scripts/verify.sh` must pass before any push.
- **Recovery**: Dirty worktrees are preserved to `autoloop/recovery/*` branches, not `main`.

### 5.2 Improvement Sources (Priority Order)

1. **GitHub Copilot/Codex review threads** — unresolved reviews are polled first.
2. **CI failures** — HLint auto-fix, then parser/build fix, then semantic fix.
3. **Engineering review hypotheses** — e.g., "regime switch improves Sharpe in chop."
4. **Operator-initiated experiments** — e.g., Candidate A execution packet.
5. **Formal verification gaps** — new invariants discovered during code review.

### 5.3 Autoloop Self-Healing Checklist

Before each bounded cycle:

- [ ] Git worktree metadata pruned
- [ ] Merged local branches detached from worktrees
- [ ] Dirty state committed to recovery branch if needed
- [ ] `stack test` passes
- [ ] `hlint app test bench` clean
- [ ] `npm run verify:automation` passes

If any check fails:
- **Auto-fixable** (HLint, dirty worktree): fix and retry.
- **Non-auto-fixable**: open `autoloop/checkpoint/YYYY-MM-DD_{description}` branch, log failure, alert operator.

---

## 6. Metrics & KPIs for SHCI

### 6.1 System Health

| Metric | Target | Measurement |
|--------|--------|-------------|
| API uptime | > 99.9% | `trader-monitor.sh` diagnostics log |
| Stale snapshot rate | < 10% | `review_bot_day.py` JSON output |
| Autoloop cycle success rate | > 95% | `autoloop-forever.sh` status |
| Mean time to recovery (MTTR) | < 5 min | Incident log timestamps |
| Mean time between failures (MTBF) | > 7 days | Incident log |

### 6.2 Improvement Velocity

| Metric | Target | Measurement |
|--------|--------|-------------|
| Backtest experiments per week | ≥ 2 | `ENGINEERING_REVIEW_*.md` count |
| Formal verification coverage | 100% of execution + risk | `stack test` formal suites |
| Regime-switch Sharpe vs. static | > 0 improvement | Walk-forward mean Sharpe |
| Code changes with baseline/result | 100% | PR description template |

---

## 7. Current Gaps & Next Actions

| Gap | Priority | Action | Owner | Deadline |
|-----|----------|--------|-------|----------|
| Stale snapshot auto-remediation | P0 | Add bot-worker pool restart to monitor script | @diegosaa | 2026-05-24 |
| Regime-switch iteration | P1 | Replace static thresholds with `RegimeScore` | @diegosaa | 2026-05-24 |
| Auto-PR for routine fixes | P2 | Wire autoloop to open PRs instead of pushing to main | @diegosaa | 2026-05-31 |
| Incident artifact directory | P2 | Create `artifacts/incidents/` template and first entry | @diegosaa | 2026-05-24 |
| Shadow live testing | P3 | Run `ta_breakout` shadow-live for 1 week vs. `ta_trend` | @diegosaa | 2026-06-07 |

---

## 8. References

- `mission.md` — Standing mission lane
- `FORMAL_METHODS.md` — Living formal contracts
- `ENGINEERING_REVIEW_*.md` — Weekly review logs
- `HEARTBEAT.md` — Monitoring configuration
- `scripts/autoloop-forever.sh` — Forever runner
- `scripts/autoloop-forever.mjs` — Bounded cycle
- `scripts/trader-monitor.sh` — Health check & auto-restart

---

*Last updated: 2026-05-21*
*Next review: 2026-05-24 or on next Sev 1/2 incident*
