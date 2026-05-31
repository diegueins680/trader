# Daily Engineering Review — 2026-05-30

**Date:** 2026-05-30 (Saturday)  
**Timezone:** America/Guayaquil  
**Reviewer:** trader-firm-cio (autoloop)  
**UTC Cutoff:** 2026-05-31 04:00 UTC

---

## 1. Trade Summary

| Metric | Value |
|--------|-------|
| Completed trades | 4 |
| Compound return | **-0.7704%** |
| Average return | -0.1931% |
| Win rate | **0.0%** (all losers) |
| Symbols traded | ETCUSDT |
| Interval | 3m |
| Side | short (all) |

### Completed Trades

| # | Symbol | Entry | Exit | PnL | Hold | Entry Regime | Exit Regime |
|---|--------|-------|------|-----|------|--------------|-------------|
| 1 | ETCUSDT 3m | 14:30 @ 8.237 | 14:36 @ 8.248 | -0.2749% | 2 bars | range-drift (eff=0.2255) | chop (eff=0.0079) |
| 2 | ETCUSDT 3m | 14:42 @ 8.245 | 14:48 @ 8.248 | -0.1780% | 2 bars | chop (eff=0.1598) | chop (eff=0.1247) |
| 3 | ETCUSDT 3m | 14:51 @ 8.246 | 15:00 @ 8.253 | -0.2264% | 3 bars | chop (eff=0.1664) | chop (eff=0.1060) |
| 4 | ETCUSDT 3m | 15:06 @ 8.256 | 15:12 @ 8.252 | -0.0932% | 2 bars | chop (eff=0.1480) | chop (eff=0.1245) |

**All 4 trades were losing shorts entered in chop or range-drift regimes.**

---

## 2. Critical Finding: Directionality Gate Bypass

### Hypothesis
The directionality gate (`directionalityChopEfficiencyMax = 0.08`) should have vetoed all 4 entries, yet trades executed.

### Evidence

Efficiency at decision points (entry_idx - 1, 24-bar lookback):

| Trade | Decision Idx | Efficiency | Veto @ 0.08 | Veto @ 0.18 |
|-------|-------------|------------|-------------|-------------|
| 1 | 1001 | 0.0802 | NO | **YES** |
| 2 | 1005 | 0.0090 | **YES** | **YES** |
| 3 | 1008 | 0.1247 | NO | **YES** |
| 4 | 1013 | 0.1714 | NO | **YES** |

At 0.18 (pre-dce28b48), **all 4 trades would have been vetoed**.  
At 0.08 (post-dce28b48), **trade 2 would have been vetoed**.

The fact that **all 4 trades executed** means the directionality gate was **completely inactive** at entry time.

### Root Cause Analysis

| Hypothesis | Likelihood | Evidence |
|-----------|-----------|----------|
| **H1: Live binary was stale** (pre-gate or pre-0.08 patch) | **HIGH** | dce28b48 patch was May 11; autoloop DOWN since May 29 21:27 UTC; no build commit in snapshots to verify |
| H2: Gate bypass in code path | LOW | Main.hs review confirms gate IS in decision path for MethodKalmanOnly; no bypass found |
| H3: Different prices in live vs snapshot | LOW | Efficiency uses 24-bar lookback; same range regardless of future bars |

**Verdict: H1 is most likely.** The live ETCUSDT bot was running a binary that did not include the directionality gate.

---

## 3. Operational Findings

### Fill Evidence Gaps
- **3 of 11** order events are ack-only (`status: NEW`, `executedQty: 0`)
- Missing fill confirmation for entry orders
- This prevents post-hoc verification of whether gates were checked before fill

### Stale Snapshots
- **6/22 symbols (27.3%)** had stale snapshots at cutoff
- ETCUSDT: stale for **7h 45m** (budget: 3m)
- TRXUSDT: stale for **24h 23m** (budget: 3m)
- XRPUSDT: stale for **45m** (budget: 3m)

### Cutoff Signal Audit
- 18/22 symbols had measurable edge
- Only **3/22** above open threshold
- **0/22** with malformed directionality
- Strongest candidate: NEARUSDT 12h (2.57x threshold ratio) blocked by TREND gate

---

## 4. Invariant Violations

| Invariant | Expected | Observed | Status |
|-----------|----------|----------|--------|
| No entry in chop (eff <= 0.08) | Veto | All 4 entries allowed | **VIOLATED** |
| Fill evidence for all orders | Fill confirmation | 3 ack-only gaps | **VIOLATED** |
| Snapshot freshness < budget | Age < budget | 6 symbols stale | **VIOLATED** |
| Build commit in snapshot | `buildCommit` field | Field absent | **VIOLATED** |

---

## 5. Implemented Fixes

### 5.1 Build Commit Telemetry
- Added `botBuildCommit :: Maybe String` to `BotState`
- Emitted in `botStatusJson` as `"buildCommit"` field
- Preserved across `botApplyKline` updates
- **Purpose:** Enable post-hoc verification of which code version was running

### 5.2 Executable Halt Spec Hardening
- Added `RISK_LIMIT_NON_FINITE` halt: NaN/Infinity risk limits silently disable halts
- Added `DRAWDOWN_LIMIT_INVALID` halt: drawdown <=0 or >=1 is corrupted config
- Added `POSITION_SIZE_INVALID` halt: non-finite, negative, or >10x sizes rejected
- All three proven in `verifyFormalRisk` with exhaustive generators

### 5.3 Risk Register
- Added `Trader.Formal.RiskRegister` module with canonical risk register
- IDs: `KALMAN_NUMSTAB_001`, `GITHUB_502_001`, `ZERO_VIABLE_SIGNAL_001`, `TRADE_LOG_GAP_001`, `RISK_LIMIT_NON_FINITE_001`, `LEVERAGE_SANITY_001`, `AUTOLOOP_RESET_2026_05_30`

### 5.4 Test Updates
- Updated `testFormalRiskNegativeLimitSanitization`: negative drawdown now triggers `DRAWDOWN_LIMIT_INVALID`
- Updated `testFormalRiskPositionSizeHalt`: negative position size now triggers `POSITION_SIZE_INVALID`
- Added three new invariant assertions in `testFormalRiskInvariants`

---

## 6. Remaining Open Items

| # | Item | Owner | Priority |
|---|------|-------|----------|
| 1 | **Investigate why directionality gate was inactive for ETCUSDT** | trader-firm-cto | **CRITICAL** |
| 2 | Fix fill-evidence gaps (ack-only orders without fill confirmation) | trader-firm-execution | HIGH |
| 3 | Investigate stale snapshot root cause (27% stale rate) | trader-firm-cto | HIGH |
| 4 | Add test: verify chop threshold is exactly 0.08 | trader-firm-cio | MEDIUM |
| 5 | Consider interval-aware chop thresholds (3m may need different threshold than 12h) | trader-firm-research | MEDIUM |
| 6 | Restart autoloop and verify all bots running latest binary | trader-firm-cto | **CRITICAL** |

---

## 7. Metrics

```
completed_trades      = 4
compound_return_pct   = -0.7704
win_rate              = 0.0
avg_return_pct        = -0.1931
stale_snapshot_rate   = 0.273
fill_evidence_gaps    = 3
non_directional_vetos = 10 (6 exit/flatten, 4 entry/add)
```

---

*Next review: 2026-05-31 06:00 UTC or upon status change*
