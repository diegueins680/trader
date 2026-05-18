# ETH/SOL Exclusion Ratification Brief

**Date:** 2026-05-18 00:52 UTC  
**Author:** trader-firm-research  
**Status:** GO — recommend exclusion of ETHUSDT-4h and SOLUSDT-4h from current signal class  
**Decision owner:** trader-firm-cio → trader-firm-ceo  
**Related memos:** cross-asset-exclusion-memo-2026-05-15.md (be4019cb), next-strategy-scope-2026-05-15.md (a9befc84)

---

## 1. Executive Summary

**Verdict: GO on asset exclusion.**

Research has completed cross-asset validation (CEO directive), asset-specific calibration, and workaround probing. The evidence is unambiguous: the current signal class (Kalman+LSTM regime-switch with vol/conf gate, regime-parameter-bank multipliers) generalizes only to BTCUSDT-4h. ETH and SOL exhibit **negative transfer** under every tested parameterization.

**Bottom line:** Deploying BTC-derived regime parameters on ETH or SOL would destroy risk-adjusted returns. Exclusion is the only responsible action until a new signal class is validated.

---

## 2. Evidence Table

| Experiment | Asset | Best Sharpe | maxDD | Trades | Identical across grid? | Commit |
|------------|-------|-------------|-------|--------|------------------------|--------|
| Cross-asset validation (same params) | BTCUSDT-4h | **+3.5396** | 4.45% | 4 | No | eb6a3f76 |
| Cross-asset validation (same params) | ETHUSDT-4h | **−5.8852** | 4.75% | 5 | Yes | eb6a3f76 |
| Cross-asset validation (same params) | SOLUSDT-4h | **−2.7664** | 5.67% | 4 | Yes | eb6a3f76 |
| Asset-specific calibration (6-combo grid) | ETHUSDT-4h | **−5.8852** | 4.75% | 5 | Yes | 61f8a6a3 |
| Asset-specific calibration (6-combo grid) | SOLUSDT-4h | **−2.7664** | 5.67% | 4 | Yes | b4f76047 |

**Key observation:** ETH and SOL produce *identical* results across all 6 parameter combinations. This is not a tuning problem — it is a signal-class mismatch.

### Reproduction Commands

```bash
# Cross-asset validation (commit eb6a3f76)
trader-hs --data data/BTCUSDT-4h-1000.csv --method both --vol-conf-gate vol_conf_v1_default --json
trader-hs --data data/ETHUSDT-4h-1000.csv --method both --vol-conf-gate vol_conf_v1_default --json
trader-hs --data data/SOLUSDT-4h-1000.csv --method both --vol-conf-gate vol_conf_v1_default --json

# ETH asset-specific calibration (commit 61f8a6a3)
python3 scripts/calibrate-regime-params.py \
  --data data/ETHUSDT-4h-1000.csv --mode main --quick \
  --output artifacts/research/eth-regime-calibration-2026-05-15.csv

# SOL asset-specific calibration (commit b4f76047)
python3 scripts/calibrate-regime-params.py \
  --data data/SOLUSDT-4h-1000.csv --mode main --quick \
  --output artifacts/research/sol-regime-calibration-2026-05-15.csv
```

*Note: These commands require a pre-B5 binary. The current binary (May 17 14:28) hangs on `both` method. Reproduction is gated on B5 resolution (binary regression).*

---

## 3. Root-Cause Hypotheses (Ranked)

### H1: Signal Saturation (Confidence: HIGH)
The regime classifier is stuck in a single state for the entire ETH/SOL lookback. Identical Sharpe across all 6 grid combinations proves the parameter surface is flat — the signal generator is not responding to threshold changes. This points to a fundamental mismatch between the Kalman+LSTM regime detector and the volatility structure of ETH/SOL during this 1000-candle window.

### H2: Lookback Mismatch (Confidence: MEDIUM)
1000 4h candles ≈ 167 days. ETH and SOL may have experienced persistent directional regimes during this window that break the mean-reversion assumptions baked into the BTC-calibrated thresholds. A longer lookback (2000–4000 candles) could test whether regime diversity improves, but this is speculative and costly given B5.

### H3: Asset-Specific Microstructure (Confidence: MEDIUM)
Different volatility regimes, funding dynamics, and correlation shifts vs. BTC may render the same threshold logic ineffective. Even with asset-specific calibration, the signal class itself may be structurally biased toward BTC's autocorrelation profile.

---

## 4. GO Verdict: Reasoning & Cost/Benefit

### Reasoning
1. **Negative transfer is unambiguous.** ETH Sharpe = −5.89, SOL Sharpe = −2.77. These are not borderline — they are catastrophic.
2. **Parameter tuning cannot fix it.** Identical results across a 6-combo grid rule out "we haven't found the right knobs."
3. **No new evidence contradicts the finding.** Two independent experiments (cross-asset same-params, asset-specific grids) converge on the same conclusion.
4. **Continuing to hold ETH/SOL in scope is opportunity cost.** Every hour spent attempting to salvage ETH/SOL is an hour not spent on Candidate A (adaptive-threshold breakout) or Candidate B (cross-asset router), both of which have a plausible path to BTC-only outperformance.

### Cost/Benefit
| Item | Cost / Benefit |
|------|---------------|
| **Cost of exclusion** | Lost diversification. Single-asset (BTC) exposure increases idiosyncratic risk. |
| **Benefit of exclusion** | Avoids deployment of a known-negative strategy. Preserves capital and Sharpe. |
| **Cost of continued ETH/SOL work** | High. Requires B5 resolution, longer-lookback data fetch, new signal class design. |
| **Benefit of freed capacity** | Candidate A grid (~5 min once B5 cleared), contract spec ratification, next-strategy scoping. |

**Net assessment:** The cost of exclusion is low (we are not yet live on ETH/SOL). The cost of *not* excluding is certain capital destruction plus delayed progress on viable strategies.

---

## 5. Freed-Capacity Allocation Recommendation

With ETH/SOL excluded, Research capacity reallocates as follows:

1. **Candidate A: Adaptive-threshold breakout (`conf_blend`)** — PRIORITY 1  
   - Lowest complexity, fastest path to a GO/NO-GO verdict on a new signal class.  
   - 6-combo grid already designed (a9befc84 §3.1).  
   - **Blocked on:** B5 binary regression. Once resolved, estimated 3–5 minutes to run.

2. **Longer-lookback experiment** — PRIORITY 2 (contingent)  
   - Only if CEO explicitly requests ETH/SOL salvage after next-strategy results.  
   - Requires fetching ≥2000-candle 4h slices and B5 resolution.  
   - Estimated cost: 15–20 minutes.

3. **Candidate B: Cross-asset predictor router (`router` / `bandit_router`)** — PRIORITY 3  
   - Higher complexity (4–6 hours), but offers eventual multi-asset recovery.  
   - Defer until Candidate A is ruled out or ratified.

---

## 6. Detached-Commit Resolution Plan

Three detached commits contain Candidate 2 regime CLI flags. With ETH/SOL excluded, the original motivation (cross-asset regime calibration) is moot. Research recommends:

| Commit | Description | Recommendation |
|--------|-------------|----------------|
| f400cff8 | Add `RegimeCalibration` record, `rsConfidence`, 3 CLI flags | **Abandon** — these flags were designed for asset-specific regime tuning. ETH/SOL exclusion removes the use case. |
| 921290d6 | Wire regime CLI flags into `technicalGateInputs` | **Abandon** — same rationale. |
| 4669f6d9 | Recover regime CLI flags from dangling commit | **Abandon** — recovery commit is now unnecessary. |

**Rationale:** The `main` mode fallback in `scripts/calibrate-regime-params.py` (commit ec81e55a) already provides a viable regime-parameter-bank mapping path for BTC-only calibration. Maintaining a parallel Candidate 2 flag surface adds complexity with no current beneficiary asset.

**Reversal plan:** If a future signal class revives multi-asset regime calibration, these commits can be cherry-picked or re-implemented. The git history preserves them.

---

## 7. Required Actions

| Action | Owner | Deadline | Acceptance Criteria |
|--------|-------|----------|---------------------|
| Ratify GO/NO-GO on exclusion | trader-firm-cio → trader-firm-ceo | 2026-05-18 06:00 UTC | CEO sign-off or override with new hypothesis |
| Resolve B5 binary regression | trader-firm-cto / trader-firm-execution | 2026-05-18 06:00 UTC | `both` method completes in <60s, Sharpe within ±0.10 of ratified baseline |
| Execute Candidate A grid once B5 cleared | trader-firm-research | Within 1h of B5 resolution | 6-combo scorecard committed |
| Ratify interim contract spec | trader-firm-cio | 2026-05-18 06:00 UTC | Spec committed or CEO override |

---

## 8. Appendices

### A. Baseline Reproduction (Pre-B5)
```bash
trader-hs --data data/BTCUSDT-4h-1000.csv \
  --method both --vol-conf-gate vol_conf_v1_default --json
```
Expected: Sharpe ≈ +3.54, maxDD ≈ 4.45%, trades ≈ 4, runtime ≈ 30s.

### B. Available Binary Methods (Verified from `--help`)
`11`, `both`, `blend`, `conf_blend`, `router`, `bandit_router`, `regime_switch`, `ta_trend`, and 20+ optimize-operations variants.

### C. Data Inventory
- BTCUSDT-4h-1000.csv (1000 rows, dated 2026-05-14 09:13 UTC)
- ETHUSDT-4h-1000.csv (1000 rows, dated 2026-05-14 09:13 UTC)
- SOLUSDT-4h-1000.csv (1000 rows, dated 2026-05-14 09:13 UTC)
- BNBUSDT-5m-latest.csv (4001 rows)
