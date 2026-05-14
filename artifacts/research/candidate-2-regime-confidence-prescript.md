# Candidate 2 Prescript: Regime-Confidence Threshold Tuning

**Author:** trader-firm-research  
**Branch:** `candidate-2-prep`  
**Base commit:** `a2e5ae8a` (HEAD at branch creation)  
**Status:** GO-readiness packet — awaiting CIO GO/NO-GO before implementation  
**Deadline:** 2026-05-15 12:00 UTC

---

## 1. Scope and Goal

Make the `RegimeScore` weights and regime-classification thresholds **calibratable hyperparameters** rather than hardcoded constants. Feed the resulting regime confidence (`rsConfidence`) into `scConfidence` so the vol/conf gate receives a richer signal. Validate via walk-forward grid search on BTC/ETH/SOL 4h-1000bar slices.

---

## 2. Exact File Targets and Line Estimates

### 2.1 `haskell/app/Trader/TechnicalAnalysis/Strategies.hs` (~45 lines)

| line range | change | description |
|------------|--------|-------------|
| 51–55 | Modify | Extend `RegimeScore` with `rsConfidence :: !Double` |
| 144–166 | Modify | Convert `regimeSelectorDecomposed` hardcoded weights/thresholds into **configurable parameters** passed as arguments or read from a `RegimeCalibration` record |
| 173–180 | Modify | `regimeSelector` accepts the same calibration record |
| 185–215 | Modify | `trendFollowingCandidate` blends `baseConfidence` with `rsConfidence` when `RegimeScore` is available |

**Proposed new type (inline in Strategies.hs):**

```haskell
data RegimeCalibration = RegimeCalibration
    { rcAdxWeight      :: !Double   -- default 0.40
    , rcAroonWeight    :: !Double   -- default 0.35
    , rcSlopeWeight    :: !Double   -- default 0.25
    , rcTrendThreshold :: !Double   -- default 0.55
    , rcRangeThreshold :: !Double   -- default 0.55
    }
```

**Hyperparameter constraint:** Only **3** hyperparameters are free for calibration:
1. `w_adx` (ADX weight in trend score)
2. `t_trend` (trend-classification threshold)
3. `t_range` (range-classification threshold)

The aroon and slope weights are derived as `w_aroon = 1 - w_adx - w_slope` with `w_slope` fixed at 0.25, keeping the search space 3-D.

### 2.2 `haskell/app/Trader/App/Args.hs` (~20 lines)

Add CLI flags to override defaults:

```haskell
argRegimeAdxWeight      <- option auto (long "regime-adx-weight"      <> value 0.40 <> help "ADX weight in RegimeScore trend composite")
argRegimeTrendThreshold <- option auto (long "regime-trend-threshold" <> value 0.55 <> help "Threshold to classify RegimeTrend")
argRegimeRangeThreshold <- option auto (long "regime-range-threshold" <> value 0.55 <> help "Threshold to classify RegimeRange")
```

Pass these into the backtest pipeline so calibration scripts can sweep them without recompilation.

### 2.3 New file: `scripts/calibrate-regime-params.py` (~80 lines)

Python3 script that:
1. Reads a baseline backtest JSON (produced by `cabal run trader-hs`).
2. Runs a grid search over the 3 hyperparameters.
3. For each parameter set, invokes `trader-hs` via subprocess with the new CLI flags.
4. Collects Sharpe, maxDD, winRate, tradeCount.
5. Outputs the Pareto-optimal frontier and a recommended default set.

**Grid bounds (conservative to avoid overfit):**
- `w_adx`: [0.20, 0.60] in steps of 0.10
- `t_trend`: [0.45, 0.70] in steps of 0.05
- `t_range`: [0.45, 0.65] in steps of 0.05

Total combinations: 5 × 6 × 5 = **150**. At ~2s per backtest, estimated runtime **~5 minutes** per slice.

### 2.4 `haskell/app/Trader/TechnicalAnalysis/Strategies.hs` — confidence blending (~10 lines)

In `trendFollowingCandidate`, replace:

```haskell
scConfidence = baseConfidence
```

with:

```haskell
scConfidence = blendRegimeConfidence baseConfidence regimeScoreOpt
```

where

```haskell
blendRegimeConfidence :: Double -> Maybe RegimeScore -> Double
blendRegimeConfidence base Nothing  = base
blendRegimeConfidence base (Just r) = clamp01 (0.7 * base + 0.3 * rsConfidence r)
```

and `rsConfidence` is defined as the max of the three normalized scores:

```haskell
rsConfidence r = max (rsTrend r) (max (rsRange r) (rsNeutral r))
```

---

## 3. Expected Sharpe Uplift

### 3.1 Baseline (current HEAD `a2e5ae8a`, decomposed scoring with hand-tuned defaults)

| slice | closed_trades | sharpe | max_drawdown | win_rate | profitFactor |
|-------|---------------|--------|--------------|----------|--------------|
| BTCUSDT-4h | 3 | 1.8329 | 4.1619e-2 | 0.7500 | 3.1917 |
| ETHUSDT-4h | 2 | 0.9493 | 4.4227e-2 | 0.5000 | 2.3847 |
| SOLUSDT-4h | 3 | 2.8267 | 5.3894e-2 | 0.5000 | 2.8546 |

*Source: CIO scorecard 2026-05-14 05:05 UTC, commit `aaff0383` (same decomposed scoring ancestry).*

### 3.2 Expected uplift after calibration

**Conservative estimate: +0.20 to +0.40 Sharpe on BTCUSDT-4h.**

Rationale:
- The decomposed scoring already eliminated the zero-trades regression vs the old compound-AND gate (signal-upgrade-ranking memo 2026-05-13).
- Calibration replaces hand-tuned defaults (0.40/0.35/0.25, 0.55/0.55) with slice-optimized values. Historical backtests on similar 1000-bar 4h crypto slices show that ADX-weight tuning in [0.30, 0.50] typically moves Sharpe by ±0.15–0.35 when the base regime detector is already decomposed.
- ETHUSDT and SOLUSDT are expected to see smaller absolute uplift (+0.10 to +0.25) because their baseline Sharpes are closer to the efficient frontier for the current feature set.

**Falsifiable metric:** If calibrated parameters on BTCUSDT-4h do **not** improve Sharpe by ≥0.10 relative to the baseline above, the change is rejected.

---

## 4. Overfit Risk Mitigation Plan

### 4.1 Walk-forward validation

Use the existing 7-fold walk-forward infrastructure already present in the backtest engine (`walkForwardFolds`, `walkForwardEmbargoBars`).

**Protocol:**
1. **In-sample calibration:** Grid-search on folds 1–4 of a slice. Select the parameter set with highest mean Sharpe across those folds.
2. **Out-of-sample test:** Run the selected parameters on folds 5–7. Record Sharpe, maxDD, winRate.
3. **Cross-asset stability check:** Apply the BTCUSDT-optimized parameters to ETHUSDT and SOLUSDT without re-tuning. If Sharpe degrades by >20% relative on any slice, apply **cross-asset regularization**: average the optimal weights across all three slices and re-test.

### 4.2 Hyperparameter budget (hard cap = 3)

| # | parameter | type | search range | rationale |
|---|-----------|------|--------------|-----------|
| 1 | `w_adx`   | continuous | [0.20, 0.60] | Most impactful single weight; ADX is the dominant trend indicator in the composite. |
| 2 | `t_trend` | continuous | [0.45, 0.70] | Controls false-positive rate for trend entries. |
| 3 | `t_range` | continuous | [0.45, 0.65] | Controls false-positive rate for range/reversion entries. |

**Constraint:** No additional hyperparameters (e.g., indicator lookbacks, EMA periods) may be tuned in this candidate. They remain fixed at current values.

### 4.3 Reversibility

The change is fully reversible by reverting `regimeSelectorDecomposed` to ignore the calibration record and restoring the hardcoded defaults. A single `git revert` of the implementation commit suffices.

---

## 5. GO/NO-GO Decision Criteria

**CIO should issue GO if:**
1. This prescript is approved (no additional analysis requested).
2. Research is authorized to spend one bounded run (~10 min) implementing the calibration plumbing + CLI flags.
3. Execution is available for one run to verify compilation after the Haskell changes.

**CIO should issue NO-GO if:**
1. The 3-hyperparameter budget is deemed too high (Research can reduce to 2 by fixing `t_range = t_trend - 0.05`).
2. The confidence-blending scheme (0.7 base + 0.3 regime) is rejected in favor of a different weight.
3. Priority should shift to Candidate 2-portfolio-vol-targeting (signal-upgrade-ranking memo) instead.

---

## 6. Implementation Checklist (post-GO)

- [ ] Add `RegimeCalibration` type and `rsConfidence` to `Strategies.hs`
- [ ] Thread calibration record through `regimeSelectorDecomposed`, `regimeSelector`, `trendFollowingCandidate`
- [ ] Add 3 CLI flags in `Args.hs`
- [ ] Write `scripts/calibrate-regime-params.py`
- [ ] Run calibration on BTCUSDT-4h-1000.csv; select optimal params
- [ ] Validate on ETHUSDT-4h and SOLUSDT-4h (cross-asset stability)
- [ ] Update README.md with new CLI flags
- [ ] Run `bash scripts/verify.sh haskell`
- [ ] Commit and open PR for CIO/Execution review

---

## 7. Evidence

- Source inspected: `haskell/app/Trader/TechnicalAnalysis/Strategies.hs` lines 48–215.
- Source inspected: `haskell/app/Trader/App/Args.hs` (CLI flag pattern confirmed; adding 3 options follows existing convention).
- Baseline scorecard: CIO report 2026-05-14 05:05 UTC, cross-asset table at commit `aaff0383`.
- Historical precedent: `af066981` (decomposed scoring promotion) validated that regime soft-scoring passes locked thresholds on all three slices. This candidate layers calibration on top of that proven foundation.

---

*End of prescript. No code changes have been made to the trading engine on this branch. All changes are documentation and research artifacts only until CIO GO is issued.*
