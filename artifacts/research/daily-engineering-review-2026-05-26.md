# Trader Daily Engineering Review — May 26, 2026

**Reviewer:** trader-firm-engineering  
**Date:** 2026-05-26 11:38 AM (America/Guayaquil) / 16:38 UTC  
**Status:** CRITICAL — Zero viable signal production across 22 symbols

---

## 1. Executive Summary

Today's live trading analysis reveals a **systemic signal generation failure**. Of 22 tracked symbols:
- **Only 1 symbol (ETC) executed trades today**
- **20 symbols remained permanently neutral**
- **1 symbol (XRP) blocked by market data gap**

Total trades: 4 across all symbols (2 from stale states, 2 from ETC today)
Net P&L: approximately flat (+0.08% on ETC, -0.02% on TRX)

---

## 2. Failure Mode Analysis

### 2.1 Kalman Neutral (7 symbols, 32%)
**Symptoms:** Kalman filter classifies as NEUTRAL despite price movement
**Evidence:**
- AVAX: kalmanPredNext drifted 50% from spot (18.36 → 9.13)
- ETH: kalmanPredNext drifted 10% from spot (2293 → 2046)
- BNB: Spot 650.73 vs Kalman 651.27 (0.08% diff, classified NEUTRAL)

**Root Cause:** 
- Default --kalman-process-var = 1e-5 is too low
- Process variance 100x smaller than measurement variance (1e-3)
- Filter becomes rigid, ignores new price data
- Posterior variance collapses → z-score unstable

**Engineering Fix:**
```haskell
-- In KalmanFusion.hs, add variance floor:
postVar = max 1e-6 (1 / postPrec)

-- In Args.hs, increase default:
--kalman-process-var 1e-3  (was 1e-5)
```

### 2.2 LSTM Neutral (4 symbols, 18%)
**Symptoms:** LSTM prediction close to spot → classified NEUTRAL
**Evidence:**
- SOL: LSTM 84.09 vs Spot 84.36 (0.3% diff)
- DOGE: LSTM -0.0003 vs Spot 0.102 (neutral classification)

**Root Cause:**
- LSTM outputs raw price levels, not returns
- No directional signal when price ≈ prediction
- DOGE open threshold = 160% (broken scaling)

**Engineering Fix:**
- LSTM should predict returns (pct change), not price levels
- Add output normalization for low-priced assets
- Cap maximum threshold at 10%

### 2.3 EDGE_HEADROOM (4 symbols, 18%)
**Symptoms:** Edge < 1.5 × openThreshold → trade blocked
**Evidence:**
- BTC: openThreshold = 2.923%, need >4.4% edge to trade
- DOT: openThreshold = 101.7% (absurd — broken config)

**Root Cause:**
- entryEdgeHeadroomMultiple = 1.5 is too conservative
- Some thresholds are clearly broken (DOGE 160%, DOT 101%)

**Engineering Fix:**
```haskell
-- In SignalGates.hs:
entryEdgeHeadroomMultiple = 1.2  -- was 1.5

-- Add threshold sanity check:
maxOpenThreshold = 0.10  -- 10% cap
```

### 2.4 NON_DIRECTIONAL_CHOP (3 symbols, 14%)
**Symptoms:** Regime detection blocks trades in "choppy" market
**Evidence:**
- BTC: Method predicts DOWN, but regime = CHOP → NEUTRAL
- ETC: Method predicts DOWN, but regime = CHOP → NEUTRAL (yet ETC traded?)

**Root Cause:**
- Regime detection may be too sensitive
- Conflict between directional prediction and regime label

---

## 3. Hypotheses and Tests

### H1: Low process variance causes Kalman filter rigidity
**Test:** Run backtest with --kalman-process-var 1e-3 vs 1e-5
**Metric:** Count of "Kalman neutral" decisions, kalman drift vs spot
**Expected:** Higher process variance → fewer neutral classifications, better tracking

### H2: Edge headroom multiple is too conservative
**Test:** Run backtest with entryEdgeHeadroomMultiple = 1.2 vs 1.5
**Metric:** Trade count, Sharpe ratio, max drawdown
**Expected:** More trades, similar or better risk-adjusted returns

### H3: Threshold calibration produces absurd values for some symbols
**Test:** Audit threshold distribution across all symbols
**Metric:** Histogram of openThreshold values
**Expected:** All thresholds < 10%; identify outliers

---

## 4. Implementation Plan

### P1: Kalman Filter Fix (CRITICAL)
- [ ] Add minimum variance floor in KalmanFusion.hs
- [ ] Increase default --kalman-process-var to 1e-3
- [ ] Add adaptive process variance option
- [ ] Backtest on BNBUSDT-5m dataset

### P2: Edge Headroom Fix (HIGH)
- [ ] Reduce entryEdgeHeadroomMultiple to 1.2
- [ ] Add threshold sanity cap (max 10%)
- [ ] Backtest with modified parameters

### P3: LSTM Output Fix (HIGH)
- [ ] Change LSTM to predict returns instead of price levels
- [ ] Add output scaling normalization
- [ ] Validate on low-priced assets (DOGE, TRX)

### P4: Monitoring (MEDIUM)
- [ ] Add "bars_since_last_trade" metric per symbol
- [ ] Add "neutral_rate_by_reason" dashboard
- [ ] Alert when neutral_rate > 95% for >100 bars

---

## 5. Validation Plan

```bash
# Test 1: Kalman parameter sweep
./trader-hs --data data/BNBUSDT-5m-2020-06_full.csv --method 10 \
  --kalman-process-var 1e-3 --kalman-measurement-var 1e-3 \
  --vol-target 0.10 --json

# Test 2: Edge headroom comparison
# (requires code change to SignalGates.hs)

# Test 3: Full backtest with fixes
./trader-hs --data data/BTCUSDT-4h-1000.csv --method 01 \
  --kalman-process-var 1e-3 --open-threshold 0.01 \
  --vol-target 0.10 --json
```

---

## 6. Risk Register Updates

| ID | Risk | Severity | Status | Action |
|---|---|---|---|---|
| KALMAN-NUMSTAB-001 | Kalman filter numerical instability / zero trades | CRITICAL | CONFIRMED | Implement P1 fixes |
| THRESHOLD-FACTOR-001 | thresholdFactor not wired into simulation config | Medium | OPEN | Add sanity caps |
| ZERO-VIABLE-SIGNAL-001 | No method achieves Sharpe >= 0.20 on >= 5000-bar data | CRITICAL | CONFIRMED | Root cause: permanent neutrality |

---

*Next review: 2026-05-27 11:00 UTC or upon implementation of P1 fixes*
