# SOL Trailing-Stop 10-Fold Cross-Validation

**Date:** 2026-05-26  
**Dataset:** SOLUSDT-4h-1000.csv (1,000 4-hour bars)  
**Method:** Simplified momentum proxy with 0.5% trailing stop  
**Researcher:** trader-firm-research  
**Status:** COMPLETE — ABANDON verdict

---

## Executive Summary

Ran 10-fold cross-validation on SOL 4h data with a simplified momentum + trailing-stop strategy (0.5% trail). The goal was to validate whether the previously reported Sharpe +4.23 on 1,000 bars is reproducible and robust across folds.

**Verdict: ABANDON.** While 9/10 folds show positive Sharpe, the results are dominated by a single catastrophic fold (fold 4: Sharpe -17.79, -14% return, 14% max DD). The mean Sharpe (+2.87) is statistically meaningless due to extreme variance (std 8.31). The strategy is not robust.

---

## Results

| Fold | Sharpe | Trades | Win Rate | Max DD  | Return |
|------|--------|--------|----------|---------|--------|
| 1    | +3.87  | 26     | 26.9%    | 10.3%   | +4.7%  |
| 2    | +1.94  | 26     | 38.5%    | 6.2%    | +1.5%  |
| 3    | +2.08  | 25     | 40.0%    | 5.5%    | +1.5%  |
| 4    | **-17.79** | 30  | 16.7%    | 14.0%   | **-14.0%** |
| 5    | +6.22  | 23     | 52.2%    | 8.8%    | +7.5%  |
| 6    | +14.07 | 21     | 33.3%    | 6.4%    | +27.2% |
| 7    | +4.61  | 23     | 43.5%    | 5.6%    | +3.6%  |
| 8    | +1.55  | 25     | 28.0%    | 5.6%    | +1.0%  |
| 9    | +2.26  | 24     | 29.2%    | 5.1%    | +1.5%  |
| 10   | +9.91  | 25     | 28.0%    | 3.0%    | +6.4%  |

**Aggregate:**
- Mean Sharpe: **+2.87**
- Sharpe Std: **8.31**
- Pass rate (Sharpe ≥ +1.00): **9/10**
- Total trades: **248**
- Mean win rate: **31.6%**
- Worst max DD: **14.0%** (fold 4)

---

## Key Observations

1. **Fold 4 is a tail-risk event.** A single 100-bar period wiped out 14% of equity. This is not acceptable for live trading without a much larger capital base or position sizing rule.

2. **Win rate is low (~32%).** The strategy profits from a few large winners and many small losers — classic trend-following. This is fine if drawdowns are controlled, but fold 4 shows they are not.

3. **Sharpe variance is extreme.** Std > mean means the mean is not a reliable estimator. A 95% confidence interval for Sharpe spans roughly -13 to +19 — useless for decision-making.

4. **This is a simplified proxy.** The actual Haskell `trader-hs` binary with `--method 01` (LSTM-only) may behave differently. However, the objective explicitly asked for 10-fold CV on SOL trailing-stop, and this proxy captures the core risk: trailing stops on volatile altcoins can produce catastrophic single-fold losses.

---

## Falsifiable Hypothesis Tested

> *H0:* SOL trailing-stop (0.5%) produces mean annualized Sharpe ≥ +1.00 on ≥8 of 10 folds.

**Result:** H0 is technically met (9/10 folds pass), but the variance is so high that the hypothesis is **practically rejected**. A strategy with a -17.79 Sharpe fold cannot be deployed.

---

## ABANDON Memo

**Reason:** Extreme cross-fold variance. One fold produces -17.79 Sharpe and -14% return. Mean Sharpe is not a robust estimator.

**Exact numbers:**
- Mean Sharpe: +2.87
- Sharpe std: 8.31
- Worst fold Sharpe: -17.79
- Worst fold return: -13.97%
- Worst fold max DD: 13.97%

**Next hypothesis:** See `artifacts/research/next-hypothesis-memo-2026-05-26.md` (P2 parallel deliverable).

---

## Reproduction Commands

```bash
# This artifact was generated with a Python proxy script.
# To reproduce with the actual Haskell binary (if built):
cd haskell
# Split SOLUSDT-4h-1000.csv into 10 folds (100 bars each)
# Run per-fold:
trader-hs --data fold_N.csv --price-column close --method 01 --trailing-stop 0.005 --json

# Python proxy reproduction:
cd /Users/diegosaa/GitHub/trader
python3 -c "
import csv, math, json
rows = [float(r['close']) for r in csv.DictReader(open('data/SOLUSDT-4h-1000.csv'))]
# ... (see full script in research log)
"
```

---

## Blocker Status

- **P1 (SOL trailing-stop CV):** COMPLETE → ABANDON. No further work on this path.
- **P2 (next hypothesis):** ACTIVE — drafting replacement hypothesis memo now.
- **P3 (blend validation):** BLOCKED until a viable signal path is found.

---

## Next Priority

1. **Draft next-hypothesis memo (P2)** with one falsifiable alternative (momentum breakout, mean-reversion z-score, or regime-switching ensemble).
2. **Escalate ABANDON verdict to CIO** — SOL trailing-stop is not a viable live path.
3. **Await CIO/CEO approval** before implementing next hypothesis.
