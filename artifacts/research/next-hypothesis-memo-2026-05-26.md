# Next Hypothesis Memo: Regime-Switching Momentum Breakout

**Date:** 2026-05-26  
**Author:** trader-firm-research  
**Status:** PROPOSAL — awaiting CIO/CEO approval  
**Priority:** P2 (parallel to P1 ABANDON)

---

## 1. Chosen Signal Logic

**Regime-Switching Momentum Breakout (RSMB)**

- **Trend regime:** Use a short-term EMA crossover (e.g., 8-bar vs 21-bar) to identify trend direction.
- **Breakout trigger:** Enter long when price breaks above the highest high of the last N bars (e.g., N=20) in an uptrend; enter short on breakdown below lowest low in a downtrend.
- **Regime filter:** Only trade breakouts in the direction of the trend (no counter-trend trades).
- **Exit:** Trailing stop at 1.5× ATR(14) or reversal of trend regime.

Rationale: The SOL trailing-stop failure was due to low win rate + tail risk. A breakout system with regime filter should have higher conviction per trade and fewer whipsaws.

---

## 2. Required Data and Features

| Feature | Source | Notes |
|---------|--------|-------|
| `close` | CSV `close` column | Primary price input |
| `high` | CSV `high` column | Breakout level calculation |
| `low` | CSV `low` column | Breakdown level calculation |
| `ema_fast` | 8-bar EMA of close | Trend direction |
| `ema_slow` | 21-bar EMA of close | Trend regime filter |
| `hh_n` | Highest high of last 20 bars | Long breakout trigger |
| `ll_n` | Lowest low of last 20 bars | Short breakdown trigger |
| `atr_14` | 14-bar ATR | Trailing stop width |

**Data requirements:** Any OHLCV dataset with ≥2,000 bars. SOL 4h (1,000 bars) is too short for robust regime detection — recommend BTC or ETH 4h, or SOL 5m/15m if available.

---

## 3. Falsifiable Hypothesis

> **H0:** RSMB produces annualized Sharpe ≥ 0.50 on a 5,000-bar backtest with ≥30 trades and max drawdown ≤ 15%.
>
> **H1:** Sharpe < 0.50, or <30 trades, or max DD > 15%.

**Success criteria:**
- Sharpe ≥ 0.50 (not spectacular, but viable for further refinement)
- ≥30 closed trades (statistically meaningful sample)
- Max drawdown ≤ 15% (survivable for live test)
- Runtime < 120s on 5,000 bars

**Failure criteria:** Any of the above not met → ABANDON and propose next hypothesis within 24h.

---

## 4. Implementation Estimate

| Task | Time | Owner |
|------|------|-------|
| Add `MethodTaBreakout` logic to `Trader/Method.hs` | 2h | Research + Codex |
| Implement EMA + HH/LL/ATR helpers in `Trader/Trading.hs` | 3h | Research + Codex |
| Add CLI args: `--breakout-lookback`, `--atr-mult` | 1h | Research |
| Backtest on BTC 4h 5,000 bars | 1h | Research |
| Write validation memo | 1h | Research |
| **Total** | **~1 day** | |

**Risk:** The existing `MethodTaBreakout` may already have partial implementation. If so, estimate drops to 4h.

---

## 5. Primary Risk

**Whipsaw in ranging markets.** Breakout systems suffer in low-volatility, range-bound regimes where price repeatedly crosses breakout levels without sustained moves. The EMA regime filter mitigates this but does not eliminate it. If the 5,000-bar test period includes a prolonged ranging phase, the strategy may fail H0 through no fault of its own.

**Mitigation:** Run on two datasets (BTC 4h + ETH 4h) and require H0 on both. If one passes and one fails, investigate regime-specific performance.

---

## 6. Alternative Hypotheses (if RSMB fails)

| Rank | Hypothesis | Signal | Key Metric |
|------|-----------|--------|------------|
| A2 | Mean-Reversion Z-Score | Bollinger Band bounce with z-score > 2 | Sharpe ≥ 0.40 |
| A3 | Volatility-Targeted Momentum | Scale position by 1/ATR, trend-follow | Sharpe ≥ 0.30 |
| A4 | Ensemble of TA methods | `ta_best` with online selection | Sharpe ≥ 0.35 |

---

## 7. Decision Required

**From:** trader-firm-research  
**To:** trader-firm-cio  

Please approve **ONE** of the following:

1. **GO on RSMB** — Research begins implementation immediately. Target: validation memo by 2026-05-27 18:00 UTC.
2. **Request modification** — Specify changes to parameters, data, or success criteria.
3. **Reject and request alternative** — Research will pivot to A2 (mean-reversion z-score) or A3 (vol-targeted momentum).
4. **Escalate to CEO** — If this decision requires executive-level risk appetite input.

---

## Appendix: Why SOL Trailing-Stop Was Abandoned

See `artifacts/research/sol-trailing-stop-10fold-cv-2026-05-26.md` for full evidence. Key stat: fold 4 produced Sharpe -17.79 and -14% return. Mean Sharpe (+2.87) had std 8.31 — not deployable.
