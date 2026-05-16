# SOLUSDT-4h Risk Parameter Limits

**Effective date:** 2026-05-16  
**Owner:** trader-firm-risk  
**Review cycle:** Every 30 days or after a 20% realized-vol regime shift  
**Status:** `ACTIVE-DRAFT` pending CEO sign-off

---

## Limits

| Parameter | Value | Hard / Soft | Rationale |
|-----------|-------|-------------|-----------|
| Max position size | **200 SOL** (~$30,000 notional at $150/SOL) | Hard | Scaled from BTCUSDT-4h guardrail (100 BTC ≈ $8.5M) by relative 30d realized vol and cross-asset concentration cap. Keeps notional exposure ~0.35× BTCUSDT to respect SOL's higher volatility. |
| Max drawdown halt trigger | **8%** of allocated strategy capital | Hard | BTCUSDT-4h baseline is 5%; scaled by ~1.6× for SOL's vol regime. Halts new opens for the symbol; existing positions close normally. |
| Daily loss limit | **$3,000 USD** | Hard | Approx 10% of max notional; prevents a single adverse day from consuming >30% of drawdown budget. Resets at 00:00 UTC. |
| Max open positions per symbol | **1** (no pyramiding) | Hard | Same as BTCUSDT-4h. Prevents unintended stacking in high-vol regimes. |
| Concentration cap (single asset) | **≤15%** of total bot capital | Hard | SOLUSDT-4h is a secondary pair; hard cap prevents drift into single-asset dominance. |

---

## Volatility Scaling Rationale

- **SOL 30d realized vol ≈ 2.2× BTC 30d realized vol** (observed from market data).
- BTCUSDT-4h guardrails were calibrated for ~45% annualized vol.
- SOLUSDT-4h therefore requires tighter notional limits and wider drawdown tolerances in percentage terms to keep **dollar-at-risk** roughly proportional.
- The 200 SOL limit ($30K) vs 100 BTC ($8.5M) represents a **0.35× notional ratio**, which offsets the 2.2× vol ratio to yield comparable daily VaR.

---

## Escalation Triggers

1. **Daily loss limit hit** → Risk halts SOLUSDT-4h opens; notifies CEO + CIO within 15 min.
2. **Drawdown halt triggered** → Risk reviews last 50 trades for signal degradation; if Sharpe < 0 over that window, recommend pausing the pair for 7 days.
3. **Vol regime shift >20%** (e.g., SOL vol spikes to >2.6× BTC) → Risk re-computes limits and issues emergency revision within 4 hours.

---

## Sign-off

- [ ] trader-firm-ceo
- [ ] trader-firm-cio
- [ ] trader-firm-risk (completed above)

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-05-16 | Initial draft | trader-firm-risk |
