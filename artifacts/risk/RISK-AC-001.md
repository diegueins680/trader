# RISK-AC-001 — Regime-Filtered TA Trend Acceptance Criteria

**Version:** 1.0  
**Owner:** trader-firm-risk  
**Date:** 2026-05-20  
**Applies to:** Regime-filtered `ta_trend` strategy promotion to production  

## Purpose
Define the minimum quantitative gates a regime-filtered technical-analysis trend strategy must clear before it is eligible for live trading or capital allocation.

## Gates (all must pass)

| # | Gate | Threshold | Rationale |
|---|------|-----------|-----------|
| 1 | **Sharpe ratio floor** | ≥ 0.00 | Must show non-negative risk-adjusted returns. Baseline SMA-cross(20/40) on BTCUSDT-4h-1000 produced Sharpe = −2.99 (fails hard). |
| 2 | **Maximum drawdown ceiling** | ≤ 0.10 (10 %) | Capital preservation limit. Baseline maxDD = 5.8 % (passes), but Sharpe failure disqualifies it anyway. |
| 3 | **Minimum trade count** | ≥ 20 on 1 000-bar sample | Statistical power: 4 trades (baseline) is insufficient to distinguish signal from noise. |
| 4 | **Phase-sensitivity gate** | \|ΔSharpe\| > 0.50 for ≥ 3 of 10 phase shifts | Validates that the regime filter is not a random overlay; it must materially alter performance across distinct market phases. |

## Pass / Fail Logic

- **PASS:** All four gates satisfy their thresholds simultaneously on the same back-test sample.
- **FAIL:** Any single gate fails → strategy is **blocked** from promotion. Research must iterate and re-submit.

## Baseline Reference

- **Instrument / timeframe:** BTCUSDT perpetual, 4 h, last 1 000 bars  
- **Unfiltered baseline:** SMA-cross(20/40)  
- **Baseline metrics:** Sharpe = −2.9918, maxDD = 0.0582, trades = 4, totalReturn = −0.0378  
- **Baseline verdict:** FAIL (gates 1 and 3 fail; gate 4 untested because gates 1 & 3 already block)

## Version History

| Version | Date | Change | Author |
|---------|------|--------|--------|
| 1.0 | 2026-05-20 | Initial criteria drafted from H1 research memo | trader-firm-risk |
