# ETHUSDT Regime Disposition Memo

Date: 2026-05-14 20:30 UTC  
Author: trader-firm-research  
Status: P1 CRITICAL — requires CIO decision before 2026-05-15 06:00 UTC

## Observation
ETHUSDT-4h (1000 bars) produces negative Sharpe under every tested regime configuration:

| config | Sharpe | maxDD | winRate | trades |
|--------|--------|-------|---------|--------|
| baseline (no regime dampening) | -5.8852 | 4.75% | 40% | 5 |
| BTC-transferred params (0.20/0.70/0.45) | -12.1944 | 1.43% | 0% | 2 |
| ETH-local best (0.40/0.50/0.50) | -2.9504 | 4.27% | 50% | 4 |

## Interpretation
Per-asset regime calibration provides a +2.93 Sharpe improvement over baseline and a +9.24 improvement over BTC-transferred params, but ETHUSDT-4h remains unprofitable under trend-following with regime dampening. This suggests the issue is not parametric (wrong weights/thresholds) but structural: ETHUSDT-4h may be inherently mean-reverting or noise-dominated on this slice, making trend-following a mismatched strategy regardless of regime gating.

## Recommendation
**Adopt per-asset regime banks for BTCUSDT and SOLUSDT immediately; exclude ETHUSDT from Candidate 2 trend-following scope pending a mean-reversion hypothesis test.** The next disconfirmation experiment is to run the same ETHUSDT-4h slice with `--method ta_mean_reversion` and the top ETH-local regime params to see if mean-reversion flips Sharpe positive. If mean-reversion also fails (Sharpe < 0), the firm should classify ETHUSDT-4h as "untradeable with current signal suite" and redirect Research bandwidth to alternate timeframes (1h or 1d) or alternate assets.

## Next exact packet
- **Hypothesis:** ETHUSDT-4h mean-reversion with regime dampening achieves Sharpe > 0.
- **Falsification threshold:** Sharpe ≤ 0 after 12-point quick grid.
- **Owner:** trader-firm-research
- **Deadline:** 2026-05-15 06:00 UTC
- **Reproduction:** `python3 scripts/calibrate-regime-params.py --data data/ETHUSDT-4h-1000.csv --quick --method ta_mean_reversion`
