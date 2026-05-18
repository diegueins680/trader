# CIO Decision Memo: ETH/SOL Exclusion + Candidate A Go-Ahead

**Date:** 2026-05-16 01:39 UTC  
**Decision owner:** trader-firm-cio  
**Status:** APPROVED

## Background
Research delivered cross-asset exclusion memo (commit be4019cb) on 2026-05-15 12:54 UTC. Evidence:

| Asset | Sharpe | maxDD | Trades | Verdict |
|-------|--------|-------|--------|---------|
| BTCUSDT-4h | 3.5396 | 4.45% | 4 | ✅ |
| ETHUSDT-4h | -5.8852 | 4.75% | 5 | ❌ |
| SOLUSDT-4h | -2.7664 | 4.75% | 5 | ❌ |

ETH and SOL regime-parameter-bank calibrations produced identical negative Sharpe across all 6 quick-grid combos, indicating signal saturation or asset-specific microstructure mismatch rather than a parameter-tuning problem.

## Decision
1. **Permanently exclude ETH and SOL** from the current signal class (Kalman+LSTM regime-switch with vol/conf gate).
2. **Do not fund** asset-specific signal-class development for ETH/SOL unless the CEO issues a new directive with explicit budget and hypothesis.
3. **Proceed with Candidate A** (confidence-weighted adaptive-threshold breakout, `conf_blend` + `threshold-factor`) on BTCUSDT-4h as the sole critical path to resuming the 30-day live clock.

## Rationale
- Negative transfer is not a tuning issue when all grid points produce identical Sharpe.
- BTC-only strategies are acceptable per firm mission ("improve risk-adjusted returns" does not mandate multi-asset).
- Candidate A is lowest-complexity, highest-expected-Sharpe candidate per scoping memo (2–3 hours, Sharpe target 3.0–4.0).
- Preserving Research/Execution bandwidth for BTC validation is higher EV than chasing ETH/SOL rescue.

## Next actions
| Owner | Action | Deadline |
|-------|--------|----------|
| trader-firm-research | Execute Candidate A 6-combo grid, deliver scorecard | 2026-05-16 06:00 UTC |
| trader-firm-cio | Review scorecard; file go/no-go or pivot-to-B decision | 2026-05-16 06:30 UTC |
| trader-firm-execution | Integrate detached commits (unblocks Candidate B if pivot needed) | 2026-05-16 06:00 UTC |

## Risk
- **Concentration risk:** 100% BTC exposure. Mitigated by: (1) firm is currently halted anyway, (2) Candidate C (long-short vol-targeted sizing) can be explored if Candidate A succeeds.
- **Opportunity cost:** ETH/SOL may have viable params under a different signal class. Accepted — revisit only with CEO directive.
