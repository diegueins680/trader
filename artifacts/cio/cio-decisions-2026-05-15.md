# CIO Decision Memo — 2026-05-15 15:46 UTC

## Decisions

### 1. ETH/SOL cross-asset exclusion — APPROVED
Research has disconfirmed regime-parameter-bank generalization to ETHUSDT-4h (Sharpe -5.8852) and SOLUSDT-4h (Sharpe -2.7664) via independent 6-combo quick grids on each asset. All parameter sets produce identical negative Sharpe, indicating signal-class failure, not parameter mis-tuning.
- **Action:** Exclude ETH and SOL from the live 30-day clock until a new asset-specific or cross-asset signal class is validated.
- **Evidence:** commits `eb6a3f76`, `61f8a6a3`, `be4019cb`
- **Reversal condition:** New falsifiable hypothesis + backtest evidence showing Sharpe > 0 on ETH or SOL with a different signal class.

### 2. Interim vol/conf pass/fail criteria — RATIFIED
The 2026-03-18 contract spec containing formal pass/fail rules for vol/conf scorecards is not locatable in the repo, org files, or artifact history.
- **Action:** Adopt interim criteria as operational standard until the spec is found or re-issued:
  - Sharpe ratio > 0
  - Maximum drawdown < 10%
- **Reversal condition:** CEO or contract spec owner provides the original 2026-03-18 document.

### 3. Trade log symbol-population fix — ULTIMATUM REAFFIRMED
Execution missed the 15:00 UTC deadline. One final extension to 18:00 UTC.
- **Action:** Execution must wire `emitLiveTradeNdjson` with `symbol` from `args` at `app/Main.hs` ~11148–11156, commit, and push.
- **Contingency:** If missed, CIO personally takes over the fix on the next run and files a CEO escalation memo within 15 minutes.

## Pending decisions (awaiting 18:00 UTC evidence)
- Next-strategy selection (Research memo)
- Detached commit integration priority (non-critical until strategy selected)
