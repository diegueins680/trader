# Intrabar protective-exit re-entry lock audit

Date: 2026-09-06 local / 2026-09-07 UTC

Risk: `TRAILING-STOP-001`

Disposition: fixed and closed

## Reproduction

The historical simulator makes a decision for index `t`, then marks the held
position across the interval ending at `t+1`. An intrabar stop-loss,
trailing-stop, or take-profit exit is therefore filled and recorded at
`t+1`. Before this change, a zero configured cooldown left the next fold step
eligible to open immediately; that step also used `t+1` as its entry index.

The former trailing-stop-only regression was generalized as
`testProtectiveExitSameEventReentryLock`. It exercises deterministic stop-loss,
trailing-stop, and full take-profit fixtures with zero configured cooldown. The
first trailing-stop test-only run failed with:

```text
Assertion failed: trailing-stop exit and fresh entry never share an event index
```

That is the exact registered failure mode, not a theoretical concern.

## Fix

The simulator now identifies the newly emitted closed trade before calculating
the next cooldown. Stop-loss, trailing-stop, and full take-profit exit reasons
impose a minimum cooldown of one event. The effective cooldown is the maximum
of this protective minimum, the existing maximum-hold minimum, and the
user-configured cooldown, so a longer operator setting is never weakened.

Signal exits are filled at the current decision index rather than the next
intrabar index and do not need the extra event. Persistent risk halts and
liquidation already prevent re-entry through their own state.

## Regression contract

The same fixture must now demonstrate all of the following:

1. A trailing-stop exit actually occurs.
2. The continuing signal remains eligible to enter later.
3. No fresh entry shares the protective exit index.
4. With configured cooldown zero, the earliest entry is exactly one index after
   the exit index.

The failing pre-fix run and passing post-fix run use the same fixture and
assertions. No market data, parameter search, development sample, or final
holdout is involved.

## Scope and safety

This changes deterministic historical-simulation timing and may remove
optimistic same-event churn and its associated return/cost accounting. It does
not claim improved returns, alter signals, weaken costs, change saved model or
configuration semantics, touch deployed environment values, authorize live
orders, or place an order. Live exchange-native protective-order handling is a
separate execution path and is unchanged.

Reproduce from the repository root:

```bash
bash scripts/verify.sh haskell
bash scripts/verify.sh full
```
