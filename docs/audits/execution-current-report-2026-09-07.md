# Current execution report — 2026-09-07

## Disposition of `EXECUTION-MISSING-001`

Close the reporting-lane risk. The 2026-05-15 and 2026-05-27 trade-log
deadlines were missed and remain historical failures; this report does not
rewrite them as timely. It satisfies the outstanding requirement to obtain a
current execution-owner report and explicitly disposes of that operational
obligation without claiming that every execution risk is closed.

The owner remains `trader-firm-execution`; reassignment is not required. This
closeout changes no order path, risk threshold, model, promotion state,
deployment setting, credential, or live authorization.

## Original obligation and current baseline

The 2026-05-15 CIO memo required the live trade-log writer to populate its
symbol. The 2026-05-26 run plan then set a 2026-05-27 hard deadline for the
`--trade-log` deliverable and required a recorded failure if it was missed.
The canonical risk register later preserved the lapse as
`EXECUTION-MISSING-001` with one close condition: obtain a current execution
report and explicitly close or reassign the obligation.

This report audits merged `main` at
`365715ee22a9e0ab1ee6ed7b9dcaf7b265b47437`. It relies on repository-owned
deterministic evidence; it does not use an exchange account, credentials,
orders, or a live runtime witness.

## Current execution evidence

| Concern | Current behavior | Evidence | Status at this boundary |
|---|---|---|---|
| Trade-log interface | `--trade-log FILE` appends NDJSON with a populated symbol. Backtest rows retain schema 1.1; applied live OPEN/CLOSE markers use additive schema 1.2. | `haskell/app/Main.hs`; `haskell/app/Trader/TradeLogRiskState.hs`; `artifacts/cio/trade-log-schema-contract-2026-05-24.md` | Implemented and regression-bound |
| Fill admission | An execution transition requires sent/live/status evidence and positive finite executed quantity, or the documented requested-quantity fallback. Explicit partial-fill evidence takes precedence over terminal canceled/expired status. | `haskell/app/Trader/OrderExecution.hs`; `haskell/app/Trader/Formal/Execution.hs` | Bounded-exhaustive reference model plus regressions |
| Exposure reconciliation | Applied base quantity is capped against intended exposure; reduce-only execution cannot increase or flip exposure. | `haskell/app/Trader/OrderExecution.hs`; `verifyFormalExecution` | Bounded-exhaustive at the pure seam |
| Reversal ordering | A futures reversal confirms the reduce-only close before submitting a separately admitted entry; close and entry quantities are reconciled independently. | `placeOrderForSignalEx`; `testLiveReversalPartialEntryScenario` | Deterministic production-seam witness |
| Maker timeout and partial fill | After timeout, cancellation is followed by a final order read so a fill racing the cancel can be retained. A partially filled terminal order applies only its confirmed fraction. | `sendPostOnlyEntry`; `testReduceOnlyPartialTakeProfitTerminalCancelScenario` | Deterministic reducer coverage; network choreography not simulated |
| Ambiguous request outcome | A failed live request with a client order identifier is queried by that identifier before the result is classified. | `sendMarketOrderWithClientOrderId` | Implemented; exchange/network failure modes remain runtime-dependent |
| Venue reconciliation | Fresh live Binance futures bars compare local position sign with exchange inventory, book a disappeared local position closed, and adopt a venue position missing locally. Startup readiness stays blocked until exchange inventory is inspected and every open symbol has an owner. | `reconcileBotPositionWithExchange`; `GET /ready`; `testSnapshotRestartRestoresMemoryWithoutExposureScenario` | Implemented with fail-closed startup ownership |
| Restart state | Persisted status restores only identity-matched bounded closed-trade memory. Persisted position and open-trade fields cannot seed startup exposure. | `haskell/app/Trader/BotSnapshotRecovery.hs`; P0 scenario matrix | Deterministic regression-bound |
| Decision-time risk observability | Schema-1.2 live event markers carry the exact pre-execution drawdown, daily/weekly loss, expectancy availability, halt state, and separate market/processing timestamps. Invalid numeric evidence is null and explicitly invalid. | `haskell/app/Trader/TradeLogRiskState.hs`; trade-log risk-state audit | Implemented and compatibility-tested |

The four named P0 composition fixtures and the canonical drawdown-flatten
direction are catalogued in
`docs/audits/p0-execution-halt-scenario-matrix.md`. The schema-1.2 compatibility,
finite-value, missing-expectancy, and close-cause boundary is catalogued in
`docs/audits/trade-log-risk-snapshot-2026-09-07.md`.

## Limits that this closeout does not erase

- A live trade-log row is an applied state-transition marker, not exchange
  accounting or fill proof. Its legacy quantity, P&L, and fee values remain
  zero-valued compatibility fields. Exchange income and account-trade evidence
  remain authoritative for realized economics.
- The deterministic scenario matrix exercises production pure seams and
  integration wiring, not actual exchange/network choreography. Partial fills,
  cancellations, retries, and reconciliation still require monitoring and
  venue evidence in operation.
- An IO-level fixture does not yet span every path from computed signal through
  order submission, replacement fill reconciliation, action labels, and final
  bot state. The narrower open/hold/flip/halt gaps remain listed in
  `docs/audits/strategy-decision-flow-spec.md`.
- A transient per-bar exchange-position query failure preserves local state; it
  is not evidence that local and venue state agree. Startup readiness has a
  stronger fail-closed inventory/ownership boundary.
- Backtest trade rows do not contain exact intrabar daily/weekly risk snapshots
  and remain schema 1.1. No approximate state is manufactured from close-only
  records.
- No historical log was upgraded, and consumers must explicitly accept live
  schema 1.2.
- This evidence is not a micro-live trial, a production-readiness attestation,
  a strategy-performance result, or permission to enable live trading.

These limitations are not a reason to retain a risk whose stated subject is a
missing owner report. They are bounded separately by `H-EXECUTION`, the
strategy decision-flow gap audit, runtime reconciliation and readiness
controls, and the pre-live assurance checklist. Any future claim about
exchange-level reliability must present new runtime evidence rather than cite
this administrative closeout.

## Verification and reproducibility

The merged implementation baseline passed `bash scripts/verify.sh full` before
this report was written. The exact report branch then passed the same canonical
wrapper on 2026-09-07, including the Haskell suite, 241 web tests and the web
production build, deployment-config validation, 161 root automation/research
tests, and a formal registry with 38 specifications, 324 named features, 246
clauses, 262 implementation files, 104 evidence links, and 32 canonical risks.

From the repository root:

```sh
bash scripts/verify.sh haskell
npm run test:formal
bash scripts/verify.sh full
```

## Decision record

- `EXECUTION-MISSING-001`: **CLOSED**.
- Reason: the current owner report now documents the delivered trade-log
  contract, fill and state-reconciliation boundary, deterministic evidence,
  and unresolved limitations, and explicitly disposes of the overdue report.
- Historical deadline outcome: **MISSED**, retained as fact.
- Owner: `trader-firm-execution`, unchanged.
- Production behavior and live authorization: unchanged.
