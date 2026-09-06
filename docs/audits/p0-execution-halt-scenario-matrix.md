# P0 execution and halt/exit scenario-proof matrix

Date: 2026-09-06  
Owners: Execution primary, Risk secondary  
Scope: issue #104 closeout at deterministic production seams

## Decision

The four named execution/restart scenarios pass deterministic regression
coverage against the reducers used by startup and the live bot. Each resulting
position is also passed to the canonical `liveRiskHaltAction` /
`specRiskHalt` path and produces the required flat target and close direction.

These tests do not contact an exchange or authorize an order. They prove state
reconciliation at the production pure seams and bind those seams to the IO
paths below. Venue availability, fill-report latency, and end-to-end exchange
sandbox behavior remain operational risks rather than claims made by this
matrix.

## Pass/fail matrix

| Scenario | Deterministic fixture and result | Exact reducer | Startup/live path | Persisted status/snapshot evidence | Halt/exit result | Status |
|---|---|---|---|---|---|---|
| Startup canceled-after-partial preserved | `testStartupCanceledAfterPartialExecutionScenario`: a live `CANCELED` order reports `0.8 / 2.0` base filled against `0.5` intended exposure; only `0.2` exposure is opened | `orderAppliedFraction` then `applyExecutedQuantity` | `Main.initBotState`, local `executedFractionFromOrder`, and startup `(desiredPos, desiredSize, closeQty, openQty)` reconciliation | `botStatusJson` persists `positions`, `orders`, `trades`, and `openTrade`; each `BotOrderEvent` retains the `ApiOrderResult` terminal status and executed quantity | A drawdown halt targets flat and emits sell direction `-1` for the resulting long | PASS |
| Live reversal with partial entry fill | `testLiveReversalPartialEntryScenario`: confirmed `0.6` long close plus a `CANCELED` entry with `0.1 / 0.4` filled produces only a `0.1` short | `orderAppliedFraction` for each leg then `applySplitReversalExecutedQuantities` | `placeOrderForSignalEx.sendFuturesTargetEntry` confirms the reduce-only close before entry; `Main.botApplyKline` reads `aorPrecedingClose` and reconciles both legs independently | Combined `ApiOrderResult` retains `aorPrecedingClose`; `BotOrderEvent`, `resultingOpenTrade`, `position`, and status snapshot preserve the observed result | A drawdown halt targets flat and emits buy direction `1` for the resulting short | PASS |
| Reduce-only partial take-profit followed by terminal cancel | `testReduceOnlyPartialTakeProfitTerminalCancelScenario`: a live `CANCELED` reduce-only order reports `0.2 / 0.5` filled; long exposure decreases from `1.0` to `0.8` with no open quantity | `orderAppliedFraction` then `applyReduceOnlyExecutedQuantity` | `Main.botApplyKline` `partialExitWanted` branch | Partial order journal/ops evidence records `partialSize`, `remainingSize`, prior `openTrade`, `resultingOpenTrade`, and raw terminal order evidence; the status snapshot persists the resulting order and position history | A drawdown halt targets flat and emits sell direction `-1` for the remaining long | PASS |
| Snapshot restart restores memory without phantom exposure | `testSnapshotRestartRestoresMemoryWithoutExposureScenario`: snapshots differing only in `positions` and `openTrade` claims restore identical closed-trade memory; a venue-flat restart yields no close order | `restoreTradeMemoryFromStatus` in `Trader.BotSnapshotRecovery` | `Main.initBotState` obtains `startPos0` from `fetchBotAccountPos` when trading is enabled (or zero in paper mode), then separately restores bounded closed-trade memory | `writeBotStatusSnapshot` atomically persists the full status, but `TradeMemorySnapshotContext` admits only identity plus a bounded `trades` array; exposure fields are outside the recovery contract | A drawdown halt while venue-flat remains flat and emits no order direction | PASS |

## Production-path linkage

- Both startup and live order paths build `OrderExecutionEvidence` from
  `aorSent`, live mode, terminal status, and `aorExecutedQty`, then call the
  same `orderAppliedFraction` implementation.
- Reversal placement refuses to send the opposite entry until the venue
  confirms the reduce-only close is flat. The confirmed close and entry fill
  remain separate in `aorPrecedingClose`, so a partial entry cannot be promoted
  to the intended size.
- The partial-take-profit branch always calls the reduce-only reducer. That
  reducer returns zero open quantity and cannot increase or flip exposure.
- Snapshot recovery is isolated in `Trader.BotSnapshotRecovery`. Its public
  context contains no position field. `Main.initBotState` remains the only
  startup exposure authority and queries the venue before restoring adaptive
  memory.
- `testMaxDrawdownHaltsSimulation` remains the integrated simulator witness
  that drawdown closes the trade with `ExitMaxDrawdown`; every new execution
  scenario additionally checks the live close direction through the same
  canonical halt reason.

## Risk sign-off

### Current-main green-watch

Before this closeout, code inspection and existing tests supported a
green-watch assessment: fill evidence was reconciled through shared reducers,
risk halts used `specRiskHalt`, and the live order path sequenced reversal close
before entry. That status was not sufficient for P0 acceptance because the four
named compositions were not deterministic fixtures and restart recovery was
embedded in the large application module.

### P0 closeout readiness

The named matrix is ready for P0 closeout. Both the canonical Haskell gate and
the full repository gate passed locally on 2026-09-06. The proof boundary is
explicit:

- deterministic reducers, snapshot-memory isolation, and halt direction are
  covered;
- snapshot exposure does not authorize or construct startup exposure;
- no live-trading flag, credential, deployment configuration, or promotion
  state changes in this closeout;
- exchange/network choreography is not simulated, so current monitoring,
  reconciliation, human authorization, and rollback controls remain required.

`EXECUTION-RESTART-001` records the phantom-exposure risk as closed at this
deterministic boundary. It does not close unrelated open risks such as
`EXECUTION-MISSING-001`, `TRADE-LOG-GAP-002`, or `TRAILING-STOP-001`.

## Reproduction

From the repository root:

```bash
bash scripts/verify.sh haskell
bash scripts/verify.sh full
```

The Haskell suite executes the four exact fixtures named in the matrix and the
existing integrated drawdown fixture. The formal verifier binds the new
snapshot-recovery module and all four regression names to `H-EXECUTION`.

Recorded result:

- `bash scripts/verify.sh haskell` — PASS.
- `bash scripts/verify.sh full` — PASS: Haskell suite, 241 web tests and web
  production build, deployment-config validation, formal registry (38 specs,
  321 named features, 237 clauses, 261 implementation files, 92 evidence
  links, 32 canonical risks), and 154 automation tests.
