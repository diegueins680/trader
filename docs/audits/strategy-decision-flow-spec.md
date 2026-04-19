# Strategy decision-flow spec

Date: 2026-04-19
Owner: Trader Firm CTO
Scope: P0 audit item 1 — strategy decision-path invariants
Sources audited:
- `haskell/app/Main.hs`
- `haskell/app/Trader/Trading.hs`
- `haskell/app/Trader/SignalGates.hs`
- `haskell/app/Trader/VolConfGate.hs`
- `haskell/test/TestMain.hs`

## Audit status

This artifact now gives the repo one canonical written decision-flow spec, but **implementation parity is not yet fully achieved**.

Current state:
- **Latest-signal** has the richest stateless prediction/gate/sizing path.
- **Live bot** mostly wraps latest-signal, then adds stateful hold / cooldown / exposure / halt / execution behavior.
- **Backtest** re-implements a similar but not identical path inside `Trader.Trading`.
- Existing tests cover several **helper invariants**, but not the full **blocked-entry / hold / flip / halt** path matrix end-to-end.

## Canonical intended flow

This is the canonical sequence the repo should converge on for all three surfaces (`latest-signal`, live bot, backtest):

1. **Prediction inputs**
   - current price / bar context
   - Kalman prediction + optional meta/confidence
   - LSTM prediction + optional confidence proxy
   - regime / conformal / quantile / volatility context

2. **Raw directional proposals**
   - derive open-direction from `openThreshold`
   - derive close-direction from `closeThreshold`
   - method-specific chooser builds one candidate open direction and one candidate close direction

3. **Entry-only edge sanity gates**
   - `EDGE_SPIKE`
   - `EDGE_HEADROOM`
   - later, after provisional sizing, `EDGE_FEE_BUFFER`

4. **Post-direction structural gates**
   - volatility gate
   - vol-target-ready gate
   - trend gate
   - cloud gate
   - price-action gate
   - signal-to-noise gate
   - non-directionality veto
   - regime-edge gate
   - MTF consensus gate
   - cross-asset gate
   - meta-label gate
   - funding/OI gate

5. **Side eligibility / positioning gate**
   - long-flat vs long-short positioning semantics
   - optional pairs overlay / method-specific overlays
   - side removed if no valid trade size exists

6. **Sizing pipeline**
   - base size
   - confidence / vol-conf multiplier
   - volatility targeting
   - signal-to-noise scaling
   - risk-per-trade scaling
   - regime / pairs / funding / Kelly overlays
   - cap to max size
   - floor by min size

7. **Vol-confidence behavior application**
   - `AllowEntry`, `Hold`, `Block`, or `AllowExitOnly`
   - stateful close behavior must preserve reduce-only semantics

8. **Execution intent**
   - hold current side
   - open from flat
   - close to flat
   - flip long<->short
   - rebalance same side

9. **Exit ladder**
   - signal exit / close-direction exit
   - bracket exit (`STOP_LOSS`, `TRAILING_STOP`, `TAKE_PROFIT`)
   - flip exits (`LSTM_FLIP`, Kalman exits)
   - `MAX_HOLD`
   - risk halts (`MAX_DRAWDOWN`, `MAX_DAILY_LOSS`, `MAX_WEEKLY_LOSS`, `NEGATIVE_EXPECTANCY`)
   - live-only order/auth halts (`MAX_ORDER_ERRORS`, `BINANCE_AUTH_INVALID`)

10. **State persistence**
   - latest signal / gate reason / size
   - orders / order evidence
   - open trade / closed trade
   - halt state / cooldown / exposure counters

## Actual current flow by surface

## 1) `latest-signal` flow (`Main.computeLatestSignal`)

### Prediction -> direction
- Build raw Kalman/LSTM open and close directions.
- For meta-aware methods, apply Kalman/meta gating before the method chooser.
- Choose `chosenDirBase` and `closeDirBase` from the requested method.

### Entry-only edge gates
- `signalEntryHeadroomOk`
- `signalEntryEdgeSpikeOk`
- If either fails, open direction is dropped before later structural gates.

### Post-direction structural gates
- `signalRunPostDirectionGates` applies, in order:
  1. `VOLATILITY`
  2. `VOL_TARGET`
  3. `TREND`
  4. `CLOUD`
  5. `PRICE_ACTION`
  6. `SIGNAL_TO_NOISE`
  7. `NON_DIRECTIONAL_*`
  8. `REGIME_EDGE`
  9. `MTF_CONSENSUS`
  10. `CROSS_ASSET`
  11. `META_LABEL`
  12. `FUNDING_OI`
- This is the clearest current statement of post-direction gate order in the repo.

### Sizing
- `tradePosSize` / method size is required for non-LSTM-only methods.
- `volConfGateCell` computes behavior + size multiplier.
- Additional size overlays: volatility targeting, SNR weighting, risk scale, regime size, pairs overlay, funding/OI size scale, Kelly-lite.
- Enforce `maxPositionSize`, then `minPositionSize`.

### Vol-conf + fee-buffer finalization
- `applyVolConfGateBehavior` is applied to the provisional side/size.
- `signalEntryFeeBufferOk` is checked **after** provisional sizing.
- Final side can still be dropped because of `EDGE_FEE_BUFFER` or `MIN_SIZE`.

### Output semantics
- Produces:
  - `lsChosenDir`
  - `lsCloseDir`
  - action string such as `LONG`, `SHORT`, or `HOLD (...)`
  - gate reason surface via hold reason text
- `latest-signal` is still mostly **stateless**: it does not own min-hold, cooldown, exposure caps, order failure halts, or trade persistence.

## 2) Live bot flow (`Main` stateful bot path)

### Upstream dependency
- Live bot first calls `computeLatestSignal`.
- So live bot inherits the stateless gate order above.

### Stateful direction mapping
- Converts `lsChosenDir` + `lsCloseDir` into `desiredPosSignal`.
- If `VOL_CONF_GATE_HOLD` is active while already in position, live bot preserves current position.
- If a close is required, live bot may force `lsChosenDir` to the exit side so an actual closing order can be placed.

### Exit / halt / hold ladder added by live bot
Live adds a second stateful layer after latest-signal:
1. bracket/partial-take-profit exits
2. halt flattening (`HALTED_*` / `EXIT_*`)
3. `HOLD_MIN_HOLD`
4. `EXIT_MAX_HOLD`
5. open-position / per-base / exposure blockers
6. no-trade-window / max-trades-per-day / perf-gate blockers
7. cooldown blocker
8. order placement and execution reconciliation
9. auth/order-error halts

### Execution
- Decides whether to close only, open only, or flip.
- Applies real order evidence to reconcile desired vs executed state.
- Persists `BotOrderEvent` with directionality snapshot from latest-signal.

### Key live conclusion
- Live bot is **not** a second independent prediction engine.
- It is a **stateful reducer around latest-signal**.
- Therefore latest-signal is the current best source of truth for pre-execution gate order, but not for execution/halt order.

## 3) Backtest flow (`Trader.Trading.simulateEnsemble...`)

### Prediction -> direction
- Builds raw Kalman/LSTM open and close directions.
- If meta is present, `gateKalmanDir` applies confidence/width/high-vol gating.
- For `MethodBoth`-style semantics it requires directional agreement for fresh entries.
- Uses close-direction to keep the position when open-direction is neutral.

### Entry-side gates
Backtest currently applies these entry-side conditions inline:
- trend gate
- volatility gate
- signal-to-noise gate
- vol-target-ready gate
- tri-layer (`cloud` + `priceAction`) gate
- `signalEntryEdgeSpikeOk`
- `signalEntryHeadroomOk`

### Missing vs latest-signal
Backtest **does not currently route through** `signalRunPostDirectionGates`, and therefore has no shared pure ordering for:
- `NON_DIRECTIONAL_*`
- `REGIME_EDGE`
- `MTF_CONSENSUS`
- `CROSS_ASSET`
- `META_LABEL`
- `FUNDING_OI`
- pairs overlay

### Sizing
- Applies LSTM confidence sizing (when enabled and vol-conf disabled)
- volatility targeting
- risk-per-trade sizing
- SNR weighting
- Kelly-lite sizing
- vol-conf size multiplier
- max/min size rules
- fee-buffer check after provisional sizing

### Vol-conf behavior
- Uses `volConfGateCell` and `applyVolConfGateBehavior`.
- Unlike latest-signal/live, backtest does **not** use `volConfStatefulCloseDirection`; it mutates desired side/size directly.

### Stateful path after entry gating
Backtest then layers:
- min-hold forced hold
- no-trade-window / max-trades-per-day / perf gate
- `MAX_HOLD`
- halted flattening
- rebalance
- bracket exits
- LSTM flip / Kalman exits
- funding application
- post-bar risk halt re-check

### Key backtest conclusion
- Backtest has broad overlap with live/latest-signal, but it is **not yet the same documented gate order**.
- It is a parallel implementation, not a shared reducer.

## Canonical map: prediction -> gate -> sizing -> execution -> exit

| Stage | Canonical rule | Latest-signal | Live bot | Backtest |
|---|---|---|---|---|
| Prediction | Build raw open/close dirs from models | Yes | Delegates to latest-signal | Yes |
| Method chooser | Pick method-specific open/close dir | Yes | Delegates | Yes |
| Entry spike/headroom | Entry-only gate before later structural gates | Yes | Delegates | Yes |
| Post-direction gates | Single shared ordered reducer | **Yes** | Delegates | **No shared reducer** |
| Non-directional veto | Shared structural gate | Yes | Delegates | **Missing** |
| Vol-conf close semantics | Shared stateful close behavior | Yes (`volConfStatefulCloseDirection`) | Yes via latest-signal output | **Not shared** |
| Fee buffer after provisional size | Required | Yes | Inherited from latest-signal size/action | Yes |
| Min-hold | Stateful wrapper after direction selection | No | Yes | Yes |
| Cooldown | Stateful wrapper | No | Yes | Yes |
| Exposure/open-position blockers | Stateful portfolio wrapper | No | Yes | Partial / different scope |
| Bracket exits | Exit ladder | No | Yes | Yes |
| Flip exits | Exit ladder | No | Partial via desired position + order path | Yes (`LSTM_FLIP`, Kalman exits) |
| Risk halts | Flatten + reason | No | Yes | Yes |
| Order/auth halts | Execution-only halt | No | Yes | No |

## Parity notes

## Latest-signal vs live

**Good parity**
- Live bot uses latest-signal as its upstream stateless decision source.
- The documented pre-execution gate order is therefore mostly determined by `computeLatestSignal`.
- Vol-conf hold semantics are visibly carried into live decision reduction.

**Non-parity / extra live behavior**
- live adds min-hold, cooldown, exposure, open-position limits, perf gates, order halts, and execution reconciliation.
- That is acceptable if documented as a second stage, but it is not currently encoded as one shared reducer function.

## Latest-signal vs backtest

**Shared pieces**
- open/close-threshold direction concept
- edge spike and edge headroom gates
- fee-buffer after provisional size
- vol-conf cell/behavior concepts
- max-hold / risk-halt concepts exist in both worlds

**Major parity gaps**
- backtest does not use `signalRunPostDirectionGates`
- backtest does not apply the new `NON_DIRECTIONAL_*` veto path from `SignalGates`
- backtest does not share `volConfStatefulCloseDirection`
- latest-signal has richer gate-reason surface than backtest
- live uses latest-signal + stateful wrapper, while backtest owns a separate inlined state machine

## Backtest vs live

**Shared stateful concepts**
- min-hold
- max-hold
- no-trade windows
- max-trades-per-day
- perf gates
- risk halts
- bracket exits

**Important differences**
- live has real execution and order failure halts; backtest does not
- live exposure blockers are cross-bot / cross-symbol; backtest is single-path simulation
- backtest includes LSTM/Kalman exit mechanics inline; live relies more on latest-signal output plus order-state logic

## Existing test coverage inventory

### Already covered well enough at helper level
- `EDGE_SPIKE`, `EDGE_HEADROOM`, `EDGE_FEE_BUFFER` helper invariants
- normalized entry-edge fail-closed behavior
- directionality helper semantics (`NON_DIRECTIONAL_CHOP`, weak-band, malformed)
- vol-confidence cell classification and reduce-only behavior
- backtest fee-buffer regression
- cost attribution consistency

### What is not yet covered end-to-end
The repo does **not** currently prove the full decision path through integrated latest-signal/live/backtest tests.

## Test gap list

## A. Blocked-entry path gaps

1. **No end-to-end latest-signal blocked-entry matrix**
   - Missing tests that call `computeLatestSignal` and assert final `lsAction` / `lsChosenDir` reason precedence for:
     - `EDGE_SPIKE`
     - `EDGE_HEADROOM`
     - `NON_DIRECTIONAL_*`
     - `VOL_CONF_GATE_BLOCK`
     - `VOL_CONF_GATE_ALLOW_EXIT_ONLY`
     - `MIN_SIZE`
     - `EDGE_FEE_BUFFER`

2. **No parity fixture between latest-signal and backtest on the same blocked entry**
   - Especially missing for `NON_DIRECTIONAL_*`, because backtest currently has no shared implementation.

3. **No live blocked-entry reducer tests**
   - Missing stateful tests for:
     - `MAX_OPEN_POSITIONS`
     - `MAX_OPEN_PER_BASE`
     - `MAX_GROSS_EXPOSURE`
     - `MAX_NET_EXPOSURE`
     - `MAX_EXPOSURE_PER_BASE`
     - `NO_TRADE_WINDOW`
     - `MAX_TRADES_PER_DAY`

4. **No canonical reason-order test**
   - `signalRunPostDirectionGates` has a defined first-failure order, but there is no regression asserting the same reason order at the latest-signal action surface.

## B. Hold path gaps

1. **No integrated `HOLD_MIN_HOLD` test**
   - Live logic has explicit min-hold state handling, but test coverage is missing.

2. **No integrated vol-conf hold-path test**
   - Helper tests cover `VolConfGateHold`, but not the actual latest-signal -> live-bot path where current position is preserved.

3. **No integrated cooldown hold test**
   - Missing live test for `HOLD_COOLDOWN` from flat.

4. **No integrated perf-gate hold test**
   - Missing proof that non-flip perf gating holds current exposure while flip attempts exit.

5. **No backtest hold-path matrix**
   - Missing targeted backtest tests for:
     - min-hold forced hold
     - no-trade-window hold
     - max-trades-per-day hold
     - perf-gate hold

## C. Flip path gaps

1. **No explicit live long->short / short->long flip test**
   - Missing end-to-end stateful tests for flip intent, closing quantity, reopening quantity, and action labeling.

2. **No explicit backtest flip regression**
   - Missing tests where a held long flips to short (or vice versa) under normal signal conditions.

3. **No tests for flip-specific forced exits**
   - Missing backtest tests for:
     - `LSTM_FLIP`
     - `KALMAN_SLOW`
     - `KALMAN_BAND`

4. **No parity test for flip semantics**
   - There is no fixture proving latest-signal close/open intent, live execution intent, and backtest position transition agree on the same scenario.

## D. Halt path gaps

1. **No direct backtest regression for halt exit reasons**
   - Missing isolated tests for:
     - `MAX_DRAWDOWN`
     - `MAX_DAILY_LOSS`
     - `MAX_WEEKLY_LOSS`
     - `NEGATIVE_EXPECTANCY`

2. **No direct live halt reducer tests**
   - Missing tests that assert:
     - in-position halt -> `EXIT_<reason>`
     - flat halt -> `HALTED_<reason>`

3. **Order/auth halts are not covered in the actual trading path**
   - Existing test coverage touches queued bot starts, not live bot decision/execution halts.
   - Missing tests for:
     - `MAX_ORDER_ERRORS`
     - `BINANCE_AUTH_INVALID`

4. **No parity test for halt flattening**
   - Missing fixture proving the same risk condition flattens a held backtest position and a held live position with consistent reason semantics.

## Acceptance readout against the P0 item

### Acceptance target
- one canonical decision-flow spec exists
- latest-signal, backtest, and live bot use the same documented gate order
- regression tests cover blocked-entry, hold, flip, and halt paths

### Status now
- **Canonical spec exists:** **Yes** (this artifact)
- **Same documented gate order across all surfaces:** **No, not yet**
- **Regression matrix for blocked-entry / hold / flip / halt:** **No, not yet**

Primary blocker to full acceptance:
- `Trader.Trading` still owns a parallel decision engine instead of reusing the latest-signal/post-direction gate reducer.

## First follow-up actions

1. **Write parity fixture tests before code refactor**
   - Add one small synthetic scenario for each class:
     - blocked-entry
     - hold
     - flip
     - halt
   - Each scenario should assert outcomes at:
     - latest-signal surface
     - live reducer surface
     - backtest surface (where applicable)

2. **Extract a shared pure post-direction reducer**
   - Make latest-signal and backtest call the same ordered gate function instead of maintaining separate inlined order.
   - `signalRunPostDirectionGates` is the best current seed.

3. **Unify vol-conf close semantics**
   - Backtest should either reuse `volConfStatefulCloseDirection` or document why it intentionally diverges.

4. **Add explicit halt-path regressions**
   - Cover both risk halts and live execution halts.

5. **Add reason-precedence tests**
   - Lock the repo to one first-failure reason order so reports and live bot action strings stay comparable.

## Bottom line

The repo now has a written canonical spec, but the code still has **one rich stateless path (`computeLatestSignal`), one live stateful wrapper, and one parallel backtest state machine**. The next step is not another doc pass; it is parity tests plus shared reducers so decision-path invariants stop depending on manual comparison.
