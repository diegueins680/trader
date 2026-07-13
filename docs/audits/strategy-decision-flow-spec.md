# Strategy decision-flow spec

Date: 2026-04-19
Updated: 2026-07-12
Owner: Trader Firm CTO
Scope: P0 audit item 1 — strategy decision-path invariants
Sources audited:
- `haskell/app/Main.hs`
- `haskell/app/Trader/Trading.hs`
- `haskell/app/Trader/SignalGates.hs`
- `haskell/app/Trader/VolConfGate.hs`
- `haskell/test/TestMain.hs`

## Audit status

This artifact gives the repo one canonical written decision-flow spec. The shared post-direction gate reducer is now used by latest-signal and by the backtest gates the simulator owns; full surface parity still excludes live-only execution state and simulator-only accounting.

Current state:
- **Latest-signal** has the richest stateless prediction/gate/sizing path.
- **Live bot** mostly wraps latest-signal, then adds stateful hold / cooldown / exposure / halt / execution behavior.
- **Backtest** still owns simulator state transitions inside `Trader.Trading`, but its available post-direction structural gates route through `signalRunPostDirectionGates`.
- Regression fixtures now cover the **blocked-entry / hold / flip / halt** semantic matrix at shared production seams; full IO-stack coverage remains intentionally narrower.

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
Backtest applies the entry-only edge checks before the structural reducer:
- `signalEntryEdgeSpikeOk`
- `signalEntryHeadroomOk`

It then routes the post-direction gates it owns through `signalRunPostDirectionGates`:
- volatility gate
- vol-target-ready gate
- trend gate
- cloud gate
- price-action gate
- signal-to-noise gate

### Surface-specific vs latest-signal
Backtest passes explicit allow/no-op checks for latest-signal-only contexts it does not own locally:
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
- Backtest now shares the ordered post-direction reducer boundary for the gates available in the simulator.
- It remains a simulator-specific state machine after direction selection, which is expected for bracket exits, rebalance, funding, and risk-halt accounting.

## Canonical map: prediction -> gate -> sizing -> execution -> exit

| Stage | Canonical rule | Latest-signal | Live bot | Backtest |
|---|---|---|---|---|
| Prediction | Build raw open/close dirs from models | Yes | Delegates to latest-signal | Yes |
| Method chooser | Pick method-specific open/close dir | Yes | Delegates | Yes |
| Entry spike/headroom | Entry-only gate before later structural gates | Yes | Delegates | Yes |
| Post-direction gates | Single shared ordered reducer | **Yes** | Delegates | **Shared for available gates** |
| Non-directional veto | Shared structural gate | Yes | Delegates | **No-op unless simulator evidence is added** |
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

**Remaining surface differences**
- backtest uses `signalRunPostDirectionGates` for available gates, but passes no-op checks for latest-signal-only evidence not present in simulator context
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

### Added parity regression matrix
- blocked-entry: the shared reducer locks first-failure precedence (`VOLATILITY` before later gates and an upstream reason before post-direction reasons), while a synthetic backtest proves its real volatility gate blocks an otherwise active entry
- hold: latest-signal's vol-confidence close-direction suppression now feeds a stateful live reducer that preserves held exposure; the fixture also checks the simulator's canonical `VolConfGateHold` side/size result
- flip: live order intent proves long-to-short direction, while a synthetic long-short backtest proves the held-long to short transition and counts entry + exit + replacement entry as three position changes
- halt: live risk evaluation now calls `specRiskHalt` through a pure halt/action helper; the max-drawdown fixture proves the same `ExitMaxDrawdown` reason, flat target, sell direction, and flat backtest result

### What is not yet covered end-to-end
The matrix now covers the four semantic paths at their shared production seams. It does **not** yet call the full `computeLatestSignal -> botApplyKline -> exchange execution` stack for every reason or every live-only portfolio/order gate.

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

2. **Shared volatility blocked-entry parity is covered; latest-signal-only evidence is not**
   - The exact shared reducer reason and an otherwise-active simulator entry blocked by the volatility gate are covered.
   - `NON_DIRECTIONAL_*` remains outside backtest parity because the simulator does not fabricate evidence it cannot provide point-in-time.

3. **No live blocked-entry reducer tests**
   - Missing stateful tests for:
     - `MAX_OPEN_POSITIONS`
     - `MAX_OPEN_PER_BASE`
     - `MAX_GROSS_EXPOSURE`
     - `MAX_NET_EXPOSURE`
     - `MAX_EXPOSURE_PER_BASE`
     - `NO_TRADE_WINDOW`
     - `MAX_TRADES_PER_DAY`

4. **Canonical shared reason order is covered; full action rendering is not**
   - The regression locks first-failure and upstream-reason precedence in the exact reducer called by latest-signal and backtest.
   - A direct `computeLatestSignal` assertion over every final `lsAction` string remains open.

## B. Hold path gaps

1. **No integrated `HOLD_MIN_HOLD` test**
   - Live logic has explicit min-hold state handling, but test coverage is missing.

2. **Vol-conf hold state transition is covered**
   - Latest-signal now carries the evaluated behavior into the actual live/one-shot reducers, and `VolConfGateHold` preserves an existing position.
   - A full IO-level `botApplyKline` fixture remains open.

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

1. **Live flip intent is covered; fill choreography is not**
   - The pure production reducer proves a held long targets short and emits a sell direction.
   - Closing quantity, replacement fill reconciliation, and action-label assertions remain open.

2. **Backtest long-to-short flip regression is covered**
   - The fixture observes both held-long and later short exposure and locks turnover at three position changes.

3. **No tests for flip-specific forced exits**
   - Missing backtest tests for:
     - `LSTM_FLIP`
     - `KALMAN_SLOW`
     - `KALMAN_BAND`

4. **Flip transition parity is covered at semantic seams**
   - Live transition intent and backtest position/turnover agree for long-to-short.
   - A single IO fixture spanning computed latest signal through exchange fill remains open.

## D. Halt path gaps

1. **Max-drawdown backtest halt is covered; other halt reasons remain**
   - Covered: `MAX_DRAWDOWN`, including canonical exit reason and final flat exposure.
   - Still missing isolated tests for:
     - `MAX_DAILY_LOSS`
     - `MAX_WEEKLY_LOSS`
     - `NEGATIVE_EXPECTANCY`

2. **Pure live halt action is covered; rendered action labels remain**
   - An in-position max-drawdown halt targets flat and emits the correct opposing order direction.
   - `EXIT_<reason>` / `HALTED_<reason>` rendering assertions remain open.

3. **Order/auth halts are not covered in the actual trading path**
   - Existing test coverage touches queued bot starts, not live bot decision/execution halts.
   - Missing tests for:
     - `MAX_ORDER_ERRORS`
     - `BINANCE_AUTH_INVALID`

4. **Max-drawdown halt flattening parity is covered**
   - Live and backtest share `specRiskHalt`, report `ExitMaxDrawdown`, and flatten held-long exposure consistently.

## Acceptance readout against the P0 item

### Acceptance target
- one canonical decision-flow spec exists
- latest-signal, backtest, and live bot use the same documented gate order for shared gates
- regression tests cover blocked-entry, hold, flip, and halt paths

### Status now
- **Canonical spec exists:** **Yes** (this artifact)
- **Same documented post-direction gate order across shared gates:** **Yes**
- **Regression matrix for blocked-entry / hold / flip / halt:** **Yes, at the shared pure/live/backtest semantic seams**

Primary remaining depth beyond this acceptance item:
- IO-level fixtures still do not span every path from `computeLatestSignal` through `botApplyKline` and exchange fill/reconciliation.
- Latest-signal-only evidence and live-only portfolio/order gates remain intentionally outside simulator parity where point-in-time evidence does not exist.

## First follow-up actions

1. **Extend the parity fixtures to IO surfaces**
   - Add bounded `computeLatestSignal` and `botApplyKline` fixtures for final action labels and fill/reconciliation, without duplicating the production reducers.

2. **Extend simulator evidence for currently no-op gate contexts**
   - Add simulator-side inputs for non-directionality, regime edge, MTF, cross-asset, meta-label, and funding/OI only where the backtest can provide point-in-time evidence.
   - Keep missing evidence explicit rather than silently fabricating latest-signal contexts.

3. **Keep vol-conf stateful semantics shared**
   - Preserve the live behavior-carrying reducer and simulator `applyVolConfGateBehavior` fixtures as either surface evolves.

4. **Add explicit halt-path regressions**
   - Cover both risk halts and live execution halts.

5. **Extend reason-precedence tests to rendered actions**
   - The shared reducer order is locked; add direct latest-signal action-string checks for the latest-signal-only gates.

## Bottom line

The repo now has a written canonical spec, a shared post-direction reducer boundary, a stateful live vol-confidence reducer, canonical live/backtest risk halts, and regression fixtures for blocked-entry, hold, flip, and halt semantics. The next depth step is bounded IO-level coverage and, where justified, point-in-time simulator evidence for currently no-op latest-signal-only gates.
