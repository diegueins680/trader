# Volatility-confidence stateful risk closeout

Date: 2026-09-06

Risk ID: `VOL-TARGET-001`

Disposition: closed as a stale test-regression report

## Decision

The reported failure does not reproduce on the current canonical Haskell
wrapper. No production behavior was changed.

The permanent risk ID says “target,” but the original risk row specifically
named `volConfStatefulCloseDirection`, which is the volatility-confidence gate
rather than the distinct volatility-target sizing path. The ID is retained
because canonical risk IDs are permanent; the projections now describe the
actual subject.

## History and causal finding

- Commit `f2644e91` on 2026-05-19 changed
  `volConfStatefulCloseDirection` so `AllowEntry` and `Hold` return `Nothing`,
  preventing a stateless close-direction value from reopening an entry.
- The risk row first entered repository history later, in commit `8d41af11` on
  2026-05-27, claiming that this helper broke `cabal test`.
- Commit `910025a7` added `testVolConfHoldPreservesLivePosition`, binding the
  helper to the live order-intent reducer and simulator semantics.
- Current `testVolConfGateMalformedInputsFailClosed` independently checks the
  stateful helper cases for `AllowEntry`, `Hold`, `Block`, and
  `AllowExitOnly`.

The chronology shows that the imported report was already stale: its named fix
predated the first tracked risk entry by eight days.

## Current contract and evidence

For `VolConfGateHold`:

1. `volConfStatefulCloseDirection` emits no stateless close direction.
2. `desiredPositionForSignalWithVolConf` preserves an existing normalized live
   position without creating a new position.
3. `applyVolConfGateBehavior` preserves the existing simulated side and size.
4. With `AllowEntry` instead of `Hold`, the same neutral live signal exits,
   proving that preservation is owned by the explicit stateful behavior.

The unchanged branch baseline was verified before this closeout edit with:

```sh
bash scripts/verify.sh haskell
```

The command exited `0`; `trader-tests` passed. This is the current canonical
equivalent of the report's historical `cd haskell && cabal test trader-tests`
reproduction instruction and includes build, Fourmolu, HLint, smoke checks,
and the complete test suite.

## Scope and residual risk

This closeout does not claim economic value for the volatility-confidence gate
and does not change its configuration, prediction, sizing, position, or order
semantics. It does not alter credentials, live authorization, bot state,
champion selection, research holdouts, or any exchange interaction.

Malformed volatility-target configuration remains separately covered by
closed risk `VOL-TARGET-INVALID-001`. Any future semantic regression in the
volatility-confidence state machine should reopen this permanent risk ID with
a failing deterministic witness.
