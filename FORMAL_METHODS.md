## Formal fee-aware entry gate contract

`signalEntryFeeBufferOk` in `haskell/app/Trader/SignalGates.hs`, as wired into entry admissibility in `haskell/app/Trader/Trading.hs`, is treated as a fail-closed marginal-entry veto.

Clauses:

1. Let `requiredHeadroom = 1.5 * normalizeSignalThreshold openThreshold` and `requiredEdge = requiredHeadroom + max 0 roundTripFeeFloor`. An entry is admissible only when the fee floor and edge are both finite and `edge >= requiredEdge` whenever `requiredEdge > 0`.
2. For fixed `openThreshold` and edge, admissibility is monotone non-increasing as `roundTripFeeFloor` rises.
3. For fixed `openThreshold` and fee floor, admissibility is monotone non-increasing as raw edge falls.
4. Non-finite fee floors or non-finite edges are fail-closed.
5. Once a state is blocked at fee floor `f`, it remains blocked for every `f' >= f`.

Bounded executable obligations in `haskell/test/TestMain.hs`:

- `testSignalGateEntryFeeBuffer`
- `testSignalGateEntryFeeBufferMonotoneFees`
- `testSignalGateEntryFeeBufferMonotoneEdge`
- `testSignalGateEntryFeeBufferFailsClosed`

Proof sketch:

- `signalEntryFeeBufferOk` normalizes negative fee floors to `0` but rejects non-finite fee floors outright, so malformed fee inputs cannot reopen entries.
- The guard compares edge against an affine requirement with unit slope in the fee floor, so increasing fees cannot reduce the minimum admissible edge.
- Lowering raw edge cannot make a blocked state admissible because the predicate is only `edge >= requiredEdge` once the inputs are well formed.
- `simulateEnsembleLongFlatVWithHLChecked` consults the new predicate only on entry attempts and combines it conjunctively with the existing threshold/headroom/spike gates, so the fee-aware gate can only veto marginal trades; it cannot turn a previously blocked state into allowed.