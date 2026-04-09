Insert the following new section after the existing "Formal autoloop safety contract" section and before "## What is not proved".

## Formal signal directionality gate contract

`signalDirectionalitySnapshot` and `signalDirectionalityEntryAllowed` in `haskell/app/Trader/SignalGates.hs` are treated as a fail-closed classifier for low-directionality and mean-reversion entry gating.

Clauses:

1. Directionality efficiency is admissible only when it is finite and in `[0,1]`; malformed efficiency inputs are non-directional and cannot reopen an entry.
2. With fixed edge/confidence outside this helper, lowering the 24-bar efficiency score must be monotone non-increasing for entry admissibility: once a state is blocked, any lower-efficiency state stays blocked.
3. Inside the mean-reversion band (`0.25 < efficiency <= 0.40`), increasing mean-reversion dominance must be monotone non-increasing for entry admissibility: once a state is blocked, stronger MR dominance stays blocked.
4. Malformed regime probabilities (`NaN`, `Infinity`, `< 0`, `> 1`) are fail-closed and no more permissive than the strong-MR blocked state, even when the price path itself still looks like a directional trend.
5. The snapshot may keep its price-path label (`trend-up`, `trend-down`, etc.) for observability, but `dsReason = NON_DIRECTIONAL_MALFORMED` and `dsNonDirectional = True` must veto entry whenever regime evidence is malformed.

The verifier in `haskell/test/TestMain.hs` now checks this contract with bounded executable cases:

- `testSignalDirectionalityMalformedEfficiencyFailsClosed` uses a synthetic snapshot to prove non-finite efficiency is rejected by `signalDirectionalityEntryAllowed`.
- `testSignalDirectionalityMalformedRegimeFailsClosed` proves a strong-trend price path with malformed regime probabilities is still vetoed as `NON_DIRECTIONAL_MALFORMED`.
- `testSignalDirectionalityEfficiencyMonotonicity` checks a three-step efficiency ladder under fixed trend-friendly regime evidence and proves the allowed/blocked sequence is monotone non-increasing as efficiency falls.
- `testSignalDirectionalityMrDominanceMonotonicity` checks a four-step MR-dominance ladder (`trend leader`, `weak MR leader below hysteresis`, `strong MR leader`, malformed probabilities) inside the MR efficiency band and proves the allowed/blocked sequence is monotone non-increasing as MR dominance strengthens.

Proof sketch:

- `directionalityEfficiencyOk` is now the shared finite-range gate for both snapshot well-formedness and the derived price-path metrics, so malformed efficiency inputs cannot survive as directional snapshots.
- `signalDirectionalityRegimeEvidence` now returns an explicit malformed-evidence flag whenever any regime probability falls outside `[0,1]`; missing regimes remain `Nothing`, but malformed regimes are distinguishable from absent evidence.
- `signalDirectionalitySnapshot` preserves the existing `NON_DIRECTIONAL_CHOP` and `NON_DIRECTIONAL_MR` thresholds, and only adds a fail-closed `NON_DIRECTIONAL_MALFORMED` veto when regime evidence is malformed or unusable. This keeps the threshold contract unchanged while ensuring malformed market-state inference is never more permissive than the blocked low-directionality states.
- `signalDirectionalityEntryAllowed` still requires both a well-formed snapshot and `dsNonDirectional = False`, so malformed persisted or synthetic snapshots fail closed even if a caller bypasses the snapshot builder.