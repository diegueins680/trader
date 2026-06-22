# Formal Specs Consistency Audit

Date audited: 2026-06-21

Scope:

- `docs/formal-specs-extracted.md`
- `FORMAL_METHODS.md`
- `README.md`
- `docs/audits/strategy-decision-flow-spec.md`
- `docs/audits/p0-item-2-data-integrity-and-leakage.md`
- Haskell modules referenced by the extracted specs
- `haskell/test/TestMain.hs`

Verification run:

- `bash scripts/verify.sh haskell`
- Result: PASS

The wrapper completed Cabal build, Fourmolu check, HLint, smoke checks, and the Haskell test suite successfully. After the repairs below, the VolConf formal-report fields that drifted are asserted by the test harness.

## Summary

Most implemented contracts map cleanly to code and tests: fresh-entry fee/headroom/spike gates, conformal intervals, quantile intervals, cost attribution, order execution, risk halts, threshold calibration, close timing, telemetry, bot adoption guards, and cost calibration all have direct implementation hooks and regression coverage.

The initial issues were spec/documentation drift, not failing production behavior. This audit now records the repaired state: README constants match the implemented entry-threshold and directionality defaults, `FORMAL_METHODS.md` matches edge-normalization behavior, `Trader.Formal.Optimization` matches VolConf malformed-confidence semantics, and the old P0 market-data audit has a superseding Binance/stale-threshold note.

## Findings

### F1. README entry-threshold feasibility contradicts code and tests

Status: Resolved.

Documents:

- `README.md:39`
- `README.md:40`

Implementation:

- `haskell/app/Trader/SignalGates.hs:103`
- `haskell/app/Trader/SignalGates.hs:104`
- `haskell/app/Trader/SignalGates.hs:486`
- `haskell/app/Trader/SignalGates.hs:487`

Tests:

- `haskell/test/TestMain.hs:3384`
- `haskell/test/TestMain.hs:3392`

README previously said the spike cap was `min(4 * openThreshold, 0.5)` and therefore `openThreshold > 1/3` was infeasible. Code uses default `sgcEntryEdgeSpikeMultiple = 1000.0` and `sgcEntryEdgeSpikeCredibleCap = 5.0`, so `signalEntryOpenThresholdFeasibilityCap = 5.0 / 1.5 = 10/3`.

The implementation, optimizer/top-combo filters, tests, and README now agree on `10/3`.

Prior impact:

- Operators reading the README would expect combos with `openThreshold > 1/3` to be rejected, but implementation permits them up to `10/3`.
- The extracted spec correctly noted the drift before the README repair, but the canonical README was misleading.

Applied repair:

- README was updated to `1000x/5.0` and `10/3`.

### F2. README directionality chop threshold contradicts code default

Status: Resolved.

Document:

- `README.md:56`

Implementation:

- `haskell/app/Trader/SignalGates.hs:109`
- `haskell/app/Trader/App/Args.hs:1320`

Tests:

- `haskell/test/TestMain.hs:6295`
- `haskell/test/TestMain.hs:6332`

README previously said entries were vetoed as `NON_DIRECTIONAL_CHOP` when 24-bar price efficiency is `<= 0.18`. Code default is `sgcDirectionalityChopEfficiencyMax = 0.08`, and the CLI default is derived from that value. README now says `<= 0.08`.

Prior impact:

- Runtime is stricter/narrower for chop classification than the README states.
- Optimizer ranges also sample `--signal-directionality-chop-efficiency-max`, so this may be tunable, but the documented default is wrong.

Applied repair:

- README was updated to `0.08`.

### F3. `FORMAL_METHODS.md` misstates non-finite edge normalization

Status: Resolved.

Document:

- `FORMAL_METHODS.md:86`
- `FORMAL_METHODS.md:104`
- `FORMAL_METHODS.md:120`

Implementation:

- `haskell/app/Trader/SignalGates.hs:187`
- `haskell/app/Trader/SignalGates.hs:188`
- `haskell/app/Trader/SignalGates.hs:190`

Tests:

- `haskell/test/TestMain.hs:3611`
- `haskell/test/TestMain.hs:3623`
- `haskell/test/TestMain.hs:7607`
- `haskell/test/TestMain.hs:7614`

`FORMAL_METHODS.md` previously said `normalizeSignalEntryEdge` collapses negative or non-finite raw edges to `Just 0`. Code preserves finite non-negative values, maps finite negatives to `Just 0`, and maps non-finite values to `Nothing`. The formal prose now matches that behavior.

The implementation is safer than the stale wording, and tests assert the current behavior.

Prior impact:

- The formal prose is internally inconsistent with adjacent fail-closed clauses that say malformed edge samples fail closed.
- A future implementer following the stale prose could weaken the gate by converting non-finite edges to explicit zero evidence.

Applied repair:

- Replaced "negative or non-finite raw edges to `Just 0`" with "finite negative raw edges to `Just 0`; non-finite raw edges to `Nothing`."

### F4. `Trader.Formal.Optimization` VolConf helper is stale and partly unmapped

Status: Resolved.

Implementation:

- `haskell/app/Trader/VolConfGate.hs:239`
- `haskell/app/Trader/VolConfGate.hs:246`
- `haskell/app/Trader/VolConfGate.hs:293`

Current production spec:

- Missing confidence is weak evidence.
- Provided malformed confidence is `AllowExitOnly 0`.

Former stale formal helper:

- `haskell/app/Trader/Formal/Optimization.hs:1043`
- `haskell/app/Trader/Formal/Optimization.hs:1054`
- `haskell/app/Trader/Formal/Optimization.hs:1116`

`canonicalizeConfidenceInput` previously mapped missing and non-finite confidence to `Just 0.0`, and `fvrVolConfMalformedConfidenceMatchesWeak` checked malformed confidence against weak evidence. The formal helper now preserves missing confidence as missing/weak, keeps malformed provided confidence malformed, renames the report field to `fvrVolConfMalformedConfidenceFailsClosed`, and checks malformed provided confidence against exit-only semantics.

Tests:

- `haskell/test/TestMain.hs:3765`
- `haskell/test/TestMain.hs:3791`

Coverage repair:

- `testVolConfGateMalformedInputsFailClosed` now asserts all active VolConf report fields: canonicalization, malformed volatility, malformed confidence, conservative malformed inputs, and bounded output.

Prior impact:

- The executable formal report can silently contain false/stale VolConf fields while the production tests pass.
- `docs/formal-specs-extracted.md` could treat stale `verifyFormalOptimization.fvrVolConf*` fields as current authority unless the formal helper and extracted spec are kept aligned.

Applied repair:

- Repaired `canonicalizeConfidenceInput` and the malformed-confidence predicate, renamed the field, added out-of-range volatility/confidence witnesses, and asserted all active `fvrVolConf*` fields.

### F5. VolConf `[0,2]` bound is a default, not a hard invariant under CLI overrides

Status: Resolved.

Documents:

- `FORMAL_METHODS.md:93`
- `README.md:46`

Implementation:

- `haskell/app/Trader/VolConfGate.hs:102`
- `haskell/app/Trader/VolConfGate.hs:127`
- `haskell/app/Trader/App/Args.hs:1248`
- `haskell/app/Trader/App/Args.hs:2401`
- `haskell/app/Main.hs:577`

Docs previously said volatility evidence must be within `[0,2]`. The preset default uses `vcgcVolatilityEvidenceMax = 2.0`, but CLI/API config can override `--vol-conf-volatility-evidence-max`, and validation only requires it to be `> 0`. README and `FORMAL_METHODS.md` now describe this as a configured evidence max with `2.0` as the default.

Prior impact:

- If `[0,2]` is intended as a hard safety invariant, the implementation does not enforce it.
- If `[0,2]` is intended as the default preset, docs should say "default max is 2.0" rather than "must be within [0,2]".

Applied repair:

- Clarified the spec as configurable with default max `2.0`.

### F6. Market-data audit is stale for Binance, and stale-threshold policy differs

Status: Resolved for the stale audit text; broader P0 gate remains a documented gap.

Old audit:

- `docs/audits/p0-item-2-data-integrity-and-leakage.md:66`
- `docs/audits/p0-item-2-data-integrity-and-leakage.md:67`
- `docs/audits/p0-item-2-data-integrity-and-leakage.md:285`

Current implementation:

- `haskell/app/Trader/Binance.hs:596`
- `haskell/app/Trader/Binance.hs:932`
- `haskell/app/Trader/Binance.hs:943`
- `haskell/app/Trader/Binance.hs:948`
- `haskell/app/Trader/Binance.hs:962`
- `haskell/app/Main.hs:12672`
- `haskell/app/Main.hs:12692`

The P0 data-integrity audit used to say Binance does not retain close time or filter closed candles. Current code parses close time, filters closed klines, validates finite/non-negative/OHLC shape, and validates strict increasing open times. The bot loop also holds on stale data and market-data gaps.

The audit also recommended stale data only after the last closed bar was older than `2 * interval`, while `marketDataFreshness` marks stale when `ageMs > interval`:

- Audit: `docs/audits/p0-item-2-data-integrity-and-leakage.md:313`
- Implementation/test: `haskell/app/Trader/MarketDataIntegrity.hs:31`, `haskell/test/TestMain.hs:3435`

Prior impact:

- The audit document under-reports current Binance safety.
- The stale threshold is stricter in code than the audit recommendation. That may be acceptable, but it should be explicit.

Applied repair:

- Added a superseding Binance note to the P0 audit.
- Documented the chosen production stale threshold as `ageMs > intervalMs`, measured from the last processed candle close time.

## Mapping Status By Spec

| Spec | Mapping status |
|---|---|
| Fresh-entry gates | Code/tests/docs map. |
| Directionality gate | Code/tests/docs map. |
| Vol-confidence gate | Production code/tests/formal helper map for active fields. |
| Conformal intervals | Code/tests map. |
| Quantile intervals | Code/tests map for malformed/dimension/finiteness behavior. |
| Backtest cost attribution | Code/tests map. |
| Order execution | Code/tests/formal helper map. |
| Risk halt decision | Code/tests/formal helper map. |
| ROI scoring/tie-breaks | Code/formal helper mostly map for asserted fields; not every `FormalVerificationReport` field is asserted. |
| Close timing | Code/tests map. |
| Threshold calibration | Code/tests/docs map. |
| Gate telemetry | Code/tests/docs map. |
| Market data integrity | Helper and Binance live path map; broader cross-loader P0 QA gate remains a documented gap. |
| Bot startup/adoption | Code/tests map. |
| Cost calibration | Code/tests map. |
| Top-combo scoring/refresh | Code/tests map. |
| Strategy decision flow | Correctly documented as intended/current-parity gap. |
| Data integrity checklist | Correctly documented as broader P0 gap, with Binance update note and production stale threshold refreshed. |

## Repairs Applied

1. Fixed `Trader.Formal.Optimization` VolConf predicates and added assertions for all active `fvrVolConf*` report fields.
2. Updated README entry-threshold feasibility constants to match code.
3. Updated README directionality chop default to match code.
4. Corrected `FORMAL_METHODS.md` edge-normalization prose.
5. Clarified VolConf evidence max as configurable with default `2.0`.
6. Refreshed the P0 data-integrity audit for Binance and documented the chosen stale threshold.
