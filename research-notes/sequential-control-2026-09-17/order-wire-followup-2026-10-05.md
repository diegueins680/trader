# Nonzero Binance wire admission — 2026-10-05

Baseline: `5cff8eb6379a46a50db6c37ac247212341d35738`.
Registration: [order-wire-engineering](../registrations/order-wire-engineering.json),
committed as `e2cb6c6f` before implementation. This is an engineering repair,
not a financial trial or candidate adoption.

## Conflict and resolution

The finite-positive gate introduced in PR #299 accepts `1e-9` and `5e-9`.
The existing `showFFloat (Just 8)` formatter renders both as zero. GHC uses
its decimal conversion when rounding the tie: Python's binary64 `.8f` is not
a valid replacement oracle at `5e-9`.

Keep the existing formatter byte-for-byte and move it into OrderNumeric. Both
validation and Binance serialization use it. After rejecting non-finite and
nonpositive inputs, parse the rendered string as `Fixed E12`: its unbounded
Integer units represent 1e-12, so eight decimal places are exact. Require a
successful parse and positive units before credential access. A zero-wire
failure appends `after eight-decimal wire formatting` to the constructor's
existing numeric message. Market base priority remains unchanged; a bad
present base cannot fall back to quote.

Public order signatures, request formats for accepted values, deployment,
permissions, leverage, risk caps, fleet and all live flags are unchanged. No
new configuration or dependency is introduced. Existing parsers, filters and
caller errors are preserved except for the intended earlier zero-wire rejection.

## Evidence and proof scope

- Seven SAT-premise/UNSAT-violation SMT queries over binary64 and unbounded
  Integer parser results. `F-ORDER-WIRE-POSITIVE` states positive parsed-unit
  admission, conditional on named formatter/parser semantics.
- Local finite guard model: five constructors, numeric/wire/context Booleans,
  40 initial plus 40 terminal states, 40 transitions, depth one. Every path to
  the terminal credential marker passes all three guards. The marker is not
  an order capability or a model of the complete trading lifecycle.
- Compiled actual source prefixes: 5728 rows (1632 boundary combinations from
  17 words plus 4096 generated rows, seed 20261005). Each covers five current,
  five pre-finite and five pre-wire prefixes: 28640 checks per version.
- 11456 comparisons against the captured old formatter preserve wire bytes.
  Selected zero-wire values reject for all five constructors. Invalid and
  ignored quote values, all markets/modes and type restrictions are included.
- Haskell generated properties run 1000 words plus edge cases and explicit
  zero-wire/tie/quantum regressions. CE-ORDER-NUM-001/002 remain; new
  CE-ORDER-WIRE-001 captures the positive-to-zero failure.
- Source mutations remove guards, reverse selection, weaken positive units,
  replace formatting, omit constructors or change the adapter delegate.
  The verifier rejects drift; a hash is not an implementation proof.

A-ORDER-WIRE explicitly assumes pinned GHC/base deterministic formatting,
Fixed E12 Read semantics, ASCII ByteString packing, ordinary immutable inputs,
source/model correspondence and adequate resources. This work does not prove
these libraries or the compiler. Compiled conformance is property-tested
engineering evidence, not a theorem about all binary64 formatting cases.

The first targeted run found a test harness roster error: it expected the new
wire counterexample in the old, deliberately unchanged fixture. Corrected the
roster to read each fixture independently. The rerun passed both targeted
tests in 15.747 seconds. No production defect was hidden by that correction.

## Illustrative overhead

GHC 9.4.8 `-O2`, 100000 base-selected calls to
`validateMarketNumbers False (Just (fromIntegral i / 10000)) Nothing` for
`i=1..100000`, strict fold count forced with `evaluate`, process CPU clock:
100000 accepted; 1.998365 CPU seconds (about 20 microseconds per call,
including input-generation/fold overhead). The Haskell property suite passed
in the same compiled driver. This single local measurement is not a worst-case
bound, exchange latency, production inference benchmark or resource guarantee.
Reproduce by compiling the pure module with `-O2`, timing that forced fold
using `System.CPUTime.getCPUTime`, and dividing picoseconds by 1e12. No credential
or network code is linked.

## Remaining scope and recommendation

Wire rounding can still increase a quantity. Tick/lot membership, full cap
preservation, upstream minimum-size promotion, caller retry composition, other
venue adapters, complete IO refinement and lifecycle/concurrency obligations
remain unresolved. No broad closure criterion is weakened: **5 scoped closures,
27 partial, 6 open**. RL-OFFLINE-001 remains HIGH/OPEN.

Financial evidence remains frozen: 108 fits, 19440 replays, 19548 registry rows;
all 108 OPE batches invalid; contaminated development; 1227 final returns
sealed. No new OOS, cost, drawdown, tail-risk, RL seed, champion or holdout
result. Recommendation: retain the champion, adopt no research candidate,
continue offline research and close remaining system obligations separately.

## Maker fallback amendment

Caller review found `sendPostOnlyEntry` catches every constructor exception and
can enter configured market fallback. A zero-wire price must be rejected before
that catch. The amendment was committed as `ddf62cc1` before its implementation.
`validOrderPrice` now delegates to the same wire-aware validator. The existing
Main no-order branch, all rounding arithmetic, fallback flag and unrelated
fallback reasons are unchanged.

CE-ORDER-WIRE-002 preserves the actual Main admission/exception branches and
previous finite-only predicate. Inert compiled continuations use the real price
validator without exchange code. Every one of 5728 rows checks current/previous
maker dispatch under both fallback flags (22912 comparisons). With price `1e-9`
and fallback enabled, the previous guard enters fallback while the new guard
returns unsent before constructor eligibility. The existing eight-state maker
model and source-bound SMT/conformance checks include wire validity explicitly.
This is a local caller repair, not complete Main refinement.

Initial pinned run 37320697846 was canceled after this additional gap was found;
its old-head results are not final evidence. The initial ordinary formal check
also predates the updated receipt. Final evidence must use the amended source.

The local `verify.py --record` attempt on the initial source stopped at the
unchanged process-bridge check: `PPO process bridge: no actual inference for
trained policy`. No receipt was written and no timing limits were widened.

## Canonical verification

Frozen implementation/proof source:
`48f054406d3c40748e1f8dfd1c9fff55231524a1`.
[Pinned run 37321662549](https://github.com/diegueins680/trader/actions/runs/37321662549),
job 111801985815, passed:

- `python3 scripts/formal/verify.py --record`: 14:06:04–14:07:14 UTC.
- `bash scripts/verify.sh formal`: 14:07:14–14:09:16 UTC. All 63 SMT
  requirement groups checked; 198 integrity tests passed in 52.591 seconds.
- `bash scripts/verify.sh full`: 14:09:16–14:17:37 UTC. Formal repeated
  (198 tests, 52.644 seconds), Haskell formatting/lint/build/smoke/test passed,
  all 241 web tests and 185 automation tests passed.
- Six targeted numeric/rounding/model/conformance tests passed locally in
  40.794 seconds; targeted HLint reported no hints. Compiled Haskell numeric
  properties also passed in the benchmark driver.

The raw generated receipt was imported unchanged, SHA256
`f95789f2008e701eded970fc82b162fee02fdf5f4ff372eddd498da8b11d8bdf`.
Every source hash was checked against both the lock and actual files. Changed
receipt sections are only orderNumbers, smt, sourceHashes and capabilityIsolation.
The capability graph adds one pure QuantityRounding → OrderNumeric edge:
294→295 local edges, 256→257 trader-hs reachable edges. Existing research
isolation conclusions are unchanged. The temporary reproduction workflow is
removed in the final receipt/report commit; implementation and proof sources
remain frozen. No proof placeholders were accepted by the formal gate.

Ordinary formal runs 37320697654 and 37321662700 passed 198 integrity tests
(49.649 and 48.217 seconds) but correctly failed the old-receipt comparison.
The actual pinned reproduction above supplies its replacement. Final-head CI
and deployment-safe merge evidence are recorded in
[PR #300](https://github.com/diegueins680/trader/pull/300).

These are scoped engineering checks, not completion of the 38-obligation mission,
a library/compiler proof, economic evidence or production authorization. No
orders, live exploration, authenticated trading experiments, live flag changes,
deployment, champion change or holdout opening occurred.
