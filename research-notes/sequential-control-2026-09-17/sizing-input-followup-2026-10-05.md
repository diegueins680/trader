# Sizing input admission — 2026-10-05

Baseline `a3cfc824af48ab5799b728538ffcfe3df2c37e06`. Registration
[sizing-input-engineering](../registrations/sizing-input-engineering.json) was
committed as `4306ae14` before implementation. No financial experiment,
exchange endpoint invocation, training or holdout access.

## Problem and resolution

Main's normalizers could skip minimum-notional checks for a NaN or missing
price, skip a NaN upper quantity bound, and publish infinity after probe minimum
expansion without a grid. Quote probe sizing could also publish infinity.
The downstream order-number guard does not establish sizing correctness.

The pure QuantityRounding validators now check effective supplied metadata
before normalization: present minimum/maximum quantities and minimum notional
must be finite and nonnegative; present quantity bounds must be ordered.
A supplied price must be finite and positive; positive minimum notional
requires an available price for base-quantity sizing. Quote-only sizing checks
its amount and notional metadata directly, without a manufactured price.

Both Main normalizers retain the existing first raw quantity/grid guard, then
validate metadata/price before comparisons or minimum expansion. The five new
errors do not match the existing minimum-size retry classifier. Probe results
also pass a finiteness check after expansion. Existing rounding formulas,
minimum-size policy on valid input, wire formatting, optional-filter meaning,
zero-filter comparison semantics, signatures, configurations and limits remain.
The error text/order changes only for newly rejected invalid inputs.

Amendment `8ecbe344` registered negative raw input rejection before changing
code. The historical raw-input validator is preserved with the old Main
functions. Negative quantities now fail before zero-clamping and minimum-size
retry; zero and negative-zero retain their prior semantics. The existing
rounding SMT preflight lemma also establishes nonnegative admitted inputs.

No new module, dependency, model, policy, service, live setting, deployment,
position owner or fleet change is introduced. The champion is preserved.

## Six synthetic regression witnesses

| Fixture | Previous pure behavior | New behavior |
|---|---|---|
| CE-SIZING-001 | NaN price skips a positive minimum-notional check | Reject before sizing or retry |
| CE-SIZING-002 | NaN maximum quantity skips its bound | Reject before sizing or retry |
| CE-SIZING-003 | Missing price skips a required probe notional check | Reject missing required price |
| CE-SIZING-004 | `1e300 / 1e-300` expansion publishes infinity without a grid | Reject the non-finite result |
| CE-SIZING-005 | Positive infinite quote amount is returned | Reject the input |
| CE-SIZING-006 | Negative raw quantity is clamped to zero and promoted by entry retry | Reject before clamping or retry |

These are synthetic code-domain failures, not evidence of historical exchange
responses, filled orders, financial losses or market prevalence. Captured old
Main fragments and corrected functions run only against pure field projections
and production numeric kernels; no exchange module or effect is linked.

## Verification and implementation conformance

- Fourteen nonvacuous SAT-premise/UNSAT-violation SMT queries over binary64,
  optional-presence Booleans and error strings. Three requirement groups cover
  admission, conditional publication bounds and non-retryable errors.
- Local minimum-retry model: 56 states, 24 initial, 24 terminal, 32 transitions,
  maximum depth three. Invalid admission cannot reach normalization or retry;
  only the existing too-small outcome permits minimum expansion.
- 6214 compiled rows: 4160 boundary combinations, 2048 generated rows (seed
  20261005), six witnesses. Four actual current and four preserved functions
  yield 24856 function cases per version.
- An independent Python oracle checks admission. Successful valid results retain
  exact quantity bits and entry-promotion flags: 3148 normalizeQty, 3385
  normalizeEntryQty, 3291 normalizeProbeQty and 4599 validateProbeQuote results.
- The actual normalizeEntryQty fragment is additionally compiled with a forbidden
  minimum-expansion marker. Invalid input must reject before that marker.
- Haskell generated properties and source/caller mutations check the pure guards.
  Existing rounding, wire and maker-fallback certificates remain in the suite.
- All eight targeted tests passed locally after the negative-input amendment
  in 36.934 seconds. Targeted HLint
  reported no hints. Source hashes match the reviewed lock.

A-SIZING-INPUT names pinned GHC/base binary64, Maybe/Either, Integer/Rational,
record-field projection, immutable non-bottom inputs, source/model mapping,
existing rounding certificates and adequate resources. The publication SMT
lemma is conditional on upstream finiteness/rounding certificates; conformance
tests connect it to the actual functions. This is not a compiler, whole-Main,
exact-real notional accounting, freshness or full risk-cap proof.

## Illustrative pure-validator overhead

GHC 9.4.8 `-O2`, one million forced strict-fold calls to
`validateSizingInputs (Just 0) (Just 10) (Just 50) (Just (fromIntegral i / 10000))`,
`i=1..1000000`: all accepted, **0.07339 process CPU seconds**, including generation
and fold overhead. Metadata is fixed in this sample and may benefit from compiler
optimization. Reproduce using `evaluate`, `Data.List.foldl'` and
`System.CPUTime.getCPUTime` around that fold, dividing picoseconds by 1e12.
This is not a worst-case, production inference or exchange-latency bound.

## Limits and decision

Provider parsing can still discard malformed fields; freshness and completeness
are not established here. Missing optional filters retain their old meaning.
Direct minTradeQty consumers (including position classification), notional
rounding, final wire/tick/lot membership, complete exposure-cap composition,
other venues and broader lifecycle/concurrency properties remain unresolved.
The existing source-bound downward/upward registries are rebound only for the
reviewed guard additions; their rounding algorithms are unchanged.

No broad status or closure criterion changes: **5 scoped closures, 27 partial,
6 open**. RL-OFFLINE-001 remains HIGH/OPEN. The research record is unchanged:
108 fits, 19440 replays, 19548 registry rows, contaminated development, all
108 OPE batches invalid, and 1227 final returns sealed. No new OOS, costs,
returns, drawdown, tail-risk, seed, champion or holdout evidence.

Recommendation: preserve the champion and adopt no research candidate. Continue
offline research and remaining specification/refinement work separately.

## Canonical checks

The local `python scripts/formal/verify.py --record` attempt stopped in the
unchanged PPO process bridge at `PPO process bridge: no actual inference for
trained policy`. No receipt was written, no timing limit was widened, and no
user process was stopped. Ordinary CI on the pre-receipt source later passed
all 200 integrity tests but correctly failed with `certificate differs from
reviewed receipt; investigate before recording`; its Haskell, web and automation
jobs passed. This stale-receipt failure was resolved by importing the generated
receipt, not by weakening comparison or hand-editing results.

Pinned reproduction [run 37330001658](https://github.com/diegueins680/trader/actions/runs/37330001658),
job 111830400641, checked out source
`5586fb2619f4c651f7a866e8c3fba339f4618c9f` and passed:

- Receipt generation: 2026-10-05 15:08:24–15:09:37 UTC.
- `bash scripts/verify.sh formal`: 15:09:37–15:11:42 UTC; 200 integrity
  tests in 50.954 seconds, then SMT/model/compiled conformance verification.
- `bash scripts/verify.sh full`: 15:11:42–15:20:03 UTC; 200 integrity
  tests in 51.411 seconds, all 66 SMT groups, Haskell suite, 241 web tests
  and 185 automation tests passed. Formal verification is included in full.

Receipt SHA256:
`15d082a6882382a884b8dec3d81e8a579cf6e4dd3f91240a3fd6bc48f17879ab`.
It was imported byte-for-byte from that job; receipt hashes equal the reviewed
lock and actual source bytes. Changed sections are sizingInputs, smt,
sourceHashes and capabilityIsolation's sourceHashes only. Other certificates
are unchanged. Preserved Main fragments and the historical raw validator were
also compared directly with baseline Git source and matched exactly.

The temporary reproduction workflow is removed from the final tree. This final
commit changes only the receipt, this report and workflow removal, preserving
all implementation/proof source hashes. Final-head CI and merge/deployment
audits are recorded in PR #301; no whole-mission completion is asserted.
