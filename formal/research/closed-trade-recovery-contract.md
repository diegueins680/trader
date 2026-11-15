# Closed-trade snapshot admission v1

Baseline d57968a66d8f85d08e6bfb430f1472781dcf2fa5. Engineering only;
no orders, deployment, dataset access, financial experiment or holdout access.

H-EXECUTION-I2 requires identity-matched bounded closed-trade memory and excludes
persisted position/open-trade claims from startup exposure. Current recovery
accepts non-finite equity, substitutes a return for a non-finite supplied return,
can derive an infinite return from finite equities, and adds holding periods in
machine Int without checking overflow. Resolve these against G-FINITE and
G-FAIL-CLOSED by rejecting malformed closed-trade records before admission.

Entry equity must be finite and positive; exit equity must be finite (negative
finite exit equity remains representable as an insolvent historical outcome).
A present return must be finite; an absent return may be derived as exit/entry-1
only if its binary64 result is finite. Do not replace explicitly invalid values
with zero or a derived return. Holding periods must be nonnegative. An optional
high-volatility probability is either absent or finite in [0,1]. Preserve all
other existing closed-trade fields, defaults, explicit finite-return semantics,
identity checks, trade-limit behavior, valid ordering and API/JSON signatures.
No proof that explicit returns reconcile external cash accounting is claimed.

After the existing last-N selection, reindex using exact Integer arithmetic.
Each trade consumes max(1,holdingPeriods) index units, then a one-index gap.
Publish only if 0 <= entry < exit <= maxBound::Int for every retained record.
Convert to Int only after that check. If the selected history cannot fit,
return no recovered history, not a truncated prefix or wrapped indices. Zero
holding periods retain their old one-index span. Retrying identical input is
pure and deterministic. No position/ownership state is restored or changed.

Proof obligations: binary64 admission implies finite selected return and valid
metadata; unbounded-integer index recurrence preserves strict ordering and
machine bounds before narrowing. Model-check identity/admission/selection/
index-validation/publication with invalid stages unable to publish, including
retry idempotence; this is pure local recovery, not durable filesystem recovery.
Compile actual pure source helpers against projected Trade fields. Differential
checks use independent Python binary64/integer oracles and preserved old helpers.
Full Haskell tests exercise the actual Aeson decoder and identity/position fields.

Budget: 2048 seeded binary64 rows (seed 20261005) plus a documented deterministic
boundary cross product; finite index histories of length 0..4 with durations
0,1,2,3 under bound 7 for model enumeration; actual Int boundary histories and
1024 seeded lists up to length 8; deterministic old-source counterexamples.
No financial metrics, seeds, policies, configurations or proof gates change.

Assumptions: pinned GHC/base/Aeson parsing and record semantics, ordinary immutable
non-bottom inputs, binary64 operations, exact Integer and checked narrowing,
source extraction correspondence and adequate resources. Pure-helper conformance
and JSON regression tests are not a compiler or complete parser/refinement proof.
Input-size/resource bounds, durable crash recovery, authenticity, snapshot
freshness, concurrent ownership and venue reconciliation remain unresolved.
No broad closure is inferred; obligations 6/10/29/37 receive scoped evidence only.
