# Immutable optimizer publication repair — 2026-10-03

The new `optimizer-snapshot-v2` fixes partial publication for a complete standalone
offline optimizer and its forward consumer. It does not replace the frozen v1 learner,
its policy artifacts or any production path. It is disabled by default and is not a
new financial experiment. The broader mission still has 35 unresolved obligations.

The initial specification was committed in `130dbb3a` before implementation. Source
freeze: `61bb43bc467c9e6b0e68ac3ddc6fd0ad8e3b93dd`. Latest main checked was
`dbd45e2691cb37f1421a23306c43b676fa82e6fc`. The branch remains
`research/sequential-review-2026-09-28`, draft PR #284 stacked on draft #281.

## Actual repair and scope

Frozen v1 stores parameters, first moments, second moments and step counter in four
separate attributes. CE-RL-016 exhibits mixed observations during those stores.
The replacement stores one frozen snapshot containing tuples of immutable bytes.
Forward and update each capture a single snapshot. Derived NumPy views do not live
inside the snapshot. Staging builds all new arrays locally and validates/finalizes
the entire candidate before publication. A nonblocking lock and expected-object
identity comparison reject busy or stale writers. One reference store publishes
parameters, moments, variances and step together.

Enabled entry points require the pinned GIL-enabled CPython/NumPy runtime, valid
controls and exact base finite arrays. Unsupported configurations return absence.
The default path returns absence before inspecting other arguments. The public
functions have no file, network, process, exchange, promotion or artifact operation.
The existing Haskell action/capability conformance remains in the verification gate;
no Haskell production code, dependency or environment configuration changes.

A failed call before publication leaves the old snapshot; after publication the new
snapshot can remain even if acknowledgement is lost. The implementation makes no
exactly-once retry promise. Interrupted lock handoff can strand future writes; they
return absence instead of waiting, while immutable snapshot readers remain usable.
This is not durable recovery, permanent-lockout prevention or hard cancellation.

## Verification and conformance

The source-bound model explores 3,970 states and 9,981 transitions to the reachable
fixed point, with two writers, two calls each and one retained reader. Maximum
shortest depth is 29; the non-stutter rank bound is 49. It includes 502 orphan-lock
states and 152 stale comparisons. All reads remain coherent and all commits have
matching expected identity and exclusive ownership. The rank is conditional on
terminating primitives and is not a wall-clock bound or scheduler guarantee.

Two SMT requirements check the actual integer publication guard and binary64
selected-slot pack predicate. The step cap is 2^31-1; snapshot payloads contain
675 or 777 float64 slots. Array scans, serialization, immutable-byte semantics,
CPython reference/lock atomicity and the ordinary API domain are explicit assumptions.
Source AST, compiled single-store shape, call roster, defaults and hashes bind the
implementation. No universal NumPy/compiler refinement or optimizer convergence is
claimed. Property tests and runtime observations remain separately classified.

Conformance covers 144 paired updates (seeds 11/23/47, actor/critic widths 3/1,
batches 1/8/256, eight steps), all-field and forward parity, numerical/staging failure,
step cap, invalid representations/controls, immutable views, two forced concurrent
writers, retained readers during update, stale publication, occupied lock behavior,
unsupported runtime and nonthrowing publication observations. The registry's original
v1 counterexample remains reproduced, not silently relabeled as repaired.

## Numerical counterexample discovered during implementation

Initial arithmetic directly consumed views into immutable byte buffers. A registered
critic fixture (seed 11, one row, after three updates) produced
`0x1.8b0a67f19f16bp-11` instead of the frozen native-array result
`0x1.8b0a67f19f16cp-11`: one ULP, exactly 2^-63. Changing data representation had
changed backend arithmetic behavior despite equal scalar parameter values.

CE-RL-023 preserves the input, parameter values and observed outputs. Arithmetic
now uses private native working copies; publication still uses immutable bytes.
The corrected fixture and registered suites pass, including ten repeated diagnostic
runs after the layout correction. CI checks the corrected path against its frozen
baseline, rather than requiring every BLAS backend to repeat the original raw-view
bit pattern. This is empirical bounded parity; no all-backend bitwise theorem or
floating-point error bound is inferred. No tolerance replaced the bitwise gate.

## Remaining work and recommendation

Obligations 33 and 34 move from open to partially verified, because this component
now has an implemented repair and source-bound concurrency evidence. Their wider
ownership, lifecycle, artifact/promotion and production synchronization scopes remain
unproved; the frozen v1 still has CE-RL-016. The aggregate remains 3 closed, 28 partial,
7 open. Existing capability/default closures were extended to cover this module's
effect and default boundaries. No broader obligation closes merely from adding this
component. Hard inference cancellation, whole-training numerical safety, persistent
recovery and complete runtime refinement remain outstanding.

A stale reproduction sentence that implied unconditional v1 atomicity now explicitly
limits its claim to prepublication numerical failures and links CE-RL-016. Canonical
clauses, ledger, source contracts, defaults, risk entries, README and changelog agree.
RL-OFFLINE-001 remains HIGH/OPEN. Preserve the champion and recommend no adoption.
No market-data retrieval, holdout access, new financial policy training, historical
OPE rerun, live exploration, order, merge or deployment occurred. Existing verification
fixtures and benchmarks exercise synthetic updates only. No new dependency was added.

## Verification receipt and engineering timings

Both `bash scripts/verify.sh formal` and `bash scripts/verify.sh full` exited 0 at
unchanged source commit `61bb43bc`: 150 integrity tests, 44 scoped SMT requirements,
existing and new finite models, Haskell conformance/format/lint/build/smoke/tests,
241 web tests and 185 automation tests. All four CI jobs passed at the same commit;
Docker/deployment jobs were skipped. The final documentation commit contains only
this report and its receipt. Existing linker and web bundle-size advisories remain.

`python scripts/formal/verify.py --require-complete` exited 1 with
`ValueError: research acceptance blocked by open obligations or research evidence gates`.
This is the expected refusal; no mission acceptance or integration readiness is claimed.

Synthetic single-thread workstation benchmarks used 100 updates and 200 forwards
for each output width. Median updates were 0.354/0.337 ms, median forwards
0.0447/0.0433 ms, payloads 5,400/6,216 bytes; model checking took 0.041 s. Observed
maxima and platform details are in the receipt. These are not hard deadlines,
production latency certification, whole-training performance or financial evidence.

[Machine-readable receipt](optimizer-snapshot-v2-evidence-2026-10-03.json) records
commands, source commit, log hashes, tool versions, bounds, CI and benchmarks.
[Formal contract](../../formal/research/optimizer-snapshot-v2-contract.md) and
[proof ledger](../../formal/research/proof-ledger.json) preserve proof classes,
assumptions, implementation links, counterexamples and remaining gaps.
