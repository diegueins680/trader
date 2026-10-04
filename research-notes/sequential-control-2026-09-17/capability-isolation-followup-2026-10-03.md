# Capability and production-learning isolation — 2026-10-03

Obligations 11 and 24 now have sufficient source-bound certificates for the delivered
research boundary. Together with 31, this gives three affected-scope closures and
35 unresolved obligations. This is engineering progress, not mission completion,
production readiness, or evidence of economic value.

The specification was committed in `9615f5b7` before implementation. Verification
source was frozen at `65957dffeed7e632259b0e93f59d8e7be649ec95`. The latest main
remained `dbd45e2691cb37f1421a23306c43b676fa82e6fc`. Work remains on
`research/sequential-review-2026-09-28`, draft PR #284 stacked on draft #281.

The earlier obligation audit required the whole delivered dependency/effect boundary,
not another isolated false-authority lemma. `ComponentGraph.hs` therefore uses the
actual pinned GHC and Cabal parsers to extract all application modules and executable
roots. No application code runs during extraction. The checker rejects unsupported
source options, source generation, foreign imports, boot imports and build branches.
It traverses the complete finite dependency closure without a depth cutoff.

| Executable | Reachable files | Traversed edges | Maximum shortest depth |
|---|---:|---:|---:|
| trader-hs | 93 | 248 | 3 |
| optimize-equity | 25 | 43 | 3 |
| merge-top-combos | 11 | 12 | 3 |
| analyze-close-timing | 2 | 1 | 1 |
| lstm-bench | 3 | 2 | 2 |
| outbox-publisher | 1 | 0 | 0 |

These overlapping graphs come from 116 sources and 286 local import edges. None
reaches or declares the research proposal module. The module imports nothing and
retains its private constructor, pure proposal API and constant false authority.
Three GHC-negative compilation fixtures reject constructor forgery, coercion and
using the proposal projection as an IO operation. Those fixtures are conformance
tests, not a universal compiler theorem.

The Python finite source audit enumerates `infer`, `Network.forward` and
`_finite_real_vector`. In their admitted ordinary Network/base-array domain they
have only reviewed numeric operations and clock reads, no parameter stores and no
order, network or process operation. Actual inference state snapshots supplement
that audit. It is not a sandbox for arbitrary caller-supplied Python programs.

Z3 checks the exact emitted policy key set against the parsed native LSTM decoder's
required-key predicate: 12 Boolean key-presence variables, a satisfiable premise,
and an unsatisfiable violation. This assumes Aeson's required-field decoding rules.
It prevents unchanged emitted offline policy artifacts from entering native LSTM
persistence; it does not rule out arbitrary administrative JSON conversion.

The ledger, canonical clauses, closure contracts, checker inputs and reverse source
mappings agree. The wrapper reproduces these certificates before admitting closure.
Source/inventory/effect/schema drift fails; future runtime integration must reopen
these obligations. Mutation regressions exercise transitive imports, module roster
changes, missing modules, unsupported syntax/build configuration, new calls/writes,
decoder/writer changes and schema overlap. All 219 source locks are enforced.

A-CAPABILITY-BUILD names trusted compiler, linking, library and runtime primitives,
unchanged installed binaries/PATH, and the absence of injected code, custom array
dispatch, external code mounts and manual policy conversion. Packaging/helper
review is engineering evidence, not verification of a deployed image. The inherited
GHC 8.10.4 optimized Docker profile remains an unresolved operational limitation;
GHC 9.4.8 is the canonical verification toolchain. Existing live authorization and
champion training are outside these closures. Obligations 13, 23, 26 and 28 are
still unresolved, as are numeric, concurrency, timeout and accounting gaps.

No financial experiment, market-data retrieval, protected-holdout access, OPE rerun,
policy training campaign, simulator change or production modification occurred.
The full wrapper still exercises existing synthetic training fixtures. Prior rejected
candidates and numerical counterexamples remain rejected/unresolved; no new market
claim follows. No new runtime dependency, flag, API or artifact semantics was added.

Recommendation remains **no adoption**. Next priority is actual repair and composition
of versioned atomic publication and cancellable inference, followed by numeric and
data-admission refinement; disconnected safe kernels cannot close those obligations.

Both `bash scripts/verify.sh formal` and `bash scripts/verify.sh full` exited 0 at
the unchanged source freeze: 139 formal integrity tests, 42 SMT obligations,
existing finite models/Haskell conformance, Haskell formatting/lint/build/smoke/test,
241 web tests and 185 automation tests. All four GitHub CI jobs passed at the same
commit; Docker and deployment jobs were skipped. The final evidence commit adds
only this report and the linked receipt. Verification's ordinary linker and web
bundle-size advisories remain; there is no failure hidden by them.

`python scripts/formal/verify.py --require-complete` exited 1 with
`ValueError: research acceptance blocked by open obligations or research evidence gates`.
This is an expected refusal, not a passing mission-acceptance result. No proof
placeholder or skipped assertion is used for the scoped verification pass.

[Machine-readable receipt](capability-isolation-evidence-2026-10-03.json) records
exact commands, source commit, log hashes, statuses, bounds, test counts and CI.
[Source-isolation contract](../../formal/research/capability-isolation-contract.md)
and [all-38 audit](../../formal/research/obligation-closure-audit.md) define the scope.
