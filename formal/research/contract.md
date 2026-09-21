# Research assurance scope v1 — specified before implementation

Baseline: `dbd45e26`, 2026-09-20. This work adds verification infrastructure for
the existing rejected offline screen. It does not register a new financial trial,
open a holdout, introduce a policy, or amend the frozen experiment protocol.

## Canonical requirements

`F-RL-SHIELD`: Let `m` be a research mode, `g` five Boolean evidence guards,
`e` elapsed milliseconds and `a` a target, both IEEE binary64. Acceptance is
exactly `m = OfflineReplayV1 AND all(g) AND finite(e) AND 0 <= e <= 20 AND
finite(a) AND a in {-1/4, 0, 1/4}`. Rejection is absence, not an instruction to
flatten. Acceptance preserves the supplied target, including signed zero.
`F-RL-AUTH`: Every admitted proposal has `orderAuthorized = False`.
`F-RL-DEFAULT`: The default mode rejects every target. These refine
`A-SEQUENTIAL-RESEARCH` and the existing `PolicyProposalV1` module, without
changing its semantics. Classification: safety, authorization, numeric,
functional correctness.

`F-RL-LIFECYCLE`: An explicit finite call-interleaving model starts disabled,
allows offline calls only while enabled and not draining, drains already-started
calls, and never gains order authority. Disabling prevents subsequent admission;
an immutable proposal returned from an earlier snapshot is still non-authorizing.
Stopping prevents new starts. Under completion fairness at most two outstanding
calls need completion to reach quiescence. There is no promotion or parameter
update operation. Classification: lifecycle, concurrency, safety, liveness.
This is an **abstract client protocol**, not an implemented server lifecycle.
Its refinement to the production server is open and blocks integration.

`F-RL-CAUSAL`: The existing market-feature slice at integer t accesses only
indices [t-24,t]. A symbol's slice cannot access another symbol's array.
Prefix-fitted scaling and the registered purge/embargo must remain separate from
evaluation. SMT can prove index/partition arithmetic, not source availability or
NumPy execution. Classification: causality, data integrity.

`F-RL-ACCOUNT`: In exact arithmetic, net wealth change is gross price P&L plus
signed funding minus execution costs minus terminal liquidation costs; subtracting
an inventory reward penalty is not a cash debit. This algebraic model does not
prove binary64 accounting, exchange fills, or simulator fidelity. The existing
ledger conformance tests remain necessary. Classification: accounting, numeric.

`F-RL-INTEGRITY`: Proof status has a closed vocabulary, every completed claim has
reproducible evidence and pinned sources, and open critical claims remain visible.
Changed source hashes invalidate receipts. SAT, UNKNOWN, timeout, missing tool,
missing traceability, skipped obligations, and stale expected results fail the
formal command. Classification: integrity, operational.

## Assumptions and refinement boundaries

- A-FP: GHC `Double` implements IEEE binary64 comparisons and classification;
  no unsafe coercion/FFI fabricates a proposal and no asynchronous exception is
  interpreted as acceptance. NaN, infinities, subnormals and signed zeros are in
  the SMT domain. Proofs are over total defined values, excluding Haskell bottom.
- A-GUARDS: Evidence is supplied by trusted deterministic callers. This boundary
  checks Boolean evidence; it does not establish its truth about the world.
- A-SOLVER: Pinned Z3 and its floating-point encoding, Python runtime, GHC and
  hardware are trusted. UNSAT is solver evidence, not an independently replayed
  Lean/Coq proof certificate.
- A-INTERLEAVING: Two callers, atomic abstract start/finish/disable/stop steps;
  snapshots do not grant capabilities. No claim of unbounded concurrency or
  correctness of actual server locks. Completion fairness is required for
  liveness; a hung inference call remains a production blocker.
- A-CAUSAL: Integer timestamps/indices are exact. Historical bar availability
  remains assumed, not observed. No fresh-data or revision witness is invented.
- A-ACCOUNT: Exact real algebra is distinct from binary64 implementation.

The abstraction of a concrete pure call is its mode, five guards, binary64
elapsed/target, and optional returned target/authority. Cross-language fixtures
exercise that relation; generated tests are not a universal compiler refinement
proof. Source pinning requires review after any change but is not semantic proof.
The state model abstracts client calls only; production order ownership,
reconciliation, readiness, persistence and shutdown are outside its proof scope.

## Consistency resolutions

1. The existing formal registry validates coverage and evidence links; it does
   not discharge the prose clauses as theorems. Preserve it and add a distinct
   executable proof gate with explicit status and assumptions.
2. Existing production profiles may authorize their current fleet. The task
   forbids changing those profiles. Report **no authorization introduced**, not
   a false claim that the whole fleet is disabled.
3. The prior RL screen is rejected contaminated-development evidence, not fresh
   out-of-sample confirmation. All protected periods remain protected, including
   the prospective carry embargo through 2027-01-20. No economic rerun is needed
   to verify a deterministic boundary.
4. TCN/PatchTST/Transformer are documented proxies with existing accurate aliases;
   their source bytes have not changed since the September 17 audit. No semantic
   replacement or duplicate migration is warranted.
5. A disabled pure function cannot erase objects held by a caller. Only absence
   of order authority is established for retained proposals; revocation of future
   executable capabilities would require a separate stateful design.

No internal contradiction was found in this affected scope. This is not a proof
of consistency of every prose requirement, runtime and deployment in the repo.
All broader mandatory obligations are tracked explicitly rather than being
silently narrowed to this proof scope. No candidate becomes eligible for review,
shadow, paper or live activation from this verification work.
