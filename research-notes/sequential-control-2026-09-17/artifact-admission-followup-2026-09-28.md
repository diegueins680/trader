# Offline artifact admission continuation — 2026-09-28

**No adoption; the broader mission remains incomplete.** This continuation adds
verification of the existing offline policy loader. It changes no loader, model,
policy, artifact schema, experiment, dataset, holdout access, production setting
or champion. No new empirical performance or out-of-sample evidence is claimed.
Latest remote main remains `dbd45e26`; work continues on the isolated branch
`research/sequential-review-2026-09-28` in draft PR #284, stacked on #281.

## Specification and source connection

The [contract](../../formal/research/artifact-admission-contract.md) was committed
at `e12e0bd4` before implementation. Existing admission behavior is authoritative:
`load_policy` accepts only the existing v1 research artifact, explicit false
`enabled`, rejected-research disposition, compatible metadata, equal supplied
digest/provenance, and valid parameter snapshots. This work adds evidence and
resolves a missing verification link; it does not change any identifier semantics.

`F-RL-ARTIFACT-PATH` binds the complete loader AST to an audited skeleton, with two
predicate slots translated separately. Thirteen ordered statement groups become
atomic nondeterministic pass/fail gates. Local function definitions and an empty
dictionary allocation are not validation gates. Unexpected returns, rebinding,
changed read bounds, skipped validation or new control flow fail source admission.
Each completed concrete statement group maps to its corresponding abstract gate;
primitive return/exception semantics and the checker itself are trusted.

Exhaustive exploration finds **27 states, 40 transitions and maximum depth 13**,
including terminal stuttering. Every returned state has all gates passed, failures
are absorbing, and terminality follows within 13 gate completions when primitives
terminate. This is not a wall-clock timeout guarantee. It does not compose with
the separate 75-state lifecycle abstraction to form a full system proof.

`F-RL-ARTIFACT-METADATA` translates the actual incompatibility and hash-rejection
predicates into Boolean/string SMT expressions. Two satisfiable passing cases and
two UNSAT negated implications establish the stated metadata/digest implications.
Exact false identity is distinct from Python numeric equality: an `enabled` value
of integer zero is rejected. Unsupported syntax fails the checker. The scoped SMT
requirement count increases from 15 to **16**, not to 17: the other new requirement
is model checked.

## Counterexamples and implementation conformance

[CE-RL-007/008](../../formal/research/artifact-counterexamples.json) are deliberate
source mutants, not bugs found in the unchanged loader. Removing digest rejection
admits a valid synthetic policy with an incorrect expected digest. Removing exact
false enforcement admits an otherwise matching artifact with `enabled=true`.
Actual compiled mutant functions return networks in these fixtures; the original
loader rejects both, and the new source-derived SMT check rejects both mutations.
A returned research network still does not imply order authority.

Six new tests cover source/control-flow drift, skipped model gates, escaped failure
states, false/vacuous/unsupported predicates and both preserved mutants. Actual
load/save conformance uses only a small, deterministic, untrained network in a
temporary directory. Thirty-two malformed variants plus a wrong expected digest
are rejected before Network construction. Cases include version/action drift,
non-false enabled values, changed provenance, invalid/non-finite/overflowing
parameters, invalid JSON, duplicate keys and oversize bytes. Valid round trips
preserve parameters, 65,536-byte padded JSON is admitted, and 65,537 bytes is
rejected. These are regression/property tests, not a proof of all JSON inputs or
helper behavior. No trained artifact or large dataset is committed.

## Assumptions and unresolved properties

`A-ARTIFACT-ADMISSION` records ordinary Path/bytes/JSON values, stable global and
expected-provenance bindings, no hostile subclass or concurrent mutation, trusted
Python exception/control-flow semantics, JSON/hashlib/NumPy, helper validators and
Network construction. String equality, action equality, exact-false identity and
canonical-provenance equality are abstraction primitives. No interpreter/compiler,
canonicalization-injectivity, cryptographic-collision or helper-internal proof is
claimed. The checker audits one supported source skeleton, not arbitrary Python.

Expected digests and provenance come from the caller; a match proves neither
identity/authenticity nor that the claimed training occurred. The artifact format
still lacks the complete future production proof/provenance admission contract
requested by the broader mission. Inference finiteness, timeout behavior, neural
region verification, Haskell artifact admission and production lifecycle/refinement
remain outside this certificate. Accepted research artifacts are not promoted.

The canonical registry, proof ledger, source locks, model results, risk register
and CI map both new requirements to code and tests. Mission obligations 29/30 gain
partial evidence; default-disabled evidence is also linked. **All 38 whole-mission
obligations remain open/partial.** `RL-OFFLINE-001` remains HIGH/OPEN. The existing
negative financial results, invalid OPE, missing matched champion comparison,
contaminated development periods and sealed holdouts remain unchanged.

## Verification receipt

Pending final formal/full commands. Targeted six-test artifact suite passed;
this is not yet a passing full-verification claim. Final receipt will preserve
failures, exact source revision, commands, versions and log hashes.
