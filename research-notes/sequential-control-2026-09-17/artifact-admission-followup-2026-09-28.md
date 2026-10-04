# Offline artifact admission continuation — 2026-09-28

**No adoption; the broader mission remains incomplete.** This continuation adds
verification of the existing offline policy loader. It changes no loader, model,
policy, artifact schema, experiment, dataset, holdout access, production setting
or champion. No new empirical performance or out-of-sample evidence is claimed.

This receipt records the artifact-admission stage. The later
[training-transition continuation](transition-admission-followup-2026-09-28.md)
adds one scoped SMT requirement and its own verification receipt; earlier stage
results remain unchanged.
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
rejected. These are deterministic regression tests, not a proof of all JSON inputs or
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

Source freeze: `91273c9f`; report revision: `2cbba834`. Subsequent receipt changes
are documentation only. Exact commands from the isolated worktree:

```bash
export PATH=/private/tmp/node-v20.19.0-darwin-x64/bin:$PATH
export TRADER_FORMAL_PYTHON=/private/tmp/trader-proof-20260928/bin/python
"$TRADER_FORMAL_PYTHON" scripts/formal/test_integrity.py ArtifactAdmissionTests
bash scripts/verify.sh formal
bash scripts/verify.sh full
"$TRADER_FORMAL_PYTHON" scripts/formal/verify.py --require-complete
```

Pinned tools: Python 3.13.3, NumPy 2.3.5, z3-solver 4.15.4.0 (solver 4.15.4),
GHC 9.4.8, Cabal 3.12.1.0, fourmolu 0.15.0.0, hlint 3.8 and Node 20.19.0.

- Targeted artifact suite: **exit 0**, six tests, 0.933 seconds.
- Formal wrapper: **exit 0**, 29 tests, 16 scoped SMT requirements; loader model
  27 states/40 transitions/depth 13 and existing lifecycle model 75 states/349
  transitions/depth 6. Existing 20,480 Haskell conformance cases and 180 rational
  accounting traces also pass. Standalone verifier: 5.992 seconds.
- Full wrapper: **exit 0**, formal checks, Haskell build/format/lint/smoke/tests,
  web typecheck/241 tests/build and 185 automation tests passed, none skipped.
  Automation took 43.105 seconds; the scoped verifier reported 8.139 seconds.
  No retries, disabled checks or altered timeouts were needed for this stage.
- Remote [CI run 36434475535](https://github.com/diegueins680/trader/actions/runs/36434475535)
  at `2cbba834`: formal, Haskell, web and automation passed. Docker build and
  deployment were skipped.
- Acceptance diagnostic: expected **exit 1**, `ValueError: research acceptance
  blocked by open obligations`. All 38 broader obligations remain open/partial.

Logs remain outside Git. SHA-256 receipts, with prefix
`/private/tmp/trader-artifact-` and suffix `-20260928.log`:

| Log | SHA-256 |
| --- | --- |
| formal | `918b24878dac99bc416225f5c9ac0418bcd4f1b09e831918842f3c262ed41ef5` |
| acceptance | `e509ee4ce17b660ea239f105f107bb6969ba0c19f404ea31f3193cb7045c7d1b` |
| full | `bc7b551907eac0e9f8e8424b6d8d50a8e16ae00cb31ba4831e6be98fdc94e310` |

No live authorization, authenticated exchange experimentation, order, live-money
exploration, holdout access, merge, deployment or champion change occurred.
No proof placeholder was introduced; unresolved obligations remain explicit.
