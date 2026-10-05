# Artifact admission composition — 2026-10-05

Registration07df0efa precedes implementation, against main09d3bdda. The v1 loader
validated expected provenance, yielded during file I/O and then compared against
the caller's mutable dictionary. The preserved loader demonstrates acceptance of
artifact B after the expected dictionary changes from A to B. The repair captures
canonical JSON before I/O, validates its private decoded snapshot and compares
against that immutable string. Only this loader and a private helper change at
runtime. Stable-input v1 format, identifiers, parameters, training and inference
semantics remain unchanged. Existing archived financial runs keep their original
commit/source identities; reproduction must check out those recorded commits.

## Composition and refinement

| Actual operation | Abstract interpretation and checked connection |
| --- | --- |
| v1 `_provenance_identity` | Ordinary builtin JSON is encoded by the pinned C encoder; private decode is validated; immutable string is returned before opening the file. |
| v1 `validate_provenance` | Required hashes, integer domains, algorithm/horizon and serializable finite JSON are checked. Extra JSON fields preserve old compatibility. |
| v1 `load_policy` | Exactly one bounded read supplies both hash and JSON decode. Digest mismatch raises before decode; duplicate/nonfinite JSON, fields/types and metadata reject before parameter construction. |
| v1 `_parameter_snapshots` | Exact keys/shapes and numeric real array kinds; copies and finite portable conversion; validated arrays replace every initial random parameter before return. |
| v4 `_provenance` | Exact dictionary/string domains; private shallow copy of immutable admitted strings. Snapshot precedes hashing/decode. External mutation after capture cannot change the reference. |
| v4 `decode_artifact_v4` | Immutable bytes, size/enablement/digest gates; unique-key decode; exact field/contract/provenance comparison; restore and canonical-byte check before return. |
| v4 `_restore`, `_snapshot`, `_buffer` | Version/type/shape/step checks dominate immutable state construction. Finite hex buffers, nonnegative second moments and matching actor/critic outputs are checked. No unchecked metadata object is returned. |
| v4 request and Haskell consumers | Existing bridge/proposal/process certificates are freshly required. Native persisted-LSTM required-key exclusion and all six production-root component checks remain required; neither research format can activate through the native decoder. |

AST locks enumerate13 reviewed reader/helper definitions and detect drift. They
are not alone a proof of implementation semantics. Actual predicates are translated
to SMT; the reviewed primitive/data-flow relation maps capture/read/hash/decode/
metadata/publication to the finite transition model. This correspondence explicitly
trusts pinned CPython3.13.3 ordinary builtins/GIL/C encoder, JSON, hashlib, NumPy and
GHC/base semantics. No hostile subclasses, runtime monkeypatching or manual native
artifact conversion. SHA outputs are compared; neither collision resistance nor
truth/authenticity of caller-supplied provenance is proved. Interrupted operations
may fail rather than return: no termination or crash-atomic persistence theorem.
The model permits arbitrary repeated path/caller replacement, including torn read
bytes represented by the returned byte identity. Scheduling instrumentation in the
regression tests is only a way to force the ordinary concurrent write deterministically.

## Exact checked scope

- Six SAT-premise/UNSAT-violation SMT queries: v1 digest/metadata, v4 digest/metadata,
  restored policy version and nested optimizer version. Outer v4 contract tuple
  equality covers all seven declared semantic versions; actual-code mutations test
  every component. Unrelated type/configuration predicates are unconstrained in
  version lemmas, so they cannot hide a version-check counterexample.
- Fixed model:848 reachable states,9072 edges,maximum shortest depth10,32 accepted
  states. Two caller equivalence classes, eight file classes (digest,provenance,
  supported version), fixed expected-digest class0. All reachable-state edges are
  enumerated to a fixed point; no retry cutoff. A bounded abstraction is not a
  whole-runtime proof. It verifies equality/control routing under the named relation.
- Legacy model:944 states,10064 edges,depth11,64 accepted states; preserved
  CE-RL-ARTIFACT-PROVENANCE-RACE and original loader from09d3bdda.
- Actual-code conformance: legacy race reproduced, fixed mismatch rejected, v4
  captured mismatch rejected, file replacement after read leaves the original
  hash/decode result intact. Three stable-input compatibility cases (including
  nested metadata and tuple-to-JSON compatibility);29 invalid/reference/version
  cases. Existing codec suites provide the broader numeric/generated-bit coverage.
- New mutation tests reject skipped snapshot validation, late provenance capture,
  rereads during decode, unvalidated parameter publication, mutable v4 reference,
  unvalidated returned record and removed nested-version gates. Solver mutation
  tests bypass AST locks to ensure omitted predicates produce counterexamples.
- Source inventory now has1736 call sites (formerly1733); no new module import,
  file destination, callback, public entry point, external service or dependency.

## Traceability and decision

Unchanged criteria29/30 now require ARTIFACT-COMPOSITION, IDENTITY, SNAPSHOT-FLOW,
existing v1/v4 gate models, v4 helper boundary, bridge codec/boundary, promotion
surface/native-schema exclusions and component/process/snapshot isolation. Every
formal constituent must reproduce in the same verifier invocation. Conformance
and property tests supplement those certificates. Scope remains the delivered
research paths and explicitly named inherited integration boundary, not all existing
production artifacts or production correctness. Reopen before any added loader,
consumer, format, helper, schema or activation path.

**9 scoped closures,24 partial,5 open.** `formalObligationsComplete=false` and
`missionComplete=false`. Other numeric, causality, lifecycle, ownership, recovery,
reliability and economic gates remain unresolved. No winner integrated. General
recommendation:no candidate passed. RL recommendation:continue offline research.
Frozen108 fits/19440 replays/19548 rows remain contaminated development evidence;
all108 OPE batches remain invalid;1227 final returns stay sealed. No matched
champion confirmation, new OOS/holdout results, market-data reads, cost/stress/tail
or performance claims. Prospective embargo2027-01-20T13:00Z remains unchanged.

## Verification and delivery

The new targeted conformance and solver/model mutation tests pass locally. Canonical
formal/full results and final-head CI are recorded below after execution. No live
flags, authenticated trading endpoints, orders, exploration, deployment, model
promotion, fleet/risk settings or financial experiments are changed.

Pinned reproduction [run37376815890/job111987880859](https://github.com/diegueins680/trader/actions/runs/37376815890/job/111987880859)
on exact head0d9ee36a86f736adf765fbaafc2d84a2890868e7 passed:

- Receipt generation21:39:00–21:40:05UTC;72 named SMT groups.
- `bash scripts/verify.sh formal`,21:40:05–21:41:59UTC;216 tests in49.345s.
- `bash scripts/verify.sh full`,21:41:59–21:49:40UTC;216 formal tests in52.029s,
  Haskell trader-tests,241 web tests and185 automation tests passed.
- Receipt SHA256`b8dc0f322a1cf16cca2e32d2ba641d1e41a382e63c7b439c42dff92fdf2016eb`.
  Imported byte-for-byte from the completed job log. Only artifactComposition,
  obligationClosure,promotionBoundary,smt,sourceHashes differ from the old receipt.
  All existing SMT results remain unchanged; every locked source hash matches disk.

Local `TRADER_FORMAL_PYTHON=/Users/diegosaa/.cache/trader-proof-20261003/bin/python
PATH=/private/tmp/node-v20.19.0-darwin-x64/bin:$PATH bash scripts/verify.sh formal`
initially failed a stale closure-list assertion and the PPO successor's stale helper
hash. Both were corrected after confirming every other learning-module statement
unchanged from09d3bdda. The local retry ran216 tests in315.645s and failed the
existing process fixture with `ValueError: PPO process bridge: no actual inference
for trained policy`. That path permits deadline-driven absence; the precise local
cause was not established. This is not a local pass. The pinned runner independently
passed the unchanged assertion/deadline in both wrappers. No check was skipped,
weakened or replaced. Ordinary pre-receipt CI ran216 tests successfully, then rejected
the stale committed receipt as intended; final CI must verify the imported receipt.
The temporary reproduction workflow is removed before final-head CI and merge.
