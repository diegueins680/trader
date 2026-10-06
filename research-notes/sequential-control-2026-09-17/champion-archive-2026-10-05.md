# Champion archive preservation — 2026-10-05

Engineering registration f1ece294 precedes implementation on main b706fa9b.
No market experiment, trained checkpoint, holdout access or live operation occurs.
The unchanged obligation 28 criterion is: "Challenger failure cannot modify
champion configuration/artifacts or selection."

## Conflict, repair and refinement

The screen runner already creates its JSON, event, return and policy files
exclusively. The report exporter created a fresh directory but then used
`write_bytes`, which opens an existing leaf for truncation. Directory creation
alone does not establish exclusive ownership of files inserted afterward. A
scheduled insertion of a regular file, symlink or hardlink reproduces the old
failure in the actual full exporter. These are synthetic engineering witnesses,
not evidence that production champion files were overwritten.

The exporter now opens every report leaf with `xb`; collisions raise
FileExistsError. No existing file is removed to retry. All seven successful
report byte streams match the legacy implementation. No schema, filename,
financial result, model identifier, training rule or current selector changes.
A failed write can leave a newly created partial report directory; a retry at the
same path rejects it. This is intentional preservation, not durable recovery or
whole-archive atomic publication.

The abstraction maps ordinary pathname bindings to file identities and maps
identities to opaque byte strings. Exclusive creation either refuses an existing
leaf or returns a handle to a new identity. Writes through that owned handle
cannot change an existing protected identity. Symbolic keys need not be disjoint:
choosing an existing champion pathname causes rejection. Leaf collisions during
publication are modeled; stable parent/open-object identity is a named assumption.

## Machine-checkable evidence

- F-RL-ARCHIVE-SOURCE checks all five actual exclusive writer sites, the full
  12-module/15-import/1,737-call research inventory, six callback bindings and
  fixed output destinations. Git subprocesses are limited to source identity;
  the separate Haskell process protocol and all six production roots retain
  their own required certificates. Hash equality detects drift; reviewed
  primitive effects and source-to-model correspondence remain assumptions.
- F-RL-ARCHIVE-PRESERVE checks eight satisfiable-premise/UNSAT queries over
  unbounded integer keys, object identities and opaque byte values: existing
  bindings/bytes, rejected collisions, new-handle ownership, retries, unrelated
  keys and distinct allocations. This is an SMT result for the stated filesystem
  semantics, not a proof of the operating-system kernel.
- F-RL-ARCHIVE-FLOW reaches a fixed point with two writers, two keys, four object
  identities and six content classes: 110 states, 359 transitions, maximum
  shortest depth 7. Retry cycles have no depth cutoff. Partial writes and failures
  are reachable. No liveness theorem follows from this preservation invariant.
- The preserved legacy model reaches 242 states and 815 transitions, depth 9.
  CE-RL-ARCHIVE-COLLISION is `insert-collision → select report → open/truncate`.
  The legacy source and tiny synthetic archive are committed regression fixtures.
- Sixteen actual conformance cases cover byte compatibility, three collision
  kinds in both versions, partial write/retry and the actual JSON/policy writers.
  Thirty-two generated existing-directory cases add property-test evidence.
  Separate automation tests exercise the complete exporter and expected bytes.
  These tests are not described as mathematical proofs.

## Closure composition and assumptions

Obligation 28 requires the archive source, SMT and model certificates together
with freshly reproduced promotion metadata/schema/flow, complete callback/effect
coverage, proposal types, process boundaries, snapshot/successor/codec boundaries,
native artifact disjointness and all six production executable exclusions. None
of those constituents alone closes champion preservation. The canonical verifier
refuses closure if any required constituent is missing or fails in that invocation.

A-CHAMPION-ARCHIVE names CPython/pathlib/io/exclusive-create semantics, ordinary
built-in objects, fixed reviewed code and registered scalar names, trusted
libraries/helper programs, and stable parent-directory/open-object identities.
It excludes malicious namespace replacement, privileged filesystem mutation,
external deletion/recreation of champion data, code injection and manual adoption
of unfinished research output. It does not assume output paths are disjoint from
existing champion files. Existing production learning by its own authorized code
is outside the candidate-caused preservation claim. No OS sandbox, deployed-image,
filesystem durability or correctness of inherited selection algorithms is claimed.

The protected inherited integration boundary is that candidate outputs cannot
invoke a production selector, replace its existing configuration/artifacts, or
enter the native model loader unchanged. Production ownership, draining,
reconciliation and recovery require their separate unresolved proofs.

All original 38 scope and closure-criterion strings are unchanged. After
successful fresh canonical reproduction, obligation 28 is conditionally
`exhaustively_checked` through the specified composition: **10 scoped closures,
24 partial, 4 open**. Overall formal and mission completion remain false. Open
obligations remain 4/10/37/38; this repair does not close crash recovery.

## Research and safety disposition

No candidate passed and no challenger is adopted. RL remains offline research.
The frozen 108 fits / 19,440 replays / 19,548 registry rows remain contaminated
development; all 108 OPE batches remain invalid. The 1,227 final returns remain
sealed and the prospective embargo stays 2027-01-20T13:00Z. Costs, statistical
results, drawdowns, tails, policies, inference benchmarks and literature findings
are unchanged. This fixture contains no real market observations.

No production Haskell code, authorization flag, environment configuration,
deployment identity, reviewed fleet, adopted UUID, leverage, ownership setting or
risk limit changes. No authenticated endpoint, order or live exploration occurs.
The user-owned original worktree and its running services remain untouched.

## Verification and delivery

Targeted proof/conformance and exporter tests pass locally. Pinned CI reproduced
the complete certificate and passed both canonical wrappers; details follow.
Final-head checks and merge-SHA deployment audits are recorded in PR 307. Merge
only the tested tree and suppress deployment triggers.

Local `bash scripts/verify.sh automation` passed all 185 tests (51.106 seconds).
The initial local formal wrapper ran 224 tests in 161.027 seconds and failed the
old explicit closure roster (`28 != 29` unresolved); all other tests passed.
The roster now names the original nine closures plus 28. An added validator
mutation test removes each required constituent in turn and requires rejection;
the new archive/closure subset passes 11 tests in 2.315 seconds. No proof or
economic threshold was weakened. Superseded runs were cancelled and are not counted as successful checks.


Pinned [run 37390460978](https://github.com/diegueins680/trader/actions/runs/37390460978)
checked exact source commit `d7a23b53d6559c3912380589577904bd91fb4088`:

- `python3 scripts/formal/verify.py --record`: passed; 68 seconds.
- `bash scripts/verify.sh formal`: passed; 130 seconds; 225 integrity tests.
- `bash scripts/verify.sh full`: passed; 490 seconds; 225 integrity tests,
  Haskell build/format/lint/smoke/test suite, 241 web tests and build, and 185
  automation tests. Integrity test durations were 61.148 and 60.447 seconds.
- All 74 SMT requirement groups reproduce; the added archive group contains
  eight satisfiable-premise/UNSAT obligations. Results retain per-property status.
- Reproduced receipt SHA256:
  `9cbde36821f1d9968c290569265ae7e1e87a57f7c13ae2fdb8ef522b1828ca5a`.
  Imported byte-for-byte from the job log, with every source hash checked against
  the toolchain manifest. Only archive, promotion, closure, SMT and source-hash
  sections differ from the previous receipt. No result was manually synthesized.

The temporary reproduction workflow is removed from the delivered tree. Final
head CI must independently reproduce this committed receipt. Aggregate results
still say `formalObligationsComplete=false` and `missionComplete=false`.
This is a conditional affected-scope closure, not overall operational readiness.
