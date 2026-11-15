# Source-linked offline artifact admission — 2026-09-28

Specification recorded before implementation. This assurance extension changes no
loader, artifact version, network, candidate eligibility or frozen experiment.

## Requirements

**F-RL-ARTIFACT-PATH (model_checked).** In the finite admission abstraction derived
from the audited `load_policy` top-level statement sequence, every path to a
returned network has passed expected-provenance validation, bounded snapshot read,
size, digest, JSON parsing, field/type checks, artifact-provenance validation,
compatibility, parameter conversion, parameter snapshot validation and construction.
Any failing gate is terminal and cannot reach return. Under terminating primitives,
every path ends in failure or return. No retry or authorization transition exists.

**F-RL-ARTIFACT-METADATA (smt_verified).** Translate the actual incompatibility
predicate's supported Boolean/comparison syntax into Z3. Passing it entails the
v1 schema, environment, observation and rejected-research disposition, exact false
enabled marker, equal actions and equal canonical provenance. Passing the actual
hash-rejection predicate entails equal snapshot and expected digests. Require SAT
passing cases and UNSAT violations. Unknown syntax fails certificate admission.

## Source connection and assumptions

A-ARTIFACT-ADMISSION: ordinary Path/bytes and JSON built-ins; stable globals,
expected provenance and primitive bindings; no hostile subclass, monkeypatch or
concurrent mutation; Python exception/control-flow semantics, JSON, hashlib,
NumPy, validation helpers and Network construction are trusted. Expected hashes
and provenance are externally supplied, not authenticated by this theorem.
Digest equality is not proof of authenticity, cryptographic collision resistance,
correct training, complete provenance, economic value or promotion eligibility.

Bind the entire loader AST to an explicitly audited source skeleton, including
its local JSON duplicate-key and numeric validators. Extract the compatibility
and hash predicates from the actual source, and verify their guard placement and
raising branches. No exception-catching or earlier successful return is allowed
by the skeleton. Primitive gates are atomic nondeterministic pass/fail steps;
state exploration overapproximates reachable outcomes. The abstraction relation
maps completion of each admitted statement group to its corresponding gate.
Audited source recognition plus model checking is not a machine-proved Python
interpreter, a proof of helper internals or full implementation refinement.

The metadata SMT domain abstracts Python string equality, exact-false identity,
action equality and canonical-provenance equality under the preceding type gates.
It does not prove canonicalization injective or prove neural outputs finite.
Source hashes tie the assumptions and conformance fixtures to this revision.

## Counterexamples, conformance and delivery

Preserve deliberate digest-bypass and enabled-marker-bypass source mutants;
show that the checker rejects them and that actual compiled mutant functions
admit the offending small synthetic artifacts. These are mutation fixtures, not
claims of defects in the unchanged loader and not financial trials.

Exercise actual load/save code with deterministic untrained networks and small
JSON artifacts in temporary directories. Include valid round trips, wrong digest,
version/action/provenance drift, enabled non-false values, malformed JSON, duplicate
keys, invalid parameters, non-finite/overflowing values and the size boundary.
Check that failures occur before Network construction where applicable. Distinguish
these property/regression tests from the abstract model and SMT results.

Map both requirements bidirectionally through the canonical specification, ledger,
assumptions, source checker, fixtures, tests, results and existing formal/full CI.
Keep whole-mission obligations open/partial; no claim of Haskell artifact admission
or production authorization follows. Keep prior receipts and economic results.
