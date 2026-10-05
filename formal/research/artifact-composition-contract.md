# Artifact admission composition v1

Preregistered 2026-10-05 against main09d3bdda237ce8a5a2f85fce9d2ce974738bf279.
Candidate closures:29 (compatibility) and30 (hash/provenance mismatch rejection).
Their original scope and closure criteria must remain unchanged. A failed or
incomplete composition retains its blocker; certificate counts cannot close it.

## Conflict and intended behavior

The v1 loader validates expected_provenance, performs file I/O, then serializes
that same caller-owned dictionary for comparison. A concurrent caller mutation
during I/O can replace the expected identity. Canonical requirement30 requires a
mismatch to reject; its next action explicitly calls for immutable expected inputs.
The authoritative identity is the immutable canonical JSON snapshot captured
before opening the artifact, not whichever dictionary contents exist after I/O.

Add a private JSON snapshot helper and bind its result at loader entry. Validate
the decoded private snapshot; compare artifact provenance against that immutable
string. Preserve the public function signature, v1 wire format, supported values,
parameter representation and numerical behavior. Under stable valid ordinary JSON
inputs, acceptance and returned parameters must equal the old loader's behavior.
Malformed expected inputs must fail before I/O. Preserve the original loader as a
small deterministic regression fixture, not an alternative active entry point.

The archive remains frozen at its original Git/source identities; no financial
campaign is rerun or relabeled. No training/inference/reward/selection function is
changed. This narrowly specified admission correction supersedes the prior blanket
source-freeze wording only for load_policy and its private snapshot helper. Existing
counterexamples and old-run provenance remain historical evidence; reproductions
of old runs must use their recorded commits, not substitute patched source hashes.

## Formal domains and obligations

A byte snapshot B is immutable. H(B) is the installed SHA256 function's output;
no hash implementation, collision-resistance or authenticity theorem is claimed.
P is the canonical JSON encoding of ordinary provenance data. E is its immutable
expected snapshot captured before artifact read/hash/decode. V is the supported
version tuple, including implicit v1 feature/action/reward semantics supplied by
the unchanged environment and observation constants. V4 declares its tuple explicitly.
Returned(B) implies H(B)=expectedDigest AND P(provenance(B))=E AND versions(B)=V.

F-RL-ARTIFACT-COMPOSITION: enumerate actual reader/helper/bridge use and source-bound
control/data bindings, including single-read bytes, expected identity capture,
private parameter construction, v4 provenance copy, native schema exclusion and
non-authorizing Haskell process/proposal consumers. Review all primitive effects;
AST/source hashes detect drift but do not alone establish refinement.
F-RL-ARTIFACT-IDENTITY: SMT verify source-derived digest/provenance predicates and
supported outer/nested version requirements for both formats. Translate actual
predicates; do not replace unproved helpers by an assumption that admission is safe.
F-RL-ARTIFACT-SNAPSHOT-FLOW: exhaustively model concurrent caller provenance and
path replacement before/after snapshot, read, hash, decode and publication; returned
identity must match the captured expectation and exact bytes read. Preserve a
legacy counterexample. Finite identity classes abstract arbitrary equal/unequal
byte/hash/provenance values; cryptographic authenticity is separate.
F-RL-ARTIFACT-COMPOSITION-CONFORMANCE: actual old/new loaders under deterministic
interleavings, immutable v4 expected identity after capture, malformed/unsupported
versions, digest/provenance mismatches, parameter helper validation, stable-input
compatibility, and mutations removing a gate or changing a source binding.

## Refinement and assumptions

A-ARTIFACT-COMPOSITION: pinned CPython3.13.3, JSON/hashlib/NumPy and GHC/base semantics;
ordinary builtin JSON values/base numeric arrays, fixed trusted modules/helpers,
no hostile subclasses, runtime monkeypatching or manual conversion into a native
production artifact. Canonical JSON capture is one non-interleaved C-encoder
operation on ordinary builtin objects under the pinned GIL runtime. Captured strings
and bytes are immutable; library hash/decode consume exactly those inputs. V4 copies
an ordinary dictionary whose admitted values are immutable strings. The reference
identity is supplied by the caller and may be factually false; the theorem proves
matching, not provenance truth or scientific validity. File reads can return changed
or torn bytes; only the actual bytes read may be hashed/decoded. Hostile external
code or filesystem writes are not granted authorization by these loaders.

No whole-language/compiler/library proof, OS sandbox, crash-atomic artifact storage,
production-wide ownership/recovery/authorization theorem or performance/economic
claim. Source-to-primitive correspondence remains a named reviewed assumption,
backed by actual-code conformance. Keep numeric, freshness, lifecycle and economic
obligations unresolved unless separately discharged. Existing v1 counterexamples
must not be erased, and no final holdout may be read.

Use the current pinned toolchain and CI wrappers. Reproduce all composing certificates
in the same invocation, reject unknown/missing results, run formal/full, remove the
temporary receipt workflow, require final-head CI, then merge without deployment.
