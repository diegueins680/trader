# Delivered research capability boundary v1

Specified 2026-10-03 at source 8700dbb8, before implementation.

## Intended closures and scope

Candidate closures are mission 11 (capability separation) and 24 (no production
learning), for the newly delivered research policy/proposal implementation. The
claim does not forbid the champion's pre-existing LSTM training. It does not
certify existing live-order authorization, ownership, reconciliation or server
concurrency. No other mission obligation closes merely because these two do.

Use the pinned GHC 9.4.8 parser and its bundled Cabal 3.8.1.0 parser to extract
actual local module imports and all six executable roots. No regex import search
is a proof. Reject unhandled source/build constructs, conditional Cabal component
branches, source preprocessors, compile-time code generation, foreign imports,
package-qualified/local boot imports, missing local modules and duplicate names.
Enumerate the complete reachable fixed point. No executable's dependency closure
or declared module roster may contain Trader.Research.PolicyProposalV1. The
research module imports nothing and exposes only the reviewed pure proposal API.
GHC compilation must reject constructor forgery, representational coercion from
Double and use of the proposal itself as IO/order-mode authority.

F-RL-COMPONENT-ISOLATION: in the source-derived finite dependency graph, all
executable roots exclude the research proposal module. The graph is a build
reachability claim; it is not a control-flow graph of the server.

F-RL-POLICY-EFFECTS: for ordinary admitted Network objects and base numeric
arrays, the source-derived reachable inference helper call set has only reviewed
numeric operations and clock reads; proposal generation neither writes parameters
nor invokes order, process, network, artifact or authorization operations. The
Haskell source surface is pure, has a private constructor and constant false
orderAuthorized. Caller-supplied arbitrary Python programs, runtime monkeypatching
and replacement packages are not policy artifacts and are outside this claim.

F-RL-ARTIFACT-DISJOINT: extract the exact top-level keys emitted by save_policy
and the required fields of the native PersistedLstmModel decoder, including its
jsonOptions transformation. Prove with SMT that the emitted object cannot satisfy
the native decoder's required-key predicate. This is about unchanged emitted
artifacts, not arbitrary user-transformed JSON or every API input format. Loading
an admitted offline artifact constructs numeric Network data, not executable code.

## Refinement and assumptions

A-CAPABILITY-BUILD: GHC parsing/type rules, bundled Cabal parser, standard linking,
Aeson generic required-field decoding and Python/NumPy primitives are trusted.
Sources, compiler options and package identities are fixed; no injected runtime
code, custom ndarray dispatch, malicious PATH/binary replacement, external code
mount or manual artifact rewriting. Explicit administrative execution of research
training is outside production execution. This is not an OS sandbox theorem.

Concrete-to-model abstraction maps each parsed import to an edge, each parsed
Cabal executable to a root, and each JSON key to its membership bit. All relevant
sources and the complete file inventory are included in deterministic receipts.
Graph mutations must produce a path or rejection, not silently drop an edge.
Inference source admission checks every traversed helper body/call and rejects
unknown calls or writes; primitive purity is named, not proved from NumPy C code.
The existing SMT authority/default proofs and compiled conformance remain required.

## Consistency and operational limits

Dockerfile uses GHC 9.4.8. Dockerfile.optimized still uses GHC 8.10.4. AGENTS.md,
.tool-versions and CI define 9.4.8 as the authoritative verification toolchain.
The legacy profile is not certified buildable by this audit. Do not modify it in
this research task; retain an operational limitation and separate follow-up.
Both recipes' runtime COPY sets are inspected for newly introduced research code;
recipe inspection is engineering evidence, not proof of a deployed image. No
Docker image is built or deployed, and no actual fleet state is asserted.

Production subprocess boundaries include trusted trader/optimizer binaries,
Cabal discovery, git metadata and the pre-existing cast helper. Import exclusion
alone does not prove those external programs safe. The delivered research code
introduces no new process target or runtime integration; comparisons bind their
unchanged source paths. No research policy is installed in a production selector.

## Verification and refusal

Model checks enumerate reachable dependency nodes to a fixed point; output bounds
and counts are measured and recorded, never invented before running. SMT uses the
existing pinned solver and independent satisfiable-premise/UNSAT-violation queries.
Malformed/parser-unsupported input, changed source inventory, unknown effects,
artifact key overlap, unexpected type-check success, missing proof or solver
UNKNOWN/timeout refuses closure. Tests include transitive import insertion,
Cabal roster insertion, parser-hostile imports, unknown effect/write injection,
artifact-field overlap and all previously established proposal fixtures.

No empirical result is produced. Existing rejected policies, protected datasets,
champion, authorization flags, risk limits and production source remain unchanged.
