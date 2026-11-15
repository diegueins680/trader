# Champion archive isolation contract v1

Engineering registration, 2026-10-05, against main
b706fa9bdcac058cce6c2a17d3a808f614270def. No financial experiment,
holdout access, live authorization, deployment, or champion replacement.

## Requirement and discovered conflict

Obligation 28 retains its exact scope and criterion: "Challenger failure cannot
modify champion configuration/artifacts or selection." Existing type/import and
promotion certificates do not alone establish filesystem preservation.

The screen runner's JSON, events, returns and policy writers create files
exclusively. The report exporter instead calls write_bytes after creating a fresh
directory. A colliding file or leaf symlink inserted before that call can be
truncated. Preserve this scheduled counterexample. Replace only the exporter
write with exclusive binary creation. Stable-input output bytes and formats must
remain identical. Never delete an existing destination to retry. Failure may
leave new partial research output; it must preserve pre-existing data.

## Formal definition and proof obligations

Let F map path identities to file objects and bytes; C is any set of existing
champion configuration/artifact identities at entry. Exclusive create rejects an
existing leaf, including symlinks; on success it returns a handle to a new object.
Only that handle can receive this writer's bytes. For every c in C, F_final(c)
retains its original object and bytes, including on errors and retries. No
assumption that an analyst selected disjoint paths substitutes for this check.
Directory creation never replaces an existing leaf. Parent-directory namespace
and file identities stay stable for an open operation/handle lifetime; malicious
parent replacement, privileged filesystem mutation, external deletion/recreation
of champion files and manual adoption of unfinished output are outside scope.

F-RL-ARCHIVE-SOURCE: bind all actual research writer modes, destinations, callbacks,
process calls and runtime consumers through the existing full promotion-source
inventory. Derive the exporter tail and runner/policy exclusive writer tails.
The call-site inventory and primitive effects must be reviewed; hashes alone are
not proofs. Candidate outputs cannot select paths or call a champion selector.

F-RL-ARCHIVE-PRESERVE: SMT-check exclusive acquisition, handle ownership, untouched
existing identities/bytes, failure and retry preservation, and other-key
preservation over unbounded symbolic keys. Check premise satisfiability.

F-RL-ARCHIVE-FLOW: model two archive writers, a protected existing object, two
destination keys, colliding insertion, exclusive acquisition, partial/full write,
failure, close and retry. Check reachable fixed point, ownership and preservation.
Allow partial new artifacts; do not infer crash durability or whole-archive
atomicity. The legacy overwriting mode must yield a preserved counterexample.

F-RL-ARCHIVE-CONFORMANCE: execute actual writer/export code on synthetic fixtures,
including existing files, leaf symlinks, colliding insertion, failed writes and
retries. Compare successful output bytes with the preserved legacy implementation.
No market data, training run or production file may be used.

## Composition and assumptions

A-CHAMPION-ARCHIVE: pinned CPython/pathlib/io and filesystem exclusive-create
semantics; ordinary built-in path/bytes/numeric objects; fixed reviewed code and
registered scalar names; no code injection, hostile subclasses or replaced helper
programs. Source-to-model effect correspondence is reviewed and regression-tested,
not a whole-language or kernel proof. Object/parent identity stability is required;
leaf collisions before acquisition are explicitly modeled and tested.

For obligation 28, require freshly reproduced archive source/SMT/model evidence,
full research callback/effect and destination coverage, non-authorizing metadata,
Haskell type/process boundaries, native model schema exclusion, and exclusion from
all six production executable roots. The inherited integration boundary is the
absence of a candidate-to-selector/order bridge, not correctness of all existing
production selection algorithms. Existing champion learning by its own authorized
production code is outside the candidate-caused preservation claim.

Close 28 only if this complete chain proves its unchanged criterion. Otherwise
retain partial status with a concrete blocker. Do not weaken any original 38 scope
or criterion strings. No other broad closure is preregistered. Source mutations,
model counterexamples and failed proof attempts must fail the formal gate.

Pinned formal/full CI must reproduce the receipt. Merge only the tested head,
suppress deployment triggers, and audit the merged tree and deployment records.
