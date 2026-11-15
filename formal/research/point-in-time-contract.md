# Point-in-time training admission v3

Engineering registration against main75571dd13991134952bd625ae6da5611ba9185ad.
This adds an unfrozen, default-disabled entry around the existing PPO successor.
It does not reinterpret frozen CSV data or authorize a financial experiment.

## Semantics before implementation

All times are integer UTC microseconds in [0,2^63-1]. A requested grid contains
121..4096 increasing completed-bar close times and explicit decision times.
Each decision is at or after its close and strictly before the next close (when
present); processing delay is a separate fixed nonnegative microsecond count. Symbols are explicit, distinct and bounded (1..8).
Each immutable record identifies symbol, price/funding kind, bar close, revision
number, release time (optional), first-seen time, collection time, revision-release
time (optional), and finite binary64 value. Initial revision is zero. Positive
revision requires a revision-release witness. First-seen and collection timestamps
are mandatory; collection >= first-seen >= bar close. Named release/revision
witnesses cannot precede bar close or follow first-seen. No missing time becomes
zero. Price must be positive; funding may be signed, including observed zero.

Availability a(r)=max(barClose, firstSeen, collection, release if present,
revisionRelease if present)+processingDelay. Visibility at grid decision d means
a(r)<=d and a matching symbol/kind/barClose. Select the unique greatest revision
among visible records. Duplicate greatest revisions reject, including identical
payload duplicates; no arbitrary tie break. Unavailable payload values are never
examined. All record headers must be well formed: unverifiable headers reject the
batch rather than being silently skipped. Unknown/unrequested symbols are rejected.

Each bar's value is frozen at its own decision vintage. A correction learned later
does not rewrite prior observations, normalization or rewards. This conservative
replay interpretation differs from a rolling vintage database that rebuilds past
features at every decision. It may reject ordinary late public data and changes
no frozen v1/v2 interpretation. Subsequent values become outcomes only at their
later grid decisions; this does not prove a market fill at that timestamp.

Training gets only independently allocated arrays from a fully admitted grid.
No partial prefixes or partial result can escape. A returned v3 envelope includes
the exact close/decision grids, processing delay and selected timestamp/revision witnesses and the immutable
v2 result. The envelope is not a v4 persisted artifact and has no promotion or
order capability. There is no inference or production consumer. A failed learner
returns absence. Existing v2 remains separately callable with its old limitations.

## Obligations and methods

F-RL-PIT-TIME (SMT): source-derived availability/admission implies every selected
release, first-seen, collection and revision-release is <=decision; bounded integer
addition cannot publish a timestamp above 2^63-1. Integer selection induction
preserves maximum visible revision. Independent SAT premises and UNSAT violations.

F-RL-PIT-BOUNDARY (source/finite checks): complete reviewed module and caller
effects, default/version gates, matching symbols, typed immutable records,
unique-revision selection, private allocation, one dominated v2 training call,
and envelope-only result. Extend the complete research import/effect inventory;
all previously closed boundary obligations require this constituent as well.

F-RL-PIT-FLOW (model checking): repeated record scans, candidate replacement,
ambiguous revision rejection, missing-slot rejection, training failure and
all-or-nothing result. Two slots and three revision ranks, fixed-point search;
no artificial retry bound. Output only follows admission of every slot.

F-RL-PIT-CONFORMANCE (property/differential tests): independent selection oracle,
late revisions, first-seen fallback, missingness, symbol mismatches, ties,
malformed/non-finite values, time overflow, future payload mutation, immutable
input/result handoff and actual successor training on a tiny synthetic fixture.
Preserve any discovered implementation counterexample. Synthetic training is an
engineering fixture, never financial or statistical evidence.

A-PIT-WITNESS: timestamps and revision identities supplied by the caller are
truthful, from a synchronized clock and consistent source; no mechanism here
certifies provider truth or fills missing historical evidence. CPython exact
integer/tuple/dataclass and NumPy copy semantics are trusted. No hostile reflection,
monkeypatching, subclass instances or concurrent source mutation. Source-to-model
mapping is reviewed and tested, not compiler refinement. No general binary64,
reward/fill, liveness, production or economic guarantee is implied.

Obligation4 may move open to partially_verified when these checks pass. It must
not close: frozen input lacks witnesses, inherited ingestion is not migrated,
and actual historical timestamp authenticity is unresolved. Original38 scope and
criteria stay fixed. No final holdout access, deployment, live flag or order change.
