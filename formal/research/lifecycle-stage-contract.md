# Delivered candidate stage preservation

Base main: ac9288ee8c7d61fa6af283f81a4bdf7b47846966. Engineering verification only;
no new policy, deployment, data access, training or financial trial.

## Requirement and consistency resolution, before implementation

Original obligation 25 is unchanged: "Candidate promotion follows all mandatory
evidence stages without skipping or self-promotion." Its original scope remains
unchanged, including the named inherited production integration boundary.

The old blocker requested a concrete persistent promotion service. The delivered
system has no such service: all policy writers label artifacts as research-only,
all readers reject promoted metadata, and the compiled production component graph
excludes the delivered research modules/artifacts. The canonical contract already
says that no candidate becomes eligible for shadow, paper or live activation from
this verification work. The negative decision memo rejects all tested candidates.
A new promotion service would add behavior that neither evidence nor requirements
authorize. The authoritative requirement is a safety restriction on promotion,
not a liveness requirement to promote a candidate or implement every future stage.

This resolves the implementation/specification mismatch explicitly: verify stage
preservation across the complete delivered path. Do not count an abstract service
or an isolated enum as implementation closure. If any consumer can advance a
candidate, retain the blocker until its evidence/history guard is verified.
Production authority, ownership, readiness and other inherited behavior remain
separate obligations, not assumed correct by this resolution.

## Formal meaning

Eligibility stages, ordered by index, are Research (0), BacktestEligible (1),
ReplayEligible (2), ShadowEligible (3), PaperEligible (4), MicroLiveReviewEligible
(5), and LiveAuthorized (6). A lawful promotion is either a stutter or a single
successor below LiveAuthorized with all prerequisite evidence and external review.
The task has no transition to LiveAuthorized, even with apparently favorable data.

Computational phases (construct, save, load, infer, evaluate, reject, disable,
retry, successor construction) are distinct from eligibility. Offline simulation
inside research, including `OfflineReplayV1`, is not a certificate of eligibility
for a separately operated historical-replay/shadow/paper stage. Such computation
cannot set promotion metadata, satisfy gates or authorize a next stage. This
interpretation is consistent with rejected research replay already in main; no
identifier, saved artifact or runtime behavior changes.

The abstraction alpha maps an unpersisted research snapshot/proposal and accepted
v1 `rejected_research_only` or v4 `research-only` metadata to Research. Rejected
metadata produces no admitted candidate. Fixed research report metadata is not a
policy artifact. Existing schema/effect/build certificates must establish that no
other format, callback, writer, process result or production consumer provides an
alternative promotion path.

For every admitted concrete transition C -> C', verify alpha(C)=alpha(C')=Research.
Consequently every prefix of any finite execution preserves eligibility, no stage
is skipped, and no policy-controlled evidence or score causes advancement. This
is stronger refusal than the permitted generic stage relation. It does not claim
that evidence gates have passed, that a candidate will progress, or that an
unimplemented promotion workflow is verified. Fairness is unnecessary for this
safety invariant. Future activation requires a new specification and reopening25.

## Proof obligations and method

- F-RL-STAGE-SOURCE: complete reviewed research import/effect closure and extracted
  writer/reader metadata predicates, composed with the existing Haskell protocol,
  artifact compatibility and six-executable exclusion certificates. Source pins
  detect drift; pins alone are not a semantic proof.
- F-RL-STAGE-STEP: SMT over arbitrary metadata strings and guard booleans, with
  satisfiable premises, proves accepted metadata projects to Research and every
  delivered step refines the lawful stage relation for arbitrary evidence/review
  input. Wrong or advanced stage labels cannot be admitted.
- F-RL-STAGE-FLOW: reachable fixed-point checking of both actual policy formats,
  save/load/infer/reject/disable/retry/successor events, retained proposals and
  attempts at every later eligibility stage. No retry-depth cutoff. A deliberately
  added stage-skipping transition must produce a preserved counterexample.
- F-RL-STAGE-CONFORMANCE: actual writers/loaders with correctly rehashed promoted
  metadata, all stage labels, malformed metadata classes, retained ordinary
  research behavior and nested data. Tests supplement the source/SMT composition.

A-STAGE-PROJECTION names the reviewed abstraction above and trusts ordinary pinned
Python/GHC/JSON primitives, fixed source/callback bindings, and the existing
A-PROMOTION-BOUNDARY/A-CAPABILITY-BUILD assumptions. It does not assume the stage
invariant as a premise: actual constructors, rejecting guards and effect exclusion
must establish it. No hostile replacement binaries, injected code, forged build
receipts, manual artifact conversion or OS/compiler correctness theorem.

Closure25 is conditional on all source, SMT, model, artifact, capability, default,
shield-consumer and added-boundary certificates reproducing in the same CI run.
Every original number/title/scope/criterion stays unchanged. Any missing consumer
or counterexample blocks closure. No other obligation closes from this audit.
Use pinned Python3.13.3, Z3 4.15.4 and GHC9.4.8, existing canonical formal/full CI.
