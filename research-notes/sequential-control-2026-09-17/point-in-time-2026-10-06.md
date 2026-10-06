# Witnessed point-in-time entry: engineering result

Base main75571dd13991134952bd625ae6da5611ba9185ad. Preregistration commits
d19b229e and2e790067 preceded implementation. This is an unfrozen research entry,
not a new financial experiment or a reinterpretation of frozen screen evidence.

## Change and interpretation

`point_in_time_v3.train_point_in_time_v3` takes immutable `Record` values, explicit
symbol/close/decision tuples, a processing delay, and the existing PPO training
parameters. Both public entries default disabled and reject unknown versions.
The [contract](../../formal/research/point-in-time-contract.md) specifies exact
UTC microseconds and each record's release, first-seen, collection and revision
witnesses. `admit_v3` chooses the unique highest revision available at that bar's
own decision. It never changes an older bar when a later correction arrives.
Unknown headers, missing slots, duplicate greatest visible revisions and invalid
selected values reject the whole batch; missing funding is never zero-filled.
Unavailable payloads are not inspected. Invalid future *headers* still reject:
metadata must be intelligible before availability can be decided.

This conservative frozen-vintage interpretation has explicit limitations. It does
not rebuild historical features using the latest revisions known at each later
decision. It can refuse late public bars. The caller specifies intended decision
times; no actual fill or collection-clock accuracy follows from admission. Release
may be absent, using mandatory first-seen and collection witnesses; a positive
revision must have a revision-release timestamp. Correct-looking forged witnesses
cannot be detected here. No timestamp was invented for the frozen CSV archive.

The accepted bytes are immutable and training receives independent arrays. A v3
result retains the exact grid and chosen witnesses with the v2 immutable result.
No persistence or v4 artifact-loader consumer accepts the v3 envelope implicitly.
No Python production service, dependency, configuration, live flag or deployment
change is introduced. Ordinary v1/v2 interfaces preserve their semantics.

## Formal and implementation evidence

F-RL-PIT-TIME: seven independent satisfiable-premise/UNSAT pairs establish integer
availability ordering, signed-63-bit published bounds and maximum-revision/tie/
unavailable-record transfer. Python intermediate integers do not overflow;
values whose availability exceeds a bounded decision cannot be selected.

F-RL-PIT-FLOW:126 reachable states,470 directed transitions, maximum shortest
path6, two slots and three ordered revision ranks, explored to a fixed point.
Repeated scans have no retry-depth cutoff. Only completely admitted slots reach
training/publication; disabled and failed calls publish absence. This is a finite
control abstraction, not a compiler or scheduler/liveness proof.

F-RL-PIT-BOUNDARY: complete ten-definition source review, actual guard/selection/
publication extraction, two default gates and one dominated training call. The
research effect/import roster grows from12 to13 modules. All11 existing scoped
closures require this new boundary certificate in the same verification run.
Original38 obligation titles, scope and criteria are unchanged.

F-RL-PIT-CONFORMANCE:64 seeded differential record cases, five unavailable-payload
cases and one actual registered synthetic PPO training step (seed31,121 constant
bars,one symbol). Additional integrity tests cover malformed timestamps, overflow,
non-finite values, missingness, symbol isolation, immutable private handoff,
training failures, source/model mutations and omission of a required certificate.
Property tests and the synthetic learner execution are engineering evidence,
not formal proof or profitability evidence. An initial test setup missed a local
`sys` import; it failed, was corrected, and is not counted as a passing run.

A-PIT-WITNESS records trusted caller timing/revision truth and Python/NumPy
semantics. Source-to-model correspondence is reviewed and tested, not universal
language refinement. No universal numeric/reward/accounting theorem is claimed.
No implementation counterexample was found in this scoped change; deliberate
bypass/availability mutants are verifier regressions, not historical data findings.

## Decision and outstanding work

Obligation4 becomes **partially_verified**, not closed. Current totals:
**11 scoped closures,24 partial,3 open**;27 remain unresolved. No candidate passes.
Timestamp authenticity, frozen-data gaps and inherited ingestion remain blockers.
Numeric obligations10, durable recovery37 and progress/isolation38 remain open.
The v3 entry must remain offline until separate empirical and operational gates
are satisfied; the present change provides no admission to shadow/paper/live use.

Financial evidence is unchanged:108 fits,19,440 replays,19,548 registry rows are
contaminated development; all108 OPE batches invalid;1,227 final returns sealed;
prospective embargo2027-01-20T13:00Z. No market data or final holdout was read.
No new OOS returns, costs, drawdown, tail risk, OPE, inference benchmark or economic
performance is claimed. Recommendation: no adoption; continue offline research.

## Reproduction and verification

Use the existing pinned toolchain and wrappers:

```sh
bash scripts/verify.sh formal
bash scripts/verify.sh full
```

No network is required after dependencies are installed. The existing proof
receipt, ledger and source locks include this entry. Verification run evidence is
recorded below after actual completion; an unrun check is not a pass.

Local targeted verification:20 tests passed in5.822s (PointInTimeTests and
IntegrityTests). The specification coverage check passed:40 specs,353 named
features,483 clauses,599 implementation files,256 evidence links,34 risks.
No proof placeholders found. Full pinned wrapper results remain pending.
