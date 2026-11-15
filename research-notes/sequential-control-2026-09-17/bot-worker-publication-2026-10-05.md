# Bot worker publication — 2026-10-05

Registration 8cddd127 precedes implementation, against main 287b888a. The current
bot-start transaction previously forked its child before reading the timestamp and
committing `BotStarting`. An exception in that interval restores the old MVar state
but cannot undo a fork. A deterministic local adapter preserves that critical order
and demonstrates child execution with no published registration. This is an
engineering counterexample, not evidence of a live exchange incident.

## Repair and canonical interpretation

`Trader.App.WorkerPublication` supplies a private gate around a masked map update.
Preparation, including timestamp acquisition, runs before fork and is interruptible.
The existing duplicate check remains inside the same transaction and uses Retain
for the unchanged result. Launch receives the real child ThreadId, builds the same
BotStarting record, and forces the publication pair/state/outcome to WHNF. Only
when the MVar replacement returns does the parent release True. Any earlier
exception releases False and propagates; the waiting child exits without action.
No blocking killThread/throwTo is used for rollback. Accepted action execution is
unmasked. Post-commit cancellation can interrupt the parent response while retaining
the committed worker; it is not described as a failed pre-commit publication.

No record fields, trading permissions, fleet settings, risk limits, ownership key,
position-adoption rules or model behaviors change. No production state is touched.
The helper does not supervise an action after successful handoff. Explicit stop,
replacement, persistent ownership, readiness freshness and drain ordering still
need composition with this boundary. There is no claim that a dead worker's later
cleanup or a multi-instance ownership conflict is fixed here.

## Exact verification scope

- Complete source binding: one production handoff, two existing duplicate checks,
  one private gate, one child fork, three publication WHNF cuts. A reviewed legacy
  function is retained in `bot-worker-publication-legacy.json`.
- Six SAT-premise/UNSAT-violation queries cover gate/failure implications and the
  actual nested-map insertion interpretation: accepted owner identity, duplicate
  preservation, other-tenant preservation and other-symbol preservation.
  SMT array indices are unbounded integers representing normalized keys; HashMap
  semantics and key correspondence are explicit trusted library assumptions.
- Fixed-point model:two callers, one protected key, empty or existing registration.
  Empty case 134 states, 274 edges, depth 12; existing case 24 states, 34 edges, depth 6.
  Total 158 states, 308 edges;initial strictly decreasing protocol rank 18. Execution
  is a terminal handoff, not a theorem that arbitrary bot actions terminate.
- Legacy model 225 states, 450 edges. CE-BOT-UNPUBLISHED-WORKER reaches
  lock → prepared → fork → execute before publication. The compiled adapter
  also schedules rollback after the child begins, with no network or order effects.
- Six compiled Haskell scenarios use the actual helper: legacy ordering,
  committed identity/unmasked execution,unchanged state, publication failure at
  pair/state/outcome forcing, cancelled preparation, two concurrent duplicate starts.
  Failed-child tests carry its ThreadId in the injected exception and wait for
  ThreadFinished/ThreadDied before asserting no action ran; no sleep-only absence
  assertion substitutes for completion. Three-second test watchdogs are fixture
  guards, not production timing guarantees.
- Source mutations reject removed masking/gating, accepting failure, omitted WHNF
  evaluation and bypassed production handoff. Model/solver mutations independently
  detect pre-publication execution and failed/vacuous predicates.

## Assumptions and limitations

A-BOT-PUBLICATION names GHC 9.4.8/base 4.17.2.1 fork/MVar/masking semantics, normal
allocation, finite ordinary admitted values and total fixed publication operations.
Pure constructors are forced to WHNF, not deep-normal form. Private gate writes
and the final uncontended MVar put finish under masking. Fork returns one thread
or fails without one. Fair scheduling and returning primitives support model
progress; no wall-clock, resource-exhaustion or arbitrary-code guarantee follows.
Repeated external cancellation and uninterruptible foreign calls remain outside
the liveness statement. Source/model correspondence is reviewed and conformance
checked; no whole-GHC IO refinement or operating-system sandbox is claimed.

The abstraction omits external stop/removal, so current-owner uniqueness during
those races is not inferred from startup uniqueness. No persistent transaction,
venue reconciliation, cross-instance owner identity or crash-restart theorem is
claimed. Existing model/type/import certificates must reproduce on the expanded
application graph; the research proposal module remains outside all six executable
root closures.

## Traceability and disposition

F-BOT-PUBLICATION-GATES/FLOW/SOURCE/CONFORMANCE map to the helper, actual Main
call, test module, source registry, preserved counterexample and CI wrappers.
They add evidence to 15/33/34/36/38. Original 38 scope/criterion strings are unchanged.
15 moves open → partial because a real startup-publication boundary is now checked;
complete live-owner uniqueness remains a blocker. **9 scoped closures, 25 partial,
4 open**; both formalObligationsComplete and missionComplete remain false.

No candidate adoption or new economic result. Frozen 108 fits / 19440 replays / 19548
registry rows remain contaminated development; all 108 OPE batches invalid; 1227
final returns sealed;prospective embargo 2027-01-20T13:00Z unchanged. No market-data
read, training campaign, authenticated endpoint, order, live exploration or deployment.
General recommendation: no candidate passed. RL: continue offline research only.

## Verification and delivery

The six targeted compiled cases and the new solver/model/source mutation tests pass
locally. Required canonical formal/full and final-head CI receipts follow after
execution. The merge must preserve the tested tree and suppress deployment triggers.

Local canonical results are not reported as passing: the first formal attempt
rejected a Haskell test source incorrectly listed in production implementation
coverage; its mapping was corrected. The next `bash scripts/verify.sh formal`
ran 219 tests in 304.566 seconds and failed one existing PPO process-bridge
fixture with `ValueError: PPO process bridge: no actual inference for trained
policy`. The host had concurrent compiler workloads; the deadline was unchanged.
This is an observed local failure, not a proof of an environmental cause.

Ordinary initial CI job 112014455403 ran all 219 tests successfully (48.345 seconds),
then rejected the stale committed receipt with `certificate differs from reviewed
receipt; investigate before recording`. The replacement receipt must come verbatim
from successful pinned reproduction, with its changed sections reviewed and all
source hashes checked. Neither failure is hidden by editing a recorded result.

Initial pinned full wrapper job 112014454763 failed HLint on three new fixture
style issues (newtype, lambda-case and rights), after formal had passed. These
were corrected in the test module without changing the production helper or
model. The test source lock was refreshed; a fresh receipt and complete pinned
wrapper run are required after this correction.

Local `bash scripts/verify.sh haskell` completed successfully after the fixture
style correction: build, fourmolu, HLint, smoke checks and the full Haskell test
suite passed. Focused `python scripts/formal/test_integrity.py
BotWorkerPublicationTests` also passed all three tests in 10.720 seconds,
including all six compiled scenarios. No new test timeout was relaxed.

Pinned reproduction run [37385663732](https://github.com/diegueins680/trader/actions/runs/37385663732),
job 112018186771, succeeded on 43a9bb6178360b07bd9090905aa6d19c8da3f94c:
receipt 23:02:42–23:03:34 UTC; canonical formal 23:03:34–23:04:56;
canonical full 23:04:56–23:10:18 on 2026-10-05. Formal ran 219 integrity tests
(33.860 seconds separately; 34.484 seconds within full), all passing. Full also
passed Haskell build/format/lint/smoke/test suite, 241 web tests and 185 automation
tests. The 73 SMT obligation groups and existing 9 closure certificates reproduced.

Receipt SHA256: `13965d8cbde6e043d7f3e55427154e0f5db37d140625dfff1abf9495d169ead2`.
Imported bytes are the runner's emitted receipt, not locally edited output. Changes
are confined to botWorkerPublication, sourceHashes, the added SMT group, the
capability graph's one added module/edge (121 files/296 local edges), and the
inventory-readiness source hash. Its readiness evidence is otherwise unchanged;
the other five executable root closures are identical. All toolchain source
hashes were checked against local bytes. The original 38 criteria and required
certificates are unchanged; the receipt still reports 29 unresolved and 9 closed,
with formalObligationsComplete=false and missionComplete=false.

The temporary PR-only reproduction workflow is removed after success. Final-head
ordinary CI and merge/deployment audits are recorded in
[PR #306](https://github.com/diegueins680/trader/pull/306); the full wrapper above
covers the same implementation, proof and test sources. There is no claim that
these checks close the broader unresolved mission or authorize production use.
