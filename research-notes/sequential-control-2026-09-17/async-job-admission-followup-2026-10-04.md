# Async job admission follow-up — 2026-10-04

Decision: repair reservation ownership and publish-before-execute behavior in
inherited `Main.startJob`. Registration `32f07754` precedes implementation; baseline
main is `025557184af9219624fda170f97a95db55529ae8`. No financial trial, data retrieval,
checkpoint, cost model, champion decision, holdout access or deployment is included.

## Findings and repair

`startJob` reserved a slot before allocation/persistence, with release installed
only in a later child. The parent could fail or be interrupted before fork and
strand the slot. It also forked a runnable callback before publishing its thread
in the job store. Failed or interrupted publication could leave untracked execution.

Two preserved control-slice witnesses reproduce these inherited orderings:

| Witness | Baseline adapter | Compiled repair |
| --- | --- | --- |
| Preparation throws | Count remains 1 | Count returns to 0 |
| Publication throws | Callback ran; count remains 1 until explicitly stopped | Callback never runs; count returns to 0 |

CE-ASYNC-001/002 are baseline control-slice refutations, not execution of the entire
HTTP application. The manifest preserves the original Main function and hash and
identifies the adapter's substitutions for preparation and publication effects.

The new small `Trader.App.AsyncJobAdmission` module owns a private bounded counter.
The masked parent releases on failure before fork; successful fork transfers release
to the child. The child's private gate receives True only after Main publishes its
entry. Failed publication sends False; the child releases without executing the
callback. Counter release is a nonblocking atomic IORef update. No dependency is
added. Main retains identifiers, queue messages, JSON contracts, payload persistence
and configured limits. A cancelled caller after successful publication may leave
its registered job running, with that job still owning its slot.

## Verification and limits

- SMT: six signed-Int arithmetic obligations, source-bound to total checked
  predicates. Acceptance cannot overflow or exceed a valid configured capacity.
- Finite model: two callers at capacities 1 and 2, 369 states, 846 transitions,
  max shortest depth 14 and initial decreasing rank 20. Exact ownership, no early
  execution, no overbooking and no stranded terminal reservation are checked.
- Compiled differential evidence: 292 numeric cases, including 256 generated.
- Seven barrier-controlled runtime tests cover preparation/publication errors and
  interruption, masked parent, queue saturation, callback cancellation and 32
  generated concurrent cases. Fixed seed 20261004. These are tests, not proofs.
- Mutation checks reject leaks, double release, early execution, overbooking,
  nontermination and changed gate behavior. Source hashes and ledger mappings bind
  the actual integration to the reviewed model and tests.

A-ASYNC-ADMISSION names trusted runtime primitives, private state, exception-safe
publication, finite allocation, and scheduler/callback progress. This is not full
Haskell IO refinement, a proof of arbitrary publisher behavior, durable job recovery,
HTTP-wide draining, callback side effects or external resource completion. A running
callback that cannot terminate legitimately retains capacity. An interrupted
preparation can still leave an inherited durable running record; this repair does
not claim that record is reconciled.

The broader mission remains 3/38 closed, 28 partial and 7 open. The new evidence
narrows obligations 10 and 33–38 without weakening their closure criteria. Economic
results, contaminated development status and the sealed final holdout are unchanged.
No candidate is adopted; continue offline research and lifecycle verification.

## Reproduction

Use the pinned tools in `formal/research/toolchain.json`; checks run without network
after dependencies are installed. Run from the repository root:

```sh
bash scripts/verify.sh formal
bash scripts/verify.sh haskell
bash scripts/verify.sh full
```

The PR records actual results for its tested head; deterministic receipts live in
`formal/research/results.json`. No new policy or production inference benchmark is
claimed. The model bound is a verification scope, not a production capacity cap.
