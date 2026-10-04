# Shared drain / pool admission follow-up — 2026-10-04

Decision: repair stale-ingress admission in the existing async/backtest pools.
Registration a803a4bd precedes implementation on main 91e96444. This is an
engineering repair, with zero financial trials and no market data, holdout,
champion, policy, authorization-setting, ownership-setting or deployment change.

## Conflict and canonical scope

The HTTP ingress snapshot and later pool reservation were separate operations.
A request could read isDraining=False, pause, then reserve after beginDrain=True.
Local async pool sealing occurs only during later cleanup and could not exclude
this schedule; the backtest pool had no shared drain guard at all.

The affected requirement is linearizable reservation ordering. A reservation that
commits before the drain transaction retains its cleanup obligation and may begin
or finish its callback afterward. This is explicitly weaker than broad obligation
14's whole-server compute/bot/order requirement; that obligation remains partial
with unchanged closure criteria. Do not use this repair to reclassify late callback
execution, synchronous signal/trade or bot/order authorization as verified.

## Implementation and compatibility

DrainController now privately holds a TVar Bool. unlessDraining takes STM rather
than arbitrary IO, so each pool composes its latch check and counter update in the
same transaction. beginDrain is monotonic; when concurrent calls complete normally, exactly
one reports the first transition. All three Main async stores and the shared backtest gate receive the
same controller. Independent offline constructors keep a private open controller.

Slot limits, masked ownership, release, local async completion acknowledgement,
execution timeout and asynchronous exception propagation remain in force. A drained
backtest returns BacktestDraining, mapped to HTTP 503 with the existing JSON error
shape; Busy/Timeout/Exception mappings are unchanged. Waiting retries only Busy,
so it exits after observing drain at its next reservation attempt. The inherited
one-second delay remains; no hard queue-wait deadline is claimed. Async closure
retains its existing error message and mapping.

STM 2.5.1.0 was already a production dependency. It is now pinned and added to the
test component; no new service or verification framework is introduced. Primary
reference: [pinned STM documentation](https://hackage-content.haskell.org/package/stm-2.5.1.0/docs/Control-Concurrent-STM.html),
which links the original PPoPP 2005 Composable Memory Transactions paper.

## Evidence and counterexamples

| Requirement | Evidence | Scope |
| --- | --- | --- |
| F-DRAIN-POOL-ORDER | Five Z3 integer/Boolean claims | Arbitrary nonnegative bounded counts, positive independent capacities and Boolean closure; runtime serializability assumed |
| F-DRAIN-POOL-LIFECYCLE | 5,376 states / 17,728 transitions | Two callers/pools/drainers, capacities 1/2, eight assignments; max shortest depth 10, decreasing rank initially 12 |
| F-DRAIN-POOL-CONFORMANCE | Four compiled tests and 32 generated concurrent cases | Seed 20261004; actual helpers plus source-bound Main constructors; not full HTTP/IO refinement |

The model checks ownership, capacity, monotonic closure and absence of reservation
after drain, including stale ingress and existing owners. Optional local close is a
superset of the backtest pool's always-unsealed projection. It abstracts a single
reservation attempt; the older model separately contains waiting retry cycles.
Conditional finite completion does not prove scheduler fairness or callback
termination in an actual server.

CE-DRAIN-POOL-001/002/003 preserve the stale-ingress async, immediate-backtest and
waiting-backtest schedules. Literal baseline modules plus a labeled scheduling
adapter accept all three after drain; current helpers reject all three. These are
not a full HTTP server replay. Existing counterexamples remain executable.

Tests cover transaction rollback, no rejected prepare/publish/callback effects,
release and acknowledgement after drain, a capacity-blocked waiter observing drain,
independent controllers, and concurrent callers/drainers. The seed generates test
configurations; OS interleavings are not seeded or exhaustively tested. Mutations reject late
reservation, reopening, leaks, nonprogress and a disconnected Main controller.
The invariant checks actual count/ownership changes as well as transition labels;
a mislabeled late-admission mutation is rejected. No proof placeholder or check
relaxation is introduced.

## Assumptions, failures and traceability

A-DRAIN-POOL trusts pinned GHC/base/STM atomicity, rollback and masking semantics,
private state and finite pure transactions. Contention and scheduling have no hard
wall-clock bound. Existing callbacks and IO still require their named assumptions.
Older counter proofs are projections with an independent open controller, with
source manifests refreshed only for reviewed changes and unchanged bodies checked.

Local metadata generation first refused the changed newJobStore body because it
was omitted from the explicit reviewed-function list; the corresponding source
mutation test then failed for the absent manifest. Adding the reviewed constructor
preserved all checks, and all three targeted integrity tests passed. A subsequent
risk-document update used the wrong JSON collection name (`risks` instead of
`entries`) and stopped after writing ledger/spec updates; only the remaining risk
updates were rerun. No duplicate requirement or weakened validation was accepted.

The first aggregate `python scripts/formal/verify.py --record` run failed in
unchanged `inference_process.py:166`: its default worker exceeded the three-second
exit limit (`subprocess.TimeoutExpired`). New drain checks completed before that
failure. Concurrent compiler/runtime load was present; causation is unproved.
No timeout or gate was relaxed. A successful complete reproduction and canonical
wrapper run are required before delivery; the PR records both attempts.

The second aggregate attempt failed at the same unchanged suite's injected
`startup_hang` case (three-second external harness timeout). An isolated run using
the default macOS temporary directory also failed at the default-mode exit check.
A separate startup diagnostic measured a fresh binary at 1.684 seconds versus
0.040/0.031 seconds for later launches. This demonstrates variable startup cost,
not its root cause; observed host load was high and causation is not established.

The unchanged isolated suite then passed with `TMPDIR=/private/tmp`: 226 guard
cases, 12 synthetic-network cases, all three default modes and all eight injected
faults; `startup_hang` took 2.334 seconds. No three-second harness, one-second
startup, twenty-millisecond request or cleanup budget was changed. This diagnostic
is not substituted for the mandatory complete/canonical runs. Local reproduction
uses the explicit temporary-directory setting; CI uses its ordinary Linux scratch
location. The complete aggregate reproduction then passed in 233.728 seconds with the same
limits and `TMPDIR=/private/tmp`; the PR records subsequent canonical outcomes.

The first targeted Haskell wrapper built and formatted successfully, then stopped
on HLint's `Use fromMaybe` hint in AsyncJobAdmission. The equivalent default
expression was corrected; the hint was not suppressed. Final wrapper outcomes are
recorded in the PR after actual completion.

The source contract, ledger, fixtures and result receipt provide bidirectional
requirement-to-implementation/test/CI traceability. Canonical registry validation
passed with 40 specifications, 419 clauses and 34 risks. The PR records the final
canonical wrapper and GitHub CI outcomes; a registration or passing unit test is
not a claim that those broader checks passed.

## Decision and reproduction

SHUTDOWN-DEADLINE-001 remains HIGH/OPEN. Broad obligations remain 3 closed / 28
partial / 7 open. Shared reservation ordering adds evidence to 14/21/33/34/36/38;
full server draining, durable recovery and full implementation refinement remain
unresolved. No candidate is adopted. Existing contaminated development, invalid
OPE, missing matched-champion confirmation and 1,227 sealed final returns are
unchanged. Continue offline research and lifecycle verification.

```sh
bash scripts/verify.sh formal
bash scripts/verify.sh haskell
bash scripts/verify.sh full
```

Use the pinned toolchain in `formal/research/toolchain.json`. Dependencies are
installed beforehand; proofs and deterministic fixtures run without network.
No production inference or economic-performance benchmark is claimed.
