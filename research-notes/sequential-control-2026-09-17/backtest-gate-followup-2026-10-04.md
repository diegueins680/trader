# Backtest gate follow-up — 2026-10-04

Decision: repair inherited backtest execution cancellation and timeout semantics.
Registration `dae95c48` precedes implementation on main `a9c6b3c0`. This is an
engineering repair with zero financial trials, no data retrieval, holdout access,
checkpoint changes, champion selection, live setting changes or deployment.

## Findings and source-grounded interpretation

Main placed `try action` inside `System.Timeout.timeout`. SomeException catches
also capture the timer exception and external cancellation. Timeout expiry could
therefore take the ordinary error branch instead of BacktestTimedOut/HTTP 504,
and shutdown cancellation could be returned as a value. Raw Int multiplication
of configured seconds could wrap negative and request unlimited execution.
Acquisition also preceded finalizer installation without an enclosing mask.

The pinned [System.Timeout source](https://hackage-content.haskell.org/package/base-4.17.2.1/docs/src/System.Timeout.html)
documents negative-duration behavior and timeout's asynchronous-exception hierarchy.
The intended behavior follows the existing distinct failure constructors and HTTP
mapping. H-INTERFACE-E1 uses semantic terminal-outcome names; existing wire JSON uses
`done`/`error`, and cancellation responses use `canceled`. This repair preserves
those wire identifiers: async backtest timeouts remain errors with timeout details,
while synchronous backtest expiry maps to 504. The contract is not interpreted as
a request to rename persisted/API statuses. Whole-interface and tenant-ownership
verification remain outside this scoped repair. No new queue fairness or queue-wait deadline is inferred. Existing waiting
mode retries once per second and starts its execution timer only after admission.

## Repair and preserved counterexamples

`Trader.App.BacktestGate` is a small base-only module using the existing proved
admission/release arithmetic. A private atomic counter and masked acquisition
protect reservation ownership until the finalizer is installed. The callback
retains the caller's original masking state. Release cannot block on a gate lock.
An outer synchronous-only handler wraps `timeout action`, so the gate's timer is
classified separately and external SomeAsyncException values propagate. Integer
conversion saturates before returning to Int. Main retains error/HTTP formatting.

| Witness | Baseline | Repair |
| --- | --- | --- |
| CE-BACKTEST-001: own timeout | ordinary error | timeout |
| CE-BACKTEST-002: ThreadKilled | ordinary error | original cancellation propagates |
| CE-BACKTEST-003: outer timer | swallowed; inner error returned | outer timeout returns Nothing |
| CE-BACKTEST-004: maxBound seconds | negative microseconds | positive saturated microseconds |
| CE-BACKTEST-005: interrupted reserve/finalizer gap | one stranded slot | zero slots after protected callback cancellation |

The first three compile literal baseline functions in a module adapter. The
fourth evaluates the literal inherited duration expression. The fifth deliberately
inserts a deterministic barrier into the unmasked baseline gap and compares it
with cancellation at the repaired callback's first interruptible point. It is a
labeled control-slice schedule, not execution of the entire Main/HTTP application.
Baseline source, hashes, adapters and expected outputs are committed.

## Verification and limitations

- Five SMT integer claims prove positive bounded duration, no overshoot of the
  sanitized exact duration, exact conversion below saturation, saturation and
  monotonicity, for arbitrary positive machine bound and representable seconds.
- Eight finite configurations combine two callers, capacities 1/2 and all pairs
  of immediate/waiting modes: 1,656 states, 2,984 edges, depth at most 8.
  Exact ownership, capacity, no callback after failed admission and exception
  routing are invariant. Every state has a path to quiescence (EF terminal).
  Retry cycles remain explicit; AF termination and fairness are not proved.
- Compiled differential checks: 264 numeric cases including 256 generated.
  Seven runtime tests cover outcomes, nested timers, cancellation, waiting retry,
  saturation, caller masking and 32 generated concurrent schedules, seed 20261004.
- Mutation checks reject slot leaks, duplicate release, swallowed cancellation,
  early callback entry, nonprogress and changed exception classification.
- Source locks preserve the actual Main handoff and unchanged error/HTTP mapping;
  the executable dependency graph and Cabal module roster remain checked.

A-BACKTEST-GATE names runtime semantics, private state, total predicates,
interruptibility and callback cooperation. Arbitrary throwTo delivery of an
exception outside SomeAsyncException remains classified by its type, not delivery
provenance. Internally swallowed exceptions, uninterruptible masking, foreign
blocking calls, surrounding callers, durable recovery and global drain admission
remain outside the guarantee. The timer covers the IO callback, not later evaluation of a lazy returned value.
Source binding and tests are not full IO refinement.

A local metadata update initially assumed all source manifests shared a nested
body-map format. The worker registry uses a flat format; the update failed and a
source-integrity test refused the absent new manifest. The updater was corrected,
existing body checks retained, and targeted mutations rerun successfully. No
verification check was disabled or weakened.

The first aggregate receipt attempt (`python scripts/formal/verify.py --record`)
failed in unchanged `inference_process.py:166`: its default worker exceeded the
three-second exit limit (`subprocess.TimeoutExpired`). New gate checks completed
before that failure. Concurrent compiler/runtime load was present; causation is
not proved. The timeout and acceptance gate are unchanged, and an actual successful
rerun is required before delivery. After the targeted Haskell wrapper passed,
the unchanged aggregate receipt rerun passed in 163.905 seconds. Both attempts
are retained in the review record.

SHUTDOWN-DEADLINE-001 remains HIGH/OPEN. The 38 broad obligations retain their
existing closure criteria: 3 closed, 28 partial, 7 open. This repair supplies
concrete evidence for 10/21/34/36/38, without closing unrelated boundaries.
Financial results, invalid OPE, contaminated development, missing matched champion
confirmation and the 1,227-return sealed holdout remain unchanged. No candidate
passed promotion; continue offline research and lifecycle verification.

## Reproduction

Use `formal/research/toolchain.json` and the canonical root commands:

```sh
bash scripts/verify.sh formal
bash scripts/verify.sh haskell
bash scripts/verify.sh full
```

Receipts are in `formal/research/results.json`; the PR records completed command
and CI results. No production inference or economic benchmark is claimed. No new
dependency, schema migration, API configuration field or deployment is introduced.
