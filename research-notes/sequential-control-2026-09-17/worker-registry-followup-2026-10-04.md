# Worker registry engineering follow-up — 2026-10-04

Decision: repair inherited supervised-worker shutdown; preserve the financial
champion and all research promotion gates. No model training, financial experiment,
market data retrieval, final-holdout opening, exchange call or deployment is part
of this change. Registration was committed as `33b5ea42` before implementation.
Baseline: main `e39fbfa8346a32361a81226e4f6d20822c3f49df`.

## Reproduced defects and corrected behavior

| Fixture | Baseline | Repaired implementation |
| --- | --- | --- |
| Uninterruptible callback; stop times out; retry | Count becomes 0; retry reports success while callback remains blocked | Count stays 1; retry reports failure until callback finishes |
| Start after successful empty-registry stop | New worker starts | Returns Nothing and creates no worker |
| Cancellation delivered while callback finalizer remains blocked | Stop reports success | Stop waits for explicit finalizer acknowledgement or reports failure |

The exact old function bodies are retained as a small, disconnected regression
fixture. The formal command compiles both baseline and repaired implementations
against the same barrier-controlled witness and checks their distinct outputs.
These counterexamples are CE-WORKER-001/002/003. The fixture is not production code.

The registry now closes persistently under its MVar lock, retains unfinished
entries across timeouts, serializes one cancellation request per worker, and uses
nondestructive completion reads. Masked publication and `forkIOWithUnmask` ensure
registration precedes callback entry and callback execution is interruptible even
when the parent starts masked. The outer completion finalizer never acquires the
registry lock. Completed entries are pruned during subsequent admitted starts or stops.

Five existing Main callers discard the internal return value, so changing it to
`Maybe ThreadId` requires no CLI/API/configuration/model-identifier migration.
The production application is compiled by the required full verification wrapper.
The worker registry is distinct from HTTP drain admission: no server-wide admission
or authorization guarantee is inferred from this lock.

## Evidence and limits

- Model checked: two workers, two stop callers, two timeout ticks/caller; 14,095
  states, 55,904 transitions, shortest depth 16, initial decreasing rank 24.
- SMT verified: seven Boolean atomic-step closure, admission, retention,
  acknowledgement and cancellation-idempotence predicates. Delivery without
  completion has a satisfiable witness.
- Property/regression tested: five compiled tests, including 32 generated start/stop
  scheduling cases with seed 20261004. Three original counterexamples are compiled
  and reproduced against the old fixture, then rejected by the repaired module.
- Mutation tests reject reopen, premature success, forgotten unfinished entries,
  duplicate helpers, delivery-as-completion and a nonterminating protocol step.
- Source locks and model/test mappings are in the machine-readable ledger and
  `worker-registry-source.json`. No proof placeholder is used.

The [contract](../../formal/research/worker-registry-contract.md) defines the
abstraction and assumptions. This is not full Haskell IO refinement, exhaustive
GHC scheduling, durable recovery, forced termination of uninterruptible actions,
OS-thread reclamation or completion of detached descendants/external resources.
A timeout before registry-lock acquisition may return False without closing it.
No unconditional wall-clock guarantee is claimed.

The 38 broad obligations remain 3 closed, 28 partial and 7 open. This repair adds
specific evidence to 14 and 33–37; none of those entire obligations is closed.
Remaining work includes HTTP admission races, async/bot/listen-key completion,
resource reconciliation, logging bounds, persistent recovery and composition of
the repaired offline components into a separately registered successor. Frozen
v1 counterexamples and contaminated economic evidence remain unchanged.

## Reproduction

Install the versions in `formal/research/toolchain.json` once. Subsequent proof
and fixture runs need no network. From the repository root:

```sh
bash scripts/verify.sh formal
bash scripts/verify.sh haskell
bash scripts/verify.sh full
```

`formal/research/results.json` holds deterministic reproduced counts. The PR
records actual command outcomes for its tested commit. These commands must fail
on drift; re-recording is a reviewed maintenance operation, not a way to accept
an unexplained proof failure. No production performance or profitability claim
is made by these engineering tests.

Final model review corrected MC-WORKER-001: a captured worker may finish before the cancellation request is dispatched. Per-caller immutable capture masks now represent this interleaving, with 1,220 post-completion dispatch edges. The preserved trace is replayed in the formal gate. This is an abstraction-coverage correction, separately recorded from the three implementation defects.
