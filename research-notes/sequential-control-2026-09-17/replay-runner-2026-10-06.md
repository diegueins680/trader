# Exact replay runner v2 — 2026-10-06

**Decision: retain a disabled composition kernel; no adoption.** No prices,
settlements, archived results, protected holdout or prospective campaign outcomes
were read. Only synthetic fixtures exercise the source. The champion, production
configuration and all frozen evidence are unchanged.

The [registration](../registrations/replay-runner-v2-engineering.json) and
[contract](../../formal/research/replay-runner-v2-contract.md) were committed as
`9696c458` before the implementation.

## Why this increment

The updated next action for obligation 10 was to compose the exact replay and
funding kernels into a checked runner. Neither kernel alone shows that bars,
funding buckets and targets line up, or that an episode's wealth adds up across
steps.

## What changed

`scripts/research/replay_runner_v2.py` (`run_v2`, `enabled=False` by default)
replays a **fixed, caller-supplied target schedule**. Step k is the transition
from bar `left+k-1` to bar `left+k`, as in frozen `sequential_replay_v1`. It
reads only `prices[left+k]`, `buckets.events[left+k]` and `targets[k-1]`. The
last step is forced terminal, and the run stops at the first terminal receipt.
Any rejected input or transition returns `None`; otherwise it returns an immutable
`Episode`. The only imports are the two v2 kernels.

## Evidence

| Requirement | Status | Scope |
|---|---|---|
| F-RL-RUNNER-V2-SOURCE | exhaustively_checked | Complete AST lock; exactly four reviewed indexed reads; forced final terminal; composition-only imports; activation-first default. |
| F-RL-RUNNER-V2-ARITH | smt_verified | 7 SAT-premise/UNSAT-violation pairs: reads in bounds and never past bar `left+k` at step k; telescoping of final equity from the per-step identity; charged bucket index ≥ 1, so an event charged at step k lies in (close[left+k-1], close[left+k]] and the bucket-0 prehistory exception is never charged. |
| F-RL-RUNNER-V2-FLOW | model_checked | Horizons 1..8 (180 states, 172 transitions): only terminal episodes with 1..H receipts publish; a mutant publishing an unterminated episode is caught. |
| F-RL-RUNNER-V2-CONFORMANCE | property_tested | 64 seeded episodes (157 receipt rows reconciled by the Haskell Rational oracle, 4 early terminations); 64 metamorphic episodes where every price and settlement after the step-k close is rewritten and receipts 1..k stay identical. |

The metamorphic check is shown to be sensitive: runner variants that read the
next bar's price, or the next bar's funding bucket, are rejected with
`future data changed an earlier receipt`. Five integrity tests cover the
end-to-end check, six source mutants, the model mutant, solver SAT/UNKNOWN
refusal, 11 malformed-input cases, a forged bucket version, interrupted and
failing transitions, bucket alignment, terminal liquidation and early insolvency.

One lemma is weak and is reported as such: "forced terminal only at the final
step" is close to a tautology once the source binding is established. The
binding itself is what the source check provides.

## Verification on this host

| Command | Result |
|---|---|
| `scripts/formal/test_integrity.py` | 300/301 OK; the only error is the environmental `PPOProcessBridgeTests` deadline failure seen on clean `main` (load average 37–57) |
| `node scripts/verify-formal-specs.mjs` | valid |

`verify.py --record` reproduces every certificate, including the load-sensitive
PPO bridge, so it was deferred until host load dropped. The commit message and PR
record the final outcome. Clean CI is the authoritative verdict.

## Limitations

A-REPLAY-RUNNER-V2 inherits A-REPLAY-ACCOUNT-V2 and A-FUNDING-EVENTS-V2, and it
treats a settlement's grid time as its availability time. That is an assumption,
not a provider release or first-seen proof. A fixed target schedule is not a
policy: no observation causality, learning, OPE, shield precedence or economic
claim follows. Obligation 10 stays **open**; counts stay 12 scoped closures /
25 partial / 1 open.

## Recommendation

Unchanged: **no candidate passed**. Next step for obligation 10: repair the
Q/CQL successor paths (CE-RL-014/015), then connect a policy to this runner only
through the existing shielded proposal boundary.
